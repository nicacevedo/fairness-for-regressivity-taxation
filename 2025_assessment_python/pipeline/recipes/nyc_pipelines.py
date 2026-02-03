import re
import inspect
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class NYCMassAppraisalPreprocessor(BaseEstimator, TransformerMixin):
    """
    NYC Mass Appraisal Preprocessor (PLUTO + Sales merge), supports:
      - mode="linear": ColumnTransformer with OHE + scaling (good for Ridge/ElasticNet/etc.)
      - mode="lgbm"  : returns DataFrame with pandas 'category' dtypes (good for LightGBM native cats)

    Philosophy (aligned with IAAO-style mass appraisal / AVM guidance):
      - Clean and verify core sale and parcel characteristics, avoid obvious leakage fields,
        and engineer stable, interpretable predictors (areas, ages, density, time, and location strata).
      - Preserve location stratification variables (borough/nta/census/tract/community board, etc.).
      - Use consistent train/test transformations without peeking (no test-learned categories/imputation).

    Important:
      - This transformer does NOT drop rows (sklearn convention). Use `sales_filter_mask(...)`
        to filter training rows (e.g., invalid sale price, missing coords) before modeling.
    """

    def __init__(
        self,
        mode: str = "linear",                  # {"linear","lgbm"}
        output: str = "sparse",                # {"sparse","array","dataframe"} for linear mode only
        target_col: str = "sale_price",
        sale_date_col: str = "sale_date",
        drop_id_cols: bool = True,             # drop BBL/BIN-like IDs to reduce memorization
        drop_high_card_cols: bool = True,      # drop address/ownername/etc and any huge-cardinality columns
        high_card_threshold: int = 500,        # if a categorical column has > threshold unique values -> drop (if enabled)
        onehot_min_frequency: Optional[float] = 0.005,  # group rare levels for linear models
        scale_numeric: bool = True,            # linear mode scaling
        add_time_features: bool = True,        # year/month/quarter + cyc month
        add_geo_grid: bool = True,             # lat/lon bin as categorical
        geo_grid_precision: int = 3,           # rounding decimals (3 ~ ~100m); 2 ~ ~1km
        add_log_area_features: bool = True,    # add log1p(area) engineered columns (helps linear & sometimes LGBM)
        eps_ratio: float = 1e-9,
        verbose: bool = False,
    ):
        if mode not in {"linear", "lgbm"}:
            raise ValueError("mode must be 'linear' or 'lgbm'")
        if output not in {"sparse", "array", "dataframe"}:
            raise ValueError("output must be one of {'sparse','array','dataframe'}")

        self.mode = mode
        self.output = output
        self.target_col = target_col
        self.sale_date_col = sale_date_col
        self.drop_id_cols = drop_id_cols
        self.drop_high_card_cols = drop_high_card_cols
        self.high_card_threshold = int(high_card_threshold)
        self.onehot_min_frequency = onehot_min_frequency
        self.scale_numeric = scale_numeric
        self.add_time_features = add_time_features
        self.add_geo_grid = add_geo_grid
        self.geo_grid_precision = int(geo_grid_precision)
        self.add_log_area_features = add_log_area_features
        self.eps_ratio = float(eps_ratio)
        self.verbose = verbose

        # Learned artifacts
        self.column_transformer_: Optional[ColumnTransformer] = None
        self.feature_names_: Optional[np.ndarray] = None

        # Learned schema
        self.numeric_cols_: Optional[List[str]] = None
        self.categorical_cols_: Optional[List[str]] = None
        self.dropped_cols_: Optional[List[str]] = None

        # LGBM categorical alignment
        self.lgbm_cat_levels_: Optional[Dict[str, List[str]]] = None
        self.feature_cols_: Optional[List[str]] = None  # final column order after prep (lgbm)

        # Train-learned medians for certain imputations if needed
        self._train_medians_: Optional[pd.Series] = None

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------
    def process_target(self, y: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """Common in mass appraisal: model log(price) for stability."""
        y_arr = np.asarray(y, dtype=float)
        return np.log(y_arr)#np.log1p(y_arr)

    def inverse_target(self, y_pred: Union[pd.Series, np.ndarray]) -> np.ndarray:
        return np.expm1(np.asarray(y_pred, dtype=float))

    def sales_filter_mask(
        self,
        df: pd.DataFrame,
        min_price: float = 10000.0,
        max_price: Optional[float] = None,
        require_coords: bool = True,
        require_positive_areas: bool = False,
    ) -> pd.Series:
        """
        Suggested TRAINING filter mask (does not run automatically).

        Typical mass appraisal / AVM practice is to exclude sales that are clearly unusable
        (e.g., nonpositive or obviously invalid price, missing essentials). :contentReference[oaicite:2]{index=2}
        """
        d = self._ensure_df(df)
        d = self._normalize_columns(d)
        y = self._parse_currency_to_float(d.get(self.target_col, np.nan))
        mask = pd.Series(True, index=d.index)

        mask &= y.notna()
        mask &= (y > min_price)
        if max_price is not None:
            mask &= (y <= max_price)

        if require_coords:
            lat = self._coalesce_first_present(d, ["latitude", "lat"])
            lon = self._coalesce_first_present(d, ["longitude", "lon", "lng"])
            if lat is not None and lon is not None:
                lat_num = pd.to_numeric(lat, errors="coerce")
                lon_num = pd.to_numeric(lon, errors="coerce")
                mask &= lat_num.notna() & lon_num.notna()

        if require_positive_areas:
            for c in ["lotarea", "lot_area", "land_square_feet", "bldgarea", "bldg_area", "gross_square_feet"]:
                if c in d.columns:
                    v = self._parse_numeric_to_float(d[c])
                    mask &= (v > 0) | v.isna()

        return mask

    def get_lgbm_categorical_features(self) -> List[str]:
        """Convenience: categorical columns (after transform) for LightGBM sklearn wrapper."""
        if self.mode != "lgbm":
            return []
        return [c for c in (self.categorical_cols_ or []) if c in (self.feature_cols_ or [])]

    # ---------------------------------------------------------------------
    # sklearn API
    # ---------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y=None):
        df = self._ensure_df(X)
        df = self._normalize_columns(df)

        # Remove target if included
        if self.target_col in df.columns:
            df = df.drop(columns=[self.target_col])

        # Prepare features (cleanup + FE)
        df_prep = self._prepare_features(df, is_fit=True)

        # Decide columns to drop
        drop_cols = self._default_drop_cols(df_prep)
        if self.drop_id_cols:
            drop_cols |= set(self._id_like_cols(df_prep))
        df_prep = df_prep.drop(columns=[c for c in drop_cols if c in df_prep.columns], errors="ignore")
        self.dropped_cols_ = sorted([c for c in drop_cols if c in df.columns or c in df_prep.columns])

        # Infer categorical vs numeric
        cat_cols = self._infer_categorical_cols(df_prep)
        num_cols = [c for c in df_prep.columns if c not in set(cat_cols)]

        # Coerce numeric columns to numeric
        df_prep = self._coerce_numeric_columns(df_prep, numeric_candidates=num_cols)

        # Recompute numeric/cat based on resulting dtypes
        cat_cols = self._infer_categorical_cols(df_prep)
        num_cols = [c for c in df_prep.columns if c not in set(cat_cols)]
        self.categorical_cols_ = sorted(cat_cols)
        self.numeric_cols_ = sorted(num_cols)

        # Store medians for optional fallback imputations (no leakage)
        if self.numeric_cols_:
            self._train_medians_ = df_prep[self.numeric_cols_].median(numeric_only=True)
        else:
            self._train_medians_ = None

        if self.verbose:
            print(f"[fit] columns: {df_prep.shape[1]}, numeric: {len(self.numeric_cols_)}, cat: {len(self.categorical_cols_)}")

        if self.mode == "lgbm":
            df_lgbm = self._fit_lgbm_categories(df_prep)
            self.feature_cols_ = df_lgbm.columns.tolist()
            return self

        # linear mode
        self.column_transformer_ = self._build_linear_column_transformer()
        self.column_transformer_.fit(df_prep)
        self.feature_names_ = self._safe_get_feature_names()
        return self

    def transform(self, X: pd.DataFrame):
        if self.mode == "linear" and self.column_transformer_ is None:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")
        if self.mode == "lgbm" and self.feature_cols_ is None:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")

        df = self._ensure_df(X)
        df = self._normalize_columns(df)

        if self.target_col in df.columns:
            df = df.drop(columns=[self.target_col])

        df_prep = self._prepare_features(df, is_fit=False)

        # Drop same columns as training
        if self.dropped_cols_:
            df_prep = df_prep.drop(columns=[c for c in self.dropped_cols_ if c in df_prep.columns], errors="ignore")

        # Ensure all train-time columns exist
        if self.mode == "lgbm":
            # Align categories and column order after coercions
            df_prep = self._coerce_numeric_columns(df_prep, numeric_candidates=self.numeric_cols_ or [])
            df_prep = self._apply_lgbm_categories(df_prep)

            for c in self.feature_cols_:
                if c not in df_prep.columns:
                    df_prep[c] = np.nan
            df_prep = df_prep[self.feature_cols_]
            # LightGBM requires no object dtypes:
            self._assert_no_object_dtypes(df_prep)
            return df_prep

        # linear mode: coerce numeric
        df_prep = self._coerce_numeric_columns(df_prep, numeric_candidates=self.numeric_cols_ or [])

        X_out = self.column_transformer_.transform(df_prep)

        if self.output == "sparse":
            return X_out
        if self.output == "array":
            return np.asarray(X_out)
        # dataframe output (densifies)
        names = self._safe_get_feature_names()
        return pd.DataFrame(np.asarray(X_out), columns=names)

    # ---------------------------------------------------------------------
    # Core preparation steps
    # ---------------------------------------------------------------------
    def _prepare_features(self, df: pd.DataFrame, is_fit: bool) -> pd.DataFrame:
        d = df.copy()

        # 1) Parse sale date if present and add time features
        if self.sale_date_col in d.columns:
            d[self.sale_date_col] = self._parse_date(d[self.sale_date_col])

        if self.add_time_features and self.sale_date_col in d.columns:
            dt = pd.to_datetime(d[self.sale_date_col], errors="coerce")
            d["sale_year"] = dt.dt.year
            d["sale_month"] = dt.dt.month
            d["sale_quarter"] = dt.dt.quarter
            # cyc month
            m = d["sale_month"].astype(float)
            d["sale_month_sin"] = np.sin(2 * np.pi * (m / 12.0))
            d["sale_month_cos"] = np.cos(2 * np.pi * (m / 12.0))

        # 2) Canonicalize key location columns (lat/lon may exist multiple times)
        lat = self._coalesce_first_present(d, ["latitude", "lat"])
        lon = self._coalesce_first_present(d, ["longitude", "lon", "lng"])
        if lat is not None:
            d["latitude"] = pd.to_numeric(lat, errors="coerce")
        if lon is not None:
            d["longitude"] = pd.to_numeric(lon, errors="coerce")

        # 3) Parse numeric-like object columns (currency, commas, etc.) conservatively later
        # Here, handle common known numeric string columns if present
        for c in [
            "sale_price", "land_square_feet", "gross_square_feet",
            "lotarea", "bldgarea", "comarea", "resarea", "officearea", "retailarea",
            "garagearea", "strgearea", "factryarea", "otherarea",
            "assessland", "assesstot", "exempttot",
            "xcoord", "ycoord",
        ]:
            if c in d.columns:
                if c == "sale_price":
                    d[c] = self._parse_currency_to_float(d[c])
                else:
                    d[c] = self._parse_numeric_to_float(d[c])

        # 4) Basic booleans to int (common PLUTO flags)
        for c in d.columns:
            if d[c].dtype == bool:
                d[c] = d[c].astype(int)

        # 5) Feature engineering (mass appraisal style)
        d = self._feature_engineering(d)

        # 6) Geo grid categorical
        if self.add_geo_grid and ("latitude" in d.columns) and ("longitude" in d.columns):
            lat_r = d["latitude"].round(self.geo_grid_precision)
            lon_r = d["longitude"].round(self.geo_grid_precision)
            # Use a stable string with a Missing token for NaNs
            d["geo_grid"] = (
                lat_r.astype("Float64").astype(str).fillna("Missing")
                + "_"
                + lon_r.astype("Float64").astype(str).fillna("Missing")
            )

        # 7) If there are duplicate synonymous columns (from merges), coalesce some key ones
        d = self._coalesce_common_duplicates(d)

        return d

    def _feature_engineering(self, d: pd.DataFrame) -> pd.DataFrame:
        df = d.copy()
        eps = self.eps_ratio

        # Helper to get column as numeric series
        def num(col: str) -> pd.Series:
            if col not in df.columns:
                return pd.Series(np.nan, index=df.index)
            return pd.to_numeric(df[col], errors="coerce")

        # Choose best available lot/building area columns
        lot_area = num("lotarea")
        if lot_area.isna().all() and "land_square_feet" in df.columns:
            lot_area = num("land_square_feet")
        bldg_area = num("bldgarea")
        if bldg_area.isna().all() and "gross_square_feet" in df.columns:
            bldg_area = num("gross_square_feet")

        df["lot_area_clean"] = lot_area
        df["bldg_area_clean"] = bldg_area

        # Density / utilization
        df["bldg_to_lot_ratio"] = bldg_area / (lot_area + eps)

        # Residential/commercial composition if present
        res_area = num("resarea")
        com_area = num("comarea")
        df["res_share_of_bldg"] = res_area / (bldg_area + eps)
        df["com_share_of_bldg"] = com_area / (bldg_area + eps)

        # Units-derived
        units_res = num("unitsres")
        if units_res.isna().all() and "residential_units" in df.columns:
            units_res = num("residential_units")
        units_total = num("unitstotal")
        if units_total.isna().all() and "total_units" in df.columns:
            units_total = num("total_units")

        df["units_res_clean"] = units_res
        df["units_total_clean"] = units_total
        df["avg_res_unit_size"] = res_area / (units_res + eps)
        df["avg_total_unit_size"] = bldg_area / (units_total + eps)

        # Ages (use sale_year if available)
        year_built = num("yearbuilt")
        if year_built.isna().all() and "year_built" in df.columns:
            year_built = num("year_built")

        sale_year = num("sale_year")
        df["age_at_sale"] = sale_year - year_built
        df["age_at_sale"] = df["age_at_sale"].clip(lower=0)

        # Alterations if present
        year_alt1 = num("yearalter1")
        year_alt2 = num("yearalter2")
        df["years_since_alt1"] = sale_year - year_alt1
        df["years_since_alt2"] = sale_year - year_alt2
        df["years_since_alt1"] = df["years_since_alt1"].clip(lower=0)
        df["years_since_alt2"] = df["years_since_alt2"].clip(lower=0)

        # FAR-related (PLUTO fields builtfar/residfar/commfar/facilfar often exist)
        builtfar = num("builtfar")
        residfar = num("residfar")
        commfar = num("commfar")
        facilfar = num("facilfar")
        df["builtfar"] = builtfar
        df["residfar"] = residfar
        df["commfar"] = commfar
        df["facilfar"] = facilfar
        # Relative intensity vs allowed (where nonzero)
        df["builtfar_to_residfar"] = builtfar / (residfar + eps)
        df["builtfar_to_commfar"] = builtfar / (commfar + eps)

        # Log area features
        if self.add_log_area_features:
            for base, s in {
                "lot_area_clean": df["lot_area_clean"],
                "bldg_area_clean": df["bldg_area_clean"],
                "resarea": res_area,
                "comarea": com_area,
            }.items():
                if base in df.columns:
                    v = pd.to_numeric(df[base], errors="coerce")
                else:
                    v = pd.to_numeric(s, errors="coerce")
                df[f"log1p_{base}"] = np.log1p(v.clip(lower=0))

        # Simple flags
        if "easements" in df.columns:
            ea = pd.to_numeric(df["easements"], errors="coerce")
            df["has_easement"] = (ea.fillna(0) > 0).astype(int)

        return df

    # ---------------------------------------------------------------------
    # Column inference, dropping, coercions
    # ---------------------------------------------------------------------
    def _default_drop_cols(self, df: pd.DataFrame) -> set:
        # Common leakage / high-card fields in NYC merged data
        drop = {
            "address", "apartment_number", "ownername", "notes", "geom",
            "plutomapid", "version", "appdate", "basempdate", "dcasdate", "edesigdate",
            "landmkdate", "masdate", "polidate", "rpaddate", "zoningdate",
        }
        # If present, these often behave like near-unique identifiers
        drop |= {"bbl_int_from_parts", "bbl10"}

        # Also remove unnamed index column from CSV exports
        drop |= {"unnamed_0", ""}

        # Optionally drop very high-card categorical columns after seeing uniques (fit only)
        if self.drop_high_card_cols:
            for c in df.columns:
                if df[c].dtype == object:
                    nun = df[c].nunique(dropna=True)
                    if nun > self.high_card_threshold:
                        drop.add(c)

        return drop

    def _id_like_cols(self, df: pd.DataFrame) -> List[str]:
        # IDs that cause memorization; keep out by default
        candidates = [
            "bbl", "bin", "block", "lot", "tax_block", "tax_lot",
            "bbl_pluto", "condono", "appbbl", "taxmap", "sanborn",
            "xcoord", "ycoord",
        ]
        out = [c for c in candidates if c in df.columns]
        # Also drop exact duplicates with different casing/old names if present
        for c in df.columns:
            if c.endswith("_id") or c.endswith("_identifier"):
                out.append(c)
        return sorted(set(out))

    def _infer_categorical_cols(self, df: pd.DataFrame) -> List[str]:
        # Known NYC/PLUTO categorical-ish fields (often codes/strings)
        known = {
            "borough", "boro", "borocode",
            "neighborhood",
            "building_class_category",
            "building_class_as_of_final_roll",
            "building_class_at_time_of_sale",
            "tax_class_as_of_final_roll",
            "tax_class_at_time_of_sale",
            "zip_code", "postcode",
            "community_board", "council_district", "schooldist", "policeprct", "firecomp",
            "healtharea", "sanitboro", "sanitsub", "sanitdistrict", "healthcenterdistrict",
            "census_tract_2020", "census_tract_2010", "tract2010", "bct2020", "bctcb2020",
            "neighborhood_tabulation_area_nta_2020", "nta", "cb2010",
            "zonedist1", "zonedist2", "zonedist3", "zonedist4",
            "overlay1", "overlay2",
            "spdist1", "spdist2", "spdist3",
            "splitzone",
            "bldgclass", "landuse", "lottype", "bsmtcode", "proxcode", "irrlotcode", "ext",
            "histdist", "landmark",
            "areasource",
            "firm07_flag", "pfirm15_flag", "dcpedited",
            "geo_grid",
        }

        cat_cols = set()

        # 1) objects => categorical
        for c in df.columns:
            if df[c].dtype == object:
                cat_cols.add(c)

        # 2) known codes even if numeric
        for c in df.columns:
            if c in known:
                cat_cols.add(c)

        # 3) Low-cardinality integer-like numeric codes -> categorical (carefully)
        for c in df.columns:
            if c in cat_cols:
                continue
            if pd.api.types.is_integer_dtype(df[c]) or pd.api.types.is_float_dtype(df[c]):
                nun = pd.Series(df[c]).dropna().nunique()
                if nun > 0 and nun <= 30 and (("district" in c) or ("board" in c) or ("borough" in c) or ("class" in c)):
                    cat_cols.add(c)

        # Remove any columns that are obviously continuous numeric
        for c in list(cat_cols):
            if c in {"latitude", "longitude"}:
                cat_cols.discard(c)

        return sorted(cat_cols)

    def _coerce_numeric_columns(self, df: pd.DataFrame, numeric_candidates: List[str]) -> pd.DataFrame:
        out = df.copy()

        # First: coerce candidates explicitly
        for c in numeric_candidates:
            if c not in out.columns:
                continue
            if out[c].dtype == object:
                out[c] = self._parse_numeric_to_float(out[c])
            else:
                out[c] = pd.to_numeric(out[c], errors="coerce")

        # Second: attempt numeric conversion for any remaining object columns
        # that look numeric (commas/$/digits) with high success rate.
        for c in out.columns:
            if out[c].dtype != object:
                continue
            s = out[c]
            parsed = self._parse_numeric_to_float(s)
            success = parsed.notna().mean()
            # If most values parse as numbers, treat as numeric
            if success >= 0.90:
                out[c] = parsed

        return out

    # ---------------------------------------------------------------------
    # LGBM categorical alignment
    # ---------------------------------------------------------------------
    # def _fit_lgbm_categories(self, df: pd.DataFrame) -> pd.DataFrame:
    #     d = df.copy()
    #     self.lgbm_cat_levels_ = {}

    #     # Ensure categorical columns exist
    #     cat_cols = self.categorical_cols_ or []
    #     for c in cat_cols:
    #         if c not in d.columns:
    #             continue
    #         s = d[c]

    #         # Cast to string for stable categories; use Missing token
    #         s = s.astype("object")
    #         s = s.where(s.notna(), "Missing")
    #         s = s.astype(str)

    #         # stable categories from train + ensure Missing exists
    #         cats = list(pd.unique(s))
    #         if "Missing" not in cats:
    #             cats = ["Missing"] + cats
    #         self.lgbm_cat_levels_[c] = cats

    #         d[c] = pd.Categorical(s, categories=cats, ordered=False)

    #     # Numeric columns to numeric
    #     for c in (self.numeric_cols_ or []):
    #         if c in d.columns:
    #             d[c] = pd.to_numeric(d[c], errors="coerce")

    #     # Final schema
    #     self.feature_cols_ = d.columns.tolist()
    #     self._assert_no_object_dtypes(d)
    #     return d

    def _fit_lgbm_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        self.lgbm_cat_levels_ = {}

        # Ensure geo_grid is treated as categorical if present
        cat_cols = list(self.categorical_cols_ or [])
        if "geo_grid" in d.columns and "geo_grid" not in cat_cols:
            cat_cols.append("geo_grid")
            # keep internal record consistent
            self.categorical_cols_ = sorted(set(cat_cols))

        for c in cat_cols:
            if c not in d.columns:
                continue
            s = d[c].astype("object")
            s = s.where(s.notna(), "Missing").astype(str)

            cats = list(pd.unique(s))
            if "Missing" not in cats:
                cats = ["Missing"] + cats

            self.lgbm_cat_levels_[c] = cats
            d[c] = pd.Categorical(s, categories=cats, ordered=False)

        # Numeric columns
        for c in (self.numeric_cols_ or []):
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")

        self.feature_cols_ = d.columns.tolist()
        self._assert_no_object_dtypes(d)
        return d


    # def _apply_lgbm_categories(self, df: pd.DataFrame) -> pd.DataFrame:
    #     d = df.copy()
    #     levels = self.lgbm_cat_levels_ or {}

    #     # Align categorical columns
    #     for c, cats in levels.items():
    #         if c not in d.columns:
    #             continue
    #         s = d[c].astype("object")
    #         s = s.where(s.notna(), "Missing").astype(str)
    #         # Unknowns -> Missing
    #         s = np.where(pd.Series(s).isin(cats), s, "Missing")
    #         d[c] = pd.Categorical(s, categories=cats, ordered=False)

    #     # Numeric columns to numeric (with fallback to train medians if desired elsewhere)
    #     for c in (self.numeric_cols_ or []):
    #         if c in d.columns:
    #             d[c] = pd.to_numeric(d[c], errors="coerce")

    #     self._assert_no_object_dtypes(d)
    #     return d

    def _apply_lgbm_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        levels = self.lgbm_cat_levels_ or {}

        # If geo_grid exists but wasn't in levels (shouldn't happen after fix), handle it
        if "geo_grid" in d.columns and "geo_grid" not in levels:
            s = d["geo_grid"].astype("object").where(d["geo_grid"].notna(), "Missing").astype(str)
            cats = list(pd.unique(s))
            if "Missing" not in cats:
                cats = ["Missing"] + cats
            levels["geo_grid"] = cats
            self.lgbm_cat_levels_ = levels

        for c, cats in levels.items():
            if c not in d.columns:
                continue
            s = d[c].astype("object")
            s = s.where(s.notna(), "Missing").astype(str)
            s = np.where(pd.Series(s).isin(cats), s, "Missing")
            d[c] = pd.Categorical(s, categories=cats, ordered=False)

        for c in (self.numeric_cols_ or []):
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")

        self._assert_no_object_dtypes(d)
        return d


    def _assert_no_object_dtypes(self, df: pd.DataFrame) -> None:
        bad = [c for c in df.columns if df[c].dtype == object]
        if bad:
            raise ValueError(
                "Found object dtypes after lgbm transform. "
                f"These must be numeric/bool/category. Bad columns: {bad[:30]}"
            )

    # ---------------------------------------------------------------------
    # Linear transformer
    # ---------------------------------------------------------------------
    def _build_linear_column_transformer(self) -> ColumnTransformer:
        num_pipe_steps = [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ]
        if self.scale_numeric:
            num_pipe_steps.append(("scaler", StandardScaler()))
        num_pipe = Pipeline(steps=num_pipe_steps)

        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                ("onehot", self._make_onehot_encoder()),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("num", num_pipe, self.numeric_cols_ or []),
                ("cat", cat_pipe, self.categorical_cols_ or []),
            ],
            remainder="drop",
            sparse_threshold=0.3,
        )

    def _make_onehot_encoder(self):
        sig = inspect.signature(OneHotEncoder)
        kwargs = dict(handle_unknown="ignore", dtype=np.float32)
        if self.onehot_min_frequency is not None:
            kwargs["min_frequency"] = self.onehot_min_frequency
        if "sparse_output" in sig.parameters:
            kwargs["sparse_output"] = True
        else:
            kwargs["sparse"] = True
        return OneHotEncoder(**kwargs)

    def _safe_get_feature_names(self) -> np.ndarray:
        try:
            return self.column_transformer_.get_feature_names_out()
        except Exception:
            return np.array([], dtype=object)

    # ---------------------------------------------------------------------
    # Parsing / normalization utilities
    # ---------------------------------------------------------------------
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()

        # If CSV had an unnamed first column index
        for c in list(d.columns):
            if isinstance(c, str) and c.strip().lower().startswith("unnamed"):
                d = d.drop(columns=[c])
        # Normalize to snake_case, keep uniqueness
        new_cols = []
        seen = {}
        for c in d.columns:
            sc = self._snake_case(str(c))
            if sc in seen:
                seen[sc] += 1
                sc = f"{sc}__dup{seen[sc]}"
            else:
                seen[sc] = 0
            new_cols.append(sc)
        d.columns = new_cols

        # Normalize some known column name variants to canonical names (coalesce later too)
        rename_map = {
            "sale_price": self.target_col,
            "sale_date": self.sale_date_col,
        }
        # If user chose non-default names, respect their target_col/sale_date_col
        # but still let original names map in.
        if self.target_col != "sale_price":
            rename_map["sale_price"] = self.target_col
        if self.sale_date_col != "sale_date":
            rename_map["sale_date"] = self.sale_date_col

        d = d.rename(columns={k: v for k, v in rename_map.items() if k in d.columns})
        return d

    def _snake_case(self, s: str) -> str:
        s = s.strip()
        s = re.sub(r"[()]", "", s)
        s = re.sub(r"[^0-9a-zA-Z]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s.lower()

    # def _parse_currency_to_float(self, s: Union[pd.Series, np.ndarray, float, int]) -> pd.Series:
    #     if isinstance(s, (float, int, np.number)):
    #         return pd.Series([s])
    #     ser = pd.Series(s)
    #     # remove $ and commas and quotes
    #     cleaned = (
    #         ser.astype(str)
    #         .str.replace(r"[\$,]", "", regex=True)
    #         .str.replace('"', "", regex=False)
    #         .str.replace(" ", "", regex=False)
    #     )
    #     cleaned = cleaned.replace({"": np.nan, "nan": np.nan, "None": np.nan, "NA": np.nan})
    #     return pd.to_numeric(cleaned, errors="coerce")

    def _parse_currency_to_float(self, s):
        ser = pd.Series(s)
        cleaned = (
            ser.astype(str)
            .str.replace(r"[\$,]", "", regex=True)
            .str.replace('"', "", regex=False)
            .str.strip()
        )
        # map empty-ish tokens to NaN without replace downcasting warnings
        cleaned = cleaned.where(~cleaned.isin(["", "nan", "None", "NA"]), np.nan)
        return pd.to_numeric(cleaned, errors="coerce")

    # def _parse_numeric_to_float(self, s: Union[pd.Series, np.ndarray]) -> pd.Series:
    #     ser = pd.Series(s)
    #     cleaned = (
    #         ser.astype(str)
    #         .str.replace(r"[,\$]", "", regex=True)
    #         .str.replace('"', "", regex=False)
    #         .str.replace(" ", "", regex=False)
    #     )
    #     cleaned = cleaned.replace({"": np.nan, "nan": np.nan, "None": np.nan, "NA": np.nan})
    #     return pd.to_numeric(cleaned, errors="coerce")

    def _parse_numeric_to_float(self, s):
        ser = pd.Series(s)
        cleaned = (
            ser.astype(str)
            .str.replace(r"[,\$]", "", regex=True)
            .str.replace('"', "", regex=False)
            .str.strip()
        )
        cleaned = cleaned.where(~cleaned.isin(["", "nan", "None", "NA"]), np.nan)
        return pd.to_numeric(cleaned, errors="coerce")

    # def _parse_date(self, s: Union[pd.Series, np.ndarray]) -> pd.Series:
    #     ser = pd.Series(s)
    #     # NYC sales often MM/DD/YYYY; also handle ISO
    #     dt = pd.to_datetime(ser, errors="coerce", infer_datetime_format=True)
    #     # Keep as datetime64[ns]
    #     return dt

    def _parse_date(self, s):
        ser = pd.Series(s)
        return pd.to_datetime(ser, errors="coerce")
    
    def _ensure_df(self, X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Please pass a pandas DataFrame.")
        return X.copy()

    def _coalesce_first_present(self, d: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
        for c in candidates:
            if c in d.columns:
                return d[c]
        return None

    def _coalesce_common_duplicates(self, d: pd.DataFrame) -> pd.DataFrame:
        df = d.copy()

        # If both "borough" and "borough__dup*" exist, keep the one with fewer missing
        # (same idea for latitude/longitude/yearbuilt, etc.)
        def best_of(prefix: str) -> Optional[str]:
            cols = [c for c in df.columns if c == prefix or c.startswith(prefix + "__dup")]
            if not cols:
                return None
            miss = [(c, df[c].isna().mean()) for c in cols]
            miss.sort(key=lambda x: x[1])
            return miss[0][0]

        for base in ["borough", "latitude", "longitude", "yearbuilt", "zip_code", "postcode"]:
            best = best_of(base)
            if best is not None and best != base and base in df.columns:
                # coalesce into base
                df[base] = df[base].where(df[base].notna(), df[best])
                # drop the other if it was only a duplicate
                # (keep best if it's base already)
        return df


# ------------------------------------------------------------------------------
# Convenience subclasses
# ------------------------------------------------------------------------------
class NYCLinearPreprocessor(NYCMassAppraisalPreprocessor):
    def __init__(self, **kwargs):
        super().__init__(mode="linear", output=kwargs.pop("output", "sparse"), **kwargs)


class NYCLGBMPreprocessor(NYCMassAppraisalPreprocessor):
    def __init__(self, **kwargs):
        super().__init__(mode="lgbm", output="dataframe", **kwargs)
