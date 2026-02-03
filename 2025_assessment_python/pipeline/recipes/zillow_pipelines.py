import inspect
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


class AmesHousingPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocessor for Kaggle Ames dataset with two modes:
      - mode="linear": sklearn ColumnTransformer (impute/log/scale/encode)
      - mode="lgbm"  : returns pandas DataFrame with categorical dtypes for LightGBM

    Fixes:
      - Feature engineering applied in fit() and transform()
      - LotFrontage imputed using train-learned Neighborhood medians (no leakage)
      - In lgbm mode: converts object columns to pandas 'category' and aligns categories train/test
    """

    def __init__(
        self,
        mode: str = "linear",
        output: str = "sparse",
        log_skewed_numeric: bool = True,
        skew_threshold: float = 0.75,
        onehot_min_frequency: Optional[float] = None,
        drop_id: bool = True,
        add_mosold_cyc: bool = True,
    ):
        if mode not in {"linear", "lgbm"}:
            raise ValueError("mode must be 'linear' or 'lgbm'")
        if output not in {"sparse", "array", "dataframe"}:
            raise ValueError("output must be 'sparse', 'array', or 'dataframe'")

        self.mode = mode
        self.output = output
        self.log_skewed_numeric = log_skewed_numeric
        self.skew_threshold = float(skew_threshold)
        self.onehot_min_frequency = onehot_min_frequency
        self.drop_id = drop_id
        self.add_mosold_cyc = add_mosold_cyc

        # NA-means-absent columns
        self.none_cols = [
            "Alley",
            "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
            "FireplaceQu",
            "GarageType", "GarageFinish", "GarageQual", "GarageCond",
            "PoolQC", "Fence", "MiscFeature",
            "MasVnrType",
        ]
        self.zero_cols = [
            "MasVnrArea",
            "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
            "BsmtFullBath", "BsmtHalfBath",
            "GarageCars", "GarageArea",
        ]

        # ordinal definition
        self.ordinal_mapping_: Dict[str, List[str]] = self._default_ordinal_mapping()

        # learned artifacts
        self.lotfrontage_median_by_neighborhood_: Optional[pd.Series] = None
        self.lotfrontage_global_median_: Optional[float] = None

        self.column_transformer_: Optional[ColumnTransformer] = None
        self.numeric_cols_: Optional[List[str]] = None
        self.ordinal_cols_: Optional[List[str]] = None
        self.nominal_cols_: Optional[List[str]] = None
        self.log1p_cols_: Optional[List[str]] = None
        self.feature_names_: Optional[np.ndarray] = None

        # lgbm categorical alignment
        self.lgbm_feature_cols_: Optional[List[str]] = None
        self.lgbm_cat_cols_: Optional[List[str]] = None
        self.lgbm_cat_levels_: Optional[Dict[str, List[str]]] = None  # column -> categories list
        self.lgbm_ord_cols_: Optional[List[str]] = None

    # -------------------------
    # sklearn API
    # -------------------------
    def fit(self, X: pd.DataFrame, y=None):
        df = self._ensure_df(X)

        if "SalePrice" in df.columns:
            df = df.drop(columns=["SalePrice"])
        if self.drop_id and "Id" in df.columns:
            df = df.drop(columns=["Id"])

        # learn LotFrontage medians from TRAIN
        if "LotFrontage" in df.columns and "Neighborhood" in df.columns:
            self.lotfrontage_median_by_neighborhood_ = df.groupby("Neighborhood")["LotFrontage"].median()
            self.lotfrontage_global_median_ = float(df["LotFrontage"].median())
        else:
            self.lotfrontage_median_by_neighborhood_ = None
            self.lotfrontage_global_median_ = None

        # cleanup + FE (IMPORTANT: in fit too)
        df = self._prepare_features(df, is_fit=True)

        # ordinals present
        self.ordinal_cols_ = [c for c in self.ordinal_mapping_.keys() if c in df.columns]

        # all object cols not ordinal -> nominal
        obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
        self.nominal_cols_ = [c for c in obj_cols if c not in self.ordinal_cols_]

        # numeric
        numeric_all = df.select_dtypes(include=["number", "bool"]).columns.tolist()
        self.numeric_cols_ = numeric_all[:]  # ordinals are object/category, so not here

        # lgbm mode: store category levels + cast to category
        if self.mode == "lgbm":
            df = self._fit_lgbm_categories(df)
            self.lgbm_feature_cols_ = df.columns.tolist()
            return self

        # linear mode: choose log columns
        self.log1p_cols_ = []
        if self.log_skewed_numeric and self.numeric_cols_:
            self.log1p_cols_ = self._select_log1p_cols(df)

        self.column_transformer_ = self._build_linear_column_transformer()
        self.column_transformer_.fit(df)

        self.feature_names_ = self._safe_get_feature_names()
        return self

    def transform(self, X: pd.DataFrame):
        if self.mode == "linear" and self.column_transformer_ is None:
            raise RuntimeError("Not fitted. Call fit() first.")
        if self.mode == "lgbm" and self.lgbm_feature_cols_ is None:
            raise RuntimeError("Not fitted. Call fit() first.")

        df = self._ensure_df(X)

        if "SalePrice" in df.columns:
            df = df.drop(columns=["SalePrice"])
        if self.drop_id and "Id" in df.columns:
            df = df.drop(columns=["Id"])

        df = self._prepare_features(df, is_fit=False)

        if self.mode == "lgbm":
            df = self._apply_lgbm_categories(df)

            # ensure column order
            for c in self.lgbm_feature_cols_:
                if c not in df.columns:
                    df[c] = np.nan
            return df[self.lgbm_feature_cols_]

        # linear: apply log1p before transformer
        if self.log1p_cols_:
            for c in self.log1p_cols_:
                if c in df.columns:
                    df[c] = np.log1p(df[c].astype(float))

        X_out = self.column_transformer_.transform(df)

        if self.output == "sparse":
            return X_out
        if self.output == "array":
            return np.asarray(X_out)

        # dataframe output (densifies)
        names = self._safe_get_feature_names()
        return pd.DataFrame(np.asarray(X_out), columns=names)

    # -------------------------
    # Target helpers
    # -------------------------
    def process_target(self, y) -> np.ndarray:
        return np.log1p(np.asarray(y, dtype=float))

    def inverse_target(self, y_pred) -> np.ndarray:
        return np.expm1(np.asarray(y_pred, dtype=float))

    # -------------------------
    # Internals
    # -------------------------
    def _ensure_df(self, X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Pass a pandas DataFrame (Ames train/test).")
        return X.copy()

    def _default_ordinal_mapping(self) -> Dict[str, List[str]]:
        return {
            # quality
            "ExterQual":   ["None", "Po", "Fa", "TA", "Gd", "Ex"],
            "ExterCond":   ["None", "Po", "Fa", "TA", "Gd", "Ex"],
            "BsmtQual":    ["None", "Po", "Fa", "TA", "Gd", "Ex"],
            "BsmtCond":    ["None", "Po", "Fa", "TA", "Gd", "Ex"],
            "HeatingQC":   ["None", "Po", "Fa", "TA", "Gd", "Ex"],
            "KitchenQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
            "FireplaceQu": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
            "GarageQual":  ["None", "Po", "Fa", "TA", "Gd", "Ex"],
            "GarageCond":  ["None", "Po", "Fa", "TA", "Gd", "Ex"],
            "PoolQC":      ["None", "Fa", "TA", "Gd", "Ex"],

            # basement exposure
            "BsmtExposure": ["None", "No", "Mn", "Av", "Gd"],

            # basement finish
            "BsmtFinType1": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
            "BsmtFinType2": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],

            # functional
            "Functional": ["None", "Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],

            # garage finish
            "GarageFinish": ["None", "Unf", "RFn", "Fin"],

            # fence
            "Fence": ["None", "MnWw", "GdWo", "MnPrv", "GdPrv"],

            # paved drive
            "PavedDrive": ["None", "N", "P", "Y"],
        }

    def _prepare_features(self, df: pd.DataFrame, is_fit: bool) -> pd.DataFrame:
        df = df.copy()

        # MSSubClass categorical-ish
        if "MSSubClass" in df.columns:
            df["MSSubClass"] = df["MSSubClass"].astype(str)

        # NA means None
        for c in self.none_cols:
            if c in df.columns:
                df[c] = df[c].fillna("None")

        # NA means 0
        for c in self.zero_cols:
            if c in df.columns:
                df[c] = df[c].fillna(0)

        # LotFrontage: use train-learned neighborhood median
        if "LotFrontage" in df.columns:
            df["LotFrontage"] = self._impute_lotfrontage(df)

        # feature engineering
        df = self._feature_engineering(df)

        # cyc MoSold
        if self.add_mosold_cyc and "MoSold" in df.columns:
            m = df["MoSold"].astype(float)
            df["MoSold_sin"] = np.sin(2 * np.pi * (m / 12.0))
            df["MoSold_cos"] = np.cos(2 * np.pi * (m / 12.0))

        return df

    def _impute_lotfrontage(self, df: pd.DataFrame) -> pd.Series:
        s = df["LotFrontage"]
        if s.isna().sum() == 0:
            return s

        if self.lotfrontage_median_by_neighborhood_ is None or "Neighborhood" not in df.columns:
            fill_val = self.lotfrontage_global_median_
            if fill_val is None:
                fill_val = float(s.median())
            return s.fillna(fill_val)

        mapped = df["Neighborhood"].map(self.lotfrontage_median_by_neighborhood_)
        fill_val = self.lotfrontage_global_median_
        if fill_val is None:
            fill_val = float(s.median())
        return s.fillna(mapped).fillna(fill_val)

    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        def col(name: str):
            return df[name] if name in df.columns else pd.Series(0, index=df.index)

        df["TotalSF"] = col("TotalBsmtSF") + col("1stFlrSF") + col("2ndFlrSF")
        df["TotalBsmtFinSF"] = col("BsmtFinSF1") + col("BsmtFinSF2")
        df["TotalPorchSF"] = col("OpenPorchSF") + col("EnclosedPorch") + col("3SsnPorch") + col("ScreenPorch")

        df["TotalBath"] = (
            col("FullBath")
            + 0.5 * col("HalfBath")
            + col("BsmtFullBath")
            + 0.5 * col("BsmtHalfBath")
        )

        if "YrSold" in df.columns and "YearBuilt" in df.columns:
            df["AgeAtSale"] = (df["YrSold"] - df["YearBuilt"]).clip(lower=0)
        if "YrSold" in df.columns and "YearRemodAdd" in df.columns:
            df["YearsSinceRemodel"] = (df["YrSold"] - df["YearRemodAdd"]).clip(lower=0)

        if "YrSold" in df.columns and "GarageYrBlt" in df.columns:
            g = df["GarageYrBlt"].replace(0, np.nan)
            df["GarageAge"] = (df["YrSold"] - g).fillna(0).clip(lower=0)

        if "PoolArea" in df.columns:
            df["HasPool"] = (df["PoolArea"].astype(float) > 0).astype(int)
        if "2ndFlrSF" in df.columns:
            df["Has2ndFloor"] = (df["2ndFlrSF"].astype(float) > 0).astype(int)
        if "GarageArea" in df.columns:
            df["HasGarage"] = (df["GarageArea"].astype(float) > 0).astype(int)
        if "TotalBsmtSF" in df.columns:
            df["HasBsmt"] = (df["TotalBsmtSF"].astype(float) > 0).astype(int)
        if "Fireplaces" in df.columns:
            df["HasFireplace"] = (df["Fireplaces"].astype(float) > 0).astype(int)

        return df

    # ---------- LGBM categorical handling ----------
    def _fit_lgbm_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Casts object columns to pandas 'category' and stores category levels
        so we can enforce the same categories at transform() time.
        """
        df = df.copy()
        self.lgbm_cat_levels_ = {}

        ord_cols = self.ordinal_cols_ or []
        self.lgbm_ord_cols_ = ord_cols[:]

        # 1) Ordinal: fixed, ordered categories from mapping
        for c in ord_cols:
            # ensure present token used by mapping
            df[c] = df[c].fillna("None")
            cats = self.ordinal_mapping_[c]
            self.lgbm_cat_levels_[c] = list(cats)
            df[c] = pd.Categorical(df[c], categories=cats, ordered=True)

        # 2) Nominal: learn categories from train + include "Missing"
        cat_cols = []
        for c in (self.nominal_cols_ or []):
            df[c] = df[c].astype("object")
            df[c] = df[c].fillna("Missing")
            # stable list of observed categories (keep order of appearance)
            seen = pd.Index(pd.unique(df[c].astype(str)))
            if "Missing" not in seen:
                seen = pd.Index(["Missing"]).append(seen)
            cats = list(seen)
            self.lgbm_cat_levels_[c] = cats
            df[c] = pd.Categorical(df[c].astype(str), categories=cats, ordered=False)
            cat_cols.append(c)

        self.lgbm_cat_cols_ = sorted(set(cat_cols + ord_cols))
        return df

    def _apply_lgbm_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies stored category levels to transform data, mapping unknowns to 'Missing'.
        """
        df = df.copy()
        if not self.lgbm_cat_levels_:
            # Shouldn't happen if fit() ran correctly
            return df

        for c, cats in self.lgbm_cat_levels_.items():
            if c not in df.columns:
                continue

            if c in (self.lgbm_ord_cols_ or []):
                # ordinal uses 'None'
                df[c] = df[c].fillna("None").astype(str)
                # map unknown -> 'None' (safer than NaN for ordinals)
                df[c] = np.where(pd.Series(df[c]).isin(cats), df[c], "None")
                df[c] = pd.Categorical(df[c], categories=cats, ordered=True)
            else:
                # nominal uses 'Missing'
                df[c] = df[c].fillna("Missing").astype(str)
                df[c] = np.where(pd.Series(df[c]).isin(cats), df[c], "Missing")
                df[c] = pd.Categorical(df[c], categories=cats, ordered=False)

        return df

    # ---------- Linear mode bits ----------
    def _select_log1p_cols(self, df: pd.DataFrame) -> List[str]:
        no_log = {
            "YrSold", "MoSold",
            "YearBuilt", "YearRemodAdd", "GarageYrBlt",
            "AgeAtSale", "YearsSinceRemodel", "GarageAge",
            "OverallQual", "OverallCond",
        }
        cols = []
        for c in (self.numeric_cols_ or []):
            if c in no_log or c not in df.columns:
                continue
            s = df[c]
            if (s.dropna() < 0).any():
                continue
            if s.dropna().nunique() <= 20:
                continue
            sk = float(s.dropna().skew()) if s.dropna().shape[0] else 0.0
            if abs(sk) > self.skew_threshold:
                cols.append(c)
        return cols

    def _build_linear_column_transformer(self) -> ColumnTransformer:
        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
            ]
        )

        ord_cols = self.ordinal_cols_ or []
        ord_categories = [self.ordinal_mapping_[c] for c in ord_cols]
        ord_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
                (
                    "encoder",
                    OrdinalEncoder(
                        categories=ord_categories,
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                ),
                ("scaler", StandardScaler()),
            ]
        )

        onehot = self._make_onehot_encoder()
        nom_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                ("onehot", onehot),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("num", num_pipe, self.numeric_cols_ or []),
                ("ord", ord_pipe, ord_cols),
                ("nom", nom_pipe, self.nominal_cols_ or []),
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
