"""
nyc_mass_appraisal_preprocessing.py

Preprocessing-only utilities for NYC sales + PLUTO-style dataframes.

What you get:
  - make_y(df): builds y from SALE PRICE (log1p or none).
  - make_linear_preprocessor(): DataFrame -> sparse matrix (one-hot for categoricals).
  - make_lgbm_preprocessor(): DataFrame -> DataFrame (categoricals kept as pandas 'category').

Important practice note (recommended):
  - To avoid leakage, split FIRST (e.g., by SALE DATE), then fit the preprocessors on train
    and transform train/test separately.
  - You SHOULD keep df["SALE DATE"] in your raw dataframe for splitting later.
  - These preprocessors do NOT mutate your original df; they work on a copy.

Design choices tailored to your case:
  - Fixes OneHotEncoder errors from mixed types by coercing categoricals to strings.
  - Drops columns that are entirely missing at fit time (prevents sklearn imputer shape drift).
  - Optionally auto-drops admin "*date" columns (except SALE DATE) and "geom"/"notes".

Python: 3.9+
sklearn: compatible with both OneHotEncoder(sparse_output=...) and older sparse=...
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import inspect
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


# ----------------------------
# Defaults (override as needed)
# ----------------------------

DEFAULT_TARGET_COL = "SALE PRICE"
DEFAULT_SALE_DATE_COL = "SALE DATE"

DEFAULT_DROP_COLS: Tuple[str, ...] = (
    # target (pipelines shouldn't use it as X)
    "SALE PRICE",
    # high-cardinality IDs (memorization risk)
    "BBL",
    "BIN",
    "BLOCK",
    "LOT",
    # raw address-like text
    "ADDRESS",
    "APARTMENT NUMBER",
    "EASE-MENT",
    # often troublesome / not useful for baseline benchmarks
    "geom",
    "notes",
)

DEFAULT_CATEGORICAL_COLS: Tuple[str, ...] = (
    "BOROUGH",
    "NEIGHBORHOOD",
    "BUILDING CLASS CATEGORY",
    "BUILDING CLASS AS OF FINAL ROLL",
    "BUILDING CLASS AT TIME OF SALE",
    "TAX CLASS AS OF FINAL ROLL",
    "TAX CLASS AT TIME OF SALE",
    "ZIP CODE",
    "Community Board",
    "Council District",
    "Census Tract 2020",
    "Neighborhood Tabulation Area (NTA) (2020)",
)


# ----------------------------
# Helpers
# ----------------------------

def _make_onehot() -> OneHotEncoder:
    """Back/forward compatible OneHotEncoder constructor."""
    sig = inspect.signature(OneHotEncoder)
    if "sparse_output" in sig.parameters:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    return OneHotEncoder(handle_unknown="ignore", sparse=True)


def _infer_num_cat_columns(df: pd.DataFrame, categorical_cols: Sequence[str]) -> Tuple[List[str], List[str]]:
    """
    Infer numeric and categorical column lists.
    - Anything in categorical_cols is treated as categorical (if present).
    - Any object/category dtype is categorical.
    - Numeric dtypes not in the categorical list are numeric.
    """
    cat = [c for c in categorical_cols if c in df.columns]

    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c]):
            if c not in cat:
                cat.append(c)

    num = [c for c in df.columns if c not in cat and pd.api.types.is_numeric_dtype(df[c])]
    return num, cat


def _has_any_numeric(df: pd.DataFrame, col: str) -> bool:
    s = pd.to_numeric(df[col], errors="coerce")
    return bool(s.notna().any())


def _auto_drop_admin_cols(
    df: pd.DataFrame,
    *,
    keep: Sequence[str] = (DEFAULT_SALE_DATE_COL,),
    drop_geom: bool = True,
    drop_notes: bool = True,
    drop_date_suffix: bool = True,
) -> List[str]:
    """
    Drops admin columns commonly present in PLUTO extracts:
      - 'geom'
      - 'notes'
      - any column whose name ends with 'date' or '_date' (except SALE DATE),
        e.g., basempdate, rpaddate, zoningdate, etc.
    """
    keep_set = set(keep)
    out: List[str] = []

    for c in df.columns:
        if c in keep_set:
            continue
        cl = c.lower()

        if drop_geom and cl == "geom":
            out.append(c)
            continue
        if drop_notes and cl == "notes":
            out.append(c)
            continue
        if drop_date_suffix and (cl.endswith("date") or cl.endswith("_date")):
            out.append(c)
            continue

    return out


# ----------------------------
# Target preparation
# ----------------------------

def make_y(
    df: pd.DataFrame,
    *,
    target_col: str = DEFAULT_TARGET_COL,
    transform: str = "log1p",
) -> np.ndarray:
    """
    Build y from df[target_col].

    transform:
      - "log1p": y = log(1 + price)
      - "none":  y = price
    """
    if target_col not in df.columns:
        raise KeyError(f"target_col='{target_col}' not found in DataFrame columns.")

    y_raw = pd.to_numeric(df[target_col], errors="coerce").astype(float).values
    if transform == "log1p":
        return np.log1p(y_raw)
    if transform == "none":
        return y_raw
    raise ValueError("transform must be one of {'log1p', 'none'}.")


# ----------------------------
# Feature engineering
# ----------------------------

class NYCFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Adds a small, standard set of appraisal features if columns exist:
      - log1p(LAND SQUARE FEET), log1p(GROSS SQUARE FEET)
      - age = sale_year - YEAR BUILT
      - sale_year, sale_month

    IMPORTANT:
      - Keeping SALE DATE in your raw df is recommended for later splitting.
      - This transformer can optionally drop raw SALE DATE from *feature output* so it
        doesn't become a model feature.
    """

    def __init__(
        self,
        *,
        sale_date_col: str = DEFAULT_SALE_DATE_COL,
        year_built_col: str = "YEAR BUILT",
        land_sqft_col: str = "LAND SQUARE FEET",
        gross_sqft_col: str = "GROSS SQUARE FEET",
        add_logs: bool = True,
        add_age: bool = True,
        add_sale_year_month: bool = True,
        drop_raw_sale_date_in_features: bool = True,
    ) -> None:
        self.sale_date_col = sale_date_col
        self.year_built_col = year_built_col
        self.land_sqft_col = land_sqft_col
        self.gross_sqft_col = gross_sqft_col
        self.add_logs = add_logs
        self.add_age = add_age
        self.add_sale_year_month = add_sale_year_month
        self.drop_raw_sale_date_in_features = drop_raw_sale_date_in_features

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("NYCFeatureEngineer expects a pandas DataFrame.")
        df = X.copy()

        if self.sale_date_col in df.columns:
            df[self.sale_date_col] = pd.to_datetime(df[self.sale_date_col], errors="coerce")

        if self.add_logs:
            if self.land_sqft_col in df.columns:
                land = pd.to_numeric(df[self.land_sqft_col], errors="coerce").clip(lower=0)
                df["log_land_sqft"] = np.log1p(land)
            if self.gross_sqft_col in df.columns:
                gross = pd.to_numeric(df[self.gross_sqft_col], errors="coerce").clip(lower=0)
                df["log_gross_sqft"] = np.log1p(gross)

        if self.add_sale_year_month and self.sale_date_col in df.columns:
            df["sale_year"] = df[self.sale_date_col].dt.year
            df["sale_month"] = df[self.sale_date_col].dt.month

        if self.add_age and (self.sale_date_col in df.columns) and (self.year_built_col in df.columns):
            sale_year = df[self.sale_date_col].dt.year
            year_built = pd.to_numeric(df[self.year_built_col], errors="coerce")
            age = sale_year - year_built
            df["age"] = age.where(age.notna(), np.nan).clip(lower=0, upper=300)

        if self.drop_raw_sale_date_in_features and self.sale_date_col in df.columns:
            df = df.drop(columns=[self.sale_date_col])

        return df


# ----------------------------
# Linear preprocessor (OneHot + scaling)
# ----------------------------

class LinearPreprocessor(BaseEstimator, TransformerMixin):
    """
    DataFrame -> sparse design matrix suitable for linear models:
      - feature engineering
      - drop columns
      - optional auto-drop (geom/notes/*date except SALE DATE)
      - drop all-missing columns learned at fit time (prevents sklearn imputer warnings/drift)
      - numeric: median impute + indicator + scale
      - categorical: most_frequent impute + cast-to-str + one-hot
    """

    def __init__(
        self,
        *,
        categorical_cols: Sequence[str] = DEFAULT_CATEGORICAL_COLS,
        drop_cols: Sequence[str] = DEFAULT_DROP_COLS,
        sale_date_col: str = DEFAULT_SALE_DATE_COL,
        add_feature_engineering: bool = True,
        drop_raw_sale_date_in_features: bool = True,
        auto_drop_admin_dates: bool = True,
        drop_all_missing_cols: bool = True,
    ) -> None:
        self.categorical_cols = list(categorical_cols)
        self.drop_cols = list(drop_cols)
        self.sale_date_col = sale_date_col
        self.add_feature_engineering = add_feature_engineering
        self.drop_raw_sale_date_in_features = drop_raw_sale_date_in_features
        self.auto_drop_admin_dates = auto_drop_admin_dates
        self.drop_all_missing_cols = drop_all_missing_cols

        self._fe: Optional[NYCFeatureEngineer] = None
        self._num_cols: List[str] = []
        self._cat_cols: List[str] = []
        self._dropped_all_missing: List[str] = []
        self._ct: Optional[ColumnTransformer] = None

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("LinearPreprocessor expects a pandas DataFrame.")

        df = X.copy()

        if self.add_feature_engineering:
            self._fe = NYCFeatureEngineer(
                sale_date_col=self.sale_date_col,
                drop_raw_sale_date_in_features=self.drop_raw_sale_date_in_features,
            )
            df = self._fe.fit_transform(df)

        # Drop explicit drop_cols
        df = df.drop(columns=[c for c in self.drop_cols if c in df.columns], errors="ignore")

        # Optional: auto-drop admin columns (helps PLUTO extracts)
        if self.auto_drop_admin_dates:
            auto = _auto_drop_admin_cols(df, keep=())
            if auto:
                df = df.drop(columns=auto, errors="ignore")

        num_cols, cat_cols = _infer_num_cat_columns(df, categorical_cols=self.categorical_cols)

        # Drop all-missing columns at fit time
        dropped_all_missing: List[str] = []
        if self.drop_all_missing_cols:
            keep_num = []
            for c in num_cols:
                if _has_any_numeric(df, c):
                    keep_num.append(c)
                else:
                    dropped_all_missing.append(c)

            keep_cat = []
            for c in cat_cols:
                if df[c].notna().any():
                    keep_cat.append(c)
                else:
                    dropped_all_missing.append(c)

            num_cols, cat_cols = keep_num, keep_cat

        self._num_cols = num_cols
        self._cat_cols = cat_cols
        self._dropped_all_missing = dropped_all_missing

        num_pipe = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler(with_mean=False)),
            ]
        )

        cat_pipe = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                # CRITICAL FIX: OneHotEncoder requires uniform dtype (avoid float/str mix)
                ("to_str", FunctionTransformer(lambda a: a.astype(str), feature_names_out="one-to-one")),
                ("onehot", _make_onehot()),
            ]
        )

        self._ct = ColumnTransformer(
            transformers=[
                ("num", num_pipe, self._num_cols),
                ("cat", cat_pipe, self._cat_cols),
            ],
            remainder="drop",
            sparse_threshold=0.3,
        )
        self._ct.fit(df)
        return self

    def transform(self, X: pd.DataFrame):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("LinearPreprocessor expects a pandas DataFrame.")

        df = X.copy()

        if self.add_feature_engineering and self._fe is not None:
            df = self._fe.transform(df)

        df = df.drop(columns=[c for c in self.drop_cols if c in df.columns], errors="ignore")

        if self.auto_drop_admin_dates:
            auto = _auto_drop_admin_cols(df, keep=())
            if auto:
                df = df.drop(columns=auto, errors="ignore")

        if self._dropped_all_missing:
            df = df.drop(columns=[c for c in self._dropped_all_missing if c in df.columns], errors="ignore")

        # Ensure expected cols exist
        for c in self._num_cols:
            if c not in df.columns:
                df[c] = np.nan
        for c in self._cat_cols:
            if c not in df.columns:
                df[c] = np.nan

        return self._ct.transform(df)

    def get_feature_names_out(self):
        if self._ct is None:
            return None
        try:
            return self._ct.get_feature_names_out()
        except Exception:
            return None


def make_linear_preprocessor(
    *,
    categorical_cols: Sequence[str] = DEFAULT_CATEGORICAL_COLS,
    drop_cols: Sequence[str] = DEFAULT_DROP_COLS,
    sale_date_col: str = DEFAULT_SALE_DATE_COL,
    add_feature_engineering: bool = True,
    drop_raw_sale_date_in_features: bool = True,
    auto_drop_admin_dates: bool = True,
    drop_all_missing_cols: bool = True,
) -> LinearPreprocessor:
    return LinearPreprocessor(
        categorical_cols=categorical_cols,
        drop_cols=drop_cols,
        sale_date_col=sale_date_col,
        add_feature_engineering=add_feature_engineering,
        drop_raw_sale_date_in_features=drop_raw_sale_date_in_features,
        auto_drop_admin_dates=auto_drop_admin_dates,
        drop_all_missing_cols=drop_all_missing_cols,
    )


# ----------------------------
# LightGBM preprocessor (native categorical)
# ----------------------------

class LGBMPreprocessor(BaseEstimator, TransformerMixin):
    """
    DataFrame -> pandas DataFrame for LightGBM:
      - feature engineering
      - drop columns
      - optional auto-drop (geom/notes/*date except SALE DATE)
      - drop all-missing columns learned at fit time (prevents imputer column dropping)
      - numeric: median impute (after coercing to numeric)
      - categorical: impute + cast to pandas 'category' (as strings for consistency)

    Returns a DataFrame with a stable set of columns after fit/transform.
    """

    def __init__(
        self,
        *,
        categorical_cols: Sequence[str] = DEFAULT_CATEGORICAL_COLS,
        drop_cols: Sequence[str] = DEFAULT_DROP_COLS,
        sale_date_col: str = DEFAULT_SALE_DATE_COL,
        add_feature_engineering: bool = True,
        drop_raw_sale_date_in_features: bool = True,
        auto_drop_admin_dates: bool = True,
        drop_all_missing_cols: bool = True,
        cat_impute_strategy: str = "most_frequent",  # or "constant"
        cat_constant_value: str = "MISSING",
    ) -> None:
        self.categorical_cols = list(categorical_cols)
        self.drop_cols = list(drop_cols)
        self.sale_date_col = sale_date_col
        self.add_feature_engineering = add_feature_engineering
        self.drop_raw_sale_date_in_features = drop_raw_sale_date_in_features
        self.auto_drop_admin_dates = auto_drop_admin_dates
        self.drop_all_missing_cols = drop_all_missing_cols
        self.cat_impute_strategy = cat_impute_strategy
        self.cat_constant_value = cat_constant_value

        self._fe: Optional[NYCFeatureEngineer] = None
        self._num_cols: List[str] = []
        self._cat_cols: List[str] = []
        self._dropped_all_missing: List[str] = []

        self._num_imputer: Optional[SimpleImputer] = None
        self._cat_imputer: Optional[SimpleImputer] = None

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("LGBMPreprocessor expects a pandas DataFrame.")

        df = X.copy()

        if self.add_feature_engineering:
            self._fe = NYCFeatureEngineer(
                sale_date_col=self.sale_date_col,
                drop_raw_sale_date_in_features=self.drop_raw_sale_date_in_features,
            )
            df = self._fe.fit_transform(df)

        df = df.drop(columns=[c for c in self.drop_cols if c in df.columns], errors="ignore")

        if self.auto_drop_admin_dates:
            auto = _auto_drop_admin_cols(df, keep=())
            if auto:
                df = df.drop(columns=auto, errors="ignore")

        num_cols, cat_cols = _infer_num_cat_columns(df, categorical_cols=self.categorical_cols)

        dropped_all_missing: List[str] = []
        if self.drop_all_missing_cols:
            keep_num = []
            for c in num_cols:
                if _has_any_numeric(df, c):
                    keep_num.append(c)
                else:
                    dropped_all_missing.append(c)

            keep_cat = []
            for c in cat_cols:
                if df[c].notna().any():
                    keep_cat.append(c)
                else:
                    dropped_all_missing.append(c)

            num_cols, cat_cols = keep_num, keep_cat

        self._num_cols = num_cols
        self._cat_cols = cat_cols
        self._dropped_all_missing = dropped_all_missing

        self._num_imputer = SimpleImputer(strategy="median")
        if self._num_cols:
            self._num_imputer.fit(df[self._num_cols].apply(pd.to_numeric, errors="coerce"))

        if self.cat_impute_strategy == "constant":
            self._cat_imputer = SimpleImputer(strategy="constant", fill_value=self.cat_constant_value)
        else:
            self._cat_imputer = SimpleImputer(strategy="most_frequent")

        if self._cat_cols:
            self._cat_imputer.fit(df[self._cat_cols])

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("LGBMPreprocessor expects a pandas DataFrame.")

        df = X.copy()

        if self.add_feature_engineering and self._fe is not None:
            df = self._fe.transform(df)

        df = df.drop(columns=[c for c in self.drop_cols if c in df.columns], errors="ignore")

        if self.auto_drop_admin_dates:
            auto = _auto_drop_admin_cols(df, keep=())
            if auto:
                df = df.drop(columns=auto, errors="ignore")

        if self._dropped_all_missing:
            df = df.drop(columns=[c for c in self._dropped_all_missing if c in df.columns], errors="ignore")

        # Ensure expected cols exist
        for c in self._num_cols:
            if c not in df.columns:
                df[c] = np.nan
        for c in self._cat_cols:
            if c not in df.columns:
                df[c] = np.nan

        # Numeric: coerce + impute
        if self._num_cols:
            num_in = df[self._num_cols].apply(pd.to_numeric, errors="coerce")
            num_out = self._num_imputer.transform(num_in)
            df.loc[:, self._num_cols] = num_out

        # Categorical: impute + cast to category (as strings for consistency)
        if self._cat_cols:
            cat_out = self._cat_imputer.transform(df[self._cat_cols])
            cat_df = pd.DataFrame(cat_out, columns=self._cat_cols, index=df.index)
            for c in self._cat_cols:
                df[c] = cat_df[c].astype(str).astype("category")

        return df


def make_lgbm_preprocessor(
    *,
    categorical_cols: Sequence[str] = DEFAULT_CATEGORICAL_COLS,
    drop_cols: Sequence[str] = DEFAULT_DROP_COLS,
    sale_date_col: str = DEFAULT_SALE_DATE_COL,
    add_feature_engineering: bool = True,
    drop_raw_sale_date_in_features: bool = True,
    auto_drop_admin_dates: bool = True,
    drop_all_missing_cols: bool = True,
    cat_impute_strategy: str = "most_frequent",
    cat_constant_value: str = "MISSING",
) -> LGBMPreprocessor:
    return LGBMPreprocessor(
        categorical_cols=categorical_cols,
        drop_cols=drop_cols,
        sale_date_col=sale_date_col,
        add_feature_engineering=add_feature_engineering,
        drop_raw_sale_date_in_features=drop_raw_sale_date_in_features,
        auto_drop_admin_dates=auto_drop_admin_dates,
        drop_all_missing_cols=drop_all_missing_cols,
        cat_impute_strategy=cat_impute_strategy,
        cat_constant_value=cat_constant_value,
    )


# ----------------------------
# Optional: config container
# ----------------------------

@dataclass(frozen=True)
class PreprocessConfig:
    categorical_cols: Tuple[str, ...] = DEFAULT_CATEGORICAL_COLS
    drop_cols: Tuple[str, ...] = DEFAULT_DROP_COLS
    sale_date_col: str = DEFAULT_SALE_DATE_COL


# ----------------------------
# Example usage (no split, no model)
# ----------------------------
if __name__ == "__main__":
    # df_final = ...  # load elsewhere
    #
    # Keep SALE DATE in df_final for splitting later:
    # sale_date = pd.to_datetime(df_final["SALE DATE"], errors="coerce")
    #
    # y = make_y(df_final, transform="log1p")
    # lin_pre = make_linear_preprocessor()
    # X_lin = lin_pre.fit_transform(df_final)   # sparse matrix
    #
    # lgbm_pre = make_lgbm_preprocessor()
    # X_lgbm = lgbm_pre.fit_transform(df_final) # pandas DataFrame
    #
    # print(y.shape, X_lin.shape, X_lgbm.shape)
    pass
