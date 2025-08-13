import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from balancing_models import BalancingResampler

# =============================================================================
# Custom Transformer Classes (Updated for Compatibility)
# =============================================================================

class InitialColumnDropper(BaseEstimator, TransformerMixin):
    """Drops initial unnecessary columns."""
    def __init__(self):
        self.drop_cols_ = []

    def fit(self, X, y=None):
        loc_drop_cols = [
            c for c in X.columns
            if c.startswith('loc_')
            and not c.startswith('loc_school_')
            and not pd.api.types.is_numeric_dtype(X[c])
        ]
        other_drop_cols = ['time_split']
        self.drop_cols_ = [c for c in loc_drop_cols + other_drop_cols if c in X.columns]
        return self

    def transform(self, X):
        X_transformed = X.drop(columns=self.drop_cols_)
        print("Step 1: Initial columns dropped.")
        return X_transformed

class TypeConverter(BaseEstimator, TransformerMixin):
    """Converts types and handles specific zero-value replacements."""
    def __init__(self):
        self.mutate_cols_ = []

    def fit(self, X, y=None):
        if 'char_bldg_sf' in X.columns:
            # This modification should happen inside transform to avoid changing the original data during fit
            pass
        self.mutate_cols_ = [
            c for c in X.columns if
            c in ['char_recent_renovation', 'time_sale_post_covid']
            or c.startswith('ind_')
            or c.startswith('ccao_is')
        ]
        return self

    def transform(self, X):
        X_transformed = X.copy()
        if 'char_bldg_sf' in X_transformed.columns:
            X_transformed['char_bldg_sf'] = X_transformed['char_bldg_sf'].replace(0, 1)
        for c in self.mutate_cols_:
            if c in X_transformed.columns:
                X_transformed[c] = pd.to_numeric(X_transformed[c], errors='coerce')
        print("Step 2: Types converted.")
        return X_transformed

class DataFrameImputer(BaseEstimator, TransformerMixin):
    """Imputes missing values while preserving the DataFrame structure."""
    def __init__(self, id_vars=[]):
        self.id_vars = id_vars
        self.imputers_ = {}
        self.num_cols_ = []
        self.nom_cols_ = []

    def fit(self, X, y=None):
        self.num_cols_ = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) and c not in self.id_vars]
        self.nom_cols_ = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c]) and c not in self.id_vars]
        
        for col in self.num_cols_:
            imputer = SimpleImputer(strategy='median')
            self.imputers_[col] = imputer.fit(X[[col]])
        for col in self.nom_cols_:
            imputer = SimpleImputer(strategy='most_frequent')
            self.imputers_[col] = imputer.fit(X[[col]])
        print("Step 3: Imputers learned.")
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col, imputer in self.imputers_.items():
            # .ravel() flattens the 2D output of .transform to a 1D array,
            # ensuring compatibility when assigning back to the DataFrame column.
            X_transformed[col] = imputer.transform(X_transformed[[col]]).ravel()
        print("Step 3: Imputation applied.")
        return X_transformed

class TargetEncoder(BaseEstimator, TransformerMixin):
    """Performs target encoding for high-cardinality categorical features."""
    def __init__(self, cols_to_encode):
        self.cols_to_encode = cols_to_encode
        self.encoding_maps_ = {}
        self.global_mean_ = 0

    def fit(self, X, y):
        data = X.copy()
        data['target'] = y
        self.global_mean_ = y.mean()
        for col in self.cols_to_encode:
            if col in data.columns and not pd.api.types.is_numeric_dtype(data[col]):
                mapping = data.groupby(col)['target'].mean()
                self.encoding_maps_[col] = mapping
        print("Step 4: Target encoding maps learned.")
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col, mapping in self.encoding_maps_.items():
            X_transformed[col] = X_transformed[col].map(mapping).fillna(self.global_mean_)
        for col in self.cols_to_encode:
            if col in X_transformed.columns and not pd.api.types.is_numeric_dtype(X_transformed[col]):
                 X_transformed[col] = pd.to_numeric(X_transformed[col], errors='coerce').fillna(self.global_mean_)
        print("Step 4: Target encoding applied.")
        return X_transformed

class NovelCategoryHandler(BaseEstimator, TransformerMixin):
    """
    Identifies categories not seen during training and maps them to 'unknown'.
    """
    def __init__(self, nom_cols=[]):
        self.nom_cols = nom_cols
        self.categories_ = {}

    def fit(self, X, y=None):
        for col in self.nom_cols:
            if col in X.columns:
                self.categories_[col] = set(X[col].dropna().unique())
        print("Step 5: Novel category handler learned known categories.")
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col, known_cats in self.categories_.items():
            # Replace values that are not in known_cats and are not NaN
            X_transformed[col] = X_transformed[col].apply(
                lambda x: x if pd.isna(x) or x in known_cats else 'unknown'
            )
        print("Step 5: Novel categories mapped to 'unknown'.")
        return X_transformed

class DataFrameOneHotEncoder(BaseEstimator, TransformerMixin):
    """Performs one-hot encoding and returns a DataFrame."""
    def __init__(self, ohe_cols, id_vars):
        self.ohe_cols = ohe_cols
        self.id_vars = id_vars
        self.ohe_ = None
        self.ohe_cols_to_use_ = []

    def fit(self, X, y=None):
        self.ohe_cols_to_use_ = [c for c in self.ohe_cols if c in X.columns]
        if self.ohe_cols_to_use_:
            # handle_unknown='ignore' will create all-zero columns for categories
            # not seen during fit, which is what we want for the 'unknown' category.
            self.ohe_ = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            self.ohe_.fit(X[self.ohe_cols_to_use_])
        print("Step 6: One-hot encoder learned.")
        return self

    def transform(self, X):
        if not self.ohe_cols_to_use_:
            return X
        
        X_transformed = X.drop(columns=self.ohe_cols_to_use_)
        
        ohe_data = self.ohe_.transform(X[self.ohe_cols_to_use_])
        ohe_df = pd.DataFrame(ohe_data, columns=self.ohe_.get_feature_names_out(self.ohe_cols_to_use_), index=X.index)
        
        print("Step 6: One-hot encoding applied.")
        return pd.concat([X_transformed, ohe_df], axis=1)

class Winsorizer(BaseEstimator, TransformerMixin):
    """
    Caps outliers using quantiles (Winsorizing).
    """
    def __init__(self, winsorize_cols):
        self.winsorize_cols = winsorize_cols
        self.quantiles_ = {}

    def fit(self, X, y=None):
        for col in self.winsorize_cols:
            if col in X.columns:
                q1 = X[col].quantile(0.01)
                q99 = X[col].quantile(0.99)
                self.quantiles_[col] = (q1, q99)
        print("Step 7: Winsorizer learned quantiles.")
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col, (q1, q99) in self.quantiles_.items():
            if col in X_transformed.columns:
                X_transformed[col] = X_transformed[col].clip(q1, q99)
        print("Step 7: Winsorizing applied.")
        return X_transformed

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Performs feature engineering (offsets, Box-Cox, polynomials)."""
    def __init__(self, offset_cols, boxcox_cols, poly_cols):
        self.offset_cols = offset_cols
        self.boxcox_cols = boxcox_cols
        self.poly_cols = poly_cols
        self.power_transformers_ = {}

    def fit(self, X, y=None):
        X_fit = X.copy()
        for col in self.offset_cols:
            if col in X_fit.columns:
                X_fit[f'{col}_1'] = X_fit[col] + 0.001
        for col in self.boxcox_cols:
            target_col = f'{col}_1' if col in self.offset_cols else col
            if target_col in X_fit.columns:
                pos_vals = X_fit[target_col] > 0
                if pos_vals.any():
                    pt = PowerTransformer(method='box-cox', standardize=False)
                    pt.fit(X_fit.loc[pos_vals, [target_col]])
                    self.power_transformers_[target_col] = pt
        print("Step 8: Feature engineering parameters learned.")
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in self.offset_cols:
            if col in X_transformed.columns:
                X_transformed[f'{col}_1'] = X_transformed[col] + 0.001
        for col, pt in self.power_transformers_.items():
            if col in X_transformed.columns:
                pos_mask = X_transformed[col] > 0
                if pos_mask.any():
                    X_transformed.loc[pos_mask, col] = pt.transform(X_transformed.loc[pos_mask, [col]]).ravel()
        for col in self.poly_cols:
            if col in X_transformed.columns:
                X_transformed[f'{col}^2'] = X_transformed[col] ** 2
        print("Step 8: Feature engineering applied.")
        return X_transformed
        
class DataFrameScaler(BaseEstimator, TransformerMixin):
    """Scales numeric columns and returns a DataFrame."""
    def __init__(self, norm_cols, id_vars):
        self.norm_cols = norm_cols
        self.id_vars = id_vars
        self.scaler_ = None
        self.cols_to_scale_ = []

    def fit(self, X, y=None):
        # Determine columns to scale based on the provided patterns
        self.cols_to_scale_ = [
            c for c in X.columns if (
                any(c.startswith(p) for p in self.norm_cols)
            ) and c not in self.id_vars and pd.api.types.is_numeric_dtype(X[c])
        ]
        if self.cols_to_scale_:
            self.scaler_ = StandardScaler()
            self.scaler_.fit(X[self.cols_to_scale_])
        print("Step 9: Scaler learned.")
        return self

    def transform(self, X):
        if not self.cols_to_scale_:
            return X
        X_transformed = X.copy()
        X_transformed[self.cols_to_scale_] = self.scaler_.transform(X[self.cols_to_scale_])
        print("Step 9: Scaling applied.")
        return X_transformed

class NearZeroVarianceRemover(BaseEstimator, TransformerMixin):
    """Removes columns with near-zero variance."""
    def __init__(self, keep_cols=[]):
        self.keep_cols = keep_cols
        self.nzv_cols_ = []

    def fit(self, X, y=None):
        self.nzv_cols_ = [c for c in X.columns if X[c].nunique(dropna=False) <= 1 and c not in self.keep_cols]
        print("Step 10: Near-zero variance columns identified.")
        return self

    def transform(self, X):
        X_transformed = X.drop(columns=self.nzv_cols_)
        print("Step 10: Near-zero variance columns removed.")
        return X_transformed

# =============================================================================
# Pipeline Assembly Function
# =============================================================================

def build_model_pipeline(pred_vars, cat_vars, id_vars, keep_cols=[]):
    """Assembles the full preprocessing pipeline."""
    lencode_cols = ['meta_nbhd_code', 'meta_township_code', 'char_class'] + [c for c in pred_vars if c.startswith('loc_school_')]
    ohe_cols = [c for c in cat_vars if c not in lencode_cols]
    
    offset_cols = ['prox_nearest_vacant_land_dist_ft', 'prox_nearest_new_construction_dist_ft', 'acs5_percent_employment_unemployed']
    boxcox_cols = ['acs5_median_income_per_capita_past_year', 'acs5_median_income_household_past_year', 'char_bldg_sf', 'char_land_sf', 'prox_nearest_vacant_land_dist_ft', 'prox_nearest_new_construction_dist_ft', 'acs5_percent_employment_unemployed', 'acs5_median_household_renter_occupied_gross_rent']
    winsorize_cols = ['char_land_sf', 'char_bldg_sf']
    poly_cols = ['char_yrblt', 'char_bldg_sf', 'char_land_sf']
    norm_cols_prefixes = ['meta_nbhd_code', 'meta_township_code', 'char_class', 'char_yrblt', 'char_bldg_sf', 'char_land_sf', 'loc_', 'prox_', 'shp_', 'acs5_', 'other_']

    pipeline = Pipeline([
        ('1_initial_drop', InitialColumnDropper()),
        ('2_type_conversion', TypeConverter()),
        ('3_imputation', DataFrameImputer(id_vars=id_vars)),
        ('4_target_encode', TargetEncoder(cols_to_encode=lencode_cols)),
        ('5_handle_novel_cats', NovelCategoryHandler(nom_cols=ohe_cols)),
        ('6_one_hot_encode', DataFrameOneHotEncoder(ohe_cols=ohe_cols, id_vars=id_vars)),
        ('7_winsorize', Winsorizer(winsorize_cols=winsorize_cols)),
        ('8_feature_engineer', FeatureEngineer(offset_cols=offset_cols, boxcox_cols=boxcox_cols, poly_cols=poly_cols)),
        ('9_normalize', DataFrameScaler(norm_cols=norm_cols_prefixes, id_vars=id_vars)),
        ('10_nzv_removal', NearZeroVarianceRemover(keep_cols=keep_cols))
    ])
    
    return pipeline

def build_model_pipeline_supress_onehot(pred_vars, cat_vars, id_vars, keep_cols=[]):
# def build_model_pipeline(pred_vars, cat_vars, id_vars, keep_cols=[]):
    """Assembles the full preprocessing pipeline."""
    lencode_cols = ['meta_nbhd_code', 'meta_township_code', 'char_class'] + [c for c in pred_vars if c.startswith('loc_school_')]
    ohe_cols = [c for c in cat_vars if c not in lencode_cols]
    
    offset_cols = ['prox_nearest_vacant_land_dist_ft', 'prox_nearest_new_construction_dist_ft', 'acs5_percent_employment_unemployed']
    boxcox_cols = ['acs5_median_income_per_capita_past_year', 'acs5_median_income_household_past_year', 'char_bldg_sf', 'char_land_sf', 'prox_nearest_vacant_land_dist_ft', 'prox_nearest_new_construction_dist_ft', 'acs5_percent_employment_unemployed', 'acs5_median_household_renter_occupied_gross_rent']
    winsorize_cols = ['char_land_sf', 'char_bldg_sf']
    poly_cols = ['char_yrblt', 'char_bldg_sf', 'char_land_sf']
    norm_cols_prefixes = ['meta_nbhd_code', 'meta_township_code', 'char_class', 'char_yrblt', 'char_bldg_sf', 'char_land_sf', 'loc_', 'prox_', 'shp_', 'acs5_', 'other_']

    pipeline = Pipeline([
        ('1_initial_drop', InitialColumnDropper()),
        ('2_type_conversion', TypeConverter()),
        ('3_imputation', DataFrameImputer(id_vars=id_vars)),
        # ('4_target_encode', TargetEncoder(cols_to_encode=lencode_cols)),
        ('5_handle_novel_cats', NovelCategoryHandler(nom_cols=ohe_cols)),
        ('6_one_hot_encode', DataFrameOneHotEncoder(ohe_cols=ohe_cols, id_vars=id_vars)),
        ('7_winsorize', Winsorizer(winsorize_cols=winsorize_cols)),
        ('8_feature_engineer', FeatureEngineer(offset_cols=offset_cols, boxcox_cols=boxcox_cols, poly_cols=poly_cols)),
        ('9_normalize', DataFrameScaler(norm_cols=norm_cols_prefixes, id_vars=id_vars)),
        ('10_nzv_removal', NearZeroVarianceRemover(keep_cols=keep_cols))
    ])
    
    return pipeline

