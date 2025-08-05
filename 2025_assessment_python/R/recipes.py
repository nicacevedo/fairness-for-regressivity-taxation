# Translation of code in: https://github.com/ccao-data/model-res-avm/blob/master/R/recipes.R
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline



from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


# Equivalent of: model_main_recipe
def model_main_pipeline(data, pred_vars, cat_vars, id_vars):
    """
    data: pd.DataFrame
    pred_vars: list of predictor column names
    cat_vars: list of categorical column names (subset of pred_vars)
    id_vars: list of ID column names
    """
    # --- Step 1: Define roles manually ---
    y = data['meta_sale_price']
    X = data[pred_vars].copy()
    IDs = data[id_vars]
    # --- Step 2: Drop unwanted columns ---
    # Drop 'time_split' if present
    if 'time_split' in data.columns:
        X = X.drop(columns=['time_split'], errors='ignore')
    # --- Step 3: Handle categorical vars ---
    # Fill NAs with 'unknown' for categorical variables
    for col in cat_vars:
        if col in X.columns:
            if pd.api.types.is_categorical_dtype(X[col]):
                if 'unknown' not in X[col].cat.categories:
                    X[col] = X[col].cat.add_categories('unknown')
            X[col] = X[col].fillna('unknown')

    # Ordinal encode categorical variables with 0-based integers
    ordinal_encoder = OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=-1,  # substitute for 'new' levels
        dtype=int
    )
    # --- Step 4: Create column transformer ---
    transformer = ColumnTransformer(
        transformers=[
            ('cat', ordinal_encoder, cat_vars),
            # Pass-through for remaining predictors
        ],
        remainder='passthrough'  # leave non-cat vars untouched
    )
    # --- Step 5: Wrap in a pipeline (no model yet) ---
    preprocessing_pipeline = Pipeline(steps=[
        ('transform', transformer)
    ])

    return preprocessing_pipeline, X, y, IDs


# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler, PowerTransformer

def model_lin_pipeline(data, pred_vars, cat_vars, id_vars, keep_cols=[]):
    # roles
    y = data['meta_sale_price']
    X = data[pred_vars].copy()
    IDs = data[id_vars].copy()

    # remove variables not outcome/predictor/ID
    X = X.drop(columns=['time_split'], errors='ignore')

    # drop extra location predictors that aren't school district
    drop_cols = [c for c in X.columns if c.startswith('loc_') and not c.startswith('loc_school_') and not pd.api.types.is_numeric_dtype(X[c])]
    X = X.drop(columns=drop_cols)

    # convert logicals to numerics and fix zeros
    if 'char_bldg_sf' in X.columns:
        X['char_bldg_sf'] = X['char_bldg_sf'].replace(0, 1)
    mutate_cols = [c for c in X.columns if c in ['char_recent_renovation','time_sale_post_covid'] or c.startswith('ind_') or c.startswith('ccao_is')]
    for c in mutate_cols:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors='coerce')

    # impute numeric (median) and nominal (mode), excluding IDs
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) and c not in id_vars]
    for c in num_cols:
        X[c] = X[c].fillna(X[c].median())
    nom_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c]) and c not in id_vars]
    for c in nom_cols:
        mode_val = X[c].mode(dropna=True)
        mode_val = mode_val.iloc[0] if len(mode_val) else 'unknown'
        X[c] = X[c].fillna(mode_val)

    # replace novel levels -> handled downstream; replace NA in factors with "unknown"
    for c in nom_cols:
        X[c] = X[c].astype(str).fillna('unknown')

    # linear encodings for specified high-cardinality nominals (target mean encoding)
    lencode_cols = ['meta_nbhd_code','meta_township_code','char_class'] + [c for c in X.columns if c.startswith('loc_school_')]
    global_mean = y.mean()
    for c in [c for c in lencode_cols if c in X.columns]:
        if not pd.api.types.is_numeric_dtype(X[c]):
            m = data.groupby(c)['meta_sale_price'].mean()
            X[c] = X[c].map(m).fillna(global_mean)
        else:
            X[c] = pd.to_numeric(X[c], errors='coerce').fillna(global_mean)

    # one-hot encode remaining nominal predictors (exclude lencode and IDs)
    ohe_cols = [c for c in nom_cols if c in X.columns and c not in lencode_cols and c not in id_vars]
    if ohe_cols:
        X = pd.get_dummies(X, columns=ohe_cols, drop_first=False)

    # small positive offsets
    if 'prox_nearest_vacant_land_dist_ft' in X.columns:
        X['prox_nearest_vacant_land_dist_ft_1'] = X['prox_nearest_vacant_land_dist_ft'] + 0.001
    if 'prox_nearest_new_construction_dist_ft' in X.columns:
        X['prox_nearest_new_construction_dist_ft_1'] = X['prox_nearest_new_construction_dist_ft'] + 0.001
    if 'acs5_percent_employment_unemployed' in X.columns:
        X['acs5_percent_employment_unemployed_1'] = X['acs5_percent_employment_unemployed'] + 0.001

    # BoxCox transforms (only strictly positive)
    boxcox_cols = [
        'acs5_median_income_per_capita_past_year',
        'acs5_median_income_household_past_year',
        'char_bldg_sf','char_land_sf',
        'prox_nearest_vacant_land_dist_ft_1',
        'prox_nearest_new_construction_dist_ft_1',
        'acs5_percent_employment_unemployed_1',
        'acs5_median_household_renter_occupied_gross_rent'
    ]
    pt = PowerTransformer(method='box-cox', standardize=False)
    for c in [c for c in boxcox_cols if c in X.columns]:
        pos = X[c] > 0
        if pos.any():
            X.loc[pos, c] = pt.fit_transform(X.loc[pos, [c]]).ravel()

    # winsorize
    for c in ['char_land_sf','char_bldg_sf']:
        if c in X.columns:
            q1, q99 = X[c].quantile(0.01), X[c].quantile(0.99)
            X[c] = X[c].clip(q1, q99)

    # polynomial features degree=2
    for c in ['char_yrblt','char_bldg_sf','char_land_sf']:
        if c in X.columns:
            X[f'{c}^2'] = X[c] ** 2

    # normalize numeric predictors (broad sets), excluding IDs
    norm_cols = [c for c in X.columns if (
        c.startswith('meta_nbhd_code') or c.startswith('meta_township_code') or c.startswith('char_class') or
        c.startswith('char_yrblt') or c.startswith('char_bldg_sf') or c.startswith('char_land_sf') or
        c.startswith('loc_') or c.startswith('prox_') or c.startswith('shp_') or
        c.startswith('acs5_') or c.startswith('other_')
    ) and c not in id_vars and pd.api.types.is_numeric_dtype(X[c])]
    if norm_cols:
        scaler = StandardScaler()
        X[norm_cols] = scaler.fit_transform(X[norm_cols])

    # near-zero variance removal
    nzv = [c for c in X.columns if X[c].nunique(dropna=False) <= 1 and c not in keep_cols]
    X = X.drop(columns=nzv)

    return X, y, IDs
