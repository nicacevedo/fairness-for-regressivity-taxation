# Translation of code in: https://github.com/ccao-data/model-res-avm/blob/master/R/recipes.R
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline



from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold

from category_encoders import TargetEncoder


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
    # MINE: missing " # Replace novel levels with "new" (to include in .fit ) "
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

    # Mine: apply the pipeline
    X = pd.DataFrame(
        preprocessing_pipeline.fit_transform(X),
        columns=preprocessing_pipeline.get_feature_names_out(),
        index=X.index
    )

    # Mine 2: replace the extra "x__" on the names of the columns
    X.columns = [col.replace("cat__","").replace("remainder__","") for col in X.columns] 

    # Mine 3: drop the ID columns (unnecesary bc of the definition of X)
    return preprocessing_pipeline, X, y, IDs


# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler, PowerTransformer

def my_model_lin_pipeline(data, pred_vars, cat_vars, id_vars):
    # Remove any variables not an outcome var or in the pred_vars vector
    objective = data["meta_sale_price"]
    data = data.drop(columns=["time_split"], errors="ignore")
    data = data[pred_vars + cat_vars + id_vars]
    # Drop extra location predictors that aren't school district
    data = data.drop(columns=[
            col for col in data.columns
            if col.startswith('loc_')
            and not (col in pred_vars and pd.api.types.is_numeric_dtype(data[col])) # Corrected line
            and not col.startswith('loc_school_')
        ]
    )
    # MINE: update pred vars and remove duplicated columns
    pred_vars = list(set([c for c in pred_vars if c in data.columns])) # update pred_vars (after dropping)
    data = data.loc[:, ~data.T.duplicated()]
    # Convert logical values to numerics and get rid of 0s
    data.loc[data['char_bldg_sf'] == 0, 'char_bldg_sf'] = 1
    cols_to_convert = ['char_recent_renovation', 'time_sale_post_covid'] + [
        col for col in data.columns if col.startswith('ind_') or col.startswith('ccao_is')
    ]
    data[list(set(cols_to_convert))] = data[list(set(cols_to_convert))].apply(pd.to_numeric)
    # Fill missing values with the median/mode (MINE: why not cat_vars?)
    numeric_predictors = list(set(data[pred_vars].select_dtypes(include='number').columns))
    data.fillna(
        data[[col for col in numeric_predictors if col not in id_vars]].median(), 
        inplace=True
    )
    nominal_predictors = list(set(data[pred_vars].select_dtypes(exclude='number').columns))
    data.fillna(
        data[[col for col in nominal_predictors if col not in id_vars]].mode().iloc[0], 
        inplace=True
    )


    # ==============
    # Replace novel levels with "new"
    # TO BE DONE: Create a new category names "new" when new test data is inputed. R code: # Replace novel levels with "new" step_novel(all_nominal_predictors(), -has_role("ID")) %>%
    # TO BE DONE: Replace NA in factors with "unknown": This is also for new data. See below the proposed code: 
    # # 1. Identify non-numeric predictor columns to process, excluding ID columns
    # # nominal_predictors = data[pred_vars].select_dtypes(exclude='number').columns
    # cols_to_process = [col for col in nominal_predictors if col not in id_vars]
    # # 2. Fill NaN values in that subset of columns with the string 'unknown'
    # data[cols_to_process] = data[cols_to_process].fillna('unknown')
    # ==============

    # Create linear encodings for certain high cardinality nominal predictors
    cols_to_encode = list(set(['meta_nbhd_code', 'meta_township_code', 'char_class'] + [
        col for col in data.columns if col.startswith('loc_school_')
    ]))
    data =  TargetEncoder(cols=cols_to_encode).fit_transform(data, objective)

    # Dummify (OHE) any remaining nominal predictors
    nominal_predictors = list(set(data[pred_vars].select_dtypes(exclude='number').columns))
    cols_to_dummify = [
        col for col in nominal_predictors if col not in cols_to_encode
    ]
    data = pd.get_dummies(data, columns=cols_to_dummify, dtype=int, drop_first=False) # one_hot = TRUE: means real one-hot-encoder

    # Normalize/transform skewed numeric predictors. Add a small fudge factor
    # so no values are zero
    cols_to_transform = [
        'prox_nearest_vacant_land_dist_ft',
        'prox_nearest_new_construction_dist_ft',
        'acs5_percent_employment_unemployed'
    ]
    for col in cols_to_transform:
        data[f'{col}_1'] = data[col] + 0.001    
    cols_to_transform = [
        'acs5_median_income_per_capita_past_year',
        'acs5_median_income_household_past_year',
        'char_bldg_sf', 'char_land_sf',
        'prox_nearest_vacant_land_dist_ft_1',
        'prox_nearest_new_construction_dist_ft_1',
        'acs5_percent_employment_unemployed_1',
        'acs5_median_household_renter_occupied_gross_rent'
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ('boxcox', PowerTransformer(method='box-cox'), cols_to_transform)
        ],
        remainder='passthrough'
    )
    data = pd.DataFrame(
        preprocessor.fit_transform(data),
        columns=preprocessor.get_feature_names_out(),
        index=data.index
    )
    # MINE: recover the original names:
    data.columns = [col.replace("boxcox__", "").replace("remainder__", "") for col in data.columns]

    # Winsorize some extreme values in important numeric vars.
    cols_to_winsorize = ['char_land_sf', 'char_bldg_sf']
    winsor_limits = {}
    for col in cols_to_winsorize:
        lower_limit = data[col].quantile(0.01)
        upper_limit = data[col].quantile(0.99)
        winsor_limits[col] = (lower_limit, upper_limit)
    for col, limits in winsor_limits.items():
        lower_limit, upper_limit = limits
        data[col] = data[col].clip(lower=lower_limit, upper=upper_limit)
    # MINE (comment):  it splitted the two loops because both are used in train, but the second only on testing

    # ====
    # MINE (uncommented part): Polynomial transformation. THIS HAS TO BE FIXED: I have empty columns and columns that changed.
    # cols_to_transform = ['char_yrblt', 'char_bldg_sf', 'char_land_sf']
    # preprocessor = ColumnTransformer(
    #     transformers=[
    #         ('poly', PolynomialFeatures(degree=2, include_bias=False), cols_to_transform)
    #     ],
    #     remainder='passthrough'
    # )
    # # 3. Fit and transform the data
    # data = pd.DataFrame(
    #     preprocessor.fit_transform(data),
    #     columns=preprocessor.get_feature_names_out(),
    #     index=data.index
    # )
    # MINE: recover the original names:
    # data.columns = [col.replace("poly__", "").replace("remainder__", "") for col in data.columns]
    # OPTION 2:
    cols_to_square = ['char_yrblt', 'char_bldg_sf', 'char_land_sf']
    # 2. Loop through the list and create a new squared column for each
    for col in cols_to_square:
        # Create a new column with the "_poly_2" suffix
        data[f'{col}_poly_2'] = data[col] ** 2
    # ====

    # Normalize basically all numeric predictors
    cols_to_normalize = ['meta_nbhd_code', 'meta_township_code', 'char_class']
    prefixes_to_normalize = [
        'char_yrblt', 'char_bldg_sf', 'char_land_sf', 'loc_', 'prox_',
        'shp_', 'acs5_', 'other_'
    ]
    for col in data.columns:
        if any(col.startswith(prefix) for prefix in prefixes_to_normalize):
            cols_to_normalize.append(col)
    cols_to_normalize = [
        col for col in list(set(cols_to_normalize)) if col not in id_vars   # Ensure the list is unique and doesn't contain ID columns
    ]
    preprocessor = ColumnTransformer(
        transformers=[
            ('scale', StandardScaler(), cols_to_normalize)
        ],
        remainder='passthrough'
    )
    data = pd.DataFrame(
        preprocessor.fit_transform(data),
        columns=preprocessor.get_feature_names_out(),
        index=data.index
    )

    # MINE: recover the original names:
    data.columns = [col.replace("poly__", "").replace("remainder__", "") for col in data.columns]

    # MINE (comment): Zero variance columns dropped
    # print("The prev data: \n", data)
    # selector = VarianceThreshold(threshold=0)
    # data = pd.DataFrame(
    #     selector.fit_transform(data),
    #     columns=selector.get_feature_names_out(),
    #     index=data.index
    # )

    # print("modified: \n", data)
    # print("dummied cols: \n", data.loc[data["prox_nearest_vacant_land_dist_ft"] < 0.01, "prox_nearest_vacant_land_dist_ft"])
    
    


    # SECOND VERSION
    # MINE: update pred vars and remove duplicated columns
    pred_vars = list(set([c for c in pred_vars if c in data.columns])) # update pred_vars (after dropping)
    print("The prev data: \n", data)
    # Define the thresholds (these are common defaults)
    freq_cutoff = 95 / 5  # Ratio of most to second most frequent value
    unique_cutoff = 10    # Percentage of unique values out of total samples

    # List to hold columns to remove
    cols_to_drop = []

    for col in pred_vars:
        # Skip non-numeric if they slip in, or handle separately
        if data[col].dtype not in ['int64', 'float64']:
            continue
            
        counts = data[col].value_counts()
        
        # Check if there is only one unique value (zero variance)
        if len(counts) == 1:
            cols_to_drop.append(col)
            continue

        # Calculate frequency ratio and percent unique
        freq_ratio = counts.iloc[0] / counts.iloc[1]
        percent_unique = data[col].nunique() / len(data) * 100
        
        # If it meets the NZV criteria, add to the drop list
        if freq_ratio > freq_cutoff and percent_unique < unique_cutoff:
            cols_to_drop.append(col)

    # Drop the identified columns from the DataFrame
    df_filtered = data.drop(columns=cols_to_drop)

    print("modified: \n", data)

    print("Columns removed for Near-Zero Variance:", cols_to_drop)

    # END SECOND VERSION

    print("NA's in final verison")
    na_values = data.drop(columns=id_vars, errors="ignore").isna().sum()
    print(na_values.loc[na_values > 0])

    
    return data.drop(columns=id_vars, errors="ignore"), objective, None#data[list(id_vars)]


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
