# Analogous of .R script in: https://github.com/ccao-data/model-res-avm/blob/master/pipeline/01-train.R 


#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 1. Setup ---------------------------------------------------------------------
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Pre-imports
import os
import sys
# Get the absolute path of the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the parent directory (project/)
# Go up one level from 'subfolder' to 'project'
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
granparent_dir = os.path.dirname(os.path.abspath(parent_dir))
granparent_dir = os.path.abspath(os.path.join(parent_dir, granparent_dir))
print("parent", parent_dir)
print("g parent", granparent_dir)
# Add the parent directory to sys.path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    sys.path.append(granparent_dir)

# Real imports 
import pandas as pd
import numpy as np
import yaml
from math import log2, floor
from time import time



# Models 
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import optuna
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# === My imports ==
# 1. Models
from src.nn_unconstrained import FeedForwardNNRegressor, FeedForwardNNRegressorWithEmbeddings
from src.nn_constrained_cpu_v2 import FeedForwardNNRegressorWithProjection # FeedForwardNNRegressorWithConstraints, 
from src.nn_constrained_cpu_v3 import ConstrainedRegressorProjectedWithEmbeddings

# 2. Pipelines
from R.recipes import model_main_pipeline, model_lin_pipeline, my_model_lin_pipeline
from src.util_functions import compute_haihao_F_metrics
from recipes.recipes_pipelined import build_model_pipeline, build_model_pipeline_supress_onehot, ModelMainRecipe, ModelMainRecipeImputer

# 3. Preprocessors
from balancing_models import BalancingResampler

# 4. Cross Validation
from temporal_bayes_cv import ModelHandler, TemporalCV#TemporalBayesCV, ModelSpec
from optuna.pruners import SuccessiveHalvingPruner



# Load YAML params file
with open('params.yaml', 'r') as file:
    params = yaml.safe_load(file)

# Inputs
assessment_year = 2025

use_sample = True
sample_size = 100000 # SAMPLE SIZE

apply_resampling = False

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 2. Prepare Data --------------------------------------------------------------
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
print("Preparing model training data")


# Load the full set of training data, then arrange by sale date in order to
# facilitate out-of-time sampling/validation

# NOTE: It is critical to trim "multicard" sales when training. Multicard means
# there is multiple buildings on a PIN. Since these sales include multiple
# buildings, they are typically higher than a "normal" sale and must be removed
training_data_full = pd.read_parquet(f"input/training_data.parquet")
training_data_full = training_data_full[
    (~training_data_full['ind_pin_is_multicard'].astype('bool').fillna(True)) &
    (~training_data_full['sv_is_outlier'].astype('bool').fillna(True))
]

# assessment_data_full = pd.read_parquet(f"input/assessment_data.parquet")
# assessment_data_full = assessment_data_full[ # Unnecessary for testing (?)
#     (~assessment_data_full['ind_pin_is_multicard'].astype('bool').fillna(True)) &
#     (~assessment_data_full['sv_is_outlier'].astype('bool').fillna(True))
# ]

if use_sample:
    print("I am using a sample")
    training_data_full = training_data_full.sample(sample_size, random_state=42)
    # assessment_data_full = assessment_data_full.sample(sample_size, random_state=42)
else:
    print("I am using full data")

# Sorting by date of sell (ir order to split)
training_data_full = training_data_full.sort_values('meta_sale_date') # Sort by 'meta_sale_date' 
training_data_full['meta_sale_price'] = np.log(training_data_full['meta_sale_price']) # Log target
training_data_full["char_recent_renovation"] = training_data_full["char_recent_renovation"].astype(bool) # QUESTION: Why is this not in cat_vars?


# Create train/test split by time, with most recent observations in the test set
# We want our best model(s) to be predictive of the future, since properties are
# assessed on the basis of past sales
split_prop = params['cv']['split_prop']
split_index = int(len(training_data_full) * split_prop)
train = training_data_full.iloc[:split_index] # Split by time
test = training_data_full.iloc[split_index:]


# # TO REFINE: main pipeline
# # Create a recipe for the training data which removes non-predictor columns and
# # preps categorical data, see R/recipes.R for details
# train_pipeline, X, y, train_IDs = model_main_pipeline(
#     data=training_data_full,
#     pred_vars=params['model']['predictor']['all'],
#     cat_vars=params['model']['predictor']['categorical'],
#     id_vars=params['model']['predictor']['id']
# )
# # # Fit and transform training data (applies categorical processing, etc.)
# # X_transformed = pd.DataFrame(train_pipeline.fit_transform(X), index=X.index)




#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 3. Linear Model --------------------------------------------------------------
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
print("Creating and fitting linear baseline model")

# Create a linear model recipe with additional imputation, transformations,
# and feature interactions
# training_data_full = training_data_full.copy()

# Clean columns
train = train[params['model']['predictor']['all'] + params['model']['predictor']['id'] + ['meta_sale_price']]
# train = train[params['model']['predictor']['all'] + params['model']['predictor']['id'] + ['meta_sale_price'] + ["meta_sale_date"]] # Sale date for CV split (?)

test = test[params['model']['predictor']['all'] + params['model']['predictor']['id'] + ['meta_sale_price']]


# Split the data in X, y
X_train_prep, y_train_fit_log = train.drop(columns=['meta_sale_price']), train['meta_sale_price'] 
X_test_prep, y_test_fit_log = test.drop(columns=['meta_sale_price']), test['meta_sale_price']

# print("COLUMNS: ", X_train_prep.columns)

# ===== New pipeline version ====
model_lin_pipeline = build_model_pipeline(
    pred_vars=params['model']['predictor']['all'],
    cat_vars=params['model']['predictor']['categorical'],
    id_vars=params['model']['predictor']['id']
)
# model_emb_pipeline = build_model_pipeline_supress_onehot( # WARNING: We only changed to this to perform changes on the pipeline
#     pred_vars=params['model']['predictor']['all'],
#     cat_vars=params['model']['predictor']['categorical'],
#     id_vars=params['model']['predictor']['id']
# )
model_emb_pipeline = ModelMainRecipe(
# model_emb_pipeline = ModelMainRecipeImputer(
    outcome= "meta_sale_price",
    pred_vars=params['model']['predictor']['all'],
    cat_vars=params['model']['predictor']['categorical'],
    id_vars=params['model']['predictor']['id']
)




# 1. Fit / transform the data for linear models
X_train_fit_lin = model_lin_pipeline.fit_transform(X_train_prep, y_train_fit_log)#.drop(columns=params['model']['predictor']['id'])
# ===== Resampler =====
if apply_resampling:
    resampler_lin = BalancingResampler( 
        n_bins=100, binning_policy='uniform', max_diff_ratio=0.5,
        undersample_policy='random', oversample_policy='smoter',
        smote_k_neighbors=5, random_state=42
    )
    X_train_fit_lin, y_train_fit_log_lin = resampler_lin.fit_resample(X_train_fit_lin, y_train_fit_log) # only train is for resampling
else:
    y_train_fit_log_lin = y_train_fit_log
X_test_fit_lin = model_lin_pipeline.transform(X_test_prep)#.drop(columns=params['model']['predictor']['id'])


# 2. Fit / transform the data for models with embeddings
X_train_fit_emb = model_emb_pipeline.fit_transform(X_train_prep, y_train_fit_log).drop(columns=params['model']['predictor']['id'], errors="ignore")
# X_train_fit_emb["char_recent_renovation"] = X_train_fit_emb["char_recent_renovation"].astype(bool) # QUESTION: Why is this not in cat_vars?
# na_columns = X_train_fit_emb.isna().sum()[X_train_fit_emb.isna().sum() > 0].index
# X_train_fit_emb[na_columns] = X_train_fit_emb[na_columns].fillna(value="unknown")
# cat_cols_emb = [i for i,col in enumerate(X_train_fit_emb.columns) if X_train_fit_emb[col].dtype == object ]
# ===== Resampler =====
if apply_resampling:
    resampler_emb = BalancingResampler( 
        n_bins=100, binning_policy='uniform', max_diff_ratio=0.5,
        undersample_policy='random', oversample_policy='smotenc',
        smote_k_neighbors=5, random_state=42,
        categorical_features=cat_cols_emb
    )
    X_train_fit_emb, y_train_fit_log_emb = resampler_emb.fit_resample(X_train_fit_emb, y_train_fit_log) # only train is for resampling
else:
    y_train_fit_log_emb = y_train_fit_log
X_test_fit_emb = model_emb_pipeline.transform(X_test_prep).drop(columns=params['model']['predictor']['id'], errors="ignore")
# X_test_fit_emb["char_recent_renovation"] = X_test_fit_emb["char_recent_renovation"].astype(bool) # QUESTION: Why is this not in cat_vars?

# # 3. Pass any object feature to category for embedded models (Light GBM, NN w/ embedding, etc.)
# for col in X_train_fit_emb:
#     if X_train_fit_emb[col].dtype == object:
#         print("To category: ", col)
#         X_train_fit_emb[col] = X_train_fit_emb[col].astype("category")
# for col in X_test_fit_emb:
#     if X_test_fit_emb[col].dtype == object:
#         print("To category: ", col)
#         X_test_fit_emb[col] = X_test_fit_emb[col].astype("category")


# ==========================================================================================
#                       Comparisson of the different models
# ==========================================================================================

model_names = [
    # "LinearRegression", 
    # "FeedForwardNNRegressor", 
    "LightGBM", 
    # "FeedForwardNNRegressorWithEmbeddings", 
    # "FeedForwardNNRegressorWithConstraints"
]
emb_model_names = ["LightGBM", "FeedForwardNNRegressorWithEmbeddings", "FeedForwardNNRegressorWithConstraints"]
lin_model_names = ["LinearRegression", "FeedForwardNNRegressor" ]

for model_name in model_names:

    print("="*100)
    print("Fitting the model: ", model_name)
    print("="*100)

    if model_name == "LightGBM":

            
        stop_iter         = params['model']['parameter']['stop_iter']
        objective         = params['model']['objective']
        verbose           = params['model']['verbose']
        validation_prop   = params['model']['parameter']['validation_prop']   # e.g., 0.2
        validation_type   = params['model']['parameter']['validation_type']   # "random" | "recent"
        validation_metric = params['model']['parameter']['validation_metric'] # e.g., "rmse"
        link_max_depth    = params['model']['parameter']['link_max_depth']    # bool
        deterministic     = params['model']['deterministic']
        force_row_wise    = params['model']['force_row_wise']
        seed              = params['model']['seed']
        categorical_cols  = params['model']['predictor']['categorical']       # names (if you keep categorical dtype)

        params_dict = params["model"]["hyperparameter"]["default"]
        max_depth = floor(log2(params_dict["num_leaves"])) + params_dict['add_to_linked_depth']
        n_estimators_static = params['model']['hyperparameter']['default']['num_iterations']

        # # Print each variable
        # print(f"stop_iter: {stop_iter}")
        # print(f"objective: {objective}")
        # print(f"verbose: {verbose}")
        # print(f"validation_prop: {validation_prop}")
        # print(f"validation_type: {validation_type}")
        # print(f"validation_metric: {validation_metric}")
        # print(f"link_max_depth: {link_max_depth}")
        # print(f"deterministic: {deterministic}")
        # print(f"force_row_wise: {force_row_wise}")
        # print(f"seed: {seed}")
        # print(f"categorical_cols: {categorical_cols}")
        # print(f"params_dict: {params_dict}")
        # print(f"max_depth: {max_depth}")
        # print(f"n_estimators_static: {n_estimators_static}")

        # model = lgb.LGBMRegressor(
        #     n_estimators=n_estimators_static,

        #     # Determinism / engine controls
        #     random_state=seed,
        #     deterministic=deterministic,
        #     force_row_wise=force_row_wise,
        #     # n_jobs=num_threads, # Check the number CPU's first
        #     verbose=verbose,

        #     # Objective
        #     objective=objective,

        #     # Core complexity / tree params
        #     learning_rate=params_dict['learning_rate'],
        #     max_bin=int(params_dict['max_bin']),
        #     num_leaves=int(params_dict["num_leaves"]),
        #     feature_fraction=params_dict['feature_fraction'],
        #     min_gain_to_split=params_dict['min_gain_to_split'],
        #     min_data_in_leaf=int(params_dict['min_data_in_leaf']),
        #     max_depth=int(max_depth) if max_depth != -1 else -1,

        #     # Categorical-specific
        #     max_cat_threshold=int(params_dict['max_cat_threshold']),
        #     min_data_per_group=int(params_dict['min_data_per_group']),
        #     cat_smooth=params_dict['cat_smooth'],
        #     cat_l2=params_dict['cat_l2'],

        #     # Regularization
        #     reg_alpha=params_dict['lambda_l1'],
        #     reg_lambda=params_dict['lambda_l2'],

        #     # Trees (n_estimators) # (missing this information)
        #     # n_estimators=int(params_dict['n_estimators'])
        #     # if params_dict.get('n_estimators') is not None
        #     # else (int(params_dict["n_estimators_static"]) if params_dict["n_estimators_static"] is not None else 1000)
        # )    

        # # 500 on 10k
        # model = lgb.LGBMRegressor(cat_l2=67.06444168895015, cat_smooth=17.09275488174234,
        #       deterministic=True, feature_fraction=0.690780591986823,
        #       force_row_wise=True, learning_rate=0.021379249622581562,
        #       max_bin=423, max_cat_threshold=47, max_depth=11,
        #       min_data_in_leaf=2, min_data_per_group=32,
        #       min_gain_to_split=0.2564010374007819, n_estimators=2500,
        #       num_leaves=226, objective='rmse', random_state=2025,
        #       reg_alpha=6.489938652852805, reg_lambda=26.226042365885338,
        #       verbose=-1)

        # # 100 on 100k
        # model = lgb.LGBMRegressor(cat_l2=77.29638412450299, cat_smooth=106.04915982308384,
        #       deterministic=True, feature_fraction=0.6515890168955135,
        #       force_row_wise=True, learning_rate=0.01925779993720067,
        #       max_bin=372, max_cat_threshold=128, max_depth=14,
        #       min_data_in_leaf=164, min_data_per_group=215,
        #       min_gain_to_split=0.206029615115229, n_estimators=2500,
        #       num_leaves=1074, objective='rmse', random_state=2025,
        #       reg_alpha=29.6577825672396, reg_lambda=60.67539891939314,
        #       verbose=-1)

        # # 500 on 100k
        # model = lgb.LGBMRegressor(cat_l2=73.51736114647082, cat_smooth=32.89188632453103,
        #       deterministic=True, feature_fraction=0.562108593472178,
        #       force_row_wise=True, learning_rate=0.05027925110046119,
        #       max_bin=382, max_cat_threshold=58, max_depth=15,
        #       min_data_in_leaf=192, min_data_per_group=69,
        #       min_gain_to_split=0.08102102469611594, n_estimators=2500,
        #       num_leaves=1311, objective='rmse', random_state=2025,
        #       reg_alpha=7.462047791346489, reg_lambda=39.33483586225005,
        #       verbose=-1)

        # # 1000 on 100k (only train (?))
        # model = lgb.LGBMRegressor(cat_l2=22.95020213396379, cat_smooth=31.14663605311536,
        #       deterministic=True, feature_fraction=0.6572673015780892,
        #       force_row_wise=True, learning_rate=0.03341584496999874,
        #       max_bin=323, max_cat_threshold=82, max_depth=12,
        #       min_data_in_leaf=68, min_data_per_group=81,
        #       min_gain_to_split=0.12374932835073327, n_estimators=2500,
        #       num_leaves=366, objective='rmse', random_state=2025,
        #       reg_alpha=0.06676779724096571, reg_lambda=30.039145583263345,
        #       verbose=-1)

        # 2000 on 10k (only train)
        model = lgb.LGBMRegressor(cat_l2=97.31432379031611, cat_smooth=48.035921397517974,
              deterministic=True, feature_fraction=0.6799312109717625,
              force_row_wise=True, learning_rate=0.021594784937466943,
              max_bin=209, max_cat_threshold=97, max_depth=15,
              min_data_in_leaf=14, min_data_per_group=66,
              min_gain_to_split=0.012846015081543094, n_estimators=2500,
              num_leaves=1157, objective='rmse', random_state=2025,
              reg_alpha=0.08117475001618887, reg_lambda=68.91922642338055,
              verbose=-1)
        model.fit(
            X_train_fit_emb, y_train_fit_log_emb,
            eval_set=[(X_test_fit_emb, y_test_fit_log)],
            eval_metric='rmse', 
            callbacks=[
                lgb.early_stopping(stopping_rounds=stop_iter),  # Early stopping here
                lgb.log_evaluation(0)  # Suppress logging (use 1 for logging every round)
            ]
        )

    elif model_name == "LinearRegression":
        model = LinearRegression()
        model.fit(X_train_fit_lin, y_train_fit_log_lin)

    elif model_name == "FeedForwardNNRegressor":
        model = FeedForwardNNRegressor(
            input_features=X_train_fit_lin.shape[1], output_size=1,  
            batch_size=16, learning_rate=0.001, num_epochs=50,
            hidden_sizes=[200, 100]
        )
        model.fit(X_train_fit_lin, y_train_fit_log_lin)

    elif model_name == "FeedForwardNNRegressorWithEmbeddings":
        pred_vars = [col for col in params['model']['predictor']['all'] if col in X_train_fit_emb.columns] 
        large_categories = ['meta_nbhd_code', 'meta_township_code', 'char_class'] + [c for c in pred_vars if c.startswith('loc_school_')]
        # cat_vars = [col for col in params['model']['predictor']['categorical'] if col in X_train_fit_emb.columns]
        # model = FeedForwardNNRegressorWithEmbeddings(
        #     categorical_features=large_categories, output_size=1, 
        #     batch_size=16, learning_rate=0.001, num_epochs=50, 
        #     hidden_sizes=[200, 100]
        # )
        model = FeedForwardNNRegressorWithEmbeddings(
            categorical_features=large_categories, output_size=1,
            **{'learning_rate': 0.0012399967836846098, 'batch_size': 25, 'num_epochs': 98, 'hidden_sizes': [489, 472]}
        )
        # print(X_train_fit_emb.isna().sum())
        # exit()
        model.fit(X_train_fit_emb, y_train_fit_log_emb)

    elif model_name == "FeedForwardNNRegressorWithConstraints":
        pred_vars = [col for col in params['model']['predictor']['all'] if col in X_train_fit_emb.columns] 
        large_categories = ['meta_nbhd_code', 'meta_township_code', 'char_class'] + [c for c in pred_vars if c.startswith('loc_school_')]
        # cat_vars = [col for col in params['model']['predictor']['categorical'] if col in X_train_fit_emb.columns]
        # model = FeedForwardNNRegressorWithConstraints(
        #     categorical_features=large_categories, output_size=1, 
        #     batch_size=16, learning_rate=0.001, num_epochs=50, 
        #     hidden_sizes=[200, 100],
        #     # Contraint inputs 
        #     n_groups=3, dev_thresh=0.15, group_thresh=0.05, 
        #     use_individual_constraint=True, use_group_constraint=True # Not really working
        # )

        model = FeedForwardNNRegressorWithProjection(
            large_categories, output_size=1, 
            batch_size=16, learning_rate=0.001, num_epochs=10, 
            hidden_sizes=[200, 100], 
            dev_thresh=0.75
        )
        # large_categories_indices = [X_train_fit_emb.columns.get_loc(col) for col in large_categories]
        # model = ConstrainedRegressorProjectedWithEmbeddings(
        #          categorical_features=large_categories_indices,
        #          output_size=1,
        #          batch_size=16,
        #          learning_rate=0.001,
        #          num_epochs=50,
        #          hidden_sizes=[200, 100],
        #          n_groups=3,
        #          dev_thresh=0.15,
        #          group_thresh=0.05,
        #          device=None,
        #          debug=False
        # )
        # model.fit(X_train_fit_emb, y_train_fit_log_emb, debug_mode=True)
        model.fit(X_train_fit_emb, y_train_fit_log_emb)


    if len(model_names) > 0: # If there is any model, predict

        # =========== Prediction phase =========== 
        print("Beggining the performance phase...")
        if model_name in emb_model_names:
            y_pred_train_log = model.predict(X_train_fit_emb)
            y_pred_test_log = model.predict(X_test_fit_emb)
            # Exponential target to recover original values
            y_train_fit = np.exp(y_train_fit_log_emb)
        elif model_name in lin_model_names:
            y_pred_train_log = model.predict(X_train_fit_lin)
            y_pred_test_log = model.predict(X_test_fit_lin)
            # Exponential target to recover original values
            y_train_fit = np.exp(y_train_fit_log_lin)

        # Exponential target to recover original values
        y_pred_train = np.exp(y_pred_train_log)
        y_pred_test = np.exp(y_pred_test_log)
        y_test_fit = np.exp(y_test_fit_log)

        # --- Evaluate Performance ---
        # For regression, we can use metrics like Mean Squared Error (MSE).
        train_mse = mean_squared_error(y_pred_train, y_train_fit)
        test_mse = mean_squared_error(y_pred_test, y_test_fit)
        train_r2 = r2_score(y_train_fit, y_pred_train)
        test_r2 = r2_score(y_test_fit, y_pred_test)
        # Ratios and fairness metrics
        n_groups_, alpha_ = 3, 2 
        r_pred_train = y_pred_train / y_train_fit
        r_pred_test = y_pred_test / y_test_fit
        f_metrics_train = compute_haihao_F_metrics(r_pred_train, y_train_fit, n_groups=n_groups_, alpha=alpha_)
        f_metrics_test = compute_haihao_F_metrics(r_pred_test, y_test_fit, n_groups=n_groups_, alpha=alpha_)

        print(f"RMSE train: {np.sqrt(train_mse):.3f}")
        print(f"RMSE val: {np.sqrt(test_mse):.3f}")
        print(fr"$R^2$ train: {train_r2:.3f}")
        print(fr"$R^2$ val: {test_r2:.3f}")
        print(fr"$F_dev$ ({alpha_}) train: {f_metrics_train['f_dev']:.3f}")
        print(fr"$F_dev$ ({alpha_}) val: {f_metrics_test['f_dev']:.3f}")
        print(fr"$F_grp$ ({n_groups_}) train: {f_metrics_train['f_grp']:.3f}")
        print(fr"$F_grp$ ({n_groups_}) val: {f_metrics_test['f_grp']:.3f}")



exit()


# ========== CV REFACTOR =============

# --- General Parameters (YAML content) ---
params = {
    'toggle': {'cv_enable': True},
    'cv': {
        'num_folds': params['cv']['num_folds'],
        'initial_set': params['cv']['initial_set'],
        'max_iterations': 100, # Reduced for faster demo
    },
    'model': {
        'name': 'FeedForwardNNRegressorWithEmbeddings', # <-- SELECT MODEL HERE
        'objective': 'regression_l1', 'verbose': -1, 'deterministic': True,
        'force_row_wise': True, 'seed': 42,
        'predictor': {
            'all': params['model']['predictor']['all'],
            'categorical': params['model']['predictor']['categorical'],
            'id': [], # params['model']['predictor']['id']
        },
        'parameter': {
            'stop_iter': 50, 'validation_prop': 0.1, 'validation_type': 'recent',
            'validation_metric': 'rmse', 'link_max_depth': True,
        },
        'hyperparameter': {
            'LightGBM': {
                'range': {
                    'learning_rate': [0.01, 0.2], 'max_bin': [64, 256], 'num_leaves': [20, 100],
                    'add_to_linked_depth': [1, 5], 'feature_fraction': [0.5, 1.0],
                    'min_gain_to_split': [1e-8, 1.0], 'min_data_in_leaf': [10, 50],
                    'max_cat_threshold': [16, 64], 'min_data_per_group': [50, 200],
                    'cat_smooth': [1.0, 30.0], 'cat_l2': [1.0, 100.0],
                    'lambda_l1': [1e-8, 10.0], 'lambda_l2': [1e-8, 10.0],
                    'n_estimators': [50, 1000]
                },
                'default': {'learning_rate': 0.05, 'num_leaves': 31}
            },
            'RandomForestRegressor': {
                'range': {
                    'n_estimators': [50, 500], 'max_depth': [5, 50],
                    'min_samples_split': [2, 20], 'min_samples_leaf': [1, 10],
                    'max_features': [0.5, 1.0]
                },
                'default': {'n_estimators': 100, 'max_depth': 10}
            },
            'LinearRegression': {
                'range': { 'fit_intercept': [True, False] },
                'default': {'fit_intercept': True}
            },
            'FeedForwardNNRegressorWithEmbeddings': {
                'range': {
                    'learning_rate': [1e-4, 1e-2],
                    'batch_size': [16, 64],
                    'num_epochs': [10, 100],
                    # Defines search space for hidden layers:
                    # [[min_layers, max_layers], [min_units, max_units]]
                    'hidden_sizes': [[1, 2], [128, 512]]
                },
                'default': {
                    'learning_rate': 0.001, 'batch_size': 32,
                    'num_epochs': 50, 'hidden_sizes': [256, 128]
                }
            }
        }
    }
}

# --- Execution ---
model_name_to_run = params['model']['name']
print(f"----- RUNNING FOR MODEL: {model_name_to_run.upper()} -----")

model_params = params['model']['parameter']
model_params.update({
    'objective': params['model']['objective'], 'verbose': params['model']['verbose'],
    'deterministic': params['model']['deterministic'], 'force_row_wise': params['model']['force_row_wise'],
    'seed': params['model']['seed'],
    'predictor': params['model']['predictor'] # Pass predictor info for NN
})
model_params['early_stopping_enable'] = model_params['validation_prop'] > 0 and model_params['stop_iter'] > 0

# pipeline = ModelMainRecipe(
pipeline = ModelMainRecipeImputer(
    outcome="meta_sale_price",
    pred_vars=params['model']['predictor']['all'],
    cat_vars=params['model']['predictor']['categorical'],
    id_vars=params['model']['predictor']['id']
)

handler = ModelHandler(
    model_name=model_name_to_run,
    model_params=model_params,
    hyperparameter_config=params['model']['hyperparameter'][model_name_to_run]
)

temporal_cv_process = TemporalCV(
    model_handler=handler,
    cv_params=params['cv'],
    cv_enable=params['toggle']['cv_enable'],
    data=training_data_full.loc[train.index, train.columns.tolist() + ["meta_sale_date"]],
    target_col='meta_sale_price',
    date_col='meta_sale_date',
    preproc_pipeline=pipeline
)
temporal_cv_process.run()


exit()


















































# ------------------------------------------------------------
# 4. LightGBM Model (Python translation of your R block)
# ------------------------------------------------------------

print("Initializing LightGBM model")


# ---- Engine / static parameters ----
stop_iter         = params['model']['parameter']['stop_iter']
objective         = params['model']['objective']
verbose           = params['model']['verbose']
validation_prop   = params['model']['parameter']['validation_prop']   # e.g., 0.2
validation_type   = params['model']['parameter']['validation_type']   # "random" | "recent"
validation_metric = params['model']['parameter']['validation_metric'] # e.g., "rmse"
link_max_depth    = params['model']['parameter']['link_max_depth']    # bool
deterministic     = params['model']['deterministic']
force_row_wise    = params['model']['force_row_wise']
seed              = params['model']['seed']
categorical_cols  = params['model']['predictor']['categorical']       # names (if you keep categorical dtype)

# MINE: Parameters to be used
cv_enable = True#params['toggle']['cv_enable'] # False on yaml
if stop_iter is not  None:
    early_stopping_enable =  validation_prop > 0 and stop_iter > 0 
else:
    early_stopping_enable = False 
print("cv_enable, early_stopping_enable: ", cv_enable, early_stopping_enable)


# num_iterations policy
if cv_enable and early_stopping_enable:
    # use upper bound as a static maximum (LightGBM will stop early)
    n_estimators_static = params['model']['hyperparameter']['range']['num_iterations'][1]
elif cv_enable and (not early_stopping_enable):
    n_estimators_static = None  # tuned later
else:
    n_estimators_static = params['model']['hyperparameter']['default']['num_iterations']

# ------------------------------------------------------------
# Helper: rolling-origin CV splitter (recent)
# Splits by time order; 'v' folds; optionally overlaps via months (approx).
# Keeps last 'validation_prop' fraction of each fold's training portion as eval set.
# ------------------------------------------------------------
def rolling_origin_splits(df, v, date_col, overlap_months=0):
    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    n = len(df_sorted)
    # Make v chronological folds of (approx) equal size
    fold_sizes = [n // v] * v
    for i in range(n % v):
        fold_sizes[i] += 1

    idx = 0
    folds = []
    for fs in fold_sizes:
        start = idx
        end = idx + fs
        folds.append((start, end))
        idx = end

    # Convert to (train_idx, test_idx) where test is the fold segment (recent)
    # Overlap: we can optionally extend the start backward by months via a simple approximation;
    # here we keep it simple & contiguous. You can refine per-month logic if needed.
    splits = []
    for i, (start, end) in enumerate(folds):
        test_idx = np.arange(start, end)
        train_idx = np.arange(0, start)  # everything before the test segment
        if len(train_idx) == 0:
            # skip very first fold if it has no prior training data
            continue
        splits.append((train_idx, test_idx))
    return df_sorted, splits

# ------------------------------------------------------------
# Helper: prepare eval split inside each training fold (for early stopping)
# Returns (X_tr, y_tr, X_val, y_val)
# 'prep_fn' should fit on tr and transform both tr/val consistently.
# If your data is already numeric, you can set prep_fn to identity.
# ------------------------------------------------------------
def split_train_eval(X_fold, y_fold, val_prop):
    if val_prop <= 0 or val_prop >= 1:
        return (X_fold, y_fold, None, None)
    n = len(X_fold)
    val_n = max(1, int(round(n * val_prop)))
    tr_n = n - val_n
    if tr_n <= 0:
        # fallback: no split
        return (X_fold, y_fold, None, None)
    X_tr = X_fold.iloc[:tr_n, :] if isinstance(X_fold, pd.DataFrame) else X_fold[:tr_n]
    y_tr = y_fold.iloc[:tr_n]     if isinstance(y_fold, pd.Series)  else y_fold[:tr_n]
    X_val = X_fold.iloc[tr_n:, :] if isinstance(X_fold, pd.DataFrame) else X_fold[tr_n:]
    y_val = y_fold.iloc[tr_n:]     if isinstance(y_fold, pd.Series)  else y_fold[tr_n:]
    return X_tr, y_tr, X_val, y_val

# ------------------------------------------------------------
# Make/fit a LightGBM model for given hyperparameters
# Applies link_max_depth if enabled.
# ------------------------------------------------------------
def make_lgbm(params_dict):
    num_leaves       = params_dict['num_leaves']
    add_link_depth   = params_dict['add_to_linked_depth']
    max_depth = -1
    if link_max_depth:
        # max_depth = floor(log2(num_leaves)) + add_to_linked_depth
        max_depth = max(1, int(floor(log2(max(2, num_leaves)))) + int(add_link_depth))

    model = lgb.LGBMRegressor(
        # Determinism / engine controls
        random_state=seed,
        deterministic=deterministic,
        force_row_wise=force_row_wise,
        # n_jobs=num_threads, # Check the number CPU's first
        verbose=verbose,

        # Objective
        objective=objective,

        # Core complexity / tree params
        learning_rate=params_dict['learning_rate'],
        max_bin=int(params_dict['max_bin']),
        num_leaves=int(num_leaves),
        feature_fraction=params_dict['feature_fraction'],
        min_gain_to_split=params_dict['min_gain_to_split'],
        min_data_in_leaf=int(params_dict['min_data_in_leaf']),
        max_depth=int(max_depth) if max_depth != -1 else -1,

        # Categorical-specific
        max_cat_threshold=int(params_dict['max_cat_threshold']),
        min_data_per_group=int(params_dict['min_data_per_group']),
        cat_smooth=params_dict['cat_smooth'],
        cat_l2=params_dict['cat_l2'],

        # Regularization
        reg_alpha=params_dict['lambda_l1'],
        reg_lambda=params_dict['lambda_l2'],

        # Trees (n_estimators)
        n_estimators=int(params_dict['n_estimators'])
        if params_dict.get('n_estimators') is not None
        else (int(n_estimators_static) if n_estimators_static is not None else 1000)
    )
    return model

# ------------------------------------------------------------
# Define search space (from params$model$hyperparameter$range)
# Mirrors your 'update(...) lightsnip::...' bounds.
# ------------------------------------------------------------
lgbm_range = params['model']['hyperparameter']['range']

# ======== MINE: we have to change the range of some hyperparameters:
lgbm_range['learning_rate'] = [10 ** x for x in lgbm_range['learning_rate']]
lgbm_range['min_gain_to_split'] = [10 ** x for x in lgbm_range['min_gain_to_split']]
print("lgbm_range['min_gain_to_split']: ", lgbm_range['min_gain_to_split'])
lgbm_range['cat_l2'] = [10 ** x for x in lgbm_range['cat_l2']]
lgbm_range['lambda_l1'] = [10 ** x for x in lgbm_range['lambda_l1']]
lgbm_range['lambda_l2'] = [10 ** x for x in lgbm_range['lambda_l2']]
# ======== END MINE

def suggest_from_range(trial, name, rng):
    # rng could be [low, high] or have fields {low, high, step}
    low, high = rng[0], rng[1]
    if isinstance(low, int) and isinstance(high, int):
        return trial.suggest_int(name, low, high)
    else:
        return trial.suggest_float(name, float(low), float(high), log=False)

def objective_optuna(trial):

    # Tune-able parameters (equivalent to tune())
    hp = {}
    hp['learning_rate']       = suggest_from_range(trial, 'learning_rate',       lgbm_range['learning_rate'])
    hp['max_bin']             = suggest_from_range(trial, 'max_bin',             lgbm_range['max_bin'])
    hp['num_leaves']          = suggest_from_range(trial, 'num_leaves',          lgbm_range['num_leaves'])
    hp['add_to_linked_depth'] = suggest_from_range(trial, 'add_to_linked_depth', lgbm_range['add_to_linked_depth'])
    hp['feature_fraction']    = suggest_from_range(trial, 'feature_fraction',    lgbm_range['feature_fraction'])
    hp['min_gain_to_split']   = suggest_from_range(trial, 'min_gain_to_split',   lgbm_range['min_gain_to_split'])
    hp['min_data_in_leaf']    = suggest_from_range(trial, 'min_data_in_leaf',    lgbm_range['min_data_in_leaf'])
    hp['max_cat_threshold']   = suggest_from_range(trial, 'max_cat_threshold',   lgbm_range['max_cat_threshold'])
    hp['min_data_per_group']  = suggest_from_range(trial, 'min_data_per_group',  lgbm_range['min_data_per_group'])
    hp['cat_smooth']          = suggest_from_range(trial, 'cat_smooth',          lgbm_range['cat_smooth'])
    hp['cat_l2']              = suggest_from_range(trial, 'cat_l2',              lgbm_range['cat_l2'])
    hp['lambda_l1']           = suggest_from_range(trial, 'lambda_l1',           lgbm_range['lambda_l1'])
    hp['lambda_l2']           = suggest_from_range(trial, 'lambda_l2',           lgbm_range['lambda_l2'])

    # Trees policy
    if cv_enable and (not early_stopping_enable):
        # tune n_estimators from range
        ni_range = lgbm_range['num_iterations']
        hp['n_estimators'] = trial.suggest_int('n_estimators', int(ni_range[0]), int(ni_range[1]))
    else:
        hp['n_estimators'] = n_estimators_static  # fixed

    # Build model with current hyperparameters
    model = make_lgbm(hp)

    # --------- Build CV folds ---------
    # X_full, y_full must be numeric matrices ready for LGBM.
    # If you have a preprocessing pipeline, fit it on fold-train and transform both train/val.
    training_data_full_log = training_data_full.loc[train.index, train.columns.tolist() + ["meta_sale_date"]].reset_index()
    X_full = training_data_full_log #.drop(["meta_sale_price"])  # <- provide your feature frame for training (preprocessed or to-be preprocessed consistently)
    # y_full = np.log(training_data_full_log['meta_sale_price'])  # <- corresponding target (numeric) # MINE: already log
    y_full = training_data_full_log['meta_sale_price'] # MINE: already log

    if validation_type == 'random':
        kf = KFold(n_splits=params['cv']['num_folds'], shuffle=True, random_state=seed)
        folds = list(kf.split(X_full))
    else:
        # recent / rolling-origin
        df_sorted, splits = rolling_origin_splits(
            training_data_full_log, v=params['cv']['num_folds'],
            date_col='meta_sale_date',
            overlap_months=params['cv']['fold_overlap']
        )
        # Align X_full/y_full order to df_sorted if needed:
        # (Assume X_full,y_full are already aligned to 'train' order)
        folds = [(tr_idx, te_idx) for (tr_idx, te_idx) in splits]

    # --------- Evaluate trial across folds ---------
    rmses = []
    for tr_idx, te_idx in folds:
        # Fold train = everything before test segment
        X_tr_full = X_full.iloc[tr_idx, :] if isinstance(X_full, pd.DataFrame) else X_full[tr_idx]
        y_tr_full = y_full.iloc[tr_idx]     if isinstance(y_full, pd.Series)  else y_full[tr_idx]

        # From that fold-train portion, carve out an eval set for early stopping
        X_tr, y_tr, X_val, y_val = split_train_eval(X_tr_full, y_tr_full, validation_prop)

         # ============= MINE: Preprocess train and test with pipeline
        # model_emb_pipeline = ModelMainRecipe(
        #     outcome= "meta_sale_price",
        #     pred_vars=params['model']['predictor']['all'],
        #     cat_vars=params['model']['predictor']['categorical'],
        #     id_vars=params['model']['predictor']['id']
        # )
        X_tr = model_emb_pipeline.fit_transform(X_tr, y_tr).drop(columns=params['model']['predictor']['id'])
        # print(X_tr.dtypes)
        X_val = model_emb_pipeline.transform(X_val).drop(columns=params['model']['predictor']['id'])
        # print(X_val.dtypes)
        # for col in X_tr.columns:
        #     if X_tr[col].dtype != X_val[col].dtype:
        #         print("Is train same dtpye as val? ", X_tr[col].dtype == X_val[col].dtype)
        #         print("diff col: ", col)
        #         print("train type: ", X_tr[col].dtype)
        #         print("val type: ", X_val[col].dtype)
        #     if X_tr[col].dtype == "category" or X_val[col].dtype == "category":
        #         if len(X_tr[col].cat.categories) != len(X_val[col].cat.categories):
        #             print("obj col:", col)
        #             print("categories train: ", len(X_tr[col].cat.categories), X_tr[col].cat.categories)
        #             print("categories val: ", len(X_val[col].cat.categories), X_val[col].cat.categories)

        # print("Preprocess check: ", check_lgbm_categorical_alignment(X_tr, X_val))

        # ============= END MINE
        

        fit_kwargs = {}
        if early_stopping_enable and (X_val is not None):
            fit_kwargs.update({
                'eval_set': [(X_val, y_val)],
                'eval_metric': validation_metric,
                # 'early_stopping_rounds': stop_iter, # Error
                'callbacks': [  # Fix
                    lgb.early_stopping(stopping_rounds=stop_iter),  # Early stopping here
                    lgb.log_evaluation(0)  # Suppress logging (use 1 for logging every round)
                ],
                # 'verbose': False # Error
            })

        model.fit(X_tr, y_tr, **fit_kwargs)

        # Predict on the fold's test segment (not the eval split)
        X_te = X_full.iloc[te_idx, :] if isinstance(X_full, pd.DataFrame) else X_full[te_idx]
        y_te = y_full.iloc[te_idx]     if isinstance(y_full, pd.Series)  else y_full[te_idx]
        # MINE: preprocess test
        X_te = model_emb_pipeline.transform(X_te).drop(columns=params['model']['predictor']['id'])
        y_pred = model.predict(X_te)

        rmse = mean_squared_error(y_te, y_pred, squared=False)
        rmses.append(rmse)

    return float(np.mean(rmses))

# ------------------------------------------------------------
# Run CV (Bayesian tuning) if enabled
# ------------------------------------------------------------
if cv_enable:
    print("Starting cross-validation")
    # sampler = optuna.samplers.TPESampler(seed=seed, ) # Fix
    study = optuna.create_study(direction='minimize', study_name='lgbm_bayes', sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=params['cv']['initial_set'])) # Fix
    study.optimize(
        objective_optuna,
        n_trials=params['cv']['max_iterations'],
        # n_startup_trials=params['cv']['initial_set'], # MINE: Error (it is already in the TPESampler)
        show_progress_bar=True
    )
    best_params = study.best_params
    print("Best params: ", best_params)

    # Materialize final model with best params
    # Trees policy: if early stopping, keep fixed upper bound; else use tuned n_estimators
    final_hp = {
        **best_params,
        'n_estimators': best_params.get('n_estimators', n_estimators_static)
    }
    print("final_hp: ", final_hp)
    best_model = make_lgbm(final_hp)
    print("best model: ", best_model)

else:
    # No CV: use defaults / fixed trees policy
    default_hp = {
        'learning_rate':       params['model']['hyperparameter']['default']['learning_rate'],
        'max_bin':             params['model']['hyperparameter']['default']['max_bin'],
        'num_leaves':          params['model']['hyperparameter']['default']['num_leaves'],
        'add_to_linked_depth': params['model']['hyperparameter']['default']['add_to_linked_depth'],
        'feature_fraction':    params['model']['hyperparameter']['default']['feature_fraction'],
        'min_gain_to_split':   params['model']['hyperparameter']['default']['min_gain_to_split'],
        'min_data_in_leaf':    params['model']['hyperparameter']['default']['min_data_in_leaf'],
        'max_cat_threshold':   params['model']['hyperparameter']['default']['max_cat_threshold'],
        'min_data_per_group':  params['model']['hyperparameter']['default']['min_data_per_group'],
        'cat_smooth':          params['model']['hyperparameter']['default']['cat_smooth'],
        'cat_l2':              params['model']['hyperparameter']['default']['cat_l2'],
        'lambda_l1':           params['model']['hyperparameter']['default']['lambda_l1'],
        'lambda_l2':           params['model']['hyperparameter']['default']['lambda_l2'],
        'n_estimators':        n_estimators_static if n_estimators_static is not None else 1000
    }
    best_model = make_lgbm(default_hp)

# Note:
# - Fit best_model on your full training set (with early stopping if desired) like:
#     X_tr, y_tr, X_val, y_val = split_train_eval(X_train_full, y_train_full, validation_prop)
#     best_model.fit(X_tr, y_tr,
#                    eval_set=[(X_val, y_val)] if (early_stopping_enable and X_val is not None) else None,
#                    eval_metric=validation_metric,
#                    early_stopping_rounds=stop_iter if early_stopping_enable else None,
#                    verbose=False)
# - If you preserve categorical dtype in pandas for `categorical_cols`, LightGBM will use its categorical handling.
