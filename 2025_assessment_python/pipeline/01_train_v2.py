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
from nn_models.nn_unconstrained import FeedForwardNNRegressor, FeedForwardNNRegressorWithEmbeddings
from nn_models.unconstrained.BaselineModels2 import FeedForwardNNRegressorWithEmbeddings2
from nn_models.unconstrained.BaselineModels3 import FeedForwardNNRegressorWithEmbeddings3
from nn_models.nn_constrained_cpu_v2 import FeedForwardNNRegressorWithProjection # FeedForwardNNRegressorWithConstraints, 
from nn_models.nn_constrained_cpu_v3 import ConstrainedRegressorProjectedWithEmbeddings
from nn_models.unconstrained.TabTransformerRegressor import TabTransformerRegressor
from nn_models.unconstrained.TabTransformerRegressor2 import TabTransformerRegressor2
from nn_models.unconstrained.TabTransformerRegressor3 import TabTransformerRegressor3
from nn_models.unconstrained.WideAndDeepRegressor import WideAndDeepRegressor
from nn_models.unconstrained.TabNetRegressor import TabNetRegressor
from nn_models.unconstrained.SpatialGNNRegressor import SpatialGNNRegressor

# 2. Pipelines
from R.recipes import model_main_pipeline, model_lin_pipeline, my_model_lin_pipeline
from src.util_functions import compute_haihao_F_metrics
from recipes.recipes_pipelined import build_model_pipeline, build_model_pipeline_supress_onehot, ModelMainRecipe, ModelMainRecipeImputer

# 3. Preprocessors
from balancing_models import BalancingResampler

# 4. Cross Validation
from temporal_bayes_cv import ModelHandler, TemporalCV#TemporalBayesCV, ModelSpec
from optuna.pruners import SuccessiveHalvingPruner


# If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  
#  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
# In command prompt: set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Load YAML params file
with open('params.yaml', 'r') as file:
    params = yaml.safe_load(file)

# Inputs
assessment_year = 2025

use_sample = False
sample_size = 1000#00 # SAMPLE SIZE

apply_resampling = False

emb_pipeline_names = ["ModelMainRecipe", "ModelMainRecipeImputer", "build_model_pipeline_supress_onehot"]
emb_pipeline_name = emb_pipeline_names[1]


model_names = [
    # "LinearRegression", 
    # "FeedForwardNNRegressor", 
    # "LightGBM", 
    # "FeedForwardNNRegressorWithEmbeddings", 
    # "FeedForwardNNRegressorWithProjection",
    # "ConstrainedRegressorProjectedWithEmbeddings",

    # Modified
    # "FeedForwardNNRegressorWithEmbeddings2",
    # "FeedForwardNNRegressorWithEmbeddings3",
    # More unconstrained
    # "TabTransformerRegressor",
    # "TabTransformerRegressor2",
    # "WideAndDeepRegressor",
    # "TabNetRegressor",
    # "SpatialGNNRegressor",
]
emb_model_names = [
    "LightGBM", 
    "FeedForwardNNRegressorWithEmbeddings", "FeedForwardNNRegressorWithProjection", "ConstrainedRegressorProjectedWithEmbeddings",
    "TabTransformerRegressor","TabTransformerRegressor2", "WideAndDeepRegressor", "TabNetRegressor", "SpatialGNNRegressor",
    "FeedForwardNNRegressorWithEmbeddings2", "FeedForwardNNRegressorWithEmbeddings3",
]
lin_model_names = ["LinearRegression", "FeedForwardNNRegressor" ]


#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 2. Prepare Data --------------------------------------------------------------
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
print("Preparing model training data")


# Load the full set of training data, then arrange by sale date in order to
# facilitate out-of-time sampling/validation

# NOTE: It is critical to trim "multicard" sales when training. Multicard means
# there is multiple buildings on a PIN. Since these sales include multiple
# buildings, they are typically higher than a "normal" sale and must be removed
desired_columns = params['model']['predictor']['all'] + params['model']['predictor']['id'] + ['meta_sale_price', 'meta_sale_date'] + ["ind_pin_is_multicard", "sv_is_outlier"]
training_data_full = pd.read_parquet(f"input/training_data.parquet", columns=desired_columns)#columns=params['model']['predictor']['all'] + params['model']['predictor']['id'] + ['meta_sale_price', 'meta_sale_date'])
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
    print(f"I am using a sample of size: {sample_size}")
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

# NEW: valudation cut
split_index = int(len(train) * split_prop)
val = train.iloc[split_index:]
train = train.iloc[:split_index]


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
train = train[params['model']['predictor']['all'] + params['model']['predictor']['id'] + ['meta_sale_price', 'meta_sale_date']]
val = val[params['model']['predictor']['all'] + params['model']['predictor']['id'] + ['meta_sale_price', 'meta_sale_date']]
# train = train[params['model']['predictor']['all'] + params['model']['predictor']['id'] + ['meta_sale_price'] + ["meta_sale_date"]] # Sale date for CV split (?)
test = test[params['model']['predictor']['all'] + params['model']['predictor']['id'] + ['meta_sale_price', 'meta_sale_date']]

# Split the data in X, y
X_train_prep, y_train_fit_log = train.drop(columns=['meta_sale_price']), train['meta_sale_price'] 
X_val_prep, y_val_fit_log = val.drop(columns=['meta_sale_price']), val['meta_sale_price'] 
X_test_prep, y_test_fit_log = test.drop(columns=['meta_sale_price']), test['meta_sale_price']

# print("COLUMNS: ", X_train_prep.columns)

# ===== New pipeline version ====
model_lin_pipeline = build_model_pipeline(
    pred_vars=params['model']['predictor']['all'],
    cat_vars=params['model']['predictor']['categorical'],
    id_vars=params['model']['predictor']['id']
)

if emb_pipeline_name == "build_model_pipeline_supress_onehot":
    model_emb_pipeline = build_model_pipeline_supress_onehot( # WARNING: We only changed to this to perform changes on the pipeline
        pred_vars=params['model']['predictor']['all'],
        cat_vars=params['model']['predictor']['categorical'],
        id_vars=params['model']['predictor']['id']
    )
elif emb_pipeline_name == "ModelMainRecipe":
    model_emb_pipeline = ModelMainRecipe(
        outcome= "meta_sale_price",
        pred_vars=params['model']['predictor']['all'],
        cat_vars=params['model']['predictor']['categorical'],
        id_vars=params['model']['predictor']['id']
    )
elif emb_pipeline_name == "ModelMainRecipeImputer":
    model_emb_pipeline = ModelMainRecipeImputer(
            outcome= "meta_sale_price",
            pred_vars=params['model']['predictor']['all'],
            cat_vars=params['model']['predictor']['categorical'],
            id_vars=params['model']['predictor']['id']
    )

# === Fitting datasets ===


# 1. Fit / transform the data for linear models
X_train_fit_lin = model_lin_pipeline.fit_transform(X_train_prep, y_train_fit_log)#.drop(columns=params['model']['predictor']['id'])
X_test_fit_lin = model_lin_pipeline.transform(X_test_prep)#.drop(columns=params['model']['predictor']['id'])
# ===== Resampler =====
if apply_resampling:
    resampler_lin = BalancingResampler( 
        n_bins=100, binning_policy='uniform', max_diff_ratio=0.7,
        undersample_policy='random', oversample_policy='smoter',
        smote_k_neighbors=5, random_state=42
    )
    X_train_fit_lin, y_train_fit_log_lin = resampler_lin.fit_resample(X_train_fit_lin, y_train_fit_log) # only train is for resampling
else:
    y_train_fit_log_lin = y_train_fit_log


# 2. Fit / transform the data for models with embeddings
X_train_fit_emb = model_emb_pipeline.fit_transform(X_train_prep, y_train_fit_log).drop(columns=params['model']['predictor']['id'], errors="ignore")
X_val_fit_emb = model_emb_pipeline.transform(X_val_prep).drop(columns=params['model']['predictor']['id'], errors="ignore")
X_test_fit_emb = model_emb_pipeline.transform(X_test_prep).drop(columns=params['model']['predictor']['id'], errors="ignore")
# X_train_fit_emb["char_recent_renovation"] = X_train_fit_emb["char_recent_renovation"].astype(bool) # QUESTION: Why is this not in cat_vars?
# na_columns = X_train_fit_emb.isna().sum()[X_train_fit_emb.isna().sum() > 0].index
# X_train_fit_emb[na_columns] = X_train_fit_emb[na_columns].fillna(value="unknown")
cat_cols_emb = [i for i,col in enumerate(X_train_fit_emb.columns) if X_train_fit_emb[col].dtype == object  or X_train_fit_emb[col].dtype == "category"]
# ===== Resampler =====
if apply_resampling:
    resampler_emb = BalancingResampler( 
        n_bins=100, binning_policy='uniform', max_diff_ratio=0.7,
        undersample_policy='random', oversample_policy='smotenc',
        smote_k_neighbors=5, random_state=42,
        categorical_features=cat_cols_emb
    )
    X_train_fit_emb, y_train_fit_log_emb = resampler_emb.fit_resample(X_train_fit_emb, y_train_fit_log) # only train is for resampling
else:
    y_train_fit_log_emb = y_train_fit_log
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



# print("All features:")
# for col in X_train_fit_emb.columns:
#     if len(X_train_fit_emb[col].unique()) < 10:
#         print(col, X_train_fit_emb[col].dtype, X_train_fit_emb[col].unique()) 
#     else:
#         print(col, X_train_fit_emb[col].dtype, X_train_fit_emb[col].unique()[:10]) 
# # print(X_train_fit_emb.columns.tolist())
# exit()
# ==========================================================================================
#                       Comparisson of the different models
# ==========================================================================================

print("Fitting models...")

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

        # 1000 on 100k (only train (?)). Note: Is it not biased bc of validation set (?).
        model = lgb.LGBMRegressor(cat_l2=22.95020213396379, cat_smooth=31.14663605311536,
              deterministic=True, feature_fraction=0.6572673015780892,
              force_row_wise=True, learning_rate=0.03341584496999874,
              max_bin=323, max_cat_threshold=82, max_depth=12,
              min_data_in_leaf=68, min_data_per_group=81,
              min_gain_to_split=0.12374932835073327, n_estimators=2500,
              num_leaves=366, objective='rmse', random_state=2025,
              reg_alpha=0.06676779724096571, reg_lambda=30.039145583263345,
              verbose=-1)

        model.fit(
            X_train_fit_emb, y_train_fit_log_emb,
            eval_set=[(X_val_fit_emb, y_val_fit_log)],
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
        # Default
        # model = FeedForwardNNRegressorWithEmbeddings(
        #     categorical_features=large_categories, output_size=1, random_state=42,
        #     batch_size=16, learning_rate=0.001, num_epochs=15, 
        #     hidden_sizes=[200, 100]
        # )
        # Temp:
        model = FeedForwardNNRegressorWithEmbeddings(
            categorical_features=large_categories, output_size=1, random_state=42,
            learning_rate= 0.003702078773155906,
            batch_size= 31,
            num_epochs= 500,#34,
            hidden_sizes=[95,82,79],
        )
        # print(X_train_fit_emb.isna().sum())
        # exit()
        model.fit(X_train_fit_emb, y_train_fit_log_emb)

    elif model_name == "FeedForwardNNRegressorWithEmbeddings2":
        pred_vars = [col for col in params['model']['predictor']['all'] if col in X_train_fit_emb.columns] 
        large_categories = ['meta_nbhd_code', 'meta_township_code', 'char_class'] + [c for c in pred_vars if c.startswith('loc_school_')]
        cat_vars = cat_vars=params['model']['predictor']['categorical']
        # Temp
        model = FeedForwardNNRegressorWithEmbeddings2(
            categorical_features=large_categories, output_size=1, random_state=42,
            # categorical_features=params['model']['predictor']['categorical'], output_size=1, random_state=42,
            learning_rate= 1e-3,
            batch_size= 40,
            num_epochs= 50,
            hidden_sizes=[1024, 512],
            patience=10,
        )
        # print(X_train_fit_emb.isna().sum())
        # exit()
        # model.fit(X_train_fit_emb, y_train_fit_log_emb)
        model.fit(X_train_fit_emb, y_train_fit_log_emb,
            X_val=X_val_fit_emb, y_val=y_val_fit_log
        )

    elif model_name == "FeedForwardNNRegressorWithEmbeddings3":
        pred_vars = [col for col in params['model']['predictor']['all'] if col in X_train_fit_emb.columns] 
        large_categories = ['meta_nbhd_code', 'meta_township_code', 'char_class'] + [c for c in pred_vars if c.startswith('loc_school_')]
        cat_vars = cat_vars=params['model']['predictor']['categorical']
        # Temp
        model = FeedForwardNNRegressorWithEmbeddings3(
            large_categories, coord_features= ["loc_longitude", "loc_latitude"], output_size=1, random_state=42,
            # cat_vars, coord_features= ["loc_longitude", "loc_latitude"], output_size=1, random_state=42,
            use_fourier_features=False,
            batch_size=25, learning_rate=0.001, num_epochs=50, 
            hidden_sizes=[1024, 512], patience=10, 
            loss_fn='mse', 
            # n_bins=10, # binned_mse # gamma = 1.5
        )
        model.fit(X_train_fit_emb, y_train_fit_log_emb,
            X_val=X_val_fit_emb, y_val=y_val_fit_log
        )


    elif model_name == "FeedForwardNNRegressorWithProjection":
        pred_vars = [col for col in params['model']['predictor']['all'] if col in X_train_fit_emb.columns] 
        large_categories = ['meta_nbhd_code', 'meta_township_code', 'char_class'] + [c for c in pred_vars if c.startswith('loc_school_')]
        # cat_vars = [col for col in params['model']['predictor']['categorical'] if col in X_train_fit_emb.columns]
        # Default
        # model = FeedForwardNNRegressorWithProjection(
        #     large_categories, output_size=1, random_state=42,
        #     batch_size=16, learning_rate=0.001, num_epochs=10, 
        #     hidden_sizes=[200, 100], 
        #     dev_thresh=0.75
        # )
        # # 100k with 10 iters
        # model = FeedForwardNNRegressorWithProjection(
        #     large_categories, output_size=1, 
        #     **{'learning_rate': 0.002911051996104486, 'batch_size': 19, 'num_epochs': 12, 'hidden_sizes': [195], 'dev_thresh': 0.8295835055926347}
        # )
        # 100k with 48 iters
        model = FeedForwardNNRegressorWithProjection(
            large_categories, output_size=1, random_state=42,
            **{'learning_rate': 0.008284479096736002, 'batch_size': 27, 'num_epochs': 14, 'hidden_sizes': [187], 'dev_thresh': 0.9914267936547978}
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

    elif model_name == "ConstrainedRegressorProjectedWithEmbeddings":
        pred_vars = [col for col in params['model']['predictor']['all'] if col in X_train_fit_emb.columns] 
        large_categories = ['meta_nbhd_code', 'meta_township_code', 'char_class'] + [c for c in pred_vars if c.startswith('loc_school_')]
        large_categories_idx = [X_train_fit_emb.columns.tolist().index(col) for col in large_categories]
        model = ConstrainedRegressorProjectedWithEmbeddings(
                categorical_features=large_categories_idx,
                hidden_sizes = [200],
                batch_size = 32,
                learning_rate= 1e-3,
                num_epochs = 0.7,
                n_groups = 3,
                dev_thresh= 1,
                group_thresh = 0.1,
                use_group_balanced_sampler = False,
                random_state = 42,
                debug= False,
        )

        model.fit(X_train_fit_emb, y_train_fit_log_emb)

    # =======================================================
    #               More Unconstrained models
    # =======================================================
    elif model_name == "TabTransformerRegressor":
        cat_vars = cat_vars=params['model']['predictor']['categorical']
        coord_vars = ["loc_longitude", "loc_latitude"]
        # model = TabTransformerRegressor(
        #     cat_vars, coord_vars, output_size=1, random_state=42,
        #         batch_size=16, learning_rate=0.001, num_epochs=30, transformer_dim=16, 
        #         transformer_heads=8, transformer_layers=6,
        #          dropout=0.1, loss_fn='focal_mse' #  'mse', 'focal_mse', or 'huber'.
        # )
        # Temp:
        model = TabTransformerRegressor(
            cat_vars, coord_vars, output_size=1, random_state=42,
            learning_rate= 0.001845632975218141,
            batch_size= 48,
            num_epochs= 31,
            transformer_dim= 16,
            transformer_heads= 8,
            transformer_layers= 6,
            dropout= 0.3171215578791728,
            loss_fn= "mse",
        )
        model.fit(X_train_fit_emb, y_train_fit_log_emb)

    elif model_name == "TabTransformerRegressor2":
        cat_vars = cat_vars=params['model']['predictor']['categorical']
        coord_vars = ["loc_longitude", "loc_latitude"]
        pred_vars = [col for col in params['model']['predictor']['all'] if col in X_train_fit_emb.columns] 
        large_categories = ['meta_nbhd_code', 'meta_township_code', 'char_class'] + [c for c in pred_vars if c.startswith('loc_school_')]
        # Temp
        model = TabTransformerRegressor2(
            cat_vars, coord_vars, output_size=1, random_state=42,
            # large_categories, coord_vars, output_size=1, random_state=42,
            learning_rate= 1e-3,
            batch_size= 16, # 16, 32 | 20, 32 |
            num_epochs= 50,
            transformer_dim= 32,
            transformer_heads= 16,
            transformer_layers= 6,
            dropout= 0.1,
            loss_fn= "huber",
            patience=10,
        )
        model.fit(X_train_fit_emb, y_train_fit_log_emb,
            X_val=X_val_fit_emb, y_val=y_val_fit_log
        )

    elif model_name == "WideAndDeepRegressor":
        cat_vars = cat_vars=params['model']['predictor']['categorical']
        # model = WideAndDeepRegressor(
        #     categorical_features=cat_vars, output_size=1, random_state=42,
        #     batch_size=16, learning_rate=0.001,
        #     num_epochs=50, hidden_sizes=[200, 100]
        # )
        # Temp:
        model = WideAndDeepRegressor(
            categorical_features=cat_vars, output_size=1, random_state=42,
            learning_rate= 0.0037121387535015093,
            batch_size= 49,
            num_epochs= 20, #10,
            hidden_sizes=[403, 277, 173, 70],
        )
        model.fit(X_train_fit_emb, y_train_fit_log_emb)

    elif model_name == "TabNetRegressor":
        cat_vars = cat_vars=params['model']['predictor']['categorical']
        coord_vars = ["loc_longitude", "loc_latitude"]
        model = TabNetRegressor(
            cat_vars, coord_vars, output_size=1, random_state=42,
            batch_size=32, learning_rate=0.001, num_epochs=20, n_steps=3, 
            feature_dim=4, attention_dim=4, sparsity_lambda=1e-5, loss_fn='mse', 
        )
        model.fit(X_train_fit_emb, y_train_fit_log_emb)

    elif model_name == "SpatialGNNRegressor":
        cat_vars = cat_vars=params['model']['predictor']['categorical']
        coord_vars = ["loc_longitude", "loc_latitude"]
        graph_cat_features = ['meta_nbhd_code', 'loc_school_elementary_district_geoid']

        model = SpatialGNNRegressor(
            cat_vars, coord_vars, graph_cat_features, output_size=1, random_state=42,
            k_neighbors=10, batch_size=32, learning_rate=0.001, num_epochs=100,
            gnn_type='gat', gat_heads=4,
            gnn_hidden_dim=64, gnn_layers=3, mlp_hidden_sizes=[32, 16], loss_fn='huber',
        )
        model.fit(X_train_fit_emb, y_train_fit_log_emb)

    if len(model_names) > 0: # If there is any model, predict

        # =========== Prediction phase =========== 
        print("Beggining the performance phase...")
        if model_name in emb_model_names:
            y_pred_train_log = model.predict(X_train_fit_emb)
            y_pred_test_log = model.predict(X_test_fit_emb)
            # print(y_test_fit_log)
            # Exponential target to recover original values
            y_train_fit = np.exp(y_train_fit_log_emb)
        elif model_name in lin_model_names:
            y_pred_train_log = model.predict(X_train_fit_lin)
            y_pred_test_log = model.predict(X_test_fit_lin)
            # Exponential target to recover original values
            y_train_fit = np.exp(y_train_fit_log_lin)

        # Exponential target to recover original values
        # Sub: quick correction
        min_log_price = np.percentile(y_pred_train_log, 1)
        max_log_price = np.percentile(y_pred_train_log, 99)
        y_pred_train_log = np.clip(y_pred_train_log, min_log_price, max_log_price)
        min_log_price = np.percentile(y_pred_test_log, 1)
        max_log_price = np.percentile(y_pred_test_log, 99)
        y_pred_test_log = np.clip(y_pred_test_log, min_log_price, max_log_price)
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
        print(f"RMSE test: {np.sqrt(test_mse):.3f}")
        print(fr"$R^2$ train: {train_r2:.3f}")
        print(fr"$R^2$ test: {test_r2:.3f}")
        print(fr"$F_dev$ ({alpha_}) train: {f_metrics_train['f_dev']:.3f}")
        print(fr"$F_dev$ ({alpha_}) test: {f_metrics_test['f_dev']:.3f}")
        print(fr"$F_grp$ ({n_groups_}) train: {f_metrics_train['f_grp']:.3f}")
        print(fr"$F_grp$ ({n_groups_}) test: {f_metrics_test['f_grp']:.3f}")




# exit()

# ========== CV REFACTOR =============

# --- General Parameters (YAML content) ---
params = {
    'toggle': {'cv_enable': True},
        'cv': { 
            'num_folds': 4, 
            'resampling_strategy': 'subsample', # 'kfold', 'subsample', or 'bootstrap'
            'num_resampling_runs': 10,          # Number of times to resample for bootstrap/subsample
            'subsample_fraction': 0.1,         # Fraction of data to use if strategy is 'subsample'
            'test_set_fraction': 0.2,          # Final holdout set
            'validation_set_fraction': 0.2,    # Validation set from the remaining data
            'initial_set': 3, 
            'max_iterations': 1000,
            'run_name_suffix': 'multi_sample_test_3'
        },
    'model': {
        'name': 'FeedForwardNNRegressorWithEmbeddings', # <-- SELECT MODEL HERE
        'objective': 'regression_l1', 'verbose': -1, 'deterministic': True,
        'force_row_wise': True, 'seed': 42,
        'predictor': {
            'all': params['model']['predictor']['all'],
            'categorical': params['model']['predictor']['categorical'],
            'id': [], # params['model']['predictor']['id'],
            'large_categories':['meta_nbhd_code', 'meta_township_code', 'char_class'] + [c for c in params['model']['predictor']['all'] if c.startswith('loc_school_')],
            'coord_features':["loc_longitude", "loc_latitude"],
        },
        'parameter': {
            'stop_iter': 50, 'validation_prop': 0.2, 'validation_type': 'recent',
            'validation_metric': 'rmse', 'link_max_depth': True, 'early_stopping_enable': True,
        },
        'hyperparameter': {
            'LightGBM': {
                'range': {
                    "n_estimators": params['model']['hyperparameter']['range']['num_iterations'],
                    "learning_rate": [10**x for x in params['model']['hyperparameter']['range']['learning_rate']],  # 10 ^ X,
                    "max_bin": params['model']['hyperparameter']['range']['max_bin'], #[50, 512],
                    "num_leaves": params['model']['hyperparameter']['range']['num_leaves'], #[32, 2048],
                    "add_to_linked_depth": params['model']['hyperparameter']['range']['add_to_linked_depth'], #[1, 7],
                    "feature_fraction": params['model']['hyperparameter']['range']['feature_fraction'], #[0.3, 0.7],
                    "min_gain_to_split": [10**x for x in params['model']['hyperparameter']['range']['min_gain_to_split']], #[-3.0, 4.0]  # 10 ^ X,,
                    "min_data_in_leaf": params['model']['hyperparameter']['range']['min_data_in_leaf'], #[2, 400],
                    "max_cat_threshold": params['model']['hyperparameter']['range']['max_cat_threshold'], #[10, 250],
                    "min_data_per_group": params['model']['hyperparameter']['range']['min_data_per_group'], #[2, 400],
                    "cat_smooth": params['model']['hyperparameter']['range']['cat_smooth'], #[10.0, 200.0],
                    "cat_l2": [10**x for x in params['model']['hyperparameter']['range']['cat_l2']], #[-3, 2]  # 10 ^ X,
                    "lambda_l1": [10**x for x in params['model']['hyperparameter']['range']['lambda_l1']], #[-3, 2]  # 10 ^ X,
                    "lambda_l2": [10**x for x in params['model']['hyperparameter']['range']['lambda_l2']], #[-3, 2]  # 10 ^ X,
                },
                'default': {'learning_rate': 0.05, 'num_leaves': 31}
            },
            'LinearRegression': {
                'range': { 'fit_intercept': [True, False] },
                'default': {'fit_intercept': True}
            },
            'FeedForwardNNRegressorWithEmbeddings': {
                'range': {
                    'learning_rate': [5e-4, 5e-2],
                    'batch_size': [16, 512],
                    'num_epochs': [10, 500],
                    # Defines search space for hidden layers:
                    # [[min_layers, max_layers], [min_units, max_units]]
                    'hidden_sizes': [[1, 6], [64, 4096]],
                    'use_fourier_features':[True, False],
                    'patience':[5, 15], 
                    'loss_fn':['mse', 'focal_mse', 'huber'], 
                    # 'n_bins' : [],
                    'gamma' : [0.75, 1.5],
                },
                'default': {
                    'learning_rate': 1e-3, 'batch_size': 16,
                    'num_epochs': 50, 'hidden_sizes': [256, 128]
                }
            },
            'FeedForwardNNRegressorWithProjection': {
                    'range': {
                        'learning_rate': [1e-3, 1e-2],
                        'batch_size': [16, 32],
                        'num_epochs': [10, 20],
                        'hidden_sizes': [[1, 2], [128, 256]],
                        'dev_thresh': [0.7, 1.0]
                    },
                    'default': {
                        'learning_rate': 0.001, 'batch_size': 32, 'num_epochs': 30,
                        'hidden_sizes': [128, 64], 'dev_thresh': 0.15
                    }
            },
            'TabTransformerRegressor': {
                'range': {
                    'learning_rate': [1e-4, 1e-3], 'batch_size': [64, 512],
                    'num_epochs': [10, 500], 'transformer_dim': [2, 8], # NOTE: heads as a divisor: transformer_dim = transformer_dim * heads
                    'transformer_heads': [4, 8], 'transformer_layers': [4, 6], 
                    'dropout':[0.05, 0.2], 'loss_fn': ['mse', 'focal_mse', 'huber'],
                    'patience':[1, 15], 
                    'fourier_type':['none', 'basic', 'positional', 'gaussian'],
                },
                'default': {
                    'learning_rate': 0.001, 'batch_size': 32, 'num_epochs': 30,
                    'mlp_hidden_dims': [64, 32], 'num_attention_layers': 2,
                    'num_attention_heads': 4
                }
            },
            'WideAndDeepRegressor': {
                'range': {
                    'learning_rate': [1e-4, 1e-2], 'batch_size': [16, 128],
                    'num_epochs': [10, 50], 'hidden_sizes': [[2, 4], [64, 512]]
                },
                'default': {
                    'learning_rate': 0.001, 'batch_size': 64, 'num_epochs': 30,
                    'hidden_sizes': [256, 128]
                }
            },
        }
    }
}

# --- Execution Logic ---
model_name_to_run = params['model']['name']
print(f"----- RUNNING FOR MODEL: {model_name_to_run.upper()} -----")

model_params = params['model']['parameter']
model_params.update({
    'objective': params['model']['objective'], 'verbose': params['model']['verbose'],
    'deterministic': params['model']['deterministic'], 'force_row_wise': params['model']['force_row_wise'],
    'seed': params['model']['seed'],
    'predictor': params['model']['predictor']
})
model_params['early_stopping_enable'] = model_params['validation_prop'] > 0 and model_params['stop_iter'] > 0

if params['model']['name'] in ["LightGBM"]:
    pipeline = ModelMainRecipe(
        outcome="meta_sale_price",
        pred_vars=params['model']['predictor']['all'],
        cat_vars=params['model']['predictor']['categorical'],
        id_vars=params['model']['predictor']['id']
    )
elif params['model']['name'] in ['TabTransformerRegressor']:
    pipeline = ModelMainRecipeImputer(
        outcome="meta_sale_price",
        pred_vars=params['model']['predictor']['all'],
        cat_vars=params['model']['predictor']['categorical'],
        id_vars=params['model']['predictor']['id']
    )
elif params['model']['name'] in ["FeedForwardNNRegressorWithEmbeddings"]:
    pipeline = build_model_pipeline_supress_onehot( # WARNING: We only changed to this to perform changes on the pipeline
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
    data=pd.concat([train, val]),
    target_col='meta_sale_price',
    date_col='meta_sale_date',
    preproc_pipeline=pipeline,
    run_name_suffix=params['cv'].get('run_name_suffix', 'default')
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
