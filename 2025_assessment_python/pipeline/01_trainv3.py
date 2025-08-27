# Analogous of .R script in: https://github.com/ccao-data/model-res-avm/blob/master/pipeline/01-train.R 

# ===============================================================================
# V3: Don't do CV, and just focus on tunning the models by hand in a proper way.
#   The idea is to split the data in train/val/test and select according to that.
# ===============================================================================

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
from nn_models.unconstrained.BaselineModels import FeedForwardNNRegressorWithEmbeddings2
from nn_models.nn_constrained_cpu_v2 import FeedForwardNNRegressorWithProjection # FeedForwardNNRegressorWithConstraints, 
from nn_models.nn_constrained_cpu_v3 import ConstrainedRegressorProjectedWithEmbeddings
from nn_models.unconstrained.TabTransformerRegressor import TabTransformerRegressor
from nn_models.unconstrained.TabTransformerRegressor2 import TabTransformerRegressor2
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

use_sample = True
sample_size = 100000 # SAMPLE SIZE

apply_resampling = True

emb_pipeline_names = ["ModelMainRecipe", "ModelMainRecipeImputer", "build_model_pipeline_supress_onehot"]
emb_pipeline_name = emb_pipeline_names[2]


model_names = [
    # "LinearRegression", 
    # "FeedForwardNNRegressor", 
    # "LightGBM", 
    # "FeedForwardNNRegressorWithEmbeddings", 
    # "FeedForwardNNRegressorWithProjection",
    # "ConstrainedRegressorProjectedWithEmbeddings",

    # Modified
    "FeedForwardNNRegressorWithEmbeddings2",
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
    "FeedForwardNNRegressorWithEmbeddings2"
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
train = train[params['model']['predictor']['all'] + params['model']['predictor']['id'] + ['meta_sale_price', 'meta_sale_date']]
# train = train[params['model']['predictor']['all'] + params['model']['predictor']['id'] + ['meta_sale_price'] + ["meta_sale_date"]] # Sale date for CV split (?)
test = test[params['model']['predictor']['all'] + params['model']['predictor']['id'] + ['meta_sale_price', 'meta_sale_date']]

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
# ===== Resampler =====
if apply_resampling:
    resampler_lin = BalancingResampler( 
        n_bins=100, binning_policy='uniform', max_diff_ratio=0.3,
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
cat_cols_emb = [i for i,col in enumerate(X_train_fit_emb.columns) if X_train_fit_emb[col].dtype == object  or X_train_fit_emb[col].dtype == "category"]
# ===== Resampler =====
if apply_resampling:
    resampler_emb = BalancingResampler( 
        n_bins=100, binning_policy='uniform', max_diff_ratio=0.3,
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
        
        # Temp try:
        num_leaves = 33
        add_to_linked_depth = 5
        model = lgb.LGBMRegressor(
            learning_rate=0.01851839183501252,
            max_bin=484, 
            num_leaves=num_leaves, 
            add_to_linked_depth=add_to_linked_depth,
            feature_fraction=0.3487421794838527,
            min_gain_to_split=0.0019132666418427095, 
            min_data_in_leaf=329, 
            max_cat_threshold=14, 
            min_data_per_group=366,
            cat_smooth=30.52247516945211,
            cat_l2=15.996397825809856, 
            lambda_l1=0.003785404852255749, # reg_alpha
            lambda_l2=0.029397028671217528,# reg_lambda
            # Static?
            n_estimators=2500,
            # Fixed ones
            random_state=42,#2025,
            deterministic=True, 
            force_row_wise=True, 
            max_depth=floor(np.log2(num_leaves)) + add_to_linked_depth,
            objective='rmse', 
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
        # Default
        # model = FeedForwardNNRegressorWithEmbeddings(
        #     categorical_features=large_categories, output_size=1, random_state=42,
        #     batch_size=16, learning_rate=0.001, num_epochs=15, 
        #     hidden_sizes=[200, 100]
        # )
        # 10k samples with 10 iters
        # model = FeedForwardNNRegressorWithEmbeddings(
        #     categorical_features=large_categories, output_size=1,
        #     **{'learning_rate': 0.0012399967836846098, 'batch_size': 25, 'num_epochs': 98, 'hidden_sizes': [489, 472]}
        # )
        # # maybe 10 or 20 iters with 100k
        # model = FeedForwardNNRegressorWithEmbeddings(
        #     categorical_features=large_categories, output_size=1,
        #     **{'learning_rate': 0.004370861069626263, 'batch_size': 32, 'num_epochs': 18, 'hidden_sizes': [148, 148]}
        # )
        # # # 50 iters with 100k
        # model = FeedForwardNNRegressorWithEmbeddings(
        #     categorical_features=large_categories, output_size=1, random_state=42,
        #     **{'learning_rate': 0.004172541632024457, 'batch_size': 24, 'num_epochs': 15, 'hidden_sizes': [184, 235]}
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
        # Temp
        model = FeedForwardNNRegressorWithEmbeddings2(
            categorical_features=large_categories, output_size=1, random_state=42,
            # categorical_features=params['model']['predictor']['categorical'], output_size=1, random_state=42,
            learning_rate= 1e-3,
            batch_size= 16,
            num_epochs= 50,
            hidden_sizes=[1024, 512],
            patience=10,
        )
        # print(X_train_fit_emb.isna().sum())
        # exit()
        model.fit(X_train_fit_emb, y_train_fit_log_emb)

    elif model_name == "FeedForwardNNRegressorWithProjection":
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
        # Temp
        model = TabTransformerRegressor2(
            cat_vars, coord_vars, output_size=1, random_state=42,
            learning_rate= 0.001845632975218141,
            batch_size= 48,
            num_epochs= 31,
            transformer_dim= 16,
            transformer_heads= 8,
            transformer_layers= 6,
            dropout= 0.3171215578791728,
            loss_fn= "mse",
            patience=5,
        )
        model.fit(X_train_fit_emb, y_train_fit_log_emb)

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
        print(f"RMSE val: {np.sqrt(test_mse):.3f}")
        print(fr"$R^2$ train: {train_r2:.3f}")
        print(fr"$R^2$ val: {test_r2:.3f}")
        print(fr"$F_dev$ ({alpha_}) train: {f_metrics_train['f_dev']:.3f}")
        print(fr"$F_dev$ ({alpha_}) val: {f_metrics_test['f_dev']:.3f}")
        print(fr"$F_grp$ ({n_groups_}) train: {f_metrics_train['f_grp']:.3f}")
        print(fr"$F_grp$ ({n_groups_}) val: {f_metrics_test['f_grp']:.3f}")

