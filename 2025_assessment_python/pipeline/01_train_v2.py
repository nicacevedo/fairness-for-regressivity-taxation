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
from nn_models.unconstrained.BaselineModels4 import FeedForwardNNRegressorWithEmbeddings4
from nn_models.nn_constrained_cpu_v2 import FeedForwardNNRegressorWithProjection # FeedForwardNNRegressorWithConstraints, 
from nn_models.nn_constrained_cpu_v3 import ConstrainedRegressorProjectedWithEmbeddings
from nn_models.unconstrained.TabTransformerRegressor import TabTransformerRegressor
from nn_models.unconstrained.TabTransformerRegressor2 import TabTransformerRegressor2
from nn_models.unconstrained.TabTransformerRegressor3 import TabTransformerRegressor3
from nn_models.unconstrained.TabTransformerRegressor4 import TabTransformerRegressor4
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
from temporal_bayes_cv import ModelHandler, TemporalCV 
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
emb_pipeline_name = emb_pipeline_names[2]


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
    # "WideAndDeepRegressor",
    # "TabNetRegressor",
    # "SpatialGNNRegressor",
]
# emb_model_names = [
#     "LightGBM", 
#     "FeedForwardNNRegressorWithEmbeddings", "FeedForwardNNRegressorWithProjection", "ConstrainedRegressorProjectedWithEmbeddings",
#     "TabTransformerRegressor","TabTransformerRegressor2", "WideAndDeepRegressor", "TabNetRegressor", "SpatialGNNRegressor",
#     "FeedForwardNNRegressorWithEmbeddings2", "FeedForwardNNRegressorWithEmbeddings3",
# ]
# lin_model_names = ["LinearRegression", "FeedForwardNNRegressor" ]


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


#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 3. Linear Model --------------------------------------------------------------
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
print("Creating the CV process...")

# Create a linear model recipe with additional imputation, transformations,
# and feature interactions
# training_data_full = training_data_full.copy()

# Clean columns
train = train[params['model']['predictor']['all'] + params['model']['predictor']['id'] + ['meta_sale_price', 'meta_sale_date']]
val = val[params['model']['predictor']['all'] + params['model']['predictor']['id'] + ['meta_sale_price', 'meta_sale_date']]
# train = train[params['model']['predictor']['all'] + params['model']['predictor']['id'] + ['meta_sale_price'] + ["meta_sale_date"]] # Sale date for CV split (?)
test = test[params['model']['predictor']['all'] + params['model']['predictor']['id'] + ['meta_sale_price', 'meta_sale_date']]



# ========== CV REFACTOR =============

pred_vars = [col for col in params['model']['predictor']['all'] if col in train.columns]

# --- General Parameters (YAML content) ---
params = {
    'toggle': {'cv_enable': True},
        'cv': { 
            'num_folds': 5, 
            'resampling_strategy': 'subsample', # 'kfold', 'subsample', or 'bootstrap'
            'num_resampling_runs': 10,          # Number of times to resample for bootstrap/subsample
            'subsample_fraction': 0.1,         # Fraction of data to use if strategy is 'subsample'
            'test_set_fraction': 0.2,          # Final holdout set
            'validation_set_fraction': 0.2,    # Validation set from the remaining data
            'initial_set': 3, 
            'max_iterations': 1000,
            'run_name_suffix': 'robustifying_test_5',#'multi_sample_test_1_2_Tab', #'multi_sample_test_4_5'
        },
    'model': {
        'name': 'FeedForwardNNRegressorWithEmbeddings', # <-- SELECT MODEL HERE
        'base_model': 'TabTransformerRegressor', # model to feed the resampler
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
                'default': {
                    'learning_rate': 0.011364186059408745,
                    'max_bin': 178,
                    'num_leaves': 167,
                    'add_to_linked_depth': 7,
                    'feature_fraction': 0.45512255576737853,
                    'min_gain_to_split': 0.0015979247898933535,
                    'min_data_in_leaf': 65,
                    'max_cat_threshold': 51,
                    'min_data_per_group': 339,
                    'cat_smooth': 120.25007901496078,
                    'cat_l2': 0.01382892133084935,
                    'lambda_l1': 0.03360945895462623,
                    'lambda_l2': 0.3003031020947675,
                }
            },
            'LinearRegression': {
                'range': { 'fit_intercept': [True, False] },
                'default': {'fit_intercept': True}
            },
            'FeedForwardNNRegressorWithEmbeddings': {
                'range': {
                    'learning_rate': [5e-4, 5e-3],
                    'batch_size': [16,128],#[20, 120],
                    'num_epochs': [200,200],#[100, 500],
                    # Defines search space for hidden layers:
                    # [[min_layers, max_layers], [min_units, max_units]]
                    # 'hidden_sizes': [[4,6], [64, 4096]],#[[1, 8], [16, 5000]],
                    # 'use_fourier_features':[True, False],
                    'patience': [8, 12], 
                    'loss_fn': ['huber', 'mse', 'quantile_weighted_mse'],#['mse',  'huber'], #'focal_mse',
                    # 'n_bins' : [],
                    # 'gamma' : [1, 1.5], # Only used in focal_mse and binned_mse
                    # V3
                    'fourier_type' : ['none', 'basic'],#, 'positional', 'gaussian'], # , 
                    'fourier_mapping_size' : [16, 35],
                    'fourier_sigma' : [1, 4],
                    #v4
                    # 'engineer_time_features':[False, True],
                    # 'bin_yrblt':[True, False],
                    # 'cross_township_class':[True, False],
                    # v5
                    'dropout' : [0.0, 0.0],
                    'l1_lambda' : [0, 1e-2],
                    'l2_lambda' : [0, 1e0],
                    'use_scaler':[True],
                    'loss_alpha':[0,10],
                    'normalization_type':['none'],
                },
                'default': {
                    'learning_rate': 0.0007360519059468381,
                    'batch_size': 26,
                    'num_epochs': 172,
                    'hidden_sizes':[1796, 193, 140, 69],
                    'fourier_type': 'basic',# 'use_fourier_features': True,
                    'patience': 11,
                    'loss_fn': 'huber',
                    'gamma': 1.4092634199672638,
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
                    'learning_rate': [1e-4, 1e-3], 'batch_size': [100, 256],
                    'num_epochs': [100, 500], 'transformer_dim': [1, 8], # NOTE: heads as a divisor: transformer_dim = transformer_dim * heads
                    'transformer_heads': [4, 8], 'transformer_layers': [4, 8], 
                    'dropout':[0.05, 0.15], 'loss_fn': ['mse', 'focal_mse', 'huber'],
                    'patience':[8, 15], 
                    # v3
                    'fourier_type' : ['gaussian', 'none', 'basic','positional'], # 'basic',
                    'fourier_mapping_size' : [35, 55],
                    'fourier_sigma' : [3, 6],
                    # v4
                    'engineer_time_features':[False, True], 
                    'bin_yrblt':[True, False], 
                    'cross_township_class':[True, False],
                },
                'default': {
                    'learning_rate': 0.0008922322100000605,
                    'batch_size': 111,
                    'num_epochs': 385,
                    'transformer_dim': 6 * 4,
                    'transformer_heads': 4,
                    'transformer_layers': 6,
                    'dropout': 0.11710575577244148,
                    'loss_fn': 'mse',
                    'patience': 13,
                    'fourier_type': 'positional',
                    'fourier_mapping_size': 45,
                    'fourier_sigma': 5,
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
            # This new entry defines the search space for the RESAMPLER's parameters
            'ResamplingPipeline': {
                'range': {
                    'n_bins': [5, 500],
                    'binning_policy': ['uniform', 'quantile', 'outlier', 'kmeans', 'decision_tree'], # possible cause of error: 
                    'max_diff_ratio': [0.05, 0.95],
                    'undersample_policy': ['random', 'outlier', 'inlier', 'tomek_links'],
                    'oversample_policy': ['smotenc', 'generalized_smotenc', 'density_smotenc'], # possible cause of error: 
                    'smote_k_neighbors': [3, 30]
                },
                'default': { # Default resampler params if CV is off
                    'n_bins': 10, 'binning_policy': 'quantile', 'max_diff_ratio': 0.5,
                    'undersample_policy': 'outlier', 'oversample_policy': 'smoter',
                    'smote_k_neighbors': 5
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

if params['model']['name'] in ["LightGBM"]: #or (params['model']['name'] == 'ResamplingPipeline' and params['model']['base_model'] in ["LightGBM"]):
    print("LGBM pipeline")
    pipeline = ModelMainRecipe(
        outcome="meta_sale_price",
        pred_vars=params['model']['predictor']['all'],
        cat_vars=params['model']['predictor']['categorical'],
        id_vars=params['model']['predictor']['id']
    )
elif params['model']['name'] in ['TabTransformerRegressor'] or (params['model']['name'] == 'ResamplingPipeline' and params['model']['base_model'] in ["TabTransformerRegressor", "LightGBM"]):
    print("NOT LGBM pipeline, but yes if imputed")
    pipeline = ModelMainRecipeImputer(
        outcome="meta_sale_price",
        pred_vars=params['model']['predictor']['all'],
        cat_vars=params['model']['predictor']['categorical'],
        id_vars=params['model']['predictor']['id']
    )
elif params['model']['name'] in ["FeedForwardNNRegressorWithEmbeddings"] or (params['model']['name'] == 'ResamplingPipeline' and params['model']['base_model'] in ["FeedForwardNNRegressorWithEmbeddings"]):
    print("NOT LGBM pipeline")
    pipeline = build_model_pipeline_supress_onehot( # WARNING: We only changed to this to perform changes on the pipeline
            pred_vars=params['model']['predictor']['all'],
            cat_vars=params['model']['predictor']['categorical'],
            id_vars=params['model']['predictor']['id']
        )

print("ALGO")

handler = ModelHandler(
    model_name=model_name_to_run,
    model_params=model_params,
    hyperparameter_config=params['model']['hyperparameter'][model_name_to_run],
    base_model_name=params['model']['base_model'],
    base_model_hyperparameter_config=params['model']['hyperparameter'][params['model']['base_model']],
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
