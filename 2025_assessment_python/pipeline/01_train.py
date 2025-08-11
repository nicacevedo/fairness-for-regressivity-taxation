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




# Models 
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import lightgbm as lgb
import optuna
from sklearn.model_selection import KFold, TimeSeriesSplit

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# My imports
from src.nn_unconstrained import FeedForwardNNRegressor
from R.recipes import model_main_pipeline, model_lin_pipeline, my_model_lin_pipeline
from src.util_functions import compute_haihao_F_metrics

# Load YAML params file
with open('params.yaml', 'r') as file:
    params = yaml.safe_load(file)

# Inputs
assessment_year = 2025

use_sample = True
sample_size = 10000

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
training_data_full = training_data_full.sort_values('meta_sale_date') # Sort by 'meta_sale_date'

if use_sample:
    print("I am using a sample")
    training_data_full = training_data_full.sample(sample_size, random_state=42)
else:
    print("I am using full data")

# Create train/test split by time, with most recent observations in the test set
# We want our best model(s) to be predictive of the future, since properties are
# assessed on the basis of past sales
split_prop = params['cv']['split_prop']
split_index = int(len(training_data_full) * split_prop)
train = training_data_full.iloc[:split_index] # Split by time
test = training_data_full.iloc[split_index:]

# Create a recipe for the training data which removes non-predictor columns and
# preps categorical data, see R/recipes.R for details
train_pipeline, X, y, train_IDs = model_main_pipeline(
    data=training_data_full,
    pred_vars=params['model']['predictor']['all'],
    cat_vars=params['model']['predictor']['categorical'],
    id_vars=params['model']['predictor']['id']
)
# # Fit and transform training data (applies categorical processing, etc.)
# X_transformed = pd.DataFrame(train_pipeline.fit_transform(X), index=X.index)




#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 3. Linear Model --------------------------------------------------------------
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
print("Creating and fitting linear baseline model")

# Create a linear model recipe with additional imputation, transformations,
# and feature interactions
training_data_full_log = training_data_full.copy()
training_data_full_log['meta_sale_price'] = np.log(training_data_full_log['meta_sale_price'])

from time import time

# t0 = time()
X_train_prep, y_train_log, train_IDs = model_lin_pipeline(
    data=training_data_full_log,
    pred_vars=params['model']['predictor']['all'],
    cat_vars=params['model']['predictor']['categorical'],
    id_vars=params['model']['predictor']['id']
)
# Create a linear model specification and workflow
lin_model = LinearRegression()
# Wrap preprocessing and model into a pipeline (equivalent to workflow + recipe)
lin_pipeline = make_pipeline(
    lin_model  # you can insert preprocessing here if needed
)
# Fit the linear model on the training data
X_train_fit, y_train_fit_log = X_train_prep.loc[train.index, ], y_train_log.loc[train.index]
lin_pipeline.fit(X_train_fit, y_train_fit_log)
# print("Time 1: ", time() - t0)

# MINE: PREDICT IN TRAIN
y_pred_train_log = lin_pipeline.predict(X_train_fit)
print("RMSE train log:", np.sqrt(mean_squared_error(y_pred_train_log, y_train_fit_log)))
print("RMSE train:", np.sqrt(mean_squared_error(np.exp(y_pred_train_log), np.exp(y_train_fit_log))))
# MINE PREDICT IN TEST
X_test_fit, y_test_fit_log = X_train_prep.loc[test.index, :], y_train_log.loc[test.index]
y_pred_test_log = lin_pipeline.predict(X_test_fit)
print("RMSE test log:", np.sqrt(mean_squared_error(y_pred_test_log, y_test_fit_log)))
print("RMSE test:", np.sqrt(mean_squared_error(np.exp(y_pred_test_log), np.exp(y_test_fit_log))))




# MINE
# for col in X_train ## CHECK TYPE OF COLUMNS WHICH SHOULD BE NUMERIC  
print(X_train_fit.shape)
model = FeedForwardNNRegressor(
    input_features=X_train_fit.shape[1], output_size=1,  
    batch_size=16, learning_rate=0.001, num_epochs=80,
    hidden_sizes=[200, 100]
)
model.fit(X_train_fit, y_train_fit_log)
# MINE: PREDICT IN TRAIN
y_pred_train_log = model.predict(X_train_fit)
print("RMSE train log:", np.sqrt(mean_squared_error(y_pred_train_log, y_train_fit_log)))
print("RMSE train:", np.sqrt(mean_squared_error(np.exp(y_pred_train_log), np.exp(y_train_fit_log))))
# MINE PREDICT IN TEST
X_test_fit, y_test_fit_log = X_train_prep.loc[test.index, :], y_train_log.loc[test.index]
y_pred_test_log = model.predict(X_test_fit)
print("RMSE test log:", np.sqrt(mean_squared_error(y_pred_test_log, y_test_fit_log)))
print("RMSE test:", np.sqrt(mean_squared_error(np.exp(y_pred_test_log), np.exp(y_test_fit_log))))


# END MINE


# exit()


# # MINE
# for col in X.columns:
#     if X[col].dtype == "object":
#         print("col object: ", col, X[col].unique())
#         X[col] = X[col].astype("category")
# X_train, y_train = X.loc[train.index, :], y.loc[train.index]
# X_test, y_test = X.loc[test.index, :], y.loc[test.index]
# y_train_log, y_test_log = np.log(y_train), np.log(y_test)

print("="*100)
print("GBM:")
gbm_pipeline, X_train_prep, y_train_log, train_IDs = model_main_pipeline(
    data=training_data_full_log,
    pred_vars=params['model']['predictor']['all'],
    cat_vars=params['model']['predictor']['categorical'],
    id_vars=params['model']['predictor']['id']
)
print(X_train_prep.head())
X_train_prep = gbm_pipeline.fit_tranform(X_train_prep) # apply the pipeline
print(X_train_prep.head())
exit()

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

model = lgb.LGBMRegressor(
    n_estimators=n_estimators_static,
    # learning_rate=0.1,
    # random_state=42,
    # CCAO's: in between their range
    # num_leaves=1000,
    # add_to_linked_depth=4,
    # feature_fraction=0.5,
    # min_gain_to_split=10,
    # min_data_in_leaf=100,
    # ax_cat_threshold=100,
    # min_data_per_group=100,
    # cat_smooth=100,
    # cat_l2=1,
    # lambda_l1=1,
    # lambda_l2=1,


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
    num_leaves=int(params_dict["num_leaves"]),
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
    # n_estimators=int(params_dict['n_estimators'])
    # if params_dict.get('n_estimators') is not None
    # else (int(params_dict["n_estimators_static"]) if params_dict["n_estimators_static"] is not None else 1000)
)
model.fit(
    X_train_fit, y_train_fit_log,
    eval_set=[(X_test_fit, y_test_fit_log)],
    eval_metric='rmse', 
    callbacks=[
        lgb.early_stopping(stopping_rounds=stop_iter),  # Early stopping here
        lgb.log_evaluation(0)  # Suppress logging (use 1 for logging every round)
    ]
)


y_pred_train_log = model.predict(X_train_fit)
print("RMSE train log:", np.sqrt(mean_squared_error(y_pred_train_log, y_train_fit_log)))
print("RMSE train:", np.sqrt(mean_squared_error(np.exp(y_pred_train_log), np.exp(y_train_fit_log))))
# MINE PREDICT IN TEST
X_test_fit, y_test_fit_log = X_train_prep.loc[test.index, :], y_train_log.loc[test.index]
y_pred_test_log = model.predict(X_test_fit)
print("RMSE test log:", np.sqrt(mean_squared_error(y_pred_test_log, y_test_fit_log)))
print("RMSE test:", np.sqrt(mean_squared_error(np.exp(y_pred_test_log), np.exp(y_test_fit_log))))

print("="*100)


# Prediction
y_pred_train = np.exp(model.predict(X_train_fit))
y_pred_test = np.exp(model.predict(X_test_fit))

# --- Evaluate Performance ---
# For regression, we can use metrics like Mean Squared Error (MSE).
train_mse = mean_squared_error(y_pred_train, np.exp(y_train_fit_log))
test_mse = mean_squared_error(y_pred_test, np.exp(y_test_fit_log))

# ratio
n_groups_, alpha_ = 3, 2 
r_pred_train = y_pred_train / np.exp(y_train_fit_log)
r_pred_test = y_pred_test / np.exp(y_test_fit_log)
f_metrics_train = compute_haihao_F_metrics(r_pred_train, np.exp(y_train_fit_log), n_groups=n_groups_, alpha=alpha_)
f_metrics_test = compute_haihao_F_metrics(r_pred_test, np.exp(y_test_fit_log), n_groups=n_groups_, alpha=alpha_)

print(f"RMSE train: {np.sqrt(train_mse):.3f}")
print(f"RMSE val: {np.sqrt(test_mse):.3f}")
print(fr"$R^2$ train: {r2_score(np.exp(y_train_fit_log), y_pred_train):.3f}")
print(fr"$R^2$ val: {r2_score(np.exp(y_test_fit_log), y_pred_test):.3f}")
print(fr"$F_dev$ ({alpha_}) train: {f_metrics_train['f_dev']:.3f}")
print(fr"$F_dev$ ({alpha_}) test: {f_metrics_test['f_dev']:.3f}")
print(fr"$F_grp$ ({n_groups_}) train: {f_metrics_train['f_grp']:.3f}")
print(fr"$F_grp$ ({n_groups_}) test: {f_metrics_test['f_grp']:.3f}")

exit()
# # END MINE


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
    training_data_full_log = training_data_full.copy()
    X_full = training_data_full_log #.drop(["meta_sale_price"])  # <- provide your feature frame for training (preprocessed or to-be preprocessed consistently)
    y_full = np.log(training_data_full_log['meta_sale_price'])  # <- corresponding target (numeric)
    if validation_type == 'random':
        kf = KFold(n_splits=params['cv']['num_folds'], shuffle=True, random_state=seed)
        folds = list(kf.split(X_full))
    else:
        # recent / rolling-origin
        df_sorted, splits = rolling_origin_splits(
            train, v=params['cv']['num_folds'],
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
        # n_startup_trials=params['cv']['initial_set'], # Error
        show_progress_bar=True
    )
    best_params = study.best_params

    # Materialize final model with best params
    # Trees policy: if early stopping, keep fixed upper bound; else use tuned n_estimators
    final_hp = {
        **best_params,
        'n_estimators': best_params.get('n_estimators', n_estimators_static)
    }
    best_model = make_lgbm(final_hp)

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
