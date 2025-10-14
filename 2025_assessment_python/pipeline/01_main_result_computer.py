import pandas as pd
import numpy as np
import yaml
from time import time
import os
import sys
from math import floor, log2

# --- Environment Setup ---
# Add parent directories to the Python path to allow for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
grandparent_dir = os.path.dirname(parent_dir)
if grandparent_dir not in sys.path:
    sys.path.append(grandparent_dir)

# --- Model and Utility Imports ---
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score

# Custom models
# from nn_models.unconstrained.BaselineModels4 import FeedForwardNNRegressorWithEmbeddings4
from nn_models.unconstrained.BaselineModels5 import FeedForwardNNRegressorWithEmbeddings5
from nn_models.unconstrained.TabTransformerRegressor4 import TabTransformerRegressor4
from nn_models.unconstrained.WideAndDeepRegressor import WideAndDeepRegressor
from nn_models.unconstrained.TabNetRegressor import TabNetRegressor

# Residual models
from nn_models.unconstrained.BaselineModels6 import FeedForwardNNRegressorWithResiduals
from residual_models.ensemble_models import LGBMRegressorWithResiduals

# Classficiation models
from nn_models.unconstrained.DiscretizedNNClassifier import DiscretizedNNClassifier
from nn_models.unconstrained.DiscretizedNNClassifier2 import DiscretizedNNClassifier2
from nn_models.unconstrained.DiscretizedTabTransformerClassifier import DiscretizedTabTransformerClassifier

# Custom utilities
from balancing_models import BalancingResampler
from src_.custom_metrics import create_diagnostic_plots, compute_all_metrics
from src_.custom_metrics import AmpMSELoss, FocalMSELoss
from recipes.recipes_pipelined import build_model_pipeline, build_model_pipeline_supress_onehot, ModelMainRecipe, ModelMainRecipeImputer


# ==============================================================================
# 1. Configuration
# ==============================================================================

def get_config():
    """Loads and returns the main configuration from params.yaml."""
    with open('params_mine.yaml', 'r') as file:
        return yaml.safe_load(file)

def get_model_configurations(params: dict, seed: int) -> dict:
    """
    Defines all models to be tested, their parameters, and required preprocessor.
    """
    cat_vars = params['model']['predictor']['categorical']
    large_categories = ['meta_nbhd_code', 'meta_township_code', 'char_class'] + [c for c in params['model']['predictor']['all'] if c.startswith('loc_school_')]
    coord_vars = ["loc_longitude", "loc_latitude"]

    # LightGBM loss
    objective = AmpMSELoss(tau=10)
    # objective = FocalMSELoss(n_exp = 0, delta=1e-3, m_amp=1)

    return {
        "LightGBM": {
            "model_class": lgb.LGBMRegressor,
            "params": {
                # 'objective': 'huber',
                'objective':objective.loss_,
                'random_state': seed,
                'n_estimators': params['model']['hyperparameter']['default']['num_iterations'],
                'max_depth': floor(log2(167)) + params["model"]["hyperparameter"]["default"]['add_to_linked_depth'],
                'verbose': -1,
                'learning_rate': 0.011364186059408745,
                'max_bin': 178,
                'num_leaves': 167,
                'feature_fraction': 0.45512255576737853,
                'min_gain_to_split': 0.0015979247898933535,
                'min_data_in_leaf': 65,
                'max_cat_threshold': 51,
                'min_data_per_group': 339,
                'cat_smooth': 120.25007901496078,
                'cat_l2': 0.01382892133084935,
                'lambda_l1': 0.03360945895462623,
                'lambda_l2': 0.3003031020947675,
            },
            "fit_params": {
                # "eval_metric" : 'huber',
                "eval_metric": objective.eval_,#
                "callbacks": [lgb.early_stopping(stopping_rounds=50, verbose=False)]
            },
            "preprocessor": "LightGBM"
        },
        "LightGBMResiduals": {
            "model_class": LGBMRegressorWithResiduals,
            "params": {
                "random_state":seed,
                "residual_model_split": 0.3,
                "lgbm_params": {
                    'objective': 'rmse',
                    'random_state': seed,
                    'n_estimators': params['model']['hyperparameter']['default']['num_iterations'],
                    'max_depth': floor(log2(167)) + params["model"]["hyperparameter"]["default"]['add_to_linked_depth'],
                    'verbose': -1,
                    'learning_rate': 0.011364186059408745,
                    'max_bin': 178,
                    'num_leaves': 167,
                    'feature_fraction': 0.45512255576737853,
                    'min_gain_to_split': 0.0015979247898933535,
                    'min_data_in_leaf': 65,
                    'max_cat_threshold': 51,
                    'min_data_per_group': 339,
                    'cat_smooth': 120.25007901496078,
                    'cat_l2': 0.01382892133084935,
                    'lambda_l1': 0.03360945895462623,
                    'lambda_l2': 0.3003031020947675,
                    },
            },
            "preprocessor": "LightGBM",            
        },
        "FeedForwardNNResiduals":{
            "model_class": FeedForwardNNRegressorWithResiduals,
            "params": {
                'learning_rate': 0.0007360519059468381,
                'categorical_features': large_categories,
                'coord_features': coord_vars,
                'random_state': seed,
                'batch_size': 26,
                'num_epochs': 172,
                'hidden_sizes': [1796, 193, 140, 69],
                'fourier_type': 'basic',
                'patience': 11,
                'gamma': 0, #1.4092634199672638, # focal mse gamma
                'loss_fn': 'huber',#'focal_mse',#'quantile_weighted_mse',
                # Mine
                'dropout': 0,
                # 'dropout': 0.0007693680499241461,
                # 'l1_lambda': 2.714100311651801e-06,
                # 'l2_lambda': 0.0031372889601764937,
                'use_scaler': True,
                'loss_alpha': 35,
                'normalization_type': 'none', #'none', 'batch_norm', or 'layer_norm'
                # Residuals
                'residual_model_split': 0.3, 
                'num_residual_epochs': 200, 
                'validation_split': 0.1, 
                'residual_loss_fn': 'huber'
            },
            "preprocessor": "embedding_linear"
        },
        "FeedForwardNN": {
            "model_class": FeedForwardNNRegressorWithEmbeddings5,
            "params": {
                'learning_rate': 0.004,#0.0007360519059468381,
                'categorical_features': large_categories,
                'coord_features': coord_vars,
                'random_state': seed,
                'batch_size': 512,#4*2048,#64,#26,
                'num_epochs': 200,#172,
                'hidden_sizes': [512*2**(i) for i in range(4)],#[1024*4, 1024*2, 1024, 512, 256, 128, 64, 32], #[1796, 193, 140, 69],
                'fourier_type': 'basic',
                'patience': 11,
                # 'gamma': 5, #1.4092634199672638, # focal mse gamma
                'loss_fn': 'huber',#'focal_mse',#'quantile_weighted_mse',
                # Mine
                'dropout': 0,
                # 'dropout': 0.0007693680499241461,
                # 'l1_lambda': 2.714100311651801e-06,
                # 'l2_lambda': 0.0031372889601764937,
                'use_scaler': True,
                'loss_alpha': 35,
                'normalization_type': 'none' #'none', 'batch_norm', or 'layer_norm'
            },
            "preprocessor": "embedding_linear"
        },
        "DiscretizedNNClassifier":{
            "model_class": DiscretizedNNClassifier2,
            "params": {
                'learning_rate': 0.004,#0.0007360519059468381,
                'categorical_features': large_categories,
                'coord_features': coord_vars,
                'random_state': seed,
                'batch_size': 1024,#4*2048,#2*16*2048,#512,#26,
                'num_epochs': 200,#172,
                'hidden_sizes': [512*2**(i) for i in range(4)][::-1], #[1024*4, 1024*2, 1024, 512],#[1796, 193, 140, 69],
                'fourier_type': 'positional',
                'fourier_mapping_size': 32, # WARNING: default=16
                'fourier_sigma': 1.25, # WARNING: default=1.25
                'patience': 10,
                # Mine
                'dropout': 1e-6, #0.0007693680499241461,
                'l1_lambda': 1e-4,#2.714100311651801e-06,
                'l2_lambda': 1e-3,#0.0031372889601764937,
                'use_scaler': True,
                'normalization_type': 'none', #'none', 'batch_norm', or 'layer_norm'
                # Classification
                "n_bins": 50,
                "binning_method":'uniform',
                'min_samples_per_bin':4,
                # v2:  loss controls
                "loss_mode":'ce', 
                "val_loss_mode":'ce', 
                "huber_delta":1.0, 
                "smoothing_sigma":1,#0.0, # standard deviation of Gaussian used to generate soft targets for prob_mse and smooth_ce. Controls how much probability mass spreads to neighboring bins.
                "ce_label_smoothing":0.0, # passes label smoothing to PyTorch’s CrossEntropyLoss (only used if loss_mode='ce').
                "use_class_weights": True, # if True, computes inverse-frequency weights per class and applies them in CE loss. Helps counteract class imbalance.
                "weight_exp": 1, # Between 0.5 and 1 for uniform w/o resamp  # Weight exponent to add to the class weight (MINE)
                "default_predict_mode":'expected', # determines what predict() returns: 'expected': regression-style output using weighted sum of bin centers. 'argmax': class → center (your original behavior).
            },
            "preprocessor": "embedding_linear"
        },
            # loss_mode ∈ {
            #     'ce',            # standard cross-entropy
            #     'ev_mse',        # [this ones suck?]expected value (p@bin_values) vs y  with MSE
            #     'ev_mae',        # [this ones suck?]... with MAE
            #     'ev_huber',      # [this ones suck?]... with Huber (delta=huber_delta)
            #     'emd',           # [this ones suck?]Earth Mover's Distance (Wasserstein-1 on ordered bins)
            #     'prob_mse',      # MSE between p and (one-hot or gaussian-smoothed) target
            #     'smooth_ce'      # cross-entropy with distance-based soft targets
            # }
        "TabTransformer": {
            "model_class": TabTransformerRegressor4,
            "params": {
                'categorical_features': cat_vars,
                'coord_features': coord_vars,
                'random_state': seed,
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
                'fourier_sigma': 5
            },
            "preprocessor": "embedding"
        },
        "DiscretizedTabTransformerClassifier":{
            "model_class": DiscretizedTabTransformerClassifier,
            "params": {
                'learning_rate': 0.0008922322100000605,
                'categorical_features': cat_vars,
                'coord_features': coord_vars,
                'random_state': seed,
                'batch_size': 2*2048,#4*2048,#2*16*2048,#512,#26,
                'num_epochs': 400,#172,
                # 'hidden_sizes': [512*2**(i) for i in range(4)], #[1024*4, 1024*2, 1024, 512],#[1796, 193, 140, 69],
                'patience': 11,
                # Loss
                # 'loss_fn': 'focal_mse',#'quantile_weighted_mse',
                # 'gamma': 0, #1.4092634199672638, # focal mse gamma
                # 'loss_alpha': 35,
                # Transformer
                'd_model': 128,#6 * 4,
                'nhead': 16,#4,
                'num_layers': 6,#6,
                # Fourier features
                'fourier_type': 'positional',
                'fourier_mapping_size': 45,
                'fourier_sigma': 5,
                # Regularization
                'dropout': 0.11,
                # 'dropout': 1e-6, #0.0007693680499241461,
                # 'l1_lambda': 1e-6,#2.714100311651801e-06,
                # 'weight_decay': 1e-4,#0.0031372889601764937, # l2_lambda here
                'use_scaler': True,
                # Classification
                "n_bins":50,
                "binning_method":'uniform',#'quantile',
                'min_samples_per_bin':4,
                # v2:  loss controls
                "loss_mode":'ce', 
                "val_loss_mode":'ce',
                "huber_delta":1.0, 
                "smoothing_sigma":1,#0.0, # standard deviation of Gaussian used to generate soft targets for prob_mse and smooth_ce. Controls how much probability mass spreads to neighboring bins.
                "ce_label_smoothing":0.0, # passes label smoothing to PyTorch’s CrossEntropyLoss (only used if loss_mode='ce').
                "use_class_weights":True, # if True, computes inverse-frequency weights per class and applies them in CE loss. Helps counteract class imbalance.
                "weight_exp": 0.75,
#                 "class_weights": [
#    193.86713 ,    43.08166  ,   28.72114  ,   17.624374 ,   12.309117,
#    9.231863  ,   8.429101   ,  4.1469884  ,  4.0180693  ,  3.3864233,
#    2.7210405 ,   2.4855745  ,  2.178381   ,  1.6895729  ,  1.8035141,
#    1.3166842 ,   1.1388197  ,  1.4126102  ,  1.0085111  ,  0.71613705,
#    0.53197116,   0.42408472 ,  0.3302269  ,  0.2966461  ,  0.23566139,
#    0.22898668,   0.27190793 ,  0.30313557 ,  0.35861504 ,  0.4838605,
#    0.585359  ,   0.7063551  ,  0.9265852  ,  1.1158814  ,  1.4715765,
#    2.215723  ,   2.4158883  ,  3.1782477  ,  4.6436214  ,  6.5718637,
#   10.479399  ,  17.624374   , 23.499132   , 29.825794   , 59.65149,
#   96.93361   , 193.86713   
#   ],
                "default_predict_mode":'expected', # determines what predict() returns: 'expected': regression-style output using weighted sum of bin centers. 'argmax': class → center (your original behavior).
            },
            "preprocessor": "embedding"
        },
        "LinearRegression": {
            "model_class": LinearRegression,
            "params": {},
            "fit_params": {},
            "preprocessor": "linear"
        }
    }

# ==============================================================================
# 2. Data Loading and Preparation
# ==============================================================================

def load_and_prepare_data(config: dict, test_on_val=False) -> tuple:
    """
    Loads, filters, samples, and splits the dataset into train, validation, and test sets.
    """
    print("Preparing model training data...")
    params = config['data_prep']
    
    desired_columns = config['model']['predictor']['all'] + \
                      ['meta_sale_price', 'meta_sale_date', "ind_pin_is_multicard", "sv_is_outlier"]

    df = pd.read_parquet(params['input_path'], columns=desired_columns)
    df['char_recent_renovation'] = df['char_recent_renovation'].astype(bool)
    df = df[
        (~df['ind_pin_is_multicard'].astype('bool').fillna(True)) &
        (~df['sv_is_outlier'].astype('bool').fillna(True))
    ]

    df = df.sort_values('meta_sale_date')
    df['meta_sale_price'] = np.log(df['meta_sale_price'])

    # Time-based split: Train -> Val -> Test
    train_val_split_idx = int(len(df) * config['cv']['split_prop'])
    train_val = df.iloc[:train_val_split_idx]
    test = df.iloc[train_val_split_idx:]

    train_split_idx = int(len(train_val) * config['cv']['split_prop'])
    train = train_val.iloc[:train_split_idx]
    val = train_val.iloc[train_split_idx:]

    if test_on_val:
        # split now the trainint+validation
        test = val.copy() # val is the new test
        train_split_idx = int(len(train) * config['cv']['split_prop'])
        val = train.iloc[train_split_idx:]
        train = train.iloc[:train_split_idx]

    # Use a sample of the training data
    if params['use_sample']:
        print(f"Using a sample of size: {params['sample_size']}")
        train = train.sample(int(params['sample_size']*config['cv']['split_prop']**(2+test_on_val)), random_state=params['seed'])
        val = val.sample(int(params['sample_size']*config['cv']['split_prop']**(1+test_on_val)*(1-config['cv']['split_prop'])), random_state=params['seed'])
        print("Size of train: ", train.shape)
        print("Size of val: ", val.shape)
        print("Size of test: ", test.shape)
    else:
        print("Using full data.")
    
    return train, val, test

# ==============================================================================
# 3. Preprocessing
# ==============================================================================

def create_preprocessors(config: dict) -> dict:
    """Creates and returns a dictionary of scikit-learn preprocessing pipelines."""
    params = config['model']
    return {
        "linear": build_model_pipeline(
            pred_vars=params['predictor']['all'],
            cat_vars=params['predictor']['categorical'],
            id_vars=params['predictor']['id']
        ),
        "embedding_linear":build_model_pipeline_supress_onehot(
            pred_vars=params['predictor']['all'],
            cat_vars=params['predictor']['categorical'],
            id_vars=params['predictor']['id']
        ),
        "embedding": ModelMainRecipeImputer(
            outcome="meta_sale_price",
            pred_vars=params['predictor']['all'],
            cat_vars=params['predictor']['categorical'],
            id_vars=params['predictor']['id']
        ),
        "LightGBM":ModelMainRecipe(
            outcome= "meta_sale_price",
            pred_vars=params['predictor']['all'],
            cat_vars=params['predictor']['categorical'],
            id_vars=params['predictor']['id']
        ),
    }

def preprocess_data(preprocessors: dict, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    Fits preprocessors on training data and transforms all data splits.
    """
    print("Applying preprocessing pipelines...")
    processed_data = {}
    
    for name, pipeline in preprocessors.items():
        y_train = train_df['meta_sale_price']
        X_train = train_df.drop(columns=['meta_sale_price'])
        
        y_val = val_df['meta_sale_price']
        X_val = val_df.drop(columns=['meta_sale_price'])
        
        y_test = test_df['meta_sale_price']
        X_test = test_df.drop(columns=['meta_sale_price'])
        
        X_train_fit = pipeline.fit_transform(X_train, y_train).drop(columns=['meta_sale_date'], errors='ignore')
        X_val_fit = pipeline.transform(X_val).drop(columns=['meta_sale_date'], errors='ignore')
        X_test_fit = pipeline.transform(X_test).drop(columns=['meta_sale_date'], errors='ignore')
        
        processed_data[name] = (X_train_fit, y_train, X_val_fit, y_val, X_test_fit, y_test)
        
    return processed_data

# ==============================================================================
# 4. Model Training and Evaluation (Refactored for Multi-Split Metrics)
# ==============================================================================

def train_and_evaluate(model_name: str, model_config: dict, data_splits: tuple, is_last_run: bool, resampling_seed=None, plot_suffix='', resample=False):
    """
    Initializes, trains, predicts, and evaluates a single model.
    Returns a dictionary of metrics for train, val, and test sets.
    """
    print("-" * 50)
    print(f"Running model: {model_name}")
    print("-" * 50)
    
    X_train, y_train, X_val, y_val, X_test, y_test = data_splits

    # MINE: resampling
    config = get_config()
    print("Checking if we do resampling: ", config['data_prep'].get('apply_resampling', False))
    if resample:#config['data_prep'].get('apply_resampling', False):
        if model_name in ["FeedForwardNN", "DiscretizedNNClassifier"]:
            cat_vars = ['meta_nbhd_code', 'meta_township_code', 'char_class'] + [c for c in config['model']['predictor']['all'] if c.startswith('loc_school_')]
            cat_vars = [X_train.columns.tolist().index(col) for col in cat_vars] # Large vars
            print("We just selected the number of the cols that are features")
        elif model_name in ["TabTransformer","DiscretizedTabTransformerClassifier", "LightGBM"]:
            cat_vars = config['model']['predictor']['categorical']
            cat_vars = [X_train.columns.tolist().index(col) for col in cat_vars] # Large vars
            print("We just selected the number of the cols that are features")
        else:
            raise print("NO MODEL NAMED: ", model_name)
        # cat_vars = [X_train.columns.tolist().index(col) for col in config['model']['predictor']['categorical']]
        resampler = BalancingResampler(
            n_bins= 200,#68, # Between 100 and 500
            binning_policy= 'kmeans',#'quantile',
            max_diff_ratio= 0.27657655868482417, # between 0.2 and 0.3
            undersample_policy= 'tomek_links',#'random', # between tomek_links and inlier
            oversample_policy= 'smotenc', # Between smotenc and density
            # generalized_smote_weighting='gravity',
            smote_k_neighbors= 16,
            categorical_features=cat_vars,
            low_count_policy=None,
            random_state=config['data_prep'].get('seed', None),
        )
        print("We are going to resample")
        X_train, y_train = resampler.fit_resample(X_train, y_train)
    # END MINE

    
    # --- Model Initialization ---
    model = model_config['model_class'](**model_config['params'])
    
    # --- Training ---
    start_time = time()
    fit_params = model_config.get('fit_params', {})
    
    if model_name == "LightGBM":
        fit_params['eval_set'] = [(X_val, y_val)]
    
    if model_name != "LightGBM":#"FeedForward" in model_name or "TabTransformer" in model_name:
         model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    else:
        model.fit(X_train, y_train, **fit_params)

    print(f"Training for {model_name} finished in {time() - start_time:.2f} seconds.")

    # --- Prediction on All Splits ---
    print("Generating predictions on all data splits...")
    y_pred_train_log = model.predict(X_train)
    y_pred_val_log = model.predict(X_val)
    y_pred_test_log = model.predict(X_test)

    # --- Helper for Post-processing and Evaluation ---
    def postprocess_and_evaluate(y_true_log, y_pred_log):
        p1, p99 = np.percentile(y_pred_log, [1, 99])
        np.clip(y_pred_log, p1, p99, out=y_pred_log)
        y_true_exp = np.exp(y_true_log)
        y_pred_exp = np.exp(y_pred_log)
        metrics = compute_all_metrics(y_true_exp, y_pred_exp)
        metrics['r2'] = r2_score(y_true_exp, y_pred_exp)
        return metrics

    # --- Evaluation for all splits ---
    train_metrics = postprocess_and_evaluate(y_train, y_pred_train_log)
    val_metrics = postprocess_and_evaluate(y_val, y_pred_val_log)
    test_metrics = postprocess_and_evaluate(y_test, y_pred_test_log)

    print(f"  Train Metrics:    RMSE={train_metrics['rmse']:.4f}, R²={train_metrics['r2']:.4f}, F_dev={train_metrics['f_dev']:.4f}, F_grp={train_metrics['f_grp']:.4f}")
    print(f"  Validation Metrics: RMSE={val_metrics['rmse']:.4f}, R²={val_metrics['r2']:.4f}, F_dev={val_metrics['f_dev']:.4f}, F_grp={val_metrics['f_grp']:.4f}")
    print(f"  Assessment Metrics: RMSE={test_metrics['rmse']:.4f}, R²={test_metrics['r2']:.4f}, F_dev={test_metrics['f_dev']:.4f}, F_grp={test_metrics['f_grp']:.4f}")
    
    # --- Plotting (Only for the last run, uses original train/test preds) ---
    if is_last_run:
        print(f"Generating diagnostic plots for {model_name} (final run)...")
        y_train_exp = np.exp(y_train)
        y_pred_train_exp = np.exp(y_pred_train_log)
        y_test_exp = np.exp(y_test)
        y_pred_test_exp = np.exp(y_pred_test_log)
        create_diagnostic_plots(
            y_train=y_train_exp, y_test=y_test_exp,
            y_pred_train=y_pred_train_exp, y_pred_test=y_pred_test_exp,
            suffix=f"_{model_name}_final_run" + plot_suffix,
            save_plots=config['plotting'].get('save_plots', True), 
            log_scale=config['plotting'].get('log_scale', False),
        )
    
    return {'train': train_metrics, 'val': val_metrics, 'test': test_metrics}

# ==============================================================================
# 5. Main Execution (Refactored for Multi-Run and Multi-Split)
# ==============================================================================

def main():
    """Main script execution function."""
    config = get_config()
    main_seed = config['model']['seed']
    num_runs = config['data_prep'].get('num_evaluation_runs', 5)
    
    models_to_run = ["DiscretizedNNClassifier"]#["DiscretizedTabTransformerClassifier"]#["TabTransformer"]#["DiscretizedNNClassifier"]#["DiscretizedNNClassifier"]# ["LightGBMResiduals"]#["FeedForwardNNResiduals"]#["FeedForwardNN", "LightGBM"]
    
    train_df, val_df, test_df = load_and_prepare_data(config, test_on_val=config['data_prep'].get('test_on_val', False))
    preprocessors = create_preprocessors(config)
    processed_data = preprocess_data(preprocessors, train_df, val_df, test_df)
    
    # Restructure to hold metrics for each split
    all_run_metrics = {
        model_name: {
            split: {metric: [] for metric in ['rmse', 'r2', 'f_dev', 'f_grp']}
            for split in ['train', 'val', 'test']
        }
        for model_name in models_to_run
    }

    np.random.seed(42)
    seeds = np.random.randint(0, 1e4, num_runs)
    np.random.seed(None)

    for i in range(num_runs):
        run_seed = seeds[i]
        print(f"\n{'='*30} STARTING RUN {i+1}/{num_runs} (Seed: {run_seed}) {'='*30}\n")
        
        model_configs = get_model_configurations(config, run_seed)
        
        for name in models_to_run:
            if name in model_configs:
                model_config = model_configs[name]
                preprocessor_key = model_config['preprocessor']
                
                if preprocessor_key in processed_data:
                    is_last_run = (i == num_runs - 1)
                    all_metrics = train_and_evaluate(name, model_config, processed_data[preprocessor_key], is_last_run, plot_suffix=config['plotting'].get('save_suffix', ''), resample=config['data_prep'].get('apply_resampling', False))
                    
                    # Store metrics for each split
                    for split_name, metrics in all_metrics.items():
                        for metric_name, value in metrics.items():
                            if metric_name in all_run_metrics[name][split_name]:
                                all_run_metrics[name][split_name][metric_name].append(value)
                else:
                    print(f"Warning: Preprocessor '{preprocessor_key}' not found for model '{name}'. Skipping.")
            else:
                print(f"Warning: Model '{name}' not found in configurations. Skipping.")

    # --- Final Results Aggregation ---
    print("\n" + "=" * 80)
    print(f"FINAL AGGREGATED RESULTS (ACROSS {num_runs} RUNS)")
    print("=" * 80)

    for model_name, splits_dict in all_run_metrics.items():
        print(f"\n--- {model_name} Performance (Mean ± Std Dev) ---")
        for split_name, metrics_dict in splits_dict.items():
            print(f"  Split: {split_name.capitalize()}")
            for metric_name, values in metrics_dict.items():
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    print(f"    {metric_name.upper():<8}: {mean_val:.4f} ± {std_val:.4f}")


if __name__ == "__main__":
    main()

