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
from nn_models.unconstrained.BaselineModels4 import FeedForwardNNRegressorWithEmbeddings4
from nn_models.unconstrained.TabTransformerRegressor4 import TabTransformerRegressor4
from nn_models.unconstrained.WideAndDeepRegressor import WideAndDeepRegressor
from nn_models.unconstrained.TabNetRegressor import TabNetRegressor

# Custom utilities
from src_.custom_metrics import create_diagnostic_plots, compute_all_metrics
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

    return {
        "LightGBM": {
            "model_class": lgb.LGBMRegressor,
            "params": {
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
            "fit_params": {
                "eval_metric": 'rmse',
                "callbacks": [lgb.early_stopping(stopping_rounds=50, verbose=False)]
            },
            "preprocessor": "LightGBM"
        },
        "FeedForwardNN": {
            "model_class": FeedForwardNNRegressorWithEmbeddings4,
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
                'loss_fn': 'huber',
                'gamma':1.4092634199672638,
            },
            "preprocessor": "embedding_linear"
        },
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

def load_and_prepare_data(config: dict) -> tuple:
    """
    Loads, filters, samples, and splits the dataset into train, validation, and test sets.
    """
    print("Preparing model training data...")
    params = config['data_prep']
    
    # Corrected: Removed the ID columns from the desired columns list
    desired_columns = config['model']['predictor']['all'] + \
                      ['meta_sale_price', 'meta_sale_date', "ind_pin_is_multicard", "sv_is_outlier"]

    df = pd.read_parquet(params['input_path'], columns=desired_columns)
    df['char_recent_renovation'] = df['char_recent_renovation'].astype(bool)
    df = df[
        (~df['ind_pin_is_multicard'].astype('bool').fillna(True)) &
        (~df['sv_is_outlier'].astype('bool').fillna(True))
    ]

    if params['use_sample']:
        print(f"Using a sample of size: {params['sample_size']}")
        df = df.sample(params['sample_size'], random_state=params['seed'])
    else:
        print("Using full data.")

    df = df.sort_values('meta_sale_date')
    df['meta_sale_price'] = np.log(df['meta_sale_price'])

    # Time-based split: Train -> Val -> Test
    train_val_split_idx = int(len(df) * config['cv']['split_prop'])
    train_val = df.iloc[:train_val_split_idx]
    test = df.iloc[train_val_split_idx:]

    train_split_idx = int(len(train_val) * config['cv']['split_prop'])
    train = train_val.iloc[:train_split_idx]
    val = train_val.iloc[train_split_idx:]
    
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
# 4. Model Training and Evaluation (Refactored for Multi-Run)
# ==============================================================================

def train_and_evaluate(model_name: str, model_config: dict, data_splits: tuple, is_last_run: bool):
    """
    Initializes, trains, predicts, and evaluates a single model.
    Returns a dictionary of test set metrics.
    """
    print("-" * 50)
    print(f"Running model: {model_name}")
    print("-" * 50)
    
    X_train, y_train, X_val, y_val, X_test, y_test = data_splits
    
    # --- Model Initialization ---
    model = model_config['model_class'](**model_config['params'])
    
    # --- Training ---
    start_time = time()
    fit_params = model_config.get('fit_params', {})
    
    if model_name == "LightGBM":
        fit_params['eval_set'] = [(X_val, y_val)]
    
    if "FeedForward" in model_name or "TabTransformer" in model_name:
         model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    else:
        model.fit(X_train, y_train, **fit_params)

    print(f"Training for {model_name} finished in {time() - start_time:.2f} seconds.")

    # --- Prediction on Assessment (Test) Data ---
    y_pred_test_log = model.predict(X_test)
    
    # --- Post-processing ---
    p1, p99 = np.percentile(y_pred_test_log, [1, 99])
    np.clip(y_pred_test_log, p1, p99, out=y_pred_test_log)
        
    y_test_exp = np.exp(y_test)
    y_pred_test_exp = np.exp(y_pred_test_log)

    # --- Evaluation ---
    test_metrics = compute_all_metrics(y_test_exp, y_pred_test_exp)
    test_metrics['r2'] = r2_score(y_test_exp, y_pred_test_exp)

    # Corrected: Added fairness metrics to the per-run printout
    print(f"Assessment Metrics for {model_name}: RMSE={test_metrics['rmse']:.4f}, R²={test_metrics['r2']:.4f}, F_dev={test_metrics['f_dev']:.4f}, F_grp={test_metrics['f_grp']:.4f}")
    
    # --- Plotting (Only for the last run) ---
    if is_last_run:
        print(f"Generating diagnostic plots for {model_name} (final run)...")
        y_pred_train_log = model.predict(X_train)
        p1_train, p99_train = np.percentile(y_pred_train_log, [1, 99])
        np.clip(y_pred_train_log, p1_train, p99_train, out=y_pred_train_log)
        y_train_exp = np.exp(y_train)
        y_pred_train_exp = np.exp(y_pred_train_log)

        create_diagnostic_plots(
            y_train=y_train_exp, y_test=y_test_exp,
            y_pred_train=y_pred_train_exp, y_pred_test=y_pred_test_exp,
            suffix=f"_{model_name}_final_run"
        )
    
    return test_metrics

# ==============================================================================
# 5. Main Execution (Refactored for Multi-Run)
# ==============================================================================

def main():
    """Main script execution function."""
    config = get_config()
    main_seed = config['model']['seed']
    num_runs = config['data_prep'].get('num_evaluation_runs', 5) # Get number of runs, default to 5
    
    # Define which models to run from the configuration
    models_to_run = ["FeedForwardNN"]#["LightGBM", "FeedForwardNN", "TabTransformer"]
    
    # --- Data Pipeline (Run once) ---
    train_df, val_df, test_df = load_and_prepare_data(config)
    preprocessors = create_preprocessors(config)
    processed_data = preprocess_data(preprocessors, train_df, val_df, test_df)
    
    # --- Multi-Run Evaluation Loop ---
    all_run_metrics = {model_name: {metric: [] for metric in ['rmse', 'r2', 'f_dev', 'f_grp']} for model_name in models_to_run}

    # --- List of seed to use ---
    np.random.seed(42)
    seeds = np.random.randint(0, 1e4, num_runs)
    np.random.seed(None)

    for i in range(num_runs):
        # run_seed = main_seed + i
        run_seed = seeds[i]
        print(f"\n{'='*30} STARTING RUN {i+1}/{num_runs} (Seed: {run_seed}) {'='*30}\n")
        
        model_configs = get_model_configurations(config, run_seed)
        
        for name in models_to_run:
            if name in model_configs:
                model_config = model_configs[name]
                preprocessor_key = model_config['preprocessor']
                
                if preprocessor_key in processed_data:
                    # Pass a flag to indicate if it's the last run for plotting
                    is_last_run = (i == num_runs - 1)
                    test_metrics = train_and_evaluate(name, model_config, processed_data[preprocessor_key], is_last_run)
                    
                    # Store metrics for this run
                    for metric_name, value in test_metrics.items():
                        if metric_name in all_run_metrics[name]:
                            all_run_metrics[name][metric_name].append(value)
                else:
                    print(f"Warning: Preprocessor '{preprocessor_key}' not found for model '{name}'. Skipping.")
            else:
                print(f"Warning: Model '{name}' not found in configurations. Skipping.")

    # --- Final Results Aggregation ---
    print("\n" + "=" * 80)
    print(f"FINAL AGGREGATED RESULTS (ACROSS {num_runs} RUNS)")
    print("=" * 80)

    for model_name, metrics_dict in all_run_metrics.items():
        print(f"\n--- {model_name} Assessment Performance (Mean ± Std Dev) ---")
        for metric_name, values in metrics_dict.items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"{metric_name.upper():<8}: {mean_val:.4f} ± {std_val:.4f}")


if __name__ == "__main__":
    main()

