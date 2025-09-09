import pandas as pd
import numpy as np
import yaml
from time import time
import os
import sys
from math import floor, log2
import matplotlib.pyplot as plt

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
from sklearn.metrics import r2_score

# Custom models
from nn_models.unconstrained.BaselineModels4 import FeedForwardNNRegressorWithEmbeddings4
from nn_models.unconstrained.TabTransformerRegressor4 import TabTransformerRegressor4

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
                'objective': 'rmse', 'random_state': seed, 'n_estimators': params['model']['hyperparameter']['default']['num_iterations'],
                'max_depth': floor(log2(167)) + params["model"]["hyperparameter"]["default"]['add_to_linked_depth'], 'verbose': -1,
                'learning_rate': 0.01136, 'max_bin': 178, 'num_leaves': 167, 'feature_fraction': 0.4551,
                'min_gain_to_split': 0.00159, 'min_data_in_leaf': 65, 'max_cat_threshold': 51,
                'min_data_per_group': 339, 'cat_smooth': 120.25, 'cat_l2': 0.0138,
                'lambda_l1': 0.0336, 'lambda_l2': 0.3003,
            },
            "fit_params": {"eval_metric": 'rmse', "callbacks": [lgb.early_stopping(stopping_rounds=50, verbose=False)]},
            "preprocessor": "LightGBM"
        },
        "FeedForwardNN": {
            "model_class": FeedForwardNNRegressorWithEmbeddings4,
            "params": {
                'learning_rate': 0.000736, 'categorical_features': large_categories, 'coord_features': coord_vars,
                'random_state': seed, 'batch_size': 26, 'num_epochs': 172, 'hidden_sizes': [1796, 193, 140, 69],
                'fourier_type': 'basic', 'patience': 11, 'loss_fn': 'huber', 'gamma': 1.409,
            },
            "preprocessor": "embedding_linear"
        },
        "TabTransformer": {
            "model_class": TabTransformerRegressor4,
            "params": {
                'categorical_features': cat_vars, 'coord_features': coord_vars, 'random_state': seed,
                'learning_rate': 0.000892, 'batch_size': 111, 'num_epochs': 385, 'transformer_dim': 24,
                'transformer_heads': 4, 'transformer_layers': 6, 'dropout': 0.117, 'loss_fn': 'mse',
                'patience': 13, 'fourier_type': 'positional', 'fourier_mapping_size': 45, 'fourier_sigma': 5.0
            },
            "preprocessor": "embedding"
        }
    }

# ==============================================================================
# 2. Data Loading and Preparation
# ==============================================================================

def load_and_prepare_data(config: dict) -> tuple:
    """Loads data and performs the initial, fixed split into a training pool and assessment set."""
    print("Loading and preparing initial data splits...")
    params = config['data_prep']
    
    desired_columns = config['model']['predictor']['all'] + \
                      ['meta_sale_price', 'meta_sale_date', "ind_pin_is_multicard", "sv_is_outlier"]

    df = pd.read_parquet(params['input_path'], columns=desired_columns)
    df['char_recent_renovation'] = df['char_recent_renovation'].astype(bool)
    df = df[(~df['ind_pin_is_multicard'].astype('bool').fillna(True)) & (~df['sv_is_outlier'].astype('bool').fillna(True))]

    df = df.sort_values('meta_sale_date')
    df['meta_sale_price'] = np.log(df['meta_sale_price'])

    # Time-based split: The assessment (test) set is fixed. The rest is the pool for training/validation.
    train_val_split_idx = int(len(df) * config['cv']['split_prop'])
    train_val_pool = df.iloc[:train_val_split_idx]
    assessment_set = df.iloc[train_val_split_idx:]
    
    return train_val_pool, assessment_set

# ==============================================================================
# 3. Preprocessing
# ==============================================================================

def create_preprocessors(config: dict) -> dict:
    """Creates and returns a dictionary of scikit-learn preprocessing pipelines."""
    params = config['model']
    return {
        "linear": build_model_pipeline(pred_vars=params['predictor']['all'], cat_vars=params['predictor']['categorical'], id_vars=params['predictor']['id']),
        "embedding_linear": build_model_pipeline_supress_onehot(pred_vars=params['predictor']['all'], cat_vars=params['predictor']['categorical'], id_vars=params['predictor']['id']),
        "embedding": ModelMainRecipeImputer(outcome="meta_sale_price", pred_vars=params['predictor']['all'], cat_vars=params['predictor']['categorical'], id_vars=params['predictor']['id']),
        "LightGBM": ModelMainRecipe(outcome="meta_sale_price", pred_vars=params['predictor']['all'], cat_vars=params['predictor']['categorical'], id_vars=params['predictor']['id']),
    }

# ==============================================================================
# 4. Model Training and Evaluation
# ==============================================================================

def train_and_evaluate(model_name: str, model_config: dict, data_splits: tuple):
    """Initializes, trains, predicts, and evaluates a single model. Returns assessment set metrics."""
    X_train, y_train, X_val, y_val, X_test, y_test = data_splits
    
    model = model_config['model_class'](**model_config['params'])
    fit_params = model_config.get('fit_params', {})
    
    if model_name == "LightGBM": fit_params['eval_set'] = [(X_val, y_val)]
    
    if "FeedForward" in model_name or "TabTransformer" in model_name:
         model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    else:
        model.fit(X_train, y_train, **fit_params)

    y_pred_test_log = model.predict(X_test)
    
    p1, p99 = np.percentile(y_pred_test_log, [1, 99])
    np.clip(y_pred_test_log, p1, p99, out=y_pred_test_log)
        
    y_test_exp = np.exp(y_test)
    y_pred_test_exp = np.exp(y_pred_test_log)

    test_metrics = compute_all_metrics(y_test_exp, y_pred_test_exp)
    test_metrics['r2'] = r2_score(y_test_exp, y_pred_test_exp)
    
    return test_metrics

# ==============================================================================
# 5. Plotting
# ==============================================================================

def plot_learning_curves(results: dict, sample_sizes: list):
    """Generates and saves learning curve plots for each metric."""
    print("\nGenerating learning curve plots...")
    metrics_to_plot = ['rmse', 'r2', 'f_dev', 'f_grp']
    models = list(results[sample_sizes[0]].keys())
    
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        
        for model_name in models:
            means = [np.mean(results[size][model_name][metric]) for size in sample_sizes]
            stds = [np.std(results[size][model_name][metric]) for size in sample_sizes]
            
            plt.plot(sample_sizes, means, marker='o', linestyle='-', label=model_name)
            plt.fill_between(sample_sizes, np.array(means) - np.array(stds), np.array(means) + np.array(stds), alpha=0.2)
            
        plt.title(f'Learning Curve for {metric.upper()}')
        plt.xlabel('Training Sample Size')
        plt.ylabel(metric.upper())
        plt.legend()
        plt.grid(True)
        plt.xscale('log')
        plt.savefig(f'plots/learning_curve_{metric}_2.png')
        plt.show()

# ==============================================================================
# 6. Main Execution
# ==============================================================================

def main():
    """Main script execution function."""
    config = get_config()
    main_seed = config['model']['seed']
    num_runs = config['data_prep'].get('num_evaluation_runs', 5)
    sample_sizes = config['cv'].get('learning_curve_sample_sizes', [int(x) for x in np.linspace(1000, 10001, 5)])
    
    models_to_run = ["LightGBM", "FeedForwardNN", "TabTransformer"]
    
    # --- Data Pipeline (Run once) ---
    train_val_pool, assessment_df = load_and_prepare_data(config)
    
    # --- Learning Curve and Multi-Run Evaluation Loop ---
    all_results = {size: {model_name: {metric: [] for metric in ['rmse', 'r2', 'f_dev', 'f_grp']} for model_name in models_to_run} for size in sample_sizes}

    np.random.seed(main_seed)
    model_seeds = np.random.randint(0, 1e4, num_runs)
    np.random.seed(None)

    for size in sample_sizes:
        print(f"\n{'='*35} PROCESSING SAMPLE SIZE: {size} {'='*35}")
        
        # Ensure we don't sample more than available data
        if size > len(train_val_pool):
            print(f"Warning: Sample size {size} is larger than the training pool size {len(train_val_pool)}. Skipping.")
            continue

        for i in range(num_runs):
            run_seed = model_seeds[i]
            print(f"\n--- Starting Run {i+1}/{num_runs} (Seed: {run_seed}) for sample size {size} ---")
            
            # 1. Sample the data for this run
            run_sample_df = train_val_pool.sample(n=size, random_state=run_seed)
            
            # 2. IMPORTANT: Sort the sample by date and create a time-based train/val split
            run_sample_df = run_sample_df.sort_values('meta_sale_date')
            train_split_idx = int(len(run_sample_df) * (1 - 0.1)) # Use 25% for validation
            train_df = run_sample_df.iloc[:train_split_idx]
            val_df = run_sample_df.iloc[train_split_idx:]

            # 3. Create preprocessors and process data for this specific run
            preprocessors = create_preprocessors(config)
            # The assessment_df remains constant and is passed for transformation
            y_assessment = assessment_df['meta_sale_price']
            X_assessment = assessment_df.drop(columns=['meta_sale_price'])
            
            processed_data = {}
            for name, pipeline in preprocessors.items():
                y_train = train_df['meta_sale_price']
                X_train = train_df.drop(columns=['meta_sale_price'])
                
                y_val = val_df['meta_sale_price']
                X_val = val_df.drop(columns=['meta_sale_price'])
                
                # Fit on the current run's training data
                pipeline.fit(X_train, y_train)
                
                # Transform all splits
                X_train_fit = pipeline.transform(X_train).drop(columns=['meta_sale_date'], errors='ignore')
                X_val_fit = pipeline.transform(X_val).drop(columns=['meta_sale_date'], errors='ignore')
                X_test_fit = pipeline.transform(X_assessment).drop(columns=['meta_sale_date'], errors='ignore')
                
                processed_data[name] = (X_train_fit, y_train, X_val_fit, y_val, X_test_fit, y_assessment)


            model_configs = get_model_configurations(config, run_seed)
            
            for name in models_to_run:
                if name in model_configs:
                    model_config = model_configs[name]
                    preprocessor_key = model_config['preprocessor']
                    
                    if preprocessor_key in processed_data:
                        test_metrics = train_and_evaluate(name, model_config, processed_data[preprocessor_key])
                        
                        # Store metrics for this run and sample size
                        for metric_name, value in test_metrics.items():
                            if metric_name in all_results[size][name]:
                                all_results[size][name][metric_name].append(value)
                    else:
                        print(f"Warning: Preprocessor '{preprocessor_key}' not found for model '{name}'. Skipping.")
                else:
                    print(f"Warning: Model '{name}' not found in configurations. Skipping.")

    # --- Final Results Aggregation and Plotting ---
    print("\n" + "=" * 80)
    print("FINAL AGGREGATED RESULTS (ACROSS ALL RUNS AND SAMPLE SIZES)")
    print("=" * 80)

    for size in sample_sizes:
        if not any(all_results[size].values()): continue
        print(f"\n--- Assessment Performance for Sample Size: {size} (Mean ± Std Dev) ---")
        for model_name, metrics_dict in all_results[size].items():
            print(f"\n  Model: {model_name}")
            for metric_name, values in metrics_dict.items():
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    print(f"    {metric_name.upper():<8}: {mean_val:.4f} ± {std_val:.4f}")

    plot_learning_curves(all_results, sample_sizes)

if __name__ == "__main__":
    main()

