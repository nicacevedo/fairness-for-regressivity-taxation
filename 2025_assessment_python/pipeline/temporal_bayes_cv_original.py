# ------------------------------------------------------------
# Generalized Temporal Cross-Validation Framework
# ------------------------------------------------------------
import pandas as pd
import numpy as np
import warnings
import lightgbm as lgb
import optuna
import yaml 
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from math import log2, floor

# === My models ===
from nn_models.nn_unconstrained import FeedForwardNNRegressorWithEmbeddings
from nn_models.nn_constrained_cpu_v2 import FeedForwardNNRegressorWithProjection

# === Missing models ===
from nn_models.nn_constrained_cpu_v3 import ConstrainedRegressorProjectedWithEmbeddings
from nn_models.unconstrained.TabTransformerRegressor import TabTransformerRegressor
from nn_models.unconstrained.WideAndDeepRegressor import WideAndDeepRegressor 
# === End of Missing models ===

from recipes.recipes_pipelined import ModelMainRecipe
from math import log2, floor

# --- PyTorch Imports for the Neural Network Model ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# --- Device Configuration for PyTorch ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ------------------------------------------------------------
# Model Handler: Updated to include the new model
# ------------------------------------------------------------
class ModelHandler:
    def __init__(self, model_name, model_params, hyperparameter_config):
        self.model_name = model_name
        self.static_params = model_params
        self.hp_config = hyperparameter_config

    def _suggest_from_range(self, trial, name, rng):
        if isinstance(rng, list) and len(rng) > 0 and isinstance(rng[0], (bool, str)):
            return trial.suggest_categorical(name, rng)
        
        if name == 'hidden_sizes' or name == 'mlp_hidden_dims':
            min_layers, max_layers = rng[0]
            min_units, max_units = rng[1]
            
            n_layers = trial.suggest_int(f'{name}_n_layers', min_layers, max_layers)
            layers = []
            
            previous_layer_size = max_units
            for i in range(n_layers):
                layer_size = trial.suggest_int(f'{name}_n_units_l{i}', min_units, previous_layer_size)
                layers.append(layer_size)
                previous_layer_size = layer_size
            return layers

        low, high = rng[0], rng[1]
        if isinstance(low, int) and isinstance(high, int):
            return trial.suggest_int(name, low, high)
        else:
            log = (high / low > 10) if low > 0 else False
            return trial.suggest_float(name, float(low), float(high), log=log)

    def suggest_hyperparameters(self, trial):
        hp = {}
        for param, range_val in self.hp_config.get('range', {}).items():
            if param in ['num_iterations', 'n_estimators']: continue
            hp[param] = self._suggest_from_range(trial, param, range_val)
        return hp

    def create_model(self, trial_params):
        model_creators = {
            'LightGBM': self._create_lgbm,
            'RandomForestRegressor': self._create_random_forest,
            'LinearRegression': self._create_linear_regression,
            'FeedForwardNNRegressorWithEmbeddings': self._create_feed_forward_nn_embedding,
            'FeedForwardNNRegressorWithProjection': self._create_feed_forward_nn_projection,
            'ConstrainedRegressorProjectedWithEmbeddings': self._create_constrained_nn_v3,
            'TabTransformerRegressor': self._create_tab_transformer,
            'WideAndDeepRegressor': self._create_wide_and_deep
        }
        creator = model_creators.get(self.model_name)
        if creator:
            return creator(trial_params)
        raise ValueError(f"Unsupported model name: {self.model_name}")

    def _create_lgbm(self, params_dict):
        lgbm_params = params_dict.copy()
        lgbm_params['random_state'] = self.static_params.get('seed')
        lgbm_params['deterministic'] = True
        lgbm_params['force_row_wise'] = True
        lgbm_params['max_depth'] = floor(np.log2(lgbm_params['num_leaves'])) + lgbm_params['add_to_linked_depth']
        lgbm_params['objective'] = 'rmse'
        lgbm_params['verbose'] = -1
        return lgb.LGBMRegressor(**lgbm_params)

    def _create_random_forest(self, params_dict):
        rf_params = params_dict.copy()
        rf_params['random_state'] = self.static_params.get('seed')
        return RandomForestRegressor(**rf_params)

    def _create_linear_regression(self, params_dict):
        return LinearRegression(**params_dict)
    
    def _create_feed_forward_nn_embedding(self, params_dict):
        nn_params = {
            'categorical_features': self.static_params['predictor']['large_categories'],
            'learning_rate': params_dict.get('learning_rate'),
            'batch_size': params_dict.get('batch_size'),
            'num_epochs': params_dict.get('num_epochs'),
            'hidden_sizes': params_dict.get('hidden_sizes'),
            'random_state': self.static_params.get('seed') # Pass seed
        }
        return FeedForwardNNRegressorWithEmbeddings(**nn_params)

    def _create_feed_forward_nn_projection(self, params_dict):
        nn_params = {
            'categorical_features': self.static_params['predictor']['large_categories'],
            'learning_rate': params_dict.get('learning_rate'),
            'batch_size': params_dict.get('batch_size'),
            'num_epochs': params_dict.get('num_epochs'),
            'hidden_sizes': params_dict.get('hidden_sizes'),
            'dev_thresh': params_dict.get('dev_thresh'),
            'random_state': self.static_params.get('seed') # Pass seed
        }
        return FeedForwardNNRegressorWithProjection(**nn_params)

    def _create_constrained_nn_v3(self, params_dict):
        nn_params = {
            'categorical_features': self.static_params['predictor']['large_categories'],
            'learning_rate': params_dict.get('learning_rate'),
            'batch_size': params_dict.get('batch_size'),
            'num_epochs': params_dict.get('num_epochs'),
            'hidden_sizes': params_dict.get('hidden_sizes'),
            'dev_thresh': params_dict.get('dev_thresh'),
            'group_thresh': params_dict.get('group_thresh'),
            'n_groups': params_dict.get('n_groups'),
            'random_state': self.static_params.get('seed') # Pass seed
        }
        return ConstrainedRegressorProjectedWithEmbeddings(**nn_params)

    def _create_tab_transformer(self, params_dict):
        nn_params = {
            'categorical_features': self.static_params['predictor']['categorical'],
            'coord_features': ["loc_longitude", "loc_latitude"],
            'batch_size': params_dict.get('batch_size'),
            'learning_rate': params_dict.get('learning_rate'),
            'num_epochs': params_dict.get('num_epochs'),
            'transformer_dim': params_dict.get('transformer_dim'),
            'transformer_heads': params_dict.get('transformer_heads'),
            'transformer_layers': params_dict.get('transformer_layers'),
            'dropout': params_dict.get('dropout'),
            'loss_fn': params_dict.get('loss_fn'),
            'random_state': self.static_params.get('seed') # Pass seed
        }
        return TabTransformerRegressor(**nn_params)

    def _create_wide_and_deep(self, params_dict):
        nn_params = {
            'categorical_features': self.static_params['predictor']['categorical'],
            'batch_size': params_dict.get('batch_size'),
            'learning_rate': params_dict.get('learning_rate'),
            'num_epochs': params_dict.get('num_epochs'),
            'hidden_sizes': params_dict.get('hidden_sizes'),
            'random_state': self.static_params.get('seed') # Pass seed
        }
        return WideAndDeepRegressor(**nn_params)

    def get_fit_kwargs(self, X_val, y_val):
        return {} # No special fit args for these models

# ------------------------------------------------------------
# Temporal CV Class
# ------------------------------------------------------------
class TemporalCV:
    def __init__(self, model_handler, cv_params, cv_enable, data, target_col, date_col, preproc_pipeline, run_name_suffix=''):
        self.model_handler = model_handler
        self.cv_params = cv_params
        self.cv_enable = cv_enable
        self.data = data
        self.target_col = target_col
        self.date_col = date_col
        self.preproc_pipeline = preproc_pipeline
        self.run_name_suffix = run_name_suffix
        self.best_params_ = None
        self.best_model_ = None
        self.study_ = None

        self.early_stopping_enable = model_handler.static_params.get('early_stopping_enable', False)
        if self.early_stopping_enable and 'n_estimators' in self.model_handler.hp_config.get('range', {}):
            self.n_estimators_static = self.model_handler.hp_config['range']['n_estimators'][1]
        else:
            self.n_estimators_static = None

    def _rolling_origin_splits(self, df, v):
        df_sorted = df.sort_values(self.date_col).reset_index(drop=True)
        n = len(df_sorted)
        fold_sizes = [n // v] * v
        for i in range(n % v): fold_sizes[i] += 1
        idx = 0
        folds = []
        for fs in fold_sizes:
            start, end = idx, idx + fs
            folds.append((start, end))
            idx = end
        splits = []
        for i, (start, end) in enumerate(folds):
            test_idx, train_idx = np.arange(start, end), np.arange(0, start)
            if len(train_idx) == 0: continue
            splits.append((train_idx, test_idx))
        return df_sorted, splits

    def _split_train_eval(self, X_fold, y_fold):
        val_prop = self.model_handler.static_params.get('validation_prop', 0.0)
        if val_prop <= 0 or val_prop >= 1: return X_fold, y_fold, None, None
        n = len(X_fold)
        val_n = max(1, int(round(n * val_prop)))
        tr_n = n - val_n
        if tr_n <= 0: return X_fold, y_fold, None, None
        X_tr, y_tr = X_fold.iloc[:tr_n], y_fold.iloc[:tr_n]
        X_val, y_val = X_fold.iloc[tr_n:], y_fold.iloc[tr_n:]
        return X_tr, y_tr, X_val, y_val

    def _objective_optuna(self, trial):
        hp = self.model_handler.suggest_hyperparameters(trial)
        if self.early_stopping_enable: hp['n_estimators'] = self.n_estimators_static
        elif 'n_estimators' in self.model_handler.hp_config.get('range', {}):
            ni_range = self.model_handler.hp_config['range']['n_estimators']
            hp['n_estimators'] = trial.suggest_int('n_estimators', int(ni_range[0]), int(ni_range[1]))
        model = self.model_handler.create_model(hp)
        validation_type = self.model_handler.static_params.get('validation_type', 'recent')
        if validation_type == 'random':
            kf = KFold(n_splits=self.cv_params['num_folds'], shuffle=True, random_state=self.model_handler.static_params.get('seed'))
            folds = list(kf.split(np.arange(len(self.data))))
            X_full = self.data.drop(columns=[self.target_col, self.date_col])
            y_full = self.data[self.target_col]
        else:
            df_sorted, folds = self._rolling_origin_splits(self.data, v=self.cv_params['num_folds'])
            X_full = df_sorted.drop(columns=[self.target_col, self.date_col])
            y_full = df_sorted[self.target_col]
        fold_scores = []
        for tr_idx, te_idx in folds:
            X_tr_full, y_tr_full = X_full.iloc[tr_idx], y_full.iloc[tr_idx]
            X_te, y_te = X_full.iloc[te_idx], y_full.iloc[te_idx]
            X_tr, y_tr, X_val, y_val = self._split_train_eval(X_tr_full, y_tr_full)
            pipeline_instance = self.preproc_pipeline
            X_tr_proc = pipeline_instance.fit_transform(X_tr, y_tr)
            X_te_proc = pipeline_instance.transform(X_te)
            fit_kwargs = self.model_handler.get_fit_kwargs(None, None)
            model.fit(X_tr_proc, y_tr, **fit_kwargs)
            y_pred = model.predict(X_te_proc)
            score = mean_squared_error(y_te, y_pred, squared=False)
            fold_scores.append(score)
        return np.mean(fold_scores)

    def run(self):
        if not self.cv_enable:
            print("CV is disabled. Using default hyperparameters.")
            final_hp = self.model_handler.hp_config['default']
            if 'n_estimators' not in final_hp and self.n_estimators_static:
                 final_hp['n_estimators'] = self.n_estimators_static
            self.best_model_ = self.model_handler.create_model(final_hp)
            return

        print(f"Starting cross-validation for {self.model_handler.model_name} model...")

        def save_study_callback(study, trial):
            if trial.number > 0 and (trial.number + 1) % 5 == 0:
                best_params_serializable = {k: (v.item() if hasattr(v, 'item') else v) for k, v in study.best_params.items()}
                
                results_to_save = {
                    'model_name': self.model_handler.model_name,
                    'best_value (RMSE)': study.best_value,
                    'best_params': best_params_serializable,
                    'trials': study.trials_dataframe().to_dict(orient='records')
                }

                filename = f"outputs/{self.model_handler.model_name}_{self.run_name_suffix}_study.yaml"
                with open(filename, 'w') as f:
                    yaml.dump(results_to_save, f, sort_keys=False)
                print(f"\n--- Study results periodically saved to {filename} at iteration {trial.number + 1} ---")

        sampler = optuna.samplers.TPESampler(seed=self.model_handler.static_params.get('seed'), n_startup_trials=self.cv_params.get('initial_set', 10))
        self.study_ = optuna.create_study(direction='minimize', study_name=f'{self.model_handler.model_name}_cv', sampler=sampler)
        self.study_.optimize(
            self._objective_optuna,
            n_trials=self.cv_params.get('max_iterations', 50),
            show_progress_bar=True,
            callbacks=[save_study_callback]
        )
        self.best_params_ = self.study_.best_params
        print("\nBest hyperparameters found:")
        print(self.best_params_)

        if self.study_:
            best_params_serializable = {k: (v.item() if hasattr(v, 'item') else v) for k, v in self.study_.best_params.items()}
            
            results_to_save = {
                'model_name': self.model_handler.model_name,
                'best_value (RMSE)': self.study_.best_value,
                'best_params': best_params_serializable,
                'trials': self.study_.trials_dataframe().to_dict(orient='records')
            }

            filename = f"outputs/{self.model_handler.model_name}_{self.run_name_suffix}_study.yaml"
            with open(filename, 'w') as f:
                yaml.dump(results_to_save, f, sort_keys=False)
            print(f"\nFinal study results saved to {filename}")

        final_hp = self.best_params_.copy()
        if self.early_stopping_enable:
            final_hp['n_estimators'] = self.n_estimators_static
        self.best_model_ = self.model_handler.create_model(final_hp)
        print("\nFinal model created with best parameters:")
        print(self.best_model_)

# ------------------------------------------------------------
# Configuration and Execution
# ------------------------------------------------------------
if __name__ == '__main__':
    # --- Create Dummy Data ---
    data_size = 500
    dummy_data = pd.DataFrame({
        'meta_sale_date': pd.to_datetime(pd.date_range(start='2022-01-01', periods=data_size, freq='D')),
        'meta_sale_price': np.log1p(np.random.rand(data_size) * 100000 + 50000),
        'feature_numeric_1': np.random.rand(data_size) * 100,
        'feature_numeric_2': np.random.randn(data_size) * 10,
        'feature_cat_1': np.random.choice(['A', 'B', 'C', 'D'], size=data_size),
        'feature_cat_2': np.random.choice(['X', 'Y', 'Z'], size=data_size),
        'id_col': range(data_size)
    })

    # --- General Parameters (YAML content) ---
    params = {
        'toggle': {'cv_enable': True},
        'cv': { 
            'num_folds': 3, 
            'initial_set': 3, 
            'max_iterations': 15,
            'run_name_suffix': 'reproducible_nn_test'
        },
        'model': {
            'name': 'WideAndDeepRegressor', # <-- SELECT MODEL HERE
            'objective': 'regression_l1', 'verbose': -1, 'deterministic': True,
            'force_row_wise': True, 'seed': 42,
            'predictor': {
                'all': ['feature_numeric_1', 'feature_numeric_2', 'feature_cat_1', 'feature_cat_2'],
                'categorical': ['feature_cat_1', 'feature_cat_2'],
                'id': ['id_col']
            },
            'parameter': {
                'stop_iter': 20, 'validation_prop': 0.2, 'validation_type': 'recent',
                'validation_metric': 'rmse', 'link_max_depth': True,
            },
            'hyperparameter': {
                'LightGBM': {},
                'RandomForestRegressor': {},
                'LinearRegression': {},
                'FeedForwardNNRegressorWithEmbeddings': {},
                'FeedForwardNNRegressorWithProjection': {},
                'ConstrainedRegressorProjectedWithEmbeddings': {},
                'TabTransformerRegressor': {
                    'range': {
                        'learning_rate': [1e-4, 1e-2], 'batch_size': [16, 64],
                        'num_epochs': [20, 50], 'mlp_hidden_dims': [[1, 3], [32, 128]],
                        'num_attention_layers': [1, 4], 'num_attention_heads': [2, 8]
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
                        'num_epochs': [20, 50], 'hidden_sizes': [[2, 4], [64, 512]]
                    },
                    'default': {
                        'learning_rate': 0.001, 'batch_size': 64, 'num_epochs': 30,
                        'hidden_sizes': [256, 128]
                    }
                }
            }
        }
    }
    
    # --- Set seeds for reproducibility ---
    seed = params['model'].get('seed', 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # The following two lines are often recommended for full reproducibility with CUDA
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

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

    pipeline = ModelMainRecipe(
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
        data=dummy_data,
        target_col='meta_sale_price',
        date_col='meta_sale_date',
        preproc_pipeline=pipeline,
        run_name_suffix=params['cv'].get('run_name_suffix', 'default')
    )
    temporal_cv_process.run()
