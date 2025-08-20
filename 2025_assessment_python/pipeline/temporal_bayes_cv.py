# ------------------------------------------------------------
# Generalized Temporal Cross-Validation Framework
# ------------------------------------------------------------
# ------------------------------------------------------------
# Generalized Temporal Cross-Validation Framework
# ------------------------------------------------------------
import pandas as pd
import numpy as np
import warnings
import lightgbm as lgb
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from nn_models.nn_unconstrained import FeedForwardNNRegressorWithEmbeddings
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
# Model Handler: Abstracting Model-Specific Logic
# ------------------------------------------------------------
class ModelHandler:
    def __init__(self, model_name, model_params, hyperparameter_config):
        self.model_name = model_name
        self.static_params = model_params
        self.hp_config = hyperparameter_config

    def _suggest_from_range(self, trial, name, rng):
        if isinstance(rng, list) and len(rng) > 0 and isinstance(rng[0], (bool, str)):
             if all(isinstance(x, type(rng[0])) for x in rng):
                return trial.suggest_categorical(name, rng)
        
        # NEW: Handle list of ints for hidden_sizes
        if name == 'hidden_sizes':
            # Example: suggest one or two layers with different sizes
            n_layers = trial.suggest_int('n_layers', rng[0][0], rng[0][1])
            layers = []
            for i in range(n_layers):
                layers.append(trial.suggest_int(f'n_units_l{i}', rng[1][0], rng[1][1]))
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
            if param in ['num_iterations', 'n_estimators']:
                continue
            hp[param] = self._suggest_from_range(trial, param, range_val)
        return hp

    def create_model(self, trial_params):
        if self.model_name == 'LightGBM':
            return self._create_lgbm(trial_params)
        elif self.model_name == 'RandomForestRegressor':
            return self._create_random_forest(trial_params)
        elif self.model_name == 'LinearRegression':
            return self._create_linear_regression(trial_params)
        elif self.model_name == 'FeedForwardNNRegressorWithEmbeddings':
            return self._create_feed_forward_nn(trial_params)
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")

    def _create_lgbm(self, params_dict):
        max_depth = -1
        if self.static_params.get('link_max_depth', False):
            num_leaves = params_dict.get('num_leaves', 31)
            add_link_depth = params_dict.get('add_to_linked_depth', 1)
            max_depth = max(1, int(floor(log2(max(2, num_leaves)))) + int(add_link_depth))
        lgbm_params = {
            'random_state': self.static_params.get('seed'), 'deterministic': self.static_params.get('deterministic'),
            'force_row_wise': self.static_params.get('force_row_wise'), 'verbose': self.static_params.get('verbose', -1),
            'objective': self.static_params.get('objective'), 'learning_rate': params_dict.get('learning_rate'),
            'max_bin': int(params_dict.get('max_bin', 255)), 'num_leaves': int(params_dict.get('num_leaves', 31)),
            'feature_fraction': params_dict.get('feature_fraction'), 'min_gain_to_split': params_dict.get('min_gain_to_split'),
            'min_data_in_leaf': int(params_dict.get('min_data_in_leaf', 20)), 'max_depth': int(max_depth),
            'reg_alpha': params_dict.get('lambda_l1'), 'reg_lambda': params_dict.get('lambda_l2'),
            'n_estimators': int(params_dict.get('n_estimators', 100)),
        }
        if 'max_cat_threshold' in params_dict:
            lgbm_params.update({
                'max_cat_threshold': int(params_dict['max_cat_threshold']),
                'min_data_per_group': int(params_dict['min_data_per_group']),
                'cat_smooth': params_dict['cat_smooth'], 'cat_l2': params_dict['cat_l2'],
            })
        return lgb.LGBMRegressor(**lgbm_params)

    def _create_random_forest(self, params_dict):
        rf_params = {
            'random_state': self.static_params.get('seed'),
            'n_estimators': int(params_dict.get('n_estimators', 100)),
            'max_depth': int(params_dict.get('max_depth', 10)) if params_dict.get('max_depth') else None,
            'min_samples_split': int(params_dict.get('min_samples_split', 2)),
            'min_samples_leaf': int(params_dict.get('min_samples_leaf', 1)),
            'max_features': params_dict.get('max_features', 1.0), 'n_jobs': -1
        }
        return RandomForestRegressor(**rf_params)

    def _create_linear_regression(self, params_dict):
        lr_params = {'n_jobs': -1, 'fit_intercept': params_dict.get('fit_intercept', True)}
        return LinearRegression(**lr_params)

    def _create_feed_forward_nn(self, params_dict):
        """Creates an instance of the FeedForwardNNRegressorWithEmbeddings."""
        nn_params = {
            'categorical_features': self.static_params['predictor']['categorical'],
            # MINE: the output size should be 1
            'output_size':1,
            'learning_rate': params_dict.get('learning_rate'),
            'batch_size': params_dict.get('batch_size'),
            'num_epochs': params_dict.get('num_epochs'),
            'hidden_sizes': params_dict.get('hidden_sizes'),
        }
        return FeedForwardNNRegressorWithEmbeddings(**nn_params)

    def get_fit_kwargs(self, X_val, y_val):
        fit_kwargs = {}
        early_stopping_enable = self.static_params.get('early_stopping_enable', False)
        if early_stopping_enable and X_val is not None:
            if self.model_name == 'LightGBM':
                fit_kwargs.update({
                    'eval_set': [(X_val, y_val)],
                    'eval_metric': self.static_params.get('validation_metric', 'rmse'),
                    'callbacks': [
                        lgb.early_stopping(stopping_rounds=self.static_params.get('stop_iter', 10)),
                        lgb.log_evaluation(0)
                    ],
                })
            else:
                print(f"Warning: Early stopping not implemented for {self.model_name} in this script.")
        return fit_kwargs

# ------------------------------------------------------------
# 2. Temporal CV Class
# ------------------------------------------------------------
class TemporalCV:
    def __init__(self, model_handler, cv_params, cv_enable, data, target_col, date_col, preproc_pipeline):
        self.model_handler = model_handler
        self.cv_params = cv_params
        self.cv_enable = cv_enable
        self.data = data
        self.target_col = target_col
        self.date_col = date_col
        self.preproc_pipeline = preproc_pipeline
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
        for i in range(n % v):
            fold_sizes[i] += 1
        idx = 0
        folds = []
        for fs in fold_sizes:
            start, end = idx, idx + fs
            folds.append((start, end))
            idx = end
        splits = []
        for i, (start, end) in enumerate(folds):
            test_idx, train_idx = np.arange(start, end), np.arange(0, start)
            if len(train_idx) == 0:
                continue
            splits.append((train_idx, test_idx))
        return df_sorted, splits

    def _split_train_eval(self, X_fold, y_fold):
        val_prop = self.model_handler.static_params.get('validation_prop', 0.0)
        if val_prop <= 0 or val_prop >= 1:
            return X_fold, y_fold, None, None
        n = len(X_fold)
        val_n = max(1, int(round(n * val_prop)))
        tr_n = n - val_n
        if tr_n <= 0:
            return X_fold, y_fold, None, None
        X_tr, y_tr = X_fold.iloc[:tr_n], y_fold.iloc[:tr_n]
        X_val, y_val = X_fold.iloc[tr_n:], y_fold.iloc[tr_n:]
        return X_tr, y_tr, X_val, y_val

    def _objective_optuna(self, trial):
        hp = self.model_handler.suggest_hyperparameters(trial)

        if self.early_stopping_enable:
            hp['n_estimators'] = self.n_estimators_static
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
        else: # 'recent'
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
            X_val_proc = pipeline_instance.transform(X_val) if X_val is not None else None

            # print("NANS per dataset:")
            # print(X_tr_proc.isna().sum())
            # print(X_val_proc.isna().sum())
            # print(X_te_proc.isna().sum())
            # exit()

            # print("Categorical columns rn (train):")
            # for c in X_tr_proc.columns:
            #     if X_tr_proc[c].dtype == "category":
            #         print(c)
            #         if not c in self.model_handler.static_params['predictor']['categorical']:
            #             print("is it in the ones passed? ", False)
            # print("Categorical columns rn (test):")
            # for c in X_te_proc.columns:
            #     if X_te_proc[c].dtype == "category":
            #         print(c)
            #         if not c in self.model_handler.static_params['predictor']['categorical']:
            #             print("is it in the ones passed? ", False)
            # print("Categorical columns rn:")
            # for c in X_val_proc.columns:
            #     if X_val_proc[c].dtype == "category":
            #         print(c)
            #         if not c in self.model_handler.static_params['predictor']['categorical']:
            #             print("is it in the ones passed? ", False)
            # print("The ones that were passed:")
            # print(self.model_handler.static_params['predictor']['categorical'])

            fit_kwargs = self.model_handler.get_fit_kwargs(X_val_proc, y_val)
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
        sampler = optuna.samplers.TPESampler(seed=self.model_handler.static_params.get('seed'), n_startup_trials=self.cv_params.get('initial_set', 10))
        self.study_ = optuna.create_study(direction='minimize', study_name=f'{self.model_handler.model_name}_cv', sampler=sampler)
        self.study_.optimize(
            self._objective_optuna,
            n_trials=self.cv_params.get('max_iterations', 50),
            show_progress_bar=True
        )
        self.best_params_ = self.study_.best_params
        print("\nBest hyperparameters found:")
        print(self.best_params_)

        final_hp = self.best_params_.copy()
        if self.early_stopping_enable:
            final_hp['n_estimators'] = self.n_estimators_static
        self.best_model_ = self.model_handler.create_model(final_hp)
        print("\nFinal model created with best parameters:")
        print(self.best_model_)

# ------------------------------------------------------------
# 3. Configuration and Execution
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
            'num_folds': 5,
            'initial_set': 5,
            'max_iterations': 10, # Reduced for faster demo
        },
        'model': {
            'name': 'FeedForwardNNRegressorWithEmbeddings', # <-- SELECT MODEL HERE
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
                        'num_epochs': [20, 100],
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
        preproc_pipeline=pipeline
    )
    temporal_cv_process.run()
