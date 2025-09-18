import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd
from collections import OrderedDict
import random
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Assume 'device' is configured (e.g., device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from nn_models.unconstrained.BaselineModels5 import FeedForwardNNRegressorWithEmbeddings5

# ==============================================================================
# 1. Fourier Features Module (Unchanged)
# ==============================================================================
class FourierFeatures(nn.Module):
    """
    A module to generate various types of Fourier features for coordinate data,
    based on the "Fourier Features Let Networks Learn High Frequency Functions" paper.
    """
    def __init__(self, in_features, mapping_size, scale=10.0, fourier_type='gaussian', sigma=1.25):
        super().__init__()
        self.in_features = in_features
        self.mapping_size = mapping_size
        self.fourier_type = fourier_type

        if fourier_type == 'gaussian':
            self.B = nn.Parameter(torch.randn(in_features, mapping_size) * scale, requires_grad=False)
            self.output_dim = 2 * mapping_size
        elif fourier_type == 'positional':
            if sigma is None:
                raise ValueError("sigma must be provided for positional Fourier features.")
            freq_bands = (sigma ** (torch.arange(mapping_size) / mapping_size))
            self.register_buffer('freq_bands', freq_bands)
            self.output_dim = 2 * in_features * mapping_size
        elif fourier_type == 'basic':
            self.output_dim = 2 * in_features
        elif fourier_type == 'none':
            return
        else:
            raise ValueError(f"Unknown fourier_type: {fourier_type}")

    def forward(self, x):
        if self.fourier_type == 'gaussian':
            x_proj = 2 * np.pi * x @ self.B
            return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        elif self.fourier_type == 'positional':
            x_proj = 2 * np.pi * x.unsqueeze(-1) * self.freq_bands.view(1, 1, -1)
            x_proj_flat = x_proj.view(x.shape[0], -1)
            return torch.cat([torch.sin(x_proj_flat), torch.cos(x_proj_flat)], dim=-1)
        elif self.fourier_type == 'basic':
            x_proj = 2 * np.pi * x
            return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        elif self.fourier_type == 'none':
            return x

# ==============================================================================
# 2. Advanced Loss Functions
# ==============================================================================
class FocalMSELoss(nn.Module):
    """Focal Mean Squared Error Loss to focus on hard-to-predict samples."""
    def __init__(self, gamma=1.0):
        super().__init__()
        self.gamma = gamma
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, y_pred, y_true):
        mse_loss = self.mse(y_pred, y_true)
        focal_weight = mse_loss.detach() ** self.gamma
        return (focal_weight * mse_loss).mean()

class BinnedMSELoss(nn.Module):
    """
    Calculates MSE loss where each bin of the target variable has equal weight,
    with an optional focal factor to emphasize harder examples within each bin.
    """
    def __init__(self, y_min, y_max, n_bins=10, gamma=1.0):
        super().__init__()
        self.n_bins = n_bins
        self.gamma = gamma
        self.bin_edges = torch.linspace(y_min, y_max + 1e-6, n_bins + 1, device=device)
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, y_pred, y_true):
        bin_indices = torch.bucketize(y_true.flatten(), self.bin_edges) - 1
        bin_indices = torch.clamp(bin_indices, 0, self.n_bins - 1)
        bin_counts = torch.bincount(bin_indices, minlength=self.n_bins)
        
        weights_per_bin = 1.0 / (bin_counts + 1e-6)
        weights_per_bin[bin_counts == 0] = 0
        bin_weights = weights_per_bin[bin_indices]
        
        per_sample_mse = self.mse(y_pred, y_true).flatten()
        focal_weights = per_sample_mse.detach() ** self.gamma
        combined_weights = bin_weights * focal_weights
        weighted_loss = per_sample_mse * combined_weights
        
        num_non_empty_bins = (bin_counts > 0).sum()
        if num_non_empty_bins == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
            
        return weighted_loss.sum() / num_non_empty_bins

class QuantileWeightedLoss(nn.Module):
    """
    A custom loss function that re-weights the base MSE loss based on the
    quantile of the true target value. This puts more emphasis on samples
    in the tails of the distribution.
    """
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, y_pred, y_true, sample_weights):
        """
        Calculates the weighted MSE loss.

        Args:
            y_pred (torch.Tensor): The model's predictions.
            y_true (torch.Tensor): The true target values.
            sample_weights (torch.Tensor): The pre-computed weights for each sample.
        """
        mse_loss = self.mse(y_pred, y_true)
        weighted_loss = mse_loss * sample_weights
        return weighted_loss.mean()

# ==============================================================================
# 3. The Core Neural Network Model (Updated with Normalization)
# ==============================================================================
class NNWithEmbeddings(nn.Module):
    """
    A neural network model that accepts categorical, numerical, and coordinate inputs,
    with flexible Fourier feature encoding for coordinates.
    """
    def __init__(self, embedding_specs, num_numerical_features, num_coord_features,
                 fourier_type, fourier_mapping_size, fourier_sigma, layer_sizes,
                 dropout=0.5, normalization_type='none'):
        super().__init__()
        
        # --- Feature Encoders ---
        self.embedding_layers = nn.ModuleList([nn.Embedding(num, dim) for num, dim in embedding_specs])
        
        if fourier_type != 'none' and num_coord_features > 0:
            self.fourier_layer = FourierFeatures(
                in_features=num_coord_features, 
                mapping_size=fourier_mapping_size,
                fourier_type=fourier_type,
                sigma=fourier_sigma
            )
            coord_dim = self.fourier_layer.output_dim
        else:
            self.fourier_layer = None
            coord_dim = num_coord_features

        # --- Input Size Calculation ---
        total_embedding_dim = sum(dim for _, dim in embedding_specs)
        input_size = total_embedding_dim + num_numerical_features + coord_dim
        
        # --- Feed-Forward Layers ---
        all_layers = []
        for i, size in enumerate(layer_sizes):
            all_layers.append((f"linear_{i}", nn.Linear(input_size, size)))
            
            # Add normalization layer here
            if normalization_type == 'batch_norm':
                all_layers.append((f"norm_{i}", nn.BatchNorm1d(size)))
            elif normalization_type == 'layer_norm':
                all_layers.append((f"norm_{i}", nn.LayerNorm(size)))

            activation = nn.ReLU() if i < len(layer_sizes) - 1 else None
            if activation:
                all_layers.append((f"relu_{i}", activation))
            # Add dropout after each layer
            if dropout > 0:
                all_layers.append((f"dropout_{i}", nn.Dropout(p=dropout)))
            input_size = size
            
        self.layers = nn.Sequential(OrderedDict(all_layers))

    def forward(self, x_cat, x_num, x_coord):
        # Process categorical features
        embeddings = [emb_layer(x_cat[:, i]) for i, emb_layer in enumerate(self.embedding_layers)]
        x_cat_emb = torch.cat(embeddings, dim=1)
        
        # Process coordinate features
        if self.fourier_layer:
            x_coord_processed = self.fourier_layer(x_coord)
        else:
            x_coord_processed = x_coord
        
        # Concatenate all features
        x = torch.cat([x_cat_emb, x_num, x_coord_processed], dim=1)
        
        return self.layers(x)
        
# ==============================================================================
# 4. The Regressor Wrapper Class with Two-Stage Residual Modeling
# ==============================================================================
"""
A two-stage regressor that uses a primary model to predict the target and a
secondary model to correct the residuals from the first stage.

Inputs:
- categorical_features: A list of column names for categorical features.
- coord_features: A list of column names for coordinate features.
- engineer_time_features: A boolean flag to enable or disable cyclical time feature engineering.
- bin_yrblt: A boolean flag to enable or disable binning of the 'char_yrblt' year built feature.
- cross_township_class: A boolean flag to enable or disable creating a new feature by combining township code and class.
- fourier_type: The type of Fourier feature to use for coordinates ('gaussian', 'positional', 'basic', or 'none').
- fourier_mapping_size: The number of Fourier features to generate for each coordinate.
- fourier_sigma: The frequency scale for positional Fourier features.
- output_size: The number of outputs from the model.
- batch_size: The number of samples per batch during training.
- learning_rate: The learning rate for the Adam optimizer.
- num_epochs: The number of training epochs.
- hidden_sizes: A list of integers representing the number of neurons in each hidden layer.
- patience: The number of epochs to wait for improvement before early stopping.
- loss_fn: The name of the loss function to use ('mse', 'huber', 'focal_mse', 'binned_mse', or 'quantile_weighted_mse').
- n_bins: The number of bins for the 'binned_mse' loss function.
- gamma: The gamma parameter for the 'focal_mse' and 'binned_mse' loss functions.
- loss_alpha: The alpha parameter for the 'quantile_weighted_mse' loss function.
- random_state: A seed for random number generators to ensure reproducibility.
- dropout: The dropout probability for regularization.
- l2_lambda: The L2 regularization strength.
- l1_lambda: The L1 regularization strength.
- use_scaler: A boolean flag to enable or disable numerical feature scaling using StandardScaler.
- normalization_type: The type of normalization to use in the hidden layers ('none', 'batch_norm', or 'layer_norm').
- residual_model_split: The fraction of the training data to hold out for training the residual model.
- num_residual_epochs: The number of epochs to train the residual model.
- validation_split: The fraction of the training data to use for validation.
- residual_loss_fn: The loss function for the residual model ('mse' or 'huber').
"""
class FeedForwardNNRegressorWithResiduals:

    def __init__(self, categorical_features, coord_features=[], 
                 engineer_time_features=False, bin_yrblt=False, cross_township_class=False,
                 fourier_type='none', fourier_mapping_size=16, fourier_sigma=1.25,
                 output_size=1, batch_size=16, learning_rate=0.001, num_epochs=10, 
                 hidden_sizes=[1024], patience=10, loss_fn='mse', n_bins=10, gamma=1.0, loss_alpha=1.0, random_state=None,
                 dropout=0.5, l2_lambda=0, l1_lambda=0, use_scaler=False, normalization_type='none',
                 residual_model_split=0.2, num_residual_epochs=20, validation_split=0.1, residual_loss_fn='mse'):
        
        self.original_categorical_features = categorical_features[:]
        self.original_coord_features = coord_features[:]
        
        # Feature Engineering Flags
        self.engineer_time_features = engineer_time_features
        self.bin_yrblt = bin_yrblt
        self.cross_township_class = cross_township_class
        self.use_scaler = use_scaler
        self.scaler = None

        # Core model parameters
        self.fourier_type = fourier_type
        self.fourier_mapping_size = fourier_mapping_size
        self.fourier_sigma = fourier_sigma
        self.numerical_features = []
        self.output_size = output_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.l2_lambda = l2_lambda
        self.l1_lambda = l1_lambda
        self.patience = patience
        self.loss_fn_name = loss_fn
        self.n_bins = n_bins
        self.gamma = gamma
        self.loss_alpha = loss_alpha
        self.model = None
        self.residual_model = None
        self.category_mappings = {}
        self.embedding_specs = []
        self.random_state = random_state
        self.normalization_type = normalization_type

        # Residual Modeling Parameters
        self.residual_model_split = residual_model_split
        self.num_residual_epochs = num_residual_epochs
        self.validation_split = validation_split
        self.residual_loss_fn = residual_loss_fn

    def _engineer_features(self, X):
        X_eng = X.copy()
        
        if self.engineer_time_features:
            cyclical_features = {
                'time_sale_month_of_year': 12,
                'time_sale_day_of_week': 7,
                'time_sale_day_of_year': 365.25,
                'time_sale_day_of_month': 30.44
            }
            for col, period in cyclical_features.items():
                if col in X_eng.columns:
                    X_eng[f'{col}_sin'] = np.sin(2 * np.pi * X_eng[col] / period)
                    X_eng[f'{col}_cos'] = np.cos(2 * np.pi * X_eng[col] / period)
            
            features_to_drop = list(cyclical_features.keys()) + ['time_sale_day']
            X_eng = X_eng.drop(columns=features_to_drop, errors='ignore')

        if self.bin_yrblt:
            if 'char_yrblt' in X_eng.columns:
                X_eng['yrblt_decade'] = (X_eng['char_yrblt'] // 10 * 10).astype(str)
                X_eng = X_eng.drop(columns=['char_yrblt'], errors='ignore')

        if self.cross_township_class:
            if 'meta_township_code' in X_eng.columns and 'char_class' in X_eng.columns:
                X_eng['township_class_interaction'] = X_eng['meta_township_code'].astype(str) + '_' + X_eng['char_class'].astype(str)
        
        return X_eng

    def fit(self, X, y, X_val=None, y_val=None):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)
            random.seed(self.random_state)

        # --- Feature Engineering ---
        X_eng = self._engineer_features(X)
        
        # --- Dynamically Determine Feature Lists After Engineering ---
        current_categorical = self.original_categorical_features[:]
        if self.engineer_time_features:
            cyclical_to_remove = ['time_sale_month_of_year', 'time_sale_day_of_week', 'time_sale_day_of_year', 'time_sale_day_of_month']
            current_categorical = [c for c in current_categorical if c not in cyclical_to_remove]
        
        if self.bin_yrblt:
            if 'char_yrblt' in current_categorical: current_categorical.remove('char_yrblt')
            current_categorical.append('yrblt_decade')

        if self.cross_township_class:
            current_categorical.append('township_class_interaction')

        self.categorical_features = [c for c in current_categorical if c in X_eng.columns]
        self.coord_features = self.original_coord_features[:]
        self.numerical_features = [col for col in X_eng.columns if col not in self.categorical_features and col not in self.coord_features]

        # --- Three-way split for base, residual, and validation sets ---
        if X_val is None or y_val is None:
            total_data_size = len(X_eng)
            val_size = int(self.validation_split * total_data_size)
            residual_size = int(self.residual_model_split * total_data_size)

            if val_size + residual_size >= total_data_size:
                raise ValueError("Sum of validation_split and residual_model_split is too large.")
            
            X_train_temp, X_val_internal, y_train_temp, y_val_internal = train_test_split(
                X_eng, y, test_size=val_size, random_state=self.random_state
            )
            X_base_train, X_residual_train, y_base_train, y_residual_train = train_test_split(
                X_train_temp, y_train_temp, test_size=residual_size, random_state=self.random_state
            )
        else:
            X_val_internal = self._engineer_features(X_val)
            y_val_internal = y_val.copy()

            total_data_size = len(X_eng)
            residual_size = int(self.residual_model_split * total_data_size)

            if residual_size >= total_data_size:
                 raise ValueError("residual_model_split is too large.")

            X_base_train, X_residual_train, y_base_train, y_residual_train = train_test_split(
                X_eng, y, test_size=residual_size, random_state=self.random_state
            )


        # --- Stage 1: Train the Base Model ---
        print("Training Stage 1: The Base Model")
        
        # We need a separate instance to avoid modifying the main model's parameters for the base fit
        base_model_params = {
            'categorical_features': self.categorical_features,
            'coord_features': self.coord_features,
            'engineer_time_features': False, 'bin_yrblt': False, 'cross_township_class': False,
            'fourier_type': self.fourier_type, 'fourier_mapping_size': self.fourier_mapping_size, 'fourier_sigma': self.fourier_sigma,
            'output_size': self.output_size, 'batch_size': self.batch_size, 'learning_rate': self.learning_rate, 
            'num_epochs': self.num_epochs, 'hidden_sizes': self.hidden_sizes, 'patience': self.patience,
            'loss_fn': self.loss_fn_name, 'n_bins': self.n_bins, 'gamma': self.gamma, 'loss_alpha': self.loss_alpha,
            'random_state': self.random_state, 'dropout': self.dropout, 'l2_lambda': self.l2_lambda, 'l1_lambda': self.l1_lambda,
            'use_scaler': self.use_scaler, 'normalization_type': self.normalization_type
        }
        
        self.model = FeedForwardNNRegressorWithEmbeddings5(**base_model_params)
        self.model.fit(X_base_train, y_base_train, X_val=X_val_internal, y_val=y_val_internal)
        
        # --- Stage 2: Train the Residual Model ---
        print("\nTraining Stage 2: The Residual Model")
        
        # Get base model predictions on the held-out set
        base_predictions = self.model.predict(X_residual_train)
        # Compute the residuals
        residuals = y_residual_train.values.reshape(-1, 1) - base_predictions.reshape(-1, 1)
        
        # Use the same data to train the residual model on the residuals
        residual_model_params = {
            'categorical_features': self.categorical_features,
            'coord_features': self.coord_features,
            'engineer_time_features': False, 'bin_yrblt': False, 'cross_township_class': False,
            'fourier_type': self.fourier_type, 'fourier_mapping_size': self.fourier_mapping_size, 'fourier_sigma': self.fourier_sigma,
            'output_size': self.output_size, 'batch_size': self.batch_size, 'learning_rate': self.learning_rate, 
            'num_epochs': self.num_residual_epochs, 'hidden_sizes': self.hidden_sizes, 'patience': self.patience,
            'loss_fn': self.residual_loss_fn, 'random_state': self.random_state, 'dropout': self.dropout,
            'l2_lambda': self.l2_lambda, 'l1_lambda': self.l1_lambda, 'use_scaler': self.use_scaler,
            'normalization_type': self.normalization_type
        }
        self.residual_model = FeedForwardNNRegressorWithEmbeddings5(**residual_model_params)
        self.residual_model.fit(X_residual_train, pd.Series(residuals.flatten()))

    def predict(self, X):
        if self.model is None or self.residual_model is None:
            raise RuntimeError("You must call fit() before calling predict().")
        
        # Get base predictions
        base_predictions = self.model.predict(X)
        
        # Get residual predictions
        residual_predictions = self.residual_model.predict(X)
        
        # Combine the predictions
        final_predictions = base_predictions + residual_predictions
        
        return final_predictions
    
    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

# This class is needed for the residual model's fit method
class FeedForwardNNRegressorWithEmbeddings6:

    def __init__(self, categorical_features, coord_features=[], 
                 engineer_time_features=False, bin_yrblt=False, cross_township_class=False,
                 fourier_type='none', fourier_mapping_size=16, fourier_sigma=1.25,
                 output_size=1, batch_size=16, learning_rate=0.001, num_epochs=10, 
                 hidden_sizes=[1024], patience=10, loss_fn='mse', n_bins=10, gamma=1.0, loss_alpha=1.0, random_state=None,
                 dropout=0.5, l2_lambda=0, l1_lambda=0, use_scaler=False, normalization_type='none'):
        
        self.original_categorical_features = categorical_features[:]
        self.original_coord_features = coord_features[:]
        
        # Feature Engineering Flags
        self.engineer_time_features = engineer_time_features
        self.bin_yrblt = bin_yrblt
        self.cross_township_class = cross_township_class
        self.use_scaler = use_scaler
        self.scaler = None

        # Core model parameters
        self.fourier_type = fourier_type
        self.fourier_mapping_size = fourier_mapping_size
        self.fourier_sigma = fourier_sigma
        self.numerical_features = []
        self.output_size = output_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.l2_lambda = l2_lambda
        self.l1_lambda = l1_lambda
        self.patience = patience
        self.loss_fn_name = loss_fn
        self.n_bins = n_bins
        self.gamma = gamma
        self.loss_alpha = loss_alpha
        self.model = None
        self.category_mappings = {}
        self.embedding_specs = []
        self.random_state = random_state
        self.normalization_type = normalization_type

    def _engineer_features(self, X):
        X_eng = X.copy()
        
        if self.engineer_time_features:
            cyclical_features = {
                'time_sale_month_of_year': 12,
                'time_sale_day_of_week': 7,
                'time_sale_day_of_year': 365.25,
                'time_sale_day_of_month': 30.44
            }
            for col, period in cyclical_features.items():
                if col in X_eng.columns:
                    X_eng[f'{col}_sin'] = np.sin(2 * np.pi * X_eng[col] / period)
                    X_eng[f'{col}_cos'] = np.cos(2 * np.pi * X_eng[col] / period)
            
            features_to_drop = list(cyclical_features.keys()) + ['time_sale_day']
            X_eng = X_eng.drop(columns=features_to_drop, errors='ignore')

        if self.bin_yrblt:
            if 'char_yrblt' in X_eng.columns:
                X_eng['yrblt_decade'] = (X_eng['char_yrblt'] // 10 * 10).astype(str)
                X_eng = X_eng.drop(columns=['char_yrblt'], errors='ignore')

        if self.cross_township_class:
            if 'meta_township_code' in X_eng.columns and 'char_class' in X_eng.columns:
                X_eng['township_class_interaction'] = X_eng['meta_township_code'].astype(str) + '_' + X_eng['char_class'].astype(str)
        
        return X_eng

    def fit(self, X, y, X_val=None, y_val=None):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)
            random.seed(self.random_state)

        # --- Feature Engineering ---
        X_eng = self._engineer_features(X)
        
        # --- Dynamically Determine Feature Lists After Engineering ---
        current_categorical = self.original_categorical_features[:]
        if self.engineer_time_features:
            cyclical_to_remove = ['time_sale_month_of_year', 'time_sale_day_of_week', 'time_sale_day_of_year', 'time_sale_day_of_month']
            current_categorical = [c for c in current_categorical if c not in cyclical_to_remove]
        
        if self.bin_yrblt:
            if 'char_yrblt' in current_categorical: current_categorical.remove('char_yrblt')
            current_categorical.append('yrblt_decade')

        if self.cross_township_class:
            current_categorical.append('township_class_interaction')

        self.categorical_features = [c for c in current_categorical if c in X_eng.columns]
        self.coord_features = self.original_coord_features[:]
        self.numerical_features = [col for col in X_eng.columns if col not in self.categorical_features and col not in self.coord_features]

        # --- Validation Split on Engineered Data ---
        # The wrapper class handles the split, so this inner class just uses the provided data
        X_train, y_train = X_eng, y.copy()
        if X_val is not None and y_val is not None:
            X_val, y_val = self._engineer_features(X_val), y_val.copy()
        
        # --- Scaling numerical features ---
        if self.use_scaler:
            self.scaler = StandardScaler()
            # Fit the scaler on the training numerical data only
            X_train_num_scaled = self.scaler.fit_transform(X_train[self.numerical_features])
            if X_val is not None:
                X_val_num_scaled = self.scaler.transform(X_val[self.numerical_features])
            else:
                X_val_num_scaled = np.empty((0, len(self.numerical_features)))
        else:
            X_train_num_scaled = X_train[self.numerical_features].values
            if X_val is not None:
                X_val_num_scaled = X_val[self.numerical_features].values
            else:
                X_val_num_scaled = np.empty((0, len(self.numerical_features)))
            
        # --- Prepare Tensors and Dataloaders ---
        self.embedding_specs = []
        for col in self.categorical_features:
            categories = X_train[col].unique()
            self.category_mappings[col] = {cat: i + 1 for i, cat in enumerate(categories)}
            self.category_mappings[col]['__UNKNOWN__'] = 0
            num_categories_with_unknown = len(categories) + 1
            embedding_dim = min(50, (num_categories_with_unknown + 1) // 2)
            self.embedding_specs.append((num_categories_with_unknown, embedding_dim))

        def _create_tensors(X_df_cat, X_num_scaled, X_df_coord, y_series):
            X_cat_tensors = [torch.tensor(X_df_cat[col].map(self.category_mappings[col]).fillna(0).values, dtype=torch.long) for col in self.categorical_features]
            X_cat = torch.stack(X_cat_tensors, dim=1) if X_cat_tensors else torch.empty(len(X_df_cat), 0, dtype=torch.long)
            X_num = torch.tensor(X_num_scaled, dtype=torch.float32)
            X_coord = torch.tensor(X_df_coord[self.coord_features].astype(float).values, dtype=torch.float32)
            y_tensor = torch.tensor(y_series.values.reshape(-1, 1), dtype=torch.float32)
            return X_cat, X_num, X_coord, y_tensor

        X_cat_train, X_num_train, X_coord_train, y_train_tensor = _create_tensors(X_train, X_train_num_scaled, X_train, y_train)
        X_cat_val, X_num_val, X_coord_val, y_val_tensor = _create_tensors(X_val, X_val_num_scaled, X_val, y_val)

        # Handle the new loss function by pre-computing sample weights
        if self.loss_fn_name == 'quantile_weighted_mse':
            # Calculate quantiles for the entire training set
            y_train_quantiles = y_train.rank(pct=True)
            # Apply the weighting formula
            train_weights = np.exp(-self.loss_alpha * y_train_quantiles) + np.exp(-self.loss_alpha * (1 - y_train_quantiles))
            train_weights_tensor = torch.tensor(train_weights.values.reshape(-1, 1), dtype=torch.float32)

            # Create the training dataset including the weights
            train_dataset = TensorDataset(X_cat_train, X_num_train, X_coord_train, y_train_tensor, train_weights_tensor)
            train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
            # Validation dataset does not need weights
            val_dataset = TensorDataset(X_cat_val, X_num_val, X_coord_val, y_val_tensor)
            val_loader = DataLoader(dataset=val_dataset, batch_size=self.batch_size * 2)
        else:
            train_dataset = TensorDataset(X_cat_train, X_num_train, X_coord_train, y_train_tensor)
            val_dataset = TensorDataset(X_cat_val, X_num_val, X_coord_val, y_val_tensor)
            train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(dataset=val_dataset, batch_size=self.batch_size * 2)

        # --- Model Initialization ---
        self.model = NNWithEmbeddings(
            embedding_specs=self.embedding_specs,
            num_numerical_features=len(self.numerical_features),
            num_coord_features=len(self.coord_features),
            fourier_type=self.fourier_type,
            fourier_mapping_size=self.fourier_mapping_size,
            fourier_sigma=self.fourier_sigma,
            layer_sizes=self.hidden_sizes + [self.output_size],
            dropout=self.dropout,
            normalization_type=self.normalization_type
        ).to(device)
        
        if self.loss_fn_name == 'huber': criterion = nn.HuberLoss()
        elif self.loss_fn_name == 'focal_mse': criterion = FocalMSELoss(gamma=self.gamma)
        elif self.loss_fn_name == 'binned_mse':
            y_min, y_max = y_train.min(), y_train.max()
            criterion = BinnedMSELoss(y_min=y_min, y_max=y_max, n_bins=self.n_bins, gamma=self.gamma)
        elif self.loss_fn_name == 'quantile_weighted_mse':
            criterion = QuantileWeightedLoss(alpha=self.loss_alpha)
        else: criterion = nn.MSELoss()
        
        if self.l2_lambda < 1e-6:
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        else:        
            optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2_lambda)
        
        # --- Training Loop ---
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(self.num_epochs):
            self.model.train()
            total_train_loss = 0
            if self.loss_fn_name == 'quantile_weighted_mse':
                for batch_cat, batch_num, batch_coord, batch_y, batch_weights in train_loader:
                    batch_cat, batch_num, batch_coord, batch_y, batch_weights = batch_cat.to(device), batch_num.to(device), batch_coord.to(device), batch_y.to(device), batch_weights.to(device)
                    outputs = self.model(batch_cat, batch_num, batch_coord)
                    loss = criterion(outputs, batch_y, batch_weights)
                    
                    if self.l1_lambda >= 1e-8:
                        l1_penalty = sum(torch.sum(torch.abs(param)) for param in self.model.parameters())
                        loss += self.l1_lambda * l1_penalty

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_train_loss += loss.item()
            else:
                for batch_cat, batch_num, batch_coord, batch_y in train_loader:
                    batch_cat, batch_num, batch_coord, batch_y = batch_cat.to(device), batch_num.to(device), batch_coord.to(device), batch_y.to(device)
                    outputs = self.model(batch_cat, batch_num, batch_coord)
                    loss = criterion(outputs, batch_y)

                    if self.l1_lambda >= 1e-8:
                        l1_penalty = sum(torch.sum(torch.abs(param)) for param in self.model.parameters())
                        loss += self.l1_lambda * l1_penalty

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_train_loss += loss.item()
            
            self.model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch_cat, batch_num, batch_coord, batch_y in val_loader:
                    batch_cat, batch_num, batch_coord, batch_y = batch_cat.to(device), batch_num.to(device), batch_coord.to(device), batch_y.to(device)
                    outputs = self.model(batch_cat, batch_num, batch_coord)
                    
                    if self.loss_fn_name == 'quantile_weighted_mse':
                         # For validation, we can't use the precomputed weights from the train set
                         # so we'll just use regular MSE for monitoring.
                         loss = nn.MSELoss()(outputs, batch_y)
                    else:
                         loss = criterion(outputs, batch_y)

                    # Add L1 regularization to validation loss (optional, for monitoring)
                    if self.l1_lambda > 0:
                        l1_penalty = sum(torch.sum(torch.abs(param)) for param in self.model.parameters())
                        loss += self.l1_lambda * l1_penalty

                    total_val_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else 0
            avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {avg_train_loss:.4f} Val Loss: {avg_val_loss:.4f}')

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs.")
                    break
        
        if best_model_state:
            self.model.load_state_dict(best_model_state)

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("You must call fit() before calling predict().")
        
        X_pred = self._engineer_features(X)
        
        if self.use_scaler and self.scaler:
            X_pred_num_scaled = self.scaler.transform(X_pred[self.numerical_features])
        else:
            X_pred_num_scaled = X_pred[self.numerical_features].values
        
        X_cat_tensors = [torch.tensor(X_pred[col].map(self.category_mappings[col]).fillna(0).values, dtype=torch.long) for col in self.categorical_features]
        X_cat = torch.stack(X_cat_tensors, dim=1) if X_cat_tensors else torch.empty(len(X_pred), 0, dtype=torch.long)
        
        X_num = torch.tensor(X_pred_num_scaled, dtype=torch.float32)
        X_coord = torch.tensor(X_pred[self.coord_features].values, dtype=torch.float32)
        
        test_dataset = TensorDataset(X_cat, X_num, X_coord)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size * 2, shuffle=False)

        self.model.eval()
        all_predictions = []
        with torch.no_grad():
            for batch_cat, batch_num, batch_coord in test_loader:
                batch_cat, batch_num, batch_coord = batch_cat.to(device), batch_num.to(device), batch_coord.to(device)
                outputs = self.model(batch_cat, batch_num, batch_coord)
                all_predictions.extend(outputs.cpu().numpy())
        
        return np.array(all_predictions).flatten()

    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
