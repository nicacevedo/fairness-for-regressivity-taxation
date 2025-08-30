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

# Assume 'device' is configured (e.g., device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 1. Fourier Features Module
# ==============================================================================
class FourierFeatures(nn.Module):
    """
    Adds Fourier features for positional encoding of coordinate data.
    """
    def __init__(self, in_features, mapping_size, scale=10.0):
        super().__init__()
        self.mapping_size = mapping_size
        # Random projection matrix for Fourier features
        self.B = nn.Parameter(torch.randn(in_features, mapping_size) * scale, requires_grad=False)

    def forward(self, x):
        # x has shape (batch_size, num_coord_features)
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

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
        # Use the loss itself as a proxy for "hardness"
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
        self.y_min = y_min
        self.y_max = y_max
        self.gamma = gamma
        # Create bin edges. Add a small epsilon to the max to include it.
        self.bin_edges = torch.linspace(y_min, y_max + 1e-6, n_bins + 1, device=device)
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, y_pred, y_true):
        # Determine which bin each true value falls into
        bin_indices = torch.bucketize(y_true.flatten(), self.bin_edges) - 1
        
        # Clamp indices to be within the valid range [0, n_bins-1]
        bin_indices = torch.clamp(bin_indices, 0, self.n_bins - 1)
        
        # Calculate the count of samples in each bin for the current batch
        bin_counts = torch.bincount(bin_indices, minlength=self.n_bins)
        
        # Calculate bin weights (inverse frequency)
        weights_per_bin = 1.0 / (bin_counts + 1e-6)
        weights_per_bin[bin_counts == 0] = 0
        bin_weights = weights_per_bin[bin_indices]
        
        # Calculate the raw MSE for each sample
        per_sample_mse = self.mse(y_pred, y_true).flatten()
        
        # Calculate focal weights based on error magnitude
        focal_weights = per_sample_mse.detach() ** self.gamma
        
        # Combine the weights
        combined_weights = bin_weights * focal_weights
        
        # The final loss is the weighted average of the per-sample MSEs
        weighted_loss = per_sample_mse * combined_weights
        
        # Normalize by the number of non-empty bins to keep the loss scale consistent
        num_non_empty_bins = (bin_counts > 0).sum()
        if num_non_empty_bins == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
            
        return weighted_loss.sum() / num_non_empty_bins

# ==============================================================================
# 3. The Core Neural Network Model with Embedding and Fourier Layers
# ==============================================================================
class NNWithEmbeddings(nn.Module):
    """
    A neural network model that accepts categorical, numerical, and coordinate inputs.
    """
    def __init__(self, embedding_specs, num_numerical_features, num_coord_features,
                 use_fourier_features, fourier_mapping_size, layer_sizes):
        super().__init__()
        self.use_fourier_features = use_fourier_features
        
        # --- Feature Encoders ---
        self.embedding_layers = nn.ModuleList([nn.Embedding(num, dim) for num, dim in embedding_specs])
        
        if self.use_fourier_features and num_coord_features > 0:
            self.fourier_layer = FourierFeatures(num_coord_features, fourier_mapping_size)
            coord_dim = 2 * fourier_mapping_size
        else:
            self.fourier_layer = None
            coord_dim = num_coord_features

        # --- Input Size Calculation ---
        total_embedding_dim = sum(dim for _, dim in embedding_specs)
        input_size = total_embedding_dim + num_numerical_features + coord_dim
        
        # --- Feed-Forward Layers ---
        all_layers = []
        for i, size in enumerate(layer_sizes):
            activation = nn.ReLU() if i < len(layer_sizes) - 1 else None
            all_layers.append((f"linear_{i}", nn.Linear(input_size, size)))
            if activation:
                all_layers.append((f"relu_{i}", activation))
            input_size = size
            
        self.layers = nn.Sequential(OrderedDict(all_layers))


    def forward(self, x_cat, x_num, x_coord):
        # Process categorical features
        embeddings = [emb_layer(x_cat[:, i]) for i, emb_layer in enumerate(self.embedding_layers)]
        x_cat_emb = torch.cat(embeddings, dim=1)
        
        # Process coordinate features
        if self.use_fourier_features and self.fourier_layer:
            x_coord_processed = self.fourier_layer(x_coord)
        else:
            x_coord_processed = x_coord
        
        # Concatenate all features
        x = torch.cat([x_cat_emb, x_num, x_coord_processed], dim=1)
        
        return self.layers(x)

# ==============================================================================
# 4. The Regressor Wrapper Class
# ==============================================================================
class FeedForwardNNRegressorWithEmbeddings3:

    def __init__(self, categorical_features, coord_features=[], use_fourier_features=False,
                 output_size=1, batch_size=16, learning_rate=0.001, num_epochs=10, 
                 hidden_sizes=[1024], patience=10, loss_fn='mse', n_bins=10, gamma=1.0, random_state=None):
        self.categorical_features = categorical_features
        self.coord_features = coord_features
        self.use_fourier_features = use_fourier_features
        self.numerical_features = []
        self.output_size = output_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.hidden_sizes = hidden_sizes
        self.patience = patience
        self.loss_fn_name = loss_fn
        self.n_bins = n_bins
        self.gamma = gamma
        self.model = None
        self.category_mappings = {}
        self.embedding_specs = []
        self.random_state = random_state

    def fit(self, X, y, X_val=None, y_val=None):
        # --- Data Preprocessing ---
        self.numerical_features = [col for col in X.columns if col not in self.categorical_features and col not in self.coord_features]
        # print(f"Categorical features: {self.categorical_features}")
        # print(f"Numerical features: {self.numerical_features}")
        # print(f"Coordinate features: {self.coord_features}")

        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)
            random.seed(self.random_state)

        if X_val is not None and y_val is not None:
            X_train, y_train = X.copy(), y.copy()
            X_val, y_val = X_val.copy(), y_val.copy()
        else:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=self.random_state)

        # BAD FIX: reset specs to empty list
        if len(self.embedding_specs) > 0:
            self.embedding_specs = []

        for col in self.categorical_features:
            categories = X_train[col].unique()
            self.category_mappings[col] = {cat: i for i, cat in enumerate(categories)}
            num_categories = len(categories)
            embedding_dim = min(50, (num_categories + 1) // 2) if num_categories > 4 else num_categories
            self.embedding_specs.append((num_categories, embedding_dim))

        def _create_tensors(X_df, y_series):
            X_cat_tensors = [torch.tensor(X_df[col].map(self.category_mappings[col]).fillna(0).values, dtype=torch.long) for col in self.categorical_features]
            X_cat = torch.stack(X_cat_tensors, dim=1)
            X_num = torch.tensor(X_df[self.numerical_features].astype(float).values, dtype=torch.float32)
            X_coord = torch.tensor(X_df[self.coord_features].astype(float).values, dtype=torch.float32)
            y_tensor = torch.tensor(y_series.values.reshape(-1, 1), dtype=torch.float32)
            return X_cat, X_num, X_coord, y_tensor

        X_cat_train, X_num_train, X_coord_train, y_train_tensor = _create_tensors(X_train, y_train)
        X_cat_val, X_num_val, X_coord_val, y_val_tensor = _create_tensors(X_val, y_val)

        train_dataset = TensorDataset(X_cat_train, X_num_train, X_coord_train, y_train_tensor)
        val_dataset = TensorDataset(X_cat_val, X_num_val, X_coord_val, y_val_tensor)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.batch_size * 2)

        self.model = NNWithEmbeddings(
            embedding_specs=self.embedding_specs,
            num_numerical_features=len(self.numerical_features),
            num_coord_features=len(self.coord_features),
            use_fourier_features=self.use_fourier_features,
            fourier_mapping_size=16, # Can be tuned
            layer_sizes=self.hidden_sizes + [self.output_size]
        ).to(device)
        
        # --- Loss Function Selection ---
        if self.loss_fn_name == 'huber':
            criterion = nn.HuberLoss()
        elif self.loss_fn_name == 'focal_mse':
            criterion = FocalMSELoss(gamma=self.gamma)
        elif self.loss_fn_name == 'binned_mse':
            y_min, y_max = y_train.min(), y_train.max()
            criterion = BinnedMSELoss(y_min=y_min, y_max=y_max, n_bins=self.n_bins, gamma=self.gamma)
        else: # Default to MSE
            criterion = nn.MSELoss()
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(self.num_epochs):
            total_train_loss = 0
            self.model.train()
            for batch_cat, batch_num, batch_coord, batch_y in train_loader:
                batch_cat, batch_num, batch_coord, batch_y = batch_cat.to(device), batch_num.to(device), batch_coord.to(device), batch_y.to(device)
                outputs = self.model(batch_cat, batch_num, batch_coord)
                loss = criterion(outputs, batch_y)
                total_train_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            self.model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch_cat, batch_num, batch_coord, batch_y in val_loader:
                    batch_cat, batch_num, batch_coord, batch_y = batch_cat.to(device), batch_num.to(device), batch_coord.to(device), batch_y.to(device)
                    outputs = self.model(batch_cat, batch_num, batch_coord)
                    loss = criterion(outputs, batch_y)
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
        
        X_processed = X.copy()
        
        X_cat_list = [X_processed[col].map(self.category_mappings[col]).fillna(0).astype(int).values.reshape(-1, 1) for col in self.categorical_features]
        X_cat_np = np.hstack(X_cat_list)
        X_cat = torch.tensor(X_cat_np, dtype=torch.long)
        X_num = torch.tensor(X_processed[self.numerical_features].values, dtype=torch.float32)
        X_coord = torch.tensor(X_processed[self.coord_features].values, dtype=torch.float32)
        
        test_dataset = TensorDataset(X_cat, X_num, X_coord)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)

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
