
# ===============================================================================
#   (3) TabTransformerRegressor
# ===============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict
import warnings

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================================================================
# 1. Specialized Modules for TabTransformer
# ==============================================================================

class NumericalFeatureTokenizer(nn.Module):
    """
    A learnable tokenizer for numerical features.
    It projects each numerical feature into a high-dimensional embedding space.
    """
    def __init__(self, num_numerical_features, embedding_dim):
        super().__init__()
        self.num_features = num_numerical_features
        self.embedding_dim = embedding_dim
        # A learnable weight for each numerical feature
        self.weights = nn.Parameter(torch.randn(num_numerical_features, embedding_dim))
        # A learnable bias for each numerical feature
        self.biases = nn.Parameter(torch.randn(num_numerical_features, embedding_dim))

    def forward(self, x_num):
        # x_num has shape (batch_size, num_numerical_features)
        # Reshape x_num to (batch_size, num_numerical_features, 1) for broadcasting
        x_num = x_num.unsqueeze(-1)
        # Apply learnable transformation: x * weight + bias
        return x_num * self.weights + self.biases

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

# ==============================================================================
# 3. The TabTransformer Model
# ==============================================================================

class TabTransformer(nn.Module):
    """
    A Transformer-based model for tabular data.
    """
    def __init__(self, embedding_specs, num_numerical_features, num_coord_features,
                 fourier_mapping_size, transformer_dim, transformer_heads,
                 transformer_layers, dropout, output_size):
        super().__init__()
        
        # --- Feature Tokenizers ---
        self.embedding_layers = nn.ModuleList([nn.Embedding(num, dim) for num, dim in embedding_specs])
        total_embedding_dim = sum(dim for _, dim in embedding_specs)
        
        self.numerical_tokenizer = NumericalFeatureTokenizer(num_numerical_features, transformer_dim)
        
        self.fourier_features = None
        if num_coord_features > 0:
            self.fourier_features = FourierFeatures(num_coord_features, fourier_mapping_size)
            # Each coordinate feature gets projected to 2 * mapping_size
            self.coord_projector = nn.Linear(2 * fourier_mapping_size, transformer_dim)

        # --- Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim, nhead=transformer_heads,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
        # --- Final MLP Head ---
        # Project categorical embeddings to the transformer dimension
        self.cat_projector = nn.Linear(total_embedding_dim, transformer_dim * len(embedding_specs))
        
        # The final layer projects the processed tokens to the output size
        self.output_layer = nn.Linear(transformer_dim * self.get_num_tokens(num_numerical_features, num_coord_features), output_size)

    def get_num_tokens(self, num_numerical, num_coord):
        return len(self.embedding_layers) + num_numerical + (1 if num_coord > 0 else 0)

    def forward(self, x_cat, x_num, x_coord):
        tokens = []
        
        # Categorical tokens
        cat_embeddings = [emb_layer(x_cat[:, i]) for i, emb_layer in enumerate(self.embedding_layers)]
        cat_embeddings = torch.cat(cat_embeddings, dim=1)
        cat_tokens = self.cat_projector(cat_embeddings).view(x_cat.size(0), len(self.embedding_layers), -1)
        tokens.append(cat_tokens)
        
        # Numerical tokens
        if x_num.size(1) > 0:
            num_tokens = self.numerical_tokenizer(x_num)
            tokens.append(num_tokens)
            
        # Coordinate tokens (as a single token)
        if x_coord.size(1) > 0:
            coord_fourier = self.fourier_features(x_coord)
            coord_token = self.coord_projector(coord_fourier).unsqueeze(1)
            tokens.append(coord_token)
            
        # Combine all tokens and pass through the Transformer
        x = torch.cat(tokens, dim=1)
        x = self.transformer_encoder(x)
        
        # Flatten the output and pass to the final layer
        x = x.flatten(start_dim=1)
        return self.output_layer(x)

# ==============================================================================
# 4. The Main User-Facing Wrapper Class
# ==============================================================================
"""
Args:
    categorical_features (list[str]): A list of column names in the input DataFrame
                                      that should be treated as categorical features.
    coord_features (list[str]): A list of column names to be treated as coordinates
                                and encoded with Fourier features.
    output_size (int, optional): The number of output neurons, typically 1 for regression.
                                 Defaults to 1.
    batch_size (int, optional): The number of samples per batch for training.
                                Defaults to 32.
    learning_rate (float, optional): The learning rate for the Adam optimizer.
                                     Defaults to 0.001.
    num_epochs (int, optional): The number of complete passes through the training dataset.
                                Defaults to 10.
    transformer_dim (int, optional): The embedding dimension for all feature tokens and the
                                     internal dimension of the Transformer layers.
                                     Defaults to 32.
    transformer_heads (int, optional): The number of attention heads in each Transformer layer.
                                       Must be a divisor of `transformer_dim`. Defaults to 8.
    transformer_layers (int, optional): The number of Transformer encoder layers.
                                        Defaults to 6.
    dropout (float, optional): The dropout rate used in the Transformer layers.
                               Defaults to 0.1.
    loss_fn (str, optional): The loss function to use. Options are 'mse', 'focal_mse',
                             or 'huber'. Defaults to 'mse'.
    random_state (int | None, optional): Seed for reproducibility. Defaults to None.
"""

class TabTransformerRegressor:

    def __init__(self, categorical_features, coord_features, output_size=1, batch_size=32, learning_rate=0.001,
                 num_epochs=10, transformer_dim=32, transformer_heads=8, transformer_layers=6,
                 dropout=0.1, loss_fn='mse', random_state=None):
        self.categorical_features = categorical_features
        self.coord_features = coord_features
        self.numerical_features = []
        self.output_size = output_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.transformer_dim = transformer_dim
        self.transformer_heads = transformer_heads
        self.transformer_layers = transformer_layers
        self.dropout = dropout
        self.loss_fn_name = loss_fn
        self.random_state = random_state
        
        self.model = None
        self.scaler = StandardScaler()
        self.category_mappings = {}

    def fit(self, X, y):
        self.numerical_features = [col for col in X.columns if col not in self.categorical_features and col not in self.coord_features]
        X_fit = X.copy()
        y_fit = y.copy()

        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)
            random.seed(self.random_state)

        if self.numerical_features:
            X_fit[self.numerical_features] = self.scaler.fit_transform(X_fit[self.numerical_features])

        for col in self.categorical_features:
            self.category_mappings[col] = {cat: i for i, cat in enumerate(X_fit[col].unique())}
        
        X_cat_tensors = [torch.tensor(X_fit[col].map(self.category_mappings[col]).values, dtype=torch.long) for col in self.categorical_features]
        X_cat = torch.stack(X_cat_tensors, dim=1)
        X_num = torch.tensor(X_fit[self.numerical_features].values, dtype=torch.float32)
        X_coord = torch.tensor(X_fit[self.coord_features].values, dtype=torch.float32)
        y_tensor = torch.tensor(y_fit.values, dtype=torch.float32).unsqueeze(1)
        
        dataset = TensorDataset(X_cat, X_num, X_coord, y_tensor)
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        embedding_specs = [(len(self.category_mappings[col]), min(50, (len(self.category_mappings[col])+1)//2)) for col in self.categorical_features]
        
        self.model = TabTransformer(
            embedding_specs=embedding_specs,
            num_numerical_features=len(self.numerical_features),
            num_coord_features=len(self.coord_features),
            fourier_mapping_size=16, # Can be tuned
            transformer_dim=self.transformer_dim,
            transformer_heads=self.transformer_heads,
            transformer_layers=self.transformer_layers,
            dropout=self.dropout,
            output_size=self.output_size
        ).to(device)
        
        # Select loss function
        if self.loss_fn_name == 'focal_mse':
            criterion = FocalMSELoss()
        elif self.loss_fn_name == 'huber':
            criterion = nn.HuberLoss()
        else: # Default to MSE
            criterion = nn.MSELoss()
            
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        print("Starting TabTransformer training...")
        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch_cat, batch_num, batch_coord, batch_y in loader:
                batch_cat, batch_num, batch_coord, batch_y = batch_cat.to(device), batch_num.to(device), batch_coord.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                y_pred = self.model(batch_cat, batch_num, batch_coord)
                loss = criterion(y_pred, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss:.4f}')
        
        print("Training finished.")

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("You must call fit() before predicting.")
        
        X_pred = X.copy()
        if self.numerical_features:
            X_pred[self.numerical_features] = self.scaler.transform(X_pred[self.numerical_features])
        for col in self.categorical_features:
            # Handle unseen categories during prediction by mapping them to a default value (e.g., 0)
            X_pred[col] = X_pred[col].map(self.category_mappings[col]).fillna(0)
        
        # Prepare tensors without moving to device yet
        X_cat = torch.stack([torch.tensor(X_pred[col].values, dtype=torch.long) for col in self.categorical_features], dim=1)
        X_num = torch.tensor(X_pred[self.numerical_features].values, dtype=torch.float32)
        X_coord = torch.tensor(X_pred[self.coord_features].values, dtype=torch.float32)
        
        # Create a DataLoader for batching predictions
        dataset = TensorDataset(X_cat, X_num, X_coord)
        # Use a larger batch size for prediction as we don't need to store gradients
        loader = DataLoader(dataset, batch_size=self.batch_size * 4, shuffle=False)

        self.model.eval()
        all_predictions = []
        with torch.no_grad():
            for batch_cat, batch_num, batch_coord in loader:
                # Move each batch to the device
                batch_cat, batch_num, batch_coord = batch_cat.to(device), batch_num.to(device), batch_coord.to(device)
                
                predictions = self.model(batch_cat, batch_num, batch_coord)
                all_predictions.append(predictions.cpu().numpy())
        
        return np.concatenate(all_predictions).flatten()

    def set_params(self, **params):
        """
        Sets the parameters of this estimator. This method is compatible with
        scikit-learn's GridSearchCV.
        """
        for key, value in params.items():
            # Use hasattr to check if the attribute exists before setting it
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                # This helps catch typos or invalid parameter names
                warnings.warn(f"Invalid parameter {key} for estimator {self.__class__.__name__}.")
        return self
