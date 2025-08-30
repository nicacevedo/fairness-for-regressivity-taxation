# V3: CLS token for the output prediction and mulitple options for the fourier features.
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
    A module to generate various types of Fourier features for coordinate data.
    """
    def __init__(self, in_features, mapping_size, scale=10.0, fourier_type='gaussian', sigma=1.25):
        super().__init__()
        self.in_features = in_features
        self.mapping_size = mapping_size
        self.fourier_type = fourier_type

        if fourier_type == 'gaussian':
            # Random Fourier Features (Gaussian)
            self.B = nn.Parameter(torch.randn(in_features, mapping_size) * scale, requires_grad=False)
            self.output_dim = 2 * mapping_size
        elif fourier_type == 'positional':
            # Positional Encoding with log-linear frequencies
            if sigma is None:
                raise ValueError("sigma must be provided for positional Fourier features.")
            freq_bands = (sigma ** (torch.arange(mapping_size) / mapping_size))
            self.register_buffer('freq_bands', freq_bands)
            self.output_dim = 2 * in_features * mapping_size
        elif fourier_type == 'basic':
            # Basic sin/cos mapping
            self.output_dim = 2 * in_features
        elif fourier_type == 'none':
            # Pass through raw coordinates
            self.output_dim = in_features
        else:
            raise ValueError(f"Unknown fourier_type: {fourier_type}")

    def forward(self, x):
        if self.fourier_type == 'gaussian':
            x_proj = 2 * np.pi * x @ self.B
            return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        elif self.fourier_type == 'positional':
            # Broadcast frequencies across input features
            x_proj = 2 * np.pi * x.unsqueeze(-1) * self.freq_bands.view(1, 1, -1)
            # Flatten last two dimensions before sin/cos
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
        # Use the loss itself as a proxy for "hardness"
        focal_weight = mse_loss.detach() ** self.gamma
        return (focal_weight * mse_loss).mean()

# ==============================================================================
# 3. The TabTransformer Model with CLS Token and Flexible Fourier Features
# ==============================================================================

class TabTransformer(nn.Module):
    """
    A Transformer-based model for tabular data, using a [CLS] token for prediction.
    """
    def __init__(self, embedding_specs, num_numerical_features, num_coord_features,
                 fourier_type, fourier_mapping_size, fourier_sigma,
                 transformer_dim, transformer_heads,
                 transformer_layers, dropout, output_size):
        super().__init__()
        
        # --- [CLS] Token ---
        self.cls_token = nn.Parameter(torch.randn(1, 1, transformer_dim))
        
        # --- Feature Tokenizers ---
        self.embedding_layers = nn.ModuleList([nn.Embedding(num, dim) for num, dim in embedding_specs])
        total_embedding_dim = sum(dim for _, dim in embedding_specs)
        
        self.numerical_tokenizer = NumericalFeatureTokenizer(num_numerical_features, transformer_dim)
        
        self.fourier_features = None
        if num_coord_features > 0:
            self.fourier_features = FourierFeatures(num_coord_features, fourier_mapping_size, 
                                                    fourier_type=fourier_type, sigma=fourier_sigma)
            self.coord_projector = nn.Linear(self.fourier_features.output_dim, transformer_dim)

        # --- Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim, nhead=transformer_heads,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
        # --- Final MLP Head ---
        self.cat_projector = nn.Linear(total_embedding_dim, transformer_dim * len(embedding_specs))
        self.output_layer = nn.Linear(transformer_dim, output_size)

    def forward(self, x_cat, x_num, x_coord):
        tokens = []
        batch_size = x_cat.size(0)

        # 1. Prepare all feature tokens
        if len(self.embedding_layers) > 0:
            cat_embeddings = [emb_layer(x_cat[:, i]) for i, emb_layer in enumerate(self.embedding_layers)]
            cat_embeddings = torch.cat(cat_embeddings, dim=1)
            cat_tokens = self.cat_projector(cat_embeddings).view(batch_size, len(self.embedding_layers), -1)
            tokens.append(cat_tokens)
        
        if x_num.size(1) > 0:
            num_tokens = self.numerical_tokenizer(x_num)
            tokens.append(num_tokens)
            
        if x_coord.size(1) > 0:
            coord_processed = self.fourier_features(x_coord)
            coord_token = self.coord_projector(coord_processed).unsqueeze(1)
            tokens.append(coord_token)
        
        # 2. Prepend the [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        all_tokens = torch.cat([cls_tokens] + tokens, dim=1)
        
        # 3. Pass through the Transformer Encoder
        x = self.transformer_encoder(all_tokens)
        
        # 4. Use only the output of the [CLS] token for prediction
        cls_output = x[:, 0, :]
        
        return self.output_layer(cls_output)

# ==============================================================================
# 4. The Main User-Facing Wrapper Class
# ==============================================================================
class TabTransformerRegressor3:

    def __init__(self, categorical_features, coord_features, fourier_type='gaussian',
                 output_size=1, batch_size=32, learning_rate=0.001,
                 num_epochs=100, transformer_dim=32, transformer_heads=8, transformer_layers=6,
                 dropout=0.1, loss_fn='mse', patience=10, random_state=None):
        self.categorical_features = categorical_features
        self.coord_features = coord_features
        self.fourier_type = fourier_type
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
        self.patience = patience
        self.random_state = random_state
        
        self.model = None
        self.scaler = StandardScaler()
        self.category_mappings = {}
        self.embedding_specs = []

    def fit(self, X, y, X_val=None, y_val=None):
        self.numerical_features = [col for col in X.columns if col not in self.categorical_features and col not in self.coord_features]
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)
            random.seed(self.random_state)

        if X_val is not None and y_val is not None:
            X_train, y_train = X.copy(), y.copy()
            X_val, y_val = X_val.copy(), y_val.copy()
        else:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=self.random_state)

        if self.numerical_features:
            self.scaler.fit(X_train[self.numerical_features])
            X_train.loc[:, self.numerical_features] = self.scaler.transform(X_train[self.numerical_features])
            X_val.loc[:, self.numerical_features] = self.scaler.transform(X_val[self.numerical_features])

        self.embedding_specs = []
        for col in self.categorical_features:
            categories = X_train[col].unique()
            self.category_mappings[col] = {cat: i + 1 for i, cat in enumerate(categories)}
            self.category_mappings[col]['__UNKNOWN__'] = 0
            num_categories_with_unknown = len(categories) + 1
            embedding_dim = min(50, (num_categories_with_unknown + 1) // 2)
            self.embedding_specs.append((num_categories_with_unknown, embedding_dim))
        
        def _create_tensors(X_df, y_series):
            X_cat_tensors = [
                torch.tensor(X_df[col].map(self.category_mappings[col]).fillna(0).values, dtype=torch.long)
                for col in self.categorical_features
            ]
            X_cat = torch.stack(X_cat_tensors, dim=1) if X_cat_tensors else torch.empty(len(X_df), 0, dtype=torch.long)
            
            X_num = torch.tensor(X_df[self.numerical_features].values, dtype=torch.float32)
            X_coord = torch.tensor(X_df[self.coord_features].values, dtype=torch.float32)
            y_tensor = torch.tensor(y_series.values, dtype=torch.float32).unsqueeze(1)
            return X_cat, X_num, X_coord, y_tensor

        X_cat_train, X_num_train, X_coord_train, y_train_tensor = _create_tensors(X_train, y_train)
        X_cat_val, X_num_val, X_coord_val, y_val_tensor = _create_tensors(X_val, y_val)

        train_dataset = TensorDataset(X_cat_train, X_num_train, X_coord_train, y_train_tensor)
        val_dataset = TensorDataset(X_cat_val, X_num_val, X_coord_val, y_val_tensor)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.batch_size * 2)

        self.model = TabTransformer(
            embedding_specs=self.embedding_specs,
            num_numerical_features=len(self.numerical_features),
            num_coord_features=len(self.coord_features),
            fourier_type=self.fourier_type,
            fourier_mapping_size=16, # Tuneable, used for 'gaussian' and 'positional'
            fourier_sigma=1.25,      # Tuneable, used for 'positional'
            transformer_dim=self.transformer_dim,
            transformer_heads=self.transformer_heads,
            transformer_layers=self.transformer_layers,
            dropout=self.dropout,
            output_size=self.output_size
        ).to(device)
        
        criterion = FocalMSELoss() if self.loss_fn_name == 'focal_mse' else nn.HuberLoss() if self.loss_fn_name == 'huber' else nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        print("Starting TabTransformer training...")
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(self.num_epochs):
            self.model.train()
            total_train_loss = 0
            for batch_cat, batch_num, batch_coord, batch_y in train_loader:
                batch_cat, batch_num, batch_coord, batch_y = batch_cat.to(device), batch_num.to(device), batch_coord.to(device), batch_y.to(device)
                optimizer.zero_grad()
                y_pred = self.model(batch_cat, batch_num, batch_coord)
                loss = criterion(y_pred, batch_y)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            self.model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch_cat, batch_num, batch_coord, batch_y in val_loader:
                    batch_cat, batch_num, batch_coord, batch_y = batch_cat.to(device), batch_num.to(device), batch_coord.to(device), batch_y.to(device)
                    y_pred = self.model(batch_cat, batch_num, batch_coord)
                    loss = criterion(y_pred, batch_y)
                    total_val_loss += loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
            if epoch%10==0:
                print(f'Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print("Early stopping triggered.")
                    break
        
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        print("Training finished.")

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("You must call fit() before predicting.")
        
        X_pred = X.copy()
        if self.numerical_features:
            X_pred.loc[:, self.numerical_features] = self.scaler.transform(X_pred[self.numerical_features])
        
        X_cat_tensors = [
            torch.tensor(X_pred[col].map(self.category_mappings[col]).fillna(0).values, dtype=torch.long)
            for col in self.categorical_features
        ]
        X_cat = torch.stack(X_cat_tensors, dim=1) if X_cat_tensors else torch.empty(len(X_pred), 0, dtype=torch.long)
        
        X_num = torch.tensor(X_pred[self.numerical_features].values, dtype=torch.float32)
        X_coord = torch.tensor(X_pred[self.coord_features].values, dtype=torch.float32)
        
        dataset = TensorDataset(X_cat, X_num, X_coord)
        loader = DataLoader(dataset, batch_size=self.batch_size * 4, shuffle=False)

        self.model.eval()
        all_predictions = []
        with torch.no_grad():
            for batch_cat, batch_num, batch_coord in loader:
                batch_cat, batch_num, batch_coord = batch_cat.to(device), batch_num.to(device), batch_coord.to(device)
                predictions = self.model(batch_cat, batch_num, batch_coord)
                all_predictions.append(predictions.cpu().numpy())
        
        return np.concatenate(all_predictions).flatten()

    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                warnings.warn(f"Invalid parameter {key} for estimator {self.__class__.__name__}.")
        return self

