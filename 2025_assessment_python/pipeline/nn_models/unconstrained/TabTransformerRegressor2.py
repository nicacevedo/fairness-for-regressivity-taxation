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
# 3. The Enhanced TabTransformer Model with [CLS] Token
# ==============================================================================

class TabTransformer(nn.Module):
    """
    A Transformer-based model for tabular data, enhanced with a [CLS] token.
    """
    def __init__(self, embedding_specs, num_numerical_features, num_coord_features,
                 fourier_mapping_size, transformer_dim, transformer_heads,
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
            self.fourier_features = FourierFeatures(num_coord_features, fourier_mapping_size)
            self.coord_projector = nn.Linear(2 * fourier_mapping_size, transformer_dim)

        # --- Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim, nhead=transformer_heads,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
        # --- Final MLP Head ---
        self.cat_projector = nn.Linear(total_embedding_dim, transformer_dim * len(embedding_specs))
        
        # The final layer now only takes the [CLS] token's output
        self.output_layer = nn.Linear(transformer_dim, output_size)

    def forward(self, x_cat, x_num, x_coord):
        tokens = []
        
        # Categorical tokens
        if x_cat.size(1) > 0:
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
            
        # Combine all feature tokens
        feature_tokens = torch.cat(tokens, dim=1)
        
        # Prepend the [CLS] token
        cls_tokens = self.cls_token.expand(feature_tokens.size(0), -1, -1)
        x = torch.cat([cls_tokens, feature_tokens], dim=1)
        
        # Pass through the Transformer
        x = self.transformer_encoder(x)
        
        # Use only the output of the [CLS] token for prediction
        cls_output = x[:, 0]
        return self.output_layer(cls_output)

# ==============================================================================
# 4. The Main User-Facing Wrapper Class with Enhancements
# ==============================================================================
class TabTransformerRegressor2:

    def __init__(self, categorical_features, coord_features, output_size=1, batch_size=32, learning_rate=0.001,
                 num_epochs=100, transformer_dim=32, transformer_heads=8, transformer_layers=6,
                 dropout=0.1, loss_fn='mse', patience=10, random_state=None):
        self.categorical_features = list(categorical_features)
        self.coord_features = list(coord_features)
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

    def _feature_engineer(self, X):
        """Applies feature engineering steps."""
        X_eng = X.copy()
        if 'char_yrblt' in X_eng.columns:
            print("Binning 'char_yrblt' into decades.")
            bins = [1900, 1920, 1940, 1960, 1980, 2000, 2025]
            labels = ['pre-1920', '1920-1940', '1940-1960', '1960-1980', '1980-2000', 'post-2000']
            X_eng['yrblt_decade'] = pd.cut(X_eng['char_yrblt'], bins=bins, labels=labels, right=False).astype(str)
            
            if 'yrblt_decade' not in self.categorical_features:
                self.categorical_features.append('yrblt_decade')
        return X_eng

    def fit(self, X, y):
        X_fit = self._feature_engineer(X)
        y_fit = y.copy()
        
        self.numerical_features = [col for col in X_fit.columns if col not in self.categorical_features and col not in self.coord_features]

        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)
            random.seed(self.random_state)

        # Split data for early stopping
        X_train, X_val, y_train, y_val = train_test_split(X_fit, y_fit, test_size=0.2, random_state=self.random_state)

        if self.numerical_features:
            self.scaler.fit(X_train[self.numerical_features])
            X_train[self.numerical_features] = self.scaler.transform(X_train[self.numerical_features])
            X_val[self.numerical_features] = self.scaler.transform(X_val[self.numerical_features])

        for col in self.categorical_features:
            self.category_mappings[col] = {cat: i for i, cat in enumerate(X_train[col].unique())}
        
        def _create_tensors(X_df, y_series):
            X_cat = torch.stack([torch.tensor(X_df[col].map(self.category_mappings[col]).fillna(0).values, dtype=torch.long) for col in self.categorical_features], dim=1)
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

        embedding_specs = [(len(self.category_mappings[col]), min(50, (len(self.category_mappings[col])+1)//2)) for col in self.categorical_features]
        
        self.model = TabTransformer(
            embedding_specs=embedding_specs, num_numerical_features=len(self.numerical_features),
            num_coord_features=len(self.coord_features), fourier_mapping_size=16,
            transformer_dim=self.transformer_dim, transformer_heads=self.transformer_heads,
            transformer_layers=self.transformer_layers, dropout=self.dropout, output_size=self.output_size
        ).to(device)
        
        criterion = FocalMSELoss() if self.loss_fn_name == 'focal_mse' else nn.HuberLoss() if self.loss_fn_name == 'huber' else nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=self.patience // 2, factor=0.5)

        print("Starting Enhanced TabTransformer training...")
        best_val_loss = float('inf')
        patience_counter = 0

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

            # Validation loop
            self.model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch_cat, batch_num, batch_coord, batch_y in val_loader:
                    batch_cat, batch_num, batch_coord, batch_y = batch_cat.to(device), batch_num.to(device), batch_coord.to(device), batch_y.to(device)
                    y_pred = self.model(batch_cat, batch_num, batch_coord)
                    loss = criterion(y_pred, batch_y)
                    total_val_loss += loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            scheduler.step(avg_val_loss)

            print(f'Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print("Early stopping triggered.")
                    break
        
        self.model.load_state_dict(torch.load('best_model.pth'))
        print("Training finished.")

    def predict(self, X):
        if self.model is None: raise RuntimeError("You must call fit() before predicting.")
        
        X_pred = self._feature_engineer(X)
        if self.numerical_features:
            X_pred[self.numerical_features] = self.scaler.transform(X_pred[self.numerical_features])
        for col in self.categorical_features:
            X_pred[col] = X_pred[col].map(self.category_mappings[col]).fillna(0)
        
        X_cat = torch.stack([torch.tensor(X_pred[col].values, dtype=torch.long) for col in self.categorical_features], dim=1)
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
