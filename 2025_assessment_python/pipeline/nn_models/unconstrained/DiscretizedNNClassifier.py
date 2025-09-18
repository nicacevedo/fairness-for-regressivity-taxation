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
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.cluster import KMeans

# Assume 'device' is configured (e.g., device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            self.output_dim = in_features
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
# 2. The Core Neural Network Model (Unchanged)
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
# 3. Discretized NN Regressor (New Class)
# ==============================================================================
class DiscretizedNNClassifier:
    """
    A two-stage model that first discretizes the target variable into bins
    and then trains a neural network classifier on these bins.
    """
    def __init__(self, categorical_features, coord_features=[],
                 engineer_time_features=False, bin_yrblt=False, cross_township_class=False,
                 fourier_type='none', fourier_mapping_size=16, fourier_sigma=1.25,
                 n_bins=10, binning_method='quantile', batch_size=16, learning_rate=0.001,
                 num_epochs=10, hidden_sizes=[1024], patience=10,
                 random_state=None, dropout=0.5, l2_lambda=0, l1_lambda=0,
                 use_scaler=False, normalization_type='none'):
        
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
        self.n_bins = n_bins
        self.binning_method = binning_method
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.l2_lambda = l2_lambda
        self.l1_lambda = l1_lambda
        self.patience = patience
        self.model = None
        self.category_mappings = {}
        self.embedding_specs = []
        self.random_state = random_state
        self.normalization_type = normalization_type
        self.bin_info = {}

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
        
        # --- Binning the target variable ---
        if self.binning_method == 'quantile':
            y_bins, bin_edges = pd.qcut(y, q=self.n_bins, labels=False, retbins=True, duplicates='drop')
            y_bin_labels = y_bins.values
            self.bin_info['edges'] = bin_edges
            self.bin_info['medians'] = [y.loc[y_bins == i].median() for i in range(len(bin_edges) - 1)]
            self.bin_info['means'] = [y.loc[y_bins == i].mean() for i in range(len(bin_edges) - 1)]
            self.bin_info['num_bins'] = len(bin_edges) - 1
            if self.n_bins != self.bin_info['num_bins']:
                warnings.warn(f"Number of bins reduced to {self.bin_info['num_bins']} due to duplicate quantiles.", UserWarning)
        elif self.binning_method == 'uniform':
            y_bins, bin_edges = pd.cut(y, bins=self.n_bins, labels=False, retbins=True, duplicates='drop')
            y_bin_labels = y_bins.values
            self.bin_info['edges'] = bin_edges
            self.bin_info['medians'] = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges) - 1)]
            self.bin_info['num_bins'] = len(bin_edges) - 1
        elif self.binning_method == 'kmeans':
            kmeans = KMeans(n_clusters=self.n_bins, random_state=self.random_state, n_init='auto')
            y_bin_labels = kmeans.fit_predict(y.values.reshape(-1, 1))
            self.bin_info['centers'] = sorted(kmeans.cluster_centers_.flatten())
            self.bin_info['labels'] = y_bin_labels
            self.bin_info['num_bins'] = self.n_bins
        else:
            raise ValueError("Invalid binning_method. Choose from 'quantile', 'uniform', 'kmeans'.")
        
        # Store bin info for the validation set
        if X_val is not None and y_val is not None:
            if self.binning_method == 'quantile' or self.binning_method == 'uniform':
                y_val_labels = pd.cut(y_val, bins=self.bin_info['edges'], labels=False, include_lowest=True).values
                # Handle NaNs from out-of-bin values
                y_val_labels = np.nan_to_num(y_val_labels, nan=self.bin_info['num_bins'] - 1).astype(int)
            elif self.binning_method == 'kmeans':
                y_val_labels = kmeans.predict(y_val.values.reshape(-1, 1))
        
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
        X_train, y_train_labels = X_eng, y_bin_labels
        if X_val is None or y_val is None:
            X_train, X_val, y_train_labels, y_val_labels = train_test_split(X_eng, y_bin_labels, test_size=0.2, random_state=self.random_state, stratify=y_bin_labels)
        else:
            X_train, X_val = X_eng, self._engineer_features(X_val)
        
        # --- Scaling numerical features ---
        if self.use_scaler:
            self.scaler = StandardScaler()
            # Fit the scaler on the training numerical data only
            X_train_num_scaled = self.scaler.fit_transform(X_train[self.numerical_features])
            X_val_num_scaled = self.scaler.transform(X_val[self.numerical_features])
        else:
            X_train_num_scaled = X_train[self.numerical_features].values
            X_val_num_scaled = X_val[self.numerical_features].values
            
        # --- Prepare Tensors and Dataloaders ---
        self.embedding_specs = []
        for col in self.categorical_features:
            categories = X_train[col].unique()
            self.category_mappings[col] = {cat: i + 1 for i, cat in enumerate(categories)}
            self.category_mappings[col]['__UNKNOWN__'] = 0
            num_categories_with_unknown = len(categories) + 1
            embedding_dim = min(50, (num_categories_with_unknown + 1) // 2)
            self.embedding_specs.append((num_categories_with_unknown, embedding_dim))

        def _create_tensors(X_df_cat, X_num_scaled, X_df_coord, y_labels):
            X_cat_tensors = [torch.tensor(X_df_cat[col].map(self.category_mappings[col]).fillna(0).values, dtype=torch.long) for col in self.categorical_features]
            X_cat = torch.stack(X_cat_tensors, dim=1) if X_cat_tensors else torch.empty(len(X_df_cat), 0, dtype=torch.long)
            X_num = torch.tensor(X_num_scaled, dtype=torch.float32)
            X_coord = torch.tensor(X_df_coord[self.coord_features].astype(float).values, dtype=torch.float32)
            y_tensor = torch.tensor(y_labels, dtype=torch.long)
            return X_cat, X_num, X_coord, y_tensor

        X_cat_train, X_num_train, X_coord_train, y_train_tensor = _create_tensors(X_train, X_train_num_scaled, X_train, y_train_labels)
        X_cat_val, X_num_val, X_coord_val, y_val_tensor = _create_tensors(X_val, X_val_num_scaled, X_val, y_val_labels)
        
        train_dataset = TensorDataset(X_cat_train, X_num_train, X_coord_train, y_train_tensor)
        val_dataset = TensorDataset(X_cat_val, X_num_val, X_coord_val, y_val_tensor)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.batch_size * 2, shuffle=False)

        # --- Model Initialization ---
        self.model = NNWithEmbeddings(
            embedding_specs=self.embedding_specs,
            num_numerical_features=len(self.numerical_features),
            num_coord_features=len(self.coord_features),
            fourier_type=self.fourier_type,
            fourier_mapping_size=self.fourier_mapping_size,
            fourier_sigma=self.fourier_sigma,
            layer_sizes=self.hidden_sizes + [self.bin_info['num_bins']],
            dropout=self.dropout,
            normalization_type=self.normalization_type
        ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        
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
                    loss = criterion(outputs, batch_y)

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

        # --- Final Report ---
        y_val_pred_bins = self._predict_classes(X_val)
        y_val_true_bins = y_val_labels
        
        print("\n" + "="*50)
        print("Final Classification Report on Validation Set")
        print("="*50)
        print(classification_report(y_val_true_bins, y_val_pred_bins))
        print("Confusion Matrix:\n", confusion_matrix(y_val_true_bins, y_val_pred_bins))
        
        y_val_pred_numeric = self._convert_to_numeric(y_val_pred_bins)
        print("\n" + "="*50)
        print("Final Regression Metrics on Validation Set")
        print("="*50)
        print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_val, y_val_pred_numeric):.4f}")
        print(f"Root Mean Squared Error (RMSE): {root_mean_squared_error(y_val, y_val_pred_numeric):.4f}")
        print(f"R-squared (R2): {r2_score(y_val, y_val_pred_numeric):.4f}")
        print("="*50)

    def _convert_to_numeric(self, y_pred_bins):
        if self.binning_method == 'kmeans':
            return np.array([self.bin_info['centers'][i] for i in y_pred_bins])
        elif self.binning_method == 'quantile':
            return np.array([self.bin_info['medians'][i] for i in y_pred_bins])
        else: # uniform
            return np.array([ (self.bin_info['edges'][i] + self.bin_info['edges'][i+1]) / 2 for i in y_pred_bins])

    def _predict_classes(self, X_df):
        if self.model is None:
            raise RuntimeError("You must call fit() before calling predict().")
        
        X_pred = self._engineer_features(X_df)
        
        # --- Scaling numerical features ---
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
        
        return np.argmax(np.array(all_predictions), axis=1)

    def predict(self, X_df):
        predicted_classes = self._predict_classes(X_df)
        return self._convert_to_numeric(predicted_classes)
