import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from collections import OrderedDict
import warnings

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================================================================
# 1. The Wide & Deep Model Architecture
# ==============================================================================

class WideAndDeep(nn.Module):
    """
    A Wide & Deep model for tabular data.
    The "Wide" part is a linear model over raw and crossed features.
    The "Deep" part is a feed-forward neural network over dense embeddings.
    """
    def __init__(self, wide_dim, embedding_specs, num_numerical_features, deep_layer_sizes, output_size):
        super().__init__()
        
        # --- Wide Component ---
        # A simple linear layer for the one-hot encoded features
        self.wide_model = nn.Linear(wide_dim, output_size)
        
        # --- Deep Component ---
        self.embedding_layers = nn.ModuleList([nn.Embedding(num, dim) for num, dim in embedding_specs])
        total_embedding_dim = sum(dim for _, dim in embedding_specs)
        deep_input_size = total_embedding_dim + num_numerical_features
        
        deep_layers = []
        for i, size in enumerate(deep_layer_sizes):
            activation = nn.ReLU() if i < len(deep_layer_sizes) - 1 else None
            deep_layers.append((f"linear_{i}", nn.Linear(deep_input_size, size)))
            if activation:
                deep_layers.append((f"relu_{i}", activation))
            deep_input_size = size
        deep_layers.append(("output_linear", nn.Linear(deep_input_size, output_size)))
        self.deep_model = nn.Sequential(OrderedDict(deep_layers))

    def forward(self, x_wide, x_cat_deep, x_num_deep):
        # Process the wide part
        wide_output = self.wide_model(x_wide)
        
        # Process the deep part
        deep_embeddings = [emb_layer(x_cat_deep[:, i]) for i, emb_layer in enumerate(self.embedding_layers)]
        x_cat_emb = torch.cat(deep_embeddings, dim=1)
        x_deep = torch.cat([x_cat_emb, x_num_deep], dim=1)
        deep_output = self.deep_model(x_deep)
        
        # Combine the outputs
        return wide_output + deep_output

# ==============================================================================
# 2. The Main User-Facing Wrapper Class
# ==============================================================================
"""
Args:
    categorical_features (list[str]): A list of column names in the input DataFrame
                                      that should be treated as categorical. These
                                      will be used in both the "wide" (one-hot encoded)
                                      and "deep" (embedding) parts of the model.

    output_size (int, optional): The number of output neurons, which is typically 1 for
                                 a regression task. Defaults to 1.

    batch_size (int, optional): The number of samples to process in each batch during
                                training. Defaults to 32.

    learning_rate (float, optional): The learning rate for the Adam optimizer.
                                     Defaults to 0.001.

    num_epochs (int, optional): The total number of times the model will iterate over
                                the entire training dataset. Defaults to 10.

    hidden_sizes (list[int], optional): A list of integers defining the size and number
                                        of hidden layers in the "deep" neural network
                                        component. For example, [50, 25] creates two
                                        hidden layers with 50 and 25 neurons, respectively.
                                        Defaults to [50, 25].

    random_state (int | None, optional): An integer seed to ensure reproducibility of
                                         weight initialization and data shuffling.
                                         Defaults to None.
"""
class WideAndDeepRegressor:

    def __init__(self, categorical_features, output_size=1, batch_size=32, learning_rate=0.001,
                 num_epochs=10, hidden_sizes=[50, 25], random_state=None):
        self.categorical_features = categorical_features
        self.numerical_features = []
        self.output_size = output_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.hidden_sizes = hidden_sizes
        self.random_state = random_state
        
        self.model = None
        self.scaler = StandardScaler()
        self.one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.category_mappings = {}

    def fit(self, X, y):
        self.numerical_features = [col for col in X.columns if col not in self.categorical_features]
        X_fit = X.copy()
        y_fit = y.copy()

        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)
            random.seed(self.random_state)

        # --- Preprocessing ---
        # Scale numerical features
        if self.numerical_features:
            X_fit[self.numerical_features] = self.scaler.fit_transform(X_fit[self.numerical_features])
        
        # Create one-hot encodings for the "wide" part
        X_wide_cat = self.one_hot_encoder.fit_transform(X_fit[self.categorical_features])
        
        # Create integer mappings for the "deep" part embeddings
        for col in self.categorical_features:
            self.category_mappings[col] = {cat: i for i, cat in enumerate(X_fit[col].unique())}
        
        # --- Tensor Conversion ---
        X_wide = torch.tensor(np.hstack([X_wide_cat, X_fit[self.numerical_features].values]), dtype=torch.float32)
        
        X_cat_deep_tensors = [torch.tensor(X_fit[col].map(self.category_mappings[col]).values, dtype=torch.long) for col in self.categorical_features]
        X_cat_deep = torch.stack(X_cat_deep_tensors, dim=1)
        X_num_deep = torch.tensor(X_fit[self.numerical_features].values, dtype=torch.float32)
        y_tensor = torch.tensor(y_fit.values, dtype=torch.float32).unsqueeze(1)
        
        dataset = TensorDataset(X_wide, X_cat_deep, X_num_deep, y_tensor)
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        # --- Model Initialization ---
        wide_dim = X_wide.shape[1]
        embedding_specs = [(len(self.category_mappings[col]), min(50, (len(self.category_mappings[col])+1)//2)) for col in self.categorical_features]
        
        self.model = WideAndDeep(
            wide_dim=wide_dim,
            embedding_specs=embedding_specs,
            num_numerical_features=len(self.numerical_features),
            deep_layer_sizes=self.hidden_sizes,
            output_size=self.output_size
        ).to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        print("Starting Wide & Deep training...")
        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch_wide, batch_cat_deep, batch_num_deep, batch_y in loader:
                batch_wide, batch_cat_deep, batch_num_deep, batch_y = batch_wide.to(device), batch_cat_deep.to(device), batch_num_deep.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                y_pred = self.model(batch_wide, batch_cat_deep, batch_num_deep)
                loss = criterion(y_pred, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss:.4f}')
        
        print("Training finished.")

    def predict(self, X):
        if self.model is None: raise RuntimeError("You must call fit() before predicting.")
        X_pred = X.copy()
        
        if self.numerical_features:
            X_pred[self.numerical_features] = self.scaler.transform(X_pred[self.numerical_features])
        
        X_wide_cat = self.one_hot_encoder.transform(X_pred[self.categorical_features])
        
        for col in self.categorical_features:
            X_pred[col] = X_pred[col].map(self.category_mappings[col]).fillna(0)
        
        X_wide = torch.tensor(np.hstack([X_wide_cat, X_pred[self.numerical_features].values]), dtype=torch.float32).to(device)
        X_cat_deep = torch.stack([torch.tensor(X_pred[col].values, dtype=torch.long) for col in self.categorical_features], dim=1).to(device)
        X_num_deep = torch.tensor(X_pred[self.numerical_features].values, dtype=torch.float32).to(device)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_wide, X_cat_deep, X_num_deep)
        return predictions.cpu().numpy().flatten()

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
