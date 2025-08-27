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
# 1. The Core Neural Network Model with Embedding Layers
# ==============================================================================
# This new model can handle both categorical and numerical features separately.
class NNWithEmbeddings(nn.Module):
    """
    A neural network model that accepts both categorical and numerical inputs.
    Categorical features are passed through embedding layers, concatenated with
    numerical features, and then fed into a series of linear layers.
    """
    def __init__(self, embedding_specs, num_numerical_features, layer_sizes):
        """
        Args:
            embedding_specs (list of tuples): Each tuple contains (num_categories, embedding_dim)
                                              for a categorical feature.
            num_numerical_features (int): The number of numerical features.
            layer_sizes (list of int): A list of hidden layer sizes and the output size.
                                       e.g., [1024, 512, 1] for two hidden layers and one output.
        """
        super().__init__()
        # Create a ModuleList to hold all embedding layers
        self.embedding_layers = nn.ModuleList([nn.Embedding(num, dim) for num, dim in embedding_specs])
        
        # Calculate the total size of the concatenated embedding vectors
        total_embedding_dim = sum(dim for _, dim in embedding_specs)
        
        # The input size for the first linear layer is the sum of all embedding dimensions
        # and the number of numerical features.
        input_size = total_embedding_dim + num_numerical_features
        
        # Create the sequential feed-forward layers
        all_layers = []
        for i, size in enumerate(layer_sizes):
            # Use ReLU for hidden layers, no activation for the output layer (typical for regression)
            activation = nn.ReLU() if i < len(layer_sizes) - 1 else None
            
            all_layers.append((f"linear_{i}", nn.Linear(input_size, size)))
            if activation:
                all_layers.append((f"relu_{i}", activation))
            input_size = size # The next layer's input is the current layer's output
            
        self.layers = nn.Sequential(OrderedDict(all_layers))

    def forward(self, x_cat, x_num):
        """
        Forward pass through the network.
        
        Args:
            x_cat (torch.Tensor): Tensor of categorical features (long integers).
                                  Shape: (batch_size, num_categorical_features)
            x_num (torch.Tensor): Tensor of numerical features (floats).
                                  Shape: (batch_size, num_numerical_features)
        
        Returns:
            torch.Tensor: The model's output.
        """
        # Get embeddings for each categorical feature
        # x_cat is sliced column by column, and each column is passed to its corresponding embedding layer
        embeddings = [emb_layer(x_cat[:, i]) for i, emb_layer in enumerate(self.embedding_layers)]
        
        # Concatenate all embedding vectors
        x_cat_emb = torch.cat(embeddings, dim=1)
        
        # Concatenate the final embedding vector with the numerical features
        x = torch.cat([x_cat_emb, x_num], dim=1)
        
        # Pass the combined tensor through the linear layers
        return self.layers(x)


# ==============================================================================
# 2. The Regressor Class Adapted for Embeddings
# ==============================================================================
# This wrapper class now identifies categorical features, prepares the data,
# and uses the NNWithEmbeddings model.
class FeedForwardNNRegressorWithEmbeddings2:

    def __init__(self, categorical_features, output_size=1, batch_size=16, learning_rate=0.001, num_epochs=10, hidden_sizes=[1024], patience=10, random_state=None):
        self.categorical_features = categorical_features
        self.numerical_features = [] # Will be determined in fit()
        self.output_size = output_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.hidden_sizes = hidden_sizes
        self.patience = patience
        self.model = None
        self.category_mappings = {} # To store mappings for prediction
        self.embedding_specs = [] # To store embedding configuration
        self.random_state = random_state

    def fit(self, X, y, X_val=None, y_val=None):
        # --- Data Preprocessing ---
        self.numerical_features = [col for col in X.columns if col not in self.categorical_features]
        print(f"Categorical features: {self.categorical_features}")
        print(f"Numerical features: {self.numerical_features}")

        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)
            random.seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)

        # Handle validation set
        if X_val is not None and y_val is not None:
            X_train, y_train = X.copy(), y.copy()
            X_val, y_val = X_val.copy(), y_val.copy()
        else:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=self.random_state)

        # Create mappings and embedding specs from training data only
        X_cat_list_train = []
        self.embedding_specs = []
        self.category_mappings = {}
        for col in self.categorical_features:
            categories = X_train[col].unique()
            self.category_mappings[col] = {cat: i for i, cat in enumerate(categories)}
            num_categories = len(categories)
            
            codes = X_train[col].map(self.category_mappings[col])
            X_cat_list_train.append(codes.values.reshape(-1, 1))

            embedding_dim = min(50, (num_categories + 1) // 2) if num_categories > 4 else num_categories # if num_categories > 10 else num_categories
            self.embedding_specs.append((num_categories, embedding_dim))
            print(f"   -> Embedding for '{col}' defined with size: ({num_categories}, {embedding_dim})")

        # Prepare Tensors for Training Set
        X_cat_np_train = np.hstack(X_cat_list_train)
        X_cat_train = torch.tensor(X_cat_np_train, dtype=torch.long, device=device)
        X_num_train = torch.tensor(X_train[self.numerical_features].astype(float).values, dtype=torch.float32, device=device)
        y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32, device=device)
        
        # Prepare Tensors for Validation Set
        X_cat_list_val = [X_val[col].map(self.category_mappings[col]).fillna(0).astype(int).values.reshape(-1, 1) for col in self.categorical_features]
        X_cat_np_val = np.hstack(X_cat_list_val)
        X_cat_val = torch.tensor(X_cat_np_val, dtype=torch.long, device=device)
        X_num_val = torch.tensor(X_val[self.numerical_features].astype(float).values, dtype=torch.float32, device=device)
        y_val_tensor = torch.tensor(y_val.values.reshape(-1, 1), dtype=torch.float32, device=device)

        # Create DataLoaders
        train_dataset = TensorDataset(X_cat_train, X_num_train, y_train_tensor)
        val_dataset = TensorDataset(X_cat_val, X_num_val, y_val_tensor)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.batch_size * 2)

        self.model = NNWithEmbeddings(
            embedding_specs=self.embedding_specs,
            num_numerical_features=len(self.numerical_features),
            layer_sizes=self.hidden_sizes + [self.output_size]
        ).to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        print("\nStarting model training with early stopping...")
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(self.num_epochs):
            self.model.train()
            total_train_loss = 0
            for batch_cat, batch_num, batch_y in train_loader:
                outputs = self.model(batch_cat, batch_num)
                loss = criterion(outputs, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(train_loader)

            # Validation loop
            self.model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch_cat, batch_num, batch_y in val_loader:
                    outputs = self.model(batch_cat, batch_num)
                    loss = criterion(outputs, batch_y)
                    total_val_loss += loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

            # Early stopping logic
            if avg_val_loss < best_val_loss: # If the last loss is better than the best one so far, reset
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs.")
                    break
        
        # Load the best model state
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        print("Training finished.")

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("You must call fit() before calling predict().")
        
        X_processed = X.copy()
        
        X_cat_list = []
        for col in self.categorical_features:
            codes = X_processed[col].map(self.category_mappings[col]).fillna(0).astype(int)
            X_cat_list.append(codes.values.reshape(-1, 1))
            
        X_cat_np = np.hstack(X_cat_list)
        X_cat = torch.tensor(X_cat_np, dtype=torch.long, device=device)

        X_num = torch.tensor(X_processed[self.numerical_features].values, dtype=torch.float32, device=device)
        
        test_dataset = TensorDataset(X_cat, X_num)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        all_predictions = []
        with torch.no_grad():
            for batch_cat, batch_num in test_loader:
                outputs = self.model(batch_cat, batch_num)
                all_predictions.extend(outputs.cpu().numpy())
        
        return np.array(all_predictions).flatten()

    def set_params(self, **params):
        """
        Sets the parameters of this estimator. This method is compatible with
        scikit-learn's GridSearchCV.
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                warnings.warn(f"Invalid parameter {key} for estimator {self.__class__.__name__}.")
        return self
