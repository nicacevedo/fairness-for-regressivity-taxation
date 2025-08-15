import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict
import warnings

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================================================================
# 1. Base Neural Network Model with Embeddings (Unchanged)
# ==============================================================================
def init_weights(m):
    """Initializes weights with Kaiming He for ReLUs for better stability."""
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class NNWithEmbeddings(nn.Module):
    """The base neural network that produces raw, unconstrained predictions."""
    def __init__(self, embedding_specs, num_numerical_features, layer_sizes):
        super().__init__()
        self.embedding_layers = nn.ModuleList([nn.Embedding(num, dim) for num, dim in embedding_specs])
        total_embedding_dim = sum(dim for _, dim in embedding_specs)
        input_size = total_embedding_dim + num_numerical_features
        
        all_layers = []
        for i, size in enumerate(layer_sizes):
            activation = nn.ReLU() if i < len(layer_sizes) - 1 else None
            all_layers.append((f"linear_{i}", nn.Linear(input_size, size)))
            if activation:
                all_layers.append((f"relu_{i}", activation))
            input_size = size
            
        self.layers = nn.Sequential(OrderedDict(all_layers))
        self.apply(init_weights)

    def forward(self, x_cat, x_num):
        embeddings = [emb_layer(x_cat[:, i]) for i, emb_layer in enumerate(self.embedding_layers)]
        x_cat_emb = torch.cat(embeddings, dim=1)
        x = torch.cat([x_cat_emb, x_num], dim=1)
        return self.layers(x)

# ==============================================================================
# 2. The Simple Projection Layer
# ==============================================================================
class RelativeDeviationClamp(nn.Module):
    """
    A simple, fast, and differentiable layer that enforces the relative
    deviation constraint using torch.clamp.
    """
    def __init__(self, dev_thresh=0.15):
        super().__init__()
        self.dev_thresh = dev_thresh

    def forward(self, y_raw, y_real):
        """
        Args:
            y_raw (torch.Tensor): The raw, unconstrained output from the base NN.
            y_real (torch.Tensor): The ground truth values for the batch.
        """
        # Calculate the lower and upper bounds for each prediction
        lower_bound = (1 - self.dev_thresh) * y_real
        upper_bound = (1 + self.dev_thresh) * y_real
        
        # Clamp the raw prediction to be within the valid range
        y_pred = torch.clamp(y_raw, min=lower_bound, max=upper_bound)
        
        return y_pred

# ==============================================================================
# 3. The Combined Meta-Model
# ==============================================================================
class ProjectedNN(nn.Module):
    """A meta-model that wraps the base NN and the simple projection layer."""
    def __init__(self, base_model, projection_layer):
        super().__init__()
        self.base_model = base_model
        self.projection_layer = projection_layer
        
    def forward(self, x_cat, x_num, y_real):
        # 1. Get the raw prediction from the base model
        y_raw = self.base_model(x_cat, x_num)
        
        # 2. Pass the raw prediction and ground truth to the projection layer
        y_pred = self.projection_layer(y_raw, y_real)
        
        return y_pred

# ==============================================================================
# 4. The Main User-Facing Wrapper Class
# ==============================================================================
class FeedForwardNNRegressorWithProjection:

    def __init__(self, categorical_features, output_size=1, batch_size=32, learning_rate=0.001, num_epochs=10, hidden_sizes=[50, 25], dev_thresh=0.15):
        self.categorical_features = categorical_features
        self.numerical_features = []
        self.output_size = output_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.hidden_sizes = hidden_sizes
        self.dev_thresh = dev_thresh
        
        self.model = None
        self.base_model = None
        self.scaler = StandardScaler()
        self.category_mappings = {}

    def fit(self, X, y):
        self.numerical_features = [col for col in X.columns if col not in self.categorical_features]
        X_fit = X.copy()
        y_fit = y.copy()

        if self.numerical_features:
            X_fit[self.numerical_features] = self.scaler.fit_transform(X_fit[self.numerical_features])

        for col in self.categorical_features:
            self.category_mappings[col] = {cat: i for i, cat in enumerate(X_fit[col].unique())}
        
        X_cat_tensors = [torch.tensor(X_fit[col].map(self.category_mappings[col]).values, dtype=torch.long) for col in self.categorical_features]
        X_cat = torch.stack(X_cat_tensors, dim=1)
        X_num = torch.tensor(X_fit[self.numerical_features].values, dtype=torch.float64)
        y_tensor = torch.tensor(y_fit.values, dtype=torch.float64).unsqueeze(1)
        
        dataset = TensorDataset(X_cat, X_num, y_tensor)
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        embedding_specs = [(len(self.category_mappings[col]), min(50, (len(self.category_mappings[col])+1)//2)) for col in self.categorical_features]
        base_nn = NNWithEmbeddings(embedding_specs, len(self.numerical_features), self.hidden_sizes + [self.output_size])
        
        projection_layer = RelativeDeviationClamp(self.dev_thresh)
        self.model = ProjectedNN(base_nn, projection_layer).to(device).double()
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        print("Starting training with simple projection layer...")
        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch_cat, batch_num, batch_y in loader:
                batch_cat, batch_num, batch_y = batch_cat.to(device), batch_num.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                
                # Pass y_real (batch_y) to the model for the projection
                y_pred = self.model(batch_cat.long(), batch_num.double(), batch_y)
                
                loss = criterion(y_pred, batch_y)
                if not torch.isnan(loss):
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

            avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss:.4f}')
        
        self.base_model = self.model.base_model
        print("Training finished.")

    def predict_constrained(self, X, y_real):
        """
        Predicts the final, projected output.
        Requires y_real to calculate the projection bounds.
        """
        if self.model is None: raise RuntimeError("You must call fit() before predicting.")
        
        X_pred = X.copy()
        if self.numerical_features:
            X_pred[self.numerical_features] = self.scaler.transform(X_pred[self.numerical_features])

        for col in self.categorical_features:
            X_pred[col] = X_pred[col].map(self.category_mappings[col]).fillna(0)
        
        X_cat_tensors = [torch.tensor(X_pred[col].values, dtype=torch.long) for col in self.categorical_features]
        X_cat = torch.stack(X_cat_tensors, dim=1)
        X_num = torch.tensor(X_pred[self.numerical_features].values, dtype=torch.float64)
        y_tensor = torch.tensor(y_real.values, dtype=torch.float64).unsqueeze(1)
        
        dataset = TensorDataset(X_cat, X_num, y_tensor)
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        all_predictions = []
        with torch.no_grad():
            for batch_cat, batch_num, batch_y in loader:
                batch_cat, batch_num, batch_y = batch_cat.to(device), batch_num.to(device), batch_y.to(device)
                
                y_pred = self.model(batch_cat.long(), batch_num.double(), batch_y)
                all_predictions.extend(y_pred.cpu().numpy())

        return np.array(all_predictions).flatten()

    def predict(self, X):
        """
        Predicts the raw, unconstrained output from the base neural network.
        """
        if self.base_model is None: raise RuntimeError("Fit the model first.")
        X_pred = X.copy()
        if self.numerical_features:
            X_pred[self.numerical_features] = self.scaler.transform(X_pred[self.numerical_features])
        for col in self.categorical_features:
            X_pred[col] = X_pred[col].map(self.category_mappings[col]).fillna(0)
        X_cat_tensors = [torch.tensor(X_pred[col].values, dtype=torch.long) for col in self.categorical_features]
        X_cat = torch.stack(X_cat_tensors, dim=1).to(device)
        X_num = torch.tensor(X_pred[self.numerical_features].values, dtype=torch.float64).to(device)
        
        self.base_model.eval()
        all_raw_predictions = []
        with torch.no_grad():
            dataset = TensorDataset(X_cat, X_num)
            loader = DataLoader(dataset, batch_size=self.batch_size)
            for batch_cat, batch_num in loader:
                all_raw_predictions.extend(self.base_model(batch_cat.long(), batch_num.double()).cpu().numpy())
        return np.array(all_raw_predictions).flatten()
