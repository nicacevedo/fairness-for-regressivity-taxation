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
# 1. Core TabNet Components
# ==============================================================================

class Sparsemax(nn.Module):
    """A differentiable sparse version of softmax."""
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        input = input.transpose(0, self.dim)
        original_size = input.size()
        input = input.reshape(input.size(0), -1)

        input = input.transpose(0, 1)
        dim = 1

        sorted_input, _ = torch.sort(input, dim=dim, descending=True)
        cumulative_sum = torch.cumsum(sorted_input, dim=dim)
        k = torch.arange(1, input.size(dim) + 1, device=input.device).float()
        sum_k = 1 + k * sorted_input
        
        indices = torch.arange(0, input.size(dim), device=input.device)
        mask = (sum_k > cumulative_sum).long()
        
        k_max = torch.max(mask * k, dim=dim, keepdim=True)[0]
        
        # tau = (torch.gather(cumulative_sum, dim, k_max - 1) - 1) / k_max
        tau = (torch.gather(cumulative_sum, dim, (k_max - 1).long()) - 1) / k_max
        output = torch.max(torch.zeros_like(input), input - tau)

        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)
        return output

class AttentiveTransformer(nn.Module):
    """The attentive transformer block for feature selection."""
    def __init__(self, input_dim, output_dim, gamma=1.5):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.sparsemax = Sparsemax(dim=1)
        self.gamma = gamma

    def forward(self, x, prior_scales):
        x = self.bn(self.fc(x))
        x = x * prior_scales
        mask = self.sparsemax(x)
        return mask

class FeatureTransformer(nn.Module):
    """The feature transformer block for processing selected features."""
    def __init__(self, input_dim, output_dim, n_glu_layers=2):
        super().__init__()
        layers = []
        for i in range(n_glu_layers):
            layers.append(nn.Linear(input_dim, 2 * output_dim))
            layers.append(nn.GLU())
            input_dim = output_dim
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

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

# ==============================================================================
# 3. The TabNet Model
# ==============================================================================

class TabNet(nn.Module):
    def __init__(self, input_dim, output_dim, n_steps=3, feature_dim=8,
                 attention_dim=8, gamma=1.5, n_glu_layers=2):
        super().__init__()
        self.n_steps = n_steps
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.attention_dim = attention_dim
        self.gamma = gamma

        self.initial_bn = nn.BatchNorm1d(input_dim)
        self.feature_transformers = nn.ModuleList()
        self.attentive_transformers = nn.ModuleList()

        for _ in range(n_steps):
            self.feature_transformers.append(
                FeatureTransformer(input_dim, feature_dim + attention_dim, n_glu_layers)
            )
            self.attentive_transformers.append(
                AttentiveTransformer(attention_dim, input_dim, gamma)
            )
        
        self.final_fc = nn.Linear(feature_dim, output_dim)
        self.initial_attention_features = nn.Parameter(torch.randn(1, self.attention_dim), requires_grad=True)

    def forward(self, x):
        x_bn = self.initial_bn(x)
        prior_scales = torch.ones_like(x_bn)
        total_output = torch.zeros(x.size(0), self.output_dim).to(x.device)
        sparsity_loss = 0.0
        
        attention_features = self.initial_attention_features.expand(x.size(0), -1)

        for step in range(self.n_steps):
            mask = self.attentive_transformers[step](attention_features, prior_scales)
            sparsity_loss -= torch.mean(torch.sum(mask * torch.log(mask + 1e-10), dim=1)) / self.n_steps
            
            prior_scales = prior_scales * (self.gamma - mask)
            
            masked_x = x_bn * mask
            features = self.feature_transformers[step](masked_x)
            
            decision_features = features[:, :self.feature_dim]
            attention_features = features[:, self.feature_dim:]

            step_output = self.final_fc(decision_features)
            total_output += nn.functional.relu(step_output)
            
        return total_output, sparsity_loss

# ==============================================================================
# 4. The Main User-Facing Wrapper Class
# ==============================================================================
"""
Args:
    categorical_features (list[str]): A list of column names in the input DataFrame
                                      that should be treated as categorical.
    coord_features (list[str]): A list of column names to be treated as coordinates.
                                (Note: The current implementation treats these as
                                standard numerical features).
    output_size (int, optional): The number of output neurons, typically 1 for regression.
                                 Defaults to 1.
    batch_size (int, optional): The number of samples per batch for training.
                                Defaults to 32.
    learning_rate (float, optional): The learning rate for the Adam optimizer.
                                     Defaults to 0.001.
    num_epochs (int, optional): The number of complete passes through the training dataset.
                                Defaults to 10.
    n_steps (int, optional): The number of sequential decision steps in the TabNet model.
                             Each step has its own feature selection mask. Defaults to 3.
    feature_dim (int, optional): The dimensionality of the feature representation that
                                 contributes to the final prediction at each step.
                                 Defaults to 8.
    attention_dim (int, optional): The dimensionality of the representation used to
                                   determine the feature selection mask for the next step.
                                   Defaults to 8.
    sparsity_lambda (float, optional): A regularization parameter that encourages the
                                       feature selection masks to be sparse, making the
                                       model more interpretable. Defaults to 1e-5.
    loss_fn (str, optional): The loss function to use. Options are 'mse', 'focal_mse',
                             or 'huber'. Defaults to 'mse'.
    random_state (int | None, optional): Seed for reproducibility. Defaults to None.
"""
class TabNetRegressor:

    def __init__(self, categorical_features, coord_features, output_size=1, batch_size=32,
                 learning_rate=0.001, num_epochs=10, n_steps=3, feature_dim=8,
                 attention_dim=8, sparsity_lambda=1e-5, loss_fn='mse', random_state=None):
        self.categorical_features = categorical_features
        self.coord_features = coord_features
        self.numerical_features = []
        self.output_size = output_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.n_steps = n_steps
        self.feature_dim = feature_dim
        self.attention_dim = attention_dim
        self.sparsity_lambda = sparsity_lambda
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

        # For TabNet, we'll treat categoricals and coordinates as numerical for the initial layer
        # Embeddings can be added, but this is a simpler starting point
        for col in self.categorical_features:
            X_fit[col] = X_fit[col].astype('category').cat.codes
            
        X_tensor = torch.tensor(X_fit.values, dtype=torch.float32)
        y_tensor = torch.tensor(y_fit.values, dtype=torch.float32).unsqueeze(1)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        input_dim = X_tensor.shape[1]
        self.model = TabNet(
            input_dim=input_dim,
            output_dim=self.output_size,
            n_steps=self.n_steps,
            feature_dim=self.feature_dim,
            attention_dim=self.attention_dim
        ).to(device)
        
        if self.loss_fn_name == 'focal_mse':
            criterion = FocalMSELoss()
        elif self.loss_fn_name == 'huber':
            criterion = nn.HuberLoss()
        else:
            criterion = nn.MSELoss()
            
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        print("Starting TabNet training...")
        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                y_pred, sparsity_loss = self.model(batch_X)
                
                prediction_loss = criterion(y_pred, batch_y)
                loss = prediction_loss + self.sparsity_lambda * sparsity_loss
                
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
        for col in self.categorical_features:
            X_pred[col] = X_pred[col].astype('category').cat.codes
        
        X_tensor = torch.tensor(X_pred.values, dtype=torch.float32).to(device)
        
        self.model.eval()
        with torch.no_grad():
            predictions, _ = self.model(X_tensor)
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
