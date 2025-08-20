import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np

import warnings


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)




# ==============================================================================
# 1. The Neural Network Class
# ==============================================================================
# This class defines our feedforward neural network.
# It's designed to be highly configurable. You can define the entire
# structure (input, hidden, and output layers) by passing a list of numbers.
# ==============================================================================
class FeedForwardNN(nn.Module):
    """
    A configurable feedforward neural network.

    Args:
        layer_sizes (list of int): A list where the first element is the input
            feature size, the last is the output size, and the numbers in
            between are the sizes of the hidden layers.
            Example: [10, 128, 64, 1] means 10 input features, two hidden
                    layers with 128 and 64 neurons, and 1 output value.
        activation (nn.Module): The activation function to use between hidden
            layers. Defaults to ReLU.
    """
    def __init__(self, layer_sizes, activation=nn.ReLU()):
        super(FeedForwardNN, self).__init__()

        # We use nn.Sequential to stack the layers.
        layers = []
        for i in range(len(layer_sizes) - 1):
            # Add a linear layer
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            
            # Add an activation function, but not after the final output layer
            if i < len(layer_sizes) - 2:
                layers.append(activation)
        
        # The '*' unpacks the list of layers into arguments for nn.Sequential
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """The forward pass of the network."""
        return self.model(x)

# ==============================================================================
# 2. Training and Prediction Functions
# ==============================================================================
# These helper functions make it easy to train the model and make predictions.
# ==============================================================================

def train_model(model, data_loader, criterion, optimizer, num_epochs=25, device=device):
    """
    Trains the neural network.

    Args:
        model (nn.Module): The neural network model to train.
        data_loader (DataLoader): DataLoader providing training data.
        criterion (nn.Module): The loss function (e.g., nn.MSELoss).
        optimizer (optim.Optimizer): The optimization algorithm (e.g., Adam).
        num_epochs (int): The number of times to iterate over the dataset.
    """
    print("Starting training...")
    # Set the model to training mode
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in data_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs.to(device))
            
            # Squeeze output and labels to match dimensions for loss calculation
            loss = criterion(outputs.squeeze(), labels.to(device))
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Print loss for the epoch
        avg_loss = running_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss (RMSE): {np.sqrt(avg_loss):.4f}")
        
    print("Finished training.")

def model_predict(model, data_loader, device=device): # Ex: predict
    """
    Makes predictions on new data.

    Args:
        model (nn.Module): The trained model.
        data_loader (DataLoader): DataLoader providing data for prediction.

    Returns:
        tuple: A tuple containing lists of predictions and actual labels.
    """
    # print("\nMaking predictions...")
    # Set the model to evaluation mode
    model.eval()
    
    all_predictions = []
    # all_labels = []
    
    # No need to track gradients for prediction
    with torch.no_grad():
        for inputs in data_loader: # , labels
            # print(inputs[0])
            outputs = model(inputs[0].to(device))
            
            # For regression, the output of the model is the prediction.
            # No sigmoid or rounding is needed.
            predicted = outputs.squeeze()
            
            all_predictions.extend(predicted.cpu().numpy())
            # all_labels.extend(labels.cpu().numpy())
            
    return all_predictions #, all_labels



class FeedForwardNNRegressor:


    # ==============================================================================
    # 3. Example Usage
    # ==============================================================================
    # Here's how to put it all together. We will:
    # - Generate some synthetic data for a regression task.
    # - Define the network architecture.
    # - Train the model.
    # - Evaluate its performance.
    # ==============================================================================

    def __init__(self, input_features, output_size, batch_size=16, learning_rate=0.001, num_epochs=10, hidden_sizes=[1024]):
        self.input_features = input_features
        self.output_size = output_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.layer_config = [input_features] + hidden_sizes + [output_size]
        self.model = None

        print(f"Network Configuration: {self.layer_config}")
        


    def fit(self, X, y):
        print("torch device: ", torch.cuda.current_device())
        # Convert numpy arrays to PyTorch tensors
        X_train_tensor = torch.tensor(X.to_numpy(dtype=np.float32), dtype=torch.float32, device=device)
        y_train_tensor = torch.tensor(y.to_numpy(dtype=np.float32), dtype=torch.float32, device=device)
        # Create TensorDatasets and DataLoaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        # --- Model Initialization ---
        # Instantiate the model with our defined configuration
        self.model = FeedForwardNN(layer_sizes=self.layer_config).to(device)
        # Define the loss function and optimizer
        # For regression, Mean Squared Error (MSE) is a common loss function.
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # --- Train ---
        # Train the model
        train_model(self.model, train_loader, criterion, optimizer, num_epochs=self.num_epochs, device=device)

    def predict(self, X):
        X_test_tensor = torch.tensor(X.to_numpy(dtype=np.float32), dtype=torch.float32, device=device)
        test_dataset = TensorDataset(X_test_tensor)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)
        # Make predictions on the test set
        predictions = model_predict(self.model, test_loader, device=device) # , actuals

        # # --- Evaluate Performance ---
        # # For regression, we can use metrics like Mean Squared Error (MSE).
        # mse = mean_squared_error(actuals, predictions)
        # train_mse = mean_squared_error(train_actuals, train_predictions)
        
        return np.array(predictions)



# ===== Adding embeddings =====

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from collections import OrderedDict

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
class FeedForwardNNRegressorWithEmbeddings:

    def __init__(self, categorical_features, output_size, batch_size=16, learning_rate=0.001, num_epochs=10, hidden_sizes=[1024]):
        self.categorical_features = categorical_features
        self.numerical_features = [] # Will be determined in fit()
        self.output_size = output_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.hidden_sizes = hidden_sizes
        self.model = None
        self.category_mappings = {} # To store mappings for prediction
        self.embedding_specs = [] # To store embedding configuration

    # In the FeedForwardNNRegressorWithEmbeddings class...
    def fit(self, X, y):
        # --- Data Preprocessing ---
        self.numerical_features = [col for col in X.columns if col not in self.categorical_features]
        print(f"Categorical features: {self.categorical_features}")
        print(f"Numerical features: {self.numerical_features}")

        # Create mappings, embedding specs, and the categorical tensor in a single, safe loop.
        X_cat_list = []
        self.embedding_specs = []
        self.category_mappings = {}

        for col in self.categorical_features:
            # 1. Get unique categories from the column
            categories = X[col].unique()
            
            # 2. Create the mapping from category to integer index
            self.category_mappings[col] = {cat: i for i, cat in enumerate(categories)}
            num_categories = len(categories)
            
            # 3. Use the MAPPING to generate the codes.
            codes = X[col].map(self.category_mappings[col])
            X_cat_list.append(codes.values.reshape(-1, 1))

            # # 4. (Optional but recommended) Debugging check - WITH .values FIX
            # print(f"Feature '{col}': Found {num_categories} unique categories. Codes range from {codes.values.min()} to {codes.values.max()}.")
            # if codes.isnull().any():
            #     raise ValueError(f"Feature '{col}' contains values not found in the training data map, resulting in NaNs after mapping.")
            # if codes.max() >= num_categories:
            #     raise ValueError(f"Code {codes.max()} generated for feature '{col}' is out of bounds for {num_categories} categories.")

            # 5. Define the embedding layer size
            # MINE: If there are less than or equal to 10 categories, keep the number.
            if num_categories > 10:
                embedding_dim = min(50, (num_categories + 1) // 2)
            else:
                embedding_dim = num_categories
            self.embedding_specs.append((num_categories, embedding_dim))
            print(f"   -> Embedding defined with size: ({num_categories}, {embedding_dim})")

        # Combine all categorical code columns into a single numpy array
        X_cat_np = np.hstack(X_cat_list)
        
        # Prepare Tensors
        X_cat = torch.tensor(X_cat_np, dtype=torch.long, device=device)
        X_num = torch.tensor(X[self.numerical_features].astype(float).values, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(y.values.reshape(-1, 1), dtype=torch.float32, device=device)

        # --- The rest of the method (DataLoader, Model Init, Training) is unchanged ---
        train_dataset = TensorDataset(X_cat, X_num, y_tensor)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)

        self.model = NNWithEmbeddings(
            embedding_specs=self.embedding_specs,
            num_numerical_features=len(self.numerical_features),
            layer_sizes=self.hidden_sizes + [self.output_size]
        ).to(device)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        print("\nStarting model training...")
        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch_cat, batch_num, batch_y in train_loader:
                outputs = self.model(batch_cat, batch_num)
                loss = criterion(outputs, batch_y)
                
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_loss:.4f}')
        print("Training finished.")

    # In the FeedForwardNNRegressorWithEmbeddings class...

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("You must call fit() before calling predict().")
        
        # Create a copy to avoid modifying the original DataFrame
        X_processed = X.copy()
        
        # ===== START: MODIFIED & MORE ROBUST SECTION =====
        
        # Process categorical features
        X_cat_list = []
        for col in self.categorical_features:
            # Map categories to the integer codes learned during training.
            # .map() will produce NaN for any category not in the mapping.
            # .fillna(0) replaces these NaNs with the default index 0.
            # .astype(int) ensures the final codes are integers.
            codes = X_processed[col].map(self.category_mappings[col]).fillna(0).astype(int)
            X_cat_list.append(codes.values.reshape(-1, 1))
            
        X_cat_np = np.hstack(X_cat_list)
        X_cat = torch.tensor(X_cat_np, dtype=torch.long, device=device)

        # ===== END: MODIFIED SECTION =====

        # Process numerical features (assuming they are already scaled if scaler was fit)
        X_num = torch.tensor(X_processed[self.numerical_features].values, dtype=torch.float32, device=device)
        
        # Create DataLoader for prediction
        test_dataset = TensorDataset(X_cat, X_num)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)

        # --- Predict ---
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
            # Use hasattr to check if the attribute exists before setting it
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                # This helps catch typos or invalid parameter names
                warnings.warn(f"Invalid parameter {key} for estimator {self.__class__.__name__}.")
        return self
