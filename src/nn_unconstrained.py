import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np


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
    print("\nMaking predictions...")
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
        X_train_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        y_train_tensor = torch.tensor(y.to_numpy(), dtype=torch.float32, device=device)
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
        X_test_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        test_dataset = TensorDataset(X_test_tensor)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)
        # Make predictions on the test set
        predictions = model_predict(self.model, test_loader, device=device) # , actuals

        # # --- Evaluate Performance ---
        # # For regression, we can use metrics like Mean Squared Error (MSE).
        # mse = mean_squared_error(actuals, predictions)
        # train_mse = mean_squared_error(train_actuals, train_predictions)
        
        return np.array(predictions)
