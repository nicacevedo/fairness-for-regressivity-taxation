import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np


import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from collections.abc import Iterable


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
# print("device: ", device)



# Custom functions
def quantiles_price_tensor(y):
    m = y.size(dim=0)
    y_tilde = torch.zeros(m)
    for i,x in enumerate(y):
        y_tilde[i] = torch.sum(x >= y) # N_x
    return y_tilde / m


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



# Custom loss
class LInfLoss(nn.Module):
    def __init__(self):
        super(LInfLoss, self).__init__()

    def forward(self, prediction, target):
        return torch.max(torch.abs(prediction - target))


class WeightedMSELoss(nn.Module):
    def __init__(self, D_matrix):
        """
        Initializes the Weighted MSE Loss module.

        Args:
            D_matrix (torch.Tensor or np.ndarray): The weighting matrix D.
                                                     Will be converted to a torch.Tensor and registered as a buffer.
                                                     Shape: (num_outputs, num_outputs).
        """
        super(WeightedMSELoss, self).__init__()

        if not isinstance(D_matrix, torch.Tensor):
            D_matrix = torch.tensor(D_matrix, dtype=torch.float32)

        if D_matrix.dim() != 2 or D_matrix.shape[0] != D_matrix.shape[1]:
            raise ValueError("D_matrix must be a square 2D tensor.")

        # Register D_matrix as a buffer. Buffers are part of the module's state_dict
        # but are not considered trainable parameters. They are moved to device with the module.
        self.register_buffer('D_matrix', D_matrix)
        self.num_outputs = D_matrix.shape[0]

    def forward(self, predictions, targets):
        predictions = torch.reshape(predictions, targets.shape)
        # targets = torch.reshape(targets, predictions.shape)
        """
        Computes the weighted MSE loss (r' D r).

        Args:
            predictions (torch.Tensor): Your model's output. Shape (batch_size, num_outputs).
            targets (torch.Tensor): The true values. Shape (batch_size, num_outputs).

        Returns:
            torch.Tensor: The scalar weighted MSE loss.
        """
        # print(predictions.shape)
        # print(predictions)
        if predictions.shape != targets.shape:
            raise ValueError(f"Predictions shape {predictions.shape} and targets shape {targets.shape} must match.")
        if predictions.shape[-1] != self.num_outputs:
            raise ValueError(f"Last dimension of predictions {predictions.shape[-1]} must match D_matrix dimension {self.num_outputs}.")

        residuals = predictions - targets  # r = predictions - targets
        # Shape of residuals: (batch_size, num_outputs)

        # Expand residuals for batch matrix multiplication:
        # (batch_size, num_outputs) -> (batch_size, 1, num_outputs)
        residuals_expanded = residuals.unsqueeze(1)
        # (batch_size, num_outputs) -> (batch_size, num_outputs, 1)
        residuals_T_expanded = residuals.unsqueeze(2)

        # Compute: residuals_expanded @ D_matrix @ residuals_T_expanded
        # (B, 1, N) @ (N, N) -> (B, 1, N)
        # (B, 1, N) @ (B, N, 1) -> (B, 1, 1)
        loss_per_sample = torch.matmul(torch.matmul(residuals_expanded, self.D_matrix), residuals_T_expanded)

        # Squeeze to get (batch_size,) and then take the mean over the batch
        loss = torch.mean(loss_per_sample.squeeze())

        return loss



class QuadraticLoss(nn.Module):
    def __init__(self, D: torch.Tensor):
        """
        Custom loss: (Ax - b)^T D (Ax - b)
        Args:
            D (Tensor): Weight matrix D of shape (n, n)
        """
        super().__init__()
        self.register_buffer('D', D)  # So it's moved with the model (to GPU, etc.)

    def forward(self, Ax: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Ax (Tensor): Output of the model or linear transformation A x, shape (batch_size, n)
            b (Tensor): Target tensor, shape (batch_size, n)
        Returns:
            loss (Tensor): Scalar tensor
        """
        r = Ax - b  # Residual, shape (batch_size, n)
        loss = torch.einsum('bi,ij,bj->b', r, self.D, r)  # Batch-wise rᵀDr
        return loss.mean()  # Mean over the batch


# 1. Define the CVXPY Problem for the Constrained Layer (MODIFIED: Output Constraint)
def create_constrained_output_layer_on_predictions(output_dim, constraint_constant, constraint_value_upper_bound, groups_constraint_upper_bound, number_of_groups, batch_size=16):
    """
    Creates a CVXPY problem for a final layer that constrains its *output predictions*.

    Constraint: y_constrained / constraint_constant <= constraint_value_upper_bound
    """
    # Parameter: The unconstrained predictions from the upstream network
    # This will be (batch_size, output_dim)
    # batch_size = constraint_constant.shape[0]
    y_unconstrained = cp.Parameter((batch_size, output_dim))

    # Variable: The constrained predictions that satisfy the constraint
    # This will also be (batch_size, output_dim)
    y_constrained = cp.Variable((batch_size, output_dim))

    # Objective: Minimize the difference between unconstrained and constrained predictions
    objective = cp.Minimize(cp.sum_squares(y_constrained - y_unconstrained)) # squared l-2 norm for the projection

    # The constraint: y_constrained / constant <= value
    # This applies element-wise for each prediction in the batch and for each output.
    constraints = []
    # constraints = [y_constrained[i] / constraint_constant[i] - 1 <= constraint_value_upper_bound for i in range(constraint_constant.shape[0])]
    # constraints += [y_constrained[i] / constraint_constant[i] - 1 >= -constraint_value_upper_bound for i in range(constraint_constant.shape[0])]

    print("Generating group constraints...")
    # Groups constraint
    y_quants = quantiles_price_tensor(constraint_constant)
    groups_intervals = np.linspace(0, 1, number_of_groups+1)
    groups_dict = dict()
    # print("y_quants:", y_quants)
    # print("groups_intervals:", groups_intervals)
    for j in range(number_of_groups):
        groups_dict[j] = torch.where((groups_intervals[j] <= y_quants) & (y_quants <= groups_intervals[j+1]) )[0]
    # print("groups_dict", groups_dict)


    # X,y = X.to_numpy(), y.to_numpy()
    # n,m = X.shape
    # beta = cp.Variable(m)
    # error = cp.Variable()
    # mean_ratio_by_group = cp.Variable(number_of_groups)


    # # constraints = [ cp.SOC(error, X @ beta - y) ] # error to minimize
    # for j in range(number_of_groups):
    #     constraints += [ mean_ratio_by_group[j] ==  (1/len(groups_dict[j])) * cp.sum( [y_constrained[i] / constraint_constant[i] for i in groups_dict[j]] )]
    # #     constraints+=[
    # #         (1/len(groups_dict[j])) * cp.sum( [X[i,:] @ beta / y[i] for i in  groups_dict[j]]) - 1<= self.delta
    # #     ]



    # Average deviation from 1
    constraints+=[
            (1/len(groups_dict[j])) * (cp.sum( [y_constrained[i] / constraint_constant[i] -1 for i in  groups_dict[j]]  ))<= groups_constraint_upper_bound for j in range(number_of_groups)
        ]
    constraints+=[
            (1/len(groups_dict[j])) * (cp.sum( [y_constrained[i] / constraint_constant[i] -1 for i in  groups_dict[j]]  ))>= -groups_constraint_upper_bound for j in range(number_of_groups)
        ]




    # # Varirance of r_i per group bound
    # for j in range(number_of_groups):
    #     print("-"*100)
    #     # print(y_constrained[groups_dict[j]])
    #     # print(constraint_constant[groups_dict[j]])
    #     # print(mean_ratio_by_group[j])
    #     # print(y_constrained[groups_dict[j]] / constraint_constant[groups_dict[j]] - mean_ratio_by_group[j])
    #     print(np.sqrt(1/len(groups_dict[j])) * ( y_constrained[groups_dict[j]] / constraint_constant[groups_dict[j]] - mean_ratio_by_group[j] ) )
    #     # constraints+=[
    #     #     cp.SOC(constraint_value_upper_bound, np.sqrt(1/len(groups_dict[j])) * ( y_constrained[groups_dict[j]] / constraint_constant[groups_dict[j]] - mean_ratio_by_group[j] ) )
    #     # ]
    # constraints+=[
    #     cp.SOC(constraint_value_upper_bound, np.sqrt(1/len(groups_dict[j])) * ( y_constrained[groups_dict[j]] / constraint_constant[groups_dict[j]] - mean_ratio_by_group[j] ) ) for j in groups_dict
    # ]


    problem = cp.Problem(objective, constraints)

    # Parameters are y_unconstrained, variables are y_constrained
    return CvxpyLayer(problem,
                      parameters=[y_unconstrained],
                      variables=[y_constrained])

# 2. Define the MLP with the CVXPYLAYER (MODIFIED to work with output constraints)
class ConstrainedMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size,  batch_size, dropout_rate, constraint_constant, constraint_value_upper_bound, groups_constraint_upper_bound, number_of_groups):
        super(ConstrainedMLP, self).__init__()

        if not isinstance(hidden_sizes, Iterable):
            raise TypeError("hidden_sizes must be a list or tuple of integers.")

        # Build the hidden layers dynamically
        layers = []
        current_input_size = input_size

        print("Looping hidden layers...")
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            current_input_size = hidden_size

        # Use nn.Sequential to chain the hidden layers
        # nn.Sequential is a special kind of Module that holds other Modules in a list.
        print("Creating sequence...")
        self.hidden_layers = nn.Sequential(*layers)

        # The input to the final linear layer is the size of the last hidden layer
        # If hidden_sizes is empty, the input is the original input_size.
        print("Linear...")
        last_hidden_size = hidden_sizes[-1] if hidden_sizes else input_size
        self.final_linear = nn.Linear(last_hidden_size, output_size)

        # self.fc1 = nn.Linear(input_size, hidden_size)
        # self.relu = nn.ReLU()

        # Add a Dropout layer after the first activation
        # self.dropout = nn.Dropout(p=dropout_rate) # p is the probability of an element being zeroed

        # # This is now just a regular final linear layer
        # # It produces the *unconstrained* predictions that will be fed to CvxpyLayer
        # self.final_linear = nn.Linear(hidden_size, output_size)

        # Create the CVXPYLayer for output constraint
        print("Creating constrained model...")
        self.constrained_layer = create_constrained_output_layer_on_predictions(
            output_dim=output_size,
            constraint_constant=constraint_constant,
            constraint_value_upper_bound=constraint_value_upper_bound,
            groups_constraint_upper_bound=groups_constraint_upper_bound,
            number_of_groups=number_of_groups,
            batch_size=batch_size
        )


    # def forward(self, x):
    #     x = self.fc1(x)
    #     x = self.relu(x)

    #     # Apply dropout after the activation function of the hidden layer
    #     x = self.dropout(x)

    #     # Get the unconstrained predictions from the final linear layer
    #     y_unconstrained = self.final_linear(x)

    #     # Pass unconstrained predictions through the CVXPY layer to get constrained ones
    #     y_constrained, = self.constrained_layer(y_unconstrained) # Note the comma for single return
    #     # y_constrained = y_unconstrained


    #     # Groups mean deviaion constraint


    #     return y_constrained # Return the constrained predictions


    def forward(self, x):
        # Pass the input through all dynamically created hidden layers
        x = self.hidden_layers(x)

        # Get the unconstrained predictions from the final linear layer
        y_unconstrained = self.final_linear(x)

        # Pass unconstrained predictions through the CVXPY layer to get constrained ones
        print(y_unconstrained.shape)
        y_constrained, = self.constrained_layer(y_unconstrained)

        return y_constrained



class ConstrainedMLPRegressor:


    # ==============================================================================
    # 3. Example Usage
    # ==============================================================================
    # Here's how to put it all together. We will:
    # - Generate some synthetic data for a regression task.
    # - Define the network architecture.
    # - Train the model.
    # - Evaluate its performance.
    # ==============================================================================

    def __init__(self, input_features, output_size, batch_size=16, learning_rate=0.001, num_epochs=10, hidden_sizes=[1024], dropout_rate=0.2):
        self.input_size = input_features
        self.output_size = output_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = num_epochs
        self.hidden_size = tuple(hidden_sizes)# + [output_size]
        self.dropout_rate = dropout_rate
        self.model = None

        print(f"Network Configuration: {[input_features] + hidden_sizes + [output_size]}")
        


    def fit(self, X, y):
        print("Fitting the model...")
        # print("torch device: ", torch.cuda.current_device())
        # Data to torch datatype
        X = torch.tensor(X, dtype=torch.float32, device=device) #torch.from_numpy(X_train.astype(np.float64).to_numpy())
        y = torch.tensor(y.to_numpy(), dtype=torch.float32, device=device) #torch.from_numpy(y_train_scaled.astype(np.float64).to_numpy())
       # constraint_constant = 2.0
        constraint_constant = y
        constraint_value_upper_bound = 100#5.0#0.5 # y_i / 2.0 <= 5.0  => y_i <= 10.0
        groups_constraint_upper_bound = 100 #0.05
        number_of_groups = 3
        # bound to normalization
        # constraint_value_upper_bound = (constraint_value_upper_bound - y_train.mean() ) / y_train.std()
        # Initialize model, loss, and optimizer
        print("Creating the model...")
        self.model = ConstrainedMLP(self.input_size, self.hidden_size, self.output_size, self.batch_size, self.dropout_rate, constraint_constant, constraint_value_upper_bound, groups_constraint_upper_bound, number_of_groups).to(device)
        print("Double...")
        self.model.double()
        print("Loss..")
        criterion = nn.MSELoss()#QuadraticLoss(D=D_full)
        print("Optimizer creation...")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # # Training loop
        # for epoch in range(self.epochs):
        #     self.model.train()
        #     optimizer.zero_grad()

        #     outputs = self.model(X_train_torch) # These are now the *constrained* outputs
        #     # print(outputs.shape)
        #     loss = criterion(outputs, y_train_torch)

        #     loss.backward()
        #     optimizer.step()

        #     if (epoch + 1) % 10 == 0:
        #         print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}")
        #         with torch.no_grad():
        #             # We can check the constraint directly on the 'outputs' tensor
        #             # which already represents the constrained values

        #             # Check constraint 1
        #             ratios = outputs / constraint_constant - 1
        #             print("max violation", torch.max(torch.abs(ratios)))
        #             print(outputs[:4])
        #             print(constraint_constant[:4])


        #             # print(f"  First 5 constrained outputs (should be: {constraint_constant[random_observation] * ( -constraint_value_upper_bound +1)} <= * <= {constraint_constant[random_observation] * ( constraint_value_upper_bound +1)}):")
        #             # print(outputs[:5].squeeze().numpy())

        # print("\nTraining complete!")

        # # Convert numpy arrays to PyTorch tensors
        # X_train_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        # y_train_tensor = torch.tensor(y.to_numpy(), dtype=torch.float32, device=device)
        # # Create TensorDatasets and DataLoaders
        print("Creating dataset...")
        train_dataset = TensorDataset(X, y)
        print("Creating data loader...")
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        # # --- Model Initialization ---
        # # Instantiate the model with our defined configuration
        # self.model = ConstrainedMLP(layer_sizes=self.layer_config).to(device)
        # # Define the loss function and optimizer
        # # For regression, Mean Squared Error (MSE) is a common loss function.
        # criterion = nn.MSELoss()
        # optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # # --- Train ---
        # # Train the model
        print("Training the model...")
        train_model(self.model, train_loader, criterion, optimizer, num_epochs=self.num_epochs, device=device)
        

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32, device=device)
        test_dataset = TensorDataset(X)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)
        # Make predictions on the test set
        predictions = model_predict(self.model, test_loader, device=device) # , actuals

        # # --- Evaluate Performance ---
        # # For regression, we can use metrics like Mean Squared Error (MSE).
        # mse = mean_squared_error(actuals, predictions)
        # train_mse = mean_squared_error(train_actuals, train_predictions)
        
        return np.array(predictions)


# 3. Training and Prediction Example (unchanged in logic, only model init params)
# if __name__ == "__main__":
    # n,m = X_train.shape
    # input_size = m
    # hidden_size = (50,)
    # output_size = 1 # Keep output_size = 1 for simpler constraint
    # learning_rate = 0.01
    # epochs = 100
    # dropout_rate = 0.5

    # # device: cpu or gpu
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # # Data to torch datatype
    # X_train_torch = torch.from_numpy(X_train.astype(np.float64).to_numpy())
    # y_train_torch = torch.from_numpy(y_train_scaled.astype(np.float64).to_numpy())

    # # print(X_train_torch.shape)
    # # print(y_train_torch.shape)


    # # constraint_constant = 2.0
    # constraint_constant = y_train_torch
    # constraint_value_upper_bound = 1e3#5.0#0.5 # y_i / 2.0 <= 5.0  => y_i <= 10.0
    # groups_constraint_upper_bound = 1e3 #0.05
    # number_of_groups = 3

    # # bound to normalization
    # # constraint_value_upper_bound = (constraint_value_upper_bound - y_train.mean() ) / y_train.std()


    # # X_test = torch.randn(20, input_size).float()
    # # y_test = torch.randn(20, output_size).float() * 10

    # # Initialize model, loss, and optimizer
    # model = ConstrainedMLP(input_size, hidden_size, output_size, dropout_rate, constraint_constant, constraint_value_upper_bound, groups_constraint_upper_bound, number_of_groups)
    # model.double()

    # # Loss function
    # # y_train_bins = get_uniform_bins(vector_range=(1749900, 12250100), n_bins=1000) # vector_range=(14000,501000), n_bins=500)
    # # print(y_train_bins)
    # # y_train_weights = get_group_weights(y_train, y_train_bins, alpha=10)
    # # print(y_train.shape)
    # # print(len(y_train_weights))
    # # print(constraint_constant.shape)
    # # # print(np.diag(y_train_weights).shape)
    # # D_full = torch.from_numpy(np.diag(y_train_weights))
    # # print(D_full.shape)
    # # # print(D_full)
    # # criterion = WeightedMSELoss(D_matrix=D_full) #LInfLoss()#nn.L1Loss()#nn.MSELoss()#LInfLoss()#nn.L1Loss()#nn.MSELoss()#LInfLoss()##nn.L1Loss()#
    # criterion = nn.MSELoss()#QuadraticLoss(D=D_full)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # # random_observation = np.random.choice(range(constraint_constant.shape[0]))
    # # print("Random observation: ", random_observation)
    # # print(f"Constraint: | Final output / {constraint_constant[random_observation] - 1} | <= {constraint_value_upper_bound}") # (i.e., final output <= {constraint_constant[random_observation] * constraint_value_upper_bound})")
    # # print("-" * 30)

    # # Training loop
    # for epoch in range(epochs):
    #     model.train()
    #     optimizer.zero_grad()

    #     outputs = model(X_train_torch) # These are now the *constrained* outputs
    #     # print(outputs.shape)
    #     loss = criterion(outputs, y_train_torch)

    #     loss.backward()
    #     optimizer.step()

    #     if (epoch + 1) % 10 == 0:
    #         print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    #         with torch.no_grad():
    #             # We can check the constraint directly on the 'outputs' tensor
    #             # which already represents the constrained values

    #             # Check constraint 1
    #             ratios = outputs / constraint_constant - 1
    #             print("max violation", torch.max(torch.abs(ratios)))
    #             print(outputs[:4])
    #             print(constraint_constant[:4])


    #             # print(f"  First 5 constrained outputs (should be: {constraint_constant[random_observation] * ( -constraint_value_upper_bound +1)} <= * <= {constraint_constant[random_observation] * ( constraint_value_upper_bound +1)}):")
    #             # print(outputs[:5].squeeze().numpy())

    # print("\nTraining complete!")

    # # --- PREDICTION PHASE ---
    # print("\n--- Making Predictions on Test Data ---")

    # model.eval()
    # with torch.no_grad():
    #     test_predictions = model(X_test_torch) # These are also the *constrained* outputs

    #     print(f"Test data shape: {X_test_torch.shape}")
    #     print(f"Predicted values shape: {test_predictions.shape}")
    #     print("First 5 test input samples:")
    #     print(X_test_torch[:5])
    #     print("\nFirst 5 predicted values (constrained):")
    #     print(test_predictions[:5])

    #     test_loss = criterion(test_predictions, y_test)
    #     print(f"\nTest Loss: {test_loss.item():.4f}")

    #     # Final check of the constraint on test predictions
    #     print(f"\nConstraint check on first 5 TEST predictions (should be <= {constraint_constant[random_observation] * constraint_value_upper_bound}):")
    #     print(test_predictions[:5].squeeze().numpy())