import torch
import torch.nn as nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer


# Util functions
def quantiles_price_tensor(y):
    m = y.size(dim=0)
    y_tilde = torch.zeros(m)
    for i,x in enumerate(y):
        y_tilde[i] = torch.sum(x >= y) # N_x
    return y_tilde / m




#
#           ===== NN Models =====
# 

class ConstrainedLayer(nn.Module):
    def __init__(self, input_dim, batch_size):
        super().__init__()
        self.batch_size = batch_size
        

        # CVXPY variables and parameters
        x = cp.Variable((batch_size, 1))  # Output (y_constrained)
        y_param = cp.Parameter((batch_size, 1))  # True values (y_real)
        pred_param = cp.Parameter((batch_size, 1))  # Unconstrained predictions (y_unconstrained) 
        print("y_param: ", y_param.shape)
        print("pred_param: ", pred_param.shape)

        # Objective: minimize squared difference between x and prediction
        objective = cp.Minimize(cp.sum_squares(x - pred_param))
        print("y_param values: ", y_param)
        # constraints = []
        # constraints = [x <= y_param]  # Constrain outputs <= true values
        # constraints  = [x / y_param - 1 <= 10]
        # constraints += [x / y_param - 1 >= -10]
        constraints = [x[i,:] / y_param[i,:] - 1 <= 10 for i in range(y_param.shape[0])]
        constraints += [x[i,:] / y_param[i,:] - 1 >= -10 for i in range(y_param.shape[0])]

        problem = cp.Problem(objective, constraints)
        print("I am almost creating the layer...")
        self.cvx_layer = CvxpyLayer(problem, parameters=[pred_param, y_param], variables=[x])

    def forward(self, predictions, y_true_batch):
       # Reshape y_true_batch if needed
        if y_true_batch.dim() == 1:
            y_true_batch = y_true_batch.unsqueeze(1)
        if predictions.dim() == 1:
            predictions = predictions.unsqueeze(1)
        print("predictions: ", predictions.shape)
        print("y_true_batch: ", y_true_batch.shape)
        solution, = self.cvx_layer(predictions, y_true_batch)
        return solution  # Constrained outputs


class ConstrainedNN(nn.Module):
    def __init__(self, input_dim, hidden_layers, batch_size):
        super().__init__()
        self.batch_size = batch_size

        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, 1))
        self.feedforward = nn.Sequential(*layers)

        self.constrained_layer = ConstrainedLayer(input_dim=input_dim, batch_size=batch_size)

    def forward(self, X_batch, y_true_batch=None, apply_constraint=True):
        pred_unconstrained = self.feedforward(X_batch)

        if apply_constraint:
            if y_true_batch is None:
                raise ValueError("y_true_batch is required for constraint")
            pred_constrained = self.constrained_layer(pred_unconstrained, y_true_batch)
            return pred_constrained
        else:
            return pred_unconstrained


class ConstrainedNNRegressor: # The main model


    # ==============================================================================
    # 3. Example Usage
    # ==============================================================================
    # Here's how to put it all together. We will:
    # - Generate some synthetic data for a regression task.
    # - Define the network architecture.
    # - Train the model.
    # - Evaluate its performance.
    # ==============================================================================

    def __init__(self, input_dim, output_size, batch_size=16, learning_rate=0.001, num_epochs=10, hidden_layers=[1024]):
        self.input_dim = input_dim
        self.output_size = output_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.hidden_layers = hidden_layers
        self.layer_config = [input_dim] + hidden_layers + [output_size]
        self.model = None

        print(f"Network Configuration: {self.layer_config}")
        


    def fit(self, X, y):

        device = torch.device("cpu")#'cuda' if torch.cuda.is_available() else 'cpu')

        # Preliminaries on data
        X = torch.tensor(X, dtype=torch.float32, device=device)
        y = torch.tensor(y.to_numpy(), dtype=torch.float32, device=device)
        train_dataset = torch.utils.data.TensorDataset(X, y)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True) # drop_last if not same bath_size

        # Model with flexible hidden layers
        self.model = ConstrainedNN(input_dim=self.input_dim, hidden_layers=self.hidden_layers, batch_size=self.batch_size)
        self.model.to(device)

        # Loss & optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        print("\nBeggining training...")
        self.model.train()
        for epoch in range(self.num_epochs):
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                preds = self.model(batch_X, batch_y)  # y_true_batch passed to constrain output
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")


    def predict(self, X):

        device = torch.device("cpu")#'cuda' if torch.cuda.is_available() else 'cpu')

        # Preliminaries on data
        X = torch.tensor(X, dtype=torch.float32, device=device)
        test_dataset = torch.utils.data.TensorDataset(X)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        # Make predictions on the test set
        print("\nMaking predictions...")

        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch_X, _ in test_loader:
                batch_X = batch_X.to(device)
                preds = self.model(batch_X, apply_constraint=False)
                all_preds.append(preds.cpu())

        all_preds = torch.cat(all_preds).numpy()
        return all_preds


