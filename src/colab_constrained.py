import torch
import torch.nn as nn
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import numpy as np
from collections.abc import Iterable


# Custom functions
def quantiles_price_tensor(y):
    m = y.size(dim=0)
    y_tilde = torch.zeros(m)
    for i,x in enumerate(y):
        y_tilde[i] = torch.sum(x >= y) # N_x
    return y_tilde / m


# Get the matrix of weights
def get_uniform_bins(vector_range=(0,100), n_bins=10):
    # Distribution of each one
    y_min, y_max = vector_range
    # print("range: ", y_min, y_max)
    # B =  30#int((y_max - y_min)/330000)+1 # number of bins
    bins = np.linspace(y_min, y_max, n_bins)
    # print(f"{n_bins} bins: ")
    # print(bins)
    # lb = bins[0]
    return bins

def get_group_weights(vector, bins, alpha=1):
    """
        Let the weights first be the size of each group for simplicity
        Let intervals be: [lb, ub)
    """
    weights = []
    lb = bins[0] # first lower bound
    i = 0
    for ub in bins[1:]:
        # print(lb, ub)
        vector_group_ids = (lb <= vector) & (vector < ub)
        # print(vector_group_ids)
        group_size = np.sum(vector_group_ids)
        # print(group_size)
        # break
        i+=1
        # print("Group", i, "size: ", group_size)
        if group_size > 0:
            weights+= [1/group_size**alpha]*group_size # inverse of size repeated the number of samples times
        lb = ub # update lb
    return weights




# 1. Define the CVXPY Problem for the Constrained Layer (MODIFIED: Output Constraint)
def create_constrained_output_layer_on_predictions(output_dim, constraint_constant, constraint_value_upper_bound, groups_constraint_upper_bound, number_of_groups, batch_size_for_cvxpy_param=1):
    """
    Creates a CVXPY problem for a final layer that constrains its *output predictions*.

    Constraint: y_constrained / constraint_constant <= constraint_value_upper_bound
    """
    # Parameter: The unconstrained predictions from the upstream network
    # This will be (batch_size, output_dim)
# constraint_constant.shape[0]
    y_unconstrained = cp.Parameter((constraint_constant.shape[0], output_dim))

    # Variable: The constrained predictions that satisfy the constraint
    # This will also be (batch_size, output_dim)
    y_constrained = cp.Variable((constraint_constant.shape[0], output_dim))

    # Objective: Minimize the difference between unconstrained and constrained predictions
    objective = cp.Minimize(cp.sum_squares(y_constrained - y_unconstrained)) # squared l-2 norm for the projection

    # The constraint: y_constrained / constant <= value
    # This applies element-wise for each prediction in the batch and for each output.
    constraints = []
    constraints = [y_constrained[i] / constraint_constant[i] - 1 <= constraint_value_upper_bound for i in range(constraint_constant.shape[0])]
    constraints += [y_constrained[i] / constraint_constant[i] - 1 >= -constraint_value_upper_bound for i in range(constraint_constant.shape[0])]


    # Groups constraint
    y_quants = quantiles_price_tensor(constraint_constant)
    groups_intervals = np.linspace(0, 1, number_of_groups+1)
    groups_dict = dict()
    # print("y_quants:", y_quants)
    # print("groups_intervals:", groups_intervals)
    for j in range(number_of_groups):
        groups_dict[j] = torch.where((groups_intervals[j] <= y_quants) & (y_quants <= groups_intervals[j+1]) )[0]
    # print("groups_dict", groups_dict)


    # # X,y = X.to_numpy(), y.to_numpy()
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



    problem = cp.Problem(objective, constraints)

    # Parameters are y_unconstrained, variables are y_constrained
    return CvxpyLayer(problem,
                      parameters=[y_unconstrained],
                      variables=[y_constrained])

# 2. Define the MLP with the CVXPYLAYER (MODIFIED to work with output constraints)
class ConstrainedMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size,  dropout_rate, constraint_constant, constraint_value_upper_bound, groups_constraint_upper_bound, number_of_groups):
        super(ConstrainedMLP, self).__init__()

        if not isinstance(hidden_sizes, Iterable):
            raise TypeError("hidden_sizes must be a list or tuple of integers.")

        # Build the hidden layers dynamically
        layers = []
        current_input_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            current_input_size = hidden_size

        # Use nn.Sequential to chain the hidden layers
        # nn.Sequential is a special kind of Module that holds other Modules in a list.
        self.hidden_layers = nn.Sequential(*layers)

        # The input to the final linear layer is the size of the last hidden layer
        # If hidden_sizes is empty, the input is the original input_size.
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
        self.constrained_layer = create_constrained_output_layer_on_predictions(
            output_dim=output_size,
            constraint_constant=constraint_constant,
            constraint_value_upper_bound=constraint_value_upper_bound,
            groups_constraint_upper_bound=groups_constraint_upper_bound,
            number_of_groups=number_of_groups,
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




# 3. Training and Prediction Example (unchanged in logic, only model init params)
if __name__ == "__main__":
    n,m = X_train.shape
    input_size = m
    hidden_size = (50,)
    output_size = 1 # Keep output_size = 1 for simpler constraint
    learning_rate = 0.01
    epochs = 100
    dropout_rate = 0.5

    # device: cpu or gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Data to torch datatype
    X_train_torch = torch.from_numpy(X_train.astype(np.float64).to_numpy())
    y_train_torch = torch.from_numpy(y_train_scaled.astype(np.float64).to_numpy())

    # print(X_train_torch.shape)
    # print(y_train_torch.shape)


    # constraint_constant = 2.0
    constraint_constant = y_train_torch
    constraint_value_upper_bound = 1e3#5.0#0.5 # y_i / 2.0 <= 5.0  => y_i <= 10.0
    groups_constraint_upper_bound = 1e3 #0.05
    number_of_groups = 3

    # bound to normalization
    # constraint_value_upper_bound = (constraint_value_upper_bound - y_train.mean() ) / y_train.std()


    # X_test = torch.randn(20, input_size).float()
    # y_test = torch.randn(20, output_size).float() * 10

    # Initialize model, loss, and optimizer
    model = ConstrainedMLP(input_size, hidden_size, output_size, dropout_rate, constraint_constant, constraint_value_upper_bound, groups_constraint_upper_bound, number_of_groups)
    model.double()

    # Loss function
    y_train_bins = get_uniform_bins(vector_range=(1749900, 12250100), n_bins=1000) # vector_range=(14000,501000), n_bins=500)
    # print(y_train_bins)
    y_train_weights = get_group_weights(y_train, y_train_bins, alpha=10)
    print(y_train.shape)
    print(len(y_train_weights))
    print(constraint_constant.shape)
    # print(np.diag(y_train_weights).shape)
    D_full = torch.from_numpy(np.diag(y_train_weights))
    print(D_full.shape)
    # print(D_full)
    # criterion = WeightedMSELoss(D_matrix=D_full) #LInfLoss()#nn.L1Loss()#nn.MSELoss()#LInfLoss()#nn.L1Loss()#nn.MSELoss()#LInfLoss()##nn.L1Loss()#
    criterion = nn.MSELoss()#QuadraticLoss(D=D_full)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    random_observation = np.random.choice(range(constraint_constant.shape[0]))
    print("Random observation: ", random_observation)
    print(f"Constraint: | Final output / {constraint_constant[random_observation] - 1} | <= {constraint_value_upper_bound}") # (i.e., final output <= {constraint_constant[random_observation] * constraint_value_upper_bound})")
    print("-" * 30)

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(X_train_torch) # These are now the *constrained* outputs
        # print(outputs.shape)
        loss = criterion(outputs, y_train_torch)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            with torch.no_grad():
                # We can check the constraint directly on the 'outputs' tensor
                # which already represents the constrained values

                # Check constraint 1
                ratios = outputs / constraint_constant - 1
                print("max violation", torch.max(torch.abs(ratios)))
                print(outputs[:4])
                print(constraint_constant[:4])


                # print(f"  First 5 constrained outputs (should be: {constraint_constant[random_observation] * ( -constraint_value_upper_bound +1)} <= * <= {constraint_constant[random_observation] * ( constraint_value_upper_bound +1)}):")
                # print(outputs[:5].squeeze().numpy())

    print("\nTraining complete!")

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