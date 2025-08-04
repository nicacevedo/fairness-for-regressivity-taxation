from src.util_functions import quantiles_price
import cvxpy as cp
import numpy as np

class ConstraintBothRegression:
    def __init__(self, solver_parameters=dict(), n_groups=3, group_thresh=1, deviation_thresh=0.5):
        self.parameters = None
        self.solver_parameters = solver_parameters
        self.group_thresh = group_thresh
        self.n_groups = n_groups
        self.deviation_thresh = deviation_thresh

    def fit(self, X, y):
        try:
            X = X.to_numpy()
        except Exception as e:
            pass
        try:
            y = y.to_numpy()
        except Exception as e:
            pass

        y_quants = quantiles_price(y)
        groups_intervals = np.linspace(0, 1, self.n_groups+1)
        groups_dict = dict()
        # print("y_quants:", y_quants)
        # print("groups_intervals:", groups_intervals)
        for j in range(self.n_groups):
            groups_dict[j] = np.where((groups_intervals[j] <= y_quants) & (y_quants <= groups_intervals[j+1]) )[0]
        # print("groups_dict", groups_dict)

        n,m = X.shape
        beta = cp.Variable(m)
        error = cp.Variable()
        constraints = [ cp.SOC(error, X @ beta - y) ] # error to minimize
        # for j in range(self.n_groups):
        #     constraints+=[
        #         (1/len(groups_dict[j])) * cp.sum( [X[i,:] @ beta / y[i] for i in  groups_dict[j]]) - 1<= self.group_thresh
        #     ]
        constraints+=[
                (1/len(groups_dict[j])) * (cp.sum( [(X[i,:] @ beta) / y[i] -1 for i in  groups_dict[j]]  ))<= self.group_thresh for j in range(self.n_groups)
            ]
        constraints+=[
                (1/len(groups_dict[j])) * (cp.sum( [(X[i,:] @ beta) / y[i] -1 for i in  groups_dict[j]]  ))>= -self.group_thresh for j in range(self.n_groups)
            ]


        # Deviation
        # constraints+=[
        #     X[i,:] @ beta / y[i] -1 <= self.deviation_thresh  for i in range(n)
        # ]
        # constraints+=[
        #     X[i,:] @ beta / y[i] -1 >= -self.deviation_thresh for i in range(n)
        # ]        
        constraints+=[
           ( X @ beta ) / y - 1 <= self.deviation_thresh # for i in range(n)
        ]
        constraints+=[
           ( X @ beta ) / y - 1 >= -self.deviation_thresh #for i in range(n)
        ]
        socp = cp.Problem(
            cp.Minimize(
                    error
                ),
            constraints
            )
        socp.solve(
            # solver="GUROBI",
            verbose=True,
            # max_iters=100
            **self.solver_parameters,
        )

        for j in range(self.n_groups):
            print("Mean of group", j, ":")
            print((1/len(groups_dict[j])) * (cp.sum( [X[i,:] @ beta.value / y[i] -1 for i in  groups_dict[j]] )))
            print("Size of group:", len(groups_dict[j]))
            # print("Indices:", groups_dict[j])

        self.parameters = beta.value

    def predict(self, X_test):
        print(self.parameters)
        try:
            X_test = X_test.to_numpy()
        except Exception as e:
            pass
        return X_test @ self.parameters
