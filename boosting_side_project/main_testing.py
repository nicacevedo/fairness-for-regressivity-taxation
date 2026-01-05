import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score

from src_.boosting_models import SimpleGradientBoosting, SimpleGradientBoosting2, SimpleGradientBoosting4, SimpleGradientBoosting4

np.random.seed(32784)

# Synthetic data with noise
n,p = 10000, 10
X = np.random.normal(0, 1, (n, p))
X[:, 0] = 1
X_noisy = X.copy()
exp_cols = np.random.choice(p, p//3, replace=False)
sin_cols = np.random.choice(p, p//3, replace=False)

X_noisy[:, exp_cols] = np.exp(X[:, exp_cols])
X_noisy[:, sin_cols] = np.sin(X[:, sin_cols])

beta = np.random.uniform(-5, 5, size=p)
y = np.abs(X_noisy @ beta) # real vector y

keep_cols = np.random.choice(p, 4*p//5, replace=False)
keep_rows = np.random.choice(n, 1*n//5, replace=False)
# Use np.ix_ to take the Cartesian product of row and column indices
rows_train = np.array(keep_rows)
cols = np.array(keep_cols)
X_train = X[np.ix_(rows_train, cols)]
y_train = y[rows_train]
# Proper test split: complement of the selected training rows
rows_test = np.setdiff1d(np.arange(n), rows_train)
X_test = X[np.ix_(rows_test, cols)]
y_test = y[rows_test]

print("X_train.shape, y_train.shape:", X_train.shape, y_train.shape)
print("X_test.shape, y_test.shape:", X_test.shape, y_test.shape)




# Models
boosting_args = {
    # Boosting
    "n_estimators":100,
    "learning_rate":1e-1,
    # Tree
    "max_depth": 3,
    # Adversarial
    "adv_learning_rate":1e-6,
    "tail_fraction":0.7,
    "tail_mode":"cvar", # "topk" (less smooth/harder ?)
    # Curvature
    "loss_type":"poisson", # "mse", "logistic", "poisson"
    "boosting_method":"gradient",  # "gradient" or "newton"
    # "gsc_hessian_term":"taylor" # "taylor", "upper", "lower"
    "gsc_M":1.0,
}
boosting_args_2 = boosting_args.copy()
boosting_args_2["boosting_method"] = "newton"
models = [
    LinearRegression(fit_intercept=True),
    SimpleGradientBoosting4(
        **boosting_args
    ),
    SimpleGradientBoosting4(
        **boosting_args_2
    ),
    SimpleGradientBoosting4(
        gsc_hessian_term="lower",
        **boosting_args_2
    ),
    SimpleGradientBoosting4(
        gsc_hessian_term="upper",
        **boosting_args_2
    ),
    # SimpleGradientBoosting4(
    #     adversarial_mode=True,
    #     **boosting_args
    # ),
    # SimpleGradientBoosting4(
    #     adversarial_mode=True,
    #     **boosting_args_2
    # ),
]





# 1. Simple linear regression
for model in models:
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    print("-"*100)
    print(model)
    print("RMSE")
    print(root_mean_squared_error(y_train, y_pred_train))
    print(root_mean_squared_error(y_test, y_pred_test))
    print("R2")
    print(r2_score(y_train, y_pred_train))
    print(r2_score(y_test, y_pred_test))
