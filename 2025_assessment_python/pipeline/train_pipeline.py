# train_pipeline.py

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import yaml
import os
import time
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor
import joblib
import optuna
from optuna.integration import OptunaSearchCV

# ------------------------------
# 1. SETUP
# ------------------------------

def load_params(path="params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

params = load_params()

def build_paths_dict(file_path="misc/file_dict.csv"):
    df = pd.read_csv(file_path, dtype=str).fillna("")
    out = {}
    for _, row in df.iterrows():
        type_ = row["type"]
        name = row["name"]
        local = os.path.normpath(row["path_local"]) if row["path_local"] else None
        if type_ not in out:
            out[type_] = {}
        out[type_][name] = {"local": local} if local else {}
    return out

paths = build_paths_dict()
cv_enable = params.get("toggle", {}).get("cv_enable", False)

start_time = time.time()
print("Run note:", params["run_note"])
print("Run type:", params["run_type"])

# ------------------------------
# 2. LOAD AND SPLIT DATA
# ------------------------------
print("Preparing model training data")
df = pd.read_parquet(paths["input"]["training"]["local"])
df = df[(~df["ind_pin_is_multicard"]) & (~df["sv_is_outlier"])].copy()
df.sort_values("meta_sale_date", inplace=True)

split_prop = params["cv"]["split_prop"]
split_index = int(len(df) * split_prop)
train = df.iloc[:split_index].copy()
test = df.iloc[split_index:].copy()

# ------------------------------
# 3. LINEAR MODEL
# ------------------------------
print("Creating and fitting linear baseline model")
pred_vars = params["model"]["predictor"]["all"]
cat_vars = params["model"]["predictor"]["categorical"]
numeric_vars = list(set(pred_vars) - set(cat_vars))

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric_vars),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_vars)
])

lin_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

train_lin = train.copy()
train_lin["meta_sale_price"] = np.log(train_lin["meta_sale_price"])
lin_pipeline.fit(train_lin[pred_vars], train_lin["meta_sale_price"])

# ------------------------------
# 4. LIGHTGBM MODEL
# ------------------------------
print("Training LightGBM model")

def get_lgbm_pipeline(hyperparams):
    model = LGBMRegressor(
        objective=params["model"]["objective"],
        random_state=params["model"]["seed"],
        verbosity=params["model"]["verbose"],
        n_jobs=os.cpu_count(),
        **hyperparams
    )
    cat_encoder = Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))])
    num_encoder = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    preprocessor = ColumnTransformer([
        ("num", num_encoder, numeric_vars),
        ("cat", cat_encoder, cat_vars)
    ])
    return Pipeline([("preprocessor", preprocessor), ("regressor", model)])

if cv_enable:
    print("Cross-validation enabled. Starting hyperparameter tuning with Optuna.")
    param_space = {
        "regressor__n_estimators": optuna.distributions.IntDistribution(*params["model"]["hyperparameter"]["range"]["num_iterations"]),
        "regressor__learning_rate": optuna.distributions.FloatDistribution(*params["model"]["hyperparameter"]["range"]["learning_rate"]),
        "regressor__num_leaves": optuna.distributions.IntDistribution(*params["model"]["hyperparameter"]["range"]["num_leaves"]),
        "regressor__feature_fraction": optuna.distributions.FloatDistribution(*params["model"]["hyperparameter"]["range"]["feature_fraction"]),
        "regressor__min_data_in_leaf": optuna.distributions.IntDistribution(*params["model"]["hyperparameter"]["range"]["min_data_in_leaf"]),
        "regressor__lambda_l1": optuna.distributions.FloatDistribution(*params["model"]["hyperparameter"]["range"]["lambda_l1"]),
        "regressor__lambda_l2": optuna.distributions.FloatDistribution(*params["model"]["hyperparameter"]["range"]["lambda_l2"]),
    }

    def rmse(y_true, y_pred):
        return mean_squared_error(y_true, y_pred, squared=False)

    scorer = make_scorer(rmse, greater_is_better=False)
    tscv = TimeSeriesSplit(n_splits=params["cv"]["num_folds"])

    base_pipeline = get_lgbm_pipeline({})
    optuna_search = OptunaSearchCV(
        estimator=base_pipeline,
        param_distributions=param_space,
        cv=tscv,
        scoring=scorer,
        n_trials=params["cv"]["max_iterations"],
        random_state=params["model"]["seed"],
        verbose=1
    )
    optuna_search.fit(train[pred_vars], train["meta_sale_price"])
    lgb_pipeline = optuna_search.best_estimator_

    pd.DataFrame(optuna_search.trials_).to_parquet(paths["output"]["parameter_raw"]["local"], index=False)
    pd.DataFrame.from_dict(param_space, orient="index").rename(columns={0: "range"}).to_parquet(paths["output"]["parameter_range"]["local"])
    pd.DataFrame([optuna_search.best_params_]).to_parquet(paths["output"]["parameter_final"]["local"], index=False)
else:
    print("Cross-validation disabled. Using default parameters.")
    default_params = {
        "n_estimators": params["model"]["hyperparameter"]["default"]["num_iterations"]
    }
    lgb_pipeline = get_lgbm_pipeline(default_params)
    lgb_pipeline.fit(train[pred_vars], train["meta_sale_price"])
    joblib.dump(pd.DataFrame(), paths["output"]["parameter_raw"]["local"])
    joblib.dump(pd.DataFrame(), paths["output"]["parameter_range"]["local"])
    joblib.dump(pd.DataFrame([default_params]), paths["output"]["parameter_final"]["local"])

# ------------------------------
# 5. SAVE PREDICTIONS
# ------------------------------
print("Saving predictions and model")

test_preds = test.copy()
test_preds["pred_card_initial_fmv"] = lgb_pipeline.predict(test[pred_vars])
test_preds["pred_card_initial_fmv_lin"] = np.exp(lin_pipeline.predict(test[pred_vars]))

keep_cols = [
    "meta_year", "meta_pin", "meta_class", "meta_card_num", "meta_triad_code",
    *params["ratio_study"]["geographies"], "char_bldg_sf",
    params["ratio_study"]["far_column"], params["ratio_study"]["near_column"],
    "pred_card_initial_fmv", "pred_card_initial_fmv_lin",
    "meta_sale_price", "meta_sale_date", "meta_sale_document_num"
]

for col in [params["ratio_study"]["far_column"], params["ratio_study"]["near_column"]]:
    test_preds[col] *= 10

output_test_path = paths["output"]["test_card"]["local"]
test_preds[keep_cols].to_parquet(output_test_path, index=False)

# Save preprocessor and model
joblib.dump(lgb_pipeline.named_steps["preprocessor"], paths["output"]["workflow_recipe"]["local"])
joblib.dump(lgb_pipeline, paths["output"]["workflow_fit"]["local"])

# ------------------------------
# 6. TIMING
# ------------------------------
end_time = time.time()
elapsed = end_time - start_time
print(f"Training completed in {elapsed:.2f} seconds")
timing_path = os.path.join(paths["intermediate"]["timing"]["local"], "model_timing_train.parquet")
pd.DataFrame([{"stage": "Train", "time_seconds": elapsed}]).to_parquet(timing_path, index=False)
