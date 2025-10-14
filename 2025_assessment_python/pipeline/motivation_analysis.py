#%% Imports
import numpy as np 
import pandas as pd
from typing import Union, List
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error, r2_score

# from src.preliminary_models import ConstraintBothRegression, ConstraintDeviationRegression, ConstraintGroupsMeanRegression, UpperBoundLossRegression

# K-means
# from sklearn.datasets import make_blobs
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA


# New imports
import yaml

from time import sleep


import gurobipy
import mosek
import cvxpy as cp 
print(cp.installed_solvers())

# Linear regression
import os
import sys
folder_path = os.path.join(os.getcwd(), "2025_assessment_python")
sys.path.append(folder_path)
from recipes.recipes_pipelined import build_model_pipeline, build_model_pipeline_supress_onehot, ModelMainRecipe, ModelMainRecipeImputer

# My models
from src_.motivation_utils import analyze_fairness_by_value, calculate_detailed_statistics, plot_tradeoff_analysis
from fairness_models.linear_fairness_models import LeastAbsoluteDeviationRegression, MaxDeviationConstrainedLinearRegression, LeastMaxDeviationRegression, GroupDeviationConstrainedLinearRegression, StableRegression, LeastProportionalDeviationRegression#LeastMSEConstrainedRegression, LeastProportionalDeviationRegression



#%% Data
source = "CCAO" # "toy_data"

if source == "toy_data":
    # Toy dataset
    df = pd.read_csv("data/toy_data.csv")
    df.head()

    y = df["Price"]
    X = df.drop(columns=["Price"])

elif source == "House":
    # Kaggle House Pricing dataset
    df = pd.read_csv("data/Housing.csv")

    # Get dummies of categorial
    df = pd.get_dummies(df, drop_first=True)

    # Add a constant columns
    # df["intercept"] = 1

    display(df.head())

    y = df["price"]
    X = df.drop(columns=["price"])
elif source == "California":
    # Data from Google Colab samples
    df = pd.read_csv("data/california_housing_train.csv")
    # Add a constant columns
    # df["intercept"] = 1

    # Drop outliers at y.max() (too many to be true. Must be a threshold)
    df = df.loc[df["median_house_value"] < df["median_house_value"].max(),:]

    y = df["median_house_value"]
    X = df.drop(columns=["median_house_value"])

elif source == "CCAO":
    df = pd.read_parquet("../data_county/2025/training_data.parquet", engine="fastparquet")#.sample(100000)
    df = df[
        (~df['ind_pin_is_multicard'].astype('bool').fillna(True)) &
        (~df['sv_is_outlier'].astype('bool').fillna(True))
    ]

#%% Preprocessing

# Get only the desired columns
with open('params.yaml', 'r') as file:
    params = yaml.safe_load(file)

desired_columns = params['model']['predictor']['all'] +  ['meta_sale_price', 'meta_sale_date'] 
df = df.loc[:,desired_columns]


# Train - test split
df.sort_values(by="meta_sale_date", ascending=True, inplace=True)
n,m = df.shape
print("shape: ", (n,m))
train_prop = 0.822871 # exact match of 2022 // 2023+2024
df_train = df.iloc[:int(train_prop*n),:]
df_test = df.iloc[int(train_prop*n):,:]

# Random sample of train
sample_size = 10000
if sample_size < df_train.shape[0]:
    df_train = df_train.sample(min(sample_size, df_train.shape[0]), random_state=42, replace=False)
else:
    sample_size = df_train.shape[0]
df_train.sort_values(by="meta_sale_date", ascending=True, inplace=True)

# Train - val split
train_prop = 0.8622 # almost exact match of 2021 // 2022, for 10k sample
df_val = df_train.iloc[int(train_prop*sample_size):,:]
df_train = df_train.iloc[:int(train_prop*sample_size),:]
df_train['meta_sale_date']


# Create proper X,y 
X_train, y_train = df_train.drop(columns=['meta_sale_date', 'meta_sale_price']), df_train['meta_sale_price']
X_val, y_val = df_val.drop(columns=['meta_sale_date', 'meta_sale_price']), df_val['meta_sale_price']
X_test, y_test = df_test.drop(columns=['meta_sale_date', 'meta_sale_price']), df_test['meta_sale_price']

# Log version of the targets
y_train_log = np.log(y_train)
y_val_log = np.log(y_val)
y_test_log = np.log(y_test)


# Preprocessing pipeline (TO BE REVISED)
linear_pipeline = build_model_pipeline(
    pred_vars=params['model']['predictor']['all'],
    cat_vars=params['model']['predictor']['categorical'],
    id_vars=[],
)

X_train = linear_pipeline.fit_transform(X_train, y_train_log)
X_val = linear_pipeline.transform(X_val)
X_test = linear_pipeline.transform(X_test)
X_train.head()

#%% Linera Models

from sklearn.linear_model import LinearRegression

model = LinearRegression(fit_intercept=True)
model.fit(X_train, y_train_log)
y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)
y_pred_test = model.predict(X_test)

print("train RMSE: ", root_mean_squared_error(y_train_log, y_pred_train) )

print(r"$R^2$ in Train: ",r2_score(y_pred_train, y_train_log))
print(r"$R^2$ in Val: ",r2_score(y_pred_val, y_val_log))
print(r"$R^2$ in Test: ",r2_score(y_pred_test, y_test_log))
print("-"*100)
print("Residual = (y - X'b)")
print("Max residual Train: ", np.max(y_train_log - y_pred_train))
print("Min residual Train: ", np.min(y_train_log - y_pred_train))
print("Avg residual Train: ", np.mean(y_train_log - y_pred_train))
print("Median residual Train: ", np.median(y_train_log - y_pred_train))
print("-"*100)
print("Max residual Val: ", np.max(y_val_log - y_pred_val))
print("Min residual Val: ", np.min(y_val_log - y_pred_val))
print("Avg residual Val: ", np.mean(y_val_log - y_pred_val))
print("Median residual Val: ", np.median(y_val_log - y_pred_val))




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


# +-------------------------------------------------+
# |             PART 3: MAIN EXPERIMENT LOOP        |
# +-------------------------------------------------+

if __name__ == '__main__':
    # --- Configuration ---
    NUM_GROUPS = 3
    percentages = np.linspace(0, .06, 11)

    # --- Data Storage ---
    train_results_list = []
    val_results_list = []
    pof_list = []
    pof_lb_list = []
    fei_list = []
    
    # --- Main Loop ---
    for rmse_percentage_increase in percentages:
        print("-" * 100)
        print(f"RUNNING EXPERIMENT FOR % DECREASE = {rmse_percentage_increase:.4f}")
        # model = LeastMSEConstrainedRegression(
        # model = LeastMaxDeviationRegression(
        # model = LeastProportionalDeviationRegression(
        # model = LeastProportionalDeviationRegression(
        # model = LeastGroupDeviationRegression(
        # model = MaxDeviationConstrainedLinearRegression(
        model = GroupDeviationConstrainedLinearRegression(
            fit_intercept=True,# add_rmse_constraint=False, 
            percentage_increase=0.1,#rmse_percentage_increase,
            max_row_norm_scaling=(1+rmse_percentage_increase),
            solver="MOSEK", # Your solver of choice
            l2_lambda=1e0,
        )
        # Robust version
        # l1_, l2_ = 1e-1, 5e3#1e2*10**(10*rmse_percentage_increase) # 1e-1/10**(10*rmse_percentage_increase), 1e-1*10**(10*rmse_percentage_increase))
        # print("l1: ", l1_," | l2: ", l2_)
        # model = StableRegression(
        #     fit_intercept=True,
        #     k_percentage=(1-0.02),#(1-rmse_percentage_increase),
        #     solver="MOSEK",
        #     lambda_l1=l1_,#1e-1/10**(10*rmse_percentage_increase),
        #     lambda_l2=l2_,#1e-1*10**(10*rmse_percentage_increase),
        #     objective="mae",
        # )

        # Train R2: 0.7954 | Val R2: 0.7876 -- l1:  1.778279410038923e-06  | l2:  5623.413251903491
        
        # Fit model and make predictions
        result, solve_time, pof, pof_lb, fei = model.fit(X_train, y_train_log)
        pof_list.append(pof)
        pof_lb_list.append(pof_lb)
        fei_list.append(fei)
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)

        # Calculate and store statistics
        train_stats = calculate_detailed_statistics(y_train_log, y_pred_train, num_groups=NUM_GROUPS)
        train_stats['percentage'] = rmse_percentage_increase
        train_results_list.append(train_stats)
        
        val_stats = calculate_detailed_statistics(y_val_log, y_pred_val, num_groups=NUM_GROUPS)
        val_stats['percentage'] = rmse_percentage_increase
        val_results_list.append(val_stats)

        print(f"Train R2: {train_stats['r2']:.4f} | Val R2: {val_stats['r2']:.4f}")
        # print(f"(True) Train R2: {r2_score(y_train, np.exp(y_pred_train)):.4f} | Val R2: {r2_score(y_val, np.exp(y_pred_val)):.4f}")
        print(f"Train RMSE: {train_stats['rmse']:.4f} | Val RMSE: {val_stats['rmse']:.4f}")
        print(f"Train Max Abs Deviation: {train_stats['max_abs_deviation']:.4f} | Val Max Abs Deviation: {val_stats['max_abs_deviation']:.4f}")
        print("-" * 100 + "\n")

        # Summary
        print(analyze_fairness_by_value(y_pred_train, y_train).round(3).T.to_csv(f"outputs/reports/motivation/summary_train_{rmse_percentage_increase:.4f}.csv",  float_format='%.3f'))
        print(analyze_fairness_by_value(y_pred_val, y_val).round(3).T.to_csv(f"outputs/reports/motivation/summary_val_{rmse_percentage_increase:.4f}.csv",  float_format='%.3f'))


    # --- Temporal Plot ---
    print(os.getcwd())
    plt.figure(figsize=(6,5))
    plt.plot(percentages, pof_list, "--o")
    # plt.xlabel("Fairness Threshold Improvement (%)")
    plt.xlabel("Max_i norm(x_i) Increase (%)")
    plt.ylabel("Price of Fairness (% MSE increase)")
    # plt.savefig("/img/motivation/tradoff_analysis/pof/pof_vs_percentages.jpg", dpi=300)
    plt.savefig("./img/motivation/pof_vs_percentages.jpg", dpi=300, bbox_inches='tight')

    # plt.plot(percentages, fei_list)

    # --- Analysis and Plotting ---
    train_results_df = pd.DataFrame(train_results_list)
    val_results_df = pd.DataFrame(val_results_list)

    print(train_results_df)
    print(val_results_df)

    # Re-map columns for plotting function compatibility
    train_results_df.columns = train_results_df.columns.str.replace('overall_', '')
    val_results_df.columns = val_results_df.columns.str.replace('overall_', '')

    print("Generating plots for training data...")
    plot_tradeoff_analysis(train_results_df, percentages, num_groups=NUM_GROUPS, save_dir="img/motivation/tradeoff_analysis/train")
    
    print("\nGenerating plots for validation data...")
    plot_tradeoff_analysis(val_results_df, percentages, num_groups=NUM_GROUPS, save_dir="img/motivation/tradeoff_analysis/val")


