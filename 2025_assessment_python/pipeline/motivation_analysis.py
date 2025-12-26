#%% Imports
import numpy as np 
import pandas as pd
from typing import Union, List
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, ElasticNet
# from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error

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
from src_.motivation_utils import analyze_fairness_by_value, calculate_detailed_statistics, plot_tradeoff_analysis, compute_taxation_metrics
from fairness_models.linear_fairness_models import LeastAbsoluteDeviationRegression, MaxDeviationConstrainedLinearRegression, LeastMaxDeviationRegression, GroupDeviationConstrainedLinearRegression, StableRegression, LeastProportionalDeviationRegression#LeastMSEConstrainedRegression, LeastProportionalDeviationRegression
from fairness_models.linear_fairness_models import MyGLMRegression, GroupDeviationConstrainedLogisticRegression, RobustStableLADPRDCODRegressor, StableAdversarialSurrogateRegressor, StableAdversarialSurrogateRegressor2

# My boosting models
import lightgbm as lgb
# from fairness_models.boosting_fairness_models import custom_objective, custom_eval
# from fairness_models.boosting_fairness_models import make_constrained_mse_objective, make_covariance_metric
from fairness_models.boosting_fairness_models import LGBMCovRatioRegressor, CovPenaltyConfig, LGBCustomObjective

# UC Irvine data
from src_.ucirvine_preprocessing import get_uci_column_names, preprocess_adult_data

# Results utils
from src_.plotting_utils import results_to_dataframe, plotting_dict_of_models_results

#%% Data
# source = "CCAO" # "toy_data"
seed = 234
source = "CCAO"

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
    target_name = "meta_sale_price"
    model_name = "linear"

    # Get only the desired columns
    with open('params.yaml', 'r') as file:
        params = yaml.safe_load(file)

    desired_columns = params['model']['predictor']['all'] +  [target_name, 'meta_sale_date'] 
    df = df.loc[:,desired_columns]
    
    # Train - test split
    df.sort_values(by="meta_sale_date", ascending=True, inplace=True)

elif source == "sklearn":
    from sklearn.datasets import load_breast_cancer, load_diabetes
    # df = load_breast_cancer(as_frame=True)
    df = load_diabetes(as_frame=True)
    X = df.data
    y = df.target

    df = X.copy()
    df["target"] = y
    target_name = "target"
    sensitive_name = "sex"
    model_name = "linear"

    np.random.seed(seed)
    shuffled_indices = df.index.to_list().copy()
    np.random.shuffle(shuffled_indices)
    df = df.iloc[shuffled_indices, :]


elif source  == "liblinear":
    from sklearn.datasets import load_svmlight_file
    X, y = load_svmlight_file(f"data/{source}/a4a.txt")
    X = X.toarray()
    df = pd.DataFrame(X)
    # for col in df.columns:
    #     print(df[col].unique())
    y[y == -1] = 0
    df["target"] = y
    target_name = "target"
    sensitive_name = 10
    model_name = "logistic"

    print(df.head())
    exit()

elif source == "ucirvine":
    data_name = "student" #"student" # adult
    if data_name == "adult":
        # column_names = get_uci_column_names(f'data/{source}/{data_name}/{data_name}.names')
        column_names = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
            'income'  # This is the target variable
        ]
        # df = pd.read_csv(, header=None, names=column_names)
        df = pd.read_csv(
            f'data/{source}/{data_name}/{data_name}.data',
            header=None,
            names=column_names,
            sep=',\s*',
            engine='python',
            na_values='?',
            skiprows=1  # Skip the first row, which often has a note/header
        )
        target_name = "income"
        sensitive_name = "race"#"cat__sex_Male" # sex is already as fair as it can
        model_name = "logistic"
        df = df.loc[df["age"] <= 65,:]
        print(df.head())

        # Preprocessing of UC Irvine
        X, y, pipe = preprocess_adult_data(df, pass_features=[sensitive_name])
        sensitive_name = f"passthrough__{sensitive_name}" # updated with preprocessing
        X = pd.DataFrame(X, columns=pipe.get_feature_names_out())

        # print(X.head())
        df = X.copy()
        df[target_name] = y
        df.drop(columns=["passthrough__fnlwgt"], inplace=True)
    elif data_name == "student":
        df = pd.read_csv(f'data/{source}/{data_name}/{data_name}-mat.csv', sep=";")
        sensitive_name = "sex"
        target_name = "G3"
        model_name = "linear"
    elif data_name == "abalone":
        column_names = [
            "Sex",
            "Length",		    #	mm	Longest shell measurement
            "Diameter",	    #	mm	perpendicular to length
            "Height",		    #	mm	with meat in shell
            "Whole weight",	#	grams	whole abalone
            "Shucked weight",	#	grams	weight of meat
            "Viscera weight",	#	grams	gut weight (after bleeding)
            "Shell weight",	#	grams	after being dried
            "Rings",		    #integer			+1.5 gives the age in years
        ]
        df = pd.read_csv(
            f'data/{source}/{data_name}/{data_name}.data',
            header=None,
            names=column_names,
            sep=',\s*',
            engine='python',
            na_values='?',
            skiprows=1  # Skip the first row, which often has a note/header
        )
        sensitive_name = "Sex"
        target_name = "Rings"
        model_name = "poisson"
    
    
    sensitive_mapping = [value for value in df[sensitive_name].unique()]
    X, y, pipe= preprocess_adult_data(df, target_name=target_name, pass_features=[sensitive_name])
    sensitive_name = f"passthrough__{sensitive_name}" # updated with preprocessing
    X = pd.DataFrame(X, columns=pipe.get_feature_names_out())
    # print(X[sensitive_name].head(10))
    sensitive_mapping = {i:value for i,value in enumerate(sensitive_mapping)}
    y = pd.Series(y)
    df = X.copy()
    df[target_name] = y


    # Shuffling
    # seed=234#123
    np.random.seed(seed)
    shuffled_indices = df.index.to_list().copy()
    np.random.shuffle(shuffled_indices)
    df = df.iloc[shuffled_indices, :]


#%% Preprocessing

# Data size
n,m = df.shape
print(df.head())
print(df[target_name].unique())

print("shape: ", (n,m))
train_prop = 0.822871 if source == "CCAO" else 0.99 # exact match of 2022 // 2023+2024
df_train = df.iloc[:int(train_prop*n),:]
df_test = df.iloc[int(train_prop*n):,:]

# Random sample of train
sample_size = 100000 # 10k samples for Abalon (?)# 1000 samples for Adult (?)
if sample_size < df_train.shape[0]:
    print("working with a sample (10k)")
    df_train = df_train.sample(min(sample_size, df_train.shape[0]), random_state=seed, replace=False)

    # if source == "ucirvine":
        # Repeat rows
        # df_train = df_train.loc[df.index.repeat(df['passthrough__fnlwgt'])].reset_index(drop=True)

    print("shape: ", (n,m))

else:
    sample_size = df_train.shape[0]

if source == "CCAO":
    df_train.sort_values(by="meta_sale_date", ascending=True, inplace=True)

# Train - val split
train_prop = 0.8622 # almost exact match of 2021 // 2022, for 10k sample
df_val = df_train.iloc[int(train_prop*sample_size):,:]
df_train = df_train.iloc[:int(train_prop*sample_size),:]
# df_train['meta_sale_date']



# Create proper X,y 
if source == "CCAO":
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

else:
    X_train, y_train = df_train.drop(columns=[target_name, sensitive_name]), df_train[target_name]
    X_val, y_val = df_val.drop(columns=[target_name, sensitive_name]), df_val[target_name]
    X_test, y_test = df_test.drop(columns=[target_name, sensitive_name]), df_test[target_name]


    # def normalize_rows(X):
    #     X_ = X.to_numpy()
    #     norm_ = np.linalg.norm(X_, axis=1, keepdims=True)
    #     return pd.DataFrame(X_ / norm_, columns=X.columns, index=X.index)

    # # WARNING: Normalization
    # scaler = MinMaxScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_val = scaler.transform(X_val)
    # X_test = scaler.transform(X_test)
    # X_train = normalize_rows(X_train)
    # X_val = normalize_rows(X_val)
    # X_test = normalize_rows(X_test)

############
# Linear Models
###########################################

# Prepare data
if source == "CCAO":
    y_train_scaled = y_train_log
    y_val_scaled = y_val_log
    y_test_scaled = y_test_log
else: 
    y_train_scaled = y_train
    y_val_scaled = y_val
    y_test_scaled = y_test


################################################################################
# Experiments on Stable Regression and Covariance Constrained Models 
################################################################################

# Inputs
random_state = 42
n_jobs =380#190
max_iter=1000

fit_intercept = True
l1,l2 = 1e-3, 1e-2
max_depth = 15
lr = 1e-1

# Model-specific
epsilons_cov = [2.25e-2, 1e-2, 8.75e-3, 7.5e-3]#, 5e-3] # Cov 2.25e-2, 
# epsilons_var = [1e-1, 7.5e-2, 5e-2, 2.5e-2] # Var: Not doing anything
rhos_cov = [1e-1, 1, 5, 10]#, 1.5] # Cov
# rhos_var = [0] #[0, 1e-1, 1, 5, 10, 20]#, 10] # Var: Not doing anything
keep_percentages = np.linspace(0.3, 1, 8)


models = [
    LinearRegression(fit_intercept=fit_intercept, n_jobs=n_jobs),
    # LeastAbsoluteDeviationRegression(fit_intercept=fit_intercept, solver="MOSEK"),
    ElasticNet(fit_intercept=fit_intercept, l1_ratio=l1/(l1 + l2), alpha=(l1 + l2), selection="random", random_state=random_state, warm_start=True),
    # RandomForestRegressor(n_estimators=n_jobs, criterion='squared_error', max_depth=max_depth, min_samples_split=50, min_samples_leaf=30, bootstrap=True, n_jobs=n_jobs, random_state=random_state, warm_start=True, ccp_alpha=1e-3),
    # GradientBoostingRegressor(loss='squared_error', learning_rate=1e-3, n_estimators=100, subsample=0.8, criterion='friedman_mse', min_samples_split=50, min_samples_leaf=20, max_depth=3, random_state=random_state, alpha=0.9, warm_start=True, validation_fraction=0.1, tol=1e-4, ccp_alpha=1e-3)
    # HistGradientBoostingRegressor(loss='squared_error', learning_rate=lr, max_iter=max_iter, max_leaf_nodes=31, max_depth=max_depth, min_samples_leaf=30, l2_regularization=l2, max_bins=255, 
    #                               warm_start=True, early_stopping='auto', scoring='loss', validation_fraction=0.2, n_iter_no_change=10, tol=1e-6, random_state=random_state),     
    lgb.LGBMRegressor(boosting_type='gbdt', num_leaves=31, max_depth=max_depth, learning_rate=lr, n_estimators=max_iter, subsample_for_bin=200000, objective="mse", 
                                class_weight=None, min_child_samples=30, colsample_bytree=1.0, reg_alpha=l1, reg_lambda=l2, random_state=random_state, n_jobs=8, importance_type='split'),
]


baseline_models = [str(model).split("(")[0] for model in models] # Save baseline models names



# # Stable Regression
# models.append(
#     StableRegression(
#         fit_intercept=fit_intercept, keep=1, lambda_l1=l1, lambda_l2=l2,
#         fit_group_intercept=False, delta_l2=0,  weight_by_group=False, solver="MOSEK",
#         sensitive_idx=None, #sensitive_feature=d.loc[X_train.index].to_numpy(),
#         cov_constraint=False, eps_cov=0,
#         var_constraint=False, eps_var=0,
#     )
# )




# # Experimental models
# # 1. Constrained Stable
# # for eps in epsilons_var:
# for eps in epsilons_cov:
#     models.append(
#         StableRegression(
#             fit_intercept=fit_intercept, keep=1, lambda_l1=l1, lambda_l2=l2,
#             fit_group_intercept=False, delta_l2=0,  weight_by_group=False, solver="MOSEK",
#             sensitive_idx=None, #sensitive_feature=d.loc[X_train.index].to_numpy(),
#             cov_constraint=True, eps_cov=eps,
#             var_constraint=False, eps_var=0,
#         )
#     )

# # 2. Adversarial Stable
# # for rho_var in rhos_var:
# for rho_cov in rhos_cov:
#     models.append(
#         StableAdversarialSurrogateRegressor(
#             fit_intercept=fit_intercept, keep=1, l1=l1, l2=l2,
#             solver="MOSEK", verbose=False, warm_start=True,
#             rho_cov=rho_cov, neg_corr_focus=False,
#             rho_var=0, 
#         )
#     )

# # 3. Separable AdversarialStable
# # for rho_var in rhos_var:
# for rho_cov in rhos_cov:
#     models.append(
#         StableAdversarialSurrogateRegressor2(
#             fit_intercept=fit_intercept, keep=1, l1=l1, l2=l2,
#             solver="MOSEK", verbose=False, warm_start=True,
#             rho_cov=rho_cov, neg_corr_focus=False,
#             rho_var=0, 
#         )
#     )


# # 4. Sof-penalized LGBM
lgbm_params = {
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "max_depth": max_depth,
    "learning_rate": lr,
    "n_estimators": max_iter,
    "subsample_for_bin": 200000,
    "objective": "mse", # To be updated inside
    "class_weight": None,
    "min_child_samples": 30,
    "colsample_bytree": 1.0,
    "reg_alpha": l1,
    "reg_lambda": l2,
    "random_state": random_state,
    "n_jobs": 1,#n_jobs,
    "importance_type": "split",
}
# for eps in [1e-3, 1e-1, 1e1, 1e3]:
rhos = [1e2, 5e2, 1e3, 1e4]#[1e1, 1e2]#, 1e3, 1e4]#, 1e5] #5e3, 1e4, 5e4] # Last ones: 5e2,5e3,
for rho in rhos:
    for adversary_type in ["overall", "individual"]:
        # for r_keep in keep_percentages:
            # cobj = LGBCustomObjective(rho=rho)
            #         # Lgbm params
            # models.append( # custom_obj # 
            #     lgb.LGBMRegressor(boosting_type='gbdt', num_leaves=31, max_depth=max_depth, learning_rate=lr, n_estimators=max_iter, subsample_for_bin=200000, objective=cobj.fobj,#"mse", 
            #                     class_weight=None, min_child_samples=30, colsample_bytree=1.0, reg_alpha=l1, reg_lambda=l2, random_state=random_state, n_jobs=n_jobs, importance_type='split'),
            # )
            models.append(
                LGBCustomObjective(rho=rho, keep=1, adversary_type=adversary_type, lgbm_params=lgbm_params)
            )





# Correction of baseline models
model_names = [str(model) for model in models]
for name_1 in baseline_models:
    for i,name_2 in enumerate(model_names):
        if name_1 in name_2:
            model_names[i] = name_1


print("models: ", model_names)

# Save results on a dict
results_train = {name:[] for name in model_names}
results_val = {name:[] for name in model_names}

for i, model in enumerate(models):
    model_name = model_names[i]

    # Loop the percentages
    r_list = [None] if model_name in baseline_models else keep_percentages
    for r_per in r_list:

        print("( r=", r_per,")")
        if model_name not in baseline_models:
            model.keep = r_per # update the percentage
        print("Fitting model: ", model_name)
        model.fit(X_train, y_train_log)

        # Prediction
        y_pred_log = model.predict(X_train)
        metrics_train = compute_taxation_metrics(y_train_log, y_pred_log, scale="log")
        results_train[model_name].append(metrics_train)

        # Out of sample Prediction
        y_pred_log = model.predict(X_val)
        metrics_val = compute_taxation_metrics(y_val_log, y_pred_log, scale="log")
        results_val[model_name].append(metrics_val)

print(results_to_dataframe(results_train, r_values=r_list))
print(results_to_dataframe(results_val, r_values=r_list))

plotting_dict_of_models_results(results_train, r_list=r_list, source="train")
plotting_dict_of_models_results(results_val, r_list=r_list, source="val")

exit()






diffs_ = [-1, -2]#, #10, 5] #5e-2, 1e-2, 5e-3]
for i,diff in enumerate(diffs_):
    keep_percentages = np.linspace(0.1, 1, 1)

    for keep_percentage in keep_percentages:
        print("-"*100)
        print("Diff=", diff)
        print("Keep=", keep_percentage)

        # model = LinearRegression(fit_intercept=True)
        if diff == -1:
            model = LinearRegression(fit_intercept=True)
            model.fit(X_train, y_train_log)
        elif diff == -2:
            model = StableAdversarialSurrogateRegressor(
                keep=keep_percentage,
                rho=1.0,
                l1=1e-4,
                l2=1e-4,
                fit_intercept=True,
                solver="MOSEK",
                # solver_opts=None,
                verbose=False,
                warm_start=True,
                neg_corr_focus=False,
            )

            model.fit(X_train, y_train_log, d=y_train_log) #s=y_train_log, v=y_train_log)
            print("Model results: ", model.status_, model.objective_value_)
        else:
            model = StableRegression(
                    fit_intercept=True, keep=keep_percentage, lambda_l1=1e-4, lambda_l2=1e-4,
                    fit_group_intercept=False, delta_l2=0, group_constraints=True, weight_by_group=False, 
                    sensitive_idx=None, #sensitive_feature=d.loc[X_train.index].to_numpy(),
                    group_percentage_diff=diff, solver="MOSEK")
            model.fit(X_train, y_train_log)
        
        y_pred_log = model.predict(X_val)
        results_val = compute_taxation_metrics(y_val_log, y_pred_log, scale="log")
        print(results_val)

        y_pred_log = model.predict(X_val)
        results_val = compute_taxation_metrics(y_val_log, y_pred_log, scale="log")
        print(results_val)


exit()



        # y_pred = model.predict(X_val)
        # print("R squared: ", r2_score(y_val_log, y_pred))
        # print("Corr of pred and y: ", np.corrcoef(y_val_log, y_pred)[0,1])
        # print("Corr of res and y: ", np.corrcoef(y_val_log, y_val_log - y_pred)[0,1])
        # print("Corr of ratio and y: ", np.corrcoef(y_val_log, y_pred/y_val_log)[0,1])
        # corrs_val.append(np.abs(np.corrcoef(y_val_log, y_pred/y_val_log)[0,1]))
        # corrs_res_y_val.append(np.abs(np.corrcoef(y_val_log, y_val_log-y_pred)[0,1]))
        # r2_val.append(r2_score(y_val_log, y_pred))
        # pred_var_val.append(np.std(y_val_log - y_pred)**2)

        # ratios = np.exp(y_pred) / y_val
        # cod_val.append( 100/np.median(ratios)*np.mean(np.abs(ratios - np.median(ratios))) )
        # print("COD val:", 100/np.median(ratios)*np.mean(np.abs(ratios - np.median(ratios))))
        # prd_val.append( np.mean(ratios) / (ratios @ y_val) *  np.sum(y_val)  )
        # print("PRD val:", np.mean(ratios) / (ratios @ y_val) *  np.sum(y_val) )
        # b_1, b_0 = np.polyfit(y_val_log, ratios, 1) # R_i = beta_0 + beta_1 * log(y_i) + eps_i 
        # prb = 2*(np.exp(b_1) - 1) * 100
        # print("PRB val:", prb)
        # prb_val.append(prb)


    # results["train_corrs"].append(corrs_train)
    # results["val_corrs"].append(corrs_val)
    # results["train_res_y_corrs"].append(corrs_res_y_train)
    # results["val_res_y_corrs"].append(corrs_res_y_val)
    # results["train_r2"].append(r2_train)
    # results["val_r2"].append(r2_val)
    # results["train_res_var"].append(pred_var_train)
    # results["val_res_var"].append(pred_var_val)
    # # CCAO
    # results["train_cod"].append(cod_train)
    # results["val_cod"].append(cod_val)
    # results["train_prd"].append(prd_train)
    # results["val_prd"].append(prd_val)
    # results["train_prb"].append(prb_train)
    # results["val_prb"].append(prb_val)


# Correlation between RtA and y
plt.figure(figsize=(10,6))
for i,diff in enumerate(diffs):
    plt.plot(keep_percentages, results["train_corrs"][i], '--o', color=f"C{i}", alpha=0.7, label=f"Diff={diff}")
plt.grid(True, which="both", color="0.85", linewidth=0.5)  # light gray
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title("Stable Regression with |Cov(f(X)/y, y)|<= Diff*std(y) [Training Set]")
plt.xlabel("Ratio of samples to keep")
plt.ylabel("|Corr(f(x)/y, y)|")
plt.tight_layout()
plt.savefig("./temp/plots/hist_train_corr.png", dpi=600)

plt.figure(figsize=(10,6))
for i,diff in enumerate(diffs):
    plt.plot(keep_percentages, results["val_corrs"][i], '--o', color=f"C{i}", alpha=0.7, label=f"Diff={diff}")
plt.grid(True, which="both", color="0.85", linewidth=0.5)  # light gray
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title("Stable Regression with |Cov(f(X)/y, y)|<= Diff*std(y) [Testing Set]")
plt.xlabel("Ratio of samples to keep")
plt.ylabel("|Corr(f(x)/y, y)|")
plt.tight_layout()
plt.savefig("./temp/plots/hist_val_corr.png", dpi=600)


# Correlation between residuals and y
plt.figure(figsize=(10,6))
for i,diff in enumerate(diffs):
    plt.plot(keep_percentages, results["train_res_y_corrs"][i], '--s', color=f"C{i}", alpha=0.7, label=f"Diff={diff}")
plt.grid(True, which="both", color="0.85", linewidth=0.5)  # light gray
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title("Stable Regression with |Cov(f(X)/y, y)|<= Diff*std(y) [Training Set]")
plt.xlabel("Ratio of samples to keep")
plt.ylabel("|Corr(res(x), y)|")
plt.tight_layout()
plt.savefig("./temp/plots/hist_train_res_y_corr.png", dpi=600)

plt.figure(figsize=(10,6))
for i,diff in enumerate(diffs):
    plt.plot(keep_percentages, results["val_res_y_corrs"][i], '--s', color=f"C{i}", alpha=0.7, label=f"Diff={diff}")
plt.grid(True, which="both", color="0.85", linewidth=0.5)  # light gray
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title("Stable Regression with |Cov(f(X)/y, y)|<= Diff*std(y) [Testing Set]")
plt.xlabel("Ratio of samples to keep")
plt.ylabel("|Corr(res(x), y)|")
plt.tight_layout()
plt.savefig("./temp/plots/hist_val_res_y_corr.png", dpi=600)


# R2 plots
plt.figure(figsize=(10,6))
for i,diff in enumerate(diffs):
    plt.plot(keep_percentages, results["train_r2"][i], '-.x', color=f"C{i}", alpha=0.7, label=f"Diff={diff}")
plt.grid(True, which="both", color="0.85", linewidth=0.5)  # light gray
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title("Stable Regression with |Cov(f(X)/y, y)|<= Diff*std(y) [Training Set]")
plt.xlabel("Ratio of samples to keep")
plt.ylabel("R2 Score")
plt.tight_layout()
plt.savefig("./temp/plots/hist_train_r2.png", dpi=600)

plt.figure(figsize=(10,6))
for i,diff in enumerate(diffs):
    plt.plot(keep_percentages, results["val_r2"][i], '-.x', color=f"C{i}", alpha=0.7, label=f"Diff={diff}")
plt.grid(True, which="both", color="0.85", linewidth=0.5)  # light gray
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title("Stable Regression with |Cov(f(X)/y, y)|<= Diff*std(y) [Testing Set]")
plt.xlabel("Ratio of samples to keep")
plt.ylabel("R2 Score")
plt.tight_layout()
plt.savefig("./temp/plots/hist_val_r2.png", dpi=600)

# Std plots
plt.figure(figsize=(10,6))
for i,diff in enumerate(diffs):
    plt.plot(keep_percentages, results["train_res_var"][i], '-.^', color=f"C{i}", alpha=0.7, label=f"Diff={diff}")
plt.grid(True, which="both", color="0.85", linewidth=0.5)  # light gray
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title("Stable Regression with |Cov(f(X)/y, y)|<= Diff*std(y) [Training Set]")
plt.xlabel("Ratio of samples to keep")
plt.ylabel("Residuals Variance")
plt.tight_layout()
plt.savefig("./temp/plots/hist_train_res_var.png", dpi=600)

plt.figure(figsize=(10,6))
for i,diff in enumerate(diffs):
    plt.plot(keep_percentages, results["val_res_var"][i], '-.^', color=f"C{i}", alpha=0.7, label=f"Diff={diff}")
plt.grid(True, which="both", color="0.85", linewidth=0.5)  # light gray
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title("Stable Regression with |Cov(f(X)/y, y)|<= Diff*std(y) [Testing Set]")
plt.xlabel("Ratio of samples to keep")
plt.ylabel("Residuals Variance")
plt.tight_layout()
plt.savefig("./temp/plots/hist_val_res_var.png", dpi=600)

# COD plots
plt.figure(figsize=(10,6))
for i,diff in enumerate(diffs):
    plt.plot(keep_percentages, results["train_cod"][i], '-v', color=f"C{i}", alpha=0.7, label=f"Diff={diff}")
plt.grid(True, which="both", color="0.85", linewidth=0.5)  # light gray
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title("Stable Regression with |Cov(f(X)/y, y)|<= Diff*std(y) [Training Set]")
plt.xlabel("Ratio of samples to keep")
plt.ylabel("COD")
plt.tight_layout()
plt.savefig("./temp/plots/hist_train_cod.png", dpi=600)

plt.figure(figsize=(10,6))
for i,diff in enumerate(diffs):
    plt.plot(keep_percentages, results["val_cod"][i], '-v', color=f"C{i}", alpha=0.7, label=f"Diff={diff}")
plt.grid(True, which="both", color="0.85", linewidth=0.5)  # light gray
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title("Stable Regression with |Cov(f(X)/y, y)|<= Diff*std(y) [Testing Set]")
plt.xlabel("Ratio of samples to keep")
plt.ylabel("COD")
plt.tight_layout()
plt.savefig("./temp/plots/hist_val_cod.png", dpi=600)

# PRD plots
plt.figure(figsize=(10,6))
for i,diff in enumerate(diffs):
    plt.plot(keep_percentages, results["train_prd"][i], '--D', color=f"C{i}", alpha=0.7, label=f"Diff={diff}")
plt.grid(True, which="both", color="0.85", linewidth=0.5)  # light gray
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title("Stable Regression with |Cov(f(X)/y, y)|<= Diff*std(y) [Training Set]")
plt.xlabel("Ratio of samples to keep")
plt.ylabel("PRD")
plt.tight_layout()
plt.savefig("./temp/plots/hist_train_prd.png", dpi=600)

plt.figure(figsize=(10,6))
for i,diff in enumerate(diffs):
    plt.plot(keep_percentages, results["val_prd"][i], '--D', color=f"C{i}", alpha=0.7, label=f"Diff={diff}")
plt.grid(True, which="both", color="0.85", linewidth=0.5)  # light gray
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title("Stable Regression with |Cov(f(X)/y, y)|<= Diff*std(y) [Testing Set]")
plt.xlabel("Ratio of samples to keep")
plt.ylabel("PRD")
plt.tight_layout()
plt.savefig("./temp/plots/hist_val_prd.png", dpi=600)

# PRB plots
plt.figure(figsize=(10,6))
for i,diff in enumerate(diffs):
    plt.plot(keep_percentages, results["train_prb"][i], '--D', color=f"C{i}", alpha=0.7, label=f"Diff={diff}")
plt.grid(True, which="both", color="0.85", linewidth=0.5)  # light gray
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title("Stable Regression with |Cov(f(X)/y, y)|<= Diff*std(y) [Training Set]")
plt.xlabel("Ratio of samples to keep")
plt.ylabel("PRB")
plt.tight_layout()
plt.savefig("./temp/plots/hist_train_prb.png", dpi=600)

plt.figure(figsize=(10,6))
for i,diff in enumerate(diffs):
    plt.plot(keep_percentages, results["val_prb"][i], '--D', color=f"C{i}", alpha=0.7, label=f"Diff={diff}")
plt.grid(True, which="both", color="0.85", linewidth=0.5)  # light gray
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title("Stable Regression with |Cov(f(X)/y, y)|<= Diff*std(y) [Testing Set]")
plt.xlabel("Ratio of samples to keep")
plt.ylabel("PRB")
plt.tight_layout()
plt.savefig("./temp/plots/hist_val_prb.png", dpi=600)
exit()
###############################



from sklearn.linear_model import LinearRegression

model = LinearRegression(fit_intercept=True)
print(X_train.head())
print(y_train_scaled.head())
model.fit(X_train, y_train_scaled)
y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)
y_pred_test = model.predict(X_test)

print("train RMSE: ", root_mean_squared_error(y_train_scaled, y_pred_train) )
ols_mse = root_mean_squared_error(y_train_scaled, y_pred_train)**2

print(r"$R^2$ in Train: ",r2_score(y_pred_train, y_train_scaled))
print(r"$R^2$ in Val: ",r2_score(y_pred_val, y_val_scaled))
print(r"$R^2$ in Test: ",r2_score(y_pred_test, y_test_scaled))

# print("-"*100)
# print("Residual = (y - X'b)")
# print("Max residual Train: ", np.max(y_train_scaled - y_pred_train))
# print("Min residual Train: ", np.min(y_train_scaled - y_pred_train))
# print("Avg residual Train: ", np.mean(y_train_scaled - y_pred_train))
# print("Median residual Train: ", np.median(y_train_scaled - y_pred_train))
# print("-"*100)
# print("Max residual Val: ", np.max(y_val_scaled - y_pred_val))
# print("Min residual Val: ", np.min(y_val_scaled - y_pred_val))
# print("Avg residual Val: ", np.mean(y_val_scaled - y_pred_val))
# print("Median residual Val: ", np.median(y_val_scaled - y_pred_val))


# Convert the train to binary
if source == "CCAO":
    if model_name in ["logistic", "svm"]:
        y_avg = np.mean(y_train_log) # same avg for all splits
        y_train_scaled = (y_train_log >= y_avg) + 0.0 
        y_val_scaled = (y_val_log >= y_avg) + 0.0 
        y_test_scaled = (y_test_log >= y_avg) + 0.0 
        if model_name == "svm": # T: [0,1] -> [-1,1]
            y_train_scaled = 2 * y_train_scaled - 1
            y_val_scaled = 2 * y_val_scaled - 1
            y_test_scaled = 2 * y_test_scaled - 1 


# Logistic Regression
print("-"*100)
print(f"{model_name.upper()} REGRESSION ")

model = MyGLMRegression(fit_intercept=True, solver="MOSEK", model_name=model_name, l2_lambda=0, eps=1e-4)
print(model)
print(X_train.shape)

model.fit(X_train, y_train_scaled)
y_pred_train_scaled = model.predict(X_train)
print(y_pred_train_scaled.size)
print(y_train_scaled[:10].to_numpy())
print(y_pred_train_scaled[:10])
# print(root_mean_squared_error(y_train_scaled, y_pred_train_scaled))
y_pred_val_scaled = model.predict(X_val)
y_pred_test_scaled = model.predict(X_test)
if model_name in ["poisson", "linear"]:
    print("Train Accuracy R2: ", r2_score(y_train_scaled, y_pred_train_scaled))
    print("Val Accuracy R2: ",  r2_score(y_val_scaled, y_pred_val_scaled))
    print("Test Accuracy R2: ", r2_score(y_test_scaled, y_pred_test_scaled))

    # SENSITIVE FEATURE ERROR
    if source != "CCAO":
        print(X_train.shape)
        for cat in df_train[sensitive_name].unique():
            mask = np.where(df_train[sensitive_name] == cat)[0]
            X_cat, y_cat = X_train.iloc[mask, :], y_train_scaled.iloc[mask].to_numpy()
            y_cat_pred = y_pred_train_scaled[mask]
            print(f"Size of {sensitive_name}={cat}: ", y_cat.size)#* X_cat['passthrough__fnlwgt'].sum())
            print(f"Accuracy of {sensitive_name}={cat}: ", r2_score(y_cat, y_cat_pred))
            # print(f"Accuracy of {sensitive_name}={cat}: ", root_mean_squared_error(y_train_scaled, y_pred_train_scaled))
elif model_name in ["logistic"]:
    print("Train Accuracy (1-MAE): ", np.mean(y_train_scaled == np.round(y_pred_train_scaled)))#1-mean_absolute_error(y_train_scaled, y_pred_train_scaled))
    print("Val Accuracy (1-MAE): ",  np.mean(y_val_scaled == np.round(y_pred_val_scaled)))
    print("Test Accuracy (1-MAE): ", 1-mean_absolute_error(y_test_scaled, np.round(y_pred_test_scaled)))

    # SENSITIVE FEATURE ERROR
    if source != "CCAO":
        for cat in df[sensitive_name].unique():
            y_train_cat = y_train_scaled.loc[df_train[sensitive_name] == cat]
            y_train_pred_cat = y_pred_train_scaled[df_train[sensitive_name] == cat]
            X_cat, y_cat = X_train.loc[df_train[sensitive_name] == cat, :], y_train_cat
            print(f"Size of {sensitive_name}={cat}: ", y_train_cat.size)#* X_cat['passthrough__fnlwgt'].sum())
            acc = y_train_cat == np.round(y_train_pred_cat)
            print(f"Accuracy of {sensitive_name}={cat}: ", np.mean(acc))


    #         # Borrar
    #         model.fit(X_cat, y_cat)
    #         y_cat_pred = model.predict(X_cat)
    #         # print(f"Size of {sensitive_name}={cat}: ", y_train_cat.size)
            # print(f"Post Accuracy of {sensitive_name}={cat}: ", np.mean(y_cat == np.round(y_cat_pred)))

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
    percentages = np.linspace(0, 0.087, 15)#linear abalon: 0.063, 3) #poisson with abalon: 0.0833, 3) #logistic with adult: 0.04, 3) 
    model_name = model_name#"poisson"
    l2_lambda = 0#1e-4
    # percentages = np.linspace(0, .02, 21)
    loss_names = {"linear":"MSE/2", "logistic":"Binary Cross Entropy", "poisson":"Poisson Deviance", "svm":"Hinge Loss"}
    accuracy_names = {"linear":"R^2 Score", "logistic":"Accuracy", "poisson":"R^2 Score", "svm":"Accuracy"}

    if source == "CCAO":
        sensitive_mapping = {i:i for i in range(NUM_GROUPS)}

    # --- Data Storage ---
    train_results_list = []
    val_results_list = []
    pof_list = []
    pof_lb_list = []
    pof_ub_list = []
    pof_exp_lb_list = []
    pof_exp_ub_list = []
    pof_taylor_list = []
    fei_list = []
    fairness_list = []
    bregman_divs_dict = None

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


        # model = GroupDeviationConstrainedLinearRegression(
        model = GroupDeviationConstrainedLogisticRegression(
            fit_intercept=True,# add_rmse_constraint=False, 
            percentage_increase=rmse_percentage_increase,
            # max_row_norm_scaling=(1+rmse_percentage_increase),
            solver="MOSEK", # Your solver of choice
            l2_lambda=l2_lambda,
            objective="mse",
            constraint="max_mse",
            model_name=model_name,
            eps=1e-4,
        )
        # Robust version
        # l1_, l2_ = 1e-1, 5e3#1e2*10**(10*rmse_percentage_increase) # 1e-1/10**(10*rmse_percentage_increase), 1e-1*10**(10*rmse_percentage_increase))
        # print("l1: ", l1_," | l2: ", l2_)
        # model = StableRegression(
        #     fit_intercept=True,
        #     keep=(1-0.02),#(1-rmse_percentage_increase),
        #     solver="MOSEK",
        #     lambda_l1=l1_,#1e-1/10**(10*rmse_percentage_increase),
        #     lambda_l2=l2_,#1e-1*10**(10*rmse_percentage_increase),
        #     objective="mae",
        # )

        # Train R2: 0.7954 | Val R2: 0.7876 -- l1:  1.778279410038923e-06  | l2:  5623.413251903491

        # Target transformation for a given model
        if source == "CCAO":
            if model_name == "linear":
                y_train_scaled = y_train_log
            elif model_name in ["logistic", "svm"]:
                y_train_scaled = ( y_train_log - y_train_log.min() ) / ( y_train_log.max() - y_train_log.min() )
                if model_name == "svm":
                    y_train_scaled = 2 * np.round(y_train_scaled) - 1
            elif model_name == "poisson":
                print("y max: ", y_train.max())
                print("y min: ", y_train.min())
                y_train_scaled = y_train // 100000 + 1
                print("y max: ", y_train_scaled.max())
                print("y min: ", y_train_scaled.min())
            result, solve_time, pof, fei, pof_exp_lb, pof_exp_ub, pof_taylor, pof_lb, pof_ub, fairness, metrics_output, group_sizes = model.fit(X_train, y_train_scaled, sensitive_feature=y_train_log, sensitive_nature="continuous")
        else:

            result, solve_time, pof, fei, pof_exp_lb, pof_exp_ub, pof_taylor, pof_lb, pof_ub, fairness, metrics_output, group_sizes = model.fit(X_train, y_train_scaled, sensitive_feature=df_train[sensitive_name], sensitive_nature="discrete")
        
        

        # Saving the metrics for this iteration
        print(metrics_output)
        if bregman_divs_dict is None:
            bregman_divs_dict = {i:[] for i in range(model.n_groups)}
            accuracies_dict = {i:[] for i in range(model.n_groups)}
        bregman_divs = metrics_output[0]
        accuracies = metrics_output[1]
        for key,val in bregman_divs.items():
            bregman_divs_dict[key].append(val)
            accuracies_dict[key].append(accuracies[key])
        # Fit model and make predictions
        # result, solve_time, pof, fei, pof_lb, pof_ub, fairness = model.fit(X_train, y_train_log)
        
        pof_list.append(pof)
        pof_lb_list.append(pof_lb)
        pof_ub_list.append(pof_ub)
        pof_exp_lb_list.append(pof_exp_lb)
        pof_exp_ub_list.append(pof_exp_ub)
        pof_taylor_list.append(pof_taylor)
        fei_list.append(fei)
        fairness_list.append(fairness)
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)

        # Calculate and store statistics
        train_stats = calculate_detailed_statistics(y_train_scaled, y_pred_train, num_groups=NUM_GROUPS)
        train_stats['percentage'] = rmse_percentage_increase
        train_results_list.append(train_stats)
        
        val_stats = calculate_detailed_statistics(y_val_scaled, y_pred_val, num_groups=NUM_GROUPS)
        val_stats['percentage'] = rmse_percentage_increase
        val_results_list.append(val_stats)

        print(f"Train R2: {train_stats['r2']:.4f} | Val R2: {val_stats['r2']:.4f}")
        # print(f"(True) Train R2: {r2_score(y_train, np.exp(y_pred_train)):.4f} | Val R2: {r2_score(y_val, np.exp(y_pred_val)):.4f}")
        print(f"Train RMSE: {train_stats['rmse']:.4f} | Val RMSE: {val_stats['rmse']:.4f}")
        print(f"Train Max Abs Deviation: {train_stats['max_abs_deviation']:.4f} | Val Max Abs Deviation: {val_stats['max_abs_deviation']:.4f}")
        print("-" * 100 + "\n")

        # Summary
        # print(analyze_fairness_by_value(y_pred_train, y_train).round(3).T.to_csv(f"outputs/reports/motivation/summary_train_{rmse_percentage_increase:.4f}.csv",  float_format='%.3f'))
        # print(analyze_fairness_by_value(y_pred_val, y_val).round(3).T.to_csv(f"outputs/reports/motivation/summary_val_{rmse_percentage_increase:.4f}.csv",  float_format='%.3f'))
    


    # --- Temporal Plot ---
    print(os.getcwd())
    plt.figure(figsize=(6,5))
    plt.grid(True, color='gray', linestyle='-', linewidth=0.8, alpha=0.3)
    plt.plot(percentages*100, 100*np.array(pof_taylor_list), "--x", label="POF Taylor (lin. + quad.)", color="lightgreen", alpha=1)
    plt.plot(percentages*100, 100*np.array(pof_list), "--o", label="POF", color="blue")
    plt.plot(percentages*100, 100*np.array(pof_lb_list), "--x", label="POF LB (lin. + quad.)", color="red", alpha=0.5)
    plt.plot(percentages*100, 100*np.array(pof_ub_list), "--x", label="POF UB (quad. + quad.)", color="black", alpha=0.5)
    plt.plot(percentages*100, 100*np.array(pof_exp_lb_list), "--x", label="POF LB (exp.) [1-D opt.]", color="red", alpha=0.9)
    plt.plot(percentages*100, 100*np.array(pof_exp_ub_list), "--x", label="POF UB (exp.) [1-D opt.]", color="red", alpha=0.9)
    # plt.xlabel("Fairness Threshold Improvement (%)")
    plt.xlabel("(Un)Fairness decrease %")
    plt.ylabel("Price of Fairness (% MSE increase)")
    plt.legend()
    # plt.savefig("/img/motivation/tradoff_analysis/pof/pof_vs_percentages.jpg", dpi=300)
    plt.savefig("./img/motivation/pof_vs_percentages.jpg", dpi=300, bbox_inches='tight')

    plt.figure(figsize=(6,5))
    plt.grid(True, color='gray', linestyle='-', linewidth=0.8, alpha=0.3)
    for key in bregman_divs_dict:
        plt.plot(percentages*100, bregman_divs_dict[key], "--o", label=f"g={sensitive_mapping[key]} ({group_sizes[key]} samples)", markerfacecolor='none') 
        plt.ylabel(loss_names[model_name])
        plt.xlabel("(Un)Fairness decrease %")
    plt.legend()
    plt.savefig("./img/motivation/bregman_vs_percentage.jpg", dpi=1200, bbox_inches='tight')

    plt.figure(figsize=(6,5))
    plt.grid(True, color='gray', linestyle='-', linewidth=0.8, alpha=0.3)
    for key in bregman_divs_dict:
        plt.plot(percentages*100, accuracies_dict[key], "--o", label=f"g={sensitive_mapping[key]} ({group_sizes[key]} samples)", markerfacecolor='none') 
        plt.ylabel(accuracy_names[model_name])
        plt.xlabel("(Un)Fairness decrease %")
    plt.legend()
    plt.savefig("./img/motivation/accuracies_vs_percentage.jpg", dpi=1200, bbox_inches='tight')
    # plt.figure(figsize=(6,5))
    # plt.plot(fairness_list, np.array(pof_list) * ols_mse + ols_mse, "--o", label="MSE", color="blue")
    # plt.plot(fairness_list, np.array(pof_lb_list) * ols_mse + ols_mse, "--x", label="MSE LB", color="red", alpha=0.5)
    # plt.plot(fairness_list, np.array(pof_ub_list) * ols_mse + ols_mse, "--x", label="MSE UB", color="red", alpha=0.5)
    # plt.plot(fairness_list, np.array(pof_taylor_list) * ols_mse + ols_mse, "--x", label="POF Taylor Approx.", color="lightgreen", alpha=0.5)
    # # plt.xlabel("Fairness Threshold Improvement (%)")
    # plt.xlabel("(Un)Fairness Measure")
    # plt.ylabel("Mean Squared Error")
    # plt.legend()
    # # plt.savefig("/img/motivation/tradoff_analysis/pof/pof_vs_percentages.jpg", dpi=300)
    # plt.savefig("./img/motivation/mse_vs_percentages.jpg", dpi=300, bbox_inches='tight')

    # plt.plot(percentages, fei_list)

    # # --- Analysis and Plotting ---
    # train_results_df = pd.DataFrame(train_results_list)
    # val_results_df = pd.DataFrame(val_results_list)

    # print(train_results_df)
    # print(val_results_df)

    # # Re-map columns for plotting function compatibility
    # train_results_df.columns = train_results_df.columns.str.replace('overall_', '')
    # val_results_df.columns = val_results_df.columns.str.replace('overall_', '')

    # print("Generating plots for training data...")
    # plot_tradeoff_analysis(train_results_df, percentages, num_groups=NUM_GROUPS, save_dir="img/motivation/tradeoff_analysis/train")
    
    # print("\nGenerating plots for validation data...")
    # plot_tradeoff_analysis(val_results_df, percentages, num_groups=NUM_GROUPS, save_dir="img/motivation/tradeoff_analysis/val")


