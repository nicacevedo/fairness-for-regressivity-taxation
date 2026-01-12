#%% Imports
import numpy as np 
import pandas as pd
from typing import Union, List
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
import scipy.sparse as sp
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
from sklearn.cluster import KMeans, MiniBatchKMeans


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
from fairness_models.mixture_boosting_fairness_models import * #MoELGBSmoothPenalty

# My boosting models
import lightgbm as lgb
# from fairness_models.boosting_fairness_models import custom_objective, custom_eval
# from fairness_models.boosting_fairness_models import make_constrained_mse_objective, make_covariance_metric
from fairness_models.boosting_fairness_models import LGBCustomObjective,  LGBPrimalDual#, FairGBMCustomObjective
from fairness_models.boosting_fairness_models import LGBSmoothPenalty, LGBPrimalDualImproved, LGBCovPenalty, LGBMomentPenalty, LGBCovDispPenalty#, LGBCovTweediePenalty, LGBCorrTweediePenalty # post primal-dual methods


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
    print(f"working with a sample ({sample_size//1000}k)")
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


###################################################
# Motivation analysis from here and upwards
###################################################

# Basic clustering for exploratory plots (train on X_train, assign to X_val)
n_clusters = 3
cluster_seed = seed
split_by_cluster = False
if sp.issparse(X_train):
    cluster_scaler = StandardScaler(with_mean=False)
    X_train_cluster = cluster_scaler.fit_transform(X_train)
    X_val_cluster = cluster_scaler.transform(X_val)
    cluster_model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=cluster_seed,
        n_init=20,
        batch_size=1024,
    )
    cluster_labels_train = cluster_model.fit_predict(X_train_cluster)
    cluster_labels_val = cluster_model.predict(X_val_cluster)
else:
    cluster_scaler = StandardScaler()
    X_train_cluster = cluster_scaler.fit_transform(X_train)
    X_val_cluster = cluster_scaler.transform(X_val)
    cluster_model = KMeans(n_clusters=n_clusters, random_state=cluster_seed, n_init=10)
cluster_labels_train = cluster_model.fit_predict(X_train_cluster)
cluster_labels_val = cluster_model.predict(X_val_cluster)
cluster_cmap = plt.cm.get_cmap("tab10", n_clusters)


from fairness_models.optimal_trees_models import OptimalTreeClassifier

OptimalTreeClassifier(
    
)

