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
from src_.motivation_utils import *#analyze_fairness_by_value, calculate_detailed_statistics, plot_tradeoff_analysis, compute_taxation_metrics
from fairness_models.linear_fairness_models import * #LeastAbsoluteDeviationRegression, MaxDeviationConstrainedLinearRegression, LeastMaxDeviationRegression, GroupDeviationConstrainedLinearRegression, StableRegression, LeastProportionalDeviationRegression#LeastMSEConstrainedRegression, LeastProportionalDeviationRegression
# from fairness_models.linear_fairness_models import MyGLMRegression, GroupDeviationConstrainedLogisticRegression, RobustStableLADPRDCODRegressor, StableAdversarialSurrogateRegressor, StableAdversarialSurrogateRegressor2
from fairness_models.mixture_boosting_fairness_models import * #MoELGBSmoothPenalty

# My boosting models
import lightgbm as lgb
# from fairness_models.boosting_fairness_models import custom_objective, custom_eval
# from fairness_models.boosting_fairness_models import make_constrained_mse_objective, make_covariance_metric
# from fairness_models.boosting_fairness_models import LGBCustomObjective,  LGBPrimalDual#, FairGBMCustomObjective
# from fairness_models.boosting_fairness_models import LGBSmoothPenalty, LGBPrimalDualImproved, LGBCovPenalty, LGBMomentPenalty, LGBCovDispPenalty#, LGBCovTweediePenalty, LGBCorrTweediePenalty # post primal-dual methods
from fairness_models.boosting_fairness_models import *

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
sample_size = 300000 # 10k samples for Abalon (?)# 1000 samples for Adult (?)
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
train_prop = 0.822871 # just copyng from above #0.8622 # almost exact match of 2021 // 2022, for 10k sample
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
# same code as motivation_analysis.py from here and upwards
###################################################

###################################################
# same code as motivation_analysis.py from here and upwards
###################################################

# os.environ["OMP_NUM_THREADS"] = "16"
# os.environ["MKL_NUM_THREADS"] = "16"
# os.environ["OPENBLAS_NUM_THREADS"] = "16"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "16"
# os.environ["NUMEXPR_NUM_THREADS"] = "16"

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.cluster import KMeans, MiniBatchKMeans
# from sklearn.preprocessing import StandardScaler
# import scipy.sparse as sp
# import os
# from joblib import Parallel, delayed

# def analyze_kmeans_extended(X_train, X_val, feature_names=None, max_k=10, seed=42, n_init=10, 
#                             scale_data=False, dummy_penalty_weight=1.0):
#     """
#     Computes Elbow curve and Feature Importance for K-Means clustering with optional
#     penalty weighting for dummy variables.
    
#     Args:
#         X_train: Training data (dense or sparse).
#         X_val: Validation data.
#         feature_names: List of feature names.
#         max_k: Maximum number of clusters to test.
#         seed: Random seed.
#         n_init: Number of random initializations.
#         scale_data (bool): If True, applies StandardScaler ONLY to numerical columns.
#         dummy_penalty_weight (float): 
#              - 1.0: Treat dummy and numerical variables equally (Default).
#              - < 1.0: Penalize/Down-weight dummy variables (focus clustering on numericals).
#              - > 1.0: Up-weight dummy variables.
             
#              This minimizes: Inertia_num + (dummy_penalty_weight * Inertia_dummy)
#     """
    
#     # Ensure directory exists for saving plots
#     os.makedirs("./temp/eda/kmeans/", exist_ok=True)

#     # -------------------------------------------------------
#     # 0. Helper: Identify Dummy Columns
#     # -------------------------------------------------------
#     def get_dummy_indices(X):
#         """
#         Robustly identifies 0/1, True/False columns.
#         """
#         dummy_indices = []
        
#         # Helper to check if a set of values is effectively binary {0, 1}
#         # Python set equality handles 0 == 0.0 == False
#         def is_binary_set(values):
#             try:
#                 # Get uniques, filtering out NaNs
#                 uniques = np.unique(values)
#                 uniques = uniques[~pd.isnull(uniques)]
                
#                 # If empty or too many unique values, not a dummy
#                 if len(uniques) == 0: return False # Empty/All-NaN
#                 if len(uniques) > 2: return False
                
#                 # Check if all values are in {0, 1}
#                 # This works for [0, 1], [0.0, 1.0], [False, True], [0], [1]
#                 return set(uniques).issubset({0, 1})
#             except Exception:
#                 return False

#         if sp.issparse(X):
#             # Efficient check for sparse matrices
#             X_csc = X.tocsc()
#             for i in range(X.shape[1]):
#                 col_data = X_csc.data[X_csc.indptr[i]:X_csc.indptr[i+1]]
                
#                 # Case 1: Column is empty (all zeros) -> It is a constant 0 dummy
#                 if len(col_data) == 0:
#                     dummy_indices.append(i)
#                     continue
                    
#                 # Case 2: Check stored values.
#                 if is_binary_set(col_data):
#                     dummy_indices.append(i)
                    
#         elif isinstance(X, pd.DataFrame):
#             for i, col in enumerate(X.columns):
#                 # Pass Series data directly
#                 if is_binary_set(X[col]):
#                     dummy_indices.append(i)
#         else:
#             # Numpy array
#             for i in range(X.shape[1]):
#                 if is_binary_set(X[:, i]):
#                     dummy_indices.append(i)
                    
#         return np.array(dummy_indices)

#     # -------------------------------------------------------
#     # 1. Identify indices (BEFORE Scaling)
#     # -------------------------------------------------------
#     dummy_idxs = get_dummy_indices(X_train)
#     all_idxs = np.arange(X_train.shape[1])
#     num_idxs = np.setdiff1d(all_idxs, dummy_idxs)
    
#     print(f"[Weight={dummy_penalty_weight}] Detected {len(dummy_idxs)} dummy features and {len(num_idxs)} numerical features.")

#     if feature_names is None:
#         if hasattr(X_train, 'columns'):
#             feature_names = list(X_train.columns)
#         else:
#             feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
#     else:
#         # Ensure it's a list for indexing later
#         feature_names = list(feature_names)

#     # -------------------------------------------------------
#     # 2. Preprocessing / Scaling Logic (Selective)
#     # -------------------------------------------------------
#     # We maintain separate paths for Sparse vs Dense to ensure efficiency
#     # and correct handling of indices.
    
#     if scale_data and len(num_idxs) > 0:
#         print(f"[Weight={dummy_penalty_weight}] Scaling ONLY numerical data (leaving dummies raw)...")
        
#         if sp.issparse(X_train):
#             # SPARSE MATRIX: Split, Scale Num, Stack
#             scaler = StandardScaler(with_mean=False)
            
#             X_train_num = X_train[:, num_idxs]
#             X_val_num = X_val[:, num_idxs]
            
#             X_train_dummy = X_train[:, dummy_idxs]
#             X_val_dummy = X_val[:, dummy_idxs]
            
#             # Scale numerical part
#             X_train_num = scaler.fit_transform(X_train_num)
#             X_val_num = scaler.transform(X_val_num)
            
#             # Recombine: [Numerical, Dummy]
#             X_train_cluster = sp.hstack([X_train_num, X_train_dummy])
#             X_val_cluster = sp.hstack([X_val_num, X_val_dummy])
            
#             # IMPORTANT: Reordering happened!
#             # New order is [All Numericals, All Dummies]
#             new_feature_names = [feature_names[i] for i in num_idxs] + [feature_names[i] for i in dummy_idxs]
#             feature_names = new_feature_names
            
#             # Update dummy indices to point to the end of the new matrix
#             dummy_idxs = np.arange(len(num_idxs), len(num_idxs) + len(dummy_idxs))
            
#         else:
#             # DENSE (DataFrame or Array): Modify in place (preserving order)
#             X_train_cluster = X_train.copy() if hasattr(X_train, 'copy') else X_train
#             X_val_cluster = X_val.copy() if hasattr(X_val, 'copy') else X_val
            
#             scaler = StandardScaler()
            
#             if isinstance(X_train_cluster, pd.DataFrame):
#                 X_train_cluster.iloc[:, num_idxs] = scaler.fit_transform(X_train_cluster.iloc[:, num_idxs])
#                 X_val_cluster.iloc[:, num_idxs] = scaler.transform(X_val_cluster.iloc[:, num_idxs])
#             else:
#                 X_train_cluster[:, num_idxs] = scaler.fit_transform(X_train_cluster[:, num_idxs])
#                 X_val_cluster[:, num_idxs] = scaler.transform(X_val_cluster[:, num_idxs])
#     else:
#         print(f"[Weight={dummy_penalty_weight}] Using data as-is (skipping internal scaling)...")
#         X_train_cluster = X_train.copy() if hasattr(X_train, 'copy') else X_train
#         X_val_cluster = X_val.copy() if hasattr(X_val, 'copy') else X_val

#     # -------------------------------------------------------
#     # 3. Apply Weighted Penalization
#     # -------------------------------------------------------
#     if dummy_penalty_weight != 1.0 and len(dummy_idxs) > 0:
#         print(f"[Weight={dummy_penalty_weight}] Applying penalty to {len(dummy_idxs)} dummy columns...")
#         # Scaling factor is sqrt(weight) because Inertia is squared distance
#         scale_factor = np.sqrt(dummy_penalty_weight)
        
#         if sp.issparse(X_train_cluster):
#             # Efficient diagonal multiplication for sparse
#             weights = np.ones(X_train_cluster.shape[1])
#             weights[dummy_idxs] = scale_factor
#             D = sp.diags(weights)
#             X_train_cluster = X_train_cluster @ D
#             X_val_cluster = X_val_cluster @ D
#         elif isinstance(X_train_cluster, pd.DataFrame):
#             X_train_cluster.iloc[:, dummy_idxs] *= scale_factor
#             X_val_cluster.iloc[:, dummy_idxs] *= scale_factor
#         else:
#             X_train_cluster[:, dummy_idxs] *= scale_factor
#             X_val_cluster[:, dummy_idxs] *= scale_factor

#     # Helper to create model dynamically
#     def get_model(k):
#         is_sparse = sp.issparse(X_train_cluster)
#         if is_sparse:
#             return MiniBatchKMeans(
#                 n_clusters=k, random_state=seed, n_init=n_init, batch_size=1024
#             )
#         else:
#             return KMeans(n_clusters=k, random_state=seed, n_init=n_init)

#     # -------------------------------------------------------
#     # 4. Elbow Method Loop
#     # -------------------------------------------------------
#     inertias = []
#     k_range = range(1, max_k + 1)
    
#     print(f"[Weight={dummy_penalty_weight}] Computing Elbow Curve for k=1 to {max_k}...")
    
#     for k in k_range:
#         model = get_model(k)
#         model.fit(X_train_cluster)
#         inertias.append(model.inertia_)

#     # Plot Elbow
#     plt.figure(figsize=(10, 5))
#     plt.plot(k_range, inertias, marker='o', linestyle='--')
#     plt.title(f'Elbow Method (Dummy Penalty = {dummy_penalty_weight})')
#     plt.xlabel('Number of clusters')
#     plt.ylabel('Inertia (Weighted)')
#     plt.grid(True)
#     # Use format string for filename to avoid overwrite in parallel execution
#     plt.savefig(f"./temp/eda/kmeans/elbow_method_w{dummy_penalty_weight}.png", dpi=600, bbox_inches='tight')
#     # plt.show() # Commented out for parallel safety
#     plt.close()

#     # -------------------------------------------------------
#     # 5. Fit Final Model
#     # -------------------------------------------------------
#     n_clusters = 3 
#     print(f"[Weight={dummy_penalty_weight}] Fitting final model with n_clusters={n_clusters}...")
#     cluster_model = get_model(n_clusters)
#     cluster_labels_train = cluster_model.fit_predict(X_train_cluster)
    
#     # -------------------------------------------------------
#     # 6. Feature Importance (Centroid Variance)
#     # -------------------------------------------------------
#     centers = cluster_model.cluster_centers_
    
#     # CRITICAL: We must UN-SCALE the centroids to interpret them correctly.
#     # The clustering was done in "Weighted Space", but we want to know
#     # the variance in the "Original Space".
#     if dummy_penalty_weight != 1.0 and len(dummy_idxs) > 0:
#         scale_factor = np.sqrt(dummy_penalty_weight)
#         # Reverse the scaling
#         centers[:, dummy_idxs] /= scale_factor
        
#     feature_variances = np.var(centers, axis=0)
    
#     # Map indices back to names (using potentially updated feature_names list)
#     importance_df = pd.DataFrame({
#         'Feature': feature_names,
#         'Importance (Variance)': feature_variances,
#         'Type': ['Dummy' if i in dummy_idxs else 'Numerical' for i in range(len(feature_names))]
#     }).sort_values(by='Importance (Variance)', ascending=False)

#     print(f"\n[Weight={dummy_penalty_weight}] --- Top 5 Relevant Features ---")
#     print(importance_df.head(5))

#     # -------------------------------------------------------
#     # 7. Visualization
#     # -------------------------------------------------------
#     plt.figure(figsize=(12, 6))
#     sns.barplot(data=importance_df.head(15), x='Importance (Variance)', y='Feature', hue='Type', dodge=False, palette='viridis')
#     plt.title(f'Top 15 Features (Dummy Penalty={dummy_penalty_weight})')
#     plt.xlabel('Variance of Centroids (Original Scale)')
#     plt.savefig(f"./temp/eda/kmeans/top_15_features_w{dummy_penalty_weight}.png", dpi=600, bbox_inches='tight')
#     # plt.show()
#     plt.close()

#     # Heatmap
#     top_features_df = importance_df.head(10)
#     top_features = top_features_df['Feature'].values
    
#     feature_map = {name: i for i, name in enumerate(feature_names)}
#     top_indices = [feature_map[f] for f in top_features]
    
#     centers_subset = centers[:, top_indices]
    
#     plt.figure(figsize=(10, 6))
#     sns.heatmap(centers_subset, annot=True, cmap="RdBu_r", center=0, 
#                 xticklabels=top_features, yticklabels=[f"Cluster {i}" for i in range(n_clusters)])
#     plt.title(f'Centroid Values (Original Scale) - Weight {dummy_penalty_weight}')
#     plt.xlabel('Features')
#     plt.ylabel('Cluster')
#     plt.savefig(f"./temp/eda/kmeans/centroid_zscore_top_features_w{dummy_penalty_weight}.png", dpi=600, bbox_inches='tight')
#     # plt.show()
#     plt.close()
    
#     return cluster_model, cluster_labels_train, importance_df



# Bias correction on regression

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Inputs
random_state = 42
n_jobs =190
max_iter=200#500 #0

fit_intercept = True
l1,l2 = 1e-3, 1e-2
max_depth = 15
lr = 1e-1

# # 4. Sof-penalized LGBM
lgbm_params = {
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "max_depth": max_depth,
    # "num_leaves":  2**(max_depth)//8, # must be at most 2^max_depth 
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

# lgbm_params = {
#     "bagging_fraction": 0.8428234162819122, 
#     "bagging_freq": 1, 
#     "boosting_type": "gbdt",
#     "feature_fraction": 0.6132763726499917,
#     "force_col_wise": True,
#     "lambda_l1": 0.003014864232968839,
#     "lambda_l2": 0.0023105360653011787,
#     "learning_rate": 0.011923049822178568,
#     "max_bin": 511,
#     "max_depth": 8,
#     "min_child_samples": 83,
#     "min_gain_to_split": 0.001958927480050991,
#     "n_estimators": 1000,#10000,
#     "n_jobs": 1,
#     "num_leaves": 147,
#     "objective": "regression", #mse?
#     "subsample_for_bin": 200000,
#     "verbosity": -1,
#     # Mine
#     "random_state": random_state,
# }
  

models = [
    # LinearRegression(fit_intercept=fit_intercept, n_jobs=n_jobs),
    # LeastAbsoluteDeviationRegression(fit_intercept=fit_intercept, solver="MOSEK"),
    # ElasticNet(fit_intercept=fit_intercept, l1_ratio=l1/(l1 + l2), alpha=(l1 + l2), selection="random", random_state=random_state, warm_start=True),
    # RandomForestRegressor(n_estimators=n_jobs, criterion='squared_error', max_depth=max_depth, min_samples_split=50, min_samples_leaf=30, bootstrap=True, n_jobs=n_jobs, random_state=random_state, warm_start=True, ccp_alpha=1e-3),
    # GradientBoostingRegressor(loss='squared_error', learning_rate=1e-3, n_estimators=100, subsample=0.8, criterion='friedman_mse', min_samples_split=50, min_samples_leaf=20, max_depth=3, random_state=random_state, alpha=0.9, warm_start=True, validation_fraction=0.1, tol=1e-4, ccp_alpha=1e-3)
    # HistGradientBoostingRegressor(loss='squared_error', learning_rate=lr, max_iter=max_iter, max_leaf_nodes=31, max_depth=max_depth, min_samples_leaf=30, l2_regularization=l2, max_bins=255, 
    #                               warm_start=True, early_stopping='auto', scoring='loss', validation_fraction=0.2, n_iter_no_change=10, tol=1e-6, random_state=random_state),     
    lgb.LGBMRegressor(**lgbm_params),
    LGBCovPenalty(rho=600, ratio_mode="div", zero_grad_tol=1e-12, eps_y=1e-12, lgbm_params=lgbm_params, verbose=False), # Min error
    LGBCovPenalty(rho=10, ratio_mode="diff", zero_grad_tol=1e-12, eps_y=1e-12, lgbm_params=lgbm_params, verbose=False), # closest to target
    LGBCovPenalty(rho=1900, ratio_mode="div", zero_grad_tol=1e-12, eps_y=1e-12, lgbm_params=lgbm_params, verbose=False),# closest to target
    LGBCovPenalty(rho=1200, ratio_mode="div", zero_grad_tol=1e-12, eps_y=1e-12, lgbm_params=lgbm_params, verbose=False),# closes to ideal (target, min_error)
    LGBCovPenalty(rho=2000, ratio_mode="div", zero_grad_tol=1e-12, eps_y=1e-12, lgbm_params=lgbm_params, verbose=False),# closes to ideal (target, min_error)
]


# for rho_ in range(100, 2101, 100):
#     models.append(
#         LGBSmoothPenalty(rho=rho_/10, ratio_mode="div", zero_grad_tol=1e-12, eps_y=1e-12, lgbm_params=lgbm_params, verbose=False),
#     )

# for rho_ in range(50, 1051, 50):
#     models.append(
#         LGBSmoothPenalty(rho=(rho_-50)/1e3, ratio_mode="diff", zero_grad_tol=1e-12, eps_y=1e-12, lgbm_params=lgbm_params, verbose=False),
#     )

# for rho_ in range(100, 2101, 100):
#     models.append(
#         LGBCovPenalty(rho=rho_, ratio_mode="div", zero_grad_tol=1e-12, eps_y=1e-12, lgbm_params=lgbm_params, verbose=False),
#     )

# for rho_ in range(50, 1051, 50):
#     models.append(
#         LGBCovPenalty(rho=(rho_-50)/1e2, ratio_mode="diff", zero_grad_tol=1e-12, eps_y=1e-12, lgbm_params=lgbm_params, verbose=False),
#     )


records = []
pred_by_model = {}

baseline_label = model_label(models[0])   # your plain LGBM baseline

for model_ in models:
    model_.fit(X_train, y_train_log)

    y_pred_log = model_.predict(X_val)
    y_pred = np.exp(y_pred_log)  # PRICE SCALE

    label = model_label(model_)
    pred_by_model[label] = y_pred

    global_stats, _ = analyze_financial_performance(
        y_val,
        y_pred,
        show_plots=False,
        verbosity=0,  # quiet during the loop
        iaao_property_class="Residential Improved",
    )

    records.append(build_scorecard_records(label, global_stats, mode="no-correction"))

# 1) Scorecard (table comparing 3–5 models)
df_score = make_comparison_scorecard(
    records,
    baseline_model=baseline_label,
    iaao_property_class="Residential Improved",
)

print_comparison_scorecard(df_score)

# 2) Pick finalists (example: best EquityVerdict, then lowest RMSE)
rank = {"PASS": 0, "WARN": 1, "FAIL": 2, "NA": 3}
tmp = df_score.copy()
tmp["_rank"] = tmp["EquityVerdict"].map(rank).fillna(3)
finalists = tmp.sort_values(["_rank", "RMSE"]).head(2)["Model"].tolist()

# 3) Print baseline + finalists full reports (the “financial analysis” you asked for)
print_baseline_and_finalists_reports(
    y_val,
    pred_by_model,
    baseline_key=baseline_label,
    finalist_keys=finalists,
    iaao_property_class="Residential Improved",
    n_quantiles=4,
    show_plots=False,  # set True if you want the plots too
)

# metrics_list = []
# n_quantiles = 4
# for model_ in models:
#     print("="*100)
#     print("Fitting: ", model_)
#     print("="*100)
#     model_.fit(X_train, y_train_log)

#     # Validaiton predict
#     y_pred_log_train = model_.predict(X_train) 
#     y_pred_log = model_.predict(X_val)

#     print("Original stats:")
#     print(y_train.describe())
    
#     # No-correction
#     y_pred = np.exp(y_pred_log)
#     y_pred = pd.Series(y_pred)
#     print("No correction:")
#     # print(y_pred.describe())
#     metrics_ = compute_taxation_metrics(y_val, y_pred.to_numpy(), scale="price")
#     metrics_list+=[{"Model":model_, "Mode":"no-correction"} | metrics_]

#     # print(compute_taxation_metrics(y_val, y_pred.to_numpy(), scale="price"))
#     analyze_financial_performance(y_val, y_pred, show_plots=False, n_quantiles=n_quantiles)
#     # print(f"COD:{cod(y_pred.to_numpy() / y_val):.4f} | PRD:{prd(y_pred.to_numpy() / y_val, y_val):.4f}")

#     # # Smearing
#     # log_residuals = y_train_log - y_pred_log_train 
#     # y_pred = np.exp(y_pred_log) * np.mean(np.exp(log_residuals))
#     # y_pred = pd.Series(y_pred)
#     # print("Smearing:")
#     # # print(y_pred.describe())  
#     # metrics_ = compute_taxation_metrics(y_val, y_pred.to_numpy(), scale="price")
#     # metrics_list+=[{"Model":model_, "Mode":"semaring"} | metrics_]
#     # analyze_financial_performance(y_val, y_pred, show_plots=False, n_quantiles=n_quantiles)


# df_metrics = pd.DataFrame(metrics_list) 
# print(df_metrics)
# df_metrics.to_csv("./temp/tables/eda_metrics.csv")


    
