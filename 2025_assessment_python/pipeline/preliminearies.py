import numpy as np 
import pandas as pd
from typing import Union, List
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
# from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, root_mean_squared_error, r2_score

# from src.preliminary_models import ConstraintBothRegression, ConstraintDeviationRegression, ConstraintGroupsMeanRegression, UpperBoundLossRegression

# K-means
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


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
sample_size = 100000
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

# embeddings_pipeline = 
model_emb_pipeline = build_model_pipeline_supress_onehot( # WARNING: We only changed to this to perform changes on the pipeline
        pred_vars=params['model']['predictor']['all'],
        cat_vars=params['model']['predictor']['categorical'],
        id_vars=params['model']['predictor']['id']
    )

X_train = linear_pipeline.fit_transform(X_train, y_train_log)
X_val = linear_pipeline.transform(X_val)
X_test = linear_pipeline.transform(X_test)

X_train_emb = model_emb_pipeline.fit_transform(X_train, y_train_log)
X_val_emb = model_emb_pipeline.transform(X_val)
X_test_emb = model_emb_pipeline.transform(X_test)
X_train.head()

from fairness_models.nn_fairness_models import FeedForwardNNRegressorWithEmbeddings6

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm  # Required for the Lowess line
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.linear_model import RidgeCV



# # 2. Fit / transform the data for models with embeddings
# X_train_fit_emb = model_emb_pipeline.fit_transform(X_train_prep, y_train_fit_log).drop(columns=params['model']['predictor']['id'], errors="ignore")
# X_val_fit_emb = model_emb_pipeline.transform(X_val_prep).drop(columns=params['model']['predictor']['id'], errors="ignore")
# X_test_fit_emb = model_emb_pipeline.transform(X_test_prep).drop(columns=params['model']['predictor']['id'], errors="ignore")
# # X_train_fit_emb["char_recent_renovation"] = X_train_fit_emb["char_recent_renovation"].astype(bool) # QUESTION: Why is this not in cat_vars?
# # na_columns = X_train_fit_emb.isna().sum()[X_train_fit_emb.isna().sum() > 0].index
# # X_train_fit_emb[na_columns] = X_train_fit_emb[na_columns].fillna(value="unknown")
# cat_cols_emb = [i for i,col in enumerate(X_train_emb.columns) if X_train_emb[col].dtype == object  or X_train_emb[col].dtype == "category"]
pred_vars = [col for col in params['model']['predictor']['all'] if col in X_train_emb.columns] 
large_categories = ['meta_nbhd_code', 'meta_township_code', 'char_class'] + [c for c in pred_vars if c.startswith('loc_school_')]
coord_vars = ["loc_longitude", "loc_latitude"]


emb_params = {
                'learning_rate': 0.000736, 
                'categorical_features': large_categories, 
                'coord_features': coord_vars,
                'batch_size': 2048, 
                'num_epochs': 500, 
                'hidden_sizes': [512, 256, 128], #[1796, 193, 140, 69],
                'fourier_type': 'basic', 
                'patience': 11, 
                'loss': 'mse',#, 'gamma': 1.409,
                'validation_split': 0.15,
                'eps_y':1e-12,
                'use_scaler':True,
                # 'mode':"div",

                'random_state': 42, 
            }


        # categorical_features,
        # coord_features=(),
        # fourier_type="none",
        # fourier_mapping_size=16,
        # fourier_sigma=1.25,
        # hidden_sizes=(256, 256),         # in resnet mode, first entry is working dim
        # dropout=0.1,
        # normalization="layer_norm",       # 'layer_norm' or 'none'
        # mlp_style="resnet",               # 'resnet' (recommended) or 'plain'
        # batch_size=256,
        # learning_rate=1e-3,
        # num_epochs=50,
        # patience=10,
        # 
        # use_scaler=True,
        # loss="mse",           # "mse" or "huber"
        # huber_delta=1.0,
        # alpha=0.0,            # explicit L2 on NN params
        # rho=0.0,              # covariance penalty weight
        # mode="diff",          # "diff" or "div"
        # eps_y=1e-6,
        # random_state=0,

# ==========================================
# 2. The Path Searcher (Metrics & Loop)
# ==========================================
class FairnessPathSearch:
    """
    Loops over rho values, fits the model, and calculates comprehensive metrics.
    """
    
    def __init__(self, estimator_class, rhos, alpha=1.0, fit_intercept=True, mode=None):
        self.estimator_class = estimator_class
        self.rhos = rhos
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.path_results_ = None
        self.mode = mode,
        self.models_ = {}

    def _compute_metrics(self, y_true_log, y_pred_log, prefix):
        metrics = {}
        
        # --- Log Space Metrics ---
        resid_log = y_pred_log - y_true_log
        metrics[f'{prefix}_mse_log'] = np.mean(resid_log**2)
        
        # Slope of Residuals vs Price (Vertical Equity Proxy)
        if np.std(y_true_log) > 1e-9:
            slope = np.polyfit(y_true_log, resid_log, 1)[0]
            cov = np.cov(resid_log, y_true_log)[0, 1]
            corr = np.corrcoef(resid_log, y_true_log)[0, 1]
        else:
            slope, cov, corr = 0.0, 0.0, 0.0
            
        metrics[f'{prefix}_slope_log'] = slope
        metrics[f'{prefix}_cov_resid_log'] = cov
        metrics[f'{prefix}_corr_resid_log'] = corr
        
        # --- Real Price Space Metrics ---
        y_true = np.exp(y_true_log)
        y_pred = np.exp(y_pred_log)
        
        # COD (IAAO Standard)
        ratios = y_pred / y_true
        median_ratio = np.median(ratios)
        avg_abs_dev = np.mean(np.abs(ratios - median_ratio))
        cod = (avg_abs_dev / median_ratio) * 100
        
        metrics[f'{prefix}_cod_real'] = cod
        metrics[f'{prefix}_mape_real'] = np.mean(np.abs((y_pred - y_true) / y_true))
        
        return metrics

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        results = []
        print(f"Starting Path Search over {len(self.rhos)} rho values...")
        
        for rho in self.rhos:
            model = self.estimator_class(
                alpha=self.alpha, 
                rho=rho, 
                # fit_intercept=self.fit_intercept,
                mode=self.mode[0],
                **emb_params
            )
            model.fit(X_train, y_train)
            self.models_[rho] = model
            
            row = {'rho': rho}
            
            # Train Stats
            y_pred_train = model.predict(X_train)
            row.update(self._compute_metrics(y_train, y_pred_train, 'train'))
            
            # Val Stats
            if X_val is not None and y_val is not None:
                y_pred_val = model.predict(X_val)
                row.update(self._compute_metrics(y_val, y_pred_val, 'val'))
            
            # row['L2_norm'] = np.sum(model.coef_**2)
            results.append(row)
            
        self.path_results_ = pd.DataFrame(results)
        return self

# ==========================================
# 3. Execution & ALL Plots
# ==========================================
if __name__ == "__main__":

    do_path_search = True
    # # --- A. Generate Synthetic Log-Normal Data ---
    # np.random.seed(42)
    # N = 1000
    # X = np.random.rand(N, 5) 
    
    # # Truth: Biased structure
    # true_log_price = 11.0 + 2.0 * X[:, 0] + 0.5 * X[:, 1]
    # y_log = true_log_price + np.random.normal(0, 0.3, N)

    # # Split
    # split = int(0.8 * N)
    # X_train, y_train_log = X[:split], y_log[:s    plit]
    # X_val, y_val_log = X[split:], y_log[split:]

    # --- B. Find Best Alpha & Run Search ---
    rcv = RidgeCV(alphas=np.logspace(-6, 6, 10)).fit(X_train, y_train_log)
    best_alpha = rcv.alpha_ * 1e-2
    print(f"Best Alpha found: {best_alpha:.5f}")

    if do_path_search:
        # rhos_to_search = np.linspace(1e2, 1e4, 20)#[0, 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
        rhos_to_search = np.logspace(-3, 3.5, 20)

        searcher = FairnessPathSearch(
            estimator_class=FeedForwardNNRegressorWithEmbeddings6,
            rhos=rhos_to_search,
            alpha=best_alpha,
            fit_intercept=True,
            mode="div",
        )
        searcher.fit(X_train_emb, y_train_log, X_val=X_val_emb, y_val=y_val_log)
        df = searcher.path_results_

        # --- C. Print Table ---
        print("\n--- Validation Results ---")
        print(df[['rho', 'val_mse_log', 'val_slope_log', 'val_cod_real']].to_string(float_format="%.4f"))

        # =======================================================
        # PLOT SET 1: Evolution of Metrics (Blue=MSE, Red=Slope)
        # =======================================================
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # --- Subplot 1: TRAIN ---
        ax_train = axes[0]
        ln1 = ax_train.semilogx(df['rho'], df['train_mse_log'], 'b-o', lw=2, label='Train MSE')
        ax_train.set_xlabel('Fairness Penalty (Rho)')
        ax_train.set_ylabel('MSE (Log Space)', color='b', fontweight='bold')
        ax_train.tick_params(axis='y', labelcolor='b')
        ax_train.set_title('TRAIN Set: Accuracy vs Fairness')
        ax_train.grid(True, alpha=0.3)

        ax_train_twin = ax_train.twinx()
        ln2 = ax_train_twin.semilogx(df['rho'], df['train_slope_log'].abs(), 'r--s', lw=2, label='|Residual Slope|')
        ax_train_twin.set_ylabel('|Slope of Residuals| (Inequity)', color='r', fontweight='bold')
        ax_train_twin.tick_params(axis='y', labelcolor='r')
        
        # Combined Legend
        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        ax_train.legend(lns, labs, loc='center right')

        # --- Subplot 2: VALIDATION ---
        ax_val = axes[1]
        ln3 = ax_val.semilogx(df['rho'], df['val_mse_log'], 'b-o', lw=2, label='Val MSE')
        ax_val.set_xlabel('Fairness Penalty (Rho)')
        ax_val.set_ylabel('MSE (Log Space)', color='b', fontweight='bold')
        ax_val.tick_params(axis='y', labelcolor='b')
        ax_val.set_title('VALIDATION Set: Accuracy vs Fairness')
        ax_val.grid(True, alpha=0.3)

        ax_val_twin = ax_val.twinx()
        ln4 = ax_val_twin.semilogx(df['rho'], df['val_slope_log'].abs(), 'r--s', lw=2, label='|Residual Slope|')
        ax_val_twin.set_ylabel('|Slope of Residuals| (Inequity)', color='r', fontweight='bold')
        ax_val_twin.tick_params(axis='y', labelcolor='r')
        
        # Combined Legend
        lns2 = ln3 + ln4
        labs2 = [l.get_label() for l in lns2]
        ax_val.legend(lns2, labs2, loc='center right')

        plt.tight_layout()
        plt.savefig("./temp/preliminaries/evolution.pdf", dpi=600)
        plt.show()

        # =======================================================
        # PLOT SET 2: Pareto Frontier & COD
        # =======================================================
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        # Plot 1: Trade-off (MSE vs Slope)
        sc = axes[0].scatter(
            df['val_mse_log'], 
            df['val_slope_log'].abs(), 
            c=np.log1p(df['rho']), cmap='viridis', s=80, edgecolors='k'
        )
        plt.colorbar(sc, ax=axes[0], label='Log(Rho+1)')
        axes[0].set_xlabel('Log MSE (Accuracy Loss)')
        axes[0].set_ylabel('|Slope of Residuals| (Inequity)')
        axes[0].set_title('Pareto Frontier (Validation)')
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Real Price Metrics (COD vs Rho)
        axes[1].semilogx(df['rho'], df['val_cod_real'], 'b-o', label='COD (Real Price)')
        axes[1].axhline(15.0, color='red', linestyle='--', label='IAAO Limit (15.0)')
        # axes[1].axhline(5.0, color='green', linestyle='--', label='IAAO Lower Limit (5.0)')
        axes[1].set_xlabel('Rho')
        axes[1].set_ylabel('COD Score')
        axes[1].set_title('IAAO Standard Compliance (COD)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("./temp/preliminaries/iaao.pdf", dpi=600)
        plt.show()

    # =======================================================
    # PLOT SET 3: Model Comparison Scatters (Lowess)
    # =======================================================
    print("\nGenerating Model Comparison Scatter Plots...")

    models_to_plot = [
        FeedForwardNNRegressorWithEmbeddings6(alpha=best_alpha, rho=0, mode="div", **emb_params),
        FeedForwardNNRegressorWithEmbeddings6(alpha=best_alpha, rho=5e2, mode="div", **emb_params),
        FeedForwardNNRegressorWithEmbeddings6(alpha=best_alpha, rho=1e3, mode="div", **emb_params),
        FeedForwardNNRegressorWithEmbeddings6(alpha=best_alpha, rho=2e3, mode="div", **emb_params),
    ]

    for model_ in models_to_plot:
        # 1. Fit & Predict
        model_.fit(X_train_emb, y_train_log)
        # ================================================
        # TRAINING SET
        # ================================================
        y_pred_log = model_.predict(X_train_emb)
        
        # Transform to Real Money for Ratio Calculation
        y_pred_money = np.exp(y_pred_log)
        y_train_money = np.exp(y_train_log)
        ratios = y_pred_money / y_train_money

        # 2. Setup Plot
        plt.figure(figsize=(6, 4))
        
        # Scatter points
        plt.scatter(y_train_log, ratios, 
            facecolors='none', 
            edgecolors='black', 
            s=50, 
            alpha=0.4,
            label='Properties'
        )
        
        # --- Gray Grid Lines ---
        plt.grid(True, which='major', axis='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.minorticks_on()
        plt.grid(True, which='minor', axis='both', color='lightgray', linestyle=':', linewidth=0.5, alpha=0.5)

        # --- Reference Line (Perfect Equity) ---
        plt.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Perfect Equity (1.0)')

        # --- Tendency Line (Lowess Smoothing) ---
        lowess = sm.nonparametric.lowess(ratios, y_train_log, frac=0.4)
        plt.plot(lowess[:, 0], lowess[:, 1], color='blue', linewidth=3, label='Trend (Lowess)')
        
        # --- Linear Trend (Slope Check) ---
        z = np.polyfit(y_train_log, ratios, 1)
        p = np.poly1d(z)
        plt.plot(y_train_log, p(y_train_log), "g-", alpha=0.6, linewidth=1.5, label=f'Linear Slope={z[0]:.4f}')

        # Formatting
        plt.ylabel("Assessment Ratio (AV / MV)")
        plt.xlabel("Log Market Value")
        plt.title(f"Vertical Equity Check\nrho={model_.rho:.1f} | Lowess should be flat")
        plt.legend(loc='upper right')
        plt.ylim(0, 3) 
        plt.savefig(f"./temp/preliminaries/scatter_train_{model_}.pdf", dpi=600)
        plt.show()

        print("DONE!!")

        # ================================================
        # VALIDATION SET
        # ================================================
        y_pred_log = model_.predict(X_val_emb)
        
        # Transform to Real Money for Ratio Calculation
        y_pred_money = np.exp(y_pred_log)
        y_val_money = np.exp(y_val_log)
        ratios = y_pred_money / y_val_money

        # 2. Setup Plot
        plt.figure(figsize=(6, 4))
        
        # Scatter points
        plt.scatter(y_val_log, ratios, 
            facecolors='none', 
            edgecolors='black', 
            s=50, 
            alpha=0.4,
            label='Properties'
        )
        
        # --- Gray Grid Lines ---
        plt.grid(True, which='major', axis='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.minorticks_on()
        plt.grid(True, which='minor', axis='both', color='lightgray', linestyle=':', linewidth=0.5, alpha=0.5)

        # --- Reference Line (Perfect Equity) ---
        plt.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Perfect Equity (1.0)')

        # --- Tendency Line (Lowess Smoothing) ---
        lowess = sm.nonparametric.lowess(ratios, y_val_log, frac=0.4)
        plt.plot(lowess[:, 0], lowess[:, 1], color='blue', linewidth=3, label='Trend (Lowess)')
        
        # --- Linear Trend (Slope Check) ---
        z = np.polyfit(y_val_log, ratios, 1)
        p = np.poly1d(z)
        plt.plot(y_val_log, p(y_val_log), "g-", alpha=0.6, linewidth=1.5, label=f'Linear Slope={z[0]:.4f}')

        # Formatting
        plt.ylabel("Assessment Ratio (AV / MV)")
        plt.xlabel("Log Market Value")
        plt.title(f"Vertical Equity Check\nrho={model_.rho:.1f} | Lowess should be flat")
        plt.legend(loc='upper right')
        plt.ylim(0, 3) 
        plt.savefig(f"./temp/preliminaries/scatter_val_{model_}.pdf", dpi=600)
        plt.show()

        print("DONE!!")


#     --- Validation Results ---
#           rho  val_mse_log  val_slope_log  val_cod_real
# 0      0.0001       0.1107        -0.2488       26.0308
# 1      0.0003       0.1041        -0.2349       26.0458
# 2      0.0007       0.1022        -0.2385       25.6792
# 3      0.0018       0.1019        -0.2330       25.7077
# 4      0.0048       0.1013        -0.2382       25.6028
# 5      0.0127       0.1017        -0.2363       25.8280
# 6      0.0336       0.1012        -0.2277       25.1928
# 7      0.0886       0.1141        -0.2115       25.4438
# 8      0.2336       0.1025        -0.2344       25.8325
# 9      0.6158       0.1021        -0.2274       25.5989
# 10     1.6238       0.1060        -0.2425       26.2230
# 11     4.2813       0.1029        -0.2396       25.9003
# 12    11.2884       0.1101        -0.2535       26.3442
# 13    29.7635       0.1025        -0.2044       25.6895
# 14    78.4760       0.1054        -0.2206       26.0965
# 15   206.9138       0.1278        -0.1862       25.4244
# 16   545.5595       0.1042        -0.1343       25.3163
# 17  1438.4499       0.1099        -0.1076       25.4771
# 18  3792.6902       0.1408        -0.1584       25.9343
# 19 10000.0000       0.2274        -0.2142       29.4823



# alpha * 5e1

# --- Validation Results ---
#           rho  val_mse_log  val_slope_log  val_cod_real
# 0      0.0001       0.1021        -0.2128       25.2678
# 1      0.0003       0.1131        -0.2210       25.7830
# 2      0.0007       0.1222        -0.2297       26.4640
# 3      0.0018       0.1092        -0.2535       26.9470
# 4      0.0048       0.1161        -0.2247       26.8259
# 5      0.0127       0.1121        -0.2311       26.8344
# 6      0.0336       0.1115        -0.2840       27.6180
# 7      0.0886       0.1206        -0.2717       27.8132
# 8      0.2336       0.1130        -0.2480       26.5099
# 9      0.6158       0.1170        -0.2730       28.1864
# 10     1.6238       0.1102        -0.2282       26.6806
# 11     4.2813       0.1446        -0.2787       27.9724
# 12    11.2884       0.1068        -0.2508       26.5099
# 13    29.7635       0.1013        -0.2204       25.2282
# 14    78.4760       0.1173        -0.2399       26.6014
# 15   206.9138       0.1050        -0.1944       24.9844
# 16   545.5595       0.1171        -0.0959       26.7875
# 17  1438.4499       0.1398        -0.0178       26.8775
# 18  3792.6902       0.1715        -0.0102       29.4279
# 19 10000.0000       0.2629        -0.0431       31.8035



# -- Validation Results ---
#           rho  val_mse_log  val_slope_log  val_cod_real
# 0      0.0000       0.1085        -0.2481       26.8352
# 1      0.0000       0.1183        -0.2510       27.4372
# 2      0.0000       0.1078        -0.2382       26.8292
# 3      0.0000       0.1072        -0.2591       26.8763
# 4      0.0000       0.1144        -0.2573       26.8904
# 5      0.0001       0.1070        -0.2354       26.4801
# 6      0.0001       0.1075        -0.2392       26.8659
# 7      0.0003       0.1099        -0.2432       26.5828
# 8      0.0006       0.1127        -0.2504       26.7296
# 9      0.0013       0.1118        -0.2427       26.5668
# 10     0.0028       0.1079        -0.2627       27.0889
# 11     0.0062       0.1075        -0.2444       26.7972
# 12     0.0137       0.1105        -0.2445       26.9901
# 13     0.0304       0.1066        -0.2482       26.4526
# 14     0.0672       0.1083        -0.2591       26.9610
# 15     0.1487       0.1033        -0.2351       26.0762
# 16     0.3290       0.1158        -0.2694       27.5709
# 17     0.7279       0.1041        -0.2334       26.0625
# 18     1.6103       0.1109        -0.2581       27.0282
# 19     3.5622       0.1218        -0.2622       27.0872
# 20     7.8805       0.1076        -0.2489       26.5817
# 21    17.4333       0.1165        -0.2834       27.5698
# 22    38.5662       0.1069        -0.2384       26.4700
# 23    85.3168       0.1046        -0.2001       25.4925
# 24   188.7392       0.1187        -0.2047       26.5381
# 25   417.5319       0.1084        -0.1688       26.2359
# 26   923.6709       0.1057        -0.1178       24.8321
# 27  2043.3597       0.1212        -0.0750       26.1651
# 28  4520.3537       0.1551        -0.0694       27.7098
# 29 10000.0000       0.2609        -0.1801       29.5953