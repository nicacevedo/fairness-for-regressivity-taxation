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


from src_.motivation_utils import compute_taxation_metrics

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
sample_size = 1000000
if sample_size < df_train.shape[0]:
    df_train = df_train.sample(min(sample_size, df_train.shape[0]), random_state=42, replace=False)
else:
    sample_size = df_train.shape[0]
df_train.sort_values(by="meta_sale_date", ascending=True, inplace=True)

# Train - val split
train_prop = 0.8622 # almost exact match of 2021 // 2022, for 10k sample
df_val = df_train.iloc[int(train_prop*sample_size):,:]
df_train = df_train.iloc[:int(train_prop*sample_size),:]
# df_train['meta_sale_date']





print(
    df_train["meta_sale_date"].min(),
    df_train["meta_sale_date"].max(),
    df_train.shape[0],
    # df_val.loc[df_val["meta_sale_date"].dt.year == 2021].shape[0],
    # df_train.loc[df_train["meta_sale_date"].dt.year == 2022].shape[0]
)
print(
    df_val["meta_sale_date"].min(),
    df_val["meta_sale_date"].max(),
    df_val.shape[0],
    # df_test.loc[df_test["meta_sale_date"].dt.year == 2022].shape[0],
    # df_val.loc[df_val["meta_sale_date"].dt.year == 2023].shape[0]
)
print(
    df_test["meta_sale_date"].min(),
    df_test["meta_sale_date"].max(),
    df_test.shape[0],
    # df_test.loc[df_test["meta_sale_date"].dt.year == 2022].shape[0],
    # df_test.loc[df_test["meta_sale_date"].dt.year == 2023].shape[0]
)

exit()

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


                        # # Plot the wieghts of surrogate one

                        # random_state = 42
                        # n_jobs =190
                        # max_iter=200#100#200#500 #0


                        # fit_intercept = True
                        # l1,l2 = 1e-3, 1e-2 #5e-1 # l1 = 1e-3
                        # num_leaves = 31#31
                        # max_depth = 15 #5
                        # lr = 1e-1

                        # lgbm_params = {
                        #     "boosting_type": "gbdt",
                        #     "num_leaves": 31,
                        #     "max_depth": max_depth,
                        #     # "num_leaves":  2**(max_depth)//8, # must be at most 2^max_depth 
                        #     "learning_rate": lr,
                        #     "n_estimators": max_iter,
                        #     "subsample_for_bin": 200000,
                        #     "objective": "mse", # To be updated inside
                        #     "class_weight": None,
                        #     "min_child_samples": 30,
                        #     "colsample_bytree": 1.0,
                        #     "reg_alpha": l1,
                        #     "reg_lambda": l2,
                        #     "random_state": random_state,
                        #     "n_jobs": 1,#n_jobs,
                        #     "importance_type": "split",
                        # }


                        # import lightgbm as lgb
                        # from fairness_models.boosting_fairness_models import LGBSmoothPenalty, LGBCovPenalty
                        # # model = lgb.LGBMRegressor(**lgbm_params)
                        # # model = LGBSmoothPenalty(rho=2, ratio_mode="diff", zero_grad_tol=1e-12, eps_y=1e-12, lgbm_params=lgbm_params)
                        # model = LGBCovPenalty(rho=1487, ratio_mode="div", zero_grad_tol=1e-12, eps_y=1e-12, lgbm_params=lgbm_params)
                        # model.fit(X_train, y_train_log)
                        # y_pred_log = model.predict(X_val)
                        # residuals = y_pred_log - y_val_log
                        # ratios = np.exp(y_pred_log) / y_val

                        # y_plot = np.abs(y_val_log - np.mean(y_val_log))
                        # x_plot = y_val_log
                        # # plt.plot(x_plot, y_plot**2, 'o')
                        # # plt.savefig("temp/plots/weights/weights_1.png")
                        # # plt.close()

                        # plt.plot(x_plot, y_plot**2/x_plot**2, 'o')
                        # plt.savefig("temp/plots/weights/weights.png")
                        # plt.close()

                        # plt.plot(x_plot, residuals**2*y_plot**2/x_plot**2, 'o')
                        # plt.savefig("temp/plots/weights/residual_weights.png")
                        # plt.close()

                        # plt.plot(x_plot, residuals**2, 'o')
                        # plt.savefig("temp/plots/weights/residuals_sqrd.png")
                        # plt.close()

                        # # from sklearn.ensemble import IsolationForest
                        # # # X = [[-1.1], [0.3], [0.5], [100]]
                        # # isolation_args = dict(n_estimators=100, max_samples='auto', contamination='auto', max_features=1.0, bootstrap=False, n_jobs=None, random_state=None, verbose=0, warm_start=False)

                        # # clf = IsolationForest(random_state=0).fit(X_val)
                        # # color_values = clf.score_samples(X_val)#clf.predict(X_val)
                        # # plt.scatter(x_plot, residuals**2, c=color_values, cmap='viridis') # 'viridis' is a common colormap
                        # # plt.savefig("temp/plots/weights/residual_outliers.png")
                        # # plt.close()

                        # import numpy as np
                        # from scipy.stats import skew

                        # import numpy as np
                        # import pandas as pd
                        # import matplotlib.pyplot as plt
                        # import seaborn as sns
                        # from scipy.stats import skew

                        # def binned_stats(x, y, n_bins, bin_type="quantile"):
                        #     """
                        #     Compute mean, std, skew, and kurtosis of y in bins of x.
                        #     """
                        #     df = pd.DataFrame({'x': x, 'y': y})
                            
                        #     if bin_type == "quantile":
                        #         df['bin'] = pd.qcut(df['x'], q=n_bins, duplicates='drop')
                        #     elif bin_type == "uniform":
                        #         df['bin'] = pd.cut(df['x'], bins=n_bins)
                        #     else:
                        #         raise ValueError("bin_type must be 'quantile' or 'uniform'")

                        #     # Aggregation functions handling NaNs
                        #     def safe_skew(series):
                        #         return skew(series, nan_policy='omit')
                            
                        #     def safe_kurtosis(series):
                        #         # Fisher=True means Normal dist has kurtosis = 0
                        #         return kurtosis(series, fisher=True, nan_policy='omit')

                        #     stats = df.groupby('bin', observed=False)['y'].agg(
                        #         count='count',
                        #         mean='mean',
                        #         std='std',
                        #         skew=safe_skew,
                        #         kurt=safe_kurtosis
                        #     ).reset_index()

                        #     # Extract bin centers/edges and CAST TO FLOAT to fix the plotting error
                        #     stats['x_min'] = stats['bin'].apply(lambda i: i.left).astype(float)
                        #     stats['x_max'] = stats['bin'].apply(lambda i: i.right).astype(float)
                        #     stats['x_center'] = stats['bin'].apply(lambda i: i.mid).astype(float)

                        #     return stats[['x_min', 'x_max', 'x_center', 'count', 'mean', 'std', 'skew', 'kurt']]

                        # def plot_binned_stats(stats_df, title="Binned Statistics Summary"):
                        #     sns.set_theme(style="whitegrid")
                            
                        #     # Switch to 3 rows: Main (2x height), Skew (1x), Kurtosis (1x)
                        #     fig, (ax1, ax2, ax3) = plt.subplots(
                        #         nrows=3, 
                        #         ncols=1, 
                        #         figsize=(10, 10), 
                        #         sharex=True, 
                        #         gridspec_kw={'height_ratios': [2, 1, 1]}
                        #     )

                        #     x = stats_df['x_center']
                            
                        #     # --- Row 1: Mean, Std, Counts ---
                        #     ax1.plot(x, stats_df['mean'], color='#1f77b4', lw=2, label='Mean')
                        #     ax1.fill_between(
                        #         x, 
                        #         stats_df['mean'] - stats_df['std'], 
                        #         stats_df['mean'] + stats_df['std'], 
                        #         color='#1f77b4', 
                        #         alpha=0.2, 
                        #         label='Mean ± 1 Std'
                        #     )
                            
                        #     # Ghost Bar Chart for Counts
                        #     ax1_count = ax1.twinx()
                        #     bar_width = (stats_df['x_max'] - stats_df['x_min']) * 0.8
                        #     ax1_count.bar(x, stats_df['count'], color='gray', alpha=0.15, width=bar_width, label='Count')
                        #     ax1_count.set_ylabel('Count', color='gray')
                        #     ax1_count.grid(False)
                            
                        #     ax1.set_ylabel('Mean $\pm$ Std')
                        #     ax1.set_title(title, fontsize=14, pad=15)
                        #     ax1.legend(loc='upper left')

                        #     # --- Row 2: Skewness ---
                        #     ax2.axhline(0, color='black', ls='--', lw=1, alpha=0.5)
                        #     ax2.plot(x, stats_df['skew'], color='#d62728', marker='o', ms=4, label='Skewness')
                        #     ax2.set_ylabel('Skewness')
                        #     ax2.legend(loc='upper left')

                        #     # --- Row 3: Kurtosis ---
                        #     # Kurtosis > 0 means "heavier tails than normal"
                        #     ax3.axhline(0, color='black', ls='--', lw=1, alpha=0.5)
                        #     ax3.plot(x, stats_df['kurt'], color='#2ca02c', marker='s', ms=4, label='Kurtosis (Fisher)')
                        #     ax3.set_ylabel('Kurtosis')
                        #     ax3.set_xlabel('X Value (Bin Center)')
                        #     ax3.legend(loc='upper left')

                        #     plt.tight_layout()
                        #     # plt.show()
                        #     plt.savefig("temp/plots/weights/bin_evolution.png")
                        #     # plt.show()

                        # # --- Example Usage ---

                        # # 2. Run the function
                        # n_bins = 50
                        # # df_summary = binned_stats(y_val_log.to_numpy(), ratios.to_numpy(), n_bins=n_bins, bin_type='quantile')
                        # df_summary = binned_stats(y_val_log.to_numpy(), residuals.to_numpy(), n_bins=n_bins, bin_type='quantile')

                        # # 3. Display DataFrame
                        # print("Summary DataFrame (First 5 bins):")
                        # print(df_summary.head(n_bins))

                        # # 4. Plot
                        # plot_binned_stats(df_summary, title="Evolution of Y Metrics by X Bins")


                        # # print("Stats")
                        # # results = binned_stats(y_val_log, residuals, bins=10, bin_type="quantile")

                        # # print("Stats 2")
                        # # print(binned_stats(y_val_log, ratios, bins=10, bin_type="quantile"))

                        # exit()



# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import statsmodels.api as sm  # Required for the Lowess line
# from sklearn.base import BaseEstimator, RegressorMixin
# from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
# from sklearn.linear_model import RidgeCV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from scipy.stats import spearmanr
from fairness_models.boosting_fairness_models import LGBCovPenalty
import lightgbm as lgb

class FairnessPathSearch:
    """
    Loops over rho values, fits the model, and calculates real-price space metrics:
    1. RMSE (Real Price)
    2. COD (Dispersion of Ratios)
    3. Corr(Ratio, Price) (Vertical Equity Proxy)
    """
    
    def __init__(self, estimator_class, rhos, alpha=1.0, fit_intercept=True, mode=None):
        self.estimator_class = estimator_class
        self.rhos = rhos
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.path_results_ = None
        self.mode = mode
        self.models_ = {}

    def _compute_metrics(self, y_true_log, y_pred_log, prefix):
        metrics = {}
        
        # --- Conversion to Real Price Space ---
        y_true = np.exp(y_true_log)
        y_pred = np.exp(y_pred_log)
        ratios = y_pred / y_true
        
        # 1. RMSE in Real Price (Dollar Space)
        mse_real = np.mean((y_pred - y_true)**2)
        metrics[f'{prefix}_rmse_real'] = np.sqrt(mse_real)
        
        # 2. COD (Coefficient of Dispersion) - Standard IAAO Metric
        median_ratio = np.median(ratios)
        avg_abs_dev = np.mean(np.abs(ratios - median_ratio))
        metrics[f'{prefix}_cod_real'] = (avg_abs_dev / median_ratio) * 100
        
        # 3. Vertical Equity Proxy: Correlation(Assessment Ratio, Real Price)
        # Positive correlation suggests Progressivity, Negative suggests Regressivity.
        if np.std(ratios) > 1e-12 and np.std(y_true) > 1e-12:
            corr_val = np.corrcoef(ratios, y_true)[0, 1]
        else:
            corr_val = 0.0
        metrics[f'{prefix}_corr_ratio_price'] = corr_val
        
        # Keep log MSE for internal reference if needed
        metrics[f'{prefix}_mse_log'] = np.mean((y_pred_log - y_true_log)**2)
        
        return metrics

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        results = []
        print(f"Starting Path Search over {len(self.rhos)} rho values...")
        
        # Shared Hyperparameters
        lgbm_params = {
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "max_depth": 15,
            "learning_rate": 0.1,
            "n_estimators": 200,
            "reg_alpha": 1e-3,
            "reg_lambda": 1e-2,
            "random_state": 42,
            "n_jobs": 1,
            "objective": "mse",
        }

        for rho in self.rhos:
            model = LGBCovPenalty(
                rho=rho, 
                ratio_mode="diff", 
                anchor_mode="target", 
                zero_grad_tol=1e-12, 
                eps_y=1e-12, 
                lgbm_params=lgbm_params
            )
            
            model.fit(X_train, y_train)
            self.models_[rho] = model
            
            row = {'rho': rho}
            y_pred_train = model.predict(X_train)
            row.update(self._compute_metrics(y_train, y_pred_train, 'train'))
            
            if X_val is not None and y_val is not None:
                y_pred_val = model.predict(X_val)
                row.update(self._compute_metrics(y_val, y_pred_val, 'val'))
            
            results.append(row)
            
        self.path_results_ = pd.DataFrame(results)
        return self


def plot_pareto_frontier(df):
    """
    Visualizes the Fairness-Accuracy Tradeoff (Pareto Frontier).
    X-axis: |Correlation| (Fairness/Equity)
    Y-axis: RMSE (Accuracy)
    Color: Rho (Penalty strength)
    """

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    
    for i, (prefix, title) in enumerate([('train', 'Train Set'), ('val', 'Validation Set')]):
        ax = axes[i]
        
        # We plot Accuracy vs Inequity (closer to origin is better)
        # Swapped: x is correlation, y is RMSE
        x = df[f'{prefix}_corr_ratio_price'].abs()
        y = df[f'{prefix}_rmse_real']
        c = np.log10(df['rho'] + 1e-10) # Log scale for better color distribution

  
        
        scatter = ax.scatter(x, y, c=c, cmap='viridis', s=60, edgecolors='black', alpha=0.8)
        ax.plot(x, y, color='gray', linestyle='--', alpha=0.4) # Path line
        
        # Annotate first and last points
        ax.annotate(f"Init (ρ={df['rho'].min():.1e})", (x.iloc[0], y.iloc[0]), xytext=(5,5), textcoords='offset points', fontsize=8)
        ax.annotate(f"End (ρ={df['rho'].max():.1e})", (x.iloc[-1], y.iloc[-1]), xytext=(5,5), textcoords='offset points', fontsize=8)
        
        # ax.set_title(f"Pareto Frontier: {title}")
        ax.set_xlabel("|Corr(Ratio, Price)| (Vetical Inequity)")
        if i == 0: ax.set_ylabel("RMSE (Real Price)")
        ax.grid(True, alpha=0.5)
        # ax_train.grid(True, alpha=0.3)
        
    # Single Colorbar
    cbar = fig.colorbar(scatter, ax=axes.ravel().tolist())
    cbar.set_label('Log10(Rho)')
    
    plt.tight_layout()
    plt.savefig("temp/slides/tradeoffs/Pareto_Frontier_6.pdf")
    plt.show()

if __name__ == "__main__":
    # Assuming X_train, y_train_log, X_val, y_val_log are defined...
    
    rhos_to_search = np.logspace(-3, 1.7, 40) 
    searcher = FairnessPathSearch(None, rhos=rhos_to_search)
    searcher.fit(X_train, y_train_log, X_val, y_val_log)
    df = searcher.path_results_

    # --- Helper Functions ---
    def normalize(series):
        if series.max() == series.min(): return series * 0.0
        return (series - series.min()) / (series.max() - series.min())

    def get_label(name, original_series):
        baseline = original_series.iloc[0]
        val_min, val_max = original_series.min(), original_series.max()
        if baseline != 0:
            pct_min = ((val_min - baseline) / baseline) * 100
            pct_max = ((val_max - baseline) / baseline) * 100
        else:
            pct_min, pct_max = 0.0, 0.0
        return f"{name}\nRange vs Init: [{pct_min:.1f}%, {pct_max:.1f}%]"

    # Define the new columns to process
    cols_to_plot = [
        ('train_rmse_real', 'norm_train_rmse'),
        ('train_corr_ratio_price', 'norm_train_corr'),
        ('train_cod_real', 'norm_train_cod'),
        ('val_rmse_real', 'norm_val_rmse'),
        ('val_corr_ratio_price', 'norm_val_corr'),
        ('val_cod_real', 'norm_val_cod')
    ]

    for col_orig, col_norm in cols_to_plot:
        # For Correlation, we normalize the absolute value because we want it close to 0
        if 'corr' in col_orig:
            df[col_norm] = normalize(df[col_orig].abs())
        else:
            df[col_norm] = normalize(df[col_orig])

    # --- Plotting Evolution ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sets = [('train', 'TRAIN Set'), ('val', 'VALIDATION Set')]

    for i, (prefix, title) in enumerate(sets):
        ax = axes[i]
        ax.plot(df['rho'], df[f'norm_{prefix}_rmse'], 'b-o', label=get_label('RMSE (Real)', df[f'{prefix}_rmse_real']))
        ax.plot(df['rho'], df[f'norm_{prefix}_corr'], 'r--s', label=get_label('|Corr(Ratio, Price)|', df[f'{prefix}_corr_ratio_price'].abs()))
        ax.plot(df['rho'], df[f'norm_{prefix}_cod'], 'g-.D', label=get_label('COD', df[f'{prefix}_cod_real']))
        
        ax.set_xscale('log')
        ax.set_xlabel('Fairness Penalty (Rho)')
        ax.set_ylabel('Normalized Metric [0-1]')
        # ax.set_title(title)
        ax.grid(True, alpha=0.5)
        ax.legend(ncol=1, fontsize='small') # loc='lower left', bbox_to_anchor=(0, 1.02),

    plt.tight_layout()
    plt.savefig("temp/slides/tradeoffs/LGBM_evolution_6.pdf")
    plt.show()

    # --- New Pareto Frontier Plot ---
    plot_pareto_frontier(df)

    # --- Summary Tables ---
    summary_data = []
    metric_map = [
        ('Train', 'RMSE (Real)', 'train_rmse_real', 'norm_train_rmse'),
        ('Train', '|Corr(Ratio, P)|', 'train_corr_ratio_price', 'norm_train_corr'),
        ('Train', 'COD', 'train_cod_real', 'norm_train_cod'),
        ('Valid', 'RMSE (Real)', 'val_rmse_real', 'norm_val_rmse'),
        ('Valid', '|Corr(Ratio, P)|', 'val_corr_ratio_price', 'norm_val_corr'),
        ('Valid', 'COD', 'val_cod_real', 'norm_val_cod')
    ]

    for set_label, metric_label, col_orig, col_norm in metric_map:
        # 1. Establish Baseline (First row)
        baseline_val = df[col_orig].iloc[0]
        if 'corr' in col_orig: baseline_val = abs(baseline_val)
        
        # 2. Determine Best/Worst based on Abs for Fairness
        data_series = df[col_orig].abs() if 'corr' in col_orig else df[col_orig]
        idx_min, idx_max = data_series.idxmin(), data_series.idxmax()

        for idx, t_type in [(idx_min, 'Min (Best)'), (idx_max, 'Max (Worst)')]:
            current_val = df.loc[idx, col_orig]
            # Use abs for pct calculation on fairness metrics to match legend logic
            comp_val = abs(current_val) if 'corr' in col_orig else current_val
            pct_diff = ((comp_val - baseline_val) / baseline_val * 100) if baseline_val != 0 else 0.0
            
            summary_data.append({
                'Set': set_label, 
                'Metric': metric_label, 
                'Type': t_type,
                'Value': current_val, 
                'Normalized': df.loc[idx, col_norm], 
                '% Change vs Init': pct_diff,
                'Rho': df.loc[idx, 'rho'],
            })

    df_summary = pd.DataFrame(summary_data)
    for s_type in ['Train', 'Valid']:
        print(f"\n{'='*25} {s_type.upper()} SUMMARY {'='*25}")
        print(df_summary[df_summary['Set'] == s_type].drop(columns=['Set']).to_string(index=False, float_format="%.4f"))
    exit()






    # =======================================================
    # PLOT SET 1: Evolution of Metrics (Blue=MSE, Red=Slope)
    # =======================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Subplot 1: TRAIN ---
    ax_train = axes[0]
    ln1 = ax_train.semilogx(df['rho'], df['train_mse_log'], 'b-o', lw=2, label='Train MSE')
    ax_train.set_xlabel('Fairness Penalty (Rho)')
    # ax_train.set_ylabel('MSE (Log Space)', color='b', fontweight='bold')
    ax_train.tick_params(axis='y', labelcolor='b')
    ax_train.set_title('TRAIN Set: Accuracy vs Fairness')
    ax_train.grid(True, alpha=0.3)

    ax_train_twin = ax_train.twinx()
    ln2 = ax_train_twin.semilogx(df['rho'], df['train_slope_log'].abs(), 'r--s', lw=2, label='|Residual Slope|')
    # ax_train_twin.set_ylabel('|Slope of Residuals| (Inequity)', color='r', fontweight='bold')
    ax_train_twin.tick_params(axis='y', labelcolor='r')

    ax_train_twin = ax_train.twinx()
    ln2 = ax_train_twin.semilogx(df['rho'], df['train_cod_real'].abs(), 'g--D', lw=2, label='Train COD')
    # ax_train_twin.set_ylabel('|Slope of Residuals| (Inequity)', color='g', fontweight='bold')
    ax_train_twin.tick_params(axis='y', labelcolor='g')
    
    # Combined Legend
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax_train.legend(lns, labs, loc='center right')

    # --- Subplot 2: VALIDATION ---
    ax_val = axes[1]
    ln3 = ax_val.semilogx(df['rho'], df['val_mse_log'], 'b-o', lw=2, label='Val MSE')
    ax_val.set_xlabel('Fairness Penalty (Rho)')
    # ax_val.set_ylabel('MSE (Log Space)', color='b', fontweight='bold')
    ax_val.tick_params(axis='y', labelcolor='b')
    ax_val.set_title('VALIDATION Set: Accuracy vs Fairness')
    ax_val.grid(True, alpha=0.3)

    ax_val_twin = ax_val.twinx()
    ln4 = ax_val_twin.semilogx(df['rho'], df['val_slope_log'].abs(), 'r--s', lw=2, label='|Residual Slope|')
    # ax_val_twin.set_ylabel('|Slope of Residuals| (Inequity)', color='r', fontweight='bold')
    ax_val_twin.tick_params(axis='y', labelcolor='r')

    ax_val_twin = ax_val.twinx()
    ln4 = ax_val_twin.semilogx(df['rho'], df['val_cod_real'].abs(), 'g--s', lw=2, label='Val COD')
    # ax_val_twin.set_ylabel('|Slope of Residuals| (Inequity)', color='r', fontweight='bold')
    ax_val_twin.tick_params(axis='y', labelcolor='g')
    
    # Combined Legend
    lns2 = ln3 + ln4
    labs2 = [l.get_label() for l in lns2]
    ax_val.legend(lns2, labs2, loc='center right')

    plt.tight_layout()
    plt.ylabel("Metric Value")
    plt.savefig("temp/slides/tradeoffs/LGBM_evolution.pdf")
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
    plt.show()

    # =======================================================
    # PLOT SET 3: Model Comparison Scatters (Lowess)
    # =======================================================
    print("\nGenerating Model Comparison Scatter Plots...")

    models_to_plot = [
        # FairnessConstrainedRidgeLog(alpha=best_alpha, rho=0, fit_intercept=True, mode="div"),
        # FairnessConstrainedRidgeLog(alpha=best_alpha, rho=20, fit_intercept=True, mode="div"),
    ]


    # --- 1. Robust K-Means Clustering Setup ---
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    # A. Prepare data: Stack X and Y into a matrix
    # X_cluster = np.column_stack((y_val_log, ratios))

    # B. Standardize: Essential so 'Log Value' doesn't dominate 'Ratio' due to scale differences
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    X_scaled_val = scaler.fit_transform(X_val)

    # C. Robust K-Means: 
    # n_init=50 runs the algo 50 times with different seeds and picks the best inertia 
    # (This satisfies "multiple runs and majority result" logic mathematically)
    K_CLUSTERS = 3  # You can adjust this number
    kmeans = KMeans(n_clusters=K_CLUSTERS, init='k-means++', n_init=50, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    cluster_labels = kmeans.predict(X_scaled_val)
    # --- 1. Robust K-Means Clustering Setup ---


    for model_ in models_to_plot:
        # 1. Fit & Predict
        model_.fit(X_train, y_train_log)
        y_pred_log = model_.predict(X_val)
        
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
        plt.show()

        display(compute_taxation_metrics(y_val_log, y_pred_log, scale="log"))








        # --- 2. Setup Plot (Mirroring your workflow) ---
        plt.figure(figsize=(6, 4))

        # Scatter points (Colored by Cluster)
        # We use a colormap (viridis) and map 'c' to the labels
        scatter = plt.scatter(y_val_log, ratios, 
                    c=cluster_labels, 
                    cmap='viridis', 
                    edgecolors='black', 
                    linewidth=0.5,
                    s=50, 
                    alpha=0.6, # Slightly more opaque to see colors
                    label='Clustered Properties'
        )

        # --- Gray Grid Lines ---
        plt.grid(True, which='major', axis='both', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.minorticks_on()
        plt.grid(True, which='minor', axis='both', color='lightgray', linestyle=':', linewidth=0.5, alpha=0.5)

        # --- Reference Line (Perfect Equity) ---
        plt.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Perfect Equity (1.0)')

        # --- Tendency Line (Lowess Smoothing on ALL data) ---
        # We still calculate this on the whole dataset to see the global trend
        lowess = sm.nonparametric.lowess(ratios, y_val_log, frac=0.4)
        plt.plot(lowess[:, 0], lowess[:, 1], color='blue', linewidth=3, label='Trend (Lowess)')

        # --- Linear Trend (Slope Check on ALL data) ---
        z = np.polyfit(y_val_log, ratios, 1)
        p = np.poly1d(z)
        plt.plot(y_val_log, p(y_val_log), "g-", alpha=0.6, linewidth=1.5, label=f'Linear Slope={z[0]:.4f}')

        # Formatting
        plt.ylabel("Assessment Ratio (AV / MV)")
        plt.xlabel("Log Market Value")
        plt.title(f"Vertical Equity (k={K_CLUSTERS} Clusters)\nrho={model_.rho:.1f} | Colored by Robust K-Means")

        # Legend Handling
        # We want the lines, but we also might want a legend for the clusters. 
        # This gathers the lines + the scatter handle
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles, labels, loc='upper right')

        plt.ylim(0, 3) 
        plt.show()

        display(compute_taxation_metrics(y_val_log, y_pred_log, scale="log"))