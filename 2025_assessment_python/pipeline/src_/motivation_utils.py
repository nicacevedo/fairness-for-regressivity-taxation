import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import os
from typing import Union, List

def analyze_fairness_by_value(
    y_pred_log: Union[np.ndarray, pd.Series],
    y_real: Union[np.ndarray, pd.Series],
    tax_rate: float = 1.9/100,
    num_groups: int = 3,
) -> pd.DataFrame:
    """
    Analyzes regression model performance with a focus on fairness across value groups.

    This function is designed to highlight how a model that regresses to the mean
    can be unfair by systematically over-predicting low-value items and
    under-predicting high-value items.

    Args:
        y_pred_log: Model predictions in log scale (e.g., log(price)).
        y_real: The true target values in the original scale (e.g., price).
        tax_rate: The tax rate to be applied to the predicted price (e.g., 0.05 for 5%).
        num_groups: The number of equally spaced bins to cut the real values into.

    Returns:
        A pandas DataFrame containing a detailed summary of performance and fairness
        metrics for the overall dataset and for each value-based subgroup.
    """
    if not isinstance(y_pred_log, pd.Series):
        y_pred_log = pd.Series(y_pred_log, name="y_pred_log")
    else:
        y_pred_log = y_pred_log.reset_index()[0]
    if not isinstance(y_real, pd.Series):
        y_real = pd.Series(y_real, name="y_real")
    else:
        y_real = y_real.reset_index()["meta_sale_price"]
    
        
    # --- 1. Data Preparation ---
    # Convert log predictions back to the original price scale
    y_pred = np.exp(y_pred_log)

    # print("Average of residuals ", np.mean(y_pred - y_real) )

    # Create a base DataFrame for analysis
    df = pd.DataFrame({
        'y_real': y_real,
        'y_real_log': np.log(y_real),
        'y_pred': y_pred,
        'y_pred_log': y_pred_log,
    })
    df.dropna(inplace=True)

    # --- 2. Helper function to compute statistics for any data slice ---
    def _compute_statistics(data_slice: pd.DataFrame) -> dict:
        """Computes all desired metrics for a given subset of data."""
        if data_slice.empty:
            return {}

        # Calculate residuals and tax differences
        residual = data_slice['y_pred'] - data_slice['y_real']
        residual_log = data_slice['y_pred_log'] - data_slice['y_real_log']
        target_deviation = data_slice['y_real'] - data_slice['y_real'].mean()
        target_deviation_log = data_slice['y_real_log'] - data_slice['y_real_log'].mean()
        pct_residual = residual / data_slice['y_real']
        tax_difference = residual * tax_rate

        # Separate overcharged and undercharged samples for specific stats
        overcharged_pct = pct_residual[residual > 0]
        undercharged_pct = pct_residual[residual < 0]

        stats = {
            # --- General & Error Metrics ---
            'count': len(data_slice),
            r'avg real price (\$)': data_slice['y_real'].mean(),
            # 'R2 (log-price)': 1- (residual_log @ residual_log)/(target_deviation_log @ target_deviation_log),
            'R2 (price)': 1- (residual @ residual)/(target_deviation @ target_deviation),
            'rmse': np.sqrt(np.mean(residual**2)),
            # 'mae': np.mean(np.abs(residual)),

            # # --- Residual Analysis ($) ---
            # r'avg residual (\$)': residual.mean(),
            # r'median residual (\$)': residual.median(),
            # r'most overcharged (\$)': residual.max(),
            # r'most undercharged (\$)': residual.min(),

            # --- Residual Analysis (%) ---
            r'highest overcharge (\%)': pct_residual.max() * 100,
            r'highest undercharge (\%)': pct_residual.min() * 100,
            r'avg charge (\%)': pct_residual.mean() * 100,
            r'avg overcharge (\%)': overcharged_pct.mean() * 100 if not overcharged_pct.empty else 0,
            r'avg undercharg (\%)': undercharged_pct.mean() * 100 if not undercharged_pct.empty else 0,


            # --- Fairness & Bias Metrics ---
            r'count overcharged': (residual > 0).sum(),
            r'count undercharged': (residual < 0).sum(),
            # r'total tax diff (\$)': tax_difference.sum(),
            # r'avg tax overpayment (\$)': tax_difference[tax_difference > 0].mean(),
            # r'avg tax underpayment (\$)': tax_difference[tax_difference < 0].mean(),
        }
        return stats

    # --- 3. Perform Analysis ---
    all_stats = []

    # Analyze the entire dataset first
    overall_stats = _compute_statistics(df)
    overall_stats['group'] = 'Overall'
    all_stats.append(overall_stats)

    # Create value-based groups using uniform cuts
    group_labels = [f'Group {i+1}' for i in range(num_groups)]
    try:
        df['value_group'] = pd.cut(
            df['y_real_log'],
            bins=num_groups,
            labels=group_labels,
            include_lowest=True
        )
        has_groups = True
    except ValueError:
        # This can happen if there are not enough unique values to create bins
        print(f"Warning: Could not create {num_groups} bins from y_real. Skipping group analysis.")
        has_groups = False

    # Analyze each group
    if has_groups:
        for group_name in group_labels:
            group_df = df[df['value_group'] == group_name]
            # print("group size" , group_df.shape)
            if not group_df.empty:
                group_stats = _compute_statistics(group_df)
                group_stats['group'] = group_name
                all_stats.append(group_stats)

    # --- 4. Format and Return Results ---
    results_df = pd.DataFrame(all_stats).set_index('group')
    
    # Format for better readability
    pd.options.display.float_format = '{:,.2f}'.format
    format_cols_pct = [col for col in results_df.columns if '%' in col]
    format_cols_money = [col for col in results_df.columns if '$' in col]
    
    for col in format_cols_pct:
        results_df[col] = results_df[col].apply(lambda x: fr"{x:,.2f}\%")
    for col in format_cols_money:
        results_df[col] = results_df[col].apply(lambda x: fr"\${x:,.2f}")
        
    return results_df




# +-------------------------------------------------+
# |          PART 1: STATISTICS CALCULATION         |
# +-------------------------------------------------+

def calculate_detailed_statistics(y_true, y_pred, num_groups=3):
    """
    Calculates a wide range of statistics, both overall and by group.
    Groups are created based on the quantiles of the true target values.

    Args:
        y_true (np.ndarray): The true target values.
        y_pred (np.ndarray): The predicted values from the model.
        num_groups (int): The number of equal-sized groups to split the data into.

    Returns:
        dict: A dictionary containing all calculated statistics.
    """
    stats = {}
    
    # Ensure y_true is a flat array for pd.qcut
    try:
        y_true_flat = y_true.flatten()
    except Exception as e:
        y_true_flat = y_true

    # Create equal-width bins across [min, max] (uniform-size intervals)
    y_min, y_max = np.min(y_true_flat), np.max(y_true_flat)
    if y_min == y_max:
        group_labels = np.zeros_like(y_true_flat, dtype=int)
        actual_num_groups = 1
    else:
        bin_edges = np.linspace(y_min, y_max, num_groups + 1)
        # pd.cut returns NaN for values falling exactly on the rightmost edge; force them into last bin
        group_series = pd.cut(y_true_flat, bins=bin_edges, labels=False, include_lowest=True)
        group_series = pd.Series(group_series).fillna(num_groups - 1).astype(int)
        group_labels = group_series.values
        actual_num_groups = len(np.unique(group_labels))

    
    # --- Overall Statistics ---
    residuals = y_true - y_pred
    # Handle division by zero for percentage deviation
    percent_residuals = np.divide(residuals, y_true, out=np.full_like(residuals, np.nan, dtype=float), where=y_true!=0)

    stats['samples'] = y_true.size
    stats['rmse'] = root_mean_squared_error(y_true, y_pred)
    stats['mae'] = mean_absolute_error(y_true, y_pred)
    stats['r2'] = r2_score(y_true, y_pred)
    stats['max_deviation'] = np.max(residuals)
    stats['min_deviation'] = np.min(residuals)
    stats['median_deviation'] = np.median(residuals)
    stats['max_abs_deviation'] = np.max(np.abs(residuals))
    stats['max_diff_deviations'] = np.max(residuals) - np.min(residuals)
    stats['max_pct_deviation'] = np.nanmax(percent_residuals) * 100
    stats['min_pct_deviation'] = np.nanmin(percent_residuals) * 100
    stats['avg_price'] = np.mean(y_true)

    # --- Grouped Statistics ---
    group_maes = []
    for i in range(actual_num_groups):
        mask = (group_labels == i)
        if np.sum(mask) == 0: continue # Skip empty groups
        
        y_true_group = y_true[mask]
        y_pred_group = y_pred[mask]
        residuals_group = y_true_group - y_pred_group
        percent_residuals_group = np.divide(residuals_group, y_true_group, out=np.full_like(residuals_group, np.nan, dtype=float), where=y_true_group!=0)
        
        stats[f'group_{i}_samples'] = y_true_group.size
        stats[f'group_{i}_rmse'] = root_mean_squared_error(y_true_group, y_pred_group)
        stats[f'group_{i}_mae'] = mean_absolute_error(y_true_group, y_pred_group)
        stats[f'group_{i}_r2'] = r2_score(y_true_group, y_pred_group)
        stats[f'group_{i}_max_deviation'] = np.max(residuals_group)
        stats[f'group_{i}_min_deviation'] = np.min(residuals_group)
        stats[f'group_{i}_median_deviation'] = np.median(residuals_group)
        stats[f'group_{i}_max_pct_deviation'] = np.nanmax(percent_residuals_group) * 100
        stats[f'group_{i}_min_pct_deviation'] = np.nanmin(percent_residuals_group) * 100
        stats[f'group_{i}_avg_price'] = np.mean(y_true_group)
        group_maes.append(stats[f'group_{i}_mae'])
        
    # --- Inter-Group Fairness Metrics ---
    if len(group_maes) > 1:
        stats['fairness_max_abs_diff_group_mae'] = np.max(group_maes) - np.min(group_maes)
    else:
        stats['fairness_max_abs_diff_group_mae'] = 0.0

    return stats


# +-------------------------------------------------+
# |        PART 2: PLOTTING & VISUALIZATION         |
# +-------------------------------------------------+

def plot_tradeoff_analysis(results_df, percentages, num_groups=3, save_dir="img/tradeoff_analysis"):
    """
    Generates and saves a comprehensive set of plots for tradeoff analysis.

    Args:
        results_df (pd.DataFrame): DataFrame containing the collected statistics for each percentage.
        percentages (np.ndarray): The array of percentage increases used in the model.
        num_groups (int): The number of groups used for analysis.
        save_dir (str): The directory where plots will be saved.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    fairness_metrics = {
        'max_abs_deviation': "Max Absolute Deviation |r_max|",
        'max_diff_deviations': "Max Deviation Difference (max(r) - min(r))",
        'fairness_max_abs_diff_group_mae': "Max Abs Diff of Group MAEs"
    }
    
    accuracy_metrics = {'rmse': 'RMSE', 'r2': 'R^2'}

    # --- Plot Type 1 & 3: Dual-Axis Evolution and Direct Trade-off ---
    for acc_key, acc_label in accuracy_metrics.items():
        for fair_key, fair_label in fairness_metrics.items():
            
            # --- Dual-Axis Plot ---
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(percentages, results_df[f'{acc_key}'], "--x", color='tab:red', label=acc_label)
            ax1.set_xlabel("Model Constraint: % Increase")
            ax1.set_ylabel(acc_label, color='tab:red')
            ax1.tick_params(axis='y', labelcolor='tab:red')
            
            ax2 = ax1.twinx()
            ax2.plot(percentages, results_df[f'{fair_key}'], ":x", color='tab:blue', label=fair_label)
            ax2.set_ylabel(fair_label, color='tab:blue')
            ax2.tick_params(axis='y', labelcolor='tab:blue')
            
            fig.tight_layout(rect=[0, 0, 0.9, 1])
            plt.title(f"{acc_label} vs. {fair_label} Evolution")
            plt.savefig(f"{save_dir}/{acc_key}_vs_{fair_key}_evolution.png", dpi=300)
            plt.close(fig)

            # --- Direct Trade-off Scatter Plot ---
            plt.figure(figsize=(8, 6))
            sc = plt.scatter(
                results_df[f'{fair_key}'], results_df[f'{acc_key}'],
                c=percentages, cmap="viridis", s=80, edgecolor="k"
            )
            plt.colorbar(sc, label="Model Constraint: % Increase")
            plt.xlabel(fair_label)
            plt.ylabel(acc_label)
            plt.title(f"Trade-off: {acc_label} vs. {fair_label}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{save_dir}/{acc_key}_vs_{fair_key}_tradeoff.png", dpi=300)
            plt.close()

    # --- Plot Type 2: Group Metric Evolution ---
    group_metrics_to_plot = ['rmse', 'r2', 'mae', 'max_deviation', 'min_deviation']

    for metric in group_metrics_to_plot:
        plt.figure(figsize=(10, 6))
        for i in range(num_groups):
            col_name = f'group_{i}_{metric}'
            avg_price_val = results_df[f"group_{i}_avg_price"].iloc[0]
            samples = results_df[f"group_{i}_samples"].iloc[0]

            avg_label = f" (avg={avg_price_val:.4f})" if avg_price_val is not None else ""
            plt.plot(percentages, results_df[col_name], "--o", label=f'Group {i}{avg_label}|{samples}', alpha=0.5)
        
        plt.xlabel("Model Constraint: % Increase")
        plt.ylabel(f"Group-wise {metric.upper()}")
        plt.title(f"Evolution of Group-wise {metric.upper()}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/group_evolution_{metric}.png", dpi=300)
        plt.close()
        
    print(f"All analysis plots saved to '{save_dir}' directory.")



def compute_taxation_metrics(y_real, y_pred, scale="log"):
        if scale == "log":
            y_real_log, y_pred_log = y_real, y_pred 
            y_real, y_pred = np.exp(y_real), np.exp(y_pred)
        else:
            y_real_log, y_pred_log = np.log(y_real), np.log(y_pred)
        metrics = dict()

        # 1. Accuracy metrics
        metrics["R2 Score"] = r2_score(y_real, y_pred)
        metrics["R2 Score (log)"] = r2_score(y_real_log, y_pred_log)
        metrics["RMSE"] = root_mean_squared_error(y_real, y_pred)
        metrics["MAE"] = mean_absolute_error(y_real, y_pred)
        metrics["MAPE"] = mean_absolute_percentage_error(y_real, y_pred)

        # 2. My metrics of interest
        ratios = y_pred / y_real
        metrics["Corr(ratio, y)"] = np.corrcoef(ratios, y_real)[0,1]
        metrics["Var(ratio)"] = np.var(ratios)

        # 3. Taxation-Domain Specific Metrics
        median_ratio = np.median(ratios)
        metrics["COD"] = 100/median_ratio*np.mean(np.abs(ratios - median_ratio))
        metrics["PRD"] =  np.mean(ratios) / np.sum(y_pred) * np.sum(y_real) #(ratios @ y_real) * np.sum(y_real)
        # PRB: Calculate the "Proxy" value first (Average of Sale Price and "Indicated" Value)
        proxy_vals = 0.5 * (y_pred / median_ratio + y_real)
        median_proxy = np.median(proxy_vals)
        y_prb = (ratios - median_ratio) / median_ratio
        x_prb = np.log2(proxy_vals) - np.log2(median_proxy) 
        metrics["PRB"] = np.polyfit(x_prb, y_prb, 1)[0]
        metrics["MKI"] = mki(y_pred, y_real)
        return metrics


# MKI helpers
def _to_1d_float_array(x):
    """Convert array-like to a 1D float numpy array."""
    a = np.asarray(x, dtype=float).reshape(-1)
    return a

def _gini_like_from_ordered(values):
    """
    Gini formula applied to a *given order* (when order isn't by itself, this is a concentration-style coefficient).
    values must be nonnegative and not all zero.
    """
    v = _to_1d_float_array(values)
    if np.any(v < 0):
        raise ValueError("Values must be nonnegative for Gini-like computation.")
    n = v.size
    s = v.sum()
    if n == 0 or s <= 0:
        return np.nan
    i = np.arange(1, n + 1, dtype=float)
    return float((2.0 * np.dot(i, v)) / (n * s) - (n + 1.0) / n)

def mki(estimate, sale_price, *, dropna=True):
    """
    Modified Kakwani Index (MKI).

    Steps (per AssessPy docs):
      1) order observations by sale_price ascending
      2) compute Gini(sale_price) in that order (this is the usual Gini)
      3) compute "Gini" of estimates while *remaining ordered by sale_price*
         (this is effectively a concentration coefficient)
      4) MKI = Gini_estimate / Gini_sale_price

    Interpretation (per docs): MKI < 1 regressive, =1 vertical equity, >1 progressive. :contentReference[oaicite:3]{index=3}
    """
    est = _to_1d_float_array(estimate)
    sale = _to_1d_float_array(sale_price)

    if est.shape[0] != sale.shape[0]:
        raise ValueError("estimate and sale_price must have the same length.")
    if dropna:
        mask = np.isfinite(est) & np.isfinite(sale)
        est, sale = est[mask], sale[mask]
    if est.size == 0:
        return np.nan
    if np.any(sale < 0) or np.any(est < 0):
        raise ValueError("estimate and sale_price should be nonnegative for MKI.")

    idx = np.argsort(sale, kind="mergesort")
    sale_ord = sale[idx]
    est_ord_by_sale = est[idx]

    g_sale = _gini_like_from_ordered(sale_ord)
    g_est_by_sale = _gini_like_from_ordered(est_ord_by_sale)

    if not np.isfinite(g_sale) or g_sale == 0:
        return np.nan
    return float(g_est_by_sale / g_sale)
