import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import kurtosis, skew, rankdata
from sklearn.metrics import r2_score

# ==============================================================================
# 1. Core Metric Calculations (Refactored for Efficiency and Readability)
# ==============================================================================

def _calculate_empirical_quantiles(values: np.ndarray) -> np.ndarray:
    """
    Efficiently computes the empirical quantile of each value in an array.
    This is equivalent to an empirical cumulative distribution function (ECDF).

    Args:
        values (np.ndarray): An array of numerical values (e.g., true house prices).

    Returns:
        np.ndarray: An array of the same size, where each element is the
                    empirical quantile of the corresponding input value.
    """
    # rankdata with method='max' counts how many items are <= to the current item.
    # This is an O(n log n) operation, much faster than the previous O(n^2) loop.
    ranks = rankdata(values, method='max')
    return ranks / len(values)

def _positive_part(x: np.ndarray) -> np.ndarray:
    """Returns the positive part of an array, setting negative values to 0."""
    return np.maximum(0, x)

def calculate_f_dev(ratio: np.ndarray, quantiles: np.ndarray, alpha: float = 2.0) -> float:
    """
    Calculates the F_dev fairness metric.
    This metric penalizes deviations from a ratio of 1, with weights based on
    the quantile of the true value. High-priced houses are weighted to not be
    under-assessed, and low-priced houses are weighted to not be over-assessed.

    Args:
        ratio (np.ndarray): The assessment ratio (y_pred / y_true).
        quantiles (np.ndarray): The empirical quantiles of the true values.
        alpha (float): A parameter controlling the exponential weighting.

    Returns:
        float: The calculated F_dev score.
    """
    w1 = np.exp(-alpha * quantiles)
    w2 = np.exp(-alpha * (1 - quantiles))
    
    over_assessment_penalty = _positive_part(ratio - 1) @ w1
    under_assessment_penalty = _positive_part(1 - ratio) @ w2
    
    return over_assessment_penalty + under_assessment_penalty

def _get_quantile_groups(quantiles: np.ndarray, n_groups: int = 3) -> dict[int, np.ndarray]:
    """
    Divides the data indices into groups based on their quantile values.

    Args:
        quantiles (np.ndarray): The empirical quantiles of the true values.
        n_groups (int): The number of quantile groups to create.

    Returns:
        dict[int, np.ndarray]: A dictionary mapping group index to an array of
                               data indices belonging to that group.
    """
    if n_groups <= 0:
        raise ValueError("n_groups must be a positive integer.")
        
    bounds = np.linspace(0, 1, n_groups + 1)
    groups = {}
    for i in range(n_groups):
        lower_bound = bounds[i]
        upper_bound = bounds[i+1]
        # Find indices where quantile is within the current bin's bounds.
        # The first group includes 0, subsequent groups are (lower, upper].
        if i == 0:
            in_group = (quantiles >= lower_bound) & (quantiles <= upper_bound)
        else:
            in_group = (quantiles > lower_bound) & (quantiles <= upper_bound)
        groups[i] = np.where(in_group)[0]
        
    return groups

def calculate_f_grp(ratio: np.ndarray, quantile_groups: dict) -> float:
    """
    Calculates the F_grp fairness metric in a memory-efficient way.
    This metric measures vertical inequity by checking if higher-priced groups
    have systematically higher assessment ratios than lower-priced groups.

    Args:
        ratio (np.ndarray): The assessment ratio (y_pred / y_true).
        quantile_groups (dict): A dictionary of quantile groups from _get_quantile_groups.

    Returns:
        float: The calculated F_grp score.
    """
    score = 0.0
    group_indices = sorted(quantile_groups.keys())
    
    for i in range(len(group_indices)):
        for j in range(i + 1, len(group_indices)):
            g1_idx = quantile_groups[group_indices[i]]
            g2_idx = quantile_groups[group_indices[j]]
            
            if len(g1_idx) == 0 or len(g2_idx) == 0:
                continue

            r_g1 = ratio[g1_idx]
            r_g2 = ratio[g2_idx]
            
            # Efficiently calculate sum(max(0, r_i - r_j)) without broadcasting
            # by sorting one group and using searchsorted.
            r_g2_sorted = np.sort(r_g2)
            r_g2_cumsum = np.cumsum(r_g2_sorted)
            
            total_positive_diff = 0
            for val_g1 in r_g1:
                # Find how many elements in r_g2 are smaller than val_g1
                count = np.searchsorted(r_g2_sorted, val_g1, side='right')
                if count > 0:
                    # Sum of differences = val_g1 * count - sum(elements in r_g2 smaller than val_g1)
                    sum_smaller_g2 = r_g2_cumsum[count - 1]
                    total_positive_diff += val_g1 * count - sum_smaller_g2
            
            score += total_positive_diff / (len(r_g1) * len(r_g2))
            
    return score

# ==============================================================================
# 2. Main Utility Functions for Analysis and Plotting
# ==============================================================================

def compute_all_metrics(y_true: pd.Series, y_pred: pd.Series, n_groups: int = 3, alpha: float = 2) -> dict:
    """
    Computes a comprehensive set of performance and fairness metrics.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        n_groups: Number of groups for the F_grp metric.
        alpha: Alpha parameter for the F_dev metric.

    Returns:
        A dictionary containing all computed metrics.
    """
    if isinstance(y_true, pd.Series): y_true = y_true.to_numpy()
    if isinstance(y_pred, pd.Series): y_pred = y_pred.to_numpy()
        
    ratio = y_pred / y_true
    quantiles = _calculate_empirical_quantiles(y_true)
    groups = _get_quantile_groups(quantiles, n_groups)
    
    metrics = {
        'rmse': np.sqrt(np.mean((y_pred - y_true)**2)),
        'r2': r2_score(y_true, y_pred),
        'f_dev': calculate_f_dev(ratio, quantiles, alpha),
        'f_grp': calculate_f_grp(ratio, groups),
        'ratio_std': np.std(ratio),
        'ratio_skew': skew(ratio),
        'ratio_kurtosis': kurtosis(ratio)
    }
    return metrics

def create_diagnostic_plots(
    y_train: pd.Series, 
    y_test: pd.Series, 
    y_pred_train: pd.Series, 
    y_pred_test: pd.Series,
    n_groups: int = 3, 
    alpha: float = 2,
    save_plots: bool = False, 
    suffix: str = "", 
    log_scale: bool = False
):
    """
    Generates and displays a set of diagnostic scatter plots for regression results.

    Args:
        y_train, y_test: True values for train and test sets.
        y_pred_train, y_pred_test: Predicted values for train and test sets.
        n_groups, alpha: Parameters for fairness metrics.
        save_plots: If True, saves the plots to the 'img/' directory.
        suffix: A string to append to saved plot filenames.
        log_scale: If True, applies a log scale to plot axes where appropriate.
    """
    # Compute metrics for both sets
    train_metrics = compute_all_metrics(y_train, y_pred_train, n_groups, alpha)
    test_metrics = compute_all_metrics(y_test, y_pred_test, n_groups, alpha)

    # --- Plotting ---
    
    def _plot_real_vs_pred(y_true, y_pred, metrics, label, color, filename):
        if y_true.size > 1000:
            sample_idx = np.random.choice(range(y_true.size), 1000, replace=False)
            y_true = y_true.iloc[sample_idx]
            y_pred = y_pred[sample_idx]
        plt.figure(figsize=(8, 5))
        plt.scatter(y_true, y_pred, facecolor='none', label=label, color=color, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color="red", label="y = x")
        plt.legend()
        plt.title(f"RMSE={metrics['rmse']:.2f} | F_dev={metrics['f_dev']:.3f}, F_grp={metrics['f_grp']:.3f}\n"
                  f"Ratio: std={metrics['ratio_std']:.3f}, skew={metrics['ratio_skew']:.3f}")
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        if log_scale:
            plt.xscale("log")
            plt.yscale("log")
        if save_plots:
            plt.savefig(f"img/versus/{filename}{suffix}.jpg", dpi=300)
        plt.show()

    def _plot_ratio_vs_price(y_true, y_pred, metrics, label, color, filename):
        if y_true.size > 1000:
            sample_idx = np.random.choice(range(y_true.size), 1000, replace=False)
            y_true = y_true.iloc[sample_idx]
            y_pred = y_pred[sample_idx]
        ratio = y_pred / y_true
        plt.figure(figsize=(8, 5))
        plt.scatter(y_true, ratio, facecolor='none', label=label, color=color, alpha=0.5)
        plt.hlines(1, y_true.min(), y_true.max(), colors="red", label="Fair Ratio (1.0)")
        plt.legend()
        plt.title(f"RMSE={metrics['rmse']:.2f} | R2={metrics['r2']:.2f} | F_dev={metrics['f_dev']:.3f}, F_grp={metrics['f_grp']:.3f}\n"
                  f"Ratio: std={metrics['ratio_std']:.3f}, skew={metrics['ratio_skew']:.3f}")
        plt.xlabel("True Values")
        plt.ylabel("Assessment Ratio (Pred / True)")
        if log_scale:
            plt.xscale("log")
            # plt.yscale("log")
        if save_plots:
            plt.savefig(f"img/ratio/{filename}{suffix}.jpg", dpi=300)
        plt.show()

    def _plot_residuals_vs_price(y_true, y_pred, metrics, label, color, filename):
        if y_true.size > 1000:
            sample_idx = np.random.choice(range(y_true.size), 1000, replace=False)
            y_true = y_true.iloc[sample_idx]
            y_pred = y_pred[sample_idx]
        residuals = y_true - y_pred
        plt.figure(figsize=(8, 5))
        plt.scatter(y_pred, -residuals, facecolor='none', label=label, color=color, alpha=0.5)
        plt.hlines(0, y_pred.min(), y_pred.max(), colors="red", label="Perfect Fit")
        plt.legend()
        plt.title(f"RMSE={metrics['rmse']:.2f} | R2={metrics['r2']:.2f} | F_dev={metrics['f_dev']:.3f}, F_grp={metrics['f_grp']:.3f}\n"
                  f"Ratio: std={metrics['ratio_std']:.3f}, skew={metrics['ratio_skew']:.3f}")
        plt.xlabel("Pred Values")
        plt.ylabel("Assessment (-)Residuals (Pred - True)")
        if log_scale:
            plt.xscale("log")
            # plt.yscale("log")
        if save_plots:
            plt.savefig(f"img/residuals/{filename}{suffix}.jpg", dpi=300)
        plt.show()

    def _plot_prc_residuals_vs_price(y_true, y_pred, metrics, label, color, filename):
        if y_true.size > 1000:
            sample_idx = np.random.choice(range(y_true.size), 1000, replace=False)
            y_true = y_true.iloc[sample_idx]
            y_pred = y_pred[sample_idx]
        residuals = y_true - y_pred
        plt.figure(figsize=(8, 5))
        plt.scatter(y_pred, -residuals/y_pred, facecolor='none', label=label, color=color, alpha=0.5)
        plt.hlines(0, y_pred.min(), y_pred.max(), colors="red", label="Perfect Fit")
        plt.legend()
        plt.title(f"RMSE={metrics['rmse']:.2f} | R2={metrics['r2']:.2f} | F_dev={metrics['f_dev']:.3f}, F_grp={metrics['f_grp']:.3f}\n"
                  f"Ratio: std={metrics['ratio_std']:.3f}, skew={metrics['ratio_skew']:.3f}")
        plt.xlabel("Pred Values")
        plt.ylabel("Assessment (-) Percent. Residuals (Pred - True)/Pred")
        if log_scale:
            plt.xscale("log")
            # plt.yscale("log")
        if save_plots:
            plt.savefig(f"img/prc_residuals/{filename}{suffix}.jpg", dpi=300)
        plt.show()

    # Generate plots for Test set
    _plot_real_vs_pred(y_test, y_pred_test, test_metrics, "Test", "blue", "real_vs_pred_test")
    _plot_ratio_vs_price(y_test, y_pred_test, test_metrics, "Test Ratio", "black", "ratio_vs_price_test")
    _plot_residuals_vs_price(y_test, y_pred_test, test_metrics, "Test Residuals", "green", "residuals_vs_price_test")
    _plot_prc_residuals_vs_price(y_test, y_pred_test, test_metrics, "Test Residuals", "saddlebrown", "residuals_vs_price_test")

    # Generate plots for Train set
    _plot_real_vs_pred(y_train, y_pred_train, train_metrics, "Train", "cyan", "real_vs_pred_train")
    _plot_ratio_vs_price(y_train, y_pred_train, train_metrics, "Train Ratio", "gray", "ratio_vs_price_train")
    _plot_residuals_vs_price(y_train, y_pred_train, train_metrics, "Train Residuals", "lime", "residuals_vs_price_train")
    _plot_prc_residuals_vs_price(y_train, y_pred_train, train_metrics, "Train Residuals", "peru", "residuals_vs_price_train")


