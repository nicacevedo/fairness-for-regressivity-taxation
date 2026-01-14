import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy
import re

# Multiprocessing
import os
from multiprocessing import Pool


def results_to_dataframe(results, r_values, round_decimals=3, source="train"):
    rows = []
    for model_name, metrics_list in results.items():
        # Loop through the list of results for this specific model
        for i, metric_dict in enumerate(metrics_list):
            row = metric_dict.copy()
            row['Model'] = model_name
            
            # If the model has multiple results, map them to r_list
            # If it's a baseline (length 1), we can set r_value to None or 'Baseline'
            if len(metrics_list) > 1:
                # Safely access r_list, or fallback to index if r_list is too short
                row['r'] = r_values[i] if i < len(r_values) else i
            else:
                row['r'] = None  # Or 'Baseline'
                
            rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)
    # Optional: Reorder columns to put Model and r_value first
    cols = ['Model', 'r'] + [c for c in df.columns if c not in ['Model', 'r']]
    df = df[cols] #.style.format(precision=round_decimals) #.round(round_decimals)

    # --- New Saving Logic ---
    file_name = f"./temp/tables/results_{source}.txt"
    try:
        # Generate the tabular content only (no caption/label yet).
        # We removed booktabs=True as requested.
        latex_tabular = df.to_latex(
            index=False,
            escape=False,     # Set to True if you want special characters escaped
            column_format=None, # Allow pandas to default the column formats
            # booktabs=False    # Removed booktabs for standard formatting
            float_format="%.3f",
        )
        
        # Manually wrap the tabular content to add font sizing (\footnotesize).
        full_latex_content = (
            "\\begin{table}[htbp]\n"
            "    \\centering\n"
            "    \\resizebox{\\textwidth}{!}{%\n"
            f"{latex_tabular}"
            "     }\n"
            f"    \\caption{{Results for {source}}}\n"
            f"    \\label{{tab:results_{source}}}\n"
            "\\end{table}"
        )
        
        # Save to .txt file
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(full_latex_content)
            
        print(f"Successfully saved LaTeX table to: {file_name}")
        
    except Exception as e:
        print(f"Error saving LaTeX file: {e}")

    return df


def _as_float_array(values):
    if isinstance(values, pd.Series):
        values = values.to_numpy()
    return np.asarray(values, dtype=float)


def _safe_log(values):
    values = _as_float_array(values)
    out = np.full(values.shape, np.nan, dtype=float)
    mask = values > 0
    out[mask] = np.log(values[mask])
    return out


def _fit_line_params(x_vals, y_vals):
    x_mean = np.mean(x_vals)
    y_mean = np.mean(y_vals)
    denom = np.sum((x_vals - x_mean) ** 2)
    if denom == 0:
        return None
    slope = np.sum((x_vals - x_mean) * (y_vals - y_mean)) / denom
    intercept = y_mean - slope * x_mean
    return slope, intercept


def _plot_trend_lines(
    ax,
    x_vals,
    y_vals,
    labels=None,
    n_clusters=0,
    cmap=None,
    overall_label="Overall Trend",
):
    valid_mask = np.isfinite(x_vals) & np.isfinite(y_vals)
    if labels is None:
        labels = None
    else:
        labels = labels[valid_mask]

    x_vals = x_vals[valid_mask]
    y_vals = y_vals[valid_mask]

    if x_vals.size < 2:
        return

    overall_params = _fit_line_params(x_vals, y_vals)
    if overall_params is not None:
        slope, intercept = overall_params
        x_line = np.array([x_vals.min(), x_vals.max()])
        ax.plot(x_line, slope * x_line + intercept, "r--", label=overall_label)

    if labels is None or n_clusters <= 0 or cmap is None:
        return

    for k in range(n_clusters):
        cluster_mask = labels == k
        if np.sum(cluster_mask) < 2:
            continue
        cluster_params = _fit_line_params(x_vals[cluster_mask], y_vals[cluster_mask])
        if cluster_params is None:
            continue
        slope, intercept = cluster_params
        x_min = x_vals[cluster_mask].min()
        x_max = x_vals[cluster_mask].max()
        x_line = np.array([x_min, x_max])
        ax.plot(
            x_line,
            slope * x_line + intercept,
            color=cmap(k),
            linewidth=1.5,
            label=f"C{k} Trend",
        )


def _clone_model(model):
    try:
        from sklearn.base import clone
        return clone(model)
    except Exception:
        try:
            return copy.deepcopy(model)
        except Exception:
            return model


def _sanitize_model_tag(model_name):
    base = str(model_name).split("(")[0].strip()
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", base)
    return safe or "model"


def _format_metric(value):
    if value is None or not np.isfinite(value):
        return "nan"
    return f"{value:.4f}"


def _metrics_title_line(metrics):
    cov_val = metrics.get("Corr ratio_y", metrics.get("Cov ratio_y"))
    prb_val = metrics.get("PRB")
    prd_val = metrics.get("PRD")
    mki_val = metrics.get("MKI")
    return (
        f"cov(r,y)={_format_metric(cov_val)} | "
        f"PRB={_format_metric(prb_val)} | "
        f"PRD={_format_metric(prd_val)} | "
        f"MKI={_format_metric(mki_val)}"
    )


def _cluster_labels(X_train, X_val, n_clusters, cluster_seed):
    import scipy.sparse as sp
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans, MiniBatchKMeans

    if sp.issparse(X_train):
        scaler = StandardScaler(with_mean=False)
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        cluster_model = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=cluster_seed,
            n_init=20,
            batch_size=1024,
        )
    else:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        cluster_model = KMeans(n_clusters=n_clusters, random_state=cluster_seed, n_init=10)

    cluster_labels_train = cluster_model.fit_predict(X_train_scaled)
    cluster_labels_val = cluster_model.predict(X_val_scaled)
    cluster_cmap = plt.cm.get_cmap("tab10", n_clusters)
    return cluster_labels_train, cluster_labels_val, cluster_cmap


def _predict_val_by_cluster(
    model,
    X_train,
    y_train,
    X_val,
    cluster_labels_train,
    cluster_labels_val,
    n_clusters,
    clone_models=True,
):
    y_pred_val = np.full(X_val.shape[0], np.nan, dtype=float)
    for k in range(n_clusters):
        train_mask = cluster_labels_train == k
        val_mask = cluster_labels_val == k
        if not np.any(val_mask):
            continue
        if not np.any(train_mask):
            print(f"Warning: cluster {k} has no training rows; skipping.")
            continue
        model_k = _clone_model(model) if clone_models else model
        model_k.fit(X_train[train_mask], y_train[train_mask])
        y_pred_val[val_mask] = model_k.predict(X_val[val_mask])
    if np.any(~np.isfinite(y_pred_val)):
        missing = np.sum(~np.isfinite(y_pred_val))
        print(f"Warning: {missing} validation rows missing predictions.")
    return y_pred_val


def _scatter_with_trends(
    x_vals,
    y_vals,
    labels,
    n_clusters,
    cmap,
    title,
    xlabel,
    ylabel,
    out_path,
    y_limits=None,
    point_label="Points",
):
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.grid(True, alpha=0.5)

    valid_mask = np.isfinite(x_vals) & np.isfinite(y_vals)
    x_vals = x_vals[valid_mask]
    y_vals = y_vals[valid_mask]
    labels = None if labels is None else labels[valid_mask]

    if labels is None:
        ax.scatter(
            x_vals,
            y_vals,
            s=8,
            alpha=0.6,
            edgecolors="none",
            label=point_label,
        )
        _plot_trend_lines(ax, x_vals, y_vals)
    else:
        scatter = ax.scatter(
            x_vals,
            y_vals,
            c=labels,
            cmap=cmap,
            s=8,
            alpha=0.6,
            edgecolors="none",
            label=point_label,
        )
        cbar = fig.colorbar(scatter, ticks=range(n_clusters))
        cbar.set_label("Cluster")
        cbar.set_ticklabels([f"C{k}" for k in range(n_clusters)])
        _plot_trend_lines(ax, x_vals, y_vals, labels, n_clusters, cmap)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if y_limits is not None:
        ax.set_ylim(y_limits)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=600)
    plt.close(fig)


def plot_scatter_diagnostics(
    models,
    model_names,
    X_train,
    y_train,
    X_val,
    y_val,
    *,
    y_train_log=None,
    y_val_log=None,
    n_clusters=3,
    cluster_seed=0,
    split_by_cluster=False,
    include_clustered=True,
    include_overall=True,
    plot_dir="./temp/plots/real_vs_pred",
    overall_subdir="overall",
    price_scale_model_tokens=None,
    fit_models=False,
):
    from src_.motivation_utils import compute_taxation_metrics

    if model_names is None:
        model_names = [str(model) for model in models]
    if len(model_names) != len(models):
        raise ValueError("model_names must match models length.")

    y_train_arr = _as_float_array(y_train)
    y_val_arr = _as_float_array(y_val)
    y_train_log_arr = _safe_log(y_train_arr) if y_train_log is None else _as_float_array(y_train_log)
    y_val_log_arr = _safe_log(y_val_arr) if y_val_log is None else _as_float_array(y_val_log)

    if price_scale_model_tokens is None:
        price_scale_model_tokens = ("LGBCorrTweediePenalty", "LGBCovTweediePenalty")

    cluster_labels_train = None
    cluster_labels_val = None
    cluster_cmap = None
    if include_clustered or split_by_cluster:
        cluster_labels_train, cluster_labels_val, cluster_cmap = _cluster_labels(
            X_train,
            X_val,
            n_clusters,
            cluster_seed,
        )

    if include_clustered:
        os.makedirs(plot_dir, exist_ok=True)
    if include_overall:
        overall_dir = plot_dir if overall_subdir is None else os.path.join(plot_dir, overall_subdir)
        os.makedirs(overall_dir, exist_ok=True)

    for model, model_name in zip(models, model_names):
        model_tag = _sanitize_model_tag(model_name)
        is_price_scale = any(token in str(model_name) for token in price_scale_model_tokens)

        if is_price_scale:
            y_train_target = y_train_arr
        else:
            y_train_target = y_train_log_arr

        if fit_models and not split_by_cluster:
            model.fit(X_train, y_train_target)

        if split_by_cluster:
            y_pred_raw = _predict_val_by_cluster(
                model,
                X_train,
                y_train_target,
                X_val,
                cluster_labels_train,
                cluster_labels_val,
                n_clusters,
            )
        else:
            y_pred_raw = model.predict(X_val)

        if is_price_scale:
            y_pred_arr = _as_float_array(y_pred_raw)
            y_pred_log_arr = _safe_log(y_pred_arr)
            metrics = compute_taxation_metrics(y_val_arr, y_pred_arr, scale="price")
        else:
            y_pred_log_arr = _as_float_array(y_pred_raw)
            y_pred_arr = np.exp(y_pred_log_arr)
            metrics = compute_taxation_metrics(y_val_log_arr, y_pred_log_arr, scale="log")

        ratio_val = y_pred_arr / y_val_arr
        log_ratio_val = np.divide(
            y_pred_log_arr,
            y_val_log_arr,
            out=np.full_like(y_pred_log_arr, np.nan, dtype=float),
            where=np.isfinite(y_val_log_arr) & (y_val_log_arr != 0),
        )
        residual_val = y_pred_arr - y_val_arr
        log_resid_val = y_pred_log_arr - y_val_log_arr

        title = f"{model_name}\n{_metrics_title_line(metrics)}"

        if include_clustered:
            _scatter_with_trends(
                y_val_arr,
                ratio_val,
                cluster_labels_val,
                n_clusters,
                cluster_cmap,
                title,
                "True value (y)",
                "Ratio (pred / y)",
                os.path.join(plot_dir, f"ratio_{model_tag}.png"),
                y_limits=(-1, 7),
                point_label="Ratios",
            )
            _scatter_with_trends(
                y_val_log_arr,
                log_ratio_val,
                cluster_labels_val,
                n_clusters,
                cluster_cmap,
                title,
                "log(y)",
                "log(pred) / log(y)",
                os.path.join(plot_dir, f"log_ratio_{model_tag}.png"),
                y_limits=(0.8, 1.2),
                point_label="Ratios",
            )
            _scatter_with_trends(
                y_val_arr,
                residual_val,
                cluster_labels_val,
                n_clusters,
                cluster_cmap,
                title,
                "True value (y)",
                "Residual (pred - y)",
                os.path.join(plot_dir, f"residuals_pred_{model_tag}.png"),
                point_label="Residuals",
            )
            _scatter_with_trends(
                y_val_log_arr,
                log_resid_val,
                cluster_labels_val,
                n_clusters,
                cluster_cmap,
                title,
                "log(y)",
                "Log residual (log(pred) - log(y))",
                os.path.join(plot_dir, f"log_residuals_pred_{model_tag}.png"),
                point_label="Residuals",
            )

        if include_overall:
            overall_dir = plot_dir if overall_subdir is None else os.path.join(plot_dir, overall_subdir)
            _scatter_with_trends(
                y_val_arr,
                ratio_val,
                None,
                0,
                None,
                title,
                "True value (y)",
                "Ratio (pred / y)",
                os.path.join(overall_dir, f"ratio_{model_tag}.png"),
                y_limits=(-1, 7),
                point_label="Ratios",
            )
            _scatter_with_trends(
                y_val_log_arr,
                log_ratio_val,
                None,
                0,
                None,
                title,
                "log(y)",
                "log(pred) / log(y)",
                os.path.join(overall_dir, f"log_ratio_{model_tag}.png"),
                y_limits=(0.8, 1.2),
                point_label="Ratios",
            )
            _scatter_with_trends(
                y_val_arr,
                residual_val,
                None,
                0,
                None,
                title,
                "True value (y)",
                "Residual (pred - y)",
                os.path.join(overall_dir, f"residuals_pred_{model_tag}.png"),
                point_label="Residuals",
            )
            _scatter_with_trends(
                y_val_log_arr,
                log_resid_val,
                None,
                0,
                None,
                title,
                "log(y)",
                "Log residual (log(pred) - log(y))",
                os.path.join(overall_dir, f"log_residuals_pred_{model_tag}.png"),
                point_label="Residuals",
            )

# def plotting_dict_of_models_results(results, r_list, source="train"):


#     # 1. Define your r_list (This must match the number of results in your experimental models)
#     # r_list = [1, 5, 10]  # Example values

#     # 2. Extract all unique metrics from the first model's first result
#     # (Assumes all models have the same set of metrics)
#     first_model = list(results.keys())[0]
#     metrics_names = list(results[first_model][0].keys())

#     # 3. Setup styling
#     # Assign a unique color to each model
#     model_names = list(results.keys())
#     colors = plt.cm.tab20b(np.linspace(0, 1, len(model_names)))
#     model_color_map = dict(zip(model_names, colors))

#     # Assign different markers/linestyles for each metric (to distinguish plots visually)
#     markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
#     linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-']

#     # 4. Generate one plot per metric
#     for i, metric in enumerate(metrics_names):
#         plt.figure(figsize=(12, 6))
        
#         # Pick style for this specific metric
#         marker = markers[i % len(markers)]
#         linestyle = linestyles[i % len(linestyles)]
        
#         for model, data_list in results.items():
#             # Get the color for this model
#             c = model_color_map[model]
            
#             # Extract the values for this specific metric
#             y_values = [res[metric] for res in data_list]
            
#             if len(data_list) == 1:
#                 # BASELINE: Plot as a constant horizontal line across the r_list
#                 # We create a list of the single value repeated len(r_list) times
#                 constant_value = y_values[0]
#                 plt.plot(r_list, [constant_value] * len(r_list), 
#                         label=f"{model} (Baseline)",
#                         color=c, linestyle='--', linewidth=2, alpha=0.7)
#             else:
#                 # EXPERIMENTAL: Plot the varying values against r_list
#                 plt.plot(r_list, y_values, 
#                         label=model,
#                         color=c, marker=marker, linestyle=linestyle, linewidth=2)
        
#         plt.title(f'Comparison of {metric} vs. Ratio to Keep (r) [{source}]', fontsize=14)
#         plt.xlabel(r'$r=K/n$ (Ratio of samples to keep)', fontsize=12)
#         plt.ylabel(metric, fontsize=12)
#         plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=6, borderpad=0.5, labelspacing=0.3, handletextpad=0.5)
#         plt.grid(True, alpha=0.5)
#         plt.tight_layout()
#         plt.savefig(f"./temp/plots/metrics/{metric}_{source}.png", dpi=600)
#         plt.close()

#     # 5. [MINE] Generate one plot per metric, but metrics vs rho
#     for i, metric in enumerate(metrics_names):
#         plt.figure(figsize=(12, 6))
        
#         # Pick style for this specific metric
#         marker = markers[i % len(markers)]
#         linestyle = linestyles[i % len(linestyles)]
        
#         for model, data_list in results.items():
#             # Get the color for this model
#             c = model_color_map[model]
            
#             # Extract the values for this specific metric
#             y_values = [res[metric] for res in data_list]
            
#             # MINE: get name of the rho and extract its value
#             if "(" in str(model):
#                 rho = str(model).split("(")[1].split(",")[0].replace(")", "") # ModelClassName(rho, ...)
#                 rho = float(rho)
#             else:
#                 rho = None
#             x_values = [rho for _ in range(len(r_list))]

#             if len(data_list) == 1:
#                 # BASELINE: Plot as a constant horizontal line across the r_list
#                 # We create a list of the single value repeated len(r_list) times
#                 constant_value = y_values[0]
#                 plt.plot(x_values, [constant_value] * len(r_list), 
#                         label=f"{model} (Baseline)",
#                         color=c, linestyle='--', linewidth=2, alpha=0.7)
#             else:
#                 # EXPERIMENTAL: Plot the varying values against r_list
#                 plt.plot(x_values, y_values, 
#                         label=model,
#                         color=c, marker=marker, linestyle=linestyle, linewidth=2)
        
#         plt.title(f'Comparison of {metric} vs. Penalization (rho) [{source}]', fontsize=14)
#         plt.xlabel(r'$\rho$ (Penalization)', fontsize=12)
#         plt.ylabel(metric, fontsize=12)
#         plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=6, borderpad=0.5, labelspacing=0.3, handletextpad=0.5)
#         plt.grid(True, alpha=0.5)
#         plt.tight_layout()
#         plt.savefig(f"./temp/plots/metrics_vs_rho/rho_{metric}_{source}.png", dpi=600)
#         plt.close()



def _process_metric_vs_r(args):
    """
    Worker function to plot a single metric vs Ratio (r_list).
    Executed in parallel.
    """
    i, metric, results, r_list, source, model_color_map, markers, linestyles = args
    
    # Set backend to Agg to avoid GUI issues in parallel processes
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    
    # Pick style for this specific metric
    marker = markers[i % len(markers)]
    linestyle = linestyles[i % len(linestyles)]
    
    for model, data_list in results.items():
        # Get the color for this model
        c = model_color_map[model]
        
        # Extract the values for this specific metric
        y_values = [res[metric] for res in data_list]
        
        if len(data_list) == 1:
            # BASELINE: Plot as a constant horizontal line
            constant_value = y_values[0]
            plt.plot(r_list, [constant_value] * len(r_list), 
                    label=f"{model} (Baseline)",
                    color=c, linestyle='--', linewidth=2, alpha=0.7)
        else:
            # EXPERIMENTAL: Plot the varying values
            plt.plot(r_list, y_values, 
                    label=model,
                    color=c, marker=marker, linestyle=linestyle, linewidth=2)
    
    plt.title(f'Comparison of {metric} vs. Ratio to Keep (r) [{source}]', fontsize=14)
    plt.xlabel(r'$r=K/n$ (Ratio of samples to keep)', fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=6, borderpad=0.5, labelspacing=0.3, handletextpad=0.5)
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(f"./temp/plots/metrics/", exist_ok=True)
    plt.savefig(f"./temp/plots/metrics/{metric}_{source}.png", dpi=600)
    plt.close()

def _process_metric_vs_rho(args):
    """
    Worker function to plot a single metric vs Penalization (rho).
    Executed in parallel.
    """
    i, metric, results, r_list, source, model_color_map, markers, linestyles = args
    
    # Set backend to Agg
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    
    # Pick style for this specific metric
    marker = markers[i % len(markers)]
    linestyle = linestyles[i % len(linestyles)]
    
    for model, data_list in results.items():
        # Get the color for this model
        c = model_color_map[model]
        
        # Extract the values for this specific metric
        y_values = [res[metric] for res in data_list]
        
        # MINE: get name of the rho and extract its value
        rho = None
        if "rho=" in str(model):
            try:
                # Extract rho from "rho=value" format
                rho_part = str(model).split("rho=")[1].split(",")[0].split(")")[0].strip()
                rho = float(rho_part)
            except (ValueError, IndexError):
                rho = None
        
        if rho is None:
            # Skip plotting if rho cannot be extracted
            print(f"Warning: Could not parse rho from model: {model}")
            continue
        
        x_values = [rho for _ in range(len(data_list))]
        
        if len(data_list) == 1:
            # BASELINE: constant horizontal line
            constant_value = y_values[0]
            plt.axhline(y=constant_value, 
                    label=f"{model} (Baseline)",
                    color=c, linestyle='--', linewidth=2, alpha=0.7)
        else:
            # EXPERIMENTAL: scatter plot (rho on x, metric values on y)
            # Each model contributes multiple points (one per r value)
            plt.scatter(x_values, y_values, 
                    label=model,
                    color=c, s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    plt.title(f'Comparison of {metric} vs. Penalization (rho) [{source}]', fontsize=14)
    plt.xlabel(r'$\rho$ (Penalization)', fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=6, borderpad=0.5, labelspacing=0.3, handletextpad=0.5)
    plt.grid(True, alpha=0.5)
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(f"./temp/plots/metrics_vs_rho/", exist_ok=True)
    plt.savefig(f"./temp/plots/metrics_vs_rho/rho_{metric}_{source}.png", dpi=600)
    plt.close()

def plotting_dict_of_models_results(results, r_list, source="train", n_jobs=1, scatter_config=None):
    # 1. Define your r_list (passed as argument)

    # 2. Extract all unique metrics from the first model's first result
    first_model = list(results.keys())[0]
    metrics_names = list(results[first_model][0].keys())

    # 3. Setup styling
    # Assign a unique color to each model
    model_names = list(results.keys())
    colors = plt.cm.tab20b(np.linspace(0, 1, len(model_names)))
    model_color_map = dict(zip(model_names, colors))

    # Assign different markers/linestyles for each metric
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-']

    # 4. Prepare arguments for parallel processing
    # We pack all necessary data into a tuple for each metric iteration
    
    tasks_r = []
    tasks_rho = []
    
    for i, metric in enumerate(metrics_names):
        # Arguments for plot vs R
        args_r = (i, metric, results, r_list, source, model_color_map, markers, linestyles)
        tasks_r.append(args_r)
        
        # Arguments for plot vs Rho
        args_rho = (i, metric, results, r_list, source, model_color_map, markers, linestyles)
        tasks_rho.append(args_rho)

    # 5. Execute in parallel
    # Determine number of processes
    if n_jobs is None or n_jobs < 1:
        num_processes = os.cpu_count() or 4
    else:
        num_processes = n_jobs
        
    print(f"Generating plots in parallel using {num_processes} cores...")

    with Pool(processes=num_processes) as pool:
        # Run first set of plots
        pool.map(_process_metric_vs_r, tasks_r)
        
        # Run second set of plots
        pool.map(_process_metric_vs_rho, tasks_rho)

    print("All plots generated successfully.")

    if scatter_config:
        plot_scatter_diagnostics(**scatter_config)
