import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def results_to_dataframe(results, r_values, round_decimals=4):
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
                row['r_value'] = r_values[i] if i < len(r_values) else i
            else:
                row['r_value'] = None  # Or 'Baseline'
                
            rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)
    # Optional: Reorder columns to put Model and r_value first
    cols = ['Model', 'r_value'] + [c for c in df.columns if c not in ['Model', 'r_value']]
    df = df[cols]
    return df.round(round_decimals)

# def plotting_dict_of_models_results(dict_of_results, label_names=None):
#     n_experiments = len(dict_of_results)
#     if label_names is None:
#         label_names = ["Experiment {}".format(i) for i in range(n_experiments)]
#     exp_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

#     # --- Plot 0: Training corr Scores ---
#     plt.figure(figsize=(8, 5))
#     i = 0
#     for key, results in dict_of_results.items():
#         color = exp_colors[i % len(exp_colors)]
#         label = label_names[i]
#         alpha_ = 0.3 if i <= 4 and i>1 else 0.8
#         plt.plot(
#             keep_percentages, 
#             corr_results_val[i], 
#             label=label,
#             color=color,
#             linestyle=linestyle,
#             marker=marker,
#             alpha=alpha_
#         )
#         i+=1

#     plt.xlabel(r"$\mathbf{Rate\;of\;Samples\;to\;Keep:\;}$ $r = K/n$")
#     plt.ylabel(r"$\mathbf{Correlation\;(Un)fairness:\;}$ $\text{Corr}(\;d,\;f(x)\;)$ ")
#     plt.title(r"$\mathbf{Impact\;of\;Correlation\;Fairness\;Constaint}$" "\n" r"$\mathbf{in\;Stable\;Regression}\;[Testing\;Set]$")
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
#     plt.grid(True, alpha=0.7)
#     plt.tight_layout()
#     plt.savefig("./temp/plots/delete_vs_0_correlation_val.pdf", dpi=1200)


def plotting_dict_of_models_results(results, r_list, source="train"):


    # 1. Define your r_list (This must match the number of results in your experimental models)
    # r_list = [1, 5, 10]  # Example values

    # 2. Extract all unique metrics from the first model's first result
    # (Assumes all models have the same set of metrics)
    first_model = list(results.keys())[0]
    metrics_names = list(results[first_model][0].keys())

    # 3. Setup styling
    # Assign a unique color to each model
    model_names = list(results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    model_color_map = dict(zip(model_names, colors))

    # Assign different markers/linestyles for each metric (to distinguish plots visually)
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-']

    # 4. Generate one plot per metric
    for i, metric in enumerate(metrics_names):
        plt.figure(figsize=(10, 6))
        
        # Pick style for this specific metric
        marker = markers[i % len(markers)]
        linestyle = linestyles[i % len(linestyles)]
        
        for model, data_list in results.items():
            # Get the color for this model
            c = model_color_map[model]
            
            # Extract the values for this specific metric
            y_values = [res[metric] for res in data_list]
            
            if len(data_list) == 1:
                # BASELINE: Plot as a constant horizontal line across the r_list
                # We create a list of the single value repeated len(r_list) times
                constant_value = y_values[0]
                plt.plot(r_list, [constant_value] * len(r_list), 
                        label=f"{model} (Baseline)",
                        color=c, linestyle='--', linewidth=2, alpha=0.7)
            else:
                # EXPERIMENTAL: Plot the varying values against r_list
                plt.plot(r_list, y_values, 
                        label=model,
                        color=c, marker=marker, linestyle=linestyle, linewidth=2)
        
        plt.title(f'Comparison of {metric} vs. Constraint Radius (r)', fontsize=14)
        plt.xlabel('r_list Value', fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"./temp/plots/metrics/{metric}_{source}.png", dpi=600)