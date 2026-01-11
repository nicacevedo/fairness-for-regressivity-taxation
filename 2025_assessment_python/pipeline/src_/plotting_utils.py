import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
        if "(" in str(model):
            try:
                rho = str(model).split("(")[1].split(",")[0].replace(")", "") # ModelClassName(rho, ...)
                rho = float(rho)
            except:
                rho = None
        else:
            rho = None
        x_values = [rho for _ in range(len(r_list))]
        
        if len(data_list) == 1:
            # BASELINE
            constant_value = y_values[0]
            plt.plot(x_values, [constant_value] * len(r_list), 
                    label=f"{model} (Baseline)",
                    color=c, linestyle='--', linewidth=2, alpha=0.7)
        else:
            # EXPERIMENTAL
            plt.plot(x_values, y_values, 
                    label=model,
                    color=c, marker=marker, linestyle=linestyle, linewidth=2)
    
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

def plotting_dict_of_models_results(results, r_list, source="train", n_jobs=1):
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