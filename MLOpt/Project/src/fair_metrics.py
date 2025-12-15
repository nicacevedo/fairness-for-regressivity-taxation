import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


# Fairness notions: 
#   1. Error disparity: E[residuals^2| d] similarity
#   2. No correlation between prediction and sensitive: Corr(d, y)
#   3. No correlation between sensitive and residuals: Corr(d, residuals) 
#   4. Distribution of y has to be similar: E(y|d) similarity | Var(y|d) similarity 

# fairness_metrics = {
#     "Error disparity: max_d E[res^2 | d] - min_d E[res^2 | d]":[],
#     "Prediction-sensitive correlation: Corr(d, f(x))":[],
#     "Residuals-sensitive correlation: Corr(d, y-f(x))":[],
#     "Pred. dist. disparity 1st: max_d E[f(x)|d] - min_d E[f(x)|d]":[],
#     "Pred. dist. disparity 2st: max_d Var[f(x)|d] - min_d Var[f(x)|d]":[],
# }

def fairness_metrics(y_real, y_pred, sensitive_idx, d_sensitive, fairness_type="max_min_error", error_type="rmse"):
    try:
        y_real = y_real.to_numpy()
        d_sensitive = d_sensitive.to_numpy()
        y_pred = y_pred.to_numpy()
    except Exception:
        pass
    # Computing errors
    fairness_metrics = {
        "r2_score":None,
        "error_disparity":None,
        "pred_sens_corr":None,
        "res_sens_corr":None,
        "pred_mean":None,
        "pred_std":None,
    }
    error_list = []
    expected_list = []
    std_list = []
    for g, g_idx in enumerate(sensitive_idx):
        # Error disparity
        if error_type == "rmse":
            # error_list.append(root_mean_squared_error(np.ones(y_pred[g_idx].size), y_pred[g_idx] / y_real[g_idx]))
            error_list.append(r2_score(y_real[g_idx], y_pred[g_idx]))
        elif error_type == "mae":
            error_list.append(mean_absolute_error(y_real[g_idx], y_pred[g_idx]))
        # Distribution parity
        expected_list.append(np.mean(y_pred[g_idx]))
        std_list.append(np.std(y_pred[g_idx]))
    # 1. Error disparity: Computing fairness
    if fairness_type == "max_min_error":
        fairness_metrics["error_disparity"] =  max(error_list) - min(error_list)
        fairness_metrics["pred_mean"] = max(expected_list) - min(expected_list)
        fairness_metrics["pred_std"] = max(std_list) - min(std_list)
    elif fairness_type == "max_error":
        fairness_metrics["error_disparity"] =  max(error_list)
        fairness_metrics["pred_mean"] = max(expected_list) 
        fairness_metrics["pred_std"] = max(std_list) 
        # return max(error_list), error_list
    elif fairness_type == "min_error":
        fairness_metrics["error_disparity"] =  min(error_list)
        fairness_metrics["pred_mean"] = min(expected_list)
        fairness_metrics["pred_std"] = min(std_list)
        # return min(error_list), error_list
    # 2. Predicted/sensitive correlation
    # print(y_pred) 
    # print(d_sensitive)
    fairness_metrics["pred_sens_corr"] = np.corrcoef(y_pred, d_sensitive)[0,1]
    # 3. Residuals/sensitive correlation
    fairness_metrics["res_sens_corr"] = np.corrcoef(y_pred, d_sensitive)[0,1]
    # R2 score of the model
    fairness_metrics["r2_score"] = r2_score(y_real, y_pred)

    return fairness_metrics, error_list, expected_list, std_list






import matplotlib.pyplot as plt
import seaborn as sns
from math import pi


def plot_grouped_bar(df):
    """
    Best for: Direct, precise comparison of values.
    """
    # Create the plot
    ax = df.plot(kind='bar', figsize=(10, 6), width=0.8, colormap='viridis')
    
    # Styling
    plt.title('Fairness Metrics by Model (Grouped Bar)', fontsize=16)
    plt.ylabel('Score', fontsize=12)
    plt.xlabel('Models', fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', title="Metrics")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    # plt.show()
    plt.savefig("./temp/plots/grouped_bar.png")

def plot_radar_chart(df):
    """
    Best for: Visualizing trade-offs (e.g., Accuracy vs Fairness).
    """
    # 1. Create background
    categories = list(df.columns)
    N = len(categories)
    
    # Determine angles for each axis
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1] # Close the loop for a circular plot
    
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    
    # 2. Setup the axes
    plt.xticks(angles[:-1], categories, color='black', size=10)
    ax.set_rlabel_position(0)
    plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5], ["0.1", "0.2", "0.3", "0.4", "0.5"], color="grey", size=7)
    plt.ylim(0, 0.5)
    
    # 3. Plot data for each model
    # Define a color palette
    palette = plt.cm.get_cmap("Set1", len(df))
    
    for i, (model_name, row) in enumerate(df.iterrows()):
        values = row.values.flatten().tolist()
        values += values[:1] # Close the loop
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name, color=palette(i))
        ax.fill(angles, values, alpha=0.1, color=palette(i))
    
    plt.title('Model Profile & Trade-offs (Radar Chart)', size=15, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    # plt.show()
    plt.tight_layout()
    plt.savefig("./temp/plots/radar_chart.png")


def plot_heatmap(df):
    """
    Best for: High-density data (many models or many metrics).
    """
    plt.figure(figsize=(8, 6))
    
    # Transpose df if you prefer Models on the Y-axis
    sns.heatmap(df, annot=True, cmap='Blues', vmin=0, vmax=1, fmt=".2f", linewidths=.5)
    
    plt.title('Heatmap of Fairness Metrics', fontsize=16)
    plt.tight_layout()
    # plt.show()
    plt.savefig("./temp/plots/plot_heatmapAAA.pdf", dpi=1200)
    