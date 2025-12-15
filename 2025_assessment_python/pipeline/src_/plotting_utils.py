import matplotlib.pyplot as plt


def plotting_dict_of_models_results(dict_of_results, label_names=None):
    n_experiments = len(dict_of_results)
    if label_names is None:
        label_names = ["Experiment {}".format(i) for i in range(n_experiments)]
    exp_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # --- Plot 0: Training corr Scores ---
    plt.figure(figsize=(8, 5))
    i = 0
    for key, results in dict_of_results.items():
        color = exp_colors[i % len(exp_colors)]
        label = label_names[i]
        alpha_ = 0.3 if i <= 4 and i>1 else 0.8
        plt.plot(
            keep_percentages, 
            corr_results_val[i], 
            label=label,
            color=color,
            linestyle=linestyle,
            marker=marker,
            alpha=alpha_
        )
        i+=1

    plt.xlabel(r"$\mathbf{Rate\;of\;Samples\;to\;Keep:\;}$ $r = K/n$")
    plt.ylabel(r"$\mathbf{Correlation\;(Un)fairness:\;}$ $\text{Corr}(\;d,\;f(x)\;)$ ")
    plt.title(r"$\mathbf{Impact\;of\;Correlation\;Fairness\;Constaint}$" "\n" r"$\mathbf{in\;Stable\;Regression}\;[Testing\;Set]$")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.grid(True, alpha=0.7)
    plt.tight_layout()
    plt.savefig("./temp/plots/delete_vs_0_correlation_val.pdf", dpi=1200)