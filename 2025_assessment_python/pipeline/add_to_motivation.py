# DELTE


# Basic clustering for exploratory plots (train on X_train, assign to X_val)
n_clusters = 3
cluster_seed = seed
split_by_cluster = False
if sp.issparse(X_train):
    cluster_scaler = StandardScaler(with_mean=False)
    X_train_cluster = cluster_scaler.fit_transform(X_train)
    X_val_cluster = cluster_scaler.transform(X_val)
    cluster_model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=cluster_seed,
        n_init=20,
        batch_size=1024,
    )
    cluster_labels_train = cluster_model.fit_predict(X_train_cluster)
    cluster_labels_val = cluster_model.predict(X_val_cluster)
else:
    cluster_scaler = StandardScaler()
    X_train_cluster = cluster_scaler.fit_transform(X_train)
    X_val_cluster = cluster_scaler.transform(X_val)
    cluster_model = KMeans(n_clusters=n_clusters, random_state=cluster_seed, n_init=10)
cluster_labels_train = cluster_model.fit_predict(X_train_cluster)
cluster_labels_val = cluster_model.predict(X_val_cluster)
cluster_cmap = plt.cm.get_cmap("tab10", n_clusters)

def _fit_line_params(x_vals, y_vals):
    x_mean = np.mean(x_vals)
    y_mean = np.mean(y_vals)
    denom = np.sum((x_vals - x_mean) ** 2)
    if denom == 0:
        return None
    slope = np.sum((x_vals - x_mean) * (y_vals - y_mean)) / denom
    intercept = y_mean - slope * x_mean
    return slope, intercept

def _plot_trend_lines(ax, x_vals, y_vals, labels, n_clusters, cmap, overall_label="Trend Line"):
    valid_mask = np.isfinite(x_vals) & np.isfinite(y_vals)
    x_vals = x_vals[valid_mask]
    y_vals = y_vals[valid_mask]
    labels = labels[valid_mask]

    overall_params = _fit_line_params(x_vals, y_vals)
    if overall_params is not None:
        slope, intercept = overall_params
        x_line = np.array([x_vals.min(), x_vals.max()])
        ax.plot(x_line, slope * x_line + intercept, "r--", label=overall_label)

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

def _predict_val_by_cluster(
    model,
    X_train,
    y_train_log,
    X_val,
    y_val_index,
    cluster_labels_train,
    cluster_labels_val,
    n_clusters,
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
        model.fit(X_train[train_mask], y_train_log[train_mask])
        y_pred_val[val_mask] = np.exp(model.predict(X_val[val_mask]))
    if np.any(~np.isfinite(y_pred_val)):
        missing = np.sum(~np.isfinite(y_pred_val))
        print(f"Warning: {missing} validation rows missing predictions.")
    return pd.Series(y_pred_val, index=y_val_index)

def _run_clustered_model_diagnostics(
    models,
    X_train,
    y_train_log,
    X_val,
    y_val,
    cluster_labels_train,
    cluster_labels_val,
    n_clusters,
    cluster_cmap,
    split_by_cluster=False,
):
    plt.figure(figsize=(15,6))
    y_val.hist(bins=500)
    plt.savefig(f"./temp/plots/histograms/test_val.png", dpi=600)
    plt.close()

    plt.figure(figsize=(15,6))
    np.log(y_val).hist(bins=500)
    # plt.xscale("log")
    plt.savefig(f"./temp/plots/histograms/log_test_val.png", dpi=600)
    plt.close()

    for model in models:
        if split_by_cluster:
            y_pred_val = _predict_val_by_cluster(
                model,
                X_train,
                y_train_log,
                X_val,
                y_val.index,
                cluster_labels_train,
                cluster_labels_val,
                n_clusters,
            )
        else:
            model.fit(X_train, y_train_log)
            y_pred_val = pd.Series(np.exp(model.predict(X_val)), index=y_val.index)
        y_val_np = y_val.to_numpy()
        y_pred_val_np = y_pred_val.to_numpy()
        log_y_val = np.log(y_val_np)
        log_y_pred_val = np.log(y_pred_val_np)
        ratio_val = y_pred_val_np / y_val_np
        log_ratio_val = log_y_pred_val / log_y_val
        log_resid_val = log_y_pred_val - log_y_val
        model_tag = str(model).split('(')[0]
        print(model)
        # print(y_train.describe())
        # print(pd.Series(y_pred_train).describe())
        print(y_val.describe())
        print(y_pred_val.describe())

        metrics_ = compute_taxation_metrics(log_y_val, log_y_pred_val, scale="log")

        plt.figure(figsize=(15,6))
        y_pred_val.hist(bins=500)
        plt.savefig(f"./temp/plots/histograms/test_pred_{model_tag}.png", dpi=600)
        plt.close()

        plt.figure(figsize=(15,6))
        np.log(y_pred_val).hist(bins=500)
        plt.savefig(f"./temp/plots/histograms/log_test_pred_{model_tag}.png", dpi=600)
        plt.close()

        plt.figure(figsize=(15,6))
        plt.grid(True, alpha=0.5)
        y_ = ratio_val
        x = y_val_np
        scatter = plt.scatter(
            x,
            y_,
            c=cluster_labels_val,
            cmap=cluster_cmap,
            s=8,
            alpha=0.6,
            edgecolors='none',
            label="Ratios",
        )
        cbar = plt.colorbar(scatter, ticks=range(n_clusters))
        cbar.set_label("Cluster")
        cbar.set_ticklabels([f"C{k}" for k in range(n_clusters)])
        _plot_trend_lines(plt.gca(), x, y_, cluster_labels_val, n_clusters, cluster_cmap)
        plt.ylim([-1, 7])
        plt.legend()
        plt.title(f"R2:{metrics_['R2']:.4f} | MAPE:{metrics_['MAPE']:.4f} | PRD:{metrics_['PRD']:.4f} | PRB:{metrics_['PRB']:.4f} | COD:{metrics_['COD']:.4f} |")
        plt.savefig(f"./temp/plots/real_vs_pred/ratio_{model_tag}.png", dpi=600)
        plt.close()

        plt.figure(figsize=(15,6))
        plt.grid(True, alpha=0.5)
        y_ = log_ratio_val
        x = log_y_val
        scatter = plt.scatter(
            x,
            y_,
            c=cluster_labels_val,
            cmap=cluster_cmap,
            s=8,
            alpha=0.6,
            edgecolors='none',
            label="Ratios",
        )
        cbar = plt.colorbar(scatter, ticks=range(n_clusters))
        cbar.set_label("Cluster")
        cbar.set_ticklabels([f"C{k}" for k in range(n_clusters)])
        _plot_trend_lines(plt.gca(), x, y_, cluster_labels_val, n_clusters, cluster_cmap)
        # plt.xscale("log")
        plt.ylim([.8, 1.2])
        plt.title(f"R2:{metrics_['R2']:.4f} | MAPE:{metrics_['MAPE']:.4f} | PRD:{metrics_['PRD']:.4f} | PRB:{metrics_['PRB']:.4f} | COD:{metrics_['COD']:.4f} |")
        plt.legend()
        plt.savefig(f"./temp/plots/real_vs_pred/log_ratio_{model_tag}.png", dpi=600)
        plt.close()

        plt.figure(figsize=(15,6))
        plt.grid(True, alpha=0.5)
        y_ = y_pred_val_np - y_val_np
        x = y_val_np
        scatter = plt.scatter(
            x,
            y_,
            c=cluster_labels_val,
            cmap=cluster_cmap,
            s=8,
            alpha=0.6,
            edgecolors='none',
            label="Residuals",
        )
        cbar = plt.colorbar(scatter, ticks=range(n_clusters))
        cbar.set_label("Cluster")
        cbar.set_ticklabels([f"C{k}" for k in range(n_clusters)])
        _plot_trend_lines(plt.gca(), x, y_, cluster_labels_val, n_clusters, cluster_cmap)
        # plt.ylim([.8, 1.2])
        plt.legend()
        plt.title(f"R2:{metrics_['R2']:.4f} | MAPE:{metrics_['MAPE']:.4f} | PRD:{metrics_['PRD']:.4f} | PRB:{metrics_['PRB']:.4f} | COD:{metrics_['COD']:.4f} |")
        plt.savefig(f"./temp/plots/real_vs_pred/residuals_pred_{model_tag}.png", dpi=600)
        plt.close()

        plt.figure(figsize=(15,6))
        plt.grid(True, alpha=0.5)
        y_ = log_resid_val
        x = log_y_val
        scatter = plt.scatter(
            x,
            y_,
            c=cluster_labels_val,
            cmap=cluster_cmap,
            s=8,
            alpha=0.6,
            edgecolors='none',
            label="Residuals",
        )
        cbar = plt.colorbar(scatter, ticks=range(n_clusters))
        cbar.set_label("Cluster")
        cbar.set_ticklabels([f"C{k}" for k in range(n_clusters)])
        _plot_trend_lines(plt.gca(), x, y_, cluster_labels_val, n_clusters, cluster_cmap)
        # plt.ylim([.8, 1.2])
        plt.title(f"R2:{metrics_['R2']:.4f} | MAPE:{metrics_['MAPE']:.4f} | PRD:{metrics_['PRD']:.4f} | PRB:{metrics_['PRB']:.4f} | COD:{metrics_['COD']:.4f} |")
        plt.legend()
        plt.savefig(f"./temp/plots/real_vs_pred/log_residuals_pred_{model_tag}.png", dpi=600)
        plt.close()

# DELETE
rho_ = 1e3
models_ = [
    # LGBCovDispPenalty(rho_cov=rho_*np.std(y_val_log), rho_disp=0, cov_mode="cov", disp_mode="l2", zero_grad_tol=zero_tol, eps_y=1e-12, eps_std=1e-12, lgbm_params=lgbm_params),
    LGBCovPenalty(rho=1000, ratio_mode="div", zero_grad_tol=zero_tol, eps_y=1e-12, lgbm_params=lgbm_params),
    # LGBSmoothPenalty(rho=rho_, zero_grad_tol=zero_tol, eps_y=1e-12, lgbm_params=lgbm_params),
    lgb.LGBMRegressor(**lgbm_params),
    LGBCondMeanVarPenalty(rho_cov=9, rho_disp=1, n_bins=500, ratio_mode="diff", 
                                                anchor_mode="target", zero_grad_tol=zero_tol, eps_y=1e-12, lgbm_params=lgbm_params),
]

_run_clustered_model_diagnostics(
    models_,
    X_train,
    y_train_log,
    X_val,
    y_val,
    cluster_labels_train,
    cluster_labels_val,
    n_clusters,
    cluster_cmap,
    split_by_cluster=split_by_cluster,
)

exit()


# DELTE
