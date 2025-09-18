
"""
Regressivity EDA Template for Housing Prices
--------------------------------------------
This script performs an exploratory data analysis (EDA) specifically aimed at diagnosing
and characterizing *regressivity* (systematic overprediction of low-priced homes and
underprediction of high-priced homes) BEFORE any predictive modeling.

Inputs
------
- A CSV file with at least a target column `price` and multiple feature columns.
- Optional columns (used if present):
    - latitude, longitude (float): for spatial scatter
    - neighborhood (str/categorical): for neighborhood-level variance

Usage
-----
python regressivity_eda_template.py --csv /path/to/your_data.csv --target price \
    --report_dir ./regressivity_report --low_quantile 0.3 --n_clusters 3

The script will create the folder `report_dir` containing PNGs and a JSON summary.
It will also print a short text summary at the end.

What it computes
----------------
1) Target distribution: histogram and log-histogram (if strictly positive).
2) Price variability by quantile bins: mean, std, coefficient of variation (CV), plots.
3) Feature vs target diagnostics:
   - Continuous features: scatter (price vs feature), and log–log if valid.
   - Categorical features: boxplots of price by category for top small-cardinality features.
4) Missingness vs price segment: Missing rate per feature across price bins.
5) Spatial/Neighborhood (if available):
   - Map-like scatter by (longitude, latitude) colored by price quantile.
   - Neighborhood-level mean and variance and their plots.
6) Multimodality / Clusters:
   - KMeans on standardized numeric features (k configurable), and within-cluster price variance.
7) Distributional overlap / Separability:
   - Train a simple classifier to distinguish "low-price" (<= quantile q) vs others using features.
   - Report ROC AUC; low AUC => features do not separate low-price well (predictive difficulty).
8) Feature strength by segment:
   - Mutual information (numeric) per segment (low/mid/high) to detect feature gaps.

Requirements
------------
pip install pandas numpy scikit-learn matplotlib
(Optionally: pip install pyproj if you want to do fancier maps; not required here.)

Notes
-----
- No seaborn used (pure matplotlib), as requested.
- All plots saved as PNGs in report_dir.
"""

import argparse
import os
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_regression

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------
# Utility functions
# -----------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def savefig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

def infer_feature_types(df: pd.DataFrame, target: str, max_cat_card: int = 30) -> Tuple[List[str], List[str]]:
    """Separate features into numeric and categorical (low-cardinality) lists."""
    feature_cols = [c for c in df.columns if c != target]
    num_cols = [c for c in feature_cols if is_numeric_series(df[c])]
    cat_cols = [c for c in feature_cols if (not is_numeric_series(df[c])) or (df[c].nunique() <= max_cat_card)]
    # Filter cat_cols to those that are not obviously numeric with large cardinality
    cat_cols = [c for c in cat_cols if c not in num_cols]
    return num_cols, cat_cols

def price_bins(df: pd.DataFrame, target: str, n_bins: int = 10) -> pd.Series:
    """Quantile-based bins for price."""
    # Ensure strictly increasing bin edges
    q = np.linspace(0, 1, n_bins + 1)
    edges = df[target].quantile(q).values
    # Deduplicate potential equal edges (can happen with many identical prices)
    edges = np.unique(edges)
    # In rare cases of massive ties, fall back to equal-width bins
    if len(edges) < 3:
        edges = np.linspace(df[target].min(), df[target].max(), n_bins + 1)
    bins = pd.cut(df[target], bins=edges, include_lowest=True, duplicates="drop")
    return bins

def coefficient_of_variation(x: np.ndarray) -> float:
    m = np.nanmean(x)
    s = np.nanstd(x, ddof=1)
    return float(s / m) if m != 0 else np.nan

def safe_log_transform(x: pd.Series) -> pd.Series:
    """Return log(x) only if x > 0. Otherwise return NaN for non-positive values."""
    x = x.astype(float)
    x_log = pd.Series(np.nan, index=x.index)
    mask = x > 0
    x_log[mask] = np.log(x[mask])
    return x_log

def plot_histograms_price(df: pd.DataFrame, target: str, outdir: str):
    fig, ax = plt.subplots()
    ax.hist(df[target].dropna().values, bins=60)
    ax.set_title("Price Histogram")
    ax.set_xlabel(target)
    ax.set_ylabel("Count")
    savefig(os.path.join(outdir, "01_price_hist.png"))

    # Log-hist if strictly positive values exist
    ylog = safe_log_transform(df[target])
    if ylog.notna().sum() > 0:
        fig, ax = plt.subplots()
        ax.hist(ylog.dropna().values, bins=60)
        ax.set_title("Log-Price Histogram")
        ax.set_xlabel(f"log({target})")
        ax.set_ylabel("Count")
        savefig(os.path.join(outdir, "02_log_price_hist.png"))

def plot_cv_by_price_bin(df: pd.DataFrame, target: str, outdir: str, n_bins: int = 10):
    bins = price_bins(df, target, n_bins=n_bins)
    g = df.groupby(bins)[target]
    stats = g.agg(["count", "mean", "std"]).reset_index()
    stats["cv"] = stats["std"] / stats["mean"]
    # Save table
    stats.to_csv(os.path.join(outdir, "cv_by_price_bin.csv"), index=False)

    # Plot CV
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(stats)), stats["cv"].values, marker="o")
    ax.set_title("Coefficient of Variation (CV) by Price Bin")
    ax.set_xlabel("Price Bin (quantile)")
    ax.set_ylabel("CV (std/mean)")
    savefig(os.path.join(outdir, "03_cv_by_price_bin.png"))

def plot_feature_relationships(df: pd.DataFrame, target: str, num_cols: List[str], cat_cols: List[str], outdir: str, top_cat: int = 5):
    # Continuous features: scatter price vs feature; log-log if valid
    for col in num_cols[:20]:  # cap to first 20 to avoid plot explosion
        fig, ax = plt.subplots()
        ax.scatter(df[col], df[target], s=6, alpha=0.5)
        ax.set_title(f"Price vs {col}")
        ax.set_xlabel(col)
        ax.set_ylabel(target)
        savefig(os.path.join(outdir, f"10_price_vs_{col}.png"))

        # log-log if both positive
        xlog = safe_log_transform(df[col])
        ylog = safe_log_transform(df[target])
        if xlog.notna().sum() > 0 and ylog.notna().sum() > 0:
            fig, ax = plt.subplots()
            ax.scatter(xlog, ylog, s=6, alpha=0.5)
            ax.set_title(f"log(Price) vs log({col})")
            ax.set_xlabel(f"log({col})")
            ax.set_ylabel(f"log({target})")
            savefig(os.path.join(outdir, f"11_log_price_vs_log_{col}.png"))

    # Categorical features: boxplots for top small-cardinality features
    cat_cards = sorted([(c, df[c].nunique()) for c in cat_cols], key=lambda x: x[1])
    for col, card in cat_cards[:top_cat]:
        # Drop NaN categories
        sub = df[[col, target]].dropna()
        if sub.empty:
            continue
        # Limit categories shown to avoid gigantic plots
        counts = sub[col].value_counts()
        keep = counts.index[:min(15, len(counts))]
        sub = sub[sub[col].isin(keep)]

        # Prepare boxplot data
        data = [sub[sub[col] == k][target].values for k in keep]
        fig, ax = plt.subplots(figsize=(max(6, len(keep)*0.4), 4))
        ax.boxplot(data, labels=[str(k) for k in keep], showfliers=False)
        ax.set_title(f"Price by {col} (top categories)")
        ax.set_xlabel(col)
        ax.set_ylabel(target)
        plt.xticks(rotation=45, ha="right")
        savefig(os.path.join(outdir, f"12_price_by_{col}.png"))

def plot_missingness_by_price_bin(df: pd.DataFrame, target: str, outdir: str, n_bins: int = 10):
    bins = price_bins(df, target, n_bins=n_bins)
    miss_rates = []
    for col in df.columns:
        if col == target:
            continue
        rate_by_bin = df[col].isna().groupby(bins).mean()
        miss_rates.append(pd.DataFrame({"feature": col, "bin": range(len(rate_by_bin)), "missing_rate": rate_by_bin.values}))
    if not miss_rates:
        return
    miss_df = pd.concat(miss_rates, ignore_index=True)
    miss_df.to_csv(os.path.join(outdir, "missingness_by_price_bin.csv"), index=False)

    # Plot average missingness per bin across features
    avg = miss_df.groupby("bin")["missing_rate"].mean().reset_index()
    fig, ax = plt.subplots()
    ax.plot(avg["bin"], avg["missing_rate"], marker="o")
    ax.set_title("Average Feature Missingness vs Price Bin")
    ax.set_xlabel("Price Bin (quantile)")
    ax.set_ylabel("Avg Missingness")
    savefig(os.path.join(outdir, "20_missingness_vs_price_bin.png"))

def plot_spatial_if_present(df: pd.DataFrame, target: str, outdir: str):
    lat_col = None
    lon_col = None
    for c in df.columns:
        lc = c.lower()
        if lc in ("latitude", "lat"):
            lat_col = c
        if lc in ("longitude", "lon", "lng"):
            lon_col = c
    if lat_col is None or lon_col is None:
        return

    # Color by price quantile
    bins = pd.qcut(df[target], q=10, labels=False, duplicates="drop")
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(df[lon_col], df[lat_col], c=bins, s=6)
    ax.set_title("Spatial scatter colored by price decile")
    ax.set_xlabel(lon_col)
    ax.set_ylabel(lat_col)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Price decile")
    savefig(os.path.join(outdir, "30_spatial_price_deciles.png"))

def neighborhood_stats_if_present(df: pd.DataFrame, target: str, outdir: str):
    neigh_col = None
    for c in df.columns:
        if c.lower() in ("neighborhood", "nbhd", "tract", "census_tract", "blockgroup"):
            neigh_col = c
            break
    if neigh_col is None:
        return

    g = df.groupby(neigh_col)[target]
    stats = g.agg(["count", "mean", "std"]).reset_index()
    stats["cv"] = stats["std"] / stats["mean"]
    stats.to_csv(os.path.join(outdir, "31_neighborhood_price_stats.csv"), index=False)

    # Plot neighborhood CV distribution
    fig, ax = plt.subplots()
    ax.hist(stats["cv"].dropna().values, bins=50)
    ax.set_title("Neighborhood-level CV of Price")
    ax.set_xlabel("CV")
    ax.set_ylabel("Count of neighborhoods")
    savefig(os.path.join(outdir, "31_neighborhood_cv_hist.png"))

def kmeans_cluster_variance(df: pd.DataFrame, target: str, num_cols: List[str], outdir: str, n_clusters: int = 3):
    if len(num_cols) == 0:
        return None
    X = df[num_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True))
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=17)
    labels = km.fit_predict(Xs)
    dfc = df.copy()
    dfc["cluster"] = labels
    stats = dfc.groupby("cluster")[target].agg(["count", "mean", "std"]).reset_index()
    stats["cv"] = stats["std"] / stats["mean"]
    stats.to_csv(os.path.join(outdir, "40_cluster_price_stats.csv"), index=False)

    # Plot cluster means with error bars
    fig, ax = plt.subplots()
    ax.errorbar(stats["cluster"].values, stats["mean"].values, yerr=stats["std"].values, fmt="o-")
    ax.set_title("Cluster-wise Price Mean ± Std")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Price")
    savefig(os.path.join(outdir, "40_cluster_mean_std.png"))
    return stats

def separability_classifier_auc(df: pd.DataFrame, target: str, num_cols: List[str], cat_cols: List[str], outdir: str, low_quantile: float = 0.3):
    # Binary label: low price vs rest
    thr = df[target].quantile(low_quantile)
    y = (df[target] <= thr).astype(int).values

    # Prepare features: numeric + categorical (one-hot)
    numeric_features = num_cols
    categorical_features = [c for c in cat_cols if df[c].nunique() > 1]

    # Avoid empty feature sets
    if len(numeric_features) == 0 and len(categorical_features) == 0:
        return None

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32), categorical_features),
        ],
        remainder="drop"
    )

    clf = Pipeline([
        ("pre", pre),
        ("lr", LogisticRegression(max_iter=200, solver="lbfgs"))
    ])

    # Drop rows with NA in selected columns
    used_cols = numeric_features + categorical_features
    sub = df[used_cols + [target]].dropna()
    if sub.empty:
        return None

    y_sub = (sub[target] <= thr).astype(int).values
    X_sub = sub[used_cols]

    try:
        clf.fit(X_sub, y_sub)
        # Predict probabilities for AUC
        yhat = clf.predict_proba(X_sub)[:, 1]
        auc = roc_auc_score(y_sub, yhat)
    except Exception:
        auc = np.nan

    # Save AUC
    with open(os.path.join(outdir, "50_separability_auc.txt"), "w") as f:
        f.write(f"AUC (low<=q{low_quantile} vs rest): {auc:.4f}\n")
    return auc

def feature_strength_by_segment(df: pd.DataFrame, target: str, num_cols: List[str], outdir: str):
    """
    Compute mutual information (numeric features only) with the target
    for three segments: low (<= 0.33), mid (0.33-0.66], high (> 0.66).
    Higher MI => stronger nonlinear association.
    """
    if len(num_cols) == 0:
        return None

    q1, q2 = df[target].quantile([0.33, 0.66])
    segments = {
        "low": df[df[target] <= q1],
        "mid": df[(df[target] > q1) & (df[target] <= q2)],
        "high": df[df[target] > q2],
    }
    mi_table = []

    for seg_name, sub in segments.items():
        sub = sub[num_cols + [target]].dropna()
        if len(sub) < 50:
            continue
        X = sub[num_cols].copy()
        y = sub[target].values
        # Replace inf and fill NA
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median(numeric_only=True))
        # Compute MI per feature
        try:
            mi = mutual_info_regression(X.values, y, random_state=17)
            for col, val in zip(num_cols, mi):
                mi_table.append({"segment": seg_name, "feature": col, "mutual_info": float(val)})
        except Exception:
            continue

    if mi_table:
        mi_df = pd.DataFrame(mi_table)
        mi_df.to_csv(os.path.join(outdir, "60_mutual_information_by_segment.csv"), index=False)

        # Plot top features by MI per segment (up to 10)
        for seg in mi_df["segment"].unique():
            seg_df = mi_df[mi_df["segment"] == seg].sort_values("mutual_info", ascending=False).head(10)
            if seg_df.empty:
                continue
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(seg_df["feature"], seg_df["mutual_info"])
            ax.set_title(f"Top MI Features — {seg} segment")
            ax.set_xlabel("Mutual Information")
            ax.invert_yaxis()
            savefig(os.path.join(outdir, f"60_top_mi_{seg}.png"))
        return mi_df
    return None

# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to input CSV with at least a 'price' column")
    parser.add_argument("--target", type=str, default="price", help="Target column name (default: price)")
    parser.add_argument("--report_dir", type=str, default="regressivity_report", help="Output directory for report artifacts")
    parser.add_argument("--low_quantile", type=float, default=0.3, help="Quantile threshold for low-price class")
    parser.add_argument("--n_bins", type=int, default=10, help="Number of price quantile bins")
    parser.add_argument("--n_clusters", type=int, default=3, help="KMeans clusters for multimodality diagnosis")
    parser.add_argument("--max_cat_card", type=int, default=30, help="Max cardinality to treat a feature as categorical")
    args = parser.parse_args()

    ensure_dir(args.report_dir)

    # Load data
    df = pd.read_csv(args.csv)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in CSV columns: {list(df.columns)[:20]}...")

    # Basic cleaning for analysis
    df = df.replace([np.inf, -np.inf], np.nan)

    # 1) Target distribution
    plot_histograms_price(df, args.target, args.report_dir)

    # 2) Variability by price bin (CV)
    plot_cv_by_price_bin(df, args.target, args.report_dir, n_bins=args.n_bins)

    # Infer feature types
    num_cols, cat_cols = infer_feature_types(df, args.target, max_cat_card=args.max_cat_card)

    # 3) Feature-target relationships
    plot_feature_relationships(df, args.target, num_cols, cat_cols, args.report_dir, top_cat=5)

    # 4) Missingness vs price bin
    plot_missingness_by_price_bin(df, args.target, args.report_dir, n_bins=args.n_bins)

    # 5) Spatial/neighborhood (optional)
    plot_spatial_if_present(df, args.target, args.report_dir)
    neighborhood_stats_if_present(df, args.target, args.report_dir)

    # 6) Clusters and within-cluster variance
    km_stats = kmeans_cluster_variance(df, args.target, num_cols, args.report_dir, n_clusters=args.n_clusters)

    # 7) Separability classifier (low vs rest)
    auc = separability_classifier_auc(df, args.target, num_cols, cat_cols, args.report_dir, low_quantile=args.low_quantile)

    # 8) Feature strength by segment (mutual information)
    mi_df = feature_strength_by_segment(df, args.target, num_cols, args.report_dir)

    # Summarize key outcomes to JSON
    summary = {
        "csv": args.csv,
        "target": args.target,
        "n_rows": int(len(df)),
        "low_quantile": args.low_quantile,
        "n_bins": args.n_bins,
        "n_clusters": args.n_clusters,
        "num_feature_count": len(num_cols),
        "cat_feature_count": len(cat_cols),
        "separability_auc": None if auc is None or (isinstance(auc, float) and np.isnan(auc)) else float(auc),
    }
    if km_stats is not None:
        summary["cluster_price_stats_csv"] = "40_cluster_price_stats.csv"
    if mi_df is not None:
        summary["mutual_information_by_segment_csv"] = "60_mutual_information_by_segment.csv"

    with open(os.path.join(args.report_dir, "SUMMARY.json"), "w") as f:
        f.write(json.dumps(summary, indent=2))

    # Print short console summary
    print("=== Regressivity EDA Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nArtifacts saved in: {os.path.abspath(args.report_dir)}")
    print("Key files include:")
    print(" - 01_price_hist.png, 02_log_price_hist.png (if applicable)")
    print(" - 03_cv_by_price_bin.png, cv_by_price_bin.csv")
    print(" - 10_*, 11_* feature relationship plots")
    print(" - 12_* category boxplots")
    print(" - 20_missingness_vs_price_bin.png, missingness_by_price_bin.csv")
    print(" - 30_spatial_price_deciles.png (if lat/lon present)")
    print(" - 31_neighborhood_cv_hist.png, 31_neighborhood_price_stats.csv (if neighborhood present)")
    print(" - 40_cluster_mean_std.png, 40_cluster_price_stats.csv")
    print(" - 50_separability_auc.txt")
    print(" - 60_top_mi_*.png, 60_mutual_information_by_segment.csv")
    print(" - SUMMARY.json")

if __name__ == "__main__":
    main()
