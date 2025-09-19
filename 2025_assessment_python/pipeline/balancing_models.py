import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import resample
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE, SMOTENC

class BalancingResampler(BaseEstimator, TransformerMixin):
    """
    A custom preprocessor to re-balance a dataset based on a continuous target (y).

    The resampler works by first binning the data based on y, and then applying
    over-sampling and under-sampling techniques to each bin to bring them toward
    a uniform size.

    Parameters
    ----------
    n_bins : int or 'auto', default=5
        Number of bins to split y into. If 'auto', uses the Freedman–Diaconis rule.

    binning_policy : {'uniform','quantile','outlier','kmeans','decision_tree'}, default='quantile'
        How to create bins.

    max_diff_ratio : float, default=0.5
        Target samples per bin is computed as:
            min_bin + (max_bin - min_bin) * max_diff_ratio
        0.0 -> match smallest bin, 1.0 -> match largest bin.

    undersample_policy : {'random','outlier','inlier','tomek_links'}, default='outlier'
        Policy to shrink large bins.

    oversample_policy : {'smote','smoter','smotenc','generalized_smote','density_smote',
                         'generalized_smotenc','density_smotenc'}, default='smoter'
        Policy to expand small bins.
        - *_smotenc variants require `categorical_features`.

    generalized_smote_weighting : {'random','gravity'}, default='random'
        Weighting for generalized_smote / *_smotenc convex combinations.

    smote_k_neighbors : int, default=5
        k_neighbors for SMOTE-like methods.

    categorical_features : list[int] or None, default=None
        Indices of categorical columns (for *_smotenc and mixed-type logic).

    random_state : int or None, default=None
        Random seed used across operations.
    """
    def __init__(self, n_bins=5, binning_policy='quantile', max_diff_ratio=0.5,
                 undersample_policy='outlier', oversample_policy='smoter',
                 generalized_smote_weighting='random', smote_k_neighbors=5,
                 categorical_features=None, random_state=None,
                 low_count_policy='neighbor_bridge', neighbor_k=6,
                 bridge_gamma_max=1.0):
        self.n_bins = n_bins
        self.binning_policy = binning_policy
        self.max_diff_ratio = max_diff_ratio
        self.undersample_policy = undersample_policy
        self.oversample_policy = oversample_policy
        self.generalized_smote_weighting = generalized_smote_weighting
        self.smote_k_neighbors = smote_k_neighbors
        self.categorical_features = categorical_features
        self.random_state = random_state
        # NEW: controls for sparse/empty bins
        self.low_count_policy = low_count_policy  # 'neighbor_bridge' or 'skip'
        self.neighbor_k = neighbor_k              # total k; ~k/2 per side
        self.bridge_gamma_max = bridge_gamma_max  # max one-sided extrapolation factor
        if self.random_state is not None:
            np.random.seed(self.random_state)

    # ------------------------------------------------------------------
    # sklearn API
    def fit(self, X, y):
        return self

    # ------------------------------------------------------------------
    # Binning helpers
    def _get_n_bins(self, y):
        if self.n_bins != 'auto':
            return self.n_bins
        iqr = np.subtract(*np.percentile(y, [75, 25]))
        bin_width = 2 * iqr * (len(y) ** (-1/3))
        if bin_width == 0:
            return 10
        num_bins = int(np.ceil((y.max() - y.min()) / bin_width))
        return max(2, min(num_bins, 50))

    def _create_bins(self, y, n_bins_actual):
        y_s = y if isinstance(y, pd.Series) else pd.Series(y)
        y_np = y_s.to_numpy()

        if self.binning_policy == 'uniform':
            bins, edges = pd.cut(y_s, n_bins_actual, labels=False, duplicates='drop', retbins=True)
            return bins, edges
        elif self.binning_policy == 'quantile':
            bins, edges = pd.qcut(y_s, n_bins_actual, labels=False, duplicates='drop', retbins=True)
            return bins, edges

        # Label-based policies; derive edges after
        if self.binning_policy == 'outlier':
            iso = IsolationForest(random_state=self.random_state)
            scores = iso.fit_predict(y_np.reshape(-1, 1))
            bin_labels, edges = pd.qcut(scores, n_bins_actual, labels=False, duplicates='drop', retbins=True)
        elif self.binning_policy == 'kmeans':
            kmeans = KMeans(n_clusters=n_bins_actual, random_state=self.random_state, n_init=10)
            bin_labels = kmeans.fit_predict(y_np.reshape(-1, 1))
        elif self.binning_policy == 'decision_tree':
            tree = DecisionTreeRegressor(max_leaf_nodes=n_bins_actual, random_state=self.random_state)
            tree.fit(y_np.reshape(-1, 1), y)
            bin_labels = tree.apply(y_np.reshape(-1, 1))
        else:
            raise ValueError(f"Unknown binning_policy: {self.binning_policy}")

        if 'edges' not in locals():
            df = pd.DataFrame({'y': y_np, 'bin': bin_labels})
            sorted_bins_max = df.groupby('bin')['y'].max().sort_index()
            edges = [y_np.min()] + sorted_bins_max.tolist()
            edges[-1] = y_np.max()
            edges = np.unique(edges)

        return pd.Series(bin_labels, index=y.index), edges

    # ------------------------------------------------------------------
    # y interpolation for SMOTER-like strategies
    def _smoter_interpolate_y(self, X_orig, y_orig, X_synthetic):
        numeric_cols = [c for i, c in enumerate(X_orig.columns)
                        if self.categorical_features is None or i not in self.categorical_features]
        X_orig_numeric = X_orig[numeric_cols]
        X_syn_numeric = X_synthetic[numeric_cols]

        k = min(self.smote_k_neighbors, len(X_orig_numeric) - 1)
        if k < 1:
            return pd.Series()

        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(X_orig_numeric)
        distances, indices = nn.kneighbors(X_syn_numeric)

        weights = 1.0 / (distances + 1e-6)
        weights = weights / np.sum(weights, axis=1, keepdims=True)
        y_neighbors = y_orig.to_numpy()[indices]
        interpolated_y = np.sum(weights * y_neighbors, axis=1)
        return pd.Series(interpolated_y)

    # ------------------------------------------------------------------
    # Generalized SMOTE (convex combinations) — numeric + optional cats (mode)
    def _generalized_smote_oversample(self, X_bin, y_bin, n_to_generate):
        new_X_list, new_y_list = [], []
        subsample_size = self.smote_k_neighbors + 1
        numeric_cols = [c for i, c in enumerate(X_bin.columns)
                        if self.categorical_features is None or i not in self.categorical_features]
        cat_cols = [c for i, c in enumerate(X_bin.columns)
                    if self.categorical_features is not None and i in self.categorical_features]

        for _ in range(n_to_generate):
            subsample_indices = np.random.choice(X_bin.index, size=min(subsample_size, len(X_bin)), replace=False)
            X_sub = X_bin.loc[subsample_indices]
            y_sub = y_bin.loc[subsample_indices]

            if self.generalized_smote_weighting == 'random':
                weights = np.random.rand(len(X_sub))
            elif self.generalized_smote_weighting == 'gravity':
                centroid = X_sub[numeric_cols].mean(axis=0)
                distances = np.linalg.norm(X_sub[numeric_cols] - centroid, axis=1)
                weights = 1.0 / (distances + 1e-6)
            else:
                raise ValueError(f"Unknown generalized_smote_weighting: {self.generalized_smote_weighting}")
            weights /= weights.sum()

            new_X_numeric = np.dot(weights, X_sub[numeric_cols])
            new_y = np.dot(weights, y_sub)
            new_sample = pd.Series(new_X_numeric, index=numeric_cols)
            for col in cat_cols:
                new_sample[col] = X_sub[col].mode().iloc[0]
            new_X_list.append(new_sample)
            new_y_list.append(new_y)

        return pd.DataFrame(new_X_list), pd.Series(new_y_list)

    # ADASYN-like density-focused oversampling (numeric focus; delegates to generalized)
    def _density_smote_oversample(self, X_bin, y_bin, n_to_generate):
        numeric_cols = [c for i, c in enumerate(X_bin.columns)
                        if self.categorical_features is None or i not in self.categorical_features]
        X_num = X_bin[numeric_cols]
        k = min(self.smote_k_neighbors, len(X_num) - 1)
        if k < 1:
            return pd.DataFrame(), pd.Series()
        nn = NearestNeighbors(n_neighbors=k + 1)
        nn.fit(X_num)
        distances, _ = nn.kneighbors(X_num)
        density = 1.0 / (distances[:, 1:].mean(axis=1) + 1e-6)
        # Selection probability could be used to bias subsampling; we reuse generalized method
        return self._generalized_smote_oversample(X_bin, y_bin, n_to_generate)

    # Advanced SMOTE for mixed data supporting *_smotenc
    def _advanced_smote_oversample(self, X_bin, y_bin, n_to_generate, policy):
        if self.categorical_features is None:
            raise ValueError(f"'{policy}' requires `categorical_features` to be specified.")

        new_X_list, new_y_list = [], []
        subsample_size = self.smote_k_neighbors + 1

        numeric_cols = [c for i, c in enumerate(X_bin.columns) if i not in self.categorical_features]
        cat_cols = [c for i, c in enumerate(X_bin.columns) if i in self.categorical_features]
        X_num = X_bin[numeric_cols]

        selection_prob = None
        if policy == 'density_smotenc':
            k = min(self.smote_k_neighbors, len(X_num) - 1)
            if k >= 1:
                nn = NearestNeighbors(n_neighbors=k + 1)
                nn.fit(X_num)
                distances, _ = nn.kneighbors(X_num)
                density = 1.0 / (distances[:, 1:].mean(axis=1) + 1e-6)
                selection_prob = 1.0 / density
                selection_prob /= selection_prob.sum()

        for _ in range(n_to_generate):
            base_idx = np.random.choice(X_bin.index, p=selection_prob) if selection_prob is not None else np.random.choice(X_bin.index)
            k_eff = min(self.smote_k_neighbors, len(X_num) - 1)
            nn_sub = NearestNeighbors(n_neighbors=k_eff + 1)
            nn_sub.fit(X_num)
            _, neighbor_indices = nn_sub.kneighbors(X_num.loc[[base_idx]])
            subsample_indices = X_bin.index[neighbor_indices.flatten()]
            subsample_size_eff = min(subsample_size, len(subsample_indices))
            final_subsample_indices = np.random.choice(subsample_indices, size=subsample_size_eff, replace=False)

            X_sub = X_bin.loc[final_subsample_indices]
            y_sub = y_bin.loc[final_subsample_indices]

            if self.generalized_smote_weighting == 'random':
                weights = np.random.rand(len(X_sub))
            else:  # 'gravity'
                centroid = X_sub[numeric_cols].mean(axis=0)
                distances = np.linalg.norm(X_sub[numeric_cols] - centroid, axis=1)
                weights = 1.0 / (distances + 1e-6)
            weights /= weights.sum()

            new_X_numeric = np.dot(weights, X_sub[numeric_cols])
            new_y = np.dot(weights, y_sub)
            new_sample = pd.Series(new_X_numeric, index=numeric_cols)
            for col in cat_cols:
                new_sample[col] = X_sub[col].mode().iloc[0]
            new_sample = new_sample[X_bin.columns]  # reorder

            new_X_list.append(new_sample)
            new_y_list.append(new_y)

        return pd.DataFrame(new_X_list), pd.Series(new_y_list)

    # ------------------------------------------------------------------
    # Neighbor-bridge generator for sparse or empty bins
    def _neighbor_bridge_generate(self, X_all, y_all, edges, edge_idx, n_to_generate):
        left_edge, right_edge = edges[edge_idx], edges[edge_idx + 1]

        def pool_for(i):
            le, re = edges[i], edges[i + 1]
            mask = (y_all > le) & (y_all <= re)
            return X_all[mask], y_all[mask]

        want_k = max(2, int(getattr(self, 'neighbor_k', 6)))
        left_X, left_y = (pd.DataFrame(), pd.Series(dtype=float))
        right_X, right_y = (pd.DataFrame(), pd.Series(dtype=float))

        if edge_idx - 1 >= 0:
            left_X, left_y = pool_for(edge_idx - 1)
        if edge_idx + 1 <= len(edges) - 2:
            right_X, right_y = pool_for(edge_idx + 1)

        expand = 2
        while (len(left_X) + len(right_X)) < want_k and (
            (edge_idx - expand) >= 0 or (edge_idx + expand) <= len(edges) - 2
        ):
            if edge_idx - expand >= 0:
                Xp, yp = pool_for(edge_idx - expand)
                left_X = pd.concat([left_X, Xp])
                left_y = pd.concat([left_y, yp])
            if edge_idx + expand <= len(edges) - 2:
                Xp, yp = pool_for(edge_idx + expand)
                right_X = pd.concat([right_X, Xp])
                right_y = pd.concat([right_y, yp])
            expand += 1

        if (len(left_X) + len(right_X)) == 0:
            left_X, left_y = X_all.copy(), y_all.copy()

        # Ensure num_cols only includes numeric columns
        num_cols = [c for c in X_all.columns if pd.api.types.is_numeric_dtype(X_all[c])]
        cat_cols = [c for i, c in enumerate(X_all.columns)
                    if self.categorical_features is not None and i in self.categorical_features]

        local_X = (pd.concat([left_X[num_cols], right_X[num_cols]])
                if len(right_X) else left_X[num_cols])
        local_std = (local_X.std(ddof=0).replace(0, 1e-6)
                    if len(local_X) else pd.Series(1.0, index=num_cols))

        rng = np.random.default_rng(self.random_state)
        X_new, y_new = [], []
        for _ in range(n_to_generate):
            y_t = rng.uniform(left_edge, right_edge)

            def closest(pool_y, pool_X):
                if len(pool_y) == 0:
                    return None, None
                idx = (pool_y - y_t).abs().idxmin()
                return pool_X.loc[idx], pool_y.loc[idx]

            xL, yL = closest(left_y, left_X)
            xR, yR = closest(right_y, right_X)

            if xL is not None and xR is not None:
                denom = (yR - yL)
                t = 0.5 if denom == 0 else float(np.clip((y_t - yL) / denom, 0.0, 1.0))
                x_num = (1 - t) * xL[num_cols].to_numpy() + t * xR[num_cols].to_numpy()
                x_num = x_num + rng.normal(0.0, 0.05 * local_std.to_numpy())
                x_cat = {c: (xL[c] if t < 0.5 else xR[c]) for c in cat_cols}
            else:
                anchor_x, anchor_y = (xL, yL) if xL is not None else (xR, yR)
                side_X  = left_X if xL is not None else right_X
                side_y  = left_y if xL is not None else right_y
                if len(side_X) >= 2:
                    ref_idx = (side_y - anchor_y).abs().idxmax()
                    ref_x = side_X.loc[ref_idx]
                else:
                    ref_x = side_X[num_cols].mean()
                direction = anchor_x[num_cols].to_numpy() - np.array(ref_x[num_cols])
                gamma_max = float(getattr(self, 'bridge_gamma_max', 1.0))
                gamma = float(np.clip(rng.uniform(0.1, gamma_max), 0.1, gamma_max))
                x_num = anchor_x[num_cols].to_numpy() + gamma * direction
                x_num = x_num + rng.normal(0.0, 0.05 * local_std.to_numpy())
                x_cat = {c: anchor_x[c] for c in cat_cols}

            x_row = pd.Series({**{c: v for c, v in zip(num_cols, x_num)}, **x_cat})
            X_new.append(x_row.reindex(X_all.columns))
            y_new.append(y_t)

        return pd.DataFrame(X_new, columns=X_all.columns), pd.Series(y_new, name=y_all.name)
    # ------------------------------------------------------------------
    # Tomek-links based undersampling across adjacent bins
    def _tomek_links_undersample(self, X_bin, y_bin, target_samples, X_all, bins, bin_label):
        n_to_remove = len(X_bin) - target_samples
        numeric_cols = [c for i, c in enumerate(X_all.columns)
                        if self.categorical_features is None or i not in self.categorical_features]
        neighbor_labels = [l for l in np.unique(bins) if abs(l - bin_label) == 1]
        if not neighbor_labels:
            return resample(X_bin, y_bin, n_samples=target_samples, random_state=self.random_state, replace=False)

        X_neighbors = X_all[np.isin(bins, neighbor_labels)]
        X_bin_num = X_bin[numeric_cols]
        X_nei_num = X_neighbors[numeric_cols]

        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(pd.concat([X_bin_num, X_nei_num]))
        _, indices = nn.kneighbors(X_bin_num)

        tomek_links_indices = X_bin.index[~np.isin(indices.ravel(), X_bin.index)]
        if len(tomek_links_indices) > n_to_remove:
            to_remove_indices = np.random.choice(tomek_links_indices, n_to_remove, replace=False)
        else:
            to_remove_indices = list(tomek_links_indices)
            remaining = n_to_remove - len(to_remove_indices)
            if remaining > 0:
                potential_random = X_bin.index.drop(to_remove_indices)
                rand_extra = np.random.choice(potential_random, remaining, replace=False)
                to_remove_indices.extend(rand_extra)

        X_res = X_bin.drop(to_remove_indices)
        y_res = y_bin.drop(to_remove_indices)
        return X_res, y_res

    # ------------------------------------------------------------------
    # Optional visualization of before/after y (requires seaborn/matplotlib)
    def plot_distributions(self, sample_size=2000, kde=True):
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("Plotting requires matplotlib and seaborn. Please install them.")
            return
        if not hasattr(self, 'y_original_'):
            raise RuntimeError("You must call fit_resample() before plotting.")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

        # BEFORE
        sample_idx = np.random.choice(len(self.y_original_), min(len(self.y_original_), sample_size), replace=False)
        y_before = self.y_original_[sample_idx]
        bins_before = self.bins_original_[sample_idx]
        sns.histplot(x=y_before, hue=bins_before, palette='viridis', kde=kde, ax=axes[0])
        axes[0].set_title(f"Before Resampling (n={len(self.y_original_)})")
        axes[0].set_xlabel("Target Value")
        axes[0].get_legend().set_title("Bin")

        # AFTER
        sample_idx_after = np.random.choice(len(self.y_resampled_), min(len(self.y_resampled_), sample_size), replace=False)
        y_after_raw = self.y_resampled_[sample_idx_after]
        finite_mask = np.isfinite(y_after_raw)
        y_after = y_after_raw[finite_mask]
        if len(y_after) == 0:
            print("Warning: No finite values in resampled y for plotting.")
            return
        sorted_edges = np.sort(self.bin_edges_)
        indices = np.searchsorted(sorted_edges, y_after, side='right')
        bins_after = np.clip(indices - 1, 0, len(sorted_edges) - 2)
        sns.histplot(x=y_after, hue=bins_after, palette='viridis', kde=kde, ax=axes[1])
        axes[1].set_title(f"After Resampling (n={len(self.y_resampled_)})")
        axes[1].set_xlabel("Target Value")
        axes[1].get_legend().set_title("Bin")

        plt.suptitle("Target Distribution Before and After Balancing", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        axes[0].get_legend().remove()
        axes[1].get_legend().remove()
        plt.savefig('plots/test.jpg', dpi=300)

    # ------------------------------------------------------------------
    # Main entry point
    def fit_resample(self, X, y):
        print("Starting balancing resampling...")
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X, columns=[f'f_{i}' for i in range(X.shape[1])])
        else:
            X_df = X.copy()
        y_s = pd.Series(y)

        # store originals for plotting
        self.y_original_ = y_s.to_numpy()

        n_bins_actual = self._get_n_bins(y_s)
        bins, self.bin_edges_ = self._create_bins(y_s, n_bins_actual)
        self.bins_original_ = bins.to_numpy()

        bin_counts = pd.Series(bins).value_counts()
        print(f"Using {n_bins_actual} bins with policy '{self.binning_policy}'.")
        print(f"Original bin counts:\n{bin_counts.sort_index()}")

        min_samples = bin_counts.min() if not bin_counts.empty else 0
        max_samples = bin_counts.max() if not bin_counts.empty else 0
        target_samples = int(min_samples + (max_samples - min_samples) * self.max_diff_ratio)
        print(f"Target samples per bin: {target_samples}")

        resampled_X, resampled_y = [], []

        for bin_label in bin_counts.index:
            bin_mask = (bins == bin_label)
            X_bin, y_bin = X_df[bin_mask], y_s[bin_mask]
            n_in_bin = len(X_bin)

            # ---------------------------
            # Undersampling (if needed)
            if n_in_bin > target_samples:
                print(f"Undersampling bin {bin_label} from {n_in_bin} to {target_samples} with policy '{self.undersample_policy}'...")
                if self.undersample_policy == 'tomek_links':
                    X_res, y_res = self._tomek_links_undersample(X_bin, y_bin, target_samples, X_df, bins, bin_label)
                elif self.undersample_policy == 'random':
                    X_res, y_res = resample(X_bin, y_bin, n_samples=target_samples, random_state=self.random_state, replace=False)
                elif self.undersample_policy == 'outlier':
                    numeric_cols = [c for i, c in enumerate(X_bin.columns)
                                    if self.categorical_features is None or i not in self.categorical_features]
                    X_num = X_bin[numeric_cols]
                    iso = IsolationForest(random_state=self.random_state)
                    scores = iso.fit(X_num).decision_function(X_num)
                    keep_idx = X_bin.index[np.argsort(scores)[-target_samples:]]
                    X_res, y_res = X_bin.loc[keep_idx], y_bin.loc[keep_idx]
                elif self.undersample_policy == 'inlier':
                    numeric_cols = [c for i, c in enumerate(X_bin.columns)
                                    if self.categorical_features is None or i not in self.categorical_features]
                    X_num = X_bin[numeric_cols]
                    iso = IsolationForest(random_state=self.random_state)
                    scores = iso.fit(X_num).decision_function(X_num)
                    keep_idx = X_bin.index[np.argsort(scores)[:target_samples]]
                    X_res, y_res = X_bin.loc[keep_idx], y_bin.loc[keep_idx]
                else:
                    raise ValueError(f"Unknown undersample_policy: {self.undersample_policy}")

            # ---------------------------
            # Oversampling (if needed)
            elif n_in_bin < target_samples and n_in_bin > 3:
                print(f"Oversampling bin {bin_label} from {n_in_bin} to {target_samples} with policy '{self.oversample_policy}'...")
                n_to_generate = target_samples - n_in_bin

                if self.oversample_policy == 'generalized_smote':
                    X_syn, y_syn = self._generalized_smote_oversample(X_bin, y_bin, n_to_generate)
                elif self.oversample_policy == 'density_smote':
                    X_syn, y_syn = self._density_smote_oversample(X_bin, y_bin, n_to_generate)
                elif self.oversample_policy in ['generalized_smotenc', 'density_smotenc']:
                    X_syn, y_syn = self._advanced_smote_oversample(X_bin, y_bin, n_to_generate, self.oversample_policy)
                else:  # SMOTE / SMOTER / SMOTENC
                    k_neighbors = min(self.smote_k_neighbors, n_in_bin - 1)
                    if k_neighbors < 1:
                        X_syn = X_bin.sample(n=n_to_generate, replace=True, random_state=self.random_state)
                        y_syn = pd.Series(np.random.choice(y_bin, size=len(X_syn)), name=y_s.name)
                    else:
                        other_mask = ~bin_mask
                        X_for_smote = pd.concat([X_bin, X_df[other_mask].head(1)])
                        y_dummy = np.array([0] * n_in_bin + [1])

                        if self.oversample_policy == 'smotenc':
                            if self.categorical_features is None:
                                raise ValueError("`categorical_features` must be specified for 'smotenc' policy.")
                            sampler = SMOTENC(sampling_strategy={0: target_samples},
                                              categorical_features=self.categorical_features,
                                              k_neighbors=k_neighbors,
                                              random_state=self.random_state)
                        else:  # 'smote' or 'smoter'
                            sampler = SMOTE(sampling_strategy={0: target_samples},
                                            k_neighbors=k_neighbors,
                                            random_state=self.random_state)

                        X_resampled_smote, _ = sampler.fit_resample(X_for_smote, y_dummy)
                        X_syn_all = pd.DataFrame(X_resampled_smote, columns=X_df.columns).iloc[:target_samples]
                        X_syn = X_syn_all.iloc[n_in_bin:]

                        if self.oversample_policy in ['smoter', 'smotenc']:
                            y_syn = self._smoter_interpolate_y(X_bin, y_bin, X_syn)
                        elif self.oversample_policy == 'smote':
                            y_syn = pd.Series(np.random.choice(y_bin, size=len(X_syn)), name=y_s.name)

                X_res = pd.concat([X_bin, X_syn])
                y_res = pd.concat([y_bin, y_syn])

            else:
                if n_in_bin <= 3:
                    if getattr(self, 'low_count_policy', 'neighbor_bridge') == 'neighbor_bridge':
                        sorted_edges = np.sort(self.bin_edges_)
                        if n_in_bin > 0:
                            y_ref = y_bin.median()
                        else:
                            centers = (sorted_edges[:-1] + sorted_edges[1:]) / 2.0
                            y_ref = float(np.median(y_s))
                            y_ref = centers[np.argmin(np.abs(centers - y_ref))]
                        i_edge = int(np.clip(np.searchsorted(sorted_edges, y_ref, side='right') - 1, 0, len(sorted_edges) - 2))
                        n_to_generate = max(0, target_samples - n_in_bin)
                        if n_to_generate > 0:
                            X_syn, y_syn = self._neighbor_bridge_generate(X_df, y_s, sorted_edges, i_edge, n_to_generate)
                            X_res = pd.concat([X_bin, X_syn])
                            y_res = pd.concat([y_bin, y_syn])
                        else:
                            X_res, y_res = X_bin, y_bin
                    else:
                        print(f"Skipping oversampling for bin {bin_label}: too few samples ({n_in_bin}).")
                        X_res, y_res = X_bin, y_bin
                else:
                    X_res, y_res = X_bin, y_bin

            resampled_X.append(X_res)
            resampled_y.append(y_res)

        # Fill completely empty edge-interval bins if any
        sorted_edges = np.sort(self.bin_edges_)
        edge_ranks = np.searchsorted(sorted_edges, y_s, side='right') - 1
        counts_by_edge = pd.Series(edge_ranks).value_counts()
        for i_edge in range(len(sorted_edges) - 1):
            if counts_by_edge.get(i_edge, 0) == 0:
                X_syn, y_syn = self._neighbor_bridge_generate(X_df, y_s, sorted_edges, i_edge, target_samples)
                resampled_X.append(X_syn)
                resampled_y.append(y_syn)

        X_final = pd.concat(resampled_X, ignore_index=True)
        y_final = pd.concat(resampled_y, ignore_index=True)

        self.y_resampled_ = y_final.to_numpy()
        print("Balancing resampling complete.")
        return X_final, y_final
