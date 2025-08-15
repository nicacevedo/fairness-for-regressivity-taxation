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
    A custom preprocessor to re-balance a dataset based on a continuous
    target variable (y).

    The resampler works by first binning the data based on the y values,
    and then applying over-sampling and under-sampling techniques
    to each bin to bring them to a more uniform size.

    Parameters
    ----------
    n_bins : int or 'auto', default=5
        The number of bins to split the target variable `y` into.
        - If 'auto', uses the Freedman-Diaconis rule to determine the
          optimal number of bins.

    binning_policy : {'uniform', 'quantile', 'outlier', 'kmeans', 'decision_tree'}, default='uniform'
        The policy for creating bins:
        - 'uniform': Creates bins of equal width across the range of `y`.
        - 'quantile': (Recommended Default) Creates bins with an equal number
          of samples in each. This is often the most stable choice.
        - 'outlier': Uses an Isolation Forest on `y` to create bins based on
          outlier scores. Useful if the primary goal is to handle outliers.
        - 'kmeans': Uses K-Means clustering on the 1D `y` values to find
          natural groupings.
        - 'decision_tree': Fits a Decision Tree Regressor on `y` and uses its
          leaf nodes as the bins. Creates bins of high internal similarity.

    max_diff_ratio : float, default=0.5
        Controls the target size for each bin after resampling. The target is:
        `min_samples + (max_samples - min_samples) * max_diff_ratio`.
        - 0.0 aims for all bins to match the smallest bin's size.
        - 1.0 aims for all bins to match the largest bin's size.
        - 0.5 is a balanced compromise.

    undersample_policy : {'random', 'outlier', 'tomek_links'}, default='random'
        The policy for under-sampling bins that are too large:
        - 'random': Randomly removes samples. Fast and simple.
        - 'outlier': Intelligently removes samples that are feature-space
          outliers within their bin, effectively cleaning noise.
        - 'tomek_links': Removes samples from the majority bin that are nearest
          neighbors to samples in adjacent minority bins, cleaning the
          decision boundary between bins.

    oversample_policy : {'smote', 'smoter', 'smotenc', 'generalized_smote', 'density_smote'}, default='smoter'
        The policy for over-sampling bins that are too small:
        - 'smote': Standard SMOTE. Duplicates target values.
        - 'smoter': SMOTE for Regression. Interpolates new target values.
        - 'smotenc': SMOTE for mixed categorical/continuous features.
        - 'generalized_smote': Creates a new sample from a random convex
          combination of a subsample of points.
        - 'density_smote': An ADASYN-like approach that generates more samples
          in sparser regions of the feature space.

    generalized_smote_weighting : {'random', 'gravity'}, default='random'
        Weighting strategy for the 'generalized_smote' policy.
        - 'random': Uses random weights for the convex combination.
        - 'gravity': Gives more weight to points closer to the local
          centroid, creating more "typical" samples.

    smote_k_neighbors : int, default=5
        The `k_neighbors` parameter for SMOTE-based algorithms.
        
    categorical_features : list of int or None, default=None
        Specifies which features are categorical. Must be a list of column
        indices. This is used by 'smotenc' and now also by the advanced
        undersampling policies to handle mixed data types.

    random_state : int, default=None
        Seed for reproducibility of all random processes.
    """
    def __init__(self, n_bins=5, binning_policy='quantile', max_diff_ratio=0.5,
                 undersample_policy='outlier', oversample_policy='smoter',
                 generalized_smote_weighting='random', smote_k_neighbors=5, 
                 categorical_features=None, random_state=None):
        self.n_bins = n_bins
        self.binning_policy = binning_policy
        self.max_diff_ratio = max_diff_ratio
        self.undersample_policy = undersample_policy
        self.oversample_policy = oversample_policy
        self.generalized_smote_weighting = generalized_smote_weighting
        self.smote_k_neighbors = smote_k_neighbors
        self.categorical_features = categorical_features
        self.random_state = random_state
        if self.random_state:
            np.random.seed(self.random_state)

    def fit(self, X, y):
        """No-op fit method to be compatible with scikit-learn pipelines."""
        return self

    def _get_n_bins(self, y):
        """Calculates number of bins if n_bins is 'auto'."""
        if self.n_bins != 'auto':
            return self.n_bins
        
        iqr = np.subtract(*np.percentile(y, [75, 25]))
        bin_width = 2 * iqr * (len(y) ** (-1/3))
        if bin_width == 0: return 10
        num_bins = int(np.ceil((y.max() - y.min()) / bin_width))
        return max(2, min(num_bins, 50))

    def _create_bins(self, y, n_bins_actual):
        """Helper function to create bins based on the chosen policy."""
        y_np = y.to_numpy()
        if self.binning_policy == 'uniform':
            return pd.cut(y, n_bins_actual, labels=False, duplicates='drop')
        elif self.binning_policy == 'quantile':
            return pd.qcut(y, n_bins_actual, labels=False, duplicates='drop')
        elif self.binning_policy == 'outlier':
            iso = IsolationForest(random_state=self.random_state)
            scores = iso.fit_predict(y_np.reshape(-1, 1))
            return pd.qcut(scores, n_bins_actual, labels=False, duplicates='drop')
        elif self.binning_policy == 'kmeans':
            kmeans = KMeans(n_clusters=n_bins_actual, random_state=self.random_state, n_init=10)
            return kmeans.fit_predict(y_np.reshape(-1, 1))
        elif self.binning_policy == 'decision_tree':
            tree = DecisionTreeRegressor(max_leaf_nodes=n_bins_actual, random_state=self.random_state)
            tree.fit(y_np.reshape(-1, 1), y)
            return tree.apply(y_np.reshape(-1, 1))
        else:
            raise ValueError(f"Unknown binning_policy: {self.binning_policy}")

    def _smoter_interpolate_y(self, X_orig, y_orig, X_synthetic):
        """Interpolates y values for synthetic samples for SMOTER."""
        numeric_cols = [c for i, c in enumerate(X_orig.columns) if self.categorical_features is None or i not in self.categorical_features]
        X_orig_numeric = X_orig[numeric_cols]
        X_synthetic_numeric = X_synthetic[numeric_cols]
        
        k = min(self.smote_k_neighbors, len(X_orig_numeric) - 1)
        if k < 1: return pd.Series()
        
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors.fit(X_orig_numeric)
        distances, indices = neighbors.kneighbors(X_synthetic_numeric)
        
        weights = 1.0 / (distances + 1e-6)
        weights = weights / np.sum(weights, axis=1, keepdims=True)
        y_neighbors = y_orig.to_numpy()[indices]
        interpolated_y = np.sum(weights * y_neighbors, axis=1)
        return pd.Series(interpolated_y)

    def _generalized_smote_oversample(self, X_bin, y_bin, n_to_generate):
        """Creates synthetic samples using a convex combination of a subsample."""
        new_X_list, new_y_list = [], []
        subsample_size = self.smote_k_neighbors + 1
        
        numeric_cols = [c for i, c in enumerate(X_bin.columns) if self.categorical_features is None or i not in self.categorical_features]
        cat_cols = [c for i, c in enumerate(X_bin.columns) if self.categorical_features is not None and i in self.categorical_features]

        for _ in range(n_to_generate):
            subsample_indices = np.random.choice(X_bin.index, size=min(subsample_size, len(X_bin)), replace=False)
            X_subsample = X_bin.loc[subsample_indices]
            y_subsample = y_bin.loc[subsample_indices]

            if self.generalized_smote_weighting == 'random':
                weights = np.random.rand(len(X_subsample))
            elif self.generalized_smote_weighting == 'gravity':
                centroid = X_subsample[numeric_cols].mean(axis=0)
                distances = np.linalg.norm(X_subsample[numeric_cols] - centroid, axis=1)
                weights = 1.0 / (distances + 1e-6)
            else:
                raise ValueError(f"Unknown generalized_smote_weighting: {self.generalized_smote_weighting}")
            weights /= weights.sum()

            new_X_numeric = np.dot(weights, X_subsample[numeric_cols])
            new_y = np.dot(weights, y_subsample)
            new_sample = pd.Series(new_X_numeric, index=numeric_cols)
            
            for col in cat_cols:
                new_sample[col] = X_subsample[col].mode().iloc[0]

            new_X_list.append(new_sample)
            new_y_list.append(new_y)
            
        return pd.DataFrame(new_X_list), pd.Series(new_y_list)
    
    def _density_smote_oversample(self, X_bin, y_bin, n_to_generate):
        """ADASYN-like oversampling focusing on sparser regions."""
        numeric_cols = [c for i, c in enumerate(X_bin.columns) if self.categorical_features is None or i not in self.categorical_features]
        X_bin_numeric = X_bin[numeric_cols]

        k = min(self.smote_k_neighbors, len(X_bin_numeric) - 1)
        if k < 1: return pd.DataFrame(), pd.Series()

        nn = NearestNeighbors(n_neighbors=k + 1)
        nn.fit(X_bin_numeric)
        distances, _ = nn.kneighbors(X_bin_numeric)
        
        density = 1.0 / (distances[:, 1:].mean(axis=1) + 1e-6)
        selection_prob = 1.0 / density
        selection_prob /= selection_prob.sum()

        return self._generalized_smote_oversample(X_bin, y_bin, n_to_generate)

    def _tomek_links_undersample(self, X_bin, y_bin, target_samples, X_all, bins, bin_label):
        """Undersamples by removing Tomek Links at bin boundaries."""
        n_to_remove = len(X_bin) - target_samples
        
        numeric_cols = [c for i, c in enumerate(X_all.columns) if self.categorical_features is None or i not in self.categorical_features]
        
        neighbor_labels = [l for l in np.unique(bins) if abs(l - bin_label) == 1]
        if not neighbor_labels:
            return resample(X_bin, y_bin, n_samples=target_samples, random_state=self.random_state, replace=False)
        
        X_neighbors = X_all[np.isin(bins, neighbor_labels)]
        
        # Perform NN search only on numeric columns
        X_bin_numeric = X_bin[numeric_cols]
        X_neighbors_numeric = X_neighbors[numeric_cols]
        
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(pd.concat([X_bin_numeric, X_neighbors_numeric]))
        
        _, indices = nn.kneighbors(X_bin_numeric)
        
        tomek_links_indices = X_bin.index[~np.isin(indices.ravel(), X_bin.index)]
        
        if len(tomek_links_indices) > n_to_remove:
            to_remove_indices = np.random.choice(tomek_links_indices, n_to_remove, replace=False)
        else:
            to_remove_indices = list(tomek_links_indices)
            remaining_to_remove = n_to_remove - len(to_remove_indices)
            if remaining_to_remove > 0:
                potential_random = X_bin.index.drop(to_remove_indices)
                random_indices = np.random.choice(potential_random, remaining_to_remove, replace=False)
                to_remove_indices.extend(random_indices)

        X_res = X_bin.drop(to_remove_indices)
        y_res = y_bin.drop(to_remove_indices)
        return X_res, y_res

    def fit_resample(self, X, y):
        """Resamples the dataset X and y."""
        print("Starting balancing resampling...")
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X, columns=[f'f_{i}' for i in range(X.shape[1])])
        else:
            X_df = X.copy()
        y_s = pd.Series(y)

        n_bins_actual = self._get_n_bins(y_s)
        bins = self._create_bins(y_s, n_bins_actual)
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
            n_samples_in_bin = len(X_bin)

            if n_samples_in_bin > target_samples:
                print(f"Undersampling bin {bin_label} from {n_samples_in_bin} to {target_samples} with policy '{self.undersample_policy}'...")
                if self.undersample_policy == 'tomek_links':
                    X_res, y_res = self._tomek_links_undersample(X_bin, y_bin, target_samples, X_df, bins, bin_label)
                elif self.undersample_policy == 'random':
                    X_res, y_res = resample(X_bin, y_bin, n_samples=target_samples, random_state=self.random_state, replace=False)
                elif self.undersample_policy == 'outlier':
                    numeric_cols = [c for i, c in enumerate(X_bin.columns) if self.categorical_features is None or i not in self.categorical_features]
                    X_bin_numeric = X_bin[numeric_cols]
                    iso = IsolationForest(random_state=self.random_state)
                    scores = iso.fit(X_bin_numeric).decision_function(X_bin_numeric)
                    keep_indices = X_bin.index[np.argsort(scores)[-target_samples:]]
                    X_res, y_res = X_bin.loc[keep_indices], y_bin.loc[keep_indices]
                else:
                    raise ValueError(f"Unknown undersample_policy: {self.undersample_policy}")
            
            elif n_samples_in_bin < target_samples and n_samples_in_bin > 3:
                print(f"Oversampling bin {bin_label} from {n_samples_in_bin} to {target_samples} with policy '{self.oversample_policy}'...")
                n_to_generate = target_samples - n_samples_in_bin
                
                if self.oversample_policy == 'generalized_smote':
                    X_synthetic, y_synthetic = self._generalized_smote_oversample(X_bin, y_bin, n_to_generate)
                elif self.oversample_policy == 'density_smote':
                    X_synthetic, y_synthetic = self._density_smote_oversample(X_bin, y_bin, n_to_generate)
                else: # Handle SMOTE, SMOTER, SMOTENC
                    k_neighbors = min(self.smote_k_neighbors, n_samples_in_bin - 1)
                    other_samples_mask = ~bin_mask
                    if np.any(other_samples_mask):
                        X_other = X_df[other_samples_mask].head(1)
                        X_for_smote = pd.concat([X_bin, X_other])
                        y_dummy = np.array([0] * n_samples_in_bin + [1])
                        
                        sampler = None
                        if self.oversample_policy == 'smotenc':
                            if self.categorical_features is None: raise ValueError("`categorical_features` must be specified for 'smotenc' policy.")
                            sampler = SMOTENC(sampling_strategy={0: target_samples}, categorical_features=self.categorical_features, k_neighbors=k_neighbors, random_state=self.random_state)
                        else:
                            sampler = SMOTE(sampling_strategy={0: target_samples}, k_neighbors=k_neighbors, random_state=self.random_state)
                        
                        X_resampled_smote, _ = sampler.fit_resample(X_for_smote, y_dummy)
                        X_synthetic_all = X_resampled_smote.iloc[:target_samples]
                        X_synthetic = X_synthetic_all.iloc[n_samples_in_bin:]
                    else:
                        X_synthetic = X_bin.sample(n=n_to_generate, replace=True, random_state=self.random_state)

                    if self.oversample_policy in ['smoter', 'smotenc']:
                        y_synthetic = self._smoter_interpolate_y(X_bin, y_bin, X_synthetic)
                    elif self.oversample_policy == 'smote':
                        y_synthetic = pd.Series(np.random.choice(y_bin, size=len(X_synthetic)), name=y_s.name)
                
                X_res = pd.concat([X_bin, X_synthetic])
                y_res = pd.concat([y_bin, y_synthetic])
            else:
                if n_samples_in_bin <= 3 and n_samples_in_bin > 0:
                    print(f"Skipping oversampling for bin {bin_label}: too few samples ({n_samples_in_bin}).")
                X_res, y_res = X_bin, y_bin

            resampled_X.append(X_res)
            resampled_y.append(y_res)

        X_final = pd.concat(resampled_X, ignore_index=True)
        y_final = pd.concat(resampled_y, ignore_index=True)

        print("Balancing resampling complete.")
        return X_final, y_final
