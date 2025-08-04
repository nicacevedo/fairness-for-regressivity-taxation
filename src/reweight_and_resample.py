import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


# Bins related functions
def get_uniform_bins(vector_range=(0,100), n_bins=10):
    # Distribution of each one
    y_min, y_max = vector_range
    # print("range: ", y_min, y_max)
    # B =  30#int((y_max - y_min)/330000)+1 # number of bins
    bins = np.linspace(y_min, y_max, n_bins+1)
    # print(f"{n_bins} bins: ")
    # print(bins)
    # lb = bins[0]
    return bins



def compute_bin_edges(data, num_bins):
    """
    Compute bin edges similar to matplotlib.pyplot.hist behavior.
    
    Parameters:
    - data: array-like, the input data
    - num_bins: int, the number of bins

    Returns:
    - bin_edges: ndarray of bin edges
    """
    data = np.asarray(data)

    if data.size == 0:
        raise ValueError("Input data must not be empty.")
    
    data_min, data_max = np.min(data), np.max(data)

    if data_min == data_max:
        # Create small range around the constant value
        data_min -= 0.5
        data_max += 0.5

    bin_edges = np.linspace(data_min, data_max, num_bins + 1)

    return bin_edges

def get_group_weights(vector, bins, alpha=1, weight_type="raw"):
    """
        Let the weights first be the size of each group for simplicity
        Let intervals be: [lb, ub)
    """
    weights = []
    lb = bins[0] # first lower bound
    i = 0
    for ub in bins[1:]:
        if i >= len(bins) -2: # one for python and one for extra bound (last interval)
            vector_group_ids = (lb <= vector) & (vector <= ub)
        else:
            vector_group_ids = (lb <= vector) & (vector < ub)
        group_size = np.sum(vector_group_ids)
        if group_size > 0:
            if weight_type == "raw": # Directly use the number of samples of the bin as weight
                weights+= [1/group_size**alpha]*group_size # inverse of size repeated the number of samples times
        i+=1
        lb = ub # update lb
    return weights


def get_bins_from_edges(vector, bin_edges):
    if type(vector) == pd.Series:
        vector = vector.to_numpy()
    bins = []
    lb = bin_edges[0]
    for i,ub in enumerate(bin_edges[1:]):
        vector_i = vector[(vector >= lb) & (vector < ub)]
        if vector_i.size == 0:
            print("No size in: ", i)
        bins.append(vector_i)
        lb = ub
    bins[i] =  np.concatenate((bins[i], vector[vector == ub]))
    return bins
        


def get_bin_indices_from_edges(vector, bin_edges):
    if type(vector) == pd.Series:
        vector = vector.to_numpy()
    bins = list()
    lb = bin_edges[0]
    for i,ub in enumerate(bin_edges[1:]):
        bin_indices = np.where((vector >= lb) & (vector < ub))[0]
        # if vector_i.size == 0:
            # print("No size in: ", i)
        bins.append(bin_indices)
        lb = ub
    bins[i] =  np.concatenate((bins[i], np.where(vector == ub)[0]))
    return bins



# Classes


class OutlierSmoteResampler:
    def __init__(self, bin_indices, k_neighbors=5, metric="minkowski", p=2, metric_params=None, random_state=None, bin_size_ratio=0.5, undersampling_policy="random"):
        self.bin_indices = bin_indices
        self.k_neighbors = k_neighbors
        self.metric = metric
        self.p = p 
        self.metric_params = metric_params
        self.random_state = np.random.RandomState(random_state)
        self.bin_size_ratio = bin_size_ratio# r: ratio of (max-min) to add to each bin (or to cut in larger bins)
        self.undersampling_policy = undersampling_policy # either delete random samples or by outlier score in the bin
    
    def _assign_bins(self, y):
        y_bins = np.zeros(y.size)
        for i, bin_indices_i in enumerate(self.bin_indices):
            y_bins[bin_indices_i] = i
        return y_bins #np.digitize(y, self.bin_edges, right=False) - 1

    def _compute_bin_counts(self, y_bins):
        bin_counts = []
        for i, bin_indices_i in enumerate(self.bin_indices):
            bin_counts.append(bin_indices_i.size)
        # bin_counts =  #np.bincount(y_bins, minlength=len(self.bin_edges) - 1)
        return np.array(bin_counts)

    def _generate_samples(self, X_bin, y_bin, n_samples):
        if len(X_bin) <= 1:
            return X_bin, y_bin  # Can't interpolate

        nn = NearestNeighbors(n_neighbors=min(self.k_neighbors, len(X_bin)), metric=self.metric, p=self.p, metric_params=self.metric_params)
        nn.fit(X_bin)
        indices = self.random_state.randint(0, len(X_bin), size=n_samples)

        X_synthetic = []
        y_synthetic = []

        for idx in indices:
            x_i = X_bin[idx]
            y_i = y_bin[idx]

            neighbors = nn.kneighbors([x_i], return_distance=False)[0]
            neighbor_idx = self.random_state.choice(neighbors[neighbors != idx])

            x_n = X_bin[neighbor_idx]
            y_n = y_bin[neighbor_idx]

            alpha = self.random_state.rand()
            x_new = x_i + alpha * (x_n - x_i)
            y_new = y_i + alpha * (y_n - y_i)

            X_synthetic.append(x_new)
            y_synthetic.append(y_new)

        return np.array(X_synthetic), np.array(y_synthetic)
    
    def _under_samples(self, X_bin, y_bin, n_samples):
        
        if self.undersampling_policy == "random":
            # Choose n_samples random observations
            synthetic_indices = np.random.choice(range(y_bin.size), size=n_samples, replace=False)

        elif self.undersampling_policy == "outlier_score":
            # Choose the n_samples obs with higher inlier score
            iso = IsolationForest(
                n_estimators=100,
                contamination='auto',  # or a float like 0.05 if you know the expected outlier fraction
                random_state=42
            )
            iso.fit(X_bin)
            inlier_score = iso.decision_function(X_bin)
            synthetic_indices = np.argsort(inlier_score)[-n_samples:]
            
        X_synthetic = X_bin[synthetic_indices,:]
        y_synthetic = y_bin[synthetic_indices]
        
        return X_synthetic, y_synthetic
        

    def fit_resample(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        
        y_bins = self._assign_bins(y)
        bin_counts = self._compute_bin_counts(y_bins)
        max_count = max(bin_counts)
        min_count = min(bin_counts[bin_counts>0])
        objective_bin_size = min_count + int( (max_count - min_count) * self.bin_size_ratio )
        print("objective_bin_size", objective_bin_size)

        X_resampled = [X]
        y_resampled = [y]

        for bin_idx in range(len(bin_counts)):
            count = bin_counts[bin_idx]
            print("Bin: ", bin_idx, " | bin count: ", count) 
            # if count == 0 or count == max_count:
            if count < min(self.k_neighbors, 5) or count == max_count:
                print("skipped")
                continue  # skip (almost) empty or already balanced bins
            
            # Filter bin's datapoints
            mask = y_bins == bin_idx
            X_bin = X[mask]
            y_bin = y[mask]
            
            
            # Check if it is larger (under-sample) or smaller (over-sample) than the desired size
            if count > objective_bin_size:
                n_to_sample = objective_bin_size
                X_synth, y_synth = self._under_samples(X_bin, y_bin, n_to_sample)
                print("Undersampled to: ", y_synth.size)
                
            else:
                n_to_sample = objective_bin_size
                X_synth, y_synth = self._generate_samples(X_bin, y_bin, n_to_sample)
                print("Oversampled to: ", y_synth.size)

            X_resampled.append(X_synth)
            y_resampled.append(y_synth)

        return np.vstack(X_resampled), np.concatenate(y_resampled)
