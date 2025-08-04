import numpy as np
from scipy.linalg import pinvh

def robust_nan_mahalanobis(X, regularize=1e-5, min_obs=5, missing_values=np.nan):
    """
    Compute pairwise Mahalanobis distances robust to NaNs.
    Covariance matrix is computed dynamically over valid dimensions and rows.

    Parameters:
    - X: np.ndarray (n_samples, n_features), can contain NaNs
    - regularize: float, added to diag of covariance to avoid singularity
    - min_obs: minimum number of valid rows to estimate covariance
    
    Returns:
    - distances: np.ndarray (n_samples, n_samples)
    """
    print("Inside of robust maha")
    print(X)
    print(X.shape)
    n_samples, n_features = X.shape
    distances = np.full((n_samples, n_samples), np.nan)

    for i in range(n_samples):
        xi = X[i]
        for j in range(i, n_samples):
            xj = X[j]
            valid_mask = ~np.isnan(xi) & ~np.isnan(xj)
            m = np.sum(valid_mask)
            if m == 0:
                print("No valid dimensions on: ", (i,j))
                continue  # No valid dimensions
            # Subvectors over valid dimensions
            xi_sub = xi[valid_mask]
            xj_sub = xj[valid_mask]

            # Filter rows with valid entries in dimensions M
            X_sub = X[:, valid_mask]
            valid_rows_mask = ~np.isnan(X_sub).any(axis=1)
            X_valid = X_sub[valid_rows_mask]

            if X_valid.shape[0] < min_obs:
                print("Not enough data to estimate covariance on: ", (i,j))
                continue  # Not enough data to estimate covariance

            # Covariance and inverse
            cov = np.cov(X_valid, rowvar=False) # columns are the variables, and rows the obs
            cov += regularize * np.eye(m) # add regularization term to avoid non-invertible problems and it is a more robust approximation of the covariance matrix
            try:
                VI = pinvh(cov)
            except np.linalg.LinAlgError:
                print("Inversion failed on: ", (i,j))
                continue  # Skip if inversion fails

            diff = xi_sub - xj_sub
            dist = np.sqrt(diff @ VI @ diff)
            dist *= np.sqrt(n_features / m) # compensation for missing dimensions (re-scaling). Like in nan_euclidean dist.
            distances[i, j] = distances[j, i] = dist

    return distances
