"""LightGBM regressor with an epsilon-soft penalty on Cov(ratio, z).

- Trains in log-space: f(x) \approx log(y)
- Predicts in original space: y_hat = exp(f)
- Ratio: r_i = y_hat_i / y_i
- Penalty: rho * softplus((|Cov_w(r, z)| - eps)/tau)
  where z is either log(y) or y.

This is intended as a sklearn-like wrapper around `lightgbm.train`.

Notes
-----
- Requires y > 0 (so log is defined).
- Uses a diagonal Hessian approximation (base squared-error Hessian only) for stability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

try:
    import lightgbm as lgb
except ImportError as e:  # pragma: no cover
    raise ImportError("This module requires lightgbm. Install via `pip install lightgbm`.") from e

try:
    from sklearn.base import BaseEstimator, RegressorMixin
except ImportError as e:  # pragma: no cover
    raise ImportError("This module requires scikit-learn. Install via `pip install scikit-learn`.") from e


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


@dataclass
class CovPenaltyConfig:
    """Configuration for the covariance penalty.

    Parameters
    ----------
    rho : float
        Penalty weight.
    eps : float
        Tolerance for |Cov|. Penalty activates smoothly when |Cov| > eps.
    tau : float
        Smoothness for the soft hinge (softplus). Smaller -> closer to max(0, |Cov|-eps).
    delta : float
        Smoothing for absolute value: |C| ~ sqrt(C^2 + delta).
    z_mode : str
        Either "logy" (default) or "y"; determines z used in Cov(r, z).
    """

    rho: float = 1.0
    eps: float = 0.0
    tau: float = 1e-3
    delta: float = 1e-12
    z_mode: str = "logy"  # "logy" or "y"


class LGBMCovRatioRegressor(BaseEstimator, RegressorMixin):
    """Sklearn-like LightGBM regressor with a covariance penalty on ratios.

    Objective (in log-space)
    ------------------------
    Base loss: 0.5 * sum_i (f_i - log y_i)^2

    Ratio: r_i = exp(f_i) / y_i

    Weighted covariance (simplified by centering z):
        C = E_w[r * (z - E_w[z])]

    Penalty:
        rho * softplus((|C| - eps)/tau)

    The custom objective supplies per-instance gradients and a stable diagonal Hessian.
    """

    def __init__(
        self,
        lgb_params: Optional[Dict[str, Any]] = None,
        num_boost_round: int = 2000,
        early_stopping_rounds: Optional[int] = 200,
        penalty: CovPenaltyConfig = CovPenaltyConfig(),
        random_state: int = 0,
        verbose_eval: Union[bool, int] = False,
    ):
        self.lgb_params = lgb_params or {}
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.penalty = penalty
        self.random_state = random_state
        self.verbose_eval = verbose_eval

        self._booster: Optional[lgb.Booster] = None

    def fit(
        self,
        X,
        y,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        sample_weight: Optional[np.ndarray] = None,
        eval_sample_weight: Optional[np.ndarray] = None,
    ):
        y = np.asarray(y, dtype=float)
        if np.any(y <= 0):
            raise ValueError("All y must be > 0 when training in log-space and using ratios.")

        y_log = np.log(y)

        train_data = lgb.Dataset(X, label=y_log, weight=sample_weight, free_raw_data=False)
        valid_sets = [train_data]
        valid_names = ["train"]

        if eval_set is not None:
            Xv, yv = eval_set
            yv = np.asarray(yv, dtype=float)
            if np.any(yv <= 0):
                raise ValueError("All eval y must be > 0.")
            valid_data = lgb.Dataset(
                Xv,
                label=np.log(yv),
                weight=eval_sample_weight,
                free_raw_data=False,
            )
            valid_sets.append(valid_data)
            valid_names.append("valid")

        params = {
            "objective": "regression",  # overridden by custom fobj
            "metric": "l2",
            "verbosity": -1,
            "seed": self.random_state,
        }
        params.update(self.lgb_params)

        pen = self.penalty

        def fobj(preds: np.ndarray, dataset: lgb.Dataset):
            # preds are f in log-space
            y_log_true = dataset.get_label().astype(float)
            y_true = np.exp(y_log_true)
            y_hat = np.exp(preds)

            ratio = y_hat / y_true

            if pen.z_mode == "y":
                z = y_true
            elif pen.z_mode == "logy":
                z = y_log_true
            else:
                raise ValueError("penalty.z_mode must be either 'y' or 'logy'.")

            # weights
            w = dataset.get_weight()
            if w is None:
                w = np.ones_like(z)
            w = np.asarray(w, dtype=float)
            w_sum = float(np.sum(w))
            if w_sum <= 0:
                raise ValueError("Sum of weights must be positive.")

            # center z
            z_bar = float(np.sum(w * z) / w_sum)
            z_c = z - z_bar

            # weighted covariance C = E_w[ ratio * z_c ]
            # C = float(np.sum(w * ratio * z_c) / w_sum)

            # smooth |C|
            # s = float(np.sqrt(C * C + pen.delta))

            # soft hinge (via softplus)
            # t = (s - pen.eps) / pen.tau
            # sig = float(_sigmoid(np.array([t]))[0])

            # base gradients/hessians for 0.5*(pred-logy)^2
            grad_base = 2 * (preds - y_log_true) / preds.size
            hess_base = 2 * np.ones_like(preds) / preds.size

            # # penalty derivative wrt C:
            # # softplus(u) with u=(s-eps)/tau: d/du = sigmoid(u)
            # # d/dC softplus((s-eps)/tau) = sigmoid(t) * (1/tau) * ds/dC
            # ds_dC = C / s
            # dphi_dC = sig * (1.0 / pen.tau) * ds_dC

            # # dC/df_i = (w_i/w_sum) * z_c_i * d(ratio_i)/df_i
            # # ratio_i = exp(f_i)/y_i => dr/df = ratio
            # dC_dfi = (w / w_sum) * (z_c * ratio)

            # grad_pen = pen.rho * dphi_dC * dC_dfi
            # MINE
            grad_pen = 2 * (preds - z) * (z_c/z) ** 2 
            hess_pen = 2 * (z_c/z) ** 2

            grad = grad_base + pen.rho * grad_pen
            hess = hess_base + pen.rho  * hess_pen # keep stable diagonal Hessian
            return grad, hess

        callbacks = []
        if eval_set is not None and self.early_stopping_rounds is not None:
            callbacks.append(
                lgb.early_stopping(
                    self.early_stopping_rounds,
                    verbose=bool(self.verbose_eval),
                )
            )

        self._booster = lgb.train(
            params=params,
            train_set=train_data,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            objective=fobj,
            callbacks=callbacks,
            # verbose_eval=self.verbose_eval,
        )

        return self

    def predict(self, X):
        if self._booster is None:
            raise RuntimeError("Model is not fit yet.")
        f = self._booster.predict(X)
        return np.exp(f)

    @property
    def booster_(self) -> lgb.Booster:
        if self._booster is None:
            raise RuntimeError("Model is not fit yet.")
        return self._booster




















































import numpy as np
import lightgbm as lgb
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def get_covariance_stats(y_true, y_pred):
    """
    Helper function to calculate the covariance between the ratio (y_pred/y_true)
    and y_true.
    
    Returns:
        covariance (float): The calculated covariance.
        w (np.array): The vector of weights needed for the gradient calculation.
    """
    # Prevent division by zero
    safe_labels = np.where(y_true == 0, 1e-6, y_true)
    
    # Calculate statistics
    real_mean = np.mean(y_true)  
    ratio = y_pred / safe_labels
    
    # C = Cov(ratio, labels)
    # Formula simplification: C = mean( ratio * (labels - mean(labels)) )
    # This simplification holds because sum(labels - mean) is 0.
    covariance = np.mean(ratio * (y_true - real_mean))
    
    # Calculate the gradient weights w_i for the chain rule
    # d(Cov)/d(pred_i) = (1/N) * (1/label_i) * (label_i - mean_label)
    N = len(y_true)
    w = (y_true - real_mean) / (N * safe_labels)
    
    return covariance, w

def make_constrained_mse_objective(penalty_weight, epsilon):
    """
    Factory function that returns a custom LightGBM objective function
    with fixed parameters for penalty_weight and epsilon.
    
    Args:
        penalty_weight (float): The strength of the regularization (lambda).
        epsilon (float): The allowed margin for the covariance (soft constraint).
        
    Returns:
        function: A function with signature (y_true, y_pred) -> (grad, hess)
    """
    
    def objective(y_true, y_pred):
        # 1. Calculate Covariance and its gradient weights
        cov_val, w = get_covariance_stats(y_true, y_pred)
        
        # 2. Calculate Penalty Gradient and Hessian
        # We use a squared hinge loss for the penalty: 
        # Loss_penalty = lambda * max(0, |Cov| - epsilon)^2
        
        diff = abs(cov_val) - epsilon
        
        if diff > 0:
            # Gradient of Penalty w.r.t Covariance
            # dP/dC = 2 * lambda * (|C| - eps) * sign(C)
            grad_penalty_wrt_cov = 2 * penalty_weight * diff * np.sign(cov_val)
            
            # Chain rule: dP/dPred_i = dP/dC * dC/dPred_i
            grad_penalty = grad_penalty_wrt_cov * w
            
            # Hessian approximation
            # We ignore off-diagonal terms (w_i * w_j) for computational efficiency
            # and numerical stability in diagonal-hessian boosting.
            # d^2P/dPred_i^2 approx 2 * lambda * w_i^2
            hess_penalty = 2 * penalty_weight * (w ** 2)
        else:
            grad_penalty = np.zeros_like(y_pred)
            hess_penalty = np.zeros_like(y_pred)

        # 3. Base Loss (MSE) derivatives
        # L_base = 0.5 * (pred - true)^2
        # grad = pred - true
        # hess = 1
        grad_mse = y_pred - y_true
        hess_mse = np.ones_like(y_pred)
        
        # 4. Combine
        grad = grad_mse + grad_penalty
        hess = hess_mse + hess_penalty
        
        return grad, hess
    
    return objective

def make_covariance_metric(epsilon):
    """
    Factory function for a custom validation metric to monitor the covariance.
    """
    def eval_metric(y_true, y_pred):
        cov_val, _ = get_covariance_stats(y_true, y_pred)
        
        # We want to check if |Cov| <= Epsilon
        is_violated = abs(cov_val) > epsilon
        
        # Return name, value, is_higher_better
        return 'cov_violation', abs(cov_val), False
    
    return eval_metric

# ==========================================
# Example Usage with Scikit-Learn API
# ==========================================

if __name__ == "__main__":
    # 1. Generate synthetic regression data
    # We add some specific noise to create a correlation between ratio and target
    X, y = make_regression(n_samples=5000, n_features=20, noise=0.1, random_state=42)
    y = np.abs(y) + 1  # Ensure y is positive for ratio calculation stability

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Define parameters
    LAMBDA = 10000.0  # High penalty to force the constraint
    EPSILON = 0.001   # Strict covariance limit

    # 3. Create the custom objective and metric functions
    custom_obj = make_constrained_mse_objective(penalty_weight=LAMBDA, epsilon=EPSILON)
    custom_eval = make_covariance_metric(epsilon=EPSILON)

    # 4. Initialize LGBMRegressor
    # Note: We pass the custom objective to the constructor or set_params
    model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.05,
        objective=custom_obj,  # <--- Injecting the custom objective
        verbose=-1
    )

    # 5. Train
    print(f"Training with Lambda={LAMBDA}, Epsilon={EPSILON}...")
    model.fit(
        X_train, 
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric=custom_eval # <--- Injecting the custom metric
    )

    # 6. Verify Results
    preds = model.predict(X_test)
    final_cov, _ = get_covariance_stats(y_test, preds)
    mse = mean_squared_error(y_test, preds)

    print("\n--- Final Results ---")
    print(f"MSE: {mse:.4f}")
    print(f"Covariance(Ratio, Real): {final_cov:.6f}")
    print(f"Constraint satisfied (|Cov| <= {EPSILON})? {abs(final_cov) <= EPSILON}")