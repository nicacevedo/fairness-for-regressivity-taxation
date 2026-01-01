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




# MINE: Light GBM custom metric
class LGBCustomObjective:
    def __init__(self, rho=1e-3, keep=0.7, adversary_type="overall", zero_grad_tol=1e-6, lgbm_params=None):
        self.rho = rho
        self.keep = keep
        self.adversary_type = adversary_type
        self.zero_grad_tol = zero_grad_tol
        self.model = lgb.LGBMRegressor(**lgbm_params)

    def fit(self, X, y):
        
        # Update lgbm params
        self.model.set_params(
            objective=self.fobj
        )
        self.model.fit(X,y)

    def predict(self, X):
        return self.model.predict(X)
        
    def fobj(self, y_true, y_pred):
        # Loss function value
        mse_value = (y_true - y_pred)**2
        cov_surr_value = (y_pred/y_true-1)**2 * (y_true - np.mean(y_true))**2
        loss_value = mse_value + self.rho * cov_surr_value  
        model_name = self.__str__()
        print(
            f"[{model_name.split('(')[0]}] "
            f"Loss value: {np.mean(loss_value):.6f} "
            f"| MSE value: {np.mean(mse_value):.6f} "
            f"| CovSurr value: {np.mean(cov_surr_value):.6f} "
            f"| Corr(r,y): {np.corrcoef(y_pred / y_true, y_true)[0, 1]:.6f} "
        )
        # Get worst r=K/n fraction 
        n = y_pred.size
        K = int(n * self.keep)

        if self.adversary_type == "overall":
            worst_K_base = np.argpartition(loss_value, -K)[-K:]
            worst_K_pen = worst_K_base
        elif self.adversary_type == "individual":
            worst_K_base = np.argpartition(mse_value, -K)[-K:]
            worst_K_pen = np.argpartition(cov_surr_value, -K)[-K:]
        else:
            raise ValueError(f"No adversary_type called: {self.adversary_type}")

        # base gradients/hessians for 0.5*(pred-logy)^2
        grad_base = 2 * (y_pred[worst_K_base] - y_true[worst_K_base]) / K
        hess_base = 2 * np.ones_like(y_pred[worst_K_base]) / K

        # MINE
        z = y_true[worst_K_pen]
        z_c = (y_true[worst_K_pen] - np.mean(y_true)) # np.mean(y_true[worst_K_pen]))
        grad_pen = 2 * (y_pred[worst_K_pen] - z) * (z_c/z) ** 2 / K
        hess_pen = 2 * (z_c/z) ** 2 / K

        # Adversarial grad/hess (?)
        grad, hess = np.zeros(n), np.zeros(n)
        grad[worst_K_base] += grad_base
        hess[worst_K_base] += hess_base
        grad[worst_K_pen] += self.rho * grad_pen
        hess[worst_K_pen] += self.rho  * hess_pen # keep stable diagonal Hessian

        # zero grad/hess tol
        grad[grad==0] += self.zero_grad_tol
        hess[hess==0] += self.zero_grad_tol

        return grad, hess
    
    def get_objective_value(self, y_true, y_pred):
        mse_value = (y_true - y_pred)**2
        cov_surr_value = (y_pred/y_true-1)**2 * (y_true - np.mean(y_true))**2
        return np.mean(mse_value + self.rho * cov_surr_value)

    def __str__(self):
        return f"LGBCustomObjective({self.rho}, {self.adversary_type})" #adversary_type={self.adversary_type})" #, tol={self.zero_grad_tol})"    
    



# ==========================================================
# Primal–Dual (smooth) version (kept very close to yours)
# ==========================================================

class LGBPrimalDual:
    def __init__(self, rho=1e-3, keep=0.7, adversary_type="overall", eta_adv=0.1, zero_grad_tol=1e-6, lgbm_params=None):
        self.rho = rho
        self.keep = keep
        self.adversary_type = adversary_type
        self.eta_adv = eta_adv
        self.zero_grad_tol = zero_grad_tol
        self.model = lgb.LGBMRegressor(**lgbm_params)

    def fit(self, X, y):
        # cache for the callback
        self.X_ = X
        self.y_ = y
        self.y_mean_ = np.mean(y)
        self.n_ = y.size
        # Update: cache current ensemble predictions so we can update them incrementally
        self.y_hat_ = np.ones(self.n_) * self.y_mean_   # matches boost_from_average=True default

        # CVaR cap: w_i <= 1/(alpha*n), with alpha = keep
        self.cap_ = 1.0 / (max(1, int(self.keep * self.n_)) )  # = 1/K

        # initialize adversary weights (uniform)
        w0 = np.ones(self.n_) / self.n_
        if self.adversary_type == "overall":
            self.w_ = w0
        elif self.adversary_type == "individual":
            self.p_ = w0
            self.q_ = w0
        else:
            raise ValueError(f"No adversary_type called: {self.adversary_type}")

        # Update lgbm params
        self.model.set_params(objective=self.fobj)
        self.model.fit(X, y, callbacks=[self._adv_callback])

    def predict(self, X):
        return self.model.predict(X)

    def _project_capped_simplex(self, w):
        """Project to {w>=0, sum w=1, w_i<=cap_} (simple cap + redistribute)."""
        w = np.maximum(w, 0)
        if w.sum() <= 0:
            w = np.ones_like(w) / w.size
        else:
            w = w / w.sum()

        cap = self.cap_
        # cap-and-redistribute until feasible (usually 1-2 passes)
        for _ in range(10):
            over = w > cap
            if not np.any(over):
                break
            excess = w[over].sum() - cap * over.sum()
            w[over] = cap
            under = ~over
            if not np.any(under):
                # everything capped -> already sums to 1 by definition of cap=1/K
                break
            w[under] += excess * (w[under] / w[under].sum())
        return w

    def _mirror_step(self, w, v):
        # exponentiated-gradient / mirror-ascent step
        z = self.eta_adv * (v - np.max(v))
        w_new = w * np.exp(z)
        return self._project_capped_simplex(w_new)

    def _adv_callback(self, env):
        # update adversary once per boosting iteration using current predictions
        it = env.iteration + 1
        # y_hat = env.model.predict(self.X_, num_iteration=it)
        # Update: predict ONLY the new tree’s contribution and add it to cached predictions
        delta = env.model.predict(self.X_, start_iteration=it-1, num_iteration=1)
        self.y_hat_ = self.y_hat_ + delta
        y_hat = self.y_hat_

        mse_value = (self.y_ - y_hat) ** 2
        cov_surr_value = (y_hat / self.y_ - 1) ** 2 * (self.y_ - self.y_mean_) ** 2

        if self.adversary_type == "overall":
            v = mse_value + self.rho * cov_surr_value
            self.w_ = self._mirror_step(self.w_, v)
        else:
            self.p_ = self._mirror_step(self.p_, mse_value)
            self.q_ = self._mirror_step(self.q_, cov_surr_value)

    def fobj(self, y_true, y_pred):
        # Loss function value (same prints as yours)
        mse_value = (y_true - y_pred) ** 2
        cov_surr_value = (y_pred / y_true - 1) ** 2 * (y_true - np.mean(y_true)) ** 2
        loss_value = mse_value + self.rho * cov_surr_value
        model_name = self.__str__()
        print(
            f"[{model_name.split('(')[0]}] "
            f"Loss value: {np.mean(loss_value):.6f} "
            f"| MSE value: {np.mean(mse_value):.6f} "
            f"| CovSurr value: {np.mean(cov_surr_value):.6f} "
            f"| Corr(r,y): {np.corrcoef(y_pred / y_true, y_true)[0, 1]:.6f} "
        )

        # base gradients/hessians for 0.5*(pred-y)^2  (now for ALL samples)
        grad_base = 2 * (y_pred - y_true)
        hess_base = 2 * np.ones_like(y_pred)

        # penalty gradients/hessians (same structure as yours, for ALL samples)
        z = y_true
        z_c = (y_true - np.mean(y_true))
        grad_pen = 2 * (y_pred - z) * (z_c / z) ** 2
        hess_pen = 2 * (z_c / z) ** 2

        n = y_pred.size

        # primal step uses current adversary weights (scaled by n so magnitudes stay reasonable)
        if self.adversary_type == "overall":
            w_eff = n * self.w_
            grad = w_eff * (grad_base + self.rho * grad_pen)
            hess = w_eff * (hess_base + self.rho * hess_pen)
        else:
            p_eff = n * self.p_
            q_eff = n * self.q_
            grad = p_eff * grad_base + self.rho * q_eff * grad_pen
            hess = p_eff * hess_base + self.rho * q_eff * hess_pen

        # zero grad/hess tol (same as yours)
        grad[grad == 0] += self.zero_grad_tol
        hess[hess == 0] += self.zero_grad_tol

        return grad, hess

    def __str__(self):
        return f"LGBPrimalDual({self.rho}, {self.adversary_type})" #adversary_type={self.adversary_type})" #, eta_adv={self.eta_adv}, tol={self.zero_grad_tol})"














# Single lambda. Weird behavior
class FairGBMCustomObjective:
    """Simple primal–dual wrapper around LightGBM.

    Trains with a custom objective:
        L(ŷ) = MSE(y, ŷ) + λ * cov_surr(y, ŷ)

    and updates a single nonnegative multiplier once per boosting iteration:
        λ <- max(0, λ + η * (E[cov_surr] - τ))

    IMPORTANT implementation detail:
    - Do NOT scale grad/hess by 1/n. LightGBM has constraints like
      min_sum_hessian_in_leaf; scaling by 1/n can make Hessians tiny and
      prevent any split (you'll see "best gain: -inf").

    This is *not* the full FairGBM paper algorithm; it is the minimal primal–dual
    scheme for your single surrogate constraint.
    """

    def __init__(
        self,
        tau: float = 10.0,
        keep: float = 1.0,
        step_size_lamb: float = 1e-3,
        max_iter: int = 100,
        lambda_init: float = 0.0,
        lambda_max: float | None = None,
        y_eps: float = 1e-8,
        hess_floor: float = 1e-12,
        use_penalty_hessian: bool = True,
        lgbm_params: dict | None = None,
        verbose: int = 1,
        update_on: str = "train",  # "train" or "valid"
        pred_delta_tol: float = 0.0,  # set e.g. 1e-10 to monitor stagnation
    ):
        self.tau = float(tau) * keep
        self.keep = keep
        self.step_size_lamb = float(step_size_lamb)
        self.max_iter = int(max_iter)
        self.lambda_init = float(lambda_init)
        self.lambda_max = lambda_max if lambda_max is None else float(lambda_max)
        self.y_eps = float(y_eps)
        self.hess_floor = float(hess_floor)
        self.use_penalty_hessian = bool(use_penalty_hessian)
        self.verbose = int(verbose)
        self.update_on = str(update_on)
        self.pred_delta_tol = float(pred_delta_tol)

        self.lgbm_params = dict(lgbm_params) if lgbm_params is not None else {}
        self.lgbm_params.setdefault("n_estimators", self.max_iter)
        self.model = lgb.LGBMRegressor(**self.lgbm_params)

        self.lamb_pen: float = max(0.0, self.lambda_init)
        self.history_: list[dict] = []

        self._X_train = None
        self._y_train = None
        self._y_mean = None

    # ----------------------- public API -----------------------
    def fit(self, X, y, eval_set=None):
        y = np.asarray(y, dtype=float).reshape(-1)
        self._X_train = X
        self._y_train = y
        self._y_mean = float(np.mean(y))

        # reset state
        self.lamb_pen = max(0.0, self.lambda_init)
        if self.lambda_max is not None:
            self.lamb_pen = min(self.lamb_pen, self.lambda_max)
        self.history_.clear()

        # Use the custom objective
        self.model.set_params(objective=self.fobj)

        callbacks = [self._make_lambda_callback(eval_set=eval_set)]

        if eval_set is None:
            self.model.fit(X, y, callbacks=callbacks)
        else:
            self.model.fit(X, y, eval_set=eval_set, callbacks=callbacks)

        return self

    def predict(self, X):
        return self.model.predict(X)

    def __str__(self):
        return f"FairGBMCustomObjective(tau={self.tau:.6})" # lambda={self.lamb_pen:.6g}, 

    # ----------------------- core math -----------------------
    def _safe_y(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        # preserve sign (not that it matters once squared), avoid exact 0
        return np.where(np.abs(y) < self.y_eps, np.sign(y) * self.y_eps + (y == 0) * self.y_eps, y)

    def _cov_surr(self, y_true: np.ndarray, y_pred: np.ndarray, y_mean: float | None = None) -> np.ndarray:
        # cov_surr_i = ((ŷ/y)-1)^2 * (y - mean(y))^2 = ((ŷ-y)^2 / y^2) * (y-μ)^2
        y_true = np.asarray(y_true, dtype=float).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
        y_safe = self._safe_y(y_true)
        mu = self._y_mean if y_mean is None else float(y_mean)
        zc = y_true - mu
        return ((y_pred - y_true) ** 2) * (zc**2) / (y_safe**2)

    # LightGBM custom objective signature: fobj(y_true, y_pred) -> (grad, hess)
    def fobj(self, y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

        # Base loss: MSE
        # IMPORTANT: do NOT divide by n; LightGBM expects per-sample grad/hess scales
        grad_loss = 2.0 * (y_pred - y_true)
        hess_loss = 2.0 * np.ones_like(y_pred)

        # Penalty: λ * cov_surr
        y_safe = self._safe_y(y_true)
        zc = y_true - (self._y_mean if self._y_mean is not None else float(np.mean(y_true)))
        w = (zc / y_safe) ** 2

        grad_pen = 2.0 * self.lamb_pen * (y_pred - y_true) * w

        if self.use_penalty_hessian:
            hess_pen = 2.0 * self.lamb_pen * w
        else:
            hess_pen = 0.0

        grad = grad_loss + grad_pen
        hess = hess_loss + hess_pen
        hess = np.maximum(hess, self.hess_floor)
        return grad, hess

    # ----------------------- λ update callback -----------------------
    def _make_lambda_callback(self, eval_set=None):
        # Choose the dataset used for λ updates
        if self.update_on == "valid" and eval_set is not None and len(eval_set) > 0:
            X_lam, y_lam = eval_set[0]
            y_lam = np.asarray(y_lam, dtype=float).reshape(-1)
            y_mean = float(np.mean(y_lam))
        else:
            X_lam, y_lam = self._X_train, self._y_train
            y_mean = float(self._y_mean)

        prev_pred = {"y_hat": None}

        def _cb(env):
            it = int(env.iteration) + 1
            y_hat = env.model.predict(X_lam, num_iteration=it)
            y_hat = np.asarray(y_hat, dtype=float).reshape(-1)

            cov_surr = self._cov_surr(y_lam, y_hat, y_mean=y_mean)
            mean_cov = float(np.mean(cov_surr))
            viol = float(mean_cov - self.tau)

            # Projected gradient ascent on λ
            self.lamb_pen = max(0.0, self.lamb_pen + self.step_size_lamb * viol)
            if self.lambda_max is not None:
                self.lamb_pen = min(self.lamb_pen, self.lambda_max)

            rmse = float(np.sqrt(np.mean((y_hat - y_lam) ** 2)))

            pred_delta = None
            if prev_pred["y_hat"] is not None:
                pred_delta = float(np.mean(np.abs(y_hat - prev_pred["y_hat"])))
            prev_pred["y_hat"] = y_hat

            rec = {
                "iter": it,
                "lambda": float(self.lamb_pen),
                "violation": viol,
                "mean_cov_surr": mean_cov,
                "rmse": rmse,
                "mean_abs_pred_delta": pred_delta,
                "corr(r, y)":np.corrcoef(y_hat/y_lam, y_lam)[0,1],
            }
            self.history_.append(rec)

            if self.verbose:
                msg = (
                    f"Iter {it:4d} | λ={rec['lambda']:.4g} | "
                    f"E[cov_surr]={rec['mean_cov_surr']:.4g} | "
                    f"viol={rec['violation']:.4g} | rmse={rec['rmse']:.4g} |"
                    f"corr(r, y)={rec['corr(r, y)']:.4g}"
                )
                if pred_delta is not None and self.pred_delta_tol > 0:
                    msg += f" | Δpred={pred_delta:.3g}"
                print(msg)

                # If you set pred_delta_tol>0, this helps you *see* stagnation.
                # We don't hard-stop training here to keep the wrapper simple.

        _cb.order = 50
        return _cb












# ==========================================================
# FairGBM-style primal–dual constrained boosting wrapper
# (sklearn-like API, trains LightGBM with per-iteration lambda
# updates via callbacks).
#
# This is an *implementation skeleton* faithful to the core
# mechanism: primal step uses proxy-constraint gradients;
# dual step updates multipliers with true constraint violations.
# ==========================================================

# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Callable, List, Optional, Sequence, Tuple, Union, Dict, Any

# import numpy as np

# try:
#     import lightgbm as lgb
# except Exception as e:
#     raise ImportError(
#         "This class requires lightgbm. Install with `pip install lightgbm`."
#     ) from e

# try:
#     from sklearn.base import BaseEstimator, RegressorMixin
# except Exception as e:
#     raise ImportError(
#         "This class requires scikit-learn for the sklearn-like API. Install with `pip install scikit-learn`."
#     ) from e


# ArrayLike = Union[np.ndarray, "np.typing.NDArray[np.float_]" ]


# @dataclass
# class FairnessConstraint:
#     """A constraint c(f) <= 0 used by FairGBM-style training.

#     - violation_fn computes the *true* constraint value c(f). Positive => violation.
#     - proxy_grad_hess_fn returns (grad, hess) of a *proxy* 	ilde{c}(f)
#       w.r.t. predictions y_pred (same length), used in the primal update.

#     Notes:
#       • You can set c(f) = metric(f) - eps so that c<=0 encodes metric<=eps.
#       • If you don’t have a Hessian for the proxy, return None and we’ll use 0.
#     """

#     name: str
#     violation_fn: Callable[[ArrayLike, ArrayLike, Optional[ArrayLike]], float]
#     proxy_grad_hess_fn: Callable[[ArrayLike, ArrayLike, Optional[ArrayLike]], Tuple[ArrayLike, Optional[ArrayLike]]]


# class FairGBMRegressor(BaseEstimator, RegressorMixin):
#     """FairGBM-style constrained gradient boosting using LightGBM.

#     This wrapper runs LightGBM boosting with a custom objective whose gradient is:
#         ∇_y ( base_loss(y, ŷ) + Σ_k λ_k * proxy_constraint_k(y, ŷ, s) )
#     and updates multipliers after each boosting iteration via:
#         λ ← Π_{λ≥0}( λ + η_λ * c_true(y, ŷ, s) )

#     Parameters
#     ----------
#     constraints:
#         List of FairnessConstraint objects.
#     num_boost_round:
#         Number of boosting iterations (T).
#     eta_lambda:
#         Step size for multiplier updates.
#     lambda_init:
#         Initial multiplier value(s). Scalar or array-like of length m.
#     lambda_max:
#         Optional cap for numerical stability (not required by theory).
#     use_validation_for_lambda:
#         If True and eval_set is provided, update λ using the first eval_set.
#         Otherwise update using training data.
#     hess_floor:
#         Small positive floor added to Hessians to keep them strictly positive.
#     lgb_params:
#         Parameters passed to LightGBM (lgb.train). You can include learning_rate,
#         num_leaves, max_depth, min_data_in_leaf, etc.

#     fit signature
#     -------------
#     fit(X, y, sensitive=None, sample_weight=None, eval_set=None, eval_sensitive=None)

#     Where:
#       • sensitive is an array aligned with y (e.g., group labels).
#       • eval_set is like sklearn: [(X_val, y_val), ...]
#       • eval_sensitive optionally provides sensitive arrays for each eval_set entry.
#     """

#     def __init__(
#         self,
#         constraints: Optional[Sequence[FairnessConstraint]] = None,
#         num_boost_round: int = 200,
#         eta_lambda: float = 0.1,
#         lambda_init: Union[float, Sequence[float]] = 0.0,
#         lambda_max: Optional[float] = None,
#         use_validation_for_lambda: bool = False,
#         hess_floor: float = 1e-12,
#         lgb_params: Optional[Dict[str, Any]] = None,
#         verbose: int = 0,
#         lambda_update_every: int = 1,
#         random_state: Optional[int] = None,
#     ):
#         self.constraints = list(constraints) if constraints is not None else []
#         self.num_boost_round = int(num_boost_round)
#         self.eta_lambda = float(eta_lambda)
#         self.lambda_init = lambda_init
#         self.lambda_max = lambda_max
#         self.use_validation_for_lambda = bool(use_validation_for_lambda)
#         self.hess_floor = float(hess_floor)
#         self.lgb_params = lgb_params if lgb_params is not None else {}
#         self.verbose = int(verbose)
#         self.lambda_update_every = int(lambda_update_every)
#         self.random_state = random_state

#         # learned attributes
#         self.booster_ = None
#         self.lambdas_ = None
#         self.history_ = None

#     # ---------------------- helpers ----------------------
#     @staticmethod
#     def _to_1d_float(x: Any, name: str) -> np.ndarray:
#         arr = np.asarray(x)
#         if arr.ndim != 1:
#             raise ValueError(f"{name} must be a 1D array-like. Got shape {arr.shape}.")
#         return arr.astype(float, copy=False)

#     @staticmethod
#     def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
#         return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

#     def _init_lambdas(self, m: int) -> np.ndarray:
#         if isinstance(self.lambda_init, (int, float)):
#             lam = np.full(m, float(self.lambda_init), dtype=float)
#         else:
#             lam = np.asarray(self.lambda_init, dtype=float)
#             if lam.shape != (m,):
#                 raise ValueError(f"lambda_init must have shape ({m},), got {lam.shape}.")
#         lam = np.maximum(lam, 0.0)
#         if self.lambda_max is not None:
#             lam = np.minimum(lam, float(self.lambda_max))
#         return lam

#     # ---------------------- sklearn API ----------------------
#     def fit(
#         self,
#         X: Any,
#         y: Any,
#         sensitive: Optional[Any] = None,
#         sample_weight: Optional[Any] = None,
#         eval_set: Optional[Sequence[Tuple[Any, Any]]] = None,
#         eval_sensitive: Optional[Sequence[Any]] = None,
#     ):
#         y_train = self._to_1d_float(y, "y")
#         n = y_train.shape[0]

#         s_train = None
#         if sensitive is not None:
#             s_train = self._to_1d_float(sensitive, "sensitive")
#             if s_train.shape[0] != n:
#                 raise ValueError("sensitive must be aligned with y (same length).")

#         w_train = None
#         if sample_weight is not None:
#             w_train = self._to_1d_float(sample_weight, "sample_weight")
#             if w_train.shape[0] != n:
#                 raise ValueError("sample_weight must be aligned with y (same length).")

#         constraints: List[FairnessConstraint] = list(self.constraints) if self.constraints else []
#         m = len(constraints)
#         self.lambdas_ = self._init_lambdas(m)
#         self.history_ = []

#         # LightGBM datasets
#         train_set = lgb.Dataset(X, label=y_train, weight=w_train, free_raw_data=False)

#         valid_sets = [train_set]
#         valid_names = ["train"]
#         X_lambda, y_lambda, s_lambda = X, y_train, s_train

#         if eval_set is not None and len(eval_set) > 0:
#             if eval_sensitive is not None and len(eval_sensitive) != len(eval_set):
#                 raise ValueError("eval_sensitive must have the same length as eval_set.")

#             for i, (Xv, yv) in enumerate(eval_set):
#                 yv = self._to_1d_float(yv, f"eval_set[{i}].y")
#                 ds = lgb.Dataset(Xv, label=yv, free_raw_data=False)
#                 valid_sets.append(ds)
#                 valid_names.append(f"valid_{i}")

#             if self.use_validation_for_lambda:
#                 # Use the FIRST validation set for λ updates
#                 X_lambda, y_lambda = eval_set[0]
#                 s_lambda = None
#                 if eval_sensitive is not None:
#                     s_lambda = self._to_1d_float(eval_sensitive[0], "eval_sensitive[0]")
#                     if s_lambda.shape[0] != self._to_1d_float(eval_set[0][1], "eval_set[0].y").shape[0]:
#                         raise ValueError("eval_sensitive[0] must align with eval_set[0].y")

#         # Make copies inside closure
#         y_lambda = self._to_1d_float(y_lambda, "y_lambda")
#         if s_lambda is not None:
#             s_lambda = self._to_1d_float(s_lambda, "s_lambda")

#         # ----------------- custom objective (primal step) -----------------
#         def fobj(y_true: np.ndarray, y_pred: np.ndarray):
#             # Base loss: (1/2) * (y-ŷ)^2  => grad = (ŷ-y), hess = 1
#             # (Using 1/2 avoids factor 2; scale doesn’t matter much for boosting.)
#             grad = (y_pred - y_true).astype(float, copy=False)
#             hess = np.ones_like(grad)

#             # Add proxy-constraint contributions
#             if m > 0:
#                 lam = self.lambdas_  # reference (mutated by callback)
#                 for k, c in enumerate(constraints):
#                     gk, hk = c.proxy_grad_hess_fn(y_true, y_pred, s_train)
#                     gk = np.asarray(gk, dtype=float)
#                     if gk.shape != grad.shape:
#                         raise ValueError(
#                             f"Constraint '{c.name}': proxy_grad has shape {gk.shape}, expected {grad.shape}."
#                         )
#                     grad = grad + lam[k] * gk

#                     if hk is not None:
#                         hk = np.asarray(hk, dtype=float)
#                         if hk.shape != hess.shape:
#                             raise ValueError(
#                                 f"Constraint '{c.name}': proxy_hess has shape {hk.shape}, expected {hess.shape}."
#                             )
#                         hess = hess + lam[k] * hk

#             # Keep Hessian strictly positive
#             if self.hess_floor is not None and self.hess_floor > 0:
#                 hess = np.maximum(hess, float(self.hess_floor))
#             return grad, hess

#         # ----------------- evaluation (optional, for logging / early stop) -----------------
#         def feval_rmse(y_true: np.ndarray, y_pred: np.ndarray):
#             return "rmse", self._rmse(y_true, y_pred), False

#         # ----------------- dual update callback -----------------
#         def make_lambda_update_callback():
#             def _callback(env):
#                 it = env.iteration  # 0-index
#                 if self.lambda_update_every > 1 and ((it + 1) % self.lambda_update_every != 0):
#                     return

#                 if m == 0:
#                     # nothing to update
#                     return

#                 # Predict with current model (up to current iteration)
#                 yhat = env.model.predict(X_lambda, num_iteration=it + 1)
#                 yhat = np.asarray(yhat, dtype=float)

#                 # True constraint violations (vector)
#                 viols = np.array(
#                     [c.violation_fn(y_lambda, yhat, s_lambda) for c in constraints],
#                     dtype=float,
#                 )

#                 # Projected gradient ascent: λ <- (λ + η * viol)_+
#                 self.lambdas_ = np.maximum(0.0, self.lambdas_ + self.eta_lambda * viols)
#                 if self.lambda_max is not None:
#                     self.lambdas_ = np.minimum(self.lambdas_, float(self.lambda_max))

#                 # Book-keeping
#                 rec = {
#                     "iter": int(it + 1),
#                     "rmse_lambda_set": self._rmse(y_lambda, yhat),
#                     "violations": viols.copy(),
#                     "lambdas": self.lambdas_.copy(),
#                 }
#                 self.history_.append(rec)

#                 if self.verbose:
#                     msg = f"[FairGBM] iter={it+1} rmse={rec['rmse_lambda_set']:.6g} "
#                     msg += " ".join([f"{constraints[k].name}:viol={viols[k]:.3g},λ={self.lambdas_[k]:.3g}" for k in range(m)])
#                     print(msg)

#             _callback.order = 50  # run after most internal callbacks
#             return _callback

#         callbacks = [make_lambda_update_callback()]

#         # ----------------- params and training -----------------
#         params = dict(self.lgb_params)
#         # sensible defaults
#         params.setdefault("objective", "regression")
#         params.setdefault("verbosity", -1 if self.verbose == 0 else 0)
#         if self.random_state is not None:
#             params.setdefault("seed", int(self.random_state))

#         self.booster_ = lgb.train(
#             params=params,
#             train_set=train_set,
#             num_boost_round=self.num_boost_round,
#             valid_sets=valid_sets,
#             valid_names=valid_names,
#             fobj=fobj,
#             feval=feval_rmse,
#             callbacks=callbacks,
#         )

#         return self

#     def predict(self, X: Any) -> np.ndarray:
#         if self.booster_ is None:
#             raise RuntimeError("Model is not fit yet. Call fit() first.")
#         return np.asarray(self.booster_.predict(X), dtype=float)

#     def get_booster(self):
#         return self.booster_


# # ---------------------- Example constraint helpers ----------------------

# def make_group_mean_residual_constraint(
#     name: str,
#     group_value: float,
#     eps: float = 0.0,
#     squared_proxy: bool = True,
# ) -> FairnessConstraint:
#     """Constraint: |E[resid | s==g]| - eps <= 0  (implemented as max(|mean|-eps, 0)).

#     True violation (scalar): max(|mean_resid_g| - eps, 0)

#     Proxy used in primal step:
#       • If squared_proxy: (mean_resid_g)^2  (smooth)
#       • Else: |mean_resid_g| (nonsmooth; we still provide a subgradient)

#     Gradient w.r.t. predictions:
#       resid = y - y_pred
#       mean_resid_g = mean(resid_g)
#       d/dy_pred of mean_resid_g = -1/|g| for samples in group g.

#     Hessian is 0 here (we rely on base loss Hessian + hess_floor).
#     """

#     def violation_fn(y_true, y_pred, s):
#         if s is None:
#             raise ValueError("sensitive is required for group constraints")
#         mask = (np.asarray(s) == group_value)
#         if mask.sum() == 0:
#             return 0.0
#         mean_resid = float(np.mean(y_true[mask] - y_pred[mask]))
#         return float(max(abs(mean_resid) - eps, 0.0))

#     def proxy_grad_hess_fn(y_true, y_pred, s):
#         if s is None:
#             raise ValueError("sensitive is required for group constraints")
#         y_true = np.asarray(y_true)
#         y_pred = np.asarray(y_pred)
#         mask = (np.asarray(s) == group_value)
#         gsz = int(mask.sum())
#         grad = np.zeros_like(y_pred, dtype=float)
#         if gsz == 0:
#             return grad, None

#         mean_resid = float(np.mean(y_true[mask] - y_pred[mask]))

#         # d mean_resid / d y_pred_i = -1/gsz if i in group
#         dmean = (-1.0 / gsz)

#         if squared_proxy:
#             # proxy = (mean_resid)^2
#             # d proxy / d y_pred_i = 2*mean_resid * dmean
#             grad[mask] = 2.0 * mean_resid * dmean
#         else:
#             # proxy = |mean_resid|  (use subgradient sign)
#             sign = 0.0
#             if mean_resid > 0:
#                 sign = 1.0
#             elif mean_resid < 0:
#                 sign = -1.0
#             grad[mask] = sign * dmean

#         return grad, None

#     return FairnessConstraint(name=name, violation_fn=violation_fn, proxy_grad_hess_fn=proxy_grad_hess_fn)


# def _sigmoid(x: np.ndarray) -> np.ndarray:
#     """Numerically stable sigmoid."""
#     x = np.asarray(x, dtype=float)
#     out = np.empty_like(x)
#     pos = x >= 0
#     out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
#     ex = np.exp(x[~pos])
#     out[~pos] = ex / (1.0 + ex)
#     return out


# @dataclass
# class CovPenaltyConfig:
#     """Configuration for the covariance penalty.

#     Parameters
#     ----------
#     rho : float
#         Penalty weight.
#     eps : float
#         Tolerance for |Cov|. Penalty activates smoothly when |Cov| > eps.
#     tau : float
#         Smoothness for the soft hinge (softplus). Smaller -> closer to max(0, |Cov|-eps).
#     delta : float
#         Smoothing for absolute value: |C| ~ sqrt(C^2 + delta).
#     z_mode : str
#         Either "logy" (default) or "y"; determines z used in Cov(r, z).
#     """

#     rho: float = 1.0
#     eps: float = 0.0
#     tau: float = 1e-3
#     delta: float = 1e-12
#     z_mode: str = "logy"  # "logy" or "y"

# class LGBMCovRatioRegressor(BaseEstimator, RegressorMixin):
#     """Sklearn-like LightGBM regressor with a covariance penalty on ratios.

#     Objective (in log-space)
#     ------------------------
#     Base loss: 0.5 * sum_i (f_i - log y_i)^2

#     Ratio: r_i = exp(f_i) / y_i

#     Weighted covariance (simplified by centering z):
#         C = E_w[r * (z - E_w[z])]

#     Penalty:
#         rho * softplus((|C| - eps)/tau)

#     The custom objective supplies per-instance gradients and a stable diagonal Hessian.
#     """

#     def __init__(
#         self,
#         lgb_params: Optional[Dict[str, Any]] = None,
#         num_boost_round: int = 2000,
#         early_stopping_rounds: Optional[int] = 200,
#         penalty: CovPenaltyConfig = CovPenaltyConfig(),
#         random_state: int = 0,
#         verbose_eval: Union[bool, int] = False,
#     ):
#         self.lgb_params = lgb_params or {}
#         self.num_boost_round = num_boost_round
#         self.early_stopping_rounds = early_stopping_rounds
#         self.penalty = penalty
#         self.random_state = random_state
#         self.verbose_eval = verbose_eval

#         self._booster: Optional[lgb.Booster] = None

#     def fit(
#         self,
#         X,
#         y,
#         eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
#         sample_weight: Optional[np.ndarray] = None,
#         eval_sample_weight: Optional[np.ndarray] = None,
#     ):
#         y = np.asarray(y, dtype=float)
#         if np.any(y <= 0):
#             raise ValueError("All y must be > 0 when training in log-space and using ratios.")

#         y_log = np.log(y)

#         train_data = lgb.Dataset(X, label=y_log, weight=sample_weight, free_raw_data=False)
#         valid_sets = [train_data]
#         valid_names = ["train"]

#         if eval_set is not None:
#             Xv, yv = eval_set
#             yv = np.asarray(yv, dtype=float)
#             if np.any(yv <= 0):
#                 raise ValueError("All eval y must be > 0.")
#             valid_data = lgb.Dataset(
#                 Xv,
#                 label=np.log(yv),
#                 weight=eval_sample_weight,
#                 free_raw_data=False,
#             )
#             valid_sets.append(valid_data)
#             valid_names.append("valid")

#         params = {
#             "objective": "regression",  # overridden by custom fobj
#             "metric": "l2",
#             "verbosity": -1,
#             "seed": self.random_state,
#         }
#         params.update(self.lgb_params)

#         pen = self.penalty

#         def fobj(preds: np.ndarray, dataset: lgb.Dataset):
#             # preds are f in log-space
#             y_log_true = dataset.get_label().astype(float)
#             y_true = np.exp(y_log_true)
#             y_hat = np.exp(preds)

#             ratio = y_hat / y_true

#             if pen.z_mode == "y":
#                 z = y_true
#             elif pen.z_mode == "logy":
#                 z = y_log_true
#             else:
#                 raise ValueError("penalty.z_mode must be either 'y' or 'logy'.")

#             # weights
#             w = dataset.get_weight()
#             if w is None:
#                 w = np.ones_like(z)
#             w = np.asarray(w, dtype=float)
#             w_sum = float(np.sum(w))
#             if w_sum <= 0:
#                 raise ValueError("Sum of weights must be positive.")

#             # center z
#             z_bar = float(np.sum(w * z) / w_sum)
#             z_c = z - z_bar

#             # weighted covariance C = E_w[ ratio * z_c ]
#             # C = float(np.sum(w * ratio * z_c) / w_sum)

#             # smooth |C|
#             # s = float(np.sqrt(C * C + pen.delta))

#             # soft hinge (via softplus)
#             # t = (s - pen.eps) / pen.tau
#             # sig = float(_sigmoid(np.array([t]))[0])

#             # base gradients/hessians for 0.5*(pred-logy)^2
#             grad_base = 2 * (preds - y_log_true) / preds.size
#             hess_base = 2 * np.ones_like(preds) / preds.size

#             # # penalty derivative wrt C:
#             # # softplus(u) with u=(s-eps)/tau: d/du = sigmoid(u)
#             # # d/dC softplus((s-eps)/tau) = sigmoid(t) * (1/tau) * ds/dC
#             # ds_dC = C / s
#             # dphi_dC = sig * (1.0 / pen.tau) * ds_dC

#             # # dC/df_i = (w_i/w_sum) * z_c_i * d(ratio_i)/df_i
#             # # ratio_i = exp(f_i)/y_i => dr/df = ratio
#             # dC_dfi = (w / w_sum) * (z_c * ratio)

#             # grad_pen = pen.rho * dphi_dC * dC_dfi
#             # MINE
#             grad_pen = 2 * (preds - z) * (z_c/z) ** 2 
#             hess_pen = 2 * (z_c/z) ** 2

#             grad = grad_base + pen.rho * grad_pen
#             hess = hess_base + pen.rho  * hess_pen # keep stable diagonal Hessian
#             return grad, hess

#         callbacks = []
#         if eval_set is not None and self.early_stopping_rounds is not None:
#             callbacks.append(
#                 lgb.early_stopping(
#                     self.early_stopping_rounds,
#                     verbose=bool(self.verbose_eval),
#                 )
#             )

#         self._booster = lgb.train(
#             params=params,
#             train_set=train_data,
#             num_boost_round=self.num_boost_round,
#             valid_sets=valid_sets,
#             valid_names=valid_names,
#             objective=fobj,
#             callbacks=callbacks,
#             # verbose_eval=self.verbose_eval,
#         )

#         return self

#     def predict(self, X):
#         if self._booster is None:
#             raise RuntimeError("Model is not fit yet.")
#         f = self._booster.predict(X)
#         return np.exp(f)

#     @property
#     def booster_(self) -> lgb.Booster:
#         if self._booster is None:
#             raise RuntimeError("Model is not fit yet.")
#         return self._booster






































# import numpy as np
# import lightgbm as lgb
# from sklearn.datasets import make_regression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error

# def get_covariance_stats(y_true, y_pred):
#     """
#     Helper function to calculate the covariance between the ratio (y_pred/y_true)
#     and y_true.
    
#     Returns:
#         covariance (float): The calculated covariance.
#         w (np.array): The vector of weights needed for the gradient calculation.
#     """
#     # Prevent division by zero
#     safe_labels = np.where(y_true == 0, 1e-6, y_true)
    
#     # Calculate statistics
#     real_mean = np.mean(y_true)  
#     ratio = y_pred / safe_labels
    
#     # C = Cov(ratio, labels)
#     # Formula simplification: C = mean( ratio * (labels - mean(labels)) )
#     # This simplification holds because sum(labels - mean) is 0.
#     covariance = np.mean(ratio * (y_true - real_mean))
    
#     # Calculate the gradient weights w_i for the chain rule
#     # d(Cov)/d(pred_i) = (1/N) * (1/label_i) * (label_i - mean_label)
#     N = len(y_true)
#     w = (y_true - real_mean) / (N * safe_labels)
    
#     return covariance, w

# def make_constrained_mse_objective(penalty_weight, epsilon):
#     """
#     Factory function that returns a custom LightGBM objective function
#     with fixed parameters for penalty_weight and epsilon.
    
#     Args:
#         penalty_weight (float): The strength of the regularization (lambda).
#         epsilon (float): The allowed margin for the covariance (soft constraint).
        
#     Returns:
#         function: A function with signature (y_true, y_pred) -> (grad, hess)
#     """
    
#     def objective(y_true, y_pred):
#         # 1. Calculate Covariance and its gradient weights
#         cov_val, w = get_covariance_stats(y_true, y_pred)
        
#         # 2. Calculate Penalty Gradient and Hessian
#         # We use a squared hinge loss for the penalty: 
#         # Loss_penalty = lambda * max(0, |Cov| - epsilon)^2
        
#         diff = abs(cov_val) - epsilon
        
#         if diff > 0:
#             # Gradient of Penalty w.r.t Covariance
#             # dP/dC = 2 * lambda * (|C| - eps) * sign(C)
#             grad_penalty_wrt_cov = 2 * penalty_weight * diff * np.sign(cov_val)
            
#             # Chain rule: dP/dPred_i = dP/dC * dC/dPred_i
#             grad_penalty = grad_penalty_wrt_cov * w
            
#             # Hessian approximation
#             # We ignore off-diagonal terms (w_i * w_j) for computational efficiency
#             # and numerical stability in diagonal-hessian boosting.
#             # d^2P/dPred_i^2 approx 2 * lambda * w_i^2
#             hess_penalty = 2 * penalty_weight * (w ** 2)
#         else:
#             grad_penalty = np.zeros_like(y_pred)
#             hess_penalty = np.zeros_like(y_pred)

#         # 3. Base Loss (MSE) derivatives
#         # L_base = 0.5 * (pred - true)^2
#         # grad = pred - true
#         # hess = 1
#         grad_mse = y_pred - y_true
#         hess_mse = np.ones_like(y_pred)
        
#         # 4. Combine
#         grad = grad_mse + grad_penalty
#         hess = hess_mse + hess_penalty
        
#         return grad, hess
    
#     return objective

# def make_covariance_metric(epsilon):
#     """
#     Factory function for a custom validation metric to monitor the covariance.
#     """
#     def eval_metric(y_true, y_pred):
#         cov_val, _ = get_covariance_stats(y_true, y_pred)
        
#         # We want to check if |Cov| <= Epsilon
#         is_violated = abs(cov_val) > epsilon
        
#         # Return name, value, is_higher_better
#         return 'cov_violation', abs(cov_val), False
    
#     return eval_metric

# # ==========================================
# # Example Usage with Scikit-Learn API
# # ==========================================

# if __name__ == "__main__":
#     # 1. Generate synthetic regression data
#     # We add some specific noise to create a correlation between ratio and target
#     X, y = make_regression(n_samples=5000, n_features=20, noise=0.1, random_state=42)
#     y = np.abs(y) + 1  # Ensure y is positive for ratio calculation stability

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # 2. Define parameters
#     LAMBDA = 10000.0  # High penalty to force the constraint
#     EPSILON = 0.001   # Strict covariance limit

#     # 3. Create the custom objective and metric functions
#     custom_obj = make_constrained_mse_objective(penalty_weight=LAMBDA, epsilon=EPSILON)
#     custom_eval = make_covariance_metric(epsilon=EPSILON)

#     # 4. Initialize LGBMRegressor
#     # Note: We pass the custom objective to the constructor or set_params
#     model = lgb.LGBMRegressor(
#         n_estimators=100,
#         learning_rate=0.05,
#         objective=custom_obj,  # <--- Injecting the custom objective
#         verbose=-1
#     )

#     # 5. Train
#     print(f"Training with Lambda={LAMBDA}, Epsilon={EPSILON}...")
#     model.fit(
#         X_train, 
#         y_train,
#         eval_set=[(X_test, y_test)],
#         eval_metric=custom_eval # <--- Injecting the custom metric
#     )

#     # 6. Verify Results
#     preds = model.predict(X_test)
#     final_cov, _ = get_covariance_stats(y_test, preds)
#     mse = mean_squared_error(y_test, preds)

#     print("\n--- Final Results ---")
#     print(f"MSE: {mse:.4f}")
#     print(f"Covariance(Ratio, Real): {final_cov:.6f}")
#     print(f"Constraint satisfied (|Cov| <= {EPSILON})? {abs(final_cov) <= EPSILON}")