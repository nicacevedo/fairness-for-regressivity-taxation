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
# Primal–Dual (smooth) version (kept very close to the other ones)
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
        return f"LGBPrimalDual({self.rho}, {self.adversary_type}, {self.eta_adv})" #adversary_type={self.adversary_type})" #, eta_adv={self.eta_adv}, tol={self.zero_grad_tol})"









### 

# Models to compare primal-dual method

### 

# ==========================================================
# 1) Plain (no dual/adversary): minimize MSE + rho * surrogate
# ==========================================================

class LGBSmoothPenalty:
    """LightGBM custom objective: per-sample MSE + rho * separable surrogate.

    This is the same objective structure as your current code, but WITHOUT
    any CVaR/top-k adversary / dual weights.

    Surrogate (separable): ((y_pred / y_true) - 1)^2 * (y_true - y_mean)^2
    """

    def __init__(self, rho=1e-3, zero_grad_tol=1e-6, eps_y=1e-12, lgbm_params=None):
        self.rho = rho
        self.zero_grad_tol = zero_grad_tol
        self.eps_y = eps_y
        self.model = lgb.LGBMRegressor(**(lgbm_params or {}))

    def fit(self, X, y):
        self.y_mean_ = float(np.mean(y))
        self.model.set_params(objective=self.fobj)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def fobj(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        z = y_true
        zc = (y_true - self.y_mean_)
        denom = np.maximum(np.abs(z), self.eps_y)

        # losses
        mse_value = (y_true - y_pred) ** 2
        cov_surr_value = ((y_pred / denom) - 1.0) ** 2 * (zc ** 2)
        loss_value = mse_value + self.rho * cov_surr_value

        # print (kept close to yours)
        model_name = self.__str__()
        r = y_pred / denom
        try:
            corr = float(np.corrcoef(r, y_true)[0, 1])
        except Exception:
            corr = float('nan')
        print(
            f"[{model_name.split('(')[0]}] "
            f"Loss value: {np.mean(loss_value):.6f} "
            f"| MSE value: {np.mean(mse_value):.6f} "
            f"| CovSurr value: {np.mean(cov_surr_value):.6f} "
            f"| Corr(r,y): {corr:.6f} "
        )

        # base gradients/hessians for (pred-y)^2
        grad_base = 2.0 * (y_pred - y_true)
        hess_base = 2.0 * np.ones_like(y_pred)

        # penalty gradients/hessians (separable)
        # cov_surr_i = ((pred/z)-1)^2 * zc^2 = (pred-z)^2 * (zc/z)^2
        scale = (zc / denom) ** 2
        grad_pen = 2.0 * (y_pred - z) * scale
        hess_pen = 2.0 * scale

        grad = grad_base + self.rho * grad_pen
        hess = hess_base + self.rho * hess_pen

        # zero tol
        grad[np.abs(grad) < self.zero_grad_tol] = self.zero_grad_tol
        hess[hess < self.zero_grad_tol] = self.zero_grad_tol

        return grad, hess

    def __str__(self):
        return f"LGBSmoothPenalty({self.rho})"


# ==========================================================
# 2) Improved primal-dual: mirror ascent adversary + KL projection onto capped simplex
# ==========================================================

class LGBPrimalDualImproved:
    """Primal-dual robust boosting with a *principled* dual step on the capped simplex.

    Objective (overall):  min_F max_{w in Delta_K} sum_i w_i [mse_i + rho*surr_i]
    Objective (individual): min_F max_p sum_i p_i mse_i + rho max_q sum_i q_i surr_i

    Dual update options:
      - dual_update="topk": exact best-response (uniform on worst-K)
      - dual_update="mirror": exponentiated-gradient + KL projection onto capped simplex

    Notes vs your current code:
      - Uses a KL/Bregman projection consistent with exponentiated-gradient:
            w = min(cap, u / Z) with Z chosen so sum w = 1
        (found by bisection on Z).
      - Adds eps_y to avoid division blow-ups.
      - Keeps everything else close to your original structure.
    """

    def __init__(
        self,
        rho=1e-3,
        keep=0.7,
        adversary_type="overall",
        dual_update="mirror",  # "mirror" or "topk"
        eta_adv=0.1,
        zero_grad_tol=1e-6,
        eps_y=1e-12,
        lgbm_params=None,
    ):
        self.rho = rho
        self.keep = keep
        self.adversary_type = adversary_type
        self.dual_update = dual_update
        self.eta_adv = eta_adv
        self.zero_grad_tol = zero_grad_tol
        self.eps_y = eps_y
        self.model = lgb.LGBMRegressor(**(lgbm_params or {}))

    def fit(self, X, y):
        # cache for the callback
        self.X_ = X
        self.y_ = np.asarray(y)
        self.y_mean_ = float(np.mean(self.y_))
        self.n_ = int(self.y_.size)

        # cached ensemble predictions (match boost_from_average=True for regression)
        self.y_hat_ = np.ones(self.n_) * self.y_mean_

        # CVaR/top-k cap: w_i <= 1/K, with K = keep*n
        self.K_ = max(1, int(self.keep * self.n_))
        self.cap_ = 1.0 / float(self.K_)

        w0 = np.ones(self.n_) / float(self.n_)
        if self.adversary_type == "overall":
            self.w_ = w0
        elif self.adversary_type == "individual":
            self.p_ = w0.copy()
            self.q_ = w0.copy()
        else:
            raise ValueError(f"No adversary_type called: {self.adversary_type}")

        self.model.set_params(objective=self.fobj)
        self.model.fit(X, y, callbacks=[self._adv_callback])
        return self

    def predict(self, X):
        return self.model.predict(X)

    # ---------- Dual helpers ----------

    def _project_capped_simplex_kl(self, u):
        """KL/Bregman projection of u onto {w>=0, sum w=1, w_i<=cap_}.

        The solution has the form w_i = min(cap, u_i / Z) with Z>0 chosen so sum w = 1.
        """
        u = np.asarray(u, dtype=float)
        u = np.maximum(u, 0.0) + 1e-300
        cap = float(self.cap_)

        # If already feasible after normalization and capping, quick exit
        # (not strictly necessary, but cheap)
        Z_hi = float(np.sum(u))
        if not np.isfinite(Z_hi) or Z_hi <= 0:
            return np.ones_like(u) / u.size

        def s(Z):
            return float(np.sum(np.minimum(cap, u / Z)))

        # We want s(Z)=1. s is decreasing in Z.
        Z_lo = 1e-300
        # Ensure s(Z_lo) >= 1 (should hold if cap*n >= 1)
        if s(Z_lo) < 1.0 - 1e-12:
            # fallback to top-k feasible point
            w = np.zeros_like(u)
            idx = np.argpartition(u, -self.K_)[-self.K_:]
            w[idx] = 1.0 / float(self.K_)
            return w

        # Z_hi gives s(Z_hi) <= 1
        if s(Z_hi) > 1.0 + 1e-12:
            # expand hi until it is above the root
            for _ in range(60):
                Z_hi *= 2.0
                if s(Z_hi) <= 1.0 + 1e-12:
                    break

        # bisection
        for _ in range(80):
            Z_mid = 0.5 * (Z_lo + Z_hi)
            sm = s(Z_mid)
            if abs(sm - 1.0) <= 1e-12:
                Z_lo = Z_hi = Z_mid
                break
            if sm > 1.0:
                Z_lo = Z_mid
            else:
                Z_hi = Z_mid

        Z_star = 0.5 * (Z_lo + Z_hi)
        w = np.minimum(cap, u / Z_star)
        w_sum = float(np.sum(w))
        if not np.isfinite(w_sum) or w_sum <= 0:
            w = np.ones_like(u) / u.size
        else:
            w /= w_sum
        return w

    def _dual_step(self, w, v):
        """Update weights for max_{w in capped simplex} <w, v>."""
        if self.dual_update == "topk":
            # exact best response: uniform mass on worst-K v_i
            idx = np.argpartition(v, -self.K_)[-self.K_:]
            w_new = np.zeros_like(w)
            w_new[idx] = 1.0 / float(self.K_)
            return w_new

        # mirror ascent (exponentiated-gradient) + KL projection
        z = self.eta_adv * (v - np.max(v))
        z = np.clip(z, -50.0, 50.0)
        u = w * np.exp(z)
        return self._project_capped_simplex_kl(u)

    # ---------- Callback (dual update) ----------

    def _adv_callback(self, env):
        it = int(env.iteration) + 1

        # Incremental prediction update: add only the latest tree contribution
        delta = env.model.predict(self.X_, start_iteration=it - 1, num_iteration=1)
        self.y_hat_ = self.y_hat_ + delta
        y_hat = self.y_hat_

        denom = np.maximum(np.abs(self.y_), self.eps_y)
        mse_value = (self.y_ - y_hat) ** 2
        cov_surr_value = ((y_hat / denom) - 1.0) ** 2 * (self.y_ - self.y_mean_) ** 2

        if self.adversary_type == "overall":
            v = mse_value + self.rho * cov_surr_value
            self.w_ = self._dual_step(self.w_, v)
        else:
            self.p_ = self._dual_step(self.p_, mse_value)
            self.q_ = self._dual_step(self.q_, cov_surr_value)

    # ---------- Objective ----------

    def fobj(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # stable pieces
        z = y_true
        zc = (y_true - self.y_mean_)
        denom = np.maximum(np.abs(z), self.eps_y)

        # per-sample values (for prints)
        mse_value = (y_true - y_pred) ** 2
        cov_surr_value = ((y_pred / denom) - 1.0) ** 2 * (zc ** 2)
        loss_value = mse_value + self.rho * cov_surr_value

        model_name = self.__str__()
        r = y_pred / denom
        try:
            corr = float(np.corrcoef(r, y_true)[0, 1])
        except Exception:
            corr = float('nan')
        print(
            f"[{model_name.split('(')[0]}] "
            f"Loss value: {np.mean(loss_value):.6f} "
            f"| MSE value: {np.mean(mse_value):.6f} "
            f"| CovSurr value: {np.mean(cov_surr_value):.6f} "
            f"| Corr(r,y): {corr:.6f} "
        )

        # base gradients/hessians
        grad_base = 2.0 * (y_pred - y_true)
        hess_base = 2.0 * np.ones_like(y_pred)

        # surrogate grads/hess (separable)
        scale = (zc / denom) ** 2
        grad_pen = 2.0 * (y_pred - z) * scale
        hess_pen = 2.0 * scale

        n = y_pred.size
        if self.adversary_type == "overall":
            w_eff = float(n) * self.w_
            grad = w_eff * (grad_base + self.rho * grad_pen)
            hess = w_eff * (hess_base + self.rho * hess_pen)
        else:
            p_eff = float(n) * self.p_
            q_eff = float(n) * self.q_
            grad = p_eff * grad_base + self.rho * q_eff * grad_pen
            hess = p_eff * hess_base + self.rho * q_eff * hess_pen

        # tolerances
        grad[np.abs(grad) < self.zero_grad_tol] = self.zero_grad_tol
        hess[hess < self.zero_grad_tol] = self.zero_grad_tol

        return grad, hess

    def __str__(self):
        return f"LGBPrimalDualImproved({self.rho}, {self.adversary_type}, {self.dual_update}, {self.eta_adv})"


# ==========================================================
# 3) Direct covariance penalty (non-separable but usable in LightGBM via global stats)
# ==========================================================

class LGBCovPenalty:
    """LightGBM objective: MSE + rho * (Cov(r, y))^2 with r = y_pred / y_true.

    This is a *direct* covariance penalty to compare against the separable proxy.

    Important:
      - Cov(r,y) uses global statistics, so the Hessian is dense in principle.
      - LightGBM expects a per-sample Hessian vector; we use a diagonal approximation.

    We scale the penalty as: 0.5 * rho * n * cov^2
    so that gradients have O(1) magnitude (otherwise they can be ~1/n).

    cov = (1/n) * sum_i r_i * (y_i - y_mean)
    dc/dy_pred_i = (1/n) * (y_i - y_mean) / y_i

    grad_pen_i = rho * n * cov * dc/dy_pred_i
    hess_pen_i (diag approx) = rho * n * (dc/dy_pred_i)^2
    """

    def __init__(self, rho=1e-3, zero_grad_tol=1e-6, eps_y=1e-12, lgbm_params=None):
        self.rho = rho
        self.zero_grad_tol = zero_grad_tol
        self.eps_y = eps_y
        self.model = lgb.LGBMRegressor(**(lgbm_params or {}))

    def fit(self, X, y):
        self.y_mean_ = float(np.mean(y))
        self.model.set_params(objective=self.fobj)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def fobj(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        n = y_pred.size

        denom = np.maximum(np.abs(y_true), self.eps_y)
        r = y_pred / denom
        yc = (y_true - self.y_mean_)

        # covariance (note: E[yc]=0)
        cov = float(np.mean(r * yc))

        # objective pieces (for prints)
        mse_value = (y_true - y_pred) ** 2
        pen_value = 0.5 * self.rho * n * (cov ** 2)

        try:
            corr = float(np.corrcoef(r, y_true)[0, 1])
        except Exception:
            corr = float('nan')

        model_name = self.__str__()
        print(
            f"[{model_name.split('(')[0]}] "
            f"MSE: {np.mean(mse_value):.6f} | Cov: {cov:.6e} | Pen: {pen_value:.6f} | Corr(r,y): {corr:.6f}"
        )

        # base MSE grads/hess
        grad_base = 2.0 * (y_pred - y_true)
        hess_base = 2.0 * np.ones_like(y_pred)

        # cov penalty grads/hess (diag approx)
        a = (yc / denom) / float(n)  # dc/dy_pred_i
        # penalty = 0.5*rho*n*cov^2
        grad_pen = self.rho * float(n) * cov * a
        hess_pen = self.rho * float(n) * (a ** 2)

        grad = grad_base + grad_pen
        hess = hess_base + hess_pen

        grad[np.abs(grad) < self.zero_grad_tol] = self.zero_grad_tol
        hess[hess < self.zero_grad_tol] = self.zero_grad_tol

        return grad, hess

    def __str__(self):
        return f"LGBCovPenalty({self.rho})"


# ==========================================================
# 4) Direct K-moments penalty (non-separable but usable in LightGBM via global stats ???)
# ==========================================================


class LGBMomentPenalty:
    r"""LightGBM custom objective: MSE + (moment-adversary penalty).

    Motivation (moment adversary, primal form)
    ----------------------------------------
    Let r_i = y_pred_i / y_true_i be the assessment ratio (or any multiplicative error proxy),
    and define u_i = r_i - 1 (ratio error). Regressivity shows up as *dependence* between u and y.

    Pick K basis functions φ_k(y) (e.g., low-order polynomials in standardized log(y)), and define
    the K "moment violations":
        m_k(F) = (1/n) * Σ_i u_i(F) * φ_k(y_i)

    If u is independent of y, then E[u φ(y)] = 0 for a rich class of φ; we approximate that
    infinite set with K moments.

    Consider the min-max objective:
        min_F  (1/n) Σ_i (y_i - y_pred_i)^2  +  max_{||λ||_* ≤ ρ}  λ^T m(F)

    Using the dual-norm identity:
        max_{||λ||_* ≤ ρ} λ^T m = ρ * ||m||,
    where ||·|| is the norm dual to ||·||_*.

    Therefore we implement the *primal* objective:
        MSE + ρ * ||m(F)||

    Norm choices (λ-norm -> primal penalty)
    --------------------------------------
    - lambda_norm="l2"   : constraint ||λ||_2  ≤ ρ  -> penalty ρ ||m||_2
    - lambda_norm="linf" : constraint ||λ||_∞  ≤ ρ  -> penalty ρ ||m||_1
    - lambda_norm="l1"   : constraint ||λ||_1  ≤ ρ  -> penalty ρ ||m||_∞

    Smoothness and LightGBM Hessian
    -------------------------------
    m(F) couples all samples => the true Hessian is dense. LightGBM expects a per-sample
    Hessian vector, so we use a diagonal approximation derived from the penalty's curvature
    in moment-space and the chain rule.

    Scaling
    -------
    Because m_k is an average (1/n), gradients can be small. We optionally multiply the penalty
    by `n` (scale_by_n=True) to keep gradients O(1). This is equivalent to reparameterizing ρ.

    Notes / Practical guidance
    --------------------------
    - Keep K small (e.g., 1..10). Larger K can chase noise.
    - Prefer basis on standardized log(y) to reduce scale issues.
    - For lambda_norm="l1" (∞-norm penalty), we use a smooth "softmax" approximation.
    """

    def __init__(
        self,
        rho=1e-3,
        K=3,
        basis="poly_log",               # "poly_log" or "poly_y"
        include_intercept_moment=True,  # include φ0(y)=1 as the first moment
        lambda_norm="l2",              # "l2", "linf", "l1" (norm on λ)
        scale_by_n=True,

        # smoothing / numerical stability
        eps_y=1e-12,                   # for y_true division
        eps_norm=1e-12,                # for smoothing norms
        softmax_beta=20.0,             # for smooth ||m||_inf approx when lambda_norm="l1"

        # lightgbm interface safety
        zero_grad_tol=1e-6,
        lgbm_params=None,
        verbose=True,
    ):
        self.rho = float(rho)
        self.K = int(K)
        self.basis = str(basis)
        self.include_intercept_moment = bool(include_intercept_moment)
        self.lambda_norm = str(lambda_norm).lower()
        self.scale_by_n = bool(scale_by_n)

        self.eps_y = float(eps_y)
        self.eps_norm = float(eps_norm)
        self.softmax_beta = float(softmax_beta)

        self.zero_grad_tol = float(zero_grad_tol)
        self.verbose = bool(verbose)

        self.model = lgb.LGBMRegressor(**(lgbm_params or {}))

        # cached at fit
        self.Phi_ = None        # (n, K_eff) basis matrix
        self.y_true_ = None
        self.y_mean_ = None
        self.logy_mean_ = None
        self.logy_std_ = None

    # -----------------------
    # Basis construction
    # -----------------------
    def _make_basis(self, y_true):
        y_true = np.asarray(y_true, dtype=float)
        n = y_true.size

        # Effective number of moments (columns)
        K_eff = self.K + (1 if self.include_intercept_moment else 0)

        Phi = np.empty((n, K_eff), dtype=float)
        col = 0

        if self.include_intercept_moment:
            Phi[:, 0] = 1.0
            col = 1

        if self.basis == "poly_log":
            # Use standardized log(y) for stability / interpretability
            y_safe = np.maximum(np.abs(y_true), self.eps_y)
            t = np.log(y_safe)
            self.logy_mean_ = float(np.mean(t))
            self.logy_std_ = float(np.std(t)) if float(np.std(t)) > 0 else 1.0
            s = (t - self.logy_mean_) / self.logy_std_
            # powers: s^1, s^2, ..., s^(K_eff-col)
            for k in range(1, K_eff - col + 1):
                Phi[:, col + (k - 1)] = s ** k

        elif self.basis == "poly_y":
            # Use centered/scaled y (less recommended when y is heavy-tailed)
            self.y_mean_ = float(np.mean(y_true))
            y_std = float(np.std(y_true)) if float(np.std(y_true)) > 0 else 1.0
            s = (y_true - self.y_mean_) / y_std
            for k in range(1, K_eff - col + 1):
                Phi[:, col + (k - 1)] = s ** k

        else:
            raise ValueError(f"Unknown basis='{self.basis}'. Use 'poly_log' or 'poly_y'.")

        # Optional: normalize columns (except intercept) to unit RMS to stabilize moment scales
        for j in range(K_eff):
            if self.include_intercept_moment and j == 0:
                continue
            rms = np.sqrt(np.mean(Phi[:, j] ** 2))
            if np.isfinite(rms) and rms > 1e-12:
                Phi[:, j] /= rms

        return Phi

    # -----------------------
    # Norm penalty utilities
    # -----------------------
    def _penalty_and_derivs(self, m):
        """
        Given moments m (shape K_eff,), return:
          pen      : scalar penalty (excluding any scale_by_n factor and excluding rho)
          dpen_dm  : gradient wrt m, shape K_eff
          diag_Hm  : diagonal of Hessian wrt m (PSD approx), shape K_eff
        """
        m = np.asarray(m, dtype=float)

        if self.lambda_norm == "l2":
            # penalty = ||m||_2 (smoothed)
            q = float(np.sum(m * m) + self.eps_norm)
            s = float(np.sqrt(q))
            pen = s
            dpen_dm = m / s
            # diag Hessian for ||m||_2:
            # H = I/s - (m m^T)/s^3  => diag = 1/s - m_k^2/s^3
            diag_Hm = (1.0 / s) - (m * m) / (s ** 3)
            diag_Hm = np.maximum(diag_Hm, 0.0)
            return pen, dpen_dm, diag_Hm

        if self.lambda_norm == "linf":
            # lambda-norm = inf  => primal penalty = ||m||_1
            # Use smooth abs: |x| ≈ sqrt(x^2 + eps)
            a = np.sqrt(m * m + self.eps_norm)
            pen = float(np.sum(a))
            dpen_dm = m / a
            # second derivative of sqrt(m^2+eps): eps / (m^2+eps)^(3/2)
            diag_Hm = self.eps_norm / (a ** 3)
            return pen, dpen_dm, diag_Hm

        if self.lambda_norm == "l1":
            # lambda-norm = 1  => primal penalty = ||m||_inf
            # Use smooth max on |m_k|:
            #   max_k |m_k| ≈ (1/beta) log Σ exp(beta * |m_k|)
            beta = max(self.softmax_beta, 1.0)
            a = np.sqrt(m * m + self.eps_norm)        # smooth |m|
            z = beta * (a - np.max(a))                # stabilized
            w = np.exp(z)
            Z = float(np.sum(w) + 1e-30)
            soft = w / Z
            pen = float((np.log(Z) + beta * np.max(a)) / beta)

            # d pen / d a_k = soft_k
            # d a_k / d m_k = m_k / a_k
            dpen_dm = soft * (m / a)

            # crude PSD diagonal curvature:
            # treat soft weights fixed (ignore coupling), keep abs curvature
            # d/dm (m/a) has diag eps/(a^3)
            diag_Hm = soft * (self.eps_norm / (a ** 3))
            return pen, dpen_dm, diag_Hm

        raise ValueError(f"Unknown lambda_norm='{self.lambda_norm}'. Use 'l2', 'linf', or 'l1'.")

    # -----------------------
    # LightGBM wrapper
    # -----------------------
    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        self.y_true_ = y
        self.y_mean_ = float(np.mean(y))

        # precompute basis matrix Φ(y)
        self.Phi_ = self._make_basis(y)

        # set custom objective and fit
        self.model.set_params(objective=self.fobj)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def fobj(self, y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        n = y_pred.size

        # ratio error u_i = r_i - 1
        denom = np.maximum(np.abs(y_true), self.eps_y)
        r = y_pred / denom
        u = r - 1.0

        # moments m_k = (1/n) Σ u_i φ_k(y_i)
        Phi = self.Phi_
        if Phi is None or Phi.shape[0] != n:
            # fallback (should not happen if fit() used)
            Phi = self._make_basis(y_true)
            self.Phi_ = Phi

        m = (u[:, None] * Phi).mean(axis=0)  # shape (K_eff,)

        # penalty from dual-norm identity: ρ * ||m||
        pen_norm, dnorm_dm, diag_Hm = self._penalty_and_derivs(m)

        # optional scaling by n (keeps gradients ~ O(1))
        scale = float(n) if self.scale_by_n else 1.0
        pen_value = self.rho * scale * pen_norm

        # base MSE objective parts (for prints)
        mse_value = (y_true - y_pred) ** 2

        if self.verbose:
            try:
                corr = float(np.corrcoef(r, y_true)[0, 1])
            except Exception:
                corr = float("nan")
            model_name = self.__str__()
            print(
                f"[{model_name.split('(')[0]}] "
                f"MSE: {np.mean(mse_value):.6f} | "
                f"||m||: {pen_norm:.6e} | Pen: {pen_value:.6f} | "
                f"Corr(r,y): {corr:.6f}"
            )

        # base grads/hess for MSE
        grad_base = 2.0 * (y_pred - y_true)
        hess_base = 2.0 * np.ones_like(y_pred)

        # Chain rule for penalty:
        # dm_k/dy_pred_i = (1/n) * φ_k(y_i) / y_i
        # grad_pen_i = ρ*scale * Σ_k (d||m||/dm_k) * dm_k/dy_pred_i
        #           = ρ*scale * (1/n) * (Φ_i · dnorm_dm) / denom_i
        v = Phi @ dnorm_dm  # shape (n,)
        grad_pen = self.rho * scale * (v / (float(n) * denom))

        # Diagonal Hessian approximation:
        # hess_pen_i ≈ ρ*scale * a_i^T H_m a_i
        # where a_i,k = (1/n) * φ_k(y_i)/denom_i and we use diag(H_m)
        # => Σ_k diag_Hm_k * a_i,k^2
        a = Phi / (float(n) * denom[:, None])  # shape (n, K_eff)
        hess_pen = self.rho * scale * np.sum((a * a) * diag_Hm[None, :], axis=1)

        grad = grad_base + grad_pen
        hess = hess_base + hess_pen

        # safety floors (LightGBM dislikes zeros)
        grad[np.abs(grad) < self.zero_grad_tol] = self.zero_grad_tol
        hess[hess < self.zero_grad_tol] = self.zero_grad_tol

        return grad, hess

    def __str__(self):
        return f"LGBMomentPenalty({self.rho}, K={self.K}, l_norm={self.lambda_norm})"


# 5) Covariance AND variance of r
# class LGBCovDispPenalty:
#     """LightGBM objective (sum-style):
#         L = sum_i (y_i - yhat_i)^2
#             + 0.5 * rho_cov  * n * (max(0, - Cov(r,y)/Std(y)))^2
#             + 0.5 * rho_disp * n * mean_i (r_i - 1)^2
#       where r_i = yhat_i / max(|y_i|, eps_y).

#     Notes:
#       - Cov(r,y) uses global stats -> dense Hessian in principle; we keep a diagonal approx
#         exactly like the original class.
#       - We standardize the covariance term by Std(y) (precomputed on training y).
#       - The one-sided hinge only penalizes "regressivity" direction (Cov/Std < 0).
#       - The dispersion term is separable; its diagonal Hessian is exact.
#     """

#     def __init__(
#         self,
#         rho_cov=1e-3,
#         rho_disp=1e-3,
#         zero_grad_tol=1e-6,
#         eps_y=1e-12,
#         eps_std=1e-12,
#         lgbm_params=None,
#     ):
#         self.rho_cov = float(rho_cov)
#         self.rho_disp = float(rho_disp)
#         self.zero_grad_tol = float(zero_grad_tol)
#         self.eps_y = float(eps_y)
#         self.eps_std = float(eps_std)
#         self.model = lgb.LGBMRegressor(**(lgbm_params or {}))

#     def fit(self, X, y):
#         y = np.asarray(y)
#         self.y_mean_ = float(np.mean(y))
#         self.y_std_ = float(np.std(y))  # ddof=0, consistent with mean() usage
#         if self.y_std_ < self.eps_std:
#             self.y_std_ = self.eps_std
#         self.model.set_params(objective=self.fobj)
#         self.model.fit(X, y)
#         return self

#     def predict(self, X):
#         return self.model.predict(X)

#     def fobj(self, y_true, y_pred):
#         y_true = np.asarray(y_true)
#         y_pred = np.asarray(y_pred)
#         n = y_pred.size

#         denom = np.maximum(np.abs(y_true), self.eps_y)
#         r = y_pred / denom
#         yc = (y_true - self.y_mean_)  # centered y

#         # Cov(r,y) = mean(r * (y - y_mean))
#         cov = float(np.mean(r * yc))
#         cov_std = cov / self.y_std_
#         hinge = max(0.0, -cov_std)  # one-sided: only penalize cov_std < 0

#         # dispersion proxy: mean((r-1)^2)
#         disp_mean = float(np.mean((r - 1.0) ** 2))

#         # objective pieces (for prints)
#         mse_value = (y_true - y_pred) ** 2
#         pen_cov_value = 0.5 * self.rho_cov * float(n) * (hinge ** 2)
#         pen_disp_value = 0.5 * self.rho_disp * float(n) * disp_mean

#         try:
#             corr = float(np.corrcoef(r, y_true)[0, 1])
#         except Exception:
#             corr = float("nan")

#         model_name = self.__str__()
#         print(
#             f"[{model_name.split('(')[0]}] "
#             f"Loss: {np.mean(mse_value) + pen_cov_value + pen_disp_value:.6f} | "
#             f"MSE: {np.mean(mse_value):.6f} | "
#             f"Cov/Std: {cov_std:.6e} | Hinge: {hinge:.6e} | "
#             f"DispMean: {disp_mean:.6e} | "
#             f"PenCov: {pen_cov_value:.6f} | PenDisp: {pen_disp_value:.6f} | "
#             f"Corr(r,y): {corr:.6f}"
#         )

#         # base MSE grads/hess (sum of squares)
#         grad_base = 2.0 * (y_pred - y_true)
#         hess_base = 2.0 * np.ones_like(y_pred)

#         # ---- Cov hinge penalty grads/hess (diag approx) ----
#         # cov_std = (1/std_y) * (1/n) * sum_i r_i * yc_i
#         # dcov_std/dy_pred_i = (1/std_y) * (1/n) * yc_i * d(r_i)/dy_pred_i
#         # d(r_i)/dy_pred_i = 1/denom_i
#         a = (yc / denom) / (float(n) * self.y_std_)  # dcov_std/dy_pred_i

#         if cov_std < 0.0:
#             # hinge = -cov_std, penalty = 0.5*rho_cov*n*hinge^2 = 0.5*rho_cov*n*cov_std^2
#             grad_cov = self.rho_cov * float(n) * cov_std * a
#             hess_cov = self.rho_cov * float(n) * (a ** 2)
#         else:
#             grad_cov = np.zeros_like(y_pred)
#             hess_cov = np.zeros_like(y_pred)

#         # ---- Dispersion penalty grads/hess (separable; exact diag) ----
#         # penalty = 0.5*rho_disp*n*mean((r-1)^2) = 0.5*rho_disp*sum((r-1)^2)
#         # grad_i = rho_disp*(r_i - 1)*d(r_i)/dy_pred_i = rho_disp*(r_i - 1)/denom_i
#         # hess_i = rho_disp*(1/denom_i^2)
#         inv_denom = 1.0 / denom
#         grad_disp = self.rho_disp * (r - 1.0) * inv_denom
#         hess_disp = self.rho_disp * (inv_denom ** 2)

#         grad = grad_base + grad_cov + grad_disp
#         hess = hess_base + hess_cov + hess_disp

#         # numerical guards (same style as your original)
#         grad[np.abs(grad) < self.zero_grad_tol] = self.zero_grad_tol
#         hess[hess < self.zero_grad_tol] = self.zero_grad_tol

#         return grad, hess

#     def __str__(self):
#         return f"LGBCovDispPenalty(rho_cov={self.rho_cov}, rho_disp={self.rho_disp})"

# class LGBCovDispPenalty:
#     """LightGBM objective:
#         MSE
#         + rho_cov * (penalty on Cov(r,y)/Std(y))^2
#         + rho_disp * mean((r - 1)^2)

#     with r = y_pred / y_true (safe-divided by eps_y).

#     Cov term workflow (cov_mode):
#       - "cov"     : penalize (Cov(r,y)/Std(y))^2   (replicates LGBCovPenalty behavior, but standardized)
#       - "neg_cov" : penalize (max(0, -Cov(r,y)/Std(y)))^2  (only regressivity direction)

#     Implementation notes:
#       - Global Cov uses diagonal Hessian approximation (same spirit as your LGBCovPenalty).
#       - Dispersion term is separable; diagonal Hessian is exact.
#       - We keep the same scaling style as LGBCovPenalty:
#             cov_pen = 0.5 * rho_cov  * n * term^2
#             disp_pen = 0.5 * rho_disp * n * mean((r-1)^2) = 0.5*rho_disp*sum((r-1)^2)
#         so gradients stay O(1) in magnitude.
#     """

#     def __init__(
#         self,
#         rho_cov=1e-3,
#         rho_disp=1e-3,
#         cov_mode="cov",          # "cov" or "neg_cov"
#         zero_grad_tol=1e-6,
#         eps_y=1e-12,
#         eps_std=1e-12,
#         lgbm_params=None,
#     ):
#         self.rho_cov = float(rho_cov)
#         self.rho_disp = float(rho_disp)
#         self.cov_mode = str(cov_mode)
#         self.zero_grad_tol = float(zero_grad_tol)
#         self.eps_y = float(eps_y)
#         self.eps_std = float(eps_std)
#         self.model = lgb.LGBMRegressor(**(lgbm_params or {}))

#     def fit(self, X, y):
#         y = np.asarray(y)
#         self.y_mean_ = float(np.mean(y))
#         self.y_std_ = float(np.std(y))  # ddof=0
#         if self.y_std_ < self.eps_std:
#             self.y_std_ = self.eps_std

#         if self.cov_mode not in ("cov", "neg_cov"):
#             raise ValueError(f"cov_mode must be 'cov' or 'neg_cov', got {self.cov_mode!r}")

#         self.model.set_params(objective=self.fobj)
#         self.model.fit(X, y)
#         return self

#     def predict(self, X):
#         return self.model.predict(X)

#     def fobj(self, y_true, y_pred):
#         y_true = np.asarray(y_true)
#         y_pred = np.asarray(y_pred)
#         n = y_pred.size

#         denom = np.maximum(np.abs(y_true), self.eps_y)
#         inv_denom = 1.0 / denom
#         r = y_pred * inv_denom
#         yc = (y_true - self.y_mean_)  # centered y

#         # --- standardized covariance term ---
#         cov = float(np.mean(r * yc))                   # Cov(r,y) since E[yc]=0
#         cov_std = cov / self.y_std_                    # Cov(r,y)/Std(y)

#         if self.cov_mode == "cov":
#             cov_term = cov_std                         # penalize cov_std^2
#             cov_active = True
#         else:
#             # "neg_cov": penalize max(0, -cov_std)^2
#             cov_term = max(0.0, -cov_std)
#             cov_active = (cov_std < 0.0)

#         # --- dispersion term ---
#         disp_mean = float(np.mean((r - 1.0) ** 2))

#         # objective pieces (for prints)
#         mse_value = (y_true - y_pred) ** 2
#         pen_cov_value = 0.5 * self.rho_cov * float(n) * (cov_term ** 2)
#         pen_disp_value = 0.5 * self.rho_disp * float(n) * disp_mean

#         try:
#             corr = float(np.corrcoef(r, y_true)[0, 1])
#         except Exception:
#             corr = float("nan")

#         model_name = self.__str__()
#         print(
#             f"[{model_name.split('(')[0]}] "
#             f"MSE: {np.mean(mse_value):.6f} | "
#             f"Cov/Std: {cov_std:.6e} | Mode: {self.cov_mode} | CovTerm: {cov_term:.6e} | "
#             f"DispMean: {disp_mean:.6e} | "
#             f"PenCov: {pen_cov_value:.6f} | PenDisp: {pen_disp_value:.6f} | "
#             f"Corr(r,y): {corr:.6f}"
#         )

#         # base MSE grads/hess
#         grad_base = 2.0 * (y_pred - y_true)
#         hess_base = 2.0 * np.ones_like(y_pred)

#         # --- Cov penalty grads/hess (diag approx) ---
#         # cov_std = (1/std_y) * (1/n) * sum_i r_i * yc_i
#         # d(cov_std)/d(y_pred_i) = (1/std_y) * (1/n) * yc_i * d(r_i)/d(y_pred_i)
#         # d(r_i)/d(y_pred_i) = 1/denom_i
#         a = (yc * inv_denom) / (float(n) * self.y_std_)  # d(cov_std)/d(y_pred_i)

#         if cov_active:
#             if self.cov_mode == "cov":
#                 # penalty = 0.5*rho_cov*n*(cov_std)^2
#                 grad_cov = self.rho_cov * float(n) * cov_std * a
#                 hess_cov = self.rho_cov * float(n) * (a ** 2)
#             else:
#                 # "neg_cov": penalty = 0.5*rho_cov*n*(max(0,-cov_std))^2
#                 # for cov_std<0, cov_term=-cov_std and penalty reduces to 0.5*rho_cov*n*(cov_std)^2
#                 grad_cov = self.rho_cov * float(n) * cov_std * a
#                 hess_cov = self.rho_cov * float(n) * (a ** 2)
#         else:
#             grad_cov = np.zeros_like(y_pred)
#             hess_cov = np.zeros_like(y_pred)

#         # --- Dispersion penalty grads/hess (separable; exact diag) ---
#         # disp penalty = 0.5*rho_disp*n*mean((r-1)^2) = 0.5*rho_disp*sum((r-1)^2)
#         # grad_i = rho_disp*(r_i - 1)/denom_i
#         # hess_i = rho_disp*(1/denom_i^2)
#         grad_disp = self.rho_disp * (r - 1.0) * inv_denom
#         hess_disp = self.rho_disp * (inv_denom ** 2)

#         grad = grad_base + grad_cov + grad_disp
#         hess = hess_base + hess_cov + hess_disp

#         # numerical guards (same style as your original)
#         grad[np.abs(grad) < self.zero_grad_tol] = self.zero_grad_tol
#         hess[hess < self.zero_grad_tol] = self.zero_grad_tol

#         return grad, hess

#     def __str__(self):
#         return f"LGBCovDispPenalty(rho_cov={self.rho_cov}, rho_disp={self.rho_disp}, mode={self.cov_mode})"

class LGBCovDispPenalty:
    """LightGBM objective:
        MSE
        + rho_cov  * (penalty on Cov(r,y)/Std(y))^2
        + rho_disp * mean_dispersion_loss(r - 1)

    with r = y_pred / y_true (safe-divided by eps_y).

    Cov term workflow (cov_mode):
      - "cov"     : penalize (Cov(r,y)/Std(y))^2
      - "neg_cov" : penalize (max(0, -Cov(r,y)/Std(y)))^2

    Dispersion workflow (disp_mode):
      - "l2"          : ell(u) = 0.5 * u^2   (so grad matches your prior scaling)
      - "pseudohuber" : ell(u) = delta^2 (sqrt(1 + (u/delta)^2) - 1)

    Scaling:
      cov_pen  = 0.5 * rho_cov  * n * cov_term^2
      disp_pen =       rho_disp * n * mean(ell(u))   (ell(u) ~ 0.5 u^2 near 0)
    """

    def __init__(
        self,
        rho_cov=1e-3,
        rho_disp=1e-3,
        cov_mode="cov",            # "cov" or "neg_cov"
        disp_mode="l2",            # "l2" or "pseudohuber"
        huber_delta=0.10,          # only used if disp_mode="pseudohuber" (u = r-1)
        zero_grad_tol=1e-6,
        eps_y=1e-12,
        eps_std=1e-12,
        eps_delta=1e-12,
        lgbm_params=None,
    ):
        self.rho_cov = float(rho_cov)
        self.rho_disp = float(rho_disp)
        self.cov_mode = str(cov_mode)
        self.disp_mode = str(disp_mode)
        self.huber_delta = float(huber_delta)
        self.zero_grad_tol = float(zero_grad_tol)
        self.eps_y = float(eps_y)
        self.eps_std = float(eps_std)
        self.eps_delta = float(eps_delta)
        self.model = lgb.LGBMRegressor(**(lgbm_params or {}))

    def fit(self, X, y):
        y = np.asarray(y)
        self.y_mean_ = float(np.mean(y))
        self.y_std_ = float(np.std(y))  # ddof=0
        if self.y_std_ < self.eps_std:
            self.y_std_ = self.eps_std

        if self.cov_mode not in ("cov", "neg_cov"):
            raise ValueError(f"cov_mode must be 'cov' or 'neg_cov', got {self.cov_mode!r}")

        if self.disp_mode not in ("l2", "pseudohuber"):
            raise ValueError(f"disp_mode must be 'l2' or 'pseudohuber', got {self.disp_mode!r}")

        if self.disp_mode == "pseudohuber" and self.huber_delta <= 0:
            raise ValueError(f"huber_delta must be > 0, got {self.huber_delta}")

        self.model.set_params(objective=self.fobj)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def _disp_loss_and_derivs(self, u):
        """Return (ell(u), ell'(u), ell''(u)) elementwise."""
        if self.disp_mode == "l2":
            # ell(u) = 0.5 u^2  => ell'(u)=u, ell''(u)=1
            ell = 0.5 * (u ** 2)
            ell_p = u
            ell_pp = np.ones_like(u)
            return ell, ell_p, ell_pp

        # pseudo-Huber:
        # ell(u) = d^2 (sqrt(1+(u/d)^2) - 1)
        # ell'(u) = u / sqrt(1+(u/d)^2)
        # ell''(u)= 1 / (1+(u/d)^2)^(3/2)
        d = max(self.huber_delta, self.eps_delta)
        t = u / d
        s = np.sqrt(1.0 + t * t)  # sqrt(1+(u/d)^2)
        ell = (d * d) * (s - 1.0)
        ell_p = u / s
        ell_pp = 1.0 / (s ** 3)
        return ell, ell_p, ell_pp

    def fobj(self, y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        n = y_pred.size

        denom = np.maximum(np.abs(y_true), self.eps_y)
        inv_denom = 1.0 / denom
        r = y_pred * inv_denom
        u = r - 1.0
        yc = (y_true - self.y_mean_)  # centered y

        # --- standardized covariance term ---
        cov = float(np.mean(r * yc))        # Cov(r,y) since E[yc]=0
        cov_std = cov / self.y_std_         # Cov(r,y)/Std(y)

        if self.cov_mode == "cov":
            cov_term = cov_std
            cov_active = True
        else:
            cov_term = max(0.0, -cov_std)
            cov_active = (cov_std < 0.0)

        # --- dispersion term (L2 or pseudo-Huber) ---
        ell_u, ell_p_u, ell_pp_u = self._disp_loss_and_derivs(u)
        disp_loss_mean = float(np.mean(ell_u))
        u2_mean = float(np.mean(u ** 2))  # helpful diagnostic

        # objective pieces (for prints)
        mse_value = (y_true - y_pred) ** 2
        pen_cov_value = 0.5 * self.rho_cov * float(n) * (cov_term ** 2)
        pen_disp_value = self.rho_disp * float(n) * disp_loss_mean

        try:
            corr = float(np.corrcoef(r, y_true)[0, 1])
        except Exception:
            corr = float("nan")

        model_name = self.__str__()
        print(
            f"[{model_name.split('(')[0]}] "
            f"MSE: {np.mean(mse_value):.6f} | "
            f"Cov/Std: {cov_std:.6e} | Mode: {self.cov_mode} | CovTerm: {cov_term:.6e} | "
            f"DispMode: {self.disp_mode} | DispLossMean: {disp_loss_mean:.6e} | (r-1)^2 mean: {u2_mean:.6e} | "
            f"PenCov: {pen_cov_value:.6f} | PenDisp: {pen_disp_value:.6f} | "
            f"Corr(r,y): {corr:.6f}"
        )

        # base MSE grads/hess
        grad_base = 2.0 * (y_pred - y_true)
        hess_base = 2.0 * np.ones_like(y_pred)

        # --- Cov penalty grads/hess (diag approx) ---
        a = (yc * inv_denom) / (float(n) * self.y_std_)  # d(cov_std)/d(y_pred_i)

        if cov_active:
            # for both "cov" and active "neg_cov", penalty behaves as 0.5*rho_cov*n*(cov_std)^2
            grad_cov = self.rho_cov * float(n) * cov_std * a
            hess_cov = self.rho_cov * float(n) * (a ** 2)
        else:
            grad_cov = np.zeros_like(y_pred)
            hess_cov = np.zeros_like(y_pred)

        # --- Dispersion penalty grads/hess (separable; exact diag) ---
        # pen_disp = rho_disp * sum ell(u_i)
        # u_i = r_i - 1, r_i = y_pred_i / denom_i
        # du/dy_pred = 1/denom
        grad_disp = self.rho_disp * ell_p_u * inv_denom
        hess_disp = self.rho_disp * ell_pp_u * (inv_denom ** 2)

        grad = grad_base + grad_cov + grad_disp
        hess = hess_base + hess_cov + hess_disp

        # numerical guards (same style as your original)
        grad[np.abs(grad) < self.zero_grad_tol] = self.zero_grad_tol
        hess[hess < self.zero_grad_tol] = self.zero_grad_tol

        return grad, hess

    def __str__(self):
        extra = f", disp={self.disp_mode}"
        if self.disp_mode == "pseudohuber":
            extra += f"(d={self.huber_delta})"
        return f"LGBCovDispPenalty(rho_cov={int(self.rho_cov)}, rho_disp={int(self.rho_disp)}, mode={self.cov_mode}{extra})"























# ###
# # 5) Teaddier( Gamma-Poisson like loss) verison with cov penalty for using in prices rather than log-prices
# ##

# class LGBCorrTweediePenalty:
#     """
#     LightGBM custom objective (raw score = s = log(mu)):

#         Tweedie NLL (log-link)  +  0.5 * rho * n * Corr(r, y)^2

#     where:
#         mu = exp(s)                                (mean on price scale)
#         r  = mu / y                                (ratio on price scale)
#         Corr(r,y) uses y-centered y and r-centered r (scale-free, avoids $-units blowups)

#     Why correlation (not covariance)?
#       - Cov(r,y) has units of dollars and can be huge even when Corr is small.
#       - Corr(r,y) is dimensionless and typically O(1), so the penalty magnitude is controlled.

#     Tweedie base derivatives (LightGBM-style in terms of score s=log(mu)):
#         grad_base = -y * exp((1-p)*s) + exp((2-p)*s)
#         hess_base = -y*(1-p)*exp((1-p)*s) + (2-p)*exp((2-p)*s)
#       with p in [1, 2).  (Commonly p ~ 1.1-1.95 for positive, skewed targets.)

#     Correlation penalty (global statistic, dense Hessian in principle):
#       We use a diagonal approximation for hess_pen:
#           hess_pen_i ≈ rho * n * (d corr / d s_i)^2

#     Optional GSC v=2-style "upper bound damping" (heuristic):
#       - Compute per-sample 1D Newton step: step_i = -g_i/h_i
#       - beta_i = M * |step_i|
#       - tau_i = log(1+beta_i)/beta_i   ("upper" damping)
#       - apply_to="grad": g_i <- tau_i * g_i
#         apply_to="hess": h_i <- h_i / tau_i
#       This shrinks the Newton step in exp-family losses to improve stability.
#     """

#     def __init__(
#         self,
#         rho=1e-3,
#         tweedie_p=1.5,
#         zero_grad_tol=1e-12,
#         eps_y=1e-12,
#         eps_var=1e-12,
#         clip_score=20.0,
#         # GSC damping (optional)
#         gsc_mode="taylor",          # "taylor" or "upper" (recommended). "lower" omitted for safety.
#         gsc_apply_to="grad",        # "grad" or "hess"
#         gsc_M=None,                 # if None: max(2-p, p-1)
#         gsc_M_mult=1.0,             # multiplies default M
#         # penalty options
#         use_corr=True,              # keep True; here mainly for clarity/compat
#         verbose_print=True,
#         lgbm_params=None,
#     ):
#         self.rho = float(rho)
#         self.p = float(tweedie_p)
#         self.zero_grad_tol = float(zero_grad_tol)
#         self.eps_y = float(eps_y)
#         self.eps_var = float(eps_var)
#         self.clip_score = float(clip_score)

#         self.gsc_mode = gsc_mode
#         self.gsc_apply_to = gsc_apply_to
#         self.gsc_M = gsc_M
#         self.gsc_M_mult = float(gsc_M_mult)

#         self.use_corr = bool(use_corr)
#         self.verbose_print = bool(verbose_print)

#         self.model = lgb.LGBMRegressor(**(lgbm_params or {}))

#         # cached from fit()
#         self.y_mean_ = None
#         self.y_std_ = None
#         self.n_ = None
#         self.gsc_M_ = None

#     def fit(self, X, y):
#         y = np.asarray(y, dtype=float)

#         if np.any(y <= 0):
#             raise ValueError("Tweedie with log-link requires y > 0 (prices must be positive).")

#         if not (1.0 <= self.p < 2.0):
#             raise ValueError("For LightGBM Tweedie, use tweedie_p in [1, 2).")

#         self.y_mean_ = float(np.mean(y))
#         self.y_std_ = float(np.std(y))
#         self.y_std_ = max(self.y_std_, self.eps_var)
#         self.n_ = int(y.size)

#         # Default 1D exp-family constant for the composed Tweedie exp terms
#         if self.gsc_M is None:
#             base_M = max(2.0 - self.p, self.p - 1.0, 1e-12)
#             self.gsc_M_ = float(self.gsc_M_mult * base_M)
#         else:
#             self.gsc_M_ = float(self.gsc_M)

#         self.model.set_params(objective=self.fobj)
#         self.model.fit(X, y)
#         return self

#     def predict_score(self, X):
#         """Return raw score s = log(mu)."""
#         return self.model.predict(X)

#     def predict(self, X):
#         """Return mean prediction mu on the price scale."""
#         s = self.model.predict(X)
#         s = np.clip(s, -self.clip_score, self.clip_score)
#         return np.exp(s)

#     # ---- GSC helper ----
#     def _tau_upper(self, beta):
#         beta = np.asarray(beta, dtype=float)
#         out = np.ones_like(beta)
#         m = beta > 1e-12
#         out[m] = np.log1p(beta[m]) / beta[m]
#         return out

#     # ---- custom objective ----
#     def fobj(self, y_true, y_score):
#         y = np.asarray(y_true, dtype=float)
#         s = np.asarray(y_score, dtype=float)
#         n = float(s.size)

#         # safety for exp()
#         s = np.clip(s, -self.clip_score, self.clip_score)

#         p = self.p

#         # ========== Tweedie base gradients/hessians (in score space) ==========
#         exp_1 = np.exp((1.0 - p) * s)   # mu^(1-p)
#         exp_2 = np.exp((2.0 - p) * s)   # mu^(2-p)

#         grad_base = (-y * exp_1) + exp_2
#         hess_base = (-y * (1.0 - p) * exp_1) + ((2.0 - p) * exp_2)
#         hess_base = np.maximum(hess_base, self.zero_grad_tol)

#         # ========== Optional GSC "upper" damping (recommended for stability) ==========
#         if self.gsc_mode is not None and self.gsc_mode != "taylor":
#             if self.gsc_mode != "upper":
#                 raise ValueError("Use gsc_mode='taylor' or 'upper' (lower is intentionally omitted).")

#             step_1d = -grad_base / (hess_base + self.zero_grad_tol)
#             beta = self.gsc_M_ * np.abs(step_1d)
#             tau = self._tau_upper(beta)
#             tau = np.maximum(tau, 1e-12)

#             if self.gsc_apply_to == "grad":
#                 grad_base = tau * grad_base
#             elif self.gsc_apply_to == "hess":
#                 hess_base = hess_base / tau
#             else:
#                 raise ValueError("gsc_apply_to must be 'grad' or 'hess'.")

#             hess_base = np.maximum(hess_base, self.zero_grad_tol)

#         # ========== Correlation penalty on r = mu/y (price-scale ratio) ==========
#         mu = np.exp(s)
#         denom = np.maximum(y, self.eps_y)
#         r = mu / denom

#         yc = y - self.y_mean_
#         sy = self.y_std_  # constant from fit()

#         # stats of r
#         c = float(np.mean(r * yc))          # mean(r * (y-mean_y))
#         m1 = float(np.mean(r))
#         m2 = float(np.mean(r * r))
#         var_r = max(m2 - m1 * m1, self.eps_var)
#         sr = np.sqrt(var_r)

#         # corr = c / (sr * sy)
#         corr = c / (sr * sy)

#         # derivatives wrt score s
#         # dr/ds = r
#         dc_ds = (yc * r) / n
#         dm1_ds = r / n
#         dm2_ds = (2.0 * r * r) / n
#         dvar_ds = dm2_ds - 2.0 * m1 * dm1_ds
#         dsr_ds = 0.5 * dvar_ds / sr

#         # dcorr/ds = ( (dc/sr) - c*(dsr/sr^2) ) / sy
#         dcorr_ds = (dc_ds / sr - c * dsr_ds / (sr * sr)) / sy

#         # penalty = 0.5 * rho * n * corr^2
#         # grad_pen = rho * n * corr * dcorr_ds
#         # hess_pen (diag approx) = rho * n * (dcorr_ds)^2
#         grad_pen = self.rho * n * corr * dcorr_ds
#         hess_pen = self.rho * n * (dcorr_ds ** 2)

#         grad = grad_base + grad_pen
#         hess = hess_base + hess_pen

#         # numerical cleanup
#         grad = np.where(np.isfinite(grad), grad, 0.0)
#         hess = np.where(np.isfinite(hess), hess, self.zero_grad_tol)
#         hess = np.maximum(hess, self.zero_grad_tol)

#         if self.verbose_print:
#             # Tweedie loss (up to constants) for monitoring
#             if abs(1.0 - p) > 1e-12 and abs(2.0 - p) > 1e-12:
#                 tweedie_loss = (-y * exp_1) / (1.0 - p) + (exp_2) / (2.0 - p)
#                 base_mean = float(np.mean(tweedie_loss))
#             else:
#                 base_mean = float("nan")

#             pen_value = 0.5 * self.rho * n * (corr ** 2)

#             print(
#                 f"[LGBCorrTweediePenalty] "
#                 f"Base(Tweedie): {base_mean:.6f} | "
#                 f"Corr(r,y): {corr:.6f} | Pen: {pen_value:.6f} | "
#                 f"p={self.p:.3f} gsc={self.gsc_mode}/{self.gsc_apply_to}"
#             )

#         return grad, hess

#     def __str__(self):
#         return f"LGBCorrTweediePenalty(rho={self.rho}, p={self.p}, mode={self.gsc_mode})"



# import numpy as np
# import lightgbm as lgb


# class LGBCovTweediePenalty:
#     """
#     LightGBM objective: Tweedie NLL (log-link) + 0.5 * rho * n * (Cov(r, y))^2
#     where:
#         score = y_pred is the raw margin
#         mu    = exp(score)  (log-link)
#         r_i   = mu_i / y_i   (ratio in ORIGINAL price space)
#         Cov(r,y) = (1/n) * sum_i r_i * (y_i - y_mean)

#     Tweedie (variance power p in (1,2) typical):
#       Using score = log(mu), LightGBM’s built-in objective uses:
#         grad_base = -y * exp((1-p)*score) + exp((2-p)*score)
#         hess_base = -y*(1-p)*exp((1-p)*score) + (2-p)*exp((2-p)*score)
#       (this corresponds to the Tweedie NLL written in terms of mu, composed with mu=exp(score)).
#       See discussion of metric-vs-objective in LightGBM Tweedie. :contentReference[oaicite:1]{index=1}

#     Cov penalty (direct, like your LGBCovPenalty but with mu=exp(score)):
#       cov = mean(r * yc), yc = y - y_mean
#       r   = mu / y
#       d r / d score = r
#       d cov / d score_i = (1/n) * yc_i * r_i

#       penalty = 0.5 * rho * n * cov^2
#       grad_pen_i = rho * n * cov * d cov/d score_i = rho * cov * yc_i * r_i
#       hess_pen_i (diag approx) = rho * n * (d cov/d score_i)^2
#                                = rho * (yc_i^2 * r_i^2) / n

#     Optional "GSC v=2 upper-bound damping" (heuristic inside LightGBM):
#       - Compute the 1D Newton step per sample: step_i = -grad_base / hess_base
#       - beta_i = M * |step_i|
#       - tau_i  = log(1+beta_i)/beta_i   (upper-bound damped Newton factor)
#       - Then either:
#           apply_to="grad": grad_base <- tau_i * grad_base   (shrinks step)
#           apply_to="hess": hess_base <- hess_base / tau_i   (also shrinks step)
#       NOTE: because LightGBM aggregates grads/hessians per leaf, this is only an approximation,
#             but it often stabilizes “exp-family” losses.

#     Parameters
#     ----------
#     rho : float
#         Penalty weight for covariance term.
#     tweedie_p : float
#         Tweedie variance power p (LightGBM calls this tweedie_variance_power).
#         Commonly p in (1,2) for compound Poisson-Gamma style severity.
#     target_is_log : bool
#         If True, y passed to fit() is log(price). The Tweedie loss still expects y in original space,
#         so we internally map y_price = exp(y_log). (This is mainly to match your current pipeline.)
#     gsc_mode : {"taylor","upper","lower"}
#         Damping choice. "taylor" = no damping. "upper" = log(1+beta)/beta. "lower" is optional
#         and requires beta < 1 (otherwise we fall back to upper).
#     gsc_apply_to : {"grad","hess"}
#         Where to apply tau (see above).
#     gsc_M : float or None
#         If None, we use max(2-p, p-1) as a simple 1D “exp-mixture” bound constant for Tweedie.
#     """

#     def __init__(
#         self,
#         rho=1e-3,
#         tweedie_p=1.5,
#         target_is_log=False,
#         zero_grad_tol=1e-12,
#         eps_y=1e-12,
#         clip_score=50.0,
#         gsc_mode="taylor",        # "taylor", "upper", "lower"
#         gsc_apply_to="grad",      # "grad" or "hess"
#         gsc_M=None,
#         lgbm_params=None,
#         verbose_print=True,
#     ):
#         self.rho = float(rho)
#         self.p = float(tweedie_p)
#         self.target_is_log = bool(target_is_log)
#         self.zero_grad_tol = float(zero_grad_tol)
#         self.eps_y = float(eps_y)
#         self.clip_score = float(clip_score)
#         self.gsc_mode = gsc_mode
#         self.gsc_apply_to = gsc_apply_to
#         self.gsc_M = gsc_M
#         self.verbose_print = bool(verbose_print)

#         self.model = lgb.LGBMRegressor(**(lgbm_params or {}))

#     def fit(self, X, y):
#         y = np.asarray(y, dtype=float)
#         if self.target_is_log:
#             self.y_price_ = np.exp(np.clip(y, -self.clip_score, self.clip_score))
#         else:
#             self.y_price_ = y

#         self.y_mean_ = float(np.mean(self.y_price_))
#         self.n_ = int(self.y_price_.size)

#         # default M if not provided
#         if self.gsc_M is None:
#             # simple safe-ish constant for Tweedie exp terms
#             self.gsc_M_ = float(max(2.0 - self.p, self.p - 1.0, 1e-12))
#         else:
#             self.gsc_M_ = float(self.gsc_M)

#         self.model.set_params(objective=self.fobj)
#         self.model.fit(X, y)
#         return self

#     def predict(self, X):
#         # LightGBM will output "score" for our custom objective;
#         # if you want mu (price-scale mean), do exp(score).
#         score = self.model.predict(X)
#         return score

#     def _tau_upper(self, beta):
#         beta = np.asarray(beta, dtype=float)
#         out = np.ones_like(beta)
#         m = beta > 1e-12
#         out[m] = np.log1p(beta[m]) / beta[m]
#         return out

#     def _tau_lower(self, beta):
#         # lower: -log(1-beta)/beta, requires beta<1; fallback to upper if not
#         beta = np.asarray(beta, dtype=float)
#         out = self._tau_upper(beta)
#         m = beta < (1.0 - 1e-12)
#         out[m] = (-np.log1p(-beta[m])) / beta[m]
#         return out

#     def fobj(self, y_true, y_score):
#         y_true = np.asarray(y_true, dtype=float)
#         y_score = np.asarray(y_score, dtype=float)
#         n = y_score.size

#         # price-space labels for Tweedie + cov penalty
#         if self.target_is_log:
#             y_price = np.exp(np.clip(y_true, -self.clip_score, self.clip_score))
#         else:
#             y_price = y_true

#         # raw score is log(mu)
#         s = np.clip(y_score, -self.clip_score, self.clip_score)

#         # ===== Tweedie base grads/hess (LightGBM-style: in terms of score=log(mu)) =====
#         p = self.p
#         exp_1 = np.exp((1.0 - p) * s)  # = mu^(1-p)
#         exp_2 = np.exp((2.0 - p) * s)  # = mu^(2-p)

#         grad_base = (-y_price * exp_1) + exp_2
#         hess_base = (-y_price * (1.0 - p) * exp_1) + ((2.0 - p) * exp_2)

#         # keep Hessian positive (LightGBM expects this)
#         hess_base = np.maximum(hess_base, self.zero_grad_tol)

#         # ===== Optional GSC-style damped-Newton heuristic for exp-family curvature =====
#         if self.gsc_mode and self.gsc_mode != "taylor":
#             step_1d = -grad_base / (hess_base + self.zero_grad_tol)
#             beta = self.gsc_M_ * np.abs(step_1d)

#             if self.gsc_mode == "upper":
#                 tau = self._tau_upper(beta)
#             elif self.gsc_mode == "lower":
#                 tau = self._tau_lower(beta)
#             else:
#                 raise ValueError(f"Unknown gsc_mode={self.gsc_mode}")

#             tau = np.maximum(tau, 1e-12)

#             if self.gsc_apply_to == "grad":
#                 grad_base = tau * grad_base
#             elif self.gsc_apply_to == "hess":
#                 hess_base = hess_base / tau
#             else:
#                 raise ValueError(f"Unknown gsc_apply_to={self.gsc_apply_to}")

#             hess_base = np.maximum(hess_base, self.zero_grad_tol)

#         # ===== Cov penalty in price space, using mu=exp(score) =====
#         mu = np.exp(s)
#         denom = np.maximum(np.abs(y_price), self.eps_y)
#         r = mu / denom
#         yc = (y_price - self.y_mean_)

#         cov = float(np.mean(r * yc))  # E[yc]=0 by definition, so this is covariance up to scaling

#         # penalty gradients/hessians (diag approx)
#         # d cov / d s_i = (1/n) * yc_i * r_i
#         a = (yc * r) / float(n)
#         grad_pen = self.rho * float(n) * cov * a
#         hess_pen = self.rho * float(n) * (a ** 2)

#         grad = grad_base + grad_pen
#         hess = hess_base + hess_pen

#         # numerical safety
#         grad = np.where(np.isfinite(grad), grad, 0.0)
#         hess = np.where(np.isfinite(hess), hess, self.zero_grad_tol)
#         hess = np.maximum(hess, self.zero_grad_tol)

#         if self.verbose_print:
#             try:
#                 corr = float(np.corrcoef(r, y_price)[0, 1])
#             except Exception:
#                 corr = float("nan")

#             # Tweedie NLL piece (up to constants)
#             # L = - y * mu^(1-p)/(1-p) + mu^(2-p)/(2-p)
#             # using exp_1, exp_2: mu^(1-p)=exp_1, mu^(2-p)=exp_2
#             if abs(1.0 - p) > 1e-12 and abs(2.0 - p) > 1e-12:
#                 tweedie_loss = (-y_price * exp_1) / (1.0 - p) + (exp_2) / (2.0 - p)
#                 base_mean = float(np.mean(tweedie_loss))
#             else:
#                 base_mean = float("nan")

#             pen_value = 0.5 * self.rho * float(n) * (cov ** 2)

#             print(
#                 f"[LGBCovTweediePenalty] "
#                 f"Base(Tweedie): {base_mean:.6f} | "
#                 f"Cov: {cov:.3e} | Pen: {pen_value:.6f} | Corr(r,y): {corr:.6f} | "
#                 f"p={self.p:.3f} gsc={self.gsc_mode}/{self.gsc_apply_to}"
#             )

#         return grad, hess

#     def __str__(self):
#         return f"LGBCovTweediePenalty(rho={self.rho}, p={self.p}, mode={self.gsc_mode})"



