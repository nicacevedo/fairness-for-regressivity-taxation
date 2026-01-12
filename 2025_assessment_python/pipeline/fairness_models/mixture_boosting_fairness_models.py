
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import lightgbm as lgb
import torch

# Optional sklearn/scipy
try:
    from sklearn.model_selection import KFold
    from sklearn.cluster import KMeans, MiniBatchKMeans
    from sklearn.decomposition import TruncatedSVD
except Exception:  # pragma: no cover
    KFold = None
    KMeans = None
    MiniBatchKMeans = None
    TruncatedSVD = None

try:  # pragma: no cover
    import scipy.sparse as sp
except Exception:  # pragma: no cover
    sp = None






# 2) Mixture (upper are 2-stage)
import numpy as np
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
import copy





# ---------------------------------------------------------
# 1. The Gating Network (PyTorch)
# ---------------------------------------------------------
class GatingNetwork(nn.Module):
    """
    A simple Neural Network that outputs mixing weights (Softmax).
    Input: Features X -> Output: K probabilities summing to 1.
    """
    def __init__(self, input_dim, n_experts):
        super().__init__()
        # Using a simple 1-hidden layer net for non-linear flexibility
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_experts),
            nn.Softmax(dim=1)  # Ensures outputs sum to 1 (convex combination)
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------
# 2. The MoE Model (Main Class)
# ---------------------------------------------------------
class MoELGBSmoothPenalty:
    """
    Mixture of Experts with LightGBM Experts and Neural Network Gate.
    Optimizes Global MSE + rho * Separable Surrogate via Block Coordinate Descent.
    """

    def __init__(self, 
                 n_experts=3, 
                 rho=1e-3, 
                 zero_grad_tol=1e-6, 
                 eps_y=1e-12, 
                 lgbm_params=None,
                 n_outer_iters=20,     # Number of Descent Steps (Gate <-> Experts)
                 trees_per_iter=5,     # How many trees to add to each expert per step
                 gate_lr=0.01,         # Learning rate for the Gate
                 gate_epochs=10):      # Epochs to train gate per step
        
        self.n_experts = n_experts
        self.rho = rho
        self.zero_grad_tol = zero_grad_tol
        self.eps_y = eps_y
        self.lgbm_params = lgbm_params or {'verbosity': -1}
        
        # Optimization Hyperparams
        self.n_outer_iters = n_outer_iters
        self.trees_per_iter = trees_per_iter
        self.gate_lr = gate_lr
        self.gate_epochs = gate_epochs
        
        # State
        self.experts = [] # Will hold lgb.Booster objects
        self.gate = None  # Will hold GatingNetwork
        self.y_mean_ = None

    def fit(self, X, y):
        # Convert inputs
        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float32)
        self.y_mean_ = float(np.mean(y_np))
        
        # 1. Initialize Gate
        input_dim = X_np.shape[1]
        self.gate = GatingNetwork(input_dim, self.n_experts)
        gate_optimizer = optim.Adam(self.gate.parameters(), lr=self.gate_lr)
        
        # 2. Initialize Experts (Empty Boosters)
        # We create datasets once to save memory
        lgb_train_set = lgb.Dataset(X_np, label=y_np, free_raw_data=False)
        self.experts = [None] * self.n_experts

        # -----------------------------------------------------
        # MAIN COORDINATE DESCENT LOOP
        # -----------------------------------------------------
        print(f"Starting Training: K={self.n_experts}, rho={self.rho}")
        
        for iteration in range(self.n_outer_iters):
            
            # --- A. PRE-COMPUTE CURRENT STATE ---
            # Get current gate probabilities (Fixed during Expert Update)
            X_tensor = torch.tensor(X_np)
            with torch.no_grad():
                gate_probs = self.gate(X_tensor).numpy() # Shape (N, K)
            
            # Get current expert predictions
            expert_preds = np.zeros((X_np.shape[0], self.n_experts))
            for k in range(self.n_experts):
                if self.experts[k] is None:
                    expert_preds[:, k] = self.y_mean_ # Initialization
                else:
                    expert_preds[:, k] = self.experts[k].predict(X_np)
            
            # --- B. UPDATE EXPERTS (Boosting Step) ---
            for k in range(self.n_experts):
                # 1. Calculate the "Offset" prediction from OTHER experts
                # y_hat = g_k * f_k + Sum_{j!=k} (g_j * f_j)
                # We fix the second term as a constant 'offset'
                other_preds_weighted = np.sum(
                    gate_probs[:, [j for j in range(self.n_experts) if j != k]] * expert_preds[:, [j for j in range(self.n_experts) if j != k]], 
                    axis=1
                )
                
                # 2. Define Custom Objective Closure for Expert k
                # This calculates the GLOBAL loss gradient w.r.t Expert k's output
# 2. Define Custom Objective Closure for Expert k
                # SETTING: Local Accuracy (Per Expert) + Global Fairness (Overall)
                def expert_fobj(y_true, y_pred_current_tree):
                    # y_pred_current_tree is the raw output of THIS expert (f_k)
                    
                    # A. Reconstruct Global Prediction (Needed ONLY for Fairness Penalty)
                    # y_hat = g_k * f_k + Sum_{j!=k} (g_j * f_j)
                    y_total = (gate_probs[:, k] * y_pred_current_tree) + other_preds_weighted
                    
                    # --- Term 1: Local Accuracy Gradient (The "Different Loss per Expert") ---
                    # Loss = Sum [ g_k * (f_k - y)^2 ]
                    # dLoss/df_k = g_k * 2 * (f_k - y)
                    # Note: We use y_pred_current_tree (f_k), NOT y_total
                    grad_local_mse = 2.0 * (y_pred_current_tree - y_true)
                    hess_local_mse = 2.0 * np.ones_like(y_pred_current_tree)
                    
                    # Apply the gating weight directly to the local loss
                    grad_accuracy = grad_local_mse * gate_probs[:, k]
                    hess_accuracy = hess_local_mse * gate_probs[:, k]

                    # --- Term 2: Global Fairness Gradient (The "Overall Penalty") ---
                    # Penalty = rho * Surrogate(y_total)
                    # dPenalty/df_k = rho * dSurrogate/dy_total * dy_total/df_k
                    # dy_total/df_k = gate_probs[:, k]
                    
                    z = y_true
                    zc = (y_true - self.y_mean_)
                    denom = np.maximum(np.abs(z), self.eps_y)
                    
                    # Calculate gradient w.r.t GLOBAL prediction first
                    scale = (zc / denom) ** 2
                    grad_surr_global = 2.0 * (y_total - z) * scale
                    hess_surr_global = 2.0 * scale
                    
                    # Project it onto the expert via the Gate
                    grad_fairness = self.rho * grad_surr_global * gate_probs[:, k]
                    hess_fairness = self.rho * hess_surr_global * (gate_probs[:, k] ** 2)
                    
                    # --- Combine ---
                    # The expert feels its own error strongly, plus a nudge from the global fairness goal
                    grad_final = grad_accuracy + grad_fairness
                    hess_final = hess_accuracy + hess_fairness
                    
                    # Zero tol
                    grad_final[np.abs(grad_final) < self.zero_grad_tol] = self.zero_grad_tol
                    hess_final[hess_final < self.zero_grad_tol] = self.zero_grad_tol
                    
                    return grad_final, hess_final

                # def expert_fobj(y_true, y_pred_current_tree):
                #     # Reconstruct Global Prediction
                #     # Note: y_pred_current_tree is the raw output of THIS expert so far
                #     y_total = (gate_probs[:, k] * y_pred_current_tree) + other_preds_weighted
                    
                #     # --- Compute Global Gradients (MSE + Penalty) ---
                #     # (This logic is copied from your original code)
                #     z = y_true
                #     zc = (y_true - self.y_mean_)
                #     denom = np.maximum(np.abs(z), self.eps_y)
                    
                #     # Base Gradients (MSE) w.r.t Global Prediction
                #     grad_global = 2.0 * (y_total - y_true)
                #     hess_global = 2.0 * np.ones_like(y_total)
                    
                #     # Penalty Gradients (Surrogate) w.r.t Global Prediction
                #     scale = (zc / denom) ** 2
                #     grad_pen = 2.0 * (y_total - z) * scale
                #     hess_pen = 2.0 * scale
                    
                #     grad_total = grad_global + self.rho * grad_pen
                #     hess_total = hess_global + self.rho * hess_pen
                    
                #     # --- CHAIN RULE ---
                #     # We need dL/df_k = dL/dy_total * dy_total/df_k
                #     # dy_total/df_k = gate_prob[k]
                #     grad_k = grad_total * gate_probs[:, k]
                #     hess_k = hess_total * (gate_probs[:, k] ** 2)
                    
                #     # Zero tol
                #     grad_k[np.abs(grad_k) < self.zero_grad_tol] = self.zero_grad_tol
                #     hess_k[hess_k < self.zero_grad_tol] = self.zero_grad_tol
                    
                #     return grad_k, hess_k

                # 3. Boost Expert k (Add trees to existing model)
                self.experts[k] = lgb.train(
                    params=self.lgbm_params,
                    train_set=lgb_train_set,
                    num_boost_round=self.trees_per_iter,
                    fobj=expert_fobj,
                    init_model=self.experts[k], # Continues training!
                    keep_training_booster=True
                )
                
                # Update preds for next expert's calculation
                expert_preds[:, k] = self.experts[k].predict(X_np)

            # --- C. UPDATE GATE (PyTorch Step) ---
            # We fix Expert predictions and update the Gate to minimize loss
            
            # Convert expert preds to tensor (Fixed Constant)
            expert_preds_t = torch.tensor(expert_preds, dtype=torch.float32)
            y_true_t = torch.tensor(y_np, dtype=torch.float32)
            y_mean_t = torch.tensor(self.y_mean_, dtype=torch.float32)
            
            for epoch in range(self.gate_epochs):
                gate_optimizer.zero_grad()
                
                # Forward Pass
                g_probs = self.gate(X_tensor) # differentiable
                
                # ... (forward pass code same as before) ...
                
                # Combine (Mixture)
                y_hat = torch.sum(g_probs * expert_preds_t, dim=1)
                
                # --- CHANGED: Local MSE Loss ---
                # We want the gate to minimize: Sum_k [ g_k * (f_k - y)^2 ]
                # This encourages the gate to pick the expert that is individually best.
                expert_sq_errors = (expert_preds_t - y_true_t.unsqueeze(1)) ** 2  # Shape (N, K)
                loss_local_mse = torch.mean(torch.sum(g_probs * expert_sq_errors, dim=1))
                
                # --- Global Fairness Penalty ---
                # This stays on the Global y_hat
                denom_t = torch.maximum(torch.abs(y_true_t), torch.tensor(self.eps_y))
                zc_t = y_true_t - y_mean_t
                cov_surr = ((y_hat / denom_t) - 1.0)**2 * (zc_t**2)
                loss_surr = torch.mean(cov_surr)
                
                total_loss = loss_local_mse + self.rho * loss_surr
                
                total_loss.backward()
                gate_optimizer.step()

            # --- D. LOGGING ---
            # Using the exact same print format as your request
            self._print_status(iteration, y_np, y_hat.detach().numpy(), total_loss.item(), loss_mse.item(), loss_surr.item())

        return self

    def predict(self, X):
        X_np = np.asarray(X, dtype=np.float32)
        X_t = torch.tensor(X_np)
        
        # 1. Get Weights
        with torch.no_grad():
            g_probs = self.gate(X_t).numpy()
            
        # 2. Get Expert Preds
        expert_preds = np.zeros((X_np.shape[0], self.n_experts))
        for k in range(self.n_experts):
            if self.experts[k] is not None:
                expert_preds[:, k] = self.experts[k].predict(X_np)
            else:
                expert_preds[:, k] = self.y_mean_
                
        # 3. Weighted Sum
        return np.sum(g_probs * expert_preds, axis=1)

    def _print_status(self, it, y_true, y_pred, loss_val, mse_val, surr_val):
        z = y_true
        denom = np.maximum(np.abs(z), self.eps_y)
        r = y_pred / denom
        try:
            corr = float(np.corrcoef(r, y_true)[0, 1])
        except Exception:
            corr = float('nan')
            
        print(
            f"[Iter {it+1}] "
            f"Loss value: {loss_val:.6f} "
            f"| MSE value: {mse_val:.6f} "
            f"| CovSurr value: {surr_val:.6f} "
            f"| Corr(r,y): {corr:.6f} "
        )

    def __str__(self):
        return f"MoELGBSmoothPenalty(K={self.n_experts}, rho={self.rho})"
    








"""regressivity_constrained_log_model.py

End-to-end model for log-price prediction with regressivity control.

This file implements:
  (1) a high-accuracy LightGBM regressor on log prices, optionally with a
      regressivity-oriented penalty (separable surrogate, covariance, or PRB-slope), and
  (2) a second-stage, low-capacity calibration layer trained via an Augmented Lagrangian
      to reduce regressivity with minimal accuracy loss.

Key ideas
---------
- Everything stays in log-space (targets are log prices).
- Base model can include your 1-step penalty objectives:
    * smooth_surr  (separable proxy; option of signal='div' or 'diff')
    * cov          (direct covariance penalty; diagonal Hess approximation)
    * prb_slope    (industry-like "slope" penalty; stable version of covariance)
- Calibration is intentionally simple: piecewise-linear correction in predicted level.
  Optionally regime-aware: hard regimes + per-regime calibrators.
- Global-cancellation protection: enforce fairness constraints per STRATUM (vector constraints).
  Strata are defined without using true y at inference time:
    * predicted-quantiles (default)
    * user-provided labels
    * regimes (NEW): use automatically-computed regime ids as strata, without you passing them.

Dependencies
------------
- numpy, lightgbm, torch
- scikit-learn (optional): KFold (for cross-fitting), MiniBatchKMeans/KMeans, TruncatedSVD
- scipy (optional): for sparse matrix detection

Notes on log-space fairness signal
----------------------------------
If you model T = log(Y), the real-price ratio is r = exp(T_hat - T).
So the principled log-space "ratio signal" is log(r) = T_hat - T, i.e. ratio_mode='diff'.
We keep ratio_mode='div' only for backwards compatibility with earlier experiments.

Author: (you)
"""


# ============================
# Utilities: types & helpers
# ============================

RatioMode = Literal["div", "diff"]
FairnessMetric = Literal["cov", "prb_slope"]
StrataMode = Literal["none", "pred_quantiles", "labels", "regimes"]
BasePenalty = Literal["mse", "smooth_surr", "cov", "prb_slope"]


def _as_1d_float(x: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def _is_sparse(X) -> bool:
    return sp is not None and sp.issparse(X)


def compute_signal(
    y_hat: np.ndarray,
    y_true: np.ndarray,
    *,
    ratio_mode: RatioMode,
    eps_y: float,
) -> np.ndarray:
    """Compute the signal used in fairness statistics."""
    y_hat = _as_1d_float(y_hat)
    y_true = _as_1d_float(y_true)

    if ratio_mode == "diff":
        return y_hat - y_true  # log-ratio = log(\hat Y / Y)
    if ratio_mode == "div":
        denom = np.maximum(np.abs(y_true), eps_y)
        return y_hat / denom
    raise ValueError(f"Unknown ratio_mode: {ratio_mode}")


def _center(y: np.ndarray) -> Tuple[np.ndarray, float]:
    m = float(np.mean(y)) if y.size else 0.0
    return y - m, m


def cov_stat(signal: np.ndarray, y: np.ndarray) -> float:
    signal = _as_1d_float(signal)
    y = _as_1d_float(y)
    yc, _ = _center(y)
    return float(np.mean(signal * yc))  # since E[yc]=0


def prb_slope_stat(signal: np.ndarray, y: np.ndarray, var_floor: float = 1e-12) -> float:
    signal = _as_1d_float(signal)
    y = _as_1d_float(y)
    yc, _ = _center(y)
    var_y = float(np.mean(yc * yc))
    var_y = max(var_y, var_floor)
    return float(np.mean(signal * yc) / var_y)


def _safe_corr(a: np.ndarray, b: np.ndarray, std_floor: float = 1e-12) -> float:
    """Correlation with a std floor to avoid numpy RuntimeWarnings in logs."""
    a = _as_1d_float(a)
    b = _as_1d_float(b)
    sa = float(np.std(a))
    sb = float(np.std(b))
    if not np.isfinite(sa) or not np.isfinite(sb) or sa < std_floor or sb < std_floor:
        return float("nan")
    # compute in a stable way
    ac = a - float(np.mean(a))
    bc = b - float(np.mean(b))
    return float(np.mean(ac * bc) / (sa * sb))


def make_strata_labels(
    pred_level: np.ndarray,
    *,
    strata_mode: StrataMode,
    n_strata: int,
    user_labels: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Create strata labels for cancellation-safe constraints.

    - none: single stratum 0 for all
    - labels: map user labels to 0..S-1
    - pred_quantiles: bin by pred_level quantiles, return edges as meta
    - regimes: handled in calibrator.fit because it needs regime ids
    """
    pred_level = _as_1d_float(pred_level)

    if strata_mode == "regimes":
        raise ValueError("strata_mode='regimes' is handled in EquityCalibratorALM.fit")

    if strata_mode == "none":
        return np.zeros_like(pred_level, dtype=int), None

    if strata_mode == "labels":
        if user_labels is None:
            raise ValueError("strata_mode='labels' requires user_labels.")
        lab = np.asarray(user_labels).reshape(-1)
        _, inv = np.unique(lab, return_inverse=True)
        return inv.astype(int), None

    if strata_mode == "pred_quantiles":
        if n_strata < 1:
            raise ValueError("n_strata must be >= 1")
        qs = np.linspace(0, 1, n_strata + 1)
        edges = np.quantile(pred_level, qs)
        edges = np.unique(edges)
        if edges.size <= 2:
            return np.zeros_like(pred_level, dtype=int), edges
        labels = np.clip(np.searchsorted(edges, pred_level, side="right") - 1, 0, edges.size - 2)
        return labels.astype(int), edges

    raise ValueError(f"Unknown strata_mode: {strata_mode}")


# Torch fairness stats (for ALM)


def torch_cov(signal: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    yc = y - y.mean()
    return (signal * yc).mean()


def torch_prb_slope(signal: torch.Tensor, y: torch.Tensor, var_floor: float = 1e-12) -> torch.Tensor:
    yc = y - y.mean()
    var_y = (yc * yc).mean().clamp_min(var_floor)
    return (signal * yc).mean() / var_y


# ============================================
# Stage 1: LightGBM base with penalty options
# ============================================

@dataclass
class LGBPenaltyConfig:
    penalty: BasePenalty = "smooth_surr"
    rho: float = 1e-3
    ratio_mode: RatioMode = "diff"
    eps_y: float = 1e-12
    hess_min: float = 1e-6
    print_every: int = 200  # fobj call frequency (0 disables)
    lgbm_params: Optional[Dict] = None


class LGBRegressivityPenalty:
    """LightGBM regressor on log targets with optional regressivity penalties."""

    def __init__(self, config: LGBPenaltyConfig):
        self.config = config
        self.model = lgb.LGBMRegressor(**(config.lgbm_params or {}))
        self._fobj_calls = 0
        self.y_mean_ = None
        self.y_var_ = None

    def fit(self, X, y):
        y = _as_1d_float(y)
        self.y_mean_ = float(np.mean(y))
        yc = y - self.y_mean_
        self.y_var_ = float(np.mean(yc * yc))

        if self.config.penalty == "mse":
            self.model.set_params(objective="regression")
        else:
            self.model.set_params(objective=self._fobj)

        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def _maybe_print(self, msg: str):
        self._fobj_calls += 1
        if self.config.print_every and (self._fobj_calls % self.config.print_every == 0):
            print(msg)

    def _fobj(self, y_true, y_pred):
        cfg = self.config
        y_true = _as_1d_float(y_true)
        y_pred = _as_1d_float(y_pred)
        n = y_true.size

        # Base MSE gradients/hessians
        grad = 2.0 * (y_pred - y_true)
        hess = 2.0 * np.ones_like(y_pred)

        if cfg.penalty == "smooth_surr":
            # Separable weighted MSE surrogate (proxy)
            yc = y_true - self.y_mean_
            signal = compute_signal(y_pred, y_true, ratio_mode=cfg.ratio_mode, eps_y=cfg.eps_y)

            if cfg.ratio_mode == "div":
                denom = np.maximum(np.abs(y_true), cfg.eps_y)
                scale = (yc / denom) ** 2
            else:
                scale = (yc ** 2)

            grad_pen = 2.0 * (y_pred - y_true) * scale
            hess_pen = 2.0 * scale

            grad = grad + cfg.rho * grad_pen
            hess = hess + cfg.rho * hess_pen

            corr = _safe_corr(signal, y_true)
            self._maybe_print(f"[LGB smooth_surr] rho={cfg.rho:g} | corr(signal,y)={corr:.6f}")

        elif cfg.penalty == "cov":
            # penalty = 0.5*rho*n*Cov(signal,y)^2
            yc = y_true - self.y_mean_
            signal = compute_signal(y_pred, y_true, ratio_mode=cfg.ratio_mode, eps_y=cfg.eps_y)
            cov = float(np.mean(signal * yc))

            if cfg.ratio_mode == "div":
                denom = np.maximum(np.abs(y_true), cfg.eps_y)
                dc_dpred = (yc / denom) / float(n)
            else:
                dc_dpred = yc / float(n)

            grad = grad + cfg.rho * float(n) * cov * dc_dpred
            hess = hess + cfg.rho * float(n) * (dc_dpred ** 2)

            corr = _safe_corr(signal, y_true)
            self._maybe_print(f"[LGB cov] rho={cfg.rho:g} | cov={cov:.3e} | corr(signal,y)={corr:.6f}")

        elif cfg.penalty == "prb_slope":
            # penalty = 0.5*rho*n*beta^2 where beta = Cov(signal,y)/Var(y)
            yc = y_true - self.y_mean_
            var_y = max(float(np.mean(yc * yc)), 1e-12)

            signal = compute_signal(y_pred, y_true, ratio_mode=cfg.ratio_mode, eps_y=cfg.eps_y)
            cov = float(np.mean(signal * yc))
            beta = cov / var_y

            if cfg.ratio_mode == "div":
                denom = np.maximum(np.abs(y_true), cfg.eps_y)
                dc_dpred = (yc / denom) / float(n)
            else:
                dc_dpred = yc / float(n)

            dbeta_dpred = dc_dpred / var_y

            grad = grad + cfg.rho * float(n) * beta * dbeta_dpred
            hess = hess + cfg.rho * float(n) * (dbeta_dpred ** 2)

            self._maybe_print(f"[LGB prb_slope] rho={cfg.rho:g} | beta={beta:.3e}")

        else:
            raise ValueError(f"Unknown penalty: {cfg.penalty}")

        # Numerical stability: floor Hessian only
        hess = np.maximum(hess, cfg.hess_min)
        return grad, hess


# =========================================
# Stage 2: Equity calibrators with ALM
# =========================================

@dataclass
class CalibratorALMConfig:
    # structure
    kind: Literal["1d", "regime"] = "1d"
    n_knots: int = 12

    # regimes (used if kind='regime' OR strata_mode='regimes')
    n_regimes: int = 4
    regime_mode: Literal["kmeans", "labels"] = "kmeans"

    # safe defaults for kmeans without passing Z
    auto_regime_from_X: bool = True
    auto_z_dim: int = 64
    auto_z_reduce_if_features_gt: int = 256
    robust_clip: Optional[float] = 10.0

    # KMeans engine
    kmeans_algo: Literal["minibatch", "full"] = "minibatch"
    kmeans_batch_size: int = 2048
    kmeans_n_init: int = 10
    kmeans_max_iter: int = 300
    kmeans_random_state: int = 0

    # fairness
    fairness_metric: FairnessMetric = "prb_slope"
    ratio_mode: RatioMode = "diff"
    eps_y: float = 1e-12

    # cancellation safeguard
    strata_mode: StrataMode = "pred_quantiles"
    n_strata: int = 5

    # ALM / optimization
    alm_iters: int = 15
    inner_steps: int = 300
    lr: float = 2e-2
    rho_init: float = 1.0
    rho_growth: float = 2.0
    stop_tol: float = 1e-4

    # regularization / guardrails
    l2: float = 1e-4
    smooth_l2: float = 1e-3
    max_abs_correction: float = 0.25

    # logging
    verbose: bool = True


class _PiecewiseLinear1D(torch.nn.Module):
    """Piecewise-linear function h(p) defined by values at knots."""

    def __init__(self, knots: np.ndarray, init_values: Optional[np.ndarray] = None):
        super().__init__()
        knots = np.asarray(knots, dtype=np.float32).reshape(-1)
        if knots.size < 2:
            raise ValueError("Need at least 2 knots")
        self.register_buffer("knots", torch.tensor(knots))

        if init_values is None:
            init_values = np.zeros_like(knots, dtype=np.float32)
        init_values = np.asarray(init_values, dtype=np.float32).reshape(-1)
        if init_values.shape != knots.shape:
            raise ValueError("init_values must match knots shape")

        self.values = torch.nn.Parameter(torch.tensor(init_values))

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        k = self.knots
        v = self.values

        p_clamped = torch.clamp(p, float(k[0].item()), float(k[-1].item()))
        idx = torch.searchsorted(k, p_clamped, right=True) - 1
        idx = torch.clamp(idx, 0, k.numel() - 2)

        k0 = k[idx]
        k1 = k[idx + 1]
        v0 = v[idx]
        v1 = v[idx + 1]
        t = (p_clamped - k0) / (k1 - k0 + 1e-12)
        return v0 + t * (v1 - v0)

    def smoothness_penalty(self) -> torch.Tensor:
        if self.values.numel() < 3:
            return torch.tensor(0.0, device=self.values.device)
        d2 = self.values[:-2] - 2.0 * self.values[1:-1] + self.values[2:]
        return (d2 * d2).mean()


class EquityCalibratorALM:
    """Equity correction layer trained with an Augmented Lagrangian.

    - kind='1d'     : a single piecewise-linear h(pred_level)
    - kind='regime' : hard regimes + per-regime h_k(pred_level)

    Strata define WHERE fairness is enforced (vector constraints per stratum).
    Regimes define HOW flexible the correction is (multiple calibrators).

    NEW: strata_mode='regimes' uses the regime ids as strata labels (S=K),
         without you passing regime ids explicitly.
    """

    def __init__(self, config: CalibratorALMConfig):
        self.cfg = config

        # fitted state
        self.strata_mode_ = None
        self.strata_meta_ = None  # edges for pred_quantiles
        self.knots_ = None

        # regimes state
        self.regime_model_ = None
        self.regime_labels_ = None

        # Auto-Z preprocessing state (kmeans + no Z provided)
        self._autoZ_used_ = False
        self._autoZ_median_ = None
        self._autoZ_iqr_ = None
        self._autoZ_clip_ = None
        self._autoZ_rp_ = None
        self._autoZ_svd_ = None

        self._models = None

        # optional diagnostics
        self._lam_ = None
        self._rho_ = None
        self._S_ = None

    # ---------- auto-Z from X ----------

    def _robust_fit_transform_dense(self, X: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        X = np.asarray(X, dtype=np.float32)

        med = np.nanmedian(X, axis=0).astype(np.float32)
        q1 = np.nanquantile(X, 0.25, axis=0).astype(np.float32)
        q3 = np.nanquantile(X, 0.75, axis=0).astype(np.float32)
        iqr = (q3 - q1).astype(np.float32)
        iqr = np.where(iqr < 1e-12, 1.0, iqr).astype(np.float32)

        Z = (X - med) / iqr
        if cfg.robust_clip is not None:
            Z = np.clip(Z, -float(cfg.robust_clip), float(cfg.robust_clip))

        self._autoZ_used_ = True
        self._autoZ_median_ = med
        self._autoZ_iqr_ = iqr
        self._autoZ_clip_ = cfg.robust_clip
        return Z.astype(np.float32)

    def _robust_transform_dense(self, X: np.ndarray) -> np.ndarray:
        if not self._autoZ_used_:
            raise RuntimeError("Auto-Z scaler not fitted.")
        med = self._autoZ_median_
        iqr = self._autoZ_iqr_
        Z = (np.asarray(X, dtype=np.float32) - med) / iqr
        if self._autoZ_clip_ is not None:
            Z = np.clip(Z, -float(self._autoZ_clip_), float(self._autoZ_clip_))
        return Z.astype(np.float32)

    def _auto_Z_fit(self, X) -> np.ndarray:
        cfg = self.cfg

        if _is_sparse(X):
            if TruncatedSVD is None:
                raise RuntimeError("scikit-learn is required for sparse auto-Z (TruncatedSVD).")
            n_features = X.shape[1]
            d = int(min(cfg.auto_z_dim, max(2, n_features - 1)))
            svd = TruncatedSVD(n_components=d, random_state=cfg.kmeans_random_state)
            Z0 = svd.fit_transform(X).astype(np.float32)
            self._autoZ_svd_ = svd
            self._autoZ_rp_ = None
            return self._robust_fit_transform_dense(Z0)

        Xd = np.asarray(X, dtype=np.float32)
        n_features = Xd.shape[1]

        self._autoZ_svd_ = None

        # Optional random projection if very wide (dense)
        if cfg.auto_z_reduce_if_features_gt and n_features > int(cfg.auto_z_reduce_if_features_gt):
            d = int(min(cfg.auto_z_dim, n_features))
            rng = np.random.default_rng(cfg.kmeans_random_state)
            R = rng.normal(0.0, 1.0 / np.sqrt(d), size=(n_features, d)).astype(np.float32)
            Z0 = (Xd @ R).astype(np.float32)
            self._autoZ_rp_ = R
        else:
            Z0 = Xd
            self._autoZ_rp_ = None

        return self._robust_fit_transform_dense(Z0)

    def _auto_Z_transform(self, X) -> np.ndarray:
        if not self._autoZ_used_:
            raise RuntimeError("Auto-Z was not used/fitted.")
        if _is_sparse(X):
            svd = self._autoZ_svd_
            if svd is None:
                raise RuntimeError("Auto-Z sparse SVD not fitted.")
            Z0 = svd.transform(X).astype(np.float32)
            return self._robust_transform_dense(Z0)

        Xd = np.asarray(X, dtype=np.float32)
        R = self._autoZ_rp_
        Z0 = (Xd @ R).astype(np.float32) if R is not None else Xd
        return self._robust_transform_dense(Z0)

    # ---------- regime helpers ----------

    def _fit_regimes(
        self,
        Z: Optional[np.ndarray],
        X: Optional[np.ndarray],
        regime_labels: Optional[np.ndarray],
    ) -> np.ndarray:
        cfg = self.cfg

        if cfg.regime_mode == "labels":
            if regime_labels is None:
                raise ValueError("regime_mode='labels' requires regime_labels.")
            lab = np.asarray(regime_labels).reshape(-1)
            uniq, inv = np.unique(lab, return_inverse=True)
            self.regime_labels_ = uniq
            self.regime_model_ = ("labels", uniq)
            return inv.astype(int)

        # kmeans mode
        if (MiniBatchKMeans is None) and (KMeans is None):
            raise RuntimeError("scikit-learn is required for kmeans regimes.")

        if Z is None:
            if cfg.auto_regime_from_X and (X is not None):
                Z = self._auto_Z_fit(X)
            else:
                raise ValueError(
                    "regime_mode='kmeans' requires regime_features Z, "
                    "or enable auto_regime_from_X and pass X."
                )
        else:
            Z = np.asarray(Z, dtype=np.float32)
            self._autoZ_used_ = False

        if cfg.kmeans_algo == "minibatch":
            if MiniBatchKMeans is None:
                raise RuntimeError("MiniBatchKMeans not available (sklearn missing).")
            km = MiniBatchKMeans(
                n_clusters=cfg.n_regimes,
                random_state=cfg.kmeans_random_state,
                batch_size=cfg.kmeans_batch_size,
                n_init=cfg.kmeans_n_init,
                max_iter=cfg.kmeans_max_iter,
                init="k-means++",
            )
        else:
            if KMeans is None:
                raise RuntimeError("KMeans not available (sklearn missing).")
            km = KMeans(
                n_clusters=cfg.n_regimes,
                random_state=cfg.kmeans_random_state,
                n_init=cfg.kmeans_n_init,
                max_iter=cfg.kmeans_max_iter,
                init="k-means++",
            )

        inv = km.fit_predict(Z)
        self.regime_model_ = km
        return inv.astype(int)

    def _predict_regimes(
        self,
        Z: Optional[np.ndarray],
        X: Optional[np.ndarray],
        regime_labels: Optional[np.ndarray],
    ) -> np.ndarray:
        cfg = self.cfg

        if cfg.regime_mode == "labels":
            if regime_labels is None:
                raise ValueError("regime_mode='labels' requires regime_labels at predict time.")
            if self.regime_labels_ is None:
                raise RuntimeError("Regime labels not fitted.")
            lab = np.asarray(regime_labels).reshape(-1)
            mapping = {u: i for i, u in enumerate(self.regime_labels_)}
            return np.array([mapping.get(v, 0) for v in lab], dtype=int)

        km = self.regime_model_
        if km is None:
            raise RuntimeError("Regime kmeans model not fitted.")

        if Z is None:
            if self.cfg.auto_regime_from_X and (X is not None):
                Z = self._auto_Z_transform(X)
            else:
                raise ValueError(
                    "regime_mode='kmeans' requires regime_features Z at predict time, "
                    "or enable auto_regime_from_X and pass X."
                )
        else:
            Z = np.asarray(Z, dtype=np.float32)
        return km.predict(Z).astype(int)

    # ---------- fitting ----------

    def fit(
        self,
        base_pred: np.ndarray,
        y_true: np.ndarray,
        *,
        strata_labels: Optional[np.ndarray] = None,
        regime_features: Optional[np.ndarray] = None,
        regime_labels: Optional[np.ndarray] = None,
        X_for_regime: Optional[np.ndarray] = None,
    ) -> "EquityCalibratorALM":
        cfg = self.cfg
        device = torch.device("cpu")

        base_pred = _as_1d_float(base_pred)
        y_true = _as_1d_float(y_true)

        need_regimes = (cfg.kind == "regime") or (cfg.strata_mode == "regimes")
        reg_id = None
        K = None

        if need_regimes:
            reg_id = self._fit_regimes(regime_features, X_for_regime, regime_labels)
            # prefer configured K in kmeans mode, even if some clusters are empty
            K = int(cfg.n_regimes) if cfg.regime_mode == "kmeans" else (int(np.max(reg_id)) + 1)

        # Strata labels (cancellation safeguard)
        if cfg.strata_mode == "regimes":
            if reg_id is None:
                raise RuntimeError("strata_mode='regimes' requires regimes (enable auto_regime_from_X or pass regime_labels/features).")
            lab = reg_id.astype(int)
            S = int(K)
            meta = None
        else:
            lab, meta = make_strata_labels(
                base_pred,
                strata_mode=cfg.strata_mode,
                n_strata=cfg.n_strata,
                user_labels=strata_labels,
            )
            S = int(np.max(lab)) + 1

        self.strata_mode_ = cfg.strata_mode
        self.strata_meta_ = meta

        # Knots on predicted level
        n_knots = max(int(cfg.n_knots), 2)
        knots = np.quantile(base_pred, np.linspace(0, 1, n_knots)).astype(np.float32)
        knots = np.unique(knots)
        if knots.size < 2:
            knots = np.array([float(np.min(base_pred)), float(np.max(base_pred))], dtype=np.float32)
        self.knots_ = knots

        # Torch tensors
        p = torch.tensor(base_pred.astype(np.float32), device=device)
        y = torch.tensor(y_true.astype(np.float32), device=device)
        lab_t = torch.tensor(lab.astype(np.int64), device=device)

        reg_id_t = None
        if need_regimes:
            reg_id_t = torch.tensor(reg_id.astype(np.int64), device=device)

        # Models
        if cfg.kind == "regime":
            if K is None:
                raise RuntimeError("Internal error: K not set for regime calibrator.")
            models: List[_PiecewiseLinear1D] = [_PiecewiseLinear1D(knots) for _ in range(K)]
            self._models = models
            params = []
            for m in models:
                params += list(m.parameters())
        else:
            model = _PiecewiseLinear1D(knots)
            self._models = model
            params = list(model.parameters())

        opt = torch.optim.Adam(params, lr=cfg.lr)

        # multipliers per stratum
        lam = torch.zeros(S, dtype=torch.float32, device=device)
        rho = float(cfg.rho_init)

        def fairness_stat(y_hat: torch.Tensor, y_true_t: torch.Tensor) -> torch.Tensor:
            if cfg.ratio_mode == "diff":
                signal = y_hat - y_true_t
            else:
                denom = torch.maximum(torch.abs(y_true_t), torch.tensor(cfg.eps_y, device=device))
                signal = y_hat / denom

            if cfg.fairness_metric == "cov":
                return torch_cov(signal, y_true_t)
            if cfg.fairness_metric == "prb_slope":
                return torch_prb_slope(signal, y_true_t)
            raise ValueError(f"Unknown fairness_metric: {cfg.fairness_metric}")

        # ALM loop
        for it in range(cfg.alm_iters):
            for _ in range(cfg.inner_steps):
                opt.zero_grad()

                # correction
                if cfg.kind == "regime":
                    corr = torch.zeros_like(p)
                    for k in range(K):
                        mask = (reg_id_t == k)
                        if mask.any():
                            corr[mask] = self._models[k](p[mask])
                else:
                    corr = self._models(p)

                corr = torch.clamp(corr, -cfg.max_abs_correction, cfg.max_abs_correction)
                y_hat = p + corr

                mse = torch.mean((y_hat - y) ** 2)

                # constraint vector per stratum
                g = torch.zeros(S, dtype=torch.float32, device=device)
                for s in range(S):
                    mask = (lab_t == s)
                    if mask.any():
                        g[s] = fairness_stat(y_hat[mask], y[mask])

                aug = torch.dot(lam, g) + 0.5 * rho * torch.sum(g * g)

                # regularization
                l2 = torch.tensor(0.0, device=device)
                smooth = torch.tensor(0.0, device=device)
                if cfg.kind == "regime":
                    for m in self._models:
                        l2 = l2 + (m.values * m.values).mean()
                        smooth = smooth + m.smoothness_penalty()
                    l2 = l2 / K
                    smooth = smooth / K
                else:
                    l2 = (self._models.values * self._models.values).mean()
                    smooth = self._models.smoothness_penalty()

                loss = mse + aug + cfg.l2 * l2 + cfg.smooth_l2 * smooth
                loss.backward()
                opt.step()

            # multiplier update
            with torch.no_grad():
                if cfg.kind == "regime":
                    corr = torch.zeros_like(p)
                    for k in range(K):
                        mask = (reg_id_t == k)
                        if mask.any():
                            corr[mask] = self._models[k](p[mask])
                else:
                    corr = self._models(p)

                corr = torch.clamp(corr, -cfg.max_abs_correction, cfg.max_abs_correction)
                y_hat = p + corr

                g = torch.zeros(S, dtype=torch.float32, device=device)
                for s in range(S):
                    mask = (lab_t == s)
                    if mask.any():
                        g[s] = fairness_stat(y_hat[mask], y[mask])

                lam = lam + rho * g
                max_violation = float(torch.max(torch.abs(g)).item())
                mse_val = float(torch.mean((y_hat - y) ** 2).item())

                if cfg.verbose:
                    print(
                        f"[Calib ALM {cfg.kind}] it={it+1}/{cfg.alm_iters} "
                        f"rho={rho:.3g} | mse={mse_val:.6g} | max|g_s|={max_violation:.3e}"
                    )

                if max_violation <= cfg.stop_tol:
                    break
                rho *= float(cfg.rho_growth)

        self._lam_ = lam.detach().cpu().numpy()
        self._rho_ = rho
        self._S_ = S
        return self

    def predict_correction(
        self,
        base_pred: np.ndarray,
        *,
        regime_features: Optional[np.ndarray] = None,
        regime_labels: Optional[np.ndarray] = None,
        X_for_regime: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        cfg = self.cfg
        base_pred = _as_1d_float(base_pred)
        p = torch.tensor(base_pred.astype(np.float32))

        need_regimes = (cfg.kind == "regime")
        reg_id = None
        K = None
        if need_regimes:
            reg_id = self._predict_regimes(regime_features, X_for_regime, regime_labels)
            K = int(cfg.n_regimes) if cfg.regime_mode == "kmeans" else (int(np.max(reg_id)) + 1)
            reg_id_t = torch.tensor(reg_id.astype(np.int64))

        with torch.no_grad():
            if cfg.kind == "regime":
                corr = torch.zeros_like(p)
                for k in range(K):
                    mask = (reg_id_t == k)
                    if mask.any():
                        corr[mask] = self._models[k](p[mask])
            else:
                corr = self._models(p)

            corr = torch.clamp(corr, -cfg.max_abs_correction, cfg.max_abs_correction)

        return corr.cpu().numpy().astype(float)


# =========================================
# Full pipeline: base + calibrator
# =========================================

@dataclass
class PipelineConfig:
    base: LGBPenaltyConfig = LGBPenaltyConfig()
    use_calibration: bool = True
    calib: CalibratorALMConfig = CalibratorALMConfig()

    # To avoid overfitting Stage 2, optionally cross-fit base predictions
    calibration_cv_folds: int = 0  # 0 disables
    random_state: int = 0


class RegressivityConstrainedLogModel:
    """End-to-end model: base + (optional) equity calibration."""

    def __init__(self, pipeline_cfg: Optional[PipelineConfig] = None):
        self.cfg = pipeline_cfg or PipelineConfig()
        self.base_model = LGBRegressivityPenalty(self.cfg.base)
        self.calibrator = EquityCalibratorALM(self.cfg.calib) if self.cfg.use_calibration else None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        strata_labels: Optional[np.ndarray] = None,
        regime_features: Optional[np.ndarray] = None,
        regime_labels: Optional[np.ndarray] = None,
    ) -> "RegressivityConstrainedLogModel":
        y = _as_1d_float(y)

        self.base_model.fit(X, y)
        if not self.cfg.use_calibration:
            return self

        # base preds for calibrator (optionally cross-fitted)
        if self.cfg.calibration_cv_folds and self.cfg.calibration_cv_folds > 1:
            if KFold is None:
                raise RuntimeError("scikit-learn is required for calibration_cv_folds")
            kf = KFold(n_splits=self.cfg.calibration_cv_folds, shuffle=True, random_state=self.cfg.random_state)
            oof = np.zeros_like(y, dtype=float)
            for tr, va in kf.split(X):
                base_fold = LGBRegressivityPenalty(self.cfg.base)
                base_fold.fit(X[tr], y[tr])
                oof[va] = base_fold.predict(X[va])
            base_pred_for_calib = oof
        else:
            base_pred_for_calib = self.base_model.predict(X)

        self.calibrator.fit(
            base_pred_for_calib,
            y,
            strata_labels=strata_labels,
            regime_features=regime_features,
            regime_labels=regime_labels,
            X_for_regime=X,
        )
        return self

    def predict(
        self,
        X: np.ndarray,
        *,
        regime_features: Optional[np.ndarray] = None,
        regime_labels: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        base_pred = self.base_model.predict(X)
        if not self.cfg.use_calibration:
            return base_pred
        corr = self.calibrator.predict_correction(
            base_pred,
            regime_features=regime_features,
            regime_labels=regime_labels,
            X_for_regime=X,
        )
        return base_pred + corr

    def __str__(self):
        return f"RegressivityConstrainedLogModel({self.base_model.config.rho})"


# ============================
# Minimal diagnostics
# ============================

def summarize_vertical_equity(
    y_hat: np.ndarray,
    y_true: np.ndarray,
    *,
    ratio_mode: RatioMode = "diff",
    fairness_metric: FairnessMetric = "prb_slope",
    eps_y: float = 1e-12,
) -> Dict[str, float]:
    y_hat = _as_1d_float(y_hat)
    y_true = _as_1d_float(y_true)
    signal = compute_signal(y_hat, y_true, ratio_mode=ratio_mode, eps_y=eps_y)
    out: Dict[str, float] = {}
    out["mse_log"] = float(np.mean((y_hat - y_true) ** 2))
    out["cov_signal_y"] = cov_stat(signal, y_true)
    out["prb_slope"] = prb_slope_stat(signal, y_true)
    return out

