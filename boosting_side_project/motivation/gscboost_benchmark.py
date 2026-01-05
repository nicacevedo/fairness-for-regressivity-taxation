#!/usr/bin/env python3
"""
gscboost_benchmark.py

Synthetic benchmark suite for stability & robustness of boosting updates
for several losses (Gamma/Poisson with log-link; Gaussian/MSE; Logistic/logloss).

What this script does
- Generates synthetic datasets designed to stress second-order behavior.
- Trains multiple boosters (XGBoost baselines, GSC custom objectives, LightGBM baselines).
- Logs per-round curves (downsampled by --log_every).
- Saves per-run summaries and plots, with optional std-deviation bands across seeds.

Key reliability design choices (HPC-friendly)
- Lazy imports: xgboost/lightgbm are imported inside each worker/job only.
- Thread limiting: OMP/MKL/OpenBLAS threads set to 1 inside each worker.
- Parallelism uses multiprocessing.Pool with --maxtasksperchild to avoid native memory buildup.
- LightGBM: trains once with num_boost_round and evaluates via predict(num_iteration=t)
  (avoids Booster.update() crash path observed in some builds).

Run examples
  # full suite with std bands
  python motivation/gscboost_benchmark.py --out results_gsc --seeds 10 --rounds 200 --jobs 8 --plot_band

  # isolate XGBoost only
  python motivation/gscboost_benchmark.py --out results_xgb --seeds 10 --rounds 200 --jobs 8 --no_lgb

  # only logistic tasks (fast sanity)
  python motivation/gscboost_benchmark.py --out results_logit --datasets logistic_step,logistic_smooth --seeds 10 --rounds 200 --jobs 8 --plot_band

Outputs (in --out folder)
  per_round.csv    : per-round metrics (for each dataset/model/seed/round)
  per_run.csv      : per-run final metrics + flags (failed/diverged)
  summary.csv      : aggregated summary (mean/std across successful non-diverged runs)
  summary_table.tex: compact LaTeX table
  PNG plots         : loss curve (with optional bands), max_abs_margin, clip_rate
"""

from __future__ import annotations

import argparse
import os
import time
import importlib.util
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------
# Environment safety
# -----------------------

_HAS_LGB = importlib.util.find_spec("lightgbm") is not None


def _set_thread_env() -> None:
    """Best-effort thread limiting to avoid oversubscription / OpenMP weirdness."""
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(k, "1")


# -----------------------
# Links / numerics
# -----------------------

def exp_clip(margin: np.ndarray, clip: float = 20.0) -> Tuple[np.ndarray, float, float]:
    """Compute mu = exp(margin) safely by clipping margin.

    Returns:
      mu        = exp(clipped_margin)
      clip_rate = fraction clipped
      max_abs   = max |margin| before clipping
    """
    m = margin.astype(np.float64, copy=False)
    max_abs = float(np.max(np.abs(m))) if m.size else 0.0
    clipped = np.clip(m, -clip, clip)
    clip_rate = float(np.mean((m > clip) | (m < -clip))) if m.size else 0.0
    return np.exp(clipped), clip_rate, max_abs


def sigmoid_clip(margin: np.ndarray, clip: float = 20.0) -> Tuple[np.ndarray, float, float]:
    """Compute p = sigmoid(margin) safely by clipping margin."""
    m = margin.astype(np.float64, copy=False)
    max_abs = float(np.max(np.abs(m))) if m.size else 0.0
    clipped = np.clip(m, -clip, clip)
    clip_rate = float(np.mean((m > clip) | (m < -clip))) if m.size else 0.0
    p = 1.0 / (1.0 + np.exp(-clipped))
    return p, clip_rate, max_abs


def identity_link(margin: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Identity link (Gaussian): yhat = margin."""
    m = margin.astype(np.float64, copy=False)
    max_abs = float(np.max(np.abs(m))) if m.size else 0.0
    return m, 0.0, max_abs


def metric_name(task: str) -> str:
    if task in ("gamma", "poisson"):
        return "deviance"
    if task == "gaussian":
        return "mse"
    if task == "logistic":
        return "logloss"
    return "loss"


# -----------------------
# Losses / metrics (mean per sample)
# -----------------------

def mape_mean(mu_true: np.ndarray, mu_pred: np.ndarray, eps: float = 1e-12) -> float:
    mu_true = np.maximum(mu_true, eps)
    return float(np.mean(np.abs(mu_pred - mu_true) / mu_true))


def mse_loss(y: np.ndarray, yhat: np.ndarray) -> float:
    y = y.astype(np.float64, copy=False)
    yhat = yhat.astype(np.float64, copy=False)
    return float(np.mean((yhat - y) ** 2))


def logloss(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    y = y.astype(np.float64, copy=False)
    p = np.clip(p.astype(np.float64, copy=False), eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def gamma_deviance(y: np.ndarray, mu: np.ndarray, eps: float = 1e-12) -> float:
    """Gamma deviance (per-sample mean).
    D = 2 * ((y - mu)/mu - log(y/mu)), with y,mu>0.
    """
    y = np.maximum(y.astype(np.float64, copy=False), eps)
    mu = np.maximum(mu.astype(np.float64, copy=False), eps)
    return float(np.mean(2.0 * ((y - mu) / mu - np.log(y / mu))))


def poisson_deviance(y: np.ndarray, mu: np.ndarray, eps: float = 1e-12) -> float:
    """Poisson deviance (per-sample mean).
    D = 2 * ( y*log(y/mu) - (y - mu) ), with convention y*log(y/mu)=0 if y=0.
    """
    y = np.maximum(y.astype(np.float64, copy=False), 0.0)
    mu = np.maximum(mu.astype(np.float64, copy=False), eps)
    t = np.zeros_like(y, dtype=np.float64)
    nz = y > 0
    t[nz] = y[nz] * np.log(y[nz] / mu[nz])
    return float(np.mean(2.0 * (t - (y - mu))))


def loss_value(task: str, y: np.ndarray, pred: np.ndarray) -> float:
    """Unified scalar loss for plotting/summary."""
    if task == "gamma":
        return gamma_deviance(y, pred)
    if task == "poisson":
        return poisson_deviance(y, pred)
    if task == "gaussian":
        return mse_loss(y, pred)
    if task == "logistic":
        return logloss(y, pred)
    raise ValueError(task)


def predict_from_margin(task: str, margin: np.ndarray, clip: float) -> Tuple[np.ndarray, float, float]:
    """Return prediction in mean/probability space + (clip_rate, max_abs_margin)."""
    if task in ("gamma", "poisson"):
        return exp_clip(margin, clip=clip)
    if task == "gaussian":
        return identity_link(margin)
    if task == "logistic":
        return sigmoid_clip(margin, clip=clip)
    raise ValueError(task)


# -----------------------
# Dataset container
# -----------------------

@dataclass(frozen=True)
class Dataset:
    name: str
    task: str  # "gamma" | "poisson" | "gaussian" | "logistic"
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    # Ground-truth mean/prob (optional) for MAPE or debugging
    mu_true_test: Optional[np.ndarray]
    # 0=safe, 1=danger (optional)
    region_test: Optional[np.ndarray]


def _alt_split(
    X: np.ndarray,
    y: np.ndarray,
    mu_true: Optional[np.ndarray],
    region: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Deterministic interleaving split: keeps both regimes in train/test."""
    idx = np.arange(X.shape[0])
    tr = idx[idx % 2 == 0]
    te = idx[idx % 2 == 1]
    return (
        X[tr], y[tr],
        X[te], y[te],
        (mu_true[te] if mu_true is not None else None),
        (region[te] if region is not None else None),
    )


# -----------------------
# Synthetic datasets
# -----------------------

def make_gamma_step(
    n: int = 8000,
    danger_start: float = 0.8,
    mu_safe: float = 1.0,
    mu_danger: float = 1000.0,
    shape: float = 0.5,
    outlier_frac: float = 0.01,
    outlier_mult: float = 30.0,
    seed: int = 2025,
) -> Dataset:
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n).astype(np.float32)
    mu_true = np.where(x < danger_start, mu_safe, mu_danger).astype(np.float64)
    y = rng.gamma(shape=shape, scale=mu_true / shape).astype(np.float64)
    y = np.maximum(y, 1e-12)

    danger_mask = x >= danger_start
    idx_danger = np.where(danger_mask)[0]
    m = int(outlier_frac * idx_danger.size)
    if m > 0:
        out_idx = rng.choice(idx_danger, size=m, replace=False)
        y[out_idx] *= outlier_mult

    X = x.reshape(-1, 1)
    region = danger_mask.astype(np.int32)
    Xtr, ytr, Xte, yte, mu_te, reg_te = _alt_split(X, y, mu_true, region)
    return Dataset("gamma_step", "gamma", Xtr, ytr, Xte, yte, mu_te, reg_te)


def make_gamma_smooth(
    n: int = 10000,
    amp: float = 3.0,
    freq: float = 4.0,
    shape: float = 1.0,
    seed: int = 2025,
) -> Dataset:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, size=n).astype(np.float32)
    mu_true = np.exp(amp * np.sin(freq * x)).astype(np.float64)
    y = rng.gamma(shape=shape, scale=mu_true / shape).astype(np.float64)
    y = np.maximum(y, 1e-12)

    X = x.reshape(-1, 1)
    Xtr, ytr, Xte, yte, mu_te, reg_te = _alt_split(X, y, mu_true, region=None)
    return Dataset("gamma_smooth", "gamma", Xtr, ytr, Xte, yte, mu_te, reg_te)


def make_poisson_step(
    n: int = 8000,
    danger_start: float = 0.8,
    mu_safe: float = 1.0,
    mu_danger: float = 500.0,
    outlier_frac: float = 0.005,
    outlier_add: float = 5000.0,
    seed: int = 2025,
) -> Dataset:
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n).astype(np.float32)
    mu_true = np.where(x < danger_start, mu_safe, mu_danger).astype(np.float64)
    y = rng.poisson(lam=mu_true).astype(np.float64)

    danger_mask = x >= danger_start
    idx_danger = np.where(danger_mask)[0]
    m = int(outlier_frac * idx_danger.size)
    if m > 0:
        out_idx = rng.choice(idx_danger, size=m, replace=False)
        y[out_idx] += outlier_add

    X = x.reshape(-1, 1)
    region = danger_mask.astype(np.int32)
    Xtr, ytr, Xte, yte, mu_te, reg_te = _alt_split(X, y, mu_true, region)
    return Dataset("poisson_step", "poisson", Xtr, ytr, Xte, yte, mu_te, reg_te)


def make_poisson_cancel(
    n: int = 10000,
    scale: float = 1500.0,
    bias: float = 1.0,
    noise: float = 0.05,
    seed: int = 2025,
) -> Dataset:
    """Cancellation regime: huge +/- logits that cancel in expectation."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=n).astype(np.float32)
    # large opposing components
    z = scale * (x**3 - x)  # odd polynomial -> large magnitude but cancels across domain
    mu_true = np.exp(noise * z + np.log(bias)).astype(np.float64)
    y = rng.poisson(lam=mu_true).astype(np.float64)

    X = x.reshape(-1, 1)
    Xtr, ytr, Xte, yte, mu_te, reg_te = _alt_split(X, y, mu_true, region=None)
    return Dataset("poisson_cancel", "poisson", Xtr, ytr, Xte, yte, mu_te, reg_te)


def make_gaussian_step(
    n: int = 8000,
    danger_start: float = 0.8,
    mu_safe: float = 0.0,
    mu_danger: float = 1000.0,
    sigma_safe: float = 1.0,
    sigma_danger: float = 50.0,
    outlier_frac: float = 0.01,
    outlier_mult: float = 10.0,
    seed: int = 2025,
) -> Dataset:
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n).astype(np.float32)
    mu_true = np.where(x < danger_start, mu_safe, mu_danger).astype(np.float64)
    sigma = np.where(x < danger_start, sigma_safe, sigma_danger).astype(np.float64)
    y = (mu_true + rng.normal(0.0, sigma, size=n)).astype(np.float64)

    danger_mask = x >= danger_start
    idx_danger = np.where(danger_mask)[0]
    m = int(outlier_frac * idx_danger.size)
    if m > 0:
        out_idx = rng.choice(idx_danger, size=m, replace=False)
        y[out_idx] += outlier_mult * sigma_danger * rng.standard_normal(size=m)

    X = x.reshape(-1, 1)
    region = danger_mask.astype(np.int32)
    Xtr, ytr, Xte, yte, mu_te, reg_te = _alt_split(X, y, mu_true, region)
    return Dataset("gaussian_step", "gaussian", Xtr, ytr, Xte, yte, mu_te, reg_te)


def make_gaussian_smooth(
    n: int = 10000,
    amp: float = 500.0,
    freq: float = 4.0,
    sigma: float = 10.0,
    seed: int = 2025,
) -> Dataset:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, size=n).astype(np.float32)
    mu_true = (amp * np.sin(freq * x)).astype(np.float64)
    y = (mu_true + rng.normal(0.0, sigma, size=n)).astype(np.float64)

    X = x.reshape(-1, 1)
    Xtr, ytr, Xte, yte, mu_te, reg_te = _alt_split(X, y, mu_true, region=None)
    return Dataset("gaussian_smooth", "gaussian", Xtr, ytr, Xte, yte, mu_te, reg_te)


def make_logistic_step(
    n: int = 8000,
    danger_start: float = 0.8,
    p_safe: float = 0.05,
    p_danger: float = 0.95,
    flip_frac_danger: float = 0.02,
    seed: int = 2025,
) -> Dataset:
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n).astype(np.float32)
    p_true = np.where(x < danger_start, p_safe, p_danger).astype(np.float64)
    y = rng.binomial(n=1, p=p_true, size=n).astype(np.float64)

    danger_mask = x >= danger_start
    idx_danger = np.where(danger_mask)[0]
    m = int(flip_frac_danger * idx_danger.size)
    if m > 0:
        flip_idx = rng.choice(idx_danger, size=m, replace=False)
        y[flip_idx] = 1.0 - y[flip_idx]

    X = x.reshape(-1, 1)
    region = danger_mask.astype(np.int32)
    Xtr, ytr, Xte, yte, p_te, reg_te = _alt_split(X, y, p_true, region)
    return Dataset("logistic_step", "logistic", Xtr, ytr, Xte, yte, p_te, reg_te)


def make_logistic_smooth(
    n: int = 10000,
    amp: float = 6.0,
    freq: float = 4.0,
    seed: int = 2025,
) -> Dataset:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, size=n).astype(np.float32)
    logits = amp * np.sin(freq * x)
    p_true, _, _ = sigmoid_clip(logits, clip=20.0)
    y = rng.binomial(n=1, p=p_true, size=n).astype(np.float64)

    X = x.reshape(-1, 1)
    Xtr, ytr, Xte, yte, p_te, reg_te = _alt_split(X, y, p_true, region=None)
    return Dataset("logistic_smooth", "logistic", Xtr, ytr, Xte, yte, p_te, reg_te)


def get_dataset_by_name(name: str, seed: int) -> Dataset:
    if name == "gamma_step":
        return make_gamma_step(seed=seed)
    if name == "gamma_smooth":
        return make_gamma_smooth(seed=seed)
    if name == "poisson_step":
        return make_poisson_step(seed=seed)
    if name == "poisson_cancel":
        return make_poisson_cancel(seed=seed)
    if name == "gaussian_step":
        return make_gaussian_step(seed=seed)
    if name == "gaussian_smooth":
        return make_gaussian_smooth(seed=seed)
    if name == "logistic_step":
        return make_logistic_step(seed=seed)
    if name == "logistic_smooth":
        return make_logistic_smooth(seed=seed)
    raise ValueError(f"Unknown dataset: {name}")


# -----------------------
# GSC objectives (instance-wise effective Hessian)
# -----------------------

def gsc_gamma_obj(margin: np.ndarray, dtrain: Any, M: float, mu0: float, clip_margin: float) -> Tuple[np.ndarray, np.ndarray]:
    """Gamma (log-link):
      grad = 1 - y/mu
      hess = y/mu  (approx for canonical-ish deviance)
    """
    y = dtrain.get_label().astype(np.float64, copy=False)
    mu, _, _ = exp_clip(margin, clip=clip_margin)
    y = np.maximum(y, 1e-12)
    mu = np.maximum(mu, 1e-12)
    g = 1.0 - (y / mu)
    h = (y / mu)
    lam = np.abs(g) / (np.sqrt(h + mu0) + 1e-16)
    h_eff = h * (1.0 + M * lam)
    return g, h_eff


def gsc_poisson_obj(margin: np.ndarray, dtrain: Any, M: float, mu0: float, clip_margin: float) -> Tuple[np.ndarray, np.ndarray]:
    """Poisson (log-link):
      grad = mu - y
      hess = mu
    """
    y = dtrain.get_label().astype(np.float64, copy=False)
    mu, _, _ = exp_clip(margin, clip=clip_margin)
    g = mu - y
    h = mu
    lam = np.abs(g) / (np.sqrt(h + mu0) + 1e-16)
    h_eff = h * (1.0 + M * lam)
    return g, h_eff


def gsc_gaussian_obj(margin: np.ndarray, dtrain: Any, M: float, mu0: float, clip_margin: float) -> Tuple[np.ndarray, np.ndarray]:
    """Squared loss (identity link):
      grad = f - y
      hess = 1
    """
    y = dtrain.get_label().astype(np.float64, copy=False)
    f = margin.astype(np.float64, copy=False)
    g = f - y
    h = np.ones_like(g)
    lam = np.abs(g) / (np.sqrt(h + mu0) + 1e-16)
    h_eff = h * (1.0 + M * lam)
    return g, h_eff


def gsc_logistic_obj(margin: np.ndarray, dtrain: Any, M: float, mu0: float, clip_margin: float) -> Tuple[np.ndarray, np.ndarray]:
    """Binary logistic loss in margin space:
      p = sigmoid(margin)
      grad = p - y
      hess = p*(1-p)
    """
    y = dtrain.get_label().astype(np.float64, copy=False)
    p, _, _ = sigmoid_clip(margin, clip=clip_margin)
    g = p - y
    h = p * (1.0 - p)
    lam = np.abs(g) / (np.sqrt(h + mu0) + 1e-16)
    h_eff = h * (1.0 + M * lam)
    return g, h_eff


# -----------------------
# Training helpers
# -----------------------

def make_base_margin(task: str, y_train: np.ndarray) -> float:
    """Return initial base margin in *margin space*."""
    eps = 1e-12
    m = float(np.mean(y_train))
    if task in ("gamma", "poisson"):
        m = max(m, eps)
        return float(np.log(m))
    if task == "gaussian":
        return float(m)
    if task == "logistic":
        # mean label -> logit
        p = min(max(m, eps), 1.0 - eps)
        return float(np.log(p / (1.0 - p)))
    raise ValueError(task)


@dataclass(frozen=True)
class JobSpec:
    dataset_name: str
    dataset_seed: int
    model_name: str
    task: str
    params: Dict[str, Any]
    rounds: int
    train_seed: int
    backend: str  # "xgb" or "lgb"
    # objective type (xgb only)
    fobj_kind: str  # "none" | "gsc_gamma" | "gsc_poisson" | "gsc_gaussian" | "gsc_logistic"
    M: float = 1.0
    mu0: float = 0.0
    # evaluation / logging
    eval_clip_margin: float = 20.0
    diverge_margin: float = 60.0
    diverge_clip_rate: float = 0.10
    log_every: int = 1


def _run_job_xgb(job: JobSpec, ds: Dataset) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    import xgboost as xgb

    dtr = xgb.DMatrix(ds.X_train, label=ds.y_train)
    dte = xgb.DMatrix(ds.X_test, label=ds.y_test)

    init_margin = make_base_margin(ds.task, ds.y_train)
    dtr.set_base_margin(np.full(ds.X_train.shape[0], init_margin, dtype=np.float64))
    dte.set_base_margin(np.full(ds.X_test.shape[0], init_margin, dtype=np.float64))

    params = dict(job.params)
    params["seed"] = int(job.train_seed)
    params["verbosity"] = 0
    params["base_score"] = 0.0  # using base_margin
    params["nthread"] = 1       # crucial for multi-process parallelism

    booster = xgb.Booster(params, [dtr])

    rows: List[Dict[str, Any]] = []
    diverged = False
    diverged_round = 0

    t0 = time.time()
    for t in range(job.rounds):
        if job.fobj_kind == "none":
            booster.update(dtr, t)
        else:
            if job.fobj_kind == "gsc_gamma":
                booster.update(dtr, t, fobj=lambda m, d: gsc_gamma_obj(m, d, job.M, job.mu0, job.eval_clip_margin))
            elif job.fobj_kind == "gsc_poisson":
                booster.update(dtr, t, fobj=lambda m, d: gsc_poisson_obj(m, d, job.M, job.mu0, job.eval_clip_margin))
            elif job.fobj_kind == "gsc_gaussian":
                booster.update(dtr, t, fobj=lambda m, d: gsc_gaussian_obj(m, d, job.M, job.mu0, job.eval_clip_margin))
            elif job.fobj_kind == "gsc_logistic":
                booster.update(dtr, t, fobj=lambda m, d: gsc_logistic_obj(m, d, job.M, job.mu0, job.eval_clip_margin))
            else:
                raise ValueError(job.fobj_kind)

        do_log = ((t + 1) % job.log_every == 0) or (t == job.rounds - 1)
        if not do_log:
            continue

        margin = booster.predict(dte, output_margin=True)
        pred, clip_rate, max_abs = predict_from_margin(ds.task, margin, clip=job.eval_clip_margin)
        loss_all = loss_value(ds.task, ds.y_test, pred)

        loss_safe = loss_danger = np.nan
        mape_all = mape_safe = mape_danger = np.nan
        if ds.mu_true_test is not None and ds.task in ("gamma", "poisson"):
            mape_all = mape_mean(ds.mu_true_test, pred)
        if ds.region_test is not None:
            safe = ds.region_test == 0
            danger = ds.region_test == 1
            if np.any(safe):
                loss_safe = loss_value(ds.task, ds.y_test[safe], pred[safe])
                if ds.mu_true_test is not None and ds.task in ("gamma", "poisson"):
                    mape_safe = mape_mean(ds.mu_true_test[safe], pred[safe])
            if np.any(danger):
                loss_danger = loss_value(ds.task, ds.y_test[danger], pred[danger])
                if ds.mu_true_test is not None and ds.task in ("gamma", "poisson"):
                    mape_danger = mape_mean(ds.mu_true_test[danger], pred[danger])

        rows.append({
            "round": int(t + 1),
            "dataset": ds.name,
            "task": ds.task,
            "model": job.model_name,
            "seed": int(job.train_seed),
            "loss_overall": float(loss_all),
            "loss_safe": float(loss_safe) if np.isfinite(loss_safe) else np.nan,
            "loss_danger": float(loss_danger) if np.isfinite(loss_danger) else np.nan,
            # legacy names (so older downstream scripts don't break)
            "deviance_overall": float(loss_all),
            "deviance_safe": float(loss_safe) if np.isfinite(loss_safe) else np.nan,
            "deviance_danger": float(loss_danger) if np.isfinite(loss_danger) else np.nan,
            "mape_overall": float(mape_all) if np.isfinite(mape_all) else np.nan,
            "mape_safe": float(mape_safe) if np.isfinite(mape_safe) else np.nan,
            "mape_danger": float(mape_danger) if np.isfinite(mape_danger) else np.nan,
            "clip_rate": float(clip_rate),
            "max_abs_margin": float(max_abs),
        })

        if (not np.isfinite(loss_all)) or (max_abs > job.diverge_margin) or (clip_rate > job.diverge_clip_rate):
            diverged = True
            diverged_round = int(t + 1)
            break

    wall = time.time() - t0

    last = rows[-1] if rows else {}
    per_run = {
        "dataset": ds.name,
        "task": ds.task,
        "model": job.model_name,
        "seed": int(job.train_seed),
        "failed": False,
        "diverged": bool(diverged),
        "diverged_round": int(diverged_round),
        "wall_time_s": float(wall),
        "final_round": int(last.get("round", 0)),
        "final_loss_overall": float(last.get("loss_overall", np.nan)),
        "final_loss_safe": float(last.get("loss_safe", np.nan)),
        "final_loss_danger": float(last.get("loss_danger", np.nan)),
        "final_mape_overall": float(last.get("mape_overall", np.nan)),
        "final_clip_rate": float(last.get("clip_rate", np.nan)),
        "final_max_abs_margin": float(last.get("max_abs_margin", np.nan)),
        # legacy field names
        "final_deviance_overall": float(last.get("deviance_overall", np.nan)),
        "final_deviance_safe": float(last.get("deviance_safe", np.nan)),
        "final_deviance_danger": float(last.get("deviance_danger", np.nan)),
    }
    return rows, per_run


def _run_job_lgb(job: JobSpec, ds: Dataset) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """LightGBM baseline (safe from Booster.update crash path)."""
    if not _HAS_LGB:
        raise RuntimeError("LightGBM is not available, but a LightGBM job was scheduled.")

    import lightgbm as lgb

    train_set = lgb.Dataset(ds.X_train, label=ds.y_train, free_raw_data=False)

    params = dict(job.params)
    # seeds for determinism
    params.setdefault("seed", int(job.train_seed))
    params.setdefault("feature_fraction_seed", int(job.train_seed))
    params.setdefault("bagging_seed", int(job.train_seed))
    params.setdefault("data_random_seed", int(job.train_seed))
    params.setdefault("deterministic", True)
    params.setdefault("verbosity", -1)
    params["num_threads"] = 1

    rows: List[Dict[str, Any]] = []
    diverged = False
    diverged_round = 0

    t0 = time.time()
    booster = lgb.train(
        params=params,
        train_set=train_set,
        num_boost_round=job.rounds,
        keep_training_booster=False,
        callbacks=[lgb.log_evaluation(period=0)],
    )
    wall_train = time.time() - t0

    t1 = time.time()
    for t in range(1, job.rounds + 1):
        do_log = (t % job.log_every == 0) or (t == job.rounds)
        if not do_log:
            continue

        margin = booster.predict(ds.X_test, raw_score=True, num_iteration=t)
        pred, clip_rate, max_abs = predict_from_margin(ds.task, margin, clip=job.eval_clip_margin)
        loss_all = loss_value(ds.task, ds.y_test, pred)

        loss_safe = loss_danger = np.nan
        mape_all = mape_safe = mape_danger = np.nan
        if ds.mu_true_test is not None and ds.task in ("gamma", "poisson"):
            mape_all = mape_mean(ds.mu_true_test, pred)
        if ds.region_test is not None:
            safe = ds.region_test == 0
            danger = ds.region_test == 1
            if np.any(safe):
                loss_safe = loss_value(ds.task, ds.y_test[safe], pred[safe])
                if ds.mu_true_test is not None and ds.task in ("gamma", "poisson"):
                    mape_safe = mape_mean(ds.mu_true_test[safe], pred[safe])
            if np.any(danger):
                loss_danger = loss_value(ds.task, ds.y_test[danger], pred[danger])
                if ds.mu_true_test is not None and ds.task in ("gamma", "poisson"):
                    mape_danger = mape_mean(ds.mu_true_test[danger], pred[danger])

        rows.append({
            "round": int(t),
            "dataset": ds.name,
            "task": ds.task,
            "model": job.model_name,
            "seed": int(job.train_seed),
            "loss_overall": float(loss_all),
            "loss_safe": float(loss_safe) if np.isfinite(loss_safe) else np.nan,
            "loss_danger": float(loss_danger) if np.isfinite(loss_danger) else np.nan,
            "deviance_overall": float(loss_all),
            "deviance_safe": float(loss_safe) if np.isfinite(loss_safe) else np.nan,
            "deviance_danger": float(loss_danger) if np.isfinite(loss_danger) else np.nan,
            "mape_overall": float(mape_all) if np.isfinite(mape_all) else np.nan,
            "mape_safe": float(mape_safe) if np.isfinite(mape_safe) else np.nan,
            "mape_danger": float(mape_danger) if np.isfinite(mape_danger) else np.nan,
            "clip_rate": float(clip_rate),
            "max_abs_margin": float(max_abs),
        })

        if (not np.isfinite(loss_all)) or (max_abs > job.diverge_margin) or (clip_rate > job.diverge_clip_rate):
            diverged = True
            diverged_round = int(t)
            break

    wall = wall_train + (time.time() - t1)

    last = rows[-1] if rows else {}
    per_run = {
        "dataset": ds.name,
        "task": ds.task,
        "model": job.model_name,
        "seed": int(job.train_seed),
        "failed": False,
        "diverged": bool(diverged),
        "diverged_round": int(diverged_round),
        "wall_time_s": float(wall),
        "final_round": int(last.get("round", 0)),
        "final_loss_overall": float(last.get("loss_overall", np.nan)),
        "final_loss_safe": float(last.get("loss_safe", np.nan)),
        "final_loss_danger": float(last.get("loss_danger", np.nan)),
        "final_mape_overall": float(last.get("mape_overall", np.nan)),
        "final_clip_rate": float(last.get("clip_rate", np.nan)),
        "final_max_abs_margin": float(last.get("max_abs_margin", np.nan)),
        "final_deviance_overall": float(last.get("deviance_overall", np.nan)),
        "final_deviance_safe": float(last.get("deviance_safe", np.nan)),
        "final_deviance_danger": float(last.get("deviance_danger", np.nan)),
    }
    return rows, per_run


def run_job(job: JobSpec) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Worker function executed in a subprocess."""
    _set_thread_env()
    ds = get_dataset_by_name(job.dataset_name, seed=job.dataset_seed)
    if job.backend == "xgb":
        return _run_job_xgb(job, ds)
    if job.backend == "lgb":
        return _run_job_lgb(job, ds)
    raise ValueError(f"Unknown backend: {job.backend}")


def run_job_safe(job: JobSpec) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Always-return wrapper for multiprocessing.Pool."""
    try:
        return run_job(job)
    except Exception as e:
        rrun = {
            "dataset": job.dataset_name,
            "task": job.task,
            "model": job.model_name,
            "seed": int(job.train_seed),
            "failed": True,
            "diverged": np.nan,
            "diverged_round": 0,
            "wall_time_s": np.nan,
            "final_round": 0,
            "final_loss_overall": np.nan,
            "final_loss_safe": np.nan,
            "final_loss_danger": np.nan,
            "final_mape_overall": np.nan,
            "final_clip_rate": np.nan,
            "final_max_abs_margin": np.nan,
            "final_deviance_overall": np.nan,
            "final_deviance_safe": np.nan,
            "final_deviance_danger": np.nan,
            "error": repr(e),
        }
        return [], rrun


# -----------------------
# Models (job creation)
# -----------------------

def _xgb_objective_for_task(task: str) -> str:
    if task == "gamma":
        return "reg:gamma"
    if task == "poisson":
        return "count:poisson"
    if task == "gaussian":
        return "reg:squarederror"
    if task == "logistic":
        return "binary:logistic"
    raise ValueError(task)


def _lgb_objective_for_task(task: str) -> str:
    if task == "gamma":
        return "gamma"
    if task == "poisson":
        return "poisson"
    if task == "gaussian":
        return "regression"
    if task == "logistic":
        return "binary"
    raise ValueError(task)


def build_jobs(
    datasets: List[str],
    seeds: int,
    rounds: int,
    log_every: int,
    include_lgb: bool,
) -> List[JobSpec]:
    jobs: List[JobSpec] = []

    # Common tree params (keep small to highlight loss geometry, not tree capacity)
    base_xgb_tree = dict(
        booster="gbtree",
        tree_method="hist",
        max_depth=3,
        min_child_weight=1.0,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_lambda=1.0,
        reg_alpha=0.0,
    )

    base_lgb_tree = dict(
        boosting_type="gbdt",
        max_depth=3,
        num_leaves=8,
        min_data_in_leaf=5,
        min_sum_hessian_in_leaf=1e-3,
        feature_fraction=1.0,
        bagging_fraction=1.0,
        bagging_freq=0,
        lambda_l2=1.0,
        lambda_l1=0.0,
        max_bin=255,
        min_data_in_bin=1,
        force_col_wise=True,
        verbose=-1,
        verbosity=-1,
        boost_from_average=True,
    )

    # Learning rates (fast/safe)
    eta_fast = 0.30
    eta_safe = 0.05

    # GSC damping configs
    gsc_variants = [
        ("M1", 1.0, 0.0),
        ("M1-mu0", 1.0, 1e-6),
        ("M2", 2.0, 0.0),
        ("M2-mu0", 2.0, 1e-6),
    ]

    for ds_name in datasets:
        for s in range(seeds):
            train_seed = 1000 + 17 * s
            ds = get_dataset_by_name(ds_name, seed=train_seed)  # just to know task
            task = ds.task

            # ---- XGB baselines ----
            obj = _xgb_objective_for_task(task)

            jobs.append(JobSpec(
                dataset_name=ds_name,
                dataset_seed=train_seed,
                model_name=f"XGB-{task}-fast",
                task=task,
                params={**base_xgb_tree, "objective": obj, "eta": eta_fast},
                rounds=rounds,
                train_seed=train_seed,
                backend="xgb",
                fobj_kind="none",
                log_every=log_every,
            ))

            jobs.append(JobSpec(
                dataset_name=ds_name,
                dataset_seed=train_seed,
                model_name=f"XGB-{task}-safe",
                task=task,
                params={**base_xgb_tree, "objective": obj, "eta": eta_safe},
                rounds=rounds,
                train_seed=train_seed,
                backend="xgb",
                fobj_kind="none",
                log_every=log_every,
            ))

            # max_delta_step heuristic can matter for poisson/logistic especially
            jobs.append(JobSpec(
                dataset_name=ds_name,
                dataset_seed=train_seed,
                model_name=f"XGB-{task}-max_delta",
                task=task,
                params={**base_xgb_tree, "objective": obj, "eta": eta_fast, "max_delta_step": 1.0},
                rounds=rounds,
                train_seed=train_seed,
                backend="xgb",
                fobj_kind="none",
                log_every=log_every,
            ))

            # ---- GSC custom objectives (XGB backend) ----
            if task == "gamma":
                kind = "gsc_gamma"
            elif task == "poisson":
                kind = "gsc_poisson"
            elif task == "gaussian":
                kind = "gsc_gaussian"
            elif task == "logistic":
                kind = "gsc_logistic"
            else:
                raise ValueError(task)

            for tag, M, mu0 in gsc_variants:
                jobs.append(JobSpec(
                    dataset_name=ds_name,
                    dataset_seed=train_seed,
                    model_name=f"GSC-{task}-{tag}",
                    task=task,
                    params={**base_xgb_tree, "objective": "reg:squarederror", "eta": eta_fast},
                    rounds=rounds,
                    train_seed=train_seed,
                    backend="xgb",
                    fobj_kind=kind,
                    M=M,
                    mu0=mu0,
                    log_every=log_every,
                ))

            # ---- LightGBM baselines ----
            if include_lgb:
                lobj = _lgb_objective_for_task(task)
                jobs.append(JobSpec(
                    dataset_name=ds_name,
                    dataset_seed=train_seed,
                    model_name=f"LGB-{task}-fast",
                    task=task,
                    params={**base_lgb_tree, "objective": lobj, "learning_rate": eta_fast},
                    rounds=rounds,
                    train_seed=train_seed,
                    backend="lgb",
                    fobj_kind="none",
                    log_every=log_every,
                ))
                jobs.append(JobSpec(
                    dataset_name=ds_name,
                    dataset_seed=train_seed,
                    model_name=f"LGB-{task}-safe",
                    task=task,
                    params={**base_lgb_tree, "objective": lobj, "learning_rate": eta_safe},
                    rounds=rounds,
                    train_seed=train_seed,
                    backend="lgb",
                    fobj_kind="none",
                    log_every=log_every,
                ))

    return jobs


# -----------------------
# Summary + plots
# -----------------------

def summarize(per_run: pd.DataFrame) -> pd.DataFrame:
    """Aggregate summary.

    - fail_rate: fraction of runs that raised a Python exception (failed=True)
    - diverge_rate: fraction of completed runs that diverged numerically (diverged=True)
    - metrics averaged over completed & non-diverged runs only
    """
    if "failed" not in per_run.columns:
        per_run = per_run.copy()
        per_run["failed"] = False

    completed = per_run[~per_run["failed"].astype(bool)].copy()

    def _div_rate(s: pd.Series) -> float:
        # Convert to numeric: True→1, False→0, invalid→NaN
        v = pd.to_numeric(s, errors="coerce").to_numpy()
        v = v[np.isfinite(v)]
        if v.size == 0:
            return float("nan")
        return float(np.mean(v))

    ok = completed[~completed["diverged"].astype(bool)].copy() if "diverged" in completed.columns else completed

    base = per_run.groupby(["dataset", "task", "model"], as_index=False).agg(
        n_runs=("seed", "count"),
        fail_rate=("failed", "mean"),
        diverge_rate=("diverged", _div_rate),
    )

    met = ok.groupby(["dataset", "task", "model"], as_index=False).agg(
        loss_mean=("final_loss_overall", "mean"),
        loss_std=("final_loss_overall", "std"),
        safe_loss_mean=("final_loss_safe", "mean"),
        danger_loss_mean=("final_loss_danger", "mean"),
        mape_mean=("final_mape_overall", "mean"),
        time_mean=("wall_time_s", "mean"),
        max_margin_mean=("final_max_abs_margin", "mean"),
        clip_rate_mean=("final_clip_rate", "mean"),
    )

    agg = base.merge(met, on=["dataset", "task", "model"], how="left")
    return agg


def _style_for_model(model: str) -> Dict[str, Any]:
    """Matplotlib style kwargs for a model name (marker/linestyle only)."""
    if model.startswith("XGB-"):
        if "-safe" in model:
            return {"linestyle": "--", "marker": "x", "markersize": 3.5, "linewidth": 1.8}
        if "-fast" in model:
            return {"linestyle": "-", "marker": "o", "markersize": 3.5, "linewidth": 1.8}
        if "-max_delta" in model:
            return {"linestyle": "-.", "marker": "s", "markersize": 3.5, "linewidth": 1.8}
        return {"linestyle": "-", "marker": "o", "markersize": 3.0, "linewidth": 1.6}
    if model.startswith("LGB-"):
        return {"linestyle": ":", "marker": "d", "markersize": 3.5, "linewidth": 2.0}
    if model.startswith("GSC-"):
        if "M2" in model:
            return {"linestyle": "-", "marker": "^", "markersize": 3.5, "linewidth": 2.2}
        return {"linestyle": "-", "marker": "v", "markersize": 3.5, "linewidth": 2.2}
    return {"linestyle": "-", "marker": None, "linewidth": 1.8}


def plot_curves(
    per_round: pd.DataFrame,
    out_dir: str,
    plot_band: bool,
    band_k: float,
    band_alpha: float,
) -> None:
    """Plots: loss curve (with optional std bands), max_abs_margin, clip_rate."""
    os.makedirs(out_dir, exist_ok=True)

    # Aggregate per round across seeds
    grp = per_round.groupby(["dataset", "task", "model", "round"], as_index=False)
    agg = grp.agg(
        loss_mean=("loss_overall", "mean"),
        loss_std=("loss_overall", "std"),
        margin_mean=("max_abs_margin", "mean"),
        margin_std=("max_abs_margin", "std"),
        clip_mean=("clip_rate", "mean"),
        clip_std=("clip_rate", "std"),
        n=("seed", "count"),
    )

    for (ds_name, task), sub in agg.groupby(["dataset", "task"]):
        metric = metric_name(task)

        def _plot_one(y_mean: str, y_std: str, title: str, fname: str) -> None:
            plt.figure()
            for model, mdf in sub.groupby("model"):
                mdf = mdf.sort_values("round")
                x = mdf["round"].to_numpy()
                y = mdf[y_mean].to_numpy()
                st = mdf[y_std].to_numpy()
                style = _style_for_model(model)
                plt.plot(x, y, label=model, **{k: v for k, v in style.items() if v is not None})
                if plot_band and np.any(np.isfinite(st)):
                    lo = y - band_k * np.nan_to_num(st, nan=0.0)
                    hi = y + band_k * np.nan_to_num(st, nan=0.0)
                    plt.fill_between(x, lo, hi, alpha=band_alpha)
            plt.xlabel("round")
            plt.ylabel(title)
            plt.title(f"{ds_name} ({task})")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, fname), dpi=180)
            plt.close()

        _plot_one("loss_mean", "loss_std", metric, f"{ds_name}_{metric}.png")
        _plot_one("margin_mean", "margin_std", "max_abs_margin", f"{ds_name}_max_abs_margin.png")
        _plot_one("clip_mean", "clip_std", "clip_rate", f"{ds_name}_clip_rate.png")


def write_latex_table(summary: pd.DataFrame, out_path: str) -> None:
    """Small LaTeX table for quick paper-ready inclusion."""
    # Keep a compact set of columns
    cols = [
        "dataset", "task", "model",
        "n_runs", "fail_rate", "diverge_rate",
        "loss_mean", "loss_std", "time_mean", "max_margin_mean", "clip_rate_mean",
    ]
    s = summary.copy()
    for c in cols:
        if c not in s.columns:
            s[c] = np.nan
    s = s[cols].sort_values(["dataset", "task", "model"])

    # Simple formatting
    def f(x: Any) -> str:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return ""
        if isinstance(x, (int, np.integer)):
            return str(int(x))
        if isinstance(x, (float, np.floating)):
            return f"{float(x):.4g}"
        return str(x)

    lines = []
    lines.append("\\begin{tabular}{lllrrrrrrr}")
    lines.append("\\toprule")
    lines.append("Dataset & Task & Model & n & fail & div & loss & std & time & |m| & clip\\\\")
    lines.append("\\midrule")
    for _, r in s.iterrows():
        lines.append(
            f"{f(r['dataset'])} & {f(r['task'])} & {f(r['model'])} & "
            f"{f(r['n_runs'])} & {f(r['fail_rate'])} & {f(r['diverge_rate'])} & "
            f"{f(r['loss_mean'])} & {f(r['loss_std'])} & {f(r['time_mean'])} & "
            f"{f(r['max_margin_mean'])} & {f(r['clip_rate_mean'])} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    with open(out_path, "w", encoding="utf-8") as fobj:
        fobj.write("\n".join(lines))


# -----------------------
# Main
# -----------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True, help="Output directory")
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=10, help="Number of runs per dataset/model")
    ap.add_argument("--jobs", type=int, default=1, help="Number of worker processes")
    ap.add_argument("--no_lgb", action="store_true", help="Disable LightGBM baselines")
    ap.add_argument("--datasets", type=str, default="gamma_step,gamma_smooth,poisson_step,poisson_cancel,gaussian_step,gaussian_smooth,logistic_step,logistic_smooth",
                    help="Comma-separated dataset names")
    ap.add_argument("--log_every", type=int, default=1, help="Log every k rounds (downsamples per_round.csv)")
    ap.add_argument("--print_every", type=int, default=10, help="Print progress every k jobs")
    ap.add_argument("--mp_start", type=str, default="spawn", choices=["spawn", "fork", "forkserver"], help="multiprocessing start method")
    ap.add_argument("--maxtasksperchild", type=int, default=1, help="respawn worker after N jobs (mitigates native memory leaks); 0 disables")
    ap.add_argument("--chunksize", type=int, default=1, help="imap_unordered chunksize for multiprocessing pool")
    ap.add_argument("--plot_band", action="store_true", help="Plot mean ± k·std band across seeds")
    ap.add_argument("--band_k", type=float, default=1.0, help="Band multiplier (k)")
    ap.add_argument("--band_alpha", type=float, default=0.15, help="Band transparency")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    ds_list = [d.strip() for d in args.datasets.split(",") if d.strip()]
    include_lgb = (not args.no_lgb)

    jobs = build_jobs(
        datasets=ds_list,
        seeds=int(args.seeds),
        rounds=int(args.rounds),
        log_every=max(1, int(args.log_every)),
        include_lgb=include_lgb,
    )
    print(f"Running {len(jobs)} jobs with jobs={args.jobs} workers...")

    all_round_rows: List[Dict[str, Any]] = []
    all_run_rows: List[Dict[str, Any]] = []

    if int(args.jobs) <= 1:
        done = 0
        for job in jobs:
            rrows, rrun = run_job_safe(job)
            all_round_rows.extend(rrows)
            all_run_rows.append(rrun)
            done += 1
            if (args.print_every and done % args.print_every == 0) or (done == len(jobs)):
                print(f"  completed {done}/{len(jobs)}")
    else:
        import multiprocessing as mp
        ctx = mp.get_context(args.mp_start)
        mtpc = None if int(args.maxtasksperchild) <= 0 else int(args.maxtasksperchild)
        chunksize = max(1, int(args.chunksize))

        with ctx.Pool(processes=int(args.jobs), maxtasksperchild=mtpc) as pool:
            done = 0
            for rrows, rrun in pool.imap_unordered(run_job_safe, jobs, chunksize=chunksize):
                all_round_rows.extend(rrows)
                all_run_rows.append(rrun)
                done += 1
                if (args.print_every and done % args.print_every == 0) or (done == len(jobs)):
                    print(f"  completed {done}/{len(jobs)}")

    per_round = pd.DataFrame(all_round_rows)
    per_run = pd.DataFrame(all_run_rows)

    per_round_path = os.path.join(out_dir, "per_round.csv")
    per_run_path = os.path.join(out_dir, "per_run.csv")
    per_round.to_csv(per_round_path, index=False)
    per_run.to_csv(per_run_path, index=False)

    summary = summarize(per_run)
    summary_path = os.path.join(out_dir, "summary.csv")
    summary.to_csv(summary_path, index=False)

    tex_path = os.path.join(out_dir, "summary_table.tex")
    write_latex_table(summary, tex_path)

    plot_curves(
        per_round=per_round,
        out_dir=out_dir,
        plot_band=bool(args.plot_band) and int(args.seeds) > 1,
        band_k=float(args.band_k),
        band_alpha=float(args.band_alpha),
    )

    print("Saved outputs to:", out_dir)
    print(" - per_round.csv / per_run.csv / summary.csv / summary_table.tex")
    print(" - PNG curves: *deviance|mse|logloss* / *max_abs_margin* / *clip_rate*")


if __name__ == "__main__":
    main()
