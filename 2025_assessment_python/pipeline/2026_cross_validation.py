#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CCAO temporal + bootstrap cross-validation (high-throughput, CPU-parallel)

What this script does (high level):
  1) Loads Cook County Assessor (CCAO) training data (parquet) and filters out
     multicard + outliers (same logic as your original file).
  2) Sorts by meta_sale_date and applies the SAME temporal split proportions you used:
        - Pre-test ("train+val"): first 82.2871% of rows (oldest dates)
        - Test:                 last 17.7129% of rows (most recent dates)
  3) Performs temporal cross-validation on the pre-test portion using forward-chaining
     with a fixed validation horizon equal to 17.7129% of the pre-test portion.
     For each fold, validation is always strictly in the future of training.
  4) Uses bootstrap / subsampling (multiple seeds) within each fold’s training window
     to (a) make CV more stable and (b) keep training fast on huge data.
  5) Trains and tunes exactly these models:
        - LightGBM (LGBMRegressor)  [main focus + extensive tuning]
        - ElasticNet
        - RandomForestRegressor
        - GradientBoostingRegressor
  6) Executes training/evaluation in parallel, aiming to utilize 100+ CPU cores efficiently,
     by doing *outer* parallelism (across trials/folds/bootstraps) while forcing each model
     to use 1 thread (avoids oversubscription).
  7) Selects the best model config by CV mean MAE (primary) / RMSE (secondary) in the
     ORIGINAL sale price scale (exp of log-target).
  8) Re-trains the best config on the full pre-test data and evaluates on the held-out test set.
     Saves the fitted preprocessing pipeline + fitted model.

Notes:
  - Preprocessing pipeline is used exactly as you had it (build_model_pipeline).
    We do NOT modify its internals. We only control how/when it is fit in CV.
  - Target: log(meta_sale_price) is used for training, and metrics are reported on both
    log scale AND original price scale (via exp()).
"""

from __future__ import annotations

import os
import json
import math
import time
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

import yaml

from joblib import Parallel, delayed

from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import lightgbm as lgb


# -----------------------------------------------------------------------------
# CPU / thread control: outer parallelism + single-threaded learners
# -----------------------------------------------------------------------------
# Avoid nested parallelism (e.g., joblib + OpenMP). You can override via env vars.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# '/orcd/home/002/nacevedo/RA/fairness-for-regressivity-taxation/2025_assessment_python/2026_cross_validation.py': [Errno 2] No such file or directory

# -----------------------------------------------------------------------------
# Your project-specific preprocessing pipeline
# -----------------------------------------------------------------------------
import sys

this_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(this_dir)
# Make imports robust regardless of where you launch python from.
for p in (project_root, this_dir):
    if p not in sys.path:
        sys.path.append(p)

from recipes.recipes_pipelined import build_model_pipeline  # noqa: E402


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class CVConfig:
    # Data
    parquet_path: str = "../data_county/2025/training_data.parquet"
    params_yaml_path: str = "params.yaml"
    target_name: str = "meta_sale_price"
    date_col: str = "meta_sale_date"

    # Optional: run CV on a subset for speed (applied AFTER temporal sort)
    # If subset_max_rows is set, we keep the most recent subset_max_rows rows.
    # Alternatively, set subset_recent_years to keep only rows with sale_date >= (max_year - subset_recent_years + 1).
    subset_max_rows: int = 0          # 0 => disabled
    subset_recent_years: int = 0      # 0 => disabled

    # SAME proportions as your script
    pretest_prop: float = 0.822871               # first chunk (train+val)
    inner_train_prop: float = 0.822871           # within pretest, the "train" part proportion
    # Derived: horizon fraction within pretest = 1 - inner_train_prop (same as your val fraction)

    # Temporal CV
    n_splits: int = 5
    min_train_fraction_of_pretest: float = 0.45  # ensures early folds have enough history

    # Bootstrap / subsampling
    bootstrap_seeds: Tuple[int, ...] = (7, 19, 41, 73, 101, 137)
    bootstrap_size: int = 220_000               # per-trial training rows (subsampled w/ replacement)
    bootstrap_method: str = "stratified_month"  # 'uniform', 'stratified_year', 'stratified_month'

    # Parallelism
    parallel_jobs: int = 120  # set to ~#physical cores; script will cap by os.cpu_count()

    # Checkpointing (write partial outputs so far)
    checkpoint_dir: str = "checkpoints"
    checkpoint_prefix: str = "ccao_cv"
    checkpoint_every_fold: bool = True
    handle_signals: bool = True

    # Tuning budget (adjust for runtime)
    lgbm_stage1_candidates: int = 80
    lgbm_stage1_folds: int = 2          # evaluate on the most recent folds first
    lgbm_stage1_topk: int = 12
    lgbm_stage2_candidates: int = 12    # (top-k from stage1)
    lgbm_early_stopping_rounds: int = 200
    lgbm_max_estimators: int = 10_000   # rely on early stopping

    # Other models tuning budgets
    elasticnet_candidates: int = 24
    rf_candidates: int = 12
    gbr_candidates: int = 12

    # Misc
    random_seed: int = 234
    verbose: bool = True


CFG = CVConfig(
    parallel_jobs=32,
    bootstrap_size=100_000,
    subset_max_rows=20_000,
    # subset_recent_years=3,
    lgbm_stage1_candidates=10000,
    lgbm_stage1_topk=8,
    lgbm_stage2_candidates=100,
    bootstrap_seeds=(123, 3432, 8354),

    # other models
    elasticnet_candidates = 0,
    rf_candidates = 0,
    gbr_candidates = 0,
)
# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------
def _log(msg: str) -> None:
    if CFG.verbose:
        print(msg, flush=True)



# -----------------------------------------------------------------------------
# Checkpoint helpers + signal handling
# -----------------------------------------------------------------------------
_STOP_REQUESTED = False

def _handle_stop_signal(signum, frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    _log(f"Received signal {signum}. Will checkpoint and stop after current fold.")

def atomic_write_json(obj: dict, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)

def append_rows_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    write_header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=write_header, index=False)

def save_leaderboard(df_rows: pd.DataFrame, out_path: str, stage: str) -> None:
    """
    Save a compact best-so-far snapshot:
      - best config overall (by MAE then RMSE, original scale)
      - top 25 leaderboard
    """
    if df_rows.empty:
        return
    agg = (
        df_rows.groupby(["model", "params"], as_index=False)
        .agg(
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            mae_log=("mae_log", "mean"),
            rmse_log=("rmse_log", "mean"),
            elapsed_s=("elapsed_s", "mean"),
            n=("mae", "size"),
        )
        .sort_values(["mae", "rmse"], ascending=True)
    )
    best = agg.iloc[0].to_dict()
    payload = {
        "stage": stage,
        "best_so_far": best,
        "top25": agg.head(25).to_dict(orient="records"),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    atomic_write_json(payload, out_path)
def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def safe_to_csr(X: Any) -> Any:
    """Make sure sparse matrices are CSR; leave dense alone."""
    if sp.issparse(X):
        return X.tocsr()
    return X



def row_slice(X: Any, idx: np.ndarray) -> Any:
    """
    Row-slice helper that works for:
      - scipy sparse matrices: X[idx]
      - numpy arrays: X[idx]
      - pandas DataFrame/Series: X.iloc[idx]
    """
    if sp.issparse(X):
        return X[idx]
    # pandas DataFrame/Series
    if hasattr(X, "iloc"):
        return X.iloc[idx]
    # numpy / list-like
    return X[idx]



def exp_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> Dict[str, float]:
    """Metrics in original sale price scale."""
    # clip to avoid inf on exp for any weird predictions
    y_true = np.exp(np.clip(y_true_log, -50, 50))
    y_pred = np.exp(np.clip(y_pred_log, -50, 50))
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return {"mae": float(mae), "rmse": float(rmse)}


def log_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true_log, y_pred_log)
    rmse = math.sqrt(mean_squared_error(y_true_log, y_pred_log))
    return {"mae_log": float(mae), "rmse_log": float(rmse)}


def stratified_bootstrap_indices(
    train_idx: np.ndarray,
    groups: np.ndarray,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Stratified bootstrap by groups.
    - train_idx: indices into full fold arrays
    - groups: group label for each row in the fold arrays
    - size: output sample size
    """
    g = groups[train_idx]
    uniq, counts = np.unique(g, return_counts=True)
    probs = counts / counts.sum()
    # sample groups, then sample within each chosen group
    sampled_groups = rng.choice(uniq, size=size, replace=True, p=probs)
    # map group -> indices
    idx_by_group: Dict[Any, np.ndarray] = {}
    for u in uniq:
        idx_by_group[u] = train_idx[g == u]
    out = np.empty(size, dtype=np.int64)
    # draw one row per drawn group
    for i, u in enumerate(sampled_groups):
        pool = idx_by_group[u]
        out[i] = pool[rng.integers(0, pool.shape[0])]
    return out


def make_bootstrap_indices(
    train_idx: np.ndarray,
    date_groups: np.ndarray,
    seed: int,
    size: int,
    method: str,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if size <= 0 or size >= train_idx.shape[0]:
        # full bootstrap size default to training size
        size = train_idx.shape[0]
    if method == "uniform":
        return rng.choice(train_idx, size=size, replace=True)
    if method == "stratified_year" or method == "stratified_month":
        return stratified_bootstrap_indices(train_idx, date_groups, size=size, rng=rng)
    raise ValueError(f"Unknown bootstrap_method={method}")


def make_temporal_folds(pretest_n: int, horizon: int, n_splits: int, min_train_n: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Forward-chaining folds with fixed validation horizon.
    For fold k:
      train = [0 : train_end)
      val   = [train_end : train_end + horizon)
    with strictly increasing train_end and last fold ending at pretest_n.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    max_train_end = pretest_n - horizon
    if max_train_end <= min_train_n:
        raise ValueError("Not enough pretest data for requested horizon/min_train_n")

    # choose train_end positions, biased toward later dates (more relevant for "next year" eval)
    # We space them linearly over [min_train_n, max_train_end]
    train_ends = np.linspace(min_train_n, max_train_end, num=n_splits).astype(int)
    # make strictly increasing unique
    train_ends = np.unique(train_ends)
    folds = []
    for te in train_ends:
        tr = np.arange(0, te, dtype=np.int64)
        va = np.arange(te, te + horizon, dtype=np.int64)
        folds.append((tr, va))

    # ensure last fold reaches the end of pretest (optional but helpful)
    if folds[-1][1][-1] != pretest_n - 1:
        te = max_train_end
        folds[-1] = (np.arange(0, te, dtype=np.int64), np.arange(te, te + horizon, dtype=np.int64))
    return folds


# -----------------------------------------------------------------------------
# Model parameter sampling
# -----------------------------------------------------------------------------
def sample_lgbm_params(rng: np.random.Generator) -> Dict[str, Any]:
    """
    Randomly sample a reasonably strong LightGBM configuration.
    Focus on the most impactful knobs: num_leaves, max_depth, min_child_samples,
    feature_fraction / colsample_bytree, bagging_fraction + bagging_freq,
    and L1/L2 regularization.
    """
    # Depth / leaves (leaf-wise growth -> num_leaves is critical)
    max_depth = int(rng.integers(6, 17))  # 6..16
    # keep leaves significantly below 2^max_depth to reduce overfitting
    max_leaves = max(31, min(2047, 2 ** max_depth - 1))
    num_leaves = int(rng.integers(31, max_leaves + 1))

    learning_rate = float(10 ** rng.uniform(math.log10(0.01), math.log10(0.15)))
    min_child_samples = int(rng.integers(20, 201))

    feature_fraction = float(rng.uniform(0.65, 1.0))
    bagging_fraction = float(rng.uniform(0.60, 1.0))
    bagging_freq = int(rng.integers(0, 6))  # 0 disables bagging
    # if bagging_fraction < 1, we generally want bagging_freq > 0
    if bagging_fraction < 0.999 and bagging_freq == 0:
        bagging_freq = int(rng.integers(1, 6))

    # regularization
    lambda_l1 = float(10 ** rng.uniform(-4, 1))  # 1e-4 .. 10
    lambda_l2 = float(10 ** rng.uniform(-4, 1))  # 1e-4 .. 10
    min_gain_to_split = float(rng.uniform(0.0, 0.5))

    max_bin = int(rng.choice([255, 511, 1023]))
    subsample_for_bin = int(rng.choice([200_000, 400_000, 800_000]))

    params = {
        "boosting_type": "gbdt",
        "objective": "regression",  # training on log-price
        "n_estimators": CFG.lgbm_max_estimators,
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "max_depth": max_depth,
        "min_child_samples": min_child_samples,
        "feature_fraction": feature_fraction,     # alias of colsample_bytree in LightGBM params
        "bagging_fraction": bagging_fraction,     # alias of subsample
        "bagging_freq": bagging_freq,
        "lambda_l1": lambda_l1,
        "lambda_l2": lambda_l2,
        "min_gain_to_split": min_gain_to_split,
        "max_bin": max_bin,
        "subsample_for_bin": subsample_for_bin,
        "verbosity": -1,
        "random_state": int(rng.integers(0, 2**31 - 1)),
        "n_jobs": 1,               # outer parallelism handles CPU utilization
        "force_col_wise": True,    # avoid row/col auto test overhead; good default for large sparse-ish data
    }
    return params

def loguniform(rng, lo, hi):
    return 10 ** rng.uniform(math.log10(lo), math.log10(hi))

def sample_lgbm_params_focused(rng):
    # Smaller-ish trees generalize better in your run
    max_depth = int(rng.choice([6, 7, 8, 9, 10]))

    # Couple leaves to depth; keep in a moderate band
    max_leaves = min(2 ** max_depth, 512)
    min_leaves = max(31, 2 ** (max_depth - 1))
    num_leaves = int(rng.integers(min_leaves, max_leaves + 1))

    # Bias min_gain toward low values (square shrinks toward 0)
    u = rng.uniform(0.0, 1.0)
    min_gain_to_split = float((u ** 2) * 0.18)

    learning_rate = loguniform(rng, 0.008, 0.035)

    # Bias lambda_l1 toward smaller values (more probability near 1e-5..1e-3)
    l1 = 10 ** rng.uniform(-5, -1)
    l2 = loguniform(rng, 1e-4, 1.0)

    return dict(
        boosting_type="gbdt",
        objective="regression",
        n_estimators=10000,
        learning_rate=learning_rate,
        min_gain_to_split=min_gain_to_split,
        min_child_samples=int(rng.integers(50, 250)),
        max_depth=max_depth,
        num_leaves=num_leaves,
        feature_fraction=float(rng.uniform(0.60, 0.85)),
        bagging_fraction=float(rng.uniform(0.60, 0.85)),
        bagging_freq=int(rng.choice([1, 2, 3])),
        lambda_l1=float(l1),
        lambda_l2=float(l2),
        max_bin=int(rng.choice([255, 511, 1023])),
        subsample_for_bin=int(rng.choice([200000, 400000, 800000])),
        force_col_wise=True,
        verbosity=-1,
        n_jobs=1,
    )



def sample_elasticnet_params(rng: np.random.Generator) -> Dict[str, Any]:
    # Alpha and l1_ratio are the key knobs
    alpha = float(10 ** rng.uniform(-5, -1))  # 1e-5 .. 1e-1
    l1_ratio = float(rng.uniform(0.05, 0.95))
    return {
        "alpha": alpha,
        "l1_ratio": l1_ratio,
        "fit_intercept": True,
        "max_iter": 2000,
        "selection": "random",
        "random_state": int(rng.integers(0, 2**31 - 1)),
    }


def sample_rf_params(rng: np.random.Generator) -> Dict[str, Any]:
    n_estimators = int(rng.choice([400, 700, 1000]))
    max_depth_choice = rng.choice([None, 18, 28, 36])
    max_depth = None if max_depth_choice is None else int(max_depth_choice)
    min_samples_leaf = int(rng.choice([10, 20, 30, 50]))
    max_features_choices = ["sqrt", "log2", None, 0.6, 0.8, 1.0]
    max_features = max_features_choices[int(rng.integers(0, len(max_features_choices)))]
    if isinstance(max_features, (np.floating,)):
        max_features = float(max_features)
    return {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
        "bootstrap": True,
        "n_jobs": 1,  # outer parallelism
        "random_state": int(rng.integers(0, 2**31 - 1)),
    }


def sample_gbr_params(rng: np.random.Generator) -> Dict[str, Any]:
    n_estimators = int(rng.choice([400, 800, 1200]))
    learning_rate = float(10 ** rng.uniform(math.log10(0.02), math.log10(0.10)))
    max_depth = int(rng.choice([2, 3, 4]))
    subsample = float(rng.uniform(0.6, 1.0))
    min_samples_leaf = int(rng.choice([10, 20, 30, 50]))
    return {
        "loss": "squared_error",
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "random_state": int(rng.integers(0, 2**31 - 1)),
    }


# -----------------------------------------------------------------------------
# Training / evaluation for one task (one model config on one fold with one bootstrap)
# -----------------------------------------------------------------------------
def fit_predict_one(
    model_name: str,
    model_params: Dict[str, Any],
    X_tr: Any,
    y_tr_log: np.ndarray,
    X_va: Any,
    y_va_log: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float], Dict[str, Any]]:
    """
    Fit model on (X_tr, y_tr_log) and predict on X_va.
    Returns predictions (log), metrics dict, and fitted metadata (e.g., best_iteration for LGBM).
    """
    meta: Dict[str, Any] = {}

    if model_name == "lgbm":
        reg = lgb.LGBMRegressor(**model_params)
        callbacks = [
            lgb.early_stopping(
                stopping_rounds=CFG.lgbm_early_stopping_rounds,
                first_metric_only=True,
                verbose=False,
            )
        ]
        reg.fit(
            X_tr,
            y_tr_log,
            eval_set=[(X_va, y_va_log)],
            eval_metric="l1",  # optimize MAE on log-scale for early stopping stability
            callbacks=callbacks,
        )
        best_it = getattr(reg, "best_iteration_", None)
        meta["best_iteration"] = int(best_it) if best_it is not None else None
        # If num_iteration=None and best_iteration exists, LightGBM uses best_iteration (docs).
        y_hat = reg.predict(X_va, num_iteration=best_it)
        y_hat = np.asarray(y_hat, dtype=float)

    elif model_name == "elasticnet":
        reg = ElasticNet(**model_params)
        reg.fit(X_tr, y_tr_log)
        y_hat = np.asarray(reg.predict(X_va), dtype=float)

    elif model_name == "rf":
        reg = RandomForestRegressor(**model_params)
        reg.fit(X_tr, y_tr_log)
        y_hat = np.asarray(reg.predict(X_va), dtype=float)

    elif model_name == "gbr":
        reg = GradientBoostingRegressor(**model_params)
        reg.fit(X_tr, y_tr_log)
        y_hat = np.asarray(reg.predict(X_va), dtype=float)

    else:
        raise ValueError(f"Unknown model_name={model_name}")

    m = {}
    m.update(log_metrics(y_va_log, y_hat))
    m.update(exp_metrics(y_va_log, y_hat))
    return y_hat, m, meta


def evaluate_task(
    model_name: str,
    model_params: Dict[str, Any],
    X_train_fold: Any,
    y_train_log_fold: np.ndarray,
    X_val_fold: Any,
    y_val_log_fold: np.ndarray,
    train_idx_fold: np.ndarray,
    date_groups_fold: np.ndarray,
    bootstrap_seed: int,
) -> Dict[str, Any]:
    # sample bootstrap indices within training window
    bs_idx = make_bootstrap_indices(
        train_idx=train_idx_fold,
        date_groups=date_groups_fold,
        seed=bootstrap_seed,
        size=CFG.bootstrap_size,
        method=CFG.bootstrap_method,
    )

    X_tr = row_slice(X_train_fold, bs_idx)
    y_tr = y_train_log_fold[bs_idx]
    # ensure CSR for sparse ops
    X_tr = safe_to_csr(X_tr)
    X_va = safe_to_csr(X_val_fold)

    t0 = time.time()
    _, metrics, meta = fit_predict_one(model_name, model_params, X_tr, y_tr, X_va, y_val_log_fold)
    elapsed = time.time() - t0

    out = {
        "model": model_name,
        "params": json.dumps(model_params, sort_keys=True),
        "bootstrap_seed": int(bootstrap_seed),
        "elapsed_s": float(elapsed),
        **metrics,
        **{f"meta_{k}": v for k, v in meta.items()},
    }
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    set_global_seeds(CFG.random_seed)

    # --- checkpoint paths ---
    run_ts = time.strftime("%Y%m%d_%H%M%S")
    ckpt_dir = CFG.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_prefix = f"{CFG.checkpoint_prefix}_{run_ts}"
    ckpt_rows_path = os.path.join(ckpt_dir, f"{ckpt_prefix}_rows_partial.csv")
    ckpt_best_path = os.path.join(ckpt_dir, f"{ckpt_prefix}_best_so_far.json")
    ckpt_note_path = os.path.join(ckpt_dir, f"{ckpt_prefix}_notes.json")

    # Register signal handlers so we can checkpoint if the job is preempted/terminated.
    if CFG.handle_signals:
        try:
            import signal
            signal.signal(signal.SIGTERM, _handle_stop_signal)
            signal.signal(signal.SIGINT, _handle_stop_signal)
        except Exception as e:
            _log(f"WARNING: could not register signal handlers: {e!r}")

    atomic_write_json(
        {"run_ts": run_ts, "checkpoint_rows": ckpt_rows_path, "checkpoint_best": ckpt_best_path, "config": CFG.__dict__},
        ckpt_note_path,
    )

    # --- load params.yaml (column lists for pipeline) ---
    with open(CFG.params_yaml_path, "r") as f:
        params_yaml = yaml.safe_load(f)

    pred_all = params_yaml["model"]["predictor"]["all"]
    pred_cat = params_yaml["model"]["predictor"]["categorical"]

    # --- load data ---
    _log(f"Loading parquet: {CFG.parquet_path}")
    df = pd.read_parquet(CFG.parquet_path, engine="fastparquet")

    # same filtering as your script
    df = df[
        (~df["ind_pin_is_multicard"].astype("bool").fillna(True))
        & (~df["sv_is_outlier"].astype("bool").fillna(True))
    ].copy()

    # select needed columns only (saves memory)
    desired_columns = pred_all + [CFG.target_name, CFG.date_col]
    df = df.loc[:, desired_columns].copy()

    # temporal sort
    df.sort_values(by=CFG.date_col, ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # --- optional subset for speed ---
    # Keep most recent rows (recommended) and/or restrict to recent years.
    if CFG.subset_recent_years and CFG.subset_recent_years > 0:
        dt_all = pd.to_datetime(df[CFG.date_col])
        max_year = int(dt_all.dt.year.max())
        min_year = max_year - int(CFG.subset_recent_years) + 1
        df = df.loc[dt_all.dt.year >= min_year, :].copy()
        df.sort_values(by=CFG.date_col, ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        _log(f"Subset (recent years): kept years >= {min_year} (max_year={max_year}) -> rows={df.shape[0]:,}")

    if CFG.subset_max_rows and CFG.subset_max_rows > 0 and df.shape[0] > CFG.subset_max_rows:
        df = df.iloc[-int(CFG.subset_max_rows):, :].copy()
        df.reset_index(drop=True, inplace=True)
        _log(f"Subset (last N rows): kept last {CFG.subset_max_rows:,} rows -> rows={df.shape[0]:,}")

    n = df.shape[0]
    _log(f"Rows after filtering: {n:,}")
    if n < 50_000:
        _log("WARNING: dataset size is unexpectedly small; check parquet path / filters.")

    # --- outer split (pretest vs test) ---
    pretest_n = int(CFG.pretest_prop * n)
    df_pretest = df.iloc[:pretest_n, :].copy()
    df_test = df.iloc[pretest_n:, :].copy()

    _log(f"Pre-test rows: {df_pretest.shape[0]:,} ({CFG.pretest_prop:.6f})")
    _log(f"Test rows:     {df_test.shape[0]:,} ({1-CFG.pretest_prop:.6f})")

    # date groups for stratified bootstrap
    dt = pd.to_datetime(df_pretest[CFG.date_col])
    if CFG.bootstrap_method == "stratified_year":
        date_groups = dt.dt.year.to_numpy()
    elif CFG.bootstrap_method == "stratified_month":
        date_groups = (dt.dt.year * 100 + dt.dt.month).to_numpy()
    else:
        date_groups = np.zeros(df_pretest.shape[0], dtype=int)

    # --- folds within pretest ---
    horizon = int((1.0 - CFG.inner_train_prop) * df_pretest.shape[0])
    min_train_n = max(int(CFG.min_train_fraction_of_pretest * df_pretest.shape[0]), 10_000)

    folds = make_temporal_folds(pretest_n=df_pretest.shape[0], horizon=horizon, n_splits=CFG.n_splits, min_train_n=min_train_n)
    _log(f"Temporal folds: {len(folds)} (horizon={horizon:,} rows each)")

    # cap parallel jobs by CPU count
    cpu = os.cpu_count() or 1
    n_jobs = int(min(CFG.parallel_jobs, cpu))
    _log(f"Parallel jobs: {n_jobs} (cpu_count={cpu})")

    # We'll store all results here
    all_rows: List[Dict[str, Any]] = []

    # -----------------------------------------------------------------------------
    # STAGE 1: LightGBM racing (many configs, few folds) to prune fast
    # -----------------------------------------------------------------------------
    rng = np.random.default_rng(CFG.random_seed)
    # lgbm_candidates = [sample_lgbm_params(rng) for _ in range(CFG.lgbm_stage1_candidates)]
    lgbm_candidates = [sample_lgbm_params_focused(rng) for _ in range(CFG.lgbm_stage1_candidates)]
    # Use the most recent folds first (closest to test distribution)
    folds_stage1 = folds[-min(CFG.lgbm_stage1_folds, len(folds)) :]

    _log(f"[Stage 1] LGBM candidates: {len(lgbm_candidates)}, folds used: {len(folds_stage1)}")

    for fold_id, (tr_idx, va_idx) in enumerate(folds_stage1, start=len(folds) - len(folds_stage1)):
        _log(f"[Stage 1] Fold {fold_id}: train={tr_idx.shape[0]:,}, val={va_idx.shape[0]:,}")

        # Fold-level preprocessing fit (NO leakage from val)
        df_tr = df_pretest.iloc[tr_idx].copy()
        df_va = df_pretest.iloc[va_idx].copy()

        y_tr_log = np.log(df_tr[CFG.target_name].to_numpy(dtype=float))
        y_va_log = np.log(df_va[CFG.target_name].to_numpy(dtype=float))

        X_tr_raw = df_tr.drop(columns=[CFG.date_col, CFG.target_name])
        X_va_raw = df_va.drop(columns=[CFG.date_col, CFG.target_name])

        pipe = build_model_pipeline(pred_vars=pred_all, cat_vars=pred_cat, id_vars=[])

        X_tr = pipe.fit_transform(X_tr_raw, y_tr_log)
        X_va = pipe.transform(X_va_raw)
        X_tr = safe_to_csr(X_tr)
        X_va = safe_to_csr(X_va)

        # tasks: (candidate params) x (bootstrap seeds)
        tasks = []
        for cand in lgbm_candidates:
            for bs in CFG.bootstrap_seeds:
                tasks.append((cand, bs))

        fold_rows = Parallel(n_jobs=n_jobs, backend="threading", prefer="threads")(
            delayed(evaluate_task)(
                "lgbm",
                cand,
                X_tr,
                y_tr_log,
                X_va,
                y_va_log,
                np.arange(X_tr.shape[0], dtype=np.int64),
                date_groups[tr_idx],
                bs,
            )
            for (cand, bs) in tasks
        )
        for r in fold_rows:
            r["fold"] = int(fold_id)
        all_rows.extend(fold_rows)
        if CFG.checkpoint_every_fold:
            for r in fold_rows:
                r["stage"] = "stage1"
            append_rows_csv(fold_rows, ckpt_rows_path)
            try:
                df_tmp = pd.DataFrame(all_rows)
                save_leaderboard(df_tmp, ckpt_best_path, stage="stage1")
            except Exception as e:
                _log(f"WARNING: checkpoint leaderboard failed: {e!r}")

        if _STOP_REQUESTED:
            _log("Stop requested. Exiting after Stage 1 fold checkpoint.")
            return

    df_stage1 = pd.DataFrame(all_rows)
    df_lgbm_stage1 = df_stage1[df_stage1["model"] == "lgbm"].copy()
    # aggregate per param string
    agg1 = (
        df_lgbm_stage1.groupby("params", as_index=False)
        .agg(
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            mae_log=("mae_log", "mean"),
            rmse_log=("rmse_log", "mean"),
            elapsed_s=("elapsed_s", "mean"),
        )
        .sort_values(["mae", "rmse"], ascending=True)
    )
    topk_params = agg1.head(CFG.lgbm_stage1_topk)["params"].tolist()
    topk_lgbm = [json.loads(s) for s in topk_params]
    _log(f"[Stage 1] Selected top-k LGBM configs: {len(topk_lgbm)}")

    # -----------------------------------------------------------------------------
    # STAGE 2: Full evaluation of top LGBM + tuned baselines across ALL folds
    # -----------------------------------------------------------------------------
    # sample candidate configs for other models
    rng2 = np.random.default_rng(CFG.random_seed + 11)
    enet_candidates = [sample_elasticnet_params(rng2) for _ in range(CFG.elasticnet_candidates)]
    rf_candidates = [sample_rf_params(rng2) for _ in range(CFG.rf_candidates)]
    gbr_candidates = [sample_gbr_params(rng2) for _ in range(CFG.gbr_candidates)]

    model_grid = (
        [("lgbm", p) for p in topk_lgbm[: CFG.lgbm_stage2_candidates]]
        + [("elasticnet", p) for p in enet_candidates]
        + [("rf", p) for p in rf_candidates]
        + [("gbr", p) for p in gbr_candidates]
    )
    _log(f"[Stage 2] Total configs: {len(model_grid)} (LGBM={sum(1 for m,_ in model_grid if m=='lgbm')})")

    stage2_rows: List[Dict[str, Any]] = []

    for fold_id, (tr_idx, va_idx) in enumerate(folds):
        _log(f"[Stage 2] Fold {fold_id}: train={tr_idx.shape[0]:,}, val={va_idx.shape[0]:,}")

        df_tr = df_pretest.iloc[tr_idx].copy()
        df_va = df_pretest.iloc[va_idx].copy()

        y_tr_log = np.log(df_tr[CFG.target_name].to_numpy(dtype=float))
        y_va_log = np.log(df_va[CFG.target_name].to_numpy(dtype=float))

        X_tr_raw = df_tr.drop(columns=[CFG.date_col, CFG.target_name])
        X_va_raw = df_va.drop(columns=[CFG.date_col, CFG.target_name])

        pipe = build_model_pipeline(pred_vars=pred_all, cat_vars=pred_cat, id_vars=[])
        X_tr = pipe.fit_transform(X_tr_raw, y_tr_log)
        X_va = pipe.transform(X_va_raw)
        X_tr = safe_to_csr(X_tr)
        X_va = safe_to_csr(X_va)

        fold_tasks = []
        for (mname, mparams) in model_grid:
            for bs in CFG.bootstrap_seeds:
                fold_tasks.append((mname, mparams, bs))

        fold_rows = Parallel(n_jobs=n_jobs, backend="threading", prefer="threads")(
            delayed(evaluate_task)(
                mname,
                mparams,
                X_tr,
                y_tr_log,
                X_va,
                y_va_log,
                np.arange(X_tr.shape[0], dtype=np.int64),
                date_groups[tr_idx],
                bs,
            )
            for (mname, mparams, bs) in fold_tasks
        )

        for r in fold_rows:
            r["fold"] = int(fold_id)
            r["pipeline_fitted_on"] = f"fold_{fold_id}"
        stage2_rows.extend(fold_rows)
        if CFG.checkpoint_every_fold:
            for r in fold_rows:
                r["stage"] = "stage2"
            append_rows_csv(fold_rows, ckpt_rows_path)
            try:
                df_tmp = pd.DataFrame(stage2_rows)
                save_leaderboard(df_tmp, ckpt_best_path, stage="stage2")
            except Exception as e:
                _log(f"WARNING: checkpoint leaderboard failed: {e!r}")

        if _STOP_REQUESTED:
            _log("Stop requested. Exiting after Stage 2 fold checkpoint.")
            return

    df_cv = pd.DataFrame(stage2_rows)

    # aggregate CV score per (model, params)
    agg = (
        df_cv.groupby(["model", "params"], as_index=False)
        .agg(
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            mae_log=("mae_log", "mean"),
            rmse_log=("rmse_log", "mean"),
            elapsed_s=("elapsed_s", "mean"),
        )
        .sort_values(["mae", "rmse"], ascending=True)
    )

    _log("\nTop 20 configs by CV mean MAE (original scale):")
    _log(agg.head(20).to_string(index=False))

    best_row = agg.iloc[0].to_dict()
    best_model = best_row["model"]
    best_params = json.loads(best_row["params"])
    _log(f"\nBEST: model={best_model}, CV_MAE={best_row['mae']:.2f}, CV_RMSE={best_row['rmse']:.2f}")

    # -----------------------------------------------------------------------------
    # Final training on full pretest, evaluate on test
    # -----------------------------------------------------------------------------
    # Use the SAME inner split proportion to create a final validation for early stopping
    # and to pick best_iteration for LGBM before fitting on full pretest.
    df_pretest = df_pretest.copy()
    split_point = int(CFG.inner_train_prop * df_pretest.shape[0])
    df_train_final = df_pretest.iloc[:split_point].copy()
    df_val_final = df_pretest.iloc[split_point:].copy()

    y_train_log = np.log(df_train_final[CFG.target_name].to_numpy(dtype=float))
    y_val_log = np.log(df_val_final[CFG.target_name].to_numpy(dtype=float))
    y_test_log = np.log(df_test[CFG.target_name].to_numpy(dtype=float))

    X_train_raw = df_train_final.drop(columns=[CFG.date_col, CFG.target_name])
    X_val_raw = df_val_final.drop(columns=[CFG.date_col, CFG.target_name])
    X_test_raw = df_test.drop(columns=[CFG.date_col, CFG.target_name])

    pipe_final = build_model_pipeline(pred_vars=pred_all, cat_vars=pred_cat, id_vars=[])
    X_train = pipe_final.fit_transform(X_train_raw, y_train_log)
    X_val = pipe_final.transform(X_val_raw)
    X_test = pipe_final.transform(X_test_raw)
    X_train = safe_to_csr(X_train)
    X_val = safe_to_csr(X_val)
    X_test = safe_to_csr(X_test)

    # Fit best model (with early stopping for LGBM)
    if best_model == "lgbm":
        reg = lgb.LGBMRegressor(**best_params)
        callbacks = [
            lgb.early_stopping(
                stopping_rounds=CFG.lgbm_early_stopping_rounds,
                first_metric_only=True,
                verbose=True,
            )
        ]
        reg.fit(
            X_train,
            y_train_log,
            eval_set=[(X_val, y_val_log)],
            eval_metric="l1",
            callbacks=callbacks,
        )
        best_it = getattr(reg, "best_iteration_", None)
        _log(f"Final LGBM best_iteration_ = {best_it}")
        # Refit on full pretest with best_iteration (optional but often slightly better):
        # Here we keep the model as trained with early stopping (already good).
        y_hat_test_log = reg.predict(X_test, num_iteration=best_it)

    elif best_model == "elasticnet":
        reg = ElasticNet(**best_params)
        reg.fit(X_train, y_train_log)
        y_hat_test_log = reg.predict(X_test)

    elif best_model == "rf":
        reg = RandomForestRegressor(**best_params)
        reg.fit(X_train, y_train_log)
        y_hat_test_log = reg.predict(X_test)

    elif best_model == "gbr":
        reg = GradientBoostingRegressor(**best_params)
        reg.fit(X_train, y_train_log)
        y_hat_test_log = reg.predict(X_test)

    else:
        raise ValueError(best_model)

    y_hat_test_log = np.asarray(y_hat_test_log, dtype=float)

    test_metrics = {}
    test_metrics.update(log_metrics(y_test_log, y_hat_test_log))
    test_metrics.update(exp_metrics(y_test_log, y_hat_test_log))

    _log("\nTEST metrics (original sale price scale):")
    _log(json.dumps({k: test_metrics[k] for k in ["mae", "rmse"]}, indent=2))
    _log("TEST metrics (log scale):")
    _log(json.dumps({k: test_metrics[k] for k in ["mae_log", "rmse_log"]}, indent=2))

    # Save artifacts
    out_dir = "artifacts"
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(out_dir, f"best_model_{best_model}_{ts}.txt" if best_model == "lgbm" else f"best_model_{best_model}_{ts}.joblib")
    pipe_path = os.path.join(out_dir, f"preprocess_pipeline_{ts}.joblib")
    results_path = os.path.join(out_dir, f"cv_results_{ts}.csv")

    # save CV results + summary
    df_cv.to_csv(results_path, index=False)
    try:
        save_leaderboard(df_cv, ckpt_best_path, stage="final")
    except Exception as e:
        _log(f"WARNING: final leaderboard save failed: {e!r}")

    _log(f"Saved CV results to: {results_path}")

    # save pipeline and model
    try:
        import joblib
        joblib.dump(pipe_final, pipe_path)
        _log(f"Saved preprocessing pipeline to: {pipe_path}")

        if best_model == "lgbm":
            # LightGBM native save_model
            reg.booster_.save_model(model_path)
            _log(f"Saved LightGBM model to: {model_path}")
        else:
            joblib.dump(reg, model_path)
            _log(f"Saved model to: {model_path}")

    except Exception as e:
        _log(f"WARNING: failed to save artifacts: {e!r}")

    # write a compact json summary
    summary_path = os.path.join(out_dir, f"run_summary_{ts}.json")
    summary = {
        "best_model": best_model,
        "best_params": best_params,
        "cv_best": best_row,
        "test_metrics": test_metrics,
        "config": CFG.__dict__,
        "paths": {"pipeline": pipe_path, "model": model_path, "cv_results": results_path},
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    _log(f"Saved run summary to: {summary_path}")


if __name__ == "__main__":
    main()
