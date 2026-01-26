from __future__ import annotations

import os
import json
import time
import hashlib
import datetime as _dt
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple, Union

import joblib


# -----------------------------
# JSON / hashing utilities
# -----------------------------
def _is_jsonable_scalar(x: Any) -> bool:
    return isinstance(x, (str, int, float, bool)) or x is None


def _to_jsonable(x: Any, *, max_list_len: int = 200) -> Any:
    """
    Best-effort conversion of objects to JSON-serializable structures.
    Keeps specs stable and avoids huge dumps.
    """
    if _is_jsonable_scalar(x):
        return x

    # Numpy scalars
    try:
        import numpy as np
        if isinstance(x, (np.integer, np.floating, np.bool_)):
            return x.item()
    except Exception:
        pass

    # dict
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if not isinstance(k, str):
                k = str(k)
            out[k] = _to_jsonable(v, max_list_len=max_list_len)
        return out

    # list/tuple
    if isinstance(x, (list, tuple)):
        seq = list(x)
        if len(seq) > max_list_len:
            seq = seq[:max_list_len] + ["<truncated>"]
        return [_to_jsonable(v, max_list_len=max_list_len) for v in seq]

    # sklearn-ish: get_params
    if hasattr(x, "get_params"):
        try:
            return _to_jsonable(x.get_params(deep=True), max_list_len=max_list_len)
        except Exception:
            pass

    # generic: repr
    return repr(x)


def _stable_json_dumps(obj: Any) -> str:
    return json.dumps(_to_jsonable(obj), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _hash_spec(spec: Dict[str, Any], *, n_chars: int = 16) -> str:
    payload = _stable_json_dumps(spec).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:n_chars]


# -----------------------------
# Run management
# -----------------------------
@dataclass(frozen=True)
class RunInfo:
    run_id: str
    run_dir: str
    created_at: str  # ISO datetime


def start_run(
    base_dir: str = "./artifacts",
    run_name: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> RunInfo:
    """
    Creates a run directory:
      artifacts/<YYYYMMDD_HHMMSS>__<run_name or 'run'>__<shortid>/

    Writes:
      run_meta.json
    """
    os.makedirs(base_dir, exist_ok=True)

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    short = hashlib.sha1(f"{time.time()}".encode("utf-8")).hexdigest()[:8]
    safe_name = (run_name or "run").replace(" ", "_")
    run_id = f"{ts}__{safe_name}__{short}"
    run_dir = os.path.join(base_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)

    run_meta = {
        "run_id": run_id,
        "created_at": _dt.datetime.now().isoformat(),
        "run_name": run_name,
        "meta": _to_jsonable(meta or {}),
    }
    with open(os.path.join(run_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=False)

    return RunInfo(run_id=run_id, run_dir=run_dir, created_at=run_meta["created_at"])


def save_preprocessor(
    run_dir: str,
    preprocessor: Any,
    *,
    filename: str = "preprocessor.joblib",
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Saves your fitted preprocessing pipeline / recipe pipeline / ColumnTransformer.
    Returns path.
    """
    path = os.path.join(run_dir, filename)
    joblib.dump(preprocessor, path, compress=3)

    if meta is not None:
        with open(os.path.join(run_dir, "preprocessor_meta.json"), "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(meta), f, indent=2, ensure_ascii=False)
    return path


# -----------------------------
# Model specs (unique identity)
# -----------------------------
def build_model_spec(
    model: Any,
    model_name: str,
    *,
    target_scale: str,          # "log" or "price"
    keep: Optional[float] = None,
    seed: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a stable, JSON-friendly spec that uniquely identifies the model.

    This is what gives you: "same class, different hyperparams => different model_id".
    """
    spec: Dict[str, Any] = {
        "model_name": str(model_name),
        "class_name": model.__class__.__name__,
        "class_module": model.__class__.__module__,
        "target_scale": target_scale,
        "keep": keep,
        "seed": seed,
    }

    # Grab params if available
    if hasattr(model, "get_params"):
        try:
            spec["params"] = _to_jsonable(model.get_params(deep=True))
        except Exception:
            spec["params"] = None
    else:
        # Fall back to a filtered __dict__ (avoid huge fields)
        try:
            d = dict(model.__dict__)
            # Keep only simple scalars and small dicts/lists
            filtered = {}
            for k, v in d.items():
                vv = _to_jsonable(v)
                # skip massive reprs
                if isinstance(vv, str) and len(vv) > 5000:
                    continue
                filtered[k] = vv
            spec["params"] = filtered
        except Exception:
            spec["params"] = None

    # If your wrapper holds an underlying estimator (common in your code)
    for attr in ["model", "estimator", "regressor", "booster_"]:
        if hasattr(model, attr):
            try:
                spec[f"has_{attr}"] = True
            except Exception:
                pass

    if extra:
        spec["extra"] = _to_jsonable(extra)

    return spec


# -----------------------------
# Saving fitted models
# -----------------------------
def _try_save_lightgbm_native(model: Any, out_dir: str) -> Optional[str]:
    """
    If model exposes a LightGBM Booster, save it natively (robust for SHAP, portability).
    Returns path if saved, else None.
    """
    try:
        import lightgbm as lgb  # noqa: F401
    except Exception:
        return None

    # sklearn API: LGBMRegressor has .booster_
    booster = None
    if hasattr(model, "booster_"):
        booster = getattr(model, "booster_", None)

    # some wrappers store the actual LGBMRegressor in .model
    if booster is None and hasattr(model, "model") and hasattr(model.model, "booster_"):
        booster = getattr(model.model, "booster_", None)

    # or native Booster stored directly
    if booster is None and model.__class__.__name__ == "Booster":
        booster = model

    if booster is None:
        return None

    out_path = os.path.join(out_dir, "lightgbm_booster.txt")
    try:
        booster.save_model(out_path)
        return out_path
    except Exception:
        return None


def save_fitted_model(
    run_dir: str,
    model: Any,
    spec: Dict[str, Any],
    *,
    save_joblib_model: bool = True,
    joblib_compress: int = 3,
    save_lightgbm_booster: bool = True,
) -> Tuple[str, str]:
    """
    Saves one fitted model under:
      <run_dir>/models/<model_id>/

    Files:
      spec.json
      model.joblib            (optional)
      lightgbm_booster.txt    (optional, if available)
    """
    model_id = _hash_spec(spec)
    model_dir = os.path.join(run_dir, "models", model_id)
    os.makedirs(model_dir, exist_ok=True)

    # spec.json
    with open(os.path.join(model_dir, "spec.json"), "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(spec), f, indent=2, ensure_ascii=False)

    # native LightGBM booster (recommended)
    booster_path = None
    if save_lightgbm_booster:
        booster_path = _try_save_lightgbm_native(model, model_dir)

    # joblib dump full object (works for sklearn + many wrappers)
    # note: if a class is defined in __main__ (not importable), loading may fail later
    if save_joblib_model:
        try:
            joblib.dump(model, os.path.join(model_dir, "model.joblib"), compress=joblib_compress)
        except Exception as e:
            # still keep spec + booster if available
            with open(os.path.join(model_dir, "joblib_error.txt"), "w", encoding="utf-8") as f:
                f.write(repr(e))

    # tiny manifest
    manifest = {
        "model_id": model_id,
        "model_dir": model_dir,
        "saved_lightgbm_booster": booster_path is not None,
        "lightgbm_booster_path": booster_path,
        "saved_joblib_model": save_joblib_model and os.path.exists(os.path.join(model_dir, "model.joblib")),
    }
    with open(os.path.join(model_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return model_id, model_dir


# -----------------------------
# Loading (optional but usually needed)
# -----------------------------
def load_fitted_model(model_dir: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Loads:
      - model.joblib if exists (preferred; preserves wrapper behavior)
      - else loads LightGBM booster if exists (returns lightgbm.Booster)
    Also returns spec dict from spec.json
    """
    spec_path = os.path.join(model_dir, "spec.json")
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    joblib_path = os.path.join(model_dir, "model.joblib")
    if os.path.exists(joblib_path):
        model = joblib.load(joblib_path)
        return model, spec

    booster_path = os.path.join(model_dir, "lightgbm_booster.txt")
    if os.path.exists(booster_path):
        import lightgbm as lgb
        booster = lgb.Booster(model_file=booster_path)
        return booster, spec

    raise FileNotFoundError(f"No loadable model found in {model_dir}")


def resolve_model_dir(run_dir: str, model_id: str) -> str:
    model_dir = os.path.join(run_dir, "models", model_id)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model dir not found: {model_dir}")
    return model_dir
