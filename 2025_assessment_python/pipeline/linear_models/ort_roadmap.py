
"""
optimal_regression_tree_roadmap.py

A practical, sklearn-like implementation roadmap for Optimal Regression Trees (ORT)
to compare against Gradient Boosting (e.g., LightGBM).

This file provides three main estimators:

Stage A (scalable heuristic):
  - OptimalRegressionTreeLS: axis-parallel splits + constant leaves
    trained via greedy CART-style growth + local search (coordinate-descent-ish)
    with random restarts and an explicit split penalty alpha.

Stage B (small-n exact MIQP sanity check):
  - OptimalRegressionTreeMIO: fixed-depth MIQP formulation (requires gurobipy).
    Intended for shallow depths / small datasets to benchmark heuristics.

Stage C (more expressive heuristic):
  - OptimalRegressionTreeHyperplaneLS: hyperplane (oblique) splits + constant leaves
    trained via greedy + local search with random hyperplane search / coordinate tweaks.

Notes / Scope:
- These are *research-grade* reference implementations focused on clarity and correctness.
- Stage A/C are heuristics; they will not match industrial ORT solvers in speed/quality.
- Stage B requires a MIQP solver (Gurobi) and is deliberately restricted to fixed-depth trees
  to keep the formulation readable and robust.

Dependencies:
  - numpy
  - scikit-learn (BaseEstimator, RegressorMixin)

Optional:
  - gurobipy (Stage B)

Author: ChatGPT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any
import copy
import math

import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils import check_random_state


# -----------------------------
# Tree node representation
# -----------------------------

@dataclass
class _Node:
    """A simple binary tree node supporting axis-parallel or hyperplane splits.

    If is_leaf=True: prediction is stored in value.
    If is_leaf=False:
      - axis split: feature is int, threshold is float, weights is None
      - hyperplane split: weights is (p,) array, threshold is float, feature is None
    """
    depth: int
    is_leaf: bool = True
    value: float = 0.0

    # Axis-parallel split parameters
    feature: Optional[int] = None
    threshold: Optional[float] = None

    # Hyperplane split parameters
    weights: Optional[np.ndarray] = None  # shape (p,)

    left: Optional["_Node"] = None
    right: Optional["_Node"] = None

    def n_splits(self) -> int:
        if self.is_leaf:
            return 0
        return 1 + (self.left.n_splits() if self.left else 0) + (self.right.n_splits() if self.right else 0)

    def copy_from(self, other: "_Node") -> None:
        """In-place replacement of this node's content (preserves references to this object)."""
        self.depth = other.depth
        self.is_leaf = other.is_leaf
        self.value = other.value
        self.feature = other.feature
        self.threshold = other.threshold
        self.weights = None if other.weights is None else np.array(other.weights, copy=True)
        self.left = other.left
        self.right = other.right


# -----------------------------
# Utility: losses and splits
# -----------------------------

def _sse(y: np.ndarray) -> float:
    """Sum of squared errors relative to the mean (i.e., variance * n)."""
    if y.size == 0:
        return 0.0
    mu = float(np.mean(y))
    diff = y - mu
    return float(np.dot(diff, diff))


def _leaf_value(y: np.ndarray) -> float:
    """Optimal constant prediction under squared loss."""
    if y.size == 0:
        return 0.0
    return float(np.mean(y))


def _scan_best_split_1d(x: np.ndarray, y: np.ndarray, min_leaf: int) -> Tuple[Optional[float], float]:
    """Given 1D feature x and targets y (aligned arrays for node samples),
    find the best threshold (midpoint between sorted unique x values) minimizing SSE.

    Returns: (best_threshold, best_sse_children)
      - best_threshold=None if no valid split exists
    """
    n = x.shape[0]
    if n < 2 * min_leaf:
        return None, float("inf")

    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ys = y[order]

    ps = np.cumsum(ys)
    ps2 = np.cumsum(ys * ys)

    total_sum = ps[-1]
    total_sum2 = ps2[-1]

    best_sse = float("inf")
    best_thr = None

    for i in range(min_leaf, n - min_leaf + 1):
        if xs[i - 1] == xs[i]:
            continue

        left_n = i
        right_n = n - i

        left_sum = ps[i - 1]
        left_sum2 = ps2[i - 1]
        right_sum = total_sum - left_sum
        right_sum2 = total_sum2 - left_sum2

        sse_left = left_sum2 - (left_sum * left_sum) / left_n
        sse_right = right_sum2 - (right_sum * right_sum) / right_n
        sse_children = float(sse_left + sse_right)

        if sse_children < best_sse:
            best_sse = sse_children
            best_thr = float(0.5 * (xs[i - 1] + xs[i]))

    return best_thr, best_sse


def _best_axis_split(
    X: np.ndarray,
    y: np.ndarray,
    idx: np.ndarray,
    min_leaf: int,
    max_features: Optional[Union[int, float]],
    rng: np.random.RandomState,
) -> Tuple[Optional[int], Optional[float], float]:
    """Find best axis-parallel (feature, threshold) split for node samples."""
    _, p = X.shape
    if idx.size < 2 * min_leaf:
        return None, None, float("inf")

    if max_features is None:
        feats = np.arange(p)
    elif isinstance(max_features, float):
        k = max(1, int(math.ceil(p * max_features)))
        feats = rng.choice(p, size=k, replace=False)
    else:
        k = max(1, min(p, int(max_features)))
        feats = rng.choice(p, size=k, replace=False)

    best_feat = None
    best_thr = None
    best_sse_children = float("inf")

    Xn = X[idx, :]
    yn = y[idx]

    for j in feats:
        thr, sse_children = _scan_best_split_1d(Xn[:, j], yn, min_leaf=min_leaf)
        if thr is None:
            continue
        if sse_children < best_sse_children:
            best_sse_children = sse_children
            best_feat = int(j)
            best_thr = float(thr)

    return best_feat, best_thr, best_sse_children


def _best_hyperplane_split(
    X: np.ndarray,
    y: np.ndarray,
    idx: np.ndarray,
    min_leaf: int,
    rng: np.random.RandomState,
    n_directions: int = 50,
    max_nonzero: Optional[int] = None,
    coord_steps: int = 0,
    coord_step_size: float = 0.25,
) -> Tuple[Optional[np.ndarray], Optional[float], float]:
    """Heuristic best hyperplane split at a node."""
    if idx.size < 2 * min_leaf:
        return None, None, float("inf")

    Xn = X[idx, :]
    yn = y[idx]
    p = X.shape[1]

    def normalize_w(w: np.ndarray) -> np.ndarray:
        s = np.sum(np.abs(w))
        if s <= 1e-12:
            j = rng.randint(0, p)
            w = np.zeros(p, dtype=float)
            w[j] = 1.0
            return w
        return w / s

    def sparsify(w: np.ndarray) -> np.ndarray:
        if max_nonzero is None or max_nonzero >= p:
            return w
        k = int(max_nonzero)
        keep = np.argsort(np.abs(w))[-k:]
        out = np.zeros_like(w)
        out[keep] = w[keep]
        return out

    def eval_direction(w: np.ndarray) -> Tuple[Optional[float], float]:
        proj = Xn @ w
        b, sse_children = _scan_best_split_1d(proj, yn, min_leaf=min_leaf)
        return b, sse_children

    best_w = None
    best_b = None
    best_sse = float("inf")

    for _ in range(int(n_directions)):
        w = rng.normal(size=p)
        w = sparsify(w)
        w = normalize_w(w)
        b, sse_children = eval_direction(w)
        if b is None:
            continue
        if sse_children < best_sse:
            best_sse = sse_children
            best_w = w
            best_b = float(b)

    if best_w is None:
        return None, None, float("inf")

    if coord_steps > 0:
        w = best_w.copy()
        for _ in range(int(coord_steps)):
            improved = False
            for j in range(p):
                for delta in (-coord_step_size, coord_step_size):
                    w_try = w.copy()
                    w_try[j] += delta
                    w_try = sparsify(w_try)
                    w_try = normalize_w(w_try)
                    b_try, sse_try = eval_direction(w_try)
                    if b_try is None:
                        continue
                    if sse_try + 1e-12 < best_sse:
                        best_sse = float(sse_try)
                        best_w = w_try
                        best_b = float(b_try)
                        w = best_w.copy()
                        improved = True
            if not improved:
                break

    return best_w, best_b, best_sse


# -----------------------------
# Building and evaluating trees
# -----------------------------

def _apply_split_axis(X: np.ndarray, idx: np.ndarray, feature: int, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    x = X[idx, feature]
    left_mask = x <= threshold
    return idx[left_mask], idx[~left_mask]


def _apply_split_hyperplane(X: np.ndarray, idx: np.ndarray, weights: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    proj = X[idx, :] @ weights
    left_mask = proj <= threshold
    return idx[left_mask], idx[~left_mask]


def _build_greedy_axis(
    X: np.ndarray,
    y: np.ndarray,
    idx: np.ndarray,
    depth: int,
    max_depth: int,
    min_leaf: int,
    alpha: float,
    baseline_sse: float,
    rng: np.random.RandomState,
    max_features: Optional[Union[int, float]],
) -> Tuple[_Node, float, int]:
    node = _Node(depth=depth, is_leaf=True)
    y_node = y[idx]
    node.value = _leaf_value(y_node)

    sse_leaf = _sse(y_node)
    obj_leaf = sse_leaf / (baseline_sse + 1e-12)

    if depth >= max_depth or idx.size < 2 * min_leaf:
        return node, sse_leaf, 0

    feat, thr, _ = _best_axis_split(X, y, idx, min_leaf=min_leaf, max_features=max_features, rng=rng)
    if feat is None:
        return node, sse_leaf, 0

    left_idx, right_idx = _apply_split_axis(X, idx, feat, thr)
    if left_idx.size < min_leaf or right_idx.size < min_leaf:
        return node, sse_leaf, 0

    left_node, sse_left, splits_left = _build_greedy_axis(
        X, y, left_idx, depth + 1, max_depth, min_leaf, alpha, baseline_sse, rng, max_features
    )
    right_node, sse_right, splits_right = _build_greedy_axis(
        X, y, right_idx, depth + 1, max_depth, min_leaf, alpha, baseline_sse, rng, max_features
    )

    sse_subtree = sse_left + sse_right
    splits_subtree = 1 + splits_left + splits_right
    obj_subtree = (sse_subtree / (baseline_sse + 1e-12)) + alpha * splits_subtree

    if obj_subtree + 1e-12 < obj_leaf:
        node.is_leaf = False
        node.feature = feat
        node.threshold = thr
        node.left = left_node
        node.right = right_node
        return node, sse_subtree, splits_subtree

    return node, sse_leaf, 0


def _build_greedy_hyperplane(
    X: np.ndarray,
    y: np.ndarray,
    idx: np.ndarray,
    depth: int,
    max_depth: int,
    min_leaf: int,
    alpha: float,
    baseline_sse: float,
    rng: np.random.RandomState,
    n_directions: int,
    max_nonzero: Optional[int],
    coord_steps: int,
    coord_step_size: float,
) -> Tuple[_Node, float, int]:
    node = _Node(depth=depth, is_leaf=True)
    y_node = y[idx]
    node.value = _leaf_value(y_node)

    sse_leaf = _sse(y_node)
    obj_leaf = sse_leaf / (baseline_sse + 1e-12)

    if depth >= max_depth or idx.size < 2 * min_leaf:
        return node, sse_leaf, 0

    w, b, _ = _best_hyperplane_split(
        X, y, idx, min_leaf=min_leaf, rng=rng,
        n_directions=n_directions, max_nonzero=max_nonzero,
        coord_steps=coord_steps, coord_step_size=coord_step_size
    )
    if w is None:
        return node, sse_leaf, 0

    left_idx, right_idx = _apply_split_hyperplane(X, idx, w, b)
    if left_idx.size < min_leaf or right_idx.size < min_leaf:
        return node, sse_leaf, 0

    left_node, sse_left, splits_left = _build_greedy_hyperplane(
        X, y, left_idx, depth + 1, max_depth, min_leaf, alpha, baseline_sse, rng,
        n_directions, max_nonzero, coord_steps, coord_step_size
    )
    right_node, sse_right, splits_right = _build_greedy_hyperplane(
        X, y, right_idx, depth + 1, max_depth, min_leaf, alpha, baseline_sse, rng,
        n_directions, max_nonzero, coord_steps, coord_step_size
    )

    sse_subtree = sse_left + sse_right
    splits_subtree = 1 + splits_left + splits_right
    obj_subtree = (sse_subtree / (baseline_sse + 1e-12)) + alpha * splits_subtree

    if obj_subtree + 1e-12 < obj_leaf:
        node.is_leaf = False
        node.weights = np.array(w, copy=True)
        node.threshold = float(b)
        node.left = left_node
        node.right = right_node
        return node, sse_subtree, splits_subtree

    return node, sse_leaf, 0


def _predict_one(x: np.ndarray, node: _Node) -> float:
    while not node.is_leaf:
        if node.weights is not None:
            node = node.left if float(np.dot(node.weights, x)) <= float(node.threshold) else node.right
        else:
            node = node.left if float(x[node.feature]) <= float(node.threshold) else node.right
        if node is None:
            return 0.0
    return float(node.value)


def _predict(X: np.ndarray, root: _Node) -> np.ndarray:
    out = np.empty(X.shape[0], dtype=float)
    for i in range(X.shape[0]):
        out[i] = _predict_one(X[i, :], root)
    return out


def _tree_objective(X: np.ndarray, y: np.ndarray, root: _Node, baseline_sse: float, alpha: float) -> Tuple[float, float, int]:
    yhat = _predict(X, root)
    resid = yhat - y
    sse = float(np.dot(resid, resid))
    splits = root.n_splits()
    obj = (sse / (baseline_sse + 1e-12)) + alpha * splits
    return obj, sse, splits


def _collect_indices_by_node(X: np.ndarray, idx: np.ndarray, node: _Node, out: Dict[int, np.ndarray]) -> None:
    out[id(node)] = idx
    if node.is_leaf:
        return
    if node.weights is not None:
        left_idx, right_idx = _apply_split_hyperplane(X, idx, node.weights, node.threshold)
    else:
        left_idx, right_idx = _apply_split_axis(X, idx, node.feature, node.threshold)
    _collect_indices_by_node(X, left_idx, node.left, out)
    _collect_indices_by_node(X, right_idx, node.right, out)


def _list_nodes_preorder(node: _Node) -> List[_Node]:
    nodes = [node]
    if not node.is_leaf:
        nodes.extend(_list_nodes_preorder(node.left))
        nodes.extend(_list_nodes_preorder(node.right))
    return nodes


# -----------------------------
# Stage A estimator
# -----------------------------

class OptimalRegressionTreeLS(BaseEstimator, RegressorMixin):
    """Axis-parallel ORT heuristic: greedy + local search + random restarts."""

    def __init__(
        self,
        max_depth: int = 3,
        min_samples_leaf: int = 20,
        alpha: float = 0.0,
        n_restarts: int = 5,
        max_iter: int = 10,
        max_features: Optional[Union[int, float]] = None,
        tol: float = 1e-6,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.alpha = alpha
        self.n_restarts = n_restarts
        self.max_iter = max_iter
        self.max_features = max_features
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: Any, y: Any):
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        rng_master = check_random_state(self.random_state)
        self.baseline_sse_ = float(np.dot(y - np.mean(y), y - np.mean(y)))

        best_tree = None
        best_obj = float("inf")

        n = X.shape[0]
        all_idx = np.arange(n, dtype=int)

        for r in range(int(self.n_restarts)):
            rng = check_random_state(rng_master.randint(0, 2**31 - 1))

            tree, _, _ = _build_greedy_axis(
                X=X, y=y, idx=all_idx, depth=0,
                max_depth=int(self.max_depth),
                min_leaf=int(self.min_samples_leaf),
                alpha=float(self.alpha),
                baseline_sse=float(self.baseline_sse_),
                rng=rng,
                max_features=self.max_features,
            )
            obj, _, splits = _tree_objective(X, y, tree, self.baseline_sse_, self.alpha)
            if self.verbose >= 2:
                print(f"[Restart {r+1}/{self.n_restarts}] init obj={obj:.6f}, splits={splits}")

            for it in range(int(self.max_iter)):
                improved_any = False
                idx_by_node: Dict[int, np.ndarray] = {}
                _collect_indices_by_node(X, all_idx, tree, idx_by_node)

                for node in _list_nodes_preorder(tree):
                    node_idx = idx_by_node.get(id(node), None)
                    if node_idx is None or node_idx.size == 0:
                        continue

                    node_backup = copy.deepcopy(node)
                    new_subtree, _, _ = _build_greedy_axis(
                        X=X, y=y, idx=node_idx, depth=node.depth,
                        max_depth=int(self.max_depth),
                        min_leaf=int(self.min_samples_leaf),
                        alpha=float(self.alpha),
                        baseline_sse=float(self.baseline_sse_),
                        rng=rng,
                        max_features=self.max_features,
                    )
                    node.copy_from(new_subtree)
                    new_obj, _, _ = _tree_objective(X, y, tree, self.baseline_sse_, self.alpha)
                    if new_obj + float(self.tol) < obj:
                        obj = new_obj
                        improved_any = True
                    else:
                        node.copy_from(node_backup)

                if not improved_any:
                    break

            obj, _, splits = _tree_objective(X, y, tree, self.baseline_sse_, self.alpha)
            if self.verbose >= 1:
                print(f"[Restart {r+1}/{self.n_restarts}] final obj={obj:.6f}, splits={splits}")

            if obj < best_obj:
                best_obj = obj
                best_tree = tree

        self.tree_ = best_tree
        self.best_objective_ = float(best_obj)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, ["tree_", "n_features_in_"])
        X = check_array(X, accept_sparse=False)
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_in_}.")
        return _predict(X, self.tree_)


# -----------------------------
# Stage B MIQP utilities + estimator
# -----------------------------

def _heap_children(t: int) -> Tuple[int, int]:
    return 2 * t + 1, 2 * t + 2


def _leaf_path_bits(D: int, leaf: int) -> List[int]:
    return [((leaf >> (D - 1 - d)) & 1) for d in range(D)]


def _leaf_ancestors_heap(D: int, leaf: int) -> List[Tuple[int, int]]:
    bits = _leaf_path_bits(D, leaf)
    out = []
    t = 0
    for b in bits:
        out.append((t, b))
        left, right = _heap_children(t)
        t = left if b == 0 else right
    return out


class OptimalRegressionTreeMIO(BaseEstimator, RegressorMixin):
    """Fixed-depth axis-parallel ORT via MIQP (requires gurobipy)."""

    def __init__(
        self,
        depth: int = 3,
        min_samples_leaf: int = 5,
        time_limit: Optional[float] = 60.0,
        mip_gap: Optional[float] = 0.01,
        eps: float = 1e-6,
        verbose: int = 0,
    ):
        self.depth = depth
        self.min_samples_leaf = min_samples_leaf
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.eps = eps
        self.verbose = verbose

    def fit(self, X: Any, y: Any):
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        n, p = X.shape
        D = int(self.depth)
        if D <= 0:
            raise ValueError("depth must be >= 1")

        try:
            import gurobipy as gp
            from gurobipy import GRB
        except Exception as e:
            raise ImportError(
                "OptimalRegressionTreeMIO requires gurobipy (Gurobi). "
                "Install/activate a Gurobi license and gurobipy."
            ) from e

        B = 2**D - 1
        L = 2**D

        x_min = X.min(axis=0)
        x_max = X.max(axis=0)
        M = float(np.max(x_max - x_min) + 1e-12)

        y_min = float(np.min(y))
        y_max = float(np.max(y))

        m = gp.Model("ORT_MIQP_fixed_depth")
        if self.verbose <= 0:
            m.Params.OutputFlag = 0
        if self.time_limit is not None:
            m.Params.TimeLimit = float(self.time_limit)
        if self.mip_gap is not None:
            m.Params.MIPGap = float(self.mip_gap)

        a = m.addVars(B, p, vtype=GRB.BINARY, name="a")
        b = m.addVars(B, lb=float(np.min(x_min)), ub=float(np.max(x_max)), vtype=GRB.CONTINUOUS, name="b")
        z = m.addVars(n, L, vtype=GRB.BINARY, name="z")
        beta = m.addVars(L, lb=y_min, ub=y_max, vtype=GRB.CONTINUOUS, name="beta")

        w = m.addVars(n, L, lb=y_min, ub=y_max, vtype=GRB.CONTINUOUS, name="w")
        yhat = m.addVars(n, vtype=GRB.CONTINUOUS, name="yhat")

        for i in range(n):
            m.addConstr(gp.quicksum(z[i, l] for l in range(L)) == 1, name=f"assign_{i}")

        for l in range(L):
            m.addConstr(gp.quicksum(z[i, l] for i in range(n)) >= int(self.min_samples_leaf), name=f"minleaf_{l}")

        for t in range(B):
            m.addConstr(gp.quicksum(a[t, j] for j in range(p)) == 1, name=f"onehot_{t}")

        for i in range(n):
            for l in range(L):
                m.addConstr(w[i, l] <= y_max * z[i, l], name=f"w_ub1_{i}_{l}")
                m.addConstr(w[i, l] >= y_min * z[i, l], name=f"w_lb1_{i}_{l}")
                m.addConstr(w[i, l] <= beta[l] - y_min * (1 - z[i, l]), name=f"w_ub2_{i}_{l}")
                m.addConstr(w[i, l] >= beta[l] - y_max * (1 - z[i, l]), name=f"w_lb2_{i}_{l}")
            m.addConstr(yhat[i] == gp.quicksum(w[i, l] for l in range(L)), name=f"yhat_{i}")

        for l in range(L):
            anc = _leaf_ancestors_heap(D, l)
            for (t, dirbit) in anc:
                for i in range(n):
                    expr = gp.quicksum(a[t, j] * float(X[i, j]) for j in range(p))
                    if dirbit == 0:
                        m.addConstr(expr <= b[t] + M * (1 - z[i, l]), name=f"pathL_{i}_{l}_{t}")
                    else:
                        m.addConstr(expr >= b[t] + float(self.eps) - M * (1 - z[i, l]), name=f"pathR_{i}_{l}_{t}")

        obj = gp.quicksum((float(y[i]) - yhat[i]) * (float(y[i]) - yhat[i]) for i in range(n))
        m.setObjective(obj, GRB.MINIMIZE)
        m.optimize()

        if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
            raise RuntimeError(f"Gurobi ended with status {m.Status}")

        a_sol = np.zeros((B, p), dtype=int)
        b_sol = np.zeros(B, dtype=float)
        for t in range(B):
            b_sol[t] = float(b[t].X)
            for j in range(p):
                a_sol[t, j] = int(round(a[t, j].X))
        beta_sol = np.array([float(beta[l].X) for l in range(L)], dtype=float)

        def build_node(t: int, depth: int, path_bits: List[int]) -> _Node:
            if depth == D:
                lid = 0
                for bb in path_bits:
                    lid = (lid << 1) | int(bb)
                return _Node(depth=depth, is_leaf=True, value=float(beta_sol[lid]))

            node = _Node(depth=depth, is_leaf=False)
            node.feature = int(np.argmax(a_sol[t, :]))
            node.threshold = float(b_sol[t])
            left_t, right_t = _heap_children(t)
            node.left = build_node(left_t, depth + 1, path_bits + [0])
            node.right = build_node(right_t, depth + 1, path_bits + [1])
            return node

        self.tree_ = build_node(0, 0, [])
        self.n_features_in_ = p
        self.objective_value_ = float(m.ObjVal)
        return self

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, ["tree_", "n_features_in_"])
        X = check_array(X, accept_sparse=False)
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_in_}.")
        return _predict(X, self.tree_)


# -----------------------------
# Stage C estimator
# -----------------------------

class OptimalRegressionTreeHyperplaneLS(BaseEstimator, RegressorMixin):
    """Hyperplane ORT heuristic: greedy + local search + random restarts."""

    def __init__(
        self,
        max_depth: int = 3,
        min_samples_leaf: int = 20,
        alpha: float = 0.0,
        n_restarts: int = 5,
        max_iter: int = 10,
        n_directions: int = 50,
        max_nonzero: Optional[int] = None,
        coord_steps: int = 0,
        coord_step_size: float = 0.25,
        tol: float = 1e-6,
        random_state: Optional[int] = None,
        verbose: int = 0,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.alpha = alpha
        self.n_restarts = n_restarts
        self.max_iter = max_iter
        self.n_directions = n_directions
        self.max_nonzero = max_nonzero
        self.coord_steps = coord_steps
        self.coord_step_size = coord_step_size
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: Any, y: Any):
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        rng_master = check_random_state(self.random_state)
        self.baseline_sse_ = float(np.dot(y - np.mean(y), y - np.mean(y)))

        best_tree = None
        best_obj = float("inf")

        n = X.shape[0]
        all_idx = np.arange(n, dtype=int)

        for r in range(int(self.n_restarts)):
            rng = check_random_state(rng_master.randint(0, 2**31 - 1))

            tree, _, _ = _build_greedy_hyperplane(
                X=X, y=y, idx=all_idx, depth=0,
                max_depth=int(self.max_depth),
                min_leaf=int(self.min_samples_leaf),
                alpha=float(self.alpha),
                baseline_sse=float(self.baseline_sse_),
                rng=rng,
                n_directions=int(self.n_directions),
                max_nonzero=self.max_nonzero,
                coord_steps=int(self.coord_steps),
                coord_step_size=float(self.coord_step_size),
            )
            obj, _, splits = _tree_objective(X, y, tree, self.baseline_sse_, self.alpha)
            if self.verbose >= 2:
                print(f"[Restart {r+1}/{self.n_restarts}] init obj={obj:.6f}, splits={splits}")

            for _ in range(int(self.max_iter)):
                improved_any = False
                idx_by_node: Dict[int, np.ndarray] = {}
                _collect_indices_by_node(X, all_idx, tree, idx_by_node)

                for node in _list_nodes_preorder(tree):
                    node_idx = idx_by_node.get(id(node), None)
                    if node_idx is None or node_idx.size == 0:
                        continue

                    node_backup = copy.deepcopy(node)
                    new_subtree, _, _ = _build_greedy_hyperplane(
                        X=X, y=y, idx=node_idx, depth=node.depth,
                        max_depth=int(self.max_depth),
                        min_leaf=int(self.min_samples_leaf),
                        alpha=float(self.alpha),
                        baseline_sse=float(self.baseline_sse_),
                        rng=rng,
                        n_directions=int(self.n_directions),
                        max_nonzero=self.max_nonzero,
                        coord_steps=int(self.coord_steps),
                        coord_step_size=float(self.coord_step_size),
                    )
                    node.copy_from(new_subtree)
                    new_obj, _, _ = _tree_objective(X, y, tree, self.baseline_sse_, self.alpha)
                    if new_obj + float(self.tol) < obj:
                        obj = new_obj
                        improved_any = True
                    else:
                        node.copy_from(node_backup)

                if not improved_any:
                    break

            obj, _, splits = _tree_objective(X, y, tree, self.baseline_sse_, self.alpha)
            if self.verbose >= 1:
                print(f"[Restart {r+1}/{self.n_restarts}] final obj={obj:.6f}, splits={splits}")

            if obj < best_obj:
                best_obj = obj
                best_tree = tree

        self.tree_ = best_tree
        self.best_objective_ = float(best_obj)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, ["tree_", "n_features_in_"])
        X = check_array(X, accept_sparse=False)
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_in_}.")
        return _predict(X, self.tree_)


if __name__ == "__main__":
    # Quick smoke test
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    rng = np.random.RandomState(0)
    n = 600
    X = rng.normal(size=(n, 6))
    y = (2.0 * (X[:, 0] > 0).astype(float)
         + 1.2 * X[:, 1]
         - 0.7 * X[:, 2]
         + 0.2 * rng.normal(size=n))

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)

    mA = OptimalRegressionTreeLS(max_depth=3, min_samples_leaf=30, alpha=1e-3,
                                n_restarts=3, max_iter=5, verbose=1, random_state=0)
    mA.fit(Xtr, ytr)
    pred = mA.predict(Xte)
    print("Stage A RMSE:", math.sqrt(mean_squared_error(yte, pred)))

    mC = OptimalRegressionTreeHyperplaneLS(max_depth=3, min_samples_leaf=30, alpha=1e-3,
                                          n_restarts=3, max_iter=5,
                                          n_directions=80, max_nonzero=3,
                                          coord_steps=1, coord_step_size=0.2,
                                          verbose=1, random_state=0)
    mC.fit(Xtr, ytr)
    pred = mC.predict(Xte)
    print("Stage C RMSE:", math.sqrt(mean_squared_error(yte, pred)))

    print("Stage B requires gurobipy; not run.")
