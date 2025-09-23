"""
Holistic Regression (MIO) as a scikit‑learn‑style estimator solved with CVXPY.

Model (for a chosen sparsity level k):
    minimize_{beta, b, z}  1/2||y - (X_t @ beta + b)||_2^2
                           + alpha1 * ||beta||_1
                           + 1/2 * alpha2 * ||beta||_2^2
    s.t.  z_i ∈ {0,1}
          -M z_i <= beta_i <= M z_i          (feature selected iff z_i = 1)
          sum_i z_i <= k                     (sparsity)
          z_i + z_j <= 1 for all (i,j) in H  (pairwise collinearity conflicts for non-interactions)

where X_t are transformed features built from the original numeric features first,
then correlations are computed on X_t (excluding interactions) to form the conflict set
H := {(i,j): |corr| >= rho_threshold}.

Options included here
---------------------
* Rich nonlinear transforms (identity, square, cube, abs, signed_sqrt, log1p_abs,
  tanh, sigmoid, sin, cos, recip1p_abs) — controlled globally or per feature.
* Optional **pairwise interactions** of original features (identity×identity only):
  - Excluded from the correlation conflict checks.
  - Hierarchical constraint: z_ij ≤ z_i and z_ij ≤ z_j.
  - Optional surrogate for the slide's sign rule: if interaction selected, force
    parent main effects β_i, β_j ≥ ε > 0.
* Optional **group sparsity** (all-or-nothing): all z's in a group equal.
* Elastic‑net objective (L1 + L2) with unpenalized intercept.
* Mixed-integer solvers: ECOS_BB (bundled), GLPK_MI/CBC/SCIP/GUROBI/CPLEX/MOSEK if installed.

Usage
-----
    model = HolisticRegressionCVX(
        k=20,
        alpha1=0.01,
        alpha2=0.1,
        rho_threshold=0.97,
        interactions=True,
        enforce_positive_main_when_interaction=True,
        use_groups=False,
        solver="ECOS_BB",
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

"""
from __future__ import annotations
from typing import Iterable, List, Optional, Tuple, Dict, Callable
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.utils.validation import check_X_y, check_array
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import cvxpy as cp
# import gurobipy as gp

# ---------- Nonlinear transform library ----------
# All transforms must be elementwise and return finite values for common inputs.
# They should be safe for negatives when possible.
TRANSFORM_FUNCS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "identity": lambda x: x,
    # polynomial
    "square": lambda x: x ** 2,
    "cube": lambda x: x ** 3,
    # magnitude/signed shapes
    "abs": lambda x: np.abs(x),
    "signed_sqrt": lambda x: np.sign(x) * np.sqrt(np.abs(x) + 1e-12),
    # logs use 1+|x| to be safe with non-positives
    "log1p_abs": lambda x: np.log1p(np.abs(x)),
    # bounded monotone / saturating
    "tanh": lambda x: np.tanh(x),
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
    # periodic (bounded)
    "sin": lambda x: np.sin(x),
    "cos": lambda x: np.cos(x),
    # reciprocal-like bounded variant
    "recip1p_abs": lambda x: 1.0 / (1.0 + np.abs(x)),
}


def _build_transforms(
    X: np.ndarray,
    feature_names: Optional[List[str]],
    transforms: Iterable[str],
    transform_menu: Optional[dict],
) -> Tuple[np.ndarray, List[str], List[Tuple[int, str]]]:
    """Apply per-feature transforms and return transformed matrix, names, and mapping.

    `transform_menu` can override `transforms` for chosen columns. Keys can be
    feature names (str) or column indices (int); values are lists of transform names.

    Returns
    -------
    X_t : ndarray (n_samples, n_new_features)
    names_t : list of str  names like f"{name}|{tname}"
    mapping : list of (orig_col_index, transform_name) for each new column
    """
    n, p = X.shape
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(p)]

    def menu_for_feature(j: int, name: str) -> List[str]:
        if not transform_menu:
            return list(transforms)
        if name in transform_menu:
            return list(transform_menu[name])
        if j in transform_menu:
            return list(transform_menu[j])
        return list(transforms)

    X_list = []
    names_t = []
    mapping: List[Tuple[int, str]] = []

    for j in range(p):
        xj = X[:, j]
        for tname in menu_for_feature(j, feature_names[j]):
            if tname not in TRANSFORM_FUNCS:
                raise ValueError(f"Unknown transform: {tname}")
            f = TRANSFORM_FUNCS[tname]
            col = f(xj)
            if not np.all(np.isfinite(col)):
                raise ValueError(
                    f"Transform {tname} produced non-finite values on feature {feature_names[j]}"
                )
            X_list.append(col.reshape(-1, 1))
            names_t.append(f"{feature_names[j]}|{tname}")
            mapping.append((j, tname))

    X_t = np.hstack(X_list) if X_list else np.empty((n, 0))
    return X_t, names_t, mapping


def _compute_conflicts(X_t: np.ndarray, rho_threshold: float) -> List[Tuple[int, int]]:
    """Compute conflict pairs (i,j) with |corr| >= rho_threshold on given columns.
       Return list of index pairs with i < j.
    """
    if X_t.shape[1] == 0:
        return []
    # center columns to compute Pearson correlation efficiently
    Xc = X_t - X_t.mean(axis=0, keepdims=True)
    std = Xc.std(axis=0, ddof=0)
    std[std == 0] = np.inf  # avoid division by zero -> correlation 0
    Xn = Xc / std
    # correlation matrix via normalized dot products
    C = (Xn.T @ Xn) / Xn.shape[0]
    # keep strict upper triangle
    p = C.shape[0]
    pairs: List[Tuple[int, int]] = []
    for i in range(p):
        for j in range(i + 1, p):
            if np.abs(C[i, j]) >= rho_threshold:
                pairs.append((i, j))
    return pairs


class HolisticRegressionCVX(BaseEstimator, RegressorMixin):
    """Holistic Regression (MIO) with nonlinear transforms, optional interactions,
    optional group sparsity, collinearity conflicts, and elastic‑net loss.

    Parameters
    ----------
    k : int
        Max number of selected transformed features.
    alpha1 : float, default=0.0
        L1 regularization strength on coefficients (encourages within-k shrinkage).
    alpha2 : float, default=0.0
        L2 regularization strength (elastic net ridge part).
    transforms : Iterable[str] or None
        Names of transforms to apply. Defaults to a rich set.
    transform_menu : dict or None
        Optional per-feature transform menu. Keys may be original feature names (str)
        or column indices (int); values are lists of transform names to apply to that
        feature. When provided, entries here override the global `transforms` for the
        specified features. Example: {"income": ["identity","log1p_abs"], 2: ["identity","square"]}.
    standardize : bool, default=True
        If True, standardize *original* features before applying transforms; improves scaling.
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set to False, no
        intercept will be used in calculations (i.e. data is expected to be centered).
    interactions : bool, default=False
        If True, add pairwise **interactions of original features (identity only)**: X_i * X_j
        for i<j. These columns are **excluded** from correlation/transform conflict checks,
        but they are tied hierarchically to their parent main effects.
    enforce_positive_main_when_interaction : bool, default=False
        If True, when an interaction is selected (z_ij=1) we also require beta_i >= eps and
        beta_j >= eps (eps>0 tiny). This is a simple surrogate for the slide's sign rule.
    rho_threshold : float, default=0.95
        Max allowed absolute correlation between selected **non-interaction** transformed
        features. If |corr| >= rho_threshold, they cannot be selected together.
    use_groups : bool, default=False
        Whether to enforce group sparsity constraints.
    groups : dict[str, list[tuple]] or None
        Group definition mapping a group name to a list of *transformed columns* to tie
        together. Items in the list can be provided as transformed column indices, or as
        tuples (orig_index, transform_name) to be resolved after building the design.
        Current mode is "all-or-nothing": all z's in a group are equal.
    M : Optional[float]
        Big‑M bound on |beta_i| when z_i=1. If None, computed heuristically from data scale.
    solver : str, default="ECOS_BB"
        CVXPY solver name supporting mixed‑integer problems.
    solver_kwargs : dict
        Extra arguments passed to solver.
    random_state : Optional[int]
        For potential solver randomness (not used by ECOS_BB).

    Attributes
    ----------
    coef_ : ndarray of shape (n_new_features,)
        Coefficients on transformed feature space.
    intercept_ : float
        Intercept term.
    feature_names_ : list of str
        Names for columns of the transformed design matrix.
    mapping_ : list of (orig_idx, transform_name)
        Back‑reference of each transformed column. Interactions are encoded as
        (i, f"*{j}") where transform_name starts with '*'.
    conflicts_ : list of (i,j)
        Index pairs constrained by collinearity rule (non-interactions only).
    scaler_ : StandardScaler or None
        Fitted scaler on original X if standardize=True.
    """

    def __init__(self,
                 k: int,
                 alpha1: float = 0.0,
                 alpha2: float = 0.0,
                 transforms: Optional[Iterable[str]] = None,
                 transform_menu: Optional[dict] = None,
                 standardize: bool = True,
                 fit_intercept: bool = True,
                 interactions: bool = False,
                 enforce_positive_main_when_interaction: bool = False,
                 rho_threshold: float = 0.95,
                 use_groups: bool = False,
                 groups: Optional[dict] = None,
                 M: Optional[float] = None,
                 solver: str = "ECOS_BB",
                 solver_kwargs: Optional[dict] = None,
                 random_state: Optional[int] = None):
        self.k = int(k)
        self.alpha1 = float(alpha1)
        self.alpha2 = float(alpha2)
        self.transforms = list(transforms) if transforms is not None else [
            "identity", "square", "cube", "abs", "signed_sqrt", "log1p_abs",
            "tanh", "sigmoid", "sin", "cos", "recip1p_abs"
        ]
        self.transform_menu = transform_menu
        self.standardize = bool(standardize)
        self.fit_intercept = bool(fit_intercept)
        self.interactions = bool(interactions)
        self.enforce_positive_main_when_interaction = bool(enforce_positive_main_when_interaction)
        self.rho_threshold = float(rho_threshold)
        self.use_groups = bool(use_groups)
        self.groups = groups
        self.M = M
        self.solver = solver
        self.solver_kwargs = solver_kwargs or {}
        self.random_state = random_state

    # ---- sklearn API ----
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None):
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)
        y = y.astype(float)

        # Standardize original X for numerical stability (optional)
        if self.standardize:
            self.scaler_ = StandardScaler(with_mean=True, with_std=True).fit(X)
            Xs = self.scaler_.transform(X)
        else:
            self.scaler_ = None
            Xs = X.copy()

        # 1) Build transformed features FIRST (per-feature menus)
        n, p = Xs.shape
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(p)]

        X_list = []
        names_t: List[str] = []
        mapping: List[Tuple[int, str]] = []
        # per-feature transforms
        for j in range(p):
            xj = Xs[:, j]
            # menu resolution uses helper in _build_transforms logic; inline here for speed
            if self.transform_menu:
                if feature_names[j] in self.transform_menu:
                    tlist = list(self.transform_menu[feature_names[j]])
                elif j in self.transform_menu:
                    tlist = list(self.transform_menu[j])
                else:
                    tlist = self.transforms
            else:
                tlist = self.transforms
            for tname in tlist:
                if tname not in TRANSFORM_FUNCS:
                    raise ValueError(f"Unknown transform: {tname}")
                col = TRANSFORM_FUNCS[tname](xj)
                if not np.all(np.isfinite(col)):
                    raise ValueError(f"Transform {tname} produced non-finite values on feature {feature_names[j]}")
                X_list.append(col.reshape(-1, 1))
                names_t.append(f"{feature_names[j]}|{tname}")
                mapping.append((j, tname))

        # Optional pairwise interactions of ORIGINAL features (identity only)
        interaction_pairs: List[Tuple[int, int]] = []
        if self.interactions and p >= 2:
            for i in range(p):
                for j in range(i + 1, p):
                    interaction_pairs.append((i, j))
                    col = (Xs[:, i] * Xs[:, j]).reshape(-1, 1)
                    X_list.append(col)
                    names_t.append(f"{feature_names[i]}*{feature_names[j]}")
                    # encode mapping as (i, f"*{j}") to distinguish
                    mapping.append((i, f"*{j}"))

        X_t = np.hstack(X_list) if X_list else np.empty((n, 0))
        self.feature_names_ = names_t
        self.mapping_ = mapping

        # 2) Compute conflicts based on correlations of transformed features (exclude interactions)
        is_interaction = np.array([m[1].startswith("*") if isinstance(m[1], str) else False for m in mapping])
        non_inter_idx = np.where(~is_interaction)[0]
        self.conflicts_ = _compute_conflicts(X_t[:, non_inter_idx], self.rho_threshold)
        # convert local indices back to global indices
        self.conflicts_ = [(int(non_inter_idx[i]), int(non_inter_idx[j])) for (i, j) in self.conflicts_]

        n, p_t = X_t.shape
        if self.k > p_t:
            raise ValueError(f"k={self.k} exceeds number of transformed features {p_t}")

        # Heuristic Big‑M if not provided
        if self.M is None:
            col_norms = np.linalg.norm(X_t, axis=0) + 1e-8
            M_vec = 10.0 * np.linalg.norm(y) / np.maximum(col_norms, 1e-8)
            self.M_ = float(np.max(M_vec))
        else:
            self.M_ = float(self.M)

        # CVXPY variables
        beta = cp.Variable(p_t)
        z = cp.Variable(p_t, boolean=True)

        if self.fit_intercept:
            b = cp.Variable()  # intercept
            residual = X_t @ beta + b - y
        else:
            b = 0  # No intercept
            residual = X_t @ beta - y

        # Objective
        obj = 0.5 * cp.sum_squares(residual)
        if self.alpha1 > 0:
            obj += self.alpha1 * cp.norm1(beta)
        if self.alpha2 > 0:
            obj += 0.5 * self.alpha2 * cp.sum_squares(beta)
        objective = cp.Minimize(obj)

        # Constraints
        cons = []
        M = self.M_
        eps = 1e-6
        cons += [beta <= M * z, beta >= -M * z]
        cons += [cp.sum(z) <= self.k]
        # Pairwise collinearity on non-interactions only
        for i, j in self.conflicts_:
            cons.append(z[i] + z[j] <= 1)
        # Interaction hierarchy and optional positivity on parents
        if self.interactions:
            # Build a quick index for main effects
            idx_by_pair: Dict[Tuple[int, int], int] = {}
            for idx, m in enumerate(mapping):
                if isinstance(m[1], str) and m[1].startswith('*'):
                    i = m[0]
                    j = int(m[1][1:])
                    idx_by_pair[(i, j)] = idx
            # For each interaction, tie to parents
            for (i, j), idx in idx_by_pair.items():
                # find columns for main effects of i and j with transform 'identity'
                try:
                    i_main = next(k for k, mm in enumerate(mapping) if mm == (i, 'identity'))
                    j_main = next(k for k, mm in enumerate(mapping) if mm == (j, 'identity'))
                except StopIteration:
                    # if identity wasn't built, fall back to first transform of each parent
                    i_main = next(k for k, mm in enumerate(mapping) if mm[0] == i and not (isinstance(mm[1], str) and mm[1].startswith('*')))
                    j_main = next(k for k, mm in enumerate(mapping) if mm[0] == j and not (isinstance(mm[1], str) and mm[1].startswith('*')))
                # z_interaction <= z_i and <= z_j (hierarchy)
                cons += [z[idx] <= z[i_main], z[idx] <= z[j_main]]
                # optional positivity for parents when interaction is selected
                if self.enforce_positive_main_when_interaction:
                    cons += [beta[i_main] >= eps * z[idx], beta[j_main] >= eps * z[idx]]

        # Group sparsity (all-or-nothing): make all z in a group equal
        if self.use_groups and self.groups:
            # resolve group column indices
            def resolve(item):
                if isinstance(item, int):
                    return item
                if isinstance(item, tuple):
                    # (orig_idx, tname) or (orig_idx, '*j')
                    for k, mm in enumerate(mapping):
                        if mm == item:
                            return k
                    raise ValueError(f"Group item {item} not found in mapping")
                raise ValueError("Group entries must be indices or (orig_idx, transform_name)")

            for gname, members in self.groups.items():
                idxs = [resolve(m) for m in members]
                if len(idxs) <= 1:
                    continue
                # enforce equality to the first member
                ref = idxs[0]
                for jj in idxs[1:]:
                    cons += [z[jj] == z[ref]]

        prob = cp.Problem(objective, cons)
        # Solve
        try:
            prob.solve(solver=self.solver, **self.solver_kwargs)
        except Exception as e:
            if self.solver != cp.ECOS_BB:
                prob.solve(solver=cp.ECOS_BB, **self.solver_kwargs)
            else:
                raise e

        if beta.value is None:
            raise RuntimeError("Optimization failed to produce a solution. Check solver availability or relax k/rho_threshold/M.")

        self.coef_ = np.asarray(beta.value).ravel()
        if self.fit_intercept:
            self.intercept_ = float(b.value)
        else:
            self.intercept_ = 0.0
        self.selected_mask_ = np.asarray(z.value > 0.5).ravel()
        self.n_features_in_ = X.shape[1]
        self.is_interaction_mask_ = is_interaction
        self.feature_names_ = names_t
        self.mapping_ = mapping
        return self

    def _transform_X(self, X: np.ndarray) -> np.ndarray:
        X = check_array(X, accept_sparse=False)
        if getattr(self, "scaler_", None) is not None:
            X = self.scaler_.transform(X)
        n, p = X.shape
        # rebuild transformed columns in stored order
        cols = []
        for j, tname in self.mapping_:
            if isinstance(tname, str) and tname.startswith('*'):
                j2 = int(tname[1:])
                cols.append((X[:, j] * X[:, j2]).reshape(-1, 1))
            else:
                cols.append(TRANSFORM_FUNCS[tname](X[:, j]).reshape(-1, 1))
        return np.hstack(cols) if cols else np.empty((X.shape[0], 0))

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xt = self._transform_X(X)
        return Xt @ self.coef_ + self.intercept_

    # Convenience getters
    def get_support(self, indices: bool = False):
        if indices:
            return np.where(self.selected_mask_)[0]
        return self.selected_mask_

    def transformed_feature_names(self) -> List[str]:
        return list(self.feature_names_)


# ---- Minimal usage example ----
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, p = 120, 5
    X = rng.normal(size=(n, p))
    true_beta = np.zeros(p)
    true_beta[0] = 2.0
    true_beta[3] = -1.2
    y = X @ true_beta + 0.3 * rng.normal(size=n)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


    # Example groups: tie together identity+square for feature 0
    groups = {
        "feat0_bundle": [(0, "identity"), (0, "square")]
    }

    l1_alpha = 0.01
    l2_alpha = 0.1
    fit_intercept_option = True  # Control for all models
    model = HolisticRegressionCVX(
        k=10,
        alpha1=l1_alpha,
        alpha2=l2_alpha,
        rho_threshold=0.8,
        fit_intercept=fit_intercept_option,
        interactions=True,
        enforce_positive_main_when_interaction=True,
        use_groups=False,
        groups=groups,
        solver=cp.ECOS_BB,
        standardize=True
    )

    # Linear Regression comparisson
    linear_model = LinearRegression(fit_intercept=fit_intercept_option)
    lasso_model = Lasso(alpha=l1_alpha, random_state=42, fit_intercept=fit_intercept_option)
    elastic_net_model = ElasticNet(alpha=2*l2_alpha + l1_alpha, l1_ratio=l1_alpha/(2*l1_alpha + l1_alpha), random_state=42, fit_intercept=fit_intercept_option)



    model.fit(X, y)
    linear_model.fit(X_train, y_train)
    lasso_model.fit(X_train, y_train)
    elastic_net_model.fit(X_train, y_train)


    # Predict on train and test sets
    models = {
        "Holistic Regression": model,
        "Linear Regression": linear_model,
        "Lasso": lasso_model,
        "Elastic Net": elastic_net_model,
    }



for name, model in models.items():

    if name == "Holistic Regression":
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
    else:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)

    # Compute metrics
    train_rmse = root_mean_squared_error(y_train, y_train_pred)
    test_rmse = root_mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Print results
    print(f"{name}:")
    print(f"  Train RMSE: {train_rmse:.4f}, Train R²: {train_r2:.4f}")
    print(f"  Test RMSE: {test_rmse:.4f}, Test R²: {test_r2:.4f}")
    print("-" * 50)

    
    if name == "Holistic Regression":
        print("Selected features:")
        names = np.array(model.transformed_feature_names())
        print(names[model.get_support()])

    else:
        print("Selected features:")
        betas = model.coef_
        print(np.where(betas>1e-6)[0])

    # yhat = model.predict(
    # print("Train RMSE:", float(np.sqrt(np.mean((yhat - y) ** 2))))

