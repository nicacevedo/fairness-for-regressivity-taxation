import numpy as np
import cvxpy as cp

def solve_optimal_binning_misocp(
    X,
    y,
    K,
    rho,
    m_min=2,
    lambda_width=0.0,
    width_max=None,
    standardize_X=True,
    solver_preference=None,
    verbose=False,
    time_limit=None,
    eps=1e-6,
):
    """
    Solve MISOCP for joint y-binning + X-clustering with outliers.
    
    Model (Option 1A):
      min sum_{i,k} r_{ik} + rho * sum_i o_i + lambda_width * sum_k (u_k - ell_k)
      s.t. assignment, min cluster size, y-interval membership,
           ordered bins, SOC distance-to-center activation,
           r=0 if not assigned.
    
    Parameters
    ----------
    X : (n, p) array
    y : (n,) array
    K : int
    rho : float
    m_min : int
    lambda_width : float (optional)
        Penalize bin widths to discourage overly wide y-intervals.
    width_max : float or None
        Optional hard cap on bin width.
    standardize_X : bool
        Recommended. Euclidean distances are scale-sensitive.
    solver_preference : list[str] or None
        Example: ["MOSEK", "GUROBI", "CPLEX", "SCIP"].
        If None, uses a sensible default order.
    verbose : bool
    time_limit : float or None
        Seconds. Passed to solver if supported.
    
    Returns
    -------
    labels : (n,) int array
        Cluster index 0..K-1, or -1 for outliers.
    ell : (K,) float array
    u : (K,) float array
    details : dict
        Contains raw z, o, mu, r, status, objective.
    """
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    n, p = X.shape
    
    if K <= 0:
        raise ValueError("K must be positive.")
    if m_min < 1:
        raise ValueError("m_min must be >= 1.")
    if K * m_min > n:
        raise ValueError("Infeasible likely: K * m_min > n.")
    
    # Optionally standardize X for geometry fairness
    X_used = X.copy().astype(float)
    X_mean = np.zeros(p)
    X_std = np.ones(p)
    if standardize_X:
        X_mean = X_used.mean(axis=0)
        X_std = X_used.std(axis=0)
        X_std[X_std == 0] = 1.0
        X_used = (X_used - X_mean) / X_std
    
    # y bounds and big-M for y-membership
    L = float(np.min(y)) - 2*eps # eps correction for the tolerance on bins
    U = float(np.max(y)) + 2*eps # eps correction for the tolerance on bins
    M_y = U - L if U > L else 1.0
    
    # X bounds and big-M for SOC activation
    x_low = X_used.min(axis=0)
    x_high = X_used.max(axis=0)
    # A safe bound on max possible distance in the (scaled) box
    M_x = float(np.linalg.norm(x_high - x_low))
    if M_x == 0:
        M_x = 1.0
    
    # Decision variables
    z = cp.Variable((n, K), boolean=True)  # assignment
    o = cp.Variable(n, boolean=True)       # outlier flags
    
    ell = cp.Variable(K)
    u = cp.Variable(K)
    
    mu = cp.Variable((K, p))               # centers in X space
    r = cp.Variable((n, K), nonneg=True)   # distance proxies
    
    constraints = []
    
    # 1) Assignment with outliers
    constraints += [cp.sum(z, axis=1) + o == 1]
    
    # 2) Minimum cluster size
    constraints += [cp.sum(z[:, k]) >= m_min for k in range(K)]
    
    # 3) y-membership activation
    # If z[i,k]=1 -> ell_k <= y_i <= u_k
    for k in range(K):
        for i in range(n):
            constraints += [ # Testing gaps
                # y[i] - eps >= ell[k] - M_y * (1 - z[i, k]),
                # y[i] + eps <= u[k]   + M_y * (1 - z[i, k]),
                y[i] + eps >= ell[k] - M_y * (1 - z[i, k]),
                y[i] - eps <= u[k]   + M_y * (1 - z[i, k]),
            ]
    
    # 4) Bin validity + ordering
    constraints += [
        ell >= L,
        u <= U,
        # (U - L)/(10*K) <= u - ell, # Minimum width? Or not
        ell <= u,
    ]
    for k in range(K - 1):
        constraints += [u[k] <= ell[k + 1]]
    
    if width_max is not None:
        constraints += [u - ell <= float(width_max)]
    
    # 5) Center bounds (tightens SOC)
    for k in range(K):
        constraints += [
            mu[k, :] >= x_low,
            mu[k, :] <= x_high,
        ]
    
    # 6) SOC distance activation
    # ||x_i - mu_k|| <= r_ik + M_x(1 - z_ik)
    # and r_ik <= M_x z_ik
    for k in range(K):
        for i in range(n):
            constraints += [
                cp.norm(X_used[i, :] - mu[k, :], 2) <= r[i, k] + M_x * (1 - z[i, k]),
                r[i, k] <= M_x * z[i, k],
            ]
    
    # Objective
    obj = cp.sum(r) + float(rho) * cp.sum(o)
    if lambda_width and lambda_width > 0:
        obj += float(lambda_width) * cp.sum(u - ell)
    
    problem = cp.Problem(cp.Minimize(obj), constraints)
    
    # Solver selection
    installed = set(cp.installed_solvers())
    if solver_preference is None:
        solver_preference = ["MOSEK", "GUROBI", "CPLEX", "SCIP"]
    
    chosen = None
    for s in solver_preference:
        if s in installed:
            chosen = s
            break
    
    if chosen is None:
        raise RuntimeError(
            "No suitable MISOCP-capable solver found. "
            "Install MOSEK, GUROBI, or CPLEX for best results."
        )
    
    # Solver kwargs (best-effort)
    solve_kwargs = {"solver": chosen, "verbose": verbose}
    if time_limit is not None:
        # Different solvers use different parameter names.
        # We pass common ones in a safe way.
        if chosen == "GUROBI":
            solve_kwargs["TimeLimit"] = float(time_limit)
        elif chosen == "CPLEX":
            solve_kwargs["timelimit"] = float(time_limit)
        elif chosen == "MOSEK":
            solve_kwargs["mosek_params"] = {"MSK_DPAR_OPTIMIZER_MAX_TIME": float(time_limit)}
        # SCIP time limit parameter support may vary in CVXPY bindings.

    print("SOLVER CHOSE: ", chosen)
    
    result = problem.solve(**solve_kwargs)
    
    status = problem.status
    
    # Extract solution
    z_val = None if z.value is None else np.asarray(z.value)
    o_val = None if o.value is None else np.asarray(o.value).reshape(-1)
    ell_val = None if ell.value is None else np.asarray(ell.value).reshape(-1)
    u_val = None if u.value is None else np.asarray(u.value).reshape(-1)
    mu_val = None if mu.value is None else np.asarray(mu.value)
    r_val = None if r.value is None else np.asarray(r.value)

    # print("SUM OF DISTANCES: ", )
    
    if z_val is None or o_val is None or ell_val is None or u_val is None:
        raise RuntimeError(f"Solver did not return a complete solution. Status: {status}")
    
    # Hard rounding for numerical tolerance
    z_bin = (z_val > 0.5).astype(int)
    o_bin = (o_val > 0.5).astype(int)
    
    labels = -np.ones(n, dtype=int)
    for i in range(n):
        if o_bin[i] == 1:
            labels[i] = -1
        else:
            # pick the assigned cluster
            ks = np.where(z_bin[i, :] == 1)[0]
            if len(ks) == 1:
                labels[i] = int(ks[0])
            else:
                # fallback: argmax in case of tiny solver tolerances
                labels[i] = int(np.argmax(z_val[i, :]))
    
    details = {
        "status": status,
        "objective": result,
        "z": z_val,
        "o": o_val,
        "ell": ell_val,
        "u": u_val,
        "mu": mu_val,
        "r": r_val,
        "X_used": X_used,
        "X_mean": X_mean,
        "X_std": X_std,
        "L": L,
        "U": U,
        "M_x": M_x,
        "M_y": M_y,
        "solver": chosen,
    }
    
    return labels, ell_val, u_val, details, np.sum(r_val)





import matplotlib.pyplot as plt

def plot_binning_results(
    X,
    y,
    labels,
    ell,
    u,
    show_pca=True,
    standardize_for_pca=True,
    title_prefix="",
):
    """
    Visualize:
    1) y vs index colored by cluster with horizontal bands for bins
    2) optional PCA(2D) scatter colored by cluster
    
    Parameters
    ----------
    X : (n,p)
    y : (n,)
    labels : (n,) with -1 for outliers
    ell, u : (K,)
    show_pca : bool
    standardize_for_pca : bool
    """
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    ell = np.asarray(ell).reshape(-1)
    u = np.asarray(u).reshape(-1)
    n = len(y)
    K = len(ell)
    
    # Color mapping
    # Use matplotlib's default cycle by indexing
    unique_labels = sorted(set(labels.tolist()))
    
    def label_color(lab):
        if lab == -1:
            return "gray"
        return f"C{lab % 10}"
    
    colors = [label_color(l) for l in labels]
    
    # 1) y vs index
    plt.figure(figsize=(10, 4))
    plt.scatter(np.arange(n), y, c=colors, s=18, alpha=0.8)
    
    # draw bin bands as horizontal lines
    for k in range(K):
        plt.axhline(ell[k], linestyle="--", linewidth=1)
        plt.axhline(u[k], linestyle="--", linewidth=1)
        # light annotation
        plt.text(
            0,
            (ell[k] + u[k]) / 2,
            f"bin {k}",
            fontsize=9,
            verticalalignment="center",
        )
    
    plt.title(f"{title_prefix}y with learned bins")
    plt.xlabel("data index")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig("./temp/plots/literal_trash/binninb_clustering.png", dpi=600)
    # plt.show()
    
    # 2) PCA scatter
    if show_pca:
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            print("scikit-learn not available; skipping PCA plot.")
            return
        
        Xp = X.astype(float)
        if standardize_for_pca:
            Xp = StandardScaler().fit_transform(Xp)
        
        Z = PCA(n_components=2).fit_transform(Xp)
        
        plt.figure(figsize=(6, 5))
        plt.scatter(Z[:, 0], Z[:, 1], c=colors, s=20, alpha=0.8)
        plt.title(f"{title_prefix}PCA view of X colored by bins")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        plt.savefig("./temp/plots/literal_trash/binninb_clustering_pca.png", dpi=600)
        # plt.show()


def validate_solution(y, labels, ell, u):
    y = np.asarray(y).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    ell = np.asarray(ell).reshape(-1)
    u = np.asarray(u).reshape(-1)
    K = len(ell)
    
    ok = True
    # ordering
    if not np.all(ell <= u):
        print("Invalid: some ell_k > u_k")
        ok = False
    if K > 1 and not np.all(u[:-1] <= ell[1:]):
        print("Warning: bins overlap or are not ordered strictly.")
    
    # membership
    for i, lab in enumerate(labels):
        if lab == -1:
            continue
        if not (ell[lab] - 1e-6 <= y[i] <= u[lab] + 1e-6):
            print(f"Mismatch: point {i} in bin {lab} but y={y[i]:.4f} outside [{ell[lab]:.4f},{u[lab]:.4f}]")
            ok = False
    
    return ok





###########################
# CUTTING PLANE???

import numpy as np
import cvxpy as cp


def solve_optimal_binning_cutting_plane(
    X,
    y,
    K,
    rho,
    m_min=2,
    lambda_width=0.0,
    width_max=None,
    standardize_X=True,
    solver_master_preference=None,
    solver_socp_preference=None,
    max_iters=50,
    max_new_cuts_per_iter=5000,
    feas_tol=1e-4,
    verbose=False,
    time_limit_master=None,
):
    """
    Cutting-plane / outer-approximation routine for the MISOCP binning+clustering model.

    Master: MILP with linear constraints + accumulated SOC tangent cuts.
    Separation: check SOC violations for assigned (i,k); add cuts and repeat.

    Returns
    -------
    labels : (n,) int array
        0..K-1 for bins, -1 for outliers.
    ell : (K,) float array
    u : (K,) float array
    details : dict
        status, objective, solver used, cuts count, z/o/mu/r, etc.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    n, p = X.shape

    if K <= 0:
        raise ValueError("K must be positive.")
    if m_min < 1:
        raise ValueError("m_min must be >= 1.")
    if K * m_min > n:
        raise ValueError("Likely infeasible: K * m_min > n.")

    # ---------- Standardize X (recommended) ----------
    X_used = X.copy()
    X_mean = np.zeros(p)
    X_std = np.ones(p)
    if standardize_X:
        X_mean = X_used.mean(axis=0)
        X_std = X_used.std(axis=0)
        X_std[X_std == 0] = 1.0
        X_used = (X_used - X_mean) / X_std

    # ---------- Big-M constants ----------
    L = float(np.min(y)) - 2*feas_tol
    U = float(np.max(y)) + 2*feas_tol
    M_y = U - L if U > L else 1.0

    x_low = X_used.min(axis=0)
    x_high = X_used.max(axis=0)
    M_x = float(np.linalg.norm(x_high - x_low))
    if M_x == 0:
        M_x = 1.0

    # ---------- Solver selection ----------
    installed = set(cp.installed_solvers())

    if solver_master_preference is None:
        # master is MILP
        solver_master_preference = ["GUROBI", "CPLEX", "SCIP", "GLPK_MI", "CBC", "ECOS_BB"]
    if solver_socp_preference is None:
        # continuous SOCP check (optional); not strictly required in this routine
        solver_socp_preference = ["MOSEK", "GUROBI", "CPLEX", "ECOS"]

    master_solver = next((s for s in solver_master_preference if s in installed), None)
    if master_solver is None:
        raise RuntimeError("No MILP solver found for the master. Install GUROBI/CPLEX/SCIP/GLPK_MI.")

    # We'll do SOC feasibility checking ourselves via numpy norms,
    # so we don't strictly need to solve a SOCP subproblem here.

    # ---------- Store cuts ----------
    # Each cut is a dict with:
    # i, k, alpha (p,), c (scalar) such that:
    # alpha @ mu[k] - r[i,k] + c <= M_x*(1 - z[i,k])
    cuts = []

    # ---------- Helper: build and solve master ----------
    def solve_master(cuts_list):
        z = cp.Variable((n, K), boolean=True)
        o = cp.Variable(n, boolean=True)
        ell = cp.Variable(K)
        u = cp.Variable(K)
        mu = cp.Variable((K, p))
        r = cp.Variable((n, K), nonneg=True)

        cons = []

        # Assignment with outliers
        cons += [cp.sum(z, axis=1) + o == 1]

        # Min cluster size
        cons += [cp.sum(z[:, k]) >= m_min for k in range(K)]

        # y-membership activation (vectorized per k)
        for k in range(K):
            cons += [
                y - feas_tol >= ell[k] - M_y * (1 - z[:, k]),
                y + feas_tol <= u[k]   + M_y * (1 - z[:, k]),
            ]

        # Bin validity + ordering
        cons += [
            ell >= L,
            u <= U,
            ell <= u,
        ]
        for k in range(K - 1):
            cons += [u[k] <= ell[k + 1]]

        if width_max is not None:
            cons += [u - ell <= float(width_max)]

        # Center bounds
        cons += [mu >= x_low, mu <= x_high]

        # Activation for r
        cons += [r <= M_x * z]

        # Add accumulated linear SOC cuts
        for cut in cuts_list:
            i = cut["i"]
            k = cut["k"]
            alpha = cut["alpha"]
            c = cut["c"]
            cons += [
                alpha @ mu[k, :] - r[i, k] + c <= M_x * (1 - z[i, k])
            ]

        # Objective
        obj = cp.sum(r) + float(rho) * cp.sum(o)
        if lambda_width and lambda_width > 0:
            obj += float(lambda_width) * cp.sum(u - ell)

        prob = cp.Problem(cp.Minimize(obj), cons)

        solve_kwargs = {"solver": master_solver, "verbose": verbose}
        if time_limit_master is not None:
            if master_solver == "GUROBI":
                solve_kwargs["TimeLimit"] = float(time_limit_master)
            elif master_solver == "CPLEX":
                solve_kwargs["timelimit"] = float(time_limit_master)
            # SCIP/GLPK_MI handling varies; omit for safety.

        val = prob.solve(**solve_kwargs)

        return {
            "problem": prob,
            "objective": val,
            "status": prob.status,
            "z": None if z.value is None else np.asarray(z.value),
            "o": None if o.value is None else np.asarray(o.value).reshape(-1),
            "ell": None if ell.value is None else np.asarray(ell.value).reshape(-1),
            "u": None if u.value is None else np.asarray(u.value).reshape(-1),
            "mu": None if mu.value is None else np.asarray(mu.value),
            "r": None if r.value is None else np.asarray(r.value),
        }

    # ---------- Helper: generate SOC tangent cut at (i,k) ----------
    def make_cut(i, k, mu_star):
        x_i = X_used[i, :]
        diff = mu_star - x_i
        norm_star = float(np.linalg.norm(diff))

        if norm_star <= 1e-12:
            alpha = np.zeros(p)
        else:
            alpha = diff / norm_star

        # Cut: ||x - mu*|| + alpha^T (mu - mu*) - r <= M_x (1 - z)
        # Rearranged to alpha^T mu - r + c <= M_x (1 - z)
        # where c = ||x - mu*|| - alpha^T mu*
        c = norm_star - float(alpha @ mu_star)

        return {"i": i, "k": k, "alpha": alpha, "c": c}

    # ---------- Main loop ----------
    best_feasible = None
    best_feasible_obj = np.inf

    last_master = None

    for it in range(1, max_iters + 1):
        last_master = solve_master(cuts)

        status = last_master["status"]
        if status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"Master not solved to optimality. Status: {status}")

        z_val = last_master["z"]
        o_val = last_master["o"]
        ell_val = last_master["ell"]
        u_val = last_master["u"]
        mu_val = last_master["mu"]
        r_val = last_master["r"]

        if any(v is None for v in [z_val, o_val, ell_val, u_val, mu_val, r_val]):
            raise RuntimeError("Incomplete master solution returned.")

        z_bin = (z_val > 0.5).astype(int)
        o_bin = (o_val > 0.5).astype(int)

        # ---- Check SOC violations for assigned pairs (i,k) ----
        new_cuts = []
        violations = 0

        # We only need to check pairs with z=1
        assigned_pairs = np.argwhere(z_bin == 1)

        for idx, (i, k) in enumerate(assigned_pairs):
            # If outlier flagged too (numerical), skip
            if o_bin[i] == 1:
                continue

            dist = float(np.linalg.norm(X_used[i, :] - mu_val[k, :]))
            rhs = float(r_val[i, k])  # because z=1 => SOC is dist <= r

            if dist > rhs + feas_tol:
                violations += 1
                new_cuts.append(make_cut(int(i), int(k), mu_val[k, :]))

                if len(new_cuts) >= max_new_cuts_per_iter:
                    break

        if verbose:
            print(f"[iter {it}] master obj={last_master['objective']:.6g} "
                  f"assigned_pairs={len(assigned_pairs)} new_violations={violations} "
                  f"cuts_total={len(cuts)}")

        # If no violations, we have an SOC-feasible solution for the original MISOCP
        if violations == 0:
            # Build labels
            labels = -np.ones(n, dtype=int)
            for i in range(n):
                if o_bin[i] == 1:
                    labels[i] = -1
                else:
                    ks = np.where(z_bin[i, :] == 1)[0]
                    labels[i] = int(ks[0]) if len(ks) else int(np.argmax(z_val[i, :]))

            best_feasible = (labels, ell_val, u_val)
            best_feasible_obj = float(last_master["objective"])
            break

        # Add cuts and continue
        cuts.extend(new_cuts)

    if best_feasible is None:
        # Return best available (even if not SOC-feasible)
        # This is honest: we didn't reach feasibility within iteration budget.
        labels = -np.ones(n, dtype=int)
        if last_master is not None and last_master["z"] is not None:
            z_val = last_master["z"]
            o_val = last_master["o"]
            z_bin = (z_val > 0.5).astype(int)
            o_bin = (o_val > 0.5).astype(int)
            for i in range(n):
                if o_bin[i] == 1:
                    labels[i] = -1
                else:
                    ks = np.where(z_bin[i, :] == 1)[0]
                    labels[i] = int(ks[0]) if len(ks) else int(np.argmax(z_val[i, :]))

        details = {
            "status": "max_iters_reached",
            "objective": None if last_master is None else last_master["objective"],
            "solver_master": master_solver,
            "cuts_count": len(cuts),
            "last_master": last_master,
            "L": L, "U": U, "M_x": M_x, "M_y": M_y,
            "X_used": X_used, "X_mean": X_mean, "X_std": X_std,
        }
        return labels, (None if last_master is None else last_master["ell"]), \
               (None if last_master is None else last_master["u"]), details

    labels, ell_val, u_val = best_feasible

    details = {
        "status": "soc_feasible_solution_found",
        "objective": best_feasible_obj,
        "solver_master": master_solver,
        "cuts_count": len(cuts),
        "last_master": last_master,
        "L": L, "U": U, "M_x": M_x, "M_y": M_y,
        "X_used": X_used, "X_mean": X_mean, "X_std": X_std,
    }

    return labels, ell_val, u_val, details, np.sum(r_val)


import matplotlib.pyplot as plt

def plot_binning_results_cutting_planes(
    X,
    y,
    labels,
    ell,
    u,
    show_pca=True,
    standardize_for_pca=True,
    title_prefix="",
):
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    labels = np.asarray(labels).reshape(-1)

    if ell is None or u is None:
        print("No bin limits to plot.")
        return

    ell = np.asarray(ell).reshape(-1)
    u = np.asarray(u).reshape(-1)
    n = len(y)
    K = len(ell)

    def label_color(lab):
        if lab == -1:
            return "gray"
        return f"C{lab % 10}"

    colors = [label_color(l) for l in labels]

    # y vs index
    plt.figure(figsize=(10, 4))
    plt.scatter(np.arange(n), y, c=colors, s=18, alpha=0.8)

    for k in range(K):
        plt.axhline(ell[k], linestyle="--", linewidth=1)
        plt.axhline(u[k], linestyle="--", linewidth=1)
        plt.text(
            0, (ell[k] + u[k]) / 2,
            f"bin {k}",
            fontsize=9,
            verticalalignment="center",
        )

    plt.title(f"{title_prefix}y with learned bins")
    plt.xlabel("data index")
    plt.ylabel("y")
    plt.tight_layout()
    plt.show()

    # PCA view
    if show_pca:
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            print("scikit-learn not available; skipping PCA plot.")
            return

        Xp = X.astype(float)
        if standardize_for_pca:
            Xp = StandardScaler().fit_transform(Xp)

        Z = PCA(n_components=2).fit_transform(Xp)

        plt.figure(figsize=(6, 5))
        plt.scatter(Z[:, 0], Z[:, 1], c=colors, s=20, alpha=0.8)
        plt.title(f"{title_prefix}PCA of X colored by bins")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        plt.show()
