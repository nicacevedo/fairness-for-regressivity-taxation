import cvxpy as cp
import gurobipy as gp
import mosek
import numpy as np

from time import time

from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import root_mean_squared_error, r2_score

class LeastAbsoluteDeviationRegression:
    def __init__(self, fit_intercept=True, solver="GUROBI", solve_dual=False):
        self.beta = None
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.solve_dual = solve_dual

    def fit(self, X, y):
        try:
            X, y = X.to_numpy(), y.to_numpy()
        except Exception as e:
            pass
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        n, m = X.shape

        if self.solve_dual:

            # CVXPY DUAL APPROACH of the LADReg (More efficient)
            theta = cp.Variable(n)
            constraints = [
                theta >= -1,
                theta <= 1
            ]
            beta = [X.T @ theta == 0] # Constraint from where to get the primal betas: L_d = y't + beta'(X't) + u'(t-1) + l'(-t-1)
            constraints += beta

            # Objective: <=> Minimize the overall Mean Absolute Error
            dual_prob = cp.Problem(
                cp.Maximize(y @ theta), 
                constraints
            )

            # Solve the optimization problem
            try:
                result = dual_prob.solve(solver=self.solver, verbose=False)
            except cp.error.SolverError:
                print("GUROBI not available, trying default solver.")
                result = dual_prob.solve(verbose=False)

            print(f"Problem status: {dual_prob.status}")
            print(f"Optimal objective (Mean Absolute Error): {result}")

            # Print the difference in MAE between groups post-optimization
            if dual_prob.status in ["optimal", "optimal_inaccurate"]:
                self.beta = beta[0].dual_value
                solve_time = dual_prob.solver_stats.solve_time
            else:
                print("Solver did not find an optimal solution. Beta coefficients not set.")
                self.beta = np.zeros(m) # Fallback beta

        else:
            # Primal approach of the problem
            beta = cp.Variable(m)
            u = cp.Variable(n, nonneg=True)
            l = cp.Variable(n, nonneg=True)
            constraints = [
                X @ beta + u - l == y
            ]

            # Objective: <=> Minimize the overall Mean Absolute Error
            e_n = np.ones(n)
            primal_prob = cp.Problem(
                cp.Minimize(e_n @ (u + l)), 
                constraints
            )

            # Solve the optimization problem
            try:
                result = primal_prob.solve(solver=self.solver, verbose=False)
            except cp.error.SolverError:
                print("GUROBI not available, trying default solver.")
                result = primal_prob.solve(verbose=False)

            print(f"Problem status: {primal_prob.status}")
            print(f"Optimal objective (Mean Absolute Error): {result}")

            # Print the difference in MAE between groups post-optimization
            if primal_prob.status in ["optimal", "optimal_inaccurate"]:
                self.beta = beta.value
                solve_time = primal_prob.solver_stats.solve_time
            else:
                print("Solver did not find an optimal solution. Beta coefficients not set.")
                self.beta = np.zeros(m) # Fallback beta

        return result, solve_time

    def predict(self, X):
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        return X @ self.beta
    
    def __str__(self):
        return f"LeastAbsoluteDeviationRegression(fit_intercept={self.fit_intercept})"





class StableRegression:
    def __init__(self, fit_intercept=True, solver="GUROBI", keep=0.7, lambda_l1=0, lambda_l2=0,
                 objective="mae", sensitive_idx=None, 
                 fit_group_intercept=False, delta_l2=0, 
                 fit_group_beta=False, group_beta_l2=0,
                 cov_constraint=False, eps_cov=0, sensitive_feature=None,
                 var_constraint=False, eps_var=0,
                 weight_by_group=False,
                 residual_cov_constraint=False, residual_cov_thresh=0,
                 ):
        self.beta = None
        self.intercept = None
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.keep = keep
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.objective = objective

        # Fairness / Shift
        self.sensitive_idx = sensitive_idx
        self.fit_group_intercept = fit_group_intercept#shift
        self.delta_l2 = delta_l2

        # Beta shift
        self.fit_group_beta = fit_group_beta
        self.group_beta_l2 = group_beta_l2

        # Alternative: Impose direct fairness constraint of max
        self.cov_constraint = cov_constraint
        self.eps_cov = eps_cov
        self.var_constraint = var_constraint
        self.eps_var = eps_var

        # Alternative: use weights by group and don't constraint the groups
        self.weight_by_group = weight_by_group

        # The Real Metric: Correlation wrt sensitive feature
        self.sensitive_feature = sensitive_feature

        # Constraint on the correlation of the residuals
        self.residual_cov_constraint = residual_cov_constraint
        self.residual_cov_thresh = residual_cov_thresh

    def fit(self, X, y):
        try:
            X, y = X.to_numpy(), y.to_numpy()
        except Exception as e:
            pass
        # if self.fit_intercept:
        #     X = np.hstack((np.ones((X.shape[0], 1)), X))
        n, m = X.shape

        if self.sensitive_idx is not None:
            n_groups = len(self.sensitive_idx)
            if self.weight_by_group:
                k_samples = [len(g_idx) * self.keep for g_idx in self.sensitive_idx]
                k_samples = [min(k_samples) for g_idx in self.sensitive_idx]
            else:
                k_samples = n * self.keep#int(n * self.keep) 
        else:
            # k_samples 
            # n_groups = 1
            k_samples = n * self.keep#int(n * self.keep) 

        # Primal approach of the problem
        beta = cp.Variable(m)
        intercept = cp.Variable()
        z = cp.Variable(m) # absolute of beta:            
        nu = cp.Variable(1) #if self.weight_by_group else nu = cp.Variable(1)
        theta = cp.Variable(n, nonneg=True)
        # if self.fit_group_intercept:
        #     delta = cp.Variable(n_groups)
        # if self.fit_group_beta:
        #     group_beta = cp.Variable((m, n_groups))
        
        # if self.cov_constraint:
        #     min_risk = cp.Variable()
        #     U = cp.Variable()
        

        # Regularizer constraints
        constraints =[
            beta <= z,
            -beta <= z,
             y - (X @ beta + intercept ) <= nu + theta,
            -y + (X @ beta + intercept ) <= nu + theta, 
        ]

        # Group constraints
        if self.cov_constraint:
            # d = self.sensitive_feature
            d = y
            f_X =  X @ beta + intercept
            constraints += [
                # cp.mean( nu[g] + theta[g_idx] ) <= U,
                #  cp.mean( ( d - np.mean(d) ) * ( f_X - cp.mean( f_X ) ) ) <= self.eps_cov * np.std(d) * np.std(y),
                # -cp.mean( ( d - np.mean(d) ) * ( f_X - cp.mean( f_X ) ) ) <= self.eps_cov * np.std(d) * np.std(y),
                 ( ( d - np.mean(d)) @ (f_X/y - cp.mean( f_X/y ) ) ) / n <= self.eps_cov * np.std(d), #* np.std(y),
                -( ( d - np.mean(d)) @ (f_X/y - cp.mean( f_X/y ) ) ) / n <= self.eps_cov * np.std(d), #* np.std(y),
            ]
        if self.var_constraint:
            f_X =  X @ beta + intercept
            mean_ratio = cp.mean(f_X / y)
            u = cp.Variable(n, nonneg=True)
            l = cp.Variable(n, nonneg=True)
            constraints += [
                f_X/y - mean_ratio == u - l,
                cp.mean( u + l ) <= self.eps_var, #* mean_ratio, 
            ]

            # self.residual_cov_thresh = residual_cov_thresh
        # Objective: <=> Minimize the overall Mean Absolute Error
        if self.weight_by_group:
            obj = cp.mean( [ nu[g] * float(k_samples[g]) + cp.sum(theta[g_idx])  for g, g_idx in enumerate(self.sensitive_idx) ] )
        else: 
            obj = ( nu * k_samples + cp.sum(theta) ) / k_samples
        if self.lambda_l1 > 0:
            obj += self.lambda_l1 * cp.sum(z) 
        if self.lambda_l2 > 0:
            obj += 0.5 * self.lambda_l2 * cp.norm2(beta)
        # if self.delta_l2 > 0:
        #     obj += self.delta_l2 * cp.norm2(delta)
        # if self.group_beta_l2 > 0:
        #     obj += self.group_beta_l2 * cp.norm2(group_beta)
        # if self.cov_constraint:
        #     obj += self.eps_cov * U
        primal_prob = cp.Problem(
            cp.Minimize( obj ), 
            constraints
        )

        # Solve the optimization problem
        # t0 = time()
        try:
            result = primal_prob.solve(solver=self.solver, verbose=False)
        except cp.error.SolverError:
            print("GUROBI not available, trying default solver.")
            result = primal_prob.solve(verbose=False)
        # solve_time = time() - t0
        print(f"Problem status: {primal_prob.status}")
        print(f"Optimal objective (Weighted Mean Absolute Error): {result}")
        # print(f"Solving time: {solve_time}")
        print(f"Selected betas: {np.sum(np.abs(beta.value) >= 1e-4)}")
        # if self.fit_group_intercept:
            # print(f"Shift delta: ", delta.value)
        print(f"Nu dual: ", nu.value)

        # Print the difference in MAE between groups post-optimization
        if primal_prob.status in ["optimal", "optimal_inaccurate"]:
            # print("THE MEAN COV?:", ( ( d - np.mean(d)) @ (f_X/d - cp.mean( f_X/d ) ) ).value / n)
            # print("RHS of COV: ", self.eps_cov * np.std(d))
            # print(" THE MEAN COV:", cp.mean( ( d - np.mean(d) ) * (cp.multiply(f_X, 1/d) - cp.mean( cp.multiply(f_X, 1/d) ) ) ).value)
            self.beta = beta.value
            self.intercept = intercept.value
        else:
            print("Solver did not find an optimal solution. Beta coefficients not set.")
            self.beta = np.zeros(m) # Fallback beta
            self.intercept = 0

        return result#, solve_time

    def predict(self, X):
        return X @ self.beta + self.intercept
    
    def __str__(self):
        if self.cov_constraint or self.var_constraint:
            return f"StableConstrained(eps_cov={self.eps_cov}, eps_var={self.eps_var})" # b0={int(self.fit_intercept)}
        else:
            return f"StableRegression"


import numpy as np
import cvxpy as cp
from sklearn.base import BaseEstimator, RegressorMixin


# class StableCovarianceUpperBoundLADRegressor(BaseEstimator, RegressorMixin):
class StableAdversarialSurrogateRegressor(BaseEstimator, RegressorMixin):
    """
    Stable (top-K) LAD regression with a *separable upper-bound* covariance penalty,
    eliminating the inner maximization.

    You (intentionally) use the separable upper bound:
        | sum_i w_i z_i | <= sum_i w_i |z_i|
    with z_i = (d_i - dbar) * (f_i - fbar),  f_i = x_i^T beta + b0,  fbar = (1/n) sum_j f_j.

    Stable/top-K adversary:
        max_{w} sum_i w_i s_i
        s.t. 0 <= w_i <= 1,  sum_i w_i = K
    which equals the sum of the K largest s_i.

    Here s_i(beta) = |y_i - f_i| + rho * | (d_i - dbar) * (f_i - fbar) |.

    Dual (single minimization):
        min_{beta,b0,nu,theta,...}  K*nu + sum_i theta_i + l1||beta||_1 + 0.5 l2||beta||_2^2
        s.t. s_i(beta) <= nu + theta_i,   theta_i >= 0,
             and epigraph linearisations of absolute values.

    Parameters
    ----------
    K : int or None
        Number of points in the stable/top-K objective (sum of K worst per-sample scores).
        If None, K is set from `keep` at fit time.
    keep : float
        Fraction in (0,1] used when K is None: K = ceil(keep * n_samples).
    rho : float
        Weight on the separable covariance-contribution penalty.
    l1, l2 : float
        L1 and L2 weights on coefficients (intercept not regularized).
    fit_intercept : bool
        Include an intercept b0.
    solver : str
        Default "MOSEK". Any solver supported by CVXPY for LP/QP.
    solver_opts : dict or None
        Passed to problem.solve(...).
    """

    def __init__(
        self,
        K=None,
        keep=0.1,
        rho_cov=1.0,
        rho_var=1.0,
        l1=0.0,
        l2=0.0,
        fit_intercept=True,
        solver="MOSEK",
        solver_opts=None,
        verbose=False,
        warm_start=False,
        neg_corr_focus=False,
    ):
        self.K = K
        self.keep = keep
        self.rho_cov = rho_cov
        self.rho_var = rho_var
        self.l1 = l1
        self.l2 = l2
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.solver_opts = solver_opts
        self.verbose = verbose
        self.warm_start = warm_start

        # learned
        self.coef_ = None
        self.intercept_ = 0.0
        self.status_ = None
        self.objective_value_ = None
        self.K_ = None
        self._last_scores_ = None
        self._last_nu_ = None

        # mine
        self.neg_corr_focus = neg_corr_focus

    @staticmethod
    def _as_2d_float(X):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2D array-like.")
        return X.astype(float)

    @staticmethod
    def _as_1d_float(x, name):
        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError(f"{name} must be 1D array-like.")
        return x.astype(float)

    def fit(self, X, y, d=None):
        X = self._as_2d_float(X)
        y = self._as_1d_float(y, "y")
        # if d is not None:
        d = self._as_1d_float(d, "d") if d is not None else y

        n, p = X.shape
        if y.shape[0] != n or d.shape[0] != n:
            raise ValueError("X, y, d must have the same number of samples.")

        # choose K
        if self.K is None:
            keep = float(self.keep)
            if not (0.0 < keep <= 1.0):
                raise ValueError("keep must be in (0,1].")
            K = int(np.ceil(keep * n))
        else:
            K = int(self.K)
        if not (1 <= K <= n):
            raise ValueError("K must satisfy 1 <= K <= n.")
        self.K_ = K

        rho_cov, rho_var = float(self.rho_cov), float(self.rho_var)
        if rho_cov < 0 or rho_var < 0:
            raise ValueError("ALL rho weights must be >= 0.")

        # centered d and mean prediction
        dbar = float(np.mean(d))
        d_center = d - dbar  # constants

        # variables
        beta = cp.Variable(p)
        b0 = cp.Variable() if self.fit_intercept else None

        nu = cp.Variable()                    # threshold in top-K epigraph
        theta = cp.Variable(n, nonneg=True)   # slacks

        # LAD abs residual linearisation
        # u = cp.Variable(n, nonneg=True)
        # ell = cp.Variable(n, nonneg=True)

        # abs covariance-contribution linearisation
        c = cp.Variable(n, nonneg=True)
        q = cp.Variable(n, nonneg=True)

        # abs variance-constribution linearisation
        r = cp.Variable(n, nonneg=True)
        # s = cp.Variable(n, nonneg=True)
        
        # model prediction
        yhat = X @ beta + (b0 if self.fit_intercept else 0.0)

        # fairness contribution: z_i = (d_i-dbar)*(yhat_i - mean(yhat))
        ratio_bar = (1.0 / n) * cp.sum(yhat / y)
        # z =  # affine in (beta,b0)

        # abs(z): z = c - q  => |z| = c + q
        constraints =  [ cp.multiply(d_center, yhat/y - ratio_bar) == c - q]
        # constraints += [ yhat/y - cp.mean(yhat/y) == r - s] # ratio mean deviation
        constraints += [ # Proxy using |e_i - mean(e)| 
             y - yhat - cp.mean(y - yhat) <= r,# - s,
            -y + yhat - cp.mean(y - yhat) <= r,# - s,
             y - yhat + cp.mean(y - yhat) <= r,
            -y + yhat + cp.mean(y - yhat) <= r,      
        ]
        # abs_z = c + q_

        # top-K epigraph constraints: score_i <= nu + theta_i
        cov_term = c + q if not self.neg_corr_focus else q 
        constraints += [
             y - yhat + rho_cov * cov_term + rho_var * (r) <= nu + theta,
            -y + yhat + rho_cov * cov_term + rho_var * (r) <= nu + theta,
        ]

        # objective: sum of K largest scores = min_{nu,theta} K*nu + sum theta
        obj = nu + (1/K) * cp.sum(theta)

        # regularization (do not regularize intercept)
        if self.l1 and self.l1 > 0:
            obj += float(self.l1) * cp.norm1(beta)
        if self.l2 and self.l2 > 0:
            obj += 0.5 * float(self.l2) * cp.sum_squares(beta)

        prob = cp.Problem(cp.Minimize(obj), constraints)

        # solve
        solver_opts = {} if self.solver_opts is None else dict(self.solver_opts)
        solve_kwargs = dict(verbose=self.verbose, warm_start=self.warm_start, **solver_opts)

        solver_map = {
            "MOSEK": cp.MOSEK,
            "ECOS": cp.ECOS,
            "SCS": cp.SCS,
            "OSQP": cp.OSQP,     # OK here (QP/LP), but may need tuning for accuracy
            "GUROBI": cp.GUROBI,
            "CPLEX": cp.CPLEX,
        }
        key = str(self.solver).upper()
        if key not in solver_map:
            raise ValueError(f"Unknown solver '{self.solver}'. Choose from {list(solver_map.keys())}.")

        prob.solve(solver=solver_map[key], **solve_kwargs)

        self.status_ = prob.status
        self.objective_value_ = prob.value

        if beta.value is None:
            raise RuntimeError(f"Optimization failed. Status: {prob.status}")
        else:
            print("Optimal Adversarial objective:", prob.value)

        self.coef_ = np.asarray(beta.value).reshape(-1)
        self.intercept_ = float(b0.value) if self.fit_intercept else 0.0

        # store diagnostics (optional)
        yhat_val = (X @ self.coef_) + self.intercept_
        fbar_val = float(np.mean(yhat_val))
        z_val = (d_center) * (yhat_val - fbar_val)
        score_val = np.abs(y - yhat_val) + rho_cov * np.abs(z_val)
        self._last_scores_ = score_val
        self._last_nu_ = float(nu.value) if nu.value is not None else None

        return self

    def predict(self, X):
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted yet.")
        X = self._as_2d_float(X)
        return X @ self.coef_ + self.intercept_

    def worst_case_indices_(self):
        """
        Returns indices of the K samples with largest composite score under the fitted model.
        This matches the intended top-K interpretation of the stable objective.
        """
        if self._last_scores_ is None or self.K_ is None:
            raise RuntimeError("Fit the model first.")
        idx = np.argsort(self._last_scores_)[::-1]
        return idx[: self.K_]

    def score(self, X, y):
        """
        sklearn-like score; returns negative MAE (since the fit objective is LAD-like).
        """
        y = self._as_1d_float(y, "y")
        yhat = self.predict(X)
        return -float(np.mean(np.abs(y - yhat)))
    
    def __str__(self):
        return f"StableAdversarial(rho_cov={self.rho_cov}, rho_var={self.rho_var})"


class StableAdversarialSurrogateRegressor2(BaseEstimator, RegressorMixin):
    """
    Stable (top-K) LAD regression with a *separable upper-bound* covariance penalty,
    eliminating the inner maximization.

    You (intentionally) use the separable upper bound:
        | sum_i w_i z_i | <= sum_i w_i |z_i|
    with z_i = (d_i - dbar) * (f_i - fbar),  f_i = x_i^T beta + b0,  fbar = (1/n) sum_j f_j.

    Stable/top-K adversary:
        max_{w} sum_i w_i s_i
        s.t. 0 <= w_i <= 1,  sum_i w_i = K
    which equals the sum of the K largest s_i.

    Here s_i(beta) = |y_i - f_i| + rho * | (d_i - dbar) * (f_i - fbar) |.

    Dual (single minimization):
        min_{beta,b0,nu,theta,...}  K*nu + sum_i theta_i + l1||beta||_1 + 0.5 l2||beta||_2^2
        s.t. s_i(beta) <= nu + theta_i,   theta_i >= 0,
             and epigraph linearisations of absolute values.

    Parameters
    ----------
    K : int or None
        Number of points in the stable/top-K objective (sum of K worst per-sample scores).
        If None, K is set from `keep` at fit time.
    keep : float
        Fraction in (0,1] used when K is None: K = ceil(keep * n_samples).
    rho : float
        Weight on the separable covariance-contribution penalty.
    l1, l2 : float
        L1 and L2 weights on coefficients (intercept not regularized).
    fit_intercept : bool
        Include an intercept b0.
    solver : str
        Default "MOSEK". Any solver supported by CVXPY for LP/QP.
    solver_opts : dict or None
        Passed to problem.solve(...).
    """

    def __init__(
        self,
        K=None,
        keep=0.1,
        rho_cov=1.0,
        rho_var=1.0,
        l1=0.0,
        l2=0.0,
        fit_intercept=True,
        solver="MOSEK",
        solver_opts=None,
        verbose=False,
        warm_start=False,
        neg_corr_focus=False,
    ):
        self.K = K
        self.keep = keep
        self.rho_cov = rho_cov
        self.rho_var = rho_var
        self.l1 = l1
        self.l2 = l2
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.solver_opts = solver_opts
        self.verbose = verbose
        self.warm_start = warm_start

        # learned
        self.coef_ = None
        self.intercept_ = 0.0
        self.status_ = None
        self.objective_value_ = None
        self.K_ = None
        self._last_scores_ = None
        self._last_nu_ = None

        # mine
        self.neg_corr_focus = neg_corr_focus

    @staticmethod
    def _as_2d_float(X):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2D array-like.")
        return X.astype(float)

    @staticmethod
    def _as_1d_float(x, name):
        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError(f"{name} must be 1D array-like.")
        return x.astype(float)

    def fit(self, X, y, d=None):
        X = self._as_2d_float(X)
        y = self._as_1d_float(y, "y")
        # if d is not None:
        d = self._as_1d_float(d, "d") if d is not None else y

        n, p = X.shape
        if y.shape[0] != n or d.shape[0] != n:
            raise ValueError("X, y, d must have the same number of samples.")

        # choose K
        if self.K is None:
            keep = float(self.keep)
            if not (0.0 < keep <= 1.0):
                raise ValueError("keep must be in (0,1].")
            K = int(np.ceil(keep * n))
        else:
            K = int(self.K)
        if not (1 <= K <= n):
            raise ValueError("K must satisfy 1 <= K <= n.")
        self.K_ = K

        rho_cov, rho_var = float(self.rho_cov), float(self.rho_var)
        if rho_cov < 0 or rho_var < 0:
            raise ValueError("ALL rho weights must be >= 0.")

        # centered d and mean prediction
        dbar = float(np.mean(d))
        d_center = d - dbar  # constants

        # variables
        beta = cp.Variable(p)
        b0 = cp.Variable() if self.fit_intercept else None

        nu_1 = cp.Variable()                    # threshold in top-K epigraph
        theta_1 = cp.Variable(n, nonneg=True)   # slacks
        nu_2 = cp.Variable()                    # threshold in top-K epigraph
        theta_2 = cp.Variable(n, nonneg=True)   # slacks

        # abs covariance-contribution linearisation
        c = cp.Variable(n, nonneg=True)
        q = cp.Variable(n, nonneg=True)
        
        # model prediction
        yhat = X @ beta + (b0 if self.fit_intercept else 0.0)

        # fairness contribution: z_i = (d_i-dbar)*(yhat_i - mean(yhat))
        ratio_mean = cp.mean(yhat / y)

        # abs(z): z = c - q  => |z| = c + q
        constraints =  [ cp.multiply(d_center, yhat/y - ratio_mean) == c - q]

        # top-K epigraph constraints: score_i <= nu_1 + theta_i
        cov_term = c + q if not self.neg_corr_focus else q 
        constraints += [
             y - yhat  <= nu_1 + theta_1,
            -y + yhat  <= nu_1 + theta_1,
            cov_term <= nu_2 + theta_2
        ]

        # objective: sum of K largest scores = min_{nu_1,theta_1} K*nu_1 + sum theta_1
        obj = nu_1 + (1/K) * cp.sum(theta_1) + rho_cov * (nu_2 + (1/K) * cp.sum(theta_2))

        # regularization (do not regularize intercept)
        if self.l1 and self.l1 > 0:
            obj += float(self.l1) * cp.norm1(beta)
        if self.l2 and self.l2 > 0:
            obj += 0.5 * float(self.l2) * cp.sum_squares(beta)

        prob = cp.Problem(cp.Minimize(obj), constraints)

        # solve
        solver_opts = {} if self.solver_opts is None else dict(self.solver_opts)
        solve_kwargs = dict(verbose=self.verbose, warm_start=self.warm_start, **solver_opts)

        solver_map = {
            "MOSEK": cp.MOSEK,
            "ECOS": cp.ECOS,
            "SCS": cp.SCS,
            "OSQP": cp.OSQP,     # OK here (QP/LP), but may need tuning for accuracy
            "GUROBI": cp.GUROBI,
            "CPLEX": cp.CPLEX,
        }
        key = str(self.solver).upper()
        if key not in solver_map:
            raise ValueError(f"Unknown solver '{self.solver}'. Choose from {list(solver_map.keys())}.")

        prob.solve(solver=solver_map[key], **solve_kwargs)

        self.status_ = prob.status
        self.objective_value_ = prob.value

        if beta.value is None:
            raise RuntimeError(f"Optimization failed. Status: {prob.status}")
        else:
            print("Optimal Adversarial objective:", prob.value)

        self.coef_ = np.asarray(beta.value).reshape(-1)
        self.intercept_ = float(b0.value) if self.fit_intercept else 0.0

        # store diagnostics (optional)
        yhat_val = (X @ self.coef_) + self.intercept_
        fbar_val = float(np.mean(yhat_val))
        z_val = (d_center) * (yhat_val - fbar_val)
        score_val = np.abs(y - yhat_val) + rho_cov * np.abs(z_val)
        self._last_scores_ = score_val
        self._last_nu_ = float(nu_1.value) if nu_1.value is not None else None

        return self

    def predict(self, X):
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted yet.")
        X = self._as_2d_float(X)
        return X @ self.coef_ + self.intercept_

    def worst_case_indices_(self):
        """
        Returns indices of the K samples with largest composite score under the fitted model.
        This matches the intended top-K interpretation of the stable objective.
        """
        if self._last_scores_ is None or self.K_ is None:
            raise RuntimeError("Fit the model first.")
        idx = np.argsort(self._last_scores_)[::-1]
        return idx[: self.K_]

    def score(self, X, y):
        """
        sklearn-like score; returns negative MAE (since the fit objective is LAD-like).
        """
        y = self._as_1d_float(y, "y")
        yhat = self.predict(X)
        return -float(np.mean(np.abs(y - yhat)))
    
    def __str__(self):
        return f"StableAdversarial2(rho_cov={self.rho_cov}, rho_var={self.rho_var})"


class RobustStableLADPRDCODRegressor(BaseEstimator, RegressorMixin):
    """
    Robust / stable LAD regression with a *single* CVaR adversary that stresses
    (i) LAD residuals, (ii) PRD-surrogate via |(v-mean(v)) * (ratio - r0)|, and
    (iii) COD-surrogate via (ratio - r0)^2, encoded with a rotated SOC.

    Objective (single-min form):
        min_{beta,b0,t,u,p,n,a,b}  t + (1/(alpha*n)) * sum(u)
                                  + l1*||beta||_1 + 0.5*l2*||beta||_2^2
    s.t.
        y - (X beta + b0) = p - n,  p,n >= 0
        e = p + n
        ratio = (X beta + b0) / s
        h = ratio - r0
        a >= (vtilde * h),  a >= -(vtilde * h)     (PRD surrogate abs)
        b >= h^2 via rotated SOC                   (COD surrogate)
        u >= e + w_prd*a + w_cod*b - t,  u >= 0

    Notes
    -----
    - Pass `s` (sale price) and `v` (value proxy) to fit().
    - Set `K` to use top-K stability; it will be converted to alpha = K/n.
    - Uses MOSEK by default; you can override with `solver=` and `solver_opts=`.
    """

    def __init__(
        self,
        # stability
        alpha=0.2,          # fraction in (0,1], used if K is None
        K=None,             # if provided, alpha := K/n at fit time
        # constraint weights inside robust objective
        w_prd=1.0,
        w_cod=1.0,
        # regularization
        l1=0.0,
        l2=0.0,
        # modeling choices
        fit_intercept=True,
        ratio_anchor=1.0,   # r0: center?
        center_v=True,
        # solver
        solver="MOSEK",
        solver_opts=None,
        verbose=False,
        warm_start=False,
    ):
        self.alpha = alpha
        self.K = K
        self.w_prd = w_prd
        self.w_cod = w_cod
        self.l1 = l1
        self.l2 = l2
        self.fit_intercept = fit_intercept
        self.ratio_anchor = ratio_anchor
        self.center_v = center_v
        self.solver = solver
        self.solver_opts = solver_opts
        self.verbose = verbose
        self.warm_start = warm_start

        # learned params
        self.coef_ = None
        self.intercept_ = 0.0
        self.status_ = None
        self.objective_value_ = None

    @staticmethod
    def _as_2d_float(X):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be 2D array-like.")
        return X.astype(float)

    @staticmethod
    def _as_1d_float(x, name):
        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError(f"{name} must be 1D array-like.")
        return x.astype(float)

    def fit(self, X, y, s=None, v=None):
        X = self._as_2d_float(X)
        y = self._as_1d_float(y, "y")

        if s is None or v is None:
            raise ValueError("You must pass s=... (sale price) and v=... (value proxy) to fit().")

        s = self._as_1d_float(s, "s")
        v = self._as_1d_float(v, "v")

        n, p = X.shape
        if y.shape[0] != n or s.shape[0] != n or v.shape[0] != n:
            raise ValueError("X, y, s, v must have the same number of samples.")
        if np.any(s <= 0):
            raise ValueError("All entries of s must be > 0 (to form ratios).")

        # stability level
        if self.K is not None:
            K = int(self.K)
            if not (1 <= K <= n):
                raise ValueError("K must satisfy 1 <= K <= n.")
            alpha = K / n
        else:
            alpha = float(self.alpha)
            if not (0.0 < alpha <= 1.0):
                raise ValueError("alpha must be in (0, 1].")

        inv_s = 1.0 / s

        # center v for PRD surrogate
        if self.center_v:
            vtilde = v - float(np.mean(v))
        else:
            vtilde = v.copy()

        # Decision variables
        beta = cp.Variable(p)
        b0 = cp.Variable() if self.fit_intercept else None

        t = cp.Variable()                 # CVaR threshold
        u = cp.Variable(n, nonneg=True)   # CVaR slacks

        ppos = cp.Variable(n, nonneg=True)
        nneg = cp.Variable(n, nonneg=True)

        a = cp.Variable(n, nonneg=True)   # abs(PRD surrogate) epigraph
        b = cp.Variable(n)                # will be constrained >= 0 by rotated SOC

        # Linear prediction and residual decomposition
        yhat = X @ beta + (b0 if self.fit_intercept else 0.0)
        constraints = []
        constraints += [y - yhat == ppos - nneg]
        e = ppos + nneg  # LAD magnitude (linear)

        # Ratio deviation h_i
        ratio = cp.multiply(inv_s, yhat)         # (Xb+b0)/s
        h = ratio - float(self.ratio_anchor)     # deviation from anchor r0

        # PRD surrogate: a_i >= | vtilde_i * h_i |
        vh = cp.multiply(vtilde, h)
        constraints += [a >= vh, a >= -vh]

        # COD surrogate: b_i >= h_i^2 via rotated SOC
        # Rotated SOC: 2 * b_i * 0.5 >= h_i^2  <=>  b_i >= h_i^2, with b_i >= 0.
        # for i in range(n):
        #     constraints += [cp.SOC(b[i], cp.hstack([h[i]]), 0.5)]
        constraints+=[ b >= cp.square(h) ]

        # CVaR epigraph for the composite per-point score
        # g_i = e_i + w_prd * a_i + w_cod * b_i
        g = e + float(self.w_prd) * a + float(self.w_cod) * b
        constraints += [u >= g - t]  # u_i >= g_i - t (and u_i >= 0 already)

        # Objective: t + (1/(alpha*n)) sum u + reg
        obj = t + (1.0 / (alpha * n)) * cp.sum(u)

        if self.l1 and self.l1 > 0:
            obj += float(self.l1) * cp.norm1(beta)
        if self.l2 and self.l2 > 0:
            obj += 0.5 * float(self.l2) * cp.sum_squares(beta)

        problem = cp.Problem(cp.Minimize(obj), constraints)

        # Solve
        solver_opts = {} if self.solver_opts is None else dict(self.solver_opts)
        solve_kwargs = dict(verbose=self.verbose, warm_start=self.warm_start, **solver_opts)

        solver_map = {
            "MOSEK": cp.MOSEK,
            "ECOS": cp.ECOS,
            "SCS": cp.SCS,
            "OSQP": cp.OSQP,  # (note: OSQP can't do SOC; included for completeness)
            "GUROBI": cp.GUROBI,
            "CPLEX": cp.CPLEX,
        }
        solver_key = str(self.solver).upper()
        if solver_key not in solver_map:
            raise ValueError(f"Unknown solver '{self.solver}'. Known: {list(solver_map.keys())}")

        problem.solve(solver=solver_map[solver_key], **solve_kwargs)

        self.status_ = problem.status
        self.objective_value_ = problem.value

        if beta.value is None:
            raise RuntimeError(f"Optimization failed. Status: {problem.status}")

        self.coef_ = np.asarray(beta.value).reshape(-1)
        self.intercept_ = float(b0.value) if self.fit_intercept else 0.0
        self.alpha_ = alpha  # learned effective alpha
        return self

    def predict(self, X):
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted yet.")
        X = self._as_2d_float(X)
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        # sklearn-like score: R^2 by default is typical, but for LAD
        # it's often more meaningful to report negative MAE.
        # We'll return negative MAE to match "higher is better".
        y = self._as_1d_float(y, "y")
        yhat = self.predict(X)
        return -float(np.mean(np.abs(y - yhat)))




class LeastMaxDeviationRegression:
    def __init__(self, fit_intercept=True, add_rmse_constraint=False, percentage_increase=0.00, solver="GUROBI"):
        self.beta = None
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.add_rmse_constraint = add_rmse_constraint
        self.percentage_increase = percentage_increase


    def fit(self, X, y):
        X, y = X.to_numpy(), y.to_numpy()
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        n, m = X.shape

        # Primal approach of the problem
        beta = cp.Variable(m)
        r_max = cp.Variable(1, nonneg=True)
        # r_min = cp.Variable(1, nonpos=True)
        residuals = y - X @ beta
        constraints = [
            residuals  <= r_max,
            -residuals  <= r_max,
            # residuals  >= r_min,
            # -residuals  >= r_min,
        ]

        # RMSE bound constraint (ONLY AVAILABLE FOR SAME TRAIN SIZE)
        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)
        min_rmse = root_mean_squared_error(y, model.predict(X))
        if self.add_rmse_constraint:
            constraints+=[
               cp.SOC(
                (1+self.percentage_increase)* min_rmse * np.sqrt(n),
                residuals
               ),
            ]

        # Objective: <=> Minimize |r_max - r_min |
        primal_prob = cp.Problem(
            cp.Minimize(r_max), 
            constraints
        )

        # Solve the optimization problem
        try:
            result = primal_prob.solve(solver=self.solver, verbose=False)
        except cp.error.SolverError:
            print(f"{self.solver} not available, trying default solver.")
            result = primal_prob.solve(verbose=False)

        print(f"Problem status: {primal_prob.status}")
        print(f"Optimal objective (Mean Absolute Error): {result}")

        # Print the difference in MAE between groups post-optimization
        if primal_prob.status in ["optimal", "optimal_inaccurate"]:
            self.beta = beta.value
            solve_time = primal_prob.solver_stats.solve_time
        else:
            print("Solver did not find an optimal solution. Beta coefficients not set.")
            self.beta = np.zeros(m) # Fallback beta


        return result, solve_time

    def predict(self, X):
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        return X @ self.beta
    
    def __str__(self):
        return f"LeastMaxDeviationRegression(fit_intercept={self.fit_intercept}, add_rmse_constraint={self.add_rmse_constraint}, percentage_increase={self.percentage_increase})"



class MaxDeviationConstrainedLinearRegression:
    def __init__(self, fit_intercept=True, add_rmse_constraint=False, percentage_increase=0.00, solver="GUROBI"):
        self.beta = None
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.add_rmse_constraint = add_rmse_constraint
        self.percentage_increase = percentage_increase


    def fit(self, X, y):
        X, y = X.to_numpy(), y.to_numpy()
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        n, m = X.shape

        # Primal approach of the problem
        beta = cp.Variable(m)
        r_max = cp.Variable(1, nonneg=True)
        # r_min = cp.Variable(1, nonpos=True)
        t = cp.Variable(1)
        residuals = y - X @ beta
        constraints = [
            residuals  <= r_max,
            -residuals  <= r_max,
            # residuals  >= r_min,
            # -residuals  >= r_min,
        ]

        # RMSE bound constraint (ONLY AVAILABLE FOR SAME TRAIN SIZE)
        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)
        # min_rmse = root_mean_squared_error(y, model.predict(X))
        max_res = np.max(np.abs( y - model.predict(X)) )
        if self.add_rmse_constraint:
            constraints+=[
            #    cp.SOC(
            #     (1+self.percentage_increase)* min_rmse * np.sqrt(n),
            #     residuals
            #    ),
                r_max <= max_res * (1 - self.percentage_increase),
                cp.SOC(t, residuals),
            ]

        # Objective: <=> Minimize |r_max - r_min |
        primal_prob = cp.Problem(
            # cp.Minimize(r_max), 
            # cp.Minimize(cp.quad_form(residuals, np.eye(n))),
            cp.Minimize(t),#**2),
            constraints
        )

        # Solve the optimization problem
        t0 = time()
        try:
            result = primal_prob.solve(solver=self.solver, verbose=False)
        except cp.error.SolverError:
            print(f"{self.solver} not available, trying default solver.")
            result = primal_prob.solve(verbose=False)
        solve_time = time() - t0

        print(f"Problem status: {primal_prob.status}")
        print(f"Optimal objective (RMSE): {np.sqrt(result/n)}")
        print(f"Time to solve: {solve_time}")


        # Print the difference in MAE between groups post-optimization
        if primal_prob.status in ["optimal", "optimal_inaccurate"]:
            self.beta = beta.value
            # solve_time = primal_prob.solver_stats.solve_time
        else:
            print("Solver did not find an optimal solution. Beta coefficients not set.")
            self.beta = np.zeros(m) # Fallback beta

        return result, solve_time

    def predict(self, X):
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        return X @ self.beta
    
    def __str__(self):
        return f"MaxDeviationConstrainedLinearRegression(fit_intercept={self.fit_intercept}, add_rmse_constraint={self.add_rmse_constraint}, percentage_increase={self.percentage_increase})"



# The current version of the Constrained Linear Regression
class GroupDeviationConstrainedLinearRegression:
    #  add_rmse_constraint=False,
    def __init__(self, fit_intercept=True, percentage_increase=0.00, n_groups=3, solver="GUROBI", max_row_norm_scaling=1, objective="mse", constraint="max_mse", l2_lambda=1e-3):
        self.beta = None
        self.fit_intercept = fit_intercept
        self.percentage_increase = percentage_increase
        # Ooptimization
        self.objective = objective # mae / mse
        self.constraint = constraint # max_mae / max_mse / max_mae_diff / max_mse_diff
        self.solver = solver
        # self.add_rmse_constraint = add_rmse_constraint
        self.l2_lambda = l2_lambda

        # Group constraints
        self.n_groups = n_groups
        self.max_row_norm_scaling = max_row_norm_scaling



    def fit(self, X, y):
        try:
            X = X.to_numpy()
        except Exception as e:
            pass
        try:
            y = y.to_numpy()
        except Exception as e:
            pass
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        n, m = X.shape

        if self.max_row_norm_scaling > 1: # Scaling of the max_i ||x_i||: alpha * y_i ~ alpha * x_i @ beta 
            max_row_norm_index = np.argmax(np.linalg.norm(X, axis=1))
            X[max_row_norm_index,:] = self.max_row_norm_scaling * X[max_row_norm_index,:]
            y[max_row_norm_index] = self.max_row_norm_scaling * y[max_row_norm_index]
            # X = self.max_row_norm_scaling * X
            # y = self.max_row_norm_scaling * y


        # Variable
        beta = cp.Variable(m)
        z = cp.Variable(n)
        u_g, l_g = cp.Variable(1), cp.Variable(1)

        # Constraints
        if self.objective == "mse" and self.l2_lambda == 0:
            model = LinearRegression(fit_intercept=False)
        elif self.objective == "mse":
            model = Ridge(fit_intercept=False, alpha=self.l2_lambda/n)
        elif self.objective == "mae":
            model = LeastAbsoluteDeviationRegression(fit_intercept=False)
        model.fit(X, y)
        if self.objective == "mse":
            beta_ols = model.coef_
        real_z = np.abs(y - model.predict(X))
        ols_mse = root_mean_squared_error(y, model.predict(X))**2
        ols_mse_plus_reg = root_mean_squared_error(y, model.predict(X))**2 + self.l2_lambda * (beta_ols @ beta_ols)
        lad_mae = np.mean(real_z)
        constraints = [
            # cp.SOC(r, y - X @ beta)
            # cp.norm(y - X @ beta, 2) <= r,  # second-order cone
            y - X @ beta <= z,
            -y + X @ beta <= z,
            # y - X @ beta == u - l,
            # z <= y - X @ beta + b * M,
            # z <= -y + X @ beta + (1-b) * M,
            # cp.mean(z) <= lad_mae * (1+self.percentage_increase),
            # cp.SOC(
            #     t,
            #     y - X @ beta
            # )
        ]

        # Fairness constraint
        # tau = 1e-10
        n_groups = 3
        interval_size = y.max() - y.min() + 1e-6 # a little bit more
        bins = y.min() + np.array([i*interval_size /n_groups for i in range(n_groups+1)])
        # X_bins, y_bins = [], []
        bin_indices_list = []
        for j,lb in enumerate(bins[:-1]):
            ub = bins[j+1]
            bin_indices = np.where((y>=lb) & (y<ub))[0]
            # print(bin_indices)
            bin_indices_list.append(bin_indices)
            # # Data
            # X_bins.append(X[bin_indices,:])
            # y_bins.append(y[bin_indices])

        # Compute the actual max difference
        tau = 0 
        for i in range(n_groups):
            print(i, len(bin_indices_list[i]))
            if "mae" in self.constraint: 
                constraints+=[
                    cp.mean(z[bin_indices_list[i]])  <= u_g,
                    cp.mean(z[bin_indices_list[i]])  >= l_g,
                ]
            elif "mse" in self.constraint:
                n_i = len(bin_indices_list[i])
                constraints+=[
                    cp.SOC(u_g, z[bin_indices_list[i]]/np.sqrt(n_i)),
                    # cp.SOC(u_g, z[bin_indices_list[i]]),
                    # cp.mean(**2)  <= u_g,
                    # cp.mean(z[bin_indices_list[i]]**2)  >= l_g,
                ]
            if not "diff" in self.constraint:
                if "mae" in self.constraint:
                    error_i = np.mean(real_z[bin_indices_list[i]])
                elif "mse" in self.constraint:
                    error_i = np.mean(real_z[bin_indices_list[i]]**2)
                    print("error_i: ", error_i)
                if error_i > tau:
                    tau = error_i
            elif "diff" in self.constraint:
                for j in range(i+1, n_groups):
                    if "mae" in self.constraint: 
                        error_i,error_j = np.mean(real_z[bin_indices_list[i]]), np.mean(real_z[bin_indices_list[j]])
                    elif "mse" in self.constraint:
                        error_i,error_j = np.mean(real_z[bin_indices_list[i]]**2), np.mean(real_z[bin_indices_list[j]]**2)
                    diff_ij = np.abs(error_i - error_j)
                    if diff_ij >  tau:
                        tau = diff_ij
        print("tau", tau)
        tau_bound = tau * (1-self.percentage_increase)
        print("bound: ", tau_bound)
        if "diff" in self.constraint:
            constraints+=[  
                u_g - l_g <= tau_bound
            ]
        else: # not "diff" in self.constraint
            if "mse" in self.constraint:
                print("Constraining u_g to: ", np.sqrt(tau_bound))
                constraints+=[u_g <= np.sqrt(tau_bound)]
            elif "mae" in self.constraint:
                print("Constraining u_g to:", tau_bound)
                constraints+=[u_g <= tau_bound]
        # Objective
        if self.objective == "mse":
            if self.l2_lambda != 0:
                print("Solving with Ridge objective...")
                obj = cp.Minimize(cp.mean(z**2) + self.l2_lambda * cp.quad_form(beta, np.eye(m)))
            else: 
                obj = cp.Minimize(cp.mean(z**2))
        elif self.objectiv == "mae":
            obj = cp.Minimize(cp.mean(z))

        primal_prob = cp.Problem(obj, constraints)

        # Solve the optimization problem
        t0 = time()
        try:
            result = primal_prob.solve(solver=self.solver, verbose=False)
        except cp.error.SolverError:
            print(f"{self.solver} not available, trying default solver.")
            result = primal_prob.solve(verbose=False)
        solve_time = time() - t0
        print(f"Problem status: {primal_prob.status}")
        if self.objective == "mse" and self.l2_lambda == 0:
            print(f"OLS objective (MSE): {ols_mse}")
            print(f"Optimal objective (MSE): {result}")
            price_of_fairness = (result-ols_mse)/ols_mse
            print(f"POF (MSE % decrease): ", price_of_fairness)
            J_0_value = ols_mse
        elif self.objective == "mse":
            beta_ols = np.linalg.inv(X.T @ X + n*self.l2_lambda*np.eye(m)) @ X.T @ y
            ridge_mse = root_mean_squared_error(y, X @ beta_ols)**2 
            ridge_plus_reg = ridge_mse + self.l2_lambda * beta_ols @ beta_ols
            print(f"My Ridge objective (MSE + reg): {ridge_plus_reg}")
            print(f"My Ridge objective (MSE-only): {ridge_mse}")
            # print(f"Ridge objective (MSE + reg): {ols_mse_plus_reg}")
            # ridge_mse = root_mean_squared_error(y, X @ beta_ols)**2
            # print(f"Ridge objective (MSE-only): {ridge_mse}")
            print(f"Optimal objective (MSE + reg): {result}")
            opt_mse = root_mean_squared_error(y, X @ beta.value)**2
            print(f"Optimal objective (MSE-only): {opt_mse}")
            price_of_fairness = (result-ridge_plus_reg)/ridge_plus_reg
            print(f"POF (MSE + reg % decrease): ", price_of_fairness)
            price_of_fairness = (opt_mse - ridge_mse) / ridge_mse
            J_0_value = ridge_plus_reg
            # exit()
        elif self.objective == "mae":
            print(f"Optimal objective (MAE): {result}")
            price_of_fairness = (result-lad_mae)/lad_mae
            print(f"POF (MAE % decrease): ", price_of_fairness)


        # Real fairness measure
        real_tau, real_tau_i, real_tau_j = 0, None, None
        for i in range(n_groups):
            # print(i, len(bin_indices_list[i]))
            if not "diff" in self.constraint:
                if "mae" in self.constraint:
                    error_i = np.mean(z.value[bin_indices_list[i]])
                elif "mse" in self.constraint:
                    error_i = np.mean(z.value[bin_indices_list[i]]**2)
                if error_i > real_tau:
                    real_tau, real_tau_i = error_i, i
            elif "diff" in self.constraint:
                for j in range(i+1, n_groups):
                    if "mae" in self.constraint:
                        error_i, error_j = np.mean(z.value[bin_indices_list[i]]), np.mean(z.value[bin_indices_list[j]] )
                    elif "mse" in self.constraint:
                        error_i, error_j = np.mean(z.value[bin_indices_list[i]]**2), np.mean(z.value[bin_indices_list[j]]**2)
                    diff_ij = np.abs(error_i - error_j)
                    if diff_ij > real_tau:
                        real_tau, real_tau_i, real_tau_j = diff_ij, i, j 

        # Fairness improvement and bounds
        fairness_improvement = np.abs(real_tau - tau)
        delta_fairness = self.percentage_increase*tau # tau - real_tau
        # virtual_fairness_improvement = np.abs(tau * (1 - self.percentage_increase) - tau)
        # Bounds on POF from MSE and MSE
        if self.objective == "mse" and self.constraint == "max_mse":
            g, indices_g = real_tau_i, bin_indices_list[real_tau_i]
            n_g, X_g, y_g = len(indices_g), X[indices_g,:], y[indices_g]

            # Lower bound # (n/(self.n_groups * n_g))
            a_0 = (2/n_g)*(-X_g.T @ y_g + X_g.T @ X_g @ beta_ols) + 2 * self.l2_lambda * beta_ols # gradient of the single J_g
            H = (2/n) * (X.T @ X) + 2 * self.l2_lambda
            eigen_vals, eigen_vecs = np.linalg.eigh(X.T @ X)
            # H_inv =  eigen_vecs.T @ ((n/2) * np.diag(1/eigen_vals) + 1/(2*self.l2_lambda ) ) @ eigen_vecs
            H_inv = np.linalg.pinv(H)
            A = a_0.T @ H_inv @ a_0 
            delta_J_lb = (fairness_improvement)**2 / (2 * A) # Lower bound

            # print(fr"Delta J lb (mse-max_mse): ", delta_J_lb )
            print(fr"POF J % lb 1 (mse-max_mse): ", delta_J_lb / J_0_value)
            d = H_inv @ a_0


            # # Looser LB (strong convexity)
            # m = (2/n) * np.linalg.eigvalsh(X.T @ X)[0] + 2 * self.l2_lambda
            # delta_J_lb_2 = m*(fairness_improvement)**2 / (2 * a_0 @ a_0) # Lower bound
            # # print(fr"Delta J lb 2 (mse-max_mse): ", delta_J_lb_2 )
            # print(fr"POF J % lb 2 (mse-max_mse): ", delta_J_lb_2 / J_0_value)

            # # Looser LB slightly tighter (strong convexity)
            # m_g = (2/n_g) * np.linalg.eigvalsh(X_g.T @ X_g)[0] + 2 * self.l2_lambda
            # if delta_fairness > 0:
            #     extra_term = ( (1-np.sqrt(1-2*m_g*delta_fairness/(a_0 @ a_0))) / (m_g*delta_fairness/(a_0 @ a_0)) )**2
            #     if extra_term > 1:
            #         # print("Extra term: ", extra_term)
            #         delta_J_lb_3 = delta_J_lb_2 * extra_term
            #         print(fr"POF J % lb 3 (mse-max_mse): ", delta_J_lb_3 / J_0_value)

            # Exponential LB
            # Checking exponential bounds
            M_phi = np.max(X @ beta_ols)
            C_phi = (np.exp(-M_phi) + M_phi - 1)/M_phi**2
            # print("M_phi: ", M_phi)
            # print("exp + M - 1 (LB): ", np.exp(-M_phi) + M_phi - 1)
            # print("Curvature constant (LB): ", C_phi)
            delta_J_lb_3 = C_phi*(fairness_improvement)**2 / ( a_0 @ H_inv @ a_0) # Lower bound
            # print(f"Delta MSE lb 4 (MSE-only): ", delta_J_lb_3)
            print(f"POF J % lb 4 (MSE % decrease): ", delta_J_lb_3 / J_0_value)
            # print("exp + M - 1 (UB): ", np.exp(M_phi) - M_phi - 1)
            # print("Curvature constant (UB): ", (np.exp(M_phi) - M_phi - 1)/M_phi**2)
            # exit()


            if self.l2_lambda > 0:
                print(f"POF (MSE % decrease): ", price_of_fairness)

                # First lower bound on delta MSE
                H_loss_inv_a_0 = H_inv @ a_0
                delta_loss = delta_fairness**2 / ( 2 * (a_0 @ H_loss_inv_a_0) )
                beta_lb = - delta_fairness * H_loss_inv_a_0 / (a_0 @ H_loss_inv_a_0)
                delta_mse_lb = delta_loss - self.l2_lambda * ( root_mean_squared_error(beta_ols, beta_lb)**2 + beta_ols @ beta_ols )
                print(f"Delta MSE lb (MSE-only): ", delta_mse_lb)
                print(f"POF lb (MSE % decrease): ", delta_mse_lb / ridge_mse)

            H_g = (2/n_g)*(X_g.T @ X_g) + 2 * self.l2_lambda 

            # # LB version 3 (strongly convex LB)
            # eig_min_g = np.min(np.linalg.eigvalsh(H_g))
            # print("Min eigen: ", eig_min_g)
            # if eig_min_g > 0:
            #     a_0 = a_0
            #     delta_beta = -delta_fairness*H_inv*a_0/(a_0.T @ H_inv @ a_0)
            #     a_0_delta_beta = a_0 @ delta_beta 
            #     m_norm_delta_beta = (eig_min_g/2)*(delta_beta @ delta_beta)
            #     print("m_norm_beta", m_norm_delta_beta)
            #     t_LB = (-a_0_delta_beta - np.sqrt( a_0_delta_beta**2 - 4*(m_norm_delta_beta)*delta_fairness ))/(2 * m_norm_delta_beta)
            #     delta_J_lb_3 = (t_LB**2/2)*(a_0.T @ H_inv @ a_0)
            #     print(fr"POF J % lb 3 (strongly convex): ", delta_J_lb_3 / ols_mse)

            # Upper bound
            C_ray = 0
            # A_max = 0
            for i in range(n_groups):
                indices_g = bin_indices_list[i]
                n_g, X_g, y_g = len(indices_g), X[indices_g,:], y[indices_g]
                H_g = (2/n_g) * (X_g.T @ X_g)
                d_H_g_d = d.T @ H_g @ d
                C_ray = d_H_g_d if d_H_g_d > C_ray else C_ray

                # UB bound 2
                # a_0 = (2/n_g)*(-X_g.T @ y_g + X_g.T @ X_g @ beta_ols)
                # A_max = a_0 @ d if a_0 @ d > A_max else A_max

                
            t_UB = (A - np.sqrt(A**2 - 2 * C_ray * delta_fairness)) / C_ray
            # t_UB_2 = (A_max - np.sqrt(A_max **2 - 2 * C_ray * delta_fairness)) / C_ray
            delta_J_ub = (1/2)*A*t_UB**2 # Upper bound
            # delta_J_ub_2 = (1/2) * A * t_UB_2**2
            print(fr"Delta J ub (mse-max_mse): ", delta_J_ub )
            print(fr"POF J % ub (mse-max_mse): ", delta_J_ub / ols_mse)
            # print(fr"POF J % ub 2 (mse-max_mse): ", delta_J_ub_2 / ols_mse)

            
            # UB bound 3 (tighter?)
            indices_g = bin_indices_list[real_tau_i]
            n_g, X_g, y_g = len(indices_g), X[indices_g,:], y[indices_g]
            H_g = (2/n_g) * (X_g.T @ X_g)
            C_g = d.T @ H_g @ d
            t_UB_3 = (A - np.sqrt(A**2 - 2 * C_g * delta_fairness)) / C_g
            delta_J_ub_3 = (1/2)*A*t_UB_3**2 # Upper bound
            print(fr"POF J % ub 3 (mse-max_mse): ", delta_J_ub_3 / ols_mse)            


            # print(fr"V - POF J % lb (mse-max_mse): ", ((virtual_fairness_improvement)**2 / 2 * A) / ols_mse)

        # print("l2_lambda: ", self.l2_lambda)
        # print("lambda_min original: ", np.min(np.linalg.eigh(X.T @ X)[0]))
        # lambdas_XX = np.linalg.eigh(X.T @ X)[0]  # min eigenvalue
        # min_lambda_XX = np.min(lambdas_XX) + self.l2_lambda
        # print("Min eigenvalue: ", min_lambda_XX)
        # print("Max eigenvalue: ", np.max(lambdas_XX))
        # max_row_norm = np.max(np.linalg.norm(X, axis=1))
        # print("Max row norm:", max_row_norm)
        # pof_lower_bound = (1/(4*n)) * min_lambda_XX / max_row_norm**2 * fairness_improvement**2 
        # print("POF lower bound (%)", pof_lower_bound)
        fairness_effective_improvement = fairness_improvement/tau
        print(f"FEI (% improvement)", fairness_effective_improvement)


        print(f"Time to solve: {solve_time}")

        # Print the difference in MAE between groups post-optimization
        if primal_prob.status in ["optimal", "optimal_inaccurate"]:
            self.beta = beta.value
        else:
            print("Solver did not find an optimal solution. Beta coefficients not set.")
            self.beta = np.zeros(m) # Fallback beta

        return result, solve_time, price_of_fairness, fairness_effective_improvement, delta_J_lb/ ols_mse, delta_J_ub_3/ ols_mse, real_tau

    def predict(self, X):
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        return X @ self.beta
    
    def __str__(self): #add_rmse_constraint={self.add_rmse_constraint},
        return f"GroupDeviationConstrainedLinearRegression(fit_intercept={self.fit_intercept},  percentage_increase={self.percentage_increase}, n_groups={self.n_groups})"


class MyGLMRegression:

    def __init__(self, fit_intercept=True, l2_lambda=1e-3, solver="GUROBI", solver_verbose=False, eps=1e-4, model_name="logistic"):
        self.beta = None
        self.fit_intercept = fit_intercept
        self.model_name = model_name
        # Ooptimization
        # self.objective = objective ||  objective="logistic",
        self.solver = solver
        self.solver_verbose = solver_verbose
        self.l2_lambda = l2_lambda
        self.eps = eps
        # Loss
        self.train_loss = None


    def fit(self, X, y):
        try:
            X = X.to_numpy()
        except Exception as e:
            pass
        try:
            y = y.to_numpy()
        except Exception as e:
            pass
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        n, m = X.shape      

        # Logistic regression optimization
        # To be used in constraints of CVXPY
        def get_bregman_divergence_cxpy(X, y, beta, model="logistic", eps=1e-4):
            theta = X @ beta
            y_ = np.array(y, dtype=np.float64).copy()
            if model == "linear":
                eta = y_ 
                psi = lambda z: z**2/2
            elif model=="logistic": 
                y_[y > 1-eps] = 1-eps
                y_[y < eps] = eps 
                eta = cp.log(y_ / (1-y_)) # explodes in 0 and 1
                psi = lambda z: cp.logistic(z) #cp.log( 1 + cp.exp(z) )
            elif model =="poisson":
                y_[y < eps] = eps
                eta = cp.log(y_) # explodes in 0
                psi = lambda z: cp.exp(z)
            elif model == "svm": # Smoothing of hinge: max(0, 1-x)
                eta = y_ # psi(1)=0 (the proper label)
                psi = lambda z: cp.pos(1 - cp.multiply(y_, z))
            psi_tilda_inv_y = y_ if model != "svm" else 0 # g = 0 (subgradient=0 always valid for svm)
            return psi(theta) - psi(eta) - cp.multiply(psi_tilda_inv_y, (theta - eta)) 

        # Variables
        beta = cp.Variable(m)
        z = cp.Variable(n)

        # Constraints
        # constraints = [cp.logistic(X @ beta) - cp.multiply(y, X @ beta) - cp.logistic(np.log(y / (1-y))) +  cp.multiply(y, eta) <= z]  # proves that we can write constraints       
        constraints = [get_bregman_divergence_cxpy(X, y, beta, model=self.model_name, eps=self.eps) <= z]

        # Objective
        if self.l2_lambda != 0:
            print("Solving with Ridge objective...")
            if self.model_name != "svm":
                obj = cp.Minimize(cp.mean( z ) + self.l2_lambda * cp.quad_form(beta, np.eye(m)))
            else: # svm: beta = [w, b]
                obj = cp.Minimize(cp.mean( z ) + self.l2_lambda * cp.quad_form(beta[1:], np.eye(m-1))) 
        else: 
            obj = cp.Minimize( cp.mean( z ) )

        primal_prob = cp.Problem(obj, constraints)

        # Solve the optimization problem
        t0 = time()
        try:
            # solver_params={
            #     "mosek_params": {
            #     # 'MSK_DPAR_INTPNT_CO_TOL_REL_FEAS': 1e-4,  # Primal feasibility tolerance
            #     # 'MSK_DPAR_INTPNT_CO_TOL_REL_FEAS': 1e-4,  # Dual feasibility tolerance
            #     'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-6, # Relative duality gap tolerance
            #     }
            # }
            result = primal_prob.solve(solver=self.solver, verbose=self.solver_verbose)#, **solver_params)
        except cp.error.SolverError:
            print(f"{self.solver} not available, trying default solver.")
            result = primal_prob.solve(verbose=self.solver_verbose)
        solve_time = time() - t0
        print(f"Problem status: {primal_prob.status}")
        print(f"Objective value: ", result)
        print(f"Solving time: {solve_time}")
        
        # Store results
        self.coef_ = beta.value
        self.train_loss = result

    def predict(self, X):
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        theta = X @ self.coef_
        if self.model_name == "linear":
            y_hat = theta
        elif self.model_name == "logistic":
            y_hat = np.exp(theta) / (1 + np.exp(theta))
        elif self.model_name == "poisson":
            y_hat = np.exp(theta)
        elif self.model_name == "svm":
            y_hat = np.sign(theta)
        return y_hat
    
    def __str__(self): 
        return f"MyGLMRegression(fit_intercept={self.fit_intercept}, l2_lambda={self.l2_lambda})"



    
def get_group_bins_indices(y, n_groups=3, nature="continuous"):
    if nature == "continuous":
        interval_size = y.max() - y.min() + 1e-6 # a little bit more
        bins = y.min() + np.array([i*interval_size /n_groups for i in range(n_groups+1)])
        bin_indices_list = []
        for j,lb in enumerate(bins[:-1]):
            ub = bins[j+1]
            bin_indices = np.where((y>=lb) & (y<ub))[0]
            bin_indices_list.append(bin_indices)
    elif nature =="discrete":
        print("doing the discete binning...")
        groups = np.unique(y)
        n_groups = groups.size
        bin_indices_list = []
        for g in groups:
            bin_indices = np.where(y == g)[0]
            bin_indices_list.append(bin_indices)
    return bin_indices_list, n_groups

class GroupDeviationConstrainedLogisticRegression:
    #  add_rmse_constraint=False,
    def __init__(self, fit_intercept=True, percentage_increase=0.00, n_groups=3, solver="GUROBI", max_row_norm_scaling=1, objective="mse", constraint="max_mse", l2_lambda=1e-3, eps=1e-4, model_name="logistic"):
        self.beta = None
        self.fit_intercept = fit_intercept
        self.percentage_increase = percentage_increase
        self.model_name=model_name
        # Optimization
        self.objective = objective # mae / mse
        self.constraint = constraint # max_mae / max_mse / max_mae_diff / max_mse_diff
        self.solver = solver
        self.eps = eps
        # self.add_rmse_constraint = add_rmse_constraint
        self.l2_lambda = l2_lambda

        # Group constraints
        self.n_groups = n_groups
        self.max_row_norm_scaling = max_row_norm_scaling



    def fit(self, X, y, sensitive_feature=None, sensitive_nature="continuous"):
        try:
            X = X.to_numpy()
        except Exception as e:
            pass
        try:
            y = y.to_numpy()
        except Exception as e:
            pass
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        n, m = X.shape

        # Compute original solution (J_0)
        # if self.model_name != "linar":
        glm = MyGLMRegression(fit_intercept=False, model_name=self.model_name, l2_lambda=self.l2_lambda, solver=self.solver)#LogisticRegression(fit_intercept=False, penalty=None, max_iter=500)
        glm.fit(X, y)
        # else:
        #     beta_0 = np.linalg.pinv(X.T @ X) @ (X.T @ y)

        # glm = LinearRegression(fit_intercept=False)
        #     # glm = Ridge(fit_intercept=False, alpha=self.l2_lambda/n)
        #     # raise("NO REGULARIZED VERSION")
        # elif self.model_name == "poisson":
        #     pass
        # elif self.objective == "mae":
        #     raise("NO MAE VERSION")
            # glm = LeastAbsoluteDeviationRegression(fit_intercept=False)
        
        # if self.objective == "mse":

        def get_psi_derivatives(X, y, beta, model="linear", gamma=1e-3):
            "Returns the approximation of y: phi'(X beta)"
            theta = X @ beta
            if model == "linear":
                return theta, np.ones(X.shape[0], dtype=int)
            elif model == "logistic":
                psi_tilda = np.exp(theta) / ( 1 + np.exp(theta) )
                return psi_tilda, psi_tilda / ( 1 + np.exp(theta) )
            elif model == "poisson":
                psi_tilda = np.exp(theta)
                return psi_tilda, psi_tilda
            elif model == "svm":
                # This if for the bounds, so it is not the real hing, but the smooth approximation.
                # theta_gamma = (1-y*theta) / gamma
                # exp_thresholding =  np.max(np.abs(theta_gamma)) * np.sign(theta_gamma) * (-1)  # to avoid exponential overflow
                # exp_thresholding = exp_thresholding if np.max(np.abs(theta_gamma)) <= 1e1 else 1e1 * np.sign(theta_gamma) * (-1)
                # print("exp_thresholding", exp_thresholding)
                # # psi_tilda = np.exp((1-y*theta) / gamma ) / ( 1 + np.exp((1-y*theta) / gamma ) )
                # psi_tilda = np.exp((1-y*theta) / gamma + exp_thresholding ) / ( np.exp(exp_thresholding) + np.exp((1-y*theta) / gamma + exp_thresholding ) )
                # # print("(1-y*theta)", (1-y*theta))
                # # print("(1-y*theta) / gamma", (1-y*theta) / gamma)
                # print("psi_tilda", np.max(psi_tilda), np.min(psi_tilda))
                # # print("1/gamma", (1/gamma))
                # return -y * psi_tilda, (1/gamma) * psi_tilda / ( 1 + np.exp((1-y*theta) / gamma) )

                # 2nd approximation: Huber smoothing
                z = np.multiply(y, theta)
                psi_tilda = np.zeros(theta.size)
                psi_tilda_2 = psi_tilda.copy()
                idx = (z < 1) & (z >= (1 - gamma))
                psi_tilda[idx] = (1 - z[idx])**2 / (2 * gamma)
                psi_tilda_2[idx] = 1 / gamma 
                psi_tilda[z < (1 - gamma)] =  -1
                return np.multiply(y, psi_tilda), psi_tilda_2 # y ** 2 = 1 for the second derivative
            else:
                raise Exception(f"No model model named: {model}!!")
        
        def get_bregman_divergence_mean_value(X, y, beta, model="linear", eps=1e-4):
            theta = X @ beta
            y_ =  np.array(y, dtype=np.float64).copy()
            if model == "linear":
                eta = y_ 
                psi = lambda z: z**2/2
            elif model=="logistic": 
                y_[y > 1-eps] -= eps
                y_[y < eps] += eps 
                eta = np.log(y_ / (1-y_)) # explodes in 0 and 1
                psi = lambda z: np.log(1+np.exp(z)) 
            elif model =="poisson":
                y_[y < eps] = eps
                eta = np.log(y_) # explodes in 0
                psi = lambda z: np.exp(z)
            elif model == "svm":
                eta = y_ # psi(1)=0 (the proper label)
                psi = lambda z: cp.pos(1 - np.multiply(y_, z)).value # psi(z) = max(0, 1 - yz)
            # psi_tilda_inv_y = y_ if model != "svm" else 0 # g = 0 (subgradient=0 always valid for svm)
            # print("psi(theta)", np.min(psi(theta)), np.max(psi(theta)))
            return np.mean( psi(theta) - psi(eta) - np.multiply(y_, (theta - eta)) ) if model != "svm" else np.mean( psi(theta) ) #- psi(eta) + np.multiply(y_, (theta - eta)) )
        
        def get_loss_value(X, y, beta, model="linear"):
            theta = X @ beta
            if model == "linear":
                psi = theta ** 2 / 2 
            elif model == "logistic":
                psi = np.log( 1 + np.exp(theta) )
            elif model == "poisson":
                psi = np.exp(theta)
            elif model == "svm":
                psi = cp.pos(1 - np.multiply(y, theta)).value
            else:
                raise Exception(f"No model named {model}!!")
            second_term = y * theta if model != "svm" else 0
            return np.mean(psi - second_term)

        # Unconstrained problem utils
        beta_0 = glm.coef_
        # J_0 =  glm.train_loss # J_0: logit loss
        J_0 = get_bregman_divergence_mean_value(X, y, beta_0, model=self.model_name, eps=self.eps) 
        w_0, w_0_2 = get_psi_derivatives(X, y, beta_0, model=self.model_name)#, gamma=self.eps)
        # w_0 = np.exp(X @ beta_0) / ( 1 + np.exp(X @ beta_0) )
        a_0 = (1/n) * X.T @ (w_0 - y) if self.model_name != "svm" else (1/n) * X.T @ w_0 # gradient of J_0
        H_0 = (1/n) * X.T @ np.diag(w_0_2) @ X # Hessian of J_0
        # H_0 = np.mean( [ np.exp(X[i,:] @ beta_0) / ( 1 + np.exp(X[i,:] @ beta_0) )**2 * np.outer(X[i,:],  X[i,:]) for i in range(n) ], axis=0 ) 
        H_0_inv = np.linalg.pinv(H_0)
        M_psi = 1 if self.model_name != "linear" else 0# for logistic (I think for SVM we maintain the 1)

        print("Predicted y's: ", np.round(glm.predict(X[:20,:])))
        print("The real  y's: ", y[:20])

        # Fairness constraints utils
        bin_indices_list, n_groups = get_group_bins_indices(sensitive_feature, n_groups=self.n_groups, nature=sensitive_nature)
        g_max = np.argmax([len(x) for x in bin_indices_list]) # The largest group
        print("bin_indices_list, n_groups: ", n_groups, [len(_) for _ in bin_indices_list])
        self.n_groups = n_groups # update in case its not continuous
        tau = 0 # Compute the max diference of the unconstrained problem
        loss_0 = get_loss_value(X, y, beta_0, model=self.model_name)#j_0#np.mean( np.log( 1 + np.exp(X @ beta_0) ) - y * (X @ beta_0) )
        print("loss_0: ", loss_0)
        b_d_0 = get_bregman_divergence_mean_value(X, y, beta_0, model=self.model_name, eps=self.eps)
        acc_0 = np.average(y == np.round(get_psi_derivatives(X, y, beta_0, model=self.model_name, gamma=self.eps)[0]))
        print("Accuracy 0: ", acc_0)
        print("Bregman divergence 0: ", b_d_0)
        g_0 = -1
        for g in range(self.n_groups):
            print("Group: ", g, len(bin_indices_list[g]))
            X_g , y_g = X[bin_indices_list[g], :], y[bin_indices_list[g]]
            # theta_0_g = X_g @ beta_0
            loss_0_g = get_loss_value(X_g, y_g, beta_0, model=self.model_name) #np.mean( np.log( 1 + np.exp(theta_0_g) ) - y_g * theta_0_g )
            print("loss_0_g: ", loss_0_g)
            # print("Accuracy 0_g: ", np.average(y_g == np.round(get_psi_derivatives(X_g, y_g, beta_0, model=self.model_name, gamma=self.eps)[0])))
            y_glm_pred = get_psi_derivatives(X_g, y_g, beta_0, model=self.model_name, gamma=self.eps)[0]
            print("Accuracy 0_g: ", r2_score(y_g, y_glm_pred))

            # bregman
            b_d = get_bregman_divergence_mean_value(X_g, y_g, beta_0, model=self.model_name, eps=self.eps)
            print("Bregman divergence_0_g: ", b_d)
            # print("Bregman 0_g OLS: ", get_bregman_divergence_mean_value(X_g, y_g, beta_ols, model=self.model_name, eps=self.eps))
            # print("MSE/2 0_g: ", root_mean_squared_error(y_g, X_g @ beta_ols)**2/2)
            if b_d > tau:
                tau, g_0 = b_d, g
        tau_bound = tau * (1-self.percentage_increase)
        print("tau 0: ", tau)
        print("fair tau bound: ", tau_bound)

        # # STOPPING CONDITION (cannot be more fair)
        # if g_0 == g_max:
        #     print("THE MODEL CANNOT BE MORE FAIR THAN BASELINE")
        #     self.beta = beta_0
        #     return b_d_0, 0, 0, 0, 0, 0, 0, 0, 0, tau


        # To be used in constraints of CVXPY
        def get_bregman_divergence_cxpy(X, y, beta, model="logistic", eps=1e-4):
            theta = X @ beta
            y_ = np.array(y, dtype=np.float64).copy()
            if model == "linear":
                eta = y_ 
                psi = lambda z: z**2/2
            elif model=="logistic": 
                y_[y > 1-eps] = 1-eps
                y_[y < eps] = eps 
                eta = cp.log(y_ / (1-y_)) # explodes in 0 and 1
                psi = lambda z: cp.logistic(z) #cp.log( 1 + cp.exp(z) )
            elif model =="poisson":
                y_[y < eps] = eps
                eta = cp.log(y_) # explodes in 0
                psi = lambda z: cp.exp(z)
            elif model == "svm": # Smoothing of hinge: max(0, 1-x)
                eta = y_ # psi(1)=0 (the proper label)
                psi = lambda z: cp.pos(1 - cp.multiply(y_, z))

            return psi(theta) - psi(eta) - cp.multiply(y_, (theta - eta)) if model != "svm" else psi(theta) #- psi(eta) + cp.multiply(y_, (theta - eta))
            # psi_tilda_inv_y = y_ if model != "svm" else 0 # g = 0 (subgradient=0 always valid for svm)
            # return psi(theta) - psi(eta) - cp.multiply(psi_tilda_inv_y, (theta - eta)) 

        # Variable
        beta = cp.Variable(m)
        z = cp.Variable(n)
        # u = cp.Variable(1)

        # Constraints
        constraints = [
            # cp.logistic(X @ beta) - cp.multiply(y, X @ beta) - cp.logistic(np.log(y / (1-y))) +  cp.multiply(y, np.log(y / (1-y))) <= z
            get_bregman_divergence_cxpy(X, y, beta, model=self.model_name, eps=self.eps) <= z
        ]

        # Fairness constraint
        tau_constraints = [
            # cp.mean( cp.logistic(X[idx_g, :] @ beta) - cp.multiply(y[idx_g], X[idx_g, :] @ beta) - cp.logistic(np.log(y[idx_g] / (1-y[idx_g]))) +  cp.multiply(y[idx_g], np.log(y[idx_g] / (1-y[idx_g]))) ) <= tau_bound for idx_g in bin_indices_list
            cp.mean( get_bregman_divergence_cxpy(X[idx_g, :], y[idx_g], beta, model=self.model_name, eps=self.eps) ) <= tau_bound for idx_g in bin_indices_list
            # cp.mean( cp.logistic(X[idx_g, :] @ beta) - cp.multiply(y[idx_g], X[idx_g, :] @ beta) ) <= u for idx_g in bin_indices_list
        ]
        constraints+=tau_constraints

  
        # Objective
        if self.objective == "mse":
            if self.l2_lambda != 0:
                print("Solving with Ridge objective...")
                # obj = cp.Minimize(cp.mean(z) + self.l2_lambda * cp.quad_form(beta, np.eye(m)))
                # Correction: On the same hypothesis space
                # constraints+=[ cp.quad_form(beta, np.eye(m)) <= np.linalg.norm(beta_0)**2 ] # bound the regularization by the same level
                constraints+=[ cp.SOC(np.linalg.norm(beta_0), beta) ]
                obj = cp.Minimize(cp.mean(z))
            else: 
                obj = cp.Minimize(cp.mean(z))
                # obj = cp.Minimize(u)
        # elif self.objectiv == "mae":
        #     obj = cp.Minimize(cp.mean(z))

        primal_prob = cp.Problem(obj, constraints)

        # Solve the optimization problem
        t0 = time()
        try:
            result = primal_prob.solve(solver=self.solver, verbose=False)
        except cp.error.SolverError:
            print(f"{self.solver} not available, trying default solver.")
            result = primal_prob.solve(verbose=False)
        solve_time = time() - t0
        print(f"Problem status: {primal_prob.status}")
        print(f"Optimal objective: {result}")

        if self.objective == "mse" and self.l2_lambda == 0:
            print(f"J_0 objective (original loss): {J_0}")
            print(f"J_F objective (current loss): {result}")
            price_of_fairness = (result-J_0)/J_0
            print(f"POF (Divergence Loss % decrease): ", price_of_fairness)
        elif self.objective == "mse": # [PENDING] Update the l2 version
            # J_0, result = get_bregman_divergence_mean_value(X, y, beta_0, model=self.model_name, eps=self.eps), get_bregman_divergence_mean_value(X, y, beta.value, model=self.model_name, eps=self.eps)
            print(f"J_0 objective (original loss): {J_0}")
            print(f"J_F objective (current loss): {result}")
            price_of_fairness = (result-J_0)/J_0
            print(f"POF (Divergence Loss % decrease): ", price_of_fairness)
            acc_F = np.average(y == np.sign(X @ beta.value)) 
            print(f"POF Accuracy: ", (acc_0 - acc_F) / acc_0)
            # pass 


        # Approximating the F function
        real_tau, real_tau_g = 0, -1
        real_taus, real_tau_gs = [], []

        b_d_F = get_bregman_divergence_mean_value(X, y, beta.value, model=self.model_name, eps=self.eps)
        print("POF (bregman divergence?): ", (b_d_F - b_d_0) / b_d_0)
        print("New (F) Bregman divergence 0: ", b_d_F)
        # print("New F Accuracy: ", acc_F)
        # print("Direct Taylor of MSE / 2: ", (beta.value - beta_0).T @ H_0 @ (beta.value - beta_0) / 2)
        bregman_g = dict()
        accuracy_g = dict()
        group_sizes = []
        for g, ind_g in enumerate(bin_indices_list):
            X_g, y_g = X[ind_g, :], y[ind_g]
            # print("Group: ", g, len(ind_g))
            loss_g = get_loss_value(X_g, y_g, beta.value, model=self.model_name)#cp.mean( cp.logistic(theta_g) - cp.multiply(y[ind_g], theta_g) ).value
            # print("New (F) group loss g: ", g, loss_g)
            # approx_error_g = np.mean( np.abs( np.exp(theta_g.value)/(1 + np.exp(theta_g.value)) - y_g ) ) 
            # print("New (F) Accuracy: ", np.average(y_g == np.round(get_psi_derivatives(X_g, y_g, beta_0, model=self.model_name, gamma=self.eps)[0])))
            y_glm_pred = get_psi_derivatives(X_g, y_g, beta.value, model=self.model_name, gamma=self.eps)[0]
            acc = r2_score(y_g, y_glm_pred)
            accuracy_g[g] = acc
            print("New (F) Accuracy 0_g: ", acc)
            b_d = get_bregman_divergence_mean_value(X_g, y_g, beta.value, model=self.model_name, eps=self.eps)#np.mean( np.log( 1 + np.exp(theta_g) ) - y_g * theta_g - (np.log(1+ np.exp(eta_y_g)) - y_g * eta_y_g) )
            bregman_g[g] = b_d
            group_sizes.append(y_g.size)
            print("New (F) Bregman divergence g: ", g, b_d)
            if np.abs(b_d - real_tau) <= self.eps:
                print("There are at least 2 active groups!!!!")
                # STOPPING CONDITION (cannot be more fair)
                print(g, g_max)
                if g_max in real_tau_gs:
                    print("THE MODEL CANNOT BE MORE FAIR!!")
                    self.beta = beta.value
                    return result, solve_time, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, real_tau
                real_taus.append(b_d)
                real_tau_gs.append(g)
                print("Argmax g's: ", real_tau_gs)

            elif b_d >= real_tau:
                real_tau = b_d
                real_tau_g = g    
                real_taus = [b_d]
                real_tau_gs = [g]

        # Metrics to return
        metrics_output = [bregman_g, accuracy_g]


        # 1) Fairness improvement approximation and bounds
        fairness_improvement = np.abs(real_tau - tau)
        delta_fairness = self.percentage_increase*tau # tau - real_tau

        # Fairness improvement
        fairness_effective_improvement = delta_fairness/tau
        print(f"FEI (% improvement)", fairness_effective_improvement)

        # if delta_fairness == 0:
        #     return result, solve_time, price_of_fairness, fairness_improvement, 0, 0, 0, 0, 0, real_tau 
        # virtual_fairness_improvement = np.abs(tau * (1 - self.percentage_increase) - tau)


        # Bounds on POF from MSE and MSE
        if self.objective == "mse" and self.constraint == "max_mse":

            # # # MULTIPLE ACITVE SETS: PENDING
            # print("ACTIVE GROUP: ", g_0)
            # # a_0_g_list = []
            # print("delta:", delta_fairness)
            # for g, indices_g in enumerate(bin_indices_list):
            #     n_g, X_g, y_g = len(indices_g), X[indices_g,:], y[indices_g]

            #     # Taylor approximation "bounds" (not secured to be bounds, is just the approximation)
            #     w_0_g, w_0_2_g = get_psi_derivatives(X_g, y_g, beta_0, model=self.model_name)
            #     a_0_g = (1/n_g) * X_g.T @ (w_0_g - y_g )
            #     H_0_g = (1/n_g) * X_g.T @ np.diag(w_0_2_g) @ X_g
            #     H_0_g_inv = np.linalg.pinv(H_0_g)

            #     # Compute best direction for this g
            #     A_0 = a_0_g.T @ H_0_inv @ a_0_g 
            #     d = -(delta_fairness) * H_0_inv @ a_0_g / A_0 # Delta beta*

            #     for g_, indices_g_ in enumerate(bin_indices_list):
            #         n_g_, X_g_, y_g_ = len(indices_g_), X[indices_g_,:], y[indices_g_]
            #         w_0_g_, w_0_2_g_ = get_psi_derivatives(X_g_, y_g_, beta_0, model=self.model_name)
            #         a_0_g_ = (1/n_g_) * X_g_.T @ (w_0_g_ - y_g_ )
            #     # J_0_g = get_bregman_divergence_mean_value(X_g, y_g, beta_0, self.model_name, self.eps) # tau - J_g(beta_0)
            #     # print("LB on J_g(beta_0 + delta) prev: ", J_0_g + a_0_g @ d)
            #     # delta_g = tau - J_0_g
            #     # d = -(delta_g) * H_0_inv @ a_0_g / A_0 # Delta beta*
            #         print("Constraint: ", (g,g_), a_0_g_ @ d, -delta_fairness)

            #     # Approximation of delta J
            #     print("Prining group: ", g, " - ",n_g)
            #     print("delta J = d.T @ H_0 @ d / 2 =", (d.T @ H_0 @ d)/2)
            #     # print("Constraint: ", a_0_g @ d, -delta_fairness)
            # # exit()
                # print("LB on J_g(beta_0 + delta): ", J_0_g + a_0_g @ d)
                # print("LB on J_g(beta_0 + delta) with F: ", tau + a_0_g @ d)

                # a_0_g_list.append(a_0_g)

            # # Combinations of pairs solution
            # a_1 = a_0_g_list[0]
            # a_2 = a_0_g_list[1]
            # a_3 = a_0_g_list[2]

            # # for i,a_i in enumerate([a_1, a_2, a_3]):
            # #     A_0 = a_1.T @ H_0_inv @ a_1 
            # #     print("a_1'd: ",i, a_i  @ (-(delta_fairness) * H_0_inv @ a_1 / A_0) <= -delta_fairness)
            # #     A_0 = a_2.T @ H_0_inv @ a_2 
            # #     print("a_2'd: ",i, a_i  @ (-(delta_fairness) * H_0_inv @ a_2 / A_0) <= -delta_fairness)
            # #     A_0 = a_3.T @ H_0_inv @ a_3 
            # #     print("a_3'd: ",i, a_i  @ (-(delta_fairness) * H_0_inv @ a_3 / A_0) <= -delta_fairness)
            # # s_1 = a_1.T @ H_0_inv @ a_1
            # # s_2 = a_2.T @ H_0_inv @ a_2
            # # s_3 = a_3.T @ H_0_inv @ a_3
            # # s_12 = a_1.T @ H_0_inv @ a_2
            # # s_13 = a_1.T @ H_0_inv @ a_3
            # # s_23 = a_2.T @ H_0_inv @ a_3
            # A_12 = np.array([a_1, a_2]).T
            # A_13 = np.array([a_1, a_3]).T
            # A_23 = np.array([a_2, a_3]).T
            # G_12 = A_12.T @ H_0_inv @ A_12
            # G_13 = A_13.T @ H_0_inv @ A_13
            # G_23 = A_23.T @ H_0_inv @ A_23
            # d_12 = -delta_fairness * H_0_inv @ A_12 @ np.linalg.pinv(G_12) @ np.ones(2)
            # d_13 = -delta_fairness * H_0_inv @ A_13 @ np.linalg.pinv(G_13) @ np.ones(2)
            # d_23 = -delta_fairness * H_0_inv @ A_23 @ np.linalg.pinv(G_23) @ np.ones(2)
            # print("Combinations of pairs: ")
            # print("d @ H_0 @ d / 2: ", (d_12.T @ H_0 @ d_12)/2)
            # print("d @ H_0 @ d / 2: ", (d_13.T @ H_0 @ d_13)/2)
            # print("d @ H_0 @ d / 2: ", (d_23.T @ H_0 @ d_23)/2)

            # A_123 = np.array([a_1, a_2, a_3]).T
            # G_123 = A_123.T @ H_0_inv @ A_123
            # d_123 = -delta_fairness * H_0_inv @ A_123 @ np.linalg.pinv(G_123) @ np.ones(3)
            # print("Triple combination")
            # print("d @ H_0 @ d / 2: ", (d_123.T @ H_0 @ d_123)/2)            


            # theta_123 = delta_fairness * np.linalg.pinv(A_123.T @ H_0_inv @ A_123) @ np.ones(3)
            # print("Theta 123: ", theta_123)


                # d_H_d = 1e3
                # for s, real_tau_g in enumerate(real_tau_gs):
                #     g_, indices_g_ = real_tau_g, bin_indices_list[real_tau_g]
                #     n_g, X_g, y_g = len(indices_g_), X[indices_g_,:], y[indices_g_]

                #     # Taylor approximation "bounds" (not secured to be bounds, is just the approximation)
                #     w_0_g, w_0_2_g = get_psi_derivatives(X_g, y_g, beta_0, model=self.model_name)
                #     a_0_g = (1/n_g) * X_g.T @ (w_0_g - y_g )
                #     H_0_g = (1/n_g) * X_g.T @ np.diag(w_0_2_g) @ X_g
                #     # H_0_g_inv = np.linalg.pinv(H_0_g)

                #     # Direction d:=Delta beta* from the Taylor approximation
                #     A_0 = a_0_g.T @ H_0_inv @ a_0_g 
                #     d = -(delta_fairness) * H_0_inv @ a_0_g / A_0 # Delta beta*
                #     if d_H_d > d.T @ H_0 @ d: # the one that minimzes ||d||_H0^2
                #         g, indices_g = g_, indices_g_ # update if better
                #         real_tau = real_taus[s]
                #         d_H_d = d.T @ H_0 @ d
                #         # d_H_inv_d = d.T @ H_0_inv @ d
                #         print("Group: ", g)
                #         print("d_H_d: ", d_H_d)
                #         print("a_0 @ d: ", a_0_g @ d)
                #         # print("d H_inv d", d_H_inv_d)
                #         # print("Best so far: ", g, real_tau, d_H_d)
                # exit()
                
            # else:
            g, indices_g = g_0, bin_indices_list[g_0] # This is the only proper LB of the delta F
            # g, indices_g = real_tau_g, bin_indices_list[real_tau_g]
            # g_LB, g_UB = g, g # they are the same best

            # indices_g_LB, indices_g_UB = indices_g, indices_g 
            print("THE CURRENT g: ", g)

            n_g, X_g, y_g = len(indices_g), X[indices_g,:], y[indices_g]

            # Taylor approximation "bounds" (not secured to be bounds, is just the approximation)
            w_0_g, w_0_2_g = get_psi_derivatives(X_g, y_g, beta_0, model=self.model_name)
            a_0_g = (1/n_g) * X_g.T @ (w_0_g - y_g )
            H_0_g = (1/n_g) * X_g.T @ np.diag(w_0_2_g) @ X_g
            H_0_g_inv = np.linalg.pinv(H_0_g)

            # print("Checking feasibility...")
            # delta_beta_ = beta.value - beta_0
            # print("1st order of F, using beta_F: ", a_0_g @ beta.value)
            # print("1st order of R, using beta_F: ", 2*beta_0 @ delta_beta_) # np.linalg.norm(beta_0)**2 + 
            # print("2nd order of R, using beta_F: ", 2*beta_0 @ delta_beta_ + delta_beta_ @ delta_beta_ ) # np.linalg.norm(beta_0)**2 + 
            # print("Exact diff of norms: ", np.linalg.norm(beta.value)**2 - np.linalg.norm(beta_0)**2)
            # print("norm of beta_F: ", np.linalg.norm(beta.value)**2)
            # print("norm of beta_0: ", np.linalg.norm(beta_0)**2)
            # print("Fairness improv: ", -delta_fairness)

            # Direction d:=Delta beta* from the Taylor approximation
            A_0 = a_0_g.T @ H_0_inv @ a_0_g 
            d = -(delta_fairness) * H_0_inv @ a_0_g / A_0 # Delta beta*
            # d = -(fairness_improvement) * H_0_inv @ a_0_g / A_0
            d_H_0_inv_norm = (delta_fairness)**2 / A_0 # norm H_0_inv of Delta beta* (final form of the term)
            # d_H_0_inv_norm = (fairness_improvement)**2 / A_0 # norm H_0_inv of Delta beta* (final form of the term)
            print("delta's")
            print(fairness_improvement)
            print(delta_fairness)
            print("J_0's")
            print(J_0)
            print(glm.train_loss)
            J_0 = glm.train_loss

            # d_old = d
            # t_d = 1 # We move 1 towards d

            # Projection onto the l2-regularized case:
            if self.l2_lambda > 0:
                rho_safe_tol = self.eps#**2
                a_0_g_norm = a_0_g @ a_0_g
                rho_hat = (beta_0 @ beta_0 - rho_safe_tol) * a_0_g_norm
                A_ = (fairness_improvement - beta_0 @ a_0_g) ** 2 - rho_hat
                B_ = 2*A_
                C_ = np.linalg.norm(beta_0 + d)**2 * a_0_g_norm - rho_hat
                theta_2 = (-B_ - np.sqrt(B_**2 - 4 * A_ * C_  )) / (2 * A_) # dual of the norm
                theta_1 = theta_2 * (fairness_improvement - beta_0 @ a_0_g) / a_0_g_norm # dual of the fairness
                d_ = (d - theta_1 * a_0_g - theta_2 * beta_0) / (1 + theta_2) # Feasible on the fairness + regularized feasible set 
                # # Aproximate the regularization penalty
                # w_0_fair, _ = get_psi_derivatives(X, y, beta_0 + d_, model=self.model_name)
                # a_0_fair = (1/n) * X.T @ (w_0_fair - y )
                # l2_lambda_fair = - a_0_fair / (2 * (beta_0 + d_) )
                print("Updated direction (l2-regularization case)")
                print("1st order of R (d): ", 2*beta_0 @ d_) # np.linalg.norm(beta_0)**2 + 
                print("2nd order of R (d): ", 2*beta_0 @ d_ + d_ @ d_ ) # np.linalg.norm(beta_0)**2 +
                print("1st order of F (d): ", a_0_g @ d_ )
                # print("LAMBDA FAIR: ", l2_lambda_fair)

                # # Approximating the lambda: lambda \(\approx \) theta_2. Closed for delta beta
                s_ = a_0_g.T @ H_0_inv @ a_0_g
                t_ = beta_0.T @ H_0_inv @ a_0_g
                u_ = beta_0.T @ H_0_inv @ beta_0
                r_a = a_0_g.T @ H_0_inv @ a_0 # original gradient (nonzero if regularized!)
                r_b = beta_0.T @ H_0_inv @ a_0 # original gradient (nonzero if regularized!)
                D_ = s_ * u_ - t_**2
                theta_1 = (u_ * (fairness_improvement - r_a) + t_* ( r_b - rho_safe_tol )) / D_
                theta_2 = (-t_ * (fairness_improvement - r_a) - s_*( r_b - rho_safe_tol  )) / D_
                d__ = - H_0_inv @ (a_0 + theta_1 * a_0_g + theta_2 * beta_0 )
                print("Updated direction (l2-regularization case)")
                print("1st order of R (d): ", 2*beta_0 @ d__) # np.linalg.norm(beta_0)**2 + 
                print("2nd order of R (d): ", 2*beta_0 @ d__ + d__ @ d__ ) # np.linalg.norm(beta_0)**2 +
                print("1st order of F (d): ", a_0_g @ d__ )
                d = d__
                
            
                # # Most efficient t to move towards the new d (if it exists)
                # t_d = -2* beta_0 @ d / (d @ d)#(beta_0 @ beta_0)
                # print("Move towards d: t=", t_d, "?")
                # if t_d >= 1: # ensures fairness constraint
                #     print("Moving towards d: t=", t_d, "...")
                #     d *= t_d # update efficient d
                #     print("1st order of R (d): ", 2*beta_0 @ d) # np.linalg.norm(beta_0)**2 + 
                #     print("2nd order of R (d): ", 2*beta_0 @ d + d @ d ) # np.linalg.norm(beta_0)**2 +
                #     print("1st order of F (d): ", a_0_g @ d )


            # d_H_0_inv_norm = d.T @ H_0 @ d #(delta_fairness)**2 / A_0 # norm H_0_inv of Delta beta* (final form of the term)
            # Hessian of J_g
            d_H_0_norm_g = d.T @ H_0_g @ d  # norm H_0_inv of Delta beta* (Computed directly with beta* instead of the previous one)

            # Taloy Approximation Bounds setting t=1, and d:=Delta beta*
            if M_psi > 0 and (d @ d) > 0: # M_phi -> 0 => C(t) -> 1/2
                M_phi = M_psi * np.max(np.abs( X @ d )) # Option 2 w./ Cauchy Schwarz (looser): M_psi * np.max(np.linalg.norm(X, axis=1))*np.linalg.norm(d)
                print("M_phi", M_phi)
                C_phi_LB = (np.exp(-M_phi) + M_phi - 1) / M_phi**2  # t = 1 (unless regularizer changes it)
                C_phi_UB = (np.exp( M_phi) - M_phi - 1) / M_phi**2  # t = 1 (unless regularizer changes it)
            else:
                M_phi, C_phi_LB, C_phi_UB = 0, .5, .5 # limit values: M_phi -> 0 => C(t) -> 1/2
            # lambda_diff_norms = self.l2_lambda * ((beta_0 + d) @ (beta_0 + d) - beta_0 @ beta_0) # zero if l2_lambda = 0
            delta_J_taylor = (1/2) * d_H_0_inv_norm #+ lambda_diff_norms
            delta_J_taylor_lb = C_phi_LB * d_H_0_inv_norm #+ lambda_diff_norms # Lower bound
            # delta_J_taylor_ub = C_phi_UB * d_H_0_inv_norm #+ lambda_diff_norms # Lower bound

            print(fr"POF J % Taylor (lin-const + taylor obj.): ", delta_J_taylor / J_0)
            print(fr"POF J % LB (lin-const + exp term.): ", delta_J_taylor_lb / J_0)
            # print(fr"POF J % Taylor UB (mse-max_mse): ", delta_J_taylor_ub / J_0)

            # Lin. + quad UB construction (model-dependent).
            print("AAAAAAAAAA") 
            print(self.model_name)
            if self.model_name == "linear":
                # Taylor constraint: t(a_0_g ' d) + t^2/2 ||d||_{H_0_g}^2 <= -delta
                # a, b, c = d_H_0_norm_g / 2, -delta_fairness, delta_fairness
                a, b, c = d_H_0_norm_g / 2, a_0_g @ d, delta_fairness
                print("a,b,c: ", (a, b, c))
                print("a_0 ' d", a_0_g @ d)
                if b**2 >= 4*a*c and a>0:
                    t_UB_1, t_UB_2 = (-b  - np.sqrt(b**2 - 4*a*c )) / (2*a), (-b  + np.sqrt(b**2 - 4*a*c )) / (2*a)
                    print("Roots for t UB: ", (t_UB_1, t_UB_2))
                    t_UB = min(max(t_UB_1,0), max(t_UB_2,0))
                    delta_J_taylor_ub = (t_UB**2/2) * d_H_0_inv_norm
                    print(fr"POF J % UB (taylor (const. + obj.): ", delta_J_taylor_ub / J_0)
                    print(fr"Exact delta F lb: ", t_UB*b + t_UB**2*a)
                else:
                    delta_J_taylor_ub = np.nan
            elif self.model_name == "logistic":
                # (1) Quadratic is UB with 1/4 of psi'': d'X'D^(1/2)'D^(1/2)Xd <= ||D^(1/2)Xd||^2 <= (1/4)||Xd||^2 
                H_0_ub = (1/n)/4 * X.T @ X
                H_0_g_ub = (1/n_g)/4 * X_g.T @ X_g
                d_H_0_norm_ub = d.T @ H_0_ub @ d
                d_H_0_norm_g_ub = d.T @ H_0_g_ub @ d
                a, b, c = d_H_0_norm_g_ub / 2, -delta_fairness, delta_fairness # From the fairness equality
                print("a,b,c: ", (a, b, c))
                # (2) Sum of L-smooth: 
                # if 1/8 < a:
                #     a=1/8 if (d@d) >= 1 else (d@d)/8  # Much looser bounds from the sum of L-smooth(?)
                #     print("a,b,c: ", (a, b, c))
                if b**2 >= 4*a*c and a>0:
                    t_UB_1, t_UB_2 = (-b  - np.sqrt(b**2 - 4*a*c )) / (2*a), (-b  + np.sqrt(b**2 - 4*a*c )) / (2*a)
                    print("Roots for t UB: ", (t_UB_1, t_UB_2))
                    t_UB = min(max(t_UB_1,0), max(t_UB_2,0))
                    delta_J_taylor_ub = t_UB**2 * a
                    # Future Note: constant can be either from exponential or from upper (already account for the 1/4 UB)
                    print(fr"POF J % UB (taylor (const. + obj.): ", delta_J_taylor_ub / J_0)
                    print(fr"POF J % UB 2 (taylor (const. + obj.): ", (t_UB**2/8) / J_0)
                    print(fr"POF J % UB 3 (taylor (const. + obj.): ", (t_UB**2/8) * (d@d) / J_0)
                else:
                    delta_J_taylor_ub = 0 if (a==0 and b==0 and c==0) else np.nan # infeasible => inf
            elif self.model_name == "poisson":
                delta_J_taylor_ub = np.nan#np.zeros(delta_J_taylor.size)
                # # Given the experiments, we are setting t\in[0,2] for now
                # t_ub = 2 # UB
                # psi_UB = np.exp(X @ (beta_0 + t_ub * d) )
                # psi_UB_g = np.exp(X_g @ (beta_0 + t_ub * d) ) 
                # a_0_ub = (1/n) * X.T @ (psi_UB - y )
                # H_0_ub = (1/n) * X.T @ np.diag(psi_UB) @ X
                # a_0_ub_g = (1/n_g) * X_g.T @ (psi_UB_g - y_g )
                # H_0_ub_g = (1/n_g) * X_g.T @ np.diag(psi_UB_g) @ X_g
                # h = lambda x: cp.quad_form(x, 1)/2 * d.T @ (H_0_ub) @ d
                # nabla_h = lambda x: x * (H_0_ub) @ d
                # h_g = lambda x: cp.quad_form(x, 1) / 2 * d.T @ (H_0_ub_g) @ d
                # nabla_h_g = lambda x: x * (H_0_ub_g) @ d

                # # Variable
                # t_UB = cp.Variable(1, nonneg=True)
                # # Constraint
                # print("-"*100)
                # print("tau_bound - tau: ", tau_bound - tau)
                # print("-"*100)
                # constraints=[t_UB * (a_0_g @ d) + h_g(t_UB) - h_g(0)  <= tau_bound - tau] #-self.percentage_increase]
                # obj = cp.Minimize( t_UB * (a_0 @ d) + h(t_UB) - h(0) - t_UB * nabla_h(0) @ d )
                # # Objective 
                # primal_prob = cp.Problem(obj, constraints)
                # # delta_J_taylor_ub = t_UB.value * (a_0_g @ d) + h(t_UB) - h(0) 
                # delta_J_taylor_ub = primal_prob.solve(solver=self.solver, verbose=False)

                # print("-"*50)
                # print(fr"POF J % UB (taylor (const. + obj.): ", delta_J_taylor_ub / J_0)
                # print("-"*50)
                pass # no upper bound for this one(?)
            
            # 2) The proper Lower Bound bound with Newton Raphson/Bijection/Opt (1 dimension)
            # d := Delta beta*
            if M_psi > 0 and (d@d)>0:
                M_phi_g = M_psi * np.max(np.abs( X_g @ d ))
                C_phi_LB_g = (np.exp(-M_phi_g) + M_phi_g - 1) / M_phi_g**2  # t = 1
                C_phi_UB_g = (np.exp( M_phi_g) - M_phi_g - 1) / M_phi_g**2  # t = 1
            else:
                M_phi_g, C_phi_LB_g, C_phi_UB_g = 0, .5, .5 # limit values
            print("-"*100)
            print("M_phi: ", M_phi_g)
            print("M_phi_g: ", M_phi)
            print("C_phi_LB_g: ", C_phi_LB_g)
            print("C_phi_UB_g: ", C_phi_UB_g)
            print("-"*100)


            # Roots finder for the LB: Newton Raphson/Bijection/Opt (1 dimension)
            # Min_{t>=0} C_phi_LB(t) * d_H_inv_norm (Delta J)
            # s.t. t*(a_0_g * Delta beta*) + d_H_0_norm_g * C_phi_LB_g(t) <= -delta (Delta F)
            print("-"*100)
            print("tau_bound - tau: ", tau_bound - tau)
            print("fairness delta: ", delta_fairness)
            print("-"*100)
            print("Solving exponential LB...")
            # Variable
            t_LB = cp.Variable(1, nonneg=True)
            # d = d_ # this one could be feasible??
            # Constraint
            if M_psi > 0 and (d @ d) > 0:
                print("Fully exponential...")
                print("a0g @ d: ", a_0_g @ d)
                print("a0 @ d: ", a_0 @ d)
                constraints=[t_LB * (a_0_g @ d)  + d_H_0_norm_g * ( cp.exp(-M_phi_g*t_LB) + t_LB*M_phi_g - 1 ) / M_phi_g**2  <= -delta_fairness ]#tau_bound - tau] #-self.percentage_increase] #
                obj = cp.Minimize( d_H_0_inv_norm * (cp.exp(-M_phi*t_LB) + t_LB*M_phi - 1 ) / M_phi**2 + (self.l2_lambda > 0) * (t_LB * a_0 @ d ))
            else: 
                print("Linear constrained...")
                constraints=[t_LB * (a_0_g @ d)  <= -delta_fairness ] #tau_bound - tau] # + t_LB ** 2 * d_H_0_norm_g / 2 
                obj = cp.Minimize(t_LB **2 * d_H_0_inv_norm / 2  + (self.l2_lambda > 0) * (t_LB * a_0 @ d ))
            # if self.l2_lambda > 0: # regularization constraint
            #     # constraints+=[cp.SOC(np.linalg.norm(beta_0), beta_0 + t_LB * d )]
            #     print("b0 @ d: ", beta_0 @ d)
            #     constraints+=[2 * t_LB * (beta_0 @ d)  <= 0 ] #-self.percentage_increase] #  + t_LB**2 * (d @ d) # still LB with the linearization
            #     # obj = cp.Minimize( d_H_0_inv_norm * (cp.exp(-M_phi*t_LB) + t_LB*M_phi - 1 ) / M_phi**2 )
            # # Objective 
            bound_prob = cp.Problem(obj, constraints)
            # Solve the roots
            t0 = time()
            try:
                delta_J_lb = bound_prob.solve(solver=self.solver, verbose=False)
            except cp.error.SolverError:
                print(f"{self.solver} not available, trying default solver.")
                delta_J_lb = bound_prob.solve(verbose=True)
            print(fr"POF J % LB (exp): ", delta_J_lb / J_0)
            print(f"Root find for t_LB={t_LB.value} in: ", time() - t0)

                
            # EXPONENTIAL UPPER BOUND
            # d = d__
            d_H_0_inv_norm = d.T @ H_0 @ d #(delta_fairness)**2 / A_0 # norm H_0_inv of Delta beta* (final form of the term)
            # Hessian of J_g
            d_H_0_norm_g = d.T @ H_0_g @ d  # norm H_0_inv of Delta beta* (Computed directly with beta* instead of the previous one)
            M_phi = M_psi * np.max(np.abs( X @ d )) if M_psi > 0 else 0
            M_phi_g = M_psi * np.max(np.abs( X_g @ d )) if M_psi > 0 else 0

            # Roots finder for the UB: Newton Raphson/Bijection/Opt (1 dimension)
            # Min_{t>=0} C_phi_UB(t) * d_H_inv_norm (Delta J)
            # s.t. t*(a_0_g * Delta beta*) + d_H_0_norm_g * C_phi_UB_g(t) <= -delta (Delta F)
            print("Solving exponential UB...")
            # Variable
            t_UB = cp.Variable(1, nonneg=True)
            if M_psi > 0 and (d@d) >0:
                print("Fully exponential...")
                print("a0 @ d: ", a_0_g @ d)
                print("M_phi: ", M_phi)
                print("M_phi_g: ", M_phi_g)
                # Constraint
                constraints=[t_UB * (a_0_g @ d) + d_H_0_norm_g * (cp.exp(M_phi_g*t_UB) - t_UB * M_phi_g - 1 ) / M_phi_g**2  <= -delta_fairness ]#tau_bound - tau] #-self.percentage_increase]
                # Objective 
                obj = cp.Minimize( (self.l2_lambda > 0) * t_LB * (a_0 @ d) + d_H_0_inv_norm * (cp.exp(M_phi*t_UB) - t_UB * M_phi - 1 ) / M_phi**2 ) # 
            else:
                print("...?")
                # Constraint
                constraints=[t_UB * (a_0_g @ d) +  t_UB **2 * d_H_0_norm_g / 2  <= -delta_fairness ]#tau_bound - tau] #-self.percentage_increase]
                # Objective 
                obj = cp.Minimize( (self.l2_lambda > 0) * t_LB * (a_0 @ d) + t_UB **2 * d_H_0_inv_norm / 2 )
            # if self.l2_lambda > 0: # regularization constraint
            #     print("b0 @ d: ", 2 * beta_0 @ d)
            #     print("(d @ d): ", (d @ d))
                # constraints+=[cp.SOC(np.linalg.norm(beta_0), beta_0 + t_UB * d )]
                constraints+=[ 2 * t_UB * (beta_0 @ d) + t_UB**2 * (d @ d) <= 0 ] #-self.percentage_increase] #   # still LB with the linearization # 
            #     # obj = cp.Minimize( d_H_0_inv_norm * (cp.exp(-M_phi*t_UB) + t_UB*M_phi - 1 ) / M_phi**2 )
            bound_prob = cp.Problem(obj, constraints)
            # Solve the roots
            t0 = time()
            try:
                delta_J_ub = bound_prob.solve(solver=self.solver, verbose=False)
            except cp.error.SolverError:
                print(f"{self.solver} not available, trying default solver.")
                delta_J_ub = bound_prob.solve(verbose=True)
            print(fr"POF J % UB (exp): ", delta_J_ub / J_0)
            print(f"Root find for t_UB={t_LB.value} in: ", time() - t0)


        print(f"Time to solve: {solve_time}")

        # Print the difference in MAE between groups post-optimization
        if primal_prob.status in ["optimal", "optimal_inaccurate"]:
            self.beta = beta.value
        else:
            print("Solver did not find an optimal solution. Beta coefficients not set.")
            self.beta = np.zeros(m) # Fallback beta

        return result, solve_time, price_of_fairness, fairness_effective_improvement, delta_J_lb/ J_0, delta_J_ub/ J_0, delta_J_taylor/ J_0, delta_J_taylor_lb / J_0, delta_J_taylor_ub / J_0, real_tau, metrics_output, group_sizes

    def predict(self, X):
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        return X @ self.beta
    
    def __str__(self): #add_rmse_constraint={self.add_rmse_constraint},
        return f"GroupDeviationConstrainedLinearRegression(fit_intercept={self.fit_intercept},  percentage_increase={self.percentage_increase}, n_groups={self.n_groups})"
















class LeastProportionalDeviationRegression:
    def __init__(self, fit_intercept=True, add_rmse_constraint=False, percentage_increase=0.00, solver="GUROBI"):
        self.beta = None
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.add_rmse_constraint = add_rmse_constraint
        self.percentage_increase = percentage_increase


    def fit(self, X, y):
        X, y = X.to_numpy(), y.to_numpy()
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        n, m = X.shape

        # OLS solution
        model = LinearRegression(fit_intercept=False)
        model.fit(X, y)
        beta_t = model.coef_
        y_pred_ols = model.predict(X)
        r_t = np.abs(y - y_pred_ols)
        ols_mse = np.mean(r_t**2)

        # Fairness measure
        n_groups = 3
        interval_size = y.max() - y.min() + 1e-6 # a little bit more
        bins = y.min() + np.array([i*interval_size /n_groups for i in range(n_groups+1)])
        X_bins, y_bins = [], []
        bin_indices_list = []
        for j,lb in enumerate(bins[:-1]):
            ub = bins[j+1]
            bin_indices = np.where((y>=lb) & (y<ub))[0]
            bin_indices_list.append(bin_indices)

        # Compute the actual max difference
        F_ols = 0 
        for i in range(n_groups):
            for j in range(i+1, n_groups):
                diff_ij = np.abs(np.mean(r_t[bin_indices_list[i]]) - np.mean(r_t[bin_indices_list[j]]))
                if diff_ij >  F_ols:
                    F_ols = diff_ij

        def constrained_version(X, y, W):
            beta = cp.Variable(m)
            residuals = y - X @ beta

            # RMSE bound constraint (ONLY AVAILABLE FOR SAME TRAIN SIZE)
            model = LinearRegression(fit_intercept=False)
            model.fit(X, y)
            min_rmse = root_mean_squared_error(y, model.predict(X))
            constraints=[
                cp.SOC((1+self.percentage_increase)* min_rmse * np.sqrt(n), residuals),
            ]
            # Objective: <=> Minimize |r_max - r_min |
            primal_prob = cp.Problem(
                cp.Minimize(cp.quad_form(residuals, W)), 
                constraints
            )
            # Solve the optimization problem
            try:
                result = primal_prob.solve(solver=self.solver, verbose=False)
            except cp.error.SolverError:
                print(f"{self.solver} not available, trying default solver.")
                result = primal_prob.solve(verbose=False)

            print(f"Problem status: {primal_prob.status}")
            print(f"Optimal objective (Mean Absolute Error): {result}")

            # Print the difference in MAE between groups post-optimization
            if primal_prob.status in ["optimal", "optimal_inaccurate"]:
                self.beta = beta.value
                # solve_time = primal_prob.solver_stats.solve_time
            else:
                print("Solver did not find an optimal solution. Beta coefficients not set.")
                self.beta = np.zeros(m) # Fallback beta

            return beta.value

        # Iteratively Reweighted Least Squares
        t0 = time()
        eps = 1
        for t in range(1,100):

            print("Starting iteration ", t)

            # Iteratively solve the weigts and betas updates
            w_t = 1/(r_t**2 + eps)
            W_t = np.diag(w_t)

            if self.add_rmse_constraint:
                # Solve min weighted squares (CONSTRAINED)
                # beta_sol = constrained_version(X, y, W_t)
            #     0 = 2 beta @ X @ W_t @ X - 2 X @ W_t @ y + lamb * ( 2 beta @ X @ X - 2 X @ y ) 
            #  X @ W_t @ y + lamb * (X @ y) = beta @ ( X @ W_t @ X + lambd * X @ X)
                beta_sol = np.linalg.pinv(X.T @ W_t @ X + self.percentage_increase * X.T @ X) @ (X.T @ W_t @ y + self.percentage_increase * X.T @ y )
            else:
                # Solve min weighted squares (UNCONSTRAINED)
                beta_sol = np.linalg.pinv(X.T @ W_t @ X) @ ( X.T @ W_t @ y )

            # Stopping criteria
            if np.linalg.norm(beta_sol - beta_t) < 1e-4:
                solve_time = time() - t0
                print("Solution encountered!! Diff: ", np.linalg.norm(beta_sol - beta_t))
                result = np.mean(np.log(np.abs(y - X @ beta_sol) + eps))
                print(f"Optimal objective: {result}")
                self.beta = beta_sol 
                prop_mse = root_mean_squared_error(y, X @ beta_sol)
                pof = (prop_mse - ols_mse) / ols_mse
                print(f"Price of Fairness (MSE % increase): ", pof)

                F_prop = 0 
                z_abs = np.abs(y - X @ beta_sol) 
                for i in range(n_groups):
                    for j in range(i+1, n_groups):
                        diff_ij = np.abs(np.mean(z_abs[bin_indices_list[i]]) - np.mean(z_abs[bin_indices_list[j]]))
                        if diff_ij >  F_prop:
                            F_prop = diff_ij
                efi = (F_ols - F_prop )/F_ols
                print(r"Effective Fairness Improvement (F_group % decrease)", efi)
                break

            # If not met, update
            print("Solution not ecountered. Diff: ", np.linalg.norm(beta_sol - beta_t))
            beta_t = beta_sol
            r_t = np.abs(y - X @ beta_sol)
            print("Current objective: ", np.mean(np.log(r_t + eps)))

        return result, solve_time, pof, None, None

    def predict(self, X):
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        return X @ self.beta
    
    def __str__(self):
        return f"LeastProportionalDeviationRegression(fit_intercept={self.fit_intercept}, add_rmse_constraint={self.add_rmse_constraint}, percentage_increase={self.percentage_increase})"





class LeastMSEConstrainedRegression:
    def __init__(self, fit_intercept=True, add_rmse_constraint=False, percentage_increase=0.00, solver="GUROBI"):
        self.beta = None
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.add_rmse_constraint = add_rmse_constraint
        self.percentage_increase = percentage_increase


    def fit(self, X, y):
        X, y = X.to_numpy(), y.to_numpy()
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        n, m = X.shape

        # Primal approach of the problem
        beta = cp.Variable(m)
        r_max = cp.Variable(1, nonneg=True)
        # r_min = cp.Variable(1, nonpos=True)
        residuals = y - X @ beta
        constraints = [
            residuals  <= r_max,
            -residuals  <= r_max,
            # residuals  >= r_min,
            # -residuals  >= r_min,
        ]

        # RMSE bound constraint (ONLY AVAILABLE FOR SAME TRAIN SIZE)
        beta_rmse = np.linalg.inv(X.T @ X) @ (X.T @ y)
        res = np.abs(y - X @ beta_rmse) # Equivalent to adding the "n*" in the constraint
        if self.add_rmse_constraint:
            constraints +=[
                r_max <= (1+self.percentage_increase) * (np.max(res) - np.min(res))
            ]

        # Objective: <=> Minimize |r_max - r_min |
        I_n = np.eye(n)
        primal_prob = cp.Problem(
            cp.Minimize(cp.quad_form(residuals, I_n)), 
            constraints
        )

        # Solve the optimization problem
        try:
            result = primal_prob.solve(solver=self.solver, verbose=False)
        except cp.error.SolverError:
            print("GUROBI not available, trying default solver.")
            result = primal_prob.solve(verbose=False)

        print(f"Problem status: {primal_prob.status}")
        print(f"Optimal objective (Mean Absolute Error): {result}")

        # Print the difference in MAE between groups post-optimization
        if primal_prob.status in ["optimal", "optimal_inaccurate"]:
            self.beta = beta.value
            solve_time = primal_prob.solver_stats.solve_time
        else:
            print("Solver did not find an optimal solution. Beta coefficients not set.")
            self.beta = np.zeros(m) # Fallback beta


        return result, solve_time

    def predict(self, X):
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        return X @ self.beta
    
    def __str__(self):
        return f"LeastMaxMinDeviationRegression(fit_intercept={self.fit_intercept})"








class ProportionalAbsoluteRegression:
    def __init__(self, fit_intercept=True, solver="GUROBI", max_iters=100, tol=1e-4, solve_method="bounded_residual", batch_size=1024, step_size=1e-3, bound_perc=1):
        self.beta = None
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.max_iters = max_iters
        self.tol = tol
        self.solve_method = solve_method
        self.batch_size = batch_size
        self.step_size = step_size
        self.bound_perc = bound_perc

    def fit(self, X, y):
        X, y = X.to_numpy(), y.to_numpy()
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        n, m = X.shape

        # initial solution
        model = LinearRegression(fit_intercept=self.fit_intercept)
        model.fit(X,y)
        beta_k = model.coef_
        ols_mse = root_mean_squared_error(y, X @ beta_k)**2

        total_solve_time = 0

        for k in range(self.max_iters):

            
            print("-"*50)
            print(f"Starting iteration #: {k+1}")

            res_k = y - X @ beta_k
            res_tau = np.max(np.abs(res_k)) # upper bound on residuals
            if self.solve_method == "exact_derivation":
                # Closed form solution for t
                t_k = 1/(np.abs(res_k) + 1e-2)

                # Primal approach of the sub-problem
                beta = cp.Variable(m)
                u = cp.Variable(n, nonneg=True)
                l = cp.Variable(n, nonneg=True)
                constraints = [
                    X @ beta + u - l == y
                ]

                # Objective: <=> Minimize the overall Mean Absolute Error
                primal_prob = cp.Problem(
                    cp.Minimize(t_k @ (u + l)), 
                    constraints
                )

                # Solve the optimization problem
                try:
                    result = primal_prob.solve(solver=self.solver, verbose=False)
                except cp.error.SolverError:
                    print("GUROBI not available, trying default solver.")
                    result = primal_prob.solve(verbose=False)

                print(f"Problem status: {primal_prob.status}")
                print(f"Optimal objective (Mean Absolute Error): {result}")

                # Print the difference in MAE between groups post-optimization
                if primal_prob.status in ["optimal", "optimal_inaccurate"]:
                    solve_time = primal_prob.solver_stats.solve_time
                    print(f"Iteration {k+1} took: {solve_time:.2f}s")
                    total_solve_time += solve_time
                    diff_k = np.linalg.norm(beta.value - beta_k) 
                    if diff_k <= self.tol:
                        print(f"Optimal solution found in iteration #{k+1}")
                        self.beta = beta.value
                        break
                    else:
                        print(f"Current diff: {diff_k:.2f}")
                        beta_k = beta.value
                else:
                    print("Solver did not find an optimal solution. Beta coefficients set to the last iteration...")
                    self.beta = beta_k
                    break


            if self.solve_method == "stochastic_exact_derivation":
                # Closed form solution for t, calculated on the full dataset
                t_k = 1/(np.abs(res_k) + 1e-4)

                # --- Mini-Batch Update Logic ---
                # Shuffle data for this iteration to ensure random batches
                indices = np.arange(n)
                np.random.shuffle(indices)
                X_shuffled, y_shuffled, t_k_shuffled = X[indices], y[indices], t_k[indices]

                batch_betas = []
                iteration_solve_time = 0

                # Iterate over the data in mini-batches
                for i in range(0, n, self.batch_size):
                    # Slice the data to create a mini-batch
                    X_batch = X_shuffled[i:i+self.batch_size]
                    y_batch = y_shuffled[i:i+self.batch_size]
                    t_k_batch = t_k_shuffled[i:i+self.batch_size]
                    
                    # Define the sub-problem for the current batch
                    beta_batch = cp.Variable(m)
                    u_batch = cp.Variable(X_batch.shape[0], nonneg=True)
                    l_batch = cp.Variable(X_batch.shape[0], nonneg=True)
                    constraints_batch = [
                        X_batch @ beta_batch + u_batch - l_batch == y_batch
                    ]

                    primal_prob_batch = cp.Problem(
                        cp.Minimize(t_k_batch @ (u_batch + l_batch)), 
                        constraints_batch
                    )

                    # Solve the optimization problem for the batch
                    try:
                        primal_prob_batch.solve(solver=self.solver, verbose=False)
                    except cp.error.SolverError:
                        primal_prob_batch.solve(verbose=False)

                    if primal_prob_batch.status in ["optimal", "optimal_inaccurate"]:
                        batch_betas.append(beta_batch.value)
                        iteration_solve_time += primal_prob_batch.solver_stats.solve_time
                
                # After processing all batches, check if any were successful
                if not batch_betas:
                    print("Solver did not find an optimal solution in any batch. Stopping.")
                    self.beta = beta_k # Revert to the last known good beta
                    break

                # Average the betas from all successful batches to get the update
                beta_avg = np.mean(batch_betas, axis=0)
                
                print(f"Iteration {k+1} took: {iteration_solve_time:.2f}s")
                total_solve_time += iteration_solve_time
                
                # Check for convergence using the averaged beta
                diff_k = np.linalg.norm(beta_avg - beta_k)
                if diff_k <= self.tol:
                    print(f"Optimal solution found in iteration #{k+1}")
                    self.beta = beta_avg
                    break

                else:
                    print(f"Current diff: {diff_k:.2f}")
                    beta_k = beta_avg


            elif self.solve_method == "gradient_descent":
                t0 = time()
                grad_k = np.zeros(m)
                for i in range(n):
                    grad_k += np.sign(res_k[i])/np.abs(res_k[i] + self.tol)*(-X[i,:])
                beta_k -= self.step_size * grad_k
                solve_time = time() - t0
                print(f"Iteration {k+1} took: {solve_time:.2f}s")
                total_solve_time += solve_time
                grad_norm =  np.linalg.norm(grad_k)
                if grad_norm <= self.tol:
                    print(f"Optimal solution found in iteration #{k+1}")
                    self.beta = beta_k
                    break
                else:
                    print(f"Current norm: {grad_norm:.4f}")
                    if k >= self.max_iters - 1:
                        self.beta = beta_k

            elif self.solve_method == "bounded_residual":
                # Closed form solution for t
                # t_k = 1/(np.abs(res_k) + 1e-2)

                # Primal approach of the sub-problem
                beta = cp.Variable(m)
                u = cp.Variable(n, nonneg=True)
                l = cp.Variable(n, nonneg=True)
                constraints = [
                    X @ beta + u - l == y,
                    # u + l <= res_tau*self.bound_perc, # upper bound on residuals 
                ]

                # Objective: <=> Minimize the overall Mean Absolute Error
                e_n = np.ones(n)
                primal_prob = cp.Problem(
                    cp.Minimize( cp.sum( cp.log( (u + l) + 1) )), 
                    constraints
                )

                # Solve the optimization problem
                try:
                    t0 = time()
                    result = primal_prob.solve(solver=self.solver, verbose=False)
                    solve_time = time() - t0
                except cp.error.SolverError:
                    print("GUROBI not available, trying default solver.")
                    result = primal_prob.solve(verbose=False)

                print(f"Problem status: {primal_prob.status}")
                print(f"Optimal objective (Mean Absolute Error): {result}")

                # Print the difference in MAE between groups post-optimization
                if primal_prob.status in ["optimal", "optimal_inaccurate"]:
                    print(f"Iteration {k+1} took: {solve_time:.2f}s")
                    total_solve_time += solve_time
                    print(f"Optimal solution found in iteration #{k+1}")
                    self.beta = beta.value
                    prop_mse = root_mean_squared_error(y, X @ beta.value) **2
                    pof_prop_reg = (prop_mse - ols_mse)/ols_mse
                    print()
                    break
                else:
                    print("Solver did not find an optimal solution. Beta coefficients set to the last iteration...")
                    self.beta = beta_k
                    break
                
        result = np.prod(np.abs(y - X @ self.beta))

        return result, total_solve_time, pof_prop_reg, None, None

    def predict(self, X):
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        return X @ self.beta
    
    def __str__(self):
        return f"ProportionalAbsoluteRegression(fit_intercept={self.fit_intercept}, max_iters={self.max_iters}, tol={self.tol})"



# After LGBM models
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class FairnessConstrainedRidgeLog(BaseEstimator, RegressorMixin):
    """Ridge regression for log-price targets with a global covariance penalty,
    plus an optional separable (upper-bound) surrogate penalty.

    Objective (sklearn-style scaling):
        (1/(2n)) * ||y - y_hat||^2
        + (alpha/2) * ||beta||^2   (intercept not penalized)
        + (rho/2) * c(beta)^2
        + (rho_sep/2) * s(beta)

    Global covariance moment (same as before):
        c(beta) = (1/n) * sum_i r_i * y_c,i

      mode='diff': r_i = y_hat_i - y_i     (target 0)
      mode='div' : r_i = y_hat_i / y_i^safe (target 1)

    Separable surrogate (Jensen upper bound on c^2), but centered at the right target:
        define r_tilde_i = r_i - t, where t=0 for diff and t=1 for div
        s(beta) := (1/n) * sum_i (r_tilde_i * y_c,i)^2
    """

    def __init__(
        self,
        alpha=1.0,
        rho=1.0,
        rho_sep=0.0,          # coefficient for separable surrogate
        mode="diff",
        fit_intercept=True,
        eps_y=1e-12
    ):
        self.alpha = alpha
        self.rho = rho
        self.rho_sep = rho_sep
        self.mode = mode
        self.fit_intercept = fit_intercept
        self.eps_y = eps_y

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=False, y_numeric=True)
        n, d = X.shape

        if self.mode not in ("diff", "div"):
            raise ValueError("mode must be one of {'diff','div'}")
        if float(self.rho) < 0 or float(self.rho_sep) < 0:
            raise ValueError("rho and rho_sep must be >= 0")

        # ---- Augment for intercept
        if self.fit_intercept:
            X_aug = np.hstack([X, np.ones((n, 1), dtype=X.dtype)])
            p = d + 1
        else:
            X_aug = X
            p = d

        # ---- Base quadratic pieces
        XtX = (X_aug.T @ X_aug) / float(n)
        Xty = (X_aug.T @ y) / float(n)

        I_reg = np.eye(p, dtype=float)
        if self.fit_intercept:
            I_reg[-1, -1] = 0.0

        A = XtX + (float(self.alpha) * I_reg)
        rhs = Xty.copy()

        # ---- Centered y
        y_mean = float(np.mean(y))
        y_c = y - y_mean
        var_y = float(np.mean(y_c * y_c))

        # ==========================================================
        # A) Separable surrogate term (CENTERED at correct target)
        #     s(beta) = (1/n) sum_i ( (r_i - t) * y_c,i )^2
        # ==========================================================
        rho_s = float(self.rho_sep)
        if rho_s > 0.0:
            if self.mode == "diff":
                # r_tilde = (Xb - y) - 0 = Xb - y
                # s = (1/n)||diag(y_c)(Xb - y)||^2
                d2 = (y_c ** 2).astype(float)
                Xw = X_aug * d2[:, None]
                A += rho_s * (X_aug.T @ Xw) / float(n)
                rhs += rho_s * (X_aug.T @ (d2 * y)) / float(n)

            else:
                # mode == "div"
                # r = (Xb)/y_safe, target t=1 => r_tilde = (Xb)/y_safe - 1 = (Xb - y_safe)/y_safe
                # s = (1/n) sum (y_c^2 / y_safe^2) * (Xb - y_safe)^2
                y_safe = self._safe_y(y)
                d2 = (y_c ** 2) / (y_safe ** 2)
                Xw = X_aug * d2[:, None]
                A += rho_s * (X_aug.T @ Xw) / float(n)
                rhs += rho_s * (X_aug.T @ (d2 * y_safe)) / float(n)

        # ==========================================================
        # B) Global covariance penalty (unchanged; shift doesn't matter)
        # ==========================================================
        rho_g = float(self.rho)
        if rho_g > 0.0:
            if self.mode == "diff":
                # c = mean((Xb - y)*y_c) = u^T b - var_y
                u = (X_aug.T @ y_c) / float(n)
                if self.fit_intercept:
                    u[-1] = 0.0
                rhs = rhs + (rho_g * var_y) * u
            else:
                # c = mean(((Xb)/y_safe)*y_c) = u^T b
                y_safe = self._safe_y(y)
                w = y_c / y_safe
                u = (X_aug.T @ w) / float(n)

            u = u.reshape(-1, 1)
            rhs_vec = rhs.reshape(-1, 1)

            beta0 = self._solve(A, rhs_vec)
            Au = self._solve(A, u)
            denom = 1.0 + rho_g * float((u.T @ Au).item())

            if (not np.isfinite(denom)) or abs(denom) < 1e-15:
                LHS = A + rho_g * (u @ u.T)
                beta = self._solve(LHS, rhs_vec)
            else:
                beta = beta0 - (rho_g / denom) * (Au * float((u.T @ beta0).item()))
            beta = beta.ravel()

        else:
            beta = self._solve(A, rhs.reshape(-1, 1)).ravel()
            u = None

        # ---- unpack
        if self.fit_intercept:
            self.coef_ = beta[:-1]
            self.intercept_ = float(beta[-1])
        else:
            self.coef_ = beta
            self.intercept_ = 0.0

        # ---- diagnostics (train moments)
        y_hat = X_aug @ beta
        if self.mode == "diff":
            r = y_hat - y
            r_tilde = r  # target 0
        else:
            y_safe = self._safe_y(y)
            r = y_hat / y_safe
            r_tilde = r - 1.0  # target 1

        a = r * y_c              # for covariance moment (same as (r-1)*y_c)
        a_tilde = r_tilde * y_c  # for separable surrogate

        self.cov_moment_ = float(np.mean(a))
        self.cov_moment_sq_ = float(self.cov_moment_ ** 2)
        self.sep_surrogate_ = float(np.mean(a_tilde ** 2))  # (1/n) sum ( (r-t)*y_c )^2

        self.y_mean_ = y_mean
        self._u_ = None if u is None else u.ravel()
        self._n_features_in_ = d

        return self

    def predict(self, X):
        check_is_fitted(self, ["coef_", "intercept_"])
        X = check_array(X)
        return X @ self.coef_ + self.intercept_

    def _safe_y(self, y):
        y = np.asarray(y, dtype=float)
        eps = float(self.eps_y)
        y_safe = y.copy()
        mask = np.abs(y_safe) < eps
        if np.any(mask):
            y_safe[mask] = np.where(y_safe[mask] >= 0.0, eps, -eps)
        return y_safe

    def __str__(self):
        return f"FairnessConstrainedRidgeLog(alpha={self.alpha}, rho={self.rho}, rho_sep={self.rho_sep}, mode={self.mode})"

    def __repr__(self):
            return f"FairnessConstrainedRidgeLog(alpha={self.alpha}, rho={self.rho}, rho_sep={self.rho_sep}, mode={self.mode})"

    @staticmethod
    def _solve(M, b):
        try:
            return np.linalg.solve(M, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(M, b, rcond=None)[0]

