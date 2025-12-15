import numpy as np
import cvxpy as cp
from typing import List, Sequence, Optional


class ConstantModel:
    def __init__(self, fit_intercept=True, sensitive_idx=None, objective="least_unfair", fair_weight=0, baseline_values=None, l1_lambda=0, l2_lambda=0, solver="MOSEK", verbose=False):
        self.fit_intercept = fit_intercept
        self.sensitive_idx = sensitive_idx
        self.objective = objective
        self.fair_weight = fair_weight
        self.baseline_values = baseline_values
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.y_mean = None
        self.coef_ = None
        # Optimization
        self.solver = solver
        self.verbose = verbose

    def fit(self, X, y):
        try:
            X = X.to_numpy()
            y = y.to_numpy()
        except Exception:
            pass
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        n,m = X.shape
        # G = len(self.sensitive_idx)
        # Optimization
        beta = cp.Variable(m)
        u = cp.Variable(n, nonneg=True)
        l = cp.Variable(n, nonneg=True)
        z = cp.Variable()
        
        constraints = [
            y - X @ beta == u - l
        ]
        for g_idx in self.sensitive_idx:
            constraints+=[ cp.mean(u[g_idx] + l[g_idx]) <= z ]
        
        baseline_values = {"f_max": 1, "f_min": 0, "l_max": 1,"l_min": 0} if self.baseline_values is None else self.baseline_values
    
        if self.objective == "least_unfair":
            obj = z
        elif self.objective == "least_error":
            obj = cp.mean( u + l )
        elif self.objective == "error_fairness_mixture":
            l_normalized = (cp.mean( u + l ) - baseline_values["l_min"]) / (baseline_values["l_max"]- baseline_values["l_min"])
            f_normalized =(z - baseline_values["f_min"])/(baseline_values["f_max"] - baseline_values["f_min"])
            obj = (1-self.fair_weight) * l_normalized + self.fair_weight * f_normalized

        # reg
        if self.l1_lambda > 0:
            obj += self.l1_lambda * cp.norm1(beta)
        if self.l2_lambda > 0:
            obj += self.l2_lambda * cp.norm2(beta)

        # Solve
        prob = cp.Problem( cp.Minimize(obj) , constraints)
        solver = (self.solver or "OSQP").upper()
        if solver != "OSQP":
            prob.solve(solver=solver, verbose=self.verbose)
        else:
            # Pure QP → OSQP handles it very fast
            prob.solve(solver=cp.OSQP, verbose=self.verbose, eps_abs=1e-6, eps_rel=1e-6)
        print("Constant Fair Model Status: ", prob.status)
        for g,g_idx in enumerate(self.sensitive_idx):
            print(f"Error for group (n_g={len(g_idx)})", g, ": ", cp.mean(u[g_idx] + l[g_idx]).value)
        self.coef_ = beta.value
    
        

    def predict(self, X):
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        return X @ self.coef_

    def __str__(self):
        return f"ConstantModel(fair_weight={self.fair_weight})"


class LeastAbsoluteDeviationRegression:
    def __init__(self, fit_intercept=True, fit_group_intercept=False, sensitive_weights=None, l2_delta=0, solver="GUROBI", verbose=False, solve_dual=False, sensitive_idx=None):
        self.beta = None
        self.fit_intercept = fit_intercept
        self.fit_group_intercept = fit_group_intercept
        self.sensitive_weights = sensitive_weights
        self.sensitive_idx = sensitive_idx
        self.l2_delta = l2_delta
        self.solver = solver
        self.verbose = verbose
        self.solve_dual = solve_dual

    def fit(self, X, y):
        try:
            X, y = X.to_numpy(), y.to_numpy()
        except Exception as e:
            pass
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        n, m = X.shape
        if self.fit_group_intercept:
            G = len(self.sensitive_idx)

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
                cp.Maximize( y @ theta ), 
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

            # Group intercept (fairness post-processing shift)
            if self.fit_group_intercept:
                d = cp.Variable(G) # intercept for each group (similar to adding senstive feature, but unweighted and not used in prediction) 
                # d_UB = cp.Variable(nonneg=True)
                # d_LB = cp.Variable(nonneg=True)
                constraints = []
                for g, g_idx in enumerate(self.sensitive_idx):
                    # print(g, len(g_idx))
                    constraints += [
                        X[g_idx, :] @ beta + u[g_idx] - l[g_idx] == y[g_idx] - d[g]  
                    ]
            else: # No group intercept
                constraints = [
                    X @ beta + u - l == y
                ]


            # Objective: <=> Minimize the overall Mean Absolute Error
            w_ = np.ones(n)/n if self.sensitive_weights is None else self.sensitive_weights 
            obj = w_ @ (u + l) 
            if self.fit_group_intercept: #and self.l2_delta > 0:
                obj += self.l2_delta * cp.norm2(d)
                # constraints += [ d == d_UB - d_LB]
                # obj += self.l2_delta * cp.sum(d_UB + d_LB) #cp.norm1(d) * 0#self.l2_delta
            primal_prob = cp.Problem(
                cp.Minimize( obj ), 
                constraints
            )

            # Solve the optimization problem
            try:
                result = primal_prob.solve(solver=self.solver, verbose=self.verbose)
            except cp.error.SolverError:
                print("GUROBI not available, trying default solver.")
                result = primal_prob.solve(verbose=self.verbose)

            print(f"Problem status: {primal_prob.status}")
            print(f"Optimal objective (Mean Absolute Error): {result}")

            # Print the difference in MAE between groups post-optimization
            if primal_prob.status in ["optimal", "optimal_inaccurate"]:
                self.beta = beta.value
                if self.fit_group_intercept:
                    print("delta: ", d.value)
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
    def __init__(self, fit_intercept=True, solver="GUROBI", k_percentage=0.7, lambda_l1=0, lambda_l2=0,
                 objective="mae", sensitive_idx=None, 
                 fit_group_intercept=False, delta_l2=0, 
                 fit_group_beta=False, group_beta_l2=0,
                 group_constraints=False, group_percentage_diff=0, sensitive_feature=None,
                 weight_by_group=False,
                 residual_cov_constraint=False, residual_cov_thresh=0,
                 ):
        self.beta = None
        self.intercept = None
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.k_percentage = k_percentage
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
        self.group_constraints = group_constraints
        self.group_percentage_diff = group_percentage_diff

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
                k_samples = [len(g_idx) * self.k_percentage for g_idx in self.sensitive_idx]
                k_samples = [min(k_samples) for g_idx in self.sensitive_idx]
            else:
                k_samples = n * self.k_percentage#int(n * self.k_percentage) 
        else:
            # k_samples 
            k_samples = n * self.k_percentage#int(n * self.k_percentage) 

        # Primal approach of the problem
        beta = cp.Variable(m)
        intercept = cp.Variable()
        z = cp.Variable(m) # absolute of beta:            
        nu = cp.Variable(n_groups) #if self.weight_by_group else nu = cp.Variable(1)
        theta = cp.Variable(n, nonneg=True)
        if self.fit_group_intercept:
            delta = cp.Variable(n_groups)
        if self.fit_group_beta:
            group_beta = cp.Variable((m, n_groups))
        
        if self.group_constraints:
            min_risk = cp.Variable()
            U = cp.Variable()
        

        # Regularizer constraints
        constraints =[
            beta <= z,
            -beta <= z,
        ]

        # Objective constraints
        for g, g_idx in enumerate(self.sensitive_idx):
            if not self.weight_by_group:
                constraints += [ nu[g] == nu[0] ] # unique nu
            if self.fit_group_beta and self.fit_group_intercept:
                constraints += [
                     y[g_idx] - (X[g_idx, :] @ (beta + group_beta[:, g]) + intercept + delta[g] ) <= nu[g] + theta[g_idx],
                    -y[g_idx] + (X[g_idx, :] @ (beta + group_beta[:, g]) + intercept + delta[g] ) <= nu[g] + theta[g_idx],
                ]
            elif self.fit_group_intercept:
                constraints += [
                     y[g_idx] - (X[g_idx, :] @ beta + intercept + delta[g] ) <= nu[g] + theta[g_idx],
                    -y[g_idx] + (X[g_idx, :] @ beta + intercept + delta[g] ) <= nu[g] + theta[g_idx],
                ]
            elif self.fit_group_beta:
                constraints += [
                     y[g_idx] - (X[g_idx, :] @ (beta + group_beta[:, g]) + intercept ) <= nu[g] + theta[g_idx],
                    -y[g_idx] + (X[g_idx, :] @ (beta + group_beta[:, g]) + intercept ) <= nu[g] + theta[g_idx],
                ]
            else:
                constraints += [
                     y[g_idx] - (X[g_idx, :] @ beta + intercept ) <= nu[g] + theta[g_idx],
                    -y[g_idx] + (X[g_idx, :] @ beta + intercept ) <= nu[g] + theta[g_idx],
                ]
                # if self.objective == "mae":
                #     constraints += [
                #         y - ( X @ beta + intercept ) <= nu + theta,
                #         -y + ( X @ beta + intercept ) <= nu + theta,
                #     ]
                # elif self.objective == "mse":
                #     constraints +=[
                #     cp.square(y[i] - X[i,:] @ beta) <= nu + theta[i] for i in range(n) 
                #     ]

        # Group constraints
        if self.group_constraints:
            d = self.sensitive_feature
            if self.fit_group_intercept:
                cov_mean = 0
                # cov_mean_neg = 0
                f_X_mean = 0
                for g, g_idx in enumerate(self.sensitive_idx): # Global mean computing (not just groups)
                    f_X_mean +=  cp.sum( X[g_idx, :] @ beta + intercept + delta[g] ) / n 
                for g, g_idx in enumerate(self.sensitive_idx):
                    cov_mean += cp.sum( ( d[g_idx] - np.mean(d) ) * ( X[g_idx, :] @ beta + intercept + delta[g] - f_X_mean ) ) / n 
                    # cov_mean_neg -= cp.sum( ( d[g_idx] - np.mean(d) ) * ( X[g_idx, :] @ beta + intercept + delta[g] - f_X_mean ) ) / n

                constraints += [
                    # cp.mean( nu[g] + theta[g_idx] ) <= U,
                        cov_mean <= self.group_percentage_diff * np.std(d) * np.std(y),
                        -cov_mean <= self.group_percentage_diff * np.std(d) * np.std(y),
                    # cov( self.sensitive_feature, X[g_idx, :] @ beta + intercept ) <= eps * std(self.sensitive_feature) * std( y )
                ]
            else:
                f_X =  X @ beta + intercept
                constraints += [
                    # cp.mean( nu[g] + theta[g_idx] ) <= U,
                     cp.mean( ( d - np.mean(d) ) * ( f_X - cp.mean( f_X ) ) ) <= self.group_percentage_diff * np.std(d) * np.std(y),
                    -cp.mean( ( d - np.mean(d) ) * ( f_X - cp.mean( f_X ) ) ) <= self.group_percentage_diff * np.std(d) * np.std(y),
                ]
        if self.residual_cov_constraint:
            d = y#self.sensitive_feature
            f_X = X @ beta + intercept   # plus delta[g] etc. if using group intercepts
            r = y - f_X
            constraints += [
                 cp.mean((d - np.mean(d)) * (r - cp.mean(r))) <= self.residual_cov_thresh * np.std(d), # Not sure what to put on the right
                -cp.mean((d - np.mean(d)) * (r - cp.mean(r))) <= self.residual_cov_thresh * np.std(d), # Not sure what to put on the right
            ]

            # self.residual_cov_thresh = residual_cov_thresh
        # Objective: <=> Minimize the overall Mean Absolute Error
        if self.weight_by_group:
            obj = cp.sum( [ nu[g] * k_samples[g] + cp.sum(theta[g_idx])  for g, g_idx in enumerate(self.sensitive_idx) ] )
        else: 
            obj = cp.sum( [ nu[0] * k_samples + cp.sum(theta) ] ) 
        if self.lambda_l1 > 0:
            obj += self.lambda_l1 * cp.sum(z) 
        if self.lambda_l2 > 0:
            obj += self.lambda_l2 * cp.norm2(beta)
        if self.delta_l2 > 0:
            obj += self.delta_l2 * cp.norm2(delta)
        if self.group_beta_l2 > 0:
            obj += self.group_beta_l2 * cp.norm2(group_beta)
        # if self.group_constraints:
        #     obj += self.group_percentage_diff * U
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
        if self.fit_group_intercept:
            print(f"Shift delta: ", delta.value)
        print(f"Nu dual: ", nu.value)

        # Print the difference in MAE between groups post-optimization
        if primal_prob.status in ["optimal", "optimal_inaccurate"]:
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
        return f"StableRegression(fit_intercept={self.fit_intercept})"


# Is just the same with another name, for the print xd
class StableRegressionOld:
    def __init__(self, fit_intercept=True, solver="GUROBI", k_percentage=0.7, lambda_l1=0, lambda_l2=0,
                 objective="mae", sensitive_idx=None, 
                 fit_group_intercept=False, delta_l2=0, 
                 fit_group_beta=False, group_beta_l2=0,
                 group_constraints=False, group_percentage_diff=0, sensitive_feature=None,
                 weight_by_group=False,
                 residual_cov_constraint=False, residual_cov_thresh=0,
                 ):
        self.beta = None
        self.intercept = None
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.k_percentage = k_percentage
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
        self.group_constraints = group_constraints
        self.group_percentage_diff = group_percentage_diff

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
                k_samples = [len(g_idx) * self.k_percentage for g_idx in self.sensitive_idx]
                k_samples = [min(k_samples) for g_idx in self.sensitive_idx]
            else:
                k_samples = n * self.k_percentage#int(n * self.k_percentage) 
        else:
            # k_samples 
            k_samples = n * self.k_percentage#int(n * self.k_percentage) 

        # Primal approach of the problem
        beta = cp.Variable(m)
        intercept = cp.Variable()
        z = cp.Variable(m) # absolute of beta:            
        nu = cp.Variable(n_groups) #if self.weight_by_group else nu = cp.Variable(1)
        theta = cp.Variable(n, nonneg=True)
        if self.fit_group_intercept:
            delta = cp.Variable(n_groups)
        if self.fit_group_beta:
            group_beta = cp.Variable((m, n_groups))
        
        if self.group_constraints:
            min_risk = cp.Variable()
            U = cp.Variable()
        

        # Regularizer constraints
        constraints =[
            beta <= z,
            -beta <= z,
        ]

        # Objective constraints
        for g, g_idx in enumerate(self.sensitive_idx):
            if not self.weight_by_group:
                constraints += [ nu[g] == nu[0] ] # unique nu
            if self.fit_group_beta and self.fit_group_intercept:
                constraints += [
                     y[g_idx] - (X[g_idx, :] @ (beta + group_beta[:, g]) + intercept + delta[g] ) <= nu[g] + theta[g_idx],
                    -y[g_idx] + (X[g_idx, :] @ (beta + group_beta[:, g]) + intercept + delta[g] ) <= nu[g] + theta[g_idx],
                ]
            elif self.fit_group_intercept:
                constraints += [
                     y[g_idx] - (X[g_idx, :] @ beta + intercept + delta[g] ) <= nu[g] + theta[g_idx],
                    -y[g_idx] + (X[g_idx, :] @ beta + intercept + delta[g] ) <= nu[g] + theta[g_idx],
                ]
            elif self.fit_group_beta:
                constraints += [
                     y[g_idx] - (X[g_idx, :] @ (beta + group_beta[:, g]) + intercept ) <= nu[g] + theta[g_idx],
                    -y[g_idx] + (X[g_idx, :] @ (beta + group_beta[:, g]) + intercept ) <= nu[g] + theta[g_idx],
                ]
            else:
                constraints += [
                     y[g_idx] - (X[g_idx, :] @ beta + intercept ) <= nu[g] + theta[g_idx],
                    -y[g_idx] + (X[g_idx, :] @ beta + intercept ) <= nu[g] + theta[g_idx],
                ]
                # if self.objective == "mae":
                #     constraints += [
                #         y - ( X @ beta + intercept ) <= nu + theta,
                #         -y + ( X @ beta + intercept ) <= nu + theta,
                #     ]
                # elif self.objective == "mse":
                #     constraints +=[
                #     cp.square(y[i] - X[i,:] @ beta) <= nu + theta[i] for i in range(n) 
                #     ]

        # Group constraints
        if self.group_constraints:
            d = self.sensitive_feature
            if self.fit_group_intercept:
                cov_mean = 0
                # cov_mean_neg = 0
                f_X_mean = 0
                for g, g_idx in enumerate(self.sensitive_idx): # Global mean computing (not just groups)
                    f_X_mean +=  cp.sum( X[g_idx, :] @ beta + intercept + delta[g] ) / n 
                for g, g_idx in enumerate(self.sensitive_idx):
                    cov_mean += cp.sum( ( d[g_idx] - np.mean(d) ) * ( X[g_idx, :] @ beta + intercept + delta[g] - f_X_mean ) ) / n 
                    # cov_mean_neg -= cp.sum( ( d[g_idx] - np.mean(d) ) * ( X[g_idx, :] @ beta + intercept + delta[g] - f_X_mean ) ) / n

                constraints += [
                    # cp.mean( nu[g] + theta[g_idx] ) <= U,
                        cov_mean <= self.group_percentage_diff * np.std(d) * np.std(y),
                        -cov_mean <= self.group_percentage_diff * np.std(d) * np.std(y),
                    # cov( self.sensitive_feature, X[g_idx, :] @ beta + intercept ) <= eps * std(self.sensitive_feature) * std( y )
                ]
            else:
                f_X =  X @ beta + intercept
                constraints += [
                    # cp.mean( nu[g] + theta[g_idx] ) <= U,
                     cp.mean( ( d - np.mean(d) ) * ( f_X - cp.mean( f_X ) ) ) <= self.group_percentage_diff * np.std(d) * np.std(y),
                    -cp.mean( ( d - np.mean(d) ) * ( f_X - cp.mean( f_X ) ) ) <= self.group_percentage_diff * np.std(d) * np.std(y),
                ]
        if self.residual_cov_constraint:
            d = y#self.sensitive_feature
            f_X = X @ beta + intercept   # plus delta[g] etc. if using group intercepts
            r = y - f_X
            constraints += [
                 cp.mean((d - np.mean(d)) * (r - cp.mean(r))) <= self.residual_cov_thresh * np.std(d), # Not sure what to put on the right
                -cp.mean((d - np.mean(d)) * (r - cp.mean(r))) <= self.residual_cov_thresh * np.std(d), # Not sure what to put on the right
            ]

            # self.residual_cov_thresh = residual_cov_thresh
        # Objective: <=> Minimize the overall Mean Absolute Error
        if self.weight_by_group:
            obj = cp.sum( [ nu[g] * k_samples[g] + cp.sum(theta[g_idx])  for g, g_idx in enumerate(self.sensitive_idx) ] )
        else: 
            obj = cp.sum( [ nu[0] * k_samples + cp.sum(theta) ] ) 
        if self.lambda_l1 > 0:
            obj += self.lambda_l1 * cp.sum(z) 
        if self.lambda_l2 > 0:
            obj += self.lambda_l2 * cp.norm2(beta)
        if self.delta_l2 > 0:
            obj += self.delta_l2 * cp.norm2(delta)
        if self.group_beta_l2 > 0:
            obj += self.group_beta_l2 * cp.norm2(group_beta)
        # if self.group_constraints:
        #     obj += self.group_percentage_diff * U
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
        if self.fit_group_intercept:
            print(f"Shift delta: ", delta.value)
        print(f"Nu dual: ", nu.value)

        # Print the difference in MAE between groups post-optimization
        if primal_prob.status in ["optimal", "optimal_inaccurate"]:
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
        return f"StableRegressionOld(fit_intercept={self.fit_intercept})"



from sklearn.base import BaseEstimator, RegressorMixin

class StableCovarianceUpperBoundLADRegressor(BaseEstimator, RegressorMixin):
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
        rho=1.0,
        l1=0.0,
        l2=0.0,
        fit_intercept=True,
        solver="MOSEK",
        solver_opts=None,
        verbose=False,
        warm_start=False,
    ):
        self.K = K
        self.keep = keep
        self.rho = rho
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

    def fit(self, X, y, d):
        X = self._as_2d_float(X)
        y = self._as_1d_float(y, "y")
        d = self._as_1d_float(d, "d")

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

        rho = float(self.rho)
        if rho < 0:
            raise ValueError("rho must be >= 0.")

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
        # c = cp.Variable(n, nonneg=True)
        # q = cp.Variable(n, nonneg=True)

        # model prediction
        yhat = X @ beta + (b0 if self.fit_intercept else 0.0)

        # residual abs: y - yhat = u - ell  => |y-yhat| = u + ell
        # constraints = [y - yhat == u - ell]
        # abs_res = u + ell

        # fairness contribution: z_i = (d_i-dbar)*(yhat_i - mean(yhat))
        fbar = (1.0 / n) * cp.sum(yhat)
        z = cp.multiply(d_center, yhat - fbar)  # affine in (beta,b0)

        # abs(z): z = c - q  => |z| = c + q
        # constraints += [z <= c] #- q]
        # abs_z = c #+ q
        constraints =[
             y - yhat + rho * z <= nu + theta,
             y - yhat - rho * z <= nu + theta,
            -y + yhat + rho * z <= nu + theta,
            -y + yhat - rho * z <= nu + theta,
        ]

        # per-sample composite score
        # score = abs_res + rho * abs_z

        # top-K epigraph constraints: score_i <= nu + theta_i
        # constraints += [score <= nu + theta]

        # objective: sum of K largest scores = min_{nu,theta} K*nu + sum theta
        obj = K * nu + cp.sum(theta)

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

        self.coef_ = np.asarray(beta.value).reshape(-1)
        self.intercept_ = float(b0.value) if self.fit_intercept else 0.0

        # store diagnostics (optional)
        yhat_val = (X @ self.coef_) + self.intercept_
        fbar_val = float(np.mean(yhat_val))
        z_val = (d_center) * (yhat_val - fbar_val)
        score_val = np.abs(y - yhat_val) + rho * np.abs(z_val)
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





class ModelStacking:
    """
    Fair model stacking (sklearn-style) using a fast QP (no integers, no abs slacks).

    We learn stacking weights W (and optional intercept b) to minimize a convex
    blend of overall MSE and the *maximum per-group MSE*:

        minimize  (1-α) * (1/n) * ||r||_2^2  +  α * u  +  0.5 * λ * ||W||_2^2
        subject to  (1/n_g) * ||r_g||_2^2  <=  u   for all groups g,
                    optional: W ≥ 0,  sum(W) == 1 (convex mixture)
        with r = y - (Y W + b),  Y_ij = model_j(X_i).

    This is a SOC/QP solved efficiently by OSQP/ECOS/MOSEK. No big-M, no binaries.

    Parameters
    ----------
    trained_models : sequence of fitted regressors with .predict(X)
    sensitive_idx  : list of index lists (one list per group)
    fair_weight    : α in [0,1], trades overall MSE vs worst-group MSE
    l2_lambda      : λ ≥ 0, ridge on stacking weights (use 0 for unregularized)
    sum_to_one     : if True, enforce sum(W) == 1 (default). If False, no sum constraint.
    nonneg         : if True (default), enforce W ≥ 0 (convex mixture). Set False to allow signed weights.
    fit_intercept  : include intercept b (recommended; default True)
    solver         : 'OSQP' (default), 'ECOS', or 'MOSEK'
    verbose        : solver verbosity flag

    Attributes
    ----------
    coef_      : (k,) stacking weights
    intercept_ : float intercept
    status_    : solver status string
    objective_ : optimal objective value
    """

    def __init__(
        self,
        trained_models: Sequence,
        sensitive_idx: Sequence[Sequence[int]],
        fair_weight: float = 0.5,
        l2_lambda: float = 1e-6,
        sum_to_one: bool = True,
        nonneg: bool = True,
        fit_intercept: bool = True,
        solver: str = "OSQP",
        verbose: bool = False,
    ):
        self.trained_models = list(trained_models)
        self.sensitive_idx = [list(g) for g in sensitive_idx]
        self.fair_weight = float(fair_weight)
        self.l2_lambda = float(l2_lambda)
        self.sum_to_one = bool(sum_to_one)
        self.nonneg = bool(nonneg)
        self.fit_intercept = bool(fit_intercept)
        self.solver = solver
        self.verbose = verbose
        # learned params
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0
        self.status_: Optional[str] = None
        self.objective_: Optional[float] = None

    # --- utilities ---
    def _stack_predictions(self, X: np.ndarray) -> np.ndarray:
        k = len(self.trained_models)
        Y = np.empty((X.shape[0], k), dtype=float)
        for j, mdl in enumerate(self.trained_models):
            Y[:, j] = np.asarray(mdl.predict(X)).ravel()
        return Y

    def _check_groups(self, n: int):
        used = set()
        for idx in self.sensitive_idx:
            for i in idx:
                if i < 0 or i >= n:
                    raise ValueError("Group index out of range")
                if i in used:
                    # Overlaps are allowed in some fairness defs; warn rather than fail
                    pass
                used.add(i)

    # --- sklearn-like API ---
    def fit(self, X, y):
        # Accept numpy / pandas
        try:
            X = X.to_numpy(copy=True)
        except Exception:
            X = np.asarray(X)
        try:
            y = y.to_numpy(copy=True).ravel()
        except Exception:
            y = np.asarray(y).ravel()

        n, p = X.shape[0], X.shape[1]
        self._check_groups(n)
        Y = self._stack_predictions(X)  # (n,k)
        n, k = Y.shape

        # Variables
        W = cp.Variable(k)
        b = cp.Variable() if self.fit_intercept else 0.0
        r = cp.Variable(n)  # residuals
        u = cp.Variable()   # upper bound on per-group MSE
        # l = cp.Variable()   # lower bound on per-group MSE

        # Residual definition
        constraints = [r == y - (Y @ W + b)]

        # Fairness constraints: group MSE <= u
        for idx in self.sensitive_idx:
            if len(idx) == 0:
                continue
            constraints.append(cp.sum_squares(r[idx]) / len(idx) <= u)
            # constraints.append(cp.sum_squares(r[idx]) / len(idx) >= l)

        # Weight constraints
        if self.nonneg:
            constraints.append(W >= 0)
        if self.sum_to_one:
            constraints.append(cp.sum(W) == 1)

        # Objective: (1-α) overall MSE + α * u + 0.5 λ ||W||^2
        obj = ((1.0 - self.fair_weight) * cp.sum_squares(r) / n
               + self.fair_weight * u
               + 0.5 * self.l2_lambda * cp.sum_squares(W))

        prob = cp.Problem(cp.Minimize(obj), constraints)

        # Warm start
        if self.coef_ is not None:
            W.value = self.coef_.copy()
        if self.fit_intercept:
            try:
                b.value = float(self.intercept_)
            except Exception:
                pass

        # Solve
        solver = (self.solver or "OSQP").upper()
        if solver == "MOSEK":
            prob.solve(solver=cp.MOSEK, verbose=self.verbose)
        elif solver == "ECOS":
            prob.solve(solver=cp.ECOS, verbose=self.verbose)
        else:
            # Pure QP → OSQP handles it very fast
            prob.solve(solver=cp.OSQP, verbose=self.verbose, eps_abs=1e-7, eps_rel=1e-7)

        # CHECKING:
        max_mse, min_mse = 0, np.inf,
        for idx in self.sensitive_idx:
            if len(idx) == 0:
                continue
            mse = (cp.sum_squares(r[idx]) / len(idx)).value.copy()
            max_mse = mse if mse > max_mse else max_mse
            min_mse = mse if mse < min_mse else min_mse
        print("Difference between bounds: ", max_mse - min_mse)
        print("The maxes: ", u.value, max_mse)
        print("The min: ", min_mse)

        self.status_ = prob.status
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"Optimization failed: status={prob.status}")

        self.coef_ = W.value.astype(float)
        self.intercept_ = float(b.value) if self.fit_intercept else 0.0
        self.objective_ = float(prob.value)
        return self

    def predict(self, X):
        if self.coef_ is None:
            raise RuntimeError("Call fit before predict().")
        try:
            X = X.to_numpy(copy=True)
        except Exception:
            X = np.asarray(X)
        Y = self._stack_predictions(X)
        return (Y @ self.coef_) + self.intercept_

    def __str__(self):
        return f"StackModels(fair_weight={self.fair_weight})"

class ContextualFairStacking:
    """
    Contextual (similarity-gated) fair stacking, sklearn-style.

    Trains group-specific stacking weights W_g (and optional intercepts b_g).
    At prediction, blends group-specific predictions with a gating function pi_g(x)
    based on similarity between the query x and training samples in each group
    (KNN or RBF kernel). The final prediction is the weighted sum over groups.

    Training objective (QP):
      minimize  sum_g (1/n_g) * || y_g - (Y_g W_g + b_g) ||^2
                + 0.5*l2_lambda * sum_g ||W_g||^2
                + 0.5*coupling * sum_g ||W_g - W0||^2
      subject to optional: W_g >= 0,  sum(W_g) == 1 for all groups

    Y is the matrix of base-model predictions (n by k). W0 is a shared reference
    weight vector to stabilize (couple) group models. This is a convex QP; solved
    fast with OSQP/ECOS/MOSEK.

    Gating at prediction:
      pi_g(x) proportional to sum over i in group g of K(x, x_i)
      with K being either RBF: exp(-gamma * ||x - x_i||^2) or a KNN kernel.
      We normalize pi to sum to 1.

    Parameters
    ----------
    trained_models : sequence of fitted regressors with .predict(X)
    sensitive_idx  : list of index lists (training indices per sensitive group)
    l2_lambda      : float >= 0, ridge penalty on each W_g
    coupling       : float >= 0, strength tying W_g to a shared W0 (stability)
    sum_to_one     : bool, enforce sum(W_g) == 1 for each group
    nonneg         : bool, enforce W_g >= 0 (convex mixtures)
    fit_intercept  : bool, fit b_g per group (recommended)
    solver         : 'OSQP' (default), 'ECOS', or 'MOSEK'
    gate           : {'rbf','knn'} gating type at prediction
    rbf_gamma      : float or None; if None, median heuristic is used
    knn_k          : int, number of neighbors for KNN gating
    gate_features  : Optional sequence of column indices to use for gating
    verbose        : bool solver verbosity

    Attributes
    ----------
    coef_g_       : array (G, k) group-specific stacking weights
    intercept_g_  : array (G,) group intercepts (0 if fit_intercept=False)
    coef_ref_     : array (k,) shared reference W0 (if coupling>0 else None)
    status_       : solver status
    objective_    : optimal objective value
    X_gate_       : array (n, q) cached training features used for gating
    group_sizes_  : array (G,) sizes of groups
    """

    def __init__(
        self,
        trained_models: Sequence,
        sensitive_idx: Sequence[Sequence[int]],
        l2_lambda: float = 1e-6,
        coupling: float = 1e-3,
        sum_to_one: bool = True,
        nonneg: bool = True,
        fit_intercept: bool = True,
        solver: str = "OSQP",
        gate: str = "rbf",
        rbf_gamma: Optional[float] = None,
        knn_k: int = 25,
        gate_features: Optional[Sequence[int]] = None,
        verbose: bool = False,
    ):
        self.trained_models = list(trained_models)
        self.sensitive_idx = [list(g) for g in sensitive_idx]
        self.l2_lambda = float(l2_lambda)
        self.coupling = float(coupling)
        self.sum_to_one = bool(sum_to_one)
        self.nonneg = bool(nonneg)
        self.fit_intercept = bool(fit_intercept)
        self.solver = solver
        self.gate = gate
        self.rbf_gamma = rbf_gamma
        self.knn_k = int(knn_k)
        self.gate_features = None if gate_features is None else list(gate_features)
        self.verbose = verbose
        # learned params
        self.coef_g_: Optional[np.ndarray] = None
        self.intercept_g_: Optional[np.ndarray] = None
        self.coef_ref_: Optional[np.ndarray] = None
        self.status_: Optional[str] = None
        self.objective_: Optional[float] = None
        # cache
        self.X_gate_: Optional[np.ndarray] = None
        self.group_sizes_: Optional[np.ndarray] = None

    # ---- helpers ----
    def _to_numpy(self, X):
        try:
            return X.to_numpy(copy=True)
        except Exception:
            return np.asarray(X)

    def _stack_predictions(self, X: np.ndarray) -> np.ndarray:
        k = len(self.trained_models)
        Y = np.empty((X.shape[0], k), dtype=float)
        for j, mdl in enumerate(self.trained_models):
            Y[:, j] = np.asarray(mdl.predict(X)).ravel()
        return Y

    def _gate_features(self, X: np.ndarray) -> np.ndarray:
        if self.gate_features is None:
            return X
        return X[:, self.gate_features]

    def _median_gamma(self, Z: np.ndarray) -> float:
        # Median heuristic for RBF bandwidth; subsample for speed
        m = Z.shape[0]
        idx = np.random.default_rng(0).choice(m, size=min(500, m), replace=False)
        Zs = Z[idx]
        D = np.sum((Zs[:, None, :] - Zs[None, :, :])**2, axis=2)
        med = np.median(D[D>0]) if np.any(D>0) else 1.0
        return 1.0 / (med + 1e-12)

    # ---- API ----
    def fit(self, X, y):
        X = self._to_numpy(X)
        y = self._to_numpy(y).ravel()
        Y = self._stack_predictions(X)
        n, k = Y.shape
        G = len(self.sensitive_idx)
        self.group_sizes_ = np.array([len(idx) for idx in self.sensitive_idx], dtype=int)
        if np.any(self.group_sizes_ == 0):
            raise ValueError("All groups must have at least one sample for training.")

        # Vars: W_g (G by k), optional W0, b_g (G)
        W = cp.Variable((G, k))
        b = cp.Variable(G) if self.fit_intercept else 0.0
        obj_terms = []
        constraints = []

        # Residual terms per group
        for g, idx in enumerate(self.sensitive_idx):
            yg = y[idx]
            Yg = Y[idx, :]
            rg = yg - (Yg @ W[g, :] + (b[g] if self.fit_intercept else 0.0))
            obj_terms.append(cp.sum_squares(rg) / max(1, len(idx)))
            # per-group weight constraints
            if self.nonneg:
                constraints.append(W[g, :] >= 0)
            if self.sum_to_one:
                constraints.append(cp.sum(W[g, :]) == 1)
            # ridge
            if self.l2_lambda > 0:
                obj_terms.append(0.5 * self.l2_lambda * cp.sum_squares(W[g, :]))

        # Coupling to shared reference W0
        if self.coupling > 0:
            W0 = cp.Variable(k)
            for g in range(G):
                obj_terms.append(0.5 * self.coupling * cp.sum_squares(W[g, :] - W0))
            self._uses_ref = True
        else:
            W0 = None
            self._uses_ref = False

        prob = cp.Problem(cp.Minimize(cp.sum(obj_terms)), constraints)

        # Solve
        solver = (self.solver or "OSQP").upper()
        if solver == "MOSEK":
            prob.solve(solver=cp.MOSEK, verbose=self.verbose)
        elif solver == "ECOS":
            prob.solve(solver=cp.ECOS, verbose=self.verbose)
        else:
            prob.solve(solver=cp.OSQP, verbose=self.verbose, eps_abs=1e-7, eps_rel=1e-7)

        self.status_ = prob.status
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"Optimization failed: status={prob.status}")

        self.coef_g_ = W.value.astype(float)
        self.intercept_g_ = (b.value.astype(float) if self.fit_intercept else np.zeros(G))
        self.coef_ref_ = (W0.value.astype(float) if self._uses_ref else None)
        self.objective_ = float(prob.value)

        # Cache gating data
        Z = self._gate_features(X)
        self.X_gate_ = Z.astype(float)
        if self.gate == "rbf" and (self.rbf_gamma is None):
            self.rbf_gamma = self._median_gamma(self.X_gate_)
        return self

    def _gate_weights(self, xq: np.ndarray) -> np.ndarray:
        """Return pi_g(xq) over groups (length G), using cached training features."""
        Z = self.X_gate_
        if Z is None:
            raise RuntimeError("fit must be called before predict; no gating cache.")
        xq = xq.reshape(1, -1)
        G = len(self.sensitive_idx)
        scores = np.zeros(G)
        if self.gate == "rbf":
            gamma = float(self.rbf_gamma if self.rbf_gamma is not None else 1.0)
            d2 = np.sum((Z - xq)**2, axis=1)
            k_all = np.exp(-gamma * d2)
            for g, idx in enumerate(self.sensitive_idx):
                if len(idx):
                    scores[g] = np.sum(k_all[idx])
        else:  # KNN (count-based with 1/(d+eps) weights)
            k = max(1, int(self.knn_k))
            d2 = np.sum((Z - xq)**2, axis=1)
            nn_idx = np.argpartition(d2, kth=min(k, d2.size-1))[:k]
            w_nn = 1.0 / (np.sqrt(d2[nn_idx]) + 1e-8)
            for g, idx in enumerate(self.sensitive_idx):
                mask = np.isin(nn_idx, idx)
                scores[g] = np.sum(w_nn[mask])
        # normalize
        s = scores.sum()
        if s <= 0:
            sizes = self.group_sizes_.astype(float)
            return sizes / sizes.sum()
        return scores / s

    def predict(self, X):
        if self.coef_g_ is None:
            raise RuntimeError("Call fit before predict().")
        X = self._to_numpy(X)
        Y = self._stack_predictions(X)
        Zq = self._gate_features(X)
        G, k = self.coef_g_.shape
        m = X.shape[0]
        yhat = np.empty(m, dtype=float)
        for i in range(m):
            pi = self._gate_weights(Zq[i])
            preds_g = Y[i] @ self.coef_g_.T + self.intercept_g_
            yhat[i] = float(np.dot(pi, preds_g))
        return yhat

    def score(self, X, y):
        y = self._to_numpy(y).ravel()
        yhat = self.predict(X)
        ss_res = float(np.sum((y - yhat)**2))
        ss_tot = float(np.sum((y - y.mean())**2)) + 1e-16
        return 1.0 - ss_res/ss_tot



from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import warnings
# The Price of Fairness paper, Min-Max Lexico
class LexicographicFairRegressor(BaseEstimator, RegressorMixin):
    """
    A Linear Regression estimator that optimizes for Lexicographic Min-Max Fairness.
    
    It minimizes the error of the worst-off group, constrains that group's error,
    then minimizes the error of the second worst-off group, and so on.
    
    Parameters
    ----------
    groups : list of lists of int
        A list where each element is a list of integers representing the 
        row indices of X that belong to a specific group. 
        Example: [[0, 1, 2], [3, 4, 5]] implies two groups.

    loss : str, default='mse'
        The loss function to minimize. Options are 'mse' (Mean Squared Error) 
        or 'mae' (Mean Absolute Error).
        
    tol : float, default=1e-4
        Tolerance for determining if a group is 'binding' (tied at the max loss)
        and for setting constraints.
    
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set to False,
        no intercept will be used in calculations (i.e. data is expected to be centered).

    alpha : float, default=0.0
        Regularization strength (L2 penalty). Must be a non-negative float.
        Higher values specify stronger regularization: 
        Minimize(Max(GroupLoss) + alpha * ||w||^2).
        
    solver : str, optional
        The cvxpy solver to use (e.g., 'ECOS', 'SCS', 'OSQP'). 
        If None, cvxpy chooses automatically.
        
    verbose : bool, default=False
        If True, prints progress of the lexicographic levels.
    """
    def __init__(self, groups, loss='mse', tol=1e-4, fit_intercept=True, alpha=0.0, solver=None, verbose=False):
        self.groups = groups
        self.loss = loss
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.alpha = alpha
        self.solver = solver
        self.verbose = verbose

    def fit(self, X, y):
        """
        Fit the model using Lexicographic Min-Max optimization.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        
        y : array-like of shape (n_samples,)
            Target values.
        """
        X, y = check_X_y(X, y)
        
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative.")

        # n_features = X.shape[1]
        n_samples, n_features = X.shape
        
        # CVXPY Variables
        w = cp.Variable(n_features)
        
        # Handle Intercept
        if self.fit_intercept:
            b = cp.Variable()
        else:
            b = 0.0
        
        # Pre-compute matrices for each group to speed up loop
        # self.groups is a list of lists of indices
        group_data = {}
        unique_groups_ids = []
        
        for g_idx, indices in enumerate(self.groups):
            # Ensure indices are valid
            indices = np.array(indices)
            if np.any(indices >= n_samples) or np.any(indices < 0):
                raise ValueError(f"Group {g_idx} contains indices out of bounds for X with {n_samples} samples.")
                
            X_g = X[indices]
            y_g = y[indices]
            
            # Use the index in the list as the group ID
            group_data[g_idx] = (X_g, y_g)
            unique_groups_ids.append(g_idx)

        # Tracking sets
        # fixed_constraints: list of (group_id, bound_value)
        fixed_constraints = [] 
        # active_groups: groups whose error we are currently trying to minimize
        active_groups = list(unique_groups_ids)
        
        iteration = 0
        
        while active_groups:
            iteration += 1
            if self.verbose:
                print(f"--- Iteration {iteration} ---")
                print(f"Optimizing for {len(active_groups)} active groups. {len(fixed_constraints)} groups fixed.")

            # The variable representing the max loss for the *current* active set
            t = cp.Variable()
            
            constraints = []
            
            # 1. Add constraints for groups that are already fixed/locked from previous levels
            for g_id, bound_val in fixed_constraints:
                X_g, y_g = group_data[g_id]
                loss_expr = self._get_loss_expression(X_g, y_g, w, b)
                # We constrain them to be <= bound + tolerance (to prevent numerical infeasibility)
                constraints.append(loss_expr <= bound_val + self.tol)
            
            # 2. Add constraints for the currently active groups
            # Their loss must be <= t
            loss_expressions_map = {}
            for g_id in active_groups:
                X_g, y_g = group_data[g_id]
                loss_expr = self._get_loss_expression(X_g, y_g, w, b)
                loss_expressions_map[g_id] = loss_expr
                constraints.append(loss_expr <= t)
            
            # Objective: Minimize the ceiling 't' + Regularization (if any)
            # Note: We keep regularization in every step to prevent weights from exploding 
            # when optimizing for secondary groups.
            objective_expr = t
            if self.alpha > 0:
                objective_expr += self.alpha * cp.sum_squares(w)

            objective = cp.Minimize(objective_expr)
            prob = cp.Problem(objective, constraints)
            
            try:
                prob.solve(solver=self.solver)
            except cp.SolverError as e:
                warnings.warn(f"Solver failed at iteration {iteration}: {e}")
                break

            if prob.status not in ["optimal", "optimal_inaccurate"]:
                warnings.warn(f"Optimization failed or incomplete: {prob.status}")
                break
            
            # Note: t.value is the theoretical upper bound, but with regularization 
            # it might not exactly match the max group loss (due to the trade-off).
            # We look at the realized losses to determine bottlenecks.
            
            # 3. Identify binding groups (Bottlenecks)
            newly_fixed = []
            
            # Calculate realized losses for active groups using current w, b
            realized_losses = []
            for g_id in active_groups:
                # Evaluate expression
                val = loss_expressions_map[g_id].value
                realized_losses.append((val, g_id))
            
            # Sort descending (highest loss first)
            realized_losses.sort(key=lambda x: x[0], reverse=True)
            
            # The highest realized loss among active groups
            max_val = realized_losses[0][0]
            
            if self.verbose:
                print(f"Max realized loss for active groups: {max_val:.5f}")

            # We fix everyone within 'tol' of that max realized loss
            for val, g_id in realized_losses:
                if val >= max_val - self.tol:
                    newly_fixed.append(g_id)
                    fixed_constraints.append((g_id, val))
            
            if self.verbose:
                print(f"Locking groups: {newly_fixed}")

            # Remove newly fixed from active
            for g_id in newly_fixed:
                if g_id in active_groups:
                    active_groups.remove(g_id)
                    
            # Safety break: If numerical issues prevent locking, force lock the worst one
            if not newly_fixed and active_groups:
                worst_g_id = realized_losses[0][1]
                worst_val = realized_losses[0][0]
                fixed_constraints.append((worst_g_id, worst_val))
                active_groups.remove(worst_g_id)
                if self.verbose:
                    print(f"Force locking group {worst_g_id} to ensure progress.")

        # Store fitted coefficients
        self.coef_ = w.value
        if self.fit_intercept:
            self.intercept_ = b.value
        else:
            self.intercept_ = 0.0
            
        self.is_fitted_ = True
        
        return self

    def _get_loss_expression(self, X, y, w, b):
        """Constructs the CVXPY expression for MSE or MAE."""
        n = X.shape[0]
        # Note: b is either a Variable or 0.0
        residuals = X @ w + b - y
        
        if self.loss == 'mse':
            # Sum of Squares / n
            return cp.sum_squares(residuals) / n
        elif self.loss == 'mae':
            # Sum of Abs / n
            return cp.norm(residuals, 1) / n
        else:
            raise ValueError("loss must be 'mse' or 'mae'")

    def predict(self, X):
        """Predict using the linear model."""
        check_is_fitted(self, ['coef_', 'intercept_'])
        X = check_array(X)
        return X @ self.coef_ + self.intercept_

    def __str__(self):
        return "LexicographicFairRegressor"

import numpy as np

import cvxpy as cp
_HAS_CVXPY = True

# Proportional Fairness on the POF paper
class ProportionalFairRegressor(BaseEstimator, RegressorMixin):
    """
    A Linear Regression estimator that optimizes for Proportional Fairness.
    
    Proportional Fairness (Nash Bargaining Solution) maximizes the sum of the 
    logarithms of "utilities". In the context of regression, Utility is defined 
    as the improvement over a baseline (disagreement point).
    
    Objective:
        Maximize Sum_g [ log( BaselineLoss_g - ModelLoss_g ) ]
        
    This incentivizes improving the model for all groups, but gives higher 
    marginal utility to groups that are close to the baseline (i.e., performing poorly).
    
    Parameters
    ----------
    groups : list of lists of int
        A list where each element is a list of integers representing the 
        row indices of X that belong to a specific group. 

    loss : str, default='mse'
        The loss function to minimize. Options are 'mse' or 'mae'.
        
    baseline_mode : str, default='variance'
        How to calculate the Baseline Loss (the disagreement point).
        - 'variance': Baseline is the best constant prediction for that SPECIFIC group 
          (Mean for MSE, Median for MAE).
        - 'global_constant': Baseline is the best constant prediction for the ENTIRE dataset
          (Global Mean for MSE, Global Median for MAE) applied to each group.
        - 'zero': Baseline is predicting 0 (useful if data is centered/standardized).
    
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.

    alpha : float, default=0.0
        Regularization strength (L2 penalty). Note that in Proportional Fairness,
        regularization is subtracted from the objective (log utility).
        
    solver : str, optional
        The cvxpy solver to use.
        
    verbose : bool, default=False
        If True, prints optimization status.
    """
    def __init__(self, groups, loss='mse', baseline_mode='variance', 
                 fit_intercept=True, alpha=0.0, solver=None, verbose=False):
        self.groups = groups
        self.loss = loss
        self.baseline_mode = baseline_mode
        self.fit_intercept = fit_intercept
        self.alpha = alpha
        self.solver = solver
        self.verbose = verbose

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative.")

        n_samples, n_features = X.shape
        w = cp.Variable(n_features)
        
        if self.fit_intercept:
            b = cp.Variable()
        else:
            b = 0.0

        # Build Objective
        log_utilities = []
        constraints = []
        
        # Pre-compute Global Constant if needed
        global_pred = 0.0
        if self.baseline_mode == 'global_constant':
            if self.loss == 'mse':
                global_pred = np.mean(y)
            else:
                global_pred = np.median(y)

        for g_idx, indices in enumerate(self.groups):
            indices = np.array(indices)
            X_g = X[indices]
            y_g = y[indices]
            
            # 1. Calculate Baseline Loss (The "Disagreement Point")
            baseline_loss = 1.0 # Default fallback
            
            if self.baseline_mode == 'variance':
                # Local Constant Prediction
                if self.loss == 'mse':
                    local_mean = np.mean(y_g)
                    baseline_loss = np.mean((y_g - local_mean)**2)
                else:
                    local_median = np.median(y_g)
                    baseline_loss = np.mean(np.abs(y_g - local_median))
                    
            elif self.baseline_mode == 'global_constant':
                # Global Constant Prediction applied to this group
                if self.loss == 'mse':
                    baseline_loss = np.mean((y_g - global_pred)**2)
                else:
                    baseline_loss = np.mean(np.abs(y_g - global_pred))
                    
            elif self.baseline_mode == 'zero':
                if self.loss == 'mse':
                    baseline_loss = np.mean(y_g**2)
                else:
                    baseline_loss = np.mean(np.abs(y_g))
            else:
                raise ValueError("Unknown baseline_mode. Choose 'variance', 'global_constant', or 'zero'.")
            
            # Avoid numerical issues if baseline is 0 (perfect data)
            baseline_loss = max(baseline_loss, 1e-6)

            # 2. Get Model Loss Expression
            loss_expr = self._get_loss_expression(X_g, y_g, w, b)
            
            # 3. Define Utility = Baseline - Loss
            # We want to Maximize log(Baseline - Loss)
            # This implicitly constrains Loss < Baseline
            utility = baseline_loss - loss_expr
            
            # We assume we can at least marginally beat the baseline. 
            # If the model is worse than the baseline, utility is negative, problem infeasible.
            log_utilities.append(cp.log(utility))

        # Sum of logs (product of utilities)
        objective_expr = cp.sum(log_utilities)
        
        # Regularization: Penalize the "utility"
        if self.alpha > 0:
            objective_expr -= self.alpha * cp.sum_squares(w)

        objective = cp.Maximize(objective_expr)
        prob = cp.Problem(objective, constraints)
        
        try:
            prob.solve(solver=self.solver)
        except cp.SolverError as e:
            warnings.warn(f"Solver failed: {e}")

        if prob.status not in ["optimal", "optimal_inaccurate"]:
            warnings.warn(f"Optimization failed or incomplete: {prob.status}. "
                          "This often happens if the model cannot beat the baseline "
                          "for one specific group (making log(utility) undefined).")

        self.coef_ = w.value
        if self.fit_intercept:
            self.intercept_ = b.value
        else:
            self.intercept_ = 0.0
            
        self.is_fitted_ = True
        return self

    def _get_loss_expression(self, X, y, w, b):
        n = X.shape[0]
        residuals = X @ w + b - y
        if self.loss == 'mse':
            return cp.sum_squares(residuals) / n
        elif self.loss == 'mae':
            return cp.norm(residuals, 1) / n
        else:
            raise ValueError("loss must be 'mse' or 'mae'")

    def predict(self, X):
        check_is_fitted(self, ['coef_', 'intercept_'])
        X = check_array(X)
        return X @ self.coef_ + self.intercept_
    
    def __str__(self):
        return "ProportionalFairRegressor"



class LogResidualRegression:
    """
    Regression model solving

        min_beta  sum_i log(r_i(beta))

    with residuals defined as:
      - loss='mse':  r_i(beta) = (y_i - x_i^T beta)^2 + eps
      - loss='mae':  r_i(beta) = |y_i - x_i^T beta| + eps

    Optimization is done by a majorization-minimization / IRLS scheme:
      - at each iteration, set weights w_i = 1 / r_i(beta^(k)),
      - solve a convex weighted regression subproblem:
          * 'mse': weighted least squares
          * 'mae': weighted LAD (via cvxpy)

    Parameters
    ----------
    loss : {'mse', 'mae'}, default='mse'
        Type of base residual.
    fit_intercept : bool, default=True
        Whether to estimate an intercept.
    l2_reg : float, default=0.0
        Optional ridge penalty (lambda) on beta.
    eps : float, default=1e-6
        Small positive constant added inside r_i to avoid log(0) and
        huge weights.
    max_iter : int, default=50
        Maximum number of IRLS iterations.
    tol : float, default=1e-5
        Relative tolerance on change in objective sum_i log(r_i).
    weight_cap : float, default=1e6
        Cap on individual weights w_i to avoid extreme influence.
    verbose : bool, default=False
        If True, prints objective values during optimization.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        Estimated coefficients.
    intercept_ : float
        Estimated intercept.
    n_iter_ : int
        Number of IRLS iterations performed.
    obj_history_ : list of float
        Objective values (sum log r_i) at each iteration.
    """

    def __init__(self,
                 loss: str = "mse",
                 fit_intercept: bool = True,
                 l2_reg: float = 0.0,
                 eps: float = 1e-6,
                 max_iter: int = 50,
                 tol: float = 1e-5,
                 weight_cap: float = 1e6,
                 verbose: bool = False):
        if loss not in ("mse", "mae"):
            raise ValueError("loss must be 'mse' or 'mae'")
        self.loss = loss
        self.fit_intercept = fit_intercept
        self.l2_reg = float(l2_reg)
        self.eps = float(eps)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.weight_cap = float(weight_cap)
        self.verbose = verbose

        # learned parameters
        self.coef_ = None
        self.intercept_ = 0.0
        self.n_iter_ = 0
        self.obj_history_ = []

    # ---- utilities ----
    def _prepare_X(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n, p = X.shape
        if self.fit_intercept:
            X_ext = np.hstack([X, np.ones((n, 1))])
        else:
            X_ext = X
        return X_ext

    def _split_beta(self, beta_ext):
        """Split extended beta into (coef_, intercept_)."""
        if self.fit_intercept:
            return beta_ext[:-1], beta_ext[-1]
        else:
            return beta_ext, 0.0

    def _objective(self, r):
        """Compute sum_i log(r_i). r must be strictly positive."""
        return float(np.sum(np.log(r)))

    def _compute_residuals(self, X_ext, y, beta_ext):
        pred = X_ext @ beta_ext
        e = y - pred
        if self.loss == "mse":
            r = e**2 + self.eps
        else:  # 'mae'
            r = np.abs(e) + self.eps
        return e, r

    # ---- core IRLS procedure ----
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        n, p = X.shape

        X_ext = self._prepare_X(X)
        _, p_ext = X_ext.shape

        beta_optimal = None

        # Initialization: plain OLS (ridge if l2_reg>0) on squared loss
        XtX = X_ext.T @ X_ext
        if self.l2_reg > 0:
            # Do not regularize intercept (last coordinate)
            reg = self.l2_reg * np.eye(p_ext)
            if self.fit_intercept:
                reg[-1, -1] = 0.0
            XtX = XtX + reg
        Xty = X_ext.T @ y
        try:
            beta_ext = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            beta_ext = np.linalg.lstsq(XtX, Xty, rcond=None)[0]

        # IRLS loop
        self.obj_history_ = []
        obj_prev = np.inf

        for it in range(self.max_iter):
            e, r = self._compute_residuals(X_ext, y, beta_ext)

            # weights = 1 / r_i
            w = 1.0 / r
            # cap weights to avoid extreme influence
            w = np.minimum(w, self.weight_cap)

            # current objective
            obj = self._objective(r)
            self.obj_history_.append(obj)
            if obj >= np.max(self.obj_history_):
                beta_optimal = beta_ext

            if self.verbose:
                print(f"[Iter {it}] obj = {obj:.6f}")

            # check convergence (relative)
            if it > 0:
                rel_change = abs(obj - obj_prev) / (abs(obj_prev) + 1e-12)
                if rel_change < self.tol:
                    break
            obj_prev = obj

            # Solve weighted subproblem
            if self.loss == "mse":
                # Weighted least squares: min sum_i w_i * e_i^2
                sqrt_w = np.sqrt(w)
                Xw = X_ext * sqrt_w[:, None]
                yw = y * sqrt_w
                XtX_w = Xw.T @ Xw
                if self.l2_reg > 0:
                    reg = self.l2_reg * np.eye(p_ext)
                    if self.fit_intercept:
                        reg[-1, -1] = 0.0
                    XtX_w = XtX_w + reg
                Xty_w = Xw.T @ yw
                try:
                    beta_ext = np.linalg.solve(XtX_w, Xty_w)
                except np.linalg.LinAlgError:
                    beta_ext = np.linalg.lstsq(XtX_w, Xty_w, rcond=None)[0]
            else:  # 'mae'
                if not _HAS_CVXPY:
                    raise RuntimeError(
                        "loss='mae' requires cvxpy to solve weighted LAD subproblems."
                    )
                # Weighted LAD: min sum_i w_i * |e_i|
                beta_var = cp.Variable(p_ext)
                residuals = y - X_ext @ beta_var
                objective = cp.Minimize(cp.sum(cp.multiply(w, cp.abs(residuals))))
                constraints = []
                if self.l2_reg > 0:
                    # Add small ridge term for stability
                    objective = cp.Minimize(
                        cp.sum(cp.multiply(w, cp.abs(residuals)))
                        + 0.5 * self.l2_reg * cp.sum_squares(beta_var[:-1 if self.fit_intercept else None])
                    )
                prob = cp.Problem(objective, constraints)
                prob.solve(solver=cp.ECOS, verbose=self.verbose)
                if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                    raise RuntimeError(f"LAD subproblem failed, status={prob.status}")
                beta_ext = beta_var.value

        # store
        self.n_iter_ = it + 1
        self.coef_, self.intercept_ = self._split_beta(beta_optimal)
        return self

    def predict(self, X):
        if self.coef_ is None:
            raise RuntimeError("Call fit before predict.")
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y_pred = X @ self.coef_
        if self.fit_intercept:
            y_pred = y_pred + self.intercept_
        return y_pred

    def score(self, X, y):
        """
        R^2 score (coefficient of determination).
        """
        y = np.asarray(y).ravel()
        y_pred = self.predict(X)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-16
        return 1.0 - ss_res / ss_tot
    
    def __str__(self):
        return f"LogResidualRegression"#(fit_intercept={self.fit_intercept}, max_iter={self.max_iter}, tol={self.tol})"
