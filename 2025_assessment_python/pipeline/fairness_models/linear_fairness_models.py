import cvxpy as cp
import gurobipy as gp
import mosek
import numpy as np

from time import time

from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import root_mean_squared_error

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
    def __init__(self, fit_intercept=True, solver="GUROBI", k_percentage=0.7, lambda_l1=1e-3, lambda_l2=1e-1,
                 objective="mae"):
        self.beta = None
        self.fit_intercept = fit_intercept
        self.solver = solver
        self.k_percentage = k_percentage
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.objective = objective

    def fit(self, X, y):
        try:
            X, y = X.to_numpy(), y.to_numpy()
        except Exception as e:
            pass
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        n, m = X.shape

        # k_samples 
        k_samples = int(n * self.k_percentage) 

        # Primal approach of the problem
        beta = cp.Variable(m)
        nu = cp.Variable(1)
        lamb = cp.Variable(n, nonneg=True)
        z = cp.Variable(m)
        u = cp.Variable(1)

        # Regularizer constraints
        constraints =[
            beta <= z,
            -beta <= z,
            cp.SOC(u, beta),
            # Fairness <= delta,
        ]

        # Objective constraints
        if self.objective == "mae":
            constraints += [
                y - X @ beta <= nu + lamb,
                -y + X @ beta <= nu + lamb,
            ]
        elif self.objective == "mse":
            constraints +=[
               cp.square(y[i] - X[i,:] @ beta) <= nu + lamb[i] for i in range(n) 
            ]

        # Objective: <=> Minimize the overall Mean Absolute Error
        primal_prob = cp.Problem(
            cp.Minimize(nu * k_samples + cp.sum(lamb) + self.lambda_l1 * cp.sum(z) + self.lambda_l2 * u**2), 
            constraints
        )

        # # Fairness constraint
        # n_groups = 3
        # interval_size = y.max() - y.min() + 1e-6 # a little bit more
        # bins = y.min() + np.array([i*interval_size /n_groups for i in range(n_groups+1)])
        # X_bins, y_bins = [], []
        # bin_indices_list = []
        # for j,lb in enumerate(bins[:-1]):
        #     ub = bins[j+1]
        #     bin_indices = np.where((y>=lb) & (y<ub))[0]
        #     # print(bin_indices)
        #     bin_indices_list.append(bin_indices)
        #     # Data
        #     X_bins.append(X[bin_indices,:])
        #     y_bins.append(y[bin_indices])

        # # Compute the actual max difference
        # tau = 0 
        # for i in range(n_groups):
        #     print(i, len(bin_indices_list[i]))
        #     constraints+=[
        #         cp.mean(z[bin_indices_list[i]])  <= u_g,
        #         cp.mean(z[bin_indices_list[i]])  >= l_g,
        #     ]
        #     for j in range(i+1, n_groups):
        #         diff_ij = np.abs(np.mean(real_z[bin_indices_list[i]]) - np.mean(real_z[bin_indices_list[j]]))
        #         if diff_ij >  tau:
        #             tau = diff_ij


        # Solve the optimization problem
        t0 = time()
        try:
            result = primal_prob.solve(solver=self.solver, verbose=False)
        except cp.error.SolverError:
            print("GUROBI not available, trying default solver.")
            result = primal_prob.solve(verbose=False)
        solve_time = time() - t0
        print(f"Problem status: {primal_prob.status}")
        print(f"Optimal objective (Weighted Mean Absolute Error): {result}")
        print(f"Solving time: {solve_time}")
        print(f"Selected betas: {np.sum(np.abs(beta.value) >= 1e-4)}")

        # Print the difference in MAE between groups post-optimization
        if primal_prob.status in ["optimal", "optimal_inaccurate"]:
            self.beta = beta.value
        else:
            print("Solver did not find an optimal solution. Beta coefficients not set.")
            self.beta = np.zeros(m) # Fallback beta

        return result, solve_time

    def predict(self, X):
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        return X @ self.beta
    
    def __str__(self):
        return f"StableRegression(fit_intercept={self.fit_intercept})"



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
            y_ = y.copy()
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



    
def get_group_bins_indices(y, n_groups=3):
    interval_size = y.max() - y.min() + 1e-6 # a little bit more
    bins = y.min() + np.array([i*interval_size /n_groups for i in range(n_groups+1)])
    bin_indices_list = []
    for j,lb in enumerate(bins[:-1]):
        ub = bins[j+1]
        bin_indices = np.where((y>=lb) & (y<ub))[0]
        bin_indices_list.append(bin_indices)
    return bin_indices_list

class GroupDeviationConstrainedLogisticRegression:
    #  add_rmse_constraint=False,
    def __init__(self, fit_intercept=True, percentage_increase=0.00, n_groups=3, solver="GUROBI", max_row_norm_scaling=1, objective="mse", constraint="max_mse", l2_lambda=1e-3, eps=1e-4, model_name="logistic"):
        self.beta = None
        self.fit_intercept = fit_intercept
        self.percentage_increase = percentage_increase
        self.model_name=model_name
        # Ooptimization
        self.objective = objective # mae / mse
        self.constraint = constraint # max_mae / max_mse / max_mae_diff / max_mse_diff
        self.solver = solver
        self.eps = eps
        # self.add_rmse_constraint = add_rmse_constraint
        self.l2_lambda = l2_lambda

        # Group constraints
        self.n_groups = n_groups
        self.max_row_norm_scaling = max_row_norm_scaling



    def fit(self, X, y, y_real_values=None):
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
        
        def get_bregman_divergence_value(X, y, beta, model="linear", eps=1e-4):
            theta = X @ beta
            y_ = y.copy()
            if model == "linear":
                eta = y_ 
                psi = lambda z: z**2/2
            elif model=="logistic": 
                y_[y > 1-eps] = 1-eps
                y_[y < eps] = eps 
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
        J_0 = get_bregman_divergence_value(X, y, beta_0, model=self.model_name, eps=self.eps) 
        w_0, w_0_2 = get_psi_derivatives(X, y, beta_0, model=self.model_name)#, gamma=self.eps)
        # w_0 = np.exp(X @ beta_0) / ( 1 + np.exp(X @ beta_0) )
        a_0 = (1/n) * X.T @ (w_0 - y) if self.model_name != "svm" else (1/n) * X.T @ w_0 # gradient of J_0
        b_0 = beta_0 # Gradient of the l2-norm term
        H_0 = (1/n) * X.T @ np.diag(w_0_2) @ X # Hessian of J_0
        # H_0 = np.mean( [ np.exp(X[i,:] @ beta_0) / ( 1 + np.exp(X[i,:] @ beta_0) )**2 * np.outer(X[i,:],  X[i,:]) for i in range(n) ], axis=0 ) 
        H_0_inv = np.linalg.pinv(H_0)
        M_psi = 1 if self.model_name != "linear" else 0# for logistic (I think for SVM we maintain the 1)

        print("Predicted y's: ", glm.predict(X[:10,:]))
        print("The real  y's: ", np.sign(y[:10]))

        # Fairness constraints utils
        bin_indices_list = get_group_bins_indices(y_real_values, n_groups=self.n_groups)
        tau = 0 # Compute the max diference of the unconstrained problem
        loss_0 = get_loss_value(X, y, beta_0, model=self.model_name)#j_0#np.mean( np.log( 1 + np.exp(X @ beta_0) ) - y * (X @ beta_0) )
        print("loss_0: ", loss_0)
        b_d_0 = get_bregman_divergence_value(X, y, beta_0, model=self.model_name, eps=self.eps)
        acc_0 = np.average(y == np.sign(X @ beta_0)) 
        print("Accuracy 0: ", acc_0)
        print("Bregman divergence 0: ", b_d_0)
        for g in range(self.n_groups):
            print("Group: ", g, len(bin_indices_list[g]))
            X_g , y_g = X[bin_indices_list[g], :], y[bin_indices_list[g]]
            # theta_0_g = X_g @ beta_0
            loss_0_g = get_loss_value(X_g, y_g, beta_0, model=self.model_name) #np.mean( np.log( 1 + np.exp(theta_0_g) ) - y_g * theta_0_g )
            print("loss_0_g: ", loss_0_g)
            print("Accuracy 0_g: ", np.average(y_g == np.sign(X_g @ beta_0)) )

            # bregman
            b_d = get_bregman_divergence_value(X_g, y_g, beta_0, model=self.model_name, eps=self.eps)
            print("Bregman divergence_0_g: ", b_d)
            # print("Bregman 0_g OLS: ", get_bregman_divergence_value(X_g, y_g, beta_ols, model=self.model_name, eps=self.eps))
            # print("MSE/2 0_g: ", root_mean_squared_error(y_g, X_g @ beta_ols)**2/2)
            if b_d > tau:
                tau = b_d
        tau_bound = tau * (1-self.percentage_increase)
        print("tau 0: ", tau)
        print("fair tau bound: ", tau_bound)

        # To be used in constraints of CVXPY
        def get_bregman_divergence_cxpy(X, y, beta, model="logistic", eps=1e-4):
            theta = X @ beta
            y_ = y.copy()
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

        # y[y<self.eps] = self.eps
        # y[y>1-self.eps] = 1-self.eps
        constraints+=[  # Fairness constraint
            # cp.mean( cp.logistic(X[idx_g, :] @ beta) - cp.multiply(y[idx_g], X[idx_g, :] @ beta) - cp.logistic(np.log(y[idx_g] / (1-y[idx_g]))) +  cp.multiply(y[idx_g], np.log(y[idx_g] / (1-y[idx_g]))) ) <= tau_bound for idx_g in bin_indices_list
            cp.mean( get_bregman_divergence_cxpy(X[idx_g, :], y[idx_g], beta, model=self.model_name, eps=self.eps) ) <= tau_bound for idx_g in bin_indices_list

            # cp.mean( cp.logistic(X[idx_g, :] @ beta) - cp.multiply(y[idx_g], X[idx_g, :] @ beta) ) <= u for idx_g in bin_indices_list
        ]
  
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
            print(f"POF (MSE % decrease): ", price_of_fairness)
        elif self.objective == "mse": # [PENDING] Update the l2 version
            J_0, result = get_bregman_divergence_value(X, y, beta_0, model=self.model_name, eps=self.eps), get_bregman_divergence_value(X, y, beta.value, model=self.model_name, eps=self.eps)
            print(f"J_0 objective (original loss): {J_0}")
            print(f"J_F objective (current loss): {result}")
            price_of_fairness = (result-J_0)/J_0
            print(f"POF (MSE % decrease): ", price_of_fairness)
            acc_F = np.average(y == np.sign(X @ beta.value)) 
            print(f"POF Accuracy: ", (acc_0 - acc_F) / acc_0)
            # pass 

        # Approximating the F function
        real_tau, real_tau_g = 0, -1

        b_d_F = get_bregman_divergence_value(X, y, beta.value, model=self.model_name, eps=self.eps)
        print("New (F) Bregman divergence 0: ", b_d_F)
        print("New F Accuracy: ", acc_F)
        # print("Direct Taylor of MSE / 2: ", (beta.value - beta_0).T @ H_0 @ (beta.value - beta_0) / 2)
        for g, ind_g in enumerate(bin_indices_list):
            X_g, y_g = X[ind_g, :], y[ind_g]
            print("Group: ", g, len(ind_g))
            loss_g = get_loss_value(X_g, y_g, beta.value, model=self.model_name)#cp.mean( cp.logistic(theta_g) - cp.multiply(y[ind_g], theta_g) ).value
            print("New (F) group loss g: ", g, loss_g)
            # approx_error_g = np.mean( np.abs( np.exp(theta_g.value)/(1 + np.exp(theta_g.value)) - y_g ) ) 
            print("New (F) Accuracy: ", np.mean(y_g == np.sign(X_g @ beta.value)))
            b_d = get_bregman_divergence_value(X_g, y_g, beta.value, model=self.model_name, eps=self.eps)#np.mean( np.log( 1 + np.exp(theta_g) ) - y_g * theta_g - (np.log(1+ np.exp(eta_y_g)) - y_g * eta_y_g) )
            print("New (F) Bregman divergence g: ", g, b_d)
            if b_d >= real_tau:
                real_tau = b_d
                real_tau_g = g            

        # 1) Fairness improvement approximation and bounds
        fairness_improvement = np.abs(real_tau - tau)
        delta_fairness = self.percentage_increase*tau # tau - real_tau
        # virtual_fairness_improvement = np.abs(tau * (1 - self.percentage_increase) - tau)
        # Bounds on POF from MSE and MSE
        if self.objective == "mse" and self.constraint == "max_mse":
            g, indices_g = real_tau_g, bin_indices_list[real_tau_g]
            n_g, X_g, y_g = len(indices_g), X[indices_g,:], y[indices_g]

            # Taylor approximation "bounds" (not secured to be bounds, is just the approximation)
            w_0_g, w_0_2_g = get_psi_derivatives(X_g, y_g, beta_0, model=self.model_name)
            # w_0_g = np.exp(X_g @ beta_0) / ( 1 + np.exp(X_g @ beta_0) )
            a_0_g = (1/n_g) * X_g.T @ (w_0_g - y_g )
            # a_0_g_ = np.mean(X_g.T * (w_0_g - y_g ), axis=1)# gradient of the single J_g in b_0
            H_0_g = (1/n_g) * X_g.T @ np.diag(w_0_2_g) @ X_g
            # H_0_g_ = np.mean( [ np.exp(X[i,:] @ beta_0) / ( 1 + np.exp(X[i,:] @ beta_0) )**2 * np.outer(X[i,:],  X[i,:]) for i in indices_g ], axis=0 )
            H_0_g_inv = np.linalg.pinv(H_0_g)

            # Direction d:=Delta beta* from the Taylor approximation
            A_0 = a_0_g.T @ H_0_inv @ a_0_g 
            # if self.l2_lambda == 0:
            d = -(fairness_improvement) * H_0_inv @ a_0_g / A_0 # Delta beta*
            # else: # Regularized solution
            s_ = a_0_g.T @ H_0_inv @ a_0_g
            u_ = b_0.T @ H_0_inv @ b_0
            t_ = a_0_g.T @ H_0_inv @ b_0
            Δ = s_*u_ - t_**2
            d_ = -(fairness_improvement) * H_0_inv @ (u_ * a_0_g - t_ * b_0) / Δ

            d_H_0_inv_norm = (fairness_improvement)**2 / A_0 # norm H_0_inv of Delta beta* (final form of the term)

            # Hessian of J_g
            # norm of beta* with H_0_g (second-order subdifferential of F(beta_0))
            # A_0_g = a_0_g .T @ H_0_inv_g @ a_0_g 
            d_H_0_norm_g = d.T @ H_0_g @ d  # norm H_0_inv of Delta beta* (Computed directly with beta* instead of the previous one)

            # Taloy Approximation Bounds setting t=1, and d:=Delta beta*
            if M_psi > 0:
                M_phi = M_psi * np.max(np.abs( X @ d )) # Option 2 w./ Cauchy Schwarz (looser): M_psi * np.max(np.linalg.norm(X, axis=1))*np.linalg.norm(d)
                C_phi_LB = (np.exp(-M_phi) + M_phi - 1) / M_phi**2  # t = 1
                C_phi_UB = (np.exp( M_phi) - M_phi - 1) / M_phi**2  # t = 1
            else:
                M_phi, C_phi_LB, C_phi_UB = 0, .5, .5 # limit values
            delta_J_taylor = (1/2) * d_H_0_inv_norm
            delta_J_taylor_lb = C_phi_LB * d_H_0_inv_norm  # Lower bound
            delta_J_taylor_ub = C_phi_UB * d_H_0_inv_norm  # Lower bound

            print(fr"POF J % Taylor (lin-const + taylor obj.): ", delta_J_taylor / J_0)
            print(fr"POF J % LB (lin-const + exp term.): ", delta_J_taylor_lb / J_0)
            # print(fr"POF J % Taylor UB (mse-max_mse): ", delta_J_taylor_ub / J_0)

            # Lin. + quad UB construction (model-dependent). 
            if self.model_name == "linear":
                # Taylor constraint: t(a_0_g ' d) + t^2/2 ||d||_{H_0_g}^2 <= -delta
                a, b, c = d_H_0_norm_g / 2, -fairness_improvement, fairness_improvement
                t_UB_1, t_UB_2 = (-b  - np.sqrt(b**2 - 4*a*c )) / (2*a), (-b  + np.sqrt(b**2 - 4*a*c )) / (2*a)
                print("Roots for t UB: ", (t_UB_1, t_UB_2))
                t_UB = min(max(t_UB_1,0), max(t_UB_2,0))
                delta_J_taylor_ub = (t_UB**2/2) * d_H_0_inv_norm
                print(fr"POF J % UB (taylor (const. + obj.): ", delta_J_taylor_ub / J_0)
            elif self.model_name == "logistic":
                # Quadratic is UB with 1/4 of psi''
                H_0_ub = (1/n)/4 * X.T @ X
                H_0_g_ub = (1/n_g)/4 * X_g.T @ X_g
                d_H_0_norm_ub = d.T @ H_0_ub @ d
                d_H_0_norm_g_ub = d.T @ H_0_g_ub @ d

                a, b, c = d_H_0_norm_g_ub / 2, -fairness_improvement, fairness_improvement
                t_UB_1, t_UB_2 = (-b  - np.sqrt(b**2 - 4*a*c )) / (2*a), (-b  + np.sqrt(b**2 - 4*a*c )) / (2*a)
                print("Roots for t UB: ", (t_UB_1, t_UB_2))
                t_UB = min(max(t_UB_1,0), max(t_UB_2,0))
                # Future Note: constant can be either from exponential or from upper 
                delta_J_taylor_ub = (t_UB**2/2) * d_H_0_norm_ub
                print(fr"POF J % UB (taylor (const. + obj.): ", delta_J_taylor_ub / J_0)
            elif self.model_name == "poisson":
                delta_J_taylor_ub = np.zeros(delta_J_taylor_ub.size)
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
            if M_psi > 0:
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
            # Variable
            t_LB = cp.Variable(1, nonneg=True)
            # Constraint
            print("-"*100)
            print("tau_bound - tau: ", tau_bound - tau)
            print("-"*100)
            if M_psi > 0:
                constraints=[t_LB * (a_0_g @ d) + d_H_0_norm_g * ( cp.exp(-M_phi_g*t_LB) + t_LB * M_phi_g - 1 ) / M_phi_g**2  <= tau_bound - tau] #-self.percentage_increase]
                obj = cp.Minimize( d_H_0_inv_norm * (cp.exp(-M_phi*t_LB) + t_LB * M_phi - 1 ) / M_phi**2 )
            else: 
                constraints=[t_LB * (a_0_g @ d) + t_LB ** 2 * d_H_0_norm_g / 2  <= tau_bound - tau]
                obj = cp.Minimize(t_LB **2 * d_H_0_inv_norm / 2 )
            # Objective 
            primal_prob = cp.Problem(obj, constraints)
            # Solve the roots
            t0 = time()
            try:
                delta_J_lb = primal_prob.solve(solver=self.solver, verbose=False)
            except cp.error.SolverError:
                print(f"{self.solver} not available, trying default solver.")
                delta_J_lb = primal_prob.solve(verbose=True)
            print(fr"POF J % LB (exp): ", delta_J_lb / J_0)
            print(f"Root find for t_LB={t_LB.value} in: ", time() - t0)

            
            print("delta beta (normal): ", np.linalg.norm(beta_0 + t_LB.value*d))
            print("delta beta (regula): ", np.linalg.norm(beta_0 + t_LB.value*d_))
            print("norm of beta_0: ", np.linalg.norm(beta_0))
            print("norm of beta_F: ", np.linalg.norm(beta.value))

                

            # Roots finder for the UB: Newton Raphson/Bijection/Opt (1 dimension)
            # Min_{t>=0} C_phi_UB(t) * d_H_inv_norm (Delta J)
            # s.t. t*(a_0_g * Delta beta*) + d_H_0_norm_g * C_phi_UB_g(t) <= -delta (Delta F)
            # Variable
            t_UB = cp.Variable(1, nonneg=True)
            if M_psi > 0:
                # Constraint
                constraints=[t_UB * (a_0_g @ d) + d_H_0_norm_g * ( cp.exp(M_phi_g*t_UB) - t_UB * M_phi_g - 1 ) / M_phi_g**2  <= tau_bound - tau] #-self.percentage_increase]
                # Objective 
                obj = cp.Minimize( d_H_0_inv_norm * (cp.exp(M_phi*t_UB) - t_UB * M_phi - 1 ) / M_phi**2 )
            else:
                # Constraint
                constraints=[t_UB * (a_0_g @ d) +  t_UB **2 * d_H_0_norm_g / 2  <= tau_bound - tau] #-self.percentage_increase]
                # Objective 
                obj = cp.Minimize( t_UB **2 * d_H_0_inv_norm / 2 )

            primal_prob = cp.Problem(obj, constraints)
            # Solve the roots
            t0 = time()
            try:
                delta_J_ub = primal_prob.solve(solver=self.solver, verbose=False)
            except cp.error.SolverError:
                print(f"{self.solver} not available, trying default solver.")
                delta_J_ub = primal_prob.solve(verbose=True)
            print(fr"POF J % UB (exp): ", delta_J_ub / J_0)
            print(f"Root find for t_LB={t_LB.value} in: ", time() - t0)


        # Fairness improvement
        fairness_effective_improvement = fairness_improvement/tau
        print(f"FEI (% improvement)", fairness_effective_improvement)

        print(f"Time to solve: {solve_time}")

        # Print the difference in MAE between groups post-optimization
        if primal_prob.status in ["optimal", "optimal_inaccurate"]:
            self.beta = beta.value
        else:
            print("Solver did not find an optimal solution. Beta coefficients not set.")
            self.beta = np.zeros(m) # Fallback beta

        return result, solve_time, price_of_fairness, fairness_effective_improvement, delta_J_lb/ J_0, delta_J_ub/ J_0, delta_J_taylor/ J_0, delta_J_taylor_lb / J_0, delta_J_taylor_ub / J_0, real_tau

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
