import cvxpy as cp
import gurobipy as gp
import mosek
import numpy as np

from time import time

from sklearn.linear_model import LinearRegression
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




class GroupDeviationConstrainedLinearRegression:
    #  add_rmse_constraint=False,
    def __init__(self, fit_intercept=True, percentage_increase=0.00, n_groups=3, solver="GUROBI", max_row_norm_scaling=1, objective="mse", l2_lambda=1e-3):
        self.beta = None
        self.fit_intercept = fit_intercept
        self.percentage_increase = percentage_increase
        # Ooptimization
        self.objective = objective
        self.solver = solver
        # self.add_rmse_constraint = add_rmse_constraint
        self.l2_lambda = l2_lambda

        # Group constraints
        self.n_groups = n_groups
        self.max_row_norm_scaling = max_row_norm_scaling



    def fit(self, X, y):
        X, y = X.to_numpy(), y.to_numpy()
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
        if self.objective == "mse":
            model = LinearRegression(fit_intercept=False)
        elif self.objective == "mae":
            model = LeastAbsoluteDeviationRegression(fit_intercept=False)
        model.fit(X, y)
        real_z = np.abs(y - model.predict(X))
        ols_mse = root_mean_squared_error(y, model.predict(X))**2
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
        X_bins, y_bins = [], []
        bin_indices_list = []
        for j,lb in enumerate(bins[:-1]):
            ub = bins[j+1]
            bin_indices = np.where((y>=lb) & (y<ub))[0]
            # print(bin_indices)
            bin_indices_list.append(bin_indices)
            # Data
            X_bins.append(X[bin_indices,:])
            y_bins.append(y[bin_indices])

        # Compute the actual max difference
        tau = 0 
        for i in range(n_groups):
            print(i, len(bin_indices_list[i]))
            constraints+=[
                cp.mean(z[bin_indices_list[i]])  <= u_g,
                cp.mean(z[bin_indices_list[i]])  >= l_g,
            ]
            for j in range(i+1, n_groups):
                diff_ij = np.abs(np.mean(real_z[bin_indices_list[i]]) - np.mean(real_z[bin_indices_list[j]]))
                if diff_ij >  tau:
                    tau = diff_ij
        print("tau", tau)
        tau_bound = tau * (1-self.percentage_increase)
        print("bound: ", tau_bound)
        constraints+=[
            u_g - l_g <= tau * (1-self.percentage_increase)
        ]

        # Objective
        if self.objective == "mse":
            if self.l2_lambda != 0:
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
        if self.objective == "mse":
            print(f"Optimal objective (RMSE): {np.sqrt(result)}")
            price_of_fairness = (result-ols_mse)/ols_mse
            print(f"POF (MSE % decrease): ", price_of_fairness)
            # price_of_fairness = (np.sqrt(result)-np.sqrt(ols_mse))/np.sqrt(ols_mse)
            # print(f"POF (RMSE % decrease): ", price_of_fairness)
        elif self.objective == "mae":
            print(f"Optimal objective (MAE): {result}")
            price_of_fairness = (result-lad_mae)/lad_mae
            print(f"POF (MAE % decrease): ", price_of_fairness)


        # Real fairness measure
        real_tau = 0
        for i in range(n_groups):
            print(i, len(bin_indices_list[i]))
            i_error = np.mean(z.value[bin_indices_list[i]])
            # print("mean: ", i_error) 
            for j in range(i+1, n_groups):
                diff_ij = np.abs(np.mean(z.value[bin_indices_list[j]] - i_error))
                if diff_ij > real_tau:
                    real_tau = diff_ij 
        fairness_improvement = np.abs(real_tau - tau)
        # print("l2_lambda: ", self.l2_lambda)
        # print("lambda_min original: ", np.min(np.linalg.eigh(X.T @ X)[0]))
        lambdas_XX = np.linalg.eigh(X.T @ X)[0]  # min eigenvalue
        min_lambda_XX = np.min(lambdas_XX) + self.l2_lambda
        print("Min eigenvalue: ", min_lambda_XX)
        # print("Max eigenvalue: ", np.max(lambdas_XX))
        max_row_norm = np.max(np.linalg.norm(X, axis=1))
        # print("Max row norm:", max_row_norm)
        pof_lower_bound = (1/(4*n)) * min_lambda_XX / max_row_norm**2 * fairness_improvement**2 
        print("POF lower bound (%)", pof_lower_bound)
        fairness_effective_improvement = fairness_improvement/tau
        print(f"FEI (% improvement)", fairness_effective_improvement)


        print(f"Time to solve: {solve_time}")

        # Print the difference in MAE between groups post-optimization
        if primal_prob.status in ["optimal", "optimal_inaccurate"]:
            self.beta = beta.value
        else:
            print("Solver did not find an optimal solution. Beta coefficients not set.")
            self.beta = np.zeros(m) # Fallback beta

        return result, solve_time, price_of_fairness, pof_lower_bound, fairness_effective_improvement

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
