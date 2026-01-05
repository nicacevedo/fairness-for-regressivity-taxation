import numpy as np
from sklearn.tree import DecisionTreeRegressor

class SimpleGradientBoosting:
    """Simple Gradient Boosting implementation following classical algorithm"""
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_leaf=5, min_impurity_decrease=1e-4, adversarial_mode=False, adv_learning_rate=1e-3, loss_type="mse"):
        # Boosting parameters
        self.n_estimators = n_estimators  # M in the algorithm
        self.learning_rate = learning_rate
        self.trees = [] 
        self.loss_type = loss_type
        self.F0 = None

        # Adversarial parameters
        self.adversarial_mode = adversarial_mode
        self.lamb = None # dual weights
        self.adv_learning_rate = adv_learning_rate 
            
        # Weak learner paramters
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf    # Prevents tiny leaves (implicit regularization)
        self.min_impurity_decrease = min_impurity_decrease # Requires minimum improvement
    
    def _loss(self, y, pred):
        """Loss function L(y, F(x)) - using squared error for simplicity"""
        if self.loss_type == "mse":
            return np.mean((y - pred) ** 2)
    
    def _gradient(self, y, F):
        """Negative gradient (pseudo-residuals): -∂L/∂F"""
        return self.lamb * (y - F)
    
    def _initialize(self, y):
        """Initialize: F0(x) = arg min_γ Σ L(yi, γ)"""
        # For squared error, this is just the mean
        self.F0 = np.mean(y)
        return np.full(len(y), self.F0)
    
    def fit(self, X, y):
        """Fit the gradient boosting model"""
        # Initialize
        F = self._initialize(y)
        self.lamb = np.ones_like(y)

        # Main boosting loop: for m = 1 to M-1
        for m in range(self.n_estimators):
            # (a) Calculate pseudo-residuals
            residuals = self._gradient(y, F)
            if self.adversarial_mode:
                dL_dlamb = 0.5 * ( F - y )**2 # loss gradient w.r.t dual
                n = y.size
                e = np.ones_like(y)
                print(f"[SimpleGradientBoosting] Loss: {np.sum(self.lamb*(F - y)**2)/2:.5f} | Mean res: {np.mean(residuals):.5f} | Cosine: {(e @ self.lamb)/n/np.linalg.norm(self.lamb):.5f}")
            else:
                n = y.size
                e = np.ones_like(y)
                print(f"[SimpleGradientBoosting] Loss: {np.sum((F - y)**2)/(2*n):.5f} | Mean res: {np.mean(residuals):.5f} | Cosine: {(e @ self.lamb)/n/np.linalg.norm(self.lamb):.5f}")


            # (b) Fit a classifier h_m(x) to pseudo residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, min_impurity_decrease=self.min_impurity_decrease)
            tree.fit(X, residuals)
            
            # (c) Compute multiplier γ_m (using fixed learning rate for simplicity)
            gamma_m = self.learning_rate
            
            # (d) Update the model: F_m(x) = F_{m-1}(x) + γ_m * h_m(x)
            F = F + gamma_m * tree.predict(X)
            if self.adversarial_mode: # Update the adversarial weights
                new_lamb = np.maximum( self.lamb + self.adv_learning_rate * dL_dlamb, 0)  # project to nonnegative (?)
                self.lamb = new_lamb / np.sum(new_lamb) # Normalize sum to 1

            self.trees.append((gamma_m, tree))
        
        return self
    
    def predict(self, X):
        """Predict using the boosted model"""
        # Start with F0
        F = np.full(X.shape[0], self.F0)
        
        # Add contributions from all trees
        for gamma_m, tree in self.trees:
            F += gamma_m * tree.predict(X)
        
        return F
    
    def __str__(self):
        return f"SimpleGradientBoosting(adversarial_mode={self.adversarial_mode})"



import numpy as np
from sklearn.tree import DecisionTreeRegressor


class SimpleGradientBoosting2:
    """Simple Gradient Boosting implementation following classical algorithm"""

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_leaf=5,
        min_impurity_decrease=1e-4,
        adversarial_mode=False,
        adv_learning_rate=1e-6,
        tail_fraction=0.1,
        tail_mode="cvar",
        loss_type="mse",
    ):
        # Boosting parameters
        self.n_estimators = n_estimators  # M in the algorithm
        self.learning_rate = learning_rate
        self.trees = []
        self.loss_type = loss_type
        self.F0 = None

        # Adversarial parameters
        self.adversarial_mode = adversarial_mode
        self.lamb = None  # dual weights (simplex)
        self.adv_learning_rate = adv_learning_rate
        self.tail_fraction = tail_fraction  # CVaR level α (e.g., 0.1 = worst 10%)
        self.tail_mode = tail_mode          # "cvar" or "topk"

        # Weak learner parameters
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease

    def _loss(self, y, pred):
        """Loss function L(y, F(x))"""
        if self.loss_type == "mse":
            return np.mean((y - pred) ** 2)

    def _gradient(self, y, F):
        """Negative gradient (pseudo-residuals): -∂L/∂F

        For squared loss: -∂/∂F [0.5 (y-F)^2] = (y-F).
        In adversarial mode we handle weights via sample_weight in the tree fit.
        """
        return (y - F)

    def _initialize(self, y):
        """Initialize: F0(x) = arg min_γ Σ L(yi, γ)"""
        self.F0 = np.mean(y)
        return np.full(len(y), self.F0)

    def _project_capped_simplex(self, v, cap, tol=1e-12, max_iter=60):
        """Project v onto {w: sum w = 1, 0 <= w <= cap} (CVaR dual set)."""
        v = np.asarray(v, dtype=float)
        if cap <= 0:
            raise ValueError("cap must be positive")

        # Bisection on tau in w = clip(v - tau, 0, cap)
        lo = np.min(v - cap)  # yields sum close to n*cap (max)
        hi = np.max(v)        # yields sum close to 0 (min)

        tau = 0.0
        for _ in range(max_iter):
            tau = 0.5 * (lo + hi)
            w = np.clip(v - tau, 0.0, cap)
            s = float(np.sum(w))
            if abs(s - 1.0) <= tol:
                break
            if s > 1.0:
                lo = tau  # need larger tau -> smaller sum
            else:
                hi = tau  # need smaller tau -> larger sum

        w = np.clip(v - tau, 0.0, cap)
        s = float(np.sum(w))
        if not np.isfinite(s) or s <= 0:
            # fallback to uniform feasible weights
            n = len(v)
            w = np.full(n, 1.0 / n)
            w = np.minimum(w, cap)
            w /= np.sum(w)
            return w

        w /= s
        return w

    def fit(self, X, y):
        """Fit the gradient boosting model"""
        # Reset state (important if fit() is called multiple times)
        self.trees = []

        # Initialize
        F = self._initialize(y)
        if self.adversarial_mode:
            self.lamb = np.full(len(y), 1.0 / len(y), dtype=float)

        # Main boosting loop
        for m in range(self.n_estimators):
            # (a) Pseudo-residuals
            residuals = self._gradient(y, F)

            if self.adversarial_mode:
                # d/dλ of weighted objective (on the simplex) is just the per-sample loss
                loss_i = 0.5 * (F - y) ** 2

            # e = np.ones_like(y)
            # if self.adversarial_mode:
            #     cos = (e @ self.lamb) / (np.linalg.norm(e) * np.linalg.norm(self.lamb))
            #     print(
            #         f"[SimpleGradientBoosting] Loss: {np.linalg.norm(F - y) ** 2 / 2:.5f} | "
            #         f"Mean res: {np.mean(residuals):.5f} | Cosine: {cos:.5f}"
            #     )
            # else:
            #     print(
            #         f"[SimpleGradientBoosting] Loss: {np.linalg.norm(F - y) ** 2 / 2:.5f} | "
            #         f"Mean res: {np.mean(residuals):.5f}"
            #     )

            # (b) Fit weak learner to pseudo-residuals
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
            )
            # In adversarial mode, fit with sample weights (correct weighted least-squares behavior)
            if self.adversarial_mode:
                tree.fit(X, residuals, sample_weight=self.lamb)
            else:
                tree.fit(X, residuals)

            # (c) Compute an optimal multiplier for squared loss (stable line-search)
            h = tree.predict(X)
            if self.adversarial_mode:
                w = self.lamb
                num = np.sum(w * h * (y - F))
                den = np.sum(w * h * h) + 1e-12
            else:
                num = np.sum(h * (y - F))
                den = np.sum(h * h) + 1e-12
            gamma_star = num / den

            # (d) Update model
            F = F + self.learning_rate * gamma_star * h

            # (e) Update adversarial weights (CVaR / top-k tail-risk)
            if self.adversarial_mode:
                loss_i = 0.5 * (F - y) ** 2

                # CVaR dual: max_{w in Δ, 0<=w_i<=1/(α n)} Σ w_i loss_i
                alpha = float(self.tail_fraction)
                alpha = min(max(alpha, 1.0 / len(y)), 1.0)
                cap = 1.0 / (alpha * len(y))

                if self.tail_mode == "topk":
                    k = max(1, int(np.ceil(alpha * len(y))))
                    idx = np.argpartition(loss_i, -k)[-k:]
                    self.lamb[:] = 0.0
                    self.lamb[idx] = 1.0 / k
                else:
                    # Mirror ascent step then projection onto capped simplex
                    z = self.adv_learning_rate * (loss_i - loss_i.max())
                    z = np.clip(z, -50.0, 0.0)  # avoid underflow
                    v = self.lamb * np.exp(z) + 1e-15
                    self.lamb = self._project_capped_simplex(v, cap)

            self.trees.append((self.learning_rate * gamma_star, tree))

        return self

    def predict(self, X):
        """Predict using the boosted model"""
        F = np.full(X.shape[0], self.F0)
        for gamma_m, tree in self.trees:
            F += gamma_m * tree.predict(X)
        return F

    def __str__(self):
        return f"SimpleGradientBoosting(adversarial_mode={self.adversarial_mode})"


import numpy as np
from sklearn.tree import DecisionTreeRegressor


class SimpleGradientBoosting3:
    """Simple Gradient Boosting with optional CVaR/top-k tail-risk reweighting.

    Notes:
      - loss_type controls the loss as a function of the raw score F.
      - predict() returns the mean parameter for GLM losses:
          * mse      -> F
          * logistic -> sigmoid(F)
          * poisson  -> exp(F)
    """

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_leaf=5,
        min_impurity_decrease=1e-4,
        adversarial_mode=False,
        adv_learning_rate=1e-6,
        tail_fraction=0.1,
        tail_mode="cvar",
        boosting_method="gradient",  # "gradient" or "newton"
        loss_type="mse",  # "mse", "logistic", "poisson"
    ):
        # Boosting parameters
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.trees = []
        self.loss_type = loss_type
        self.boosting_method = boosting_method
        self.F0 = None

        # Tail-risk parameters
        self.adversarial_mode = adversarial_mode
        self.lamb = None
        self.adv_learning_rate = adv_learning_rate
        self.tail_fraction = tail_fraction
        self.tail_mode = tail_mode

        # Weak learner parameters
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease

    def _sigmoid(self, t):
        t = np.clip(t, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-t))

    def _per_sample_loss(self, y, F):
        """Per-sample loss as a function of raw prediction F."""
        if self.loss_type == "mse":
            return 0.5 * (y - F) ** 2
        if self.loss_type == "logistic":
            # negative log-likelihood for Bernoulli with logit F
            return np.logaddexp(0.0, F) - y * F
        if self.loss_type == "poisson":
            # negative log-likelihood up to a constant: exp(F) - y F
            Fc = np.clip(F, -50.0, 50.0)
            mu = np.exp(Fc)
            return mu - y * Fc
        raise ValueError(f"Unknown loss_type={self.loss_type}")

    def _gradient(self, y, F):
        """Negative gradient (pseudo-residuals): -∂L/∂F."""
        if self.loss_type == "mse":
            return (y - F)
        if self.loss_type == "logistic":
            p = self._sigmoid(F)
            return (y - p)
        if self.loss_type == "poisson":
            Fc = np.clip(F, -50.0, 50.0)
            mu = np.exp(Fc)
            return (y - mu)
        raise ValueError(f"Unknown loss_type={self.loss_type}")

    def _hessian(self, y, F):
        """Second derivative ∂²L/∂F² (for Newton boosting)."""
        if self.loss_type == "mse":
            return np.ones_like(y, dtype=float)
        if self.loss_type == "logistic":
            p = self._sigmoid(F)
            return np.maximum(p * (1.0 - p), 1e-12)
        if self.loss_type == "poisson":
            Fc = np.clip(F, -50.0, 50.0)
            mu = np.exp(Fc)
            return np.maximum(mu, 1e-12)
        raise ValueError(f"Unknown loss_type={self.loss_type}")

    def _initialize(self, y):
        """Initialize F0 for the chosen loss."""
        if self.loss_type == "mse":
            self.F0 = float(np.mean(y))
        elif self.loss_type == "logistic":
            p0 = float(np.mean(y))
            p0 = min(max(p0, 1e-6), 1.0 - 1e-6)
            self.F0 = float(np.log(p0 / (1.0 - p0)))
        elif self.loss_type == "poisson":
            mu0 = float(np.mean(y))
            mu0 = max(mu0, 1e-12)
            self.F0 = float(np.log(mu0))
        else:
            raise ValueError(f"Unknown loss_type={self.loss_type}")
        return np.full(len(y), self.F0)

    def _project_capped_simplex(self, v, cap, tol=1e-12, max_iter=60):
        """Project v onto {w: sum w = 1, 0 <= w <= cap}."""
        v = np.asarray(v, dtype=float)
        if cap <= 0:
            raise ValueError("cap must be positive")

        lo = np.min(v - cap)
        hi = np.max(v)

        tau = 0.0
        for _ in range(max_iter):
            tau = 0.5 * (lo + hi)
            w = np.clip(v - tau, 0.0, cap)
            s = float(np.sum(w))
            if abs(s - 1.0) <= tol:
                break
            if s > 1.0:
                lo = tau
            else:
                hi = tau

        w = np.clip(v - tau, 0.0, cap)
        s = float(np.sum(w))
        if not np.isfinite(s) or s <= 0:
            n = len(v)
            w = np.full(n, 1.0 / n)
            w = np.minimum(w, cap)
            w /= np.sum(w)
            return w

        w /= s
        return w

    def fit(self, X, y):
        """Fit the gradient boosting model."""
        self.trees = []

        # Basic sanity checks for GLM losses
        if self.loss_type == "logistic":
            if np.any((y < 0) | (y > 1)):
                raise ValueError("For logistic loss, y must be in [0,1] (typically {0,1}).")
        if self.loss_type == "poisson":
            if np.any(y < 0):
                raise ValueError("For poisson loss, y must be >= 0.")

        F = self._initialize(y)
        if self.adversarial_mode:
            self.lamb = np.full(len(y), 1.0 / len(y), dtype=float)

        for _m in range(self.n_estimators):
            # Residuals / Newton pseudo-response
            if self.boosting_method == "newton":
                neg_g = self._gradient(y, F)
                hess = self._hessian(y, F)
                residuals = neg_g / (hess + 1e-12)  # z = -g/h = (-g)/h
                fit_w = hess
                if self.adversarial_mode:
                    fit_w = self.lamb * fit_w
            else:
                residuals = self._gradient(y, F)
                fit_w = self.lamb if self.adversarial_mode else None

            # Progress print
            loss_i_print = self._per_sample_loss(y, F)
            # e = np.ones_like(y)
            # if self.adversarial_mode:
            #     cos = (e @ self.lamb) / (np.linalg.norm(e) * np.linalg.norm(self.lamb))
            #     print(
            #         f"[SimpleGradientBoosting] Loss: {np.mean(loss_i_print):.5f} | "
            #         f"Mean res: {np.mean(residuals):.5f} | Cosine: {cos:.5f}"
            #     )
            # else:
            #     print(
            #         f"[SimpleGradientBoosting] Loss: {np.mean(loss_i_print):.5f} | "
            #         f"Mean res: {np.mean(residuals):.5f}"
            #     )

            # Fit weak learner
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
            )
            if fit_w is not None:
                tree.fit(X, residuals, sample_weight=fit_w)
            else:
                tree.fit(X, residuals)

            tree_pred = tree.predict(X)

            # Update model
            if self.boosting_method == "newton":
                # tree_pred approximates the Newton step
                F = F + self.learning_rate * tree_pred
                self.trees.append((self.learning_rate, tree))
            else:
                if self.loss_type == "mse":
                    # exact 1D line-search along tree_pred for squared loss
                    if self.adversarial_mode:
                        w = self.lamb
                        num = np.sum(w * tree_pred * (y - F))
                        den = np.sum(w * tree_pred * tree_pred) + 1e-12
                    else:
                        num = np.sum(tree_pred * (y - F))
                        den = np.sum(tree_pred * tree_pred) + 1e-12
                    gamma_star = num / den
                    F = F + self.learning_rate * gamma_star * tree_pred
                    self.trees.append((self.learning_rate * gamma_star, tree))
                else:
                    # simple first-order step for non-quadratic losses
                    F = F + self.learning_rate * tree_pred
                    self.trees.append((self.learning_rate, tree))

            # Tail-risk reweighting (CVaR / top-k) using the chosen loss
            if self.adversarial_mode:
                loss_i = self._per_sample_loss(y, F)

                alpha = float(self.tail_fraction)
                alpha = min(max(alpha, 1.0 / len(y)), 1.0)
                cap = 1.0 / (alpha * len(y))

                if self.tail_mode == "topk":
                    k = max(1, int(np.ceil(alpha * len(y))))
                    idx = np.argpartition(loss_i, -k)[-k:]
                    self.lamb[:] = 0.0
                    self.lamb[idx] = 1.0 / k
                else:
                    z = self.adv_learning_rate * (loss_i - loss_i.max())
                    z = np.clip(z, -50.0, 0.0)
                    v = self.lamb * np.exp(z) + 1e-15
                    self.lamb = self._project_capped_simplex(v, cap)

        return self

    def predict(self, X):
        """Predict mean parameter for the selected loss."""
        F = np.full(X.shape[0], self.F0)
        for gamma_m, tree in self.trees:
            F += gamma_m * tree.predict(X)

        if self.loss_type == "logistic":
            return self._sigmoid(F)
        if self.loss_type == "poisson":
            return np.exp(np.clip(F, -50.0, 50.0))
        return F

    def __str__(self):
        return f"SimpleGradientBoosting(adversarial_mode={self.adversarial_mode}, boosting_method={self.boosting_method})"


class SimpleGradientBoosting4:
    """Simple Gradient Boosting with optional CVaR/top-k tail-risk reweighting.

    Notes:
      - loss_type controls the loss as a function of the raw score F.
      - predict() returns the mean parameter for GLM losses:
          * mse      -> F
          * logistic -> sigmoid(F)
          * poisson  -> exp(F)

      - In boosting_method="newton", you can optionally damp the Newton step using
        generalized self-concordance (v=2) style surrogates via gsc_hessian_term:
          * "taylor": standard quadratic (2nd-order Taylor) => tau = 1
          * "upper" : minimize the GSC *upper* bound surrogate => tau = ln(1+beta)/beta
          * "lower" : minimize the GSC *lower* bound surrogate => tau = -ln(1-beta)/beta (beta<1)
        where beta = gsc_M * ||step||_2.

        This is a scalar damping factor tau multiplying the Newton step.
    """

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_leaf=5,
        min_impurity_decrease=1e-4,
        adversarial_mode=False,
        adv_learning_rate=1e-6,
        tail_fraction=0.1,
        tail_mode="cvar",
        boosting_method="gradient",  # "gradient" or "newton"
        loss_type="mse",  # "mse", "logistic", "poisson"
        gsc_hessian_term="taylor",  # "taylor", "upper", "lower"
        gsc_M=1.0,
    ):
        # Boosting parameters
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.trees = []
        self.loss_type = loss_type
        self.boosting_method = boosting_method
        self.F0 = None

        # GSC damping (only used in Newton mode)
        self.gsc_hessian_term = gsc_hessian_term
        self.gsc_M = float(gsc_M)

        # Tail-risk parameters
        self.adversarial_mode = adversarial_mode
        self.lamb = None
        self.adv_learning_rate = adv_learning_rate
        self.tail_fraction = tail_fraction
        self.tail_mode = tail_mode

        # Weak learner parameters
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease

    def _sigmoid(self, t):
        t = np.clip(t, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-t))

    def _per_sample_loss(self, y, F):
        """Per-sample loss as a function of raw prediction F."""
        if self.loss_type == "mse":
            return 0.5 * (y - F) ** 2
        if self.loss_type == "logistic":
            # negative log-likelihood for Bernoulli with logit F
            return np.logaddexp(0.0, F) - y * F
        if self.loss_type == "poisson":
            # negative log-likelihood up to a constant: exp(F) - y F
            Fc = np.clip(F, -50.0, 50.0)
            mu = np.exp(Fc)
            return mu - y * Fc
        raise ValueError(f"Unknown loss_type={self.loss_type}")

    def _gradient(self, y, F):
        """Negative gradient (pseudo-residuals): -∂L/∂F."""
        if self.loss_type == "mse":
            return (y - F)
        if self.loss_type == "logistic":
            p = self._sigmoid(F)
            return (y - p)
        if self.loss_type == "poisson":
            Fc = np.clip(F, -50.0, 50.0)
            mu = np.exp(Fc)
            return (y - mu)
        raise ValueError(f"Unknown loss_type={self.loss_type}")

    def _hessian(self, y, F):
        """Second derivative ∂²L/∂F² (for Newton boosting)."""
        if self.loss_type == "mse":
            return np.ones_like(y, dtype=float)
        if self.loss_type == "logistic":
            p = self._sigmoid(F)
            return np.maximum(p * (1.0 - p), 1e-12)
        if self.loss_type == "poisson":
            Fc = np.clip(F, -50.0, 50.0)
            mu = np.exp(Fc)
            return np.maximum(mu, 1e-12)
        raise ValueError(f"Unknown loss_type={self.loss_type}")

    def _initialize(self, y):
        """Initialize F0 for the chosen loss."""
        if self.loss_type == "mse":
            self.F0 = float(np.mean(y))
        elif self.loss_type == "logistic":
            p0 = float(np.mean(y))
            p0 = min(max(p0, 1e-6), 1.0 - 1e-6)
            self.F0 = float(np.log(p0 / (1.0 - p0)))
        elif self.loss_type == "poisson":
            mu0 = float(np.mean(y))
            mu0 = max(mu0, 1e-12)
            self.F0 = float(np.log(mu0))
        else:
            raise ValueError(f"Unknown loss_type={self.loss_type}")
        return np.full(len(y), self.F0)

    def _project_capped_simplex(self, v, cap, tol=1e-12, max_iter=60):
        """Project v onto {w: sum w = 1, 0 <= w <= cap}."""
        v = np.asarray(v, dtype=float)
        if cap <= 0:
            raise ValueError("cap must be positive")

        lo = np.min(v - cap)
        hi = np.max(v)

        tau = 0.0
        for _ in range(max_iter):
            tau = 0.5 * (lo + hi)
            w = np.clip(v - tau, 0.0, cap)
            s = float(np.sum(w))
            if abs(s - 1.0) <= tol:
                break
            if s > 1.0:
                lo = tau
            else:
                hi = tau

        w = np.clip(v - tau, 0.0, cap)
        s = float(np.sum(w))
        if not np.isfinite(s) or s <= 0:
            n = len(v)
            w = np.full(n, 1.0 / n)
            w = np.minimum(w, cap)
            w /= np.sum(w)
            return w

        w /= s
        return w

    def _gsc_tau(self, step_vec):
        """Scalar damping factor tau for Newton steps based on v=2 GSC surrogates.

        beta = gsc_M * ||step||_2.

        - "taylor": tau = 1
        - "upper" : tau = ln(1+beta)/beta
        - "lower" : tau = -ln(1-beta)/beta (requires beta < 1)

        If "lower" is requested but beta>=1, we fall back to the "upper" choice.
        """
        mode = self.gsc_hessian_term
        if mode is None or mode == "taylor":
            return 1.0

        beta = float(self.gsc_M) * float(np.linalg.norm(step_vec))
        if not np.isfinite(beta) or beta <= 1e-12:
            return 1.0

        if mode == "upper":
            return float(np.log1p(beta) / beta)

        if mode == "lower":
            if beta >= 1.0 - 1e-12:
                return float(np.log1p(beta) / beta)
            return float((-np.log1p(-beta)) / beta)

        raise ValueError(f"Unknown gsc_hessian_term={mode}")

    def fit(self, X, y):
        """Fit the gradient boosting model."""
        self.trees = []

        # Basic sanity checks for GLM losses
        if self.loss_type == "logistic":
            if np.any((y < 0) | (y > 1)):
                raise ValueError("For logistic loss, y must be in [0,1] (typically {0,1}).")
        if self.loss_type == "poisson":
            if np.any(y < 0):
                raise ValueError("For poisson loss, y must be >= 0.")

        F = self._initialize(y)
        if self.adversarial_mode:
            self.lamb = np.full(len(y), 1.0 / len(y), dtype=float)

        for _m in range(self.n_estimators):
            # Residuals / Newton pseudo-response
            if self.boosting_method == "newton":
                neg_g = self._gradient(y, F)
                hess = self._hessian(y, F)
                residuals = neg_g / (hess + 1e-12)  # z = -g/h = (-g)/h
                fit_w = hess
                if self.adversarial_mode:
                    fit_w = self.lamb * fit_w
            else:
                residuals = self._gradient(y, F)
                fit_w = self.lamb if self.adversarial_mode else None

            # Progress print
            loss_i_print = self._per_sample_loss(y, F)
            # e = np.ones_like(y)
            # if self.adversarial_mode:
            #     cos = (e @ self.lamb) / (np.linalg.norm(e) * np.linalg.norm(self.lamb))
            #     print(
            #         f"[SimpleGradientBoosting] Loss: {np.mean(loss_i_print):.5f} | "
            #         f"Mean res: {np.mean(residuals):.5f} | Cosine: {cos:.5f}"
            #     )
            # else:
            #     print(
            #         f"[SimpleGradientBoosting] Loss: {np.mean(loss_i_print):.5f} | "
            #         f"Mean res: {np.mean(residuals):.5f}"
            #     )

            # Fit weak learner
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
            )
            if fit_w is not None:
                tree.fit(X, residuals, sample_weight=fit_w)
            else:
                tree.fit(X, residuals)

            tree_pred = tree.predict(X)

            # Update model
            if self.boosting_method == "newton":
                # tree_pred approximates the Newton step
                tau = 1.0
                if self.loss_type in ("logistic", "poisson"):
                    tau = self._gsc_tau(tree_pred)
                step = self.learning_rate * tau

                F = F + step * tree_pred
                self.trees.append((step, tree))
            else:
                if self.loss_type == "mse":
                    # exact 1D line-search along tree_pred for squared loss
                    if self.adversarial_mode:
                        w = self.lamb
                        num = np.sum(w * tree_pred * (y - F))
                        den = np.sum(w * tree_pred * tree_pred) + 1e-12
                    else:
                        num = np.sum(tree_pred * (y - F))
                        den = np.sum(tree_pred * tree_pred) + 1e-12
                    gamma_star = num / den
                    F = F + self.learning_rate * gamma_star * tree_pred
                    self.trees.append((self.learning_rate * gamma_star, tree))
                else:
                    # simple first-order step for non-quadratic losses
                    F = F + self.learning_rate * tree_pred
                    self.trees.append((self.learning_rate, tree))

            # Tail-risk reweighting (CVaR / top-k) using the chosen loss
            if self.adversarial_mode:
                loss_i = self._per_sample_loss(y, F)

                alpha = float(self.tail_fraction)
                alpha = min(max(alpha, 1.0 / len(y)), 1.0)
                cap = 1.0 / (alpha * len(y))

                if self.tail_mode == "topk":
                    k = max(1, int(np.ceil(alpha * len(y))))
                    idx = np.argpartition(loss_i, -k)[-k:]
                    self.lamb[:] = 0.0
                    self.lamb[idx] = 1.0 / k
                else:
                    z = self.adv_learning_rate * (loss_i - loss_i.max())
                    z = np.clip(z, -50.0, 0.0)
                    v = self.lamb * np.exp(z) + 1e-15
                    self.lamb = self._project_capped_simplex(v, cap)

        return self

    def predict(self, X):
        """Predict mean parameter for the selected loss."""
        F = np.full(X.shape[0], self.F0)
        for gamma_m, tree in self.trees:
            F += gamma_m * tree.predict(X)

        if self.loss_type == "logistic":
            return self._sigmoid(F)
        if self.loss_type == "poisson":
            return np.exp(np.clip(F, -50.0, 50.0))
        return F

    def __str__(self):
        return f"SimpleGradientBoosting(adversarial_mode={self.adversarial_mode}, boosting_method={self.boosting_method}, gsc_hessian_term={self.gsc_hessian_term})"
