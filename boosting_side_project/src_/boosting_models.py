import numpy as np
from sklearn.tree import DecisionTreeRegressor

class SimpleGradientBoosting:
    """Simple Gradient Boosting implementation following classical algorithm"""
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, adversarial_mode=False, loss_type="mse"):
        # Boosting parameters
        self.n_estimators = n_estimators  # M in the algorithm
        self.learning_rate = learning_rate
        self.trees = [] 
        self.loss_type = loss_type
        self.F0 = None

        # Adversarial parameters
        self.adversarial_mode = adversarial_mode
        self.lamb = None # dual weights
            
        # Weak learner paramters
        self.max_depth = max_depth
    
    def _loss(self, y, pred):
        """Loss function L(y, F(x)) - using squared error for simplicity"""
        if self.loss_type == "mse":
            return np.mean((y - pred) ** 2)
        # elif self.loss_type == "logistic":
        #     # sigmoid
        #     p = 1.0 / (1.0 + np.exp(-pred))
        #     # clip to avoid log(0)
        #     eps = 1e-15
        #     p = np.clip(p, eps, 1 - eps)
        #     # binary cross-entropy
        #     loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))  
        #     return loss      
    # elif self.loss_type == "poisson":
    #         return 
    
    def _gradient(self, y, F):
        """Negative gradient (pseudo-residuals): -∂L/∂F"""
        return self.lamb * y - F
    
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
                # print(f"[SimpleGradientBoosting] Current mean residual: {np.mean(residuals)}")
            
            # (b) Fit a classifier h_m(x) to pseudo residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            
            # (c) Compute multiplier γ_m (using fixed learning rate for simplicity)
            gamma_m = self.learning_rate
            
            # (d) Update the model: F_m(x) = F_{m-1}(x) + γ_m * h_m(x)
            F = F + gamma_m * tree.predict(X)
            if self.adversarial_mode: # Update the adversarial weights
                new_lamb = np.maximum( self.lamb + gamma_m * dL_dlamb, 0)  # project to nonnegative (?)
                # new_lamb /= np.sum(new_lamb)
                self.lamb = new_lamb / np.sum(new_lamb) # Normalize sum to 1
                # print("Current lambda: ", self.lamb)

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
