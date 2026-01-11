import numpy as np
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
import copy

# ---------------------------------------------------------
# 1. The Gating Network (PyTorch)
# ---------------------------------------------------------
class GatingNetwork(nn.Module):
    """
    A simple Neural Network that outputs mixing weights (Softmax).
    Input: Features X -> Output: K probabilities summing to 1.
    """
    def __init__(self, input_dim, n_experts):
        super().__init__()
        # Using a simple 1-hidden layer net for non-linear flexibility
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_experts),
            nn.Softmax(dim=1)  # Ensures outputs sum to 1 (convex combination)
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------
# 2. The MoE Model (Main Class)
# ---------------------------------------------------------
class MoELGBSmoothPenalty:
    """
    Mixture of Experts with LightGBM Experts and Neural Network Gate.
    Optimizes Global MSE + rho * Separable Surrogate via Block Coordinate Descent.
    """

    def __init__(self, 
                 n_experts=3, 
                 rho=1e-3, 
                 zero_grad_tol=1e-6, 
                 eps_y=1e-12, 
                 lgbm_params=None,
                 n_outer_iters=20,     # Number of Descent Steps (Gate <-> Experts)
                 trees_per_iter=5,     # How many trees to add to each expert per step
                 gate_lr=0.01,         # Learning rate for the Gate
                 gate_epochs=10):      # Epochs to train gate per step
        
        self.n_experts = n_experts
        self.rho = rho
        self.zero_grad_tol = zero_grad_tol
        self.eps_y = eps_y
        self.lgbm_params = lgbm_params or {'verbosity': -1}
        
        # Optimization Hyperparams
        self.n_outer_iters = n_outer_iters
        self.trees_per_iter = trees_per_iter
        self.gate_lr = gate_lr
        self.gate_epochs = gate_epochs
        
        # State
        self.experts = [] # Will hold lgb.Booster objects
        self.gate = None  # Will hold GatingNetwork
        self.y_mean_ = None

    def fit(self, X, y):
        # Convert inputs
        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float32)
        self.y_mean_ = float(np.mean(y_np))
        
        # 1. Initialize Gate
        input_dim = X_np.shape[1]
        self.gate = GatingNetwork(input_dim, self.n_experts)
        gate_optimizer = optim.Adam(self.gate.parameters(), lr=self.gate_lr)
        
        # 2. Initialize Experts (Empty Boosters)
        # We create datasets once to save memory
        lgb_train_set = lgb.Dataset(X_np, label=y_np, free_raw_data=False)
        self.experts = [None] * self.n_experts

        # -----------------------------------------------------
        # MAIN COORDINATE DESCENT LOOP
        # -----------------------------------------------------
        print(f"Starting Training: K={self.n_experts}, rho={self.rho}")
        
        for iteration in range(self.n_outer_iters):
            
            # --- A. PRE-COMPUTE CURRENT STATE ---
            # Get current gate probabilities (Fixed during Expert Update)
            X_tensor = torch.tensor(X_np)
            with torch.no_grad():
                gate_probs = self.gate(X_tensor).numpy() # Shape (N, K)
            
            # Get current expert predictions
            expert_preds = np.zeros((X_np.shape[0], self.n_experts))
            for k in range(self.n_experts):
                if self.experts[k] is None:
                    expert_preds[:, k] = self.y_mean_ # Initialization
                else:
                    expert_preds[:, k] = self.experts[k].predict(X_np)
            
            # --- B. UPDATE EXPERTS (Boosting Step) ---
            for k in range(self.n_experts):
                # 1. Calculate the "Offset" prediction from OTHER experts
                # y_hat = g_k * f_k + Sum_{j!=k} (g_j * f_j)
                # We fix the second term as a constant 'offset'
                other_preds_weighted = np.sum(
                    gate_probs[:, [j for j in range(self.n_experts) if j != k]] * expert_preds[:, [j for j in range(self.n_experts) if j != k]], 
                    axis=1
                )
                
                # 2. Define Custom Objective Closure for Expert k
                # This calculates the GLOBAL loss gradient w.r.t Expert k's output
# 2. Define Custom Objective Closure for Expert k
                # SETTING: Local Accuracy (Per Expert) + Global Fairness (Overall)
                def expert_fobj(y_true, y_pred_current_tree):
                    # y_pred_current_tree is the raw output of THIS expert (f_k)
                    
                    # A. Reconstruct Global Prediction (Needed ONLY for Fairness Penalty)
                    # y_hat = g_k * f_k + Sum_{j!=k} (g_j * f_j)
                    y_total = (gate_probs[:, k] * y_pred_current_tree) + other_preds_weighted
                    
                    # --- Term 1: Local Accuracy Gradient (The "Different Loss per Expert") ---
                    # Loss = Sum [ g_k * (f_k - y)^2 ]
                    # dLoss/df_k = g_k * 2 * (f_k - y)
                    # Note: We use y_pred_current_tree (f_k), NOT y_total
                    grad_local_mse = 2.0 * (y_pred_current_tree - y_true)
                    hess_local_mse = 2.0 * np.ones_like(y_pred_current_tree)
                    
                    # Apply the gating weight directly to the local loss
                    grad_accuracy = grad_local_mse * gate_probs[:, k]
                    hess_accuracy = hess_local_mse * gate_probs[:, k]

                    # --- Term 2: Global Fairness Gradient (The "Overall Penalty") ---
                    # Penalty = rho * Surrogate(y_total)
                    # dPenalty/df_k = rho * dSurrogate/dy_total * dy_total/df_k
                    # dy_total/df_k = gate_probs[:, k]
                    
                    z = y_true
                    zc = (y_true - self.y_mean_)
                    denom = np.maximum(np.abs(z), self.eps_y)
                    
                    # Calculate gradient w.r.t GLOBAL prediction first
                    scale = (zc / denom) ** 2
                    grad_surr_global = 2.0 * (y_total - z) * scale
                    hess_surr_global = 2.0 * scale
                    
                    # Project it onto the expert via the Gate
                    grad_fairness = self.rho * grad_surr_global * gate_probs[:, k]
                    hess_fairness = self.rho * hess_surr_global * (gate_probs[:, k] ** 2)
                    
                    # --- Combine ---
                    # The expert feels its own error strongly, plus a nudge from the global fairness goal
                    grad_final = grad_accuracy + grad_fairness
                    hess_final = hess_accuracy + hess_fairness
                    
                    # Zero tol
                    grad_final[np.abs(grad_final) < self.zero_grad_tol] = self.zero_grad_tol
                    hess_final[hess_final < self.zero_grad_tol] = self.zero_grad_tol
                    
                    return grad_final, hess_final

                # def expert_fobj(y_true, y_pred_current_tree):
                #     # Reconstruct Global Prediction
                #     # Note: y_pred_current_tree is the raw output of THIS expert so far
                #     y_total = (gate_probs[:, k] * y_pred_current_tree) + other_preds_weighted
                    
                #     # --- Compute Global Gradients (MSE + Penalty) ---
                #     # (This logic is copied from your original code)
                #     z = y_true
                #     zc = (y_true - self.y_mean_)
                #     denom = np.maximum(np.abs(z), self.eps_y)
                    
                #     # Base Gradients (MSE) w.r.t Global Prediction
                #     grad_global = 2.0 * (y_total - y_true)
                #     hess_global = 2.0 * np.ones_like(y_total)
                    
                #     # Penalty Gradients (Surrogate) w.r.t Global Prediction
                #     scale = (zc / denom) ** 2
                #     grad_pen = 2.0 * (y_total - z) * scale
                #     hess_pen = 2.0 * scale
                    
                #     grad_total = grad_global + self.rho * grad_pen
                #     hess_total = hess_global + self.rho * hess_pen
                    
                #     # --- CHAIN RULE ---
                #     # We need dL/df_k = dL/dy_total * dy_total/df_k
                #     # dy_total/df_k = gate_prob[k]
                #     grad_k = grad_total * gate_probs[:, k]
                #     hess_k = hess_total * (gate_probs[:, k] ** 2)
                    
                #     # Zero tol
                #     grad_k[np.abs(grad_k) < self.zero_grad_tol] = self.zero_grad_tol
                #     hess_k[hess_k < self.zero_grad_tol] = self.zero_grad_tol
                    
                #     return grad_k, hess_k

                # 3. Boost Expert k (Add trees to existing model)
                self.experts[k] = lgb.train(
                    params=self.lgbm_params,
                    train_set=lgb_train_set,
                    num_boost_round=self.trees_per_iter,
                    fobj=expert_fobj,
                    init_model=self.experts[k], # Continues training!
                    keep_training_booster=True
                )
                
                # Update preds for next expert's calculation
                expert_preds[:, k] = self.experts[k].predict(X_np)

            # --- C. UPDATE GATE (PyTorch Step) ---
            # We fix Expert predictions and update the Gate to minimize loss
            
            # Convert expert preds to tensor (Fixed Constant)
            expert_preds_t = torch.tensor(expert_preds, dtype=torch.float32)
            y_true_t = torch.tensor(y_np, dtype=torch.float32)
            y_mean_t = torch.tensor(self.y_mean_, dtype=torch.float32)
            
            for epoch in range(self.gate_epochs):
                gate_optimizer.zero_grad()
                
                # Forward Pass
                g_probs = self.gate(X_tensor) # differentiable
                
                # ... (forward pass code same as before) ...
                
                # Combine (Mixture)
                y_hat = torch.sum(g_probs * expert_preds_t, dim=1)
                
                # --- CHANGED: Local MSE Loss ---
                # We want the gate to minimize: Sum_k [ g_k * (f_k - y)^2 ]
                # This encourages the gate to pick the expert that is individually best.
                expert_sq_errors = (expert_preds_t - y_true_t.unsqueeze(1)) ** 2  # Shape (N, K)
                loss_local_mse = torch.mean(torch.sum(g_probs * expert_sq_errors, dim=1))
                
                # --- Global Fairness Penalty ---
                # This stays on the Global y_hat
                denom_t = torch.maximum(torch.abs(y_true_t), torch.tensor(self.eps_y))
                zc_t = y_true_t - y_mean_t
                cov_surr = ((y_hat / denom_t) - 1.0)**2 * (zc_t**2)
                loss_surr = torch.mean(cov_surr)
                
                total_loss = loss_local_mse + self.rho * loss_surr
                
                total_loss.backward()
                gate_optimizer.step()

            # --- D. LOGGING ---
            # Using the exact same print format as your request
            self._print_status(iteration, y_np, y_hat.detach().numpy(), total_loss.item(), loss_mse.item(), loss_surr.item())

        return self

    def predict(self, X):
        X_np = np.asarray(X, dtype=np.float32)
        X_t = torch.tensor(X_np)
        
        # 1. Get Weights
        with torch.no_grad():
            g_probs = self.gate(X_t).numpy()
            
        # 2. Get Expert Preds
        expert_preds = np.zeros((X_np.shape[0], self.n_experts))
        for k in range(self.n_experts):
            if self.experts[k] is not None:
                expert_preds[:, k] = self.experts[k].predict(X_np)
            else:
                expert_preds[:, k] = self.y_mean_
                
        # 3. Weighted Sum
        return np.sum(g_probs * expert_preds, axis=1)

    def _print_status(self, it, y_true, y_pred, loss_val, mse_val, surr_val):
        z = y_true
        denom = np.maximum(np.abs(z), self.eps_y)
        r = y_pred / denom
        try:
            corr = float(np.corrcoef(r, y_true)[0, 1])
        except Exception:
            corr = float('nan')
            
        print(
            f"[Iter {it+1}] "
            f"Loss value: {loss_val:.6f} "
            f"| MSE value: {mse_val:.6f} "
            f"| CovSurr value: {surr_val:.6f} "
            f"| Corr(r,y): {corr:.6f} "
        )

    def __str__(self):
        return f"MoELGBSmoothPenalty(K={self.n_experts}, rho={self.rho})"