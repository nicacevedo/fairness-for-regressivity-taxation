import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import OneHotEncoder

class OptimalTreeClassifier(BaseEstimator, ClassifierMixin):
    """
    Optimal Classification Tree (OCT) solver using Proximal Block Coordinate Descent 
    and ADMM.
    
    Optimized for maximizing Training Accuracy (fidelity) for tasks like 
    clustering interpretation.

    Parameters
    ----------
    max_depth : int, default=3
        The maximum depth of the tree.
        
    rho : float, default=0.001
        ADMM penalty parameter. Lower values allow the model to explore more freely 
        before snapping to grid.
        
    lr : float, default=0.1
        Learning rate for the Primal (gradient descent) step.
        
    n_admm_steps : int, default=10
        Number of outer ADMM iterations.
        
    n_primal_steps : int, default=200
        Number of inner gradient descent epochs per ADMM step.
        
    batch_size : int, default=None
        Batch size for gradient descent. If None, uses full batch (recommended for high accuracy).
        
    warm_start : bool, default=True
        If True, initializes variables using the best of multiple decision trees.
        
    n_warm_start_trees : int, default=20
        Number of random decision trees to evaluate during warm start initialization.
        The optimizer will begin from the best one found.
        
    random_state : int, default=None
        Seed for reproducibility.
        
    alpha_start : float, default=0.1
        Initial sigmoid scaling factor (temperature).
        
    alpha_end : float, default=10.0
        Final sigmoid scaling factor.
    """
    
    def __init__(self, max_depth=3, rho=0.001, lr=0.1, 
                 n_admm_steps=10, n_primal_steps=200, 
                 batch_size=None, warm_start=True, n_warm_start_trees=20,
                 random_state=None, alpha_start=0.1, alpha_end=10.0):
        self.max_depth = max_depth
        self.rho = rho
        self.lr = lr
        self.n_admm_steps = n_admm_steps
        self.n_primal_steps = n_primal_steps
        self.batch_size = batch_size
        self.warm_start = warm_start
        self.n_warm_start_trees = n_warm_start_trees
        self.random_state = random_state
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end

    def fit(self, X, y):
        """
        Build the optimal classification tree.
        """
        # 1. Validation and Setup
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_ = X.shape[1]
        
        rng = np.random.default_rng(self.random_state)
        
        # One-hot encode targets for loss calculation
        enc = OneHotEncoder(sparse_output=False, categories='auto')
        Y_onehot = enc.fit_transform(y.reshape(-1, 1))
        
        # Tree Structure: Complete Binary Tree
        self.n_internal_ = 2**self.max_depth - 1
        self.n_leaves_ = 2**self.max_depth
        
        # 2. Variable Initialization (Random Fallback)
        self.W_ = rng.standard_normal((self.n_internal_, self.n_features_)) * 0.01
        self.b_ = rng.standard_normal(self.n_internal_) * 0.01
        self.L_ = np.ones((self.n_leaves_, self.n_classes_)) / self.n_classes_
        
        # Initialize current_alpha before warm start because _update_leaves_exact needs it via _forward
        self.current_alpha = self.alpha_end

        # Warm Start Logic (Best of N)
        if self.warm_start:
            best_init_acc = -1.0
            best_init_vars = None
            
            # We will try:
            # 1. Standard Greedy CART (splitter='best') - Baseline
            # 2. N Random Trees (splitter='random') - Exploration
            
            candidate_seeds = rng.integers(0, 100000, size=self.n_warm_start_trees)
            
            # Check Standard CART first
            clf_greedy = DecisionTreeClassifier(max_depth=self.max_depth, splitter='best', random_state=self.random_state)
            clf_greedy.fit(X, y)
            
            W_g, b_g = self._convert_sklearn_tree(clf_greedy)
            L_g = self._update_leaves_exact(X, Y_onehot, W_g, b_g)
            acc_g = self._score_hard_simulated(X, y, W_g, b_g, L_g)
            
            best_init_acc = acc_g
            best_init_vars = (W_g, b_g, L_g)
            
            # Check Random Trees
            for i in range(self.n_warm_start_trees):
                clf_rand = DecisionTreeClassifier(max_depth=self.max_depth, splitter='random', random_state=candidate_seeds[i])
                clf_rand.fit(X, y)
                
                W_r, b_r = self._convert_sklearn_tree(clf_rand)
                L_r = self._update_leaves_exact(X, Y_onehot, W_r, b_r)
                acc_r = self._score_hard_simulated(X, y, W_r, b_r, L_r)
                
                if acc_r > best_init_acc:
                    best_init_acc = acc_r
                    best_init_vars = (W_r, b_r, L_r)
            
            # Apply best found initialization
            self.W_, self.b_, self.L_ = best_init_vars
            
            # Add small perturbation to help continuous optimizer escape the exact discrete corner
            # But kept small to preserve the quality of the found tree
            noise_scale = 0.01 
            self.W_ += rng.normal(0, noise_scale, self.W_.shape)
            self.b_ += rng.normal(0, noise_scale, self.b_.shape)

        # Initialize ADMM variables based on the chosen W
        self.Z_ = self._project_hard(self.W_)
        self.U_ = np.zeros_like(self.W_)
        
        # Track best model
        best_acc = self._score_hard(X, y, self.Z_, self.b_)
        best_state = (self.Z_.copy(), self.b_.copy(), self.L_.copy())
        
        # Adam Optimizer State
        m_W, v_W = np.zeros_like(self.W_), np.zeros_like(self.W_)
        m_b, v_b = np.zeros_like(self.b_), np.zeros_like(self.b_)
        t = 0
        
        # 3. Optimization Loop
        for admm_iter in range(self.n_admm_steps):
            
            # Anneal Alpha
            progress = admm_iter / max(1, self.n_admm_steps - 1)
            self.current_alpha = self.alpha_start + progress * (self.alpha_end - self.alpha_start)
            
            # --- Block A: Primal Update (Gradient Descent) ---
            for epoch in range(self.n_primal_steps):
                t += 1
                
                # Batching
                if self.batch_size is None:
                    X_batch, Y_batch = X, Y_onehot
                else:
                    idx = rng.choice(X.shape[0], size=self.batch_size, replace=False)
                    X_batch, Y_batch = X[idx], Y_onehot[idx]
                
                # Forward Pass
                path_probs, node_probs_list = self._forward(X_batch, self.W_, self.b_)
                y_pred_soft = path_probs @ self.L_ 
                y_pred_soft = np.clip(y_pred_soft, 1e-7, 1.0 - 1e-7)
                
                # Backward Pass
                grad_path_probs = - (Y_batch / y_pred_soft) @ self.L_.T
                dW_ce, db_ce = self._backward(X_batch, node_probs_list, grad_path_probs)
                
                # ADMM Penalty
                dW_admm = self.rho * (self.W_ - self.Z_ + self.U_)
                
                grad_W = dW_ce + dW_admm
                grad_b = db_ce 
                
                # Update
                m_W, v_W, self.W_ = self._adam_step(self.W_, grad_W, m_W, v_W, t)
                m_b, v_b, self.b_ = self._adam_step(self.b_, grad_b, m_b, v_b, t)

            # --- Block B: Leaves Update ---
            self.L_ = self._update_leaves_exact(X, Y_onehot, self.W_, self.b_)
            
            # --- Block C: Consensus ---
            Target = self.W_ + self.U_
            self.Z_ = self._project_hard(Target)
            
            # --- Block D: Dual Update ---
            self.U_ = self.U_ + (self.W_ - self.Z_)
            
            # Check Training Accuracy (Hard)
            current_hard_acc = self._score_hard(X, y, self.Z_, self.b_)
            
            # Allow update if equal (drift) or better
            if current_hard_acc >= best_acc:
                best_acc = current_hard_acc
                best_state = (self.Z_.copy(), self.b_.copy(), self.L_.copy())
        
        # 4. Finalize
        if best_state is not None:
            self.W_, self.b_, self.L_ = best_state
        else:
            self.W_ = self.Z_.copy()
            self.L_ = self._update_leaves_exact(X, Y_onehot, self.W_, self.b_)
            
        return self

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        hard_paths = self._forward_hard(X)
        return hard_paths @ self.L_

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
    
    # --- Helper Methods ---

    def _convert_sklearn_tree(self, clf):
        """Convert a fitted sklearn DecisionTreeClassifier to OCT parameters (W, b)."""
        tree = clf.tree_
        W = np.zeros((self.n_internal_, self.n_features_))
        b = np.zeros(self.n_internal_)
        
        # Queue: (sklearn_node_id, oct_node_id)
        queue = [(0, 0)]
        
        while queue:
            sk_id, my_id = queue.pop(0)
            if my_id >= self.n_internal_: continue
            
            if tree.feature[sk_id] != -2:
                feat = tree.feature[sk_id]
                thresh = tree.threshold[sk_id]
                
                W[my_id, feat] = 1.0
                b[my_id] = thresh
                
                queue.append((tree.children_left[sk_id], 2*my_id + 1))
                queue.append((tree.children_right[sk_id], 2*my_id + 2))
            else:
                # Sklearn leaf, but OCT internal node. 
                # Effectively prune by creating impossible splits or routing all one way.
                # Here we just leave as zeros (feature 0 > 0?), effectively random/static
                pass
        return W, b

    def _score_hard_simulated(self, X, y, W, b, L):
        """Calculate hard accuracy for a candidate set of parameters without storing them."""
        # Need to simulate forward pass with these params
        decisions = (X @ W.T - b) > 0
        n_samples = X.shape[0]
        curr_node = np.zeros(n_samples, dtype=int)
        for _ in range(self.max_depth):
            move_right = decisions[np.arange(n_samples), curr_node]
            curr_node = 2 * curr_node + 1 + move_right.astype(int)
        leaf_indices = curr_node - self.n_internal_
        path_matrix = np.zeros((n_samples, self.n_leaves_))
        path_matrix[np.arange(n_samples), leaf_indices] = 1.0
        
        probs = path_matrix @ L
        preds = self.classes_[np.argmax(probs, axis=1)]
        return np.mean(preds == y)

    def _forward(self, X, W, b):
        logits = (X @ W.T - b) * self.current_alpha
        mu = np.zeros_like(logits)
        pos_mask = logits >= 0
        neg_mask = ~pos_mask
        mu[pos_mask] = 1.0 / (1.0 + np.exp(-logits[pos_mask]))
        z = np.exp(logits[neg_mask])
        mu[neg_mask] = z / (1.0 + z)
        
        n_samples = X.shape[0]
        path_probs = np.ones((n_samples, self.n_leaves_))
        for leaf_idx in range(self.n_leaves_):
            current = leaf_idx + self.n_internal_
            while current > 0:
                parent = (current - 1) // 2
                is_right_child = (current % 2 == 0)
                prob_right = mu[:, parent]
                if is_right_child:
                    path_probs[:, leaf_idx] *= prob_right
                else:
                    path_probs[:, leaf_idx] *= (1.0 - prob_right)
                current = parent
        return path_probs, mu

    def _backward(self, X, mu, grad_path_probs):
        path_probs, _ = self._forward(X, self.W_, self.b_) 
        n_samples = X.shape[0]
        d_logit = np.zeros((n_samples, self.n_internal_))
        
        for j in range(self.n_internal_):
            left_child_leaves = self._get_descendant_leaves(2*j + 1)
            right_child_leaves = self._get_descendant_leaves(2*j + 2)
            sum_grad_p_right = np.sum(path_probs[:, right_child_leaves] * grad_path_probs[:, right_child_leaves], axis=1)
            sum_grad_p_left  = np.sum(path_probs[:, left_child_leaves] * grad_path_probs[:, left_child_leaves], axis=1)
            d_logit[:, j] = self.current_alpha * (sum_grad_p_right * (1.0 - mu[:, j]) - sum_grad_p_left * mu[:, j])
            
        dW = d_logit.T @ X
        db = -np.sum(d_logit, axis=0)
        return dW / n_samples, db / n_samples

    def _adam_step(self, params, grads, m, v, t, beta1=0.9, beta2=0.999, eps=1e-8):
        m = beta1 * m + (1 - beta1) * grads
        v = beta2 * v + (1 - beta2) * (grads ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        params = params - self.lr * m_hat / (np.sqrt(v_hat) + eps)
        return m, v, params

    def _get_descendant_leaves(self, node_idx):
        if node_idx >= self.n_internal_:
            return [node_idx - self.n_internal_]
        return self._get_descendant_leaves(2*node_idx + 1) + \
               self._get_descendant_leaves(2*node_idx + 2)

    def _project_hard(self, W):
        Z = np.zeros_like(W)
        best_feats = np.argmax(W, axis=1)
        for i in range(W.shape[0]):
            Z[i, best_feats[i]] = 1.0
        return Z

    def _update_leaves_exact(self, X, Y_onehot, W, b):
        path_probs, _ = self._forward(X, W, b)
        leaf_weights = path_probs.T @ Y_onehot
        leaf_counts = path_probs.sum(axis=0)[:, None] + 1e-10
        return leaf_weights / leaf_counts

    def _forward_hard(self, X):
        decisions = (X @ self.W_.T - self.b_) > 0
        n_samples = X.shape[0]
        curr_node = np.zeros(n_samples, dtype=int)
        for _ in range(self.max_depth):
            move_right = decisions[np.arange(n_samples), curr_node]
            curr_node = 2 * curr_node + 1 + move_right.astype(int)
        leaf_indices = curr_node - self.n_internal_
        path_matrix = np.zeros((n_samples, self.n_leaves_))
        path_matrix[np.arange(n_samples), leaf_indices] = 1.0
        return path_matrix

    def _score_hard(self, X, y, W, b):
        old_W, old_b = self.W_, self.b_
        self.W_, self.b_ = W, b
        y_pred = self.predict(X)
        acc = np.mean(y_pred == y)
        self.W_, self.b_ = old_W, old_b
        return acc

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, 
                               n_classes=3, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Optimal Classification Tree (Multi-Start Warm Initialized)...")
    
    oct_model = OptimalTreeClassifier(max_depth=5, 
                                      rho=1e-2, 
                                      lr=1e-2, 
                                      n_admm_steps=20, 
                                      n_primal_steps=200, 
                                      warm_start=True,
                                      n_warm_start_trees=50, # Try 20 different initial trees
                                      alpha_start=1.0,
                                      alpha_end=2.0,
                                    #   random_state=42
                                      )
    oct_model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, oct_model.predict(X_train))
    test_acc = accuracy_score(y_test, oct_model.predict(X_test))

    print(f"OCT Train Accuracy: {train_acc:.4f}")
    print(f"OCT Test Accuracy:  {test_acc:.4f}")

    cart = DecisionTreeClassifier(max_depth=5, 
                                #   random_state=42
                                  )
    cart.fit(X_train, y_train)
    cart_train_acc = accuracy_score(y_train, cart.predict(X_train))
    cart_test_acc = accuracy_score(y_test, cart.predict(X_test))
    
    print(f"CART Train Accuracy: {cart_train_acc:.4f}")
    print(f"CART Test Accuracy:  {cart_test_acc:.4f}")