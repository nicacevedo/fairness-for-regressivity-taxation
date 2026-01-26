import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FourierFeatures(nn.Module):
    def __init__(self, in_features, mapping_size, scale=10.0, fourier_type="gaussian", sigma=1.25):
        super().__init__()
        self.fourier_type = fourier_type
        self.in_features = in_features
        self.mapping_size = mapping_size

        if fourier_type == "gaussian":
            self.B = nn.Parameter(torch.randn(in_features, mapping_size) * scale, requires_grad=False)
            self.output_dim = 2 * mapping_size
        elif fourier_type == "positional":
            freq_bands = (sigma ** (torch.arange(mapping_size) / mapping_size))
            self.register_buffer("freq_bands", freq_bands)
            self.output_dim = 2 * in_features * mapping_size
        elif fourier_type == "basic":
            self.output_dim = 2 * in_features
        elif fourier_type == "none":
            self.output_dim = in_features
        else:
            raise ValueError(f"Unknown fourier_type: {fourier_type}")

    def forward(self, x):
        if self.fourier_type == "gaussian":
            x_proj = 2 * np.pi * x @ self.B
            return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        if self.fourier_type == "positional":
            x_proj = 2 * np.pi * x.unsqueeze(-1) * self.freq_bands.view(1, 1, -1)
            x_proj_flat = x_proj.view(x.shape[0], -1)
            return torch.cat([torch.sin(x_proj_flat), torch.cos(x_proj_flat)], dim=-1)
        if self.fourier_type == "basic":
            x_proj = 2 * np.pi * x
            return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return x  # "none"


class ResBlock(nn.Module):
    """Simple tabular ResNet-style block (keeps dimension)."""

    def __init__(self, dim, dropout=0.0, normalization="layer_norm"):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()

        if normalization == "layer_norm":
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
        elif normalization == "none":
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        else:
            raise ValueError("normalization must be 'layer_norm' or 'none'")

    def forward(self, x):
        h = self.fc1(x)
        h = self.norm1(h)
        h = self.act(h)
        h = self.drop(h)

        h = self.fc2(h)
        h = self.norm2(h)
        h = self.drop(h)

        return self.act(x + h)


class NNWithEmbeddings(nn.Module):
    def __init__(
        self,
        embedding_specs,
        num_numerical_features,
        num_coord_features,
        fourier_type="none",
        fourier_mapping_size=16,
        fourier_sigma=1.25,
        hidden_sizes=(512, 256),
        dropout=0.1,
        normalization="layer_norm",  # prefer layer_norm for tabular stability
        mlp_style="resnet",          # "resnet" (recommended) or "plain"
        output_size=1,
    ):
        super().__init__()

        self.embedding_layers = nn.ModuleList([nn.Embedding(num, dim) for num, dim in embedding_specs])

        if num_coord_features > 0 and fourier_type != "none":
            self.fourier_layer = FourierFeatures(
                in_features=num_coord_features,
                mapping_size=fourier_mapping_size,
                fourier_type=fourier_type,
                sigma=fourier_sigma,
            )
            coord_dim = self.fourier_layer.output_dim
        else:
            self.fourier_layer = None
            coord_dim = num_coord_features

        total_emb_dim = sum(dim for _, dim in embedding_specs)
        in_dim = total_emb_dim + num_numerical_features + coord_dim

        if normalization not in ("layer_norm", "none"):
            raise ValueError("normalization must be 'layer_norm' or 'none'")
        if mlp_style not in ("resnet", "plain"):
            raise ValueError("mlp_style must be 'resnet' or 'plain'")

        self.mlp_style = mlp_style
        self.normalization = normalization

        if mlp_style == "plain":
            # Plain MLP, but NO dropout/norm on the final output layer.
            layers = []
            sizes = list(hidden_sizes) + [output_size]
            for i, out_dim in enumerate(sizes):
                layers.append(nn.Linear(in_dim, out_dim))
                is_last = (i == len(sizes) - 1)
                if not is_last:
                    if normalization == "layer_norm":
                        layers.append(nn.LayerNorm(out_dim))
                    layers.append(nn.ReLU())
                    if dropout and dropout > 0:
                        layers.append(nn.Dropout(p=dropout))
                in_dim = out_dim
            self.net = nn.Sequential(*layers)

        else:
            # ResNet-style MLP:
            # - project to a working dimension (hidden_sizes[0])
            # - apply residual blocks at that dimension
            # - output head (linear), no dropout/norm on output
            if len(hidden_sizes) < 1:
                raise ValueError("hidden_sizes must have at least one element for resnet style")

            dim = int(hidden_sizes[0])
            self.in_proj = nn.Linear(in_dim, dim)
            self.in_norm = nn.LayerNorm(dim) if normalization == "layer_norm" else nn.Identity()
            self.in_act = nn.ReLU()
            self.in_drop = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()

            # number of blocks: use remaining entries as a hint; default to 2 if only one size given
            n_blocks = max(2, len(hidden_sizes) - 1)
            self.blocks = nn.Sequential(*[
                ResBlock(dim=dim, dropout=dropout, normalization=normalization)
                for _ in range(n_blocks)
            ])

            self.out = nn.Linear(dim, output_size)

    def forward(self, x_cat, x_num, x_coord):
        if len(self.embedding_layers) > 0:
            embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embedding_layers)]
            x_cat_emb = torch.cat(embs, dim=1)
        else:
            x_cat_emb = torch.empty((x_num.shape[0], 0), device=x_num.device)

        if self.fourier_layer is not None:
            x_coord = self.fourier_layer(x_coord)

        x = torch.cat([x_cat_emb, x_num, x_coord], dim=1)

        if self.mlp_style == "plain":
            return self.net(x)

        x = self.in_proj(x)
        x = self.in_norm(x)
        x = self.in_act(x)
        x = self.in_drop(x)
        x = self.blocks(x)
        return self.out(x)  # final output: no dropout / norm

# Updated with reproducibility 
# class FeedForwardNNRegressorWithEmbeddings6:
#     """
#     Tabular NN for log-price regression with optional covariance regularization:
#       - mode="diff": penalize Cov(yhat - y, y)
#       - mode="div" : penalize Cov(yhat / y, y)   (safe division)

#     Objective (per-sample ridge-like scaling):
#         0.5 * mean((yhat - y)^2) + 0.5*alpha*||theta||^2 + 0.5*rho*c(theta)^2

#     Architecture improvements applied:
#       - Prefer LayerNorm (or none) for tabular stability; no BatchNorm.
#       - ResNet-style MLP blocks (default) for stronger tabular baselines.
#       - Fourier features preserved for coordinate inputs.
#       - Fairness penalty applied ONLY to final log prediction yhat.
#     """

#     def __init__(
#         self,
#         categorical_features,
#         coord_features=(),
#         fourier_type="none",
#         fourier_mapping_size=16,
#         fourier_sigma=1.25,
#         hidden_sizes=(256, 256),         # in resnet mode, first entry is working dim
#         dropout=0.1,
#         normalization="layer_norm",       # 'layer_norm' or 'none'
#         mlp_style="resnet",               # 'resnet' (recommended) or 'plain'
#         batch_size=256,
#         learning_rate=1e-3,
#         num_epochs=50,
#         patience=10,
#         validation_split=0.1,
#         use_scaler=True,
#         loss="mse",           # "mse" or "huber"
#         huber_delta=1.0,
#         alpha=0.0,            # explicit L2 on NN params
#         rho=0.0,              # covariance penalty weight
#         mode="diff",          # "diff" or "div"
#         eps_y=1e-6,
#         random_state=0,
#     ):
#         self.original_categorical_features = list(categorical_features)
#         self.original_coord_features = list(coord_features)

#         self.fourier_type = fourier_type
#         self.fourier_mapping_size = fourier_mapping_size
#         self.fourier_sigma = fourier_sigma
#         self.hidden_sizes = tuple(hidden_sizes)
#         self.dropout = float(dropout)
#         self.normalization = normalization
#         self.mlp_style = mlp_style

#         self.batch_size = int(batch_size)
#         self.learning_rate = float(learning_rate)
#         self.num_epochs = int(num_epochs)
#         self.patience = int(patience)
#         self.validation_split = float(validation_split)
#         self.use_scaler = bool(use_scaler)

#         self.loss = loss
#         self.huber_delta = float(huber_delta)

#         self.alpha = float(alpha)
#         self.rho = float(rho)
#         self.mode = mode
#         self.eps_y = float(eps_y)

#         self.random_state = int(random_state)

#         # fitted artifacts
#         self.model = None
#         self.scaler = None
#         self.category_mappings = {}
#         self.embedding_specs = []
#         self.categorical_features = []
#         self.coord_features = []
#         self.numerical_features = []

#         # fairness constants (train-set)
#         self._y_mean = None
#         self._var_y = None

#     def _split_if_needed(self, X, y, X_val, y_val):
#         if X_val is not None and y_val is not None:
#             return X, y, X_val, y_val
#         X_tr, X_va, y_tr, y_va = train_test_split(
#             X, y, test_size=self.validation_split, random_state=self.random_state
#         )
#         return X_tr, y_tr, X_va, y_va

#     def _build_mappings(self, X_train):
#         self.category_mappings = {}
#         self.embedding_specs = []
#         for col in self.categorical_features:
#             cats = pd.Series(X_train[col].astype("object")).unique()
#             self.category_mappings[col] = {cat: i + 1 for i, cat in enumerate(cats)}
#             self.category_mappings[col]["__UNKNOWN__"] = 0
#             n = len(cats) + 1
#             dim = min(50, (n + 1) // 2)
#             self.embedding_specs.append((n, dim))

#     def _tensorize(self, X_df, X_num_scaled, y_series=None):
#         # categorical
#         if len(self.categorical_features) > 0:
#             cat_cols = []
#             for col in self.categorical_features:
#                 idx = X_df[col].map(self.category_mappings[col]).fillna(0).astype(int).values
#                 cat_cols.append(torch.tensor(idx, dtype=torch.long))
#             X_cat = torch.stack(cat_cols, dim=1)
#         else:
#             X_cat = torch.empty((len(X_df), 0), dtype=torch.long)

#         # numerical
#         X_num = torch.tensor(X_num_scaled, dtype=torch.float32)

#         # coords
#         if len(self.coord_features) > 0:
#             X_coord = torch.tensor(X_df[self.coord_features].astype(float).values, dtype=torch.float32)
#         else:
#             X_coord = torch.empty((len(X_df), 0), dtype=torch.float32)

#         if y_series is None:
#             return X_cat, X_num, X_coord, None

#         y_t = torch.tensor(np.asarray(y_series).reshape(-1, 1), dtype=torch.float32)
#         return X_cat, X_num, X_coord, y_t

#     def _base_loss(self, yhat, y):
#         if self.loss == "mse":
#             return 0.5 * torch.mean((yhat - y) ** 2)
#         if self.loss == "huber":
#             return torch.mean(nn.functional.huber_loss(yhat, y, delta=self.huber_delta, reduction="none"))
#         raise ValueError("loss must be 'mse' or 'huber'")

#     def _l2_penalty(self):
#         if self.alpha <= 0:
#             return torch.tensor(0.0, device=device)
#         s = torch.tensor(0.0, device=device)
#         for p in self.model.parameters():
#             if not p.requires_grad:
#                 continue
#             # exclude biases / 1D params (common ridge convention)
#             if p.ndim == 1:
#                 continue
#             s = s + torch.sum(p ** 2)
#         return 0.5 * self.alpha * s

#     def _cov_penalty(self, yhat, y):
#         # Applied ONLY on final log prediction yhat (as desired).
#         if self.rho <= 0:
#             return torch.tensor(0.0, device=yhat.device)

#         y_mean = self._y_mean
#         var_y = self._var_y
#         y_c = y - y_mean  # training-set mean anchor

#         if self.mode == "diff":
#             m = torch.mean(yhat * y_c)
#             c = m - var_y
#             return 0.5 * self.rho * (c ** 2)

#         if self.mode == "div":
#             y_safe = torch.clamp(y, min=self.eps_y)
#             w = y_c / y_safe
#             c = torch.mean(yhat * w)
#             return 0.5 * self.rho * (c ** 2)

#         raise ValueError("mode must be 'diff' or 'div'")

#     def fit(self, X, y, X_val=None, y_val=None):
#         X = X.copy()
#         y = pd.Series(y).copy()

#         print(device)

#         # feature lists
#         self.categorical_features = [c for c in self.original_categorical_features if c in X.columns]
#         self.coord_features = [c for c in self.original_coord_features if c in X.columns]
#         self.numerical_features = [
#             c for c in X.columns if c not in self.categorical_features and c not in self.coord_features
#         ]

#         X_train, y_train, X_val, y_val = self._split_if_needed(X, y, X_val, y_val)

#         # scaler
#         if self.use_scaler and len(self.numerical_features) > 0:
#             self.scaler = StandardScaler()
#             X_train_num = self.scaler.fit_transform(X_train[self.numerical_features])
#             X_val_num = self.scaler.transform(X_val[self.numerical_features])
#         else:
#             self.scaler = None
#             X_train_num = (
#                 X_train[self.numerical_features].values.astype(np.float32)
#                 if len(self.numerical_features) > 0 else np.zeros((len(X_train), 0), dtype=np.float32)
#             )
#             X_val_num = (
#                 X_val[self.numerical_features].values.astype(np.float32)
#                 if len(self.numerical_features) > 0 else np.zeros((len(X_val), 0), dtype=np.float32)
#             )

#         # mappings
#         self._build_mappings(X_train)

#         # tensors
#         X_cat_tr, X_num_tr, X_coord_tr, y_tr = self._tensorize(X_train, X_train_num, y_train)
#         X_cat_va, X_num_va, X_coord_va, y_va = self._tensorize(X_val, X_val_num, y_val)

#         # fairness constants from TRAIN y (stable anchors)
#         y_tr_dev = y_tr.to(device)
#         self._y_mean = torch.mean(y_tr_dev)
#         self._var_y = torch.mean((y_tr_dev - self._y_mean) ** 2)

#         train_ds = TensorDataset(X_cat_tr, X_num_tr, X_coord_tr, y_tr)
#         val_ds = TensorDataset(X_cat_va, X_num_va, X_coord_va, y_va)
#         train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
#         val_loader = DataLoader(val_ds, batch_size=self.batch_size * 2, shuffle=False)

#         # model (ResNet-style default; LayerNorm or none; Fourier preserved)
#         self.model = NNWithEmbeddings(
#             embedding_specs=self.embedding_specs,
#             num_numerical_features=X_num_tr.shape[1],
#             num_coord_features=X_coord_tr.shape[1],
#             fourier_type=self.fourier_type,
#             fourier_mapping_size=self.fourier_mapping_size,
#             fourier_sigma=self.fourier_sigma,
#             hidden_sizes=self.hidden_sizes,
#             dropout=self.dropout,
#             normalization=self.normalization,
#             mlp_style=self.mlp_style,
#             output_size=1,
#         ).to(device)

#         optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

#         best_val = float("inf")
#         best_state = None
#         patience = 0

#         for _epoch in range(self.num_epochs):
#             self.model.train()
#             for xb_cat, xb_num, xb_coord, yb in train_loader:
#                 xb_cat = xb_cat.to(device)
#                 xb_num = xb_num.to(device)
#                 xb_coord = xb_coord.to(device)
#                 yb = yb.to(device)

#                 yhat = self.model(xb_cat, xb_num, xb_coord)

#                 base = self._base_loss(yhat, yb)
#                 reg = self._l2_penalty()
#                 fair = self._cov_penalty(yhat, yb)

#                 loss = base + reg + fair

#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#             # validation on FULL objective
#             self.model.eval()
#             val_loss = 0.0
#             n_batches = 0
#             with torch.no_grad():
#                 for xb_cat, xb_num, xb_coord, yb in val_loader:
#                     xb_cat = xb_cat.to(device)
#                     xb_num = xb_num.to(device)
#                     xb_coord = xb_coord.to(device)
#                     yb = yb.to(device)

#                     yhat = self.model(xb_cat, xb_num, xb_coord)
#                     base = self._base_loss(yhat, yb)
#                     reg = self._l2_penalty()
#                     fair = self._cov_penalty(yhat, yb)

#                     val_loss += (base + reg + fair).item()
#                     n_batches += 1

#             val_loss /= max(n_batches, 1)

#             if val_loss < best_val:
#                 best_val = val_loss
#                 best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
#                 patience = 0
#             else:
#                 patience += 1
#                 if patience >= self.patience:
#                     break

#         if best_state is not None:
#             self.model.load_state_dict(best_state)

#         return self

#     def predict(self, X):
#         if self.model is None:
#             raise RuntimeError("Call fit() before predict().")

#         X = X.copy()

#         # numerical scaling
#         if self.scaler is not None and len(self.numerical_features) > 0:
#             X_num = self.scaler.transform(X[self.numerical_features])
#         else:
#             X_num = (
#                 X[self.numerical_features].values.astype(np.float32)
#                 if len(self.numerical_features) > 0 else np.zeros((len(X), 0), dtype=np.float32)
#             )

#         X_cat, X_num_t, X_coord_t, _ = self._tensorize(X, X_num, None)

#         ds = TensorDataset(X_cat, X_num_t, X_coord_t)
#         loader = DataLoader(ds, batch_size=self.batch_size * 2, shuffle=False)

#         self.model.eval()
#         preds = []
#         with torch.no_grad():
#             for xb_cat, xb_num, xb_coord in loader:
#                 xb_cat = xb_cat.to(device)
#                 xb_num = xb_num.to(device)
#                 xb_coord = xb_coord.to(device)
#                 yhat = self.model(xb_cat, xb_num, xb_coord)
#                 preds.append(yhat.detach().cpu().numpy())

#         return np.concatenate(preds, axis=0).reshape(-1)

# Add these imports at top of your file (minimal + necessary):
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FeedForwardNNRegressorWithEmbeddings6:
    """
    Same as before, but now includes strict reproducibility controls.

    IMPORTANT (GPU note):
      For *true* determinism on CUDA, PyTorch may require CUBLAS_WORKSPACE_CONFIG to be set
      BEFORE the first CUDA context is created. We set it here if missing, but if CUDA was
      already initialized earlier in your process, you should also set it in your shell:
          export CUBLAS_WORKSPACE_CONFIG=:4096:8
      and start a fresh Python process.
    """

    def __init__(
        self,
        categorical_features,
        coord_features=(),
        fourier_type="none",
        fourier_mapping_size=16,
        fourier_sigma=1.25,
        hidden_sizes=(256, 256),
        dropout=0.1,
        normalization="layer_norm",
        mlp_style="resnet",
        batch_size=256,
        learning_rate=1e-3,
        num_epochs=50,
        patience=10,
        validation_split=0.1,
        use_scaler=True,
        loss="mse",
        huber_delta=1.0,
        alpha=0.0,
        rho=0.0,
        mode="diff",
        eps_y=1e-6,
        random_state=0,
        verbose=True,
        log_every=1,
        # --- NEW reproducibility knobs ---
        deterministic=True,      # force deterministic algorithms where possible
        num_workers=0,           # keep 0 for strict determinism; >0 needs worker seeding
        pin_memory=False,        # no effect on determinism; keep simple
        cpu_threads=1,           # set torch CPU threads; helps determinism on some BLAS setups
    ):
        self.original_categorical_features = list(categorical_features)
        self.original_coord_features = list(coord_features)

        self.fourier_type = fourier_type
        self.fourier_mapping_size = fourier_mapping_size
        self.fourier_sigma = fourier_sigma
        self.hidden_sizes = tuple(hidden_sizes)
        self.dropout = float(dropout)
        self.normalization = normalization
        self.mlp_style = mlp_style

        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.num_epochs = int(num_epochs)
        self.patience = int(patience)
        self.validation_split = float(validation_split)
        self.use_scaler = bool(use_scaler)

        self.loss = loss
        self.huber_delta = float(huber_delta)

        self.alpha = float(alpha)
        self.rho = float(rho)
        self.mode = mode
        self.eps_y = float(eps_y)

        self.random_state = int(random_state)
        self.verbose = bool(verbose)
        self.log_every = int(log_every)

        # reproducibility
        self.deterministic = bool(deterministic)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.cpu_threads = int(cpu_threads)

        # fitted artifacts
        self.model = None
        self.scaler = None
        self.category_mappings = {}
        self.embedding_specs = []
        self.categorical_features = []
        self.coord_features = []
        self.numerical_features = []

        # fairness constants (train-set)
        self._y_mean = None
        self._var_y = None

    # -------------------- NEW: reproducibility helpers --------------------
    def _seed_everything(self):
        """Seed Python/NumPy/PyTorch (+ CUDA) deterministically."""
        seed = int(self.random_state)

        # Python / NumPy
        random.seed(seed)
        np.random.seed(seed)

        # PyTorch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Reduce nondeterminism from parallel CPU reductions
        if self.cpu_threads is not None and self.cpu_threads > 0:
            torch.set_num_threads(self.cpu_threads)

        if self.deterministic:
            # cuDNN determinism
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            # Disable TF32 (can change numeric results across runs/hardware)
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False

            # Enforce deterministic algorithms (will error if something nondeterministic is used)
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                # Older torch versions may not support this; ignore.
                pass

            # Ensure CuBLAS deterministic workspace (best effort)
            # NOTE: for strictest behavior, set this BEFORE starting Python.
            if torch.cuda.is_available() and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    @staticmethod
    def _seed_worker(worker_id):
        """Deterministic seeding for DataLoader workers (only used if num_workers>0)."""
        # torch.initial_seed() is already set from the DataLoader generator
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # -------------------- unchanged helpers (but with one deterministic tweak) --------------------
    def _split_if_needed(self, X, y, X_val, y_val):
        if X_val is not None and y_val is not None:
            return X, y, X_val, y_val
        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y, test_size=self.validation_split, random_state=self.random_state, shuffle=True
        )
        return X_tr, y_tr, X_va, y_va

    def _build_mappings(self, X_train):
        """Deterministic category ordering: sort unique categories."""
        self.category_mappings = {}
        self.embedding_specs = []
        for col in self.categorical_features:
            # Force deterministic mapping independent of row order
            cats = pd.Series(X_train[col].astype("object")).dropna().unique()
            cats = sorted(list(cats))
            self.category_mappings[col] = {cat: i + 1 for i, cat in enumerate(cats)}
            self.category_mappings[col]["__UNKNOWN__"] = 0
            n = len(cats) + 1
            dim = min(50, (n + 1) // 2)
            self.embedding_specs.append((n, dim))

    def _tensorize(self, X_df, X_num_scaled, y_series=None):
        if len(self.categorical_features) > 0:
            cat_cols = []
            for col in self.categorical_features:
                idx = X_df[col].map(self.category_mappings[col]).fillna(0).astype(int).values
                cat_cols.append(torch.tensor(idx, dtype=torch.long))
            X_cat = torch.stack(cat_cols, dim=1)
        else:
            X_cat = torch.empty((len(X_df), 0), dtype=torch.long)

        X_num = torch.tensor(X_num_scaled, dtype=torch.float32)

        if len(self.coord_features) > 0:
            X_coord = torch.tensor(X_df[self.coord_features].astype(float).values, dtype=torch.float32)
        else:
            X_coord = torch.empty((len(X_df), 0), dtype=torch.float32)

        if y_series is None:
            return X_cat, X_num, X_coord, None

        y_t = torch.tensor(np.asarray(y_series).reshape(-1, 1), dtype=torch.float32)
        return X_cat, X_num, X_coord, y_t

    def _base_loss(self, yhat, y):
        if self.loss == "mse":
            return 0.5 * torch.mean((yhat - y) ** 2)
        if self.loss == "huber":
            return torch.mean(nn.functional.huber_loss(yhat, y, delta=self.huber_delta, reduction="none"))
        raise ValueError("loss must be 'mse' or 'huber'")

    def _l2_penalty(self):
        if self.alpha <= 0:
            return torch.tensor(0.0, device=device)
        s = torch.tensor(0.0, device=device)
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1:  # exclude biases / LN scales
                continue
            s = s + torch.sum(p ** 2)
        return 0.5 * self.alpha * s

    def _cov_penalty(self, yhat, y):
        if self.rho <= 0:
            return torch.tensor(0.0, device=yhat.device)

        y_mean = self._y_mean
        var_y = self._var_y
        y_c = y - y_mean

        if self.mode == "diff":
            m = torch.mean(yhat * y_c)
            c = m - var_y
            return 0.5 * self.rho * (c ** 2)

        if self.mode == "div":
            y_safe = torch.clamp(y, min=self.eps_y)
            w = y_c / y_safe
            c = torch.mean(yhat * w)
            return 0.5 * self.rho * (c ** 2)

        raise ValueError("mode must be 'diff' or 'div'")

    # -------------------- main fit/predict --------------------
    def fit(self, X, y, X_val=None, y_val=None):
        # NEW: seed EVERYTHING at the start of fit()
        self._seed_everything()

        X = X.copy()
        y = pd.Series(y).copy()

        self.categorical_features = [c for c in self.original_categorical_features if c in X.columns]
        self.coord_features = [c for c in self.original_coord_features if c in X.columns]
        self.numerical_features = [c for c in X.columns if c not in self.categorical_features and c not in self.coord_features]

        X_train, y_train, X_val, y_val = self._split_if_needed(X, y, X_val, y_val)

        if self.use_scaler and len(self.numerical_features) > 0:
            self.scaler = StandardScaler()
            X_train_num = self.scaler.fit_transform(X_train[self.numerical_features])
            X_val_num = self.scaler.transform(X_val[self.numerical_features])
        else:
            self.scaler = None
            X_train_num = X_train[self.numerical_features].values.astype(np.float32) if len(self.numerical_features) > 0 else np.zeros((len(X_train), 0), dtype=np.float32)
            X_val_num = X_val[self.numerical_features].values.astype(np.float32) if len(self.numerical_features) > 0 else np.zeros((len(X_val), 0), dtype=np.float32)

        # deterministic categorical mappings
        self._build_mappings(X_train)

        X_cat_tr, X_num_tr, X_coord_tr, y_tr = self._tensorize(X_train, X_train_num, y_train)
        X_cat_va, X_num_va, X_coord_va, y_va = self._tensorize(X_val, X_val_num, y_val)

        y_tr_dev = y_tr.to(device)
        self._y_mean = torch.mean(y_tr_dev)
        self._var_y = torch.mean((y_tr_dev - self._y_mean) ** 2)

        train_ds = TensorDataset(X_cat_tr, X_num_tr, X_coord_tr, y_tr)
        val_ds = TensorDataset(X_cat_va, X_num_va, X_coord_va, y_va)

        # NEW: deterministic DataLoader shuffling
        g = torch.Generator()
        g.manual_seed(int(self.random_state))

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            generator=g,
            worker_init_fn=self._seed_worker if self.num_workers > 0 else None,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            generator=g,
            worker_init_fn=self._seed_worker if self.num_workers > 0 else None,
        )

        # Build model AFTER seeding (critical!)
        self.model = NNWithEmbeddings(
            embedding_specs=self.embedding_specs,
            num_numerical_features=X_num_tr.shape[1],
            num_coord_features=X_coord_tr.shape[1],
            fourier_type=self.fourier_type,
            fourier_mapping_size=self.fourier_mapping_size,
            fourier_sigma=self.fourier_sigma,
            hidden_sizes=self.hidden_sizes,
            dropout=self.dropout,
            normalization=self.normalization,
            mlp_style=self.mlp_style,
            output_size=1,
        ).to(device)

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        best_val = float("inf")
        best_state = None
        patience = 0

        if self.verbose:
            print(
                "epoch | train_total train_mse train_fair c_train corr(r,y)_tr | "
                "val_total val_mse val_fair c_val corr(r,y)_va"
            )

        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            tr_total = tr_base = tr_fair = 0.0
            tr_n = 0

            tr_sum_y = tr_sum_y2 = 0.0
            tr_sum_r = tr_sum_r2 = 0.0
            tr_sum_ry = 0.0
            tr_sum_yhat_yc = 0.0
            tr_sum_yhat_w = 0.0

            y_mean = float(self._y_mean.detach().cpu().item())
            var_y = float(self._var_y.detach().cpu().item())

            for xb_cat, xb_num, xb_coord, yb in train_loader:
                xb_cat = xb_cat.to(device)
                xb_num = xb_num.to(device)
                xb_coord = xb_coord.to(device)
                yb = yb.to(device)

                yhat = self.model(xb_cat, xb_num, xb_coord)

                base = self._base_loss(yhat, yb)
                reg = self._l2_penalty()
                fair = self._cov_penalty(yhat, yb)
                loss = base + reg + fair

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                bsz = int(yb.shape[0])
                tr_total += loss.item() * bsz
                tr_base += base.item() * bsz
                tr_fair += fair.item() * bsz
                tr_n += bsz

                with torch.no_grad():
                    yb_cpu = yb.view(-1).detach().cpu().numpy()
                    yhat_cpu = yhat.view(-1).detach().cpu().numpy()

                    if self.mode == "diff":
                        r = (yhat_cpu - yb_cpu)
                        y_c = (yb_cpu - y_mean)
                        tr_sum_yhat_yc += float(np.sum(yhat_cpu * y_c))
                    else:
                        y_safe = np.maximum(yb_cpu, self.eps_y)
                        r = (yhat_cpu / y_safe)
                        w = (yb_cpu - y_mean) / y_safe
                        tr_sum_yhat_w += float(np.sum(yhat_cpu * w))

                    tr_sum_y += float(np.sum(yb_cpu))
                    tr_sum_y2 += float(np.sum(yb_cpu * yb_cpu))
                    tr_sum_r += float(np.sum(r))
                    tr_sum_r2 += float(np.sum(r * r))
                    tr_sum_ry += float(np.sum(r * yb_cpu))

            tr_total /= max(tr_n, 1)
            tr_base /= max(tr_n, 1)
            tr_fair /= max(tr_n, 1)
            tr_mse = 2.0 * tr_base

            def _corr(sum_a, sum_a2, sum_b, sum_b2, sum_ab, n_):
                if n_ <= 1:
                    return 0.0
                mean_a = sum_a / n_
                mean_b = sum_b / n_
                var_a = max(sum_a2 / n_ - mean_a * mean_a, 0.0)
                var_b = max(sum_b2 / n_ - mean_b * mean_b, 0.0)
                if var_a <= 0.0 or var_b <= 0.0:
                    return 0.0
                cov_ab = sum_ab / n_ - mean_a * mean_b
                return cov_ab / (var_a ** 0.5 * var_b ** 0.5)

            tr_corr = _corr(tr_sum_r, tr_sum_r2, tr_sum_y, tr_sum_y2, tr_sum_ry, tr_n)

            if self.mode == "diff":
                c_tr = (tr_sum_yhat_yc / max(tr_n, 1)) - var_y
            else:
                c_tr = (tr_sum_yhat_w / max(tr_n, 1))

            # VALIDATION
            self.model.eval()
            va_total = va_base = va_fair = 0.0
            va_n = 0

            va_sum_y = va_sum_y2 = 0.0
            va_sum_r = va_sum_r2 = 0.0
            va_sum_ry = 0.0
            va_sum_yhat_yc = 0.0
            va_sum_yhat_w = 0.0

            with torch.no_grad():
                for xb_cat, xb_num, xb_coord, yb in val_loader:
                    xb_cat = xb_cat.to(device)
                    xb_num = xb_num.to(device)
                    xb_coord = xb_coord.to(device)
                    yb = yb.to(device)

                    yhat = self.model(xb_cat, xb_num, xb_coord)
                    base = self._base_loss(yhat, yb)
                    reg = self._l2_penalty()
                    fair = self._cov_penalty(yhat, yb)
                    loss = base + reg + fair

                    bsz = int(yb.shape[0])
                    va_total += loss.item() * bsz
                    va_base += base.item() * bsz
                    va_fair += fair.item() * bsz
                    va_n += bsz

                    yb_cpu = yb.view(-1).detach().cpu().numpy()
                    yhat_cpu = yhat.view(-1).detach().cpu().numpy()

                    if self.mode == "diff":
                        r = (yhat_cpu - yb_cpu)
                        y_c = (yb_cpu - y_mean)
                        va_sum_yhat_yc += float(np.sum(yhat_cpu * y_c))
                    else:
                        y_safe = np.maximum(yb_cpu, self.eps_y)
                        r = (yhat_cpu / y_safe)
                        w = (yb_cpu - y_mean) / y_safe
                        va_sum_yhat_w += float(np.sum(yhat_cpu * w))

                    va_sum_y += float(np.sum(yb_cpu))
                    va_sum_y2 += float(np.sum(yb_cpu * yb_cpu))
                    va_sum_r += float(np.sum(r))
                    va_sum_r2 += float(np.sum(r * r))
                    va_sum_ry += float(np.sum(r * yb_cpu))

            va_total /= max(va_n, 1)
            va_base /= max(va_n, 1)
            va_fair /= max(va_n, 1)
            va_mse = 2.0 * va_base
            va_corr = _corr(va_sum_r, va_sum_r2, va_sum_y, va_sum_y2, va_sum_ry, va_n)

            if self.mode == "diff":
                c_va = (va_sum_yhat_yc / max(va_n, 1)) - var_y
            else:
                c_va = (va_sum_yhat_w / max(va_n, 1))

            if self.verbose and (self.log_every > 0) and (epoch % self.log_every == 0):
                print(
                    f"{epoch:>5d} | "
                    f"{tr_total:10.6f} {tr_mse:9.6f} {tr_fair:9.6f} {c_tr: .3e} {tr_corr: .4f} | "
                    f"{va_total:9.6f} {va_mse:8.6f} {va_fair:8.6f} {c_va: .3e} {va_corr: .4f}"
                )

            if va_total < best_val:
                best_val = va_total
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("Call fit() before predict().")

        X = X.copy()

        if self.scaler is not None and len(self.numerical_features) > 0:
            X_num = self.scaler.transform(X[self.numerical_features])
        else:
            X_num = X[self.numerical_features].values.astype(np.float32) if len(self.numerical_features) > 0 else np.zeros((len(X), 0), dtype=np.float32)

        X_cat, X_num_t, X_coord_t, _ = self._tensorize(X, X_num, None)

        ds = TensorDataset(X_cat, X_num_t, X_coord_t)

        # Deterministic order: shuffle=False, generator fixed anyway
        g = torch.Generator()
        g.manual_seed(int(self.random_state))

        loader = DataLoader(
            ds,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            generator=g,
            worker_init_fn=self._seed_worker if self.num_workers > 0 else None,
        )

        self.model.eval()
        preds = []
        with torch.no_grad():
            for xb_cat, xb_num, xb_coord in loader:
                xb_cat = xb_cat.to(device)
                xb_num = xb_num.to(device)
                xb_coord = xb_coord.to(device)
                yhat = self.model(xb_cat, xb_num, xb_coord)
                preds.append(yhat.detach().cpu().numpy())

        return np.concatenate(preds, axis=0).reshape(-1)

    def __str__(self):
        return f"FeedForwardNNRegressorWithEmbeddings(rho={self.rho}, mode={self.mode})"