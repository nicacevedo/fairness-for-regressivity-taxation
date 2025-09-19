import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from collections import OrderedDict
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.cluster import KMeans

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================================
# 1) Fourier Features (unchanged API)
# ======================================================================
class FourierFeatures(nn.Module):
    def __init__(self, in_features, mapping_size, scale=10.0, fourier_type='gaussian', sigma=1.25):
        super().__init__()
        self.in_features = in_features
        self.mapping_size = mapping_size
        self.fourier_type = fourier_type

        if fourier_type == 'gaussian':
            self.B = nn.Parameter(torch.randn(in_features, mapping_size) * scale, requires_grad=False)
            self.output_dim = 2 * mapping_size
        elif fourier_type == 'positional':
            if sigma is None:
                raise ValueError("sigma must be provided for positional Fourier features.")
            freq_bands = (sigma ** (torch.arange(mapping_size) / mapping_size))
            self.register_buffer('freq_bands', freq_bands)
            self.output_dim = 2 * in_features * mapping_size
        elif fourier_type == 'basic':
            self.output_dim = 2 * in_features
        elif fourier_type == 'none':
            self.output_dim = in_features
        else:
            raise ValueError(f"Unknown fourier_type: {fourier_type}")

    def forward(self, x):
        if self.fourier_type == 'gaussian':
            x_proj = 2 * np.pi * x @ self.B
            return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        elif self.fourier_type == 'positional':
            x_proj = 2 * np.pi * x.unsqueeze(-1) * self.freq_bands.view(1, 1, -1)
            x_proj_flat = x_proj.view(x.shape[0], -1)
            return torch.cat([torch.sin(x_proj_flat), torch.cos(x_proj_flat)], dim=-1)
        elif self.fourier_type == 'basic':
            x_proj = 2 * np.pi * x
            return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        elif self.fourier_type == 'none':
            return x

# ======================================================================
# 2) Core NN w/ Embeddings
#    - No norm/dropout on the output layer
#    - Safe when there are zero categorical or coordinate features
# ======================================================================
class NNWithEmbeddings(nn.Module):
    def __init__(self, embedding_specs, num_numerical_features, num_coord_features,
                 fourier_type, fourier_mapping_size, fourier_sigma, layer_sizes,
                 dropout=0.5, normalization_type='none'):
        super().__init__()
        self.embedding_layers = nn.ModuleList([nn.Embedding(num, dim) for num, dim in embedding_specs])

        # Fourier for coords
        if fourier_type != 'none' and num_coord_features > 0:
            self.fourier_layer = FourierFeatures(
                in_features=num_coord_features,
                mapping_size=fourier_mapping_size,
                fourier_type=fourier_type,
                sigma=fourier_sigma
            )
            coord_dim = self.fourier_layer.output_dim
        else:
            self.fourier_layer = None
            coord_dim = num_coord_features

        total_embedding_dim = sum(dim for _, dim in embedding_specs)
        input_size = total_embedding_dim + num_numerical_features + coord_dim

        # Build MLP: apply norm/activation/dropout only on hidden layers
        layers = []
        for i, size in enumerate(layer_sizes):
            layers.append((f"linear_{i}", nn.Linear(input_size, size)))
            is_last = (i == len(layer_sizes) - 1)
            if not is_last:
                if normalization_type == 'batch_norm':
                    layers.append((f"norm_{i}", nn.BatchNorm1d(size)))
                elif normalization_type == 'layer_norm':
                    layers.append((f"norm_{i}", nn.LayerNorm(size)))
                layers.append((f"relu_{i}", nn.ReLU()))
                if dropout and dropout > 0:
                    layers.append((f"dropout_{i}", nn.Dropout(p=dropout)))
            input_size = size
        self.layers = nn.Sequential(OrderedDict(layers))

    def forward(self, x_cat, x_num, x_coord):
        # Categorical embeddings (safe if 0 cat cols)
        if len(self.embedding_layers) > 0:
            emb_list = [emb(x_cat[:, i]) for i, emb in enumerate(self.embedding_layers)]
            x_cat_emb = torch.cat(emb_list, dim=1)
        else:
            x_cat_emb = x_cat.new_zeros((x_cat.size(0), 0), dtype=torch.float32)
            x_cat_emb = x_cat_emb.to(x_num.dtype)

        # Coordinates
        if self.fourier_layer is not None:
            x_coord_processed = self.fourier_layer(x_coord)
        else:
            x_coord_processed = x_coord

        x = torch.cat([x_cat_emb, x_num, x_coord_processed], dim=1)
        return self.layers(x)  # logits

# ======================================================================
# 3) Discretized Classifier with Geometry-Aware Losses
# ======================================================================
class DiscretizedNNClassifier2:
    """
    Discretize y into bins, train a classifier, and optionally recover a
    regression-like geometry with several loss options.

    loss_mode ∈ {
        'ce',            # standard cross-entropy
        'ev_mse',        # expected value (p@bin_values) vs y  with MSE
        'ev_mae',        # ... with MAE
        'ev_huber',      # ... with Huber (delta=huber_delta)
        'emd',           # Earth Mover's Distance (Wasserstein-1 on ordered bins)
        'prob_mse',      # MSE between p and (one-hot or gaussian-smoothed) target
        'smooth_ce'      # cross-entropy with distance-based soft targets
    }
    """
    def __init__(self, categorical_features, coord_features=None,
                 engineer_time_features=False, bin_yrblt=False, cross_township_class=False,
                 fourier_type='none', fourier_mapping_size=16, fourier_sigma=1.25,
                 n_bins=10, min_samples_per_bin=3, binning_method='quantile',
                 batch_size=16, learning_rate=1e-3, num_epochs=10, hidden_sizes=[1024],
                 patience=10, random_state=None, dropout=0.5, l2_lambda=0.0, l1_lambda=0.0,
                 use_scaler=False, normalization_type='none',
                 # NEW: loss controls
                 loss_mode='ce', huber_delta=1.0, smoothing_sigma=0.0, ce_label_smoothing=0.0,
                 use_class_weights=False,
                 # inference
                 default_predict_mode='expected'  # 'expected' or 'argmax'
                 ):
        self.original_categorical_features = (categorical_features or [])[:]
        self.original_coord_features = (coord_features or [])[:]

        # FE flags
        self.engineer_time_features = engineer_time_features
        self.bin_yrblt = bin_yrblt
        self.cross_township_class = cross_township_class

        # Core params
        self.use_scaler = use_scaler
        self.scaler = None
        self.fourier_type = fourier_type
        self.fourier_mapping_size = fourier_mapping_size
        self.fourier_sigma = fourier_sigma
        self.numerical_features = []
        self.n_bins = n_bins
        self.min_samples_per_bin = min_samples_per_bin
        self.binning_method = binning_method
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.l2_lambda = l2_lambda
        self.l1_lambda = l1_lambda
        self.patience = patience
        self.model = None
        self.category_mappings = {}
        self.embedding_specs = []
        self.random_state = random_state
        self.normalization_type = normalization_type
        self.bin_info = {}

        # Loss controls
        self.loss_mode = loss_mode
        self.huber_delta = float(huber_delta)
        self.smoothing_sigma = float(smoothing_sigma)
        self.ce_label_smoothing = float(ce_label_smoothing)
        self.use_class_weights = bool(use_class_weights)

        # Inference behavior
        self.default_predict_mode = default_predict_mode

    # -------------------- Feature engineering --------------------
    def _engineer_features(self, X):
        X_eng = X.copy()
        if self.engineer_time_features:
            cyclical_features = {
                'time_sale_month_of_year': 12,
                'time_sale_day_of_week': 7,
                'time_sale_day_of_year': 365.25,
                'time_sale_day_of_month': 30.44
            }
            for col, period in cyclical_features.items():
                if col in X_eng.columns:
                    X_eng[f'{col}_sin'] = np.sin(2 * np.pi * X_eng[col] / period)
                    X_eng[f'{col}_cos'] = np.cos(2 * np.pi * X_eng[col] / period)
            X_eng = X_eng.drop(columns=list(cyclical_features.keys()) + ['time_sale_day'], errors='ignore')

        if self.bin_yrblt and 'char_yrblt' in X_eng.columns:
            X_eng['yrblt_decade'] = (X_eng['char_yrblt'] // 10 * 10).astype(str)
            X_eng = X_eng.drop(columns=['char_yrblt'], errors='ignore')

        if self.cross_township_class and 'meta_township_code' in X_eng.columns and 'char_class' in X_eng.columns:
            X_eng['township_class_interaction'] = X_eng['meta_township_code'].astype(str) + '_' + X_eng['char_class'].astype(str)
        return X_eng

    # -------------------- Binning helpers --------------------
    @staticmethod
    def _compute_class_weights(y_labels, num_classes):
        counts = np.bincount(y_labels, minlength=num_classes).astype(np.float32)
        counts[counts == 0] = 1.0
        weights = counts.sum() / (counts * num_classes)
        return torch.tensor(weights, dtype=torch.float32)

    @staticmethod
    def _gaussian_targets(indices, num_classes, sigma):
        # Build a gaussian over class indices centered at each target index
        # indices: (B,) int64 tensor; returns (B, C)
        device_local = indices.device
        C = num_classes
        grid = torch.arange(C, device=device_local).view(1, C)
        idx = indices.view(-1, 1)
        dist2 = (grid - idx).float().pow(2)
        t = torch.exp(-dist2 / (2.0 * (sigma ** 2)))
        t = t / (t.sum(dim=1, keepdim=True) + 1e-12)
        return t

    def _build_bin_values(self, y, y_bins, unique_labels):
        # Build per-bin numeric representative aligned with remapped labels order
        bin_values = []
        for old_label in unique_labels:
            vals = y[y_bins == old_label]
            if len(vals) == 0:
                bin_values.append(float('nan'))
            else:
                bin_values.append(float(vals.median()))
        return np.array(bin_values, dtype=np.float32)

    # -------------------- Fit --------------------
    def fit(self, X, y, X_val=None, y_val=None):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)
            random.seed(self.random_state)

        # FE
        X_eng = self._engineer_features(X)

        # Binning
        if self.binning_method == 'quantile':
            y_bins, bin_edges = pd.qcut(y, q=self.n_bins, labels=False, retbins=True, duplicates='drop')
        elif self.binning_method == 'uniform':
            y_bins, bin_edges = pd.cut(y, bins=self.n_bins, labels=False, retbins=True, duplicates='drop')
        elif self.binning_method == 'kmeans':
            km = KMeans(n_clusters=self.n_bins, random_state=self.random_state, n_init='auto')
            y_bins = pd.Series(km.fit_predict(y.values.reshape(-1, 1)), index=y.index)
            bin_edges = km.cluster_centers_.flatten()  # not edges; centers
        else:
            raise ValueError("Invalid binning_method. Choose from 'quantile', 'uniform', 'kmeans'.")

        # Filter small bins
        bin_counts = y_bins.value_counts()
        valid_bins = bin_counts[bin_counts >= self.min_samples_per_bin].index
        if len(valid_bins) < 2:
            raise ValueError("Not enough valid bins after filtering. Try reducing `min_samples_per_bin` or `n_bins`.")

        mask = y_bins.isin(valid_bins)
        X_eng = X_eng[mask].reset_index(drop=True)
        y = y[mask].reset_index(drop=True)
        y_bins = y_bins[mask].reset_index(drop=True)

        unique_labels = sorted(y_bins.unique())
        label_mapping = {old: new for new, old in enumerate(unique_labels)}
        y_labels = y_bins.map(label_mapping).values

        # Store bin info
        self.bin_info['num_bins'] = len(unique_labels)
        # Build robust bin values (medians per valid bin) regardless of method
        bin_values_np = self._build_bin_values(y, y_bins.map(int), unique_labels)
        self.bin_info['bin_values'] = bin_values_np  # length = num_bins

        # For predict(class->value), keep simple representative per class
        if self.binning_method == 'kmeans':
            # align centers to remapped order via medians too (more robust)
            self.bin_info['centers'] = bin_values_np.copy()
        elif self.binning_method in ['quantile', 'uniform']:
            self.bin_info['centers'] = bin_values_np.copy()

        # Validation processing
        if X_val is not None and y_val is not None:
            X_val_eng = self._engineer_features(X_val)
            # Assign val to clusters/bins using same rule
            if self.binning_method in ['quantile', 'uniform']:
                # Use pandas cut with original edges derived from y (for ref only)
                # but we will map to closest center to be robust if edges mismatch
                # Here we simply assign by nearest center
                centers_all = np.array(self.bin_info['centers'])
                y_val_labels_raw = pd.Series(np.argmin(np.abs(y_val.values.reshape(-1,1) - centers_all.reshape(1,-1)), axis=1), index=y_val.index)
            else:  # kmeans
                centers_all = np.array(self.bin_info['centers'])
                y_val_labels_raw = pd.Series(np.argmin(np.abs(y_val.values.reshape(-1,1) - centers_all.reshape(1,-1)), axis=1), index=y_val.index)

            valid_mask_val = y_val_labels_raw.isin(range(self.bin_info['num_bins']))
            X_val_eng = X_val_eng[valid_mask_val].reset_index(drop=True)
            y_val = y_val[valid_mask_val].reset_index(drop=True)
            y_val_labels = y_val_labels_raw[valid_mask_val].astype(int).values
        else:
            X_val_eng = None
            y_val_labels = None

        # Determine feature lists
        current_categorical = self.original_categorical_features[:]
        if self.engineer_time_features:
            cyc_rm = ['time_sale_month_of_year', 'time_sale_day_of_week', 'time_sale_day_of_year', 'time_sale_day_of_month']
            current_categorical = [c for c in current_categorical if c not in cyc_rm]
        if self.bin_yrblt:
            if 'char_yrblt' in current_categorical:
                current_categorical.remove('char_yrblt')
            current_categorical.append('yrblt_decade')
        if self.cross_township_class:
            current_categorical.append('township_class_interaction')

        self.categorical_features = [c for c in current_categorical if c in X_eng.columns]
        self.coord_features = self.original_coord_features[:]
        self.numerical_features = [c for c in X_eng.columns if c not in self.categorical_features + self.coord_features]

        # Split if val not provided
        if X_val_eng is None:
            X_train, X_val_eng, y_train_labels, y_val_labels = train_test_split(
                X_eng, y_labels, test_size=0.2, random_state=self.random_state, stratify=y_labels
            )
            y_train_cont = y.iloc[X_train.index].reset_index(drop=True).values.astype(np.float32)
            y_val_cont = y.iloc[X_val_eng.index].reset_index(drop=True).values.astype(np.float32)
            X_train = X_train.reset_index(drop=True)
            X_val_eng = X_val_eng.reset_index(drop=True)
        else:
            X_train = X_eng
            y_train_labels = y_labels
            y_train_cont = y.values.astype(np.float32)
            y_val_cont = y_val.values.astype(np.float32)

        # Scaling
        if self.use_scaler and len(self.numerical_features) > 0:
            self.scaler = StandardScaler()
            X_train_num_scaled = self.scaler.fit_transform(X_train[self.numerical_features])
            X_val_num_scaled = self.scaler.transform(X_val_eng[self.numerical_features])
        else:
            X_train_num_scaled = X_train[self.numerical_features].values if len(self.numerical_features) > 0 else np.zeros((len(X_train), 0), dtype=np.float32)
            X_val_num_scaled = X_val_eng[self.numerical_features].values if len(self.numerical_features) > 0 else np.zeros((len(X_val_eng), 0), dtype=np.float32)

        # Embedding specs
        self.embedding_specs = []
        self.category_mappings = {}
        for col in self.categorical_features:
            cats = X_train[col].astype(str).unique()
            mapping = {cat: i + 1 for i, cat in enumerate(cats)}
            mapping['__UNKNOWN__'] = 0
            self.category_mappings[col] = mapping
            num_with_unk = len(cats) + 1
            emb_dim = min(50, (num_with_unk + 1) // 2)
            self.embedding_specs.append((num_with_unk, emb_dim))

        # Tensor builders
        def _to_tensors(X_df, X_num_scaled, y_lbls, y_cont):
            if len(self.categorical_features) > 0:
                X_cat_tensors = [torch.tensor(X_df[col].astype(str).map(self.category_mappings[col]).fillna(0).values, dtype=torch.long) for col in self.categorical_features]
                X_cat = torch.stack(X_cat_tensors, dim=1)
            else:
                X_cat = torch.empty(len(X_df), 0, dtype=torch.long)
            X_num = torch.tensor(X_num_scaled, dtype=torch.float32)
            X_coord = torch.tensor(X_df[self.coord_features].astype(float).values if len(self.coord_features) > 0 else np.zeros((len(X_df), 0), dtype=np.float32), dtype=torch.float32)
            y_lbls_t = torch.tensor(y_lbls, dtype=torch.long)
            y_cont_t = torch.tensor(y_cont, dtype=torch.float32)
            return X_cat, X_num, X_coord, y_lbls_t, y_cont_t

        X_cat_tr, X_num_tr, X_coord_tr, y_tr_lbl, y_tr_cont = _to_tensors(X_train, X_train_num_scaled, y_train_labels, y_train_cont)
        X_cat_va, X_num_va, X_coord_va, y_va_lbl, y_va_cont = _to_tensors(X_val_eng, X_val_num_scaled, y_val_labels, y_val_cont)

        train_loader = DataLoader(TensorDataset(X_cat_tr, X_num_tr, X_coord_tr, y_tr_lbl, y_tr_cont), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_cat_va, X_num_va, X_coord_va, y_va_lbl, y_va_cont), batch_size=max(1, self.batch_size * 2), shuffle=False)

        # Model
        num_classes = self.bin_info['num_bins']
        self.model = NNWithEmbeddings(
            embedding_specs=self.embedding_specs,
            num_numerical_features=len(self.numerical_features),
            num_coord_features=len(self.coord_features),
            fourier_type=self.fourier_type,
            fourier_mapping_size=self.fourier_mapping_size,
            fourier_sigma=self.fourier_sigma,
            layer_sizes=self.hidden_sizes + [num_classes],
            dropout=self.dropout,
            normalization_type=self.normalization_type
        ).to(device)

        # Class weights (optional)
        if self.use_class_weights:
            class_weights = self._compute_class_weights(y_train_labels, num_classes).to(device)
        else:
            class_weights = None

        # Base CE criterion (for 'ce')
        ce_kwargs = {}
        if self.ce_label_smoothing > 0:
            ce_kwargs['label_smoothing'] = float(self.ce_label_smoothing)
        if class_weights is not None:
            ce_kwargs['weight'] = class_weights
        ce_criterion = nn.CrossEntropyLoss(**ce_kwargs)

        # Optimizer
        if self.l2_lambda < 1e-8:
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        else:
            optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2_lambda)

        # Precompute bin_values tensor for EV losses
        bin_values = torch.tensor(self.bin_info['bin_values'], dtype=torch.float32, device=device).view(1, -1)  # (1,C)

        def _huber(a, b, delta):
            diff = a - b
            abs_diff = torch.abs(diff)
            quad = torch.minimum(abs_diff, torch.tensor(delta, device=diff.device))
            lin = abs_diff - quad
            return 0.5 * quad * quad + delta * lin

        def _emd_loss(p, y_idx):
            # p: (B,C) softmax
            B, C = p.shape
            onehot = torch.zeros(B, C, device=p.device)
            onehot.scatter_(1, y_idx.view(-1, 1), 1.0)
            cdf_diff = torch.cumsum(p - onehot, dim=1)
            return torch.mean(torch.sum(torch.abs(cdf_diff), dim=1))

        def _soft_target(indices):
            if self.smoothing_sigma > 0:
                return self._gaussian_targets(indices, num_classes, self.smoothing_sigma)
            else:
                onehot = torch.zeros(indices.size(0), num_classes, device=indices.device)
                onehot.scatter_(1, indices.view(-1, 1), 1.0)
                return onehot

        # Training loop
        best_val = float('inf')
        best_state = None
        patience_cnt = 0

        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss_sum = 0.0
            for b_cat, b_num, b_coord, b_y_lbl, b_y_cont in train_loader:
                b_cat, b_num, b_coord = b_cat.to(device), b_num.to(device), b_coord.to(device)
                b_y_lbl = b_y_lbl.to(device)
                b_y_cont = b_y_cont.to(device)

                logits = self.model(b_cat, b_num, b_coord)
                if self.loss_mode == 'ce':
                    loss = ce_criterion(logits, b_y_lbl)
                else:
                    p = torch.softmax(logits, dim=1)
                    if self.loss_mode == 'ev_mse':
                        y_hat = (p * bin_values).sum(dim=1)
                        loss = torch.mean((y_hat - b_y_cont) ** 2)
                    elif self.loss_mode == 'ev_mae':
                        y_hat = (p * bin_values).sum(dim=1)
                        loss = torch.mean(torch.abs(y_hat - b_y_cont))
                    elif self.loss_mode == 'ev_huber':
                        y_hat = (p * bin_values).sum(dim=1)
                        loss = torch.mean(_huber(y_hat, b_y_cont, self.huber_delta))
                    elif self.loss_mode == 'emd':
                        loss = _emd_loss(p, b_y_lbl)
                    elif self.loss_mode == 'prob_mse':
                        t = _soft_target(b_y_lbl)
                        loss = torch.mean((p - t) ** 2)
                    elif self.loss_mode == 'smooth_ce':
                        t = _soft_target(b_y_lbl)
                        logp = torch.log_softmax(logits, dim=1)
                        loss = -torch.mean(torch.sum(t * logp, dim=1))
                    else:
                        raise ValueError(f"Unknown loss_mode: {self.loss_mode}")

                if self.l1_lambda >= 1e-12:
                    l1_pen = sum(param.abs().sum() for param in self.model.parameters())
                    loss = loss + self.l1_lambda * l1_pen

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_sum += float(loss.item())

            # Validation
            self.model.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for b_cat, b_num, b_coord, b_y_lbl, b_y_cont in val_loader:
                    b_cat, b_num, b_coord = b_cat.to(device), b_num.to(device), b_coord.to(device)
                    b_y_lbl = b_y_lbl.to(device)
                    b_y_cont = b_y_cont.to(device)
                    logits = self.model(b_cat, b_num, b_coord)
                    if self.loss_mode == 'ce':
                        vloss = ce_criterion(logits, b_y_lbl)
                    else:
                        p = torch.softmax(logits, dim=1)
                        if self.loss_mode == 'ev_mse':
                            y_hat = (p * bin_values).sum(dim=1)
                            vloss = torch.mean((y_hat - b_y_cont) ** 2)
                        elif self.loss_mode == 'ev_mae':
                            y_hat = (p * bin_values).sum(dim=1)
                            vloss = torch.mean(torch.abs(y_hat - b_y_cont))
                        elif self.loss_mode == 'ev_huber':
                            y_hat = (p * bin_values).sum(dim=1)
                            vloss = torch.mean(_huber(y_hat, b_y_cont, self.huber_delta))
                        elif self.loss_mode == 'emd':
                            vloss = _emd_loss(p, b_y_lbl)
                        elif self.loss_mode == 'prob_mse':
                            t = _soft_target(b_y_lbl)
                            vloss = torch.mean((p - t) ** 2)
                        elif self.loss_mode == 'smooth_ce':
                            t = _soft_target(b_y_lbl)
                            logp = torch.log_softmax(logits, dim=1)
                            vloss = -torch.mean(torch.sum(t * logp, dim=1))
                        else:
                            raise ValueError(f"Unknown loss_mode: {self.loss_mode}")
                    val_loss_sum += float(vloss.item())

            avg_tr = train_loss_sum / max(1, len(train_loader))
            avg_va = val_loss_sum / max(1, len(val_loader))
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.num_epochs}] Train: {avg_tr:.4f} | Val: {avg_va:.4f}")

            if avg_va < best_val - 1e-9:
                best_val = avg_va
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}.")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        # ===== Final validation report (classification + regression) =====
        y_val_pred_bins = self._predict_classes_df(X_val_eng)
        y_val_true_bins = y_val_labels
        print("\n" + "="*50)
        print("Final Classification Report on Validation Set")
        print("="*50)
        print(classification_report(y_val_true_bins, y_val_pred_bins, zero_division=0))
        print("Confusion Matrix:\n", confusion_matrix(y_val_true_bins, y_val_pred_bins))

        y_val_pred_numeric = self._convert_to_numeric(y_val_pred_bins)
        print("\n" + "="*50)
        print("Final Regression Metrics on Validation Set")
        print("="*50)
        print(f"Mean Absolute Error (MAE): {mean_absolute_error(y.iloc[X_val_eng.index], y_val_pred_numeric):.4f}")
        print(f"Root Mean Squared Error (RMSE): {root_mean_squared_error(y.iloc[X_val_eng.index], y_val_pred_numeric):.4f}")
        print(f"R-squared (R2): {r2_score(y.iloc[X_val_eng.index], y_val_pred_numeric):.4f}")
        print("="*50)

    # -------------------- Class->numeric conversion --------------------
    def _convert_to_numeric(self, y_pred_bins):
        centers = np.asarray(self.bin_info['centers'])
        return centers[np.asarray(y_pred_bins, dtype=int)]

    # -------------------- Predict helpers --------------------
    def _prepare_predict_tensors(self, X_df):
        Xp = self._engineer_features(X_df)
        # Scale nums
        if self.use_scaler and self.scaler is not None and len(self.numerical_features) > 0:
            X_num_scaled = self.scaler.transform(Xp[self.numerical_features])
        else:
            X_num_scaled = Xp[self.numerical_features].values if len(self.numerical_features) > 0 else np.zeros((len(Xp), 0), dtype=np.float32)

        # Cats
        if len(self.categorical_features) > 0:
            X_cat_tensors = [torch.tensor(Xp[col].astype(str).map(self.category_mappings[col]).fillna(0).values, dtype=torch.long) for col in self.categorical_features]
            X_cat = torch.stack(X_cat_tensors, dim=1)
        else:
            X_cat = torch.empty(len(Xp), 0, dtype=torch.long)

        # Coords
        X_coord = torch.tensor(Xp[self.coord_features].astype(float).values if len(self.coord_features) > 0 else np.zeros((len(Xp), 0), dtype=np.float32), dtype=torch.float32)

        ds = TensorDataset(X_cat, torch.tensor(X_num_scaled, dtype=torch.float32), X_coord)
        loader = DataLoader(ds, batch_size=max(1, self.batch_size * 2), shuffle=False)
        return loader, Xp.index

    def _predict_logits(self, X_df):
        loader, _ = self._prepare_predict_tensors(X_df)
        self.model.eval()
        outs = []
        with torch.no_grad():
            for b_cat, b_num, b_coord in loader:
                b_cat, b_num, b_coord = b_cat.to(device), b_num.to(device), b_coord.to(device)
                logits = self.model(b_cat, b_num, b_coord)
                outs.append(logits.cpu().numpy())
        return np.vstack(outs)

    def _predict_classes_df(self, X_df):
        logits = self._predict_logits(X_df)
        return np.argmax(logits, axis=1)

    # Public predict APIs
    def predict_argmax(self, X_df):
        classes = self._predict_classes_df(X_df)
        return self._convert_to_numeric(classes)

    def predict_expected(self, X_df):
        logits = self._predict_logits(X_df)
        p = torch.softmax(torch.tensor(logits), dim=1).numpy()
        centers = np.asarray(self.bin_info['centers']).reshape(1, -1)
        y_hat = (p * centers).sum(axis=1)
        return y_hat

    def predict(self, X_df, mode=None):
        mode = mode or self.default_predict_mode
        if mode == 'expected':
            return self.predict_expected(X_df)
        elif mode == 'argmax':
            return self.predict_argmax(X_df)
        else:
            raise ValueError("mode must be 'expected' or 'argmax'")
