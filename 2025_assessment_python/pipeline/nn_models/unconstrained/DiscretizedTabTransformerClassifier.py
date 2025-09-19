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
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, mean_absolute_error, r2_score, root_mean_squared_error

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================================
# 1) Fourier Features (shared with MLP version)
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
# 2) Tokenizers & TabTransformer backbone
# ======================================================================
class NumericalFeatureTokenizer(nn.Module):
    """Project each numerical feature to a token of size d_model."""
    def __init__(self, num_numerical_features, d_model):
        super().__init__()
        self.num_features = num_numerical_features
        self.d_model = d_model
        self.weights = nn.Parameter(torch.randn(num_numerical_features, d_model))
        self.biases = nn.Parameter(torch.randn(num_numerical_features, d_model))

    def forward(self, x_num):
        if x_num.numel() == 0:
            return x_num.new_zeros((x_num.size(0), 0, self.d_model))
        x_num = x_num.unsqueeze(-1)
        return x_num * self.weights + self.biases

class TabTransformer(nn.Module):
    def __init__(self, embedding_specs, num_numerical_features, num_coord_features,
                 fourier_type, fourier_mapping_size, fourier_sigma,
                 d_model=64, nhead=8, num_layers=4, dropout=0.1, num_classes=10):
        super().__init__()
        self.d_model = d_model
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Categorical -> embeddings -> tokens
        self.embedding_layers = nn.ModuleList([nn.Embedding(num, dim) for num, dim in embedding_specs])
        total_emb_dim = sum(dim for _, dim in embedding_specs)
        self.cat_projector = None
        if len(embedding_specs) > 0:
            # Project concatenated embeddings to one token per categorical feature
            self.cat_projector = nn.Linear(total_emb_dim, d_model * len(embedding_specs))

        # Numeric -> tokens
        self.num_tokenizer = NumericalFeatureTokenizer(num_numerical_features, d_model)

        # Coordinates -> optional Fourier -> projector -> one token
        self.fourier = None
        self.coord_projector = None
        if num_coord_features > 0 and fourier_type != 'none':
            self.fourier = FourierFeatures(num_coord_features, fourier_mapping_size, fourier_type=fourier_type, sigma=fourier_sigma)
            self.coord_projector = nn.Linear(self.fourier.output_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True, dim_feedforward=4*d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, num_classes)

    def forward(self, x_cat, x_num, x_coord):
        B = x_num.size(0)
        tokens = []

        # Categorical tokens
        if len(self.embedding_layers) > 0:
            cat_embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embedding_layers)]
            cat_embs = torch.cat(cat_embs, dim=1)  # (B, sum_dim)
            cat_tokens = self.cat_projector(cat_embs).view(B, len(self.embedding_layers), self.d_model)
            tokens.append(cat_tokens)

        # Numeric tokens
        if x_num.size(1) > 0:
            tokens.append(self.num_tokenizer(x_num))

        # Coord token (optional)
        if x_coord.size(1) > 0 and self.fourier is not None and self.coord_projector is not None:
            coord_feat = self.fourier(x_coord)
            coord_token = self.coord_projector(coord_feat).unsqueeze(1)  # (B,1,d)
            tokens.append(coord_token)

        # If no tokens at all, make a dummy zero token to avoid errors
        if len(tokens) == 0:
            tokens.append(torch.zeros(B, 1, self.d_model, device=x_num.device, dtype=x_num.dtype))

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls] + tokens, dim=1)
        h = self.encoder(x)
        cls_out = h[:, 0, :]
        return self.output(cls_out)  # logits (B, C)

# ======================================================================
# 3) Discretized TabTransformer Classifier (mirrors MLP classifier logic)
# ======================================================================
class DiscretizedTabTransformerClassifier:
    """
    A bin-based classifier using a TabTransformer backbone and geometry-aware losses.

    loss_mode in {
        'ce', 'ev_mse', 'ev_mae', 'ev_huber', 'emd', 'prob_mse', 'smooth_ce'
    }
    """
    def __init__(self, categorical_features, coord_features=None,
                 engineer_time_features=False, bin_yrblt=False, cross_township_class=False,
                 # transformer
                 d_model=64, nhead=8, num_layers=4, dropout=0.1,
                 # fourier / inputs
                 fourier_type='none', fourier_mapping_size=16, fourier_sigma=1.25,
                 # binning
                 n_bins=10, min_samples_per_bin=3, binning_method='quantile',
                 # training
                 batch_size=32, learning_rate=1e-3, num_epochs=50, patience=10, random_state=None,
                 # regularization
                 l2_lambda=0.0, l1_lambda=0.0, use_scaler=False,
                 # loss controls
                 loss_mode='ce', huber_delta=1.0, smoothing_sigma=0.0, ce_label_smoothing=0.0,
                 use_class_weights=False,
                 # inference
                 default_predict_mode='expected'):
        # Feature lists
        self.original_categorical_features = (categorical_features or [])[:]
        self.original_coord_features = (coord_features or [])[:]

        # FE flags
        self.engineer_time_features = engineer_time_features
        self.bin_yrblt = bin_yrblt
        self.cross_township_class = cross_township_class

        # Transformer params
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout

        # Fourier
        self.fourier_type = fourier_type
        self.fourier_mapping_size = fourier_mapping_size
        self.fourier_sigma = fourier_sigma

        # Binning
        self.n_bins = n_bins
        self.min_samples_per_bin = min_samples_per_bin
        self.binning_method = binning_method

        # Training
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.patience = patience
        self.random_state = random_state

        # Regularization / scaling
        self.l2_lambda = l2_lambda
        self.l1_lambda = l1_lambda
        self.use_scaler = use_scaler
        self.scaler = None

        # Loss controls
        self.loss_mode = loss_mode
        self.huber_delta = float(huber_delta)
        self.smoothing_sigma = float(smoothing_sigma)
        self.ce_label_smoothing = float(ce_label_smoothing)
        self.use_class_weights = bool(use_class_weights)

        # Inference
        self.default_predict_mode = default_predict_mode

        # Internal state
        self.model = None
        self.category_mappings = {}
        self.embedding_specs = []
        self.categorical_features = []
        self.numerical_features = []
        self.coord_features = []
        self.bin_info = {}

    # ---------------- Feature engineering ----------------
    def _engineer_features(self, X):
        X_eng = X.copy()
        if self.engineer_time_features:
            cyc = {
                'time_sale_month_of_year': 12,
                'time_sale_day_of_week': 7,
                'time_sale_day_of_year': 365.25,
                'time_sale_day_of_month': 30.44,
            }
            for col, period in cyc.items():
                if col in X_eng.columns:
                    X_eng[f'{col}_sin'] = np.sin(2 * np.pi * X_eng[col] / period)
                    X_eng[f'{col}_cos'] = np.cos(2 * np.pi * X_eng[col] / period)
            X_eng = X_eng.drop(columns=list(cyc.keys()) + ['time_sale_day'], errors='ignore')
        if self.bin_yrblt and 'char_yrblt' in X_eng.columns:
            X_eng['yrblt_decade'] = (X_eng['char_yrblt'] // 10 * 10).astype(str)
            X_eng = X_eng.drop(columns=['char_yrblt'], errors='ignore')
        if self.cross_township_class and 'meta_township_code' in X_eng.columns and 'char_class' in X_eng.columns:
            X_eng['township_class_interaction'] = X_eng['meta_township_code'].astype(str) + '_' + X_eng['char_class'].astype(str)
        return X_eng

    # ---------------- Binning helpers ----------------
    @staticmethod
    def _compute_class_weights(y_labels, num_classes):
        counts = np.bincount(y_labels, minlength=num_classes).astype(np.float32)
        counts[counts == 0] = 1.0
        weights = counts.sum() / (counts * num_classes)
        return torch.tensor(weights, dtype=torch.float32)

    @staticmethod
    def _gaussian_targets(indices, num_classes, sigma):
        device_local = indices.device
        C = num_classes
        grid = torch.arange(C, device=device_local).view(1, C)
        idx = indices.view(-1, 1)
        dist2 = (grid - idx).float().pow(2)
        t = torch.exp(-dist2 / (2.0 * (sigma ** 2)))
        t = t / (t.sum(dim=1, keepdim=True) + 1e-12)
        return t

    def _build_bin_values(self, y, y_bins, unique_labels):
        bin_values = []
        for old_label in unique_labels:
            vals = y[y_bins == old_label]
            if len(vals) == 0:
                bin_values.append(float('nan'))
            else:
                bin_values.append(float(vals.median()))
        return np.array(bin_values, dtype=np.float32)

    # ---------------- Fit ----------------
    def fit(self, X, y, X_val=None, y_val=None):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            torch.manual_seed(self.random_state)
            random.seed(self.random_state)

        X_eng = self._engineer_features(X)

        # Binning y
        if self.binning_method == 'quantile':
            y_bins, _ = pd.qcut(y, q=self.n_bins, labels=False, retbins=True, duplicates='drop')
        elif self.binning_method == 'uniform':
            y_bins, _ = pd.cut(y, bins=self.n_bins, labels=False, retbins=True, duplicates='drop')
        elif self.binning_method == 'kmeans':
            km = KMeans(n_clusters=self.n_bins, random_state=self.random_state, n_init='auto')
            y_bins = pd.Series(km.fit_predict(y.values.reshape(-1, 1)), index=y.index)
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
        bin_values_np = self._build_bin_values(y, y_bins.map(int), unique_labels)
        self.bin_info['bin_values'] = bin_values_np
        self.bin_info['centers'] = bin_values_np.copy()

        # Validation
        if X_val is not None and y_val is not None:
            X_val_eng = self._engineer_features(X_val)
            centers_all = np.array(self.bin_info['centers'])
            y_val_labels_raw = pd.Series(np.argmin(np.abs(y_val.values.reshape(-1,1) - centers_all.reshape(1,-1)), axis=1), index=y_val.index)
            valid_mask_val = y_val_labels_raw.isin(range(self.bin_info['num_bins']))
            X_val_eng = X_val_eng[valid_mask_val].reset_index(drop=True)
            y_val = y_val[valid_mask_val].reset_index(drop=True)
            y_val_labels = y_val_labels_raw[valid_mask_val].astype(int).values
        else:
            X_val_eng = None
            y_val_labels = None

        # Feature lists
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

        # Split if no val provided
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

        # Build tensors
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
        self.model = TabTransformer(
            embedding_specs=self.embedding_specs,
            num_numerical_features=len(self.numerical_features),
            num_coord_features=len(self.coord_features),
            fourier_type=self.fourier_type,
            fourier_mapping_size=self.fourier_mapping_size,
            fourier_sigma=self.fourier_sigma,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=self.dropout,
            num_classes=num_classes,
        ).to(device)

        # Loss setup
        if self.use_class_weights:
            class_weights = self._compute_class_weights(y_train_labels, num_classes).to(device)
        else:
            class_weights = None

        ce_kwargs = {}
        if self.ce_label_smoothing > 0:
            ce_kwargs['label_smoothing'] = float(self.ce_label_smoothing)
        if class_weights is not None:
            ce_kwargs['weight'] = class_weights
        ce_criterion = nn.CrossEntropyLoss(**ce_kwargs)

        if self.l2_lambda < 1e-8:
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        else:
            optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.l2_lambda)

        bin_values = torch.tensor(self.bin_info['bin_values'], dtype=torch.float32, device=device).view(1, -1)

        def _huber(a, b, delta):
            diff = a - b
            abs_diff = torch.abs(diff)
            quad = torch.minimum(abs_diff, torch.tensor(delta, device=diff.device))
            lin = abs_diff - quad
            return 0.5 * quad * quad + delta * lin

        def _emd_loss(p, y_idx):
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

        # Train
        best_val = float('inf')
        best_state = None
        patience_cnt = 0

        for epoch in range(self.num_epochs):
            self.model.train()
            tr_sum = 0.0
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
                tr_sum += float(loss.item())

            # Validation
            self.model.eval()
            va_sum = 0.0
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
                    va_sum += float(vloss.item())

            avg_tr = tr_sum / max(1, len(train_loader))
            avg_va = va_sum / max(1, len(val_loader))
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

        # Final reports
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

    # ---------------- Inference helpers ----------------
    def _convert_to_numeric(self, y_pred_bins):
        centers = np.asarray(self.bin_info['centers'])
        return centers[np.asarray(y_pred_bins, dtype=int)]

    def _prepare_predict_tensors(self, X_df):
        Xp = self._engineer_features(X_df)
        if self.use_scaler and self.scaler is not None and len(self.numerical_features) > 0:
            X_num_scaled = self.scaler.transform(Xp[self.numerical_features])
        else:
            X_num_scaled = Xp[self.numerical_features].values if len(self.numerical_features) > 0 else np.zeros((len(Xp), 0), dtype=np.float32)

        if len(self.categorical_features) > 0:
            X_cat_tensors = [torch.tensor(Xp[col].astype(str).map(self.category_mappings[col]).fillna(0).values, dtype=torch.long) for col in self.categorical_features]
            X_cat = torch.stack(X_cat_tensors, dim=1)
        else:
            X_cat = torch.empty(len(Xp), 0, dtype=torch.long)

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
