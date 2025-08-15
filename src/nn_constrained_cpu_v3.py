import numpy as np
import pandas as pd
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler


# =========================
# Utilities
# =========================
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# =========================
# MLP with Embeddings
# =========================
class TabularMLPWithEmbeddings(nn.Module):
    def __init__(self, embedding_specs, num_numeric, hidden_sizes, out_dim=1, activation=nn.ReLU):
        super().__init__()
        self.use_embeddings = len(embedding_specs) > 0
        if self.use_embeddings:
            self.emb_layers = nn.ModuleList([nn.Embedding(n_cat, emb_dim) for (n_cat, emb_dim) in embedding_specs])
            total_emb = sum(emb_dim for _, emb_dim in embedding_specs)
        else:
            self.emb_layers = None
            total_emb = 0

        in_dim = total_emb + num_numeric
        layers = []
        for i, h in enumerate(hidden_sizes):
            layers.append((f"lin_{i}", nn.Linear(in_dim, h)))
            layers.append((f"act_{i}", activation()))
            in_dim = h
        layers.append(("lin_out", nn.Linear(in_dim, out_dim)))
        self.net = nn.Sequential(OrderedDict(layers))
        self.apply(init_weights)

    def forward(self, x_num, x_cat=None):
        if self.use_embeddings and x_cat is not None and x_cat.shape[1] > 0:
            embs = [emb(x_cat[:, j]) for j, emb in enumerate(self.emb_layers)]
            x = torch.cat([*embs, x_num], dim=1)
        else:
            x = x_num
        return self.net(x).squeeze(-1)


# =========================
# Pure-PyTorch hard projector (no cvxpy)
# =========================
class RatioBoxGroupProjector(nn.Module):
    def __init__(self, tau: float, gamma: float, max_iters: int = 30):
        super().__init__()
        self.tau = float(tau)
        self.gamma = float(gamma)
        self.max_iters = int(max_iters)

    @staticmethod
    def _clamp_ratio(y, y_real, tau):
        l = (1.0 - tau) * y_real
        u = (1.0 + tau) * y_real
        return torch.clamp(y, l, u), l, u

    @staticmethod
    def _group_bisection(y_raw, y_real, mask_g, target_interval, l_g, u_g, iters):
        y0 = y_raw[mask_g]
        yr = y_real[mask_g]
        w = 1.0 / torch.clamp(yr, min=1e-9)
        l = l_g[mask_g]
        u = u_g[mask_g]

        def S(alpha):
            y_alpha = torch.clamp(y0 + alpha / w, l, u)
            return torch.sum(w * y_alpha), y_alpha

        S0, y_clip0 = S(torch.tensor(0.0, dtype=y0.dtype, device=y0.device))
        L, U = target_interval
        if (S0 >= L) and (S0 <= U):
            return y_clip0

        alpha_to_low = (l - y0) * w
        alpha_to_high = (u - y0) * w
        alpha_lo = torch.min(alpha_to_low) - 1.0
        alpha_hi = torch.max(alpha_to_high) + 1.0

        T = torch.where(S0 < L, L, U)
        lo = alpha_lo.clone()
        hi = alpha_hi.clone()
        y_mid = None
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            Smid, ymid = S(mid)
            y_mid = ymid
            lo = torch.where(Smid < T, mid, lo)
            hi = torch.where(Smid >= T, mid, hi)
        return y_mid

    def forward(self, y_raw: torch.Tensor, y_real: torch.Tensor, group_ids: torch.Tensor, n_groups: int):
        eps = 1e-9
        y_real = torch.clamp(y_real, min=eps)

        y_clipped, l, u = self._clamp_ratio(y_raw, y_real, self.tau)

        dtype = y_raw.dtype
        y_out = y_clipped.clone()
        for g in range(n_groups):
            mask = (group_ids == g)
            if not torch.any(mask):
                continue
            n_g = mask.sum().to(dtype=dtype)
            L = (1.0 - self.gamma) * n_g
            U = (1.0 + self.gamma) * n_g

            y_proj_g = self._group_bisection(
                y_raw, y_real, mask,
                target_interval=(L, U),
                l_g=l, u_g=u,
                iters=self.max_iters
            )
            y_out[mask] = y_proj_g
        return y_out


# =========================
# sklearn-like wrapper with embeddings & projection
# =========================
class ConstrainedRegressorProjectedWithEmbeddings:
    """
    Args:
        categorical_features: list[int] indices of columns to embed
        output_size: int (default 1)
        batch_size, learning_rate, num_epochs
        hidden_sizes: list[int]
        n_groups, dev_thresh (tau), group_thresh (gamma)
    """
    def __init__(self,
                 categorical_features,
                 output_size=1,
                 batch_size=16,
                 learning_rate=0.001,
                 num_epochs=50,
                 hidden_sizes=[200, 100],
                 n_groups=3,
                 dev_thresh=0.15,
                 group_thresh=0.05,
                 device=None,
                 debug=False):
        self.categorical_features = list(categorical_features) if categorical_features is not None else []
        self.output_size = int(output_size)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.num_epochs = int(num_epochs)
        self.hidden_sizes = list(hidden_sizes)
        self.n_groups = int(n_groups)
        self.tau = float(dev_thresh)
        self.gamma = float(group_thresh)
        self.debug = bool(debug)

        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.scaler = StandardScaler()

        # learned parts
        self.base = None
        self.projector = RatioBoxGroupProjector(self.tau, self.gamma, max_iters=30)

        # metadata for embeddings
        self.cat_idx = None
        self.num_idx = None
        self.cat_maps = []      # list[dict[value->id]]
        self.cat_unk_ids = []   # list[int]
        self.embedding_specs = []  # list[(vocab, emb_dim)]

    @staticmethod
    def _emb_dim(vocab):
        return int(min(50, np.ceil(vocab ** 0.25) * 4))

    @staticmethod
    def _to_array(X):
        if isinstance(X, pd.DataFrame):
            return X.values, list(X.columns)
        elif isinstance(X, np.ndarray):
            return X, [f"x{i}" for i in range(X.shape[1])]
        else:
            raise TypeError("X must be a pandas DataFrame or numpy ndarray.")

    def _split_columns(self, n_cols):
        cat = sorted(set([i for i in self.categorical_features if 0 <= i < n_cols]))
        num = [i for i in range(n_cols) if i not in cat]
        return cat, num

    def _fit_encoders(self, X_array):
        self.cat_maps = []
        self.cat_unk_ids = []
        self.embedding_specs = []
        for idx in self.cat_idx:
            col = X_array[:, idx]
            # build mapping
            uniq = pd.Series(col).astype("category").cat.categories.tolist()
            mp = {v: i for i, v in enumerate(uniq)}
            # add UNK
            unk_id = len(mp)
            mp["__UNK__"] = unk_id
            self.cat_maps.append(mp)
            self.cat_unk_ids.append(unk_id)
            vocab = len(mp)
            self.embedding_specs.append((vocab, self._emb_dim(vocab)))

    def _transform_X(self, X, fit=False):
        X_arr, _ = self._to_array(X)
        n_cols = X_arr.shape[1]
        if fit:
            self.cat_idx, self.num_idx = self._split_columns(n_cols)
            self._fit_encoders(X_arr)
            # fit scaler on numeric
            if len(self.num_idx) > 0:
                self.scaler.fit(X_arr[:, self.num_idx].astype(float))
        # numeric
        if len(self.num_idx) > 0:
            Xn = self.scaler.transform(X_arr[:, self.num_idx].astype(float))
            x_num = torch.tensor(Xn, dtype=torch.float32, device=self.device)
        else:
            x_num = torch.empty((X_arr.shape[0], 0), dtype=torch.float32, device=self.device)
        # categorical -> ids
        if len(self.cat_idx) > 0:
            cat_cols = []
            for ci, mp, unk in zip(self.cat_idx, self.cat_maps, self.cat_unk_ids):
                col = pd.Series(X_arr[:, ci])
                ids = col.map(lambda v: mp.get(v, unk)).astype(int).values
                cat_cols.append(torch.tensor(ids, dtype=torch.long, device=self.device))
            x_cat = torch.stack(cat_cols, dim=1)
        else:
            x_cat = torch.empty((X_arr.shape[0], 0), dtype=torch.long, device=self.device)
        return x_num, x_cat

    def _groups_from_y(self, y: pd.Series):
        gids = pd.qcut(pd.Series(y), q=self.n_groups, labels=False, duplicates='drop')
        uniq = np.sort(pd.unique(gids))
        remap = {u: i for i, u in enumerate(uniq)}
        gids = np.array([remap[g] for g in gids], dtype=int)
        self.n_groups = len(uniq)
        return gids

    def _build_model(self, num_numeric):
        self.base = TabularMLPWithEmbeddings(
            embedding_specs=self.embedding_specs,
            num_numeric=num_numeric,
            hidden_sizes=self.hidden_sizes,
            out_dim=self.output_size
        ).to(self.device)
        # stabilize last layer a bit
        with torch.no_grad():
            for mod in reversed(self.base.net):
                if isinstance(mod, nn.Linear):
                    mod.bias.zero_()
                    nn.init.uniform_(mod.weight, -1e-4, 1e-4)
                    break

    # =========================
    # Public API
    # =========================
    def fit(self, X, y):
        y = pd.Series(y).astype(float)
        gids = self._groups_from_y(y)

        x_num, x_cat = self._transform_X(X, fit=True)
        y_t = torch.tensor(y.values, dtype=torch.float32, device=self.device)
        gid_t = torch.tensor(gids, dtype=torch.long, device=self.device)

        self._build_model(num_numeric=x_num.shape[1])

        opt = optim.Adam(self.base.parameters(), lr=self.learning_rate)
        mse = nn.MSELoss()

        ds = TensorDataset(x_num, x_cat, y_t, gid_t)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=True)

        self.base.train()
        for epoch in range(self.num_epochs):
            tot = 0.0
            for xb_num, xb_cat, yb, gb in loader:
                opt.zero_grad()
                y_raw = self.base(xb_num, xb_cat)  # (B,)
                y_proj = self.projector(y_raw, yb, gb, n_groups=self.n_groups)
                loss = mse(y_proj, yb)
                loss.backward()
                opt.step()
                tot += float(loss.item())

                if self.debug:
                    with torch.no_grad():
                        ratio = y_proj / torch.clamp(yb, min=1e-9)
                        dev = ratio - 1.0
                        print(f"[epoch {epoch+1}] batch loss={loss.item():.6f}, "
                              f"dev[min={dev.min().item():.4f}, max={dev.max().item():.4f}]")
            print(f"Epoch {epoch+1}/{self.num_epochs} | Loss: {tot / max(1, len(loader)):.6f}")

        self.base.eval()
        return self

    def predict_constrained(self, X, y_real, group_ids=None):
        y_real = pd.Series(y_real).astype(float)
        if group_ids is None:
            gids = self._groups_from_y(y_real)
        else:
            gids = np.asarray(group_ids, dtype=int)

        x_num, x_cat = self._transform_X(X, fit=False)
        y_ref = torch.tensor(y_real.values, dtype=torch.float32, device=self.device)
        gid_t = torch.tensor(gids, dtype=torch.long, device=self.device)

        ds = TensorDataset(x_num, x_cat, y_ref, gid_t)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)

        preds = []
        with torch.no_grad():
            for xb_num, xb_cat, yb, gb in loader:
                y_raw = self.base(xb_num, xb_cat)
                y_proj = self.projector(y_raw, yb, gb, n_groups=self.n_groups)
                preds.append(y_proj.cpu().numpy())
        return np.concatenate(preds, axis=0)

    def predict(self, X):
        x_num, x_cat = self._transform_X(X, fit=False)
        outs = []
        self.base.eval()
        with torch.no_grad():
            ds = TensorDataset(x_num, x_cat)
            loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
            for xb_num, xb_cat in loader:
                outs.append(self.base(xb_num, xb_cat).cpu().numpy())
        return np.concatenate(outs, axis=0)
