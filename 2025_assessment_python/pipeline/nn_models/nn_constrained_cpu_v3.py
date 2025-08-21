# constrained_nn_with_embeddings.py
#
# A sklearn-like PyTorch model that ALWAYS applies embeddings to the
# specified categorical columns, and projects each minibatch prediction
# to satisfy fairness-style ratio constraints — no cvxpy, no DPP.
#
# Key features
# - Always-on embeddings for `categorical_features` (list of column indices)
# - Group-balanced BatchSampler (optional)
# - Hard projection layer enforcing per-item and per-group ratio constraints
# - Numeric features are standardized with sklearn's StandardScaler
#
# API (example)
# model = ConstrainedRegressorProjectedWithEmbeddings(
#     categorical_features=[0, 3, 5],  # indices to embed
#     hidden_sizes=[200, 100],
#     batch_size=16,
#     learning_rate=1e-3,
#     num_epochs=50,
#     n_groups=3,
#     dev_thresh=0.15,
#     group_thresh=0.05,
#     use_group_balanced_sampler=True,
# )
# model.fit(X_train, y_train)
# yhat = model.predict(X_valid, y_real=y_valid, use_projection=True)

from __future__ import annotations
import math
import numpy as np
import pandas as pd
from collections import OrderedDict
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Sampler
from sklearn.preprocessing import StandardScaler


# ---------------------------
# Group-balanced BatchSampler
# ---------------------------
class GroupBalancedBatchSampler(Sampler[List[int]]):
    """Yields batches with a fixed (near-equal by default) per-group composition.

    Args:
        group_ids: 1D array-like of ints in [0, n_groups-1] for each dataset item
        batch_size: total batch size
        n_groups: number of distinct groups (if None, inferred)
        per_group_counts: optional list of ints summing to batch_size; if None -> near-equal split
        shuffle: shuffle indices each epoch
        drop_last: drop incomplete final batch
        replacement: sample with replacement when a group runs out (keeps composition at cost of repeats)
        generator: optional torch.Generator for deterministic shuffling
    """
    def __init__(
        self,
        group_ids,
        batch_size: int,
        n_groups: Optional[int] = None,
        per_group_counts: Optional[List[int]] = None,
        shuffle: bool = True,
        drop_last: bool = True,
        replacement: bool = False,
        generator: Optional[torch.Generator] = None,
    ):
        group_ids = np.asarray(group_ids, dtype=int)
        assert group_ids.ndim == 1, "group_ids must be 1D"
        self.N = len(group_ids)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.replacement = bool(replacement)
        self.gen = generator

        self.n_groups = int(n_groups if n_groups is not None else (group_ids.max() + 1 if self.N > 0 else 0))
        self.indices_by_group = [np.where(group_ids == g)[0].tolist() for g in range(self.n_groups)]
        self.group_sizes = np.array([len(ix) for ix in self.indices_by_group], dtype=int)

        if per_group_counts is not None:
            k = np.asarray(per_group_counts, dtype=int)
            assert len(k) == self.n_groups, "per_group_counts length != n_groups"
            assert k.sum() == self.batch_size, "per_group_counts must sum to batch_size"
            assert np.all(k >= 0)
            self.k = k
        else:
            # near-equal split
            base = np.full(self.n_groups, self.batch_size // self.n_groups, dtype=int)
            base[: self.batch_size % self.n_groups] += 1
            self.k = base

        if self.replacement:
            self._num_batches_est = math.ceil(self.N / max(1, self.batch_size))
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                per_group_full = np.array(
                    [(self.group_sizes[g] // max(1, self.k[g])) if self.k[g] > 0 else np.inf for g in range(self.n_groups)],
                    dtype=float,
                )
            full_batches = int(np.min(per_group_full[np.isfinite(per_group_full)]) if np.isfinite(per_group_full).any() else 0)
            self._num_batches_est = full_batches

    def __len__(self):
        return self._num_batches_est

    def _rng(self):
        if self.gen is None:
            return np.random.default_rng()
        seed = int(torch.randint(0, 2**31 - 1, (1,), generator=self.gen).item())
        return np.random.default_rng(seed)

    def __iter__(self):
        rng = self._rng()
        pools = []
        for g in range(self.n_groups):
            pool = self.indices_by_group[g].copy()
            if self.shuffle:
                rng.shuffle(pool)
            pools.append(pool)

        while True:
            batch = []
            for g in range(self.n_groups):
                need = int(self.k[g])
                if need == 0:
                    continue
                if self.replacement:
                    if len(pools[g]) == 0:
                        continue
                    choice = rng.choice(pools[g], size=need, replace=True)
                    batch.extend(choice.tolist())
                else:
                    take = []
                    while need > 0 and len(pools[g]) > 0:
                        take.append(pools[g].pop())
                        need -= 1
                    if need > 0:
                        if self.drop_last:
                            return
                    batch.extend(take)

            if len(batch) == 0:
                return
            if self.drop_last and len(batch) < self.batch_size and not self.replacement:
                return
            if len(batch) > self.batch_size:
                batch = batch[: self.batch_size]
            yield batch


# ---------------------------
# Embedding block (ALWAYS used for specified columns)
# ---------------------------
class CategoricalEncoder:
    """Builds per-column vocabularies with [UNK] and [NA] tokens and maps values -> ids."""
    def __init__(self, columns: List[int]):
        self.columns = sorted(list(columns))
        self.vocabs: List[dict] = []
        self.unk_ids: List[int] = []
        self.na_ids: List[int] = []

    def fit(self, X: np.ndarray):
        self.vocabs.clear(); self.unk_ids.clear(); self.na_ids.clear()
        for idx in self.columns:
            col = pd.Series(X[:, idx])
            uniq = col.dropna().astype("category").cat.categories.tolist()
            mp = {v: i for i, v in enumerate(uniq)}
            unk = len(mp); na = unk + 1
            mp["__UNK__"] = unk
            mp["__NA__"]  = na
            self.vocabs.append(mp)
            self.unk_ids.append(unk)
            self.na_ids.append(na)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        ids = []
        for (idx, mp, unk, na) in zip(self.columns, self.vocabs, self.unk_ids, self.na_ids):
            col = pd.Series(X[:, idx])
            def map_val(v):
                if pd.isna(v):
                    return na
                return mp.get(v, unk)
            ids.append(col.map(map_val).astype(int).values)
        return np.stack(ids, axis=1) if len(ids) > 0 else np.empty((X.shape[0], 0), dtype=int)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def vocab_sizes(self) -> List[int]:
        return [len(mp) for mp in self.vocabs]


def default_emb_dim(vocab: int) -> int:
    return int(min(50, np.ceil(vocab ** 0.25) * 4))


class EmbeddingBlock(nn.Module):
    """ModuleList of embeddings for each categorical column; always applied."""
    def __init__(self, vocab_sizes: List[int], emb_dims: Optional[List[int]] = None):
        super().__init__()
        if emb_dims is None:
            emb_dims = [default_emb_dim(v) for v in vocab_sizes]
        assert len(vocab_sizes) == len(emb_dims)
        self.emb_layers = nn.ModuleList([nn.Embedding(v, d) for v, d in zip(vocab_sizes, emb_dims)])
        self.out_dim = int(sum(emb_dims))

    def forward(self, x_cat_ids: torch.Tensor) -> torch.Tensor:
        # Ensure x_cat_ids is on the same device as the embedding layers
        if x_cat_ids.device != self.emb_layers[0].weight.device:
            x_cat_ids = x_cat_ids.to(self.emb_layers[0].weight.device)
        embs = [emb(x_cat_ids[:, j]) for j, emb in enumerate(self.emb_layers)]
        return torch.cat(embs, dim=1) if len(embs) > 0 else torch.empty((x_cat_ids.shape[0], 0), device=x_cat_ids.device)

# ---------------------------
# MLP backbone
# ---------------------------
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], out_dim: int = 1, activation=nn.ReLU):
        super().__init__()
        layers = []
        d = in_dim
        for i, h in enumerate(hidden):
            layers.append((f"lin_{i}", nn.Linear(d, h)))
            layers.append((f"act_{i}", activation()))
            d = h
        layers.append(("lin_out", nn.Linear(d, out_dim)))
        self.net = nn.Sequential(OrderedDict(layers))
        self.apply(init_weights)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------
# Hard projection (per minibatch)
# ---------------------------
class RatioBoxGroupProjector(nn.Module):
    """min ||y - y_raw||^2 s.t.
         (1-τ) y_real <= y <= (1+τ) y_real
         sum_{i in G_j} (y_i / y_real_i) in [(1-γ) n_j, (1+γ) n_j]
       Closed-form via groupwise bisection; differentiable through unrolled ops.
    """
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

        def S(alpha: torch.Tensor):
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
        lo = alpha_lo.clone(); hi = alpha_hi.clone(); y_mid = None
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
            y_proj_g = self._group_bisection(y_raw, y_real, mask, (L, U), l, u, self.max_iters)
            y_out[mask] = y_proj_g
        return y_out


# ---------------------------
# Main sklearn-like model
# ---------------------------
class ConstrainedRegressorProjectedWithEmbeddings:
    """Neural regressor with ALWAYS-ON embeddings for specified columns and
    a hard projection layer enforcing fairness-style ratio constraints per minibatch.

    Constraints
    ----------
    dev_thresh (tau): per-sample |y_pred/y_real - 1| <= tau
    group_thresh (gamma): per-group mean ratio in [1-gamma, 1+gamma]

    Groups are y-quantiles with `n_groups` bins.
    """
    def __init__(
        self,
        categorical_features: List[int],
        hidden_sizes: List[int] = [200, 100],
        batch_size: int = 16,
        learning_rate: float = 1e-3,
        num_epochs: int = 50,
        n_groups: int = 3,
        dev_thresh: float = 0.15,
        group_thresh: float = 0.05,
        use_group_balanced_sampler: bool = False,
        replacement: bool = False,
        random_state: Optional[int] = None,
        device: Optional[str] = None,
        debug: bool = False,
    ):
        assert len(categorical_features) >= 0, "Provide list (possibly empty) of categorical column indices"
        self.cat_cols = sorted(list(categorical_features))
        self.hidden_sizes = list(hidden_sizes)
        self.batch_size = int(batch_size)
        self.lr = float(learning_rate)
        self.epochs = int(num_epochs)
        self.n_groups = int(n_groups)
        self.tau = float(dev_thresh)
        self.gamma = float(group_thresh)
        self.use_balanced = bool(use_group_balanced_sampler)
        self.replacement = bool(replacement)
        self.random_state = random_state
        self.debug = bool(debug)

        self.device = torch.device(device) if device is not None else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.scaler = StandardScaler()

        # encoders/embeddings
        self.cat_encoder = CategoricalEncoder(self.cat_cols)
        self.emb_block: Optional[EmbeddingBlock] = None
        self.num_idx: List[int] = []

        # model parts
        self.backbone: Optional[MLP] = None
        self.projector = RatioBoxGroupProjector(self.tau, self.gamma, max_iters=30)

    # ---- utilities ----
    @staticmethod
    def _as_array(X) -> Tuple[np.ndarray, int]:
        if isinstance(X, pd.DataFrame):
            return X.values, X.shape[1]
        elif isinstance(X, np.ndarray):
            return X, X.shape[1]
        else:
            raise TypeError("X must be a pandas DataFrame or numpy ndarray.")

    def _split_indices(self, n_cols: int):
        cat = self.cat_cols
        num = [i for i in range(n_cols) if i not in cat]
        self.num_idx = num

    def _build_embeddings(self, X_arr: np.ndarray):
        cat_ids = self.cat_encoder.fit_transform(X_arr)
        vocab_sizes = self.cat_encoder.vocab_sizes()
        self.emb_block = EmbeddingBlock(vocab_sizes)
        return cat_ids

    def _numeric_tensor(self, X_arr: np.ndarray, fit: bool) -> torch.Tensor:
        if len(self.num_idx) > 0:
            Xn = X_arr[:, self.num_idx].astype(float)
            if fit:
                self.scaler.fit(Xn)
            Xn = self.scaler.transform(Xn)
            return torch.tensor(Xn, dtype=torch.float32, device=self.device)
        else:
            return torch.empty((X_arr.shape[0], 0), dtype=torch.float32, device=self.device)

    def _categorical_tensor(self, X_arr: np.ndarray, fit: bool) -> torch.Tensor:
        if len(self.cat_cols) == 0:
            return torch.empty((X_arr.shape[0], 0), dtype=torch.long, device=self.device)
        if fit:
            cat_ids = self._build_embeddings(X_arr)
        else:
            cat_ids = self.cat_encoder.transform(X_arr)
        return torch.tensor(cat_ids, dtype=torch.long, device=self.device)

    def _build_backbone(self, num_dim: int):
        emb_dim = 0 if self.emb_block is None else self.emb_block.out_dim
        in_dim = num_dim + emb_dim
        self.backbone = MLP(in_dim, self.hidden_sizes, out_dim=1).to(self.device)
        if self.emb_block:
            self.emb_block.to(self.device)
        with torch.no_grad():
            # stabilize final layer
            for mod in reversed(self.backbone.net):
                if isinstance(mod, nn.Linear):
                    mod.bias.zero_(); nn.init.uniform_(mod.weight, -1e-4, 1e-4)
                    break

    def _groups_from_y(self, y: np.ndarray) -> np.ndarray:
        gids = pd.qcut(pd.Series(y), q=self.n_groups, labels=False, duplicates="drop")
        uniq = np.sort(pd.unique(gids))
        remap = {u: i for i, u in enumerate(uniq)}
        gids = np.array([remap[g] for g in gids], dtype=int)
        self.n_groups = len(uniq)
        return gids

    # ---- sklearn-like API ----
    def fit(self, X, y):
        X_arr, n_cols = self._as_array(X)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        assert X_arr.shape[0] == y_arr.shape[0]
        self._split_indices(n_cols)

        # build tensors
        x_num = self._numeric_tensor(X_arr, fit=True)
        x_cat = self._categorical_tensor(X_arr, fit=True)
        y_t   = torch.tensor(y_arr, dtype=torch.float32, device=self.device)

        # groups by y-quantiles
        gids = self._groups_from_y(y_arr)
        g_t  = torch.tensor(gids, dtype=torch.long, device=self.device)

        # model
        self._build_backbone(num_dim=x_num.shape[1])
        assert self.emb_block is not None or x_num.shape[1] > 0, "No features provided."

        # dataset
        ds = TensorDataset(x_num, x_cat, y_t, g_t)

        # loader
        if self.use_balanced and self.n_groups > 0:
            base = np.full(self.n_groups, self.batch_size // self.n_groups, dtype=int)
            base[: self.batch_size % self.n_groups] += 1
            gen = torch.Generator()
            if self.random_state is not None:
                gen.manual_seed(int(self.random_state))
            batch_sampler = GroupBalancedBatchSampler(
                group_ids=gids,
                batch_size=self.batch_size,
                n_groups=self.n_groups,
                per_group_counts=base.tolist(),
                shuffle=True,
                drop_last=True,
                replacement=False,
                generator=gen,
            )
            loader = DataLoader(ds, batch_sampler=batch_sampler)
        else:
            loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=True)

        # training
        opt = optim.Adam(list(self.backbone.parameters()) + list(self.emb_block.parameters() if self.emb_block else []), lr=self.lr)
        mse = nn.MSELoss()
        self.backbone.train();  
        if self.emb_block: self.emb_block.train()

        for epoch in range(self.epochs):
            tot = 0.0
            for xb_num, xb_cat, yb, gb in loader:
                opt.zero_grad()
                if self.emb_block and xb_cat.shape[1] > 0:
                    x_emb = self.emb_block(xb_cat)
                    feats = torch.cat([x_emb, xb_num], dim=1)
                else:
                    feats = xb_num
                y_raw = self.backbone(feats)
                y_proj = self.projector(y_raw, yb, gb, n_groups=self.n_groups)
                loss = mse(y_proj, yb)
                loss.backward(); opt.step()
                tot += float(loss.item())
            print(f"Epoch {epoch+1}/{self.epochs} | Loss: {tot / max(1, len(loader)):.6f}")

        self.backbone.eval(); 
        if self.emb_block: self.emb_block.eval()
        return self

    def _forward_blocks(self, xb_num: torch.Tensor, xb_cat: torch.Tensor) -> torch.Tensor:
        if self.emb_block and xb_cat.shape[1] > 0:
            x_emb = self.emb_block(xb_cat)
            feats = torch.cat([x_emb, xb_num], dim=1)
        else:
            feats = xb_num
        return self.backbone(feats)

    def predict_constrained(self, X, y_real, use_projection: bool = True) -> np.ndarray:
        X_arr, _ = self._as_array(X)
        y_arr = np.asarray(y_real, dtype=float).reshape(-1)
        assert X_arr.shape[0] == y_arr.shape[0]

        x_num = self._numeric_tensor(X_arr, fit=False)
        x_cat = self._categorical_tensor(X_arr, fit=False)
        y_t   = torch.tensor(y_arr, dtype=torch.float32, device=self.device)

        gids = self._groups_from_y(y_arr)
        g_t  = torch.tensor(gids, dtype=torch.long, device=self.device)

        ds = TensorDataset(x_num, x_cat, y_t, g_t)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)

        outs = []
        with torch.no_grad():
            for xb_num, xb_cat, yb, gb in loader:
                y_raw = self._forward_blocks(xb_num, xb_cat)
                y_out = self.projector(y_raw, yb, gb, n_groups=self.n_groups) if use_projection else y_raw
                outs.append(y_out.cpu().numpy())
        return np.concatenate(outs, axis=0)

    def predict(self, X) -> np.ndarray:
        X_arr, _ = self._as_array(X)
        x_num = self._numeric_tensor(X_arr, fit=False)
        x_cat = self._categorical_tensor(X_arr, fit=False)
        ds = TensorDataset(x_num, x_cat)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        outs = []
        self.backbone.eval();  
        if self.emb_block: self.emb_block.eval()
        with torch.no_grad():
            for xb_num, xb_cat in loader:
                y_raw = self._forward_blocks(xb_num, xb_cat)
                outs.append(y_raw.cpu().numpy())
        return np.concatenate(outs, axis=0)
