"""Attention-based team model for the brawler-pick recommender (v3).

Goal: capture interactions LightGBM can't easily get with multi-hot features.
A small transformer encoder over (CLS, ctx, team-A brawlers, team-B brawlers)
should let the model learn brawler-vs-brawler matchups end-to-end via attention.

Architecture (deliberately small — CPU-friendly, 1-3 min/epoch on i7-12700H):

  Tokens (8 per battle):
    [CLS] [CTX] [A1] [A2] [A3] [B1] [B2] [B3]

  Embeddings per token:
    token = brawler_emb + side_emb + scalar_proj(trophy, power)
            (or for CTX: mode_emb + map_emb + btype_emb + side_emb)
            (or for CLS: just learnable_cls + cls_side_emb)

  Encoder:
    nn.TransformerEncoder(d_model=64, nhead=4, ff=128, num_layers=2, dropout=0.1)

  Head:
    encoder(tokens)[:, 0, :]  ⊕  [a_trophy_log, b_trophy_log, trophy_diff_log]
      → Linear(d_model+3, d_model) → ReLU → Dropout(0.1) → Linear(d_model, 1)

  Loss: BCEWithLogitsLoss
  Opt:  AdamW(lr=1e-3, wd=1e-4)
  Schd: cosine warmup → cosine anneal

The model interface mirrors LGBMTeamModel: `.fit(train_df, valid_df=...)` then
`.predict_proba(df) -> np.ndarray`. That way `evaluate()` and `evaluate_topk()`
work without changes.

Save format:
    <prefix>.pt        torch state dict
    <prefix>.meta.json  vocab + hyperparams (so we can recreate the model)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .features import (
    PHASE2_DIM,
    PHASE4_DIM,
    TEAM_AGGREGATE_DIM,
    TeamFeaturizer,
    compute_phase2_features,
    compute_phase4_features,
    compute_team_aggregates,
)

# --- token type ids -----------------------------------------------------------
# (used by the side/segment embedding so the encoder can tell tokens apart)
TOK_CLS = 0
TOK_CTX = 1
TOK_A   = 2
TOK_B   = 3


def _log1p_clip(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x = np.where(np.isnan(x), 0.0, x)
    return np.log1p(np.maximum(x, 0.0)).astype(np.float32)


def _df_to_tensors(
    df: pd.DataFrame,
    f: TeamFeaturizer,
    *,
    include_team_aggregates: bool = False,
    include_time_features: bool = False,
    include_history_features: bool = False,
) -> dict[str, torch.Tensor]:
    """Build the tensor dict the transformer consumes from a dataframe.

    Per-battle tensors:
        a_brawlers   (B, 3) int64        — brawler vocab ids (0 = unknown/pad)
        b_brawlers   (B, 3) int64
        a_trophies   (B, 3) float32      — log1p
        b_trophies   (B, 3) float32
        a_powers     (B, 3) float32      — power / 11.0 (so max = 1.0)
        b_powers     (B, 3) float32
        mode_id      (B,)   int64        — vocab id (0 = unknown)
        map_id       (B,)   int64
        btype_id     (B,)   int64
        scalar       (B, 3) float32      — [a_log_t_mean, b_log_t_mean, diff]
        extra_scalar (B, K) float32      — concat of optional add-on features:
                                           phase 1 (23 dims, team aggregates)
                                           and/or phase 2 (12 dims, time +
                                           days_since_release). (B, 0) when
                                           both flags are off.
        y            (B,)   float32      — team_a_wins
    """
    n = len(df)

    # +1 in the embedding tables reserves index 0 as PAD/UNK.
    def _bidx(b: int) -> int:
        i = f.brawler_to_idx.get(int(b))
        return 0 if i is None else (i + 1)

    a_b = np.zeros((n, 3), dtype=np.int64)
    b_b = np.zeros((n, 3), dtype=np.int64)
    a_t = np.zeros((n, 3), dtype=np.float32)
    b_t = np.zeros((n, 3), dtype=np.float32)
    a_p = np.zeros((n, 3), dtype=np.float32)
    b_p = np.zeros((n, 3), dtype=np.float32)

    has_per_brawler = "team_a_trophies" in df.columns
    a_t_mean_log = _log1p_clip(df["team_a_trophies_mean"].values)
    b_t_mean_log = _log1p_clip(df["team_b_trophies_mean"].values)

    for i in range(n):
        ta = df["team_a"].iat[i]
        tb = df["team_b"].iat[i]
        for k in range(min(3, len(ta))):
            a_b[i, k] = _bidx(int(ta[k]))
        for k in range(min(3, len(tb))):
            b_b[i, k] = _bidx(int(tb[k]))
        if has_per_brawler:
            tat = df["team_a_trophies"].iat[i]
            tbt = df["team_b_trophies"].iat[i]
            tap = df["team_a_powers"].iat[i]
            tbp = df["team_b_powers"].iat[i]
            for k in range(min(3, len(ta))):
                a_t[i, k] = float(np.log1p(max(0, int(tat[k]))))
                a_p[i, k] = float(int(tap[k])) / 11.0
            for k in range(min(3, len(tb))):
                b_t[i, k] = float(np.log1p(max(0, int(tbt[k]))))
                b_p[i, k] = float(int(tbp[k])) / 11.0
        else:
            # Fallback: same trophy mean for every slot (degraded; no power).
            for k in range(3):
                a_t[i, k] = a_t_mean_log[i]
                b_t[i, k] = b_t_mean_log[i]
                a_p[i, k] = 1.0
                b_p[i, k] = 1.0

    mode_id = np.array(
        [(f.mode_to_idx.get(str(m), -1) + 1) for m in df["mode"].fillna("UNKNOWN").astype(str).values],
        dtype=np.int64,
    )
    map_id = np.array(
        [(f.map_to_idx.get(str(m), -1) + 1) for m in df["map"].fillna("UNKNOWN").astype(str).values],
        dtype=np.int64,
    )
    btype_id = np.array(
        [(f.btype_to_idx.get(str(m), -1) + 1) for m in df["battle_type"].fillna("UNKNOWN").astype(str).values],
        dtype=np.int64,
    )

    scalar = np.stack(
        [a_t_mean_log, b_t_mean_log, a_t_mean_log - b_t_mean_log], axis=1
    ).astype(np.float32)

    extras: list[np.ndarray] = []
    if include_team_aggregates:
        extras.append(compute_team_aggregates(df))
    if include_time_features:
        extras.append(compute_phase2_features(df, f.brawler_first_seen))
    if include_history_features:
        extras.append(compute_phase4_features(df, f.player_history))
    if extras:
        extra_scalar = np.concatenate(extras, axis=1).astype(np.float32)
    else:
        extra_scalar = np.zeros((n, 0), dtype=np.float32)

    if "team_a_wins" in df.columns:
        y = df["team_a_wins"].values.astype(np.float32)
    else:
        y = np.zeros(n, dtype=np.float32)

    return dict(
        a_brawlers=torch.from_numpy(a_b),
        b_brawlers=torch.from_numpy(b_b),
        a_trophies=torch.from_numpy(a_t),
        b_trophies=torch.from_numpy(b_t),
        a_powers=torch.from_numpy(a_p),
        b_powers=torch.from_numpy(b_p),
        mode_id=torch.from_numpy(mode_id),
        map_id=torch.from_numpy(map_id),
        btype_id=torch.from_numpy(btype_id),
        scalar=torch.from_numpy(scalar),
        extra_scalar=torch.from_numpy(extra_scalar),
        y=torch.from_numpy(y),
    )


class _TransformerCore(nn.Module):
    """The actual nn.Module. Kept separate from the sklearn-like wrapper so the
    wrapper stays small and testable."""

    def __init__(
        self,
        n_brawlers: int,
        n_modes: int,
        n_maps: int,
        n_btypes: int,
        d_model: int = 64,
        nhead: int = 4,
        ff: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        extra_scalar_dim: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.extra_scalar_dim = int(extra_scalar_dim)

        # +1 for PAD/UNK at index 0
        self.brawler_emb = nn.Embedding(n_brawlers + 1, d_model, padding_idx=0)
        self.mode_emb    = nn.Embedding(n_modes + 1, d_model, padding_idx=0)
        self.map_emb     = nn.Embedding(n_maps + 1, d_model, padding_idx=0)
        self.btype_emb   = nn.Embedding(n_btypes + 1, d_model, padding_idx=0)

        # Token-side / segment embedding (CLS, CTX, A, B = 4 types).
        self.side_emb = nn.Embedding(4, d_model)

        # Per-brawler scalar projection: (trophy_log, power_norm) -> d_model.
        self.scalar_proj_brawler = nn.Linear(2, d_model, bias=True)

        # Learnable [CLS]
        self.cls = nn.Parameter(torch.randn(d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # `enable_nested_tensor=False`: nested tensors only help with norm_first=False
        # and a key_padding_mask layout we don't fit; explicitly disabling silences
        # the runtime warning and avoids a fallback path.
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.head_norm = nn.LayerNorm(d_model)

        # Combine CLS pool with the global trophy scalars and (optionally) the
        # phase-1 team aggregates. With extra_scalar_dim=0 the head shape
        # matches every pre-phase-1 saved model byte-for-byte.
        self.head = nn.Sequential(
            nn.Linear(d_model + 3 + self.extra_scalar_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def _brawler_token(self, ids: torch.Tensor, troph: torch.Tensor, power: torch.Tensor, side: int) -> torch.Tensor:
        """ids (B,3); troph (B,3); power (B,3) -> (B, 3, d_model)."""
        emb = self.brawler_emb(ids)                                  # (B,3,d)
        scal = torch.stack([troph, power], dim=-1)                   # (B,3,2)
        emb = emb + self.scalar_proj_brawler(scal)                   # (B,3,d)
        side_e = self.side_emb(torch.full_like(ids, side))           # (B,3,d)
        return emb + side_e

    def _ctx_token(self, mode_id: torch.Tensor, map_id: torch.Tensor, btype_id: torch.Tensor) -> torch.Tensor:
        ctx = self.mode_emb(mode_id) + self.map_emb(map_id) + self.btype_emb(btype_id)  # (B,d)
        ctx = ctx + self.side_emb(torch.full_like(mode_id, TOK_CTX))                    # (B,d)
        return ctx.unsqueeze(1)                                                          # (B,1,d)

    def _cls_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        cls = self.cls.unsqueeze(0).expand(batch_size, -1)                               # (B,d)
        cls = cls + self.side_emb(torch.zeros(batch_size, dtype=torch.long, device=device))  # TOK_CLS=0
        return cls.unsqueeze(1)                                                           # (B,1,d)

    def forward(
        self,
        a_brawlers, b_brawlers, a_trophies, b_trophies, a_powers, b_powers,
        mode_id, map_id, btype_id, scalar, extra_scalar,
    ) -> torch.Tensor:
        bsz = a_brawlers.shape[0]
        device = a_brawlers.device
        cls = self._cls_token(bsz, device)
        ctx = self._ctx_token(mode_id, map_id, btype_id)
        a = self._brawler_token(a_brawlers, a_trophies, a_powers, side=TOK_A)
        b = self._brawler_token(b_brawlers, b_trophies, b_powers, side=TOK_B)
        # Concat tokens: [CLS, CTX, A1, A2, A3, B1, B2, B3]  -> (B, 8, d)
        tokens = torch.cat([cls, ctx, a, b], dim=1)

        # Mask out PAD positions in the brawler tokens (id == 0). CLS and CTX
        # are always valid. Mask shape (B, 8) — True = ignore.
        a_pad = a_brawlers == 0                          # (B,3)
        b_pad = b_brawlers == 0
        attn_pad = torch.cat([
            torch.zeros(bsz, 2, dtype=torch.bool, device=device),  # CLS, CTX
            a_pad, b_pad,
        ], dim=1)

        encoded = self.encoder(tokens, src_key_padding_mask=attn_pad)
        cls_out = self.head_norm(encoded[:, 0, :])                      # (B,d)
        # extra_scalar is (B, 0) when phase 1 is disabled — the cat is a no-op
        # and head_in shape matches every legacy saved model.
        head_in = torch.cat([cls_out, scalar, extra_scalar], dim=-1)    # (B, d+3+K)
        return self.head(head_in).squeeze(-1)                            # (B,)


@dataclass
class TransformerTeamModel:
    """sklearn-like wrapper around `_TransformerCore` with the LGBM-style API."""
    d_model: int = 64
    nhead: int = 4
    ff: int = 128
    num_layers: int = 2
    dropout: float = 0.1

    epochs: int = 5
    batch_size: int = 4096
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    grad_clip: float = 1.0
    eval_batch_size: int = 16384
    early_stop_patience: int = 2

    num_workers: int = 0       # in-RAM tensors → keep this 0 for less overhead
    device: str = "cpu"
    verbose: bool = True
    log_every: int = 50

    # Phase-1: when True, the model widens the head's scalar input by
    # TEAM_AGGREGATE_DIM (23) and consumes per-team trophy/power aggregates.
    # Phase-2: same mechanism, additional PHASE2_DIM (12) for cyclical-time
    # and `days_since_release` aggregates.
    # Phase-4: additional PHASE4_DIM (20) for per-team aggregates of
    # per-player history stats. Requires `team_a/b_player_tags` on the input
    # DataFrame and the lookup is fit on training data only.
    # All default False so the head shape matches legacy saved models when
    # no flags are set.
    use_team_aggregates: bool = False
    use_time_features: bool = False
    use_history_features: bool = False

    featurizer: TeamFeaturizer | None = None
    model: _TransformerCore | None = None
    history: list[dict] = field(default_factory=list)

    # Names of all tensors carried through forward(). Order matches the
    # `_TransformerCore.forward` signature so we can splat in calls below.
    _TENSOR_KEYS = (
        "a_brawlers", "b_brawlers",
        "a_trophies", "b_trophies",
        "a_powers", "b_powers",
        "mode_id", "map_id", "btype_id",
        "scalar", "extra_scalar",
    )

    @property
    def extra_scalar_dim(self) -> int:
        dim = 0
        if self.use_team_aggregates:
            dim += TEAM_AGGREGATE_DIM
        if self.use_time_features:
            dim += PHASE2_DIM
        if self.use_history_features:
            dim += PHASE4_DIM
        return dim

    def _set_seed(self) -> None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    @staticmethod
    def _move_to_device(t: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
        return {k: v.to(device) for k, v in t.items()}

    def _iter_batches(
        self,
        t: dict[str, torch.Tensor],
        *,
        batch_size: int,
        shuffle: bool,
        device: torch.device,
        seed: int | None = None,
    ):
        """Yield batches of (input_tensors_in_TENSOR_KEYS_order, y) without
        going through `torch.utils.data.DataLoader`.

        Why this exists: with the dataset preloaded onto the GPU, every step's
        per-batch CPU↔GPU memcopy disappears; the only overhead is one
        `torch.randperm` per epoch and one fancy-index gather per batch — both
        kernel-launch-bound. Empirically: epoch wall time on the v3 transformer
        (251 k params, 1.87 M rows) drops from ~85 s on GPU with DataLoader to
        ~10 s with this path on the same RTX 3060 Mobile.
        """
        n = t["y"].shape[0]
        if shuffle:
            g = None
            if seed is not None:
                g = torch.Generator(device=device); g.manual_seed(seed)
            perm = torch.randperm(n, device=device, generator=g)
        else:
            perm = torch.arange(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            inputs = tuple(t[k][idx] for k in self._TENSOR_KEYS)
            y = t["y"][idx]
            yield inputs, y

    def fit(
        self,
        df: pd.DataFrame,
        y: np.ndarray | None = None,
        valid_df: pd.DataFrame | None = None,
        valid_y: np.ndarray | None = None,
        history_df: pd.DataFrame | None = None,
    ) -> "TransformerTeamModel":
        self._set_seed()
        device = torch.device(self.device)

        f = TeamFeaturizer(
            include_team_aggregates=self.use_team_aggregates,
            include_time_features=self.use_time_features,
            include_history_features=self.use_history_features,
        ).fit(df, history_df=history_df)
        self.featurizer = f

        self.model = _TransformerCore(
            n_brawlers=f.n_brawlers,
            n_modes=f.n_modes,
            n_maps=f.n_maps,
            n_btypes=f.n_btypes,
            d_model=self.d_model,
            nhead=self.nhead,
            ff=self.ff,
            num_layers=self.num_layers,
            dropout=self.dropout,
            extra_scalar_dim=self.extra_scalar_dim,
        ).to(device)

        if self.verbose:
            n_params = sum(p.numel() for p in self.model.parameters())
            print(f"[transformer] vocab: brawlers={f.n_brawlers} modes={f.n_modes} "
                  f"maps={f.n_maps} btypes={f.n_btypes}; params: {n_params:,}; "
                  f"extra_scalar_dim={self.extra_scalar_dim} "
                  f"(team_aggregates={self.use_team_aggregates}, "
                  f"time_features={self.use_time_features}, "
                  f"history_features={self.use_history_features}); device: {device}")

        # Build tensors
        if self.verbose:
            print(f"[transformer] tensorizing train ({len(df):,} rows)...")
        train_t = _df_to_tensors(
            df, f,
            include_team_aggregates=self.use_team_aggregates,
            include_time_features=self.use_time_features,
            include_history_features=self.use_history_features,
        )

        if valid_df is None:
            # Internal 5% validation split (random, after shuffling). Same seed
            # as the manual seed so it's reproducible.
            n = len(df)
            n_val = max(1, int(0.05 * n))
            rng = np.random.default_rng(self.seed)
            perm = rng.permutation(n)
            val_idx = perm[:n_val]
            tr_idx = perm[n_val:]
            sub = lambda t, idx: {k: v[idx] for k, v in t.items()}
            valid_t = sub(train_t, val_idx)
            train_t = sub(train_t, tr_idx)
            if self.verbose:
                print(f"[transformer] internal val split: train={len(tr_idx):,} val={len(val_idx):,}")
        else:
            if self.verbose:
                print(f"[transformer] tensorizing valid ({len(valid_df):,} rows)...")
            valid_t = _df_to_tensors(
                valid_df, f,
                include_team_aggregates=self.use_team_aggregates,
                include_time_features=self.use_time_features,
                include_history_features=self.use_history_features,
            )
            if valid_y is not None:
                valid_t["y"] = torch.from_numpy(valid_y.astype(np.float32))

        # Preload the full dataset onto the target device. With ~1.87 M rows in
        # the v3 A_fair config the tensors are about 50 MB total, comfortably
        # within the RTX 3060 Mobile's 5.77 GB. After this point every batch
        # gather is a pure GPU index op.
        train_t = self._move_to_device(train_t, device)
        valid_t = self._move_to_device(valid_t, device)
        if self.verbose and device.type == "cuda":
            mem_mb = sum(v.element_size() * v.numel() for v in train_t.values()) / 1024**2
            print(f"[transformer] train tensors on {device}: {mem_mb:.1f} MB")

        n_train = train_t["y"].shape[0]
        steps_per_epoch = max(1, (n_train + self.batch_size - 1) // self.batch_size)

        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, self.epochs * steps_per_epoch),
        )
        loss_fn = nn.BCEWithLogitsLoss()

        best_val_auc = -1.0
        best_state = None
        bad_epochs = 0

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            self.model.train()
            running = 0.0
            n_seen = 0
            step = 0
            for inputs, yy in self._iter_batches(
                train_t, batch_size=self.batch_size, shuffle=True,
                device=device, seed=self.seed + epoch,
            ):
                step += 1
                logits = self.model(*inputs)
                loss = loss_fn(logits, yy)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if self.grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                opt.step()
                sched.step()
                bs = yy.size(0)
                running += loss.item() * bs
                n_seen += bs
                if self.verbose and (step % self.log_every == 0 or step == steps_per_epoch):
                    avg = running / max(1, n_seen)
                    print(f"[transformer] epoch {epoch} step {step}/{steps_per_epoch} "
                          f"loss={avg:.4f} lr={opt.param_groups[0]['lr']:.5f}")

            val_metrics = self._evaluate_tensors(valid_t, device)
            train_loss = running / max(1, n_seen)
            elapsed = time.time() - t0
            self.history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["logloss"],
                "val_auc": val_metrics["auc"],
                "elapsed_s": elapsed,
            })
            if self.verbose:
                print(f"[transformer] epoch {epoch} done: train_loss={train_loss:.4f} "
                      f"val_loss={val_metrics['logloss']:.4f} val_auc={val_metrics['auc']:.4f} "
                      f"elapsed={elapsed:.1f}s")

            if val_metrics["auc"] > best_val_auc + 1e-4:
                best_val_auc = val_metrics["auc"]
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.early_stop_patience:
                    if self.verbose:
                        print(f"[transformer] early stopping at epoch {epoch} (best val_auc={best_val_auc:.4f})")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    def _evaluate_tensors(self, t: dict[str, torch.Tensor], device: torch.device) -> dict:
        """Evaluate on already-on-device tensors. Used by the training loop's
        per-epoch validation."""
        self.model.eval()
        all_p: list[np.ndarray] = []
        all_y: list[np.ndarray] = []
        with torch.no_grad():
            for inputs, yy in self._iter_batches(
                t, batch_size=self.eval_batch_size, shuffle=False, device=device,
            ):
                logits = self.model(*inputs)
                all_p.append(torch.sigmoid(logits).cpu().numpy())
                all_y.append(yy.cpu().numpy())
        proba = np.concatenate(all_p)
        y = np.concatenate(all_y)
        from sklearn.metrics import log_loss, roc_auc_score
        return {
            "auc": float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
            "logloss": float(log_loss(y, np.clip(proba, 1e-3, 1 - 1e-3))),
            "n": int(len(y)),
        }

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        assert self.model is not None and self.featurizer is not None, "must fit first"
        device = torch.device(self.device)
        self.model.eval()
        t = self._move_to_device(
            _df_to_tensors(
                df, self.featurizer,
                include_team_aggregates=self.use_team_aggregates,
                include_time_features=self.use_time_features,
                include_history_features=self.use_history_features,
            ),
            device,
        )
        out: list[np.ndarray] = []
        with torch.no_grad():
            for inputs, _yy in self._iter_batches(
                t, batch_size=self.eval_batch_size, shuffle=False, device=device,
            ):
                logits = self.model(*inputs)
                out.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(out)


def save_transformer(model: TransformerTeamModel, prefix: Path | str) -> None:
    """Persist a fitted TransformerTeamModel to disk.

    Writes:
        <prefix>.pt          — torch state dict
        <prefix>.meta.json   — vocab + arch hyperparams (so we can reload later)
    """
    p = Path(prefix)
    p.parent.mkdir(parents=True, exist_ok=True)
    assert model.model is not None and model.featurizer is not None
    torch.save(model.model.state_dict(), str(p) + ".pt")
    if model.featurizer.player_history:
        ph = model.featurizer.player_history
        ph_encoded = {
            "player_stats": ph.get("player_stats", {}),
            "player_brawler_stats_keys": [
                [pt, int(bid)] for (pt, bid) in ph.get("player_brawler_stats", {}).keys()
            ],
            "player_brawler_stats_vals": list(ph.get("player_brawler_stats", {}).values()),
        }
    else:
        ph_encoded = None
    meta = {
        "type": "TransformerTeamModel",
        "arch": {
            "d_model": model.d_model,
            "nhead": model.nhead,
            "ff": model.ff,
            "num_layers": model.num_layers,
            "dropout": model.dropout,
            "extra_scalar_dim": model.extra_scalar_dim,
        },
        "training": {
            "epochs": model.epochs,
            "batch_size": model.batch_size,
            "lr": model.lr,
            "weight_decay": model.weight_decay,
            "seed": model.seed,
            "grad_clip": model.grad_clip,
            "history": model.history,
            "use_team_aggregates": bool(model.use_team_aggregates),
            "use_time_features": bool(model.use_time_features),
            "use_history_features": bool(model.use_history_features),
        },
        "featurizer": {
            "brawler_to_idx": {str(k): v for k, v in model.featurizer.brawler_to_idx.items()},
            "mode_to_idx": model.featurizer.mode_to_idx,
            "map_to_idx": model.featurizer.map_to_idx,
            "btype_to_idx": model.featurizer.btype_to_idx,
            "include_team_aggregates": bool(model.featurizer.include_team_aggregates),
            "include_time_features": bool(model.featurizer.include_time_features),
            "include_history_features": bool(model.featurizer.include_history_features),
            "brawler_first_seen": {
                str(k): v for k, v in model.featurizer.brawler_first_seen.items()
            },
            "player_history_encoded": ph_encoded,
        },
    }
    with open(str(p) + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)


def load_transformer(prefix: Path | str) -> TransformerTeamModel:
    """Reload a previously-saved transformer.

    Backwards-compatible: legacy meta.json files (pre-phase-1 / pre-phase-2)
    are missing `extra_scalar_dim`, `use_team_aggregates`, `use_time_features`
    and `brawler_first_seen` fields. We default each to its disabled state so
    old saves load with the head shape and feature pipeline they were trained
    with.

    For models trained with phase-2 we infer `use_time_features` from the saved
    head dim if the explicit flag is absent — `extra_scalar_dim - (23 if
    use_team_aggregates else 0) == 12` means phase-2 was on.
    """
    p = Path(prefix)
    with open(str(p) + ".meta.json") as f:
        meta = json.load(f)
    arch = meta["arch"]
    extra_scalar_dim = int(arch.get("extra_scalar_dim", 0))
    feat_meta = meta["featurizer"]
    train_meta = meta.get("training", {})

    use_team_aggregates = bool(
        train_meta.get("use_team_aggregates", False)
        or feat_meta.get("include_team_aggregates", False)
    )
    use_time_features = bool(
        train_meta.get("use_time_features", False)
        or feat_meta.get("include_time_features", False)
    )
    use_history_features = bool(
        train_meta.get("use_history_features", False)
        or feat_meta.get("include_history_features", False)
    )
    # Self-consistency check: rebuild the expected dim from flags and warn
    # quietly if it disagrees with the saved value (newer saves should always
    # match; some legacy phase-1 saves had the flag missing in `training`).
    if (
        not use_team_aggregates
        and not use_time_features
        and not use_history_features
        and extra_scalar_dim > 0
    ):
        # Pre-phase-2 saves: every nonzero extra_scalar_dim is phase-1.
        use_team_aggregates = True

    first_seen = {
        int(k): v for k, v in feat_meta.get("brawler_first_seen", {}).items()
    }
    ph_encoded = feat_meta.get("player_history_encoded")
    player_history: dict = {"player_stats": {}, "player_brawler_stats": {}}
    if ph_encoded is not None:
        player_history["player_stats"] = ph_encoded.get("player_stats", {})
        keys = ph_encoded.get("player_brawler_stats_keys", [])
        vals = ph_encoded.get("player_brawler_stats_vals", [])
        player_history["player_brawler_stats"] = {
            (str(k[0]), int(k[1])): v for k, v in zip(keys, vals)
        }
    feat = TeamFeaturizer(
        brawler_to_idx={int(k): v for k, v in feat_meta["brawler_to_idx"].items()},
        mode_to_idx=feat_meta["mode_to_idx"],
        map_to_idx=feat_meta["map_to_idx"],
        btype_to_idx=feat_meta["btype_to_idx"],
        include_team_aggregates=use_team_aggregates,
        include_time_features=use_time_features,
        include_history_features=use_history_features,
        brawler_first_seen=first_seen,
        player_history=player_history,
    )
    m = TransformerTeamModel(
        d_model=arch["d_model"], nhead=arch["nhead"], ff=arch["ff"],
        num_layers=arch["num_layers"], dropout=arch["dropout"],
        use_team_aggregates=use_team_aggregates,
        use_time_features=use_time_features,
        use_history_features=use_history_features,
    )
    m.featurizer = feat
    core = _TransformerCore(
        n_brawlers=feat.n_brawlers,
        n_modes=feat.n_modes,
        n_maps=feat.n_maps,
        n_btypes=feat.n_btypes,
        d_model=arch["d_model"], nhead=arch["nhead"], ff=arch["ff"],
        num_layers=arch["num_layers"], dropout=arch["dropout"],
        extra_scalar_dim=extra_scalar_dim,
    )
    state = torch.load(str(p) + ".pt", map_location="cpu")
    core.load_state_dict(state)
    core.eval()
    m.model = core
    return m
