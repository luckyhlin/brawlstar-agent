"""Feature engineering for the team-completion model.

Two flavors:
- `TeamFeaturizer.transform_sparse`: scipy.sparse for logistic regression
- `TeamFeaturizer.transform_dense`:  numpy float for LightGBM (smallish dense),
  with categorical columns indicated for the LGBM API

Both share the same `fit` (which learns vocabulary from training data) so a
single featurizer instance can drive both model families.

Features:
    [team_a multi-hot brawler indicators]   (n_brawlers dims)
    [team_b multi-hot brawler indicators]   (n_brawlers dims)
    [mode one-hot]                          (n_modes dims)
    [map  one-hot]                          (n_maps dims)
    [battle_type one-hot]                   (n_btypes dims)
    [team_a_trophies_mean]                  (1 dim, log-scaled)
    [team_b_trophies_mean]                  (1 dim, log-scaled)
    [trophy_diff]                           (1 dim, log-scaled, A - B)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse


def _index_map(values) -> dict[object, int]:
    """Stable index map: sorted unique values → contiguous ints. NaN-safe."""
    cleaned: set = set()
    for v in values:
        if v is None:
            continue
        if isinstance(v, float) and np.isnan(v):
            continue
        cleaned.add(v)
    return {v: i for i, v in enumerate(sorted(cleaned, key=lambda x: str(x)))}


@dataclass
class TeamFeaturizer:
    brawler_to_idx: dict[int, int] = field(default_factory=dict)
    mode_to_idx: dict[str, int] = field(default_factory=dict)
    map_to_idx: dict[str, int] = field(default_factory=dict)
    btype_to_idx: dict[str, int] = field(default_factory=dict)

    @property
    def n_brawlers(self) -> int:
        return len(self.brawler_to_idx)

    @property
    def n_modes(self) -> int:
        return len(self.mode_to_idx)

    @property
    def n_maps(self) -> int:
        return len(self.map_to_idx)

    @property
    def n_btypes(self) -> int:
        return len(self.btype_to_idx)

    @property
    def n_features(self) -> int:
        # team_a brawlers + team_b brawlers + mode + map + btype + trophies (3)
        return (
            self.n_brawlers * 2
            + self.n_modes
            + self.n_maps
            + self.n_btypes
            + 3
        )

    def feature_names(self) -> list[str]:
        names: list[str] = []
        b_names = sorted(self.brawler_to_idx.keys())
        names.extend(f"a_b{b}" for b in b_names)
        names.extend(f"b_b{b}" for b in b_names)
        names.extend(f"mode_{m}" for m in sorted(self.mode_to_idx.keys())) 
        names.extend(f"map_{m}"  for m in sorted(self.map_to_idx.keys()))
        names.extend(f"bt_{m}"   for m in sorted(self.btype_to_idx.keys()))
        names.extend(["a_trophies_log", "b_trophies_log", "trophy_diff_log"])
        return names

    def fit(self, df: pd.DataFrame) -> "TeamFeaturizer":
        all_brawlers: set[int] = set()
        for t in df["team_a"]:
            all_brawlers.update(int(b) for b in t)
        for t in df["team_b"]:
            all_brawlers.update(int(b) for b in t)
        self.brawler_to_idx = _index_map(all_brawlers)
        # Replace NaN in categorical columns with the literal string "UNKNOWN"
        # so downstream `.get(str(value))` lookups work consistently.
        self.mode_to_idx  = _index_map(df["mode"].fillna("UNKNOWN").astype(str).tolist())
        self.map_to_idx   = _index_map(df["map"].fillna("UNKNOWN").astype(str).tolist())
        self.btype_to_idx = _index_map(df["battle_type"].fillna("UNKNOWN").astype(str).tolist())
        return self

    # -- numeric helpers --

    @staticmethod
    def _log1p(x: float | np.ndarray) -> np.ndarray:
        """Log1p with NaN→0; trophies are usually positive, but be safe."""
        x = np.asarray(x, dtype=float)
        x = np.where(np.isnan(x), 0.0, x)
        return np.log1p(np.maximum(x, 0.0))

    def transform_sparse(self, df: pd.DataFrame) -> sparse.csr_matrix:
        n = len(df)
        cols_total = self.n_features

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []

        team_a_offset = 0
        team_b_offset = self.n_brawlers
        mode_offset   = self.n_brawlers * 2
        map_offset    = mode_offset + self.n_modes
        btype_offset  = map_offset + self.n_maps
        num_offset    = btype_offset + self.n_btypes  # numeric features start here

        a_trophies = self._log1p(df["team_a_trophies_mean"].values)
        b_trophies = self._log1p(df["team_b_trophies_mean"].values)

        modes_s  = df["mode"].fillna("UNKNOWN").astype(str).values
        maps_s   = df["map"].fillna("UNKNOWN").astype(str).values
        btypes_s = df["battle_type"].fillna("UNKNOWN").astype(str).values

        for i, (ta, tb, mode, mp, bt) in enumerate(
            zip(df["team_a"].values, df["team_b"].values,
                modes_s, maps_s, btypes_s)
        ):
            for b in ta:
                idx = self.brawler_to_idx.get(int(b))
                if idx is not None:
                    rows.append(i); cols.append(team_a_offset + idx); data.append(1.0)
            for b in tb:
                idx = self.brawler_to_idx.get(int(b))
                if idx is not None:
                    rows.append(i); cols.append(team_b_offset + idx); data.append(1.0)
            mi = self.mode_to_idx.get(mode)
            if mi is not None:
                rows.append(i); cols.append(mode_offset + mi); data.append(1.0)
            mpi = self.map_to_idx.get(mp)
            if mpi is not None:
                rows.append(i); cols.append(map_offset + mpi); data.append(1.0)
            bti = self.btype_to_idx.get(bt)
            if bti is not None:
                rows.append(i); cols.append(btype_offset + bti); data.append(1.0)
            # Trophy features
            rows.append(i); cols.append(num_offset + 0); data.append(float(a_trophies[i]))
            rows.append(i); cols.append(num_offset + 1); data.append(float(b_trophies[i]))
            rows.append(i); cols.append(num_offset + 2); data.append(float(a_trophies[i] - b_trophies[i]))

        return sparse.csr_matrix((data, (rows, cols)), shape=(n, cols_total), dtype=np.float32)

    def transform_dense(self, df: pd.DataFrame) -> tuple[np.ndarray, list[int]]:
        """Dense feature matrix + indices of categorical columns (for LightGBM).

        Categorical columns (single int per row, -1 for unknown):
            mode_idx, map_idx, btype_idx
        Plus brawler multi-hots and trophy floats stay numeric.
        """
        n = len(df)
        # Brawler multi-hots
        ax = np.zeros((n, self.n_brawlers), dtype=np.float32)
        bx = np.zeros((n, self.n_brawlers), dtype=np.float32)
        for i, ta in enumerate(df["team_a"].values):
            for b in ta:
                j = self.brawler_to_idx.get(int(b))
                if j is not None:
                    ax[i, j] = 1.0
        for i, tb in enumerate(df["team_b"].values):
            for b in tb:
                j = self.brawler_to_idx.get(int(b))
                if j is not None:
                    bx[i, j] = 1.0

        mode_idx  = np.array([self.mode_to_idx.get(str(m), -1)
                              for m in df["mode"].fillna("UNKNOWN").astype(str).values],  dtype=np.int32)
        map_idx   = np.array([self.map_to_idx.get(str(m), -1)
                              for m in df["map"].fillna("UNKNOWN").astype(str).values],   dtype=np.int32)
        btype_idx = np.array([self.btype_to_idx.get(str(m), -1)
                              for m in df["battle_type"].fillna("UNKNOWN").astype(str).values], dtype=np.int32)

        a_trophies = self._log1p(df["team_a_trophies_mean"].values)
        b_trophies = self._log1p(df["team_b_trophies_mean"].values)
        trophy_diff = a_trophies - b_trophies

        # Order: brawler_a (n_brawlers) | brawler_b (n_brawlers) | cat_mode | cat_map | cat_btype | a_trophy | b_trophy | trophy_diff
        X = np.hstack([
            ax,
            bx,
            mode_idx.reshape(-1, 1).astype(np.float32),
            map_idx.reshape(-1, 1).astype(np.float32),
            btype_idx.reshape(-1, 1).astype(np.float32),
            a_trophies.reshape(-1, 1).astype(np.float32),
            b_trophies.reshape(-1, 1).astype(np.float32),
            trophy_diff.reshape(-1, 1).astype(np.float32),
        ])

        cat_cols = [
            self.n_brawlers * 2,        # mode
            self.n_brawlers * 2 + 1,    # map
            self.n_brawlers * 2 + 2,    # battle_type
        ]
        return X, cat_cols
