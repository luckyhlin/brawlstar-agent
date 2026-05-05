"""Baseline brawler-strength estimators.

Three escalating context levels:

1. GlobalWilsonBaseline       — P(win | brawler) ignoring all context
2. ModeWilsonBaseline         — P(win | brawler, mode)
3. ModeMapWilsonBaseline      — P(win | brawler, mode, map) with Bayesian
                                shrinkage toward mode-level rate when data is sparse

Each baseline answers "rank candidate brawlers" and is also usable as a *team-strength*
estimator (mean over the team's brawlers) — that's how we score a team-completion model
against a no-context baseline in `team_model.py::evaluate`.

All baselines are FIT on a training DataFrame and then PREDICT on a test DataFrame.
That keeps them honest under temporal CV.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..models import wilson_interval


def _wins_total(df: pd.DataFrame, brawler_col: str = "team_a") -> pd.DataFrame:
    """Expand team rows to per-brawler rows and aggregate wins/total per brawler.

    Each row in `df` has team_a (3 brawlers) and team_a_wins (0/1). We explode
    so each brawler in team_a contributes one observation.
    """
    rows = []
    for team, win in zip(df[brawler_col].values, df["team_a_wins"].values):
        for b in team:
            rows.append((int(b), int(win)))
    if not rows:
        return pd.DataFrame(columns=["brawler_id", "wins", "total"])
    out = pd.DataFrame(rows, columns=["brawler_id", "win"])
    agg = out.groupby("brawler_id").agg(wins=("win", "sum"), total=("win", "size")).reset_index()
    return agg


@dataclass
class GlobalWilsonBaseline:
    """Per-brawler Wilson-CI win rate, ignoring mode/map/anything else."""
    rates: dict[int, float] | None = None
    n: dict[int, int] | None = None
    fallback: float = 0.5

    def fit(self, train: pd.DataFrame) -> "GlobalWilsonBaseline":
        agg = _wins_total(train)
        self.rates = {}
        self.n = {}
        for r in agg.itertuples(index=False):
            _, center, _ = wilson_interval(int(r.wins), int(r.total))
            self.rates[int(r.brawler_id)] = center
            self.n[int(r.brawler_id)] = int(r.total)
        # global mean for unknown brawlers
        if agg["total"].sum() > 0:
            self.fallback = float(agg["wins"].sum() / agg["total"].sum())
        return self

    def score_brawler(self, brawler_id: int, **_) -> float:
        return self.rates.get(int(brawler_id), self.fallback) if self.rates else self.fallback

    def score_team(self, team: tuple[int, ...], **_) -> float:
        """Mean per-brawler Wilson rate. Used as a P(team wins) baseline."""
        if not team:
            return self.fallback
        return float(np.mean([self.score_brawler(b) for b in team]))

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return P(team_a wins) per row of df. Pure heuristic:
        sigmoid-style relative strength of team_a vs team_b given per-brawler scores.
        """
        scores_a = df["team_a"].apply(lambda t: self.score_team(t)).values
        scores_b = df["team_b"].apply(lambda t: self.score_team(t)).values
        # Map raw rates ([~0.3, ~0.7]) to a calibrated win-prob via diff.
        # Larger A-strength minus B-strength → higher P(A wins).
        diff = scores_a - scores_b
        return 1.0 / (1.0 + np.exp(-6.0 * diff))


@dataclass
class ModeWilsonBaseline:
    """Per-(mode, brawler) Wilson rate. Falls back to global if unseen."""
    rates: dict[tuple[str, int], float] | None = None
    mode_means: dict[str, float] | None = None
    fallback: float = 0.5

    def fit(self, train: pd.DataFrame) -> "ModeWilsonBaseline":
        self.rates = {}
        self.mode_means = {}
        for mode, sub in train.groupby("mode"):
            agg = _wins_total(sub)
            for r in agg.itertuples(index=False):
                _, center, _ = wilson_interval(int(r.wins), int(r.total))
                self.rates[(str(mode), int(r.brawler_id))] = center
            if agg["total"].sum() > 0:
                self.mode_means[str(mode)] = float(agg["wins"].sum() / agg["total"].sum())
        if train["team_a_wins"].size > 0:
            self.fallback = float(train["team_a_wins"].mean())
        return self

    def score_brawler(self, brawler_id: int, mode: str | None = None, **_) -> float:
        if mode is not None and self.rates is not None:
            v = self.rates.get((str(mode), int(brawler_id)))
            if v is not None:
                return v
            return self.mode_means.get(str(mode), self.fallback) if self.mode_means else self.fallback
        return self.fallback

    def score_team(self, team: tuple[int, ...], mode: str | None = None, **_) -> float:
        if not team:
            return self.fallback
        return float(np.mean([self.score_brawler(b, mode=mode) for b in team]))

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        scores_a = np.array([
            self.score_team(t, mode=m)
            for t, m in zip(df["team_a"].values, df["mode"].values)
        ])
        scores_b = np.array([
            self.score_team(t, mode=m)
            for t, m in zip(df["team_b"].values, df["mode"].values)
        ])
        diff = scores_a - scores_b
        return 1.0 / (1.0 + np.exp(-6.0 * diff))


@dataclass
class ModeMapWilsonBaseline:
    """Per-(mode, map, brawler) win rate with Bayesian shrinkage to mode-level rate.

    Shrinkage: posterior_rate = (wins + alpha * mode_rate * k) / (total + alpha * k)
    where alpha is the strength of the prior. For very sparse cells this falls back
    cleanly to the mode-level rate.
    """
    map_rates: dict[tuple[str, str, int], float] | None = None
    mode_rates: dict[tuple[str, int], float] | None = None
    mode_means: dict[str, float] | None = None
    fallback: float = 0.5
    alpha: float = 30.0  # prior strength in pseudo-observations

    def fit(self, train: pd.DataFrame) -> "ModeMapWilsonBaseline":
        # Mode-level Wilson rates (used as prior)
        mode_baseline = ModeWilsonBaseline().fit(train)
        self.mode_rates = {k: v for k, v in (mode_baseline.rates or {}).items()}
        self.mode_means = mode_baseline.mode_means or {}
        self.fallback = mode_baseline.fallback

        # Per (mode, map, brawler)
        self.map_rates = {}
        for (mode, mp), sub in train.groupby(["mode", "map"]):
            agg = _wins_total(sub)
            for r in agg.itertuples(index=False):
                bid = int(r.brawler_id)
                wins = int(r.wins)
                total = int(r.total)
                prior_rate = self.mode_rates.get((str(mode), bid), self.fallback)
                shrunk = (wins + self.alpha * prior_rate) / (total + self.alpha)
                self.map_rates[(str(mode), str(mp), bid)] = float(shrunk)
        return self

    def score_brawler(
        self,
        brawler_id: int,
        mode: str | None = None,
        map: str | None = None,
        **_,
    ) -> float:
        if mode is not None and map is not None and self.map_rates is not None:
            v = self.map_rates.get((str(mode), str(map), int(brawler_id)))
            if v is not None:
                return v
        if mode is not None and self.mode_rates is not None:
            v = self.mode_rates.get((str(mode), int(brawler_id)))
            if v is not None:
                return v
        if mode is not None and self.mode_means:
            return self.mode_means.get(str(mode), self.fallback)
        return self.fallback

    def score_team(
        self,
        team: tuple[int, ...],
        mode: str | None = None,
        map: str | None = None,
        **_,
    ) -> float:
        if not team:
            return self.fallback
        return float(np.mean([self.score_brawler(b, mode=mode, map=map) for b in team]))

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        scores_a = np.array([
            self.score_team(t, mode=m, map=mp)
            for t, m, mp in zip(df["team_a"].values, df["mode"].values, df["map"].values)
        ])
        scores_b = np.array([
            self.score_team(t, mode=m, map=mp)
            for t, m, mp in zip(df["team_b"].values, df["mode"].values, df["map"].values)
        ])
        diff = scores_a - scores_b
        return 1.0 / (1.0 + np.exp(-6.0 * diff))


def rank_brawlers(
    baseline,
    candidates: list[int],
    *,
    mode: str | None = None,
    map: str | None = None,
    top_k: int | None = None,
) -> list[tuple[int, float]]:
    """Apply a fitted baseline to rank candidate brawlers by P(win) they bring.

    Returns list of (brawler_id, score) sorted descending.
    """
    scored = [
        (int(b), float(baseline.score_brawler(b, mode=mode, map=map)))
        for b in candidates
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    if top_k is not None:
        scored = scored[:top_k]
    return scored
