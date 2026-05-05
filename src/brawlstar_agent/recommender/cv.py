"""Temporal cross-validation harness.

Designed for the meta-drifting domain: we want to know "if we train on weeks
[t-W, t), can we predict weeks [t, t+H)?" — and we want this to be averaged
across multiple t values to be robust to a single bad split.

Right now (May 2026) we only have ~2 days of clean data, so the harness
exercises ONE temporal split. Re-running this monthly produces a meaningful
"transferability" curve over time. Same code, no edits.

Usage:
    folds = make_temporal_folds(df, train_days=1, test_days=1, step_hours=12)
    for tr, te in folds:
        ...
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterator

import numpy as np
import pandas as pd


def _parse_iso(ts: str) -> datetime:
    """Parse ISO timestamp; accepts trailing 'Z' or '+00:00'."""
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts)


def make_temporal_folds(
    df: pd.DataFrame,
    train_days: float = 1.0,
    test_days: float = 0.5,
    step_hours: float = 6.0,
    min_train_rows: int = 5_000,
    min_test_rows: int = 1_000,
    gap_hours: float = 0.0,
) -> list[tuple[pd.DataFrame, pd.DataFrame, dict]]:
    """Generate sliding (train, test) windows over `battle_time_iso`.

    For each fold:
        train = battles with time in [t - train_days, t - gap_hours)
        test  = battles with time in [t, t + test_days)

    `t` slides across the data range in `step_hours` increments, starting at
    `min(time) + train_days + gap_hours` and stopping when test window can't
    fit anymore.

    Folds with fewer than `min_train_rows` or `min_test_rows` rows are skipped.
    """
    if df.empty:
        return []

    times = pd.to_datetime(df["battle_time_iso"], utc=True)
    df = df.assign(_t=times).sort_values("_t").reset_index(drop=True)

    t_min = df["_t"].min().to_pydatetime().astimezone(timezone.utc)
    t_max = df["_t"].max().to_pydatetime().astimezone(timezone.utc)

    train_delta = timedelta(days=train_days)
    test_delta = timedelta(days=test_days)
    gap_delta = timedelta(hours=gap_hours)
    step = timedelta(hours=step_hours)

    t = t_min + train_delta + gap_delta
    folds: list[tuple[pd.DataFrame, pd.DataFrame, dict]] = []
    while t + test_delta <= t_max + timedelta(seconds=1):
        train_lo = t - train_delta - gap_delta
        train_hi = t - gap_delta
        test_lo = t
        test_hi = t + test_delta

        tr_mask = (df["_t"] >= train_lo) & (df["_t"] < train_hi)
        te_mask = (df["_t"] >= test_lo)  & (df["_t"] < test_hi)
        tr = df.loc[tr_mask].drop(columns=["_t"])
        te = df.loc[te_mask].drop(columns=["_t"])

        if len(tr) >= min_train_rows and len(te) >= min_test_rows:
            folds.append((tr, te, {
                "train_lo": train_lo.isoformat(),
                "train_hi": train_hi.isoformat(),
                "test_lo": test_lo.isoformat(),
                "test_hi": test_hi.isoformat(),
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
            }))
        t += step

    return folds


def evaluate_models_on_folds(
    folds: list[tuple[pd.DataFrame, pd.DataFrame, dict]],
    model_factories: dict[str, callable],
    label_col: str = "team_a_wins",
) -> pd.DataFrame:
    """Train each `model_factory` on each fold's train, evaluate on test.

    `model_factories` is {name: zero-arg callable returning a fresh model
    instance with .fit(df) and .predict_proba(df)}.

    Returns a long-form DataFrame with one row per (fold, model).
    """
    from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, brier_score_loss
    rows = []
    for fold_idx, (tr, te, meta) in enumerate(folds):
        for name, factory in model_factories.items():
            m = factory()
            m.fit(tr)
            proba = m.predict_proba(te)
            y = te[label_col].values
            proba_clip = np.clip(proba, 1e-3, 1 - 1e-3)
            rows.append({
                "fold": fold_idx,
                "train_lo": meta["train_lo"],
                "train_hi": meta["train_hi"],
                "test_lo": meta["test_lo"],
                "test_hi": meta["test_hi"],
                "n_train": meta["n_train"],
                "n_test": meta["n_test"],
                "model": name,
                "auc":   float(roc_auc_score(y, proba)),
                "logloss": float(log_loss(y, proba_clip)),
                "accuracy": float(accuracy_score(y, (proba > 0.5).astype(int))),
                "brier": float(brier_score_loss(y, proba_clip)),
            })
    return pd.DataFrame(rows)
