"""Tiered evaluation slices for the recommender's stable test set.

Brawl Stars `battle_type` semantics in our DB (see also `techContext.md`):

  - `'ranked'`     = the in-game **Unranked** trophy ladder. No draft, no tier
                     system; `brawler_trophies` is the cumulative trophy count
                     (0-4951).
  - `'soloRanked'` = the in-game **Ranked** queue. Has tier system
                     Bronze→...→Pro and a strict 1-2-2-1 ban/pick draft from
                     **Mythic (>= 13)** upward. `brawler_trophies` is overloaded
                     to mean *the player's tier number* (1-22) in this
                     battle_type.

So the model's "all-test" AUC mixes three quite different game shapes
(no-draft trophy ladder, low-tier soloRanked with simultaneous pick, and
high-tier soloRanked with strict draft). Splitting them out with these slicers
gives much more honest reads.

Slices produced by `make_slice_masks(test_df)` (in priority order):

  - 'all'                   : full test set (backwards-compat with reports
                              that didn't slice)
  - 'ranked'                : trophy-ladder battles only
  - 'soloRanked'            : ranked queue (any tier)
  - 'soloRanked_diamondplus': both teams Diamond+ (>= 10) — has *some* draft
                              (simultaneous pick at Diamond)
  - 'soloRanked_mythicplus' : both teams Mythic+ (>= 13) — strict 1-2-2-1
                              ban/pick draft. **The "competitive" subset.**
  - 'soloRanked_legendaryplus': both teams Legendary+ (>= 16) — top-tier slice;
                              smaller and noisier but maximally clean draft.

The "both teams" qualifier: a slice keeps a row only if EVERY one of the 6
brawler_trophies values (3 on each team) is >= the threshold. Matchmaking
groups players by tier so this is how the in-game lobby actually composes.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

# `brawler_trophies` thresholds in `soloRanked` (the field is the tier number
# 1-22 in this battle_type, NOT the cumulative trophy count).
SOLO_RANK_DIAMOND = 10
SOLO_RANK_MYTHIC = 13
SOLO_RANK_LEGENDARY = 16
SOLO_RANK_MASTERS = 19
SOLO_RANK_PRO = 20


def _team_min_tiers(test_df: pd.DataFrame) -> np.ndarray | None:
    """Return per-row min(brawler_trophies across all 6 slots) as int32, or
    None if the per-brawler tuple columns are missing (legacy code path).
    """
    if "team_a_trophies" not in test_df.columns:
        return None
    a_t = np.asarray(test_df["team_a_trophies"].tolist(), dtype=np.int32)
    b_t = np.asarray(test_df["team_b_trophies"].tolist(), dtype=np.int32)
    return np.minimum(a_t.min(axis=1), b_t.min(axis=1))


def make_slice_masks(test_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Boolean masks (one per slice) over a *positionally-indexed* test_df.

    Caller is responsible for `test_df = test_df.reset_index(drop=True)`
    BEFORE calling this — otherwise predictions and masks won't align.
    """
    n = len(test_df)
    if n == 0:
        return {"all": np.zeros(0, dtype=bool)}

    is_ranked = (test_df["battle_type"] == "ranked").values
    is_solo = (test_df["battle_type"] == "soloRanked").values

    masks: dict[str, np.ndarray] = {
        "all": np.ones(n, dtype=bool),
        "ranked": is_ranked,
        "soloRanked": is_solo,
    }

    min_tier = _team_min_tiers(test_df)
    if min_tier is not None:
        masks["soloRanked_diamondplus"] = is_solo & (min_tier >= SOLO_RANK_DIAMOND)
        masks["soloRanked_mythicplus"] = is_solo & (min_tier >= SOLO_RANK_MYTHIC)
        masks["soloRanked_legendaryplus"] = is_solo & (min_tier >= SOLO_RANK_LEGENDARY)
    return masks


def evaluate_slices(
    model,
    test_df: pd.DataFrame,
    *,
    label_col: str = "team_a_wins",
    min_n_per_slice: int = 200,
    proba: np.ndarray | None = None,
) -> dict:
    """Compute per-slice binary metrics for `model` on `test_df`.

    Runs `model.predict_proba(test_df)` ONCE on the full test set and slices
    the resulting probability vector with boolean masks; far cheaper than
    re-predicting per slice (especially for the transformer, which re-tensorizes
    the full DataFrame inside `predict_proba`).

    Args:
        model: any model with `predict_proba(df) -> np.ndarray`.
        test_df: positionally-indexed test DataFrame
                 (call `.reset_index(drop=True)` first).
        proba: optional pre-computed probabilities. If provided, skip the
               `model.predict_proba` call. Useful when slicing several
               metrics from the same prediction pass.

    Returns dict keyed by slice name with sub-dicts of {auc, logloss,
    accuracy, brier, n}. Slices with too few rows or single-class labels
    return {n, skipped: True}.
    """
    from sklearn.metrics import (  # imported lazily to avoid module load cost
        accuracy_score,
        brier_score_loss,
        log_loss,
        roc_auc_score,
    )

    if len(test_df) == 0:
        raise ValueError("empty test_df")

    if proba is None:
        proba = model.predict_proba(test_df)
    proba = np.asarray(proba)
    y = test_df[label_col].values

    masks = make_slice_masks(test_df)
    out: dict[str, dict] = {}
    proba_clip = np.clip(proba, 1e-3, 1.0 - 1e-3)
    for slice_name, mask in masks.items():
        n = int(mask.sum())
        if n < min_n_per_slice:
            out[slice_name] = {"n": n, "skipped": True, "reason": "too_few_rows"}
            continue
        s_y = y[mask]
        if len(np.unique(s_y)) < 2:
            out[slice_name] = {"n": n, "skipped": True, "reason": "single_class"}
            continue
        s_proba = proba[mask]
        s_proba_clip = proba_clip[mask]
        out[slice_name] = {
            "n": n,
            "auc": float(roc_auc_score(s_y, s_proba)),
            "logloss": float(log_loss(s_y, s_proba_clip)),
            "accuracy": float(accuracy_score(s_y, (s_proba > 0.5).astype(int))),
            "brier": float(brier_score_loss(s_y, s_proba_clip)),
            "label_rate": float(s_y.mean()),
        }
    return out


def format_slice_table(slices: Mapping[str, Mapping]) -> str:
    """Pretty-print a single-model slice metrics dict for terminal output."""
    lines = [
        f"{'slice':28s} {'n':>10s} {'AUC':>7s} {'logloss':>8s} {'acc':>7s} {'brier':>7s}",
    ]
    for name, m in slices.items():
        if m.get("skipped"):
            lines.append(f"{name:28s} {m['n']:>10,d}  (skipped: {m.get('reason', 'n/a')})")
        else:
            lines.append(
                f"{name:28s} {m['n']:>10,d}  {m['auc']:.4f}  {m['logloss']:.4f}  "
                f"{m['accuracy']:.4f}  {m['brier']:.4f}"
            )
    return "\n".join(lines)
