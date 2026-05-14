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

Optional v3.1 add-ons (each guarded by an opt-in flag on `TeamFeaturizer` and
piped through both LGBM and the v3 transformer's `extra_scalar`):

- Phase 1 (`include_team_aggregates=True`): 23 numeric columns of per-team
  aggregates over the per-brawler trophy/power tuples. See
  `compute_team_aggregates` + `TEAM_AGGREGATE_NAMES`.
- Phase 2 (`include_time_features=True`): 12 numeric columns capturing
  cyclical time-of-day / day-of-week and per-team `days_since_release`
  aggregates. The release lookup (`brawler_first_seen`) is fit on training
  data only to avoid leakage and is round-tripped through save/load. See
  `compute_phase2_features` + `PHASE2_NAMES`.
- Phase 4 (`include_history_features=True`): 20 numeric columns derived from
  per-player stats (n_games, overall_wr, per-(player, brawler) wr and count,
  per-player main brawler). The lookup is fit on training data only and
  frozen onto the featurizer; same-window self-leakage is small (~1/n_games
  per row) and accepted for v1 simplicity. Requires `team_a/b_player_tags`
  on the input DataFrame (added by `dataset.load_clean_battles`). See
  `compute_player_history` + `compute_phase4_features` + `PHASE4_NAMES`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse


# Phase 1: per-team aggregates over trophy/power tuples.
# Order matches the columns produced by compute_team_aggregates.
TEAM_AGGREGATE_NAMES: list[str] = [
    "a_t_min_log", "a_t_max_log", "a_t_std_log",
    "b_t_min_log", "b_t_max_log", "b_t_std_log",
    "a_p_mean", "a_p_min", "a_p_max", "a_p_std",
    "b_p_mean", "b_p_min", "b_p_max", "b_p_std",
    "a_n_max_p", "a_n_low_p",
    "b_n_max_p", "b_n_low_p",
    "t_min_diff_log", "t_max_diff_log",
    "p_mean_diff", "p_min_diff", "n_max_p_diff",
]
TEAM_AGGREGATE_DIM = len(TEAM_AGGREGATE_NAMES)  # 23


def compute_team_aggregates(df: pd.DataFrame) -> np.ndarray:
    """Phase 1 per-team aggregates derived from `team_a/b_trophies` and
    `team_a/b_powers` tuples (already on every clean row from
    `dataset.load_clean_battles`).

    Returns float32 array of shape (n, TEAM_AGGREGATE_DIM=23).

    Trophy stats are log1p-scaled to match the existing scalar features.
    Power stats are normalized to [0, 1] (divide by 11). Count features
    (`n_max_p`, `n_low_p`) are normalized to [0, 1] (divide by team size 3).
    Diff features are A − B on already-scaled inputs.

    If the per-brawler tuple columns are missing (e.g., legacy upstream code
    path), returns zeros — this gracefully degrades phase-1-aware models on
    legacy data without crashing.
    """
    n = len(df)
    if n == 0:
        return np.zeros((0, TEAM_AGGREGATE_DIM), dtype=np.float32)
    if "team_a_trophies" not in df.columns:
        return np.zeros((n, TEAM_AGGREGATE_DIM), dtype=np.float32)

    a_t = np.asarray(df["team_a_trophies"].tolist(), dtype=np.float32)
    b_t = np.asarray(df["team_b_trophies"].tolist(), dtype=np.float32)
    a_p = np.asarray(df["team_a_powers"].tolist(), dtype=np.float32)
    b_p = np.asarray(df["team_b_powers"].tolist(), dtype=np.float32)

    a_t_min = np.log1p(np.maximum(a_t.min(axis=1), 0.0))
    a_t_max = np.log1p(np.maximum(a_t.max(axis=1), 0.0))
    a_t_std = np.log1p(np.maximum(a_t.std(axis=1), 0.0))
    b_t_min = np.log1p(np.maximum(b_t.min(axis=1), 0.0))
    b_t_max = np.log1p(np.maximum(b_t.max(axis=1), 0.0))
    b_t_std = np.log1p(np.maximum(b_t.std(axis=1), 0.0))

    a_p_mean = a_p.mean(axis=1) / 11.0
    a_p_min = a_p.min(axis=1) / 11.0
    a_p_max = a_p.max(axis=1) / 11.0
    a_p_std = a_p.std(axis=1) / 11.0
    b_p_mean = b_p.mean(axis=1) / 11.0
    b_p_min = b_p.min(axis=1) / 11.0
    b_p_max = b_p.max(axis=1) / 11.0
    b_p_std = b_p.std(axis=1) / 11.0

    a_n_max_p = (a_p == 11).sum(axis=1).astype(np.float32) / 3.0
    a_n_low_p = (a_p < 8).sum(axis=1).astype(np.float32) / 3.0
    b_n_max_p = (b_p == 11).sum(axis=1).astype(np.float32) / 3.0
    b_n_low_p = (b_p < 8).sum(axis=1).astype(np.float32) / 3.0

    t_min_diff = a_t_min - b_t_min
    t_max_diff = a_t_max - b_t_max
    p_mean_diff = a_p_mean - b_p_mean
    p_min_diff = a_p_min - b_p_min
    n_max_p_diff = a_n_max_p - b_n_max_p

    out = np.stack([
        a_t_min, a_t_max, a_t_std,
        b_t_min, b_t_max, b_t_std,
        a_p_mean, a_p_min, a_p_max, a_p_std,
        b_p_mean, b_p_min, b_p_max, b_p_std,
        a_n_max_p, a_n_low_p,
        b_n_max_p, b_n_low_p,
        t_min_diff, t_max_diff,
        p_mean_diff, p_min_diff, n_max_p_diff,
    ], axis=1).astype(np.float32)

    assert out.shape == (n, TEAM_AGGREGATE_DIM), f"shape {out.shape}"
    return out


# ---------------------------------------------------------------------------
# Phase 2 — time features + per-brawler `days_since_release` aggregates.
# ---------------------------------------------------------------------------

PHASE2_NAMES: list[str] = [
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "a_dsr_min_log", "a_dsr_mean_log",
    "b_dsr_min_log", "b_dsr_mean_log",
    "a_n_new_brawlers", "b_n_new_brawlers",
    "dsr_min_diff_log", "n_new_diff",
]
PHASE2_DIM = len(PHASE2_NAMES)  # 12

# Brawlers with `days_since_release < NEW_BRAWLER_DAYS` count as "brand new"
# for the `n_new_brawlers` count. Two weeks captures the documented DAMIAN
# release-meta inflation window.
NEW_BRAWLER_DAYS = 14.0


def compute_brawler_first_seen(df: pd.DataFrame) -> dict[int, str]:
    """For each brawler_id appearing in `df`, the earliest battle_time_iso it
    appears in (in either team_a or team_b).

    Built from training data only and frozen onto the featurizer (see
    `TeamFeaturizer.brawler_first_seen`), so the lookup at inference time is
    identical to what we used at fit time. The actual game-release date for
    a brawler that existed before our DB lookback is approximated as the first
    time we observed it post-`CLEAN_CUTOFF_ISO` — that approximation is fine
    for capturing the "released within our window" inflation effect.

    Returns: `{brawler_id: ISO_string}`. ISO strings, not pd.Timestamps, so
    the lookup serializes cleanly to JSON.
    """
    if df.empty:
        return {}
    a = df[["team_a", "battle_time_iso"]].rename(columns={"team_a": "team"})
    b = df[["team_b", "battle_time_iso"]].rename(columns={"team_b": "team"})
    flat = pd.concat([a, b], ignore_index=True)
    flat = flat.explode("team").rename(columns={"team": "brawler_id"})
    flat["brawler_id"] = flat["brawler_id"].astype(int)
    grouped = flat.groupby("brawler_id")["battle_time_iso"].min()
    return {int(k): str(v) for k, v in grouped.items()}


def _first_seen_to_array(
    first_seen: dict[int, str],
    max_bid: int,
    fallback: np.datetime64,
) -> np.ndarray:
    """Build a (max_bid + 1,) array of first-seen timestamps indexed by bid.
    Brawlers absent from the lookup default to `fallback` (typically the
    battle's own time -> days_since_release == 0)."""
    arr = np.full(max_bid + 1, fallback, dtype="datetime64[s]")
    for bid, fs_iso in first_seen.items():
        try:
            arr[int(bid)] = np.datetime64(pd.Timestamp(fs_iso).floor("s").tz_localize(None))
        except (ValueError, TypeError, OverflowError):
            continue
    return arr


def compute_phase2_features(
    df: pd.DataFrame,
    brawler_first_seen: dict[int, str],
) -> np.ndarray:
    """12-column phase-2 feature matrix, see `PHASE2_NAMES` for the column order.

    Cyclical encoding for hour and dow (sin/cos pair handles wrap). Time-since-
    release is `log1p(days_since_release)` so the scale matches the existing
    log1p trophy features in the head. `n_new_brawlers` counts brawlers with
    `days_since_release < NEW_BRAWLER_DAYS` per side, normalised by team size.

    Robust to: missing/unknown brawler ids (default first-seen = battle time
    -> dsr = 0), missing per-brawler trophy tuples (degraded mode returns
    zeros), and rows whose battle_time_iso is older than the brawler's
    first-seen entry (clamped to >= 0 days).
    """
    n = len(df)
    if n == 0:
        return np.zeros((0, PHASE2_DIM), dtype=np.float32)

    # Battle timestamps — handle pd-friendly ISO strings.
    battle_dt = pd.to_datetime(df["battle_time_iso"], utc=True, errors="coerce")
    battle_dt_naive = battle_dt.dt.tz_convert(None).to_numpy(dtype="datetime64[s]")
    battle_pd = battle_dt.dt.tz_convert(None)

    hour = battle_pd.dt.hour.to_numpy(dtype=np.float32)
    dow = battle_pd.dt.dayofweek.to_numpy(dtype=np.float32)
    hour_sin = np.sin(2.0 * np.pi * hour / 24.0).astype(np.float32)
    hour_cos = np.cos(2.0 * np.pi * hour / 24.0).astype(np.float32)
    dow_sin = np.sin(2.0 * np.pi * dow / 7.0).astype(np.float32)
    dow_cos = np.cos(2.0 * np.pi * dow / 7.0).astype(np.float32)

    if "team_a" not in df.columns or len(brawler_first_seen) == 0:
        # Degraded mode: phase-2 still emits the 4 time scalars + zeros for dsr.
        return np.concatenate([
            np.stack([hour_sin, hour_cos, dow_sin, dow_cos], axis=1),
            np.zeros((n, PHASE2_DIM - 4), dtype=np.float32),
        ], axis=1).astype(np.float32)

    a_b = np.asarray(df["team_a"].tolist(), dtype=np.int64)
    b_b = np.asarray(df["team_b"].tolist(), dtype=np.int64)
    max_bid = int(max(a_b.max(initial=0), b_b.max(initial=0), max(brawler_first_seen.keys())))

    # Default fallback = each battle's own time => dsr = 0 (we know nothing
    # earlier about that brawler, so treat it as freshly-released).
    fallback = battle_dt_naive.min() if n > 0 else np.datetime64("2026-01-01")
    fs_arr = _first_seen_to_array(brawler_first_seen, max_bid=max_bid, fallback=fallback)

    a_fs = fs_arr[a_b]                                  # (n, 3) datetime64[s]
    b_fs = fs_arr[b_b]
    bt = battle_dt_naive[:, None]                        # (n, 1)

    a_dsr_sec = (bt - a_fs).astype("timedelta64[s]").astype(np.float64)
    b_dsr_sec = (bt - b_fs).astype("timedelta64[s]").astype(np.float64)
    a_dsr_days = np.maximum(a_dsr_sec / 86400.0, 0.0).astype(np.float32)
    b_dsr_days = np.maximum(b_dsr_sec / 86400.0, 0.0).astype(np.float32)

    a_dsr_min_log = np.log1p(a_dsr_days.min(axis=1))
    a_dsr_mean_log = np.log1p(a_dsr_days.mean(axis=1))
    b_dsr_min_log = np.log1p(b_dsr_days.min(axis=1))
    b_dsr_mean_log = np.log1p(b_dsr_days.mean(axis=1))

    a_n_new = ((a_dsr_days < NEW_BRAWLER_DAYS).sum(axis=1) / 3.0).astype(np.float32)
    b_n_new = ((b_dsr_days < NEW_BRAWLER_DAYS).sum(axis=1) / 3.0).astype(np.float32)

    dsr_min_diff = a_dsr_min_log - b_dsr_min_log
    n_new_diff = a_n_new - b_n_new

    out = np.stack([
        hour_sin, hour_cos,
        dow_sin, dow_cos,
        a_dsr_min_log, a_dsr_mean_log,
        b_dsr_min_log, b_dsr_mean_log,
        a_n_new, b_n_new,
        dsr_min_diff, n_new_diff,
    ], axis=1).astype(np.float32)
    assert out.shape == (n, PHASE2_DIM), f"shape {out.shape}"
    return out


# ---------------------------------------------------------------------------
# Phase 4 — per-player history aggregates.
# ---------------------------------------------------------------------------

PHASE4_NAMES: list[str] = [
    # Per side A (5 features) — frequency-only, no WR. WR features were
    # initially included but dropped: when the lookup is built from
    # pre-cutoff April data, the legacy team-result bug (DEC-010) makes
    # overall_wr and brawler_wr 50% noise; when built from training-window
    # data they leak the label. Frequency features (counts, main brawler)
    # are unaffected by the legacy bug and only mildly leaky.
    "a_n_known_players", "a_mean_n_games_log", "a_mean_brawler_count_log",
    "a_max_brawler_count_log", "a_n_main_picks",
    # Per side B (5 features)
    "b_n_known_players", "b_mean_n_games_log", "b_mean_brawler_count_log",
    "b_max_brawler_count_log", "b_n_main_picks",
    # Diffs A − B (2 features)
    "n_known_diff", "n_main_picks_diff",
]
PHASE4_DIM = len(PHASE4_NAMES)  # 12


# Defaults applied when a player_tag is unknown (not seen in training data).
# `is_known` is False in that case so the model can downweight these slots.
_HIST_DEFAULT_OVERALL_WR = 0.5
_HIST_DEFAULT_BRAWLER_WR = 0.5
# Minimum n_games for a player's stats to count as "real" — below this we
# still emit the values but `is_known` is False (used by the count features).
HIST_MIN_GAMES = 2


def compute_player_history(df: pd.DataFrame) -> dict:
    """Build per-player and per-(player, brawler) aggregates from `df`.

    Built from the TRAINING data only and frozen onto the featurizer (see
    `TeamFeaturizer.player_history`). Same-window self-leakage at training
    time is small (a battle's outcome influences at most ~1/n_games of its
    own player's `overall_wr`) and accepted for v1. Test-time inference uses
    the frozen training-window lookup, so test-window leakage is zero.

    Returns a dict of two sub-dicts:
      - `player_stats`         : {player_tag: {n_games, n_wins, overall_wr, main_brawler_id}}
      - `player_brawler_stats` : {(player_tag, brawler_id): {count, wins}}

    `team_a_wins` is required (it's a TRAINING dataframe). For inference,
    the lookup is reused from a previously-fit featurizer; this function is
    not called.
    """
    if df.empty:
        return {"player_stats": {}, "player_brawler_stats": {}}

    # Explode the doubled-perspective DataFrame back into one row per
    # (battle_id, slot). Each row has player_tag, brawler_id, and the binary
    # outcome (team_a_wins for the team that includes that slot).
    a = pd.DataFrame({
        "player_tag": [t for tup in df["team_a_player_tags"] for t in tup],
        "brawler_id": [int(b) for tup in df["team_a"] for b in tup],
        "win": np.repeat(df["team_a_wins"].values.astype(np.int32), 3),
    })
    b = pd.DataFrame({
        "player_tag": [t for tup in df["team_b_player_tags"] for t in tup],
        "brawler_id": [int(b) for tup in df["team_b"] for b in tup],
        # Team B wins iff team A loses.
        "win": np.repeat((1 - df["team_a_wins"].values).astype(np.int32), 3),
    })
    flat = pd.concat([a, b], ignore_index=True)
    flat = flat[flat["player_tag"].astype(bool)]  # drop empty tags defensively

    # Per-player rollup
    g = flat.groupby("player_tag", sort=False)
    n_games = g.size().astype(np.int32)
    n_wins = g["win"].sum().astype(np.int32)
    overall_wr = (n_wins / n_games).astype(np.float32)
    # Main brawler = the most-played brawler by this player in our window.
    # Ties broken by the smallest id (stable / deterministic).
    main_lookup = (
        flat.groupby(["player_tag", "brawler_id"], sort=False)
        .size()
        .reset_index(name="count")
        .sort_values(["player_tag", "count", "brawler_id"], ascending=[True, False, True])
        .drop_duplicates("player_tag", keep="first")
        .set_index("player_tag")["brawler_id"]
        .astype(np.int64)
    )
    player_stats = {
        pt: {
            "n_games": int(n_games.loc[pt]),
            "n_wins": int(n_wins.loc[pt]),
            "overall_wr": float(overall_wr.loc[pt]),
            "main_brawler_id": int(main_lookup.loc[pt]) if pt in main_lookup.index else 0,
        }
        for pt in n_games.index
    }

    # Per-(player, brawler) rollup
    pb = (
        flat.groupby(["player_tag", "brawler_id"], sort=False)
        .agg(count=("win", "size"), wins=("win", "sum"))
        .reset_index()
    )
    player_brawler_stats: dict[tuple[str, int], dict] = {}
    # Build with raw dict indexing for speed (DataFrame iteration is slow at scale).
    pt_arr = pb["player_tag"].to_numpy()
    bid_arr = pb["brawler_id"].to_numpy(dtype=np.int64)
    cnt_arr = pb["count"].to_numpy(dtype=np.int32)
    win_arr = pb["wins"].to_numpy(dtype=np.int32)
    for i in range(len(pb)):
        player_brawler_stats[(str(pt_arr[i]), int(bid_arr[i]))] = {
            "count": int(cnt_arr[i]),
            "wins": int(win_arr[i]),
        }

    return {"player_stats": player_stats, "player_brawler_stats": player_brawler_stats}


def _per_slot_history_arrays(
    df: pd.DataFrame,
    side: str,                       # 'a' or 'b'
    player_history: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (n_games, overall_wr, brawler_count, brawler_wr, is_main, is_known)
    each shaped (n, 3), one row per battle, one column per slot in side.

    Uses defaults when the player or (player, brawler) pair is missing from
    the lookup. `is_known` is True when n_games >= HIST_MIN_GAMES.
    """
    n = len(df)
    pt_col = f"team_{side}_player_tags"
    bid_col = f"team_{side}"
    if pt_col not in df.columns:
        zeros = np.zeros((n, 3), dtype=np.float32)
        return zeros, np.full((n, 3), _HIST_DEFAULT_OVERALL_WR, dtype=np.float32), \
               zeros, np.full((n, 3), _HIST_DEFAULT_BRAWLER_WR, dtype=np.float32), \
               np.zeros((n, 3), dtype=np.float32), np.zeros((n, 3), dtype=np.float32)

    pt_tuples = df[pt_col].tolist()
    bid_tuples = df[bid_col].tolist()

    n_games = np.zeros((n, 3), dtype=np.float32)
    overall_wr = np.full((n, 3), _HIST_DEFAULT_OVERALL_WR, dtype=np.float32)
    b_count = np.zeros((n, 3), dtype=np.float32)
    b_wr = np.full((n, 3), _HIST_DEFAULT_BRAWLER_WR, dtype=np.float32)
    is_main = np.zeros((n, 3), dtype=np.float32)
    is_known = np.zeros((n, 3), dtype=np.float32)

    ps = player_history.get("player_stats", {})
    pbs = player_history.get("player_brawler_stats", {})

    for i in range(n):
        pts = pt_tuples[i]
        bids = bid_tuples[i]
        for k in range(min(3, len(pts))):
            pt = str(pts[k]) if pts[k] else ""
            bid = int(bids[k])
            stats = ps.get(pt)
            if stats is None:
                # Unknown player — leave defaults (n_games=0, overall_wr=0.5,
                # b_count=0, b_wr=0.5, is_main=0, is_known=0).
                continue
            ng = stats["n_games"]
            n_games[i, k] = float(ng)
            # Only record real overall_wr / is_main / is_known when we have
            # at least HIST_MIN_GAMES games — otherwise a 1-game player's
            # 0/1 WR pollutes the per-side mean/min aggregates and the
            # "is_main" flag is trivially true for a player who's played
            # exactly one brawler. Keeping defaults (overall_wr=0.5, is_main=0)
            # produces cleaner aggregates.
            if ng >= HIST_MIN_GAMES:
                is_known[i, k] = 1.0
                overall_wr[i, k] = float(stats["overall_wr"])
                if int(stats.get("main_brawler_id", 0)) == bid:
                    is_main[i, k] = 1.0
            pb = pbs.get((pt, bid))
            if pb is not None and pb["count"] >= HIST_MIN_GAMES:
                b_count[i, k] = float(pb["count"])
                b_wr[i, k] = float(pb["wins"]) / float(pb["count"])

    return n_games, overall_wr, b_count, b_wr, is_main, is_known


def compute_phase4_features(df: pd.DataFrame, player_history: dict) -> np.ndarray:
    """Phase-4 (n, 12) team-level aggregates of per-player history stats.

    Frequency-only: n_games, brawler-pair count, main-brawler alignment
    aggregated across the 3 slots per side, plus 2 A−B diffs. WR-derived
    features were dropped because (a) when the lookup comes from training
    data, WR leaks the label, and (b) when it comes from pre-cutoff April
    data, the legacy team-result bug (DEC-010) makes WR 50 % noise.

    Robust to missing `team_*_player_tags` columns (returns zeros) and to
    unknown players (defaults applied + `is_known` flag exposed via the
    aggregates).
    """
    n = len(df)
    if n == 0:
        return np.zeros((0, PHASE4_DIM), dtype=np.float32)

    a_ng, _a_owr, a_bc, _a_bwr, a_main, a_known = _per_slot_history_arrays(
        df, "a", player_history
    )
    b_ng, _b_owr, b_bc, _b_bwr, b_main, b_known = _per_slot_history_arrays(
        df, "b", player_history
    )

    # Per side aggregates (frequency-only).
    a_n_known = a_known.sum(axis=1) / 3.0                       # in [0, 1]
    a_mean_ng_log = np.log1p(a_ng.mean(axis=1))
    a_mean_bc_log = np.log1p(a_bc.mean(axis=1))
    a_max_bc_log = np.log1p(a_bc.max(axis=1))
    a_n_main = a_main.sum(axis=1) / 3.0

    b_n_known = b_known.sum(axis=1) / 3.0
    b_mean_ng_log = np.log1p(b_ng.mean(axis=1))
    b_mean_bc_log = np.log1p(b_bc.mean(axis=1))
    b_max_bc_log = np.log1p(b_bc.max(axis=1))
    b_n_main = b_main.sum(axis=1) / 3.0

    out = np.stack([
        a_n_known, a_mean_ng_log, a_mean_bc_log, a_max_bc_log, a_n_main,
        b_n_known, b_mean_ng_log, b_mean_bc_log, b_max_bc_log, b_n_main,
        a_n_known - b_n_known,
        a_n_main - b_n_main,
    ], axis=1).astype(np.float32)

    assert out.shape == (n, PHASE4_DIM), f"shape {out.shape}"
    return out


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
    # Phase 1 toggle. When True, transform_dense appends per-team trophy/power
    # aggregates; transform_sparse is unaffected so LogReg keeps its old shape.
    include_team_aggregates: bool = False
    # Phase 2 toggle. When True, transform_dense appends 12 cyclical-time and
    # per-team `days_since_release` aggregates. The release lookup is fit on
    # training data inside `fit()` and stored on `brawler_first_seen` so that
    # serialized featurizers reproduce phase-2 output deterministically.
    include_time_features: bool = False
    brawler_first_seen: dict[int, str] = field(default_factory=dict)
    # Phase 4 toggle. When True, transform_dense appends 20 per-team
    # aggregates of per-player history stats. The lookup is fit on training
    # data and stored on `player_history` (two sub-dicts: per-player and
    # per-(player, brawler)). Round-trips via save/load. Requires
    # `team_a/b_player_tags` columns on the input DataFrame.
    include_history_features: bool = False
    player_history: dict = field(default_factory=dict)

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
        # + phase-1 team aggregates (23) + phase-2 time/release scalars (12)
        # + phase-4 history aggregates (20) when each flag is enabled.
        return (
            self.n_brawlers * 2
            + self.n_modes
            + self.n_maps
            + self.n_btypes
            + 3
            + (TEAM_AGGREGATE_DIM if self.include_team_aggregates else 0)
            + (PHASE2_DIM if self.include_time_features else 0)
            + (PHASE4_DIM if self.include_history_features else 0)
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
        if self.include_team_aggregates:
            names.extend(TEAM_AGGREGATE_NAMES)
        if self.include_time_features:
            names.extend(PHASE2_NAMES)
        if self.include_history_features:
            names.extend(PHASE4_NAMES)
        return names

    def fit(self, df: pd.DataFrame, history_df: pd.DataFrame | None = None) -> "TeamFeaturizer":
        """Learn vocabulary from `df` and freeze any opt-in lookups onto self.

        `history_df` (optional, only consulted when `include_history_features`
        is True) is a SEPARATE DataFrame used to build the per-player history
        lookup. Pass pre-cutoff data (e.g. April battles) so the lookup is
        DISJOINT from the training rows being predicted on — this is the
        cleanest leakage-free source. If omitted, the lookup is built from
        `df` itself, which is leaky (see DEC-018) and emits a warning.
        """
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
        # Phase 2: freeze the brawler -> first-seen lookup from training data.
        # The lookup is what makes `compute_phase2_features` deterministic at
        # inference time, so it has to live on the featurizer. Skip if phase 2
        # is disabled to keep saved meta clean for legacy phase-1-only paths.
        if self.include_time_features:
            self.brawler_first_seen = compute_brawler_first_seen(df)
        # Phase 4: freeze the per-player and per-(player, brawler) lookup.
        # Use `history_df` if supplied — that should be a different time
        # window from training (e.g. pre-cutoff April data). If absent,
        # fall back to `df` with a leakage warning.
        if self.include_history_features:
            if history_df is not None:
                self.player_history = compute_player_history(history_df)
            else:
                import warnings
                warnings.warn(
                    "include_history_features=True but no history_df provided; "
                    "the lookup will be built from the training df itself, "
                    "which leaks each row's outcome through its own player_tag "
                    "stats. Pass `history_df=` (e.g. pre-cutoff April data) "
                    "for clean methodology. See DEC-018.",
                    stacklevel=2,
                )
                self.player_history = compute_player_history(df)
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

        # Order: brawler_a (n_brawlers) | brawler_b (n_brawlers) | cat_mode | cat_map | cat_btype | a_trophy | b_trophy | trophy_diff [| team aggregates (23)]
        parts = [
            ax,
            bx,
            mode_idx.reshape(-1, 1).astype(np.float32),
            map_idx.reshape(-1, 1).astype(np.float32),
            btype_idx.reshape(-1, 1).astype(np.float32),
            a_trophies.reshape(-1, 1).astype(np.float32),
            b_trophies.reshape(-1, 1).astype(np.float32),
            trophy_diff.reshape(-1, 1).astype(np.float32),
        ]
        if self.include_team_aggregates:
            parts.append(compute_team_aggregates(df))
        if self.include_time_features:
            parts.append(compute_phase2_features(df, self.brawler_first_seen))
        if self.include_history_features:
            parts.append(compute_phase4_features(df, self.player_history))
        X = np.hstack(parts)

        cat_cols = [
            self.n_brawlers * 2,        # mode
            self.n_brawlers * 2 + 1,    # map
            self.n_brawlers * 2 + 2,    # battle_type
        ]
        return X, cat_cols
