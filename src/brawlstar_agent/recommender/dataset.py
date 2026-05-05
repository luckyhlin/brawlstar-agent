"""Dataset loading and train/test splits for the recommender.

Hard rules:

- Training data MUST be filtered by `battle_time_iso >= CLEAN_CUTOFF_ISO` (the
  team-result bug fix). See DEC-010 — the bug is not recoverable from stored
  columns, so legacy data has ~50% inverted labels for some battles.
- Showdown is excluded (different shape: ranks 1-10, no teams).
- Friendly / challenge / tournament battles are excluded by default — different
  player population, often unsuited to ranked-meta inference.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..db import DEFAULT_DB_PATH

# Cutoff: battles at or after this ISO timestamp use the post-fix code path
# in db.py and have correct team_index → result attribution.
CLEAN_CUTOFF_ISO = "2026-05-03T01:00:00Z"

# Battle types we consider "competitive" (have meta signal):
#   - ranked: trophy ladder
#   - soloRanked: competitive ranked (Bronze..Masters)
COMPETITIVE_BATTLE_TYPES = ("ranked", "soloRanked")


@dataclass(frozen=True)
class BattleRow:
    """One battle in the team-completion shape: A_brawlers vs B_brawlers, A wins yes/no.

    `team_a_*` is the team that we're trying to predict the win probability for.
    For training we expand each battle to BOTH perspectives (A=team0 then A=team1)
    so the model sees a balanced 50/50 win/loss distribution.
    """

    battle_id: str
    battle_time_iso: str
    mode: str
    map: str
    battle_type: str

    team_a: tuple[int, ...]      # sorted brawler IDs
    team_b: tuple[int, ...]
    team_a_wins: int             # 1 or 0

    # Skill features (means over team)
    team_a_trophies_mean: float
    team_b_trophies_mean: float


def _connect(db_path: Path | str) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def load_clean_battles(
    db_path: Path | str = DEFAULT_DB_PATH,
    after: str = CLEAN_CUTOFF_ISO,
    before: str | None = None,
    modes: tuple[str, ...] | None = None,
    battle_types: tuple[str, ...] = COMPETITIVE_BATTLE_TYPES,
    require_complete_teams: bool = True,
    expand_both_perspectives: bool = True,
) -> pd.DataFrame:
    """Load 3v3 battles from clean window into team-completion shape.

    Returns DataFrame with one row per (battle, perspective) when
    `expand_both_perspectives=True`, else one row per battle (team 0 = team A).

    Columns:
        battle_id, battle_time_iso, mode, map, battle_type,
        team_a (tuple of 3 sorted brawler IDs),
        team_b (tuple of 3 sorted brawler IDs),
        team_a_wins (0/1),
        team_a_trophies_mean, team_b_trophies_mean
    """
    conn = _connect(db_path)
    try:
        params: list = [after]
        bt_filter = ""
        if battle_types:
            placeholders = ",".join("?" * len(battle_types))
            bt_filter = f"AND b.battle_type IN ({placeholders})"
            params.extend(battle_types)
        before_filter = ""
        if before:
            before_filter = "AND b.battle_time_iso < ?"
            params.append(before)
        mode_filter = ""
        if modes:
            placeholders = ",".join("?" * len(modes))
            mode_filter = f"AND b.mode IN ({placeholders})"
            params.extend(modes)

        # Pull battles + battle_players in one go via JOIN.
        # We ORDER BY so that within a battle, rows come out grouped by team_index.
        sql = f"""
            SELECT
                b.battle_id, b.battle_time_iso, b.mode, b.map, b.battle_type,
                bp.team_index, bp.brawler_id, bp.brawler_trophies, bp.result
            FROM battles b
            JOIN battle_players bp ON bp.battle_id = b.battle_id
            WHERE b.is_showdown = 0
              AND b.battle_time_iso >= ?
              {bt_filter}
              {before_filter}
              {mode_filter}
              AND bp.result IN ('victory', 'defeat')
            ORDER BY b.battle_id, bp.team_index, bp.brawler_id
        """
        df = pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()

    if df.empty:
        return df

    # Group by (battle_id, team_index) → team rows
    grp = df.groupby(["battle_id", "team_index"], sort=False)
    teams = grp.agg(
        battle_time_iso=("battle_time_iso", "first"),
        mode=("mode", "first"),
        map=("map", "first"),
        battle_type=("battle_type", "first"),
        brawlers=("brawler_id", lambda s: tuple(sorted(s.tolist()))),
        n_players=("brawler_id", "size"),
        trophies_mean=("brawler_trophies", "mean"),
        team_result=("result", "first"),  # all rows in a (battle, team) share same result post-fix
    ).reset_index()

    if require_complete_teams:
        teams = teams[teams["n_players"] == 3].copy()

    # Pivot so each battle has team0 and team1 rows side by side
    # We need both teams to have 3 players for a 3v3.
    # Drop battles with anything other than exactly 2 teams.
    battle_team_counts = teams.groupby("battle_id").size()
    valid_battles = battle_team_counts[battle_team_counts == 2].index
    teams = teams[teams["battle_id"].isin(valid_battles)].copy()
    teams = teams.sort_values(["battle_id", "team_index"]).reset_index(drop=True)

    # Reshape: even rows = team0, odd rows = team1
    if len(teams) % 2 != 0:
        # Defensive: drop any orphan
        teams = teams.iloc[:-1].copy()

    t0 = teams.iloc[0::2].reset_index(drop=True)
    t1 = teams.iloc[1::2].reset_index(drop=True)
    assert (t0["battle_id"].values == t1["battle_id"].values).all(), "team pairing failed"

    # Drop battles where the two teams disagree pathologically (both victory or both defeat).
    same_label = t0["team_result"].values == t1["team_result"].values
    keep = ~same_label
    t0 = t0[keep].reset_index(drop=True)
    t1 = t1[keep].reset_index(drop=True)

    # Team A perspective = team0 first
    base = pd.DataFrame({
        "battle_id": t0["battle_id"].values,
        "battle_time_iso": t0["battle_time_iso"].values,
        "mode": t0["mode"].values,
        "map": t0["map"].values,
        "battle_type": t0["battle_type"].values,
        "team_a": t0["brawlers"].values,
        "team_b": t1["brawlers"].values,
        "team_a_wins": (t0["team_result"].values == "victory").astype(np.int8),
        "team_a_trophies_mean": t0["trophies_mean"].values,
        "team_b_trophies_mean": t1["trophies_mean"].values,
    })

    if not expand_both_perspectives:
        return base

    # Mirror perspective: team1 = team A
    mirror = pd.DataFrame({
        "battle_id": t0["battle_id"].values,
        "battle_time_iso": t0["battle_time_iso"].values,
        "mode": t0["mode"].values,
        "map": t0["map"].values,
        "battle_type": t0["battle_type"].values,
        "team_a": t1["brawlers"].values,
        "team_b": t0["brawlers"].values,
        "team_a_wins": (t1["team_result"].values == "victory").astype(np.int8),
        "team_a_trophies_mean": t1["trophies_mean"].values,
        "team_b_trophies_mean": t0["trophies_mean"].values,
    })

    return pd.concat([base, mirror], ignore_index=True)


def split_temporal(
    df: pd.DataFrame,
    train_end: str,
    test_start: str | None = None,
    test_end: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-based split: train on [start, train_end), test on [test_start, test_end).

    If `test_start` is None, defaults to `train_end` (no gap).
    If `test_end` is None, includes everything from `test_start` onward.

    The whole point of this split is to evaluate "train on month N, predict month N+1"
    style transferability. Use this instead of random splits whenever possible —
    random splits leak future information into training in a meta-drifting domain.
    """
    if test_start is None:
        test_start = train_end

    train = df[df["battle_time_iso"] < train_end].copy()
    if test_end is None:
        test = df[df["battle_time_iso"] >= test_start].copy()
    else:
        test = df[
            (df["battle_time_iso"] >= test_start)
            & (df["battle_time_iso"] < test_end)
        ].copy()
    return train, test


def split_random(
    df: pd.DataFrame, test_frac: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random split, grouped by battle_id so a battle's two perspectives never
    end up on different sides of the split (would be a label-leak).
    """
    rng = np.random.default_rng(seed)
    battle_ids = np.array(df["battle_id"].drop_duplicates().tolist(), dtype=object)
    rng.shuffle(battle_ids)
    n_test = int(len(battle_ids) * test_frac)
    test_ids = set(battle_ids[:n_test].tolist())
    train_mask = ~df["battle_id"].isin(test_ids)
    return df[train_mask].copy(), df[~train_mask].copy()


def load_brawler_names(db_path: Path | str = DEFAULT_DB_PATH) -> dict[int, str]:
    """Resolve brawler IDs to names. Prefers `brawlers` table; falls back to
    `battle_players.brawler_name` for any IDs missing there (handles new
    brawlers that landed in battles before the `brawlers` table was refreshed).
    """
    conn = _connect(db_path)
    try:
        names = {int(r[0]): str(r[1]) for r in conn.execute("SELECT id, name FROM brawlers").fetchall()}
        # Find any brawler IDs in battle_players not in `brawlers`
        rows = conn.execute(
            """
            SELECT brawler_id, brawler_name, COUNT(*) AS n
            FROM battle_players
            WHERE brawler_id IS NOT NULL AND brawler_name IS NOT NULL
            GROUP BY brawler_id, brawler_name
            """
        ).fetchall()
        # Pick the most common name per id
        per_id: dict[int, dict[str, int]] = {}
        for bid, bname, n in rows:
            per_id.setdefault(int(bid), {})
            per_id[int(bid)][str(bname)] = int(n)
        for bid, names_for_id in per_id.items():
            if bid not in names:
                top_name = max(names_for_id.items(), key=lambda kv: kv[1])[0]
                names[bid] = top_name
    finally:
        conn.close()
    return names


def battle_count_summary(db_path: Path | str = DEFAULT_DB_PATH) -> dict:
    """Quick counts for sanity-checking the clean window."""
    conn = _connect(db_path)
    try:
        total = conn.execute(
            "SELECT COUNT(*) FROM battles WHERE is_showdown=0"
        ).fetchone()[0]
        clean = conn.execute(
            "SELECT COUNT(*) FROM battles WHERE is_showdown=0 AND battle_time_iso >= ?",
            (CLEAN_CUTOFF_ISO,),
        ).fetchone()[0]
        latest = conn.execute(
            "SELECT MAX(battle_time_iso) FROM battles WHERE is_showdown=0"
        ).fetchone()[0]
        per_mode = conn.execute(
            """
            SELECT mode, COUNT(*) AS n, COUNT(DISTINCT map) AS maps
            FROM battles
            WHERE is_showdown=0 AND battle_time_iso >= ?
            GROUP BY mode ORDER BY n DESC
            """,
            (CLEAN_CUTOFF_ISO,),
        ).fetchall()
    finally:
        conn.close()
    return {
        "total_team_battles": total,
        "clean_team_battles": clean,
        "latest_battle": latest,
        "per_mode": [dict(r) for r in per_mode],
    }
