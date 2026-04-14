"""Statistical models for brawler strength estimation.

- Wilson score interval: corrects for sample size uncertainty
- Tier-adjusted win rate: corrects for player skill composition bias
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

from .db import DEFAULT_DB_PATH

# soloRanked brawler_trophies = ranked tier points (2-22)
RANKED_TIERS = [
    ("Bronze",    2,  5),
    ("Gold",      6,  9),
    ("Diamond",  10, 13),
    ("Mythic",   14, 16),
    ("Legendary", 17, 19),
    ("Masters",  20, 22),
]

# Regular ranked ladder brawler_trophies
LADDER_TIERS = [
    ("Starter",  0,    499),
    ("Mid",      500,  749),
    ("High",     750,  999),
    ("Elite",    1000, 1249),
    ("Pro",      1250, 99999),
]


def wilson_interval(wins: int, total: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score confidence interval for a binomial proportion.

    Returns (lower, center, upper) as fractions in [0, 1].
    z=1.96 gives a 95% CI.
    """
    if total == 0:
        return (0.0, 0.0, 0.0)
    p = wins / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    spread = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total) / denom
    return (max(0.0, center - spread), center, min(1.0, center + spread))


def ranked_points_to_tier(points: int) -> str | None:
    """Map soloRanked brawler_trophies (2-22) to a ranked tier name."""
    for name, lo, hi in RANKED_TIERS:
        if lo <= points <= hi:
            return name
    return None


def _query_per_tier_stats(
    conn: sqlite3.Connection,
    tiers: list[tuple[str, int, int]],
    battle_type: str,
    mode: str | None = None,
) -> tuple[dict[str, dict[str, dict]], dict[str, dict]]:
    """Query win/loss counts per brawler per tier.

    Returns:
        brawler_stats: {brawler_name: {tier_name: {wins, total}}}
        global_stats:  {tier_name: {wins, total}}
    """
    brawler_stats: dict[str, dict[str, dict]] = {}
    global_stats: dict[str, dict] = {}

    for tier_name, lo, hi in tiers:
        params: list = [battle_type, lo, hi]
        mode_cond = ""
        if mode:
            mode_cond = "AND b.mode = ?"
            params.append(mode)

        rows = conn.execute(f"""
            SELECT
                bp.brawler_name,
                COUNT(*) as total,
                SUM(CASE WHEN bp.result = 'victory' THEN 1 ELSE 0 END) as wins
            FROM battle_players bp
            JOIN battles b ON bp.battle_id = b.battle_id
            WHERE b.is_showdown = 0
              AND b.battle_type = ?
              AND bp.brawler_trophies >= ? AND bp.brawler_trophies <= ?
              AND bp.result IN ('victory', 'defeat')
              {mode_cond}
            GROUP BY bp.brawler_name
        """, params).fetchall()

        tier_total_wins = 0
        tier_total_games = 0
        for r in rows:
            bname = r[0]
            if bname not in brawler_stats:
                brawler_stats[bname] = {}
            brawler_stats[bname][tier_name] = {"wins": r[2], "total": r[1]}
            tier_total_wins += r[2]
            tier_total_games += r[1]

        global_stats[tier_name] = {"wins": tier_total_wins, "total": tier_total_games}

    return brawler_stats, global_stats


def tier_adjusted_win_rate(
    brawler_tier_stats: dict[str, dict],
    global_tier_stats: dict[str, dict],
) -> float | None:
    """Compute a tier-standardized win rate for a brawler.

    Re-weights the brawler's per-tier WR by the global tier distribution,
    so brawlers that are mostly played at one tier don't get inflated/deflated
    by the skill level of their player base.
    """
    grand_total = sum(g["total"] for g in global_tier_stats.values())
    if grand_total == 0:
        return None

    adjusted = 0.0
    weight_sum = 0.0

    for tier_name, g in global_tier_stats.items():
        weight = g["total"] / grand_total
        if weight == 0:
            continue

        b = brawler_tier_stats.get(tier_name)
        if b and b["total"] >= 3:
            wr = b["wins"] / b["total"]
        elif g["total"] > 0:
            wr = g["wins"] / g["total"]
        else:
            continue

        adjusted += weight * wr
        weight_sum += weight

    if weight_sum == 0:
        return None
    return adjusted / weight_sum


def score_brawlers(
    db_path: Path | str = DEFAULT_DB_PATH,
    mode: str | None = None,
) -> list[dict]:
    """Compute comprehensive brawler scores combining all models.

    For each brawler returns:
    - raw_wr, wins, total (overall team-mode stats)
    - wilson_lower, wilson_upper (95% CI bounds)
    - adjusted_wr (tier-standardized from ranked ladder data)
    - ranked_adjusted_wr (tier-standardized from soloRanked data, if enough data)
    - per-tier win rates for soloRanked (Bronze..Masters)
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    mode_cond = ""
    params: list = []
    if mode:
        mode_cond = "AND b.mode = ?"
        params.append(mode)

    # 1. Raw overall win rates (all team modes, all battle types)
    raw_rows = conn.execute(f"""
        SELECT
            bp.brawler_id,
            bp.brawler_name,
            COUNT(*) as total,
            SUM(CASE WHEN bp.result = 'victory' THEN 1 ELSE 0 END) as wins
        FROM battle_players bp
        JOIN battles b ON bp.battle_id = b.battle_id
        WHERE b.is_showdown = 0
          AND bp.result IN ('victory', 'defeat')
          {mode_cond}
        GROUP BY bp.brawler_id, bp.brawler_name
    """, params).fetchall()

    brawlers = {}
    for r in raw_rows:
        name = r["brawler_name"]
        wins = r["wins"]
        total = r["total"]
        wl, wc, wu = wilson_interval(wins, total)
        brawlers[name] = {
            "brawler_id": r["brawler_id"],
            "brawler_name": name,
            "wins": wins,
            "total": total,
            "raw_wr": round(100.0 * wins / total, 2) if total else 0,
            "wilson_lower": round(100.0 * wl, 2),
            "wilson_upper": round(100.0 * wu, 2),
        }

    # 2. Tier-adjusted WR from ranked ladder
    ladder_bstats, ladder_gstats = _query_per_tier_stats(conn, LADDER_TIERS, "ranked", mode)
    for name, b in brawlers.items():
        bts = ladder_bstats.get(name, {})
        adj = tier_adjusted_win_rate(bts, ladder_gstats)
        b["adjusted_wr"] = round(100.0 * adj, 2) if adj is not None else b["raw_wr"]

    # 3. Tier-adjusted WR from soloRanked + per-tier breakdown
    ranked_bstats, ranked_gstats = _query_per_tier_stats(conn, RANKED_TIERS, "soloRanked", mode)
    for name, b in brawlers.items():
        bts = ranked_bstats.get(name, {})
        adj = tier_adjusted_win_rate(bts, ranked_gstats)
        b["ranked_adjusted_wr"] = round(100.0 * adj, 2) if adj is not None else None

        # Per ranked tier WR
        for tier_name, _, _ in RANKED_TIERS:
            ts = bts.get(tier_name)
            if ts and ts["total"] >= 3:
                b[f"wr_{tier_name}"] = round(100.0 * ts["wins"] / ts["total"], 1)
                b[f"n_{tier_name}"] = ts["total"]
            else:
                b[f"wr_{tier_name}"] = None
                b[f"n_{tier_name}"] = 0

    conn.close()

    result = list(brawlers.values())
    result.sort(key=lambda x: x["raw_wr"], reverse=True)
    return result
