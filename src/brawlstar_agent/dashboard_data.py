"""Dashboard analytics data collection — shared between dashboard server
and the precompute cron job.

Heavy SQL (matchup self-join, synergy, etc.) is expensive on a 1-CPU droplet.
The dashboard server reads precomputed results from `data/analytics_cache.json`
when present; the precompute script writes the cache periodically. Both ends
import `collect_all_data` from here so the schema stays in sync.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .analytics import TROPHY_TIERS, BattleAnalytics
from .collector import load_pinned_tags
from .db import DEFAULT_DB_PATH
from .models import RANKED_TIERS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = PROJECT_ROOT / "data" / "analytics_cache.json"

log = logging.getLogger(__name__)


def collect_all_data(db_path: str | Path) -> dict[str, Any]:
    """Run all dashboard analytics queries and return a single nested dict.

    This is the heavy work. On a 1-CPU $6/mo droplet with ~1M battle_player
    rows, expect 5-15 minutes. On a multi-core laptop, expect 5-30 seconds.
    """
    a = BattleAnalytics(str(db_path))
    summary = a.summary()
    modes = [m for m in summary["mode_distribution"].keys()
             if m not in ("soloShowdown", "duoShowdown")]

    log.info("Computing brawler win rates...")
    brawler_rates: dict[str, list[dict]] = {}
    brawler_rates["all"] = a.brawler_win_rates(min_sample=5, limit=999)
    for mode in modes:
        rows = a.brawler_win_rates(mode=mode, min_sample=3, limit=999)
        if rows:
            brawler_rates[mode] = rows

    brawler_rates["ladder_all"] = a.brawler_win_rates(battle_type="ranked", min_sample=5, limit=999)
    tier_data = a.brawler_win_rates_by_tier(min_sample=10, limit=999)
    for tier_name, rows in tier_data.items():
        brawler_rates[f"ladder_{tier_name}"] = rows

    brawler_rates["competitive_all"] = a.brawler_win_rates(
        battle_type="soloRanked", min_sample=5, limit=999,
    )
    for tier_name, _lo, _hi in RANKED_TIERS:
        rows = a.brawler_win_rates(
            battle_type="soloRanked", ranked_tier=tier_name,
            min_sample=3, limit=999,
        )
        if rows:
            brawler_rates[f"competitive_{tier_name}"] = rows

    log.info("Computing combo win rates...")
    combos: dict[str, list[dict]] = {}
    combos["all"] = a.combo_win_rates(min_sample=3, limit=50)
    for mode in modes:
        rows = a.combo_win_rates(mode=mode, min_sample=2, limit=50)
        if rows:
            combos[mode] = rows

    log.info("Computing matchup matrix (self-join, slowest)...")
    matchups = a.matchup_win_rates(min_sample=50, limit=300)

    log.info("Computing synergy matrix...")
    synergies = a.synergy_win_rates(min_sample=50, limit=300)

    log.info("Computing brawler scores...")
    brawler_scores = a.brawler_scores()

    log.info("Computing personal data (if MAJOR_ACCOUNT_TAG set)...")
    load_dotenv(PROJECT_ROOT / "api.env")
    my_tag = os.getenv("MAJOR_ACCOUNT_TAG", "")
    my_data = _collect_personal_data(a._conn, my_tag) if my_tag else None

    log.info("Computing watched-player data from pinned_tags.txt...")
    watched_tags = [t for t in load_pinned_tags() if t != my_tag]
    watched_data = [_watched_player_entry(a._conn, t) for t in watched_tags]
    log.info("  watched players: %d", len(watched_data))

    a.close()

    trophy_tiers = [{"name": n, "lo": lo, "hi": hi} for n, lo, hi in TROPHY_TIERS]
    ranked_tiers = [{"name": n, "lo": lo, "hi": hi} for n, lo, hi in RANKED_TIERS]

    return {
        "summary": summary,
        "modes": modes,
        "trophy_tiers": trophy_tiers,
        "ranked_tiers": ranked_tiers,
        "brawler_rates": brawler_rates,
        "brawler_scores": brawler_scores,
        "combos": combos,
        "matchups": matchups,
        "synergies": synergies,
        "my_data": my_data,
        "watched_data": watched_data,
    }


def _watched_player_entry(conn: sqlite3.Connection, tag: str) -> dict:
    """Build a watched-player entry. Always returns a dict with at least
    {tag, name, battle_count, ...}, even when the tag has no battles in the
    DB yet (e.g., newly added side accounts) so the dashboard can render an
    empty-state card rather than skipping the player."""
    full = _collect_personal_data(conn, tag)
    if full:
        return full

    # No battles yet — fall back to whatever player profile exists.
    player = conn.execute(
        "SELECT tag, name, trophies, highest_trophies, exp_level, club_name "
        "FROM players WHERE tag = ?",
        (tag,),
    ).fetchone()
    if player:
        return {
            "tag": tag,
            "name": player["name"],
            "trophies": player["trophies"],
            "highest_trophies": player["highest_trophies"],
            "exp_level": player["exp_level"],
            "club": player["club_name"],
            "battle_count": 0,
            "battle_log": [],
            "brawler_stats": [],
            "mode_stats": [],
        }

    # Tag not yet crawled at all (timer hasn't fired since it was added).
    return {
        "tag": tag,
        "name": "(awaiting first crawl)",
        "trophies": None,
        "highest_trophies": None,
        "exp_level": None,
        "club": None,
        "battle_count": 0,
        "battle_log": [],
        "brawler_stats": [],
        "mode_stats": [],
    }


def _collect_personal_data(conn: sqlite3.Connection, tag: str) -> dict | None:
    """Build the personal-stats blob for one player tag."""
    player = conn.execute(
        "SELECT tag, name, trophies, highest_trophies, exp_level, club_name "
        "FROM players WHERE tag = ?",
        (tag,),
    ).fetchone()
    if not player:
        return None

    battles_raw = conn.execute("""
        SELECT
            b.battle_id, b.battle_time_iso, b.mode, b.map, b.battle_type,
            b.duration, b.is_showdown, b.star_player_tag,
            bp.brawler_name, bp.brawler_trophies, bp.result,
            bp.trophy_change, bp.is_star_player, bp.team_index
        FROM battle_players bp
        JOIN battles b ON bp.battle_id = b.battle_id
        WHERE bp.player_tag = ?
        ORDER BY b.battle_time_iso DESC
    """, (tag,)).fetchall()

    battle_log = []
    for br in battles_raw:
        bid = br["battle_id"]
        my_team_idx = br["team_index"]

        teammates = conn.execute("""
            SELECT player_tag, brawler_name, brawler_trophies, is_star_player
            FROM battle_players
            WHERE battle_id = ? AND team_index = ? AND player_tag != ?
        """, (bid, my_team_idx, tag)).fetchall()

        opponents = conn.execute("""
            SELECT player_tag, brawler_name, brawler_trophies, is_star_player
            FROM battle_players
            WHERE battle_id = ? AND team_index != ?
        """, (bid, my_team_idx)).fetchall()

        battle_log.append({
            "time": br["battle_time_iso"],
            "mode": br["mode"],
            "map": br["map"],
            "type": br["battle_type"],
            "duration": br["duration"],
            "brawler": br["brawler_name"],
            "trophies": br["brawler_trophies"],
            "result": br["result"],
            "trophy_change": br["trophy_change"],
            "star_player": bool(br["is_star_player"]),
            "teammates": [
                {"brawler": t["brawler_name"], "trophies": t["brawler_trophies"]}
                for t in teammates
            ],
            "opponents": [
                {"brawler": o["brawler_name"], "trophies": o["brawler_trophies"]}
                for o in opponents
            ],
        })

    brawler_stats = conn.execute("""
        SELECT
            bp.brawler_name,
            COUNT(*) as total,
            SUM(CASE WHEN bp.result = 'victory' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN bp.is_star_player THEN 1 ELSE 0 END) as star_count,
            ROUND(100.0 * SUM(CASE WHEN bp.result = 'victory' THEN 1 ELSE 0 END) / COUNT(*), 1) as win_rate
        FROM battle_players bp
        JOIN battles b ON bp.battle_id = b.battle_id
        WHERE bp.player_tag = ? AND bp.result IN ('victory', 'defeat')
        GROUP BY bp.brawler_name
        ORDER BY total DESC
    """, (tag,)).fetchall()

    mode_stats = conn.execute("""
        SELECT
            b.mode,
            COUNT(*) as total,
            SUM(CASE WHEN bp.result = 'victory' THEN 1 ELSE 0 END) as wins,
            ROUND(100.0 * SUM(CASE WHEN bp.result = 'victory' THEN 1 ELSE 0 END) / COUNT(*), 1) as win_rate
        FROM battle_players bp
        JOIN battles b ON bp.battle_id = b.battle_id
        WHERE bp.player_tag = ? AND bp.result IN ('victory', 'defeat')
        GROUP BY b.mode
        ORDER BY total DESC
    """, (tag,)).fetchall()

    return {
        "tag": tag,
        "name": player["name"],
        "trophies": player["trophies"],
        "highest_trophies": player["highest_trophies"],
        "exp_level": player["exp_level"],
        "club": player["club_name"],
        "battle_count": len(battle_log),
        "battle_log": battle_log,
        "brawler_stats": [dict(r) for r in brawler_stats],
        "mode_stats": [dict(r) for r in mode_stats],
    }


def write_cache(db_path: str | Path = DEFAULT_DB_PATH) -> dict:
    """Compute and atomically write the analytics cache to disk.

    Returns the cache dict that was written. Used by precompute-analytics.py.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found at {db_path}")

    log.info("Starting precompute against %s", db_path)
    started = time.monotonic()
    data = collect_all_data(db_path)
    elapsed = time.monotonic() - started

    cache = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "computed_in_seconds": round(elapsed, 1),
        "db_size_mb": round(db_path.stat().st_size / 1024 / 1024, 1),
        "battle_count": data["summary"].get("total_battles"),
        "data": data,
    }

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cache, default=str))
    tmp.replace(CACHE_PATH)

    cache_size_mb = CACHE_PATH.stat().st_size / 1024 / 1024
    log.info("Wrote cache to %s (%.2f MB) — compute %.1fs",
             CACHE_PATH, cache_size_mb, elapsed)
    return cache


def read_cache() -> dict | None:
    """Read the analytics cache. Returns None if missing or corrupt."""
    if not CACHE_PATH.exists():
        return None
    try:
        return json.loads(CACHE_PATH.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Cache at %s is unreadable: %s", CACHE_PATH, exc)
        return None
