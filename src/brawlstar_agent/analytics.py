"""Battle analytics: win rates, combos, matchups, synergies.

All queries run against the SQLite database populated by the collector.
Results are returned as plain dicts/lists for easy printing or further processing.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .db import DEFAULT_DB_PATH

# Brawler trophy tiers for ranked (trophy ladder) battles.
# In ranked, brawler_trophies reflects real skill; in soloRanked it's rank points.
TROPHY_TIERS = [
    ("Starter",   0,    499),
    ("Mid",       500,  749),
    ("High",      750,  999),
    ("Elite",     1000, 1249),
    ("Pro",       1250, 99999),
]

# Player-level trophies (total across all brawlers) from profile data.
PLAYER_TIERS = [
    ("Casual",     0,     9999),
    ("Regular",    10000, 19999),
    ("Veteran",    20000, 34999),
    ("Expert",     35000, 49999),
    ("Top",        50000, 999999),
]


class BattleAnalytics:
    """Query engine for battle statistics."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row

    def _where_clause(
        self,
        mode: str | None = None,
        battle_type: str | None = None,
        min_trophies: int | None = None,
        max_trophies: int | None = None,
        trophy_tier: str | None = None,
        ranked_tier: str | None = None,
        player_tier: str | None = None,
        after: str | None = None,
        before: str | None = None,
    ) -> tuple[str, list, bool]:
        """Build reusable WHERE conditions and params for filtering.

        Returns (where_sql, params, needs_player_join).
        """
        from .models import RANKED_TIERS

        conditions = ["b.is_showdown = 0"]
        params: list = []
        needs_player_join = False

        if mode:
            conditions.append("b.mode = ?")
            params.append(mode)
        if battle_type:
            conditions.append("b.battle_type = ?")
            params.append(battle_type)
        if min_trophies is not None:
            conditions.append("bp.brawler_trophies >= ?")
            params.append(min_trophies)
        if max_trophies is not None:
            conditions.append("bp.brawler_trophies <= ?")
            params.append(max_trophies)
        if trophy_tier:
            for name, lo, hi in TROPHY_TIERS:
                if name == trophy_tier:
                    conditions.append("bp.brawler_trophies >= ? AND bp.brawler_trophies <= ?")
                    params.extend([lo, hi])
                    break
        if ranked_tier:
            for name, lo, hi in RANKED_TIERS:
                if name == ranked_tier:
                    conditions.append("bp.brawler_trophies >= ? AND bp.brawler_trophies <= ?")
                    params.extend([lo, hi])
                    break
        if player_tier:
            needs_player_join = True
            for name, lo, hi in PLAYER_TIERS:
                if name == player_tier:
                    conditions.append("p.trophies >= ? AND p.trophies <= ?")
                    params.extend([lo, hi])
                    break
        if after:
            conditions.append("b.battle_time_iso >= ?")
            params.append(after)
        if before:
            conditions.append("b.battle_time_iso <= ?")
            params.append(before)

        return " AND ".join(conditions), params, needs_player_join

    def _player_join_sql(self, needs_player_join: bool) -> str:
        if needs_player_join:
            return "LEFT JOIN players p ON bp.player_tag = p.tag"
        return ""

    # -- 1. Win rate per brawler --

    def brawler_win_rates(
        self,
        mode: str | None = None,
        min_trophies: int | None = None,
        min_sample: int = 10,
        limit: int = 50,
        **filters,
    ) -> list[dict]:
        """Win rate for each brawler, optionally filtered by mode/trophy range.

        Returns sorted by win_rate descending.
        """
        where, params, pj = self._where_clause(mode=mode, min_trophies=min_trophies, **filters)
        pjoin = self._player_join_sql(pj)

        query = f"""
            SELECT
                bp.brawler_id,
                bp.brawler_name,
                COUNT(*) as total,
                SUM(CASE WHEN bp.result = 'victory' THEN 1 ELSE 0 END) as wins,
                ROUND(100.0 * SUM(CASE WHEN bp.result = 'victory' THEN 1 ELSE 0 END) / COUNT(*), 2) as win_rate
            FROM battle_players bp
            JOIN battles b ON bp.battle_id = b.battle_id
            {pjoin}
            WHERE {where}
              AND bp.result IN ('victory', 'defeat')
            GROUP BY bp.brawler_id, bp.brawler_name
            HAVING COUNT(*) >= ?
            ORDER BY win_rate DESC
            LIMIT ?
        """
        params.extend([min_sample, limit])
        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    # -- 2. Win rate per 3-brawler combo --

    def combo_win_rates(
        self,
        mode: str | None = None,
        min_trophies: int | None = None,
        min_sample: int = 3,
        limit: int = 50,
        **filters,
    ) -> list[dict]:
        """Win rate for each 3-brawler team composition.

        Normalizes combos by sorting brawler names alphabetically.
        """
        where, params, pj = self._where_clause(mode=mode, min_trophies=min_trophies, **filters)
        pjoin = self._player_join_sql(pj)

        query = f"""
            WITH team_comp AS (
                SELECT
                    bp.battle_id,
                    bp.team_index,
                    GROUP_CONCAT(bp.brawler_name, '|') as raw_comp,
                    bp.result
                FROM battle_players bp
                JOIN battles b ON bp.battle_id = b.battle_id
                {pjoin}
                WHERE {where}
                  AND bp.result IN ('victory', 'defeat')
                GROUP BY bp.battle_id, bp.team_index
                HAVING COUNT(*) = 3
            )
            SELECT
                raw_comp,
                COUNT(*) as total,
                SUM(CASE WHEN result = 'victory' THEN 1 ELSE 0 END) as wins,
                ROUND(100.0 * SUM(CASE WHEN result = 'victory' THEN 1 ELSE 0 END) / COUNT(*), 2) as win_rate
            FROM team_comp
            GROUP BY raw_comp
            HAVING COUNT(*) >= ?
            ORDER BY win_rate DESC
            LIMIT ?
        """
        params.extend([min_sample, limit])
        rows = self._conn.execute(query, params).fetchall()

        results = []
        for r in rows:
            combo = sorted(r["raw_comp"].split("|"))
            results.append({
                "combo": " + ".join(combo),
                "brawlers": combo,
                "total": r["total"],
                "wins": r["wins"],
                "win_rate": r["win_rate"],
            })
        # Re-sort after normalization (in case raw order affected grouping)
        # Actually we should normalize BEFORE grouping. Let's do it in Python
        # since SQL GROUP_CONCAT order isn't guaranteed to be sorted.
        return self._merge_normalized_combos(results, min_sample, limit)

    @staticmethod
    def _merge_normalized_combos(results: list[dict], min_sample: int, limit: int) -> list[dict]:
        """Merge combos that are the same after sorting brawler names."""
        merged: dict[str, dict] = {}
        for r in results:
            key = r["combo"]
            if key in merged:
                merged[key]["total"] += r["total"]
                merged[key]["wins"] += r["wins"]
            else:
                merged[key] = {
                    "combo": r["combo"],
                    "brawlers": r["brawlers"],
                    "total": r["total"],
                    "wins": r["wins"],
                }
        for v in merged.values():
            v["win_rate"] = round(100.0 * v["wins"] / v["total"], 2) if v["total"] else 0

        out = [v for v in merged.values() if v["total"] >= min_sample]
        out.sort(key=lambda x: x["win_rate"], reverse=True)
        return out[:limit]

    # -- 3. Brawler matchup matrix --

    def matchup_win_rates(
        self,
        mode: str | None = None,
        min_trophies: int | None = None,
        min_sample: int = 5,
        limit: int = 200,
        **filters,
    ) -> list[dict]:
        """Win rate of brawler_A (my team) against brawler_B (opposing team).

        Returns rows of {brawler_a, brawler_b, total, wins, win_rate}.
        """
        where, params, pj = self._where_clause(mode=mode, min_trophies=min_trophies, **filters)
        # Matchup self-join uses bp alias 'a', so player join goes on 'a'
        pjoin_a = "LEFT JOIN players p ON a.player_tag = p.tag" if pj else ""

        query = f"""
            SELECT
                a.brawler_name as brawler_a,
                opp.brawler_name as brawler_b,
                COUNT(*) as total,
                SUM(CASE WHEN a.result = 'victory' THEN 1 ELSE 0 END) as wins,
                ROUND(100.0 * SUM(CASE WHEN a.result = 'victory' THEN 1 ELSE 0 END) / COUNT(*), 2) as win_rate
            FROM battle_players a
            JOIN battle_players opp
                ON a.battle_id = opp.battle_id AND a.team_index != opp.team_index
            JOIN battles b ON a.battle_id = b.battle_id
            {pjoin_a}
            WHERE {where}
              AND a.result IN ('victory', 'defeat')
            GROUP BY a.brawler_name, opp.brawler_name
            HAVING COUNT(*) >= ?
            ORDER BY a.brawler_name, win_rate DESC
            LIMIT ?
        """
        params.extend([min_sample, limit])
        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    # -- 4. Brawler synergy matrix --

    def synergy_win_rates(
        self,
        mode: str | None = None,
        min_trophies: int | None = None,
        min_sample: int = 5,
        limit: int = 200,
        **filters,
    ) -> list[dict]:
        """Win rate when brawler_A and brawler_B are on the same team.

        Returns rows of {brawler_a, brawler_b, total, wins, win_rate}.
        Pairs are normalized (A < B alphabetically) to avoid double-counting.
        """
        where, params, pj = self._where_clause(mode=mode, min_trophies=min_trophies, **filters)
        pjoin_a = "LEFT JOIN players p ON a.player_tag = p.tag" if pj else ""

        query = f"""
            SELECT
                a.brawler_name as brawler_a,
                b_ally.brawler_name as brawler_b,
                COUNT(*) as total,
                SUM(CASE WHEN a.result = 'victory' THEN 1 ELSE 0 END) as wins,
                ROUND(100.0 * SUM(CASE WHEN a.result = 'victory' THEN 1 ELSE 0 END) / COUNT(*), 2) as win_rate
            FROM battle_players a
            JOIN battle_players b_ally
                ON a.battle_id = b_ally.battle_id
                AND a.team_index = b_ally.team_index
                AND a.brawler_name < b_ally.brawler_name
            JOIN battles b ON a.battle_id = b.battle_id
            {pjoin_a}
            WHERE {where}
              AND a.result IN ('victory', 'defeat')
            GROUP BY a.brawler_name, b_ally.brawler_name
            HAVING COUNT(*) >= ?
            ORDER BY win_rate DESC
            LIMIT ?
        """
        params.extend([min_sample, limit])
        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    # -- 5. Brawler scores (statistical model) --

    def brawler_scores(self, mode: str | None = None) -> list[dict]:
        """Comprehensive brawler scoring with Wilson CI and tier-adjusted WR.

        Delegates to models.score_brawlers.
        """
        from .models import score_brawlers
        # Extract the DB path from our connection
        db_path = self._conn.execute("PRAGMA database_list").fetchone()[2]
        return score_brawlers(db_path=db_path, mode=mode)

    # -- 6. Brawler win rates by trophy tier --

    def brawler_win_rates_by_tier(
        self,
        mode: str | None = None,
        min_sample: int = 10,
        limit: int = 101,
    ) -> dict[str, list[dict]]:
        """Win rates for each brawler, broken down by trophy tier.

        Only uses 'ranked' battles where brawler_trophies is meaningful.
        Returns {tier_name: [brawler rows]}.
        """
        result = {}
        for tier_name, lo, hi in TROPHY_TIERS:
            rows = self.brawler_win_rates(
                mode=mode,
                battle_type="ranked",
                trophy_tier=tier_name,
                min_sample=min_sample,
                limit=limit,
            )
            if rows:
                result[tier_name] = rows
        return result

    # -- Summary --

    def summary(self) -> dict:
        """Quick overview of what's in the database."""
        total_battles = self._conn.execute("SELECT COUNT(*) FROM battles").fetchone()[0]
        team_battles = self._conn.execute(
            "SELECT COUNT(*) FROM battles WHERE is_showdown = 0"
        ).fetchone()[0]
        showdown_battles = total_battles - team_battles
        total_players = self._conn.execute("SELECT COUNT(*) FROM players").fetchone()[0]
        total_brawlers = self._conn.execute("SELECT COUNT(*) FROM brawlers").fetchone()[0]

        modes = self._conn.execute(
            "SELECT mode, COUNT(*) as cnt FROM battles GROUP BY mode ORDER BY cnt DESC"
        ).fetchall()
        btypes = self._conn.execute(
            "SELECT battle_type, COUNT(*) as cnt FROM battles GROUP BY battle_type ORDER BY cnt DESC"
        ).fetchall()

        time_range = self._conn.execute(
            "SELECT MIN(battle_time_iso), MAX(battle_time_iso) FROM battles"
        ).fetchone()

        return {
            "total_battles": total_battles,
            "team_battles": team_battles,
            "showdown_battles": showdown_battles,
            "total_players": total_players,
            "total_brawlers": total_brawlers,
            "mode_distribution": {r["mode"]: r["cnt"] for r in modes},
            "battle_type_distribution": {r["battle_type"]: r["cnt"] for r in btypes},
            "earliest_battle": time_range[0],
            "latest_battle": time_range[1],
        }

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
