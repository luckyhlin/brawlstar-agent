#!/usr/bin/env python3
"""Empirically verify the legacy team-result bug rate.

Approach (read-only — does NOT mutate the DB):

For each legacy battle (pre-2026-05-03), the bug we want to detect is:
team_index=0's stored `result` may not equal team_index=0's actual result —
they only agree if the player whose battlelog we originally fetched happened
to be on team 0.

Verification trick: pick ANY player Q from the battle's team 0. Re-fetch Q's
battlelog with the post-fix API client. If the same battle is still in Q's
last-25, the API returns `battle.result` from Q's perspective — which is
team 0's ACTUAL result. Compare to our stored value for team 0. Mismatch
proves the bug fired on this battle.

Sampling strategy:
- Pick legacy battles from 2026-04-25 onward (the bug's last week is the
  only window where re-fetch is still possible at all — the API only returns
  each player's last 25 battles, and older ones have aged out).
- For each candidate battle, prefer team-0 players who appear in FEW total
  battles in our DB (proxy for "low activity, this battle is still in their
  recent feed").

Output: a JSON report with the empirical bug rate and confidence interval,
plus per-battle details for spot-checking.

Usage:
    PYTHONPATH=src uv run python scripts/verify-bug.py --n-samples 80
    PYTHONPATH=src uv run python scripts/verify-bug.py --n-samples 80 --rps 1.5
    PYTHONPATH=src uv run python scripts/verify-bug.py --max-age-days 7 --since 2026-04-26
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from brawlstar_agent.api_client import APIError, BrawlStarsAPI  # noqa: E402
from brawlstar_agent.db import parse_battle_time  # noqa: E402

log = logging.getLogger("verify-bug")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson 95% CI for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    spread = z * math.sqrt((p * (1 - p) + z2 / (4 * n)) / n) / denom
    return (max(0.0, center - spread), center, min(1.0, center + spread))


def select_candidates(
    db_path: Path,
    *,
    since_iso: str,
    until_iso: str,
    max_total_battles: int,
    n_samples: int,
) -> list[tuple[str, str, str]]:
    """Return list of (battle_id, battle_time_iso, player_tag) candidates.

    Picks team_index=0 players from legacy battles, preferring players with
    few total battles in our DB (more likely the legacy battle is still in
    their last-25).
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            WITH p_count AS (
                SELECT player_tag, COUNT(*) AS n_total
                FROM battle_players
                GROUP BY player_tag
            )
            SELECT b.battle_id, b.battle_time_iso, bp.player_tag,
                   bp.result AS stored_result, p_count.n_total
            FROM battles b
            JOIN battle_players bp
              ON bp.battle_id = b.battle_id AND bp.team_index = 0
            JOIN p_count ON p_count.player_tag = bp.player_tag
            WHERE b.is_showdown = 0
              AND b.battle_time_iso >= ?
              AND b.battle_time_iso < ?
              AND p_count.n_total <= ?
              AND bp.result IN ('victory', 'defeat')
            ORDER BY p_count.n_total ASC, b.battle_time_iso DESC
            LIMIT ?
            """,
            (since_iso, until_iso, max_total_battles, n_samples),
        ).fetchall()
    finally:
        conn.close()
    return [(r["battle_id"], r["battle_time_iso"], r["player_tag"]) for r in rows]


def get_stored_result_for_team(db_path: Path, battle_id: str, team_index: int) -> str | None:
    """Look up the canonical stored result for a team in a battle."""
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            """
            SELECT result
            FROM battle_players
            WHERE battle_id = ? AND team_index = ? AND result IN ('victory', 'defeat')
            LIMIT 1
            """,
            (battle_id, team_index),
        ).fetchone()
    finally:
        conn.close()
    return row[0] if row else None


def find_matching_battle(api_battles: list[dict], target_battle_time_iso: str) -> dict | None:
    """Find the battle whose normalized time matches our stored battle_time_iso."""
    for entry in api_battles:
        bt_raw = entry.get("battleTime", "")
        bt_iso = parse_battle_time(bt_raw)
        if bt_iso == target_battle_time_iso:
            return entry
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/brawlstars.db")
    parser.add_argument(
        "--since", default="2026-04-25T00:00:00+00:00",
        help="Earliest legacy battle time to consider (ISO with offset).",
    )
    parser.add_argument(
        "--until", default="2026-05-03T01:00:00+00:00",
        help="Cutoff (the team-result bug was fixed at this time).",
    )
    parser.add_argument(
        "--n-samples", type=int, default=80,
        help="Max number of legacy battles to attempt verification on.",
    )
    parser.add_argument(
        "--max-total-battles", type=int, default=4,
        help="Skip candidates whose chosen player appears in more than this many "
             "battles total in our DB (low-activity heuristic).",
    )
    parser.add_argument("--rps", type=float, default=1.5)
    parser.add_argument(
        "--report-to", default="reports/verify_bug.json",
        help="JSON output path.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"DB not found: {db_path}")
        sys.exit(1)

    print(f"[verify-bug] sampling up to {args.n_samples} legacy battles "
          f"({args.since[:10]} to {args.until[:10]}) "
          f"from players with ≤ {args.max_total_battles} battles total in DB")

    candidates = select_candidates(
        db_path,
        since_iso=args.since,
        until_iso=args.until,
        max_total_battles=args.max_total_battles,
        n_samples=args.n_samples,
    )
    print(f"[verify-bug] picked {len(candidates)} candidate (battle, team-0 player) pairs")

    if not candidates:
        print("[verify-bug] no candidates; widen --max-total-battles or push --since earlier.")
        sys.exit(0)

    api = BrawlStarsAPI(requests_per_second=args.rps)

    n_attempted = 0
    n_player_not_found = 0
    n_player_no_match = 0   # player exists, but battle aged out of their feed
    n_recovered = 0          # legacy battle still in player's last-25
    n_match = 0              # stored = actual (no bug fire OR genuine no-flip)
    n_flipped = 0            # stored ≠ actual (bug fire confirmed)
    details: list[dict] = []

    t0 = time.time()
    for i, (battle_id, battle_time_iso, player_tag) in enumerate(candidates):
        n_attempted += 1
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(candidates)}] verified={n_recovered} flipped={n_flipped} "
                  f"match={n_match}  elapsed={time.time()-t0:.0f}s")

        try:
            api_battles = api.get_battlelog(player_tag)
        except APIError as exc:
            log.debug("API error for %s: %s", player_tag, exc)
            if exc.reason == "notFound":
                n_player_not_found += 1
            details.append({
                "battle_id": battle_id, "player": player_tag,
                "outcome": f"api_error:{exc.reason}",
            })
            continue

        match = find_matching_battle(api_battles, battle_time_iso)
        if match is None:
            n_player_no_match += 1
            details.append({
                "battle_id": battle_id, "player": player_tag,
                "outcome": "battle_aged_out",
                "fresh_battlelog_size": len(api_battles),
            })
            continue

        n_recovered += 1
        # API result is from this player's perspective; player is on team 0.
        actual_team0_result = match.get("battle", {}).get("result")
        stored_team0_result = get_stored_result_for_team(db_path, battle_id, team_index=0)

        if actual_team0_result not in ("victory", "defeat"):
            details.append({
                "battle_id": battle_id, "player": player_tag,
                "outcome": "non_binary_result",
                "actual": actual_team0_result, "stored": stored_team0_result,
            })
            continue

        if actual_team0_result == stored_team0_result:
            n_match += 1
            outcome = "match"
        else:
            n_flipped += 1
            outcome = "FLIPPED"
        details.append({
            "battle_id": battle_id, "player": player_tag,
            "battle_time_iso": battle_time_iso,
            "stored": stored_team0_result, "actual": actual_team0_result,
            "outcome": outcome,
        })

    elapsed = time.time() - t0
    api.close()

    flip_lo, flip_center, flip_hi = wilson_interval(n_flipped, n_recovered)

    print("\n=== Verification summary ===")
    print(f"  attempted:           {n_attempted}")
    print(f"  player-not-found:    {n_player_not_found}")
    print(f"  battle-aged-out:     {n_player_no_match}")
    print(f"  recoverable:         {n_recovered}")
    print(f"  bug-fired (FLIPPED): {n_flipped}")
    print(f"  matches (no-fire):   {n_match}")
    if n_recovered > 0:
        print(f"  empirical bug rate:  {flip_center:.1%} "
              f"  (95% CI: [{flip_lo:.1%}, {flip_hi:.1%}])")
        print(f"  expected from code analysis: ~50%")
    print(f"  elapsed: {elapsed:.0f}s, API requests: {api.total_requests}")

    out = {
        "attempted": n_attempted,
        "recovered": n_recovered,
        "flipped": n_flipped,
        "matches": n_match,
        "player_not_found": n_player_not_found,
        "battle_aged_out": n_player_no_match,
        "empirical_bug_rate": flip_center if n_recovered else None,
        "ci_95_lo": flip_lo if n_recovered else None,
        "ci_95_hi": flip_hi if n_recovered else None,
        "since": args.since,
        "until": args.until,
        "max_total_battles": args.max_total_battles,
        "elapsed_seconds": elapsed,
        "details": details,
    }

    out_path = Path(args.report_to)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[verify-bug] wrote {out_path}")


if __name__ == "__main__":
    main()
