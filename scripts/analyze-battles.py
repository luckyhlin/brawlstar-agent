#!/usr/bin/env python3
"""CLI for analyzing collected Brawl Stars battle data.

Usage:
    # Show database summary
    PYTHONPATH=src uv run python scripts/analyze-battles.py

    # Brawler win rates (all modes)
    PYTHONPATH=src uv run python scripts/analyze-battles.py --brawlers

    # Brawler win rates for a specific mode
    PYTHONPATH=src uv run python scripts/analyze-battles.py --brawlers --mode gemGrab

    # Best team compositions
    PYTHONPATH=src uv run python scripts/analyze-battles.py --combos

    # Matchup analysis: which brawlers beat which
    PYTHONPATH=src uv run python scripts/analyze-battles.py --matchups

    # Synergy analysis: which brawlers pair well together
    PYTHONPATH=src uv run python scripts/analyze-battles.py --synergies

    # Filter by trophy range
    PYTHONPATH=src uv run python scripts/analyze-battles.py --brawlers --min-trophies 500

    # All analyses at once
    PYTHONPATH=src uv run python scripts/analyze-battles.py --all
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from brawlstar_agent.analytics import BattleAnalytics
from brawlstar_agent.db import DEFAULT_DB_PATH


def print_table(rows: list[dict], columns: list[str], headers: list[str] | None = None):
    """Simple table printer."""
    if not rows:
        print("  (no data meets criteria)\n")
        return
    if headers is None:
        headers = columns

    widths = [max(len(h), max(len(str(r.get(c, ""))) for r in rows)) for c, h in zip(columns, headers)]

    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(f"  {header_line}")
    print(f"  {'  '.join('-' * w for w in widths)}")
    for r in rows:
        line = "  ".join(str(r.get(c, "")).ljust(w) for c, w in zip(columns, widths))
        print(f"  {line}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze Brawl Stars battle data")
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB_PATH))
    parser.add_argument("--mode", type=str, help="Filter by game mode (e.g. gemGrab, knockout)")
    parser.add_argument("--min-trophies", type=int, help="Minimum brawler trophies filter")
    parser.add_argument("--min-sample", type=int, default=5, help="Minimum sample size")
    parser.add_argument("--limit", type=int, default=30, help="Max rows per analysis")
    parser.add_argument("--brawlers", action="store_true", help="Show brawler win rates")
    parser.add_argument("--combos", action="store_true", help="Show combo win rates")
    parser.add_argument("--matchups", action="store_true", help="Show matchup win rates")
    parser.add_argument("--synergies", action="store_true", help="Show synergy win rates")
    parser.add_argument("--all", action="store_true", help="Run all analyses")
    parser.add_argument("--json", action="store_true", help="Output as JSON instead of tables")
    args = parser.parse_args()

    show_all = args.all or not (args.brawlers or args.combos or args.matchups or args.synergies)

    analytics = BattleAnalytics(args.db)
    filters = {}
    if args.mode:
        filters["mode"] = args.mode
    if args.min_trophies:
        filters["min_trophies"] = args.min_trophies

    # Always show summary
    summary = analytics.summary()
    print("\n=== Database Summary ===")
    print(f"  Battles:  {summary['total_battles']} ({summary['team_battles']} team, {summary['showdown_battles']} showdown)")
    print(f"  Players:  {summary['total_players']}")
    print(f"  Brawlers: {summary['total_brawlers']}")
    print(f"  Time range: {summary['earliest_battle'] or 'N/A'} .. {summary['latest_battle'] or 'N/A'}")
    print(f"  Modes: {json.dumps(summary['mode_distribution'], indent=4)}")
    print()

    filter_desc = ""
    if args.mode:
        filter_desc += f" mode={args.mode}"
    if args.min_trophies:
        filter_desc += f" min_trophies={args.min_trophies}"
    if filter_desc:
        print(f"  Filters:{filter_desc}\n")

    if args.brawlers or show_all:
        print("=== Brawler Win Rates (team modes) ===")
        rows = analytics.brawler_win_rates(min_sample=args.min_sample, limit=args.limit, **filters)
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            print_table(rows, ["brawler_name", "win_rate", "wins", "total"], ["Brawler", "Win%", "Wins", "Total"])

    if args.combos or show_all:
        print("=== Team Composition Win Rates ===")
        rows = analytics.combo_win_rates(min_sample=args.min_sample, limit=args.limit, **filters)
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            print_table(rows, ["combo", "win_rate", "wins", "total"], ["Composition", "Win%", "Wins", "Total"])

    if args.matchups or show_all:
        print("=== Matchup Win Rates (brawler A vs opposing brawler B) ===")
        rows = analytics.matchup_win_rates(min_sample=args.min_sample, limit=args.limit, **filters)
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            print_table(rows, ["brawler_a", "brawler_b", "win_rate", "wins", "total"],
                       ["Brawler", "vs Opponent", "Win%", "Wins", "Total"])

    if args.synergies or show_all:
        print("=== Synergy Win Rates (brawler A + brawler B on same team) ===")
        rows = analytics.synergy_win_rates(min_sample=args.min_sample, limit=args.limit, **filters)
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            print_table(rows, ["brawler_a", "brawler_b", "win_rate", "wins", "total"],
                       ["Brawler A", "Brawler B", "Win%", "Wins", "Total"])

    analytics.close()


if __name__ == "__main__":
    main()
