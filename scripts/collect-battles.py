#!/usr/bin/env python3
"""CLI for collecting Brawl Stars battle data from the API.

Usage:
    # Full pipeline: seed brawlers + rankings + fetch battlelogs
    PYTHONPATH=src uv run python scripts/collect-battles.py

    # Snowball: fetch battlelogs for discovered tags (the main scaling method)
    PYTHONPATH=src uv run python scripts/collect-battles.py --collect-only --battlelog-limit 1000 --rps 2

    # Seed rankings only (no battlelog fetching)
    PYTHONPATH=src uv run python scripts/collect-battles.py --seed-only

    # Fetch player profiles (trophies/rank/level) for players missing that data
    PYTHONPATH=src uv run python scripts/collect-battles.py --profiles --profile-limit 500

    # Add country rankings for diversity
    PYTHONPATH=src uv run python scripts/collect-battles.py --countries global US KR BR

    # Faster rate (if you're confident about rate limits)
    PYTHONPATH=src uv run python scripts/collect-battles.py --rps 2.0
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from brawlstar_agent.api_client import BrawlStarsAPI
from brawlstar_agent.collector import Collector
from brawlstar_agent.db import BrawlDB, DEFAULT_DB_PATH


def main():
    parser = argparse.ArgumentParser(description="Collect Brawl Stars battle data")
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB_PATH), help="SQLite database path")
    parser.add_argument("--rps", type=float, default=1.0, help="API requests per second (default: 1.0)")
    parser.add_argument("--ranking-limit", type=int, default=200, help="Players to fetch per ranking")
    parser.add_argument("--battlelog-limit", type=int, default=200, help="Max players to fetch battlelogs for")
    parser.add_argument("--countries", nargs="+", default=["global"], help="Country codes for rankings")
    parser.add_argument("--older-than", type=float, default=6.0, help="Re-fetch battlelogs older than N hours")
    parser.add_argument("--profile-limit", type=int, default=200, help="Max players to fetch profiles for")
    parser.add_argument("--seed-only", action="store_true", help="Only seed from rankings, don't fetch battlelogs")
    parser.add_argument("--collect-only", action="store_true", help="Only fetch battlelogs for existing tags")
    parser.add_argument("--profiles", action="store_true", help="Fetch player profiles (trophies/rank) for discovered tags")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug-level logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    db = BrawlDB(args.db)
    api = BrawlStarsAPI(requests_per_second=args.rps)

    try:
        collector = Collector(api, db)

        if args.profiles:
            stats = collector.collect_profiles(max_players=args.profile_limit)
        elif args.collect_only:
            stats = collector.collect_battlelogs(
                max_players=args.battlelog_limit,
                older_than_hours=args.older_than,
            )
        elif args.seed_only:
            collector.seed_brawlers()
            seeded = collector.seed_rankings(
                countries=args.countries, limit=args.ranking_limit
            )
            stats = {
                "players_seeded": seeded,
                "total_players_in_db": db.get_player_count(),
            }
        else:
            stats = collector.run_full_pipeline(
                countries=args.countries,
                ranking_limit=args.ranking_limit,
                battlelog_limit=args.battlelog_limit,
                older_than_hours=args.older_than,
            )

        print("\n=== Collection Results ===")
        print(json.dumps(stats, indent=2))
        print(f"\nDB: {args.db}")
        print(f"Total API requests this session: {api.total_requests}")
        print(f"Total battles in DB: {db.get_battle_count()}")
        print(f"Total players in DB: {db.get_player_count()}")
        print(f"\nMode distribution:")
        for mode, cnt in db.get_mode_distribution():
            print(f"  {mode:20s} {cnt:>6d}")

    finally:
        api.close()
        db.close()


if __name__ == "__main__":
    main()
