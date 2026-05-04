#!/usr/bin/env python3
"""Pre-compute all dashboard analytics and write to data/analytics_cache.json.

Heavy SQL (matchup self-join, synergy, brawler_scores, etc.) is expensive on a
1-CPU droplet. Running it on every dashboard launch is unusable. This script
runs the queries once and writes the result; the dashboard server reads the
cache for instant load.

Designed for periodic execution via systemd timer (every 1h is a good default).
The dashboard shows the cache's `computed_at` timestamp so users see freshness.

Watchdog: if compute exceeds ~30-45 min, something has degraded (DB grew too
much, missing index, query plan regression). The systemd unit
`brawl-analytics.service` sets `TimeoutStartSec=2700` (45 min) to hard-kill any
runaway run; the failed unit will appear in `systemctl --failed` and the
dashboard will paint the compute-time badge red if `computed_in_seconds > 2700`.

Usage:
    PYTHONPATH=src uv run python scripts/precompute-analytics.py

Exits with status 0 on success, 1 on failure.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from brawlstar_agent.dashboard_data import write_cache
from brawlstar_agent.db import DEFAULT_DB_PATH


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        cache = write_cache(DEFAULT_DB_PATH)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logging.exception("Precompute failed: %s", exc)
        return 1

    print(
        f"OK · {cache['battle_count']:,} battles · "
        f"compute {cache['computed_in_seconds']}s · "
        f"DB {cache['db_size_mb']} MB"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
