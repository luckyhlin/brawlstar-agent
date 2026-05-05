#!/usr/bin/env python3
"""Crawl battlelogs for a small set of pinned player tags.

The bulk crawler (scripts/collect-battles.py) prioritizes top-trophy players
and snowballs from there. With ~500k tags discovered, low-trophy tags you
care about (your own account, friends, players under inspection) effectively
never get fetched because they're far down the trophy ranking.

This script complements the bulk crawler by ALWAYS crawling tags listed in
`data/pinned_tags.txt`. The file is gitignored and the parser is shared with
the dashboard (see src/brawlstar_agent/collector.py::load_pinned_tags for
the format spec).

If the file is missing or empty, this script logs and exits with status 0.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from brawlstar_agent.api_client import APIError, BrawlStarsAPI
from brawlstar_agent.collector import PINNED_TAGS_FILE, load_pinned_tags
from brawlstar_agent.db import BrawlDB

log = logging.getLogger("collect-pinned")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    tags = load_pinned_tags()
    if not tags:
        log.info("No pinned tags configured at %s", PINNED_TAGS_FILE)
        return 0

    log.info("Crawling %d pinned tag(s)", len(tags))

    db = BrawlDB()
    api = BrawlStarsAPI(requests_per_second=1.0)

    total_new = 0
    total_dupes = 0
    failures: list[tuple[str, str]] = []

    try:
        for tag in tags:
            try:
                profile = api.get_player(tag)
                db.upsert_player_profile(profile)
            except APIError as exc:
                log.warning("%s profile: %s", tag, exc)

            try:
                battles = api.get_battlelog(tag)
            except APIError as exc:
                log.error("%s battlelog: %s", tag, exc)
                failures.append((tag, str(exc)))
                continue

            new, dupes = db.insert_battles(battles, tag)
            db.mark_battlelog_fetched(tag)
            db._conn.commit()
            total_new += new
            total_dupes += dupes
            log.info(
                "%s: fetched %d battles, %d new, %d dupes",
                tag, len(battles), new, dupes,
            )

        log.info(
            "Done. tags=%d new=%d dupes=%d failures=%d api_requests=%d",
            len(tags), total_new, total_dupes, len(failures), api.total_requests,
        )
        return 0 if not failures else 1
    finally:
        api.close()
        db.close()


if __name__ == "__main__":
    sys.exit(main())
