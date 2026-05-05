"""Battle data collection orchestrator.

Coordinates seeding from rankings, fetching battlelogs, and snowballing
new player tags from discovered battles.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .api_client import APIError, BrawlStarsAPI
from .db import BrawlDB

log = logging.getLogger(__name__)

# Single source of truth for the pinned-tags file (shared by collect-pinned
# crawler and the dashboard's watched-players tab).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PINNED_TAGS_FILE = PROJECT_ROOT / "data" / "pinned_tags.txt"


def load_pinned_tags() -> list[str]:
    """Read player tags from data/pinned_tags.txt.

    File format (gitignored, lives only on droplet & local data/):
        # Comments start with '# ' (hash + space). Whole-line comments are skipped.
        # One player tag per line, e.g.:
        #RYY9LJVL                # Personal account
        #280YJ0R80  # PolyMentos
        #2GY9CCUQR0 # psyduck
        #2CR92JQG92 # PolyMentosBB

    Both whole-line comments AND inline comments (anything after the tag) are
    supported. Returns a list of tags in file order; duplicates are deduped
    while preserving first occurrence.
    """
    if not PINNED_TAGS_FILE.exists():
        return []
    tags: list[str] = []
    seen: set[str] = set()
    for raw in PINNED_TAGS_FILE.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("# "):
            continue
        # First whitespace-separated token; everything after is treated as inline comment.
        token = line.split(None, 1)[0]
        if token.startswith("#") and len(token) >= 4 and token not in seen:
            tags.append(token)
            seen.add(token)
    return tags


class Collector:
    """Orchestrates data collection from the Brawl Stars API into SQLite."""

    def __init__(self, api: BrawlStarsAPI, db: BrawlDB):
        self.api = api
        self.db = db

    def seed_brawlers(self) -> int:
        """Fetch canonical brawler list and store in DB."""
        log.info("Fetching brawler catalog...")
        brawlers = self.api.get_brawlers()
        count = self.db.upsert_brawlers(brawlers)
        self.db.log_collection("seed_brawlers", "all", "ok", f"{count} brawlers")
        log.info("Stored %d brawlers", count)
        return count

    def seed_rankings(
        self,
        countries: list[str] | None = None,
        limit: int = 200,
    ) -> int:
        """Seed player tags from ranking leaderboards."""
        if countries is None:
            countries = ["global"]

        total = 0
        for country in countries:
            log.info("Fetching %s rankings (limit=%d)...", country, limit)
            try:
                data = self.api.get_player_rankings(country=country, limit=limit)
            except APIError as exc:
                log.error("Failed to fetch %s rankings: %s", country, exc)
                self.db.log_collection("seed_rankings", country, "error", str(exc))
                continue

            items = data.get("items", [])
            for p in items:
                self.db.upsert_player_tag(
                    tag=p["tag"],
                    name=p.get("name"),
                    source="rankings",
                    trophies=p.get("trophies"),
                )
            self.db._conn.commit()
            total += len(items)
            self.db.log_collection("seed_rankings", country, "ok", f"{len(items)} players")
            log.info("Seeded %d players from %s rankings", len(items), country)

        return total

    def collect_battlelogs(
        self,
        tags: list[str] | None = None,
        max_players: int = 200,
        older_than_hours: float = 6.0,
    ) -> dict:
        """Fetch battlelogs for players and store battles.

        If tags is None, picks players from DB whose battlelogs are stale.
        Returns stats dict.
        """
        if tags is None:
            tags = self.db.get_tags_needing_fetch(
                older_than_hours=older_than_hours, limit=max_players
            )

        stats = {
            "players_attempted": 0,
            "players_ok": 0,
            "players_error": 0,
            "new_battles": 0,
            "skipped_dupes": 0,
            "api_requests": self.api.total_requests,
        }

        log.info("Collecting battlelogs for %d players...", len(tags))

        for i, tag in enumerate(tags):
            stats["players_attempted"] += 1
            if (i + 1) % 25 == 0 or i == 0:
                log.info(
                    "Progress: %d/%d players, %d new battles so far, %d API requests",
                    i + 1, len(tags), stats["new_battles"], self.api.total_requests,
                )

            try:
                battles = self.api.get_battlelog(tag)
            except APIError as exc:
                if exc.reason == "notFound":
                    log.debug("Player %s not found, skipping", tag)
                else:
                    log.warning("Error fetching battlelog for %s: %s", tag, exc)
                stats["players_error"] += 1
                self.db.log_collection("fetch_battlelog", tag, "error", str(exc))
                continue

            new, skipped = self.db.insert_battles(battles, tag)
            self.db.mark_battlelog_fetched(tag)
            self.db._conn.commit()
            stats["players_ok"] += 1
            stats["new_battles"] += new
            stats["skipped_dupes"] += skipped
            self.db.log_collection(
                "fetch_battlelog", tag, "ok", f"{new} new, {skipped} dupes"
            )

        stats["api_requests"] = self.api.total_requests - stats["api_requests"]
        log.info(
            "Collection done: %d players OK, %d errors, %d new battles, %d dupes skipped",
            stats["players_ok"], stats["players_error"],
            stats["new_battles"], stats["skipped_dupes"],
        )
        return stats

    def collect_profiles(
        self,
        max_players: int = 200,
    ) -> dict:
        """Fetch player profiles for tags missing trophy data.

        Fills in trophies, highest_trophies, exp_level, club for discovered players.
        """
        rows = self.db._conn.execute(
            """SELECT tag FROM players
               WHERE trophies IS NULL AND last_profile_at IS NULL
               ORDER BY created_at ASC
               LIMIT ?""",
            (max_players,),
        ).fetchall()
        tags = [r[0] for r in rows]

        stats = {
            "profiles_attempted": 0,
            "profiles_ok": 0,
            "profiles_error": 0,
            "api_requests": self.api.total_requests,
        }

        log.info("Fetching profiles for %d players...", len(tags))

        for i, tag in enumerate(tags):
            stats["profiles_attempted"] += 1
            if (i + 1) % 50 == 0 or i == 0:
                log.info(
                    "Profiles: %d/%d, %d API requests",
                    i + 1, len(tags), self.api.total_requests,
                )

            try:
                profile = self.api.get_player(tag)
            except APIError as exc:
                if exc.reason == "notFound":
                    log.debug("Player %s not found, skipping", tag)
                else:
                    log.warning("Error fetching profile for %s: %s", tag, exc)
                stats["profiles_error"] += 1
                self.db.log_collection("fetch_profile", tag, "error", str(exc))
                continue

            self.db.upsert_player_profile(profile)
            stats["profiles_ok"] += 1

        stats["api_requests"] = self.api.total_requests - stats["api_requests"]
        log.info(
            "Profiles done: %d OK, %d errors",
            stats["profiles_ok"], stats["profiles_error"],
        )
        return stats

    def run_full_pipeline(
        self,
        countries: list[str] | None = None,
        ranking_limit: int = 200,
        battlelog_limit: int = 200,
        older_than_hours: float = 6.0,
    ) -> dict:
        """Run the full collection pipeline: brawlers -> rankings -> battlelogs.

        Returns combined stats.
        """
        results = {}

        brawler_count = self.seed_brawlers()
        results["brawlers_stored"] = brawler_count

        seeded = self.seed_rankings(countries=countries, limit=ranking_limit)
        results["players_seeded"] = seeded

        stats = self.collect_battlelogs(
            max_players=battlelog_limit,
            older_than_hours=older_than_hours,
        )
        results.update(stats)

        results["total_players_in_db"] = self.db.get_player_count()
        results["total_battles_in_db"] = self.db.get_battle_count()
        return results
