"""SQLite storage layer for Brawl Stars battle data."""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "brawlstars.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS brawlers (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS players (
    tag                 TEXT PRIMARY KEY,
    name                TEXT,
    trophies            INTEGER,
    highest_trophies    INTEGER,
    exp_level           INTEGER,
    club_name           TEXT,
    source              TEXT,          -- how we discovered: 'rankings', 'battlelog', 'manual'
    last_battlelog_at   TEXT,          -- ISO timestamp of last battlelog fetch
    last_profile_at     TEXT,          -- ISO timestamp of last profile fetch
    created_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS battles (
    battle_id       TEXT PRIMARY KEY,  -- battleTime + sorted player tags for dedup
    battle_time     TEXT NOT NULL,     -- original API timestamp
    battle_time_iso TEXT NOT NULL,     -- parsed ISO 8601 for easy querying
    event_id        INTEGER,
    mode            TEXT NOT NULL,
    map             TEXT,
    battle_type     TEXT,              -- 'ranked', 'soloRanked', 'friendly', etc.
    duration        INTEGER,
    is_showdown     INTEGER NOT NULL DEFAULT 0,
    star_player_tag TEXT
);

CREATE TABLE IF NOT EXISTS battle_players (
    battle_id       TEXT NOT NULL REFERENCES battles(battle_id),
    player_tag      TEXT NOT NULL,
    team_index      INTEGER NOT NULL,  -- 0/1 for teams; 0..9 rank for showdown
    brawler_id      INTEGER NOT NULL,
    brawler_name    TEXT NOT NULL,
    brawler_power   INTEGER,
    brawler_trophies INTEGER,
    is_star_player  INTEGER NOT NULL DEFAULT 0,
    result          TEXT,              -- 'victory'/'defeat' for team; rank (1-10) for showdown
    trophy_change   INTEGER,
    PRIMARY KEY (battle_id, player_tag)
);

CREATE TABLE IF NOT EXISTS collection_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    action      TEXT NOT NULL,        -- 'seed_rankings', 'fetch_battlelog', 'fetch_profile'
    target      TEXT,                 -- player tag or country code
    status      TEXT NOT NULL,        -- 'ok', 'error', 'not_found'
    detail      TEXT,
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_battle_players_brawler   ON battle_players(brawler_id);
CREATE INDEX IF NOT EXISTS idx_battle_players_player    ON battle_players(player_tag);
CREATE INDEX IF NOT EXISTS idx_battle_players_team      ON battle_players(battle_id, team_index);
CREATE INDEX IF NOT EXISTS idx_battle_players_result    ON battle_players(result, brawler_name);
CREATE INDEX IF NOT EXISTS idx_battles_mode             ON battles(mode);
CREATE INDEX IF NOT EXISTS idx_battles_mode_showdown    ON battles(mode, is_showdown);
CREATE INDEX IF NOT EXISTS idx_battles_time             ON battles(battle_time_iso);
CREATE INDEX IF NOT EXISTS idx_players_trophies         ON players(trophies DESC);
CREATE INDEX IF NOT EXISTS idx_players_last_battlelog   ON players(last_battlelog_at);
"""


def parse_battle_time(raw: str) -> str:
    """Convert API timestamp '20260413T185800.000Z' to ISO 8601."""
    try:
        dt = datetime.strptime(raw, "%Y%m%dT%H%M%S.%fZ").replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except ValueError:
        return raw


def _make_battle_id(battle_time: str, player_tags: list[str]) -> str:
    """Create a deterministic dedup key from battle time + sorted player tags."""
    sorted_tags = "|".join(sorted(player_tags))
    return f"{battle_time}:{sorted_tags}"


class BrawlDB:
    """SQLite database manager for battle data."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(SCHEMA_SQL)
        self._conn.commit()

    # -- Brawlers --

    def upsert_brawlers(self, brawlers: list[dict]) -> int:
        """Insert or update canonical brawler list from /brawlers API."""
        count = 0
        for b in brawlers:
            self._conn.execute(
                "INSERT INTO brawlers (id, name) VALUES (?, ?) ON CONFLICT(id) DO UPDATE SET name=excluded.name",
                (b["id"], b["name"]),
            )
            count += 1
        self._conn.commit()
        return count

    # -- Players --

    def upsert_player_tag(
        self,
        tag: str,
        name: str | None = None,
        source: str = "battlelog",
        trophies: int | None = None,
    ) -> None:
        """Insert a player tag if not exists, or update name/trophies if provided."""
        self._conn.execute(
            """INSERT INTO players (tag, name, trophies, source)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(tag) DO UPDATE SET
                   name = COALESCE(excluded.name, players.name),
                   trophies = COALESCE(excluded.trophies, players.trophies)""",
            (tag, name, trophies, source),
        )

    def upsert_player_profile(self, profile: dict) -> None:
        """Update player row with full profile data."""
        tag = profile["tag"]
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """INSERT INTO players (tag, name, trophies, highest_trophies, exp_level, club_name, last_profile_at, source)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'profile')
               ON CONFLICT(tag) DO UPDATE SET
                   name = excluded.name,
                   trophies = excluded.trophies,
                   highest_trophies = excluded.highest_trophies,
                   exp_level = excluded.exp_level,
                   club_name = excluded.club_name,
                   last_profile_at = excluded.last_profile_at""",
            (
                tag,
                profile.get("name"),
                profile.get("trophies"),
                profile.get("highestTrophies"),
                profile.get("expLevel"),
                profile.get("club", {}).get("name"),
                now,
            ),
        )
        self._conn.commit()

    def mark_battlelog_fetched(self, tag: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "UPDATE players SET last_battlelog_at = ? WHERE tag = ?", (now, tag)
        )

    def get_tags_needing_fetch(self, older_than_hours: float = 6.0, limit: int = 200) -> list[str]:
        """Get player tags whose battlelog hasn't been fetched recently."""
        cutoff = datetime.now(timezone.utc).isoformat()
        # Calculate cutoff time
        from datetime import timedelta
        cutoff_dt = datetime.now(timezone.utc) - timedelta(hours=older_than_hours)
        cutoff = cutoff_dt.isoformat()
        rows = self._conn.execute(
            """SELECT tag FROM players
               WHERE last_battlelog_at IS NULL OR last_battlelog_at < ?
               ORDER BY trophies DESC NULLS LAST
               LIMIT ?""",
            (cutoff, limit),
        ).fetchall()
        return [r["tag"] for r in rows]

    def get_player_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM players").fetchone()[0]

    # -- Battles --

    def insert_battles(self, battles: list[dict], fetched_for_tag: str) -> tuple[int, int]:
        """Normalize and insert battles from a battlelog response.

        Returns (new_battles, skipped_dupes).
        """
        new = 0
        skipped = 0

        for entry in battles:
            battle_time = entry.get("battleTime", "")
            battle_time_iso = parse_battle_time(battle_time)
            event = entry.get("event", {})
            battle = entry.get("battle", {})
            mode = battle.get("mode", event.get("mode", "unknown"))
            is_showdown = mode in ("soloShowdown", "duoShowdown", "trioShowdown")

            all_tags = self._extract_all_tags(battle, is_showdown)
            if not all_tags:
                skipped += 1
                continue

            battle_id = _make_battle_id(battle_time, all_tags)

            # Skip if already stored
            exists = self._conn.execute(
                "SELECT 1 FROM battles WHERE battle_id = ?", (battle_id,)
            ).fetchone()
            if exists:
                skipped += 1
                continue

            star_tag = None
            star_player = battle.get("starPlayer")
            if star_player:
                star_tag = star_player.get("tag")

            self._conn.execute(
                """INSERT INTO battles
                   (battle_id, battle_time, battle_time_iso, event_id, mode, map,
                    battle_type, duration, is_showdown, star_player_tag)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    battle_id, battle_time, battle_time_iso,
                    event.get("id"), mode, event.get("map"),
                    battle.get("type"), battle.get("duration"),
                    int(is_showdown), star_tag,
                ),
            )

            # Insert player rows
            self._insert_battle_players(battle_id, battle, mode, is_showdown, fetched_for_tag)
            new += 1

        self._conn.commit()
        return new, skipped

    def _extract_all_tags(self, battle: dict, is_showdown: bool) -> list[str]:
        tags = []
        if "teams" in battle:
            for team in battle["teams"]:
                for p in team:
                    tags.append(p["tag"])
        elif "players" in battle:
            for p in battle["players"]:
                tags.append(p["tag"])
        return tags

    def _insert_battle_players(
        self, battle_id: str, battle: dict, mode: str, is_showdown: bool,
        fetched_for_tag: str = "",
    ) -> None:
        star_player = battle.get("starPlayer")
        star_tag = star_player.get("tag") if star_player else None
        trophy_change = battle.get("trophyChange")

        if "teams" in battle:
            result = battle.get("result", "unknown")

            # Determine which team the fetched player is on.
            # battle.result is from that player's perspective.
            fetched_team_idx = 0
            for ti, team in enumerate(battle["teams"]):
                if any(p["tag"] == fetched_for_tag for p in team):
                    fetched_team_idx = ti
                    break

            for team_idx, team in enumerate(battle["teams"]):
                if team_idx == fetched_team_idx:
                    team_result = result
                else:
                    team_result = "defeat" if result == "victory" else "victory" if result == "defeat" else result

                for p in team:
                    brawler = p.get("brawler", {})
                    self._conn.execute(
                        """INSERT OR IGNORE INTO battle_players
                           (battle_id, player_tag, team_index, brawler_id, brawler_name,
                            brawler_power, brawler_trophies, is_star_player, result, trophy_change)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            battle_id, p["tag"], team_idx,
                            brawler.get("id", 0), brawler.get("name", "UNKNOWN"),
                            brawler.get("power"), brawler.get("trophies"),
                            int(p["tag"] == star_tag) if star_tag else 0,
                            team_result,
                            trophy_change,
                        ),
                    )
                    self.upsert_player_tag(p["tag"], p.get("name"))
        elif "players" in battle:
            result = battle.get("result", "unknown")
            rank = battle.get("rank")
            for i, p in enumerate(battle["players"]):
                brawler = p.get("brawler", {})
                if is_showdown:
                    p_result = str(rank) if rank else str(i + 1)
                else:
                    p_result = result
                self._conn.execute(
                    """INSERT OR IGNORE INTO battle_players
                       (battle_id, player_tag, team_index, brawler_id, brawler_name,
                        brawler_power, brawler_trophies, is_star_player, result, trophy_change)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        battle_id, p["tag"], i,
                        brawler.get("id", 0), brawler.get("name", "UNKNOWN"),
                        brawler.get("power"), brawler.get("trophies"),
                        int(p["tag"] == star_tag) if star_tag else 0,
                        p_result,
                        trophy_change if i == 0 else None,
                    ),
                )
                self.upsert_player_tag(p["tag"], p.get("name"))

    def log_collection(self, action: str, target: str, status: str, detail: str | None = None) -> None:
        self._conn.execute(
            "INSERT INTO collection_log (action, target, status, detail) VALUES (?, ?, ?, ?)",
            (action, target, status, detail),
        )
        self._conn.commit()

    # -- Stats / diagnostics --

    def get_battle_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM battles").fetchone()[0]

    def get_battle_player_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM battle_players").fetchone()[0]

    def get_mode_distribution(self) -> list[tuple[str, int]]:
        rows = self._conn.execute(
            "SELECT mode, COUNT(*) as cnt FROM battles GROUP BY mode ORDER BY cnt DESC"
        ).fetchall()
        return [(r["mode"], r["cnt"]) for r in rows]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
