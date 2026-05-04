"""Brawl Stars API client with rate limiting and retry logic."""

from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path
from urllib.parse import quote

import httpx
from dotenv import load_dotenv

log = logging.getLogger(__name__)

BASE_URL = "https://api.brawlstars.com/v1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _encode_tag(tag: str) -> str:
    """URL-encode a player/club tag (# -> %23)."""
    if tag.startswith("#"):
        return quote(tag, safe="")
    return tag


class RateLimiter:
    """Token-bucket style rate limiter with jitter."""

    def __init__(self, requests_per_second: float = 1.0):
        self.min_interval = 1.0 / requests_per_second
        self._last_request: float = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_request
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed + random.uniform(0.05, 0.15)
            time.sleep(sleep_time)
        self._last_request = time.monotonic()


class BrawlStarsAPI:
    """Thin wrapper around the Brawl Stars REST API.

    Handles auth, rate limiting, and exponential backoff on 429/5xx.
    """

    def __init__(
        self,
        api_key: str | None = None,
        requests_per_second: float = 1.0,
        max_retries: int = 5,
    ):
        if api_key is None:
            api_key = self._load_key()
        self._client = httpx.Client(
            base_url=BASE_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0,
        )
        self._limiter = RateLimiter(requests_per_second)
        self._max_retries = max_retries
        self._total_requests = 0

    @staticmethod
    def _load_key() -> str:
        env_path = PROJECT_ROOT / "api.env"
        load_dotenv(env_path)
        # Per-machine override: set BRAWL_API_KEY_VAR=BRAWL_STAR_API_DO on the
        # DO droplet (or any other host with a different IP-locked key) so the
        # same api.env can carry multiple keys without code changes.
        var_name = os.getenv("BRAWL_API_KEY_VAR", "BRAWL_STAR_API")
        key = os.getenv(var_name)
        if not key:
            raise RuntimeError(f"{var_name} not found in {env_path}")
        return key

    def _request(self, path: str, params: dict | None = None) -> dict | list:
        """Make a rate-limited GET request with retry on transient errors."""
        for attempt in range(self._max_retries):
            self._limiter.wait()
            self._total_requests += 1
            try:
                resp = self._client.get(path, params=params)
            except httpx.TransportError as exc:
                log.warning("Transport error on %s (attempt %d): %s", path, attempt + 1, exc)
                self._backoff(attempt)
                continue

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code == 429:
                log.warning("Rate limited (429) on %s, backing off (attempt %d)", path, attempt + 1)
                self._backoff(attempt, base=5.0)
                continue

            if resp.status_code >= 500:
                log.warning("Server error %d on %s (attempt %d)", resp.status_code, path, attempt + 1)
                self._backoff(attempt)
                continue

            # 4xx client errors (except 429) are not retryable
            body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            reason = body.get("reason", resp.status_code)
            message = body.get("message", resp.text[:200])
            log.error("API error %s: %s (path=%s)", reason, message, path)
            raise APIError(resp.status_code, reason, message)

        raise APIError(429, "max_retries", f"Exhausted {self._max_retries} retries for {path}")

    @staticmethod
    def _backoff(attempt: int, base: float = 2.0) -> None:
        delay = base * (2 ** attempt) + random.uniform(0.5, 2.0)
        delay = min(delay, 120.0)
        log.info("Sleeping %.1fs before retry", delay)
        time.sleep(delay)

    # -- Player endpoints --

    def get_player(self, tag: str) -> dict:
        return self._request(f"/players/{_encode_tag(tag)}")

    def get_battlelog(self, tag: str) -> list[dict]:
        data = self._request(f"/players/{_encode_tag(tag)}/battlelog")
        return data.get("items", [])

    # -- Rankings --

    def get_player_rankings(
        self,
        country: str = "global",
        limit: int = 200,
        after: str | None = None,
        before: str | None = None,
    ) -> dict:
        params: dict = {"limit": limit}
        if after:
            params["after"] = after
        if before:
            params["before"] = before
        return self._request(f"/rankings/{country}/players", params=params)

    # -- Brawlers --

    def get_brawlers(self, limit: int | None = None) -> list[dict]:
        params = {"limit": limit} if limit else None
        data = self._request("/brawlers", params=params)
        return data.get("items", [])

    def get_brawler(self, brawler_id: int) -> dict:
        return self._request(f"/brawlers/{brawler_id}")

    # -- Events / Game modes --

    def get_game_modes(self) -> list[dict]:
        data = self._request("/gamemodes")
        return data.get("items", [])

    def get_event_rotation(self) -> list[dict]:
        return self._request("/events/rotation")

    # -- Diagnostics --

    @property
    def total_requests(self) -> int:
        return self._total_requests

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class APIError(Exception):
    def __init__(self, status: int, reason: str, message: str):
        self.status = status
        self.reason = reason
        self.message = message
        super().__init__(f"[{status}] {reason}: {message}")
