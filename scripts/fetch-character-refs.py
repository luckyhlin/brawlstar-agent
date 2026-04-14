#!/usr/bin/env python3
"""Download Brawl Stars character reference images from BrawlAPI + Brawlify CDN.

Uses the public BrawlAPI to get the full brawler list with numeric IDs,
then downloads bordered and borderless portraits from the Brawlify CDN.

Run with: uv run python scripts/fetch-character-refs.py
"""

import json
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path

PROJECT_ROOT = Path("/media/lin/disk2/brawlstar-agent")
REFS_DIR = PROJECT_ROOT / "datasets" / "character_refs"
PORTRAITS_DIR = REFS_DIR / "portraits"
INDEX_FILE = REFS_DIR / "brawlers_index.json"

API_URL = "https://api.brawlapi.com/v1/brawlers"


def fetch_brawler_list() -> list[dict]:
    """Fetch all brawlers from the BrawlAPI."""
    req = urllib.request.Request(API_URL, headers={"User-Agent": "brawlstar-agent/0.1"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())
    raw = data.get("list", data) if isinstance(data, dict) else data
    return raw


def download_file(url: str, dest: Path) -> bool:
    """Download a file, return True on success."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "brawlstar-agent/0.1"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            content = resp.read()
        if len(content) > 500:
            dest.write_bytes(content)
            return True
    except Exception:
        pass
    return False


def main():
    PORTRAITS_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching brawler list from BrawlAPI...")
    raw_brawlers = fetch_brawler_list()
    print(f"Found {len(raw_brawlers)} brawlers from API.\n")

    index = []
    downloaded = 0
    failed = []

    for b in raw_brawlers:
        brawler_id = b["id"]
        name = b.get("name", "Unknown").title()
        hash_name = b.get("hash", "")
        role = b.get("class", {}).get("name", "unknown") if isinstance(b.get("class"), dict) else "unknown"

        bordered_url = f"https://cdn.brawlify.com/brawlers/borders/{brawler_id}.png"
        borderless_url = f"https://cdn.brawlify.com/brawlers/borderless/{brawler_id}.png"

        bordered_file = PORTRAITS_DIR / f"{brawler_id}_bordered.png"
        borderless_file = PORTRAITS_DIR / f"{brawler_id}_borderless.png"

        ok_bordered = bordered_file.exists() or download_file(bordered_url, bordered_file)
        ok_borderless = borderless_file.exists() or download_file(borderless_url, borderless_file)
        ok = ok_bordered or ok_borderless

        entry = {
            "id": brawler_id,
            "name": name,
            "hash": hash_name,
            "role": role,
            "bordered_file": bordered_file.name if ok_bordered else None,
            "borderless_file": borderless_file.name if ok_borderless else None,
        }
        index.append(entry)

        if ok:
            downloaded += 1
            tag = "bordered" if ok_bordered else ""
            tag += ("+borderless" if ok_borderless else "") if ok_bordered else ("borderless" if ok_borderless else "")
            print(f"  OK  {name:20s} (id={brawler_id}) [{tag}]")
        else:
            failed.append(name)
            print(f"  FAIL {name:20s} (id={brawler_id})")

    with open(INDEX_FILE, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nDownloaded: {downloaded}/{len(raw_brawlers)} brawlers")
    print(f"Index saved to: {INDEX_FILE}")
    print(f"Portraits in: {PORTRAITS_DIR}")

    if failed:
        print(f"\nMissing ({len(failed)}): {', '.join(failed)}")


if __name__ == "__main__":
    main()
