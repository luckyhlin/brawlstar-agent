# Decisions

## DEC-001: Emulator — All Paths Failed
- AVD: anti-cheat self-kills Brawl Stars (x86 CPU detected, obfuscated integrity check)
- Genymotion: ARM translation removed in v3.9, game can't install
- Waydroid: needs Wayland + binder/ashmem modules, not viable on X11
- **Outcome**: pivoted to YouTube gameplay recordings

## DEC-002: Project Location
- `/media/lin/disk2/brawlstar-agent/` — 295GB free NVMe
- Avoid /home (27GB) and / (8GB)

## DEC-003: Data Source — YouTube Gameplay
- Download with yt-dlp, extract frames with ffmpeg
- Human review via browser-based hub (auto-label + click to correct)
- Works well. 308 frames from 24 clips.
- Future: physical Android device for live capture if needed

## DEC-004: Brawler Identification — Needs Real Approach
- Color histogram matching against portrait catalog does not work
- Blob detection by saturation is too noisy
- **Next decision needed**: what approach for brawler ID?
  - Option A: train a small classifier (ResNet/MobileNet) on labeled in-game crops
  - Option B: YOLO-style object detector fine-tuned on Brawl Stars frames
  - Option C: contrastive embeddings (portrait → in-game matching)
  - All require labeled training data (brawler crops with identity labels)

## DEC-006: API-Based Battle Analytics Pipeline
- CV-based brawler identification is hard (needs labeled data, training, etc.)
- Official API gives structured battle data directly: who played what brawler, win/loss, mode, map
- Pipeline: rankings → player tags → battlelogs → SQLite → analytics queries
- Storage: SQLite at `data/brawlstars.db` (16MB after 4,628 battles from 200 top players)
- Rate limiting: 1-2 req/s with exponential backoff, conservative to avoid bans
- Snowball discovery: each battlelog yields ~6 new player tags from opponents/teammates
- **First run**: 200 global top players → 4,628 battles → 19,845 discovered players
- Analytics: brawler win rates, team comp win rates, matchup matrix, synergy matrix
- **Outcome**: much faster path to useful insights than CV pipeline

## DEC-005: Dataset Backup Strategy — Deferred
- `datasets/` is gitignored for now (images too large for normal git)
- When a high-quality dataset is ready, we need a proper backup strategy
- Options to evaluate later:
  - Git LFS (track large files in git via pointers)
  - External storage + download script (e.g. tarball on local NAS or cloud, `scripts/fetch-dataset.sh`)
  - DVC (data version control — git-like tracking for large files)
- **No action now** — revisit when dataset quality justifies preservation
