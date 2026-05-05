# Architecture

## Directory Layout

```
/media/lin/disk2/brawlstar-agent/
├── memory-bank/                # Cross-session memory (6 files)
├── .cursor/rules/              # memory-bank.mdc (auto-read/update)
├── pyproject.toml              # uv project (Python 3.12)
├── src/brawlstar_agent/        # Python package
│   ├── api_client.py           # Brawl Stars API wrapper (rate-limited, auth, retry)
│   ├── db.py                   # SQLite storage (schema, insert/dedup, queries)
│   ├── collector.py            # Data collection orchestrator (seed → fetch → snowball)
│   ├── analytics.py            # Win rate queries (brawler, combo, matchup, synergy)
│   ├── models.py               # Statistical baselines: Wilson CI, tier-adjusted WR, score_brawlers
│   ├── dashboard_data.py       # Shared analytics collection + cache read/write (used by dashboard + precompute cron)
│   ├── recommender/            # Phase-6 brawler-pick recommendation models (Session 8)
│   │   ├── dataset.py          # Clean-window loader + perspective-doubling + splits + name resolver
│   │   ├── features.py         # TeamFeaturizer (sparse + dense modes for sklearn / LGBM)
│   │   ├── baselines.py        # Random / TrophyOnly / Global / Mode / ModeMap baselines
│   │   ├── team_model.py       # LogRegTeamModel + LGBMTeamModel + evaluate + save/load
│   │   ├── inference.py        # rank_brawlers_for_map / complete_team / last_pick
│   │   ├── cv.py               # Sliding temporal-fold harness
│   │   └── topk_eval.py        # Top-K recommendation eval: hit@K, MRR, win-rate uplift
│   ├── capture.py              # Frame extraction, video reading
│   ├── crop.py                 # Auto-detect + batch crop game region
│   ├── perception.py           # Color analysis, template matching, MSER
│   ├── ui_regions.py           # Normalized UI regions, mode detection, overlays
│   ├── ocr.py                  # Tesseract OCR for timer/score/names
│   └── character_match.py      # Portrait matching baseline (weak)
├── scripts/
│   ├── download-gameplay.sh    # Single YouTube download
│   ├── download-batch.sh       # Batch download with dedup
│   ├── extract-frames.sh       # Video → frames (ffmpeg)
│   ├── extract-batch.sh        # Extract all clips
│   ├── auto-label-and-review.py # Auto-classify + generate HTML review pages
│   ├── generate-review-hub.py  # Browser-based review hub (local server)
│   ├── review-all.sh           # Terminal-guided sequential review
│   ├── crop-reviewed-frames.py # Export gameplay frames from manifests
│   ├── run-perception.py       # Full perception pipeline runner
│   ├── fetch-character-refs.py # Download brawler portraits from BrawlAPI
│   ├── collect-battles.py      # CLI: bulk crawler (every 6h on droplet via systemd)
│   ├── collect-pinned.py       # CLI: pinned-tags crawler (every 1h on droplet, reads data/pinned_tags.txt)
│   ├── precompute-analytics.py # CLI: writes data/analytics_cache.json (every 1h on droplet, with 45 min watchdog)
│   ├── analyze-battles.py      # CLI: run analytics queries on collected data
│   ├── dashboard.py            # Local web dashboard; --remote-cache HOST rsyncs cache from droplet on launch
│   ├── train-recommender.py    # End-to-end train + eval; --cutoff makes it re-runnable monthly
│   ├── analyze-recommender.py  # Plots, feature importance, DAMIAN deep-dive from saved report
│   └── eval-topk.py            # Top-K recommendation eval: hit@K, MRR, win uplift
├── capture/
│   ├── clips/                  # Downloaded YouTube videos
│   ├── frames/                 # Extracted frames + review manifests per clip
│   └── download_history.txt    # yt-dlp dedup archive
├── datasets/
│   ├── gameplay_cropped/       # 308 reviewed gameplay frames (21 clip dirs)
│   ├── character_refs/         # 200 brawler portraits + brawlers_index.json
│   └── perception/             # Pipeline outputs (calibration, ocr, summary)
├── data/
│   ├── brawlstars.db           # SQLite: battles, players, brawlers (git-ignored)
│   ├── pinned_tags.txt         # Pinned crawler input (git-ignored)
│   └── analytics_cache.json    # Pre-computed dashboard cache (git-ignored)
├── emulator/                   # Android SDK, AVD, Genymotion (legacy)
├── notebooks/
│   └── recommender_v1.ipynb    # Executed companion to docs/recommender-v1.md
├── models/
│   ├── recommender_v1.lgb.txt  # Trained LightGBM (git-ignored via *.bin et al.)
│   └── recommender_v1.meta.json
├── reports/
│   ├── recommender_v1.json     # Latest train+eval metrics
│   └── recommender_v1/         # Plots and DAMIAN deep-dive
├── logs/                       # Pipeline logs
└── docs/
    ├── data-sources.md         # Video/image data sources guide
    ├── brawlstars-api.md       # Full API reference (7 endpoints)
    ├── deployment.md           # Fresh-VPS deploy runbook (DigitalOcean)
    ├── analytics-notes.md      # Handoff guide for ML/analytics on the battle data
    ├── recommender-v1.md       # Phase-6 v1 methodology, results, and how-to-retrain
    └── api-examples/           # Live API responses (git-ignored)
```

## Data Pipeline

```
YouTube → yt-dlp → capture/clips/*.mp4
  → ffmpeg 2fps → capture/frames/<clip>/frame_*.jpg
  → auto-label-and-review.py → review_manifest.json + review.html
  → user reviews in browser hub (generate-review-hub.py)
  → crop-reviewed-frames.py → datasets/gameplay_cropped/<clip>/
  → run-perception.py → datasets/perception/ (calibration, ocr, characters)
```

## API Battle Analytics Pipeline (production, on droplet)

```
Rankings API → seed top global player tags → players table
  → for each tag: GET /players/{tag}/battlelog
  → normalize: 1 battle → battles row + 6 battle_players rows
  → snowball: discover new tags from opponents/teammates → players table
  → dedup on (battleTime + sorted tags) — INSERT OR IGNORE on battle_id

Three independent systemd timers run on the droplet (DEC-007):
  brawl-collect.timer         every 6h    bulk crawler (top trophies, 1500/run, --collect-only)
  brawl-collect-pinned.timer  every 1h    pinned tags from data/pinned_tags.txt
  brawl-analytics.timer       every 1h    precompute → data/analytics_cache.json (45 min watchdog)
```

DB: `data/brawlstars.db` (SQLite, WAL mode), gitignored.
Tables: brawlers, players, battles, battle_players, collection_log.

**Local viewing**: `scripts/dashboard.py --remote-cache brawl` rsyncs the precomputed cache
from the droplet and serves the HTML on `localhost:8765` — DB stays on droplet, no transfer
needed for the read path. See `docs/deployment.md` "View the dashboard" for full details.

**Data caveats** (read before training models): see `docs/analytics-notes.md`.

## Perception Pipeline (per frame)

```
Frame
 ├─ detect_game_mode() → showdown / brawl_ball / gem_grab / heist
 ├─ crop timer region → tesseract → "2:02" (works on brawl_ball/gem_grab)
 ├─ crop top-left region → tesseract → "100%" or "7"
 ├─ crop game area → detect_brawler_blobs() → candidate bboxes
 └─ per blob → compare color histogram vs 100 portraits → ranked guesses (WEAK)
```
