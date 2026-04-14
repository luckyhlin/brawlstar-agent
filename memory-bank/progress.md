# Progress

## Current Phase

**API Battle Analytics**: Pipeline built and running. First collection pass done (4,628 battles from 200 top global players). Analytics queries operational.

## Done

### Infrastructure
- Machine assessed, project on disk2 (295GB free NVMe)
- Memory bank + cursor rule for cross-session continuity
- Python env via uv (3.12, opencv 4.13, numpy, yt-dlp, tesseract, matplotlib)

### Emulator (dead end)
- AVD: game installs but Supercell anti-cheat self-kills (DEC-003)
- Genymotion: ARM translation removed in v3.9, can't install game (DEC-004)
- Waydroid: needs Wayland + kernel modules we don't have

### Data Pipeline
- 24 YouTube clips downloaded (batch + manual)
- Frames extracted at 2fps, 480 sampled frames reviewed via browser hub
- 308 gameplay frames exported to `datasets/gameplay_cropped/` (20 clips, 12 resolutions)
- 100 brawler portraits (bordered + borderless) from BrawlAPI

### Brawl Stars API
- API key obtained and stored in `api.env` (IP-locked JWT, git-ignored)
- Full API documentation written: `docs/brawlstars-api.md` (7 endpoints)
- Live example responses dumped to `docs/api-examples/` (9 files)
- 101 brawlers cataloged, 25-battle log format analyzed, event rotation mapped

### Perception Baseline
- Game mode detection: showdown, brawl_ball, gem_grab, heist (heuristic, decent)
- UI region calibration: normalized coordinates, visual overlays verified
- OCR (tesseract): timer extraction works well on Brawl Ball (95%+ confidence), poor on Showdown (different UI layout)
- Character matching: color histogram baseline — **does not work** (distances ~0.85, not discriminative)
- Blob detection: finds saturated regions — **noisy**, picks up UI elements too

## Not Done / Known Gaps

- **Brawler identification**: needs a real pipeline (classifier or embedding), not color histograms
- **OCR verification**: results not human-verified yet
- **Showdown text parsing**: "Brawlers left: N" in different region, not handled
- **Minimap parsing**: not started
- **Structured game-state extraction**: not started
- **More data**: 308 frames is small, pipeline can scale easily

## Session Log

### Session 1 — 2025-03-29
- Machine inspection, memory bank created, emulator decision (AVD)

### Session 2 — 2025-03-29
- Android SDK installed, AVD booted, Brawl Stars anti-cheat blocks it
- Genymotion tried, no ARM translation
- Pivoted to YouTube gameplay, built data pipeline, first 13 frames cropped

### Session 3 — 2026-04-03
- Batch frame extraction, headless review prep, crop export helper

### Session 4 — 2026-04-06
- Browser review hub built, all 24 clips reviewed by user
- 308 gameplay frames exported
- Perception pipeline built and run: mode detection, OCR, blob detection, character matching
- Tesseract installed, OCR re-run with results

### Session 5 — 2026-04-13
- Brawl Stars API key obtained, documented 7 endpoints in `docs/brawlstars-api.md`
- Fetched and saved live API responses for all endpoints into `docs/api-examples/`
- Built full battle analytics pipeline (DEC-006):
  - `api_client.py`: rate-limited API wrapper with auth, retry, backoff
  - `db.py`: SQLite schema with dedup, battle normalization, player tracking
  - `collector.py`: seed rankings → fetch battlelogs → snowball tags
  - `analytics.py`: win rate per brawler, combo, matchup matrix, synergy matrix
  - `scripts/collect-battles.py` + `scripts/analyze-battles.py` CLI tools
- First collection: 200 global top players, 4,628 battles, 19,845 discovered players
- Analytics validated: all 4 query types produce meaningful results
