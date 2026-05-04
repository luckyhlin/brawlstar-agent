# Progress

## Current Phase

**Production deploy**: Crawler is now hosted on a DigitalOcean droplet with a 6-hour systemd timer. Collection runs autonomously, accumulating fresh battle data without an always-on home machine. Local machine remains the dev environment; droplet pulls code via git (DEC-008).

Current dataset: **160,764 battles** across 11 modes, **553k+ player tags**, time range now caught up to **2026-05-04**.

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

### Session 6 — 2026-05-03 — Production deploy on DigitalOcean
- Provisioned DigitalOcean Basic droplet ($6/mo, Ubuntu 24.04 LTS, US region) — DEC-007
- Reserved IP `209.38.4.212` attached; BS API key whitelisted both reserved + anchor public IP `64.23.171.86` (DO Reserved IP is inbound-only; outbound uses the anchor)
- Hardened: non-root sudo user `lin`, key-only SSH, root login disabled, UFW (allow OpenSSH only), fail2ban with home-IP allowlist
- Defused needrestart auto-restart of services on `apt install` (`/etc/needrestart/needrestart.conf` → `restart = 'l'`)
- Python 3.12 + uv installed; project rsynced; `uv sync` succeeded after overriding the hardcoded `cache-dir` from `pyproject.toml` via `UV_CACHE_DIR` env var
- Made `api_client.py` portable across machines via `BRAWL_API_KEY_VAR` indirection — same `api.env` file works on both local (uses `BRAWL_STAR_API`) and droplet (uses `BRAWL_STAR_API_DO`)
- systemd service + 6h timer for `scripts/collect-battles.py --collect-only --battlelog-limit 1500 --rps 2`; Persistent=true so missed runs catch up
- First scheduled run: +35,860 new battles in ~5 min; total now **160,764 battles**, latest battle `2026-05-04` (caught up to today)
- Adopted local-primary git deploy workflow — DEC-008
- Cleaned `pyproject.toml`: removed hardcoded `cache-dir`, replaced with `UV_CACHE_DIR` env var per machine
- Wrote `docs/deployment.md` as a step-by-step runbook for fresh-VPS migration
- **Pitfalls captured** (now documented in deployment runbook):
  - DO Reserved IPs are inbound-only by default — must whitelist anchor public IP for outbound API calls
  - needrestart on Ubuntu 24.04 will auto-restart services mid-run on every `apt install`
  - fail2ban's first restart races with its socket; `sleep 2` before `fail2ban-client`
  - `pyproject.toml`'s `[tool.uv] cache-dir` is non-portable; use env var instead
  - `Type=oneshot` services block `systemctl start` until the run completes — use `--no-block` or run from another shell
