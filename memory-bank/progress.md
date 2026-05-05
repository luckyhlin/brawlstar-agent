# Progress

## Current Phase

**Phase 6 v1 shipped**: brawler-pick recommender trained on clean post-fix data. LightGBM team-completion model achieves AUC 0.730 (random) / 0.704 (temporal) — beats ModeMap baseline by ~3 AUC points. Inference helpers cover all three user scenarios (pre-draft tier list, mid-draft completion, last pick).

**Production crawl** continues on DigitalOcean droplet. Local DB now has **210k battles, latest 2026-05-05, 78k post-fix clean** (rsync'd this session).

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

### Session 7 — 2026-05-04 — Pinned tags + analytics precompute cache
- Identified two real gaps in the day-1 deploy:
  - Bulk crawler ranks by trophies and only fetches top-1500 stale tags per run; with 553k tags discovered, low-trophy tags (personal account, watchlist) effectively never get fetched.
  - Dashboard launch did full SQL self-joins on every load; ~5-15 min on the 1-CPU droplet, unusable.
- Added a **pinned-tags crawler** (`scripts/collect-pinned.py` + `data/pinned_tags.txt` gitignored) on a 1h timer. Always fetches a small explicit list independent of the bulk snowball.
- Refactored dashboard to support **precomputed analytics cache** at `data/analytics_cache.json`:
  - Extracted `collect_all_data()` and `_collect_personal_data()` from `scripts/dashboard.py` into `src/brawlstar_agent/dashboard_data.py` so both the server and the cron job share the schema.
  - New `scripts/precompute-analytics.py` runs queries and atomically writes the cache.
  - `scripts/dashboard.py` reads the cache by default; `--recompute` forces fresh, `--no-cache` skips entirely.
  - Dashboard header now paints cache age + compute time with color thresholds (orange >30 min compute, red >45 min).
- New systemd unit pair: `brawl-collect-pinned.service` + `.timer` (every 1h).
- New systemd unit pair: `brawl-analytics.service` + `.timer` (every 1h, `TimeoutStartSec=2700` watchdog at 45 min).
- Documented IP semantics (anchor IPv4 = outbound, reserved IPv4 = inbound, both WAN) and SSH config alias pattern in deployment.md.
- Verified the dashboard refactor compiles cleanly; cache helpers (`read_cache`, `write_cache`) imported successfully.
- Added `--remote-cache HOST` flag to `scripts/dashboard.py`: auto-rsyncs cache from a remote SSH host before launching. Makes the local-laptop workflow trivial — no SSH tunnel, no DB sync, just `uv run python scripts/dashboard.py --remote-cache brawl`. Falls back to local cache if rsync fails (offline-tolerant).

### Session 8 (continued) — 2026-05-04 — Recommender v1.1 — top-K eval, baselines floor, collector bug fix
- **Found and fixed a real collector bug**: `scripts/collect-battles.py --collect-only` (the systemd-timer command on the droplet) skipped `seed_brawlers()`. So the `brawlers` table was frozen at 101 rows since the May-3 deploy, even though new brawlers (DAMIAN id 16000104, first seen in battles **2026-04-24**) had been showing up in `battle_players`. Patch: `--collect-only` now seeds brawlers once at the start of every run. One extra API call per 6 hours, idempotent UPSERT.
- **Confirmed 4,820 `UNKNOWN` (id=0) brawler_player rows** — these are battles where the API returned a malformed/missing brawler dict. Edge case, not a model concern (recommender filters them out via vocab).
- **Added Random + TrophyOnly baselines** to make the AUC=0.5 floor explicit. **Critical finding**: `TrophyOnly` AUC = 0.497 — i.e., trophy diff alone is essentially random. Ranked matchmaking equalizes trophies within a match, so the AUC>0.5 we see in every other model is *all genuine brawler-pick signal*, not skill-tier leakage. Documents that the 0.65–0.73 AUC numbers are honestly earned.
- **Added top-K recommendation evaluation** (`recommender/topk_eval.py` + `scripts/eval-topk.py`). For each test row, mask team A's last pick, score all ~97 legal candidates, find rank of actually-played brawler. Reports hit@K, MRR, mean/median rank, and win-rate uplift conditioned on actual pick being in top-K.
- **Top-K results** (random split, n=2,500, last_pick mode):
  - Random: hit@1=0.001, hit@10=0.09 (matches K/N theoretical floor)
  - Global Wilson: hit@1=0.006, hit@3=0.19, hit@10=0.22 (hit@3 is decent — top global brawlers are picked often regardless of context)
  - ModeMap: hit@1=0.008, hit@5=0.15, hit@10=0.20
  - **LightGBM: hit@1=0.150 (15× random), hit@3=0.194, hit@5=0.225, hit@10=0.293**, MRR=0.205
  - Win uplift: when actual pick is in LightGBM top-1, team WR = 62.5% (vs 51.1% baseline, **+11.4 pp**)
  - Winners-only top-K (cleanest meta-quality test): LightGBM hit@1=**0.21** (20× random), hit@10=**0.39**
- **Player-flexibility tradeoff** (top-1 → top-10): hit rate doubles, expected WR drops 3 pp. Top-5 is a reasonable default.
- **Documented** why legacy bug labels can't be recovered: the bug preserves all internal invariants; trophy_change/result alignment is identical pre- and post-fix; verifying via re-crawl is only possible for very recent battles (<2 weeks) due to the 25-battle API window. Even verification doesn't enable correction — we still couldn't tell which specific battles are flipped.
- Updated `docs/recommender-v1.md` with new tables, top-K section, win-uplift section, collector-bug note, and the long-form legacy-bug-recovery answer.
- Saved `reports/recommender_v1_topk.json` for downstream consumption.

### Session 8 — 2026-05-04 — Brawler-pick recommender v1 (Phase 6 kickoff)
- Rsync'd droplet DB to local: 210k battles, latest 2026-05-05, **78k clean post-fix**.
- Tried to implement the legacy label-flip detector floated in `analytics-notes.md`. **It doesn't work** — the bug swaps team labels symmetrically and preserves the "1W+1L per battle" invariant. Recorded as **DEC-010**: legacy data is unusable for training and evaluation; strict post-2026-05-03 filter is the only path.
- Built `src/brawlstar_agent/recommender/` subpackage:
  - `dataset.py`: clean-window loader, both-perspectives expansion, temporal/random splits, `load_brawler_names` that falls back to `battle_players` for new brawlers (the `brawlers` table didn't have DAMIAN id 16000104).
  - `features.py`: `TeamFeaturizer` with sparse and dense modes (sklearn vs LGBM).
  - `baselines.py`: Global / Mode / ModeMap Wilson-CI baselines with the same `predict_proba` interface as the trained models.
  - `team_model.py`: `LogRegTeamModel` + `LGBMTeamModel` + `evaluate` + `save_model` / `load_model`.
  - `inference.py`: `rank_brawlers_for_map` (Monte Carlo over empirical teammates), `complete_team`, `last_pick`.
  - `cv.py`: sliding temporal-fold harness + `evaluate_models_on_folds`.
- Wrote `scripts/train-recommender.py` (re-runnable with `--cutoff` for the transferable-algorithm property) and `scripts/analyze-recommender.py` (plots + DAMIAN deep-dive).
- Built `notebooks/recommender_v1.ipynb` (executed, 116 KB with outputs baked in) and `docs/recommender-v1.md` (full methodology + how-to-retrain).
- **Random-split results** (n_test=21k):
  - Global Wilson AUC=0.655 → ModeMap AUC=0.697 → LightGBM **AUC=0.730**
  - LogReg with multi-hot teams underperforms ModeMap (0.661 vs 0.697); needs interaction features to compete.
- **Temporal CV** (4 sub-daily folds, ~2 days clean data): LightGBM AUC=0.704; ModeMap drops to 0.666 from 0.697 (random) — exactly the gap a temporal-vs-random comparison should expose.
- **Findings noted in doc**:
  - Release-meta inflation: DAMIAN (newest brawler, id 16000104) has **64.5% raw WR over 41k appearances** — most-played AND highest-WR brawler. Model recommends DAMIAN with extreme confidence; this is meta truth, not a model bug, but worth flagging for inference robustness.
  - LogReg without interaction features can't beat the ModeMap aggregate; tree models (LGBM) capture brawler×map and brawler-pair interactions for free.
  - Brier score 0.21 on LightGBM suggests reasonably calibrated probabilities for a v1.
- Added deps: `scikit-learn`, `lightgbm`, `pandas`, `pyarrow` via `uv add`.
- Per-machine `UV_CACHE_DIR=/media/lin/disk2/.uv-cache` is owned by root locally; worked around by setting `UV_CACHE_DIR=/media/lin/disk2/brawlstar-agent/.uv-cache-local` for this session (gitignored).

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
