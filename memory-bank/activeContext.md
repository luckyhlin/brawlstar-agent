# Active Context

## Current State

Crawler is **deployed and running on a DigitalOcean droplet** (DEC-007) on a 6-hour systemd timer. Local-primary workflow with git pull on the droplet (DEC-008). Data is fresh through today.

**Production droplet**: `lin@209.38.4.212` (reserved IP) / `64.23.171.86` (anchor / outbound IP)
**DB on droplet**: `~/brawlstar-agent/data/brawlstars.db` — independent copy from local; the droplet is the source of truth for production collection.

## Operating principle (DEC-009)

- **Remote**: routine/periodic — crawlers, scheduled analytics precompute, backups
- **Local**: interactive/heavy — dashboard, ad-hoc queries, ML training, exploration
- Cache flows local from droplet via `dashboard.py --remote-cache`; DB rsync only for ad-hoc deep-dives

## What Works
- **API client**: rate-limited (1-2 req/s), auth from api.env, retry with exponential backoff
- **SQLite storage**: normalized schema, dedup by (battleTime + sorted tags), WAL mode
- **Collector**: seed from rankings → fetch battlelogs → snowball discovered tags, resume-safe
- **Analytics**: 4 query types all working with filter support (mode, trophy range, time window)
  1. Brawler win rates per mode
  2. Team composition (3-brawler combo) win rates
  3. Matchup matrix (brawler A vs opposing brawler B)
  4. Synergy matrix (brawler A + brawler B on same team)

## Current Data (2026-05-02 snapshot)
- 124,904 battles (11 modes; brawlBall, soloShowdown, duoShowdown lead)
- 553,915 player tags discovered (snowball from top-200 seeds; only 6,168 have trophy data, 5,968 have full profile)
- 971,341 battle_player rows
- 101 brawlers cataloged
- Time range: 2021-12-16 .. 2026-04-15 (last ingestion 2026-04-15, ~17 days idle)
- DB size: 617 MB at `data/brawlstars.db` (largest consumers: battle_players + indexes ~50%)
- Dashboard: `scripts/dashboard.py` (local HTTP with portraits, 4 analysis tabs)

## Next Steps

**Phase 6 starts next session: brawler-pick recommendation model.** Required reading before any modeling work: [`docs/analytics-notes.md`](../docs/analytics-notes.md). It covers the schema, data caveats (especially the team-result bug — filter `battle_time_iso >= '2026-05-03T01:00:00Z'`), existing baselines (`models.py::score_brawlers`, matchup/synergy matrices), problem framing, and starting points.

Deferred / parallel tracks:
1. **Cloudflare R2 nightly backups** — free 10 GB tier, `sqlite3 .backup` → zstd → rclone copyto. Protects against droplet failure. ~15 min when wanted.
2. **Country diversity in seeding** — currently global-only; adding US/KR/BR/JP rankings could broaden meta coverage.
3. **First-month observation** — let the three timers run for ~30 days, then revisit cadence/limits.

## Completed
- Droplet provisioned, hardened, deployed (DEC-007)
- Git deploy active (droplet has its own deploy key); updates via `git pull` (DEC-008)
- 6h systemd timer for bulk crawler running
- Pinned-tags crawler + 1h timer (script + tags file ready, systemd unit not yet installed on droplet)
- Analytics precompute + 1h timer with 45 min watchdog (script + dashboard refactor ready, systemd unit not yet installed on droplet)
- Dashboard reads cached JSON for instant load; shows cache age + compute time in header
- `--remote-cache HOST` flag in `scripts/dashboard.py` auto-rsyncs the cache from the droplet so the dashboard runs entirely on local laptop without any DB transfer (cache is the only thing needed)

## Deferred (post-Phase-1)
- Public dashboard via Cloudflare Tunnel + Pages frontend (only when actually wanted)
- Postgres migration (only if/when SQLite stops fitting; `pgloader` makes it a 1-command job)
- Trophy-range segmentation, CSV/JSON exports of analytics
- More profiles backfill, country diversity (US/KR/BR/JP rankings) — easy once cadence is observed
