# Active Context

## Current State

Crawler is **deployed and running on a DigitalOcean droplet** (DEC-007) on a 6-hour systemd timer. Local-primary workflow with git pull on the droplet (DEC-008). Data is fresh through today.

**Production droplet**: `lin@209.38.4.212` (reserved IP) / `64.23.171.86` (anchor / outbound IP)
**DB on droplet**: `~/brawlstar-agent/data/brawlstars.db` — independent copy from local; the droplet is the source of truth for production collection.

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
1. **Switch droplet from rsync'd code to git clone** — generate SSH key on droplet, register as GitHub deploy key, re-clone repo so future updates flow via `git pull`
2. **Cloudflare R2 nightly backups** — free 10 GB tier, `sqlite3 .backup` → zstd → rclone copyto. Protects against droplet failure.
3. **SSH polish** — verify your IPv4 home address ends up in fail2ban `ignoreip`
4. **First weekly observation** — let the timer run for ~7 days, then check growth rate, mode coverage, idle-player ratio. Tune `--battlelog-limit` and cadence if needed.

## Deferred (post-Phase-1)
- Public dashboard via Cloudflare Tunnel + Pages frontend (only when actually wanted)
- Postgres migration (only if/when SQLite stops fitting; `pgloader` makes it a 1-command job)
- Trophy-range segmentation, CSV/JSON exports of analytics
- More profiles backfill, country diversity (US/KR/BR/JP rankings) — easy once cadence is observed
