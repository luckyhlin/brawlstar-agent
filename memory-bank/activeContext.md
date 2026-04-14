# Active Context

## Current State

API battle analytics pipeline is built and operational. First data collection done.

## What Works
- **API client**: rate-limited (1-2 req/s), auth from api.env, retry with exponential backoff
- **SQLite storage**: normalized schema, dedup by (battleTime + sorted tags), WAL mode
- **Collector**: seed from rankings → fetch battlelogs → snowball discovered tags, resume-safe
- **Analytics**: 4 query types all working with filter support (mode, trophy range, time window)
  1. Brawler win rates per mode
  2. Team composition (3-brawler combo) win rates
  3. Matchup matrix (brawler A vs opposing brawler B)
  4. Synergy matrix (brawler A + brawler B on same team)

## Current Data (2026-04-13)
- 23,219 battles (team + showdown, 11 modes)
- 87,535 player tags discovered (snowballed from top-200 seeds)
- 1,200 players with trophy data, 1,000 with full profile
- 101 brawlers cataloged
- Time range: 2026-03-30 .. 2026-04-13
- DB size: 72MB at `data/brawlstars.db`
- Dashboard: `scripts/dashboard.py` (local HTTP with portraits, 4 analysis tabs)

## Next Steps
1. **More profiles**: 86k players still missing trophy data, fetch in batches
2. **Add country diversity**: seed US, KR, BR, JP rankings for broader meta coverage
3. **Periodic cron**: set up a daily/weekly collection pass to build time-series data
4. **Trophy-range segmentation**: add trophy bracket filter to dashboard
5. **Export**: CSV/JSON export of analytics for further visualization
