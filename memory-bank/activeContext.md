# Active Context

## Current State

**Phase 6 v1 shipped (Session 8, 2026-05-04)**: brawler-pick recommender trained, evaluated, documented, and committed. LightGBM team-completion model achieves AUC 0.730 (random) / 0.704 (temporal) on clean post-fix data.

**Production crawl** still running on droplet (3 systemd timers from Session 7). Local DB synced this session: 210k battles, latest 2026-05-05, **78k clean post-fix** battles.

## Operating principle (DEC-009)

- **Remote**: routine/periodic — crawlers, scheduled analytics precompute, backups
- **Local**: interactive/heavy — dashboard, ad-hoc queries, ML training, exploration
- Cache flows local from droplet via `dashboard.py --remote-cache`; DB rsync only for ad-hoc deep-dives or model retraining

## What Works

### Crawler & analytics infra (Sessions 6-7, unchanged)
- API client, SQLite collector, snowball discovery, three timers on droplet
- Dashboard reads pre-computed `analytics_cache.json` for instant load
- `--remote-cache` flag rsyncs cache from droplet for laptop-only viewing

### Recommender v1 (this session)
- `src/brawlstar_agent/recommender/` subpackage
- `scripts/train-recommender.py` — re-runnable end-to-end (key for the "transferable algorithm" framing across months)
- `scripts/analyze-recommender.py` — plots, feature importance, DAMIAN deep-dive
- `notebooks/recommender_v1.ipynb` — executed, with outputs
- `docs/recommender-v1.md` — full methodology, results, how-to-retrain
- `models/recommender_v1.lgb.txt` — saved trained model
- `reports/recommender_v1.json` + `reports/recommender_v1/*.png` — metrics & plots

### Inference scenarios — all covered
- `rank_brawlers_for_map(model, mode, map, train_df=...)` — pre-draft tier list
- `complete_team(model, my_team, opp_team, ...)` — mid-draft completion
- `last_pick(model, my_partial_team, opp_team, ...)` — end-of-draft

## Next Steps (Phase 6 v2 candidates, no order yet)

1. **Rolling-window retrain**: `train-recommender.py --cutoff $(date -d '30 days ago' +%Y-%m-%d)` once per month; compare to full-history model. Direct meta-drift mitigation.
2. **Interaction features for LogReg**: brawler × map, brawler × mode crosses → cheap inference.
3. **Calibration on temporal splits** — verify Brier/log-loss hold up out-of-time, especially on release-meta brawlers (DAMIAN being the canonical example).
4. **Per-tier conditioning**: stratify training by ranked tier (Bronze..Masters) so the recommendation respects skill level. Currently mean-trophy is a coarse proxy.
5. **Star Power / Hyper Charge ingestion** (much bigger): they're not in `battlelog`; needs a periodic profile snapshot to attribute "this brawler had X at battle time T". Separate design.

## Deferred / parallel tracks (unchanged)

1. Cloudflare R2 nightly backups
2. Country diversity in seeding (US/KR/BR/JP rankings)
3. First-month observation of the three timers
4. Public dashboard via Cloudflare Tunnel
5. Postgres migration (only if SQLite stops fitting)

## Important caveats inherited

- **DEC-010**: legacy team-result bug is not recoverable. Recommender uses strict post-2026-05-03 filter. Don't try to backfill the legacy data again.
- **Release-meta inflation**: DAMIAN (id 16000104, newest brawler) has 64.5% raw WR over 41k games. Model recommends DAMIAN with extreme confidence. This reflects the live meta, not a model bug, but cap your trust in `P(win) > 0.85` predictions.
- **Sample window**: clean post-fix data is concentrated in 2026-05-03..2026-05-05. Real "month N → month N+1" temporal evaluation needs to wait ~30 days for enough disjoint windows; the harness handles this without code changes.

## DB state on local

- Path: `/media/lin/disk2/brawlstar-agent/data/brawlstars.db` (~970 MB)
- Synced 2026-05-04 19:something UTC
- Refresh before major retrains: `rsync -avz --progress brawl:brawlstar-agent/data/brawlstars.db data/brawlstars.db`
