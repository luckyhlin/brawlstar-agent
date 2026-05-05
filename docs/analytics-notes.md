# Analytics & ML Handoff Notes

> Quick-start guide for someone picking up data analysis or model training on the collected battle data. Read this **before** you start training anything on `data/brawlstars.db` — there are non-obvious data quality caveats that will quietly ruin your model if you miss them.

## What you have

- **DB**: `data/brawlstars.db`, SQLite, ~600 MB, growing ~150-200 MB/month
  - Lives on the **droplet** (production, fresh) — `rsync` it to local for analysis (DEC-009)
  - Local copy may be stale; refresh before serious work
- **Cache**: `data/analytics_cache.json` (~400 KB), updated every 1h on droplet by `brawl-analytics.timer`
- **Counts (as of 2026-05-04)**: 200k+ battles, 970k+ battle_player rows, 553k player tags discovered, 104 brawlers
- **Time coverage**: ~1.5 months of dense data (post-deploy, 2026-05-03 onward); pre-deploy data is sparse and partly broken (see caveats below)

## Schema (cheat sheet)

```sql
brawlers(id PK, name)                    -- 104 rows, IDs 16000000-16000103

players(tag PK, name, trophies, highest_trophies, exp_level,
        club_name, source, last_battlelog_at, last_profile_at, created_at)
        -- 553k+ rows; only ~6k have trophies/profile data filled in
        -- source: 'rankings' | 'battlelog' | 'profile' | 'manual'

battles(battle_id PK, battle_time, battle_time_iso, event_id, mode, map,
        battle_type, duration, is_showdown, star_player_tag)
        -- 200k+ rows; battle_id = battleTime + sorted player tags (deterministic dedup key)
        -- mode: brawlBall, gemGrab, heist, knockout, bounty, hotZone, basketBrawl,
        --       siege, wipeout, trophyThieves, soloShowdown, duoShowdown
        -- battle_type: 'ranked' (trophy ladder), 'soloRanked' (competitive ranked),
        --              'friendly', null, etc.

battle_players(battle_id FK, player_tag, team_index, brawler_id, brawler_name,
               brawler_power, brawler_trophies, is_star_player, result, trophy_change)
        PRIMARY KEY (battle_id, player_tag)
        -- 970k+ rows. team_index: 0/1 for 3v3 modes, 0..9 for showdown
        -- result: 'victory' / 'defeat' for team modes, '1'..'10' (rank) for showdown

collection_log(id PK, action, target, status, detail, created_at)
        -- audit trail; ignore for analytics
```

Indexes already exist on `battle_players(brawler_id)`, `(player_tag)`, `(battle_id, team_index)`, `(result, brawler_name)`, plus `battles(mode)`, `battles(mode, is_showdown)`, `battles(battle_time_iso)`, `players(trophies DESC)`. Most analytics queries you'd want are already cheap. The expensive one is the matchup self-join on `battle_players × battle_players` (~5-15 min on the 1-CPU droplet, seconds on a laptop).

## ⚠ CRITICAL DATA CAVEATS — read before training anything

### 1. Team-result label bug in pre-2026-05-03 battles (FIXED, NOT BACKFILLED)

Bug history: `db.py::_insert_battle_players` originally assumed the player whose battlelog we fetched was always on `team_index = 0`, and labeled team 0 with the API's `result` and team 1 with the inverse. **In reality, the fetched player can be on either team.** When they were on team 1, the labels got inverted: team 0 got the wrong result, team 1 got the wrong result.

- **Affected**: every battle ingested before commit `dde58a4` (2026-05-03 ~01:00 UTC), where the snowballed-into player happened to be on team 1.
- **Magnitude**: roughly half of pre-deploy battles, since team assignment is approximately random. So ~60-70k battles in the DB have inverted labels for some/all players.
- **Fixed forward**: post-`dde58a4` battles use the correct logic. About **130k+ battles since 2026-05-03 are clean.**
- **Not backfilled**: we did NOT go back and fix the historical labels. They're still wrong.

**For ML training**: the simplest safe filter is `WHERE battle_time_iso >= '2026-05-03T01:00:00Z'`. You'll lose ~70k battles but every label is trustworthy. Don't train on the full dataset without this filter — half your bad-team labels will silently flip.

If you really need the old data, the fixable invariant is: in any single battle, exactly one team's `result` is `'victory'` and the other is `'defeat'`. So you can detect "did one team have its result flipped" and write a fix. But it's easier to just filter.

Showdown battles (`is_showdown = 1`) use ranks 1-10 instead of victory/defeat — they're not affected by this bug.

### 2. Per-player API window: only last 25 battles

Brawl Stars API returns the **25 most recent battles per player**, period. Older ones are gone. Implications:

- **Active high-trophy players play 25+ battles in 6 hours.** At our 6h crawl cadence, we miss anything beyond their last 25. The bulk crawler covers ~1500 stale players per run; for a player ranked ~20,000 by trophies, time-between-fetches can be days, and we'll miss most of their games.
- **Pre-deploy days look sparse** (`SELECT date(battle_time_iso), COUNT(*) FROM battles GROUP BY 1`) — May 1-2 each have <5k rows because by the time we got around to those players' battlelogs on May 4-5, their May 1-2 games had aged out.
- **Steady-state going forward** is ~40-45k battles/day, growing in proportion to crawl cadence.

For meta-trend analysis ("did Brawler X get nerfed in the patch on date D?"), you have <1 day resolution only on highly active top players. For long-term win rates, post-deploy data is fine.

### 3. Trophy data is sparse

Of 553k discovered tags, only ~6k have `trophies`/`highest_trophies`/`exp_level` populated. The other 547k are tag-only (we know they exist because they showed up in someone's battlelog, but we never fetched their profile).

- `battle_players.brawler_trophies` is *per-brawler trophy*, populated for every row from the battle data itself. Meaningful in `ranked` (trophy ladder), where it reflects skill at that brawler. In `soloRanked` it's *rank points* (2-22 mapping to Bronze..Masters via `RANKED_TIERS`).
- `players.trophies` is *total trophies across all brawlers*, populated only when we fetched the profile.

For player-skill features, **prefer `brawler_trophies`** (per-row, always present) over `players.trophies` (often null). Use `RANKED_TIERS` from `models.py` to convert rank points to tier names for soloRanked battles.

### 4. Mode/battle_type interactions

- `battle_type='ranked'` = trophy ladder, `brawler_trophies` = real trophy count
- `battle_type='soloRanked'` = competitive ranked, `brawler_trophies` = rank points (2-22)
- `battle_type='friendly'` = no trophy change, lower meta signal
- `mode='soloShowdown' | 'duoShowdown' | 'trioShowdown'` = `is_showdown=1`, different shape (ranks 1-10, no team)

**For team-comp / matchup analysis, exclude showdown** (`is_showdown = 0`). For "absolute brawler strength", consider stratifying by `battle_type` because the player population differs.

## Existing analytics modules (don't reinvent)

### `src/brawlstar_agent/analytics.py` — `BattleAnalytics` class

| Method | What it does | Sample size advice |
|---|---|---|
| `summary()` | mode/battle_type counts, time range, totals | always cheap |
| `brawler_win_rates(mode=, battle_type=, ranked_tier=, ...)` | per-brawler WR with sample size | use `min_sample=5+` |
| `brawler_win_rates_by_tier(min_sample=10)` | per-brawler WR per ladder trophy tier | requires `battle_type='ranked'` |
| `combo_win_rates(mode=, min_sample=3)` | 3-brawler team comp WR | sparse; min_sample=2-3 |
| `matchup_win_rates(min_sample=50)` | brawler A (your team) vs brawler B (opp). Self-join, **slowest query** | min_sample=50 keeps it tractable |
| `synergy_win_rates(min_sample=50)` | brawler A + brawler B on same team. Also a self-join. | min_sample=50 |
| `brawler_scores()` | calls `models.py::score_brawlers` (Wilson CI + tier-adjusted WR) | always cheap |

### `src/brawlstar_agent/models.py` — statistical baselines

- `wilson_interval(wins, total)` — 95% Wilson CI for a binomial proportion. **Use this** instead of raw win-rate for sample-size honesty.
- `tier_adjusted_win_rate(brawler_per_tier_stats, global_per_tier_stats)` — re-weights a brawler's WR by the global tier distribution. Corrects for "this brawler is mostly played at high tier so it looks strong."
- `score_brawlers()` — combines raw WR, Wilson CI, ladder tier-adjusted WR, soloRanked tier-adjusted WR, per-tier breakdown. Already used by the dashboard.

### `src/brawlstar_agent/dashboard_data.py` — precomputed cache schema

If you don't want to query the DB, just load `data/analytics_cache.json` — it has `summary`, `brawler_rates` (per-mode + per-tier), `combos`, `matchups`, `synergies`, `brawler_scores`, `my_data`. ~400 KB, refreshed every 1h on droplet.

## Problem framing: "best brawler to use" prediction

This is a **contextual recommendation problem**, not just a global ranking. Same brawler is great in one matchup and bad in another.

Input you'll have at recommendation time:
- mode + map (always known)
- battle_type (always known: ranked / soloRanked / etc.)
- player's tier or brawler trophies
- partial team (the user has picked their teammates' brawlers? or not?)
- partial opponent info (in ranked draft, opponents are known; in trophy ladder, often not)

Target: predicted P(victory | brawler choice, context) per candidate brawler → rank.

### Baselines to beat (in order of complexity)

1. **Global Wilson WR** (`brawler_scores`) — ignores context entirely. Easy floor.
2. **Mode-conditional WR** — `brawler_win_rates(mode=...)`. Filters context by mode.
3. **Mode + map-conditional WR** — extend the queries with a map filter. Gets sparse fast.
4. **Linear matchup model** — for each candidate brawler X, predict P(win) = base_WR(X|mode,tier) + Σ matchup_effect(X, opponent_brawler) + Σ synergy_effect(X, teammate_brawler). Effects come from `matchup_win_rates` and `synergy_win_rates`. Cheap to score, often surprisingly hard to beat.
5. **Embedding / factorization model** — learn brawler embeddings such that team_emb · opp_emb predicts win prob. Handles sparsity better; needs training infrastructure.
6. **Gradient boosted trees on engineered features** — features = (your trophies, your brawler one-hot, opponent brawlers one-hot, mode one-hot, map one-hot, ...). Probably the highest-ROI starting point if you want a model rather than a heuristic.

### Realistic gotchas

- **Class imbalance**: every battle is "victory" for one side and "defeat" for the other. Ensure both perspectives are in training data, OR train only from team_index=0's perspective and use the inverse for evaluation. Either is fine but be consistent.
- **Sparsity**: 104 brawlers × ~50 maps × 11 modes × ~6 tiers = ~340k cells, with maybe 130k clean battles in this cell space → most cells have <1 sample. Embedding models or hierarchical priors help; pure tabular WR will overfit.
- **Meta drift**: Supercell rebalances brawlers periodically. A model trained on data older than ~2 months may be off by a tier. Include `battle_time_iso` as a feature or train on rolling windows.
- **Selection bias**: high-trophy players play more, so they dominate the data. The "best brawler" in their data is the best brawler at their skill level, not yours. Stratify by tier.
- **Star Power / Hyper Charge / Gear / Gadget choice are not in the schema.** A given brawler's effectiveness depends heavily on these meta-progression items, which we don't track. This is a real limitation; `brawler_power` (1-11) is a weak proxy.

### What the data is and isn't good for

| Question | Quality |
|---|---|
| "Which brawlers have the highest win rate this season?" | Good (use `score_brawlers` with Wilson CI) |
| "How does brawler X do against brawler Y?" | OK (matchup matrix; sparse for rare pairs) |
| "What's the best partner for brawler X?" | OK (synergy matrix) |
| "Which brawler should I pick on map M in mode N at trophy tier T?" | Need a model; data exists but is sparse per cell |
| "Did the recent balance patch buff brawler X?" | Limited (need before/after windows; meta drift inferences need careful temporal splits) |
| "What's the optimal Star Power for brawler X?" | **Not possible** with current data; not collected |

## Workflow tips for ML on local laptop

1. **Get the latest DB**: `rsync -avz --progress brawl:brawlstar-agent/data/brawlstars.db data/brawlstars.db`
2. **Filter for clean labels**: `WHERE battle_time_iso >= '2026-05-03T01:00:00Z'` in every training query
3. **Use the existing modules first**: don't rewrite Wilson CI, matchup queries, etc. They're in `analytics.py` and `models.py`.
4. **Keep notebooks in `notebooks/`** (gitignored by default per `.venv/` policy; if you want them tracked, add an explicit rule).
5. **Heavy training jobs run on local** (DEC-009). RTX 3060 + 62 GB RAM is the right place.
6. **If you want to ship a feature back to production** (a new periodic analytics report, a recommendation API), follow DEC-008: edit on local → commit → push → `git pull` on droplet → install/restart relevant systemd unit.

## Quick orientation queries

```sql
-- Clean training set size
SELECT COUNT(*) FROM battles WHERE battle_time_iso >= '2026-05-03T01:00:00Z';

-- Sample size per (mode, brawler) — most-played first
SELECT b.mode, bp.brawler_name, COUNT(*) AS n
FROM battle_players bp JOIN battles b ON b.battle_id = bp.battle_id
WHERE b.battle_time_iso >= '2026-05-03T01:00:00Z'
  AND b.is_showdown = 0
GROUP BY 1, 2 ORDER BY n DESC LIMIT 30;

-- Mode-conditional brawler WR (use models.score_brawlers for Wilson-corrected version)
SELECT bp.brawler_name,
       COUNT(*) AS total,
       SUM(CASE WHEN bp.result='victory' THEN 1 ELSE 0 END) AS wins,
       ROUND(100.0*SUM(CASE WHEN bp.result='victory' THEN 1 ELSE 0 END)/COUNT(*),2) AS wr
FROM battle_players bp JOIN battles b ON b.battle_id = bp.battle_id
WHERE b.battle_time_iso >= '2026-05-03T01:00:00Z'
  AND b.is_showdown = 0
  AND b.mode = 'brawlBall'
  AND bp.result IN ('victory','defeat')
GROUP BY 1 HAVING total >= 50 ORDER BY wr DESC LIMIT 20;

-- Available maps per mode
SELECT mode, map, COUNT(*) AS battles
FROM battles WHERE battle_time_iso >= '2026-05-03T01:00:00Z'
GROUP BY 1, 2 ORDER BY 3 DESC LIMIT 20;
```

## Things explicitly NOT in scope for the next session

- Don't extend the schema to track Star Power / Hyper Charge / Gear / Gadget — the API doesn't return them in battlelogs (only on the `/players/{tag}` profile, statically). Adding them is a separate ingestion design.
- Don't backfill the team-result bug; just filter post-2026-05-03.
- Don't run training on the droplet; it'll OOM. Local only.
- Don't add live-match features (DEC-001 era constraint: research only, no live botting).

## Where to write findings

- New decisions → `memory-bank/decisions.md` as DEC-010, DEC-011, etc.
- Session log → `memory-bank/progress.md` as Session 8.
- Methodology / model docs → new file like `docs/recommender-v1.md` (sibling to this file).
- Trained models → `models/` directory (gitignored by default for `.pt`/`.pth`/`.onnx`/`.bin`/`.h5`/`.pkl`/`.npy` per current `.gitignore`).
