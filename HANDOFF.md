# Handoff — 2026-05-06 mid-Session 8

> Snapshot for the next agent. The memory-bank files are the canonical record;
> this document is a thin pointer + the next-action checklist.

## Read first (in this order)

1. **`memory-bank/memory-bank.md`** — project overview, hard constraints, phase status
2. **`memory-bank/activeContext.md`** — current focus + immediate state
3. **`memory-bank/progress.md`** — full session log; Session 8 entries are most relevant
4. **`memory-bank/decisions.md`** — DEC-001..010 (DEC-010 is critical: the legacy team-result bug is unrecoverable; do NOT try again)
5. **`docs/recommender-v1.md`** — Phase 6 methodology, results, and how-to-retrain (the canonical doc for the recommender)
6. **`docs/deployment.md`** § 16 — fish/zellij gotcha table on droplet (only relevant if you ssh in)

Optional / situational:
- `docs/analytics-notes.md` — data caveats; the "verification" subsection in `recommender-v1.md` supersedes the legacy-data-recovery suggestions
- `memory-bank/architecture.md` — directory layout, module inventory
- `memory-bank/techContext.md` — droplet config, host machine

## Where the recommender stands

| Run | Cutoff | Battles | LightGBM AUC (random / temporal) | hit@1 | Status |
|---|---|---:|---:|---:|---|
| v1 baseline | 2026-05-03 | 78k | 0.730 / 0.704 | 0.150 | shipped earlier in Session 8 |
| **Run A (v2 default)** | **2026-05-03** | **1.78M** | **0.7382 / 0.7281** | TBD | **just done** |
| Run B (all-data) | 2021-01-01 | 2.26M | TBD | TBD | **pending** |
| Run C (30-day window) | 2026-04-06 | TBD | TBD | TBD | **pending** |

Reports/models live at `reports/recommender_*.json`, `models/recommender_*.lgb.txt`. The training CLI is `scripts/train-recommender.py`; top-K is `scripts/eval-topk.py`.

## What to do, in priority order

### 1. Run C — last 30 days (PRIORITY: this is the production-realistic window)

```bash
cd /media/lin/disk2/brawlstar-agent
PYTHONPATH=src uv run python scripts/train-recommender.py \
    --cutoff 2026-04-06T00:00:00Z \
    --save-to models/recommender_v2_30d \
    --report-to reports/recommender_v2_30d.json
```

Estimated time: ~10-20 min. Compare to Run A — if 30-day matches or exceeds the all-cutoff result, that's strong evidence the production model should use a rolling 30-day window.

### 2. Run B — all data (sanity ceiling, slower)

```bash
PYTHONPATH=src uv run python scripts/train-recommender.py \
    --cutoff 2021-01-01T00:00:00Z \
    --save-to models/recommender_v2_all \
    --report-to reports/recommender_v2_all.json
```

Estimated: ~45-75 min (LightGBM fit alone will be 15+ min on ~2.3M battles). Skip if it's blocking other work; can be done overnight.

### 3. Top-K eval against the winning model

```bash
PYTHONPATH=src uv run python scripts/eval-topk.py
```

`scripts/eval-topk.py` currently hardcodes `load_clean_battles()` with the default cutoff. To evaluate on a different cutoff, edit the call in `main()` (one line). Compare the new hit@1 / hit@5 / MRR to v1's (0.150 / 0.225 / 0.205).

### 4. Update `docs/recommender-v1.md` with the v2 numbers

Add a "v2 results" section comparing all four runs (v1, A, B, C). The takeaway is a story: where does more data help? Where does the time window matter? Where do we hit the feature-set ceiling?

If v2's biggest run still doesn't break ~0.74 AUC, the data ceiling is real and feature engineering (the v2 plan in `docs/recommender-v1.md`'s "Things to add in v2" section) is the next big lever.

### 5. Droplet shrinkage cleanup verification

User is currently shrinking the droplet's DB to fit on the 24 GB disk. Tasks pending:
- Confirm droplet disk usage is < 80% before any timer restart
- The DELETE pattern (purge pre-2026-04 battles) is in `docs/deployment.md` § 16's troubleshooting table and in this session's recent assistant turns
- After local has a smaller, compacted version, optionally rsync it back: `rsync -avz --progress data/brawlstars.db brawl:~/brawlstar-agent/data/brawlstars.db` (with timers stopped)
- Then re-enable timers: `sudo systemctl start brawl-collect.timer brawl-collect-pinned.timer brawl-analytics.timer`

### 6. Consider relaxing CLEAN_CUTOFF_ISO

Because the cold-start guarantees ALL data in the DB is post-fix-ingested, the current `CLEAN_CUTOFF_ISO = '2026-05-03T01:00:00Z'` is unnecessarily conservative. Two options:
  - Drop the constant entirely (any rows in the DB are clean)
  - Keep the constant as a safety net but document that it's now optional

Defer this until after Runs B/C results inform the decision.

## Hard constraints — do NOT violate

- **DEC-010**: legacy team-result bug is not recoverable from stored data. The cold-start was the cleanup. Don't write any code that "tries to fix" pre-fix battles — they no longer exist (purged + replaced).
- **DEC-009**: training/inference live on local laptop only. Don't attempt to run training on the droplet (1 GB RAM, 1 vCPU; will OOM). Droplet only crawls.
- **DEC-008**: code changes flow local → GitHub → droplet `git pull`. Never edit code on droplet.
- **Hard memory bank rules**: no live multiplayer botting, everything local on Linux, isolated from personal accounts.

## Gotchas you must know

- **DB is 18.6 GB on local**. Most queries are fast because of indexes, but `pd.read_sql_query` for the full battle_players table loads ~28M rows / ~7 GB into memory. Doable on the 62 GB RAM, but be mindful with parallel jobs.
- **Fish auto-launches on droplet ssh interactively** — see `docs/deployment.md` § 16. If you `ssh -t brawl 'bash heredoc with <<EOF'`, the heredoc fails. Use `NO_FISH=1 ssh -t brawl '...'` or pass SQL as a quoted multi-line string. Non-interactive `ssh brawl 'cmd'` is unaffected.
- **DAMIAN release-meta inflation**: brawler ID 16000104, ~65% raw WR over its games. Real meta truth, not a model bug, but flag for users when shipping recommendations. Filed in `docs/recommender-v1.md`.
- **rsync of live SQLite DB without `.backup` or stopped writers** produces a malformed copy. Use `bash scripts/rsync-db-from-droplet.sh` (default) or `--direct` (when timers are stopped + WAL checkpointed).
- **DB row counts can lie**: `players` table has 822k rows but most have only `tag` populated (snowball-discovered tags); only ~6k have full profile data. `battle_players.brawler_trophies` is the per-row trophy snapshot and is always populated.

## Working pattern (per memory-bank protocol)

- Update `memory-bank/progress.md` Session 8 entries as you go (append; don't rewrite history)
- Update `memory-bank/activeContext.md` if focus changes
- Add new decisions as DEC-011, DEC-012, etc. to `memory-bank/decisions.md`
- Commit incrementally per DEC-008 (local → GitHub → droplet pull when shipping crawler/server changes)
- Don't write throwaway notes in this `HANDOFF.md`; if you have things to record permanently, put them in memory-bank instead

## Where to write findings

- New decisions → `memory-bank/decisions.md` as DEC-011, DEC-012, etc.
- Session log → `memory-bank/progress.md` (append to Session 8 (continued) or open Session 9)
- Methodology → update `docs/recommender-v1.md`, or new `docs/recommender-v2.md` if substantive
- Models → `models/` (`.lgb.txt` is gitignored; `.meta.json` is committed)
- Notebooks → `notebooks/`
- Reports (JSON metrics, plots) → `reports/recommender_*/` (size-conscious; under 1 MB per artifact)

## Last good commit

`bd1869b` — verification + cold-start orchestrator + (later) helper scripts. Run `git log --oneline | head -10` to see recent commits including the v2 work in progress.
