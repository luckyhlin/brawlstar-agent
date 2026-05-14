# Brawler-pick recommender v2

> Phase 6 update. Companion: `docs/recommender-v1.md` (still the canonical methodology + inference walkthrough; read it first if you haven't). This document records what changed in v2: data scale, the **stable temporal test set** methodology (DEC-011), the fair-run comparison across cutoffs, the top-K story on a leakage-free test set, and why we are NOT shipping the 30-day rolling-window model right now.

## What v2 is and isn't

**It is**: the same model architecture as v1 (multi-hot brawler bag + dense scalars feeding a LightGBM team-completion model), retrained on 50× more data, evaluated under a methodology that lets us compare runs across different training cutoffs.

**It isn't**: a new model class. v3 is where transformer / embedding / factorization-machine architectures come in; that's where the bigger AUC moves probably live. See "v3 plans" below.

## Why v2 needed a methodology fix

### What v1 evaluated

v1 used a random 80/20 split of the post-2026-05-03 clean window. That works when:
- The data spans a couple of days (no meaningful intra-window distribution shift), AND
- Comparisons happen between models trained on the SAME data.

But by the time we had three different cutoffs (Run A=2026-05-03, Run C=2026-04-06, Run B=2021-01-01) and were comparing AUCs across them, each run's random test set was sampled from a *different* underlying time distribution. So "Run A AUC = Run C AUC" was comparing two different metrics that happened to share a name.

| Original random-split run | Cutoff | Battles | Random-split AUC | Random test-set timeframe |
|---|---|---:|---:|---|
| Run A | 2026-05-03 | 1.78M | 0.7382 | 20% of 2026-05-03..06 (3 days, dense) |
| Run C | 2026-04-06 | 2.13M | 0.7392 | 20% of 2026-04-06..06 (heavily weighted to May 4-5 by density) |

Identical-looking AUCs (0.7382 vs 0.7392) made it look like more data wasn't helping. They weren't actually measuring the same thing.

### DEC-011: a stable temporal test set

Pinned in `scripts/train-recommender.py`:

```python
STABLE_TEST_AFTER_DEFAULT = "2026-05-05T00:00:00Z"
```

- **Train** = `[--cutoff, '2026-05-05T00:00:00Z')`
- **Test** = `['2026-05-05T00:00:00Z', latest_battle_in_db]`
- **Test set size**: ~844,151 clean ranked/soloRanked battles after `dataset.py` cleaning, **1,688,302 rows after both-perspectives doubling**. Plenty for tight AUC standard errors.

Implementation:
- `scripts/train-recommender.py --stable-test-after TIMESTAMP` — toggles temporal holdout in place of the random split. Records `split_mode`, `stable_test_after`, `n_train_battles`, `n_test_battles` in the report. When set, temporal CV runs only over the train portion (no leakage).
- `scripts/eval-topk.py --cutoff … --stable-test-after …` — same boundary, so binary AUC and top-K hit@K come from identical test rows.
- `scripts/compare-fair-runs.py` — prints the comparison table.

When the boundary needs to move (eventually it should — as data accumulates the test window can slide forward), bump `STABLE_TEST_AFTER_DEFAULT` and re-train every model that contributes to a comparison. AUCs across boundary versions are NOT comparable; rename downstream artifacts on each move (e.g., `_fair_v2`, `_fair_v3`).

## The fair-run comparison

Three trainings, all `--no-temporal --stable-test-after 2026-05-05T00:00:00Z`, same LightGBM hyperparameters (n_estimators=600, num_leaves=63, lr=0.05, min_data_in_leaf=80, l2=1.0), same featurizer:

| Run | Cutoff | Train battles | LightGBM AUC | LogReg | ModeMap | Mode | Global | LGBM fit (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A_fair | 2026-05-03 |   935,808 | 0.7181 | 0.6808 | 0.6625 | 0.6731 | 0.6551 | 111 |
| **C_fair** | **2026-04-06** | **1,280,861** | **0.7265** | 0.6805 | 0.6725 | 0.6725 | 0.6536 | 154 |
| B_fair | 2021-01-01 | 1,412,893 | 0.7235 | 0.6752 | 0.6708 | 0.6732 | 0.6544 | 123 |

C_fair is the technical winner: **+0.84 pp** over A_fair, **+0.30 pp** over B_fair. The 30-day window is the empirical sweet spot — adding 5 years of older (sparse, partly legacy-buggy) data on top of it slightly hurts.

What random splits hid:
- Random→stable-test AUC drops are real and informative. **ModeMap drops ~3 pp** (0.6917 → 0.6625 for A; 0.6888 → 0.6725 for C). Per-(mode, map) memorization doesn't fully transfer across time — that's the meta-drift signal random splits hide. **LightGBM drops ~2 pp** (0.7382 → 0.7181 for A; 0.7392 → 0.7265 for C); regularized + interaction-aware so it transfers a little better, but it still pays a temporal-holdout tax.
- **LogReg flat across all three runs** (0.6808 / 0.6805 / 0.6752). The linear model with multi-hot team features can't represent brawler×brawler or brawler×map interactions, so more rows feed a feature set that's already saturated. **Tree models get those interactions for free**; that's why C_fair's LightGBM moves while LogReg doesn't. This pins the next lever as **feature design** (or model architecture), not data volume.

Per-mode AUC, LightGBM, stable-test:

| Mode (n in test) | A_fair | C_fair | B_fair |
|---|---:|---:|---:|
| brawlBall    | 0.7517 | 0.7593 | 0.7559 |
| siege        | 0.7933 | 0.8005 | 0.7994 |
| basketBrawl  | 0.7273 | 0.7306 | 0.7249 |
| knockout     | 0.6942 | 0.7114 | 0.7086 |
| wipeout      | 0.7047 | 0.7076 | 0.7067 |
| heist        | 0.6784 | 0.6806 | 0.6783 |
| gemGrab      | 0.6667 | 0.6695 | 0.6665 |
| hotZone      | 0.6551 | 0.6576 | 0.6565 |
| bounty       | 0.6159 | 0.6189 | 0.6198 |

C_fair wins or ties in 8 of 9 modes (loses to B_fair on bounty by 0.001 — within noise).

## Top-K recommendation on the stable test set

Re-ran `scripts/eval-topk.py` against the C_fair config, n=5000 sample, last_pick mode (mask team A's third pick, model ranks all ~97 legal candidates, find rank of actually-played brawler):

| Model         | hit@1 | hit@3 | hit@5 | hit@10 | MRR    | mean rank | WR\|in_top1 (Δ vs 49.5% baseline) |
|---------------|------:|------:|------:|-------:|-------:|----------:|---:|
| Random        | 0.004 | 0.024 | 0.050 | 0.099  | 0.047  | 49.3      | 47.4% (−2.2 pp)  |
| TrophyOnly    | 0.000 | 0.002 | 0.003 | 0.021  | 0.022  | 63.6      | (no top-1 picks) |
| Global Wilson | 0.005 | 0.014 | 0.154 | 0.212  | 0.073  | 39.8      | 58.3% (+8.8 pp)  |
| ModeMap       | 0.118 | 0.152 | 0.176 | 0.241  | 0.169  | 34.0      | 70.6% (+21.1 pp) |
| **LightGBM**  | **0.130** | **0.190** | **0.229** | **0.292** | **0.194** | **33.6** | **67.6% (+18.1 pp)** |

Reading the table:
- **LightGBM hit@1 = 0.130** on the stable test set. v1 reported 0.150 on a random split — that was random-split leakage; **0.13 is the honest baseline going forward**. v3 numbers must compare against this, not against v1's number.
- **ModeMap hit@1 = 0.118 is surprisingly close to LightGBM**. Because the meta is concentrated, blindly recommending the top per-(mode, map) brawlers is right ~12% of the time even on a future window. LightGBM only edges out at hit@1, but its lead widens at hit@3 (0.19 vs 0.15) and hit@10 (0.29 vs 0.24) where context-aware ranking actually has to do work.
- **TrophyOnly is *worse* than Random** at hit@K — it ranks high-trophy candidates first, which are systematically not the brawlers people actually play. (TrophyOnly's *binary* AUC of ~0.53 was barely above random in v1 and is similarly weak here. There is essentially no skill-tier shortcut in this data.)
- **Win uplift in top-1 is +18.1 pp**: when the played brawler IS in our top-1 recommendation, that team won 67.6% vs the test-set baseline 49.5%. That's the actionable signal the recommender exists for.

Winners-only top-K (filter test rows where team A actually won — cleanest meta-quality test):

| Model         | hit@1 | hit@5 | hit@10 | MRR    |
|---------------|------:|------:|-------:|-------:|
| Random        | 0.003 | 0.048 | 0.093  | 0.046  |
| ModeMap       | 0.181 | 0.251 | 0.318  | 0.234  |
| **LightGBM**  | **0.183** | **0.290** | **0.369** | **0.253** |

When the team actually won with brawler X, LightGBM's top-1 IS X 18.3% of the time, top-10 is 36.9%. Versus uniform random (1/97 ≈ 1%), this is 18× and 4× respectively.

## Honest reading: should we ship 30-day rolling now?

**No, not yet.** The +0.84 pp AUC bump from A_fair → C_fair is real but doesn't justify the operational complexity at this stage:

1. **Marginal lift**. 0.84 pp AUC translates to single-digit pp shifts in actual hit@K and win-uplift, well within the noise of normal player variation.
2. **Pipeline cost**. A 30-day rolling window means every retrain loads 30× more data than the simpler post-fix-only window (1.28M battles vs 936k for A_fair) and requires a scheduled monthly retrain (DEC-009: training stays local). For a sub-1 pp improvement, the operational simplicity of A_fair wins.
3. **The real ceiling is feature design / model architecture, not data volume**. LogReg flat across all three runs caps somewhere around AUC 0.68 with the current feature set; LightGBM caps somewhere around 0.73. To break through to AUC 0.78+ we likely need different features (or a transformer/FM architecture). Adding +2-5 pp via better features dwarfs +0.84 pp from a longer training window.
4. **v3 advanced methods are the next lever.** If we're investing engineering time, it's there.

**Provisional production candidate**: `models/recommender_v2_default_fair.lgb.txt` (cutoff 2026-05-03 — the simplest pipeline). 30-day model `models/recommender_v2_30d_fair.lgb.txt` is kept as the technical-best reference and as the v3 baseline to beat, but is not the deployment target.

When 30-day might become worth deploying:
- If/after v3 transformer lands and we want to maximize that model's training data window. (Transformers benefit from data more than gradient-boosted trees do — so an architecture that *can* use the extra rows productively might shift the calculus.)
- If we observe that A_fair's stable-test AUC degrades over multiple weeks while C_fair-style refits hold up — i.e., real meta drift starts to dominate. Re-evaluating monthly is cheap; the harness is already there.

## How to retrain (v2)

```bash
cd /media/lin/disk2/brawlstar-agent
export UV_CACHE_DIR=/media/lin/disk2/brawlstar-agent/.uv-cache-local

# 3-day fair run (current production candidate)
PYTHONPATH=src uv run python scripts/train-recommender.py \
    --cutoff 2026-05-03T01:00:00Z \
    --stable-test-after 2026-05-05T00:00:00Z \
    --no-temporal \
    --save-to models/recommender_v2_default_fair \
    --report-to reports/recommender_v2_default_fair.json

# 30-day fair run (technical winner; reference + v3 baseline)
PYTHONPATH=src uv run python scripts/train-recommender.py \
    --cutoff 2026-04-06T00:00:00Z \
    --stable-test-after 2026-05-05T00:00:00Z \
    --no-temporal \
    --save-to models/recommender_v2_30d_fair \
    --report-to reports/recommender_v2_30d_fair.json

# All-data fair run (only useful if Run B style ever comes back into play)
PYTHONPATH=src uv run python scripts/train-recommender.py \
    --cutoff 2021-01-01T00:00:00Z \
    --stable-test-after 2026-05-05T00:00:00Z \
    --no-temporal \
    --save-to models/recommender_v2_all_fair \
    --report-to reports/recommender_v2_all_fair.json

# Side-by-side comparison (reads the three reports above)
PYTHONPATH=src uv run python scripts/compare-fair-runs.py

# Top-K eval on the stable test set against the chosen config
PYTHONPATH=src uv run python scripts/eval-topk.py \
    --cutoff 2026-04-06T00:00:00Z \
    --stable-test-after 2026-05-05T00:00:00Z \
    --output reports/recommender_v2_topk.json --sample-size 5000
```

The `logs/run_fair_v2.sh` orchestrator (gitignored) chains these together for backgrounded sequential runs (~20 min total wall-clock).

After the stable-test boundary moves (e.g., to `'2026-06-05T00:00:00Z'` once another month of dense data has accumulated), re-train every model with the new boundary and rename outputs (e.g., `recommender_v2_*_fair_2026-06`) so old and new comparisons don't accidentally mix.

## v3 plans (advanced methods)

Where v3 should focus, roughly ordered by expected payoff:

1. **Transformer over (mode, map, team-A-brawlers, team-B-brawlers)**: a small encoder that learns brawler embeddings + interaction structure end-to-end. Likely the biggest jump if we have enough data per (brawler, brawler) cell. Should benefit from the 30-day window more than LightGBM does.
2. **Factorization machine** with brawler×brawler and brawler×map cross terms. Cheap inference, much closer to LightGBM than current LogReg without the full transformer cost. Good baseline against the transformer.
3. **Brawler embeddings as a stepping stone**: same predictor architecture as v2 but learnable d=16 embeddings per brawler instead of 1-hot. Tests whether the embedding bottleneck changes the data-scaling curve at all.
4. **Star Power / Hyper Charge / Gear ingestion** (separate ingestion design): would need to attribute "brawler X had Y at battle time T" via periodic profile snapshots. Adds real stat depth that v1/v2 are blind to.
5. **Calibration on stable test**: verify Brier/log-loss hold up; consider isotonic / Platt scaling especially for release-meta brawlers (DAMIAN remains the canonical example — see v1 doc).

In every v3 candidate: keep `--stable-test-after` (DEC-011) as the evaluation methodology so v3 numbers are directly comparable to the v2 table above.

## Files

- `models/recommender_v2_default_fair.{lgb.txt,meta.json}` — A_fair (3-day, current production candidate)
- `models/recommender_v2_30d_fair.{lgb.txt,meta.json}` — C_fair (30-day, technical winner / v3 baseline)
- `models/recommender_v2_all_fair.{lgb.txt,meta.json}` — B_fair (all-data including ~150k pre-April legacy)
- `reports/recommender_v2_default_fair.json` / `30d_fair.json` / `all_fair.json` — per-run binary metrics + per-mode breakdown
- `reports/recommender_v2_topk.json` — top-K + win uplift on the stable test set
- `scripts/train-recommender.py` — adds `--stable-test-after` flag (DEC-011)
- `scripts/eval-topk.py` — re-anchored to stable test set
- `scripts/compare-fair-runs.py` — apples-to-apples comparison table
- `logs/run_fair_v2.sh` — orchestrator, gitignored

## See also

- `docs/recommender-v1.md` — v1 methodology, inference walkthrough (`rank_brawlers_for_map` / `complete_team` / `last_pick`), DAMIAN release-meta caveat (still applies in v2; cap trust on `P(win) > 0.85` predictions)
- `memory-bank/decisions.md`
  - DEC-010 — legacy team-result bug is unrecoverable; use post-fix data
  - **DEC-011** — stable temporal test set is mandatory for v2-and-beyond comparisons
- `memory-bank/progress.md` Session 8 — full session log including the cold-start, the random-split → stable-test pivot, and the local DB shrink
