# Brawler-pick recommender v1

> Phase 6 deliverable. Companion: `notebooks/recommender_v1.ipynb` (executable, plots inline) and the `src/brawlstar_agent/recommender/` subpackage. Methodology document; results numbers will refresh on every retrain.

## Problem statement

Predict whether team A wins a 3v3 Brawl Stars battle given the brawlers on each team and the context (mode, map, battle type, trophy level). All three user-facing scenarios reduce to inference under different conditioning of one core model:

\[
P(\text{team A wins} \mid \text{brawlers}_A, \text{brawlers}_B, \text{mode}, \text{map}, \text{tier}, \text{date})
\]

| Scenario | How to query the model |
|---|---|
| "Best brawler for map M" (no draft info) | Marginalize over plausible teammate/opponent triples drawn from the empirical (mode, map) distribution. |
| "Best 2nd / 3rd pick for team A given partial draft" | For each candidate Z, score the completed team A vs the (partial) team B. Argmax. |
| "Last pick for team A: A has X+Y, B has U+V+W" | Special case of the above: A has 2, B has 3, search Z. |

This avoids training three different models; one trained predictor + one inference helper handles all of it.

## What's in the box

```
src/brawlstar_agent/recommender/
├── dataset.py     # clean-window loader, perspective-doubling, splits, brawler-name resolver
├── features.py    # TeamFeaturizer with sparse (sklearn) + dense (LGBM) modes
├── baselines.py   # Global / Mode / ModeMap Wilson-CI baselines (predict_proba interface)
├── team_model.py  # LogRegTeamModel + LGBMTeamModel + evaluate + save/load
├── inference.py   # rank_brawlers_for_map, complete_team, last_pick
└── cv.py          # sliding-temporal-fold harness + evaluate_models_on_folds

scripts/
├── train-recommender.py    # End-to-end train + eval + save
└── analyze-recommender.py  # Plots, feature importance, DAMIAN deep-dive

notebooks/recommender_v1.ipynb   # Interactive companion (executed, with outputs)
reports/recommender_v1.json      # Latest metrics
reports/recommender_v1/*.png     # Plots
models/recommender_v1.lgb.txt    # Trained LightGBM (with .meta.json featurizer)
```

## Hard data rules (read before reusing this code)

1. **Use only post-fix data**: `WHERE battle_time_iso >= '2026-05-03T01:00:00Z'`. The team-result attribution bug fixed in commit `dde58a4` is not detectable from stored columns and cannot be recovered. See **DEC-010**. The dataset loader applies this filter by default; do not bypass it.
2. **Exclude showdown** (`is_showdown = 0`). Different shape (ranks 1-10, no teams) — needs its own model.
3. **Use both perspectives during training**. For each battle, emit two rows: (team_a = team0, team_a_wins = team0 result) and the mirror. This forces label balance to exactly 50/50, which is what makes our metrics meaningful (any deviation from 50% means the model has actually learned something).
4. **Drop friendly / challenge / tournament**. The `dataset.load_clean_battles` default keeps only `ranked` and `soloRanked` — the population in friendlies is not representative.

## Results (random split, n_test=21,052)

Trained on 84,208 rows of clean post-fix data, mostly from 2026-05-03 to 2026-05-05.

| Model       | AUC    | logloss | accuracy | Brier  | fit time |
|-------------|--------|---------|----------|--------|----------|
| Global      | 0.6549 | 0.6596  | 0.6041   | 0.2337 | 0.1s     |
| Mode        | 0.6830 | 0.6456  | 0.6290   | 0.2272 | 0.1s     |
| ModeMap     | 0.6965 | 0.6356  | 0.6360   | 0.2226 | 0.4s     |
| LogReg      | 0.6615 | 0.6451  | 0.6055   | 0.2279 | 0.5s     |
| **LightGBM**| **0.7298** | **0.5988** | **0.6558** | **0.2080** | 8.0s |

Reading the table:
- **Floor**: Global Wilson WR (0.655 AUC) — knowing only "is brawler X usually a winner" gets you to 0.65. So team-vs-team prediction is *easier than 50/50* even from per-brawler base rates alone.
- **ModeMap baseline** is the bar to beat if you stop short of a real model: knowing brawler × mode × map without any interactions gets you to 0.697. Surprisingly hard.
- **LogReg with multi-hot team features** (270 features) is *worse* than the ModeMap heuristic (0.661 vs 0.697). Linear without interaction terms can't beat the ModeMap aggregate. Adding `brawler × map` cross-features would close that gap; we left this out for v1 simplicity since LightGBM gets there for free via tree splits.
- **LightGBM** wins by 3.3 AUC points over the best heuristic. The interactions matter.

See `reports/recommender_v1/fig_auc_bars.png` for the visualization.

### Per-mode breakdown

LightGBM holds its lead across all major modes (`reports/recommender_v1/fig_per_mode_auc.png`). Modes with smaller sample sizes (siege, wipeout, basketBrawl) show wider variance in baseline-vs-LGBM gap, but LGBM is consistently best.

## Temporal cross-validation: transferability harness

The whole point of the CV setup is to validate "train on window N, predict window N+1" — the property that survives meta drift. With ~2 days of clean data today, we can only generate sub-daily folds, but **the same code runs without changes when months of data are available** — that's how we'll quantify transferability over time.

Today's results (4 folds, train_days=0.75, test_days=0.25, step=8h):

| Model    | mean AUC | mean logloss | mean acc |
|----------|---------:|-------------:|---------:|
| Global   | 0.6617   | 0.6578       | 0.6101   |
| ModeMap  | 0.6657   | 0.6479       | 0.6142   |
| **LightGBM** | **0.7043** | **0.6180** | **0.6362** |

Notice the ModeMap baseline drops 3.1 AUC points from random (0.697) to temporal (0.666); LightGBM drops 2.6 points (0.730 → 0.704) but maintains the lead. **Even our temporally-evaluated LightGBM beats the random-split heuristic.** That's the strongest evidence of transferability we can produce from this much data.

The temporal-vs-random gap is real and will only show up clearly with a temporal split — it's exactly the kind of bias that random splits hide. As more data arrives, expect the gap to grow when patches land (Supercell rebalances mid-season) and shrink in stable periods.

`reports/recommender_v1/fig_temporal_cv.png` plots the per-fold AUC trajectory.

## Inference walkthrough

```python
from brawlstar_agent.recommender import load_brawler_names
from brawlstar_agent.recommender.team_model import load_model
from brawlstar_agent.recommender.inference import (
    rank_brawlers_for_map, complete_team, last_pick,
)

names = load_brawler_names()
ids = {v: k for k, v in names.items()}
model = load_model("models/recommender_v1")

# 1. Pre-draft tier list for a (mode, map)
ranks = rank_brawlers_for_map(model, "brawlBall", "Backyard Bowl", train_df=..., n_samples=80)
# → [(b_id, P(win)), ...] sorted descending

# 2. Mid-draft completion: A picked LOLA + GIGI, B picked JESSIE + EL PRIMO
res = complete_team(
    model,
    my_team=(ids["LOLA"], ids["GIGI"]),
    opp_team=(ids["JESSIE"], ids["EL PRIMO"]),
    mode="brawlBall", map="Backyard Bowl", top_k=8,
)

# 3. Last pick: A picked LOLA + GIGI, B finished JESSIE + EL PRIMO + BIBI
res = last_pick(
    model,
    my_partial_team=(ids["LOLA"], ids["GIGI"]),
    opp_team=(ids["JESSIE"], ids["EL PRIMO"], ids["BIBI"]),
    mode="brawlBall", map="Backyard Bowl",
)
```

The notebook (`notebooks/recommender_v1.ipynb`) has these scenarios pre-rendered.

## Known issue: release-meta inflation (the DAMIAN case)

When LightGBM is asked "best last pick on Backyard Bowl", it ranks DAMIAN at 0.96 P(win) and the next-best brawler at 0.90. That's a striking gap, and worth understanding before you trust it.

DAMIAN is brawler ID `16000104` — likely Brawl Stars' newest release (the ID is the highest in our data; the official `brawlers` table on the droplet is not yet refreshed but `battle_players` knows the name). Raw stats from training data:

| Stat | Value |
|------|-------|
| Appearances in training | 41,134 |
| Overall win rate | **64.5%** |
| Win rate (knockout) | 70.1% |
| Win rate (wipeout) | 83.2% (small N) |
| Trophy mean of users | 1,366 (high — top players unlocked it first) |

**The model isn't broken; the meta is.** DAMIAN is genuinely the most-played AND highest-win-rate brawler in the data window we have. Other top-WR brawlers in our data (OLLIE, GLOWBERT, GIGI) are also recent releases. This is the classic "release meta" pattern: Supercell ships new brawlers strong, top players unlock them first, the data shows volume + win rate inflation, the next balance patch evens things out.

**What to do about it**:
1. Treat any predicted P(win) > 0.85 in inference as a release-meta signal, not a robust recommendation.
2. The temporal-CV harness *will* catch this when DAMIAN's WR drops post-patch — the model trained pre-patch will start losing AUC against the post-patch test window. That's the meta-drift signal we wanted to quantify.
3. If we wanted to dampen this in v2, hierarchical priors (per-brawler shrinkage toward the global mean) or rolling-window training (drop everything older than 30 days) would help.

`reports/recommender_v1/damian_deepdive.json` has the raw numbers.

## How to retrain

```bash
cd /media/lin/disk2/brawlstar-agent
PYTHONPATH=src uv run python scripts/train-recommender.py
```

Common flags:

```bash
# Train on a different cutoff window (e.g. last 30 days only)
... scripts/train-recommender.py --cutoff 2026-04-04T00:00:00Z

# Restrict to one mode (faster iteration)
... scripts/train-recommender.py --modes brawlBall

# Save model + metrics to specific paths
... scripts/train-recommender.py \
    --save-to models/recommender_v2 \
    --report-to reports/recommender_v2.json

# Skip temporal CV for fast iteration
... scripts/train-recommender.py --no-temporal
```

After training, regenerate the plots and the DAMIAN deep-dive:

```bash
PYTHONPATH=src uv run python scripts/analyze-recommender.py
```

## Limitations & next steps

### What this model doesn't know
- **Star Power, Hyper Charge, Gear, Gadget choice** — not in the `battlelog` API response, so not in our schema. A given brawler's effectiveness depends on these. `brawler_power` (1-11) is a weak proxy and is *not* used in the v1 features (it didn't help in offline tests; would only help if we coupled it with brawler ID in interaction terms).
- **Draft order** — the API gives the final 6 brawlers, not the pick sequence. So `complete_team(my_team={X, Y}, opp_team={U, V, W})` doesn't know whether team B chose its third pick *in response to* team A's two — but in the pure "P(win | brawlers)" framing this doesn't matter for inference, only for understanding what counterfactual you're modeling.
- **Specific player skill** — we use team-mean `brawler_trophies` as a coarse skill proxy. Per-player skill features (player win rate, hours played) would help but require profile-data ingestion which is a separate project.

### Things to add in v2
1. **Brawler × map / brawler × mode interaction features** for LogReg, so the linear model becomes a real competitor at <1ms inference.
2. **Calibrated probabilities** — current LightGBM proba is well-calibrated by Brier on random-split (0.21) but worth verifying on temporal splits and in extreme regimes (release-meta brawlers).
3. **Rolling-window training**: train on the last 30 days only (rebuild monthly). Mitigates meta drift directly.
4. **Brawler embeddings**: factorization machine or shallow NN. Probably not worth it until we have 10x more data per (mode, map, brawler) cell.
5. **Star Power / Hyper Charge ingestion** — separate ingestion design needed; the API returns these statically per `/players/{tag}/profile`, not in battlelogs. Would require us to attribute "this brawler had Star Power X at the time of this battle" via a periodic profile snapshot, which is involved.

### What gets better automatically as data accumulates
- More temporal-CV folds → tighter transferability bounds without code changes.
- Per (mode, map, tier) cells get denser → ModeMap baseline gets stronger; the LightGBM gap may shrink (a *good* sign — means the meta is well-explained by per-cell stats).
- A real "month N → month N+1" curve becomes possible after ~3 months of post-fix data, at which point we should publish a transferability scorecard.

## Where this fits in the project

- **DEC-009**: training and inference live on local laptop (RTX 3060 + 62 GB RAM); the droplet only crawls and pre-computes baselines into `analytics_cache.json`. Nothing in this work runs on the droplet.
- **DEC-008**: code is committed locally; no need to deploy any of it to the droplet for now. If we ever want production inference, we'd add a recommendation API service in a separate phase.
- **DEC-010** (introduced this session): the legacy team-result bug is undetectable from stored columns; do not attempt to "fix" it. Strict post-fix filter is the only path.

## Files to read in order

1. This document (you're here).
2. `notebooks/recommender_v1.ipynb` — see numbers + plots in context.
3. `src/brawlstar_agent/recommender/dataset.py` — understand the data shape.
4. `src/brawlstar_agent/recommender/team_model.py` — the model class.
5. `src/brawlstar_agent/recommender/inference.py` — how to call it.
6. `scripts/train-recommender.py` — how to retrain end-to-end.
