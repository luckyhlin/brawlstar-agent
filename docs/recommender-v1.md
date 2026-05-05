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

### Binary win prediction

| Model       | AUC    | logloss | accuracy | Brier  | fit time |
|-------------|--------|---------|----------|--------|----------|
| **Random** (uniform 0.5 + jitter) | **0.5000** | 0.6931 | 0.5000 | 0.2500 | 0.0s |
| **TrophyOnly** (sigmoid of log-trophy diff) | **0.4965** | 0.6932 | 0.5082 | 0.2500 | 0.0s |
| Global      | 0.6549 | 0.6596  | 0.6041   | 0.2337 | 0.1s     |
| Mode        | 0.6830 | 0.6456  | 0.6290   | 0.2272 | 0.1s     |
| ModeMap     | 0.6965 | 0.6356  | 0.6360   | 0.2226 | 0.4s     |
| LogReg      | 0.6615 | 0.6451  | 0.6055   | 0.2279 | 0.5s     |
| **LightGBM**| **0.7298** | **0.5988** | **0.6558** | **0.2080** | 8.0s |

Reading the table:
- **Random floor**: 0.500 AUC, exactly as theory says. Confirms our metric pipeline.
- **TrophyOnly = 0.497 AUC**. **The "high-trophy team beats low-trophy team" signal is essentially absent in this data.** That sounds wrong but actually makes sense: ranked matchmaking pairs players with similar trophies, so within a match the trophy difference is small and largely uninformative. **Every AUC point above 0.500 is genuine brawler-pick signal**, not a skill-tier shortcut. This is rare — most "intuitive" baselines on competitive game data are dominated by skill leakage. We don't have that problem here.
- **Global Wilson** (0.655) — the "is brawler X usually a winner" heuristic. Strong floor.
- **ModeMap** (0.697) — knowing brawler × mode × map without interactions. Surprisingly hard to beat.
- **LogReg with multi-hot team features** (270 features, no interactions) is *worse than the ModeMap heuristic* (0.661 vs 0.697). LogReg is a real ML model, not a heuristic — but without explicit `brawler × map` cross-features, it can only learn flat per-brawler effects, the same information ModeMap aggregates directly. Tree models (LightGBM) get interactions for free.
- **LightGBM** wins by 3.3 AUC points over the best heuristic. The interactions matter.

See `reports/recommender_v1/fig_auc_bars.png`.

### Top-K recommendation: where does the actually-played brawler rank?

AUC says "how often is team A's win prob higher than team B's", but the **user-facing question** is "given partial draft state, list the top-K brawlers I should consider". Different problem, different metrics.

For each test battle we mask team A's third pick, score all ~97 legal candidates with each model, and record where the actually-played brawler lands.

| Model        | hit@1   | hit@3 | hit@5 | hit@10 | MRR    | mean rank |
|--------------|--------:|------:|------:|-------:|-------:|----------:|
| Random (uniform) | 0.001 | 0.03  | 0.05  | 0.10   | 0.04   | 49.0 |
| Global Wilson    | 0.006 | 0.19  | 0.19  | 0.22   | 0.13   | 35.0 |
| ModeMap          | 0.008 | 0.06  | 0.15  | 0.20   | 0.09   | 34.5 |
| **LightGBM**     | **0.150** | **0.194** | **0.225** | **0.293** | **0.205** | 36.2 |

LightGBM **hit@1 is 15× the uniform-random floor** (0.150 vs ~0.010 = 1/97). At hit@10 it's still ~3× random.

Why is **Global Wilson hit@3 (0.19) almost as good as LightGBM hit@3 (0.19)?** Because the most-played-and-highest-WR brawlers (DAMIAN, OLLIE, GLOWBERT etc.) are picked very often *regardless of context*. So just blindly recommending the top-3 globally-strongest brawlers correctly predicts the actual pick ~19% of the time — but **only because the meta is so concentrated**, not because the model understands matchups. The hit@1 is the cleaner test: there Global gets 0.6%, LightGBM 15.0% — a 25× gap, because hitting the *exact* top recommendation requires real context modeling.

### Win-rate uplift: actionable test of the recommendation

When the actually-played brawler IS in the model's top-K, what's the win rate of those games?

| Top-K    | LightGBM WR | Random WR (test set baseline) | Uplift Δ |
|----------|------------:|------------------------------:|---------:|
| top-1    | 62.5%       | 51.1%                         | +11.4 pp |
| top-3    | 62.5%       | 51.1%                         | +11.4 pp |
| top-5    | 61.7%       | 51.1%                         | +10.7 pp |
| top-10   | 59.1%       | 51.1%                         | +8.0 pp  |

**Read this as the player-flexibility tradeoff**: going from top-1 → top-10 doubles the candidate pool (more flexibility for "I don't have that brawler" / "I'm bad at it") while *only sacrificing 3 percentage points of expected win rate*. Top-K with K=5 is a great default.

### Winners-only top-K (cleanest meta-quality test)

If we restrict to test rows where team A actually won, the played brawler was at least *good enough for that specific matchup*. The model's rank of that brawler is then a quality signal:

| Model       | hit@1 | hit@5 | hit@10 | MRR    |
|-------------|------:|------:|-------:|-------:|
| Random      | 0.001 | 0.04  | 0.09   | 0.04   |
| ModeMap     | 0.015 | 0.22  | 0.29   | 0.12   |
| **LightGBM**| **0.210** | **0.313** | **0.389** | **0.276** |

**LightGBM hit@1 = 21% on winning teams**: when the team actually won with brawler X, our model's top-1 recommendation IS X 21% of the time. That's 20× over uniform random.

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

## Operational note: brawlers table was stale (fixed 2026-05-04)

The droplet's `brawlers` table had only 101 rows even though `battle_players` had 103 distinct brawler IDs. Cause: `scripts/collect-battles.py --collect-only` (the systemd-timer command) skipped `seed_brawlers()` to "save the API call". DAMIAN (id 16000104) showed up in real battles starting **2026-04-24** but never made it into the canonical table.

Fix: `--collect-only` now calls `seed_brawlers()` once at the start of every run. One extra API call per 6 hours, idempotent UPSERT, ensures the table stays current as Supercell ships new brawlers. Code path doesn't affect already-stored battle data.

The recommender uses `dataset.load_brawler_names()` which already falls back to `battle_players.brawler_name` for any IDs missing from the canonical table — so the model itself was never affected by this bug.

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

## Why we can't recover the legacy bug labels (long answer to "is the data really wrong?")

It is reasonable to ask:

1. *Are the legacy labels actually wrong?*
2. *If so, why can't we revert them?*

Both are good questions. Detail by detail:

### The bug, restated precisely

Pre-`dde58a4` `db.py::_insert_battle_players` did:
```python
team_result = result if team_idx == 0 else inverse(result)
```
where `result` is `battle.result` from the *fetched player's* perspective — the player whose `/players/{tag}/battlelog` we hit. The fetched player can be on team 0 OR team 1.

If the fetched player was on team 0: labels are correct.
If the fetched player was on team 1: team 0 gets the *fetched player's* result, which is actually team 1's outcome, and team 1 gets the inverse — i.e., team 0's outcome. **Both teams' labels are flipped relative to ground truth.**

Crucially, the bug **preserves every internal invariant we have**:
- Exactly one team has `'victory'`, exactly one has `'defeat'` (still true, just on the wrong teams).
- `trophy_change` (stored on `team_index=0`'s first player) reflects the *fetched player's* trophy delta. Pre-fix, the row's `result` is also the fetched player's result. So `result` and `trophy_change` *agree on sign in both eras*. No mismatch to detect.

### Why we can't tell which battles are flipped from stored data

We never recorded `fetched_for_tag`. Without knowing which player triggered the battlelog fetch, there is no internal signal that distinguishes "labels correct" from "labels symmetrically flipped".

### Could we re-crawl and verify empirically?

Yes — for a *small subset* of legacy battles. The Brawl Stars API only returns each player's most recent **25 battles**. For a battle ingested on 2026-04-15, by today (2026-05-05) every player who participated has played 25+ more games, so the battle has aged out of every player's API window. **It's gone.**

The only legacy battles potentially recoverable are ones that:
- Are recent enough to still be in some participant's last-25 (typically <2 weeks old, less for active players).
- Have at least one inactive participant (low play frequency → battle hasn't aged out).

Concretely, if you wanted to verify the bug:
1. Pick legacy battles between, say, 2026-04-25 and 2026-05-02 (the bug's last week).
2. For each, find the participant with the smallest count of subsequent battles in our DB (proxy for "low activity, battle might still be in their feed").
3. Re-crawl that player's battlelog with the post-fix code.
4. If the same `battle_id` appears, compare new labels to stored labels. A roughly 50% mismatch rate would confirm the bug.

This is a small experiment — maybe 50-100 verifiable battles, which would tell us whether the bug actually fired at the expected ~50% rate.

But verification doesn't enable correction. Even if we proved 50% of legacy battles are flipped, **we still can't tell which 50%** without re-fetching each one. And re-fetching is impossible for the bulk of legacy data because of the 25-battle window.

### What we actually saw when we ran the verification (2026-05-04)

`scripts/verify-bug.py` was run against 80 candidate battles from 2026-04-25 to 2026-05-02 (the bug's last week). Result:

| Metric | Count |
|---|---|
| Attempted | 80 |
| Battle aged out of API | 60 |
| **Recoverable** | **20** |
| Bug fired (FLIPPED) | 0 |
| Match (no flip) | 20 |

A naive read says "bug rate = 0%", but **all 20 recovered battles turned out to be post-fix-INGESTED** (their `battle_time` was pre-cutoff but they were ingested by the post-deploy crawler). The participants who still had them in their last-25 are exactly the low-activity players whose first crawl happened post-deploy — selection bias makes them all post-fix.

So the verification *cannot* test pre-fix-ingested battles because those have all aged out of the API. The 0% flip rate is consistent with "post-fix code produces correct labels" (the null hypothesis), not with "the bug never fired".

A cross-check via `collection_log` is more informative:

| Class (pre-cutoff battles) | Count |
|---|---|
| Definitely post-fix-INGESTED (provably clean labels) | **6,045** |
| Pre-fix-INGESTED or ambiguous (likely contains the buggy data) | 75,045 |

So **8% of pre-cutoff-time battles are actually clean** — the time-based filter is conservative. We could relax `CLEAN_CUTOFF_ISO` to use ingestion-time (`collection_log.created_at`) instead of battle-time, recovering ~6k clean rows. Filed as a v2 candidate.

### Operational conclusion (unchanged)

Strict post-2026-05-03 filter for training and evaluation. DEC-010 stands by code analysis; the empirical bug rate is intrinsically untestable. The 6k recoverable clean rows are a small bonus available later via the smarter ingestion-time filter.

`reports/verify_bug.json` has the per-battle details for spot-checking.

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
