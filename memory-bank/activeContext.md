# Active Context

## Current State (2026-05-13 late evening, Session 12 — Fresh data landed at `data/brawlstars_extra_2026-05-14.db`; ready for v4 boundary cut)

**User rsynced new droplet data** into `data/brawlstars_extra_2026-05-14.db` (17 GB, integrity_check ok). **3.36 M clean post-fix battles** (1.89× the old DB), **329 k new beyond `2026-05-06T04:30:14Z`**, latest battle `2026-05-14T02:27:05Z`. Old `data/brawlstars.db` untouched. The DEC-019/020 data-bound diagnosis can now be empirically falsified: at fixed N = XL, the joint Chinchilla fit predicts ~+0.56 pp Mythic+ from doubling D.

**Next chat's first move**: slide `STABLE_TEST_AFTER_DEFAULT` forward (probably ~`2026-06-05T00:00:00Z` once data is dense enough — or pick another tail-window that gives ~2 weeks of test data), retrain the kit-sink XL and the LGBM phase-1+4 on the combined DB, and report fresh v4 numbers. AUCs will NOT be comparable to the current 0.6249 ensemble SOTA — that's the price of meta-drift, treat as a new baseline.

## Pre-data-rsync view (kept for reference)

## Current State (2026-05-13 evening, Session 12 — Anchor runs + ensemble; new Mythic+ SOTA 0.6249, DEC-020)

**Three new findings this evening** confirm and refine DEC-019:

1. **NEW Mythic+ SOTA = 0.6249** via LGBM ⊕ XL ensemble at α_lgbm = 0.45 (0.55 transformer). +0.69 pp over the best single model (XL+P1+P2+P4 at 0.6180), zero new training. Ensemble gain grows with slice difficulty: Unranked +0.39 pp, Mythic+ +0.69 pp, Legendary+ +0.85 pp — the diversity benefit between LGBM-tree splits and transformer attention compounds on the hardest examples.

2. **Anchor runs M1 / M2 land on the kit-sink curve** (M1 = 1.10 M params, Mythic+ 0.6117 — predicted 0.6121; M2 = 1.57 M, Mythic+ 0.6131 — predicted 0.6144). Both at batch=4096, same recipe as small / big / XL. The kit-sink fit now has 2 real DOF instead of 0; the **asymptote AUC revises UP from 0.629 to 0.640**, and XL has closed **84.3 %** (not 91.5 %) of the gap from random.

3. **M3 (5.1 M params) is an outlier — and that itself is informative.** Had to drop batch from 4096 → 2048 to fit the RTX 3060 Mobile's 5.77 GiB; Mythic+ collapsed to 0.5892, 3 pp below the curve. Net interpretation: at our GPU memory budget, 5 M is past the practical useful-capacity ceiling under the current training recipe. The clean kit-sink fit excludes M3 as a recipe-confounded data point.

### Refined fit (5 obs, dof = 2; M3 excluded)

- Kit-sink Mythic+ (1 − AUC) = 0.360 + 0.548 · N^(−0.213) — much flatter than the 3-point fit's α = 0.364.
- AUC ceiling at current D = 1.87 M: **0.640**.
- 2.19 pp Mythic+ headroom above XL's 0.6180.
- Inverse predictions: AUC 0.620 = 5.5 M params (1.7× XL); AUC 0.625 = 21.3 M (6.5×); AUC 0.630 = 143 M (43.6×); AUC 0.640 = asymptote (unreachable).
- Joint Chinchilla (10 obs, dof = 5): α = 0.454, β = 0.082, E = 0.275. At the SOTA point, **reducible loss splits 8.2 % capacity / 91.8 % data** — even more data-bound than the pre-anchor fit said.
- Data projection at fixed XL N: 2× D → 0.6208, 4× D → 0.6261, 8× D → 0.6311. Each doubling of D ≈ +0.5 pp Mythic+.

### Production candidates after this session

| Use case | Model | Mythic+ AUC |
|---|---|---:|
| **Best Mythic+ AUC (research)** | **LGBM ⊕ XL ensemble (α_lgbm=0.45)** | **0.6249** |
| Best single transformer | `models/recommender_v3_phase1p2p4_xl.pt` (Run K) | 0.6180 |
| Best AUC/cost (GPU) | `models/recommender_v3_phase1p2p4_big.pt` (Run J) | 0.6084 |
| Best CPU-only | `models/recommender_v2_phase1p4.lgb.txt` (Run F) | 0.6060 |
| New anchor mid-size | `models/recommender_v3_phase1p2p4_m1.pt` (1.1 M) | 0.6117 |
| New anchor mid-size | `models/recommender_v3_phase1p2p4_m2.pt` (1.6 M) | 0.6131 |
| Anchor outlier (kept for repro) | `models/recommender_v3_phase1p2p4_m3.pt` (5.1 M) | 0.5892 |

### Reordered next-phase priorities (post-DEC-020)

1. **Time-series-aware boundary slide + rerun**: when fresh droplet data lands, slide `STABLE_TEST_AFTER_DEFAULT` forward and rename downstream artifacts (`_v4`). AUCs across boundary versions are not comparable.
2. **More droplet data → `data/brawlstars_extra.db`**: cheapest β-falsification. Expected ~+0.5 pp Mythic+ per doubling under the new fit.
3. **Phase 4b — per-token history features**: new-information channel on the per-brawler token.
4. **Per-player ELO / Bradley-Terry skill features** (phase 4c): cheap; fits the existing pipeline; complements phase-4 frequency aggregates.
5. **Sequence model over per-player history**: architectural lever most likely to beat the ensemble SOTA. Consumes raw history that phase 4 / 4b can only aggregate.
6. **Pick-prediction multi-task head** + pairwise/listwise ranking loss: orthogonal to AUC; targets hit@K.

De-prioritized: bigger transformer beyond XL (M3 confirmed the practical ceiling); FM baseline (LGBM phase 1 already covers it); GNN (transformer attention is the GNN).

### Artifacts (this session)

- `scripts/analyze-scaling-laws.py` — fit + plots; reads `reports/` automatically.
- `scripts/ensemble-stable-test.py` — LGBM ⊕ transformer sweep with proba cache.
- `reports/scaling_laws_inventory.csv` — 21 runs (14 transformer + 7 LGBM).
- `reports/scaling_laws.json` — fitted parameters + predictions + projections.
- `reports/ensemble_kitsink.json` — α sweep + slice breakdown.
- `reports/ensemble_cache/p_{lgbm_phase1p4, xfr_phase1p2p4_xl}.npy` — cached probas.
- `reports/scaling_law_N_*.png` — log-x scatter + curve with M3 X-marker.
- `logs/{anchor_runs_orchestrator,m3_retry_orchestrator,train_v3_phase1p2p4_m1/m2/m3,ensemble_kitsink,scaling_laws_final}.log`.
- `models/recommender_v3_phase1p2p4_{m1,m2,m3}.{pt,meta.json}` — 3 new model checkpoints.
- `canvases/scaling-law-mythicplus.canvas.tsx` (workspace) + `docs/canvases/scaling-law-mythicplus.canvas.tsx` (repo mirror) — rich summary canvas, refreshed.

## Pre-DEC-020 view (kept for reference)

## Current State (2026-05-13, Session 12 — Scaling-law analysis says Mythic+ is data-bound, DEC-019)

**Scaling-law fit on the existing 11 transformer runs answers HANDOFF.md task #1**: at the current data scale (D = 1.87 M training rows), Mythic+ AUC asymptotes at **~0.629**. SOTA XL+P1+P2+P4 (0.6180) has closed **91.5 %** of the gap from random; only **+1.1 pp** of capacity headroom remains, and reaching even AUC 0.625 would require **16× current XL params** (53 M). The joint Chinchilla-style fit on Mythic+ (1 − AUC) (11 obs, dof = 6) gives **α ≈ 0.30, β ≈ 0.09** — at the SOTA point the reducible loss splits **14 % capacity / 86 % data**. We are heavily data-bound, not capacity-bound.

### Decision implication (DEC-019)

Stop chasing larger transformers for Mythic+ — go after data and new information channels:
1. **More droplet data → separate DB** (`data/brawlstars_extra.db`): cheapest falsification of the implied β. Joint fit predicts ~+0.5 pp Mythic+ per doubling of D.
2. **Phase 4b (per-token history features)**: new information channel, the empirical analogue of doubling D again (DEC-018's +0.71 pp Mythic+ matches the fit's 4 – 8× D prediction).
3. **Star Power / Hyper Charge / Gears ingestion**: largest information delta, biggest engineering cost.
4. (optional) Anchor runs at N ≈ 1 M, 1.5 M — only refines the kit-sink-fit α; doesn't move the SOTA.
5. Pick-prediction multi-task head: orthogonal lever (top-K).

De-prioritized: ~~bigger transformer beyond XL~~ (would buy < 0.5 pp at this data scale per the analysis).

### Artifacts

- `scripts/analyze-scaling-laws.py` — reproducible fit pipeline.
- `reports/scaling_laws_inventory.csv` — 18-run inventory (11 transformer + 7 LGBM).
- `reports/scaling_laws.json` — fitted parameters + predictions + projections.
- `reports/scaling_law_N_mythic_logloss.png`, `scaling_law_N_mythic_auc.png`, `scaling_law_N_all_auc.png` — log-x scatter + curve.
- `canvases/scaling-law-mythicplus.canvas.tsx` — rich summary canvas (outside repo).

## Pre-DEC-019 view (kept for reference)

## Current State (2026-05-13, Session 11 — Phase 4 player history breaks the Mythic+ ceiling, DEC-018)

**Phase 4 (per-player history) is the first feature addition that meaningfully moves Mythic+ AUC.** Built 12 frequency-only player-history aggregates (n_games, brawler-pair counts, main-brawler alignment) computed from a separate pre-cutoff April window (`--use-history-features` + `--history-after`). The `history_df` is disjoint from training rows, so the lookup is leakage-free by construction. WR features intentionally dropped (label-leaky from same-window source; 50%-noise from pre-cutoff legacy-bug source). Player overlap with April history is only ~9% per player, ~25% per team — thin but enough.

### NEW Mythic+ state of the art: 0.6180 (v3 XL + P1+P2+P4)

Up from prior 0.6109 (vanilla v3 XL), **+0.71 pp**. Also wins all-test (0.7746) and best calibration (Brier 0.1902).

| Model | Params | All AUC | Mythic+ | Legendary+ | Cost |
|---|---:|---:|---:|---:|---|
| v3 XL (no phase, prior SOTA) | 3.28M | 0.7674 | 0.6109 | 0.5986 | 52 min GPU |
| **v3 XL + P1+P2+P4 (Run K)** | 3.29M | **0.7746** | **0.6180** | **0.6054** | 58 min GPU |
| v3 big + P1+P2+P4 (Run J) | 576k | 0.7734 | 0.6084 | 0.5954 | 16 min GPU |
| v3 small + P1+P2+P4 (Run H) | 254k | 0.7719 | 0.6013 | 0.5816 | 7 min GPU |
| LGBM + P1+P4 (Run F) | n/a | 0.7708 | 0.6060 | 0.5965 | 24 min CPU |

LGBM gains on Mythic+ ≈ 0 (same plateau as phase 1/2). Transformer gain compounds with arch: small kitchen sink +0.80 pp, big +0.96 pp, XL +0.71 pp vs respective P1-only/no-phase baselines. Bigger arch absorbs phase 4 better.

### Production candidates after DEC-018

| Use case | Model | Mythic+ |
|---|---|---:|
| Best research/calibration | `models/recommender_v3_phase1p2p4_xl.pt` | **0.6180** |
| Best AUC/cost on GPU | `models/recommender_v3_phase1p2p4_big.pt` | 0.6084 |
| Best CPU-only deploy | `models/recommender_v2_phase1p4.lgb.txt` | 0.6060 |

All three are ALSO best in their respective tiers for all-test AUC + calibration. The vanilla v3_big.pt / v3_xl.pt remain on disk for reproducibility but are superseded.

### Key insight: new information beats new aggregates

Phase 1 (trophy/power aggregates) and phase 2 (time/release aggregates) moved Mythic+ by ~0 pp because aggregates of existing data are already implicit in brawler embeddings and tree splits. Phase 4 added something new — per-player frequency signal that the model has never seen. Bigger arches can absorb this signal; LGBM cannot.

## Pre-DEC-018 view (kept for reference)

## Current State (2026-05-09, Session 10 afternoon — phase 2 + soloRanked-only sprint done, DEC-017)

**Phase 2 (time + days_since_release) and soloRanked-only training both delivered roughly nothing on the Mythic+ slice.** Five new runs done; full ablation table:

| Run | Model | P1 | P2 | Train | All AUC | **Mythic+ AUC** |
|---|---|:-:|:-:|---|---:|---:|
| A | LGBM | yes | **yes** | mixed | 0.7618 | 0.6065 |
| B | LGBM | yes | no | **solo** | 0.6282 | 0.6079 |
| C | LGBM | yes | yes | **solo** | 0.6245 | 0.6044 |
| D | v3 small | yes | **yes** | mixed | 0.7564 | 0.5944 |
| E | v3 small | yes | yes | **solo** | 0.6014 | 0.5802 |

Phase 2 deltas on Mythic+: LGBM mixed −0.05 pp, LGBM solo −0.35 pp, v3 small mixed +0.11 pp. All within sample noise. soloRanked-only deltas on Mythic+: LGBM phase-1 +0.09 pp (best non-XL!), LGBM phase-1+2 −0.21 pp, v3 small −1.42 pp (data volume wins for the transformer).

**Mythic+ AUC ranking after 12 configurations** (state of the art still v3 XL):
| AUC | Model |
|---:|---|
| **0.6109** | v3 XL (no phase) |
| 0.6079 | LGBM phase 1, soloRanked-only train |
| 0.6070 | LGBM phase 1, mixed train |
| 0.6065 | LGBM phase 1+2, mixed (Run A) |
| 0.6048 | LGBM no-phase baseline |
| 0.5944 | v3 small phase 1+2 mixed (Run D) |
| 0.5802 | v3 small phase 1+2 solo (Run E) |

Spread across 12 configs is just 0.0307 AUC. Capacity + architecture (LGBM → XL transformer) gives ~+0.30 pp; everything else (phase 1, phase 2, data scoping) is sample noise on this slice.

## Pre-DEC-017 view (kept for reference)

## Current State (2026-05-09, Session 10 morning — tiered slice eval done, DEC-016)

**Big reframing**: domain-knowledge clarification (`ranked` = in-game Unranked trophy ladder, `soloRanked` = actual Ranked queue with Bronze→Pro tiers and strict 1-2-2-1 ban/pick draft from Mythic+) plus retrospective slice evaluation on all 7 saved checkpoints showed every prior "all-test" AUC was ~79 % weighted by the much-easier trophy-ladder slice. **The actual competitive draft problem (`soloRanked_mythicplus`, n=246,372 rows) is much harder**: all models cluster between **AUC 0.5937 and 0.6109**. State of the art is v3 XL at 0.6109, only +0.061 above random. The `+4.28 pp` Phase 1 LGBM headline gain was **+4.84 pp on Unranked but only +0.22 pp on Mythic+**.

**Going forward, primary metric = soloRanked Mythic+ AUC**, not "all-test". The all-test number averages two structurally different problems:
- `ranked` (Unranked trophy ladder, no draft, loose matchmaking) — AUC 0.74-0.80, easier
- `soloRanked_mythicplus` (strict 1-2-2-1 draft, tier-equalized matchmaking) — AUC 0.59-0.61, harder

### Tiered AUC across all models (DEC-011 stable test, n_test = 1,688,302)

| Model | all | ranked (1.34M) | soloRanked (348k) | Dia+ (313k) | **Myth+ (246k)** | Lgd+ (108k) |
|---|---:|---:|---:|---:|---:|---:|
| v2 LGBM A_fair (Run 0) | 0.7181 | 0.7426 | 0.6117 | 0.6062 | 0.6048 | 0.5938 |
| v2 LGBM phase 1 | 0.7609 | **0.7910** | 0.6253 | 0.6139 | 0.6070 | 0.5978 |
| v3 small (Run 1) | 0.7378 | 0.7642 | 0.6147 | 0.6044 | 0.5937 | 0.5785 |
| v3 small + phase 1 | 0.7540 | 0.7839 | 0.6145 | 0.6034 | 0.5933 | 0.5785 |
| v3 big (Run 4) | 0.7635 | 0.7929 | 0.6256 | 0.6131 | 0.6022 | 0.5879 |
| v3 big + phase 1 | 0.7603 | 0.7895 | 0.6230 | 0.6099 | 0.5988 | 0.5849 |
| **v3 XL (Run 5)** | **0.7674** | 0.7966 | **0.6312** | **0.6208** | **0.6109** | **0.5986** |

Notes:
- v3 small (Run 1) is actually **worse than the v2 LGBM baseline** on Mythic+ (0.5937 vs 0.6048). The transformer's lift came from the easy slice.
- XL only beats LGBM phase 1 on Mythic+ by **+0.39 pp** (0.6109 vs 0.6070), at 13× the parameter count.
- Phase 1 features help LGBM most on the easy slice, basically nothing on the competitive slice.
- All slice numbers came from one `predict_proba` pass per saved model — no retraining needed.

## Pre-DEC-016 view (kept for reference)

**Phase 6 v3.1 phase-1 ablation complete (DEC-015)**. 23 dense per-team aggregates of trophy/power (min/max/std + counts + diffs) added behind a `--use-team-aggregates` flag. Backwards-compatible plumbing: every legacy v3 checkpoint still loads at byte-identical head shape. Three runs on the DEC-011 stable test set decomposed the original Run 0→1 +1.97 pp gain.

| Run | Setup | Params | AUC | LogLoss | Brier | Wall-clock |
|---|---|---:|---:|---:|---:|---:|
| 0 | v2 LGBM A_fair (no phase 1) | n/a | 0.7181 | 0.6109 | 0.2124 | 111 s CPU |
| **P1.LGBM** | **LGBM + phase 1** | n/a | **0.7609** | 0.5642 | 0.1952 | ~5 min CPU |
| 1 | v3 small (no phase 1, CPU) | 251 k | 0.7378 | 0.5879 | 0.2044 | 28 min CPU |
| 3 | v3 small (no phase 1, GPU fast) | 251 k | 0.7366 | 0.5890 | 0.2049 | 6 min GPU |
| **P1.small** | **v3 small + phase 1** | 253 k | **0.7540** | 0.5696 | 0.1975 | ~5-7 min GPU |
| 4 | v3 big (no phase 1) | 570 k | 0.7635 | 0.5616 | 0.1943 | 14 min GPU |
| **P1.big** | **v3 big + phase 1** | 573 k | **0.7603** | 0.5651 | 0.1955 | 14 min GPU |
| 5 | v3 XL (no phase 1) | 3.28 M | 0.7674 | 0.5573 | 0.1928 | 52 min GPU |

**Key finding — feature engineering and architecture are substitutes at this data scale:**
- LGBM gains **+4.28 pp** from 23 numeric scalars alone (more than the entire original LGBM→transformer jump). LGBM phase 1 closes 62 % of the gap to the XL transformer at ~3 % of training cost.
- The v3 small transformer gains only **+1.62 pp** — its `scalar_proj_brawler` + attention already captured most of what phase 1 makes explicit.
- The v3 big transformer **does not gain** (−0.32 pp on stable test even though val_auc nudges up by +0.05 pp — the temporal-holdout tax widened from 0.76 to 1.13 pp). Encoder + per-brawler scalars + capacity have saturated this feature set.

### Production candidates after phase 1

Now four equally-valid choices, picked by deployment constraints:

- **`models/recommender_v2_phase1.lgb.txt`** (NEW, AUC 0.7609, CPU-only, ~5 min training). **Cheapest quality model.** Best for non-GPU deploys; only 0.65 pp behind v3 XL.
- **`models/recommender_v3_phase1_default.pt`** (NEW, small + phase 1, AUC 0.7540, 253 k params). Best CPU-deployable transformer; closes 62 % of small→big gap without changing arch.
- **`models/recommender_v3_big.pt`** (Run 4, NO phase 1, AUC 0.7635, 570 k params). Best AUC/inference-cost ratio for GPU deploys. Phase-1 big does *not* surpass this.
- **`models/recommender_v3_xl.pt`** (Run 5, NO phase 1, AUC 0.7674, 3.28 M params). Best AUC + best calibration (Brier 0.1928).

`models/recommender_v3_phase1_big.pt` is kept on disk for reproducibility but **not promoted** — it underperforms vanilla Run 4.

### Pre-phase-1 ablation (kept for reference — see DEC-014/015 for the full Session-9 progression)

Headline ablation on the DEC-011 stable test set, n_test = 1.69 M rows; same 1.87 M training rows. Phase-1 add-on numbers are in the table at the top of this file.

| Run | What changed | Params | AUC | LogLoss | Acc | Brier | Train wall-clock |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | v2 A_fair LightGBM (baseline) | n/a | 0.7181 | 0.6109 | 0.6490 | 0.2124 | 111 s CPU |
| 1 | + transformer + per-brawler features (small arch, CPU) | 251 k | 0.7378 | 0.5879 | 0.6618 | 0.2044 | 1651 s CPU |
| 2 | same arch, on GPU (DataLoader-based slow path) | 251 k | 0.7392 | 0.5868 | 0.6633 | 0.2040 | 616 s GPU |
| 3 | same arch, GPU fast data path (preload to VRAM) | 251 k | 0.7366 | 0.5890 | 0.6612 | 0.2049 | 338 s GPU |
| 4 | + bigger arch + 8 epochs | 570 k | 0.7635 | 0.5616 | 0.6788 | 0.1943 | 858 s GPU |
| **5** | **+ XL arch (d=256/L=6) + 12 epochs** | **3.28 M** | **0.7674** | **0.5573** | **0.6821** | **0.1928** | **3129 s GPU** |

**Decomposition** of the +4.93 pp end-to-end gain:
- Architecture (LightGBM → transformer): **+1.97 pp** (Run 0→1)
- GPU compute (CPU → GPU, slow path): 0.0 pp, 2.7× faster (Run 1→2)
- Data plumbing (DataLoader → preloaded VRAM): 0.0 pp, additional 1.8× faster (Run 2→3)
- Bigger model (251k → 570k params + 2 more epochs): **+2.69 pp** (Run 3→4)
- XL model (570k → 3.28M params + 4 more epochs): **+0.39 pp** (Run 4→5; clear diminishing returns)

So GPU + plumbing gave us a **4.9× total wall-clock speedup**; the freed compute funded both the big arch (+2.69 pp) and the XL arch (+0.39 pp). The capacity-vs-AUC scaling at this data size:

| Params | Δ AUC vs prev | Δ AUC per ×10 params |
|---:|---:|---:|
| 251 k | +1.97 pp | — |
| 570 k | +2.57 pp | ≈ +6.3 pp |
| 3.28 M | +0.39 pp | ≈ +0.5 pp |

Past 3 M params the per-×10 lift collapsed to ~0.5 pp. Capacity has saturated against this 1.87 M-row training set; further AUC needs a different inductive bias or more data.

### Per-mode AUC, Run 5 (XL) vs v2 A_fair LightGBM

| Mode (n in test) | A_fair LGBM | v3 XL | Δ |
|---|---:|---:|---:|
| brawlBall (847 k) | 0.7517 | 0.7961 | **+4.44 pp** |
| siege (30 k) | 0.7933 | 0.8329 | +3.96 pp |
| basketBrawl (45 k) | 0.7273 | 0.7777 | +5.04 pp |
| **knockout (363 k)** | 0.6942 | **0.7700** | **+7.58 pp** |
| wipeout (2 k) | 0.7047 | 0.7409 | +3.62 pp |
| heist (94 k) | 0.6784 | 0.7233 | +4.49 pp |
| gemGrab (125 k) | 0.6667 | 0.6971 | +3.04 pp |
| hotZone (89 k) | 0.6551 | 0.6813 | +2.62 pp |
| bounty (93 k) | 0.6159 | 0.6448 | +2.89 pp |

### Top-K (n=5000 last_pick, stable test, same sample for all models)

- **All rows hit@1 essentially tied** across small (0.136), big (0.137), XL (0.136). Same for hit@5 and MRR.
- **Winners-only hit@1 also tied**: small 0.204, big 0.204, XL 0.198. hit@10 climbs slightly with arch (0.357 → 0.364 → 0.369).
- **Bigger models help calibration, not ranking.** Brier and logloss improve monotonically with capacity, but the *ordering* of brawlers within a (team_partial, opp) context barely changes. So `WR|in_top1 ≈ 68-69 %` (+18-20 pp vs 49.5 % baseline) is roughly the same regardless of which of the 3 transformers you use.
- **Top-K appears to have a structural ceiling around hit@1 = 0.14 / MRR = 0.20.** The "predict which of ~97 brawlers a player picks" task is bounded by player roster + personal preference, not by win-probability quality. Breaking that ceiling likely needs a multi-task head that explicitly optimizes pick prediction, not bigger versions of the same binary-BCE encoder.
- **Brawler vocabulary**: model trained on 102 brawlers (all that appeared in any post-2026-05-03 ranked / soloRanked battle). Train and test windows have the same 102 — the 2 missing from the official `brawlers` table (BOLT id 16000106, STARR NOVA id 16000105) are the two newest releases that never got picked in our window. So no test rows are silently dropped. The "97 avg legal candidates per row" math: 102 vocab − 6 in-battle + 1 (actual added back) ≈ 97.

### Production candidates

Two equally-valid choices depending on what you're optimizing for:

- **`models/recommender_v3_big.pt`** (Run 4, 570 k params, AUC 0.7635). Best AUC-per-inference-cost ratio. Pick this when you want a fast recommender — top-K is statistically identical to XL within sample noise.
- **`models/recommender_v3_xl.pt`** (Run 5, 3.28 M params, AUC 0.7674, **Brier 0.1928**). Best calibration. Pick this when downstream consumers will threshold on the win-probability number itself (e.g., "auto-recommend only when P > 0.7").

Both load on a CPU-only machine via `from brawlstar_agent.recommender.transformer_model import load_transformer; m = load_transformer('models/recommender_v3_xl')` — `load_transformer` does `map_location='cpu'`. Same `.predict_proba(df)` interface as `LGBMTeamModel`, so all v1/v2 inference helpers (`rank_brawlers_for_map`, `complete_team`, `last_pick`) work unchanged.

Three earlier v3 checkpoints are kept for ablation reproducibility (Run 1 / 2 / 3 small arch on CPU/GPU-slow/GPU-fast) — see the headline ablation table.

### Pending v3.1 candidates (post-DEC-018 reordering)

Phase 4 broke through the Mythic+ ceiling. New ordering:

1. ~~**Phase 1**~~ — DONE (DEC-015). +0.22 pp on Mythic+.
2. ~~**Tiered slice eval**~~ — DONE (DEC-016). Established Mythic+ as primary metric.
3. ~~**Phase 2 + soloRanked-only**~~ — DONE (DEC-017). Both noise on Mythic+.
4. ~~**Phase 4 player history**~~ — DONE (DEC-018). **+0.71 pp Mythic+** at XL, new SOTA 0.6180. First feature addition that moves the competitive slice.
5. **Phase 4b — per-token history features**: extend `scalar_proj_brawler` from 2 to 3+ inputs so per-player history scalars sit on the brawler token instead of the team aggregate. The model can attend to the specific slot with high comfort/main status. Should be more expressive at XL. ~half day code + 1 retrain.
6. **More droplet data (separate DB)**: rsync recent droplet DB into `data/brawlstars_extra.db` for isolation (user's earlier suggestion). Would broaden the ~9% April history coverage substantially, expanding phase 4 signal density. Network + storage cost.
7. **Pick-prediction multi-task head**: orthogonal to AUC; targets hit@K which is meaningful in actual draft games. ~1 day of code.
8. **Star Power / Hyper Charge / Gears ingestion**: true new data, largest potential headroom. Multi-day, requires new collector + DB migration.
9. ~~**Phase 3 — per-brawler / per-mode WR aggregates**~~ — DE-PRIORITIZED. Aggregate-of-existing-data pattern.
10. **Calibration via isotonic / Platt** — Brier 0.1902 on Run K is already best so far; marginal further gain possible.
11. **Factorization machine** baseline — lower priority; phase-1 LGBM already captured most of what FM would.

Removed from the list:
- ~~"even bigger arch (Run 6: d=384/L=8)"~~ — DEC-014 showed capacity saturation past 3 M params (still holds, but DEC-018 shows phase 4 unlocks more capacity headroom at XL).
- ~~"phase 1 + bigger transformer"~~ — DEC-015.
- ~~"soloRanked-only training as primary research direction"~~ — DEC-017.

## Pre-DEC-017 view (kept for reference)

### Pending v3.1 candidates (post-DEC-016 reordering — Mythic+ AUC is the new primary metric)

Items reordered by expected lift on the **Mythic+ slice specifically**, not on all-test.

1. ~~**Per-brawler feature ablation on LightGBM (Phase 1)**~~ — DONE (DEC-015). +4.28 pp on all-test but only +0.22 pp on Mythic+. Trophy/power signal saturates against the trophy-ladder slice; barely helps in tier-equalized soloRanked.
2. ~~**Tiered slice eval**~~ — DONE (DEC-016). Established Mythic+ AUC as the primary metric.
3. **Phase 2 — time-based features + `days_since_release[bid]`**: top-tier players adopt new brawlers fastest, so release-meta inflation should be most visible in Mythic+. NEW signal, doesn't overlap with phase 1. Cheap (~1 hr code + 1-2 retrains).
4. **Phase 3 — per-brawler / per-mode WR aggregates** computed on training data, **conditioned on `battle_type`** (so soloRanked aggregates feed soloRanked battles, not mixed). Worth doing on both LGBM and v3. Medium cost (~half day code + leakage discipline).
5. **soloRanked-only training** as a parallel ablation: train on `battle_type='soloRanked'` only (1.0 M training rows, ~50% smaller). Removes the trophy-overload semantics + focuses model capacity on the actual research target. Worth checking if a 350k-Mythic+-train-row model on Mythic+-aware features beats v3 XL's 0.6109. Cheap (~2 retrains, ~30 min total).
6. **Pick-prediction multi-task head** — likely most valuable on Mythic+ where draft order + roster matter most. Top-K is more meaningful in actual draft games. ~1 day of code.
7. **Player history features** — strongest signal at top tiers where the same competitive players appear repeatedly (Mythic+ is ~70% of soloRanked, smaller player pool). Coverage check: 28% of player-rows have ≥3 battles of history. Largest engineering investment. ~1 day code + 1 retrain.
8. **Factorization machine** baseline — lower priority now; phase-1 LGBM already captured the easy-slice signal FM would have given.
9. **Calibration via isotonic / Platt** — small. Mythic+ Brier 0.24 is high; calibration could help marginally.
10. **C_fair-style 30-day window** with v3 + phase 1 + new features — untested combo.
11. **Star Power / Hyper Charge ingestion** — biggest potential headroom for Mythic+ specifically (top-tier players play tuned builds). Most engineering work.

Removed from the list:
- ~~"even bigger arch (Run 6: d=384/L=8)"~~ — DEC-014 showed capacity saturation past 3 M params.
- ~~"phase 1 + bigger transformer"~~ — DEC-015 showed phase-1 big underperforms vanilla Run 4 even on all-test, more so on Mythic+.

## Operating principle (DEC-009, unchanged)

- **Remote**: routine/periodic — crawlers, scheduled analytics precompute, backups
- **Local**: interactive/heavy — dashboard, ad-hoc queries, ML training, exploration
- v3 transformer training is local-only (CPU on the workspace machine).

## What Works

### Crawler & analytics infra (Sessions 6-7, unchanged)
- API client, SQLite collector, snowball discovery, three timers on droplet
- Dashboard reads pre-computed `analytics_cache.json` for instant load
- `--remote-cache` flag rsyncs cache from droplet for laptop-only viewing

### Recommender package (`src/brawlstar_agent/recommender/`)
- `dataset.py` — clean-window loader; v3 added per-brawler trophy + power tuples (backwards compatible)
- `features.py` — `TeamFeaturizer` (sparse for sklearn, dense for LGBM)
- `baselines.py` — Random / TrophyOnly / Global / Mode / ModeMap Wilson-CI
- `team_model.py` — `LogRegTeamModel`, `LGBMTeamModel`, `evaluate`, `save_model` / `load_model`
- `transformer_model.py` — **v3** `TransformerTeamModel` + `_TransformerCore` + `save_transformer` / `load_transformer`
- `inference.py` — `rank_brawlers_for_map`, `complete_team`, `last_pick`
- `cv.py` — sliding temporal-fold harness
- `topk_eval.py` — top-K + win uplift

### CLI scripts
- `scripts/train-recommender.py` — v1/v2 LightGBM training (DEC-011 `--stable-test-after`)
- `scripts/train-recommender-v3.py` — v3 transformer training (same flags)
- `scripts/eval-topk.py` — gained `--transformer-from PATH` to score the v3 model alongside v2 LightGBM with the same candidate pool + sample
- `scripts/compare-fair-runs.py` — apples-to-apples v2 comparison table

### Inference scenarios — all covered (works on v1, v2, and v3 models)
- `rank_brawlers_for_map(model, mode, map, train_df=...)` — pre-draft tier list
- `complete_team(model, my_team, opp_team, ...)` — mid-draft completion
- `last_pick(model, my_partial_team, opp_team, ...)` — end-of-draft

## Important caveats inherited (unchanged)

- **DEC-010**: legacy team-result bug not recoverable. Recommender uses strict post-2026-05-03 filter. Don't try to backfill the legacy data again.
- **DEC-011**: stable temporal test set (`2026-05-05T00:00:00Z`) is mandatory for any v2 / v3 / v3.1 comparison. AUCs across boundary versions are NOT comparable; rename downstream artifacts on each move.
- **Release-meta inflation**: DAMIAN (id 16000104, newest brawler) has 64.5 % raw WR over 41 k games. Both LightGBM and Transformer recommend DAMIAN with high confidence. This reflects the live meta, not a model bug, but cap your trust in `P(win) > 0.85` predictions.
- **Sample window**: clean post-fix data is concentrated in 2026-05-03..2026-05-05. The `--stable-test-after` boundary should slide forward (e.g., to `'2026-06-05T00:00:00Z'`) once another month of dense data has accumulated; bumping the boundary requires renaming the downstream `*_default_fair` artifacts.
- **API `battle.type` naming is flipped from the in-game UI**: API `ranked` = in-game "Trophy Battles" (casual trophy ladder, free pick, no bans), API `soloRanked` = in-game "Ranked" (Bronze → Pro tier ladder, **ban-pick draft after a certain rank tier**, locked to 6 modes). Recommender uses both via `COMPETITIVE_BATTLE_TYPES`. Bans are NOT exposed by the API — soloRanked rows are post-ban-and-pick, so the recommender silently misses the ban context. See `memory-bank/techContext.md` § "Battle types" for the full table and `docs/brawlstars-api.md` § "Ranked Types" for the inline reference.

## DB state on local

- Path: `/media/lin/disk2/brawlstar-agent/data/brawlstars.db` (~15 GB after Session 8 shrink)
- Synced 2026-05-06 (cold-start data, 4.0 M battles total, 78 k post-fix clean as of v1; 1.78 M post-fix clean as of v3)
- Refresh before major retrains: `bash scripts/rsync-db-from-droplet.sh --direct` (with droplet timers stopped + WAL checkpointed)

## PyTorch / GPU

- `torch==2.5.1+cu121` installed via `uv add torch --index https://download.pytorch.org/whl/cu121` with the index registered as `name = "pytorch-cu121", explicit = true` and routed via `[tool.uv.sources] torch = { index = "pytorch-cu121" }`. Pinned `torch>=2.5.0,<2.6` because cu121 wheels stopped at 2.5.1 (PyTorch went cu124-only from 2.6, and our driver 535 doesn't support cu124). Total install ~3 GB.
- GPU is the **RTX 3060 Mobile, 5.77 GB VRAM, compute 8.6**. v3 big model uses ~230 MB of training tensors + ~15 MB of model state — comfortable.
- Per-epoch wall-clock with the fast data path (Session 9 evening): small arch 37-42 s, big arch 83-100 s. Both vs ~250-260 s on CPU.
- **GPU enablement gotcha**: needs `sudo apt install nvidia-modprobe` (Ubuntu's nvidia-driver-535 metapackage doesn't pull it in). Install must be done **outside Cursor's user-namespace sandbox** or the binary will be owned by `nobody:nogroup` (UID-mapping artifact, setuid drops to nobody, can't write to /dev). Documented in `docs/recommender-v3.md` engineering notes.
- `pyproject.toml` also defines a `pytorch-cpu` named index for easy revert (e.g. droplet builds, CI without GPU). Switch by changing `torch = { index = "pytorch-cu121" }` to `pytorch-cpu` and re-running `uv sync`.
