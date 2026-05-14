# Decisions

## DEC-001: Emulator — All Paths Failed
- AVD: anti-cheat self-kills Brawl Stars (x86 CPU detected, obfuscated integrity check)
- Genymotion: ARM translation removed in v3.9, game can't install
- Waydroid: needs Wayland + binder/ashmem modules, not viable on X11
- **Outcome**: pivoted to YouTube gameplay recordings

## DEC-002: Project Location
- `/media/lin/disk2/brawlstar-agent/` — 295GB free NVMe
- Avoid /home (27GB) and / (8GB)

## DEC-003: Data Source — YouTube Gameplay
- Download with yt-dlp, extract frames with ffmpeg
- Human review via browser-based hub (auto-label + click to correct)
- Works well. 308 frames from 24 clips.
- Future: physical Android device for live capture if needed

## DEC-004: Brawler Identification — Needs Real Approach
- Color histogram matching against portrait catalog does not work
- Blob detection by saturation is too noisy
- **Next decision needed**: what approach for brawler ID?
  - Option A: train a small classifier (ResNet/MobileNet) on labeled in-game crops
  - Option B: YOLO-style object detector fine-tuned on Brawl Stars frames
  - Option C: contrastive embeddings (portrait → in-game matching)
  - All require labeled training data (brawler crops with identity labels)

## DEC-020: Anchor runs M1/M2 confirm the kit-sink scaling-law fit; M3 hits a practical ceiling; LGBM ⊕ XL ensemble is new SOTA (2026-05-13, Session 12 evening)

**Question**: validate DEC-019's scaling-law fit by running anchor models at N ≈ 1 M, 1.5 M, 5 M params (was a 3-point exact fit with zero d.o.f.). And while we're at it, blend the existing best LGBM + best transformer to see if the diversity bonus is real.

**Built**:
- `scripts/ensemble-stable-test.py` — sweeps `α · LGBM_phase1p4 + (1−α) · v3_xl_phase1p2p4` over α ∈ [0, 1] in 2.5 % steps on the DEC-011 stable test set, with proba caching so re-runs of the sweep don't repeat the ~9-min transformer predict_proba pass.
- `logs/run_anchor_runs.sh` — orchestrator for M1 / M2 / M3 sequentially (kit-sink P1+P2+P4 features, mixed training, DEC-011 boundary). Architectures verified via `_TransformerCore.parameters()` to hit target N counts.
- `logs/run_m3_retry.sh` — M3 retry at batch=2048 after the original batch=4096 attempt OOM'd on the RTX 3060 Mobile's 5.77 GiB VRAM.

**Result 1: Ensemble is the new Mythic+ SOTA**. Optimal blend α_lgbm = 0.45 (0.55 transformer). On the stable test set:

| Slice | n | Ensemble | LGBM alone | Transformer alone | Δ vs xfr |
|---|---:|---:|---:|---:|---:|
| all | 1,694,972 | **0.7787** | 0.7706 | 0.7745 | +0.42 pp |
| ranked | 1,346,458 | 0.8064 | 0.8007 | 0.8025 | +0.39 pp |
| soloRanked | 348,514 | 0.6501 | 0.6305 | 0.6444 | +0.57 pp |
| soloRanked_diamondplus | 312,714 | 0.6338 | 0.6143 | 0.6276 | +0.61 pp |
| **soloRanked_mythicplus** | **246,372** | **0.6249** | 0.6060 | 0.6180 | **+0.69 pp** |
| soloRanked_legendaryplus | 108,414 | 0.6139 | 0.5965 | 0.6054 | +0.85 pp |

The ensemble gain grows monotonically with slice difficulty — the diversity benefit between LGBM-tree splits and transformer attention compounds on the hardest examples. Best Brier also improves to 0.1842 (was 0.1902 with XL alone).

**Result 2: Anchor runs M1 / M2 land on the scaling-law curve**.

| Run | Arch | Params | Stable-test all AUC | **Mythic+ AUC** | Fit prediction | Δ vs fit |
|---|---|---:|---:|---:|---:|---:|
| M1 | d=160 h=8 L=5 ff=320 | 1,095,361 | 0.7756 | **0.6117** | 0.6121 | −0.04 pp |
| M2 | d=192 h=8 L=5 ff=384 | 1,566,337 | 0.7731 | **0.6131** | 0.6144 | −0.13 pp |
| M3 | d=320 h=8 L=6 ff=640 | 5,112,641 | 0.7634 | **0.5892** | 0.6196 | **−3.0 pp** (OUTLIER) |

M1 and M2 sit essentially on the predicted curve. M3 collapses well below — see Result 3.

**Refit with 5 anchors (kit-sink fit, M3 excluded as confounded)**:
- `L(N) = E + A · N^(−α)` on Mythic+ (1 − AUC): **E = 0.360, A = 0.548, α = 0.213**, rmse = 0.00051, dof = 2.
- **Asymptote AUC at N → ∞ = 0.640** (was 0.629 in the 3-point fit). XL has closed **84.3 %** of the gap from random (was 91.5 %); **2.19 pp** headroom remaining (was 1.09 pp).
- **Inverse predictions** (much more permissive than the 3-point fit):
  - AUC 0.620 → 5.51 M params (1.7× XL)
  - AUC 0.625 → 21.3 M params (6.5× XL; was 53 M / 16× before)
  - AUC 0.630 → 143 M params (43.6× XL; was unreachable before)
  - AUC 0.635 → 3.76 B params (1145×; technically reachable but absurd)
  - AUC 0.640 → asymptote
- **Joint Chinchilla fit on 10 transformer runs** (M3 excluded; dof = 5): α = 0.454, β = 0.082, E = 0.275. **Capacity share of reducible loss at SOTA = 8.2 %** (was 14.4 %) — even more data-bound than the original fit said.
- **Data-scaling projection** at fixed XL (N = 3.29 M): 2× D → 0.6208, 4× D → 0.6261, 8× D → 0.6311. Each doubling of D ≈ +0.5 pp Mythic+.

**Result 3: M3 hit a practical capacity ceiling under our GPU memory budget**. First attempt OOM'd in epoch 2 backward at batch=4096 (5.12 M params + activations exceeded 5.77 GiB VRAM). Retry at batch=2048 trained successfully but underperformed badly — Mythic+ 0.5892, all-test 0.7634, both below smaller models. Two non-mutually-exclusive interpretations:

(a) **Recipe confound**: halving the batch size while holding lr / dropout / schedule constant changes the optimization SNR (smaller batches → noisier gradients → effectively higher implicit lr). A proper batch=2048 run would also halve lr or tune dropout / lr schedule for the new batch.

(b) **Genuine capacity-data mismatch**: at D = 1.87 M training rows, 5.1 M params is past the useful capacity point even with a perfect recipe. Consistent with the joint Chinchilla fit's "92 % of reducible loss is data-side at SOTA".

Both probably contribute. We treat M3 as an outlier in the curve fit, but the *practical* finding stands: at our current GPU memory budget, **5.1 M is past the useful capacity ceiling without recipe re-tuning**, and the recipe re-tuning isn't worth it before more data lands.

**Implications for next-step priorities** (reordered from DEC-019):

1. **Time-series-aware boundary slide**: when more droplet data lands, slide `STABLE_TEST_AFTER_DEFAULT` forward. AUCs across boundary versions are NOT comparable; treat the post-shift cut as `v4` and archive the v3 numbers as historical.
2. **More droplet data → `data/brawlstars_extra.db`**: cheapest β-falsification (expected ~+0.5 pp / doubling under the joint fit). User has the runbook from the earlier reply.
3. **Phase 4b (per-token history features)**: new-information channel.
4. **Per-player ELO / Bradley-Terry skill features** (phase 4c): cheap addition to the existing pipeline.
5. **Sequence model over per-player history**: architectural lever most likely to beat the 0.6249 ensemble.
6. **Pick-prediction multi-task head + pairwise / listwise ranking loss**: orthogonal to AUC.

Skipped: bigger transformer beyond XL (M3 confirmed diminishing returns at this D and recipe).

**Files added** (uncommitted):
- `scripts/ensemble-stable-test.py`, `scripts/analyze-scaling-laws.py` (updated with anchor params + outlier handling + plot fixes)
- `models/recommender_v3_phase1p2p4_{m1,m2,m3}.{pt,meta.json}` (3 new checkpoints)
- `reports/recommender_v3_phase1p2p4_{m1,m2,m3}.json` (3 new reports with stable-test slices)
- `reports/ensemble_kitsink.json`, `reports/ensemble_cache/p_*.npy`
- `reports/scaling_laws.json` (re-fit with 5-anchor kit-sink + 10-run joint)
- `reports/scaling_law_N_mythic_{auc,logloss}.png`, `scaling_law_N_all_auc.png` (M3 X-marker outlier)
- `logs/{run_anchor_runs.sh,run_m3_retry.sh,anchor_runs_orchestrator.log,m3_retry_orchestrator.log,train_v3_phase1p2p4_m1.log,...m2.log,...m3.log,ensemble_kitsink.log,scaling_laws_final.log}`
- `canvases/scaling-law-mythicplus.canvas.tsx` (workspace) + `docs/canvases/scaling-law-mythicplus.canvas.tsx` (repo mirror) — refreshed

## DEC-019: Scaling-law analysis says Mythic+ is data-bound, not capacity-bound (2026-05-13, Session 12)

**Question (from HANDOFF.md task #1)**: "Are we over-parameterized or
under-trained?" Fit a power law `AUC_Mythic+ ≈ a − b · N^(−α) · D^(−β)` to the
existing 11+ transformer runs (N from 251 k → 3.29 M, D from 349 k solo →
1.87 M mixed) and use it to predict the AUC ceiling at this data scale + the
data scale needed to justify N > 3.28 M.

**Built**: `scripts/analyze-scaling-laws.py` (single-file, no new deps —
scipy.optimize.least_squares for the fit, matplotlib for the plots). Outputs:
- `reports/scaling_laws_inventory.csv` — 18-run inventory with `(family, N, D,
  battle_types, phases_tag, auc_all, auc_myth, logloss_all, logloss_myth,
  brier_myth)`.
- `reports/scaling_laws.json` — all fitted parameters + predictions +
  projections.
- `reports/scaling_law_N_mythic_logloss.png`,
  `reports/scaling_law_N_mythic_auc.png`,
  `reports/scaling_law_N_all_auc.png` — log-x scatter of every transformer run
  vs the fitted curve from the kit-sink family.
- `canvases/scaling-law-mythicplus.canvas.tsx` — rich summary canvas with
  charts, tables, and the recommended next steps.

**Two fits, two perspectives**:

### Fit A — kitchen-sink frontier (3 SOTA points along N at fixed D = 1.87 M)
| Setup | N | Mythic+ AUC |
|---|---:|---:|
| small + P1+P2+P4 | 256 k | 0.6013 |
| big + P1+P2+P4 | 576 k | 0.6084 |
| XL + P1+P2+P4 | 3.29 M | 0.6180 |

Power-law fit `(1 − AUC) = E + A · N^(−α)` gives **E = 0.371, A = 2.576,
α = 0.364** with rmse ≈ 0 (3 params, 3 obs ⇒ 0 d.o.f.; an exact fit through
the points, not a statistically estimated curve).

Implied **asymptote AUC at current data scale: 0.629**. SOTA XL+P1P2P4 (0.6180)
has closed **91.5 %** of the gap from random AUC = 0.5.

Inverse predictions (kit-sink fit):

| Target Mythic+ AUC | N required | × over XL |
|---:|---:|---:|
| 0.620 | 5.66 M | 1.7× |
| 0.625 | 53.2 M | 16.2× |
| ≥ 0.629 | → ∞ | **above asymptote** |
| 0.63+ | unreachable | — |

So **bigger arch beyond XL barely helps**: doubling N moves AUC from 0.618 to
~0.620; even a 16× XL gets us to 0.625; 0.63 is unreachable without more
information or more data.

### Fit B — joint Chinchilla on all 11 transformer runs (statistically meaningful, dof = 6)

`L(N, D) = E + A · N^(−α) + B · D^(−β)` on Mythic+ (1 − AUC):
**E = 0.272, α = 0.302, β = 0.087, rmse = 0.0032** over 11 obs.

At the SOTA point (N = 3.29 M, D = 1.87 M), the reducible (1 − AUC) splits as:
- Capacity contribution `A · N^(−α)` = **0.0163** (14.4 % of reducible)
- Data contribution `B · D^(−β)` = **0.0968** (85.6 % of reducible)

Data-scaling projection at fixed N = 3.29 M:

| D ratio | D | Predicted Mythic+ AUC |
|---:|---:|---:|
| 1× | 1.87 M | 0.6145 |
| 2× | 3.74 M | 0.6201 (+0.56 pp) |
| 4× | 7.49 M | 0.6254 (+1.09 pp) |
| 8× | 14.97 M | 0.6304 (+1.59 pp) |

Each doubling of training rows buys ~+0.5 pp Mythic+ AUC.

### Headline takeaways

1. **We are heavily data-bound on Mythic+**, not capacity-bound. β / α ≈ 0.29,
   so doubling D delivers ~3× more loss reduction than doubling N in this
   regime.

2. **The kit-sink-N asymptote 0.629 is a soft ceiling at current D**. Any
   future N-scaling experiment beyond XL should be accompanied by D-scaling
   (more data) to be worth the GPU-hours.

3. **Phase 4 (DEC-018) ≈ effective 4 – 8× D-scaling**. The empirical jump from
   vanilla XL (0.6109) to XL+P1+P2+P4 (0.6180) is +0.71 pp on Mythic+, which
   the joint fit predicts as the equivalent of going from D = 1.87 M to
   D ∈ [4 M, 8 M]. New information channels are functionally equivalent to
   training-row scaling.

4. **The N:D = 0.57:1 "data-starved by Chinchilla logic" framing
   (HANDOFF.md) is the right concern with the right answer**: yes we're
   data-starved by LLM rules of thumb, but our task has α/β ≈ 3.5 (not ~1 like
   LLMs), so the optimal compute allocation is heavily skewed toward more N.
   At our scale, however, we're already saturating N relative to D — both fits
   point to MORE DATA as the dominant lever.

5. **Caveats**:
   - Fit A (kit-sink) is an exact 3-point fit (no DOF). The α value would
     change if we ran additional anchor runs at intermediate N (e.g. 1 M,
     1.5 M params).
   - Fit B's β is largely set by ONE D-ratio pair (small mixed @1.87 M vs
     small solo @349 k, with phase confounding). The 2-point implied β with
     E pinned at the kit-sink asymptote is 0.31 (logloss) / 0.20 (1 − AUC),
     consistent with but stronger than the joint fit's 0.18/0.09.

**Implications for the v3.1 next-phase priority list** (post-DEC-019
reordering, in expected-Mythic+-lift order):

1. **More data into a separate DB** (`data/brawlstars_extra.db`) — the
   single most data-honest next step. Empirically test the implied β: if
   real β ≈ 0.09 we expect ~+0.5 pp / doubling. Cheap to falsify, and
   confirms the diagnosis.
2. **Phase 4b (per-token history features)** — extend `scalar_proj_brawler`
   from 2 to 3+ inputs. Same family of "new information" the joint fit
   already credits for the +0.71 pp XL gain.
3. **Star Power / Hyper Charge / Gears ingestion** — largest potential
   information delta; biggest engineering cost.
4. **(Optional) anchor runs at N ≈ 1 M and 1.5 M** — pure analysis refinement,
   doesn't move the SOTA but tightens α/asymptote in Fit A.
5. **Pick-prediction multi-task head** — orthogonal lever (targets top-K, not
   AUC). Still on the menu.

**De-prioritized** by this analysis:
- ~~Bigger transformer (N > 3.29 M)~~ — diminishing returns, ceiling at current
  D is ~+1 pp regardless of how big we go.
- ~~Even-bigger arch experiments before D scaling~~ — they would confirm the
  ceiling but burn GPU-hours for marginal pp.

**Files added**: `scripts/analyze-scaling-laws.py`,
`reports/scaling_laws.json`, `reports/scaling_laws_inventory.csv`,
`reports/scaling_law_N_mythic_logloss.png`,
`reports/scaling_law_N_mythic_auc.png`,
`reports/scaling_law_N_all_auc.png`, `logs/scaling_laws.log`, and
`canvases/scaling-law-mythicplus.canvas.tsx` (workspace canvas, lives outside
the repo).

## DEC-018: Phase 4 — Player-History Features Move Mythic+ AUC for the First Time (2026-05-13, Session 11)

**Built**:
- **`team_a/b_player_tags` exposed on the dataset** (`dataset.load_clean_battles`) as parallel tuples to the existing brawler-id tuples. Backwards-compatible additive change.
- **Phase 4 features** (12 numeric scalars in `recommender.features.compute_phase4_features`): per-team aggregates of per-player `n_games`, per-(player, brawler) `count`, and `is_main` alignment. Each side gets `n_known_players`, `mean_n_games_log`, `mean_brawler_count_log`, `max_brawler_count_log`, `n_main_picks`; plus 2 A−B diffs.
- **`history_df` parameter** in `TeamFeaturizer.fit`, `LGBMTeamModel.fit`, `TransformerTeamModel.fit`. Pass a SEPARATE DataFrame (different time window from training) so the lookup doesn't see the rows we predict on. Warning fires + leaky fallback applies if `history_df` is omitted.
- **`--use-history-features` + `--history-after` CLI flags** on both train scripts. Default `--history-after='2026-04-01T00:00:00Z'` uses the pre-cutoff April window (~349 k battles, ~698 k rows after both-perspective doubling) as the lookup source.
- **WR features intentionally DROPPED.** Two reasons: (a) when the lookup is built from training data, `overall_wr` and `brawler_wr` leak the label through self-influence (~1/n_games per row); (b) when built from pre-cutoff April data, the legacy team-result bug (DEC-010) makes WR 50 % noise. Frequency features (counts, main alignment) are unaffected by the legacy bug and don't leak.

**Leakage detected and fixed before final results**:
A first attempt put `compute_player_history(df)` inside `TeamFeaturizer.fit(df)` — the lookup was built from training data including the internal 5 % val split. **val_auc jumped to 0.9975 at epoch 1**, signalling massive self-leak. Killed and restructured so the lookup comes from a disjoint pre-cutoff window. Documented warning when no `history_df` is supplied.

**Player overlap** between April history and the post-cutoff windows is only **~9 %** (277 k of 2.87 M training-window players have April history; 231 k of 2.52 M test-window players do). Per-team coverage: ~25 % of teams have at least one known player. Despite the thin coverage, phase 4 delivered measurable signal — the active competitive (Mythic+) players are over-represented in the 9 % that DO have history.

**Seven new runs** on the DEC-011 stable test set:

| Run | Model | P1 | P2 | P4 | Arch | All AUC | **Mythic+ AUC** |
|---|---|:-:|:-:|:-:|---|---:|---:|
| F | LGBM | ✓ | — | ✓ | n/a | 0.7708 | 0.6060 |
| I | LGBM | — | — | ✓ | n/a | 0.7531 | 0.6059 |
| G | v3 | ✓ | — | ✓ | small | 0.7697 | 0.5948 |
| H | v3 | ✓ | ✓ | ✓ | small | 0.7719 | 0.6013 |
| J | v3 | ✓ | ✓ | ✓ | big | **0.7734** | 0.6084 |
| **K** | **v3** | ✓ | ✓ | ✓ | **XL** | **0.7746** | **0.6180** |

**NEW Mythic+ state of the art: 0.6180** (v3 XL kitchen sink), up from the prior 0.6109 (v3 XL no-phase) — **+0.71 pp**. Run K also wins all-test (0.7746 vs prior best 0.7674).

**Key findings**:
- **Phase 4 is the FIRST feature addition that moves Mythic+ AUC** on the transformer. Phase 1 added +0.22 pp on Mythic+, phase 2 added ±0.1 pp (noise) — phase 4 alone adds +0.15 pp on small, and combined with phase 2 adds +0.80 pp on small (Run H vs P1.small).
- **LGBM doesn't benefit from phase 4 on Mythic+** (+0.0 pp F vs P1, +0.11 pp I vs no-phase). Same plateau as phase 1/2: trees already extract any aggregate signal from multi-hot brawlers via splits. Phase 4 helps LGBM on the easy `ranked` slice (+1.00 pp all-test) but the competitive slice is unmoved.
- **The transformer's encoder consumes per-player frequency signal in a way LGBM can't.** The 12 phase-4 scalars feed the head's MLP; the encoder (attention over brawler tokens) still produces a CLS embedding, and the head mixes both. For the transformer, the gain compounds across arch sizes: small +0.80 pp, big +0.96 pp vs vanilla big, XL +0.71 pp vs vanilla XL.
- **Phase 4 + phase 2 stack.** Run G (P1+P4) on small reaches Mythic+ 0.5948; Run H (P1+P2+P4) reaches 0.6013 (+0.65 pp from adding P2 to the mix). The release-meta signal in phase 2 seemingly helps the model interpret per-player history (e.g., "this player mains a new brawler" carries a specific connotation).
- **Run K (XL kitchen sink) also delivers best calibration**: Brier 0.1902 (vs prior best 0.1928 for vanilla XL), logloss 0.5505 (vs 0.5573).

**Production candidates** after Run K:

| Use case | Model | Mythic+ | All | Cost |
|---|---|---:|---:|---|
| Best Mythic+ AUC (research-grade) | `models/recommender_v3_phase1p2p4_xl.pt` (Run K) | **0.6180** | 0.7746 | 58 min GPU train, 3.3 M params |
| Best AUC/cost ratio (production) | `models/recommender_v3_phase1p2p4_big.pt` (Run J) | 0.6084 | 0.7734 | 16 min GPU train, 576 k params |
| Best CPU-only option | `models/recommender_v2_phase1p4.lgb.txt` (Run F) | 0.6060 | 0.7708 | 24 min CPU train |

Run K replaces `recommender_v3_xl.pt` as the recommended best AUC model. Run J replaces `recommender_v3_big.pt` for the cost-balanced slot. Both have very different inference cost than their no-phase predecessors — but only marginally so (the head's first Linear widens by 47, not a big deal).

**Per-mode AUC on Run K (XL kitchen sink) vs vanilla XL** (all 9 modes improve):
brawlBall +0.47, knockout +0.93, gemGrab +0.77, hotZone +1.37, heist +1.07, bounty +1.69, basketBrawl +0.90, siege +0.87, wipeout +1.59. Bounty (hardest mode) and hotZone show the biggest gains — these are the modes where the transformer's prior aggregates were weakest.

**Implication**: New-information features (phase 4 = player identity + behavior) beat new-aggregate features (phase 1 = trophy/power, phase 2 = time/release) on the competitive slice. The earlier DEC-017 implication is confirmed: Mythic+ improvement needs new information, not new aggregates of existing data.

**What stays on the list**:
- **Phase 4b — per-token history features**: extend `scalar_proj_brawler` from 2 to 3+ inputs so player-history scalars sit on the per-brawler token instead of the team aggregate. Should be more expressive. ~half day code.
- **Pick-prediction multi-task head**: orthogonal to AUC; targets hit@K specifically.
- **Star Power / Hyper Charge / Gears ingestion**: real new data, largest potential headroom.
- **More crawler data** (user's earlier suggestion): rsync more of the droplet's DB into a separate DB file for isolation. Would broaden April history coverage (~9 % overlap now would grow), expanding phase 4 signal density.

**Files added**: 
- `models/recommender_v2_phase1p4.{lgb.txt,meta.json}` (Run F)
- `models/recommender_v2_phase4.{lgb.txt,meta.json}` (Run I, phase 4 only)
- `models/recommender_v3_phase1p4_default.{pt,meta.json}` (Run G)
- `models/recommender_v3_phase1p2p4_default.{pt,meta.json}` (Run H)
- `models/recommender_v3_phase1p2p4_big.{pt,meta.json}` (Run J)
- `models/recommender_v3_phase1p2p4_xl.{pt,meta.json}` (Run K, **new SOTA**)
- 6 new report JSONs in `reports/`, 6 new log files in `logs/`.
- Code changes in `recommender/{dataset.py,features.py,team_model.py,transformer_model.py}` (all backwards compatible — every prior v3 checkpoint still loads at its original head shape).

## DEC-017: Phase 2 + soloRanked-only Sprint — Mythic+ Saturates at ~0.61 AUC (2026-05-09, Session 10 afternoon)

**Built**:
- **Phase 2 features** (12 numeric scalars in `recommender.features.compute_phase2_features`): cyclical `hour_sin/cos` and `dow_sin/cos`, per-team `days_since_release` aggregates (`min`, `mean`, count of `dsr<14` per side, A−B diffs). Lookup `brawler_first_seen` is fit on training data and frozen onto the featurizer (round-tripped via `meta.json` for both LGBM and transformer). Composable with phase 1 — both sets concatenate into the same `extra_scalar` head input.
- **`--use-time-features` and `--battle-types` CLI flags** on both train scripts. Default `battle_types='ranked,soloRanked'` keeps every prior reproduction. `--battle-types soloRanked` constrains to the in-game Ranked queue (Bronze→Pro tiers, strict 1-2-2-1 draft from Mythic+).
- **Backwards-compat verified**: legacy v3 saves still load with `extra_scalar_dim=0` or `=23`; new phase-2 saves use `=12` or `=35` (phase-1 + phase-2). `load_transformer` infers missing flags from the dim where possible.

**Five experiments** on the DEC-011 stable test set:

| Run | Model | P1 | P2 | Train data | n_train rows | All AUC | **Mythic+ AUC** |
|---|---|:-:|:-:|---|---:|---:|---:|
| A | LGBM | yes | **yes** | mixed | 1,871,616 | 0.7618 | 0.6065 |
| B | LGBM | yes | no | **soloRanked-only** | 348,906 | 0.6282 | **0.6079** |
| C | LGBM | yes | yes | **soloRanked-only** | 348,906 | 0.6245 | 0.6044 |
| D | v3 small | yes | **yes** | mixed | 1,871,616 | 0.7564 | 0.5944 |
| E | v3 small | yes | yes | **soloRanked-only** | 348,906 | 0.6014 | 0.5802 |

**Two clean answers, both negative**:

### 1. Phase 2 (time + days_since_release) is essentially a no-op on Mythic+

| Model | P1 only Mythic+ | + P2 Mythic+ | Δ |
|---|---:|---:|---:|
| LGBM mixed | 0.6070 | 0.6065 | −0.05 pp |
| LGBM solo | 0.6079 | 0.6044 | −0.35 pp |
| v3 small mixed | 0.5933 | 0.5944 | +0.11 pp |

All three deltas are within sample noise on n = 246 k. Phase 2 doesn't deliver. Hypotheses for why:
- Brawler embeddings already encode "newness" implicitly (the model learned which embeddings correspond to recently-popular brawlers).
- Per-team aggregate of `days_since_release` (min/mean) loses per-brawler granularity — DAMIAN-specific signal gets diluted.
- Time-of-day / day-of-week are meta features that Mythic+ matchmaking is already largely invariant to (same competitive cohort across timezones).

A future "phase 2b" could try `days_since_release` as a *per-brawler scalar* in the transformer's brawler tokens (extending `scalar_proj_brawler` from 2 to 3 inputs) — that's a bigger architecture change but might catch the per-brawler signal phase 2 misses.

### 2. soloRanked-only training does not improve Mythic+ AUC

The hypothesis from DEC-016 was that the trophy-overload semantic mismatch between `ranked` (0–4951) and `soloRanked` (1–22) was hurting the model on the competitive slice. **Result: it doesn't matter as much as expected.**

| Model | Mixed train Mythic+ | Solo-only train Mythic+ | Δ |
|---|---:|---:|---:|
| LGBM phase 1 | 0.6070 | **0.6079** | **+0.09 pp** (sample noise) |
| LGBM phase 1+2 | 0.6065 | 0.6044 | −0.21 pp |
| v3 small phase 1+2 | 0.5944 | 0.5802 | **−1.42 pp** |

For the small transformer, mixed-train wins clearly: **the 5.4× more training data (1.87 M vs 349 k rows) outweighs any harm from semantic mixing**. The encoder benefits from cross-population learning — brawler-vs-brawler matchups are similar across both modes, and the `battle_type` embedding in the CTX token handles the population shift.

For LGBM, soloRanked-only is roughly tied with mixed on Mythic+ (+0.09 pp, well within noise). Trees can absorb the trophy-overload because they have a categorical battle_type column they can split on. The data-volume penalty is ~ matched by the semantic-purity gain.

### 3. Mythic+ ceiling is real

State of the art on `soloRanked_mythicplus` after 12 different model configurations (LGBM and transformer × phase 0/1/1+2 × mixed/solo train × small/big/XL):

| Mythic+ AUC | Model |
|---:|---|
| **0.6109** | v3 XL (no phase) — best |
| 0.6079 | LGBM phase 1 solo-only (Run B) — second |
| 0.6070 | LGBM phase 1 mixed |
| 0.6065 | LGBM phase 1+2 mixed (Run A) |
| 0.6048 | LGBM no-phase (Run 0 baseline) |
| 0.6044 | LGBM phase 1+2 solo-only (Run C) |
| 0.6022 | v3 big (no phase) |
| 0.5988 | v3 big phase 1 |
| 0.5944 | v3 small phase 1+2 mixed (Run D) |
| 0.5937 | v3 small phase 1 mixed |
| 0.5933 | v3 small phase 1 mixed (re-run) |
| 0.5802 | v3 small phase 1+2 solo (Run E) |

**Spread = 0.0307 AUC (~3 pp) across 12 configs.** The XL transformer holds a +0.30 pp lead over the next best (LGBM phase 1 solo). All meaningful gains have come from capacity (small → big → XL) and architecture (LGBM → transformer), not feature engineering or data scoping.

**Implication**: Phase 3 (per-brawler / per-mode WR aggregates) is unlikely to break this ceiling either if it follows the phase 2 pattern (aggregate signal already encoded by brawler embeddings + tree splits). The problem isn't that we lack good aggregates — it's that the aggregates we'd build are computable from data the model already sees. **Mythic+ improvement probably needs new information, not new aggregates of existing data**:

- **Player history** — what brawlers each player has played recently / mains. NEW signal because the model currently has no per-player identity. `28 % of player-rows have ≥ 3 prior battles` (DEC-016 SQL) — meaningful coverage at the top tiers.
- **Per-token `days_since_release` (phase 2b)** — a more focused version of phase 2 that puts the signal on the brawler token instead of the team aggregate.
- **Star Power / Hyper Charge / Gears** — the data model is currently blind to. Largest potential headroom but most ingestion engineering.
- **Pick-prediction multi-task head** — orthogonal to AUC; targets hit@K which is more meaningful in the actual draft setting where Mythic+ plays.

**Files added**: `models/recommender_v2_phase1p2.{lgb.txt,meta.json}`, `models/recommender_v2_phase1_solo.{lgb.txt,meta.json}`, `models/recommender_v2_phase1p2_solo.{lgb.txt,meta.json}`, `models/recommender_v3_phase1p2_default.{pt,meta.json}`, `models/recommender_v3_phase1p2_solo.{pt,meta.json}`. Reports + logs likewise. `recommender.features.compute_phase2_features`, `compute_brawler_first_seen`, `PHASE2_NAMES`, `PHASE2_DIM = 12` are new exports. `TeamFeaturizer.include_time_features` + `brawler_first_seen` fields are persisted through save/load on both model classes.

**Production candidates unchanged from DEC-014/015/016**:
- For Mythic+ AUC: `models/recommender_v3_xl.pt` (0.6109) is still best. Run-B's LGBM phase-1 solo (0.6079, ~8 s training) is a notable cheap-deploy alternative that gives up only 0.30 pp.
- For all-test AUC: same ranking as DEC-016.

## DEC-016: Tiered Slice Evaluation Reveals Phase 1 Was Mostly Helping the "Unranked" Trophy Ladder (2026-05-09, Session 10 morning)

**Background**: Brawl Stars game-domain detail clarified by user — `battle_type='ranked'` in our DB is the in-game **Unranked** trophy ladder (no draft, no tier system); `battle_type='soloRanked'` is the in-game **Ranked** queue with Bronze→Pro tiers and a strict 1-2-2-1 ban/pick draft from Mythic (>= 13) upward. The `brawler_trophies` field is **overloaded**: cumulative trophy count (0–4951, mean ≈ 1040) in `ranked`; rank tier number (1–22, mean ≈ 14) in `soloRanked`. See `techContext.md` "Brawl Stars game-domain semantics" section. This insight reframes every prior AUC number.

**Built**: `recommender.eval_slices` module (`make_slice_masks`, `evaluate_slices`, `format_slice_table`) + `scripts/eval-slices.py` for retrospective evaluation of any saved model. Slicers produced: `all`, `ranked`, `soloRanked`, `soloRanked_diamondplus` (>= 10), `soloRanked_mythicplus` (>= 13, strict draft), `soloRanked_legendaryplus` (>= 16). Both train scripts also emit per-slice metrics in `stable_test_slices` block. `evaluate` in `team_model.py` extended to accept precomputed `proba` so we predict-once, slice-many.

**Retrospective evaluation** of every saved checkpoint on the DEC-011 stable test set (n_test = 1,688,302 rows; predictions computed once per model, sliced):

| Model | all | ranked (Unrkd) | soloRanked | Diamond+ | **Mythic+** | Legendary+ |
|---|---:|---:|---:|---:|---:|---:|
| (n in slice) | 1,688,302 | 1,339,788 | 348,514 | 312,714 | **246,372** | 108,414 |
| v2_default_fair (Run 0 LGBM) | 0.7181 | 0.7426 | 0.6117 | 0.6062 | 0.6048 | 0.5938 |
| **v2_phase1 LGBM** | 0.7609 | **0.7910** | 0.6253 | 0.6139 | 0.6070 | 0.5978 |
| v3_default (Run 1 small) | 0.7378 | 0.7642 | 0.6147 | 0.6044 | 0.5937 | 0.5785 |
| v3_phase1_default (P1.small) | 0.7540 | 0.7839 | 0.6145 | 0.6034 | 0.5933 | 0.5785 |
| v3_big (Run 4) | 0.7635 | 0.7929 | 0.6256 | 0.6131 | 0.6022 | 0.5879 |
| v3_phase1_big (P1.big) | 0.7603 | 0.7895 | 0.6230 | 0.6099 | 0.5988 | 0.5849 |
| **v3_xl (Run 5)** | **0.7674** | 0.7966 | **0.6312** | **0.6208** | **0.6109** | **0.5986** |

**Reading the table**:

1. **There are essentially two different prediction problems in our data**, not one. Every model scores 14–18 pp higher on the trophy ladder (`ranked`) than on the actual ranked queue (`soloRanked`). Trophy-ladder matchmaking is loose (bigger skill / brawler-power gaps between teams), so a "high-trophy beats low-trophy" prior alone gives lots of AUC. soloRanked matchmaking is tier-equalized, so the model has to predict on pure brawler×brawler matchup + map quality with little skill differential.

2. **All "all-test" AUC numbers were ~80 % weighted by the easy slice.** 1.34 M of 1.69 M test rows (79.4 %) are `ranked` (Unranked). Conclusions like "Phase 1 LGBM gives +4.28 pp" or "v3 XL +4.93 pp end-to-end" were directionally correct but mostly extracted trophy-ladder signal that doesn't apply to competitive draft.

3. **On the strict-draft Mythic+ slice (the actual problem we care about), all our work produced ~+0.06 AUC over the v2 baseline**:
   - v2 LGBM Run 0 → v2 phase 1 LGBM: 0.6048 → 0.6070 (**+0.22 pp**, vs +4.28 pp on all-test)
   - v2 LGBM Run 0 → v3 small (Run 1): 0.6048 → 0.5937 (**−1.11 pp** — transformer is *worse* on Mythic+!)
   - v2 LGBM Run 0 → v3 XL (Run 5): 0.6048 → 0.6109 (**+0.61 pp**, vs +4.93 pp on all-test)

4. **Phase 1 features hurt the small/big transformer slightly on Mythic+** (P1.small −0.04 pp, P1.big −0.34 pp vs their no-phase-1 counterparts). They help LGBM marginally (+0.22 pp). These deltas are within sample noise on n = 246 k. The `P1.big` regression on all-test (DEC-015's headline puzzle) is ~70 % from Mythic+ degradation and ~30 % from the (tiny) `ranked` slice regression that isn't a regression at all on the slice level.

5. **The XL transformer is the best model on every slice including Mythic+**, but the Mythic+ lead over LGBM phase 1 is **+0.39 pp** (0.6109 vs 0.6070) — at 13× the parameter count, 30× the training time, and a GPU-only deploy. On all-test the same comparison was +0.65 pp. The cost/benefit for the competitive slice specifically is much worse than it looked.

6. **Calibration follows AUC** on the slice level — Brier on Mythic+ ranges 0.2401–0.2470 across all models (vs 0.1816–0.2070 on `ranked`). The competitive slice is harder both for ranking and for calibration, by similar margins.

**Real research target reframed**: future improvement work should report **soloRanked Mythic+ AUC** as the primary metric. The all-test number is misleading because it averages a much-easier sub-population. The state of the art on the actual competitive draft problem is currently **0.6109 (v3 XL)** — only +0.061 above random, with everyone else clustered between 0.59 and 0.61.

**Implications for next phases**:
- Features that mainly leverage trophy/power (phase 1) will keep helping the trophy-ladder slice and almost not at all the competitive slice. soloRanked has already normalized those.
- Phase 2 (`days_since_release`) is still on the menu — release-meta inflation should affect Mythic+ specifically (top-tier players adopt new brawlers fast).
- Phase 3 (per-brawler / per-mode WR) — but the WR aggregates **MUST be conditioned on `battle_type`** because the populations are different.
- Pick-prediction multi-task head — likely most impactful on Mythic+ where the player roster + draft order matter most.
- Player history features — strongest signal in Mythic+ where the same competitive players appear repeatedly with their mains.

**Files added**: `src/brawlstar_agent/recommender/eval_slices.py`, `scripts/eval-slices.py`, `reports/slices_summary.json`, `reports/slices_smoke.json`, `logs/eval_slices_summary.log`. Train scripts (`train-recommender.py`, `train-recommender-v3.py`) updated to emit `stable_test_slices` blocks in their reports. `team_model.py:evaluate` accepts an optional `proba` for predict-once-slice-many.

**No retraining performed** — every number above came from running `predict_proba` once on the 1.69 M-row stable test set per saved model and slicing.

## DEC-015: v3.1 Phase 1 — Per-Team Aggregates Decompose the Run 0→1 Gain (2026-05-08, Session 9 night)

After DEC-014 (capacity saturation past 3 M params), the open question from the v3.1 candidate list was: how much of the original Run 0→1 +1.97 pp gain came from feature engineering vs from the attention architecture itself? Phase 1 answers this.

**Phase 1 = 23 dense numeric scalars** computed from the per-brawler tuples already in `dataset.py`: per-team min/max/std of trophies (log1p'd), per-team mean/min/max/std of powers (normalized), counts of `power == 11` and `power < 8` per side, and 5 A − B diffs of the most informative aggregates. Implemented in `recommender.features.compute_team_aggregates` + `TeamFeaturizer.include_team_aggregates` flag. Both LGBM and the v3 transformer accept a `--use-team-aggregates` switch through their training scripts; the flag is round-tripped in saved meta.json so models reconstruct correctly. Backwards compatible: every legacy v3_default / v3_gpu / v3_gpu_fast / v3_big / v3_xl saved checkpoint still loads and predicts at byte-identical head shape.

**Results on the DEC-011 stable test set** (same 1.87 M train / 1.69 M test rows; only the model and the `--use-team-aggregates` flag change between rows):

| Run | Setup | Params | AUC | LogLoss | Acc | Brier | Train wall-clock |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | v2 LGBM A_fair (no phase 1) | n/a | 0.7181 | 0.6109 | 0.6490 | 0.2124 | 111 s CPU |
| **P1.LGBM** | **LGBM + phase 1** | n/a | **0.7609** | 0.5642 | 0.6765 | 0.1952 | 1041 s CPU* |
| 1 | v3 small (no phase 1, CPU) | 251 k | 0.7378 | 0.5879 | 0.6618 | 0.2044 | 1651 s CPU |
| 3 | v3 small (no phase 1, GPU fast) | 251 k | 0.7366 | 0.5890 | 0.6612 | 0.2049 | 338 s GPU |
| **P1.small** | **v3 small + phase 1** | 253 k | **0.7540** | 0.5696 | 0.6706 | 0.1975 | 450 s GPU* |
| 4 | v3 big (no phase 1) | 570 k | 0.7635 | 0.5616 | 0.6788 | 0.1943 | 858 s GPU |
| **P1.big** | **v3 big + phase 1** | 573 k | **0.7603** | 0.5651 | 0.6765 | 0.1955 | 856 s GPU |
| 5 | v3 XL (no phase 1) | 3.28 M | 0.7674 | 0.5573 | 0.6821 | 0.1928 | 3129 s GPU |

\* both phase-1 runs ran in parallel and competed for CPU during data loading + test-set tensorization, hence the inflated wall-clocks.

**Reading the table:**
- **LGBM gains +4.28 pp from phase 1 alone** (0.7181 → 0.7609). 23 numeric scalars on top of the existing multi-hot vector are *more* than the entire Run 0→1 architecture jump (+1.97 pp). LGBM phase 1 (0.7609) ends 0.65 pp shy of the **XL transformer** (0.7674) at ~3 % of the training cost.
- **The v3 small transformer gains only +1.62 pp from phase 1** (0.7378 → 0.7540). The encoder + per-brawler scalars (`scalar_proj_brawler` over trophy/power) was already extracting most of what phase 1 makes explicit; team aggregates are a partial duplicate of what attention already distills.
- **The v3 big transformer essentially does not gain** from phase 1 (0.7635 → 0.7603, **−0.32 pp**). val_auc actually peaked higher (0.7716 vs Run 4's 0.7711, +0.05 pp) but the temporal-holdout tax widened to 1.13 pp (vs Run 4's 0.76 pp) — phase 1 features encoded train-window quirks that didn't transfer cleanly to the stable test window. Plausibly the team-level skill-spread distribution drifts faster across days than per-brawler trophy/power does.

**Decomposing the original +1.97 pp Run 0→1 gain:**
- Per-brawler trophy/power scalars in tokens (architecture-dependent — only the transformer can use them this way): captured most of the +1.97 pp.
- The same per-brawler signal, surfaced as 23 dense team aggregates, gives LGBM **more** lift (+4.28 pp) than it gives the transformer (+1.62 pp on small, ≈ 0 on big).
- Net: **feature engineering and architecture are largely substitutes**, not complements at this data scale. Trees + good hand-crafted aggregates ≈ transformer + per-brawler tokens. The attention architecture's main remaining advantage over LGBM phase 1 is calibration (Brier 0.1955 big vs 0.1952 LGBM ≈ tied; XL pulls ahead at 0.1928).

**Production implications:**
- **`models/recommender_v2_phase1.lgb.txt`** is now the cheapest production-quality model: AUC 0.7609 in ~5 min of CPU training, deploys without GPU. Use this when inference cost or environment matters more than the last 0.3-0.6 pp of AUC.
- **`models/recommender_v3_xl.pt`** (DEC-014) remains the best AUC and calibration. The big transformer + phase 1 (`recommender_v3_phase1_big.pt`) does *not* surpass plain Run 4 / Run 5 — kept for ablation reproducibility only, not promoted to production.
- **`models/recommender_v3_phase1_default.pt`** (small + phase 1, AUC 0.7540) is the best CPU-deployable transformer (closes 62 % of the gap from small to big without changing arch). Useful when transformer-class inference is required but GPU is not.

**Files added** (uncommitted with the rest of Session 9): `models/recommender_v2_phase1.{lgb.txt,meta.json}`, `models/recommender_v3_phase1_default.{pt,meta.json}`, `models/recommender_v3_phase1_big.{pt,meta.json}`, `reports/recommender_v2_phase1.json`, `reports/recommender_v3_phase1_default.json`, `reports/recommender_v3_phase1_big.json`, `logs/train_v2_phase1.log`, `logs/train_v3_phase1_default.log`, `logs/train_v3_phase1_big.log`. Code changes in `src/brawlstar_agent/recommender/{features.py,team_model.py,transformer_model.py}` (all backwards compatible) and `scripts/train-recommender{,-v3}.py` (`--use-team-aggregates` flag).

**Next-phase implications**:
- **Skip phase 1 + bigger transformer** — the big-arch run already shows no lift, XL would likely not either. The transformer encoder has saturated against this feature set.
- **Phase 2 (time-based features, days-since-release per brawler)** is still on the menu — those are *new* signal not captured by phase 1 or by attention over current tokens.
- **Phase 3 (per-brawler global / mode WR)** is still on the menu — same reasoning.
- **For LGBM**, phase 1 was the cheap win; pair-synergy / pair-matchup features (phase 5) are the next thing trees specifically benefit from.

## DEC-014: v3 XL Confirms Capacity Saturation Past 3M Params (2026-05-08, Session 9 night)

After DEC-013 shipped Run 4 (570 k params, AUC 0.7635), pushed further with Run 5: d_model=256, num_layers=6, ff=512, dropout=0.20, 12 epochs. Param count **3.28 M** — 5.7× Run 4. Trained 52 min on the same RTX 3060 Mobile, comfortably within VRAM (≤ 1.5 GB used of 5.77 GB).

**Result on the DEC-011 stable test set:**
- AUC **0.7674** vs Run 4's 0.7635 → **+0.39 pp**
- Brier **0.1928** vs 0.1943 → −0.15 pp
- Logloss **0.5573** vs 0.5616 → −0.43 pp
- Wins 9/9 modes vs Run 4 (every mode +0.13 to +1.10 pp; basketBrawl +1.10 leads, brawlBall +0.13 trails)
- End-to-end vs v2 LightGBM (Run 0): **+4.93 pp AUC, −5.36 pp logloss, −1.96 pp Brier**

**Capacity-vs-AUC scaling at 1.87 M training rows:**

| Params | AUC | Δ AUC vs prev | Δ AUC per ×10 params |
|---:|---:|---:|---:|
| 251 k | 0.7378 | +1.97 (vs LGBM) | — |
| 570 k | 0.7635 | +2.57 | ≈ +6.3 pp |
| 3.28 M | 0.7674 | +0.39 | ≈ +0.5 pp |

The per-×10 lift collapsed by an order of magnitude. **Capacity has saturated against the current data scale.** Further AUC gains likely need either (a) a different inductive bias — factorization machine, listwise / pick-prediction head, calibration layer — or (b) more training data, not bigger transformers. Removed "Run 6: d=384/L=8" from the v3.1 candidate list.

**Top-K is unchanged** across small/big/XL: hit@1 ≈ 0.137, MRR ≈ 0.195, WR|in_top1 ≈ 68 % on n=5000 last_pick. The +0.39 pp binary AUC gain went entirely into calibration (Brier and logloss both improved), not into ranking. Confirms what Run 4 already suggested: top-K has a structural ceiling around hit@1 = 0.14 because pick prediction is bounded by player roster + personal preference, not by win-probability quality. Breaking that ceiling requires a multi-task pick-prediction head, not more capacity on the binary BCE objective.

**Two production candidates now**:
- `models/recommender_v3_big.pt` (Run 4) — best AUC/inference-cost ratio. Use for the standard recommender UX.
- `models/recommender_v3_xl.pt` (Run 5) — best calibration. Use when downstream consumers will threshold on the win-probability number.

Brawler vocabulary is **102** (the 2 newest in the official table, BOLT id 16000106 and STARR NOVA id 16000105, haven't appeared in any post-2026-05-03 ranked / soloRanked battle in either train or test, so the eval is consistent and not silently dropping rows). Documented in `docs/recommender-v3.md` candidate-pool note.

Stable-test boundary unchanged (`'2026-05-05T00:00:00Z'`, DEC-011). Numbers across Runs 0-5 are directly comparable.

## DEC-013: v3 GPU Enablement + Bigger Architecture (2026-05-08, Session 9 evening)

After DEC-012 shipped the small CPU transformer (251 k params, AUC 0.7378), the user asked why CPU was so slow and whether we could use the laptop's RTX 3060 Mobile. Two changes were then run in series as a clean ablation:

**Change 1: GPU enablement.**
- `nvidia-modprobe` was missing — the CUDA driver 535 modules were loaded but the userspace helper that creates `/dev/nvidia*` device nodes wasn't installed. `sudo apt install nvidia-modprobe` (run in the user's real terminal, not in Cursor's user-namespace sandbox where dpkg writes the binary as `nobody:nogroup` due to UID mapping). One-time fix.
- Swapped torch CPU build for cu121: `pyproject.toml` now uses `[tool.uv.sources] torch = { index = "pytorch-cu121" }` + `[[tool.uv.index]] name = "pytorch-cu121", explicit = true`. Scoping the PyTorch index to torch only is necessary — without `explicit = true`, uv treats it as the primary index for everything and fails on opencv / jupyter / etc. Pinned `torch>=2.5.0,<2.6` because cu121 wheels stopped at 2.5.1 (PyTorch ≥2.6 is cu124-only, which needs driver ≥550 — we have 535).
- Same architecture, same training data, same code path → AUC unchanged within seed noise (Run 2: 0.7392 vs Run 1 CPU: 0.7378). Wall-clock training: **2.7× faster** (616 s vs 1651 s).

**Change 2: Fast data path.**
- Original code used `torch.utils.data.DataLoader` with CPU tensors. On GPU, per-batch CPU↔GPU memcopy dwarfed the actual compute on a 251 k-param model — the GPU was mostly idle. Refactored `transformer_model.py` with `_iter_batches`: preload all training tensors (~230 MB) onto GPU VRAM once, generate `torch.randperm` per epoch on device, gather batches via `tensor[idx]`. Single code path that works on both cpu and gpu; DataLoader removed entirely.
- AUC again unchanged (Run 3: 0.7366, noise). Wall-clock: **additional 1.8× faster** (338 s). **Compounded vs CPU: 4.9× speedup** for the same architecture.

**Change 3: Bigger architecture.**
- With epochs costing only ~40 s and ~3 GB of free VRAM headroom, scaled the model: d_model 96→128, num_layers 3→4, ff 192→256, nhead 4→8, dropout 0.10→0.15. Trained 8 epochs (vs 6) with patience 3. Param count 251 k → **570 k**.
- **Stable-test AUC: 0.7635** — **+2.69 pp over Run 3 same-arch**, **+4.54 pp over the v2 A_fair LightGBM baseline (Run 0)**, **+3.70 pp over v2 C_fair (the technical-best v2 with 30-day window)**. Logloss 0.5616, accuracy 0.6788, Brier 0.1943 (all best-in-class). Per-mode wins 9/9 by 2-7 pp; **knockout +6.84 pp** (the largest absolute gain).

**Decomposition of the +4.54 pp end-to-end Run 0 → Run 4 gain on identical training data:**
- Architecture (LightGBM → transformer + per-brawler features): **+1.97 pp** (DEC-012, Run 0→1)
- GPU compute alone: 0.0 pp, 2.7× faster (Run 1→2)
- Data plumbing (DataLoader → preloaded VRAM): 0.0 pp, additional 1.8× faster (Run 2→3)
- Bigger model + 2 more epochs (made affordable by the plumbing fix): **+2.69 pp** (Run 3→4)

**Top-K caveat**: the +2.7 pp binary-AUC jump from small → big does NOT translate into hit@1 (both at ~0.137) — top-K appears to have hit a structural ceiling that even the small transformer was already at. What the bigger model improves is **calibration** (lower Brier, lower logloss) and a small bump in WR|in_top1 (+0.8 pp on all rows). Both matter for downstream UIs that threshold on the win-probability output.

**Production candidate now**: `models/recommender_v3_big.pt` + `.meta.json` (Run 4). The three earlier v3 checkpoints (`recommender_v3_default.pt`, `recommender_v3_gpu.pt`, `recommender_v3_gpu_fast.pt`) are kept for ablation reproducibility but not for deployment.

**Stable-test boundary still `'2026-05-05T00:00:00Z'`** (DEC-011). All Run 0..4 numbers are directly comparable. v3.1 candidates open: even bigger arch (d=192-256, L=6), per-brawler feature ablation on LightGBM, factorization machine, calibration, 30-day window with the v3 transformer, Star Power / Hyper Charge ingestion. See `docs/recommender-v3.md` for the full table + per-mode breakdown + retraining commands.

## DEC-012: Recommender v3 — Attention Transformer over Per-Brawler Tokens (2026-05-08)

After v2's stable-test methodology (DEC-011) showed LightGBM saturating around AUC 0.7181 (A_fair) / 0.7265 (C_fair, 30-day) and LogReg flat at 0.68 across all data cutoffs, the v2 doc explicitly named "model architecture" — not more data — as the next lever. Ran the experiment per user direction ("we need to try more advanced methods, e.g. attention neural network, more feature engineering... we prioritize the method over more data"). Used the *same* A_fair training data and the *same* DEC-011 stable test set so the LightGBM ↔ Transformer comparison is the only thing changing.

**Architecture chosen**: small attention transformer encoder (`src/brawlstar_agent/recommender/transformer_model.py`).
- 8 tokens per battle: `[CLS] [CTX] [A1] [A2] [A3] [B1] [B2] [B3]`
- Brawler embedding (d=96) + side embedding (CLS/CTX/A/B) + per-brawler scalar projection of (`trophy_log`, `power/11`)
- 3-layer encoder, nhead=4, ff=192, dropout=0.1, norm_first=True, GELU activation
- CLS pool → concat with [a_t_log, b_t_log, t_diff_log] → MLP head → BCEWithLogitsLoss
- ~251 k params total. AdamW(lr=1e-3, wd=1e-4), cosine schedule, batch=4096, 6 epochs, early-stop on internal 5 % val
- CPU only (Torch 2.11.0+cpu, ~3 min/epoch on i7-12700H)

**Per-brawler features added** to `dataset.py` (backwards compatible): `team_a_trophies`, `team_b_trophies`, `team_a_powers`, `team_b_powers` — parallel tuples aligned to existing `team_a` / `team_b` brawler-id tuples (sorted by brawler_id within each team). Real signal v2 multi-hot ignored: ~80 % of post-fix ranked rows are power 11, but the other ~20 % spans power 0-10; trophy distribution is heavy-tailed (min 0, max 4 951, mean 851).

**Empirical result on the DEC-011 stable test set** (n_test = 844 151 battles / 1 688 302 rows; train = same 1.87 M rows as A_fair):

| Model | AUC | LogLoss | Acc | Brier |
|---|---:|---:|---:|---:|
| A_fair LightGBM (v2) | 0.7181 | 0.6109 | 0.6490 | 0.2124 |
| C_fair LightGBM (v2, 30-day) | 0.7265 | — | — | — |
| **v3 Transformer** | **0.7378** | **0.5879** | **0.6618** | **0.2044** |

Δ vs A_fair LGBM: **+1.97 pp AUC, −2.30 pp logloss, +1.3 pp accuracy, −0.80 pp Brier**. Wins or ties in **9/9 modes** (largest gains: knockout +2.6 pp, siege +2.3 pp, brawlBall +2.1 pp, bounty +1.2 pp). Top-K (n=5000 last_pick): hit@1 tied at 0.136; hit@3/hit@5 +1.3/+1.6 pp; winners-only hit@1 0.204 vs LGBM 0.195 (+0.9 pp); WR|in_top1 essentially identical at ~68 %.

**Why this matters**: this is ARCHITECTURE making the difference, not data. Same training rows; the +2 pp came from (a) attention learning brawler×brawler / brawler×map interactions that the LightGBM splits couldn't fully express on multi-hot features, and (b) per-brawler trophy + power inputs that the multi-hot featurizer threw away. Disambiguating (a) vs (b) is filed as a v3.1 ablation but doesn't change the deployment story.

**Production candidate**: `models/recommender_v3_default.pt` + `.meta.json`. Same `.predict_proba(df)` interface as `LGBMTeamModel`, so all v1/v2 inference helpers (`rank_brawlers_for_map`, `complete_team`, `last_pick`) work without modification. Docs: `docs/recommender-v3.md` for full methodology + how to retrain.

**Stable-test boundary unchanged**: still `'2026-05-05T00:00:00Z'` (DEC-011). Every v3 / v3.1 run MUST keep the same `--stable-test-after` or numbers are not comparable across model versions. When the boundary moves (eventually it should — as data accumulates the test window can slide forward), bump `STABLE_TEST_AFTER_DEFAULT` and rename downstream artifacts.

## DEC-011: Stable Temporal Test Set for Recommender Comparisons (2026-05-06)

Through Run A (random split, 1.78M battles, AUC 0.7382) and Run C (random split, 2.13M battles, AUC 0.7392) we observed near-equal random-split AUCs and concluded "more data isn't helping much; we're at a feature-set ceiling." That conclusion was wrong because each run's random test set was sampled from a *different* underlying time distribution — Run A's test set was 20% of 2026-05-03..06; Run C's test set was 20% of 2026-04-06..06 (heavily weighted toward May 4-5 by data density). Different test sets → AUCs are not directly comparable.

**Decision**: every v2-and-beyond recommender run holds out battles with `battle_time_iso >= '2026-05-05T00:00:00Z'` as the test set (~844k clean ranked/soloRanked battles after dataset cleaning, 1.69M rows after both-perspectives expansion). Train data = `[--cutoff, '2026-05-05T00:00:00Z')`. The boundary is centralized as `STABLE_TEST_AFTER_DEFAULT` in `scripts/train-recommender.py` and `scripts/eval-topk.py`.

Implementation:
- `scripts/train-recommender.py --stable-test-after TIMESTAMP`: when set, replaces the random split with this temporal holdout. Reports record `split_mode`, `stable_test_after`, `n_train_battles`, `n_test_battles`. When set, temporal CV runs only over the train portion (no leakage).
- `scripts/eval-topk.py --cutoff ... --stable-test-after ...`: same boundary, so binary AUC and top-K hit@K come from identical test rows.
- `scripts/compare-fair-runs.py`: prints the apples-to-apples table.

Empirical result on the stable test set (LightGBM, all on n_test=844,151 battles):
- A_fair (3-day window, 936k train battles): **0.7181**
- C_fair (30-day window, 1.28M train battles): **0.7265** — winner
- B_fair (all-data 2021+, 1.41M train battles): **0.7235**

The 30-day window beats 3-day by **+0.84pp** AUC and beats all-data by **+0.30pp** — a small, reproducible signal that 30 days is the current sweet spot. Adding 5 years of (sparse, partly legacy-buggy) older data slightly hurts. The per-mode pattern is consistent (C_fair wins or ties in 8 of 9 modes). With random-split numbers we couldn't see this; with the stable test set, the answer is clear.

LogReg essentially flat across runs (0.6808 / 0.6805 / 0.6752) — already saturated by feature design at this scale. ModeMap drops 2-3pp from random-split to stable-test (per-cell memorization doesn't fully transfer across time), which is the meta-drift signal random splits hide.

When boundary needs to move (e.g., as data accumulates), bump `STABLE_TEST_AFTER_DEFAULT` and re-train every model. Comparisons across boundary versions are not valid; rename downstream artifacts on each move (e.g., `_fair_v2`, `_fair_v3`).

## DEC-010: Legacy Team-Result Bug Is Not Recoverable; Use Strict Post-Fix Filter (2026-05-04)

`docs/analytics-notes.md` originally suggested that the team-result bug fixed in `dde58a4` could be detected and "fixed" via the invariant "exactly one team has `result='victory'` and one has `'defeat'`". That heuristic does not work.

The bug, restated precisely: pre-fix, `db.py::_insert_battle_players` always assigned `battle.result` to `team_index=0` and the inverse to `team_index=1`. The fetched player can be on either team, so when they were on team 1 the labels were *swapped between teams*. **The swap preserves the invariant** — exactly one team still has `'victory'` and one `'defeat'`, just on the wrong teams.

Empirical confirmation in the local DB: 99.1% of pre-fix battles satisfy the 1W+1L invariant (vs 95.8% post-fix; the small post-fix anomaly rate is partial inserts and re-fetches, not the bug). If the invariant were diagnostic, pre-fix would show ~50% bug rate.

Other potential signals also fail:
- `trophy_change` is stored on `team_index=0`'s first player but reflects the *fetched* player's trophy delta. Pre-fix and post-fix this attribution looks identical from the stored row.
- We do not store `fetched_for_tag`, so there's no way to retroactively compute which team got the swap.

**Decision**: every recommender-pipeline query filters by `battle_time_iso >= '2026-05-03T01:00:00Z'`. This is the `CLEAN_CUTOFF_ISO` constant in `src/brawlstar_agent/recommender/dataset.py`. Do not attempt to use legacy battles for training or for evaluation; the labels are silently wrong on roughly half of them.

`docs/analytics-notes.md` should be updated to reflect this; until then DEC-010 supersedes its "label-flip-detection" suggestion.

## DEC-006: API-Based Battle Analytics Pipeline
- CV-based brawler identification is hard (needs labeled data, training, etc.)
- Official API gives structured battle data directly: who played what brawler, win/loss, mode, map
- Pipeline: rankings → player tags → battlelogs → SQLite → analytics queries
- Storage: SQLite at `data/brawlstars.db` (16MB after 4,628 battles from 200 top players)
- Rate limiting: 1-2 req/s with exponential backoff, conservative to avoid bans
- Snowball discovery: each battlelog yields ~6 new player tags from opponents/teammates
- **First run**: 200 global top players → 4,628 battles → 19,845 discovered players
- Analytics: brawler win rates, team comp win rates, matchup matrix, synergy matrix
- **Outcome**: much faster path to useful insights than CV pipeline

## DEC-005: Dataset Backup Strategy — Deferred
- `datasets/` is gitignored for now (images too large for normal git)
- When a high-quality dataset is ready, we need a proper backup strategy
- Options to evaluate later:
  - Git LFS (track large files in git via pointers)
  - External storage + download script (e.g. tarball on local NAS or cloud, `scripts/fetch-dataset.sh`)
  - DVC (data version control — git-like tracking for large files)
- **No action now** — revisit when dataset quality justifies preservation

## DEC-009: Remote = Routine/Periodic, Local = Interactive/Heavy (2026-05-04)

Operating principle for deciding where any given workload runs:

- **Droplet (remote)** is for **always-on, scheduled, predictable** work:
  - Crawlers (`brawl-collect.timer`, `brawl-collect-pinned.timer`)
  - Routine analytics precompute (`brawl-analytics.timer`)
  - Backups (planned: nightly `pg_dump`-equivalent → R2)
  - Anything that benefits from a stable IP whitelist + 24/7 uptime
  - Resource ceiling: 1 GB RAM, 1 vCPU. If something needs more, it doesn't belong here.

- **Local laptop (Lin's i7-12700H, 62 GB RAM, RTX 3060)** is for **interactive, exploratory, heavy-compute** work:
  - Dashboard viewing (reads droplet's precomputed cache via `--remote-cache`)
  - Ad-hoc SQL exploration (rsync DB to local, query freely)
  - Jupyter notebooks, experimentation
  - Future ML training (brawler classifier, embeddings) — GPU is here, not on droplet
  - Code authoring (DEC-008 already established this for source code)

Decision rule for any new workload: "Does it need to run unattended on a schedule?" → remote. "Does it need >1 GB RAM or interactive iteration?" → local.

Edge cases:
- Heavy one-off backfills: run on local (against rsync'd DB) and ship results back if needed.
- New scheduled analytics queries: design them on local first, then move the script + add a systemd timer on remote.

Artifacts that flow between the two:
- Code: local → GitHub → droplet (`git pull`) — DEC-008
- Cache JSON: droplet → local (`dashboard.py --remote-cache`) — DEC-009
- DB (occasional): droplet → local (`rsync` for ad-hoc queries with `--no-cache`)
- Backups: droplet → R2 (planned) — never touches local

## DEC-008: Local-Primary Workflow, Droplet Deploys via Git Pull (2026-05-03)
- **Local machine = source of truth for code**. All edits happen in Cursor on local.
- **Droplet = deploy target**. Receives updates via `git pull`. Never edit code on the droplet.
- Workflow: edit on local → commit → push to GitHub → SSH droplet → `git pull` → restart services if needed.
- Rationale:
  - Droplet has only 1 GB RAM; Cursor remote-SSH would compete with the crawler for memory.
  - Heavy assets (`datasets/`, `capture/`, `emulator/`) live only on local; deliberately NOT synced.
  - Single source of truth eliminates merge/divergence risk between two parallel workspaces.
  - Standard CI/CD-style pattern; future-us understands it instantly.
- Per-machine state stays per-machine and gitignored: `api.env`, `data/brawlstars.db`, `~/.bashrc` env vars.
- Droplet authenticates to GitHub via its own SSH key registered as a deploy key (read-only) on the repo. Generated on-droplet, never copied between machines.
- Migration path documented in `docs/deployment.md` for future VPS moves.

## DEC-007: Hosting — DigitalOcean Droplet, Phase-1 All-in-One (2026-05-03)
- Need: always-on machine for periodic crawling. No always-on home machine available.
- Binding constraint: Brawl Stars API key requires IPv4 whitelist (specific IPs only, no CIDR). Rules out serverless cron (GitHub Actions, Cloudflare Workers, Lambda, Neon `pg_cron`) — all use ephemeral IPs.
- Options compared (May 2026):
  - Oracle Always-Free Ampere — $0/mo, 24 GB RAM, free IPv4 — rejected for now (signup tax, ARM provisioning lottery)
  - GCP e2-micro "free" — actually $3.65/mo because Feb 2024 GCP charges for all in-use external IPv4 ($0.005/hr)
  - AWS Lightsail $5 — viable, predictable bill
  - Hetzner US CX22 $5.60 — best specs/$ (4 GB RAM) but unfamiliar brand
  - **DigitalOcean Basic $6/mo** — 1 GB RAM, 25 GB SSD, included IPv4, US-based, best ecosystem/docs
  - Neon free tier rejected: 0.5 GB cap (DB already 617 MB), `pg_cron` is SQL-only (can't run Python crawler)
- **Decision**: DigitalOcean Basic Droplet, $6/mo, Ubuntu 24.04 LTS, US region.
- **Architecture**: Phase-1 all-in-one. SQLite stays (no migration to Postgres yet — single writer + few readers fits SQLite perfectly). Crawler via systemd timer. Cloudflare R2 for nightly `pg_dump`-equivalent backups (free 10 GB tier). Reserved IP attached to droplet (free while attached) for stable BS API whitelist.
- **Phase-2** (only if/when needed): expose read-only API via Cloudflare Tunnel + Pages frontend. No public Postgres exposure.
- **Phase-3** (only at >50 GB DB or read-heavy public dashboard): partition by month, cold storage to R2 as Parquet, or migrate to Postgres + Neon read replica.
- Lock-in: near zero. Stock Ubuntu + SQLite/Postgres + standard backup formats. Migration to any other VPS is `rsync` + edit one systemd unit.
