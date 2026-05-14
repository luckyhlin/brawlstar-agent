# Brawl Stars AI Research — Memory Bank

> Master index. Read this file first in every new session.

## Project Identity

- **Name**: Brawl Stars AI Research Sandbox
- **Location**: `/media/lin/disk2/brawlstar-agent/`
- **Purpose**: Offline CV/ML research on Brawl Stars gameplay — perception, strategy analysis, dataset generation.
- **Data source**: YouTube gameplay recordings (emulator path failed due to anti-cheat).
- **Scope**: Offline analysis only. NO live-match botting.

## Memory Bank Files

| File | Purpose |
|------|---------|
| `memory-bank.md` | This file — project overview and index |
| `progress.md` | What's done, what's next, session log |
| `activeContext.md` | Current focus and immediate actions |
| `techContext.md` | Machine specs and installed software |
| `decisions.md` | Key decisions with rationale |
| `architecture.md` | Directory layout, pipeline design, module inventory |

## Hard Constraints

1. Everything local on Linux.
2. No personal accounts or personal phone.
3. All data stays local.
4. No live multiplayer automation — research/offline only.

## Project Phases

1. ~~Environment Setup~~ — emulator failed, pivoted to YouTube capture
2. ~~Data Pipeline~~ — DONE: download → extract → review → crop → 308 gameplay frames
3. ~~Perception Baseline~~ — DONE (partial): OCR works on timer, brawler detection is weak
4. ~~API Battle Analytics Pipeline~~ — DONE (DEC-006): SQLite + collector + analytics queries; ~200k battles, 553k tags
5. ~~Production Deploy~~ — DONE (DEC-007/008/009): always-on DigitalOcean droplet with 3 systemd timers (bulk crawl + pinned tags + analytics precompute), local-primary git workflow, dashboard reads precomputed cache
6. ~~Brawler-pick recommendation model v1~~ — DONE (Session 8, DEC-010): LightGBM team-completion model. Random-split AUC 0.730, temporal AUC 0.704 (vs ModeMap baseline 0.697 / 0.666). Inference covers all three pick-scenarios. Read `docs/recommender-v1.md`. Re-runnable monthly via `scripts/train-recommender.py` for transferability tracking.
7. ~~Brawler-pick model v2~~ — DONE (Session 8, DEC-011): cold-start + 50× more clean data + stable temporal test set. Three fair runs (A_fair / C_fair / B_fair) on the same held-out 844k-battle test set: LightGBM stable-test AUC 0.7181 / 0.7265 / 0.7235. 30-day window is the technical winner but +0.84pp doesn't justify rolling-window pipeline complexity; production candidate stays at the simpler 3-day cutoff (A_fair). Read `docs/recommender-v2.md`. Top-K hit@1=0.130, win uplift +18.1pp on the stable test set.
8. ~~Brawler-pick model v3~~ — DONE (Session 9, 2026-05-08): attention transformer over [CLS, CTX, A1, A2, A3, B1, B2, B3] tokens with per-brawler trophy + power features. **6-step ablation** (same A_fair training data, same DEC-011 stable test set): (Run 0) v2 LightGBM baseline AUC **0.7181** → (Run 1) small CPU transformer **0.7378** (+1.97pp from architecture) → (Run 2) same on GPU **0.7392** (2.7× faster, AUC noise) → (Run 3) GPU fast data path **0.7366** (additional 1.8× faster, 4.9× total over CPU; AUC noise) → (Run 4) **big arch** (d=128, L=4, ff=256, 570k params, 8 epochs on GPU) **AUC 0.7635** (+4.54pp vs LightGBM, +2.69pp vs small) → (Run 5 = best AUC) **XL arch** (d=256, L=6, ff=512, 3.28M params, 12 epochs on GPU) **AUC 0.7674, Brier 0.1928** (+4.93pp AUC vs LightGBM, +0.39pp vs big — diminishing returns; Brier scaling well). Wins 9/9 modes vs LightGBM by 2.6-7.6pp. Knockout +7.58pp is biggest mode-level gain. Top-K hit@1 essentially tied across small/big/XL (~0.137) — structural ceiling; XL improvements go into calibration, not ranking. **Production candidates**: `models/recommender_v3_big.pt` (sweet spot: best AUC/inference-cost ratio) and `models/recommender_v3_xl.pt` (best calibration when you need to threshold on P(win)). Brawler vocab is 102 (the 2 newest, BOLT + STARR NOVA, haven't appeared in ranked play yet — same in both train and test, so eval not stale). Read `docs/recommender-v3.md` for the full ablation table + per-mode breakdown + training history.
9. ~~**Brawler-pick model v3.1 phase 1**~~ — DONE (Session 9 night, DEC-015). 23 dense per-team aggregates of trophy/power (min/max/std + counts + diffs). **LGBM stable-test AUC 0.7181 → 0.7609 (+4.28pp)**, closing 62% of the gap to the XL transformer at ~3% of training cost. **v3 small + phase 1: 0.7378 → 0.7540 (+1.62pp)**. **v3 big + phase 1: 0.7635 → 0.7603 (−0.32pp; substitution effect — encoder already extracted aggregate signal via attention)**. Decomposed the original Run 0→1 +1.97pp gain: feature engineering and architecture are largely substitutes at this data scale. New cheapest-quality production candidate: `models/recommender_v2_phase1.lgb.txt` (CPU-only, ~5min training). Read DEC-015 in `memory-bank/decisions.md`.

10. ~~**Brawler-pick model v3.1 phase 2 + soloRanked-only**~~ — DONE (Session 10 afternoon, DEC-017). Phase 2 + soloRanked-only roughly noise on Mythic+. Confirmed implication: Mythic+ improvement needs new information, not new aggregates of existing data.

11. ~~**Brawler-pick model v3.1 phase 4 — player history**~~ — DONE (Session 11, 2026-05-13, DEC-018). 12 frequency-only per-player history aggregates (n_games, brawler-pair counts, main-brawler alignment), lookup built from a SEPARATE pre-cutoff April window (`--history-after`) so disjoint from training rows. **NEW Mythic+ SOTA 0.6180** (v3 XL + phase 1+2+4 kitchen sink, Run K), up from prior 0.6109 — **+0.71 pp on Mythic+, +0.72 pp on all-test, best Brier 0.1902**. First feature addition that meaningfully moves Mythic+. LGBM doesn't benefit on Mythic+; transformer gain compounds with arch (small +0.80 / big +0.96 / XL +0.71 pp). Confirms DEC-017's implication: new information beats new aggregates on the competitive slice. New production candidates: `recommender_v3_phase1p2p4_{xl,big}.pt` and `recommender_v2_phase1p4.lgb.txt`.

12. ~~**Brawler-pick model v3.1 scaling-law analysis**~~ — DONE (Session 12, 2026-05-13, DEC-019 + DEC-020). DEC-019: original fit on 11 transformer runs said α ≈ 0.30, β ≈ 0.09 on Mythic+ (1 − AUC); 86 % of reducible loss is data-side; asymptote AUC ~0.629 at D = 1.87 M. DEC-020: anchor runs M1 (1.10 M) and M2 (1.57 M) confirmed the curve (within 0.05 – 0.13 pp), refining the kit-sink fit (5 obs / dof = 2) to **α = 0.213, asymptote 0.640, XL closed 84.3 %**. M3 (5.11 M, batch=2048 due to GPU OOM at batch=4096) collapsed to Mythic+ 0.5892, 3 pp below the curve — practical capacity ceiling under our memory budget. **NEW Mythic+ SOTA: 0.6249** via LGBM ⊕ XL ensemble (α_lgbm = 0.45, zero new training). Implication unchanged: bigger transformer beyond XL gives diminishing returns; data and new-information channels are the binding levers. Artifacts: `scripts/analyze-scaling-laws.py`, `scripts/ensemble-stable-test.py`, `reports/scaling_laws.json`, `reports/ensemble_kitsink.json`, 3 plots, canvas `docs/canvases/scaling-law-mythicplus.canvas.tsx`.

13. **Brawler-pick model v3.1 next phases** — reordered after DEC-020:
    (a) **Time-series-aware boundary slide** when more data lands — slide `STABLE_TEST_AFTER_DEFAULT` forward, treat as v4 cut, AUCs not comparable to current 0.6249. (b) **More droplet data → `data/brawlstars_extra.db`** — empirical β-falsification; expected ~+0.5 pp / doubling. (c) **Phase 4b — per-token history features** (move history scalars from team aggregate to per-brawler-token scalar projection — should be more expressive at XL; "new information" channel ≈ effective D-multiplier). (d) **Per-player ELO / Bradley-Terry skill features (phase 4c)** — cheap addition complementing phase-4 frequency aggregates. (e) **Sequence model over per-player battle history** — architectural lever most likely to beat the ensemble SOTA. (f) **Pick-prediction multi-task head + pairwise / listwise ranking loss** — orthogonal to AUC; targets hit@K. (g) **Star Power / Hyper Charge / Gears ingestion** — largest information delta, most engineering work.
    De-prioritized by DEC-020: ~~bigger transformer beyond XL~~ (M3 confirmed practical ceiling at this D and recipe). Phase 3 (per-brawler / per-mode WR aggregates) still de-prioritized — aggregate-of-existing-data pattern.
14. Brawler Identification (CV) — deferred; API gives structured data, no need for visual ID right now
15. Strategy Analysis / Coach Overlay — stretch goal

## For agents starting a new session

If you're picking up the project, read in this order:
1. `memory-bank/memory-bank.md` (this file) — overview
2. `memory-bank/progress.md` — what's done, session log, current phase
3. `memory-bank/activeContext.md` — current focus + next steps
4. `memory-bank/decisions.md` — DEC-001..020. Load-bearing: DEC-010 (legacy bug unrecoverable), DEC-011 (stable temporal test set is mandatory for v2-and-beyond comparisons), DEC-016 (Mythic+ AUC is the primary metric for the competitive draft slice), DEC-018 (phase 4 = first feature addition that moved Mythic+), DEC-019 (scaling-law analysis: data-bound, not capacity-bound), DEC-020 (anchor runs validate fit; LGBM ⊕ XL ensemble is new Mythic+ SOTA 0.6249; M3 marks the practical capacity ceiling under current GPU memory).
5. If touching infrastructure: `memory-bank/techContext.md` + `docs/deployment.md`
6. **If doing analytics or model training**: `docs/analytics-notes.md` for data caveats, `docs/recommender-v1.md` for the v1 model methodology and inference walkthrough, `docs/recommender-v2.md` for the v2 update (DEC-011 stable test set, fair-run comparison across cutoffs, top-K on the stable test set), then `docs/recommender-v3.md` for the v3 attention transformer (current production candidate, beats v2 LightGBM by +1.97pp AUC on the same data + same test set; CPU training; per-brawler trophy/power features). **Notes**: the legacy data label-flip suggestion in `analytics-notes.md` is superseded by DEC-010. Random-split AUC numbers in v1's results table are pre-DEC-011; the honest baseline is in v2's table; v3 is the current state of the art.
