# Handoff — Scaling-law nailed; fresh data ready for v4 boundary cut (2026-05-13, end of Session 12)

> Replaces the 2026-05-13 (end of Session 11) handoff. Of its 5 candidates,
> two are done and live in `memory-bank/progress.md` (Session 12) +
> `memory-bank/decisions.md` (DEC-019, DEC-020): the scaling-law analysis and
> the "more droplet data into a separate DB" rsync. The remaining three
> (phase 4b per-token, pick-prediction multi-task head, Star Power /
> Hyper Charge / Gears ingestion) carry forward unchanged. New candidates
> emerged this chat: a batch-size sweep on small / big (and optionally XL),
> the v4 boundary slide + retrain on the new DB, and per-player ELO / a
> sequence model over history.

## Project at a glance

Brawl Stars AI Research Sandbox at `/media/lin/disk2/brawlstar-agent/`.
Phase 6 (brawler-pick recommender) is on **v3.1 — kit-sink transformer
(phase 1 + 2 + 4) plus an LGBM ⊕ transformer ensemble**. The state of the
art on the competitive draft slice (soloRanked Mythic+, the in-game Ranked
queue ≥ Mythic tier, strict 1-2-2-1 ban/pick draft) is now **0.6249 AUC**
via the ensemble (no new training), up from **0.6180** (XL+P1+P2+P4 single
model, Run K). All numbers are on the DEC-011 stable test set
(`battle_time_iso >= '2026-05-05T00:00:00Z'`, 844 k battles / 1.69 M rows) —
that boundary should slide forward in the next chat now that fresh data has
landed (see below).

## Read these first

1. `memory-bank/memory-bank.md` — master index, hard constraints, phase
   status (entries 12 and 13 cover the scaling-law and next-step state)
2. `memory-bank/activeContext.md` — current state + fresh-DB notice at top
3. `memory-bank/progress.md` — **Session 12 has three entries (morning
   scaling-law, evening anchor runs + ensemble, late-evening DB landing)**
4. `memory-bank/decisions.md` — **DEC-019** (scaling-law data-bound
   diagnosis) and **DEC-020** (anchor runs + ensemble SOTA + M3 outlier) are
   load-bearing. DEC-010 (legacy bug unrecoverable) and DEC-011 (stable test
   mandatory) still apply.
5. `memory-bank/techContext.md` — **"Brawl Stars game-domain semantics"**
   section + **"Ops convention — sudo on the droplet"** note (don't compose
   `ssh brawl 'sudo …'`; give bare `sudo …` lines the user pastes after
   `ssh brawl`).
6. `docs/canvases/scaling-law-mythicplus.canvas.tsx` — the rich scaling-law
   summary canvas. Workspace-managed copy lives at
   `~/.cursor/projects/media-lin-disk2-brawlstar-agent/canvases/`.

## What just happened (this chat)

- **DEC-019 — scaling-law analysis** answered HANDOFF #1: fit
  `L(N, D) = E + A·N^(−α) + B·D^(−β)` on Mythic+ on all available transformer
  runs. Original 3-point kit-sink fit said α = 0.364, asymptote AUC = 0.629,
  XL closed 91.5 % of the gap. Joint Chinchilla on 11 transformer runs said
  capacity-share of reducible loss is 14 %, data-share 86 % — heavily
  data-bound. Built `scripts/analyze-scaling-laws.py`, CSV inventory + JSON
  fits + 3 matplotlib plots.
- **DEC-020 — anchor runs + ensemble** (Session 12 evening):
  - 3 new transformer runs at kit-sink (P1+P2+P4) on the same DEC-011
    boundary: **M1** (1.10 M params, Mythic+ 0.6117), **M2** (1.57 M, 0.6131),
    **M3** (5.11 M, 0.5892 — outlier because batch had to drop to 2048 to
    fit the RTX 3060 Mobile's 5.77 GiB VRAM).
  - M1 and M2 land essentially on the fit's predicted curve; M3 collapses
    3 pp below. Refit excluding M3 gives **α = 0.213, asymptote AUC = 0.640**;
    XL has closed only **84.3 %** of the gap (was 91.5 %); 2.19 pp Mythic+
    headroom remaining. Joint Chinchilla on 10 runs: data-share of reducible
    loss is now **91.8 %**, capacity-share **8.2 %** — even more data-bound.
  - **LGBM ⊕ XL ensemble** (`scripts/ensemble-stable-test.py`): blending at
    `α_lgbm = 0.45` yields **Mythic+ AUC 0.6249** (+0.69 pp vs XL alone),
    all-test 0.7787 (+0.42 pp), Brier 0.1842 (best ever). Ensemble gain
    grows with slice difficulty: Unranked +0.39 pp → Mythic+ +0.69 pp →
    Legendary+ +0.85 pp. **NEW Mythic+ SOTA**, zero new training.
- **Fresh data landed** (user-driven, late evening): rsynced the droplet DB
  into `data/brawlstars_extra_2026-05-14.db` (17 GB, integrity_check ok,
  3.36 M clean post-fix battles ≈ **1.89× the old DB**, 329 k new beyond
  `2026-05-06T04:30:14Z`, latest battle `2026-05-14T02:27:05Z`). Original
  `data/brawlstars.db` is untouched. This is the empirical β-falsification
  setup for DEC-019/020 — the joint fit predicts ~+0.56 pp Mythic+ from
  doubling D at fixed XL N.

## Current state

- **Production candidates** (no change from DEC-020):
  - **Best Mythic+ (research)** — LGBM phase 1+4 ⊕ XL phase 1+2+4 ensemble
    at α_lgbm = 0.45. Mythic+ **0.6249**. Recipe is in
    `scripts/ensemble-stable-test.py` + cached probas at
    `reports/ensemble_cache/p_*.npy`.
  - **Best single transformer** — `models/recommender_v3_phase1p2p4_xl.pt`
    (Run K, 3.29 M params). Mythic+ 0.6180.
  - **Best AUC/cost (GPU)** — `models/recommender_v3_phase1p2p4_big.pt`
    (Run J, 576 k params). Mythic+ 0.6084.
  - **Best CPU-only** — `models/recommender_v2_phase1p4.lgb.txt` (Run F).
    Mythic+ 0.6060.
- **New anchor models on disk** (kept for the scaling-law repro; not
  promoted):
  - `models/recommender_v3_phase1p2p4_m1.pt` (1.10 M params, Mythic+ 0.6117)
  - `models/recommender_v3_phase1p2p4_m2.pt` (1.57 M params, Mythic+ 0.6131)
  - `models/recommender_v3_phase1p2p4_m3.pt` (5.11 M params, Mythic+ 0.5892
    — batch=2048 outlier)
- **Fresh data**: `data/brawlstars_extra_2026-05-14.db` is parked, untouched
  by any pipeline. Old `data/brawlstars.db` still feeds every existing repro.
- **Reports / logs**: `reports/{scaling_laws_inventory.csv, scaling_laws.json,
  ensemble_kitsink.json, ensemble_cache/p_*.npy, scaling_law_N_*.png}` and
  the 3 new training reports under `reports/recommender_v3_phase1p2p4_*.json`.
  Training logs at `logs/{train_v3_phase1p2p4_m{1,2,3}.log,
  anchor_runs_orchestrator.log, m3_retry_orchestrator.log,
  ensemble_kitsink.log, scaling_laws_final.log}`.
- **No background jobs running.** Anchor orchestrator + ensemble both
  terminated cleanly.
- **Anything open / incomplete**:
  - **Uncommitted work**: now seven sessions of changes (Sessions 10, 11,
    12 + earlier) on `main`. Last on-disk commit predates Session 10.
    Session-level commit when convenient.
  - **Droplet timer status** — unknown; not blocking model work. If you
    care, run the inspection commands from the chat (read-only
    `ssh brawl 'systemctl list-timers --all | grep brawl'`; sudo commands
    only after `ssh brawl` interactively).
  - **`docs/recommender-v3.md` is still stale** (pre-DEC-016 / 017 / 018 /
    019 / 020). Memory-bank has the canonical record; the v3 doc is
    historical.

## Next-step candidates

All forward-looking. Items 1 and 2 are the highest-signal: item 1 directly
addresses the M3 recipe-confound finding from DEC-020; item 2 cashes in the
data-bound prediction from DEC-019/020 using the freshly-landed DB.

1. **Batch-size sweep on small / big (and optionally XL)** — disambiguate
   whether M3's underperformance was a recipe artifact or a true capacity
   ceiling at this D. Cheap on small (~7 min) and big (~15 min); somewhat
   expensive at XL (~50 min). Try batch ∈ {512, 1024, 2048, 4096, 8192}
   where memory permits. **Watch for OOM at XL+P1P2P4 — anything past
   batch=4096 will likely fail; the activation memory ceiling is the same
   constraint that bit M3.** Track val_auc and stable-test Mythic+ per
   batch. Cost: small. Risk: marginal AUC moves at small/big are sample
   noise; the real value is the methodology — find the optimal batch for
   each N, then re-anchor the scaling-law curve. ~half day.

2. **v4 boundary slide + retrain on the combined DB** — wire
   `--db data/brawlstars_extra_2026-05-14.db` through the train scripts
   (the `db_path` argument already exists in `load_clean_battles`), pick a
   new `STABLE_TEST_AFTER_DEFAULT` (probably `'2026-06-05T00:00:00Z'` or
   based on actual density — eyeball with a per-day battle count query),
   and re-train (a) `recommender_v2_phase1p4` LGBM and (b)
   `recommender_v3_phase1p2p4_xl`. Rename downstream artifacts with a `_v4`
   suffix or similar; DO NOT overwrite the existing models. AUCs across
   boundary versions are NOT comparable — treat as a fresh baseline.
   Falsifies the joint Chinchilla β ≈ 0.082: expected lift on Mythic+ is
   ~+0.56 pp from doubling D at fixed XL N. Cost: medium (a few hours of
   wall-time + a careful artifact-renaming pass). Risk: meta drift may
   eclipse the scaling-law prediction, which is itself a finding.

3. **Phase 4b — per-token history features** — extend
   `scalar_proj_brawler` from 2 inputs (trophy_log, power/11) to 3+ inputs
   so per-player history scalars (`n_games_log`, `this-brawler-count_log`,
   `is_main`) sit on the per-brawler token instead of the team aggregate.
   The model can attend to the specific slot with high comfort / main
   status. Should be more expressive at XL specifically. Composes with #2
   (do it on the v4 boundary). Cost: medium (~half day code + 1 XL retrain).

4. **Per-player ELO / Bradley-Terry skill features (phase 4c)** — compute
   per-player ELO incrementally from the pre-cutoff window, surface as
   another scalar feature alongside phase 4's frequency aggregates. Cheap
   to add (fits the existing `compute_player_history` pattern); complements
   phase 4 because frequency ≠ skill. Sparsity caveat: same ~9 % April
   history overlap, but Mythic+ players are over-represented in it. Cost:
   small. Risk: largely subsumed by phase 4 + tier embedding signal.

5. **Sequence model over per-player battle history** — replace the
   phase-4 aggregates with a real sequence model that consumes the player's
   last K battles directly. Largest architectural bet; should specifically
   beat the ensemble SOTA on Mythic+ where competitive-player history is
   densest. Cost: large (multi-day; player-tag sequence builder + new
   architecture + retrain).

6. **Pick-prediction multi-task head + pairwise / listwise ranking loss** —
   carried forward unchanged. Targets the top-K hit@1 ≈ 0.137 ceiling that
   all binary-BCE transformers share. Cost: medium (~1 day).

7. **Star Power / Hyper Charge / Gears ingestion** — carried forward
   unchanged. Real stat depth the model is blind to. Highest Mythic+
   headroom in principle (top-tier players run tuned builds) but largest
   engineering investment (new collector + DB migration). Cost: large
   (multi-day).

Ensembling (item from chat discussion) is **explicitly deprioritized** as a
focus: it's a near-free quality knob, but we'll just re-blend whatever the
next round of better-trained models produces. No standalone work needed.

Bigger transformer beyond XL is **off the menu** per DEC-020 — M3 confirmed
the practical capacity ceiling at our current GPU memory budget. Only
revisit after gradient checkpointing or mixed precision lands in the
training stack.

## How to start (for the new agent)

- Read the files in "Read these first" in order.
- Then read this entire HANDOFF.md.
- If a follow-up prompt was provided alongside this handoff, follow it.
  Otherwise, candidates 1 and 2 are the highest-signal next moves: candidate
  1 (batch-size sweep) is the cheap methodology fix, candidate 2 (v4 boundary
  slide on the new DB) is the cash-in on the data-bound prediction.
- Honor the always-applied memory-bank protocol in
  `.cursor/rules/memory-bank.mdc` (also inlined in `CLAUDE.md`): read
  `memory-bank/{memory-bank, progress, activeContext}.md` first, then update
  them as you go. Append to `progress.md` and `decisions.md`; never rewrite
  history. Add new decisions as DEC-021, DEC-022, etc.
- **Hard constraints** (DEC-009): training and inference run on the local
  workspace machine only. The droplet has 1 GB RAM / 1 vCPU and is for
  crawlers only — don't try to run training there.
- **uv quirk**: always set
  `UV_CACHE_DIR=/media/lin/disk2/brawlstar-agent/.uv-cache-local` before
  `uv` commands (the default `/media/lin/disk2/.uv-cache` is owned by
  root locally).
- **GPU access from inside Cursor**: CUDA works under
  `required_permissions=["all"]` in the Shell tool; from a normal user
  terminal it just works.
- **Droplet sudo convention** (DEC-020-era addition in `techContext.md`):
  do NOT compose `ssh brawl 'sudo …'`. Read-only inspection via
  `ssh brawl 'cmd'` is fine; for anything that needs sudo, give the user
  the bare `sudo …` line to paste after they `ssh brawl` interactively.
- **Phase 4 leakage gotcha** (still load-bearing from DEC-018): when
  calling `TeamFeaturizer.fit(df)` / `model.fit(df)` with
  `include_history_features=True` / `use_history_features=True`, **always**
  pass a `history_df` from a disjoint time window. Omitting it inflates
  val_auc to ~0.99 from self-leak. See DEC-018 for the full story.
- When you reach a natural pause point in this chat's work, run `/handoff`
  to package the work for the next session and replace this HANDOFF.md —
  per the project's handoff convention, **replace, don't append**.
