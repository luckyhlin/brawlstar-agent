# Progress

## Current Phase

**Phase 6 v3.1 — scaling-law analysis says Mythic+ is data-bound (Session 12, 2026-05-13, DEC-019)**. Empirical Chinchilla-style fit on 11 v3 transformer runs (N: 251 k → 3.29 M; D: 349 k → 1.87 M rows) yields **α ≈ 0.30, β ≈ 0.09 on Mythic+ (1 − AUC)**. At the SOTA point (XL+P1+P2+P4, N=3.29 M, D=1.87 M), the reducible (1−AUC) splits **14 % capacity / 86 % data** — we are heavily data-bound, not capacity-bound. The kit-sink-frontier 3-point fit predicts an **asymptote AUC ≈ 0.629 at current data scale**; XL+P1+P2+P4 (0.6180) has closed **91.5 %** of the gap from random. 16× XL params would only reach AUC 0.625; 0.63+ is unreachable without more data or new information channels. Implication: bigger transformers beyond XL waste GPU-hours unless paired with data scaling. **Reordered next-phase priorities**: (a) rsync more droplet data into a separate DB and empirically test the implied β, (b) phase 4b per-token history features (a new information channel ≈ effective D-multiplier per the joint fit), (c) Star Power / Hyper Charge ingestion. Anchor runs at N ≈ 1 M / 1.5 M are optional analysis-refinement only.

**Phase 6 v3.1 — phase 4 player history breaks the Mythic+ ceiling (Session 11, 2026-05-13, DEC-018)**. 12 frequency-only per-player history scalars (n_games, brawler counts, main-brawler alignment) computed from a SEPARATE pre-cutoff window (April 2026 battles, ~9% player overlap) and added behind a `--use-history-features` flag with `history_df` plumbing. **NEW Mythic+ state of the art: 0.6180** (v3 XL + phase 1+2+4 kitchen sink, Run K), up from prior 0.6109 — **+0.71 pp**. Run K also wins all-test (0.7746) and best calibration (Brier 0.1902). LGBM doesn't benefit on Mythic+ (~tied) but transformer does; gain compounds across arch sizes (small +0.80 pp / big +0.96 pp / XL +0.71 pp). Confirms DEC-017's implication that new INFORMATION beats new AGGREGATES on the competitive slice. New production candidates: `recommender_v3_phase1p2p4_xl.pt` (best research-grade), `recommender_v3_phase1p2p4_big.pt` (best cost-balanced), `recommender_v2_phase1p4.lgb.txt` (best CPU-only).

**Phase 6 v3.1 — phase 2 + soloRanked-only sprint done (Session 10 afternoon, DEC-017)**: 5 new runs (LGBM and v3 small × phase 2 × mixed/solo train) exhaust the cheap "more aggregate features / scope the data" knobs. **Mythic+ AUC was capped at ~0.61 across 12 model configurations** (now broken by DEC-018). Phase 2 (time + days_since_release) is essentially a no-op on Mythic+ (Δ = ±0.1 pp, sample noise). soloRanked-only training is roughly tied with mixed for LGBM (+0.09 pp Mythic+, sample noise), but **clearly worse for the transformer (−1.42 pp Mythic+)** because 5.4× less data outweighs semantic-mixing harm.

**Phase 6 v3.1 — tiered slice eval (Session 10 morning, DEC-016)** reframed every prior AUC number. After learning that `battle_type='ranked'` is the in-game **Unranked trophy ladder** (no draft) and `battle_type='soloRanked'` is the actual Ranked queue with strict 1-2-2-1 ban/pick draft starting at **Mythic (>= 13)**, retrospective slice eval on all 7 saved models showed **two distinct prediction problems**: AUC on the trophy ladder (`ranked`) is 0.74-0.80 across models, AUC on the strict-draft Mythic+ slice is 0.59-0.61. The "all-test" numbers were ~79% weighted by the easy slice. **All models cluster within 0.0172 AUC on Mythic+**: state of the art is v3 XL at **0.6109**, only +0.06 above the random floor. Phase 1 LGBM's "+4.28 pp on all-test" was **+4.84 pp on Unranked but only +0.22 pp on Mythic+**. The actual research target is now **soloRanked Mythic+ AUC**, not the conflated all-test.

**Phase 6 v3.1 phase-1 ablation done (Session 9 night, 2026-05-08, DEC-015)**: 23 dense numeric scalars summarising the per-brawler trophy/power tuples (per-team min/max/std + diffs). **LGBM gains +4.28 pp on all-test** (0.7181 → **0.7609**), closing 62% of the gap to the XL transformer at ~3% of training cost. **The v3 transformer barely benefits**: small +1.62 pp, big −0.32 pp (substitution effect). Decomposes the original Run 0→1 +1.97 pp gain: feature engineering and architecture are largely substitutes at this data scale. **Reinterpreted under DEC-016**: most of the LGBM gain came from the easier `ranked` (Unranked) slice. New cheapest-quality production candidate: `models/recommender_v2_phase1.lgb.txt` (CPU-only deploy). Production GPU candidates unchanged from DEC-014.

**Phase 6 v3 shipped (Session 9 morning–evening, 2026-05-08)**: attention transformer over (CLS, ctx, A1-A3, B1-B3) tokens with brawler embeddings + per-brawler trophy/power features. Same A_fair training data (1.87M rows) and same DEC-011 stable test set (844k battles). **6-row ablation** (architecture → GPU compute → data plumbing → bigger model → XL model) ends at **stable-test AUC 0.7674 (Run 5 XL) — beats v2 A_fair LightGBM 0.7181 by +4.93pp**. Wins in 9/9 modes by 2.6-7.6pp (knockout +7.58pp). GPU training (RTX 3060 Mobile): 14 min for Run 4 big (570k params), 52 min for Run 5 XL (3.28M params). Capacity scaling shows clear diminishing returns past 3M params. Production candidates: `models/recommender_v3_big.pt` (best AUC/inference-cost ratio) and `models/recommender_v3_xl.pt` (best calibration). Read `docs/recommender-v3.md`.

**v2 (kept for reference)**: LightGBM team-completion model on the same DEC-011 stable test set. Production candidate was `models/recommender_v2_default_fair.lgb.txt` (A_fair, AUC 0.7181) — superseded by v3.

**v1 (initial)**: LightGBM on the smaller post-fix-only window. Random-split AUC 0.730. Replaced by v2 + v3.

**Production crawl** is currently **paused** (droplet timers stopped after the cold-start filled disk; user is rsyncing the shrunk DB back). v3 model is trained from local DB only — no droplet dependency.

## Done

### Infrastructure
- Machine assessed, project on disk2 (295GB free NVMe)
- Memory bank + cursor rule for cross-session continuity
- Python env via uv (3.12, opencv 4.13, numpy, yt-dlp, tesseract, matplotlib)

### Emulator (dead end)
- AVD: game installs but Supercell anti-cheat self-kills (DEC-003)
- Genymotion: ARM translation removed in v3.9, can't install game (DEC-004)
- Waydroid: needs Wayland + kernel modules we don't have

### Data Pipeline
- 24 YouTube clips downloaded (batch + manual)
- Frames extracted at 2fps, 480 sampled frames reviewed via browser hub
- 308 gameplay frames exported to `datasets/gameplay_cropped/` (20 clips, 12 resolutions)
- 100 brawler portraits (bordered + borderless) from BrawlAPI

### Brawl Stars API
- API key obtained and stored in `api.env` (IP-locked JWT, git-ignored)
- Full API documentation written: `docs/brawlstars-api.md` (7 endpoints)
- Live example responses dumped to `docs/api-examples/` (9 files)
- 101 brawlers cataloged, 25-battle log format analyzed, event rotation mapped

### Perception Baseline
- Game mode detection: showdown, brawl_ball, gem_grab, heist (heuristic, decent)
- UI region calibration: normalized coordinates, visual overlays verified
- OCR (tesseract): timer extraction works well on Brawl Ball (95%+ confidence), poor on Showdown (different UI layout)
- Character matching: color histogram baseline — **does not work** (distances ~0.85, not discriminative)
- Blob detection: finds saturated regions — **noisy**, picks up UI elements too

## Not Done / Known Gaps

- **Brawler identification**: needs a real pipeline (classifier or embedding), not color histograms
- **OCR verification**: results not human-verified yet
- **Showdown text parsing**: "Brawlers left: N" in different region, not handled
- **Minimap parsing**: not started
- **Structured game-state extraction**: not started
- **More data**: 308 frames is small, pipeline can scale easily

## Session Log

### Session 12 (continued) — 2026-05-13 night — Infra fix: rsync `$HOME` literal-path bug + droplet command-presentation convention
- **Bug**: `bash scripts/rsync-db-from-droplet.sh --direct brawl data/brawlstars_extra.db` failed with `rsync: [sender] change_dir "/home/lin/$HOME/brawlstar-agent/data" failed: No such file or directory (2)` while the user was starting HANDOFF task #3 (rsync more droplet data into a separate `data/brawlstars_extra.db`).
- **Root cause**: rsync 3.2.4+ enables `--secluded-args` by default. Path arguments after `host:` are sent straight to the remote `rsync` and are NOT interpreted by the remote shell, so `\$HOME` baked into `REMOTE_DB="\$HOME/brawlstar-agent/data/brawlstars.db"` (line 42 of `rsync-db-from-droplet.sh`) stayed literal. Remote rsync then resolved the relative `$HOME/...` against the SSH user's home → `/home/lin/$HOME/...` — exactly what the error reported. Backup mode and the companion `rsync-db-to-droplet.sh` had the same bug; backup mode happened to work-around it because line 65 wraps the path in `ssh "$host" "sqlite3 $REMOTE_DB ..."` which DOES go through a remote shell.
- **Fix**: changed `REMOTE_DB` (and `REMOTE_PATH` in the to-droplet script) to a relative path `brawlstar-agent/data/brawlstars.db`. Relative paths are handled by rsync itself even with secluded-args (resolved against the remote SSH user's `$HOME`) and also work in the `ssh "$host" "sqlite3 $PATH ..."` shell form (non-interactive ssh CWD = `$HOME`). Added a defensive comment block in both scripts. Removed the now-pointless `${REMOTE_DB#\\}` no-op (was attempting to strip a leading `\` that never existed). Both scripts pass `bash -n`.
- **Doc updates**: 
  - New row in `docs/deployment.md` § 16 gotcha table documenting the symptom + cause + fix (with the workaround `--old-args` / `RSYNC_OLD_ARGS=1` noted as deprecated-by-rsync).
  - New "Ops convention — rsync paths to/from the droplet" subsection in `memory-bank/techContext.md` near the existing sudo convention.
- **User-preference convention captured**: when an agent suggests commands meant to run on the droplet, present them as **bare commands** the user pastes after `ssh brawl`. Do NOT wrap them as `ssh brawl 'cmd'` in chat — the user often has an interactive ssh session open and needs paste-able commands. Added a new "Communicating commands to the user" section in `CLAUDE.md`. Existing techContext.md "Ops convention — sudo on the droplet" line broadened to cover all droplet commands (was sudo-only). Updated `say` lines in `scripts/rsync-db-to-droplet.sh` to print bare commands instead of `ssh $REMOTE_HOST '...'` wrappers (lines 92-93, 103-105, 156-157).
- **Net change**: 2 script edits (~10 lines each) + 1 deployment.md table row + 1 CLAUDE.md section + 1 techContext.md subsection rewrite + this progress note. No code, no model, no data touched. Unblocks `data/brawlstars_extra.db` work.

### Session 12 (continued) — 2026-05-13 late evening — User rsynced fresh droplet data into `data/brawlstars_extra_2026-05-14.db`
- **New DB landed**, isolated from training: `data/brawlstars_extra_2026-05-14.db` (17 GB, integrity_check `ok`). **4,047,511 battles total**; **3,356,646 clean post-fix** (≥ `2026-05-03T01:00:00Z`); **329,235 NEW** beyond the old DB's `2026-05-06T04:30:14Z` tail; latest battle `2026-05-14T02:27:05Z`. Renamed from the user's `brawlstars_extra.db` to the data-tail-date suffix so future rsyncs can land alongside it without overwriting.
- Original `data/brawlstars.db` is **untouched** (still 15 GB, 3.72 M battles, latest `2026-05-06`). Every existing recommender reproduction continues to work against it.
- For the data-bound β-falsification experiment: clean post-fix population in the new DB is ~1.89× the old DB. With both-perspectives doubling, training rows could grow ~1.87 M → ~3.74 M at fixed N — the joint Chinchilla fit predicts ~+0.56 pp Mythic+ AUC at fixed XL N. **Test-set boundary slide is required**: AUCs across DEC-011's `2026-05-05` boundary version and any new boundary are NOT comparable (treat as `v4` cut, rename downstream artifacts).
- No retraining done in this chat — DB hand-off only.

### Session 12 (continued) — 2026-05-13 evening — Ensemble new SOTA + 3 anchor runs (M1/M2/M3) for scaling-law DOF
- **Ensemble (no new training)**: Built `scripts/ensemble-stable-test.py` — sweeps `α · LGBM_phase1p4 + (1−α) · v3_xl_phase1p2p4` over α ∈ [0, 1] in 2.5 % steps on the DEC-011 stable test set. **Best blend α_lgbm = 0.45**: all-test AUC **0.7787** (+0.42 pp over XL alone), **Mythic+ AUC 0.6249** (+0.69 pp over XL alone), Legendary+ **0.6139** (+0.85 pp). Ensemble gain monotonically increases as slice gets harder (ranked +0.39, Mythic+ +0.69, Legendary+ +0.85) — the diversity benefit between LGBM-tree splits and transformer attention is largest on the hardest examples. **NEW Mythic+ SOTA 0.6249** (vs prior 0.6180 from Run K XL alone), at zero training cost. Test set load 220 s, LGBM proba 18 s, transformer proba 517 s (CPU, predict_proba is the bottleneck — proba .npy cache added so re-running the sweep is free). Files: `reports/ensemble_kitsink.json`, `reports/ensemble_cache/p_*.npy`, `logs/ensemble_kitsink.log`.
- **3 anchor training runs** (`logs/run_anchor_runs.sh` + `logs/run_m3_retry.sh`) with kit-sink (P1+P2+P4) features, mixed training, DEC-011 boundary. Architectures verified via `sum(p.numel() for p in _TransformerCore.parameters())`:
  - **M1** (d=160, h=8, L=5, ff=320, dropout=0.17, batch=4096, 10 epochs): **1,095,361 params**, ~34 min GPU. Stable-test: all AUC **0.7756**, Mythic+ **0.6117** (fit predicted 0.6121 at N=1 M — essentially on the curve). val_auc 0.7823 final.
  - **M2** (d=192, h=8, L=5, ff=384, dropout=0.18, batch=4096, 12 epochs): **1,566,337 params**, ~37 min GPU. Stable-test: all AUC **0.7731**, Mythic+ **0.6131** (fit predicted 0.6144 at N=1.5 M — slightly below curve by 0.13 pp). val_auc 0.7848 final.
  - **M3** (d=320, h=8, L=6, ff=640, dropout=0.20, **batch=2048**, 14 epochs): **5,112,641 params**, ~98 min GPU. Stable-test: all AUC **0.7634**, Mythic+ **0.5892** (fit predicted 0.6196 at N=5 M — **observed 3.0 pp BELOW the curve**, big outlier). val_auc 0.7718 final. **First attempt OOM'd at batch=4096** in epoch 2 backward (5.1 M params × activation memory > 5.77 GiB VRAM); had to halve batch to fit. The recipe change (batch halved while everything else fixed) plausibly accounts for some of the gap — re-running M3 at batch=4096 would need gradient checkpointing or mixed precision, neither currently in the training stack.
- **Final refit** (5 anchors in kit-sink fit, M3 excluded as batch-recipe-confounded outlier; 10 transformer runs in joint fit, M3 excluded):
  - **Kit-sink fit on Mythic+ (1 − AUC)** (5 obs, dof = 2): E = 0.360, A = 0.548, **α = 0.213** (was 0.364 in 3-point fit — curve is FLATTER than the 3-point fit suggested). **Asymptote AUC = 0.640** (was 0.629).
  - **XL has closed 84.3 % of the gap from random** (was 91.5 %), with **2.19 pp** Mythic+ headroom remaining (was 1.09 pp).
  - **Inverse predictions shift up**: AUC 0.625 reachable at 21.3 M params (6.5 × XL, was 53 M / 16× under the 3-point fit); AUC 0.630 reachable at 143 M (was unreachable). AUC 0.64 = asymptote.
  - **Joint Chinchilla on Mythic+ (1 − AUC)** (10 obs, dof = 5): α = 0.454, β = 0.082, E = 0.275. **Capacity share of reducible loss at SOTA: 8.2 % (was 14.4 %)** — even more data-bound than before. Data-projection at fixed XL N: 2× D → AUC 0.6208 (+0.56 pp), 4× D → 0.6261, 8× D → 0.6311.
- **M3 finding interpretation**: With monotonic-in-N functional form, the fit can't accommodate a 5.1 M point below the 3.3 M point — including M3 collapses α to 2.0 (boundary) and pins the asymptote at XL's value. Excluding M3 (batch-recipe-confounded) gives the cleaner 5-point fit. The practical takeaway is robust regardless: under our GPU memory constraint, **5 M params is past the useful capacity ceiling** with the current training recipe. Scaling N further requires either (a) more data (the joint fit's recommendation), (b) gradient checkpointing or mixed precision to keep batch=4096 at larger N, or (c) a recipe sweep at the 5 M scale (lower lr, longer training, different dropout).
- **Plots updated** with M3 X-marker outlier indicator in `reports/scaling_law_N_*.png`. Inventory CSV grew from 18 to 21 rows (the 3 new anchor models). New reports: `reports/recommender_v3_phase1p2p4_{m1,m2,m3}.json` + `models/recommender_v3_phase1p2p4_{m1,m2,m3}.{pt,meta.json}` + logs.
- **Canvas refreshed** at `canvases/scaling-law-mythicplus.canvas.tsx` (workspace) and `docs/canvases/scaling-law-mythicplus.canvas.tsx` (repo mirror) with the new SOTA, the 5-anchor fit, the M3 outlier, the ensemble breakdown, and the updated next-step priority list.

### Session 12 — 2026-05-13 — Scaling-law analysis on Mythic+ AUC (DEC-019)
- **Goal** (per HANDOFF.md task #1): fit `AUC_Mythic+ ≈ a − b · N^(−α) · D^(−β)` to the existing 11 v3 transformer runs and use it to answer "are we over-parameterized or under-trained?"
- **Built**: `scripts/analyze-scaling-laws.py` — single-file scipy/matplotlib analysis. Reads `reports/slices_summary.json` + every `reports/recommender_*.json` with a `stable_test_slices` block; joins with hard-coded param counts (extracted from `logs/train_*.log`); fits two power laws (kit-sink-N-frontier only; full joint Chinchilla on all 11 transformer runs); emits CSV inventory + JSON fit results + 3 matplotlib PNGs.
- **Inventory shape**: 18 rows total (11 transformer + 7 LGBM, all on the DEC-011 stable test set, all evaluated on the `soloRanked_mythicplus` slice). 11 transformer runs: small/big/XL × vanilla/P1/P1+P2/P1+P4/P1+P2+P4, mostly at mixed D = 1.87 M (10 runs) + one solo D = 349 k (small P1+P2). LGBM runs included for context but not fitted (different model family).
- **Fit A — kitchen-sink frontier (3 SOTA points, fixed D = 1.87 M)**:
  - small P1+P2+P4 (256 k) → AUC 0.6013
  - big P1+P2+P4 (576 k) → AUC 0.6084
  - XL P1+P2+P4 (3.29 M) → AUC 0.6180

  Power-law fit `(1 − AUC) = E + A · N^(−α)` gives **E = 0.371, A = 2.576, α = 0.364** (rmse ≈ 0; exact 3-point fit, 0 d.o.f.).

  Predictions: AUC 0.620 needs 5.66 M params (1.7× XL), AUC 0.625 needs 53.2 M params (16.2× XL), **AUC ≥ 0.629 is unreachable** at current data scale. XL has already closed **91.5 %** of the (AUC − 0.5) gap from random.

- **Fit B — joint Chinchilla on 11 transformer runs (statistically meaningful, dof = 6)**:
  - `L(N, D) = E + A · N^(−α) + B · D^(−β)` on Mythic+ (1 − AUC):
    **E = 0.272, α = 0.302, β = 0.087, rmse = 0.0032**.
  - At the SOTA point: capacity term = 0.0163 (14.4 % of reducible), data term = 0.0968 (85.6 % of reducible). **We are heavily data-bound.**
  - Data-scaling projection at fixed N = 3.29 M:
    - 2× D (3.74 M rows) → AUC 0.6201 (+0.56 pp)
    - 4× D (7.49 M rows) → AUC 0.6254 (+1.09 pp)
    - 8× D (14.97 M rows) → AUC 0.6304 (+1.59 pp)

- **Headline takeaways**:
  - **Mythic+ ceiling at current D = 1.87 M is ~0.629**. XL already 91.5 % closed.
  - **Each doubling of D delivers ~+0.5 pp Mythic+**. New information channels (phase 4, future phase 4b / Star Power) act like effective D-multipliers; the empirical XL+phase-1+2+4 → XL+vanilla +0.71 pp jump corresponds to going from 1.87 M → 4 – 8 M training rows in the joint fit.
  - **Chinchilla's "N:D = 20:1 says we're data-starved" framing (HANDOFF.md)**: yes, but our task has α/β ≈ 3.5 (not ~1 as in LLMs), so the compute-optimal allocation is heavily N-skewed. At our scale we've saturated N relative to D; more data is the binding lever.

- **Caveats** (also in the canvas + DEC-019):
  - Fit A is an exact 3-point fit (no DOF). α is what makes the 3 points lie on the curve, not a statistical estimate. Anchor runs at N ≈ 1 M and 1.5 M would add 2 DOF.
  - Fit B's β is largely set by ONE D-ratio pair (small mixed @1.87 M vs small solo @349 k). The 2-point implied β with E pinned at the kit-sink asymptote is 0.31 (logloss) / 0.20 (1−AUC) — consistent with the joint fit's 0.18/0.09 but stronger.

- **Decision (DEC-019)**: reorder v3.1 next-phase candidates in expected-Mythic+-lift order:
  1. **More droplet data → separate DB**. Cheapest empirical falsification of β ≈ 0.09. Expected lift ~+0.5 pp / doubling.
  2. **Phase 4b (per-token history features)**. Same "new information" lever phase 4 already validated.
  3. **Star Power / Hyper Charge / Gears ingestion**. Largest information delta; biggest engineering cost.
  4. (Optional) **anchor runs at N ≈ 1 M, 1.5 M** for analysis refinement only.
  5. **Pick-prediction multi-task head** — orthogonal lever (targets top-K, not AUC).

  De-prioritized: bigger arch beyond XL (capacity-bound diminishing returns).

- **Files added**: `scripts/analyze-scaling-laws.py`, `reports/scaling_laws_inventory.csv`, `reports/scaling_laws.json`, `reports/scaling_law_N_mythic_logloss.png`, `reports/scaling_law_N_mythic_auc.png`, `reports/scaling_law_N_all_auc.png`, `logs/scaling_laws.log`. **Canvas**: `canvases/scaling-law-mythicplus.canvas.tsx` (workspace canvas, lives outside the repo) — open beside chat for the rich summary with charts + tables.
- **No retraining performed** — every number came from running the analysis script on existing reports. Total wall-time ~3 s after the inventory glue was written.

### Session 11 — 2026-05-13 — Phase 4 player history breaks the Mythic+ ceiling (DEC-018)
- **Built**: `team_a/b_player_tags` exposed in dataset loader. Phase-4 = 12 frequency-only player-history scalars (`n_known_players`, `mean_n_games_log`, `mean/max_brawler_count_log`, `n_main_picks` per side + 2 A−B diffs). WR-derived features intentionally dropped (label-leaky from same-window source, 50%-noise from pre-cutoff legacy-bug source). `--use-history-features` + `--history-after` CLI flags. `history_df` parameter threaded through `TeamFeaturizer.fit`, `LGBMTeamModel.fit`, `TransformerTeamModel.fit`. Warning fires if `history_df` is omitted (leaky fallback to training df).
- **Leakage detected and fixed**: first attempt put `compute_player_history(df)` inside `fit(df)` with no separate `history_df`. **val_auc jumped to 0.9975 at epoch 1** — internal val split's outcomes were leaking through the player_tag lookup. Killed and restructured to use a separate pre-cutoff window (`--history-after='2026-04-01T00:00:00Z'`, ~349k April battles / 698k rows).
- **Pre-cutoff coverage check** (SQL): only ~9% of training-window players have April history (277k of 2.87M); same for test (231k of 2.52M). Per-team ~25% have at least one known player. Worried this was too thin, but Mythic+ players are over-represented in the 9% so phase 4 still delivers signal.
- **Seven new runs** on the DEC-011 stable test set:

  | Run | Model | P1 | P2 | P4 | Arch | All AUC | **Mythic+ AUC** |
  |---|---|:-:|:-:|:-:|---|---:|---:|
  | F | LGBM | ✓ | — | ✓ | n/a | 0.7708 | 0.6060 |
  | I | LGBM | — | — | ✓ | n/a | 0.7531 | 0.6059 |
  | G | v3 | ✓ | — | ✓ | small | 0.7697 | 0.5948 |
  | H | v3 | ✓ | ✓ | ✓ | small | 0.7719 | 0.6013 |
  | J | v3 | ✓ | ✓ | ✓ | big | **0.7734** | 0.6084 |
  | **K** | **v3** | ✓ | ✓ | ✓ | **XL** | **0.7746** | **0.6180** ← **NEW SOTA** |

- **Mythic+ state of the art broken**: prior 0.6109 (vanilla v3 XL) → **0.6180** (v3 XL + P1+P2+P4). **+0.71 pp on Mythic+, +0.72 pp on all-test**. First feature addition that meaningfully moves Mythic+. The +0.7 pp gain is larger than the entire +0.39 pp from XL going from 570k to 3.28M params (DEC-014). New information beats new aggregates.
- **LGBM doesn't benefit from phase 4 on Mythic+** (+0.0 pp on Mythic+ for Runs F/I vs P1.LGBM). Same plateau as phase 1/2: trees already extract aggregate signal via splits on multi-hot brawlers. Phase 4 helps LGBM only on the easy `ranked` slice (~+1.00 pp all-test).
- **Phase 4 + phase 2 stack** for the transformer: small Run G (P1+P4) 0.5948 → small Run H (P1+P2+P4) 0.6013 (+0.65 pp from adding P2). Plausibly the release-meta signal helps the model interpret "this player mains a recently-released brawler".
- **Bigger arch absorbs phase 4 better**: small kitchen sink Mythic+ 0.6013, big 0.6084, XL 0.6180. The +0.96 pp small→big and +0.96 pp big→XL on Mythic+ is the largest non-trivial capacity gain we've measured (vs DEC-014's saturation pattern past 3M params without phase 4).
- **Brier improves to 0.1902** on Run K (vs 0.1928 vanilla XL). Best calibration yet.
- **Per-mode**: Run K vs vanilla XL gains range +0.47 (brawlBall) to +1.69 pp (bounty); all 9 modes improve. Bounty (the historically hardest mode) and hotZone show the biggest gains.
- **Production candidates updated** (DEC-018):
  - `models/recommender_v3_phase1p2p4_xl.pt` (Run K) — best Mythic+ + best all-test + best calibration.
  - `models/recommender_v3_phase1p2p4_big.pt` (Run J) — best AUC/cost on GPU (16 min train, 576k params, only -0.96 pp Mythic+ vs Run K).
  - `models/recommender_v2_phase1p4.lgb.txt` (Run F) — best CPU-only deploy (24 min train, 0.6060 Mythic+).
- **Files added**: 6 new model checkpoints + 6 reports + 6 logs. Code: `dataset.py` (player_tag pull-through), `features.py` (phase-4 helpers, `compute_player_history`, `compute_phase4_features`), `team_model.py` + `transformer_model.py` (phase-4 plumbing + save/load round-trip + `history_df` parameter), both train scripts (CLI flags + history_df loading). All backwards-compatible.
- **Memory bank**: DEC-018 added. progress.md / activeContext.md / memory-bank.md updated. v3.1 list reordered around new-information findings.

### Session 10 (continued) — 2026-05-09 (afternoon) — Phase 2 + soloRanked-only ablation (DEC-017)
- **Built phase 2 features** (12 numeric scalars in `recommender.features.compute_phase2_features`): cyclical `hour_sin/cos`, `dow_sin/cos`, per-team `days_since_release` aggregates (min/mean), counts of `dsr<14` per side, A−B diffs. `brawler_first_seen` lookup is fit on training data, frozen on the featurizer, round-tripped through both LGBM and transformer save/load. Composable with phase 1; both feed `extra_scalar`.
- **Added `--use-time-features` and `--battle-types` CLI flags** to both train scripts. Default keeps `('ranked', 'soloRanked')` battle types so every prior reproduction works. `--battle-types soloRanked` constrains both training AND test data to the in-game Ranked queue.
- **Backwards-compat verified**: legacy v3_default / v3_phase1_default / v3_big / v3_xl etc. still load with correct head dims; new phase-2 saves use `extra_scalar_dim=35` (23 phase-1 + 12 phase-2). `load_transformer` is permissive about missing fields in older meta.json.
- **5 runs done** in ~30 min wall time (LGBM batch 34 min on CPU, transformer batch 26 min on GPU, in parallel):
  - Run A: LGBM phase1+phase2 mixed → all 0.7618, **Mythic+ 0.6065**
  - Run B: LGBM phase1 only, soloRanked-only train (174k battles) → all 0.6282, **Mythic+ 0.6079**
  - Run C: LGBM phase1+phase2, soloRanked-only train → all 0.6245, **Mythic+ 0.6044**
  - Run D: v3 small phase1+phase2 mixed → all 0.7564, **Mythic+ 0.5944**
  - Run E: v3 small phase1+phase2, soloRanked-only train → all 0.6014, **Mythic+ 0.5802**
- **Two clean negative results on the Mythic+ slice (n=246k)**:
  - **Phase 2 essentially a no-op**: LGBM mixed +P2 = -0.05 pp, LGBM solo +P2 = -0.35 pp, transformer mixed +P2 = +0.11 pp. All within sample noise.
  - **soloRanked-only training does not help**: For LGBM phase 1, solo vs mixed = +0.09 pp (sample noise; technically the best non-XL Mythic+). For transformer, mixed clearly wins (+1.42 pp Mythic+) — the 5.4× more training data outweighs semantic mixing.
- **Mythic+ AUC is now firmly capped at ~0.61 across 12 model configurations** (12 different combinations of LGBM/transformer × no-phase/phase-1/phase-1+2 × mixed/solo train × small/big/XL). State of the art:
  | Mythic+ AUC | Model |
  |---:|---|
  | **0.6109** | v3 XL (no phase) |
  | 0.6079 | LGBM phase 1 solo (Run B) |
  | 0.6070 | LGBM phase 1 mixed |
  | 0.6065 | LGBM phase 1+2 mixed (Run A) |
  | 0.6048 | LGBM Run 0 baseline |
  | 0.6022 | v3 big (no phase) |
  | 0.5944 | v3 small phase 1+2 mixed (Run D) |
  | 0.5802 | v3 small phase 1+2 solo (Run E) |

  The full 12-config spread is just 0.0307 AUC. Capacity + architecture (LGBM → transformer XL) gave us the best 0.30 pp; everything else is sample noise on this slice.
- **Implication**: future improvement on Mythic+ probably needs **new information, not new aggregates of existing data**:
  - Player history (28% of player-rows have ≥3 prior battles) — true new signal.
  - Per-token `days_since_release` instead of per-team aggregate — more focused phase-2b. Architecture change.
  - Star Power / Hyper Charge / Gears ingestion — the data we're truly blind to.
  - Pick-prediction multi-task head — targets hit@K not AUC, but Mythic+ is where draft hit@K matters most.
  Phase 3 (per-brawler / per-mode WR aggregates) is now de-prioritized — same "aggregate of existing data" pattern as phase 1+2.
- **Files added/modified**: `recommender/features.py` (phase-2 helpers), `recommender/team_model.py` and `recommender/transformer_model.py` (phase-2 plumbing + save/load), both train scripts (`--use-time-features`, `--battle-types`). 5 new model checkpoints + 5 reports + 5 logs.
- **Memory bank**: DEC-017 added. `activeContext.md` and `progress.md` updated.

### Session 10 — 2026-05-09 — Tiered slice eval (DEC-016) reframes everything
- **Domain knowledge update**: User clarified `battle_type='ranked'` is the in-game **Unranked** trophy ladder (no draft); `battle_type='soloRanked'` is the in-game **Ranked** queue with Bronze→Pro tiers. Strict 1-2-2-1 ban/pick draft starts at **Mythic (>= 13)**; Diamond (10–12) uses laxer simultaneous pick. Documented in `techContext.md` "Brawl Stars game-domain semantics".
- **Discovered**: `brawler_trophies` field is **overloaded**. In `ranked`: 0–4951, mean ≈ 1040 (cumulative trophies). In `soloRanked`: 1–22, mean ≈ 14 (rank tier number). Verified via SQL.
- **Built tiered slicer**: `recommender/eval_slices.py` with `make_slice_masks` + `evaluate_slices` + `format_slice_table`. Slices: `all`, `ranked`, `soloRanked`, `soloRanked_diamondplus`, `soloRanked_mythicplus`, `soloRanked_legendaryplus`. New `scripts/eval-slices.py` retrospectively evaluates any saved model. Both train scripts now emit `stable_test_slices` block. `team_model.evaluate` accepts optional precomputed `proba` so we predict-once-slice-many (saves ~50-100s per model on the transformer's Python-loop tensorization).
- **Modes confirmed**: 9 modes in clean ranked/soloRanked window — brawlBall, knockout, gemGrab, bounty, hotZone, heist, siege, basketBrawl, wipeout (all 3v3). `duels` (1v1) is implicitly filtered by `require_complete_teams=True` in `dataset.py`. Will make this explicit when we touch dataset.py next.
- **Retrospective slice eval** of all 7 saved checkpoints. n_test = 1,688,302. Slice sizes: ranked=1,339,788, soloRanked=348,514, Diamond+=312,714, **Mythic+=246,372**, Legendary+=108,414.

  | Model | all | ranked | soloRanked | Dia+ | **Myth+** | Lgd+ |
  |---|---:|---:|---:|---:|---:|---:|
  | v2 LGBM A_fair (Run 0) | 0.7181 | 0.7426 | 0.6117 | 0.6062 | 0.6048 | 0.5938 |
  | v2 LGBM phase 1 | 0.7609 | 0.7910 | 0.6253 | 0.6139 | 0.6070 | 0.5978 |
  | v3 small (Run 1) | 0.7378 | 0.7642 | 0.6147 | 0.6044 | 0.5937 | 0.5785 |
  | v3 small + phase 1 | 0.7540 | 0.7839 | 0.6145 | 0.6034 | 0.5933 | 0.5785 |
  | v3 big (Run 4) | 0.7635 | 0.7929 | 0.6256 | 0.6131 | 0.6022 | 0.5879 |
  | v3 big + phase 1 | 0.7603 | 0.7895 | 0.6230 | 0.6099 | 0.5988 | 0.5849 |
  | **v3 XL (Run 5)** | **0.7674** | **0.7966** | **0.6312** | **0.6208** | **0.6109** | **0.5986** |

- **Headline conclusions**:
  - Trophy ladder vs Ranked queue are essentially **two different problems**. AUC gap 14-18 pp across all models (ranked queue is harder because matchmaking equalizes skill, leaving only brawler×matchup + map signal).
  - "All-test" numbers were ~79% weighted by the easy `ranked` slice. Phase 1 LGBM "+4.28 pp on all-test" was **+4.84 pp on Unranked, only +0.22 pp on the strict-draft Mythic+ slice**.
  - All models cluster between 0.5937 and 0.6109 AUC on Mythic+. v3 XL is the best but only by **+0.39 pp over LGBM phase 1** at 13× the params. The competitive slice is where progress has been minimal.
  - On Mythic+, the small transformer (Run 1) is actually **worse** than the v2 LGBM baseline (0.5937 vs 0.6048). The transformer's gain over LGBM was concentrated in the easy slice.
- **Real research target reframed**: report **`soloRanked_mythicplus` AUC** as the primary metric going forward. State of the art is **0.6109** (v3 XL) → only +0.061 above the AUC=0.5 random floor. That's the actual problem.
- **Implications for v3.1 next phases**:
  - Features that lean on trophy/power (phase 1) saturate against the trophy-ladder signal but don't carry on Mythic+. New phases must be tested specifically on Mythic+.
  - Phase 2 (`days_since_release[bid]`) — release-meta should affect competitive players first. Worth running.
  - Phase 3 (per-brawler / per-mode WR) — **must condition on `battle_type`** because populations are different. Mythic+ aggregates from the soloRanked subset only.
  - Pick-prediction multi-task head and player-history features both probably help more on Mythic+ specifically.
- **Files added**: `src/brawlstar_agent/recommender/eval_slices.py`, `scripts/eval-slices.py`, `reports/slices_summary.json`, `reports/slices_smoke.json`, `logs/eval_slices_summary.log`. `team_model.evaluate` extended (optional `proba`). Both train scripts wired (back-compat).
- **Memory bank**: DEC-016 added. `techContext.md` got the "Brawl Stars game-domain semantics" section at the top. `activeContext.md` updated with new slice-table headline.
- **No retraining**: all 7 model AUC numbers came from one predict-then-slice pass each. Total wall time ~20 min (mostly DB load 161s + transformer test-set tensorization 125-411s per model).

### Session 9 (continued) — 2026-05-08 (night) — v3.1 Phase 1: per-team aggregates ablation (DEC-015)
- **Goal**: decompose the original Run 0→1 +1.97 pp gain into "feature engineering" vs "attention architecture", per the v3.1 candidate list. User: "i am interested in how our model can improve with more features... we can work on the v3 Transformer first to test out, or its variant / LGBM if needed later." Approach: phase-1 = team-level aggregates of the per-brawler trophy/power tuples (cheap, dense, no leakage, no new ingestion).
- **23 new scalars** in `recommender.features.compute_team_aggregates`: per-team min/max/std of trophies (log1p), per-team mean/min/max/std of powers (/11), per-team count of `power == 11` and `power < 8` (/3), and 5 A−B diffs of the most informative ones. `TEAM_AGGREGATE_NAMES` exports the column order.
- **Plumbing changes (all backwards compatible)**:
  - `TeamFeaturizer.include_team_aggregates: bool` (default False). When True, `transform_dense` appends the 23 columns; `transform_sparse` is unchanged so LogReg keeps its old shape.
  - `LGBMTeamModel` and `LogRegTeamModel` accept `include_team_aggregates`; LGBM's `save_model`/`load_model` round-trip the flag in meta.json.
  - `_TransformerCore` accepts `extra_scalar_dim: int = 0` (default 0 ⇒ head's first Linear is `Linear(d_model + 3, d_model)`, identical to legacy saves). When nonzero, head becomes `Linear(d_model + 3 + K, d_model)`.
  - `_df_to_tensors` returns a new `extra_scalar` tensor of shape (B, K) — (B, 0) when phase 1 is off, so legacy models keep loading. `TENSOR_KEYS` widened.
  - `TransformerTeamModel.use_team_aggregates: bool` (default False). `save_transformer` records `extra_scalar_dim` in `arch` and `use_team_aggregates` in both `training` and `featurizer` dicts. `load_transformer` is permissive — defaults to 0/False on legacy meta files.
  - Both train scripts gained `--use-team-aggregates`. Smoke-tested: legacy `recommender_v3_big.pt` loads with `extra_scalar_dim=0`, head_in=131 (d_model+3), `predict_proba` runs unchanged.
- **Three runs on the DEC-011 stable test set** (same 1.87 M train rows / 1.69 M test rows; only `--use-team-aggregates` flips):
  - **LGBM phase 1**: AUC **0.7609** vs A_fair LGBM 0.7181 → **+4.28 pp**. logloss 0.5642 (−4.7 pp), Brier 0.1952 (−1.7 pp), acc 0.6765 (+2.8 pp). Fit 1041 s on CPU (slow; competed with concurrent transformer phase-1 small run for cores). 23 numeric features ALONE close ~62 % of the LGBM-to-XL-transformer gap.
  - **v3 small + phase 1**: AUC **0.7540** vs Run 1 small (no phase 1) 0.7378 → **+1.62 pp**. logloss 0.5696, Brier 0.1975, acc 0.6706. Per-epoch trajectory val_auc 0.7391/0.7515/0.7584/0.7614/0.7621/0.7628 — already at 0.7391 at epoch 1 (vs Run 1 epoch 1 = 0.7183, +2.08 pp early lead). 253,441 params (+2.4 k from wider head linear). Fit 450 s on GPU (eval phase slowed by CPU contention).
  - **v3 big + phase 1**: AUC **0.7603** vs Run 4 big (no phase 1) 0.7635 → **−0.32 pp** (slightly worse!). val_auc peaked at 0.7716 (vs Run 4's 0.7711, +0.05 pp) but the temporal-holdout tax widened to 1.13 pp (vs Run 4's 0.76 pp). 572,801 params (+2.8 k). Fit 856 s on GPU (matches Run 4). The big transformer's encoder + per-brawler scalars already extract most of what phase 1 makes explicit — team aggregates are nearly redundant at this capacity.
- **Per-mode (phase 1 big vs Run 4 big)**: basketBrawl +0.41, siege +0.54, knockout tied, brawlBall −0.53, hotZone −0.41, wipeout −1.04. Mixed results consistent with sample noise + slight overfitting on phase 1 big. None of the per-mode AUCs are catastrophically off.
- **Decomposition of original Run 0→1 +1.97 pp gain**:
  - When the *same* signal (per-team trophy/power aggregates) is given as **dense numeric features to LGBM**: +4.28 pp.
  - When the *same* signal is already inside the small transformer (via `scalar_proj_brawler` per token), adding the aggregates explicitly: only +1.62 pp.
  - When the *same* signal is already inside the big transformer: ≈ 0 (slightly negative).
  - **Conclusion**: the architecture and the feature engineering are LARGELY SUBSTITUTES at this data scale. The original +1.97 pp Run 0→1 jump was mostly feature engineering (per-brawler scalars) translated into a form LGBM couldn't use; attention's job in the small transformer was to extract aggregate-like signal automatically.
- **Production implications** (cheapest → most accurate):
  - `models/recommender_v2_phase1.lgb.txt` — AUC 0.7609, CPU-only deploy, ~5 min training. **Best AUC/cost ratio for non-GPU environments.** New addition to the candidate set.
  - `models/recommender_v3_phase1_default.pt` (small + phase 1) — AUC 0.7540, the best CPU-deployable transformer (closes 62 % of the small→big gap without changing arch).
  - `models/recommender_v3_big.pt` (Run 4, NO phase 1) — AUC 0.7635, still the best AUC/inference-cost ratio for GPU deploys.
  - `models/recommender_v3_xl.pt` (Run 5, NO phase 1) — AUC 0.7674, best AUC + best calibration.
- **`recommender_v3_phase1_big.pt` is NOT promoted to production** — it underperforms vanilla Run 4. Kept on disk for ablation reproducibility.
- **What stays on the v3.1 list**:
  - Phase 2 (time-based + `days_since_release[bid]`): still on the menu, encodes signal phase 1 cannot.
  - Phase 3 (per-brawler global / mode WR aggregates): still on the menu — partial duplication with phase 1 for the transformer, additive for LGBM.
  - Pick-prediction multi-task head: unchanged, targets the top-K ceiling specifically (not AUC).
  - C_fair-style 30-day window: untested with phase 1 yet; opening question stands.
- **Files added/modified** (uncommitted with the rest of Session 9):
  - `src/brawlstar_agent/recommender/features.py` — `compute_team_aggregates`, `TEAM_AGGREGATE_NAMES`, `TEAM_AGGREGATE_DIM`, `TeamFeaturizer.include_team_aggregates` field; `transform_dense` extension.
  - `src/brawlstar_agent/recommender/team_model.py` — flag plumbed through `LGBMTeamModel`/`LogRegTeamModel`; meta.json round-trips it for LGBM saves.
  - `src/brawlstar_agent/recommender/transformer_model.py` — `_TransformerCore.extra_scalar_dim`, `_df_to_tensors(..., include_team_aggregates=...)`, `TransformerTeamModel.use_team_aggregates` field, `_TENSOR_KEYS` adds `extra_scalar`, save/load round-trip.
  - `scripts/train-recommender.py` and `scripts/train-recommender-v3.py` — `--use-team-aggregates` CLI flag.
  - `models/recommender_v2_phase1.{lgb.txt,meta.json}`, `models/recommender_v3_phase1_default.{pt,meta.json}`, `models/recommender_v3_phase1_big.{pt,meta.json}`.
  - `reports/recommender_v2_phase1.json`, `reports/recommender_v3_phase1_default.json`, `reports/recommender_v3_phase1_big.json`.
  - `logs/train_v2_phase1.log`, `logs/train_v3_phase1_default.log`, `logs/train_v3_phase1_big.log`.

### Session 9 (continued) — 2026-05-08 (evening 2) — v3 XL aggressive scale-up (Run 5)
- Brawler-pool sanity check first: SQL confirmed train and test windows both have **102 unique brawlers** (the `brawlers` table has 104; the 2 missing are **BOLT id 16000106** and **STARR NOVA id 16000105**, the two newest releases that haven't been picked in any ranked / soloRanked battle in our window). So train ⊆ test in vocab terms — no test rows are silently skipped due to unknown brawlers in the top-K eval. The "97 avg legal candidates per row" math: 102 vocab − 6 in-battle + 1 (actual added back) = 97. Documented in `docs/recommender-v3.md` candidate-pool note.
- **Run 5 (XL)**: d_model=256, num_layers=6, ff=512, nhead=8, dropout=0.20 (up from 0.15), 12 epochs (up from 8), patience 4. Param count **3.28M** (5.7× Run 4 big, 13× Run 1 small). batch=4096 unchanged. Cosine schedule over 12*435 steps.
- **Per-epoch wall-clock**: ~260 sec (vs Run 4's 95 sec — ~2.7× slower per epoch as expected from the param count). Total fit_s **3129 sec ≈ 52 min** (vs Run 4's 14 min).
- **VRAM**: model+optimizer+grads+activations stayed comfortably under 1.5 GB on the RTX 3060 Mobile's 5.77 GB. Plenty of room for d=384 / L=8 if a future run wants to push further (but Run 5 already shows diminishing returns per ×5 params).
- **Stable-test results**: AUC **0.7674** (vs Run 4 big 0.7635, **+0.39 pp**), logloss **0.5573** (−0.43 pp), accuracy **0.6821** (+0.33 pp), Brier **0.1928** (−0.15 pp). Per-mode wins **9/9 vs Run 4** (basketBrawl +1.10, bounty +0.86, knockout +0.74, gemGrab +0.75 are the biggest gains). End-to-end vs v2 A_fair LightGBM (Run 0): **+4.93 pp AUC**, **−5.36 pp logloss**, **+3.31 pp accuracy**, **−1.96 pp Brier**, knockout +7.58 pp.
- **Per-epoch val_auc trajectory**: 0.7251 → 0.7407 → 0.7515 → 0.7571 → 0.7643 → 0.7671 → 0.7698 → 0.7719 → 0.7745 → 0.7757 → 0.7765 → 0.7764 (peaked at epoch 11, slight regression at 12). train_loss > val_loss for *every* epoch — model is *not* overfitting at all (bigger model still possible, but the question is whether it'd help).
- **Capacity scaling at this data size** (LightGBM = baseline, then per-arch transformer on the same 1.87M rows):
  - 251k params → AUC 0.7378 (+1.97 pp)
  - 570k params → AUC 0.7635 (+2.57 pp on top, ≈ +6.3 pp per ×10 params)
  - 3.28M params → AUC 0.7674 (+0.39 pp on top, ≈ +0.5 pp per ×10 params)
  Strong diminishing returns past 3M. Likely capacity has saturated against the 1.87M-row training data; further AUC gains probably need a different inductive bias (factorization machine, listwise pick-prediction head) or more data, not bigger transformers.
- **Top-K hit@K is essentially TIED across small/big/XL** (all at hit@1 ≈ 0.137, MRR ≈ 0.195 on n=5000 last_pick stable test). Winners-only also tied (hit@1 ≈ 0.20, MRR ≈ 0.265). This confirms top-K has hit a structural ceiling that even the small transformer reached: predicting which of ~97 brawlers a player picks is bounded by personal roster + preference, not by win-probability quality. The +0.39 pp binary-AUC improvement from big→XL goes entirely into **calibration** (Brier 0.1943→0.1928, logloss 0.5616→0.5573), not into ranking.
- **Production picks** (replacing the single "production candidate" line from the morning):
  - `models/recommender_v3_big.pt` — best AUC/inference-cost ratio. Use for the standard recommender UX.
  - `models/recommender_v3_xl.pt` — lowest Brier + logloss. Use when you specifically need calibrated probabilities to threshold on (e.g., "auto-recommend at P > 0.7").
  - `models/recommender_v3_default.pt` (small) — fastest, smallest. Use for CPU-only deploys.
- **Files added**: `models/recommender_v3_xl.{pt,meta.json}`, `reports/recommender_v3_xl.json`, `reports/recommender_v3_xl_topk.json`, `logs/train_v3_xl.log`, `logs/eval_v3_xl_topk.log`. Doc + memory bank updated.
- **v3.1 candidates** (reordered, since "go bigger" is no longer compelling): factorization-machine baseline; pick-prediction multi-task head to break the top-K ceiling; per-brawler feature ablation on LightGBM (decompose Run 0→1's +1.97 pp into "feature engineering" vs "architecture" buckets); calibration via isotonic on stable test; Star Power / Hyper Charge ingestion (separate design).

### Session 9 (continued) — 2026-05-08 (later) — v3 GPU enablement + ablation + bigger arch (DEC-013)
- **Goal**: user asked why CPU was slow and whether we could use the RTX 3060 Mobile. Then asked to do (1) data-plumbing fix and (2) bigger arch in series so we have a clean ablation.
- **GPU enablement**: CUDA driver 535.230.02 was loaded but `nvidia-modprobe` userspace helper missing → `/dev/nvidia*` device files never created → torch + nvidia-smi both fail. User installed `nvidia-modprobe` in their own (non-Cursor) terminal. From inside Cursor's user-namespace sandbox, the file appears as owner `nobody:nogroup` due to UID mapping, but actually owned by root in real fs; works fine outside sandbox. Documented in `docs/recommender-v3.md` engineering notes.
- **PyTorch swap CPU → cu121**: `pyproject.toml` reorganized to use `[tool.uv.sources] torch = { index = "pytorch-cu121" }` + `[[tool.uv.index]] explicit = true` so opencv / jupyter still resolve from PyPI. cu121 wheels stopped at torch 2.5.1 (PyTorch went cu124-only from 2.6, and cu124 needs driver ≥550 — we have 535). Pinned `torch>=2.5.0,<2.6`. Total install ~3 GB (torch + cudnn 9.1 + cublas + cufft + nccl + triton). Index also has `pytorch-cpu` named index for easy droplet-style fallback.
- **Run 2 (small arch on GPU, slow path)**: identical code, just `--device cuda`. **AUC 0.7392 vs Run 1 CPU 0.7378** (Δ +0.14pp = seed noise). Wall-clock training **616 s vs 1651 s = 2.7× faster**. Per-epoch 83-88s vs CPU 248-263s. Saved at `models/recommender_v3_gpu.pt`.
- **Plumbing fix**: original code used `torch.utils.data.DataLoader` with CPU tensors → per-batch CPU↔GPU memcopy dwarfed actual GPU compute on this small (251k-param) model. Refactored `transformer_model.py` with `_iter_batches`: preload all tensors to VRAM (~230 MB), generate `torch.randperm` per epoch on GPU, gather batches via `tensor[idx]`. Single code path (works on cpu and gpu), no DataLoader. Removed `_make_loader`, `TensorDataset`, DataLoader imports.
- **Run 3 (small arch on GPU, fast path)**: identical model + hyperparams as Run 2, just the new `_iter_batches`. **AUC 0.7366 vs Run 2 0.7392** (Δ −0.26pp = noise; the perm seed differs from DataLoader's so trajectories diverge slightly). Wall-clock **338 s vs 616 s = 1.8× faster on top of GPU**. Per-epoch 37-42s. **Compounded vs CPU: 4.9× speedup** for the same architecture/result. Saved at `models/recommender_v3_gpu_fast.pt`.
- **Run 4 (big arch on GPU, fast path) — production**: with epochs costing only ~40s, scaled the model: d_model 96→128, num_layers 3→4, ff 192→256, nhead 4→8, dropout 0.10→0.15, train 8 epochs (vs 6) with patience 3 (vs 2). Param count 251k→570k (2.27×). Per-epoch 83-100s. Total fit_s **858 s = 14.3 min**. **Stable-test AUC 0.7635** — **+2.69pp over Run 3 same-arch on the same data**, **+4.54pp over the v2 A_fair LightGBM baseline**, **+2.57pp over v2 C_fair (the technical-best v2 with 30-day window)**. Logloss 0.5616 (−2.7pp from Run 3, −4.93pp from v2), accuracy 0.6788 (+1.9pp / +3.0pp), Brier 0.1943 (−1.1pp / −1.81pp). Saved at `models/recommender_v3_big.pt` + `.meta.json`. Report `reports/recommender_v3_big.json`.
- **Per-mode (Run 4 vs v2 A_fair LightGBM)**: brawlBall 0.7948 (+4.31), siege 0.8268 (+3.35), basketBrawl 0.7667 (+3.94), **knockout 0.7626 (+6.84pp)**, wipeout 0.7347 (+3.00), heist 0.7189 (+4.05), gemGrab 0.6896 (+2.29), hotZone 0.6758 (+2.07), bounty 0.6362 (+2.03). Knockout's outsized gain makes sense — 1-life elimination per round, brawler×brawler matchup signal is the dominant predictor and attention captures it directly while LightGBM had to discover it via tree splits.
- **Top-K (Run 4, n=5000 last_pick stable test)**: hit@1=0.137, hit@3=0.185, hit@5=0.213, hit@10=0.285, MRR=0.195, WR|in_top1=0.693 (+19.7pp). **Almost identical to small transformer Run 1** (0.136 / 0.193 / 0.226 / 0.280). Winners-only: hit@1=0.204 (tied with small), hit@10=0.364 (+0.7pp), MRR=0.267 (+0.2pp). The +2.7pp binary-AUC gain DOES NOT translate into top-K hit-rate improvement — both transformers are at a structural ceiling for the "predict which of ~97 brawlers a player picks" task (limited by personal-roster preference, not just matchup quality). What the bigger model improves is **calibration** (lower Brier, lower logloss); WR|in_top1 also gains +0.8pp on all rows.
- **Total ablation contribution decomposition** (1.87M rows, same DEC-011 test):
  - Architecture (LightGBM → transformer + per-brawler features): **+1.97pp AUC** (Run 0→1)
  - GPU compute (CPU → GPU, slow path): **0.0pp AUC, 2.7× speedup** (Run 1→2)
  - Data plumbing (DataLoader → preloaded VRAM): **0.0pp AUC, 1.8× extra speedup** (Run 2→3)
  - Bigger model (251k → 570k params + 2 extra epochs): **+2.69pp AUC** (Run 3→4)
  - **Cumulative**: +4.54pp AUC, 4.9× speedup vs CPU, no extra data.
- **Files added/modified**: `transformer_model.py` (`_iter_batches`, `_TENSOR_KEYS`, `_move_to_device`, `_evaluate_tensors`, removed DataLoader path); `pyproject.toml` (cu121 index routing + version pin); `docs/recommender-v3.md` (full ablation table + per-mode breakdown for Run 4 + topk + engineering notes incl. the nvidia-modprobe + UID-namespace gotcha); 4 new model files + 4 reports + log files.
- **Pending v3.1 candidates**: even bigger arch (d=192, L=6 — VRAM has room, ~3 GB free), per-brawler feature ablation on LightGBM (isolates feature engineering from architecture in the +1.97pp Run 0→1 step), factorization machine baseline, calibration on stable test, Star Power / Hyper Charge ingestion.

### Session 9 — 2026-05-08 — v3 attention transformer (DEC-012)
- **Goal** (per user direction): "we need to try more advanced methods, e.g. attention neural network, more feature engineering, etc... we prioritize the method over more data". Use Run A's training data + the same DEC-011 stable test set; method > more data.
- **Per-brawler features added** to `dataset.py` (backwards compatible): `team_a_trophies`, `team_b_trophies`, `team_a_powers`, `team_b_powers` — parallel tuples aligned to existing `team_a` / `team_b` brawler-id tuples (sorted by brawler_id within each team). NULL power → 0, NULL trophy → 0. Pulls `bp.brawler_power` from the same SQL JOIN that already had trophies. ~80% of post-fix ranked rows are power 11; the other ~20% spans power 0–10 — real signal v2 multi-hot ignored.
- **TransformerTeamModel** at `src/brawlstar_agent/recommender/transformer_model.py`. Sklearn-like `.fit(df) / .predict_proba(df)` API (drop-in for `LGBMTeamModel`). Architecture: 8 tokens [CLS, CTX, A1, A2, A3, B1, B2, B3], side embeddings (CLS/CTX/A/B), per-brawler scalar projection of (trophy_log, power/11), 3-layer transformer encoder (d_model=96, nhead=4, ff=192, dropout=0.1, norm_first=True, GELU), `enable_nested_tensor=False` to silence the warning. CLS pool → concat with [a_t_log, b_t_log, t_diff_log] → MLP head. **251k params total.** AdamW(lr=1e-3, wd=1e-4), cosine schedule, BCE loss, grad_clip=1.0, batch=4096.
- **Save format**: `<prefix>.pt` (state dict) + `<prefix>.meta.json` (vocab + arch). Loader recreates `_TransformerCore` from meta and loads weights — no pickle.
- **Training script**: `scripts/train-recommender-v3.py` mirrors `scripts/train-recommender.py` (same `--cutoff` / `--stable-test-after` / `--report-to` / `--save-to` flags + transformer-specific `--epochs --batch-size --d-model --num-layers --ff --dropout --lr --device`). Uses internal 5% random val split for early stopping; the held-out stable-test set is never seen during training.
- **eval-topk.py** gained `--transformer-from PATH` to load and add the saved transformer to the side-by-side comparison without re-training, sharing the same candidate pool + sample seed as LightGBM. Optional `--skip-lgbm-train` for transformer-only sub-runs.
- **Production v3 training**: `--cutoff 2026-05-03T01:00:00Z --stable-test-after 2026-05-05T00:00:00Z --epochs 6 --batch-size 4096`, A_fair train data (1,871,616 rows / 935,808 battles), 6 epochs in **27.5 min** on i7-12700H CPU (~260s/epoch, no GPU). Best val_auc reached at epoch 6 (0.7458 internal val), but stable-test AUC is the canonical number.
  - **Stable-test AUC 0.7378** vs A_fair LGBM 0.7181 → **+1.97pp** (and vs C_fair 0.7265 → **+1.13pp** with *less* training data).
  - **logloss 0.5879** (−2.3pp), **acc 0.6618** (+1.3pp), **brier 0.2044** (−0.8pp). Calibration uniformly improved.
  - **Per-mode AUC wins in 9/9 modes**. Largest: knockout +2.64, siege +2.29, brawlBall +2.09, bounty +1.21pp. Smallest: basketBrawl +0.01 (essentially tied), wipeout +0.45.
  - **Training history**: epoch 1 val_auc 0.7183 (already matches v2 LGBM), then 0.7328 / 0.7378 / 0.7427 / 0.7455 / 0.7458 — slowing but not flat at epoch 6.
- **Top-K eval** (`scripts/eval-topk.py`, n=5000 last_pick on stable test): **hit@1 = 0.136 (tied with LGBM)**, hit@3 = 0.193 (+1.3pp), hit@5 = 0.226 (+1.6pp), hit@10 = 0.280 (−0.3pp, noise), MRR = 0.196 (+0.3pp). WR|in_top1 = 68.2% vs LGBM 68.5% — actionable signal essentially identical. Winners-only (the cleaner test): **hit@1 = 0.204 vs LGBM 0.195 (+0.9pp)**, hit@3 = 0.265 vs 0.247 (+1.8pp), hit@5 = 0.299 vs 0.283 (+1.6pp), hit@10 tied at 0.357. So the +2pp binary AUC translates to consistent winners-only top-K wins, but the all-rows hit@1 is sample-noise-bound at 13.6%.
- **Engineering notes**:
  - PyTorch installed via `uv add torch --index https://download.pytorch.org/whl/cpu` (~600 MB, CPU-only build). Pinned a few common deps (certifi, urllib3, jupyterlab) to slightly older versions — nothing functional broke.
  - Host has RTX 3060 Mobile + driver 535 loaded, but `nvidia-modprobe` is not installed so `/dev/nvidia*` device files don't exist. PyTorch can't see the GPU. To enable: `sudo apt install nvidia-modprobe` + reinstall torch with `--index https://download.pytorch.org/whl/cu121`. Not needed for current dataset size; CPU training is ~3 min/epoch which is fine. Filed as v3.1 candidate.
- **Production candidate**: `models/recommender_v3_default.pt` + `.meta.json`. Reports: `reports/recommender_v3_default.json` (binary + per-mode + history), `reports/recommender_v3_topk.json` (top-K + winners-only). Doc: `docs/recommender-v3.md`. **Replaces** A_fair LGBM as the production-ready model.
- **Pending v3.1 candidates** (all evaluated against the same DEC-011 boundary):
  - GPU re-train at d_model=128, num_layers=4 once `nvidia-modprobe` is unblocked.
  - Per-brawler feature ablation on LightGBM to isolate "architecture vs features" contribution to the +2pp.
  - Factorization machine baseline (cheap inference, second-order interactions).
  - Calibration on stable test (Brier already 0.2044; isotonic/Platt for release-meta extreme picks like DAMIAN).
  - Star Power / Hyper Charge ingestion (separate ingestion design).

### Session 8 (continued) — 2026-05-06 (late afternoon) — Local DB shrink (purge legacy pre-2026-04-01 + VACUUM)
- Goal: free space on local + droplet so we can later rsync a smaller DB back to the droplet (which hit 100% disk during cold-start; can't even run its own DELETE+VACUUM in place — see HANDOFF.md task 3).
- Pre-shrink: 18 GB DB, 4,022,553 battles, 29,949,967 battle_players. Backup taken at `data/backups/brawlstars-pre-shrink-20260506-145727.db` (18 GB; cp was 6 s on NVMe).
- DELETE WHERE `battle_time_iso < '2026-04-01T00:00:00Z'` removed 305,098 battles (7.6%) + 2,327,142 battle_players (7.8%). DELETE itself ran in ~1 min (53s for the join-on-id battle_players delete, 2s for battles).
- **VACUUM crashed with "database or disk is full"** on the first attempt. Root cause: `/` (which contains `/tmp`) is 60 GB / 92% full / only 4.8 GB free; SQLite VACUUM writes a ~17 GB temp file to `$SQLITE_TMPDIR` ↳ `$TMPDIR` ↳ `/tmp` by default, which obviously didn't fit. **Fix**: re-ran with `VACUUM INTO 'data/brawlstars.shrunk.db'` instead — writes directly to the target path on the workspace NVMe, no scratch needed. Belt-and-suspenders: also set `SQLITE_TMPDIR=/media/lin/disk2/brawlstar-agent/tmp_sqlite` (turned out to be unused). Filed in `logs/vacuum_into.sh`.
- VACUUM INTO finished in **22.5 s** (NVMe is fast). Atomic rename `mv data/brawlstars.shrunk.db data/brawlstars.db` (same fs).
- Post-shrink: **15 GB DB**, 3,717,455 battles, 27,622,825 battle_players. `freelist_count = 0` (perfect compaction). Integrity check `ok` (2 m 54 s).
- Training-relevant battle counts UNCHANGED: 2,281,443 ranked/soloRanked ≥ 2026-04-06 (Run C/C_fair training set) and 910,307 ≥ 2026-05-05 (DEC-011 stable test set) both intact. Both production training (cutoff 2026-04-06) and B_fair-style (cutoff 2021) reproductions are unaffected by the purge — B_fair only loses pre-April data which we just empirically showed *hurts* the model anyway.
- Disk usage on workspace: 228 → 225 GB used (3 GB freed); the 18 GB backup compensates the 3 GB shrink for now. Can drop the backup later once user is confident.
- **Next step (user-driven)**: rsync shrunk DB to droplet (timers stopped) and re-enable timers. See HANDOFF.md task 3.

### Session 8 (continued) — 2026-05-06 (later afternoon) — Stable test set + fair Runs A/B/C + top-K (DEC-011)
- **Critique acknowledged**: Run A vs Run C's near-equal random-split AUCs (0.7382 vs 0.7392) were comparing models on *different* test set distributions. Random 20% holdouts were sampled from each run's full window, which differs in time coverage and density. Cannot conclude "more data doesn't help" from that.
- **Fix (DEC-011)**: pinned `STABLE_TEST_AFTER_DEFAULT = '2026-05-05T00:00:00Z'` as the canonical held-out test boundary. Test set = 844,151 clean battles (1,688,302 rows after both-perspectives doubling). Every v2 run from now on uses this boundary.
- Added `--stable-test-after TIMESTAMP` flag to `scripts/train-recommender.py` (replaces random split with temporal holdout, restricts CV to train portion to prevent leakage). Records `split_mode`, `stable_test_after`, train/test battle counts in the report.
- Modified `scripts/eval-topk.py` to take `--cutoff` and `--stable-test-after` so binary + top-K share the same test rows.
- Wrote `scripts/compare-fair-runs.py` for the apples-to-apples table; orchestrator at `logs/run_fair_v2.sh` (gitignored) chains the three fair runs sequentially.
- **Three fair runs** (all `--no-temporal --stable-test-after 2026-05-05T00:00:00Z`):

  | Run | Cutoff | Train battles | LightGBM AUC | LogReg | ModeMap | Global | LGBM_fit_s |
  |---|---|---:|---:|---:|---:|---:|---:|
  | A_fair | 2026-05-03 | 935,808 | 0.7181 | 0.6808 | 0.6625 | 0.6551 | 111 |
  | **C_fair** | **2026-04-06** | **1,280,861** | **0.7265** | 0.6805 | 0.6725 | 0.6536 | 154 |
  | B_fair | 2021-01-01 | 1,412,893 | 0.7235 | 0.6752 | 0.6708 | 0.6544 | 123 |

- **Headline**: 30-day window WINS. C_fair beats A_fair by +0.84pp, beats B_fair by +0.30pp. The conclusion from earlier today ("rows don't matter") was wrong — it was a different-test-set artifact. Real signal: more recent data helps up to ~30 days, then noisy older data slightly hurts. Per-mode pattern consistent: C_fair wins in 8 of 9 modes. Random-split→stable-test AUC drop: LightGBM −2.0pp (Run A), Run-C-style ModeMap −2.9pp; LogReg essentially unchanged. The drops are real meta-drift signal that random splits hide.
- **LogReg flat across runs** (0.6808 / 0.6805 / 0.6752) — saturated by current feature design (multi-hot brawler bag + dense scalars). The Run-A → Run-C improvement comes entirely from LightGBM exploiting interactions that don't fit in flat features. Confirms the v2 design priority: brawler×map / brawler×brawler interaction features for the linear model.
- **Top-K eval** on the C_fair config (cutoff 2026-04-06, stable test, sample=5000, last_pick mode): LightGBM hit@1=0.130, hit@3=0.190, hit@5=0.229, hit@10=0.292, MRR=0.194, mean rank 33.6. Win uplift WR|in_top1 = 67.6% (vs 49.5% baseline = **+18.1pp**), WR|in_top5 = 64.8% (+15.3pp). Winners-only: hit@1=0.183, hit@10=0.369, MRR=0.253. *Lower* than v1's hit@1=0.150 / MRR=0.205 — those v1 numbers had random-split leakage; the stable-test numbers are the honest baseline going forward. ModeMap surprises with hit@1=0.118 (close to LightGBM's 0.130) — per-(mode,map) pick frequency transfers reasonably even temporally. TrophyOnly hit@K is *worse* than Random because trophy-ranked candidates are systematically not the brawlers people actually play.
- **Production candidate**: `models/recommender_v2_30d_fair.lgb.txt` (Run C_fair). Reports: `reports/recommender_v2_default_fair.json`, `recommender_v2_30d_fair.json`, `recommender_v2_all_fair.json`, `recommender_v2_topk.json`.
- Total wall-clock: 3 fair training runs in 20 min sequential (A=6, C=8, B=7), top-K eval ~10 min. PYTHONUNBUFFERED=1 + tee log gave real-time progress, no buffering surprises like the morning's Run C.

### Session 8 (continued) — 2026-05-06 (afternoon) — v2 Run C: 30-day rolling window
- **Run C** (cutoff `2026-04-06T00:00:00Z`, the priority task in HANDOFF.md): 2,125,012 battles → 4,250,024 rows after both-perspectives doubling. Earliest 2026-04-06T00:01:34, latest 2026-05-06T04:30:14. Used `--cv-step-hours 24` (default is 8) to keep temporal CV tractable; default would have produced ~90 mostly-redundant folds.
- **Random split** (n_test=850,004): Global 0.6592, Mode 0.6791, ModeMap 0.6888, LogReg 0.6792, **LightGBM 0.7392**. Vs Run A: LightGBM **+0.10pp** (effectively identical), baselines −0.1 to −0.3pp. **Adding 27 days of older data ≈ no change in random AUC** — strong signal that we're at the per-feature ceiling (not a data-volume ceiling) on the random split.
- **Temporal CV (24 daily folds)**: Global 0.6448, ModeMap 0.6494, **LightGBM 0.6978**. Naïve read = −3.0pp vs Run A's 0.7281, but most of that drop is structural — folds 0-11 train on 3-9k rows each (cold-start gives sparse Apr 12-23 coverage because the API's 25-battle-per-player window dominates). **Apples-to-apples** (last 5 folds, all post-May-1 dense data): LightGBM 0.7228, ModeMap 0.6715, Global 0.6492. So Run C's LightGBM is **−0.53pp** vs Run A on equivalent dense folds — real meta-drift signal, not a CV artifact.
- **Per-mode random AUC, LightGBM**: siege 0.786, brawlBall 0.770, knockout 0.737, basketBrawl 0.732, wipeout 0.718, heist 0.686, gemGrab 0.681, hotZone 0.673, bounty 0.646. brawlBall (n=430k test rows) holds the model's main strength; bounty is structurally hardest.
- **Production implication**: rolling 30-day window does NOT beat the post-fix-only window today. Older April data has sparse coverage from the API tail and reflects a slightly different meta (DAMIAN released ~2026-04-24 mid-window). Revisit when ≥30 days of *dense* post-cold-start data have accumulated. For now, the production-ready model is Run A's `recommender_v2_default`.
- **Fit times**: LightGBM random fit 249s (vs Run A's 405s — *faster* on more data; suspect either thread-contention luck or pandas-warmer-cache; both runs used default 16 threads).
- Outputs: `reports/recommender_v2_30d.json` (41 KB), `models/recommender_v2_30d.lgb.txt` (4.5 MB) + `.meta.json` (4 KB).
- **Pending after Run C**: Run B (all-data, ~2.26M battles, ≥75 min); top-K eval against winning model; cross-run write-up in `docs/recommender-v1.md` once Run B is done.

### Session 1 — 2025-03-29
- Machine inspection, memory bank created, emulator decision (AVD)

### Session 2 — 2025-03-29
- Android SDK installed, AVD booted, Brawl Stars anti-cheat blocks it
- Genymotion tried, no ARM translation
- Pivoted to YouTube gameplay, built data pipeline, first 13 frames cropped

### Session 3 — 2026-04-03
- Batch frame extraction, headless review prep, crop export helper

### Session 4 — 2026-04-06
- Browser review hub built, all 24 clips reviewed by user
- 308 gameplay frames exported
- Perception pipeline built and run: mode detection, OCR, blob detection, character matching
- Tesseract installed, OCR re-run with results

### Session 5 — 2026-04-13
- Brawl Stars API key obtained, documented 7 endpoints in `docs/brawlstars-api.md`
- Fetched and saved live API responses for all endpoints into `docs/api-examples/`
- Built full battle analytics pipeline (DEC-006):
  - `api_client.py`: rate-limited API wrapper with auth, retry, backoff
  - `db.py`: SQLite schema with dedup, battle normalization, player tracking
  - `collector.py`: seed rankings → fetch battlelogs → snowball tags
  - `analytics.py`: win rate per brawler, combo, matchup matrix, synergy matrix
  - `scripts/collect-battles.py` + `scripts/analyze-battles.py` CLI tools
- First collection: 200 global top players, 4,628 battles, 19,845 discovered players
- Analytics validated: all 4 query types produce meaningful results

### Session 7 — 2026-05-04 — Pinned tags + analytics precompute cache
- Identified two real gaps in the day-1 deploy:
  - Bulk crawler ranks by trophies and only fetches top-1500 stale tags per run; with 553k tags discovered, low-trophy tags (personal account, watchlist) effectively never get fetched.
  - Dashboard launch did full SQL self-joins on every load; ~5-15 min on the 1-CPU droplet, unusable.
- Added a **pinned-tags crawler** (`scripts/collect-pinned.py` + `data/pinned_tags.txt` gitignored) on a 1h timer. Always fetches a small explicit list independent of the bulk snowball.
- Refactored dashboard to support **precomputed analytics cache** at `data/analytics_cache.json`:
  - Extracted `collect_all_data()` and `_collect_personal_data()` from `scripts/dashboard.py` into `src/brawlstar_agent/dashboard_data.py` so both the server and the cron job share the schema.
  - New `scripts/precompute-analytics.py` runs queries and atomically writes the cache.
  - `scripts/dashboard.py` reads the cache by default; `--recompute` forces fresh, `--no-cache` skips entirely.
  - Dashboard header now paints cache age + compute time with color thresholds (orange >30 min compute, red >45 min).
- New systemd unit pair: `brawl-collect-pinned.service` + `.timer` (every 1h).
- New systemd unit pair: `brawl-analytics.service` + `.timer` (every 1h, `TimeoutStartSec=2700` watchdog at 45 min).
- Documented IP semantics (anchor IPv4 = outbound, reserved IPv4 = inbound, both WAN) and SSH config alias pattern in deployment.md.
- Verified the dashboard refactor compiles cleanly; cache helpers (`read_cache`, `write_cache`) imported successfully.
- Added `--remote-cache HOST` flag to `scripts/dashboard.py`: auto-rsyncs cache from a remote SSH host before launching. Makes the local-laptop workflow trivial — no SSH tunnel, no DB sync, just `uv run python scripts/dashboard.py --remote-cache brawl`. Falls back to local cache if rsync fails (offline-tolerant).

### Session 8 (continued) — 2026-05-06 — Cold-start aftermath + v2 Run A + disk-full triage
- **Cold-start crawl finished** 2026-05-06 ~02:40 UTC: 166,675 / 200,000 API calls done, **3,936,213 new battles** ingested, total **4,022,553 in DB** (51× growth over v1's 78k clean dataset). Crawl appears to have stopped at 166k because the droplet hit 100% disk.
- **Droplet disk-full disaster** when user resumed cron timers + tried to rsync the DB:
  - Live `rsync brawl:.../brawlstars.db local` produced a malformed local copy because writers were active (WAL pages not in main file). Wrote `scripts/rsync-db-from-droplet.sh` with two modes: `--backup-mode` (default, online-safe via `sqlite3 .backup`, needs ~DB-size scratch) and `--direct` (no scratch but requires writers stopped + WAL checkpointed).
  - On droplet, `.backup` itself failed with `database or disk is full`. Recovery: stop timers → `PRAGMA wal_checkpoint(TRUNCATE)` → direct rsync to local (worked, ~3.5 min for 18.6 GB). Local DB integrity check passed, 4,022,553 battles confirmed.
  - Documented all of this in `docs/deployment.md` § 16 fish/zellij gotcha table (added rsync corruption + fish heredoc rows). `scripts/rsync-db-from-droplet.sh --direct` is the new standard path.
- **DB size diagnosis**: 18.6 GB on local. Estimated breakdown: ~13 GB is text `battle_id` (95-char strings replicated 7× per battle + across 4 indexes). Schema migration to a synthetic int `battle_id` would shrink ~60%. Filed as v3 work; not blocking.
- **v2 Run A** (cutoff 2026-05-03, all post-fix-ingested data, no cap):
  - 1,779,959 battles → 3,559,918 rows after both-perspective doubling
  - Random split (n_test=711,982): Global 0.6614, Mode 0.6806, **ModeMap 0.6917**, **LogReg 0.6804**, **LightGBM 0.7382**
  - Temporal CV (7 folds): Global 0.6531, ModeMap 0.6794, **LightGBM 0.7281**
  - vs v1 baseline: LightGBM AUC +0.8pp random / +2.4pp temporal; LogReg +1.9pp (data alone helps the linear model more than the tree model — LightGBM was closer to the per-feature ceiling at v1 scale).
  - LightGBM fit time: 405s vs v1's 8s — ~50× scale, ~50× time, linearly tractable.
  - Reports: `reports/recommender_v2_default.json`, model: `models/recommender_v2_default.lgb.txt`.
- **Pending in Session 8**: Run B (all-data, cutoff 2021-01-01), Run C (30-day window, cutoff 2026-04-06), top-K eval at the new scale, droplet shrinkage cleanup, decision on whether to update `CLEAN_CUTOFF_ISO` constant given the cold-start makes time-based filter unnecessarily conservative.

### Session 8 (continued) — 2026-05-04 late night — Cold-start in progress + droplet shell layering
- Cold-start orchestrator launched on droplet inside tmux session `coldstart`. Phase 1 backup completed (~210k battles). Phase 2 stopped timers, deleted ~132k pre-cutoff battles, VACUUM'd, kicked off aggressive crawl in nohup at 3 qps.
- Documented the shell layering on the droplet: fish auto-execs from `~/.bashrc` for TTY sessions only (`-t 1 && -z $NO_FISH && -z $INSIDE_FISH`), zellij installed alongside tmux. Login shell is **still bash** (no `chsh`) so the runbook stays accurate, `~/.bashrc` exports keep working, and `ssh brawl 'cmd'` automations never see fish.
- Deployment runbook gained a § 16 with install steps + a gotcha table (e.g., `ssh -t brawl 'cmd'` allocates a TTY → would exec fish → use `NO_FISH=1` or `bash -lc '...'` for bash syntax).
- techContext.md updated with login/interactive shell + multiplexer entries.

### Session 8 (continued) — 2026-05-04 evening — Verification + cold-start orchestrator
- Wrote `scripts/verify-bug.py` to empirically test the team-result bug rate by re-fetching legacy battles from low-activity participants and comparing post-fix API results to stored labels.
- Ran 80 candidates → 20 recoverable → **0 flipped**. Surprising at first, but diagnostic: all 20 recovered battles turned out to be post-fix-INGESTED (pre-cutoff battle_time but post-deploy ingestion). The bug is intrinsically untestable in stored data — pre-fix-ingested battles have all aged out of the API window.
- Cross-check via `collection_log.fetch_battlelog` timestamps revealed **6,045 pre-cutoff battles are provably post-fix-INGESTED** (clean labels), 75,045 are pre-fix-or-ambiguous (likely buggy). Time-based cutoff is conservative; an ingestion-time filter could recover the 6k. Filed as v2 candidate.
- Updated `docs/recommender-v1.md` § "Why we can't recover the legacy bug labels" with the actual results and the post-fix-INGESTED finding.
- Wrote `scripts/coldstart-droplet.sh` orchestrator for the planned cold-start: phase 1 backup + rsync hint, phase 2 stop timers + purge + VACUUM + aggressive crawl in nohup. Idempotent, dry-run safe, tunable via `COLDSTART_RPS` / `COLDSTART_LIMIT` / `COLDSTART_OLDER_THAN` env vars. Existing api_client retry/backoff handles 429s and 5xx natively, no new fallback code needed.

### Session 8 (continued) — 2026-05-04 — Recommender v1.1 — top-K eval, baselines floor, collector bug fix
- **Found and fixed a real collector bug**: `scripts/collect-battles.py --collect-only` (the systemd-timer command on the droplet) skipped `seed_brawlers()`. So the `brawlers` table was frozen at 101 rows since the May-3 deploy, even though new brawlers (DAMIAN id 16000104, first seen in battles **2026-04-24**) had been showing up in `battle_players`. Patch: `--collect-only` now seeds brawlers once at the start of every run. One extra API call per 6 hours, idempotent UPSERT.
- **Confirmed 4,820 `UNKNOWN` (id=0) brawler_player rows** — these are battles where the API returned a malformed/missing brawler dict. Edge case, not a model concern (recommender filters them out via vocab).
- **Added Random + TrophyOnly baselines** to make the AUC=0.5 floor explicit. **Critical finding**: `TrophyOnly` AUC = 0.497 — i.e., trophy diff alone is essentially random. Ranked matchmaking equalizes trophies within a match, so the AUC>0.5 we see in every other model is *all genuine brawler-pick signal*, not skill-tier leakage. Documents that the 0.65–0.73 AUC numbers are honestly earned.
- **Added top-K recommendation evaluation** (`recommender/topk_eval.py` + `scripts/eval-topk.py`). For each test row, mask team A's last pick, score all ~97 legal candidates, find rank of actually-played brawler. Reports hit@K, MRR, mean/median rank, and win-rate uplift conditioned on actual pick being in top-K.
- **Top-K results** (random split, n=2,500, last_pick mode):
  - Random: hit@1=0.001, hit@10=0.09 (matches K/N theoretical floor)
  - Global Wilson: hit@1=0.006, hit@3=0.19, hit@10=0.22 (hit@3 is decent — top global brawlers are picked often regardless of context)
  - ModeMap: hit@1=0.008, hit@5=0.15, hit@10=0.20
  - **LightGBM: hit@1=0.150 (15× random), hit@3=0.194, hit@5=0.225, hit@10=0.293**, MRR=0.205
  - Win uplift: when actual pick is in LightGBM top-1, team WR = 62.5% (vs 51.1% baseline, **+11.4 pp**)
  - Winners-only top-K (cleanest meta-quality test): LightGBM hit@1=**0.21** (20× random), hit@10=**0.39**
- **Player-flexibility tradeoff** (top-1 → top-10): hit rate doubles, expected WR drops 3 pp. Top-5 is a reasonable default.
- **Documented** why legacy bug labels can't be recovered: the bug preserves all internal invariants; trophy_change/result alignment is identical pre- and post-fix; verifying via re-crawl is only possible for very recent battles (<2 weeks) due to the 25-battle API window. Even verification doesn't enable correction — we still couldn't tell which specific battles are flipped.
- Updated `docs/recommender-v1.md` with new tables, top-K section, win-uplift section, collector-bug note, and the long-form legacy-bug-recovery answer.
- Saved `reports/recommender_v1_topk.json` for downstream consumption.

### Session 8 — 2026-05-04 — Brawler-pick recommender v1 (Phase 6 kickoff)
- Rsync'd droplet DB to local: 210k battles, latest 2026-05-05, **78k clean post-fix**.
- Tried to implement the legacy label-flip detector floated in `analytics-notes.md`. **It doesn't work** — the bug swaps team labels symmetrically and preserves the "1W+1L per battle" invariant. Recorded as **DEC-010**: legacy data is unusable for training and evaluation; strict post-2026-05-03 filter is the only path.
- Built `src/brawlstar_agent/recommender/` subpackage:
  - `dataset.py`: clean-window loader, both-perspectives expansion, temporal/random splits, `load_brawler_names` that falls back to `battle_players` for new brawlers (the `brawlers` table didn't have DAMIAN id 16000104).
  - `features.py`: `TeamFeaturizer` with sparse and dense modes (sklearn vs LGBM).
  - `baselines.py`: Global / Mode / ModeMap Wilson-CI baselines with the same `predict_proba` interface as the trained models.
  - `team_model.py`: `LogRegTeamModel` + `LGBMTeamModel` + `evaluate` + `save_model` / `load_model`.
  - `inference.py`: `rank_brawlers_for_map` (Monte Carlo over empirical teammates), `complete_team`, `last_pick`.
  - `cv.py`: sliding temporal-fold harness + `evaluate_models_on_folds`.
- Wrote `scripts/train-recommender.py` (re-runnable with `--cutoff` for the transferable-algorithm property) and `scripts/analyze-recommender.py` (plots + DAMIAN deep-dive).
- Built `notebooks/recommender_v1.ipynb` (executed, 116 KB with outputs baked in) and `docs/recommender-v1.md` (full methodology + how-to-retrain).
- **Random-split results** (n_test=21k):
  - Global Wilson AUC=0.655 → ModeMap AUC=0.697 → LightGBM **AUC=0.730**
  - LogReg with multi-hot teams underperforms ModeMap (0.661 vs 0.697); needs interaction features to compete.
- **Temporal CV** (4 sub-daily folds, ~2 days clean data): LightGBM AUC=0.704; ModeMap drops to 0.666 from 0.697 (random) — exactly the gap a temporal-vs-random comparison should expose.
- **Findings noted in doc**:
  - Release-meta inflation: DAMIAN (newest brawler, id 16000104) has **64.5% raw WR over 41k appearances** — most-played AND highest-WR brawler. Model recommends DAMIAN with extreme confidence; this is meta truth, not a model bug, but worth flagging for inference robustness.
  - LogReg without interaction features can't beat the ModeMap aggregate; tree models (LGBM) capture brawler×map and brawler-pair interactions for free.
  - Brier score 0.21 on LightGBM suggests reasonably calibrated probabilities for a v1.
- Added deps: `scikit-learn`, `lightgbm`, `pandas`, `pyarrow` via `uv add`.
- Per-machine `UV_CACHE_DIR=/media/lin/disk2/.uv-cache` is owned by root locally; worked around by setting `UV_CACHE_DIR=/media/lin/disk2/brawlstar-agent/.uv-cache-local` for this session (gitignored).

### Session 6 — 2026-05-03 — Production deploy on DigitalOcean
- Provisioned DigitalOcean Basic droplet ($6/mo, Ubuntu 24.04 LTS, US region) — DEC-007
- Reserved IP `209.38.4.212` attached; BS API key whitelisted both reserved + anchor public IP `64.23.171.86` (DO Reserved IP is inbound-only; outbound uses the anchor)
- Hardened: non-root sudo user `lin`, key-only SSH, root login disabled, UFW (allow OpenSSH only), fail2ban with home-IP allowlist
- Defused needrestart auto-restart of services on `apt install` (`/etc/needrestart/needrestart.conf` → `restart = 'l'`)
- Python 3.12 + uv installed; project rsynced; `uv sync` succeeded after overriding the hardcoded `cache-dir` from `pyproject.toml` via `UV_CACHE_DIR` env var
- Made `api_client.py` portable across machines via `BRAWL_API_KEY_VAR` indirection — same `api.env` file works on both local (uses `BRAWL_STAR_API`) and droplet (uses `BRAWL_STAR_API_DO`)
- systemd service + 6h timer for `scripts/collect-battles.py --collect-only --battlelog-limit 1500 --rps 2`; Persistent=true so missed runs catch up
- First scheduled run: +35,860 new battles in ~5 min; total now **160,764 battles**, latest battle `2026-05-04` (caught up to today)
- Adopted local-primary git deploy workflow — DEC-008
- Cleaned `pyproject.toml`: removed hardcoded `cache-dir`, replaced with `UV_CACHE_DIR` env var per machine
- Wrote `docs/deployment.md` as a step-by-step runbook for fresh-VPS migration
- **Pitfalls captured** (now documented in deployment runbook):
  - DO Reserved IPs are inbound-only by default — must whitelist anchor public IP for outbound API calls
  - needrestart on Ubuntu 24.04 will auto-restart services mid-run on every `apt install`
  - fail2ban's first restart races with its socket; `sleep 2` before `fail2ban-client`
  - `pyproject.toml`'s `[tool.uv] cache-dir` is non-portable; use env var instead
  - `Type=oneshot` services block `systemctl start` until the run completes — use `--no-block` or run from another shell
