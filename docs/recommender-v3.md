# Brawler-pick recommender v3 — attention/transformer

> Phase 6 v3. Companion: `docs/recommender-v2.md` (read it first). v3 keeps the
> same data window (A_fair: cutoff `2026-05-03T01:00:00Z`) and the same stable
> test boundary (DEC-011: `2026-05-05T00:00:00Z`) as the v2 production candidate
> so AUC, log-loss, and top-K numbers are directly comparable across versions.
> What changes is the model class — from LightGBM on multi-hot features to a
> small transformer encoder over the team / context tokens, with per-brawler
> trophy and power-level features added to the input.

## Why v3 is a different model class, not a different dataset

The v2 fair-run study (DEC-011) gave a clear read on where the lift is:

- LogReg saturates at AUC ~0.68 across all data cutoffs (3-day, 30-day, all-time).
  The flat curve says the linear model has run out of representable interactions
  in its multi-hot feature space.
- LightGBM moves between cutoffs (0.7181 → 0.7265 → 0.7235), but the absolute
  ceiling at the same feature design is around 0.73. Its lift over LogReg comes
  almost entirely from automatically learning brawler×brawler and brawler×map
  interactions.
- Adding more data (3-day → 30-day → 5-yr) gives at most +0.84 pp. That's small
  relative to the gap between LogReg and LightGBM (+3.7 pp), which is itself the
  payoff of a richer interaction representation.

So v3 invests in two changes that should let the model learn *better* interactions
on the *same* training data:

1. **Attention over team tokens** — let the model decide which brawler-pair
   matchups matter most, instead of trees having to discover them via splits.
2. **Per-brawler features** — every brawler now carries its own trophy and
   power-level into the encoder, so the model can downweight a power-3 SHELLY
   inside a 1.87 M-row training set instead of treating it identically to a
   power-11 SHELLY.

## Architecture

```
inputs per battle:
  team_a:        3 brawler ids
  team_b:        3 brawler ids
  team_a/b_trophies:  3 ints each (post-fix data: 0..4951; mean ≈ 850)
  team_a/b_powers:    3 ints each (1..11; max = 11; ~80% of ranked rows are 11)
  mode, map, battle_type    (vocab ids; +1 reserved for UNK/PAD at index 0)
  team_a/b_trophies_mean    (log1p, mean over team)

token sequence (8 tokens):
  [CLS]  [CTX]  [A1] [A2] [A3]  [B1] [B2] [B3]

  CLS    = learnable + side_emb(CLS)
  CTX    = mode_emb + map_emb + btype_emb + side_emb(CTX)
  Ai     = brawler_emb + scalar_proj(trophy_log, power/11) + side_emb(A)
  Bi     = brawler_emb + scalar_proj(trophy_log, power/11) + side_emb(B)

encoder:
  TransformerEncoder(num_layers=3, d_model=96, nhead=4, ff=192,
                     dropout=0.1, norm_first=True, activation=gelu)
  src_key_padding_mask: PAD positions in brawler tokens (id == 0)

head:
  cls_out = LayerNorm(encoder(tokens)[:, 0, :])              # (B, d_model)
  head_in = concat(cls_out, [a_t_log, b_t_log, t_diff_log])   # (B, d_model+3)
  logit   = Linear → GELU → Dropout → Linear(d_model→1)

loss:  BCEWithLogitsLoss
opt:   AdamW(lr=1e-3, wd=1e-4), grad_clip=1.0
sched: CosineAnnealingLR over (epochs × steps_per_epoch)
batch: 4096 train, 16384 eval
```

Total params: ~251k (dominated by the brawler embedding ≈ 102×96 and the encoder
layer params). The model fits comfortably in CPU memory and trains in roughly
3 min/epoch on i7-12700H (14 threads, no GPU). We use `enable_nested_tensor=False`
on the encoder to avoid the warning that comes with `norm_first=True`.

## Per-brawler trophy + power features (the "more feature engineering" part)

Before v3, every team brawler was represented as a single multi-hot bit — a
power-3 SHELLY looked identical to a power-11 SHELLY to the model. v3 changes
two things:

- `dataset.py` now exposes `team_a_trophies`, `team_b_trophies`, `team_a_powers`,
  `team_b_powers` as parallel tuples aligned to the existing `team_a` / `team_b`
  brawler-id tuples (sorted by brawler_id within each team). Backwards-compatible:
  the old `team_a_trophies_mean` / `team_b_trophies_mean` columns and the old
  `team_a` / `team_b` tuple shapes are unchanged.
- The transformer's per-brawler token consumes (`brawler_emb`, `trophy_log`,
  `power/11`) — so the model can scale its prediction by per-slot quality.

Empirical justification: of the ~12 M post-fix battle-player rows in the clean
window, ~80 % are power 11 (max), but the remaining 20 % spans power 0–10
including ~110 k power-1 rows. Trophy distribution is heavy-tailed (min 0, max
4 951, mean 851). Both are real signal that v2's multi-hot ignored.

## Results

All numbers below are on the canonical stable test set
(`stable_test_after = '2026-05-05T00:00:00Z'`, 844 151 battles, 1 688 302 rows).
Train data: 1 871 616 rows (935 808 battles), exactly matching A_fair across
all transformer variants so the only differences are the model class
(Run 0 → 1) and the training plumbing / arch (Run 1 → 2 → 3 → 4).

### Headline ablation (binary win-prediction, stable test, all modes)

| Run | What changed | Arch | Params | Device | Epochs | Train time | **AUC** | LogLoss | Acc | Brier |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| **0** v2 baseline | LightGBM, multi-hot features | tree | n/a | CPU | n/a | 111 s | 0.7181 | 0.6109 | 0.6490 | 0.2124 |
| **1** v3 CPU | + transformer + per-brawler trophy/power | d=96, L=3, ff=192 | 251 k | CPU | 6 | 1651 s | 0.7378 | 0.5879 | 0.6618 | 0.2044 |
| **2** v3 GPU (slow path) | identical code, `.to('cuda')` | d=96, L=3, ff=192 | 251 k | GPU 3060 | 6 | 616 s | 0.7392 | 0.5868 | 0.6633 | 0.2040 |
| **3** v3 GPU (fast path) | preload to VRAM, drop DataLoader | d=96, L=3, ff=192 | 251 k | GPU 3060 | 6 | 338 s | 0.7366 | 0.5890 | 0.6612 | 0.2049 |
| **4** v3 GPU (fast + big) | bigger model + 2 more epochs | d=128, L=4, ff=256, h=8, p=0.15 | 570 k | GPU 3060 | 8 | 858 s | 0.7635 | 0.5616 | 0.6788 | 0.1943 |
| **5** v3 GPU (fast + XL) | even bigger + 4 more epochs + p=0.20 | **d=256, L=6, ff=512, h=8, p=0.20** | **3.28 M** | GPU 3060 | 12 | 3129 s | **0.7674** | **0.5573** | **0.6821** | **0.1928** |

Reading the ablation:

- **0 → 1 (architecture matters)**: same 1.87 M training rows, same DEC-011
  test, swap LightGBM for the small transformer + per-brawler trophy/power
  features. **+1.97 pp AUC**, calibration improves across the board (logloss
  −2.3 pp, Brier −0.8 pp). This is the headline v3-vs-v2 result; the
  architecture change is doing the work, not the data.
- **1 → 2 (GPU compute = 2.7×)**: identical code path (DataLoader-based), just
  `.to('cuda')`. AUC essentially identical (Δ +0.14 pp = seed noise).
  Wall-clock training drops 27.5 → 10.3 min.
- **2 → 3 (data plumbing = +1.8× on top)**: ditch DataLoader entirely; preload
  all ~230 MB of training tensors onto GPU VRAM once, iterate via
  `torch.randperm` + `tensor[idx]` (see `_iter_batches` in
  `transformer_model.py`). AUC again unchanged (Δ −0.26 pp = noise).
  Wall-clock 10.3 → 5.6 min. **Compounded vs CPU: 4.9× speedup.** At this
  model size the per-step GPU compute is microseconds; the win comes from
  removing per-batch CPU↔GPU memcopy overhead, not from doing more math.
- **3 → 4 (architecture lift, attempt 2)**: with epochs costing ~40 s and
  headroom on the 5.77 GB GPU, scale the model up: d_model 96 → 128, layers
  3 → 4, ff 192 → 256, nhead 4 → 8, dropout 0.10 → 0.15, train 8 epochs.
  Param count 251 k → 570 k. **AUC jumps +2.69 pp on top of Run 3.** The
  small transformer was undersized; the GPU plumbing fix is what made trying
  a 2.3× bigger model cheap (14 min training instead of ~75 min projected on
  CPU).
- **4 → 5 (scale further)**: same procedure, bigger again. d_model 128 → 256,
  layers 4 → 6, ff 256 → 512, dropout 0.15 → 0.20 (slightly more reg for the
  bigger model), train 12 epochs (vs 8). Param count 570 k → 3.28 M (5.7×
  Run 4, 13× Run 1). **AUC gains +0.39 pp** on top of Run 4 — wins 9/9 modes
  vs Run 4, but **strong diminishing returns** vs the +2.69 pp Run 3→4 jump.
  Brier still improves (0.1943 → 0.1928, best calibration so far). val_auc
  flatlined at epoch 11-12 (0.7765 → 0.7764), so going to 16+ epochs at the
  same arch likely won't help; capacity is approaching a real ceiling at this
  data scale.

**End-to-end Run 0 (v2 prod) → Run 5 (v3 XL):**
**+4.93 pp AUC** (0.7181 → 0.7674), **−5.36 pp logloss** (0.6109 → 0.5573),
**+3.31 pp accuracy** (0.6490 → 0.6821), **−1.96 pp Brier** (0.2124 → 0.1928),
all on the same 1.87 M training rows and the same 844 k stable-test battles.
No new data was added; the lift is entirely from the architecture (attention +
per-brawler features) and from being able to actually use a model big enough
to express it.

**Capacity-vs-AUC scaling at this data size:**

| Params | AUC | Δ AUC vs prev | Δ AUC per 10× params |
|---:|---:|---:|---:|
| n/a (LGBM) | 0.7181 | — | — |
| 251 k | 0.7378 | +1.97 pp | — |
| 570 k | 0.7635 | +2.57 pp | ≈ +6.3 pp |
| 3.28 M | 0.7674 | +0.39 pp | ≈ +0.5 pp |

The jump from "tiny" to "small-but-right-shape" gave us most of the gain.
Going from "small" to "big" doubled it. Going from "big" to "XL" returned a
fifth as much per ×10 params. Looks like capacity has nearly saturated against
this data scale — pushing past 3 M params probably needs more data or a
different inductive bias (factorization machine, multi-task pick-prediction
head, etc.) rather than just bigger transformers.

### Per-mode AUC, stable test (Run 0 vs Run 4 (big) vs Run 5 (XL))

| Mode (n in test) | A_fair LGBM | v3 big | v3 XL | Δ XL−big | Δ XL−LGBM |
|---|---:|---:|---:|---:|---:|
| brawlBall (847 k) | 0.7517 | 0.7948 | **0.7961** | +0.13 | **+4.44 pp** |
| siege (30 k)      | 0.7933 | 0.8268 | **0.8329** | +0.61 | **+3.96 pp** |
| basketBrawl (45 k)| 0.7273 | 0.7667 | **0.7777** | +1.10 | **+5.04 pp** |
| **knockout (363 k)** | 0.6942 | 0.7626 | **0.7700** | +0.74 | **+7.58 pp** |
| wipeout (2 k)     | 0.7047 | 0.7347 | **0.7409** | +0.62 | **+3.62 pp** |
| heist (94 k)      | 0.6784 | 0.7189 | **0.7233** | +0.44 | **+4.49 pp** |
| gemGrab (125 k)   | 0.6667 | 0.6896 | **0.6971** | +0.75 | **+3.04 pp** |
| hotZone (89 k)    | 0.6551 | 0.6758 | **0.6813** | +0.55 | **+2.62 pp** |
| bounty (93 k)     | 0.6159 | 0.6362 | **0.6448** | +0.86 | **+2.89 pp** |

**v3 XL wins 9/9 modes vs v3 big and 9/9 modes vs A_fair LightGBM.** Largest
absolute gain over LightGBM is still **knockout (+7.58 pp)** — 1-life
elimination per round makes brawler×brawler matchup signal dominant, and
attention captures it natively while the trees had to discover it via splits
on multi-hot features.

The big → XL gains are evenly spread (every mode +0.1 to +1.1 pp) with the
biggest movers in noisier modes: **basketBrawl +1.10**, **bounty +0.86**,
**knockout +0.74**, **gemGrab +0.75**. The strongest modes (siege, brawlBall)
already had high AUC and gained the least, consistent with capacity hitting a
ceiling on signal that's already extracted.

bounty stays the hardest mode (test AUC 0.6448) — the team-composition signal
is genuinely weakest there because of many short rounds and gear-driven
turnover. Even so, +2.9 pp on the floor over LightGBM is real progress.

### Top-K (last_pick mode, n=5000 sample, stable test)

All three "production-grade" models are evaluated on the same 5000 randomly
sampled test rows (seed = 42), the same candidate pool of ~97 brawlers per
row, and the same masked-last-pick task. v3-small and v3-big are loaded from
disk inside `scripts/eval-topk.py` via `--transformer-from`; LightGBM is
retrained inline. So everything except the model is identical.

**Candidate pool**: the model's vocabulary contains **102 brawlers** — every
brawler that has appeared in a `ranked` or `soloRanked` battle in the post-fix
window. The official `brawlers` table has 104 entries; the 2 missing from our
training data (and from the test set) are the newest releases **BOLT
(id 16000106)** and **STARR NOVA (id 16000105)**, which haven't been picked in
ranked play in our data window. So they're absent from both train and test —
no test rows are silently dropped because the actual brawler is unknown to the
model. The reported `avg legal candidates per row: 97` comes from
`102 vocab − 6 in-battle + 1 (the actual brawler added back) ≈ 97`.

All test rows:

| Model | hit@1 | hit@3 | hit@5 | hit@10 | MRR | mean rank | WR\|in_top1 (Δ vs 49.5 %) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random      | 0.004 | 0.024 | 0.050 | 0.099 | 0.047 | 49.3 | 47.4 % (−2.2 pp) |
| TrophyOnly  | 0.000 | 0.002 | 0.003 | 0.021 | 0.022 | 63.6 | (n/a) |
| Global      | 0.005 | 0.014 | 0.154 | 0.212 | 0.073 | 40.3 | 58.3 % (+8.8 pp) |
| ModeMap     | 0.048 | 0.132 | 0.163 | 0.247 | 0.130 | 33.6 | 69.6 % (+20.0 pp) |
| A_fair LGBM (v2 prod) | 0.136 | 0.180 | 0.210 | 0.283 | 0.193 | 34.2 | 68.5 % (+18.9 pp) |
| v3 small (Run 1) | 0.136 | 0.193 | **0.226** | 0.280 | **0.196** | 35.9 | 68.2 % (+18.6 pp) |
| v3 big (Run 4)   | 0.137 | 0.185 | 0.213 | 0.285 | 0.195 | 34.7 | **69.3 % (+19.7 pp)** |
| **v3 XL (Run 5)** | **0.136** | 0.187 | 0.219 | **0.287** | 0.195 | 35.1 | 68.2 % (+18.6 pp) |

Winners-only (test rows where team A actually won — cleanest meta-quality test):

| Model | hit@1 | hit@3 | hit@5 | hit@10 | MRR |
|---|---:|---:|---:|---:|---:|
| Random      | 0.003 | 0.020 | 0.048 | 0.093 | 0.046 |
| ModeMap     | 0.074 | 0.199 | 0.239 | 0.324 | 0.175 |
| A_fair LGBM | 0.195 | 0.247 | 0.283 | 0.357 | 0.256 |
| v3 small    | 0.204 | **0.265** | 0.299 | 0.357 | 0.265 |
| v3 big      | **0.204** | 0.263 | **0.300** | 0.364 | **0.267** |
| **v3 XL**   | 0.198 | 0.260 | 0.298 | **0.369** | 0.262 |

Reading these tables:

- **The big → XL +0.39 pp binary AUC gain does NOT translate into top-K either.**
  hit@1 actually tied or slightly *regressed* (0.137 → 0.136 on all rows;
  0.204 → 0.198 winners-only). hit@10 ticked up marginally on both rows-
  variants (+0.2 pp / +0.5 pp). MRR essentially flat. WR|in_top1 dropped
  slightly (0.693 → 0.682) — within sample noise for n=5000.
- **What the bigger models DO improve is calibration**: Brier 0.2049 (small) →
  0.1943 (big) → 0.1928 (XL). Logloss 0.5890 → 0.5616 → 0.5573. Each step
  produces sharper, better-calibrated win-probabilities even when the
  *ordering* of candidates within a single (team_partial, opp) context barely
  changes.
- **The structural top-K ceiling looks real** — small / big / XL all sit at
  hit@1 ≈ 0.14 and MRR ≈ 0.20 within sample noise. The "predict which of ~97
  brawlers a player picks" task is bounded by *player roster + personal
  preference*, not by win-probability quality. A perfect win-prob oracle would
  recommend the brawler that maximizes win rate; the player may pick a
  different brawler because they don't own the optimal one or because they
  trust their muscle memory on something else. That gap caps hit@K well below
  1.0 regardless of model capacity.
- **Bottom line for downstream consumers**:
  - Want calibrated probabilities (e.g., "auto-recommend only when P(win) > 0.7")?
    Use **v3 XL** (lowest Brier, lowest logloss, but 5.7× the inference cost
    of big and 13× of small).
  - Want a sensible recommendation list (top-K UI)?
    **v3 big** is the best practical pick — the top-K is identical to XL but
    inference is much cheaper.
  - Want fastest possible inference with a still-strong model?
    **v3 small** has the best hit@5 of all three (0.226), is small enough to
    serve from CPU, and gives up only ~2.5 pp of binary AUC vs XL.

### Training history

Per-epoch validation AUC on a 5 % internal split of the training data (the test
set itself is held out completely, never touched during training).

**Run 5 — XL, 12 epochs on GPU (3.28 M params):**

| Epoch | train_loss | val_loss | val_auc | elapsed |
|---:|---:|---:|---:|---:|
| 1  | 0.6379 | 0.5982 | 0.7251 | 204 s |
| 2  | 0.5961 | 0.5909 | 0.7407 | 230 s |
| 3  | 0.5857 | 0.5756 | 0.7515 | 244 s |
| 4  | 0.5777 | 0.5733 | 0.7571 | 255 s |
| 5  | 0.5710 | 0.5640 | 0.7643 | 259 s |
| 6  | 0.5660 | 0.5587 | 0.7671 | 260 s |
| 7  | 0.5619 | 0.5567 | 0.7698 | 262 s |
| 8  | 0.5586 | 0.5540 | 0.7719 | 261 s |
| 9  | 0.5557 | 0.5516 | 0.7745 | 263 s |
| 10 | 0.5532 | 0.5490 | 0.7757 | 262 s |
| 11 | 0.5518 | 0.5481 | 0.7765 | 263 s |
| 12 | 0.5510 | 0.5483 | 0.7764 | 264 s |

**Run 4 — big, 8 epochs on GPU (570 k params):**

| Epoch | train_loss | val_loss | val_auc | elapsed |
|---:|---:|---:|---:|---:|
| 1 | 0.6351 | 0.6021 | 0.7172 | 83 s |
| 2 | 0.5980 | 0.6014 | 0.7361 | 90 s |
| 3 | 0.5846 | 0.5803 | 0.7503 | 93 s |
| 4 | 0.5739 | 0.5671 | 0.7614 | 96 s |
| 5 | 0.5669 | 0.5587 | 0.7676 | 97 s |
| 6 | 0.5626 | 0.5546 | 0.7698 | 98 s |
| 7 | 0.5597 | 0.5535 | 0.7710 | 99 s |
| 8 | 0.5586 | 0.5533 | 0.7711 | 100 s |

**Run 1 — small, 6 epochs on CPU (251 k params):**

| Epoch | train_loss | val_loss | val_auc | elapsed (CPU) |
|---:|---:|---:|---:|---:|
| 1 | 0.6317 | 0.6129 | 0.7183 | 248 s |
| 2 | 0.5998 | 0.5911 | 0.7328 | 257 s |
| 3 | 0.5925 | 0.5875 | 0.7378 | 258 s |
| 4 | 0.5876 | 0.5830 | 0.7427 | 260 s |
| 5 | 0.5843 | 0.5805 | 0.7455 | 261 s |
| 6 | 0.5828 | 0.5796 | 0.7458 | 263 s |

Three observations across all three runs:

1. **`train_loss > val_loss` consistently for every epoch.** This is the
   *opposite* of the classic overfitting signature. Two compounding causes:
   (a) dropout is active at training time but not during validation, so the
   train forward pass has additive noise that BCE penalizes; (b) `train_loss`
   is a running average over the epoch and uses the model state that's behind
   in time, while `val_loss` is computed once at the end of the epoch with
   the freshest weights. The implication is that we still have *capacity
   headroom* — none of the runs are memorizing training data.
2. **Convergence behavior matches model size**, as you'd expect:
   - Run 1 (small) plateaued at ~0.7458 val AUC (Δ epoch 5→6 = +0.03 pp);
   - Run 4 (big) plateaued at ~0.7711 (Δ epoch 7→8 = +0.01 pp);
   - Run 5 (XL) plateaued at ~0.7765 (Δ epoch 11→12 = −0.01 pp).
   Each scale-up unlocks ~3 pp of additional val AUC headroom and converges
   at a higher value, but the jumps are getting smaller (Δ small→big +2.53 pp,
   big→XL +0.54 pp on val_auc).
3. **Temporal-holdout tax** (best val AUC − stable-test AUC):
   Run 1: 0.7458 − 0.7378 = 0.80 pp drop.
   Run 4: 0.7711 − 0.7635 = 0.76 pp drop.
   Run 5: 0.7765 − 0.7674 = 0.91 pp drop.
   Bigger models pay roughly the same temporal-holdout tax (~0.8 pp), much
   smaller than the LightGBM random→stable-test drop of ~2 pp. The transformer
   transfers across time better than the trees did.

## How to retrain

```bash
cd /media/lin/disk2/brawlstar-agent
export UV_CACHE_DIR=/media/lin/disk2/brawlstar-agent/.uv-cache-local

# Run 5 — XL arch on GPU. Best AUC + Brier; ~52 min training. Pick this if you
# care about calibrated probabilities. Requires CUDA + nvidia-modprobe.
PYTHONUNBUFFERED=1 PYTHONPATH=src uv run python scripts/train-recommender-v3.py \
    --cutoff 2026-05-03T01:00:00Z \
    --stable-test-after 2026-05-05T00:00:00Z \
    --epochs 12 \
    --batch-size 4096 \
    --d-model 256 --num-layers 6 --ff 512 --nhead 8 --dropout 0.20 \
    --early-stop-patience 4 \
    --device cuda \
    --save-to models/recommender_v3_xl \
    --report-to reports/recommender_v3_xl.json

# Run 4 — big arch on GPU. Sweet spot for inference cost; ~14 min training.
# Top-K is identical to XL within sample noise; AUC is 0.39 pp lower.
PYTHONUNBUFFERED=1 PYTHONPATH=src uv run python scripts/train-recommender-v3.py \
    --cutoff 2026-05-03T01:00:00Z \
    --stable-test-after 2026-05-05T00:00:00Z \
    --epochs 8 \
    --batch-size 4096 \
    --d-model 128 --num-layers 4 --ff 256 --nhead 8 --dropout 0.15 \
    --early-stop-patience 3 \
    --device cuda \
    --save-to models/recommender_v3_big \
    --report-to reports/recommender_v3_big.json

# Run 1 — small arch. Cheapest model, best for CPU-only deploys.
PYTHONUNBUFFERED=1 PYTHONPATH=src uv run python scripts/train-recommender-v3.py \
    --cutoff 2026-05-03T01:00:00Z \
    --stable-test-after 2026-05-05T00:00:00Z \
    --epochs 6 \
    --batch-size 4096 \
    --d-model 96 --num-layers 3 --ff 192 --dropout 0.1 \
    --early-stop-patience 2 \
    --device cuda \
    --save-to models/recommender_v3_default \
    --report-to reports/recommender_v3_default.json

# Top-K + win uplift on the same stable test set (compares LightGBM + transformer).
# Swap --transformer-from to point at any of the v3 checkpoints.
PYTHONUNBUFFERED=1 PYTHONPATH=src uv run python scripts/eval-topk.py \
    --cutoff 2026-05-03T01:00:00Z \
    --stable-test-after 2026-05-05T00:00:00Z \
    --transformer-from models/recommender_v3_xl \
    --output reports/recommender_v3_xl_topk.json \
    --sample-size 5000

# Transformer-only top-K (skip ~2 min LightGBM retrain when you only need v3 numbers):
PYTHONUNBUFFERED=1 PYTHONPATH=src uv run python scripts/eval-topk.py \
    --cutoff 2026-05-03T01:00:00Z \
    --stable-test-after 2026-05-05T00:00:00Z \
    --transformer-from models/recommender_v3_xl \
    --skip-lgbm-train \
    --output reports/recommender_v3_xl_topk.json \
    --sample-size 5000
```

`--transformer-from` makes `scripts/eval-topk.py` load the saved transformer and
add it to the side-by-side comparison alongside `LightGBM` and the heuristic
baselines, with the same candidate pool and sample so hit@K is apples-to-apples.

## Engineering notes

- **GPU enablement**: host is a Lenovo Legion 5 with an RTX 3060 Mobile
  (5.77 GB VRAM, compute 8.6, NVIDIA driver 535.230.02 → CUDA 12.2 max API).
  Two install gotchas:
  1. `nvidia-modprobe` is required to create `/dev/nvidia*` device nodes
     on CUDA init. The Ubuntu nvidia-driver-535 metapackage doesn't pull it
     in; install separately: `sudo apt install nvidia-modprobe`.
  2. If installed from inside Cursor's user-namespace sandbox, dpkg may write
     the binary as owned by `nobody:nogroup` (UID 65534 outside the
     namespace). The setuid bits then drop privileges to `nobody` instead of
     `root` and `/dev/nvidia*` are never created. Fix: install from a real
     terminal, OR `sudo chown root:root /usr/bin/nvidia-modprobe` after the
     fact.
- **PyTorch install**: `uv add torch --index https://download.pytorch.org/whl/cu121`,
  with the index registered as `name = "pytorch-cu121"`, `explicit = true`, and
  routed via `[tool.uv.sources] torch = { index = "pytorch-cu121" }`. This
  scopes the PyTorch index to torch only, so opencv / jupyter / etc. still
  resolve from PyPI. Without `explicit = true` uv treats the PyTorch index as
  primary and fails on every package not mirrored there. Pinned to
  `torch>=2.5.0,<2.6` because cu121 wheels stopped at 2.5.1 (PyTorch went
  cu124-only from 2.6, and cu124 needs driver ≥550 which we don't have). Total
  install size ~3 GB (torch + cudnn 9.1 + cublas + cufft + nccl + triton).
- **Fast data path** (Run 3+): the original DataLoader-based path was actually
  slower per-step on GPU than CPU because per-batch CPU↔GPU memcopy dwarfed
  the GPU compute on this small model. `_iter_batches` in
  `transformer_model.py` preloads the full ~230 MB of training tensors into
  VRAM once and iterates with `torch.randperm` + `tensor[idx]`, eliminating
  every per-step transfer. Drops epoch wall time from ~85 s (DataLoader on GPU)
  to ~40 s (small arch) / ~100 s (big arch) on the same RTX 3060. The Linux
  CPU path (no DataLoader workers either) is slightly slower but still works
  unchanged.
- **Backwards compat**: existing v1/v2 scripts still work — `dataset.py`'s new
  per-brawler tuple columns are additive only.
- **Model file format**: `models/recommender_v3_big.pt` is the torch state
  dict; `models/recommender_v3_big.meta.json` carries vocab + arch hyperparams
  needed to reconstruct the model (cuda or cpu). Use
  `from brawlstar_agent.recommender.transformer_model import load_transformer`.
  `load_transformer` does `torch.load(..., map_location="cpu")`, so the saved
  model loads on machines without a GPU; move to cuda after loading if needed.

## Files

- `src/brawlstar_agent/recommender/transformer_model.py` — `TransformerTeamModel`
  + `_TransformerCore` + `_iter_batches` (GPU-native fast path) +
  `save_transformer` / `load_transformer`
- `src/brawlstar_agent/recommender/dataset.py` — extended to expose per-brawler
  trophy + power columns (backwards compatible)
- `scripts/train-recommender-v3.py` — v3 training CLI mirroring v2
- `scripts/eval-topk.py` — gained `--transformer-from PATH` to load v3 models;
  `--skip-lgbm-train` for transformer-only sub-runs
- `models/recommender_v3_default.pt` + `.meta.json` — Run 1, small arch, CPU baseline
- `models/recommender_v3_gpu.pt` + `.meta.json` — Run 2, small arch, GPU slow path
- `models/recommender_v3_gpu_fast.pt` + `.meta.json` — Run 3, small arch, GPU fast path
- `models/recommender_v3_big.pt` + `.meta.json` — Run 4, big arch on GPU; **practical production candidate** (best AUC/inference-cost ratio, top-K identical to XL)
- `models/recommender_v3_xl.pt` + `.meta.json` — **Run 5, XL arch on GPU; best calibration** (lowest Brier + logloss). Pick this only when you need calibrated probabilities to threshold on.
- `reports/recommender_v3_default.json` / `recommender_v3_gpu.json` /
  `recommender_v3_gpu_fast.json` / `recommender_v3_big.json` /
  `recommender_v3_xl.json` — binary metrics + per-mode + per-epoch history
  for each ablation row
- `reports/recommender_v3_topk.json` — top-K for the small arch
- `reports/recommender_v3_big_topk.json` — top-K + winners-only for the big arch
- `reports/recommender_v3_xl_topk.json` — top-K + winners-only for the XL arch
- `pyproject.toml` — `torch>=2.5.0,<2.6`, `[tool.uv.sources]` routing torch through
  the named `pytorch-cu121` explicit index; `pytorch-cpu` index also defined for
  CPU-only deploys (e.g. droplet)

## See also

- `docs/recommender-v2.md` — DEC-011 stable-test methodology, A_fair / C_fair /
  B_fair baselines, why we did not ship 30-day rolling
- `docs/recommender-v1.md` — original methodology + inference walkthrough
  (`rank_brawlers_for_map` / `complete_team` / `last_pick`); DAMIAN release-meta
  caveat (still applies in v3 — newest brawler tends to dominate predictions)
- `memory-bank/decisions.md` — DEC-010 (legacy bug unrecoverable), DEC-011
  (stable test boundary mandatory)
