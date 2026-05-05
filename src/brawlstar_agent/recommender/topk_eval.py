"""Top-K recommendation evaluation.

The team-completion model's `predict_proba` outputs a binary win prob, but the
USER question is "give me the top-K brawlers ranked by goodness for this slot."
That's a ranking problem with different metrics than AUC/log-loss.

Evaluation framing — for each completed battle in the test set we pretend we
need to pick ONE brawler and have the model rank candidates:

- **last_pick mode**: assume team A's third pick is unknown. Mask it. The model
  scores all candidate Z's against (A's two picks, B's three picks). We record
  where the actually-played Z lands in the model's ranking.

- **random_position mode**: same idea but mask any one of A's three picks at
  random (more general; simulates mid-draft as well as last-pick).

Metrics we report:

- **hit@K** for K ∈ {1, 3, 5, 10}: P(actual brawler ranks in model's top-K
  among legal candidates). The natural floor is K / N_legal_candidates ~~ K/100.
- **MRR** (mean reciprocal rank): 1/rank averaged. Rewards getting the right
  answer at #1; partially credits getting it lower.
- **mean_rank**, **median_rank**: where the actual brawler lands. Smaller = better.
- **win_uplift_topK**: among games where the played brawler IS in our top-K,
  what was the actual win rate? Compared to baseline win rate. Tells us
  "if we constrained ourselves to model's top-K, how much higher would WR be?"
- **only_winners** variant: filter to test rows where the actual team WON. The
  played brawler was at least good enough to win, so its model rank is more
  meaningful as a quality signal.

Note: this is *evaluation*, not training. The training objective stays "predict
team_a_wins"; ranking is a downstream consequence of the calibrated win-prob.
A model that predicts probabilities well will also rank well, in expectation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TopKResult:
    n_evaluated: int
    hit_at: dict[int, float]       # {k: hit_rate}
    mrr: float                     # mean reciprocal rank
    mean_rank: float
    median_rank: float
    win_rate_overall: float        # baseline: win-rate of actual played brawler in eval set
    win_uplift_topk: dict[int, float]  # {k: WR among rows where actual ∈ top-K}
    n_legal_candidates_mean: float  # how many candidates we ranked among


def _all_brawler_ids(model) -> list[int]:
    f = getattr(model, "featurizer", None)
    if f is not None and getattr(f, "brawler_to_idx", None):
        return sorted(f.brawler_to_idx.keys())
    return []


def evaluate_topk(
    model,
    test_df: pd.DataFrame,
    *,
    mode: str = "last_pick",
    sample_size: int | None = 4000,
    only_winners: bool = False,
    seed: int = 42,
    candidate_pool: list[int] | None = None,
    verbose: bool = False,
) -> TopKResult:
    """Score a model's brawler rankings on `test_df`.

    Args:
        mode: 'last_pick' masks team A's last brawler (alphabetical ordering).
              'random' masks a random one of team A's three brawlers.
        sample_size: cap test rows (full eval can be slow; ~4k rows is plenty).
        only_winners: restrict to rows where team_a_wins=1.
        candidate_pool: vocabulary of brawlers to rank from. Defaults to model's
                         featurizer vocabulary.
    """
    rng = np.random.default_rng(seed)

    if candidate_pool is None:
        candidate_pool = _all_brawler_ids(model)
    cand = np.array(sorted(set(int(x) for x in candidate_pool)), dtype=np.int64)
    if len(cand) == 0:
        raise ValueError("Empty candidate pool")

    df = test_df
    if only_winners:
        df = df[df["team_a_wins"] == 1].copy()
    df = df.dropna(subset=["mode", "map", "battle_type"])
    if sample_size is not None and len(df) > sample_size:
        df = df.sample(sample_size, random_state=seed)
    df = df.reset_index(drop=True)

    n = len(df)
    if n == 0:
        raise ValueError("No rows to evaluate after filtering")

    # For each test row, build N_candidate scoring queries: replace the masked
    # slot with each candidate, compute P(win), rank, find actual brawler's rank.
    rows: list[dict] = []
    masked_targets = np.empty(n, dtype=np.int64)
    legal_counts = np.empty(n, dtype=np.int32)

    for i in range(n):
        r = df.iloc[i]
        team_a = list(r["team_a"])
        team_b = list(r["team_b"])
        if mode == "last_pick":
            mask_idx = len(team_a) - 1
        elif mode == "random":
            mask_idx = int(rng.integers(0, len(team_a)))
        else:
            raise ValueError(f"unknown mode {mode}")
        actual = int(team_a[mask_idx])
        partial = team_a[:mask_idx] + team_a[mask_idx + 1:]
        masked_targets[i] = actual
        used = set(partial) | set(team_b)
        legal_mask = ~np.isin(cand, list(used))
        # Ensure the actual brawler is in candidate pool (for ranking it).
        if actual not in cand:
            # Skip; we can't rank an unknown brawler
            masked_targets[i] = -1
            legal_counts[i] = 0
            continue
        legal_for_row = cand[legal_mask | (cand == actual)]
        legal_counts[i] = len(legal_for_row)
        for c in legal_for_row:
            full_a = tuple(sorted(partial + [int(c)]))
            rows.append({
                "row_id": i,
                "candidate": int(c),
                "battle_id": f"EVAL_{i}_{c}",
                "battle_time_iso": r["battle_time_iso"],
                "mode": r["mode"],
                "map": r["map"],
                "battle_type": r["battle_type"],
                "team_a": full_a,
                "team_b": tuple(sorted(team_b)),
                "team_a_wins": 0,
                "team_a_trophies_mean": float(r["team_a_trophies_mean"]),
                "team_b_trophies_mean": float(r["team_b_trophies_mean"]),
            })

    if not rows:
        raise ValueError("No legal candidates produced")

    big = pd.DataFrame(rows)
    if verbose:
        print(f"  scoring {len(big):,} (row × candidate) combinations...")
    proba = model.predict_proba(big.drop(columns=["row_id", "candidate"]))
    big["proba"] = proba

    # Per-row rank of the actual brawler.
    ranks = np.full(n, -1, dtype=np.int32)
    for i, sub in big.groupby("row_id", sort=False):
        actual = int(masked_targets[i])
        if actual < 0:
            continue
        # Rank in descending P(win): 1 = best
        sub_sorted = sub.sort_values("proba", ascending=False)
        order = sub_sorted["candidate"].tolist()
        try:
            r = order.index(actual) + 1
        except ValueError:
            r = -1
        ranks[i] = r

    valid = ranks > 0
    valid_ranks = ranks[valid]
    n_eval = int(valid.sum())
    if n_eval == 0:
        raise ValueError("No valid ranks (all targets unknown to model?)")

    ks = (1, 3, 5, 10)
    hit_at: dict[int, float] = {}
    win_uplift_topk: dict[int, float] = {}
    df_winsubset = df[valid].copy()
    for k in ks:
        hits = (valid_ranks <= k)
        hit_at[k] = float(hits.mean())
        # Win uplift: among rows where actual brawler IS in top-K, what's actual WR?
        wins_when_in = df_winsubset.loc[hits.tolist(), "team_a_wins"].values
        if len(wins_when_in):
            win_uplift_topk[k] = float(np.mean(wins_when_in))
        else:
            win_uplift_topk[k] = float("nan")

    return TopKResult(
        n_evaluated=n_eval,
        hit_at=hit_at,
        mrr=float(np.mean(1.0 / valid_ranks)),
        mean_rank=float(np.mean(valid_ranks)),
        median_rank=float(np.median(valid_ranks)),
        win_rate_overall=float(df_winsubset["team_a_wins"].mean()),
        win_uplift_topk=win_uplift_topk,
        n_legal_candidates_mean=float(legal_counts[valid].mean()),
    )


def random_baseline_topk(n_legal: float, k: int) -> float:
    """Hit@k for a random ranker over n_legal candidates."""
    return min(1.0, k / max(n_legal, 1))


def format_result(res: TopKResult, label: str = "") -> str:
    """Pretty-print a TopKResult."""
    lines = [f"  [{label}]" if label else "  [topK eval]"]
    lines.append(f"    n_evaluated: {res.n_evaluated}    "
                 f"avg legal candidates per row: {res.n_legal_candidates_mean:.0f}")
    lines.append(f"    Mean rank: {res.mean_rank:.1f}    "
                 f"Median rank: {res.median_rank:.0f}    MRR: {res.mrr:.4f}")
    parts = [f"hit@{k}={res.hit_at[k]:.4f}" for k in (1, 3, 5, 10)]
    lines.append("    " + "    ".join(parts))
    if res.n_legal_candidates_mean > 0:
        floors = [f"random@{k}={random_baseline_topk(res.n_legal_candidates_mean, k):.4f}"
                  for k in (1, 3, 5, 10)]
        lines.append("    " + "    ".join(floors))
    base = res.win_rate_overall
    parts2 = []
    for k in (1, 3, 5, 10):
        wu = res.win_uplift_topk.get(k, float("nan"))
        delta = wu - base if not np.isnan(wu) else float("nan")
        parts2.append(f"WR|in_top{k}={wu:.3f} (Δ {delta:+.3f})")
    lines.append("    " + "    ".join(parts2))
    return "\n".join(lines)
