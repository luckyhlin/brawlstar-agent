#!/usr/bin/env python3
"""Top-K recommendation evaluation against a fixed temporal test holdout.

Answers the question: "If you ask the model to rank candidate brawlers for the
last pick, where does the actually-played brawler land?"

Reports hit@K, MRR, mean/median rank, and win-rate uplift for picks where the
played brawler IS in the model's top-K. Compares LightGBM to all heuristics
including a true random baseline and a trophy-only baseline that exposes how
much of the AUC is just "high-trophy team usually wins low-trophy team".

The test set is the same `--stable-test-after` boundary used by
`scripts/train-recommender.py`, so binary AUC and top-K numbers come from the
same held-out battles across all v2 runs (DEC-011).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from brawlstar_agent.recommender import (  # noqa: E402
    CLEAN_CUTOFF_ISO,
    load_clean_battles,
    split_random,
)
from brawlstar_agent.recommender.baselines import (  # noqa: E402
    GlobalWilsonBaseline,
    ModeMapWilsonBaseline,
    RandomBaseline,
    TrophyOnlyBaseline,
)
from brawlstar_agent.recommender.team_model import (  # noqa: E402
    LGBMTeamModel,
    evaluate,
)
from brawlstar_agent.recommender.topk_eval import (  # noqa: E402
    evaluate_topk,
    format_result,
)

STABLE_TEST_AFTER_DEFAULT = "2026-05-05T00:00:00Z"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cutoff", default=CLEAN_CUTOFF_ISO,
                    help=f"Earliest battle to include in the train portion (default: {CLEAN_CUTOFF_ISO})")
    ap.add_argument("--stable-test-after", default=STABLE_TEST_AFTER_DEFAULT,
                    help=("Test set is battles with battle_time_iso >= this. "
                          f"Default: {STABLE_TEST_AFTER_DEFAULT}"))
    ap.add_argument("--output", default=str(REPO / "reports" / "recommender_v2_topk.json"),
                    help="JSON output path")
    ap.add_argument("--sample-size", type=int, default=2500,
                    help="Random sample of test rows for top-K evaluation (default: 2500)")
    ap.add_argument("--transformer-from", default=None,
                    help=("Optional path prefix of a saved TransformerTeamModel "
                          "(e.g. 'models/recommender_v3_default'). When set, the "
                          "v3 transformer is added to the comparison without re-training. "
                          "Skip retraining LightGBM by also passing --skip-lgbm-train if "
                          "you only want a transformer-vs-baselines run."))
    ap.add_argument("--skip-lgbm-train", action="store_true",
                    help="Skip training LightGBM (saves ~2 min). Useful for transformer-only runs.")
    args = ap.parse_args()

    out_path = Path(args.output)
    out: dict = {
        "cutoff": args.cutoff,
        "stable_test_after": args.stable_test_after,
        "sample_size": args.sample_size,
        "binary": {},
        "topk_all": {},
        "topk_winners_only": {},
    }
    print(f"Loading data with cutoff={args.cutoff}, stable_test_after={args.stable_test_after}...")
    df = load_clean_battles(after=args.cutoff).dropna(subset=["mode", "map", "battle_type"])
    train_mask = df["battle_time_iso"] < args.stable_test_after
    train = df[train_mask].copy()
    test = df[~train_mask].copy()
    if train.empty or test.empty:
        raise SystemExit(
            f"stable-test split produced empty side: train={len(train)} test={len(test)}"
        )
    print(f"  train={len(train):,} ({train['battle_id'].nunique():,} battles, "
          f"{train['battle_time_iso'].min()} → {train['battle_time_iso'].max()})")
    print(f"  test ={len(test):,} ({test['battle_id'].nunique():,} battles, "
          f"{test['battle_time_iso'].min()} → {test['battle_time_iso'].max()})")
    out["n_train_rows"] = int(len(train))
    out["n_test_rows"] = int(len(test))

    factories = {
        "Random":      lambda: RandomBaseline(),
        "TrophyOnly":  lambda: TrophyOnlyBaseline(),
        "Global":      lambda: GlobalWilsonBaseline(),
        "ModeMap":     lambda: ModeMapWilsonBaseline(),
    }
    if not args.skip_lgbm_train:
        factories["LightGBM"] = lambda: LGBMTeamModel(
            n_estimators=600, num_leaves=63,
            learning_rate=0.05, min_data_in_leaf=80,
        )

    print("\n=== Binary win-prediction (stable-test split) ===")
    print(f"{'Model':12s}  {'AUC':>7s}  {'logloss':>7s}  {'acc':>7s}  {'brier':>7s}  {'fit_s':>6s}")
    fitted = {}
    for name, factory in factories.items():
        t0 = time.time()
        m = factory().fit(train)
        elapsed = time.time() - t0
        res = evaluate(m, test)
        res["fit_seconds"] = float(elapsed)
        out["binary"][name] = res
        print(f"{name:12s}  {res['auc']:7.4f}  {res['logloss']:7.4f}  {res['accuracy']:7.4f}  "
              f"{res['brier']:7.4f}  {elapsed:6.1f}")
        fitted[name] = m

    if args.transformer_from:
        from brawlstar_agent.recommender.transformer_model import load_transformer
        print(f"\n[eval-topk] loading transformer from {args.transformer_from}*")
        t0 = time.time()
        tr = load_transformer(args.transformer_from)
        load_s = time.time() - t0
        res = evaluate(tr, test)
        res["fit_seconds"] = float(load_s)  # really "load seconds"; reused field for table consistency
        out["binary"]["Transformer"] = res
        print(f"{'Transformer':12s}  {res['auc']:7.4f}  {res['logloss']:7.4f}  {res['accuracy']:7.4f}  "
              f"{res['brier']:7.4f}  load={load_s:6.1f}")
        fitted["Transformer"] = tr

    # Pick a stable candidate pool. Prefer LightGBM's featurizer (matches v2);
    # fall back to the transformer if LightGBM was skipped.
    if "LightGBM" in fitted:
        cand_pool = sorted(fitted["LightGBM"].featurizer.brawler_to_idx.keys())
    elif "Transformer" in fitted:
        cand_pool = sorted(fitted["Transformer"].featurizer.brawler_to_idx.keys())
    else:
        cand_pool = sorted(fitted["ModeMap"].featurizer.brawler_to_idx.keys())

    topk_models = [n for n in ("Random", "TrophyOnly", "Global", "ModeMap", "LightGBM", "Transformer")
                   if n in fitted]
    print("\n=== Top-K recommendation (last_pick mode, all test rows) ===")
    print("Question: 'Mask team A's third pick, ask model to rank all 100+ candidates,"
          " find rank of actually-played brawler.'")
    for name in topk_models:
        m = fitted[name]
        try:
            res = evaluate_topk(m, test, mode="last_pick",
                                sample_size=args.sample_size,
                                candidate_pool=cand_pool, seed=42)
        except Exception as exc:
            print(f"  [{name}] error: {exc}")
            continue
        out["topk_all"][name] = asdict(res)
        print(f"\n{format_result(res, label=name)}")

    print("\n=== Top-K (only winning teams, harder + more meaningful) ===")
    print("Same setup, but only test rows where team A actually won — so the "
          "played brawler was at least \"good enough\" for that specific matchup.")
    winners_models = [n for n in ("Random", "TrophyOnly", "ModeMap", "LightGBM", "Transformer")
                      if n in fitted]
    for name in winners_models:
        m = fitted[name]
        res = evaluate_topk(m, test, mode="last_pick",
                            sample_size=args.sample_size,
                            candidate_pool=cand_pool,
                            only_winners=True, seed=42)
        out["topk_winners_only"][name] = asdict(res)
        print(f"\n{format_result(res, label=f'{name} (winners-only)')}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
