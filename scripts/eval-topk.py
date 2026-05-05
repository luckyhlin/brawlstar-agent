#!/usr/bin/env python3
"""Top-K recommendation evaluation against the held-out test set.

Answers the question: "If you ask the model to rank candidate brawlers for the
last pick, where does the actually-played brawler land?"

Reports hit@K, MRR, mean/median rank, and win-rate uplift for picks where the
played brawler IS in the model's top-K. Compares LightGBM to all heuristics
including a true random baseline and a trophy-only baseline that exposes how
much of the AUC is just "high-trophy team usually wins low-trophy team".
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from brawlstar_agent.recommender import (  # noqa: E402
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


def main() -> None:
    out_path = REPO / "reports" / "recommender_v1_topk.json"
    out: dict = {"binary": {}, "topk_all": {}, "topk_winners_only": {}}
    print("Loading data...")
    df = load_clean_battles().dropna(subset=["mode", "map", "battle_type"])
    train, test = split_random(df, test_frac=0.2, seed=42)
    print(f"  train={len(train):,}  test={len(test):,}")

    factories = {
        "Random":      lambda: RandomBaseline(),
        "TrophyOnly":  lambda: TrophyOnlyBaseline(),
        "Global":      lambda: GlobalWilsonBaseline(),
        "ModeMap":     lambda: ModeMapWilsonBaseline(),
        "LightGBM":    lambda: LGBMTeamModel(n_estimators=600, num_leaves=63,
                                              learning_rate=0.05, min_data_in_leaf=80),
    }

    print("\n=== Binary win-prediction (random split) ===")
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

    print("\n=== Top-K recommendation (last_pick mode, all test rows) ===")
    print("Question: 'Mask team A's third pick, ask model to rank all 100+ candidates,"
          " find rank of actually-played brawler.'")
    for name in ("Random", "TrophyOnly", "Global", "ModeMap", "LightGBM"):
        m = fitted[name]
        cand_pool = sorted(fitted["LightGBM"].featurizer.brawler_to_idx.keys())
        try:
            res = evaluate_topk(m, test, mode="last_pick",
                                sample_size=2500, candidate_pool=cand_pool, seed=42)
        except Exception as exc:
            print(f"  [{name}] error: {exc}")
            continue
        out["topk_all"][name] = asdict(res)
        print(f"\n{format_result(res, label=name)}")

    print("\n=== Top-K (only winning teams, harder + more meaningful) ===")
    print("Same setup, but only test rows where team A actually won — so the "
          "played brawler was at least \"good enough\" for that specific matchup.")
    for name in ("Random", "TrophyOnly", "ModeMap", "LightGBM"):
        m = fitted[name]
        cand_pool = sorted(fitted["LightGBM"].featurizer.brawler_to_idx.keys())
        res = evaluate_topk(m, test, mode="last_pick",
                            sample_size=2500, candidate_pool=cand_pool,
                            only_winners=True, seed=42)
        out["topk_winners_only"][name] = asdict(res)
        print(f"\n{format_result(res, label=f'{name} (winners-only)')}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
