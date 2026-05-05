#!/usr/bin/env python3
"""Train and evaluate the brawler-pick recommender.

The same script works at any cutoff date — that's the "transferable algorithm"
property. By default it trains on all clean post-2026-05-03 data and evaluates
both random and (when enough data) temporal splits.

Usage:
    PYTHONPATH=src uv run python scripts/train-recommender.py
    PYTHONPATH=src uv run python scripts/train-recommender.py --modes brawlBall
    PYTHONPATH=src uv run python scripts/train-recommender.py --save-to models/recommender_v1
    PYTHONPATH=src uv run python scripts/train-recommender.py --eval-only --report-to reports/v1.json

Outputs (to --report-to / --save-to):
    reports/v1.json   ← per-model metrics (random + temporal CV)
    models/recommender_v1.lgb.txt + .meta.json   ← trained LightGBM
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from brawlstar_agent.recommender import (  # noqa: E402
    CLEAN_CUTOFF_ISO,
    load_clean_battles,
    load_brawler_names,
    split_random,
)
from brawlstar_agent.recommender.baselines import (  # noqa: E402
    GlobalWilsonBaseline,
    ModeWilsonBaseline,
    ModeMapWilsonBaseline,
)
from brawlstar_agent.recommender.cv import (  # noqa: E402
    evaluate_models_on_folds,
    make_temporal_folds,
)
from brawlstar_agent.recommender.team_model import (  # noqa: E402
    LGBMTeamModel,
    LogRegTeamModel,
    evaluate,
    save_model,
)


def parse_modes(s: str | None) -> tuple[str, ...] | None:
    if not s:
        return None
    return tuple(m.strip() for m in s.split(",") if m.strip())


def per_mode_eval(model, test_df: pd.DataFrame, label_col: str = "team_a_wins") -> dict[str, dict]:
    """Evaluate `model` separately on each mode in test_df."""
    out = {}
    for mode, sub in test_df.groupby("mode"):
        if len(sub) < 200:
            continue
        out[str(mode)] = evaluate(model, sub, label_col=label_col)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="data/brawlstars.db", help="Path to SQLite DB")
    ap.add_argument("--cutoff", default=CLEAN_CUTOFF_ISO,
                    help=f"Earliest battle to include (ISO). Default: {CLEAN_CUTOFF_ISO}")
    ap.add_argument("--before", default=None, help="Latest battle (exclusive); None for everything")
    ap.add_argument("--modes", default=None, help="Comma-separated mode filter; None for all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-frac", type=float, default=0.2, help="Random-split holdout fraction")

    ap.add_argument("--save-to", default=None, help="Save the LightGBM model to this prefix")
    ap.add_argument("--report-to", default="reports/recommender_v1.json",
                    help="Write metrics JSON here")
    ap.add_argument("--no-train", action="store_true", help="Skip training, only print data summary")
    ap.add_argument("--no-temporal", action="store_true", help="Skip temporal CV (fast iteration)")

    # LGBM hyperparameters
    ap.add_argument("--lgbm-n-estimators", type=int, default=600)
    ap.add_argument("--lgbm-num-leaves", type=int, default=63)
    ap.add_argument("--lgbm-lr", type=float, default=0.05)
    ap.add_argument("--lgbm-min-leaf", type=int, default=80)
    ap.add_argument("--lgbm-l2", type=float, default=1.0)

    # Temporal CV
    ap.add_argument("--cv-train-days", type=float, default=0.75)
    ap.add_argument("--cv-test-days", type=float, default=0.25)
    ap.add_argument("--cv-step-hours", type=float, default=8)
    ap.add_argument("--cv-min-train", type=int, default=3000)
    ap.add_argument("--cv-min-test", type=int, default=1000)

    args = ap.parse_args()

    print(f"[train-recommender] cutoff={args.cutoff} modes={args.modes} db={args.db}")
    modes = parse_modes(args.modes)
    df = load_clean_battles(
        db_path=args.db, after=args.cutoff, before=args.before, modes=modes,
    ).dropna(subset=["mode", "map", "battle_type"])
    print(f"[train-recommender] loaded {len(df):,} rows ({df['battle_id'].nunique():,} battles)")
    if df.empty:
        print("[train-recommender] no data; exiting"); return

    if args.no_train:
        print(json.dumps({
            "n_rows": len(df),
            "n_battles": int(df["battle_id"].nunique()),
            "modes": df["mode"].value_counts().to_dict(),
            "battle_types": df["battle_type"].value_counts().to_dict(),
            "earliest": df["battle_time_iso"].min(),
            "latest": df["battle_time_iso"].max(),
        }, indent=2, default=str))
        return

    # Random split
    train, test = split_random(df, test_frac=args.test_frac, seed=args.seed)
    print(f"[train-recommender] random split: train={len(train):,} test={len(test):,}")

    report: dict = {
        "cutoff": args.cutoff,
        "modes": modes,
        "n_rows_total": int(len(df)),
        "n_battles": int(df["battle_id"].nunique()),
        "earliest": df["battle_time_iso"].min(),
        "latest": df["battle_time_iso"].max(),
        "test_frac": args.test_frac,
        "seed": args.seed,
        "random_split": {},
        "per_mode": {},
        "temporal_cv": [],
    }

    print("\n=== RANDOM SPLIT ===")
    print(f"{'Model':12s}  {'AUC':>7s}  {'logloss':>7s}  {'acc':>7s}  {'brier':>7s}  {'fit_s':>6s}")

    factories = {
        "Global":   lambda: GlobalWilsonBaseline(),
        "Mode":     lambda: ModeWilsonBaseline(),
        "ModeMap":  lambda: ModeMapWilsonBaseline(),
        "LogReg":   lambda: LogRegTeamModel(),
        "LightGBM": lambda: LGBMTeamModel(
            n_estimators=args.lgbm_n_estimators,
            num_leaves=args.lgbm_num_leaves,
            learning_rate=args.lgbm_lr,
            min_data_in_leaf=args.lgbm_min_leaf,
            reg_lambda=args.lgbm_l2,
            seed=args.seed,
        ),
    }

    fitted_models = {}
    for name, factory in factories.items():
        t0 = time.time()
        m = factory()
        m.fit(train)
        elapsed = time.time() - t0
        res = evaluate(m, test)
        res["fit_seconds"] = float(elapsed)
        report["random_split"][name] = res
        # Per-mode breakdown
        report["per_mode"][name] = per_mode_eval(m, test)
        print(f"{name:12s}  {res['auc']:7.4f}  {res['logloss']:7.4f}  {res['accuracy']:7.4f}  "
              f"{res['brier']:7.4f}  {elapsed:6.1f}")
        fitted_models[name] = m

    if not args.no_temporal:
        folds = make_temporal_folds(
            df,
            train_days=args.cv_train_days,
            test_days=args.cv_test_days,
            step_hours=args.cv_step_hours,
            min_train_rows=args.cv_min_train,
            min_test_rows=args.cv_min_test,
        )
        print(f"\n=== TEMPORAL CV ({len(folds)} folds) ===")
        if folds:
            cv_factories = {n: factories[n] for n in ("Global", "ModeMap", "LightGBM")}
            cv_df = evaluate_models_on_folds(folds, cv_factories)
            print(cv_df.groupby("model")[["auc", "logloss", "accuracy"]].mean().round(4).to_string())
            report["temporal_cv"] = cv_df.to_dict(orient="records")
        else:
            print("  (not enough data for any fold; skipping)")

    # Save trained model
    if args.save_to:
        save_path = Path(args.save_to)
        if "LightGBM" in fitted_models:
            save_model(fitted_models["LightGBM"], save_path)
            print(f"\n[train-recommender] saved LightGBM to {save_path}.lgb.txt + .meta.json")

    # Write JSON report
    out = Path(args.report_to)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[train-recommender] wrote report to {out}")


if __name__ == "__main__":
    main()
