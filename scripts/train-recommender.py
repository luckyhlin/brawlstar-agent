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
from brawlstar_agent.recommender.eval_slices import (  # noqa: E402
    evaluate_slices,
    format_slice_table,
)
from brawlstar_agent.recommender.team_model import (  # noqa: E402
    LGBMTeamModel,
    LogRegTeamModel,
    evaluate,
    save_model,
)

# Stable test boundary: every fair-comparison v2 run holds out battles with
# battle_time_iso >= this timestamp as the test set. Pinned 2026-05-06 once the
# random-split test sets across Runs A/C were found to be confounded by their
# different cutoffs. Going forward, do not change this without renaming
# downstream artifacts; comparisons across runs depend on a single boundary.
STABLE_TEST_AFTER_DEFAULT = "2026-05-05T00:00:00Z"


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
    ap.add_argument(
        "--stable-test-after",
        default=None,
        help=(
            "If set, hold out all battles with battle_time_iso >= this timestamp "
            "as the test set (replaces the random split). Train data becomes "
            "[--cutoff, --stable-test-after); test data becomes [--stable-test-after, --before]. "
            f"Recommended canonical value: {STABLE_TEST_AFTER_DEFAULT!r}. "
            "When set, temporal CV runs only over the train portion (no leakage)."
        ),
    )

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

    # Phase-1 (v3.1) feature engineering toggle. When set, both LGBM and
    # LogReg featurizers append per-team trophy/power aggregates to the dense
    # feature matrix (sparse matrix is unchanged for LogReg). See
    # `recommender.features.compute_team_aggregates` for the column layout.
    ap.add_argument("--use-team-aggregates", action="store_true",
                    help="Phase 1: append 23 per-team trophy/power aggregates to LGBM dense features.")
    # Phase-2 — cyclical time + per-team `days_since_release` aggregates.
    # Composable with phase 1.
    ap.add_argument("--use-time-features", action="store_true",
                    help="Phase 2: append 12 cyclical-time + days_since_release scalars.")
    # Phase-4 — per-team aggregates of per-player history stats. The lookup
    # is fit from PRE-CUTOFF data (`--history-after`..`--cutoff`) so that
    # training rows are not in the lookup (no leakage). Frequency-only
    # features (no WR) since pre-cutoff data has the legacy team-result bug.
    ap.add_argument("--use-history-features", action="store_true",
                    help="Phase 4: append 12 per-team aggregates of per-player history stats.")
    ap.add_argument("--history-after", default="2026-04-01T00:00:00Z",
                    help="When --use-history-features is set, the lookup uses battles "
                         "with battle_time_iso in [history_after, cutoff). Default Apr 1.")
    # Restrict training and test data to a subset of battle_types. Default is
    # both. Use 'soloRanked' alone to focus the model on the actual Ranked
    # queue (where strict 1-2-2-1 draft happens at Mythic+).
    ap.add_argument("--battle-types", default="ranked,soloRanked",
                    help="Comma-separated battle_types to include (default both).")

    # Temporal CV
    ap.add_argument("--cv-train-days", type=float, default=0.75)
    ap.add_argument("--cv-test-days", type=float, default=0.25)
    ap.add_argument("--cv-step-hours", type=float, default=8)
    ap.add_argument("--cv-min-train", type=int, default=3000)
    ap.add_argument("--cv-min-test", type=int, default=1000)

    args = ap.parse_args()

    print(f"[train-recommender] cutoff={args.cutoff} modes={args.modes} db={args.db}")
    if args.stable_test_after:
        print(f"[train-recommender] STABLE-TEST mode: test = battles with battle_time_iso >= {args.stable_test_after}")
    modes = parse_modes(args.modes)
    battle_types = tuple(b.strip() for b in args.battle_types.split(",") if b.strip())
    if battle_types != ("ranked", "soloRanked"):
        print(f"[train-recommender] battle_types restricted to: {battle_types}")
    df = load_clean_battles(
        db_path=args.db, after=args.cutoff, before=args.before, modes=modes,
        battle_types=battle_types,
    ).dropna(subset=["mode", "map", "battle_type"])
    print(f"[train-recommender] loaded {len(df):,} rows ({df['battle_id'].nunique():,} battles)")
    if df.empty:
        print("[train-recommender] no data; exiting"); return

    # Optionally load a separate history DataFrame for phase 4. We use
    # battles in [history_after, cutoff) — i.e. PRE-CUTOFF data — so the
    # training rows aren't in the lookup. Note: pre-cutoff battles have the
    # legacy team-result bug (DEC-010), but phase-4 features are frequency-only
    # so the bug doesn't affect them.
    history_df: pd.DataFrame | None = None
    if args.use_history_features:
        history_df = load_clean_battles(
            db_path=args.db, after=args.history_after, before=args.cutoff,
            modes=modes, battle_types=battle_types,
        ).dropna(subset=["mode", "map", "battle_type"])
        print(f"[train-recommender] history_df: {len(history_df):,} rows "
              f"({history_df['battle_id'].nunique():,} battles, "
              f"{args.history_after} → {args.cutoff})")

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

    # Choose split mode: stable temporal holdout, or random split.
    if args.stable_test_after:
        train_mask = df["battle_time_iso"] < args.stable_test_after
        train = df[train_mask].copy()
        test = df[~train_mask].copy()
        if test.empty or train.empty:
            raise SystemExit(
                f"stable-test split produced empty side: train={len(train)} test={len(test)}. "
                f"Check --cutoff / --before / --stable-test-after."
            )
        split_mode = "stable_test"
        eval_section = "stable_test"
        print(
            f"[train-recommender] stable-test split: "
            f"train={len(train):,} ({train['battle_id'].nunique():,} battles, "
            f"{train['battle_time_iso'].min()} → {train['battle_time_iso'].max()}) | "
            f"test={len(test):,} ({test['battle_id'].nunique():,} battles, "
            f"{test['battle_time_iso'].min()} → {test['battle_time_iso'].max()})"
        )
    else:
        train, test = split_random(df, test_frac=args.test_frac, seed=args.seed)
        split_mode = "random"
        eval_section = "random_split"
        print(f"[train-recommender] random split: train={len(train):,} test={len(test):,}")

    report: dict = {
        "cutoff": args.cutoff,
        "before": args.before,
        "modes": modes,
        "n_rows_total": int(len(df)),
        "n_battles": int(df["battle_id"].nunique()),
        "earliest": df["battle_time_iso"].min(),
        "latest": df["battle_time_iso"].max(),
        "test_frac": args.test_frac,
        "seed": args.seed,
        "split_mode": split_mode,
        "stable_test_after": args.stable_test_after,
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
        "n_train_battles": int(train["battle_id"].nunique()),
        "n_test_battles": int(test["battle_id"].nunique()),
        eval_section: {},
        "per_mode": {},
        "temporal_cv": [],
    }

    section_label = "STABLE-TEST SPLIT" if split_mode == "stable_test" else "RANDOM SPLIT"
    print(f"\n=== {section_label} ===")
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
            include_team_aggregates=args.use_team_aggregates,
            include_time_features=args.use_time_features,
            include_history_features=args.use_history_features,
        ),
    }
    report["battle_types"] = list(battle_types)
    if args.use_team_aggregates:
        report["use_team_aggregates"] = True
        print("[train-recommender] phase 1: per-team trophy/power aggregates ENABLED for LightGBM")
    if args.use_time_features:
        report["use_time_features"] = True
        print("[train-recommender] phase 2: cyclical-time + days_since_release ENABLED for LightGBM")
    if args.use_history_features:
        report["use_history_features"] = True
        print("[train-recommender] phase 4: per-player history aggregates ENABLED for LightGBM")

    fitted_models = {}
    fitted_proba: dict[str, np.ndarray] = {}
    for name, factory in factories.items():
        t0 = time.time()
        m = factory()
        # Pass history_df only to model classes that accept it (LGBM,
        # LogReg). Wilson baselines don't take it.
        if history_df is not None and hasattr(m, "include_history_features"):
            m.fit(train, history_df=history_df)
        else:
            m.fit(train)
        elapsed = time.time() - t0
        proba = m.predict_proba(test)
        fitted_proba[name] = proba
        res = evaluate(m, test, proba=proba)
        res["fit_seconds"] = float(elapsed)
        report[eval_section][name] = res
        # Per-mode breakdown
        report["per_mode"][name] = per_mode_eval(m, test)
        print(f"{name:12s}  {res['auc']:7.4f}  {res['logloss']:7.4f}  {res['accuracy']:7.4f}  "
              f"{res['brier']:7.4f}  {elapsed:6.1f}")
        fitted_models[name] = m

    # Tiered slice metrics — only meaningful in stable-test mode (random
    # splits don't preserve battle_type composition consistently).
    if split_mode == "stable_test":
        print("\n=== TIERED SLICES (stable test) ===")
        slices_section: dict[str, dict] = {}
        for name, m in fitted_models.items():
            slice_metrics = evaluate_slices(m, test, proba=fitted_proba[name])
            slices_section[name] = slice_metrics
            print(f"\n  [{name}]")
            for line in format_slice_table(slice_metrics).split("\n"):
                print("  " + line)
        report[f"{eval_section}_slices"] = slices_section

    if not args.no_temporal:
        # In stable-test mode, run CV only over training data so we never leak
        # into the held-out set; in random-split mode, fold over the full df.
        cv_source = train if split_mode == "stable_test" else df
        folds = make_temporal_folds(
            cv_source,
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
