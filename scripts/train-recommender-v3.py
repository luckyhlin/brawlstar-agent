#!/usr/bin/env python3
"""Train and evaluate the v3 (attention/transformer) brawler-pick recommender.

This script is the v3 sibling of `scripts/train-recommender.py`. Same data
loading, same stable-test methodology (DEC-011), same per-mode breakdown — so
the resulting AUC / log-loss numbers are directly comparable to the A_fair /
C_fair LightGBM rows in `docs/recommender-v2.md`.

Default config trains the production v3 candidate at the A_fair cutoff
(`2026-05-03T01:00:00Z`) with the canonical stable test boundary
(`2026-05-05T00:00:00Z`). All v3 runs MUST keep the same `--stable-test-after`
or AUCs are not comparable to v2.

Usage:
    PYTHONPATH=src uv run python scripts/train-recommender-v3.py \
        --cutoff 2026-05-03T01:00:00Z \
        --stable-test-after 2026-05-05T00:00:00Z \
        --epochs 5 --batch-size 4096 \
        --save-to models/recommender_v3_default \
        --report-to reports/recommender_v3_default.json

Outputs:
    models/recommender_v3_default.pt           torch state dict
    models/recommender_v3_default.meta.json    vocab + arch + training history
    reports/recommender_v3_default.json        AUC / per-mode metrics
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
)
from brawlstar_agent.recommender.eval_slices import (  # noqa: E402
    evaluate_slices,
    format_slice_table,
)
from brawlstar_agent.recommender.team_model import evaluate  # noqa: E402
from brawlstar_agent.recommender.transformer_model import (  # noqa: E402
    TransformerTeamModel,
    save_transformer,
)

# Pinned alongside scripts/train-recommender.py — DEC-011.
STABLE_TEST_AFTER_DEFAULT = "2026-05-05T00:00:00Z"


def per_mode_eval(model, test_df: pd.DataFrame, label_col: str = "team_a_wins") -> dict[str, dict]:
    out = {}
    for mode, sub in test_df.groupby("mode"):
        if len(sub) < 200:
            continue
        out[str(mode)] = evaluate(model, sub, label_col=label_col)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="data/brawlstars.db")
    ap.add_argument("--cutoff", default=CLEAN_CUTOFF_ISO,
                    help=f"Earliest battle to include (ISO). Default: {CLEAN_CUTOFF_ISO}")
    ap.add_argument("--before", default=None)
    ap.add_argument(
        "--stable-test-after", default=STABLE_TEST_AFTER_DEFAULT,
        help=("Hold out battles with battle_time_iso >= this as the test set. "
              f"Default (DEC-011): {STABLE_TEST_AFTER_DEFAULT}. "
              "Pass '' to disable and use a 5%% internal validation split for held-out metrics."))

    ap.add_argument("--save-to", default=None, help="Save trained transformer to this prefix")
    ap.add_argument("--report-to", default="reports/recommender_v3_default.json")
    ap.add_argument("--no-train", action="store_true",
                    help="Just print data summary and exit")

    # Transformer architecture
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--ff", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)

    # Training
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--early-stop-patience", type=int, default=2)
    ap.add_argument("--device", default="cpu", help="cpu or cuda (if available)")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)

    # Phase-1 (v3.1) feature engineering toggle: appends per-team trophy/power
    # aggregates (23 scalars) to the head's scalar input. The encoder is
    # unchanged; only the head's first Linear widens by TEAM_AGGREGATE_DIM.
    ap.add_argument("--use-team-aggregates", action="store_true",
                    help="Phase 1: feed per-team trophy/power aggregates into the head.")
    # Phase-2: cyclical hour/dow + per-team days_since_release aggregates.
    # Composable with phase 1; both go through the same `extra_scalar` head input.
    ap.add_argument("--use-time-features", action="store_true",
                    help="Phase 2: feed cyclical-time + days_since_release scalars into the head.")
    # Phase-4: per-team aggregates of per-player history stats. Lookup is
    # built from a separate pre-cutoff window (`--history-after`..`--cutoff`)
    # so training rows aren't in the lookup. Frequency-only features.
    ap.add_argument("--use-history-features", action="store_true",
                    help="Phase 4: feed per-player history aggregates into the head.")
    ap.add_argument("--history-after", default="2026-04-01T00:00:00Z",
                    help="When --use-history-features is set, build the lookup from "
                         "battles in [history_after, cutoff). Default Apr 1.")
    # Restrict training/test data to a subset of battle_types. Default both.
    # Use 'soloRanked' alone to focus on the strict-draft competitive subset.
    ap.add_argument("--battle-types", default="ranked,soloRanked",
                    help="Comma-separated battle_types to include (default both).")

    args = ap.parse_args()
    stable_test_after = args.stable_test_after or None
    battle_types = tuple(b.strip() for b in args.battle_types.split(",") if b.strip())

    print(f"[train-v3] cutoff={args.cutoff} stable_test_after={stable_test_after} db={args.db}")
    if battle_types != ("ranked", "soloRanked"):
        print(f"[train-v3] battle_types restricted to: {battle_types}")
    df = load_clean_battles(
        db_path=args.db, after=args.cutoff, before=args.before,
        battle_types=battle_types,
    ).dropna(subset=["mode", "map", "battle_type"])
    print(f"[train-v3] loaded {len(df):,} rows ({df['battle_id'].nunique():,} battles, "
          f"{df['battle_time_iso'].min()} → {df['battle_time_iso'].max()})")
    if df.empty:
        print("[train-v3] no data; exiting"); return

    history_df: pd.DataFrame | None = None
    if args.use_history_features:
        history_df = load_clean_battles(
            db_path=args.db, after=args.history_after, before=args.cutoff,
            battle_types=battle_types,
        ).dropna(subset=["mode", "map", "battle_type"])
        print(f"[train-v3] history_df: {len(history_df):,} rows "
              f"({history_df['battle_id'].nunique():,} battles, "
              f"{args.history_after} → {args.cutoff})")

    if args.no_train:
        print(json.dumps({
            "n_rows": len(df),
            "n_battles": int(df["battle_id"].nunique()),
            "modes": df["mode"].value_counts().to_dict(),
        }, indent=2, default=str))
        return

    if stable_test_after:
        train_mask = df["battle_time_iso"] < stable_test_after
        train = df[train_mask].copy()
        test = df[~train_mask].copy()
        if test.empty or train.empty:
            raise SystemExit(
                f"stable-test split produced empty side: train={len(train)} test={len(test)}"
            )
        split_mode = "stable_test"
        eval_section = "stable_test"
        print(f"[train-v3] stable-test split: train={len(train):,} ({train['battle_id'].nunique():,} battles) "
              f"test={len(test):,} ({test['battle_id'].nunique():,} battles)")
    else:
        from brawlstar_agent.recommender import split_random
        train, test = split_random(df, test_frac=0.2, seed=args.seed)
        split_mode = "random"
        eval_section = "random_split"
        print(f"[train-v3] random split: train={len(train):,} test={len(test):,}")

    report: dict = {
        "model": "TransformerTeamModel",
        "cutoff": args.cutoff,
        "before": args.before,
        "n_rows_total": int(len(df)),
        "n_battles": int(df["battle_id"].nunique()),
        "earliest": df["battle_time_iso"].min(),
        "latest": df["battle_time_iso"].max(),
        "seed": args.seed,
        "split_mode": split_mode,
        "stable_test_after": stable_test_after,
        "n_train_rows": int(len(train)),
        "n_test_rows": int(len(test)),
        "n_train_battles": int(train["battle_id"].nunique()),
        "n_test_battles": int(test["battle_id"].nunique()),
        "arch": {
            "d_model": args.d_model, "nhead": args.nhead, "ff": args.ff,
            "num_layers": args.num_layers, "dropout": args.dropout,
        },
        "training": {
            "epochs": args.epochs, "batch_size": args.batch_size,
            "lr": args.lr, "weight_decay": args.weight_decay,
            "early_stop_patience": args.early_stop_patience,
            "device": args.device,
            "use_team_aggregates": bool(args.use_team_aggregates),
            "use_time_features": bool(args.use_time_features),
            "use_history_features": bool(args.use_history_features),
            "battle_types": list(battle_types),
        },
        eval_section: {},
        "per_mode": {},
        "history": [],
    }

    extras = []
    if args.use_team_aggregates:
        extras.append("team_aggregates")
    if args.use_time_features:
        extras.append("time_features")
    if args.use_history_features:
        extras.append("history_features")
    extras_str = (" + " + " + ".join(extras)) if extras else ""
    print(f"\n=== TRANSFORMER (d_model={args.d_model} nhead={args.nhead} layers={args.num_layers} ff={args.ff}"
          f"{extras_str}) ===")

    t0 = time.time()
    m = TransformerTeamModel(
        d_model=args.d_model, nhead=args.nhead, ff=args.ff,
        num_layers=args.num_layers, dropout=args.dropout,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, seed=args.seed,
        early_stop_patience=args.early_stop_patience,
        device=args.device, num_workers=args.num_workers,
        log_every=args.log_every, verbose=True,
        use_team_aggregates=args.use_team_aggregates,
        use_time_features=args.use_time_features,
        use_history_features=args.use_history_features,
    )
    if history_df is not None:
        m.fit(train, history_df=history_df)
    else:
        m.fit(train)
    fit_seconds = time.time() - t0

    print(f"\n=== {('STABLE-TEST SPLIT' if split_mode == 'stable_test' else 'RANDOM SPLIT')} ===")
    # Compute predictions ONCE on the full test set; reuse for binary,
    # per-mode, and per-slice metrics. Tensorizing 1.69 M rows is the
    # expensive bit for the transformer; doing it once saves ~50-100 s.
    proba_full = m.predict_proba(test)
    res = evaluate(m, test, proba=proba_full)
    res["fit_seconds"] = float(fit_seconds)
    report[eval_section]["Transformer"] = res
    report["history"] = m.history
    print(f"Transformer  AUC={res['auc']:.4f}  logloss={res['logloss']:.4f}  "
          f"acc={res['accuracy']:.4f}  brier={res['brier']:.4f}  fit_s={fit_seconds:.1f}")

    print("\n=== PER-MODE ===")
    pm = per_mode_eval(m, test)
    report["per_mode"]["Transformer"] = pm
    for mode in sorted(pm.keys()):
        r = pm[mode]
        print(f"  {mode:14s}  AUC={r['auc']:.4f}  n={r['n']:>8,d}")

    if split_mode == "stable_test":
        print("\n=== TIERED SLICES (stable test) ===")
        slice_metrics = evaluate_slices(m, test, proba=proba_full)
        report[f"{eval_section}_slices"] = {"Transformer": slice_metrics}
        print(format_slice_table(slice_metrics))

    if args.save_to:
        save_transformer(m, args.save_to)
        print(f"\n[train-v3] saved transformer to {args.save_to}.pt + .meta.json")

    out = Path(args.report_to)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"[train-v3] wrote report to {out}")


if __name__ == "__main__":
    main()
