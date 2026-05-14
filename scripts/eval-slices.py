#!/usr/bin/env python3
"""Apply tiered evaluation slicing to one or more saved models.

Slices reported per model:
    all                       — full DEC-011 stable test set (back-compat)
    ranked                    — trophy-ladder (Unranked) battles only
    soloRanked                — competitive Ranked queue (any tier)
    soloRanked_diamondplus    — both teams Diamond+ (>= 10), simultaneous-pick draft
    soloRanked_mythicplus     — both teams Mythic+  (>= 13), strict 1-2-2-1 ban/pick
    soloRanked_legendaryplus  — both teams Legendary+ (>= 16), top-tier slice

This script does not retrain anything. It just runs `predict_proba` on the
existing stable test set and computes per-slice binary metrics.

Usage:
    PYTHONPATH=src uv run python scripts/eval-slices.py \\
        --model models/recommender_v3_xl \\
        --output reports/recommender_v3_xl_slices.json

    # Multiple models in one run; output is a single combined JSON:
    PYTHONPATH=src uv run python scripts/eval-slices.py \\
        --model models/recommender_v3_big \\
        --model models/recommender_v3_xl \\
        --model models/recommender_v3_phase1_big \\
        --model models/recommender_v2_phase1 \\
        --output reports/slices_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

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

STABLE_TEST_AFTER_DEFAULT = "2026-05-05T00:00:00Z"


def _load_any_model(prefix: str | Path):
    """Load either an LGBM or transformer checkpoint by prefix.

    LGBM artifacts:    `<prefix>.lgb.txt` + `<prefix>.meta.json`
    Transformer:       `<prefix>.pt`      + `<prefix>.meta.json`
    """
    p = Path(prefix)
    pt_path = Path(str(p) + ".pt")
    lgb_path = Path(str(p) + ".lgb.txt")
    if pt_path.exists():
        from brawlstar_agent.recommender.transformer_model import load_transformer
        return load_transformer(str(p)), "transformer"
    if lgb_path.exists():
        from brawlstar_agent.recommender.team_model import load_model
        return load_model(str(p)), "lightgbm"
    raise FileNotFoundError(
        f"No model artifact found at prefix '{prefix}' "
        f"(tried '{pt_path}' and '{lgb_path}')."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True,
                    help="Model prefix (no extension). Pass multiple times to evaluate several models.")
    ap.add_argument("--db", default="data/brawlstars.db")
    ap.add_argument("--cutoff", default=CLEAN_CUTOFF_ISO)
    ap.add_argument("--before", default=None)
    ap.add_argument("--stable-test-after", default=STABLE_TEST_AFTER_DEFAULT)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    t_load = time.time()
    print(f"[eval-slices] loading clean battles from {args.db} ...")
    df = load_clean_battles(
        db_path=args.db, after=args.cutoff, before=args.before,
    ).dropna(subset=["mode", "map", "battle_type"])
    test_df = df[df["battle_time_iso"] >= args.stable_test_after].copy().reset_index(drop=True)
    print(f"[eval-slices] full clean window: {len(df):,} rows; "
          f"stable-test (>= {args.stable_test_after}): {len(test_df):,} rows "
          f"({test_df['battle_id'].nunique():,} battles), "
          f"loaded in {time.time() - t_load:.1f}s")

    combined: dict[str, dict] = {
        "stable_test_after": args.stable_test_after,
        "cutoff": args.cutoff,
        "n_test_total": int(len(test_df)),
        "models": {},
    }

    for prefix in args.model:
        t_model = time.time()
        try:
            model, kind = _load_any_model(prefix)
        except FileNotFoundError as e:
            print(f"[eval-slices] SKIP {prefix}: {e}")
            combined["models"][prefix] = {"error": str(e)}
            continue
        print(f"\n[eval-slices] === {prefix}  ({kind}) ===")
        slices_metrics = evaluate_slices(model, test_df)
        elapsed = time.time() - t_model
        combined["models"][prefix] = {
            "kind": kind,
            "elapsed_seconds": float(elapsed),
            "slices": slices_metrics,
        }
        print(format_slice_table(slices_metrics))
        print(f"[eval-slices] {prefix} done in {elapsed:.1f}s")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"\n[eval-slices] wrote combined report to {args.output}")

    # Cross-model summary table — one row per model, AUC per slice. Slice n
    # is the same across models, so it gets its own row at the top.
    print("\n=== AUC SUMMARY (rows = models, cols = slices) ===")
    slice_order = [
        "all", "ranked", "soloRanked",
        "soloRanked_diamondplus", "soloRanked_mythicplus", "soloRanked_legendaryplus",
    ]
    short_names = {
        "all":                       "all",
        "ranked":                    "rk(unrkd)",
        "soloRanked":                "soloRkd",
        "soloRanked_diamondplus":    "Dia+",
        "soloRanked_mythicplus":     "Myth+",
        "soloRanked_legendaryplus":  "Lgd+",
    }
    first_valid = next(
        (info for info in combined["models"].values() if "slices" in info),
        None,
    )
    header = f"{'model':40s}" + "".join(f"  {short_names[s]:>10s}" for s in slice_order)
    print(header)
    if first_valid is not None:
        n_row = f"{'(n)':40s}" + "".join(
            f"  {first_valid['slices'].get(s, {}).get('n', 0):>10,d}" for s in slice_order
        )
        print(n_row)
    print("-" * len(header))
    for prefix, info in combined["models"].items():
        if "error" in info:
            continue
        cells = [f"{prefix:40s}"]
        slices = info["slices"]
        for s in slice_order:
            v = slices.get(s, {})
            if v.get("skipped") or v.get("auc") is None:
                cells.append(f"  {'(skip)':>10s}")
            else:
                cells.append(f"  {v['auc']:>10.4f}")
        print("".join(cells))


if __name__ == "__main__":
    main()
