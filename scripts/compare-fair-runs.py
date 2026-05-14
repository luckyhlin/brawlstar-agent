#!/usr/bin/env python3
"""Compare v2 'fair' recommender runs on the shared stable test set.

Reads `reports/recommender_v2_default_fair.json`, `recommender_v2_30d_fair.json`,
and `recommender_v2_all_fair.json` (DEC-011) and prints an apples-to-apples
table. Each run trained on a different cutoff but evaluated on the SAME held-out
window, so AUC differences are directly attributable to training-data choice.

Usage:
    PYTHONPATH=src uv run python scripts/compare-fair-runs.py
    PYTHONPATH=src uv run python scripts/compare-fair-runs.py --output reports/v2_fair_comparison.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

REPORTS = {
    "A_fair (3-day, cutoff 2026-05-03)": REPO / "reports" / "recommender_v2_default_fair.json",
    "C_fair (30-day, cutoff 2026-04-06)": REPO / "reports" / "recommender_v2_30d_fair.json",
    "B_fair (all-data, cutoff 2021-01-01)": REPO / "reports" / "recommender_v2_all_fair.json",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default=None, help="If set, write a comparison JSON here")
    args = ap.parse_args()

    loaded: dict[str, dict] = {}
    for label, path in REPORTS.items():
        if not path.exists():
            print(f"  [missing] {label}: {path}")
            continue
        with open(path) as f:
            loaded[label] = json.load(f)

    if not loaded:
        raise SystemExit("No fair-run reports found.")

    boundaries = {r.get("stable_test_after") for r in loaded.values()}
    if len(boundaries) != 1:
        print(f"  WARNING: stable_test_after differs across reports: {boundaries}")
    print(f"\nStable test boundary: {boundaries.pop() if len(boundaries)==1 else 'mixed'}")
    print()
    print(f"{'Run':45s} {'train_btls':>10s} {'test_btls':>10s} {'min_train':>10s} {'max_train':>10s}")
    for label, r in loaded.items():
        print(f"  {label:43s} {r['n_train_battles']:>10,} {r['n_test_battles']:>10,} "
              f"{r['earliest'][:10]:>10s} {r['latest'][:10] if r.get('latest') else '?':>10s}")

    print()
    print(f"=== AUC on stable test set (binary win prediction) ===")
    print(f"{'Run':45s}  {'Global':>7s}  {'Mode':>7s}  {'ModeMap':>7s}  {'LogReg':>7s}  {'LightGBM':>9s}  {'LGBM_fit_s':>10s}")
    for label, r in loaded.items():
        seg = r.get("stable_test", r.get("random_split", {}))
        line = f"  {label:43s}"
        for k in ("Global", "Mode", "ModeMap", "LogReg"):
            v = seg.get(k, {}).get("auc")
            line += f"  {v:7.4f}" if v is not None else f"  {'?':>7s}"
        v = seg.get("LightGBM", {}).get("auc")
        line += f"  {v:9.4f}" if v is not None else f"  {'?':>9s}"
        fs = seg.get("LightGBM", {}).get("fit_seconds")
        line += f"  {fs:10.1f}" if fs is not None else f"  {'?':>10s}"
        print(line)

    print()
    print("=== Per-mode LightGBM AUC on stable test set ===")
    modes_seen = set()
    for r in loaded.values():
        modes_seen.update(r.get("per_mode", {}).get("LightGBM", {}).keys())
    modes_seen = sorted(
        modes_seen,
        key=lambda m: -max((
            r.get("per_mode", {}).get("LightGBM", {}).get(m, {}).get("n", 0)
            for r in loaded.values()
        ), default=0),
    )
    header = f"{'Mode':14s}  " + "  ".join(f"{label[:18]:>9s}" for label in loaded.keys())
    print(header)
    for mode in modes_seen:
        row = f"{mode:14s}"
        for label, r in loaded.items():
            d = r.get("per_mode", {}).get("LightGBM", {}).get(mode)
            if d is None:
                row += f"  {'-':>9s}"
            else:
                row += f"  {d['auc']:>9.4f}"
        print(row)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(loaded, f, indent=2, default=str)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
