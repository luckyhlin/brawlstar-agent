"""
Ensemble the best LGBM (Run F, phase 1+4) and the best v3 transformer (Run K,
XL + phase 1+2+4) on the DEC-011 stable test set, both globally and on the
soloRanked_mythicplus slice. Reports the optimal blend weight, the AUC at
every blend ratio in 5% increments, and the slice breakdown for the best
blend so we can compare apples-to-apples with the individual models.

Both saved models have their phase-4 player-history lookup baked into the
.meta.json (~200 MB for LGBM, ~390 MB for XL), so `predict_proba(df)` just
works without re-passing a `history_df`.

Run with:
  UV_CACHE_DIR=/media/lin/disk2/brawlstar-agent/.uv-cache-local \\
    PYTHONPATH=src \\
    uv run python scripts/ensemble-stable-test.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from brawlstar_agent.recommender import CLEAN_CUTOFF_ISO, load_clean_battles
from brawlstar_agent.recommender.eval_slices import (
    make_slice_masks,
    evaluate_slices,
)
from brawlstar_agent.recommender.team_model import load_model as load_lgbm
from brawlstar_agent.recommender.transformer_model import load_transformer

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
MODELS_DIR = REPO_ROOT / "models"

STABLE_TEST_AFTER = "2026-05-05T00:00:00Z"
CUTOFF = CLEAN_CUTOFF_ISO  # 2026-05-03T01:00:00Z


def main() -> None:
    t0 = time.time()
    print("[ensemble] loading stable test slice...", flush=True)
    df_all = load_clean_battles(
        db_path=str(REPO_ROOT / "data" / "brawlstars.db"),
        after=CUTOFF, before=None,
        battle_types=("ranked", "soloRanked"),
    )
    test = df_all[df_all["battle_time_iso"] >= STABLE_TEST_AFTER].copy()
    print(f"[ensemble] test rows: {len(test):,} (battles: {test['battle_id'].nunique():,})", flush=True)
    print(f"[ensemble] load took {time.time()-t0:.1f}s", flush=True)

    cache_dir = REPORTS_DIR / "ensemble_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    lgbm_cache = cache_dir / "p_lgbm_phase1p4.npy"
    xfr_cache  = cache_dir / "p_xfr_phase1p2p4_xl.npy"
    n_test = len(test)

    def load_or_predict(cache_path, loader, name):
        if cache_path.exists():
            arr = np.load(cache_path)
            if arr.shape == (n_test,):
                print(f"  reusing cached {name} probas from {cache_path} (n={len(arr)})", flush=True)
                return arr
            else:
                print(f"  cached {name} has stale shape {arr.shape}, recomputing", flush=True)
        # Actually load the model and compute.
        t_l = time.time()
        m = loader()
        print(f"  {name} loaded in {time.time()-t_l:.1f}s", flush=True)
        t_p = time.time()
        arr = np.asarray(m.predict_proba(test), dtype=np.float64)
        print(f"  {name} proba in {time.time()-t_p:.1f}s", flush=True)
        np.save(cache_path, arr)
        return arr

    print("[ensemble] computing / loading probas...", flush=True)
    p_lgbm = load_or_predict(lgbm_cache, lambda: load_lgbm(MODELS_DIR / "recommender_v2_phase1p4"), "LGBM phase1p4")
    p_xfr  = load_or_predict(xfr_cache,  lambda: load_transformer(MODELS_DIR / "recommender_v3_phase1p2p4_xl"), "Transformer phase1p2p4_xl")
    assert p_lgbm.shape == p_xfr.shape == (n_test,), \
        f"shape mismatch: {p_lgbm.shape} vs {p_xfr.shape} vs ({n_test},)"

    y = test["team_a_wins"].astype(int).values

    # Slice masks (so we can evaluate ensemble on Mythic+ too)
    slices = make_slice_masks(test)

    # Sweep blend alpha = weight on LGBM in [0..1] step 0.025
    alphas = np.round(np.arange(0.0, 1.0001, 0.025), 4)
    rows = []
    for a in alphas:
        p_blend = a * p_lgbm + (1.0 - a) * p_xfr
        # clip to (1e-7, 1-1e-7) to keep logloss finite
        p_blend = np.clip(p_blend, 1e-7, 1 - 1e-7)
        auc_all  = roc_auc_score(y, p_blend)
        # slice eval: mythicplus mask
        myth_mask = slices["soloRanked_mythicplus"]
        n_myth = int(myth_mask.sum())
        auc_myth = roc_auc_score(y[myth_mask], p_blend[myth_mask]) if n_myth > 100 else float("nan")
        ll_all  = log_loss(y, p_blend)
        ll_myth = log_loss(y[myth_mask], p_blend[myth_mask]) if n_myth > 100 else float("nan")
        rows.append({
            "alpha_lgbm": float(a),
            "alpha_xfr": float(1 - a),
            "auc_all": auc_all,
            "auc_mythicplus": auc_myth,
            "logloss_all": ll_all,
            "logloss_mythicplus": ll_myth,
        })

    sweep = pd.DataFrame(rows)
    best_all_idx = sweep["auc_all"].idxmax()
    best_myth_idx = sweep["auc_mythicplus"].idxmax()
    print()
    print(sweep.to_string(index=False))
    print()
    print(f"Best blend by all-test AUC:   alpha_lgbm = {sweep.loc[best_all_idx, 'alpha_lgbm']:.3f}  "
          f"AUC_all = {sweep.loc[best_all_idx, 'auc_all']:.5f}  "
          f"AUC_myth = {sweep.loc[best_all_idx, 'auc_mythicplus']:.5f}")
    print(f"Best blend by Mythic+ AUC:    alpha_lgbm = {sweep.loc[best_myth_idx, 'alpha_lgbm']:.3f}  "
          f"AUC_all = {sweep.loc[best_myth_idx, 'auc_all']:.5f}  "
          f"AUC_myth = {sweep.loc[best_myth_idx, 'auc_mythicplus']:.5f}")
    print(f"Pure LGBM:     AUC_all = {sweep.iloc[-1]['auc_all']:.5f}  AUC_myth = {sweep.iloc[-1]['auc_mythicplus']:.5f}")
    print(f"Pure XL+P1P2P4:AUC_all = {sweep.iloc[0]['auc_all']:.5f}  AUC_myth = {sweep.iloc[0]['auc_mythicplus']:.5f}")

    # Now compute the full slice breakdown at the best-mythic blend.
    a_best = sweep.loc[best_myth_idx, "alpha_lgbm"]
    p_best = np.clip(a_best * p_lgbm + (1.0 - a_best) * p_xfr, 1e-7, 1 - 1e-7)

    slice_rows = []
    for name, mask in slices.items():
        n = int(mask.sum())
        if n < 100:
            slice_rows.append({"slice": name, "n": n, "skipped": True})
            continue
        slice_rows.append({
            "slice": name,
            "n": n,
            "auc": float(roc_auc_score(y[mask], p_best[mask])),
            "logloss": float(log_loss(y[mask], p_best[mask])),
            "auc_lgbm_only":  float(roc_auc_score(y[mask], p_lgbm[mask])),
            "auc_xfr_only":   float(roc_auc_score(y[mask], p_xfr[mask])),
            "ensemble_delta_vs_xfr_pp": float(100 * (roc_auc_score(y[mask], p_best[mask]) - roc_auc_score(y[mask], p_xfr[mask]))),
        })
    print()
    print(f"--- slice breakdown at alpha_lgbm = {a_best:.3f} (best-on-Mythic+) ---")
    for r in slice_rows:
        if r.get("skipped"):
            continue
        print(f"  {r['slice']:<32s} n={r['n']:>9,d}  ensemble={r['auc']:.5f}  "
              f"lgbm={r['auc_lgbm_only']:.5f}  xfr={r['auc_xfr_only']:.5f}  "
              f"Δ_vs_xfr={r['ensemble_delta_vs_xfr_pp']:+.3f} pp")

    out = {
        "model_a": "recommender_v2_phase1p4 (LGBM, Run F)",
        "model_b": "recommender_v3_phase1p2p4_xl (Transformer, Run K)",
        "cutoff": CUTOFF,
        "stable_test_after": STABLE_TEST_AFTER,
        "n_test_rows": int(len(test)),
        "n_test_battles": int(test["battle_id"].nunique()),
        "sweep": rows,
        "best_by_all_auc": {
            "alpha_lgbm": float(sweep.loc[best_all_idx, "alpha_lgbm"]),
            "auc_all": float(sweep.loc[best_all_idx, "auc_all"]),
            "auc_mythicplus": float(sweep.loc[best_all_idx, "auc_mythicplus"]),
        },
        "best_by_mythicplus_auc": {
            "alpha_lgbm": float(sweep.loc[best_myth_idx, "alpha_lgbm"]),
            "auc_all": float(sweep.loc[best_myth_idx, "auc_all"]),
            "auc_mythicplus": float(sweep.loc[best_myth_idx, "auc_mythicplus"]),
        },
        "slice_breakdown_at_best_mythicplus": slice_rows,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "ensemble_kitsink.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[ensemble] wrote {out_path} in {time.time()-t0:.1f}s total")


if __name__ == "__main__":
    main()
