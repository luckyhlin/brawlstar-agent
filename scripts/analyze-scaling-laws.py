"""
Scaling-law analysis for the Brawl Stars recommender v3 transformer family.

Given the 12+ trained models in `reports/` (vanilla, phase1, phase1+2, phase1+2+4,
mixed vs solo data), fit scaling laws of the Chinchilla form

    L(N, D) = E + A / N^alpha + B / D^beta

separately on (a) Mythic+ logloss and (b) Mythic+ (1 - AUC). Generate plots,
predict the AUC ceiling at the current data scale, and predict the param scale
that would saturate the existing 1.87 M-row training set on the Mythic+ slice.

Inputs:
  - reports/slices_summary.json: retrospective slice eval of the 7 pre-DEC-018 models
  - reports/recommender_*.json: per-run reports with stable_test_slices (post-DEC-016)
  - hardcoded param counts (pulled from logs/train_*.log -- see below)
Outputs:
  - reports/scaling_laws_inventory.csv: clean per-run inventory
  - reports/scaling_laws.json: fit results + predictions
  - reports/scaling_law_N_mythic.png: capacity-scaling curve on Mythic+
  - reports/scaling_law_N_all.png: capacity-scaling curve on all-test (more points)
  - reports/scaling_law_residuals.png: residual diagnostics

Run with:
  PYTHONPATH=src UV_CACHE_DIR=/media/lin/disk2/brawlstar-agent/.uv-cache-local \\
    uv run python scripts/analyze-scaling-laws.py
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"

# Hard-coded param counts (sum(p.numel()) for the transformer; lifted from
# logs/train_*.log lines like "params: 251,233"). LGBM has no comparable N.
PARAM_COUNTS: dict[str, int] = {
    "recommender_v3_default":            251_233,
    "recommender_v3_gpu":                251_233,
    "recommender_v3_gpu_fast":           251_233,
    "recommender_v3_big":                569_857,
    "recommender_v3_xl":                 3_275_265,
    "recommender_v3_phase1_default":     253_441,
    "recommender_v3_phase1_big":         572_801,
    "recommender_v3_phase1p2_default":   254_593,
    "recommender_v3_phase1p2_solo":      251_617,
    "recommender_v3_phase1p4_default":   254_593,
    "recommender_v3_phase1p2p4_default": 255_745,
    "recommender_v3_phase1p2p4_big":     575_873,
    "recommender_v3_phase1p2p4_xl":      3_287_297,
    # Anchor runs added Session 12 to give the kit-sink fit real DOF.
    # Verified by sum(p.numel() for p in _TransformerCore.parameters()).
    "recommender_v3_phase1p2p4_m1":      1_095_361,  # d=160 h=8 L=5 ff=320
    "recommender_v3_phase1p2p4_m2":      1_566_337,  # d=192 h=8 L=5 ff=384
    "recommender_v3_phase1p2p4_m3":      5_112_641,  # d=320 h=8 L=6 ff=640
}

# Human-friendly short labels for the v3 transformer family.
SHORT_LABELS: dict[str, str] = {
    "recommender_v3_default":            "small",
    "recommender_v3_big":                "big",
    "recommender_v3_xl":                 "XL",
    "recommender_v3_phase1_default":     "small+P1",
    "recommender_v3_phase1_big":         "big+P1",
    "recommender_v3_phase1p2_default":   "small+P1P2",
    "recommender_v3_phase1p2_solo":      "small+P1P2 solo",
    "recommender_v3_phase1p4_default":   "small+P1P4",
    "recommender_v3_phase1p2p4_default": "small+P1P2P4",
    "recommender_v3_phase1p2p4_big":     "big+P1P2P4",
    "recommender_v3_phase1p2p4_xl":      "XL+P1P2P4",
    "recommender_v3_phase1p2p4_m1":      "M1+P1P2P4 (1.1M)",
    "recommender_v3_phase1p2p4_m2":      "M2+P1P2P4 (1.6M)",
    "recommender_v3_phase1p2p4_m3":      "M3+P1P2P4 (5.1M)",
}


@dataclass
class Run:
    """One training run with everything we need for the scaling fit."""
    name: str
    family: str  # "transformer" or "lightgbm"
    short_label: str
    n_params: int | None  # None for LGBM
    n_train_rows: int
    battle_types: str  # "mixed" or "solo"
    phases: tuple[bool, bool, bool]  # (use_team_aggregates P1, use_time P2, use_history P4)
    auc_all: float
    auc_myth: float
    logloss_all: float
    logloss_myth: float
    brier_myth: float


def _phases_tag(p1: bool, p2: bool, p4: bool) -> str:
    parts = []
    if p1: parts.append("P1")
    if p2: parts.append("P2")
    if p4: parts.append("P4")
    return "+".join(parts) if parts else "vanilla"


def _phases_from_name(name: str) -> tuple[bool, bool, bool]:
    """Infer (P1, P2, P4) from the model filename — useful when the report
    itself doesn't store a `training` dict (e.g. v2/LGBM reports)."""
    body = name.lower()
    p1 = ("phase1" in body) or ("phase1p" in body)
    p2 = ("phase1p2" in body) or ("p2_" in body)
    p4 = ("phase4" in body) or ("phase1p4" in body) or ("p4" in body and "phase" in body)
    return (p1, p2, p4)


def _read_individual_report(path: Path) -> Run | None:
    """Pull a Run out of a `reports/recommender_*.json` file. Returns None if it
    isn't a recommender training report (e.g. summary/aux files).
    """
    if path.stem in {"slices_summary", "slices_smoke", "verify_bug"}:
        return None
    if path.stem.endswith("_topk"):
        return None
    if path.stem == "recommender_v1":
        return None
    if path.stem in {"recommender_v3_smoke", "recommender_v3_smoke2"}:
        return None
    try:
        with path.open() as f:
            r = json.load(f)
    except Exception:
        return None
    if "stable_test" not in r and "stable_test_slices" not in r:
        return None

    name = path.stem
    # Family by file prefix is robust; v3 reports also have model="TransformerTeamModel".
    family = "transformer" if name.startswith("recommender_v3") else "lightgbm"

    training = r.get("training", {})
    if family == "transformer":
        phases = (
            bool(training.get("use_team_aggregates", False)),
            bool(training.get("use_time_features", False)),
            bool(training.get("use_history_features", False)),
        )
        battle_types_field = training.get("battle_types", ["ranked", "soloRanked"])
        btype = "solo" if list(battle_types_field) == ["soloRanked"] else "mixed"
    else:
        # LGBM reports don't have a `training` dict; infer from filename.
        phases = _phases_from_name(name)
        btype = "solo" if name.endswith("_solo") else "mixed"

    n_train_rows = int(r.get("n_train_rows", 0))
    n_params = PARAM_COUNTS.get(name) if family == "transformer" else None

    # Slice key: LGBM reports nest under "LightGBM" alongside Global/Mode/etc.
    # Transformer reports key only under "Transformer".
    slice_key = "Transformer" if family == "transformer" else "LightGBM"
    slices = r.get("stable_test_slices", {}).get(slice_key, {})
    if slices and "soloRanked_mythicplus" in slices and "all" in slices:
        myth = slices["soloRanked_mythicplus"]
        allslc = slices["all"]
        auc_myth = float(myth["auc"])
        auc_all = float(allslc["auc"])
        ll_myth = float(myth["logloss"])
        ll_all = float(allslc["logloss"])
        brier_myth = float(myth["brier"])
    else:
        # Pre-DEC-016 report -- skip; we'll pull from slices_summary instead.
        return None

    short_label = SHORT_LABELS.get(name, name.replace("recommender_", ""))

    return Run(
        name=name,
        family=family,
        short_label=short_label,
        n_params=n_params,
        n_train_rows=n_train_rows,
        battle_types=btype,
        phases=phases,
        auc_all=auc_all,
        auc_myth=auc_myth,
        logloss_all=ll_all,
        logloss_myth=ll_myth,
        brier_myth=brier_myth,
    )


def _read_slices_summary() -> list[Run]:
    """Reads `reports/slices_summary.json` (retrospective slice eval of the
    7 pre-DEC-018 saved models, all trained on mixed 1.87 M rows)."""
    path = REPORTS_DIR / "slices_summary.json"
    with path.open() as f:
        summary = json.load(f)
    n_train_rows_assumed = 1_871_616  # all 7 used A_fair mixed training
    out: list[Run] = []
    for full_name, blob in summary.get("models", {}).items():
        name = full_name.split("/")[-1]
        slices = blob.get("slices", {})
        if "soloRanked_mythicplus" not in slices or "all" not in slices:
            continue
        family = "transformer" if blob.get("kind") == "transformer" else "lightgbm"
        n_params = PARAM_COUNTS.get(name) if family == "transformer" else None
        phases = _phases_from_name(name)
        out.append(Run(
            name=name,
            family=family,
            short_label=SHORT_LABELS.get(name, name.replace("recommender_", "")),
            n_params=n_params,
            n_train_rows=n_train_rows_assumed,
            battle_types="mixed",
            phases=phases,
            auc_all=float(slices["all"]["auc"]),
            auc_myth=float(slices["soloRanked_mythicplus"]["auc"]),
            logloss_all=float(slices["all"]["logloss"]),
            logloss_myth=float(slices["soloRanked_mythicplus"]["logloss"]),
            brier_myth=float(slices["soloRanked_mythicplus"]["brier"]),
        ))
    return out


def load_inventory() -> pd.DataFrame:
    """Combine slices_summary (retrospective, pre-DEC-018) with newer individual
    reports that already have a stable_test_slices block."""
    runs_by_name: dict[str, Run] = {}
    # 1. Older models via the summary
    for r in _read_slices_summary():
        runs_by_name[r.name] = r
    # 2. Newer models via their own reports. Overrides the summary entry if both
    #    exist (the individual report tends to be richer + always has the right
    #    n_train_rows for solo runs).
    for p in sorted(REPORTS_DIR.glob("recommender_*.json")):
        r = _read_individual_report(p)
        if r is None:
            continue
        # Skip recommender_v2_default fair (random split, pre-DEC-011) -- we
        # only want the post-DEC-011 LGBM saves.
        if r.name in {"recommender_v2_default", "recommender_v2_30d"}:
            continue
        runs_by_name[r.name] = r

    rows = []
    for r in runs_by_name.values():
        rows.append({
            "name": r.name,
            "family": r.family,
            "short_label": r.short_label,
            "n_params": r.n_params,
            "n_train_rows": r.n_train_rows,
            "battle_types": r.battle_types,
            "phases_tag": _phases_tag(*r.phases),
            "auc_all": r.auc_all,
            "auc_myth": r.auc_myth,
            "logloss_all": r.logloss_all,
            "logloss_myth": r.logloss_myth,
            "brier_myth": r.brier_myth,
        })
    df = pd.DataFrame(rows).sort_values(["family", "battle_types", "phases_tag", "n_params"])
    return df.reset_index(drop=True)


# ---- scaling-law fitting ------------------------------------------------


def fit_power_law_1d(N: np.ndarray, L: np.ndarray) -> dict:
    """Fit L(N) = E + A * N^(-alpha) via least-squares on (E, log A, alpha).

    Use parameterization x = (E, log_A, alpha) so all are unconstrained reals.
    Bound alpha in (0, 2] (positive scaling, sanity).
    """
    N = np.asarray(N, dtype=float)
    L = np.asarray(L, dtype=float)

    def resid(x):
        E, log_A, alpha = x
        A = math.exp(log_A)
        pred = E + A * N ** (-alpha)
        return pred - L

    # Reasonable initial guess: E ~ min(L) - small, A * Nmid^-0.3 ~ range(L)
    x0 = [float(np.min(L)) * 0.95, math.log(0.3), 0.3]
    res = least_squares(
        resid, x0=x0,
        bounds=([-np.inf, -20.0, 1e-3], [float(np.min(L)), 20.0, 2.0]),
        method="trf",
        max_nfev=20000,
    )
    E, log_A, alpha = res.x
    A = math.exp(log_A)
    pred = E + A * N ** (-alpha)
    rss = float(np.sum((pred - L) ** 2))
    n = len(N)
    dof = max(n - 3, 0)
    rmse = math.sqrt(rss / max(n, 1))
    return {
        "form": "L(N) = E + A * N^(-alpha)",
        "E": float(E),
        "A": float(A),
        "alpha": float(alpha),
        "rmse": rmse,
        "rss": rss,
        "n_points": n,
        "dof": dof,
        "fitted": pred.tolist(),
        "observed": L.tolist(),
        "N": N.tolist(),
        "ok": bool(res.success),
    }


def fit_chinchilla_joint(N: np.ndarray, D: np.ndarray, L: np.ndarray) -> dict:
    """Joint fit L(N, D) = E + A*N^(-alpha) + B*D^(-beta).

    5 params, sparse data: only meaningful if we have at least one D variation.
    """
    N = np.asarray(N, dtype=float)
    D = np.asarray(D, dtype=float)
    L = np.asarray(L, dtype=float)

    def resid(x):
        E, log_A, alpha, log_B, beta = x
        A = math.exp(log_A); B = math.exp(log_B)
        pred = E + A * N ** (-alpha) + B * D ** (-beta)
        return pred - L

    x0 = [float(np.min(L)) * 0.95, math.log(0.3), 0.3, math.log(0.1), 0.2]
    res = least_squares(
        resid, x0=x0,
        bounds=([-np.inf, -20.0, 1e-3, -20.0, 1e-3], [float(np.min(L)), 20.0, 2.0, 20.0, 2.0]),
        method="trf",
        max_nfev=20000,
    )
    E, log_A, alpha, log_B, beta = res.x
    A = math.exp(log_A); B = math.exp(log_B)
    pred = E + A * N ** (-alpha) + B * D ** (-beta)
    rss = float(np.sum((pred - L) ** 2))
    n = len(N)
    return {
        "form": "L(N, D) = E + A*N^(-alpha) + B*D^(-beta)",
        "E": float(E),
        "A": float(A),
        "alpha": float(alpha),
        "B": float(B),
        "beta": float(beta),
        "rmse": math.sqrt(rss / max(n, 1)),
        "rss": rss,
        "n_points": n,
        "dof": max(n - 5, 0),
        "fitted": pred.tolist(),
        "observed": L.tolist(),
        "N": N.tolist(),
        "D": D.tolist(),
        "ok": bool(res.success),
    }


# ---- plotting -----------------------------------------------------------


def _plot_capacity_curve(
    runs_for_fit: list[Run], fit: dict, all_runs: list[Run],
    metric: str, out_path: Path, title: str,
    outlier_names: set[str] | None = None,
) -> None:
    """One-axis log-N plot with the fitted curve + every run as a dot.

    metric: "logloss_myth" or "one_minus_auc_myth" -- which quantity to plot.
    outlier_names: model names to mark distinctly (rendered as an X with an
                  "(outlier)" annotation; e.g. M3 was trained at batch=2048
                  due to GPU memory limits, making it not a clean N-scaling
                  comparison).
    """
    outlier_names = outlier_names or set()
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    N_curve = np.geomspace(1e5, 5e7, 200)
    L_curve = fit["E"] + fit["A"] * N_curve ** (-fit["alpha"])

    fam_colors = {
        "vanilla": "#0EA5E9",
        "P1":      "#22C55E",
        "P1+P2":   "#A855F7",
        "P1+P2+P4":"#EF4444",
        "P1+P4":   "#F59E0B",
    }
    ax.plot(N_curve, L_curve, "-", color="#9CA3AF", lw=1.5,
            label=f"fit: E + A·N^(-α)  (E={fit['E']:.4f}, A={fit['A']:.3g}, α={fit['alpha']:.3f})")

    # Track plotted y-values so we can manually pad the y-axis to include outliers
    ys = []
    for r in all_runs:
        if r.family != "transformer" or r.n_params is None:
            continue
        if r.battle_types == "solo":
            continue
        tag = _phases_tag(*r.phases)
        y = _value_for_metric(r, metric)
        ys.append(y)
        is_outlier = r.name in outlier_names
        if is_outlier:
            ax.scatter(r.n_params, y, s=200, c=fam_colors.get(tag, "#666"),
                       marker="X", edgecolor="black", linewidth=1.2, zorder=5,
                       label=f"outlier ({r.short_label})"
                       if "outlier" not in ax.get_legend_handles_labels()[1] else None)
        else:
            ax.scatter(r.n_params, y, s=90, c=fam_colors.get(tag, "#666"),
                       edgecolor="black", linewidth=0.5, zorder=3,
                       label=tag if tag not in ax.get_legend_handles_labels()[1] else None)
        ax.annotate(r.short_label, (r.n_params, y),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=8, color="#374151")
    for r in runs_for_fit:
        y = _value_for_metric(r, metric)
        ax.scatter(r.n_params, y, s=180, facecolor="none",
                   edgecolor="red", linewidth=1.5, zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(1.5e5, 5e7)
    ax.set_xlabel("Trainable parameters N (log)")
    ax.set_ylabel({"logloss_myth": "Mythic+ log-loss",
                   "one_minus_auc_myth": "Mythic+ (1 − AUC)",
                   "logloss_all": "All-test log-loss",
                   "one_minus_auc_all": "All-test (1 − AUC)"}[metric])
    ax.set_title(title)
    ax.axhline(fit["E"], color="#9CA3AF", linestyle=":", lw=1,
               label=f"asymptote at N → ∞: {fit['E']:.4f}")
    if ys:
        ymin = min(ys + [fit["E"]]) - 0.005
        ymax = max(ys) + 0.005
        ax.set_ylim(ymin, ymax)
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    keep = []
    for h, l in zip(handles, labels):
        if l in seen: continue
        seen.add(l); keep.append((h, l))
    ax.legend(*zip(*keep), fontsize=8, loc="upper right", framealpha=0.9)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _value_for_metric(r: Run, metric: str) -> float:
    if metric == "logloss_myth":
        return r.logloss_myth
    if metric == "logloss_all":
        return r.logloss_all
    if metric == "one_minus_auc_myth":
        return 1.0 - r.auc_myth
    if metric == "one_minus_auc_all":
        return 1.0 - r.auc_all
    raise ValueError(metric)


# ---- predictions --------------------------------------------------------


def _empirical_auc_from_logloss(L_pred: float, runs: list[Run], metric_logloss: str, metric_auc: str) -> float:
    """Given a predicted log-loss, interpolate to the corresponding AUC using
    the empirical (log-loss, AUC) pairs. Lower logloss ⇒ higher AUC, so we
    sort by logloss ascending and 1-AUC ascending; both should monotonically
    decrease together (the better calibrated model usually also ranks better).
    """
    pairs = sorted(
        ((_value_for_metric(r, metric_logloss),
          1.0 - r.auc_myth if "myth" in metric_logloss else 1.0 - r.auc_all)
         for r in runs if r.family == "transformer"),
        key=lambda p: p[0],
    )
    xs = np.array([p[0] for p in pairs])
    ys = np.array([p[1] for p in pairs])  # 1-AUC, monotone with logloss (mostly)
    if L_pred <= xs[0]:
        # Extrapolate using the slope of the LOWEST-logloss pair (best models)
        if len(xs) >= 2:
            slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
            one_minus_auc = ys[0] + slope * (L_pred - xs[0])
        else:
            one_minus_auc = ys[0]
    elif L_pred >= xs[-1]:
        if len(xs) >= 2:
            slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
            one_minus_auc = ys[-1] + slope * (L_pred - xs[-1])
        else:
            one_minus_auc = ys[-1]
    else:
        one_minus_auc = float(np.interp(L_pred, xs, ys))
    return float(1.0 - one_minus_auc)


def predict_curve(fit: dict, N_targets: list[int]) -> list[dict]:
    out = []
    for N in N_targets:
        L = fit["E"] + fit["A"] * N ** (-fit["alpha"])
        out.append({"N": N, "L_pred": float(L)})
    return out


def main() -> None:
    inv = load_inventory()
    inv_path = REPORTS_DIR / "scaling_laws_inventory.csv"
    inv.to_csv(inv_path, index=False)
    print(f"wrote {inv_path} with {len(inv)} runs")
    print()
    print(inv.to_string(index=False))
    print()

    # -------- pick fitting subsets ----------
    # Subset A (most data-honest single-family curve): VANILLA v3 transformers,
    # mixed training, no-phase. (3 N points along the same D = 1.87M.)
    vanilla = inv[(inv.family == "transformer") & (inv.phases_tag == "vanilla") &
                  (inv.battle_types == "mixed")].copy()
    # Drop the gpu / gpu_fast small reruns to avoid double-counting -- they're
    # the SAME architecture as default; keep just 'default' as the 251k anchor.
    vanilla = vanilla[~vanilla.name.isin(["recommender_v3_gpu", "recommender_v3_gpu_fast"])]
    vanilla = vanilla.sort_values("n_params").reset_index(drop=True)

    # Subset B: kitchen-sink (phase1+2+4) transformers, mixed training. The
    # current state-of-the-art family. We EXCLUDE the M3 anchor from the fit
    # because it was trained with batch=2048 (others use batch=4096) due to
    # GPU memory limits on the RTX 3060 Mobile, and the recipe shift makes
    # its result not a clean N-scaling comparison. M3 is reported as an
    # observed outlier; the curve is fit on the 5 comparable batch=4096 runs.
    KITSINK_FIT_EXCLUDED = {"recommender_v3_phase1p2p4_m3"}
    kitsink_all = inv[(inv.family == "transformer") & (inv.phases_tag == "P1+P2+P4") &
                      (inv.battle_types == "mixed")].copy().sort_values("n_params").reset_index(drop=True)
    kitsink = kitsink_all[~kitsink_all["name"].isin(KITSINK_FIT_EXCLUDED)].copy().reset_index(drop=True)

    print("=" * 72)
    print("SUBSET A: vanilla v3 transformers, mixed training (no phase features)")
    print("=" * 72)
    print(vanilla[["short_label", "n_params", "n_train_rows", "auc_myth", "logloss_myth"]].to_string(index=False))
    print()
    print("SUBSET B: kitchen-sink (P1+P2+P4) transformers, mixed training")
    print(kitsink[["short_label", "n_params", "n_train_rows", "auc_myth", "logloss_myth"]].to_string(index=False))

    # -------- fits ----------
    fits = {}
    # We always fit *both* metrics: logloss (the training objective) and (1-AUC)
    # (the deployment ranking metric).
    for name, df_sub in [("vanilla", vanilla), ("kitsink", kitsink)]:
        if len(df_sub) < 3:
            print(f"skip fit '{name}': only {len(df_sub)} points")
            continue
        N = df_sub["n_params"].to_numpy()
        L_ll = df_sub["logloss_myth"].to_numpy()
        L_au = (1.0 - df_sub["auc_myth"]).to_numpy()
        # also all-test for cross-check
        L_ll_all = df_sub["logloss_all"].to_numpy()
        L_au_all = (1.0 - df_sub["auc_all"]).to_numpy()
        fits[name] = {
            "subset_size": int(len(df_sub)),
            "myth_logloss": fit_power_law_1d(N, L_ll),
            "myth_one_minus_auc": fit_power_law_1d(N, L_au),
            "all_logloss":  fit_power_law_1d(N, L_ll_all),
            "all_one_minus_auc": fit_power_law_1d(N, L_au_all),
        }
        print()
        print(f"--- {name} fit: Mythic+ logloss ---")
        for k in ("E", "A", "alpha", "rmse", "dof"):
            print(f"  {k} = {fits[name]['myth_logloss'][k]}")
        print(f"--- {name} fit: Mythic+ (1-AUC) ---")
        for k in ("E", "A", "alpha", "rmse", "dof"):
            print(f"  {k} = {fits[name]['myth_one_minus_auc'][k]}")

    # -------- joint Chinchilla (illustrative; very few D-points) -----------
    # Exclude M3 from the joint fit (batch=2048 anomaly — see kitsink note above).
    transformer_df = inv[(inv.family == "transformer") & (inv.n_params.notna()) &
                         (~inv["name"].isin(KITSINK_FIT_EXCLUDED))].copy()
    if len(transformer_df) >= 5:
        N = transformer_df["n_params"].to_numpy()
        D = transformer_df["n_train_rows"].to_numpy()
        L_ll = transformer_df["logloss_myth"].to_numpy()
        L_au = (1.0 - transformer_df["auc_myth"].to_numpy())
        joint_ll = fit_chinchilla_joint(N, D, L_ll)
        joint_au = fit_chinchilla_joint(N, D, L_au)
        fits["joint_all_transformers_myth_logloss"] = joint_ll
        fits["joint_all_transformers_myth_one_minus_auc"] = joint_au
        print()
        print("--- joint Chinchilla fit (all 11 transformer runs, Mythic+ logloss) ---")
        for k in ("E", "A", "alpha", "B", "beta", "rmse", "dof"):
            print(f"  {k} = {joint_ll[k]}")
        print()
        print("--- joint Chinchilla fit (all 11 transformer runs, Mythic+ 1-AUC) ---")
        for k in ("E", "A", "alpha", "B", "beta", "rmse", "dof"):
            print(f"  {k} = {joint_au[k]}")
        # Relative contributions at the XL+P1P2P4 SOTA point (N=3.29M, D=1.87M).
        Nsota, Dsota = 3_287_297.0, 1_871_616.0
        capacity_term_ll = joint_ll["A"] * Nsota ** (-joint_ll["alpha"])
        data_term_ll     = joint_ll["B"] * Dsota ** (-joint_ll["beta"])
        capacity_term_au = joint_au["A"] * Nsota ** (-joint_au["alpha"])
        data_term_au     = joint_au["B"] * Dsota ** (-joint_au["beta"])
        fits["joint_sota_decomposition"] = {
            "N_sota": Nsota,
            "D_sota": Dsota,
            "logloss": {
                "irreducible_E": joint_ll["E"],
                "capacity_term_A_over_N^alpha": float(capacity_term_ll),
                "data_term_B_over_D^beta": float(data_term_ll),
                "total_predicted": float(joint_ll["E"] + capacity_term_ll + data_term_ll),
                "observed_xl_kitsink": 0.67254,
                "capacity_share_of_reducible": float(capacity_term_ll / (capacity_term_ll + data_term_ll)),
                "data_share_of_reducible": float(data_term_ll / (capacity_term_ll + data_term_ll)),
            },
            "one_minus_auc": {
                "irreducible_E": joint_au["E"],
                "capacity_term_A_over_N^alpha": float(capacity_term_au),
                "data_term_B_over_D^beta": float(data_term_au),
                "total_predicted": float(joint_au["E"] + capacity_term_au + data_term_au),
                "observed_xl_kitsink_one_minus_auc": 1.0 - 0.61804,
                "capacity_share_of_reducible": float(capacity_term_au / (capacity_term_au + data_term_au)),
                "data_share_of_reducible": float(data_term_au / (capacity_term_au + data_term_au)),
            },
        }
        print()
        print("--- joint-fit SOTA decomposition (N=3.29M, D=1.87M) ---")
        print(f"  logloss:    E={joint_ll['E']:.4f}, +cap={capacity_term_ll:.4f}, +data={data_term_ll:.4f}, ")
        print(f"              => capacity share of reducible = {capacity_term_ll/(capacity_term_ll+data_term_ll):.1%}")
        print(f"  1-AUC:      E={joint_au['E']:.4f}, +cap={capacity_term_au:.4f}, +data={data_term_au:.4f}, ")
        print(f"              => capacity share of reducible = {capacity_term_au/(capacity_term_au+data_term_au):.1%}")
        # "What if we doubled D?" projection
        D_targets = [Dsota, 2 * Dsota, 4 * Dsota, 8 * Dsota]
        D_proj = []
        for Dt in D_targets:
            L_ll_pred = joint_ll["E"] + capacity_term_ll + joint_ll["B"] * Dt ** (-joint_ll["beta"])
            L_au_pred = joint_au["E"] + capacity_term_au + joint_au["B"] * Dt ** (-joint_au["beta"])
            D_proj.append({"D": float(Dt), "D_x_over_current": float(Dt/Dsota),
                           "myth_logloss_pred": float(L_ll_pred),
                           "myth_auc_pred": float(1.0 - L_au_pred)})
        fits["joint_data_projection_at_xl"] = D_proj
        print()
        print("--- joint-fit data-scaling projection at fixed N=3.29M (XL) ---")
        for p in D_proj:
            print(f"  D={p['D']:>10,.0f} ({p['D_x_over_current']:.1f}×): "
                  f"AUC ≈ {p['myth_auc_pred']:.5f}, logloss ≈ {p['myth_logloss_pred']:.5f}")

    # -------- predictions ----------
    N_TARGETS = [251_233, 569_857, 1_000_000, 1_500_000, 3_287_297, 5_000_000, 10_000_000, 20_000_000, 100_000_000]
    # use kitsink fit (preferred — it's the most current SOTA family)
    if "kitsink" in fits:
        ks = fits["kitsink"]
        all_runs = [Run(
            name=row["name"], family=row["family"], short_label=row["short_label"],
            n_params=row["n_params"] if pd.notna(row["n_params"]) else None,
            n_train_rows=row["n_train_rows"], battle_types=row["battle_types"],
            phases=(row["phases_tag"].count("P1") > 0,
                    row["phases_tag"].count("P2") > 0,
                    row["phases_tag"].count("P4") > 0),
            auc_all=row["auc_all"], auc_myth=row["auc_myth"],
            logloss_all=row["logloss_all"], logloss_myth=row["logloss_myth"],
            brier_myth=row["brier_myth"],
        ) for _, row in inv.iterrows()]

        preds = []
        for N in N_TARGETS:
            L_ll = ks["myth_logloss"]["E"] + ks["myth_logloss"]["A"] * N ** (-ks["myth_logloss"]["alpha"])
            L_au = ks["myth_one_minus_auc"]["E"] + ks["myth_one_minus_auc"]["A"] * N ** (-ks["myth_one_minus_auc"]["alpha"])
            preds.append({
                "N": int(N),
                "myth_logloss_pred": float(L_ll),
                "myth_auc_pred_from_oneminus": float(1.0 - L_au),
                "myth_auc_pred_from_logloss_interp": _empirical_auc_from_logloss(
                    L_ll, all_runs, "logloss_myth", "one_minus_auc_myth"),
            })
        fits["predictions_kitsink_Mythic+"] = preds
        print()
        print("--- predictions (kitsink fit on Mythic+) ---")
        for p in preds:
            print(f"  N={p['N']:>12,d} → logloss {p['myth_logloss_pred']:.5f}, "
                  f"AUC ≈ {p['myth_auc_pred_from_oneminus']:.5f} "
                  f"(via 1-AUC fit) / {p['myth_auc_pred_from_logloss_interp']:.5f} (via logloss interp)")

        # AUC ceiling on Mythic+ (kitchen-sink fit, fixed D = 1.87M)
        auc_ceiling_myth = 1.0 - ks["myth_one_minus_auc"]["E"]
        fits["asymptote_kitsink_Mythic+"] = {
            "auc_at_N_inf": float(auc_ceiling_myth),
            "logloss_at_N_inf": float(ks["myth_logloss"]["E"]),
            "note": "Asymptote at infinite N, fixed D = 1.87 M training rows. This is the irreducible loss/AUC the current data distribution can support with this architecture family.",
        }
        # how-far-from-ceiling at SOTA N=3.29M
        L_au_xl = 1.0 - 0.61804
        ceiling_gap_at_xl = (L_au_xl - ks["myth_one_minus_auc"]["E"]) / L_au_xl
        fits["asymptote_kitsink_Mythic+"]["xl_remaining_gap_pp"] = float(100 * (1.0 - 0.61804 - ks["myth_one_minus_auc"]["E"]))
        fits["asymptote_kitsink_Mythic+"]["xl_pct_of_gap_closed"] = float(100 * (1 - ceiling_gap_at_xl))
        print()
        print(f"--- asymptote (kitsink fit on Mythic+, fixed D=1.87M) ---")
        print(f"  AUC ceiling at N → ∞: {auc_ceiling_myth:.5f}")
        print(f"  Logloss floor at N → ∞: {ks['myth_logloss']['E']:.5f}")
        print(f"  At SOTA N=3.29M: AUC=0.6180; remaining ceiling gap = "
              f"{100*(1-0.61804-ks['myth_one_minus_auc']['E']):.3f} pp")
        print(f"  Pct of gap-from-random closed by XL: "
              f"{100*(0.61804 - 0.5)/(auc_ceiling_myth - 0.5):.1f}%")

        # Inverse question: what N gives AUC X, holding D=1.87M?
        AUC_TARGETS = [0.620, 0.625, 0.630, 0.635, 0.640, 0.650]
        inv_preds = []
        for target in AUC_TARGETS:
            target_one_minus_auc = 1.0 - target
            # L = E + A * N^(-alpha) → N = (A / (L - E))^(1/alpha)
            E = ks["myth_one_minus_auc"]["E"]
            A = ks["myth_one_minus_auc"]["A"]
            alpha = ks["myth_one_minus_auc"]["alpha"]
            if target_one_minus_auc <= E:
                # unreachable -- past asymptote
                inv_preds.append({"target_auc": target, "N_required": None,
                                  "note": f"unreachable (above asymptote {1-E:.5f})"})
                continue
            N_req = (A / (target_one_minus_auc - E)) ** (1.0 / alpha)
            inv_preds.append({"target_auc": target, "N_required": float(N_req),
                              "x_over_xl": float(N_req / 3_287_297)})
        fits["inverse_predictions_kitsink_Mythic+"] = inv_preds
        print()
        print("--- inverse: N required to reach Mythic+ AUC target (kitsink fit) ---")
        for p in inv_preds:
            if p["N_required"] is None:
                print(f"  AUC {p['target_auc']}: {p['note']}")
            else:
                print(f"  AUC {p['target_auc']}: N = {p['N_required']:,.0f} ({p['x_over_xl']:.1f}× current XL 3.29M)")

    # -------- save fits ----------
    out_json = REPORTS_DIR / "scaling_laws.json"
    with out_json.open("w") as f:
        json.dump(fits, f, indent=2)
    print(f"\nwrote {out_json}")

    # -------- plots ----------
    runs_all = []
    for _, row in inv.iterrows():
        if row["family"] != "transformer" or pd.isna(row["n_params"]):
            continue
        runs_all.append(Run(
            name=row["name"], family=row["family"], short_label=row["short_label"],
            n_params=int(row["n_params"]),
            n_train_rows=int(row["n_train_rows"]),
            battle_types=row["battle_types"],
            phases=(row["phases_tag"].count("P1") > 0,
                    row["phases_tag"].count("P2") > 0,
                    row["phases_tag"].count("P4") > 0),
            auc_all=row["auc_all"], auc_myth=row["auc_myth"],
            logloss_all=row["logloss_all"], logloss_myth=row["logloss_myth"],
            brier_myth=row["brier_myth"],
        ))

    runs_kitsink = [r for r in runs_all if _phases_tag(*r.phases) == "P1+P2+P4"
                    and r.battle_types == "mixed"
                    and r.name not in KITSINK_FIT_EXCLUDED]
    if "kitsink" in fits:
        _plot_capacity_curve(
            runs_for_fit=runs_kitsink,
            fit=fits["kitsink"]["myth_logloss"],
            all_runs=runs_all,
            metric="logloss_myth",
            out_path=REPORTS_DIR / "scaling_law_N_mythic_logloss.png",
            title="Mythic+ log-loss vs N (Chinchilla-style fit on kitchen-sink runs)",
            outlier_names=KITSINK_FIT_EXCLUDED,
        )
        _plot_capacity_curve(
            runs_for_fit=runs_kitsink,
            fit=fits["kitsink"]["myth_one_minus_auc"],
            all_runs=runs_all,
            metric="one_minus_auc_myth",
            out_path=REPORTS_DIR / "scaling_law_N_mythic_auc.png",
            title="Mythic+ (1 − AUC) vs N (Chinchilla-style fit on kitchen-sink runs)",
            outlier_names=KITSINK_FIT_EXCLUDED,
        )
        _plot_capacity_curve(
            runs_for_fit=runs_kitsink,
            fit=fits["kitsink"]["all_one_minus_auc"],
            all_runs=runs_all,
            metric="one_minus_auc_all",
            out_path=REPORTS_DIR / "scaling_law_N_all_auc.png",
            title="All-test (1 − AUC) vs N (Chinchilla-style fit on kitchen-sink runs)",
            outlier_names=KITSINK_FIT_EXCLUDED,
        )

    # data-axis cross-section: small mixed vs small solo, P1+P2 (D ratio 5.4×)
    small_mixed = inv[inv.name == "recommender_v3_phase1p2_default"]
    small_solo  = inv[inv.name == "recommender_v3_phase1p2_solo"]
    if len(small_mixed) and len(small_solo):
        D_mixed = float(small_mixed["n_train_rows"].iloc[0])
        D_solo  = float(small_solo ["n_train_rows"].iloc[0])
        L_mixed_ll = float(small_mixed["logloss_myth"].iloc[0])
        L_solo_ll  = float(small_solo ["logloss_myth"].iloc[0])
        # 2-point implied beta: L = E + B/D^beta  → if E is from the kitsink fit
        E_pin = fits["kitsink"]["myth_logloss"]["E"] if "kitsink" in fits else 0.665
        beta_implied = math.log((L_solo_ll - E_pin) / (L_mixed_ll - E_pin)) / math.log(D_mixed / D_solo)
        # also AUC analog
        L_mixed_au = 1.0 - float(small_mixed["auc_myth"].iloc[0])
        L_solo_au  = 1.0 - float(small_solo ["auc_myth"].iloc[0])
        E_pin_au = fits["kitsink"]["myth_one_minus_auc"]["E"] if "kitsink" in fits else 0.35
        beta_implied_au = math.log((L_solo_au - E_pin_au) / (L_mixed_au - E_pin_au)) / math.log(D_mixed / D_solo)
        fits["data_scaling_2point"] = {
            "small_mixed": {"D": D_mixed, "logloss_myth": L_mixed_ll, "auc_myth": float(small_mixed["auc_myth"].iloc[0])},
            "small_solo":  {"D": D_solo,  "logloss_myth": L_solo_ll,  "auc_myth": float(small_solo ["auc_myth"].iloc[0])},
            "implied_beta_logloss_with_kitsink_E": float(beta_implied),
            "implied_beta_oneminusauc_with_kitsink_E": float(beta_implied_au),
            "ratio_D_mixed_over_D_solo": D_mixed / D_solo,
            "note": "These are 2-point implied exponents pinning E at the kitchen-sink-N-fit asymptote. They are NOT statistically estimated (0 d.o.f.).",
        }
        print()
        print(f"--- 2-point implied data exponent ---")
        print(f"  D_mixed/D_solo = {D_mixed/D_solo:.3f}")
        print(f"  beta (logloss) = {beta_implied:.4f}")
        print(f"  beta (1-AUC)   = {beta_implied_au:.4f}")

    # Re-save fits with the data-scaling block added.
    with out_json.open("w") as f:
        json.dump(fits, f, indent=2)


if __name__ == "__main__":
    main()
