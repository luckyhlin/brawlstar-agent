#!/usr/bin/env python3
"""Generate plots and detailed analysis from a trained recommender.

Outputs (under reports/recommender_v1/):
    fig_auc_bars.png            ← AUC by model (random vs temporal)
    fig_per_mode_auc.png        ← per-mode AUC heatmap
    fig_temporal_cv.png         ← AUC over time per fold
    fig_lgb_importance.png      ← top LightGBM feature importances
    fig_top_brawlers_<mode>.png ← LightGBM marginal P(win) ranking per mode
    damian_deepdive.json        ← support / win-rate diagnostics for DAMIAN
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from brawlstar_agent.recommender import (  # noqa: E402
    load_clean_battles,
    load_brawler_names,
    split_random,
)
from brawlstar_agent.recommender.baselines import (  # noqa: E402
    GlobalWilsonBaseline,
    ModeMapWilsonBaseline,
)
from brawlstar_agent.recommender.team_model import LGBMTeamModel  # noqa: E402
from brawlstar_agent.recommender.inference import rank_brawlers_for_map  # noqa: E402

OUT = REPO / "reports" / "recommender_v1"
OUT.mkdir(parents=True, exist_ok=True)


def plot_auc_bars(report: dict) -> None:
    """Random vs temporal AUC by model."""
    rs = report["random_split"]
    rs_models = list(rs.keys())
    rs_auc = [rs[m]["auc"] for m in rs_models]

    cv_records = report.get("temporal_cv", [])
    cv_df = pd.DataFrame(cv_records)
    cv_models = sorted(cv_df["model"].unique()) if not cv_df.empty else []
    cv_auc = [float(cv_df[cv_df["model"] == m]["auc"].mean()) for m in cv_models]

    fig, ax = plt.subplots(figsize=(10, 5))
    x_rs = np.arange(len(rs_models))
    width = 0.4
    bars1 = ax.bar(x_rs - width/2, rs_auc, width, label="Random split", color="#3b82f6")
    # Align temporal-CV bars with the matching model index
    cv_aligned = [float(cv_df[cv_df["model"] == m]["auc"].mean()) if m in cv_models else np.nan for m in rs_models]
    bars2 = ax.bar(x_rs + width/2, cv_aligned, width, label="Temporal CV (mean)", color="#f59e0b")

    ax.set_xticks(x_rs)
    ax.set_xticklabels(rs_models, rotation=15)
    ax.set_ylabel("AUC")
    ax.set_ylim(0.5, 0.78)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    ax.set_title("Win-prediction AUC: random vs temporal evaluation")
    ax.legend()
    for b, v in zip(bars1, rs_auc):
        ax.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    for b, v in zip(bars2, cv_aligned):
        if not np.isnan(v):
            ax.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_auc_bars.png", dpi=140)
    plt.close(fig)


def plot_per_mode_auc(report: dict) -> None:
    """Per-mode AUC heatmap (model × mode)."""
    pm = report.get("per_mode", {})
    if not pm:
        return
    models = list(pm.keys())
    modes = sorted({m for k in pm for m in pm[k].keys()})
    data = np.full((len(models), len(modes)), np.nan)
    for i, mname in enumerate(models):
        for j, mode in enumerate(modes):
            if mode in pm[mname]:
                data[i, j] = pm[mname][mode]["auc"]
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(modes)), 0.7 * len(models) + 1.5))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0.5, vmax=0.78, aspect="auto")
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels(modes, rotation=30, ha="right")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    for i in range(len(models)):
        for j in range(len(modes)):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        color="black" if v > 0.65 else "white", fontsize=9)
    ax.set_title("AUC per mode (random split)")
    fig.colorbar(im, ax=ax, label="AUC")
    fig.tight_layout()
    fig.savefig(OUT / "fig_per_mode_auc.png", dpi=140)
    plt.close(fig)


def plot_temporal_cv(report: dict) -> None:
    cv = pd.DataFrame(report.get("temporal_cv", []))
    if cv.empty:
        return
    cv["test_lo_dt"] = pd.to_datetime(cv["test_lo"], utc=True)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for model, sub in cv.sort_values("test_lo_dt").groupby("model"):
        ax.plot(sub["test_lo_dt"], sub["auc"], "-o", label=model)
    ax.set_ylabel("AUC")
    ax.set_xlabel("Test window start")
    ax.set_title("Temporal CV: AUC across sliding train→test windows")
    ax.set_ylim(0.5, 0.78)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT / "fig_temporal_cv.png", dpi=140)
    plt.close(fig)


def plot_lgb_importance(model: LGBMTeamModel, names: dict[int, str]) -> None:
    """Top 25 LightGBM feature importances (gain)."""
    booster = model.model
    f = model.featurizer
    feat_names = []
    bn = sorted(f.brawler_to_idx.keys())
    feat_names.extend(f"a_{names.get(b, b)}" for b in bn)
    feat_names.extend(f"b_{names.get(b, b)}" for b in bn)
    feat_names.extend(["mode_idx", "map_idx", "btype_idx",
                       "a_trophy_log", "b_trophy_log", "trophy_diff_log"])
    importances = booster.feature_importance(importance_type="gain")
    if len(importances) != len(feat_names):
        # Defensive truncate
        n = min(len(importances), len(feat_names))
        importances = importances[:n]; feat_names = feat_names[:n]
    order = np.argsort(importances)[::-1][:25]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(range(len(order)), [importances[i] for i in order][::-1], color="#0ea5e9")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feat_names[i] for i in order][::-1], fontsize=9)
    ax.set_xlabel("Gain")
    ax.set_title("LightGBM feature importance (top 25 by gain)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_lgb_importance.png", dpi=140)
    plt.close(fig)


def damian_deepdive(df: pd.DataFrame, names: dict[int, str], damian_id: int = 16000104) -> dict:
    """Why does the model rate DAMIAN so highly? Let's check raw stats."""
    rows = []
    for _, r in df.iterrows():
        if damian_id in r["team_a"]:
            rows.append((r["mode"], r["map"], 1, int(r["team_a_wins"]), float(r["team_a_trophies_mean"])))
        if damian_id in r["team_b"]:
            rows.append((r["mode"], r["map"], 1, 1 - int(r["team_a_wins"]), float(r["team_b_trophies_mean"])))
    if not rows:
        return {"name": names.get(damian_id, str(damian_id)), "support": 0}
    bdf = pd.DataFrame(rows, columns=["mode", "map", "n", "win", "trophy_mean"])
    overall_n = len(bdf)
    overall_wr = float(bdf["win"].mean())
    per_mode = bdf.groupby("mode").agg(n=("win", "size"), wr=("win", "mean"), tro=("trophy_mean", "mean")).round(4).reset_index()
    per_mode = per_mode.sort_values("n", ascending=False).head(8).to_dict(orient="records")

    # How DAMIAN compares to other top-N most-played brawlers
    all_players = []
    for _, r in df.iterrows():
        for b in r["team_a"]:
            all_players.append((int(b), int(r["team_a_wins"])))
        for b in r["team_b"]:
            all_players.append((int(b), 1 - int(r["team_a_wins"])))
    big = pd.DataFrame(all_players, columns=["brawler_id", "win"])
    rank = big.groupby("brawler_id").agg(n=("win", "size"), wr=("win", "mean")).round(4)
    rank = rank.sort_values("wr", ascending=False)
    rank["name"] = rank.index.map(lambda b: names.get(b, str(b)))
    rank = rank[rank["n"] >= 200].head(15)
    return {
        "name": names.get(damian_id, str(damian_id)),
        "support": int(overall_n),
        "overall_win_rate": overall_wr,
        "trophy_mean_when_played": float(bdf["trophy_mean"].mean()),
        "per_mode": per_mode,
        "top15_by_winrate": rank.reset_index().to_dict(orient="records"),
    }


def plot_top_brawlers_per_mode(model, train_df: pd.DataFrame, names: dict[int, str], modes: list[str]) -> None:
    """LightGBM marginal P(win) ranking per mode (averaged over maps via Monte Carlo)."""
    for mode in modes:
        # Pick top-3 most-played maps in mode; rank brawlers per map via MC
        cell_counts = (
            train_df[train_df["mode"] == mode]
            .groupby("map").size().sort_values(ascending=False).head(3)
        )
        if cell_counts.empty:
            continue
        fig, axes = plt.subplots(1, len(cell_counts), figsize=(5 * len(cell_counts), 5), sharey=True)
        if len(cell_counts) == 1:
            axes = [axes]
        for ax, (mp, _n) in zip(axes, cell_counts.items()):
            ranks = rank_brawlers_for_map(model, mode, mp, train_df=train_df, n_samples=60, seed=0)
            top10 = ranks[:10]
            labels = [names.get(b, str(b)) for b, _ in top10][::-1]
            scores = [s for _, s in top10][::-1]
            ax.barh(range(len(top10)), scores, color="#10b981")
            ax.set_yticks(range(len(top10)))
            ax.set_yticklabels(labels, fontsize=9)
            ax.set_xlim(0.4, max(0.85, max(scores) + 0.05))
            ax.axvline(0.5, color="gray", linestyle=":", linewidth=0.8)
            ax.set_title(f"{mode} / {mp}")
            ax.set_xlabel("P(win) marginal")
        fig.suptitle(f"Top-10 brawlers per map (LightGBM marginal)", y=1.02)
        fig.tight_layout()
        fig.savefig(OUT / f"fig_top_brawlers_{mode}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    report_path = REPO / "reports" / "recommender_v1.json"
    if not report_path.exists():
        print(f"Run scripts/train-recommender.py first; missing {report_path}")
        return
    with open(report_path) as f:
        report = json.load(f)

    print("Plotting AUC bars...")
    plot_auc_bars(report)
    print("Plotting per-mode AUC heatmap...")
    plot_per_mode_auc(report)
    print("Plotting temporal CV trace...")
    plot_temporal_cv(report)

    print("Loading data + retraining LightGBM for plots...")
    df = load_clean_battles().dropna(subset=["mode", "map", "battle_type"])
    train, _ = split_random(df, test_frac=0.2, seed=42)
    names = load_brawler_names()
    model = LGBMTeamModel(n_estimators=600, num_leaves=63, learning_rate=0.05, min_data_in_leaf=80).fit(train)

    print("Plotting LGBM importance...")
    plot_lgb_importance(model, names)

    print("Top brawlers per top-3 modes...")
    top_modes = (
        train.groupby("mode").size().sort_values(ascending=False).head(3).index.tolist()
    )
    plot_top_brawlers_per_mode(model, train, names, top_modes)

    print("DAMIAN deep dive...")
    deep = damian_deepdive(df, names)
    with open(OUT / "damian_deepdive.json", "w") as f:
        json.dump(deep, f, indent=2, default=str)

    print(f"\nAll outputs under {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
