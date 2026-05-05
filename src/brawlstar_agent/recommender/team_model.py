"""Team-completion model: P(team_a wins | team_a, team_b, mode, map, ...).

Two implementations behind the same interface:

- LogRegTeamModel: scikit-learn LogisticRegression on sparse multi-hot features.
  Fast, interpretable, sets per-brawler "lift" coefficients that make sense
  to look at directly.

- LGBMTeamModel: LightGBM on dense features with categorical mode/map/battle_type.
  Captures interactions (which brawler-pair matchups beat which others) that the
  linear model can't.

Both expose .fit(train_df, train_y), .predict_proba(df) → np.array of P(team_a wins).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:  # pragma: no cover - LightGBM is in deps
    HAS_LGB = False

from .features import TeamFeaturizer


@dataclass
class LogRegTeamModel:
    """Logistic regression on sparse multi-hot team features."""
    C: float = 1.0
    max_iter: int = 200
    featurizer: TeamFeaturizer | None = None
    model: LogisticRegression | None = None

    def fit(self, df: pd.DataFrame, y: np.ndarray | None = None) -> "LogRegTeamModel":
        if y is None:
            y = df["team_a_wins"].values
        f = TeamFeaturizer().fit(df)
        X = f.transform_sparse(df)
        m = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver="liblinear",
        )
        m.fit(X, y)
        self.featurizer = f
        self.model = m
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        assert self.featurizer is not None and self.model is not None, "must fit first"
        X = self.featurizer.transform_sparse(df)
        return self.model.predict_proba(X)[:, 1]

    def per_brawler_lift(self) -> dict[str, dict[int, float]]:
        """Return per-brawler coefficients for interpretation.

        Positive `team_a_lift[X]` = X contributes to team A winning when on team A.
        Negative `team_b_lift[X]` = X contributes to team A winning when X is on team B
            (i.e., X is weak when faced).

        Both should be near-mirrored: a brawler's "as ally" lift ≈ -(its "as opponent" lift)
        if the model is symmetric. We expand both perspectives in training so this
        symmetry should be automatic.
        """
        assert self.featurizer is not None and self.model is not None
        f = self.featurizer
        coef = self.model.coef_[0]
        out_a: dict[int, float] = {}
        out_b: dict[int, float] = {}
        for bid, idx in f.brawler_to_idx.items():
            out_a[bid] = float(coef[idx])
            out_b[bid] = float(coef[f.n_brawlers + idx])
        return {"team_a_lift": out_a, "team_b_lift": out_b}


@dataclass
class LGBMTeamModel:
    """LightGBM on dense features with categorical mode/map/battle_type."""
    num_leaves: int = 63
    learning_rate: float = 0.05
    n_estimators: int = 400
    min_data_in_leaf: int = 50
    feature_fraction: float = 0.9
    reg_lambda: float = 1.0
    seed: int = 42
    early_stopping_rounds: int = 30
    featurizer: TeamFeaturizer | None = None
    model: Optional["lgb.Booster"] = None  # type: ignore[name-defined]
    cat_cols: list[int] = field(default_factory=list)

    def fit(
        self,
        df: pd.DataFrame,
        y: np.ndarray | None = None,
        valid_df: pd.DataFrame | None = None,
        valid_y: np.ndarray | None = None,
    ) -> "LGBMTeamModel":
        if not HAS_LGB:
            raise RuntimeError("LightGBM is not installed")
        if y is None:
            y = df["team_a_wins"].values

        f = TeamFeaturizer().fit(df)
        X, cat_cols = f.transform_dense(df)

        # Internal validation split if no explicit one given
        if valid_df is None:
            n = len(df)
            n_val = max(1, int(0.1 * n))
            rng = np.random.default_rng(self.seed)
            idx = rng.permutation(n)
            val_idx = idx[:n_val]; tr_idx = idx[n_val:]
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_va, y_va = X[val_idx], y[val_idx]
        else:
            X_tr, y_tr = X, y
            X_va, _cc = f.transform_dense(valid_df)
            y_va = valid_y if valid_y is not None else valid_df["team_a_wins"].values

        train_set = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols, free_raw_data=False)
        valid_set = lgb.Dataset(X_va, label=y_va, categorical_feature=cat_cols, free_raw_data=False, reference=train_set)
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "min_data_in_leaf": self.min_data_in_leaf,
            "feature_fraction": self.feature_fraction,
            "lambda_l2": self.reg_lambda,
            "verbose": -1,
            "seed": self.seed,
        }
        booster = lgb.train(
            params,
            train_set,
            num_boost_round=self.n_estimators,
            valid_sets=[valid_set],
            callbacks=[
                lgb.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        self.model = booster
        self.featurizer = f
        self.cat_cols = cat_cols
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        assert self.model is not None and self.featurizer is not None
        X, _ = self.featurizer.transform_dense(df)
        return self.model.predict(X, num_iteration=self.model.best_iteration)


def evaluate(
    model,
    test_df: pd.DataFrame,
    label_col: str = "team_a_wins",
) -> dict:
    """Standard binary metrics for our team-prediction setting."""
    from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, brier_score_loss

    proba = model.predict_proba(test_df)
    y = test_df[label_col].values
    proba_clip = np.clip(proba, 1e-3, 1.0 - 1e-3)
    return {
        "auc": float(roc_auc_score(y, proba)),
        "logloss": float(log_loss(y, proba_clip)),
        "accuracy": float(accuracy_score(y, (proba > 0.5).astype(int))),
        "brier": float(brier_score_loss(y, proba_clip)),
        "n": int(len(test_df)),
    }


def save_model(model: LogRegTeamModel | LGBMTeamModel, path: Path | str) -> None:
    """Persist a fitted model to disk. .joblib for sklearn, native LightGBM dump
    for LGBMTeamModel.
    """
    import joblib
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(model, LGBMTeamModel):
        # We save featurizer separately + booster as text file
        booster_path = p.with_suffix(".lgb.txt")
        meta_path = p.with_suffix(".meta.json")
        if model.model is not None:
            model.model.save_model(str(booster_path))
        meta = {
            "type": "LGBMTeamModel",
            "cat_cols": model.cat_cols,
            "featurizer": {
                "brawler_to_idx": {str(k): v for k, v in model.featurizer.brawler_to_idx.items()},
                "mode_to_idx": model.featurizer.mode_to_idx,
                "map_to_idx": model.featurizer.map_to_idx,
                "btype_to_idx": model.featurizer.btype_to_idx,
            } if model.featurizer is not None else None,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f)
    else:
        joblib.dump(model, p)


def load_model(path: Path | str):
    """Load a previously saved model."""
    import joblib
    p = Path(path)
    meta_path = p.with_suffix(".meta.json")
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("type") == "LGBMTeamModel":
            booster = lgb.Booster(model_file=str(p.with_suffix(".lgb.txt")))
            f = TeamFeaturizer(
                brawler_to_idx={int(k): v for k, v in meta["featurizer"]["brawler_to_idx"].items()},
                mode_to_idx=meta["featurizer"]["mode_to_idx"],
                map_to_idx=meta["featurizer"]["map_to_idx"],
                btype_to_idx=meta["featurizer"]["btype_to_idx"],
            )
            m = LGBMTeamModel()
            m.model = booster
            m.featurizer = f
            m.cat_cols = meta["cat_cols"]
            return m
    return joblib.load(p)
