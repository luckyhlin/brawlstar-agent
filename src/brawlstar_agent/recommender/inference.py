"""Inference helpers: answer the user-facing questions from a trained model.

Three scenarios all reduce to scoring candidate brawlers under different
conditioning:

1. `rank_brawlers_for_map(model, mode, map, ...)`:
       "Best brawler on map M, mode N" — no draft info.
       Uses Monte Carlo: sample plausible teammate/opponent triples and
       average P(win) given candidate brawler X is on team A.

2. `complete_team(model, my_team, opp_team, mode, map, ...)`:
       "Team A has X, Y already. Team B has U, V, W. What's the best Z?"
       Score every Z, return ranked list.

3. `last_pick(model, my_team, opp_team, mode, map, ...)`:
       Same shape as `complete_team` but with my_team having exactly 2
       brawlers and opp_team having 3. Returns ranked Z.
       (Convenience alias.)

All three rely on a single `score_candidate(model, partial_a, fixed_b, candidate, ...)`
primitive.

Notes on Monte Carlo for `rank_brawlers_for_map`:
- Sample plausible teammates from the empirical brawler distribution in that
  (mode, map) cell from the training data — this avoids making the candidate
  fight uniformly random teammates that no one would actually pick.
- Average P(team_a wins) over many samples; rank candidates by mean.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _make_query_row(
    team_a: tuple[int, ...],
    team_b: tuple[int, ...],
    mode: str,
    map: str,
    battle_type: str = "ranked",
    team_a_trophies_mean: float = 1000.0,
    team_b_trophies_mean: float = 1000.0,
) -> pd.DataFrame:
    return pd.DataFrame([{
        "battle_id": "QUERY",
        "battle_time_iso": "2099-01-01T00:00:00+00:00",
        "mode": mode,
        "map": map,
        "battle_type": battle_type,
        "team_a": tuple(sorted(int(x) for x in team_a)),
        "team_b": tuple(sorted(int(x) for x in team_b)),
        "team_a_wins": 0,  # placeholder
        "team_a_trophies_mean": team_a_trophies_mean,
        "team_b_trophies_mean": team_b_trophies_mean,
    }])


def score_candidate(
    model,
    partial_a: tuple[int, ...],
    fixed_b: tuple[int, ...],
    candidate: int,
    *,
    mode: str,
    map: str,
    battle_type: str = "ranked",
    team_a_trophies_mean: float = 1000.0,
    team_b_trophies_mean: float = 1000.0,
) -> float:
    """P(team A wins) if `candidate` joins team A given partial_a + fixed_b."""
    full_a = tuple(sorted(list(partial_a) + [int(candidate)]))
    if len(full_a) > 3 or len(set(full_a)) != len(full_a):
        return float("nan")
    row = _make_query_row(
        full_a, fixed_b, mode, map, battle_type,
        team_a_trophies_mean, team_b_trophies_mean,
    )
    return float(model.predict_proba(row)[0])


def complete_team(
    model,
    my_team: tuple[int, ...],
    opp_team: tuple[int, ...],
    *,
    mode: str,
    map: str,
    candidates: list[int] | None = None,
    battle_type: str = "ranked",
    team_a_trophies_mean: float = 1000.0,
    team_b_trophies_mean: float = 1000.0,
    top_k: int | None = 10,
    available_brawlers: list[int] | None = None,
) -> list[tuple[int, float]]:
    """Score every legal candidate to fill out `my_team` to 3 vs `opp_team`.

    `candidates` overrides the search space. `available_brawlers` is the
    universe of legal brawlers (defaults to the model's featurizer vocab).

    Returns [(brawler_id, P(win)), ...] sorted descending.
    """
    if candidates is None:
        if available_brawlers is None:
            f = getattr(model, "featurizer", None)
            available_brawlers = sorted(f.brawler_to_idx.keys()) if f is not None else []
        candidates = list(available_brawlers)

    used = set(my_team) | set(opp_team)
    legal = [c for c in candidates if int(c) not in used]
    if not legal:
        return []

    # Batch the scoring for speed
    rows = []
    for c in legal:
        full_a = tuple(sorted(list(my_team) + [int(c)]))
        rows.append({
            "battle_id": f"QUERY_{c}",
            "battle_time_iso": "2099-01-01T00:00:00+00:00",
            "mode": mode,
            "map": map,
            "battle_type": battle_type,
            "team_a": full_a,
            "team_b": tuple(sorted(int(x) for x in opp_team)),
            "team_a_wins": 0,
            "team_a_trophies_mean": team_a_trophies_mean,
            "team_b_trophies_mean": team_b_trophies_mean,
        })
    df = pd.DataFrame(rows)
    proba = model.predict_proba(df)
    out = list(zip([int(c) for c in legal], [float(p) for p in proba]))
    out.sort(key=lambda x: x[1], reverse=True)
    if top_k is not None:
        out = out[:top_k]
    return out


def last_pick(
    model,
    my_partial_team: tuple[int, ...],
    opp_team: tuple[int, ...],
    *,
    mode: str,
    map: str,
    **kwargs,
) -> list[tuple[int, float]]:
    """Convenience: predict the last (third) brawler for team A.

    Requires `len(my_partial_team) == 2` and `len(opp_team) == 3`.
    """
    if len(my_partial_team) != 2:
        raise ValueError("last_pick expects exactly 2 brawlers in my_partial_team")
    if len(opp_team) != 3:
        raise ValueError("last_pick expects exactly 3 brawlers in opp_team")
    return complete_team(model, my_partial_team, opp_team, mode=mode, map=map, **kwargs)


def rank_brawlers_for_map(
    model,
    mode: str,
    map: str,
    *,
    train_df: pd.DataFrame | None = None,
    candidates: list[int] | None = None,
    n_samples: int = 200,
    battle_type: str = "ranked",
    seed: int = 42,
) -> list[tuple[int, float]]:
    """Marginal "best brawler on this map" via Monte Carlo over plausible teams.

    For each candidate brawler X:
        Sample N (teammates_2, opponents_3) from the empirical (mode, map) draw
        in train_df. Compute mean P(team_a wins | team_a={X}+teammates).
        Rank candidates by mean.

    Without train_df we fall back to uniform random teammates from the model's
    vocabulary, which is honest but produces less actionable rankings.
    """
    rng = np.random.default_rng(seed)
    f = getattr(model, "featurizer", None)
    if candidates is None:
        candidates = sorted(f.brawler_to_idx.keys()) if f is not None else []
    candidates = list(candidates)
    if not candidates:
        return []

    # Build the empirical pool of teams for this (mode, map).
    if train_df is not None:
        cell = train_df[(train_df["mode"] == mode) & (train_df["map"] == map)]
        if len(cell) > 0:
            # Each row has team_a and team_b. Use their combination as the empirical pool.
            sample_idx = rng.integers(0, len(cell), size=n_samples)
            cell_rows = cell.iloc[sample_idx]
            sampled_teammates = []
            sampled_opponents = []
            for _, r in cell_rows.iterrows():
                # Use team_a as teammates pool, team_b as opponents
                ta = list(r["team_a"])
                tb = list(r["team_b"])
                # Drop a random element from team_a to make room for the candidate
                drop_idx = int(rng.integers(0, 3))
                teammates = tuple(sorted(ta[:drop_idx] + ta[drop_idx+1:]))
                sampled_teammates.append(teammates)
                sampled_opponents.append(tuple(sorted(tb)))
        else:
            sampled_teammates, sampled_opponents = _uniform_sample(rng, candidates, n_samples)
    else:
        sampled_teammates, sampled_opponents = _uniform_sample(rng, candidates, n_samples)

    # Build a giant DataFrame: candidate × n_samples.
    # Only score on samples where the candidate isn't already on either team.
    all_rows = []
    for c in candidates:
        for k in range(n_samples):
            tm = sampled_teammates[k]
            opp = sampled_opponents[k]
            if int(c) in tm or int(c) in opp:
                continue
            full_a = tuple(sorted(list(tm) + [int(c)]))
            all_rows.append({
                "candidate": int(c),
                "battle_id": f"MC_{c}_{k}",
                "battle_time_iso": "2099-01-01T00:00:00+00:00",
                "mode": mode,
                "map": map,
                "battle_type": battle_type,
                "team_a": full_a,
                "team_b": opp,
                "team_a_wins": 0,
                "team_a_trophies_mean": 1000.0,
                "team_b_trophies_mean": 1000.0,
            })
    if not all_rows:
        return []
    df = pd.DataFrame(all_rows)
    proba = model.predict_proba(df.drop(columns=["candidate"]))
    df["proba"] = proba
    means = df.groupby("candidate")["proba"].agg(["mean", "size"]).reset_index()
    means = means.sort_values("mean", ascending=False)
    return [(int(getattr(r, "candidate")), float(getattr(r, "mean"))) for r in means.itertuples(index=False)]


def _uniform_sample(rng: np.random.Generator, vocab: list[int], n: int):
    teammates, opponents = [], []
    for _ in range(n):
        idx = rng.choice(len(vocab), size=5, replace=False)
        chosen = [int(vocab[i]) for i in idx]
        teammates.append(tuple(sorted(chosen[:2])))
        opponents.append(tuple(sorted(chosen[2:])))
    return teammates, opponents
