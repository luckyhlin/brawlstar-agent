"""Brawler-pick recommendation models.

Phase 6 work (memory-bank/progress.md). The core model is

    P(team_A wins | brawlers_A, brawlers_B, mode, map, tier, date)

Three user-facing scenarios all reduce to inference under different conditioning:

- "Best brawler on map M" → marginalize over plausible opponents/teammates.
- "Last pick Z for team A given X, Y on A and U, V, W on B" → argmax_Z over candidates.
- "Pre-draft tier list for map M" → marginalize over partial teams.

See `docs/recommender-v1.md` for methodology.
"""

from .dataset import (
    CLEAN_CUTOFF_ISO,
    BattleRow,
    load_clean_battles,
    load_brawler_names,
    split_temporal,
    split_random,
)

__all__ = [
    "CLEAN_CUTOFF_ISO",
    "BattleRow",
    "load_clean_battles",
    "load_brawler_names",
    "split_temporal",
    "split_random",
]
