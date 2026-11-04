"""Hierarchical shooting-skill taxonomy.

Every shot is mapped to one category per level. Each level has a parent level used as the
shrinkage target (a player's corner-3 skill shrinks toward his three-point skill, which shrinks
toward his general skill, which shrinks toward the league).

  L1 general   all            -> league
  L1 type      two / three    -> all
  L2 zone      rim, paint, short_mid, long_mid, corner3, ab3  -> type
  L3 action    action family (jump, pullup, stepback, ...)    -> type
  L4 zone x action                                            -> zone + action - type
  L5 contest   type x closest-defender range (tight/close/open/wide) -> type   (tracking shots only)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SHORT_MID_MAX_FT = 16.0
SKILL_ZONES = ["rim", "paint", "short_mid", "long_mid", "corner3", "ab3"]
CONTEST_RANGES = ["tight", "close", "open", "wide"]

# level -> parent level ("league" is the root)
LEVELS = {
    "all": "league",
    "type": "all",
    "zone": "type",
    "action": "type",
    "zone_action": "zone",
    "contest": "type",
}
SHOT_LEVELS = ["all", "type", "zone", "action", "zone_action"]


def skill_zone(zone_group: pd.Series, shot_distance_ft: pd.Series) -> pd.Series:
    z = zone_group.astype(str).to_numpy()
    d = pd.to_numeric(shot_distance_ft, errors="coerce").to_numpy()
    out = np.where(z == "mid", np.where(d < SHORT_MID_MAX_FT, "short_mid", "long_mid"), z)
    return pd.Series(out, index=zone_group.index)


def contest_range(contest_distance_in: pd.Series) -> pd.Series:
    """PlayerDashPtShots closest-defender ranges applied to Blueprint contest distance (inches)."""
    ft = pd.to_numeric(contest_distance_in, errors="coerce") / 12.0
    out = np.select([ft < 2, ft < 4, ft < 6, ft >= 6], CONTEST_RANGES, default="unknown")
    return pd.Series(out, index=contest_distance_in.index)


def categories(shots: pd.DataFrame) -> pd.DataFrame:
    """Category keys per level. Needs is_three, zone_group, shot_distance_ft, action_family;
    contest_distance_in optional."""
    typ = pd.Series(np.where(shots["is_three"].astype(int) == 1, "three", "two"), index=shots.index)
    zone = skill_zone(shots["zone_group"], shots["shot_distance_ft"])
    fam = shots["action_family"].astype(str).fillna("other")
    out = pd.DataFrame({
        "all": "all",
        "type": typ,
        "zone": zone,
        "action": typ + "|" + fam,
        "zone_action": zone + "|" + fam,
    }, index=shots.index)
    if "contest_distance_in" in shots:
        rng = contest_range(shots["contest_distance_in"])
        out["contest"] = np.where(rng == "unknown", None, typ + "|" + rng)
    return out


def parent_key(level: str, cats: pd.DataFrame) -> pd.Series:
    parent = LEVELS[level]
    if parent == "league":
        return pd.Series("league", index=cats.index)
    return cats[parent]


__all__ = ["CONTEST_RANGES", "LEVELS", "SHOT_LEVELS", "SKILL_ZONES", "categories", "contest_range", "parent_key",
           "skill_zone"]
