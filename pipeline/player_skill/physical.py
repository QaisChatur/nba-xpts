"""Physical proxies and data-driven shot-profile archetypes.

Roster heights / weights: the public NBA.com players index answered HTTP 403 (access denied) from this
environment and no licensed roster file is available, so no listed heights are used. Proxies that are
already in our licensed-for-use data:
  * Blueprint `shot_vertical_in` (shooter vertical at release) -> shooter athleticism proxy
  * Blueprint `contest_height_in` (defender's contest height)  -> defender length proxy (shot level,
    already a Master V1 tracking feature)

Archetypes: k-means on each player's shrunk historical shot diet (location shares, action-family
shares). Diet is outcome-free, so archetypes never encode makes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from pipeline.config import RANDOM_SEED
from pipeline.player_skill import taxonomy as tx

DIET_K = 50.0  # pseudo-attempts toward league diet shares


def diet_matrix(h: pd.DataFrame, min_att: int = 0) -> pd.DataFrame:
    """Shrunk location and action-family shares per player."""
    h = h.reset_index(drop=True)
    zone = tx.skill_zone(h["zone_group"], h["shot_distance_ft"])
    fam = h["action_family"].astype(str)
    parts = []
    for name, col in (("z", zone), ("a", fam)):
        ct = pd.crosstab(h["PLAYER_ID"], col)
        league = ct.sum() / ct.sum().sum()
        n = ct.sum(axis=1)
        sh = (ct + DIET_K * league).div(n + DIET_K, axis=0)
        parts.append(sh.add_prefix(f"{name}_"))
    out = pd.concat(parts, axis=1)
    out["n"] = h.groupby("PLAYER_ID").size()
    return out[out["n"] >= min_att]


def fit_archetypes(diet: pd.DataFrame, ks=range(4, 11)) -> dict:
    X = diet.drop(columns="n")
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    scores = {}
    for k in ks:
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_SEED).fit(Z)
        scores[k] = float(silhouette_score(Z, km.labels_))
    best = max(scores, key=scores.get)
    km = KMeans(n_clusters=best, n_init=20, random_state=RANDOM_SEED).fit(Z)
    return {"k": best, "silhouette": scores, "model": km, "scaler": scaler, "columns": list(X.columns),
            "labels": pd.Series(km.labels_, index=diet.index, name="archetype")}


def assign(arch: dict, diet: pd.DataFrame) -> pd.Series:
    X = diet.reindex(columns=arch["columns"]).fillna(0.0)
    return pd.Series(arch["model"].predict(arch["scaler"].transform(X)), index=diet.index, name="archetype")


def archetype_summary(h: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Per archetype: players, shots, diet highlights, FG, mean over-expected and its between-player SD."""
    d = h.join(labels, on="PLAYER_ID").dropna(subset=["archetype"])
    d["res"] = d["made"] - d["xb"]
    zone = tx.skill_zone(d["zone_group"], d["shot_distance_ft"])
    g = d.groupby("archetype")
    out = pd.DataFrame({
        "players": g["PLAYER_ID"].nunique(), "shots": g.size(), "fg": g["made"].mean(),
        "over_expected": g["res"].mean(), "three_rate": g["is_three"].mean(),
        "rim_rate": (zone == "rim").groupby(d["archetype"]).mean(),
        "pullup_stepback_rate": d["action_family"].isin(["pullup", "stepback"]).groupby(d["archetype"]).mean(),
    })
    return out


def release_vertical_prior(shots: pd.DataFrame, k: float = 20.0) -> pd.Series:
    """As-of-date shrunk mean of Blueprint shooter vertical (jump shots), from earlier game dates only."""
    s = shots[["PLAYER_ID", "game_date", "shot_vertical_in", "action_family"]].copy()
    s["v"] = pd.to_numeric(s["shot_vertical_in"], errors="coerce")
    s = s[s["action_family"].isin(["jump", "pullup", "stepback"]) & s["v"].notna()]
    league = float(s["v"].mean()) if len(s) else np.nan
    out = pd.Series(np.nan, index=shots.index)
    for day in sorted(shots["game_date"].unique()):
        e = s[s["game_date"] < day].groupby("PLAYER_ID")["v"].agg(["sum", "count"])
        rows = shots.index[shots["game_date"] == day]
        pid = shots.loc[rows, "PLAYER_ID"].astype(int)
        su = pid.map(e["sum"]).fillna(0.0)
        n = pid.map(e["count"]).fillna(0.0)
        out.loc[rows] = ((su + k * league) / (n + k)).to_numpy()
    return out


__all__ = ["archetype_summary", "assign", "diet_matrix", "fit_archetypes", "release_vertical_prior"]
