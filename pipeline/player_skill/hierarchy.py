"""Hierarchical empirical-Bayes player shooting skill, computed as of a date.

Evidence rows (shots, or aggregate rows from PlayerDashPtShots) carry attempts `n`, makes, an expected
make sum `xbsum` (shot-difficulty baseline, out-of-fold for history), `pqsum` = sum xb(1-xb) and a base
weight `w0`. As of date D only rows dated strictly before D are used, with recency weight
w = w0 * 0.5 ** (age_days / half_life).

For a player and the shot's category at each level (see taxonomy.LEVELS), with node sums
  S_w = sum w n,  S_w2 = sum w^2 n,  S_r = sum w (made - xb)
  n_eff = S_w^2 / S_w2,  r_bar = S_r / S_w,  reliability = n_eff / (n_eff + k_level)
the context-adjusted skill (shot-making over expected, probability scale) is shrunk top-down:
  theta_root  = mu_root (league / newcomer mean)
  theta_node  = theta_parent + reliability * (r_bar - theta_parent)
(the parent of a location x action cell is theta_zone + theta_action - theta_type). A node without evidence inherits its parent's estimate (parent-group fallback; never zero-filled).
k_level = sigma^2 / tau_level^2 is tuned on chronological validation (`tune`).

The logit-scale player effect for a shot is delta = theta / pq_node, where pq_node is the node's mean
xb(1-xb) (smoothed toward the league value): mean(sigmoid(l + delta) - sigmoid(l)) ~= delta * mean(pq).

The same machinery with expected = league category rate gives *raw* hierarchical FG skill.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pipeline.player_skill import taxonomy as tx

LEVELS = ["all", "type", "zone", "action", "zone_action"]
PQ_SMOOTH = 20.0


@dataclass
class SkillParams:
    k: dict = field(default_factory=lambda: {"all": 300.0, "type": 300.0, "zone": 300.0, "action": 400.0,
                                             "zone_action": 600.0})
    k_raw: dict = field(default_factory=lambda: {"all": 150.0, "type": 200.0, "zone": 150.0, "action": 200.0,
                                                 "zone_action": 300.0})
    half_life_days: float | None = 730.0
    mu_veteran: float = 0.0
    mu_newcomer: float = 0.0
    w_dash: float = 0.5

    def to_dict(self) -> dict:
        return {"k": self.k, "k_raw": self.k_raw, "half_life_days": self.half_life_days, "mu_veteran": self.mu_veteran,
                "mu_newcomer": self.mu_newcomer, "w_dash": self.w_dash}

    @classmethod
    def from_dict(cls, d: dict) -> SkillParams:
        return cls(**d)


# --------------------------------------------------------------------------- evidence


def evidence_from_shots(df: pd.DataFrame, xb_col: str, date_col: str = "game_date") -> pd.DataFrame:
    """One evidence row per shot. df needs PLAYER_ID, made, xb_col, is_three, zone_group,
    shot_distance_ft, action_family, date_col."""
    xb = df[xb_col].to_numpy(float)
    cats = tx.categories(df)
    ev = pd.DataFrame({
        "PLAYER_ID": df["PLAYER_ID"].astype(int).to_numpy(), "date": pd.to_datetime(df[date_col]).to_numpy(),
        "n": 1.0, "made": df["made"].to_numpy(float), "xbsum": xb, "pqsum": xb * (1 - xb), "w0": 1.0,
    })
    for lvl in LEVELS:
        ev[f"cat_{lvl}"] = cats[lvl].to_numpy()
    return ev


def evidence_from_dash(dash: pd.DataFrame, season: str, date: str, w0: float) -> pd.DataFrame:
    """2PT/3PT aggregate rows for one PlayerDashPtShots season. Expected makes use the same season's
    league FG% in each closest-defender range, so the residual is contest-adjusted (not
    location-adjusted); `w0` discounts it relative to shot-level evidence. Feeds levels all and type."""
    from pipeline.player_skill.contest import dash_long
    d = dash_long(dash)
    d = d[d["SEASON"] == season]
    lg = d.groupby(["type", "range"])[["fgm", "fga"]].sum()
    lg = (lg["fgm"] / lg["fga"]).rename("lg")
    d = d.join(lg, on=["type", "range"])
    d["xbsum"] = d["fga"] * d["lg"]
    d["pqsum"] = d["fga"] * d["lg"] * (1 - d["lg"])
    g = d.groupby(["PLAYER_ID", "type"])[["fga", "fgm", "xbsum", "pqsum"]].sum().reset_index()
    g = g[g["fga"] > 0]
    ev = pd.DataFrame({
        "PLAYER_ID": g["PLAYER_ID"].astype(int), "date": pd.Timestamp(date), "n": g["fga"].astype(float),
        "made": g["fgm"].astype(float), "xbsum": g["xbsum"], "pqsum": g["pqsum"], "w0": float(w0),
        "cat_all": "all", "cat_type": g["type"],
    })
    for lvl in ("zone", "action", "zone_action"):
        ev[f"cat_{lvl}"] = None
    return ev.reset_index(drop=True)


# --------------------------------------------------------------------------- node statistics


def node_stats(ev: pd.DataFrame, as_of, half_life_days: float | None) -> dict[str, pd.DataFrame]:
    """Recency-weighted node sums per (PLAYER_ID, category) for each level, from rows before as_of."""
    as_of = pd.Timestamp(as_of)
    e = ev[ev["date"] < as_of]
    age = (as_of - e["date"]).dt.days.to_numpy(float)
    w = e["w0"].to_numpy(float) * (0.5 ** (age / half_life_days) if half_life_days else 1.0)
    base = pd.DataFrame({
        "PLAYER_ID": e["PLAYER_ID"].to_numpy(), "Sw": w * e["n"].to_numpy(), "Sw2": w * w * e["n"].to_numpy(),
        "Sr": w * (e["made"].to_numpy() - e["xbsum"].to_numpy()), "Spq": w * e["pqsum"].to_numpy(),
        "Sm": w * e["made"].to_numpy(), "N": e["n"].to_numpy(),
    })
    out = {}
    for lvl in LEVELS:
        b = base.assign(cat=e[f"cat_{lvl}"].to_numpy())
        b = b[b["cat"].notna()]
        out[lvl] = b.groupby(["PLAYER_ID", "cat"], sort=False)[["Sw", "Sw2", "Sr", "Spq", "Sm", "N"]].sum()
    return out


def lookup(stats: dict[str, pd.DataFrame], pid: np.ndarray, cats: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Node sums aligned to shots (zeros where the player has no evidence in that node)."""
    out = {}
    for lvl in LEVELS:
        idx = pd.MultiIndex.from_arrays([pid, cats[lvl].astype(str).to_numpy()])
        out[lvl] = stats[lvl].reindex(idx).fillna(0.0).reset_index(drop=True)
    return out


# --------------------------------------------------------------------------- shrinkage


@dataclass
class League:
    """League reference values per level category (from history)."""
    rate: dict      # level -> Series cat -> FG%
    pq: dict        # level -> Series cat -> mean xb(1-xb)

    @classmethod
    def from_history(cls, h: pd.DataFrame, xb_col: str = "xb") -> League:
        cats = tx.categories(h)
        pq = h[xb_col] * (1 - h[xb_col])
        return cls(rate={lvl: h["made"].groupby(cats[lvl]).mean() for lvl in LEVELS},
                   pq={lvl: pq.groupby(cats[lvl]).mean() for lvl in LEVELS})


def shrink(nodes: dict[str, pd.DataFrame], cats: pd.DataFrame, k: dict, root: np.ndarray, league: League,
           raw: bool = False) -> pd.DataFrame:
    """Top-down hierarchical shrinkage. Returns per-shot theta / reliability / n_eff / attempts per
    level, and (context-adjusted only) the logit-scale effect delta per level."""
    n = len(cats)
    out = {}
    theta = {"league": np.asarray(root, float) if not raw else np.zeros(n)}
    for lvl in LEVELS:
        s = nodes[lvl]
        sw, sw2 = s["Sw"].to_numpy(), s["Sw2"].to_numpy()
        neff = np.where(sw2 > 0, sw ** 2 / np.where(sw2 > 0, sw2, 1), 0.0)
        if raw:
            lg = cats[lvl].astype(str).map(league.rate[lvl]).to_numpy(float)
            rbar = np.where(sw > 0, s["Sm"].to_numpy() / np.where(sw > 0, sw, 1) - lg, 0.0)
        else:
            rbar = np.where(sw > 0, s["Sr"].to_numpy() / np.where(sw > 0, sw, 1), 0.0)
        rel = neff / (neff + k[lvl])
        if lvl == "zone_action":  # parent of a location x action cell: additive location + action effects
            tp = theta["zone"] + theta["action"] - theta["type"]
        else:
            tp = theta[tx.LEVELS[lvl]]
        th = tp + rel * (rbar - tp)
        theta[lvl] = th
        out[f"theta_{lvl}"] = th
        out[f"rel_{lvl}"] = rel
        out[f"neff_{lvl}"] = neff
        out[f"att_{lvl}"] = s["N"].to_numpy()
        if not raw:
            lpq = cats[lvl].astype(str).map(league.pq[lvl]).fillna(0.22).to_numpy(float)
            pq = (s["Spq"].to_numpy() + PQ_SMOOTH * lpq) / (sw + PQ_SMOOTH)
            out[f"delta_{lvl}"] = th / np.clip(pq, 0.03, None)
        else:
            out[f"fg_{lvl}"] = cats[lvl].astype(str).map(league.rate[lvl]).to_numpy(float) + th
    return pd.DataFrame(out)


def root_prior(pid: np.ndarray, has_history: np.ndarray, params: SkillParams) -> np.ndarray:
    return np.where(has_history, params.mu_veteran, params.mu_newcomer)


def skill_as_of(ev: pd.DataFrame, shots: pd.DataFrame, as_of, params: SkillParams, league: League,
                preseason_cutoff=None) -> pd.DataFrame:
    """Context-adjusted and raw hierarchical skill for `shots` using evidence before `as_of`.
    `preseason_cutoff`: rows before this date define 'has history' (veteran vs newcomer root)."""
    pid = shots["PLAYER_ID"].astype(int).to_numpy()
    cats = tx.categories(shots)
    st = node_stats(ev, as_of, params.half_life_days)
    nodes = lookup(st, pid, cats)
    if preseason_cutoff is None:
        has_hist = nodes["all"]["N"].to_numpy() > 0
    else:
        pre = ev[ev["date"] < pd.Timestamp(preseason_cutoff)]
        has_hist = np.isin(pid, pre["PLAYER_ID"].unique())
    ctx = shrink(nodes, cats, params.k, root_prior(pid, has_hist, params), league)
    raw = shrink(nodes, cats, params.k_raw, np.zeros(len(pid)), league, raw=True).add_prefix("raw_")
    res = pd.concat([ctx, raw[[c for c in raw.columns if c.startswith(("raw_theta", "raw_fg"))]]], axis=1)
    res["has_history"] = has_hist.astype(int)
    res.index = shots.index
    return res


# --------------------------------------------------------------------------- chronological tuning


def _ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _sig(x):
    return 1 / (1 + np.exp(-x))


def validation_blocks(h: pd.DataFrame, season: str, freq: str = "MS") -> list[tuple[pd.Timestamp, np.ndarray]]:
    """Monthly as-of blocks inside `season`: shots of each month are scored with evidence before it."""
    d = pd.to_datetime(h["game_date"])
    m = (h["SEASON"] == season).to_numpy()
    starts = d[m].dt.to_period("M").dt.start_time
    blocks = []
    for s in sorted(starts.unique()):
        idx = np.flatnonzero(m & (d.dt.to_period("M").dt.start_time == s).to_numpy())
        blocks.append((pd.Timestamp(s), idx))
    return blocks


def prepare_validation(h: pd.DataFrame, ev: pd.DataFrame, season: str, half_life_days: float | None,
                       preseason_cutoff) -> dict:
    """Precompute node sums for every validation shot (depends only on the half-life)."""
    parts = []
    pre = ev[ev["date"] < pd.Timestamp(preseason_cutoff)]["PLAYER_ID"].unique()
    for start, idx in validation_blocks(h, season):
        shots = h.iloc[idx]
        st = node_stats(ev, start, half_life_days)
        cats = tx.categories(shots)
        nodes = lookup(st, shots["PLAYER_ID"].astype(int).to_numpy(), cats)
        parts.append((idx, cats.reset_index(drop=True), nodes))
    idx = np.concatenate([p[0] for p in parts])
    cats = pd.concat([p[1] for p in parts], ignore_index=True)
    nodes = {lvl: pd.concat([p[2][lvl] for p in parts], ignore_index=True) for lvl in LEVELS}
    pid = h["PLAYER_ID"].to_numpy()[idx]
    return {"idx": idx, "cats": cats, "nodes": nodes, "has_hist": np.isin(pid, pre),
            "y": h["made"].to_numpy(float)[idx], "xb": h["xb"].to_numpy(float)[idx]}


def validation_loss(prep: dict, params: SkillParams, league: League, level: str = "zone_action") -> float:
    root = root_prior(None, prep["has_hist"], params)
    s = shrink(prep["nodes"], prep["cats"], params.k, root, league)
    xb = np.clip(prep["xb"], 1e-4, 1 - 1e-4)
    return _ll(prep["y"], _sig(np.log(xb / (1 - xb)) + s[f"delta_{level}"].to_numpy()))


def tune(prep: dict, league: League, start: SkillParams, level: str = "zone_action", rounds: int = 2) -> SkillParams:
    """Coordinate search of k per level (log-grid) minimizing validation log loss."""
    p = SkillParams.from_dict({**start.to_dict(), "k": dict(start.k)})
    grid = [25, 50, 100, 150, 200, 300, 450, 700, 1000, 1500, 2500, 4000, 8000]
    for _ in range(rounds):
        for lvl in LEVELS:
            best = (validation_loss(prep, p, league, level), p.k[lvl])
            for g in grid:
                p.k[lvl] = float(g)
                v = validation_loss(prep, p, league, level)
                if v < best[0] - 1e-7:
                    best = (v, float(g))
            p.k[lvl] = best[1]
    return p


__all__ = ["LEVELS", "League", "SkillParams", "evidence_from_dash", "evidence_from_shots", "lookup", "node_stats",
           "prepare_validation", "shrink", "skill_as_of", "tune", "validation_blocks", "validation_loss"]
