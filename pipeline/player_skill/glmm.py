"""Penalized hierarchical logistic player model (approximate GLMM, posterior mode).

  logit P(make) = offset + b0 + u_p + u_{p,type} + u_{p,zone} + u_{p,action} + u_{p,zone x action}
                  [+ u_{p,type x contest range}]
  u_g ~ N(0, 1 / lambda_g)   (ridge penalty per effect group; nested effects -> parent fallback)

`offset` is the logit of a player-agnostic expectation: the out-of-fold shot-difficulty baseline for
historical shots, and the league FG% of the (type, closest-defender range) cell for PlayerDashPtShots
aggregate rows (binomial rows with n attempts). Rows carry recency weights. Variance components are
chosen by chronological validation (`tune_lambda`), not REML. The fitted effects for a new shot sum to
a logit-scale player adjustment that is added to any base model's logit (approach D / candidate 6).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize

GROUPS = ["player", "type", "zone", "action", "zone_action", "contest"]
DEFAULT_LAMBDA = {"player": 50.0, "type": 100.0, "zone": 150.0, "action": 150.0, "zone_action": 400.0, "contest": 300.0}


def _keys(df: pd.DataFrame) -> dict[str, np.ndarray]:
    pid = df["PLAYER_ID"].astype(int).astype(str).to_numpy()
    out = {"player": pid}
    for g, col in (("type", "cat_type"), ("zone", "cat_zone"), ("action", "cat_action"),
                   ("zone_action", "cat_zone_action"), ("contest", "cat_contest")):
        v = df[col].to_numpy() if col in df else np.full(len(df), None, dtype=object)
        out[g] = np.where(pd.isna(v), None, np.char.add(np.char.add(pid.astype(str), "|"), v.astype(str)))
    return out


@dataclass
class GLMMFit:
    index: dict          # group -> {key: column}
    coef: np.ndarray
    intercept: float
    lam: dict

    def effect(self, df: pd.DataFrame, groups=GROUPS) -> np.ndarray:
        keys = _keys(df)
        out = np.zeros(len(df))
        for g in groups:
            ix = self.index[g]
            cols = np.array([ix.get(k, -1) if k is not None else -1 for k in keys[g]])
            ok = cols >= 0
            out[ok] += self.coef[cols[ok]]
        return out


def fit(rows: pd.DataFrame, lam: dict, groups=GROUPS, maxiter: int = 300) -> GLMMFit:
    """rows: PLAYER_ID, cat_* keys, offset, n, made, w."""
    keys = _keys(rows)
    index, blocks, pen = {}, [], []
    start = 0
    for g in groups:
        k = keys[g]
        ok = np.array([x is not None for x in k])
        uniq = pd.unique(k[ok])
        ix = {u: start + i for i, u in enumerate(uniq)}
        index[g] = ix
        r = np.flatnonzero(ok)
        c = np.array([ix[x] for x in k[ok]], dtype=int)
        blocks.append((r, c))
        pen.append(np.full(len(uniq), lam[g]))
        start += len(uniq)
    for g in GROUPS:
        index.setdefault(g, {})
    r = np.concatenate([b[0] for b in blocks])
    c = np.concatenate([b[1] for b in blocks])
    X = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(len(rows), start))
    pen = np.concatenate(pen) if pen else np.zeros(0)
    off = rows["offset"].to_numpy(float)
    n = rows["n"].to_numpy(float)
    m = rows["made"].to_numpy(float)
    w = rows["w"].to_numpy(float)
    XT = X.T.tocsr()

    def f(theta):
        b0, beta = theta[0], theta[1:]
        eta = off + b0 + X @ beta
        ll = w * (n * np.logaddexp(0, eta) - m * eta)
        p = 1 / (1 + np.exp(-eta))
        g_eta = w * (n * p - m)
        val = ll.sum() + 0.5 * np.sum(pen * beta ** 2)
        grad = np.concatenate([[g_eta.sum()], XT @ g_eta + pen * beta])
        return val, grad

    res = minimize(f, np.zeros(start + 1), jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
    return GLMMFit(index=index, coef=res.x[1:], intercept=float(res.x[0]), lam=dict(lam))


def shot_rows(ev: pd.DataFrame, as_of, half_life_days: float | None) -> pd.DataFrame:
    """Evidence (hierarchy.evidence_from_shots format, optionally with cat_contest / offset) -> fit rows."""
    as_of = pd.Timestamp(as_of)
    e = ev[ev["date"] < as_of].copy()
    age = (as_of - e["date"]).dt.days.to_numpy(float)
    e["w"] = e["w0"].to_numpy(float) * (0.5 ** (age / half_life_days) if half_life_days else 1.0)
    xb = np.clip(e["xbsum"] / e["n"], 1e-4, 1 - 1e-4)
    shot_off = np.log(xb / (1 - xb))
    e["offset"] = e["offset"].fillna(shot_off) if "offset" in e else shot_off
    return e


def dash_rows(dash: pd.DataFrame, seasons: list[str], w0: float = 1.0) -> pd.DataFrame:
    """PlayerDashPtShots (player, type, range) binomial rows with offset logit(league cell FG%)."""
    from pipeline.player_skill.contest import SEASON_DATE, dash_long
    d = dash_long(dash)
    d = d[d["SEASON"].isin(seasons) & (d["fga"] > 0)].copy()
    lg = d.groupby(["SEASON", "type", "range"])[["fgm", "fga"]].sum()
    d = d.join((lg["fgm"] / lg["fga"]).rename("lg"), on=["SEASON", "type", "range"])
    lgc = d["lg"].clip(1e-3, 1 - 1e-3)
    return pd.DataFrame({
        "PLAYER_ID": d["PLAYER_ID"].astype(int), "date": pd.to_datetime(d["SEASON"].map(SEASON_DATE)),
        "n": d["fga"].astype(float), "made": d["fgm"].astype(float), "w0": float(w0),
        "offset": np.log(lgc / (1 - lgc)), "cat_all": "all", "cat_type": d["type"], "cat_zone": None,
        "cat_action": None, "cat_zone_action": None, "cat_contest": d["type"] + "|" + d["range"],
        "xbsum": d["fga"] * d["lg"], "pqsum": d["fga"] * d["lg"] * (1 - d["lg"]),
    }).reset_index(drop=True)


__all__ = ["DEFAULT_LAMBDA", "GLMMFit", "GROUPS", "dash_rows", "fit", "shot_rows"]
