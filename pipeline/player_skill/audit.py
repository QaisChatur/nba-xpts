"""Diagnostics of player-skill signal and of the Master V1 player features.

  skill_variance      between-player variance of true shot-making (raw FG and over-expected), per level
  season_reliability  season-to-season correlation of player over-expected rates (stability of skill)
  ceiling             log-loss gain available from *perfect* knowledge of player skill, and from a
                      realistic estimate with reliability R:  gain ~= R * tau^2 / (2 p (1 - p))
  coverage            history attempts behind every 2026-27 shot, per level
  prior_signal        does each V1 player feature explain the residuals of the V1 tracking model?
                      logistic regression  y ~ offset(logit p_base) + b * z(feature), fitted on the
                      calibration partition and scored on the test partition
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from pipeline.player_skill import taxonomy as tx


def _mom_tau2(g: pd.DataFrame, min_att: int) -> tuple[float, int]:
    """Method-of-moments between-player variance of rate r = sum(x)/n with sampling var v/n."""
    g = g[g["n"] >= min_att]
    if len(g) < 5:
        return float("nan"), int(len(g))
    w = g["n"].to_numpy(float)
    r = g["r"].to_numpy()
    mu = np.average(r, weights=w)
    var_obs = np.average((r - mu) ** 2, weights=w) * len(g) / (len(g) - 1)
    samp = np.average(g["v"] / g["n"], weights=w)
    return float(max(var_obs - samp, 0.0)), int(len(g))


def skill_variance(h: pd.DataFrame, xb_col: str = "xb", min_att: int = 50) -> pd.DataFrame:
    """tau (SD of true player rates around the category mean) for raw FG and made - xb."""
    rows = []
    for lvl in tx.SHOT_LEVELS:
        key = f"cat_{lvl}"
        d = h.assign(res=h["made"] - h[xb_col], pv=h[xb_col] * (1 - h[xb_col]))
        lg = d.groupby(key)["made"].transform("mean")
        d = d.assign(dev=d["made"] - lg)
        g = d.groupby([key, "PLAYER_ID"]).agg(n=("made", "size"), r_fg=("dev", "mean"), r_res=("res", "mean"),
                                              pv=("pv", "mean"), fg=("made", "mean"))
        for cat, gc in g.groupby(level=0):
            if gc["n"].sum() < 2000:
                continue
            t_fg, k = _mom_tau2(gc.assign(r=gc["r_fg"], v=gc["fg"] * (1 - gc["fg"])), min_att)
            t_res, _ = _mom_tau2(gc.assign(r=gc["r_res"], v=gc["pv"]), min_att)
            rows.append({"level": lvl, "cat": cat, "shots": int(gc["n"].sum()), "players": k,
                         "tau_raw_fg": np.sqrt(t_fg), "tau_over_expected": np.sqrt(t_res),
                         "mean_pq": float(np.average(gc["pv"], weights=gc["n"]))})
    return pd.DataFrame(rows)


def season_reliability(h: pd.DataFrame, xb_col: str = "xb", min_att: int = 200, level: str = "all") -> pd.DataFrame:
    key = f"cat_{level}"
    d = h.assign(res=h["made"] - h[xb_col])
    g = d.groupby([key, "PLAYER_ID", "SEASON"]).agg(n=("res", "size"), r=("res", "mean"), fg=("made", "mean")).reset_index()
    rows = []
    seasons = sorted(h["SEASON"].unique())
    for a, b in zip(seasons[:-1], seasons[1:]):
        for cat, gc in g.groupby(key):
            x = gc[gc["SEASON"] == a].set_index("PLAYER_ID")
            y = gc[gc["SEASON"] == b].set_index("PLAYER_ID")
            j = x.join(y, lsuffix="_a", rsuffix="_b", how="inner")
            j = j[(j["n_a"] >= min_att) & (j["n_b"] >= min_att)]
            if len(j) < 20:
                continue
            rows.append({"level": level, "cat": cat, "seasons": f"{a}->{b}", "players": len(j),
                         "median_att": float(np.median(np.minimum(j["n_a"], j["n_b"]))),
                         "r_over_expected": float(np.corrcoef(j["r_a"], j["r_b"])[0, 1]),
                         "r_raw_fg": float(np.corrcoef(j["fg_a"], j["fg_b"])[0, 1])})
    return pd.DataFrame(rows)


def ceiling(tau: float, p: np.ndarray, reliability: np.ndarray | float = 1.0) -> float:
    """Expected per-shot log-loss reduction (nats) from adding a player effect with SD `tau`
    (probability scale) estimated with `reliability` (share of true variance recovered)."""
    pq = np.clip(p * (1 - p), 1e-3, None)
    return float(np.mean(np.asarray(reliability) * tau ** 2 / (2 * pq)))


def coverage(shots: pd.DataFrame, h: pd.DataFrame) -> pd.DataFrame:
    """Unweighted 2022-25 history attempts behind each 2026-27 shot, per level."""
    cats = tx.categories(shots)
    out = pd.DataFrame(index=shots.index)
    for lvl in tx.SHOT_LEVELS:
        n = h.groupby(["PLAYER_ID", f"cat_{lvl}"]).size()
        idx = pd.MultiIndex.from_arrays([shots["PLAYER_ID"].astype(int).to_numpy(), cats[lvl].to_numpy()])
        out[f"hist_att_{lvl}"] = n.reindex(idx).fillna(0).to_numpy()
    return out


def _offset_fit(y: np.ndarray, off: np.ndarray, z: np.ndarray) -> float:
    def nll(b):
        p = 1 / (1 + np.exp(-(off + b * z)))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x)


def logloss(y, p) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def prior_signal(df: pd.DataFrame, base_col: str, features: list[str], fit_mask: np.ndarray,
                 eval_mask: np.ndarray) -> pd.DataFrame:
    """One-parameter offset regression per feature: fitted on `fit_mask`, scored on `eval_mask`."""
    y = df["made"].to_numpy(float)
    p0 = np.clip(df[base_col].to_numpy(float), 1e-6, 1 - 1e-6)
    off = np.log(p0 / (1 - p0))
    rows = []
    for f in features:
        x = df[f].to_numpy(float)
        ok = np.isfinite(x)
        mu, sd = np.nanmean(x[fit_mask & ok]), np.nanstd(x[fit_mask & ok])
        z = np.where(ok, (x - mu) / (sd if sd > 0 else 1.0), 0.0)
        b = _offset_fit(y[fit_mask], off[fit_mask], z[fit_mask])
        p1 = 1 / (1 + np.exp(-(off + b * z)))
        rows.append({"feature": f, "coef_per_sd": b,
                     "fit_delta": logloss(y[fit_mask], p1[fit_mask]) - logloss(y[fit_mask], p0[fit_mask]),
                     "eval_delta": logloss(y[eval_mask], p1[eval_mask]) - logloss(y[eval_mask], p0[eval_mask]),
                     "corr_with_base_logit": float(np.corrcoef(z[fit_mask], off[fit_mask])[0, 1])})
    return pd.DataFrame(rows)


__all__ = ["ceiling", "coverage", "logloss", "prior_signal", "season_reliability", "skill_variance"]
