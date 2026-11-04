"""Contest-specific shooting ability.

PlayerDashPtShots reports each player's 2PT/3PT makes and attempts by closest-defender distance
range (0-2, 2-4, 4-6, 6+ ft) per season. Blueprint reports `contest_distance_in` per shot. Before
using the Dash ranges as priors for Blueprint shots, `compatibility_audit` checks that the two
definitions agree (league attempt shares and FG% per type x range).

Approaches (compared in pipeline.player_skill.experiments):
  A  contest-category prior   player FG in the shot's (type, range) relative to his own type-level FG,
                              shrunk toward 0  ->  `contest_dev`
  B  contest sensitivity      player's (open+wide minus tight+close) FG gap relative to the league gap,
                              shrunk toward 0  ->  `contest_sens`, applied as sens * (open indicator - share)
  C  base + regularized adjustment using A/B with coefficients fitted on the calibration partition
  D  joint hierarchical model (pipeline.player_skill.glmm) with player x type x range effects
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.player_skill import taxonomy as tx

DASH_RANGES = {"0-2 Feet - Very Tight": "tight", "2-4 Feet - Tight": "close", "4-6 Feet - Open": "open",
               "6+ Feet - Wide Open": "wide"}
SEASON_DATE = {"2021-22": "2022-04-10", "2022-23": "2023-04-09", "2023-24": "2024-04-14", "2024-25": "2025-04-13",
               "2025-26": "2026-04-12"}
OPEN = ("open", "wide")


def dash_long(dash: pd.DataFrame) -> pd.DataFrame:
    """Long table: PLAYER_ID, SEASON, type (two/three), range, fgm, fga."""
    d = dash.copy()
    d["range"] = d["CLOSE_DEF_DIST_RANGE"].map(DASH_RANGES)
    d = d[d["range"].notna()]
    rows = []
    for typ, m, a in (("two", "FG2M", "FG2A"), ("three", "FG3M", "FG3A")):
        x = d[["PLAYER_ID", "SEASON", "range"]].copy()
        x["type"] = typ
        x["fgm"] = pd.to_numeric(d[m], errors="coerce").fillna(0)
        x["fga"] = pd.to_numeric(d[a], errors="coerce").fillna(0)
        rows.append(x)
    out = pd.concat(rows, ignore_index=True)
    out["PLAYER_ID"] = out["PLAYER_ID"].astype(int)
    return out.groupby(["PLAYER_ID", "SEASON", "type", "range"], as_index=False)[["fgm", "fga"]].sum()


def compatibility_audit(dash: pd.DataFrame, shots: pd.DataFrame, dash_season: str = "2025-26",
                        thresholds: dict | None = None) -> pd.DataFrame:
    """League attempt share and FG% per type x range: Dash (one season) vs Blueprint tracked shots,
    using the nominal foot thresholds or, if given, quantile-mapped thresholds."""
    d = dash_long(dash)
    d = d[d["SEASON"] == dash_season].groupby(["type", "range"])[["fgm", "fga"]].sum()
    d["dash_share"] = d["fga"] / d.groupby(level="type")["fga"].transform("sum")
    d["dash_fg"] = d["fgm"] / d["fga"]
    s = shots[pd.to_numeric(shots["contest_distance_in"], errors="coerce").notna()]
    rng = tx.contest_range(s["contest_distance_in"]) if thresholds is None else mapped_range(s, thresholds)
    b = pd.DataFrame({"type": np.where(s["is_three"] == 1, "three", "two"), "range": rng, "made": s["made"].astype(int)})
    b = b.groupby(["type", "range"]).agg(bp_fga=("made", "size"), bp_fg=("made", "mean"))
    b["bp_share"] = b["bp_fga"] / b.groupby(level="type")["bp_fga"].transform("sum")
    out = d[["fga", "dash_share", "dash_fg"]].rename(columns={"fga": "dash_fga"}).join(b, how="outer")
    out["share_diff"] = out["bp_share"] - out["dash_share"]
    out["fg_diff"] = out["bp_fg"] - out["dash_fg"]
    se = np.sqrt(out["bp_fg"] * (1 - out["bp_fg"]) / out["bp_fga"])
    out["fg_diff_z"] = out["fg_diff"] / se
    return out.reset_index()


RANGE_ORDER = ["tight", "close", "open", "wide"]


def quantile_thresholds(dash: pd.DataFrame, shots: pd.DataFrame, dash_season: str = "2025-26") -> dict:
    """Blueprint contest-distance cut points (inches) per type that reproduce the Dash league range
    shares. Uses only the contest-distance distribution of `shots` (no outcomes)."""
    d = dash_long(dash)
    d = d[d["SEASON"] == dash_season].groupby(["type", "range"])["fga"].sum()
    s = shots[pd.to_numeric(shots["contest_distance_in"], errors="coerce").notna()]
    out = {}
    for typ, is3 in (("two", 0), ("three", 1)):
        share = d.loc[typ].reindex(RANGE_ORDER).fillna(0)
        cum = (share.cumsum() / share.sum()).to_numpy()[:-1]
        x = pd.to_numeric(s.loc[s["is_three"].astype(int) == is3, "contest_distance_in"]).to_numpy()
        out[typ] = [float(np.quantile(x, q)) for q in cum]
    return out


def mapped_range(shots: pd.DataFrame, thresholds: dict) -> pd.Series:
    """Dash-equivalent closest-defender range for Blueprint shots via quantile thresholds."""
    x = pd.to_numeric(shots["contest_distance_in"], errors="coerce").to_numpy()
    is3 = shots["is_three"].astype(int).to_numpy() == 1
    out = np.full(len(x), "unknown", dtype=object)
    for typ, mask in (("two", ~is3), ("three", is3)):
        t1, t2, t3 = thresholds[typ]
        v = x[mask]
        out[mask] = np.select([v < t1, v < t2, v < t3, v >= t3], RANGE_ORDER, default="unknown")
    return pd.Series(out, index=shots.index)


def _season_weights(seasons: pd.Series, as_of, half_life_days: float | None) -> np.ndarray:
    age = (pd.Timestamp(as_of) - pd.to_datetime(seasons.map(SEASON_DATE))).dt.days.to_numpy(float)
    return 0.5 ** (age / half_life_days) if half_life_days else np.ones(len(seasons))


def contest_tables(dash: pd.DataFrame, as_of, seasons: list[str], half_life_days: float | None = 730.0,
                   k_dev: float = 200.0, k_sens: float = 400.0) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Shrunk contest deviations (approach A) and sensitivities (approach B) per player and type.

    dev[p, t, r]  = shrink( (FG_ptr - lg_tr) - (FG_pt - lg_t) )         toward 0 with k_dev
    sens[p, t]    = shrink( (FG_open - FG_tight)_pt - (lg_open - lg_tight)_t ) toward 0 with k_sens
    League rates are season-specific (attempt-weighted) so era drift does not leak into skill.
    Returns (dev, sens, open_share[type]).
    """
    d = dash_long(dash)
    d = d[d["SEASON"].isin(seasons)].copy()
    lg = d.groupby(["SEASON", "type", "range"])[["fgm", "fga"]].sum()
    d = d.join((lg["fgm"] / lg["fga"]).rename("lg_tr"), on=["SEASON", "type", "range"])
    d["w"] = _season_weights(d["SEASON"], as_of, half_life_days)
    d["wa"] = d["w"] * d["fga"]
    d["wm"] = d["w"] * d["fgm"]
    d["wx"] = d["w"] * d["fga"] * d["lg_tr"]
    cell = d.groupby(["PLAYER_ID", "type", "range"])[["wa", "wm", "wx"]].sum()
    typ = cell.groupby(level=["PLAYER_ID", "type"]).sum()
    c = cell.join(typ, rsuffix="_t")
    n = c["wa"]
    over_cell = (c["wm"] - c["wx"]) / n.where(n > 0)
    over_type = (c["wm_t"] - c["wx_t"]) / c["wa_t"].where(c["wa_t"] > 0)
    dev = ((over_cell - over_type) * n / (n + k_dev)).fillna(0.0).rename("contest_dev")
    # sensitivity: open+wide vs tight+close, each side relative to league expectations
    side = np.where(c.index.get_level_values("range").isin(OPEN), "open", "tight")
    s = c[["wa", "wm", "wx"]].groupby([c.index.get_level_values("PLAYER_ID"), c.index.get_level_values("type"), side]).sum()
    s.index.names = ["PLAYER_ID", "type", "side"]
    s = s.unstack("side").fillna(0.0)
    o_rate = (s[("wm", "open")] - s[("wx", "open")]) / s[("wa", "open")].where(s[("wa", "open")] > 0)
    t_rate = (s[("wm", "tight")] - s[("wx", "tight")]) / s[("wa", "tight")].where(s[("wa", "tight")] > 0)
    n_h = 1.0 / (1.0 / s[("wa", "open")].clip(lower=1) + 1.0 / s[("wa", "tight")].clip(lower=1))  # harmonic size
    sens = ((o_rate - t_rate) * n_h / (n_h + k_sens)).fillna(0.0).rename("contest_sens")
    share = d.assign(o=d["range"].isin(OPEN) * d["fga"]).groupby("type")[["o", "fga"]].sum()
    open_share = share["o"] / share["fga"]
    return dev, sens, open_share


def contest_features(shots: pd.DataFrame, dev: pd.DataFrame, sens: pd.Series, open_share: pd.Series,
                     thresholds: dict) -> pd.DataFrame:
    """Per-shot approach A / B values; rows without a contest distance get NaN (not 0)."""
    pid = shots["PLAYER_ID"].astype(int).to_numpy()
    typ = np.where(shots["is_three"].astype(int) == 1, "three", "two")
    rng = mapped_range(shots, thresholds).to_numpy()
    known = rng != "unknown"
    a = dev.reindex(pd.MultiIndex.from_arrays([pid, typ, rng])).to_numpy()
    sv = sens.reindex(pd.MultiIndex.from_arrays([pid, typ])).to_numpy()
    is_open = np.isin(rng, OPEN).astype(float)
    out = pd.DataFrame(index=shots.index)
    out["ps_contest_dev"] = np.where(known, np.nan_to_num(a, nan=0.0), np.nan)
    out["ps_contest_sens"] = np.nan_to_num(sv, nan=0.0)
    out["ps_contest_sens_x_open"] = np.where(known, out["ps_contest_sens"] * (is_open - pd.Series(typ).map(open_share).to_numpy()), np.nan)
    out["ps_contest_has_dash"] = (~np.isnan(sv)).astype(int)
    return out


def tune_contest(dash: pd.DataFrame, target_season: str, history_seasons: list[str], half_life_days: float | None,
                 grid=(25, 50, 100, 200, 400, 800, 1600, 3200)) -> dict:
    """Pick k_dev / k_sens by binomial log loss on next-season Dash cells:
    p_ptr = lg_tr(target) + over_type_pt(history) + adjustment(history)."""
    d = dash_long(dash)
    tgt = d[d["SEASON"] == target_season]
    lg = tgt.groupby(["type", "range"])[["fgm", "fga"]].sum()
    lg_tr = (lg["fgm"] / lg["fga"]).rename("lg")
    hist = d[d["SEASON"].isin(history_seasons)].copy()
    lgh = hist.groupby(["SEASON", "type", "range"])[["fgm", "fga"]].sum()
    hist = hist.join((lgh["fgm"] / lgh["fga"]).rename("lg_tr"), on=["SEASON", "type", "range"])
    as_of = SEASON_DATE[target_season]
    hist["w"] = _season_weights(hist["SEASON"], as_of, half_life_days)
    t = hist.assign(wa=hist["w"] * hist["fga"], wr=hist["w"] * (hist["fgm"] - hist["fga"] * hist["lg_tr"]))
    t = t.groupby(["PLAYER_ID", "type"])[["wa", "wr"]].sum()
    over_t = (t["wr"] / (t["wa"] + 300)).rename("over_t")   # type-level skill (fixed modest shrinkage)
    y = tgt.join(lg_tr, on=["type", "range"]).join(over_t, on=["PLAYER_ID", "type"])
    y["over_t"] = y["over_t"].fillna(0.0)
    y = y[y["fga"] > 0]

    def binll(p):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return float(-(y["fgm"] * np.log(p) + (y["fga"] - y["fgm"]) * np.log(1 - p)).sum() / y["fga"].sum())

    base = binll(y["lg"] + y["over_t"])
    res = {"base_type_only": base, "dev": {}, "sens": {}}
    for k in grid:
        dev, sens, share = contest_tables(dash, as_of, history_seasons, half_life_days, k_dev=k, k_sens=k)
        a = dev.reindex(pd.MultiIndex.from_frame(y[["PLAYER_ID", "type", "range"]])).fillna(0).to_numpy()
        sv = sens.reindex(pd.MultiIndex.from_frame(y[["PLAYER_ID", "type"]])).fillna(0).to_numpy()
        is_open = y["range"].isin(OPEN).to_numpy(float) - y["type"].map(share).to_numpy()
        res["dev"][k] = binll(y["lg"] + y["over_t"] + a) - base
        res["sens"][k] = binll(y["lg"] + y["over_t"] + sv * is_open) - base
    res["best_k_dev"] = min(res["dev"], key=res["dev"].get)
    res["best_k_sens"] = min(res["sens"], key=res["sens"].get)
    res["target_season"], res["history_seasons"], res["cells"] = target_season, history_seasons, int(len(y))
    return res


__all__ = ["DASH_RANGES", "compatibility_audit", "contest_features", "contest_tables", "dash_long", "tune_contest"]
