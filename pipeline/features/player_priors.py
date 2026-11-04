"""Player and defender priors with empirical-Bayes shrinkage, computed strictly from earlier data.

Sources, by availability time:
  * pre-season, shot level: ShotChartDetail 2022-23 .. 2024-25 (recency-weighted),
  * pre-season, aggregate: PlayerDashPtShots 2021-22 .. 2025-26 (2PT/3PT by closest-defender range;
    2025-26 is the only source for last season),
  * in-season: official 2026-27 play-by-play shots from games dated strictly before the shot's
    game date (same-day games are excluded, so no prior can contain the shot or its game).

For a player p and category c (all / two / three / zone group / action family):
  fg_post   = (made_w + k_fg * league_fg[c]) / (att_w + k_fg)
  smoe_post = sum_w(made - baseline) / (att_w + k_smoe)          (shrunk toward 0)
where `baseline` is the historical shot-difficulty model (pipeline.features.baseline).
Defenders: smoe allowed as closest Blueprint defender, in-season only, shrunk toward 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.features.shot_context import ZONE_GROUP_ORDER

CATEGORY_KINDS = ("all", "type", "zone", "family")
DASH_RANGES = {"0-2 Feet - Very Tight": "tight", "2-4 Feet - Tight": "close", "4-6 Feet - Open": "open",
               "6+ Feet - Wide Open": "wide"}
RANGE_ORDER = ["tight", "close", "open", "wide"]


def contest_range(contest_distance_in: pd.Series) -> pd.Series:
    ft = pd.to_numeric(contest_distance_in, errors="coerce") / 12.0
    return pd.Series(np.select([ft < 2, ft < 4, ft < 6, ft >= 6], RANGE_ORDER, default="unknown"), index=ft.index)


def long_categories(shots: pd.DataFrame) -> pd.DataFrame:
    """Explode shots into (category kind, category value) rows: one row per kind per shot."""
    base = shots[["PLAYER_ID", "w", "made", "resid"]]
    parts = [
        base.assign(kind="all", cat="all"),
        base.assign(kind="type", cat=np.where(shots["is_three"] == 1, "three", "two")),
        base.assign(kind="zone", cat=shots["zone_group"].to_numpy()),
        base.assign(kind="family", cat=shots["action_family"].to_numpy()),
    ]
    return pd.concat(parts, ignore_index=True)


def aggregate(shots: pd.DataFrame, key: str = "PLAYER_ID") -> pd.DataFrame:
    """Weighted attempts / makes / residual sums per (key, kind, cat)."""
    long = long_categories(shots.rename(columns={key: "PLAYER_ID"}) if key != "PLAYER_ID" else shots)
    long["wm"] = long["w"] * long["made"]
    long["wr"] = long["w"] * long["resid"]
    g = long.groupby(["PLAYER_ID", "kind", "cat"], sort=False).agg(att=("w", "sum"), made=("wm", "sum"), resid=("wr", "sum"))
    return g


def league_rates(shots: pd.DataFrame) -> dict[tuple[str, str], float]:
    long = long_categories(shots)
    r = long.groupby(["kind", "cat"]).apply(lambda d: np.average(d["made"], weights=d["w"]), include_groups=False)
    return {k: float(v) for k, v in r.items()}


def dash_tables(dash: pd.DataFrame, last_season: str = "2025-26") -> tuple[pd.DataFrame, pd.DataFrame]:
    """(last-season 2PT/3PT totals per player, all-season 2PT/3PT by defender range per player)."""
    d = dash.copy()
    d["range"] = d["CLOSE_DEF_DIST_RANGE"].map(DASH_RANGES)
    num = ["FG2M", "FG2A", "FG3M", "FG3A"]
    d[num] = d[num].apply(pd.to_numeric, errors="coerce").fillna(0)
    last = d[d["SEASON"] == last_season].groupby("PLAYER_ID")[num].sum()
    by_range = d.groupby(["PLAYER_ID", "range"])[num].sum()
    return last, by_range


class PriorBuilder:
    """Combines pre-season tables and in-season shots into per-shot prior features."""

    def __init__(self, hist_agg: pd.DataFrame, league: dict, dash_last: pd.DataFrame, dash_range: pd.DataFrame,
                 hist_season_smoe_sd: pd.Series, params: dict):
        self.hist = hist_agg
        self.league = league
        self.dash_last = dash_last
        self.dash_range = dash_range
        self.season_sd = hist_season_smoe_sd
        self.p = params
        rng = dash_range.groupby("range").sum()
        self.league_range = {("two", r): rng.loc[r, "FG2M"] / max(rng.loc[r, "FG2A"], 1) for r in rng.index}
        self.league_range |= {("three", r): rng.loc[r, "FG3M"] / max(rng.loc[r, "FG3A"], 1) for r in rng.index}

    def _lookup(self, table: pd.DataFrame, pid: pd.Series, kind: str, cat) -> pd.DataFrame:
        cat = pd.Series(cat, index=pid.index) if not isinstance(cat, pd.Series) else cat
        idx = pd.MultiIndex.from_arrays([pid.to_numpy(), np.repeat(kind, len(pid)), cat.astype(str).to_numpy()])
        return table.reindex(idx).astype(float).fillna(0.0).set_axis(pid.index)

    def features(self, shots: pd.DataFrame, season_agg: pd.DataFrame | None, def_agg: pd.DataFrame | None) -> pd.DataFrame:
        """Prior features for `shots` (needs PLAYER_ID, is_three, zone_group, action_family and, for the
        tracking interaction, contest_distance_in / defender_id). `season_agg` / `def_agg` must only
        contain games dated before these shots."""
        p = self.p
        pid = shots["PLAYER_ID"].astype(int)
        typ = pd.Series(np.where(shots["is_three"] == 1, "three", "two"), index=shots.index)
        empty = pd.DataFrame(columns=["att", "made", "resid"], index=pd.MultiIndex.from_arrays([[], [], []]))
        season_agg = empty if season_agg is None else season_agg
        out = pd.DataFrame(index=shots.index)

        def combined(kind, cat):
            h = self._lookup(self.hist, pid, kind, cat)
            s = self._lookup(season_agg, pid, kind, cat)
            return h, s

        for kind, cat, k_fg in (("all", "all", p["k_fg_all"]), ("type", typ, p["k_fg_type"]),
                                ("zone", shots["zone_group"], p["k_fg_zone"]), ("family", shots["action_family"], p["k_fg_family"])):
            h, s = combined(kind, cat)
            att, made, resid = h["att"] + s["att"], h["made"] + s["made"], h["resid"] + s["resid"]
            catv = pd.Series(cat, index=shots.index) if not isinstance(cat, pd.Series) else cat
            lg = catv.astype(str).map(lambda c, kind=kind: self.league.get((kind, c), np.nan)).astype(float)
            if kind == "type":  # add last season's (2025-26) aggregate counts, which have no residuals
                dl = self.dash_last.reindex(pid.to_numpy()).astype(float).fillna(0).set_axis(shots.index)
                w = p["w_dash_last"]
                made = made + w * np.where(typ == "three", dl["FG3M"], dl["FG2M"])
                att_fg = att + w * np.where(typ == "three", dl["FG3A"], dl["FG2A"])
            else:
                att_fg = att
            name = kind if kind == "all" else f"{kind}"
            out[f"pl_fg_{name}_post"] = (made + k_fg * lg) / (att_fg + k_fg)
            out[f"pl_smoe_{name}_post"] = resid / (h["att"] + s["att"] + p["k_smoe"])
            out[f"pl_log_att_{name}"] = np.log1p(att_fg)
            if kind == "all":
                out["pl_season_att"] = s["att"]
                out["pl_has_history"] = (h["att"] > 0).astype(int)

        # shot diet (pre-season history + in-season), shrunk toward league shares
        all_att = self._lookup(self.hist, pid, "all", "all")["att"] + self._lookup(season_agg, pid, "all", "all")["att"]
        three_att = self._lookup(self.hist, pid, "type", "three")["att"] + self._lookup(season_agg, pid, "type", "three")["att"]
        rim_att = self._lookup(self.hist, pid, "zone", "rim")["att"] + self._lookup(season_agg, pid, "zone", "rim")["att"]
        k = p["k_rate"]
        out["pl_three_rate"] = (three_att + k * p["league_three_rate"]) / (all_att + k)
        out["pl_rim_rate"] = (rim_att + k * p["league_rim_rate"]) / (all_att + k)

        # consistency: SD of per-season smoe in 2022-25 (league median where unavailable)
        out["pl_smoe_season_sd"] = pid.map(self.season_sd).fillna(float(self.season_sd.median())).to_numpy()

        # player x tracking interaction: FG% in the current shot's closest-defender range (PlayerDashPtShots)
        if "contest_distance_in" in shots:
            rng = contest_range(shots["contest_distance_in"])
            idx = pd.MultiIndex.from_arrays([pid.to_numpy(), rng.to_numpy()])
            dr = self.dash_range.reindex(idx).astype(float).fillna(0).set_axis(shots.index)
            m = np.where(typ == "three", dr["FG3M"], dr["FG2M"])
            a = np.where(typ == "three", dr["FG3A"], dr["FG2A"])
            lg = np.array([self.league_range.get((t, r), np.nan) for t, r in zip(typ, rng)], dtype=float)
            out["pl_fg_def_range_post"] = (m + p["k_fg_range"] * lg) / (a + p["k_fg_range"])
            out["pl_fg_def_range_vs_league"] = out["pl_fg_def_range_post"] - lg

        # defender: in-season smoe allowed when closest defender (Blueprint), shrunk to 0
        if "defender_id" in shots:
            did = pd.to_numeric(shots["defender_id"], errors="coerce").fillna(-1).astype(int)
            if def_agg is None or len(def_agg) == 0:
                d = pd.DataFrame({"att": 0.0, "resid": 0.0}, index=shots.index)
            else:
                d = self._lookup(def_agg, did, "all", "all")
            out["def_smoe_allowed_post"] = d["resid"] / (d["att"] + p["k_def"])
            out["def_contests_season"] = d["att"]
        return out


def season_smoe_sd(hist_shots: pd.DataFrame, min_att: int = 100) -> pd.Series:
    g = hist_shots.groupby(["PLAYER_ID", "SEASON"]).agg(n=("resid", "size"), r=("resid", "mean"))
    g = g[g["n"] >= min_att]
    sd = g.groupby("PLAYER_ID")["r"].std()
    return sd.dropna()


__all__ = ["PriorBuilder", "aggregate", "contest_range", "dash_tables", "league_rates", "season_smoe_sd",
           "ZONE_GROUP_ORDER"]
