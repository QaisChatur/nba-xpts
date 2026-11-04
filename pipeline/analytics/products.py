"""Product analytics foundation: game timelines, shot rankings, shot explorer, expected standings.

Conventions
-----------
* "FG points" are points from field goals only. Total score (field goals + free throws) is kept
  in separate fields and never mixed with expected field-goal points.
* NBA xFG exists only for shots present in Blueprint. Cumulative NBA expected points are therefore
  reported together with the actual FG points *on the same covered shots* and a coverage count.
* Every shot carries stable references for video sync: shot_uid, official GAME_EVENT_ID
  (play-by-play actionNumber), pbp_actionId and Blueprint ngss_event_id.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from pipeline.matching import clock as clk
from pipeline.matching import court
from pipeline.utils import records, require, require_columns

TIMELINE_REQUIRED = ["shot_uid", "GAME_ID", "GAME_EVENT_ID", "PERIOD", "pbp_clock_remaining_s", "TEAM_ID",
                     "pbp_teamTricode", "PLAYER_ID", "PLAYER_NAME", "SHOT_MADE_FLAG", "SHOT_TYPE",
                     "ACTION_TYPE", "SHOT_ZONE_BASIC", "SHOT_DISTANCE", "LOC_X", "LOC_Y", "model_a_xfg"]


def prepare_shots(unified: pd.DataFrame) -> pd.DataFrame:
    """Official-universe shots with analysis-ready columns (one row per official FGA)."""
    u = unified[unified["in_official"]].copy()
    require_columns(u, TIMELINE_REQUIRED, "unified shots")
    u["shot_value"] = np.where(u["SHOT_TYPE"].str.startswith("3"), 3, 2)
    u["made"] = u["SHOT_MADE_FLAG"].astype(int)
    u["actual_fg_points"] = u["made"] * u["shot_value"]
    u["elapsed_s"] = clk.elapsed_series(u["PERIOD"], u["pbp_clock_remaining_s"])
    u["model_a_xpts_fg"] = u["model_a_xfg"] * u["shot_value"]
    u["nba_xpts_fg"] = u["nba_xfg"] * u["shot_value"]
    x_ft, y_ft = court.court_feet(u["LOC_X"], u["LOC_Y"])
    u["court_x_ft"], u["court_y_ft"] = x_ft, y_ft
    u["team"] = u["pbp_teamTricode"]
    return u.sort_values(["GAME_ID", "elapsed_s", "GAME_EVENT_ID"]).reset_index(drop=True)


SHOT_EVENT_FIELDS = [
    "shot_uid", "GAME_EVENT_ID", "ngss_event_id", "pbp_actionId", "PERIOD", "pbp_clock", "pbp_clock_remaining_s",
    "elapsed_s", "team", "TEAM_ID", "PLAYER_ID", "PLAYER_NAME", "defender_id", "defender_name", "ACTION_TYPE",
    "SHOT_TYPE", "shot_value", "SHOT_ZONE_BASIC", "SHOT_DISTANCE", "bp_shot_distance_ft", "LOC_X", "LOC_Y",
    "court_x_ft", "court_y_ft", "made", "actual_fg_points", "nba_xfg", "nba_xpts_fg", "model_a_xfg",
    "model_a_xpts_fg", "model_b_xfg_exploratory", "master_xfg", "master_xpts_fg", "master_source", "contest_distance_in", "shot_clock_s", "pbp_description",
    "pbp_videoAvailable", "in_blueprint",
]
RENAME_OUT = {"GAME_EVENT_ID": "official_event_id", "pbp_actionId": "pbp_action_id", "PERIOD": "period",
              "pbp_clock": "clock", "pbp_clock_remaining_s": "clock_remaining_s", "TEAM_ID": "team_id",
              "PLAYER_ID": "shooter_id", "PLAYER_NAME": "shooter_name", "ACTION_TYPE": "action_type",
              "SHOT_TYPE": "shot_type", "SHOT_ZONE_BASIC": "shot_zone_basic", "SHOT_DISTANCE": "shot_distance_ft_official",
              "bp_shot_distance_ft": "shot_distance_ft_tracking", "LOC_X": "loc_x", "LOC_Y": "loc_y",
              "pbp_description": "description", "pbp_videoAvailable": "video_available"}


def _events(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in SHOT_EVENT_FIELDS if c in df.columns]
    return df[cols].rename(columns=RENAME_OUT)


def game_timeline(shots: pd.DataFrame, game_id: str, header: dict, score_events: pd.DataFrame,
                  team_box: pd.DataFrame | None = None) -> dict:
    g = shots[shots["GAME_ID"] == game_id].copy()
    require(len(g) > 0, f"no shots for game {game_id}")
    covered = g["nba_xfg"].notna()
    g["covered_actual_fg_points"] = np.where(covered, g["actual_fg_points"], 0)
    for col, src in [("cum_actual_fg_points", "actual_fg_points"), ("cum_model_a_xpts_fg", "model_a_xpts_fg"),
                     ("cum_nba_xpts_fg_covered", "nba_xpts_fg"), ("cum_actual_fg_points_covered", "covered_actual_fg_points")]:
        g[col] = g.groupby("TEAM_ID")[src].transform(lambda s: s.fillna(0).cumsum())
    g["cum_nba_covered_shots"] = g.groupby("TEAM_ID")["nba_xfg"].transform(lambda s: s.notna().cumsum())
    if "master_xpts_fg" in g:
        g["cum_master_xpts_fg"] = g.groupby("TEAM_ID")["master_xpts_fg"].transform(lambda s: s.fillna(0).cumsum())
    if "model_b_xfg_exploratory" in g:
        g["_model_b_xpts"] = g["model_b_xfg_exploratory"] * g["shot_value"]
        g["cum_model_b_xpts_fg_exploratory"] = g.groupby("TEAM_ID")["_model_b_xpts"].transform(lambda s: s.fillna(0).cumsum())
    events = _events(g)
    for c in [c for c in g.columns if c.startswith("cum_")]:
        events[c] = g[c].to_numpy()

    teams = []
    for side in ("away", "home"):
        tid = header[f"{side}_team_id"]
        t = g[g["TEAM_ID"] == tid]
        cov = t["nba_xfg"].notna()
        box = team_box.set_index("team_id").loc[tid] if team_box is not None else None
        teams.append({
            "side": side, "team_id": tid, "team": header[f"{side}_tricode"], "name": header[f"{side}_name"],
            "final_score": header[f"{side}_score"],
            "free_throw_points": int(box["freeThrowsMade"]) if box is not None else None,
            "fga": int(len(t)), "actual_fg_points": int(t["actual_fg_points"].sum()),
            "model_a_xpts_fg": float(t["model_a_xpts_fg"].sum()),
            "master_xpts_fg": float(t["master_xpts_fg"].sum()) if "master_xpts_fg" in t else None,
            "nba_covered_fga": int(cov.sum()),
            "nba_xpts_fg_covered": float(t.loc[cov, "nba_xpts_fg"].sum()),
            "actual_fg_points_covered": int(t.loc[cov, "actual_fg_points"].sum()),
        })
    score = score_events.rename(columns={"actionNumber": "official_event_id", "actionId": "pbp_action_id",
                                         "teamTricode": "team", "actionType": "action_type"})
    return {
        "schema_version": "timeline.v1",
        "game": header,
        "definitions": {
            "actual_fg_points": "points from made field goals only (no free throws)",
            "model_a_xpts_fg": "sum of Model A make probability x shot value over all official FGA",
            "master_xpts_fg": "sum of Master xFG x shot value over all official FGA (tracked shots: Master model; "
                              "untracked shots: no-tracking fallback, see master_source)",
            "nba_xpts_fg_covered": "sum of NBA xFG x shot value over FGA present in Blueprint only",
            "actual_fg_points_covered": "actual FG points on the same Blueprint-covered FGA (compare with nba_xpts_fg_covered)",
            "total_score": "official running score including free throws (score_progression)",
        },
        "teams": teams,
        "shots": records(events),
        "score_progression": records(score[["official_event_id", "pbp_action_id", "period", "clock", "elapsed_s",
                                            "team", "action_type", "home_score", "away_score"]]),
    }


# --------------------------------------------------------------------------- rankings


RANK_FIELDS = ["rank", "shot_uid", "GAME_ID", "game_date", "official_event_id", "ngss_event_id", "period", "clock",
               "team", "shooter_id", "shooter_name", "defender_id", "defender_name", "action_type", "shot_type",
               "shot_zone_basic", "shot_distance_ft_official", "contest_distance_in", "nba_xfg", "model_a_xfg",
               "made", "description", "video_available"]


def shot_rankings(shots: pd.DataFrame, n: int = 5, date: str | None = None, prob_col: str = "nba_xfg") -> dict:
    """Hardest made shots (lowest probability) and easiest misses (highest probability)."""
    d = shots[shots[prob_col].notna()].copy()
    if date is not None:
        d = d[d["game_date"] == date]
    d = pd.concat([d[["GAME_ID", "game_date"]], _events(d)], axis=1)

    def top(frame, ascending):
        t = frame.sort_values([prob_col, "shot_uid"], ascending=[ascending, True]).head(n).copy()
        t.insert(0, "rank", range(1, len(t) + 1))
        return records(t[[c for c in RANK_FIELDS if c in t.columns]])

    return {
        "schema_version": "rankings.v1", "scope": f"date={date}" if date else "all available games",
        "probability": prob_col, "n_eligible_shots": int(len(d)),
        "hardest_makes": top(d[d["made"] == 1], True),
        "easiest_misses": top(d[d["made"] == 0], False),
        "note": "Only shots with a probability are eligible; NBA xFG covers Blueprint shots only.",
    }


# --------------------------------------------------------------------------- shot explorer


@dataclass
class CourtRect:
    """Axis-aligned court region in frontend feet (x across [-25, 25], y from the baseline)."""
    x_min: float
    x_max: float
    y_min: float
    y_max: float


def filter_shots(shots: pd.DataFrame, player_id: int | None = None, team: str | None = None,
                 date_from: str | None = None, date_to: str | None = None, season: str | None = None,
                 zones: list[str] | None = None, shot_type: str | None = None,
                 action_contains: str | None = None, action_types: list[str] | None = None,
                 region: CourtRect | None = None) -> pd.DataFrame:
    """Intersectional filtering on official source labels (no invented action labels)."""
    m = pd.Series(True, index=shots.index)
    if player_id is not None:
        m &= shots["PLAYER_ID"] == player_id
    if team is not None:
        m &= shots["team"] == team
    if date_from is not None:
        m &= shots["game_date"] >= date_from
    if date_to is not None:
        m &= shots["game_date"] <= date_to
    if season is not None:
        m &= shots["SEASON"] == season
    if zones:
        m &= shots["SHOT_ZONE_BASIC"].isin(zones)
    if shot_type is not None:
        m &= shots["SHOT_TYPE"].str.startswith(shot_type[0])
    if action_contains:
        m &= shots["ACTION_TYPE"].str.contains(action_contains, case=False, na=False)
    if action_types:
        m &= shots["ACTION_TYPE"].isin(action_types)
    if region is not None:
        m &= shots["court_x_ft"].between(region.x_min, region.x_max) & shots["court_y_ft"].between(region.y_min, region.y_max)
    return shots[m]


def aggregate_shots(shots: pd.DataFrame, by: list[str], min_attempts: int = 1, prob_col: str = "model_a_xfg") -> pd.DataFrame:
    """Attempts, makes, FG%, average xFG, expected/actual FG points and points above expected."""
    d = shots[shots[prob_col].notna()].copy()
    d["_xpts"] = d[prob_col] * d["shot_value"]
    agg = d.groupby(by, dropna=False).agg(
        attempts=("made", "size"), makes=("made", "sum"), avg_xfg=(prob_col, "mean"),
        actual_fg_points=("actual_fg_points", "sum"), expected_fg_points=("_xpts", "sum"),
    ).reset_index()
    agg["fg_pct"] = agg["makes"] / agg["attempts"]
    agg["fg_pct_above_expected"] = agg["fg_pct"] - agg["avg_xfg"]
    agg["points_above_expected"] = agg["actual_fg_points"] - agg["expected_fg_points"]
    agg["points_above_expected_per_shot"] = agg["points_above_expected"] / agg["attempts"]
    agg["probability_source"] = prob_col
    return agg[agg["attempts"] >= min_attempts].sort_values("points_above_expected", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- expected standings


class ExpectedResultModel(Protocol):
    """Interface for expected-result methods (deterministic now, probabilistic later).

    `expected_result` receives one game's FGA (team_id, shot_value, probability) and actual free-throw
    points per team and returns {team_id: expected_wins} with values summing to 1.
    """

    name: str

    def expected_result(self, game_shots: pd.DataFrame, ft_points: dict[int, int]) -> dict[int, float]: ...


class DeterministicExpectedScore:
    """Expected score = sum of xFG x value over all FGA + actual FT points; higher expected score = 1 win.

    Limitations: free throws are taken as observed; shot outcomes are treated as independent of what
    happened earlier (in reality a different outcome changes possessions, rebounds, fouls and game
    state); FGA volume is taken as given; ties (|margin| < 0.5) split the win.
    """

    name = "deterministic_expected_score.v1"

    def expected_result(self, game_shots, ft_points):
        exp = (game_shots["prob"] * game_shots["shot_value"]).groupby(game_shots["team_id"]).sum()
        teams = list(ft_points)
        score = {t: float(exp.get(t, 0.0)) + ft_points[t] for t in teams}
        a, b = teams
        margin = score[a] - score[b]
        if abs(margin) < 0.5:
            return {a: 0.5, b: 0.5}
        return {a: float(margin > 0), b: float(margin < 0)}


class IndependentShotWinProbability:
    """EXPERIMENTAL: P(win) under independent Bernoulli shot outcomes (exact convolution).

    Same limitations as the deterministic method regarding possession dependence; included only to
    exercise the probabilistic interface, not as a validated expected-wins model.
    """

    name = "independent_shot_win_probability.experimental"

    @staticmethod
    def _points_pmf(probs, values) -> np.ndarray:
        pmf = np.array([1.0])
        for p, v in zip(probs, values):
            new = np.zeros(len(pmf) + v)
            new[: len(pmf)] += pmf * (1 - p)
            new[v:] += pmf * p
            pmf = new
        return pmf

    def expected_result(self, game_shots, ft_points):
        a, b = list(ft_points)
        pa = self._points_pmf(*[game_shots.loc[game_shots["team_id"] == a, c].to_numpy() for c in ("prob", "shot_value")])
        pb = self._points_pmf(*[game_shots.loc[game_shots["team_id"] == b, c].to_numpy() for c in ("prob", "shot_value")])
        sa = np.arange(len(pa)) + ft_points[a]
        sb = np.arange(len(pb)) + ft_points[b]
        diff = sa[:, None] - sb[None, :]
        joint = pa[:, None] * pb[None, :]
        p_win = float(joint[diff > 0].sum() + 0.5 * joint[diff == 0].sum())
        return {a: p_win, b: 1 - p_win}


def expected_standings(shots: pd.DataFrame, team_boxes: pd.DataFrame, prob_col: str,
                       method: ExpectedResultModel | None = None) -> dict:
    """Actual vs expected record per team. Games lacking a probability for any FGA are excluded and listed."""
    method = method or DeterministicExpectedScore()
    games, excluded = [], []
    for gid, g in shots.groupby("GAME_ID"):
        box = team_boxes[team_boxes["game_id"] == gid]
        require(len(box) == 2, f"{gid}: expected 2 team box rows")
        if g[prob_col].isna().any():
            excluded.append({"game_id": gid, "reason": f"{int(g[prob_col].isna().sum())} of {len(g)} FGA lack {prob_col}"})
            continue
        ft = {int(r.team_id): int(r.freeThrowsMade) for r in box.itertuples()}
        gs = pd.DataFrame({"team_id": g["TEAM_ID"].astype(int), "shot_value": g["shot_value"], "prob": g[prob_col]})
        xw = method.expected_result(gs, ft)
        exp_pts = {t: float((gs.loc[gs.team_id == t, "prob"] * gs.loc[gs.team_id == t, "shot_value"]).sum()) + ft[t] for t in ft}
        actual = {int(r.team_id): int(r.score) for r in box.itertuples()}
        for r in box.itertuples():
            t, o = int(r.team_id), [x for x in ft if x != int(r.team_id)][0]
            games.append({"game_id": gid, "team_id": t, "team": r.team, "actual_points": actual[t],
                          "opp_points": actual[o], "expected_points": exp_pts[t], "opp_expected_points": exp_pts[o],
                          "win": int(actual[t] > actual[o]), "expected_wins": xw[t]})
    gdf = pd.DataFrame(games)
    table = pd.DataFrame()
    if len(gdf):
        table = gdf.groupby(["team_id", "team"]).agg(
            games=("game_id", "size"), wins=("win", "sum"), expected_wins=("expected_wins", "sum"),
            points_for=("actual_points", "sum"), points_against=("opp_points", "sum"),
            expected_points_for=("expected_points", "sum"), expected_points_against=("opp_expected_points", "sum"),
        ).reset_index()
        table["losses"] = table["games"] - table["wins"]
        table["expected_losses"] = table["games"] - table["expected_wins"]
        table["point_diff"] = table["points_for"] - table["points_against"]
        table["expected_point_diff"] = table["expected_points_for"] - table["expected_points_against"]
        table["wins_over_expected"] = table["wins"] - table["expected_wins"]
        table = table.sort_values(["expected_wins", "expected_point_diff"], ascending=False)
    return {"schema_version": "standings.v1", "method": method.name, "probability": prob_col,
            "limitations": (method.__doc__ or "").strip(), "games": records(gdf) if len(gdf) else [],
            "standings": records(table) if len(table) else [], "excluded_games": excluded}
