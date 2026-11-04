"""Shot-context features: action vocabulary, geometry, game time and pre-shot play-by-play context.

Everything here is known before the ball leaves the shooter's hand. Explicitly excluded:
  * the play-by-play score on a shot row (it is the score *after* the shot),
  * `description` (contains "MISS", points and assists), assist/block fields,
  * the "Bank" token of ACTION_TYPE. Historical audit (540,627 ShotChartDetail shots):
    "Step Back Bank Jump Shot" 59.2% FG at 18.6 ft vs "Step Back Jump shot" 37.6% at 22.2 ft,
    "Jump Bank Shot" 49.6% at 16.2 ft vs mid-range jump shots ~40%, i.e. scorers record "bank"
    predominantly when the shot goes in. Bank variants are folded into their base action.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from pipeline.matching import clock as clk

ACTION_FLAGS = {
    "act_driving": r"driving",
    "act_running": r"running",
    "act_cutting": r"cutting",
    "act_pullup": r"pull",
    "act_stepback": r"step back",
    "act_fadeaway": r"fadeaway",
    "act_turnaround": r"turnaround",
    "act_floater": r"float",
    "act_hook": r"hook",
    "act_layup": r"layup",
    "act_dunk": r"dunk",
    "act_tip_putback": r"tip|putback",
    "act_alley_oop": r"alley oop",
    "act_reverse": r"reverse",
    "act_finger_roll": r"finger roll",
}

ZONE_GROUPS = {
    "Restricted Area": "rim", "In The Paint (Non-RA)": "paint", "Mid-Range": "mid",
    "Left Corner 3": "corner3", "Right Corner 3": "corner3", "Above the Break 3": "ab3", "Backcourt": "ab3",
}
ZONE_GROUP_ORDER = ["rim", "paint", "mid", "corner3", "ab3"]


def normalize_action(action: pd.Series) -> pd.Series:
    """Lower-case ACTION_TYPE with the outcome-biased "bank" token removed."""
    s = action.fillna("unknown").astype(str).str.lower()
    s = s.str.replace(r"\bbank\b", " ", regex=True)
    s = s.str.replace(r"\bshot\b", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
    s = s.replace({"turnaround": "turnaround jump", "fadeaway": "fadeaway jump", "turnaround fadeaway jump": "turnaround fadeaway",
                   "jump": "jump", "hook": "hook"})
    return s


def action_family(norm: pd.Series) -> pd.Series:
    """Coarse family used for player priors (stable sample sizes)."""
    conds = [
        norm.str.contains("dunk"), norm.str.contains("tip|putback"), norm.str.contains("alley oop"),
        norm.str.contains("layup|finger roll"), norm.str.contains("hook"), norm.str.contains("float"),
        norm.str.contains("step back"), norm.str.contains("pull"),
        norm.str.contains("fadeaway|turnaround"),
    ]
    names = ["dunk", "tip_putback", "alley_oop", "layup", "hook", "floater", "stepback", "pullup", "post_fade"]
    return pd.Series(np.select(conds, names, default="jump"), index=norm.index)


def shot_features(df: pd.DataFrame) -> pd.DataFrame:
    """Static shot features from ShotChartDetail-schema columns (works for historical and 2026-27 shots)."""
    out = pd.DataFrame(index=df.index)
    x = pd.to_numeric(df["LOC_X"], errors="coerce").astype(float)
    y = pd.to_numeric(df["LOC_Y"], errors="coerce").astype(float)
    is3 = df["SHOT_TYPE"].astype(str).str.startswith("3")
    out["is_three"] = is3.astype(int)
    out["shot_distance_ft"] = np.hypot(x, y) / 10.0
    out["loc_x"] = x
    out["loc_y"] = y
    out["abs_loc_x"] = x.abs()
    out["shot_angle_deg"] = np.degrees(np.arctan2(x.abs(), y.clip(lower=-52.5) + 1e-6))
    zone = df["SHOT_ZONE_BASIC"].astype(str)
    out["zone_group"] = zone.map(ZONE_GROUPS).fillna("mid")
    out["is_corner3"] = (out["zone_group"] == "corner3").astype(int)
    out["is_backcourt"] = (zone == "Backcourt").astype(int)
    norm = normalize_action(df["ACTION_TYPE"])
    out["action_norm"] = norm
    out["action_family"] = action_family(norm)
    for name, pat in ACTION_FLAGS.items():
        out[name] = norm.str.contains(pat, regex=True).astype(int)

    period = pd.to_numeric(df["PERIOD"], errors="coerce").astype(int)
    remaining = pd.to_numeric(df["MINUTES_REMAINING"], errors="coerce") * 60 + pd.to_numeric(df["SECONDS_REMAINING"], errors="coerce")
    if "clock_remaining_s" in df:
        remaining = pd.to_numeric(df["clock_remaining_s"], errors="coerce").fillna(remaining)
    out["period"] = period
    out["is_overtime"] = (period > 4).astype(int)
    out["period_seconds_remaining"] = remaining
    plen = np.where(period <= 4, clk.REGULATION_PERIOD_S, clk.OVERTIME_PERIOD_S)
    start = np.where(period <= 4, (period - 1) * 720, 2880 + (period - 5) * 300)
    out["game_seconds_elapsed"] = start + (plen - remaining)
    out["end_of_period_heave_window"] = (remaining <= 2).astype(int)
    return out


# --------------------------------------------------------------------------- play-by-play sequence context


def _action_team(a: pd.DataFrame) -> pd.Series:
    team = pd.to_numeric(a.get("teamId"), errors="coerce")
    person = pd.to_numeric(a.get("personId"), errors="coerce")
    is_team_person = person.between(1610612700, 1610612799)
    return team.where(team.notna() & (team > 0), person.where(is_team_person))


def pbp_context(actions: pd.DataFrame, home_team_id: int, away_team_id: int) -> pd.DataFrame:
    """Pre-shot context for every field-goal attempt of one game, from earlier play-by-play actions.

    Only actions *before* the shot in feed order are used. Possession tracking: a possession starts at
    a period start / jump ball, a defensive or offensive rebound, an opponent turnover, or an opponent
    score (made FG or last made free throw). Returns one row per field goal keyed by GAME_EVENT_ID.
    """
    a = actions.reset_index(drop=True).copy()
    a["team"] = _action_team(a)
    a["remaining"] = clk.parse_clock_series(a["clock"].fillna("PT12M00.00S"))
    sh = pd.to_numeric(a["scoreHome"], errors="coerce").to_numpy()
    sa = pd.to_numeric(a["scoreAway"], errors="coerce").to_numpy()

    def other(t):
        return away_team_id if t == home_team_id else home_team_id

    rows = []
    last_home = last_away = 0
    poss_team, start_type, start_rem, start_period, timeouts = None, "period_start", 720.0, 1, 0
    last_miss_team = None
    fga: dict = {}
    fgm: dict = {}

    def new_poss(team, kind, rem, per):
        nonlocal poss_team, start_type, start_rem, start_period, timeouts
        poss_team, start_type, start_rem, start_period, timeouts = team, kind, rem, per, 0

    for i, r in enumerate(a.itertuples(index=False)):
        at = r.actionType if isinstance(r.actionType, str) else ""
        team = None if pd.isna(r.team) else int(r.team)
        rem, per = float(r.remaining), int(r.period)
        sub = str(getattr(r, "subType", "") or "")

        if at == "period" and sub.lower() == "start":
            new_poss(None, "period_start", rem, per)
            last_miss_team = None
        elif r.isFieldGoal == 1 and team is not None:
            pid = int(r.personId)
            is_home = int(team == home_team_id)
            ok = poss_team in (team, None) and start_period == per
            rows.append({
                "GAME_EVENT_ID": int(r.actionNumber),
                "is_home": is_home,
                "score_margin_pre": (last_home - last_away) * (1 if is_home else -1),
                "score_total_pre": last_home + last_away,
                "poss_start_type": start_type if ok else "unknown",
                "seconds_since_poss_start": (start_rem - rem) if ok else np.nan,
                "second_chance": int(ok and start_type == "off_rebound"),
                "after_timeout": int(ok and timeouts > 0),
                "shooter_fga_before_game": fga.get(pid, 0),
                "shooter_fgm_before_game": fgm.get(pid, 0),
            })
            made = str(r.shotResult) == "Made"
            fga[pid] = fga.get(pid, 0) + 1
            fgm[pid] = fgm.get(pid, 0) + int(made)
            if made:
                new_poss(other(team), "opp_score", rem, per)
                last_miss_team = None
            else:
                poss_team, last_miss_team = team, team
        elif at == "Rebound" and team is not None:
            if last_miss_team is not None:
                new_poss(team, "off_rebound" if team == last_miss_team else "def_rebound", rem, per)
            last_miss_team = None
        elif at == "Turnover" and team is not None:
            new_poss(other(team), "opp_turnover", rem, per)
            last_miss_team = None
        elif at == "Free Throw" and team is not None:
            m = re.search(r"(\d) of (\d)", sub)
            if m and m.group(1) == m.group(2):
                if "MISS" in str(r.description).upper():
                    last_miss_team = team
                else:
                    new_poss(other(team), "opp_score", rem, per)
                    last_miss_team = None
        elif at == "Timeout":
            timeouts += 1
        elif at == "Jump Ball":
            new_poss(None, "jump_ball", rem, per)

        if not (np.isnan(sh[i]) or np.isnan(sa[i])):
            last_home, last_away = int(sh[i]), int(sa[i])

    return pd.DataFrame(rows)
