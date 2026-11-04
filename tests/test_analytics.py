import numpy as np
import pandas as pd
import pytest

from pipeline import analytics


def shots_frame():
    rows = [
        # game, event, period, clock, team_id, tricode, player, made, type, prob_a, prob_nba
        ("G1", 1, 1, 700.0, 1, "AAA", 11, 1, "3PT Field Goal", 0.35, 0.30),
        ("G1", 2, 1, 690.0, 2, "BBB", 21, 0, "2PT Field Goal", 0.60, 0.70),
        ("G1", 3, 2, 500.0, 1, "AAA", 12, 1, "2PT Field Goal", 0.50, np.nan),
        ("G1", 4, 4, 10.0, 2, "BBB", 21, 1, "2PT Field Goal", 0.40, 0.45),
    ]
    df = pd.DataFrame(rows, columns=["GAME_ID", "GAME_EVENT_ID", "PERIOD", "pbp_clock_remaining_s", "TEAM_ID",
                                     "pbp_teamTricode", "PLAYER_ID", "SHOT_MADE_FLAG", "SHOT_TYPE", "model_a_xfg", "nba_xfg"])
    df["shot_uid"] = df["GAME_ID"] + "_" + df["GAME_EVENT_ID"].astype(str).str.zfill(4)
    df["PLAYER_NAME"] = "p" + df["PLAYER_ID"].astype(str)
    df["ACTION_TYPE"] = "Jump Shot"
    df["SHOT_ZONE_BASIC"] = np.where(df["SHOT_TYPE"].str.startswith("3"), "Above the Break 3", "Mid-Range")
    df["SHOT_DISTANCE"] = np.where(df["SHOT_TYPE"].str.startswith("3"), 25, 15)
    df["LOC_X"], df["LOC_Y"] = 0, df["SHOT_DISTANCE"] * 10
    df["in_official"] = True
    df["game_date"] = "2026-10-28"
    return analytics.prepare_shots(df)


HEADER = {"game_id": "G1", "away_team_id": 1, "home_team_id": 2, "away_tricode": "AAA", "home_tricode": "BBB",
          "away_name": "A", "home_name": "B", "away_score": 8, "home_score": 6}
TEAM_BOX = pd.DataFrame({"game_id": ["G1", "G1"], "team_id": [1, 2], "team": ["AAA", "BBB"], "freeThrowsMade": [3, 4],
                         "score": [8, 6]})
SCORE = pd.DataFrame({"actionNumber": [1], "actionId": [1], "period": [1], "clock": ["PT11M40.00S"], "elapsed_s": [20.0],
                      "teamTricode": ["AAA"], "actionType": ["3pt"], "home_score": [0], "away_score": [3]})


def test_timeline_separates_fg_points_and_coverage():
    s = shots_frame()
    tl = analytics.game_timeline(s, "G1", HEADER, SCORE, TEAM_BOX)
    teams = {t["team"]: t for t in tl["teams"]}
    assert teams["AAA"]["actual_fg_points"] == 5
    assert teams["AAA"]["actual_fg_points"] + teams["AAA"]["free_throw_points"] == teams["AAA"]["final_score"]
    assert teams["AAA"]["model_a_xpts_fg"] == pytest.approx(0.35 * 3 + 0.5 * 2)
    # NBA xFG covers only one AAA shot; the comparable actual points are on the same shot.
    assert teams["AAA"]["nba_covered_fga"] == 1
    assert teams["AAA"]["nba_xpts_fg_covered"] == pytest.approx(0.9)
    assert teams["AAA"]["actual_fg_points_covered"] == 3
    last_aaa = [x for x in tl["shots"] if x["team_id"] == 1][-1]
    assert last_aaa["cum_actual_fg_points"] == 5
    assert last_aaa["cum_model_a_xpts_fg"] == pytest.approx(2.05)


def test_rankings_order_and_eligibility():
    r = analytics.shot_rankings(shots_frame(), n=5, date="2026-10-28")
    assert r["n_eligible_shots"] == 3
    assert [x["nba_xfg"] for x in r["hardest_makes"]] == [0.30, 0.45]
    assert [x["nba_xfg"] for x in r["easiest_misses"]] == [0.70]


def test_filter_and_aggregate():
    s = shots_frame()
    threes = analytics.filter_shots(s, shot_type="3PT")
    assert len(threes) == 1
    region = analytics.filter_shots(s, region=analytics.CourtRect(-1, 1, 0, 25))
    assert set(region["SHOT_TYPE"]) == {"2PT Field Goal"}
    agg = analytics.aggregate_shots(s, ["team"])
    a = agg.set_index("team").loc["AAA"]
    assert a["points_above_expected"] == pytest.approx(5 - 2.05)


def test_independent_shot_pmf_convolution():
    pmf = analytics.IndependentShotWinProbability._points_pmf([0.5, 0.5], [2, 3])
    assert pmf.tolist() == pytest.approx([0.25, 0, 0.25, 0.25, 0, 0.25])
    assert pmf.sum() == pytest.approx(1.0)


def test_win_probability_symmetry():
    gs = pd.DataFrame({"team_id": [1, 2], "shot_value": [2, 2], "prob": [0.5, 0.5]})
    res = analytics.IndependentShotWinProbability().expected_result(gs, {1: 0, 2: 0})
    assert res[1] == pytest.approx(0.5)


def test_standings_exclude_games_with_missing_probabilities():
    s = shots_frame()
    det = analytics.expected_standings(s, TEAM_BOX, "model_a_xfg")
    # AAA: 0.35*3 + 0.5*2 + 3 FT = 5.05; BBB: 0.6*2 + 0.4*2 + 4 FT = 6.0 (BBB lost the actual game 6-8).
    table = {r["team"]: r for r in det["standings"]}
    assert table["AAA"]["expected_points_for"] == pytest.approx(5.05)
    assert table["BBB"]["expected_points_for"] == pytest.approx(6.0)
    assert (table["AAA"]["wins"], table["AAA"]["expected_wins"]) == (1, 0.0)
    assert (table["BBB"]["wins"], table["BBB"]["expected_wins"]) == (0, 1.0)
    nba = analytics.expected_standings(s, TEAM_BOX, "nba_xfg")
    assert nba["standings"] == [] and nba["excluded_games"][0]["game_id"] == "G1"
