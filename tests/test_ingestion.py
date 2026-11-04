"""Season ingestion: Blueprint extraction, listing parsing and per-game verification on the committed sample."""

import json

import pytest

from pipeline.config import SAMPLES_DIR
from pipeline.ingestion import season
from pipeline.ingestion.blueprint_scrape import discover_games_in_html, extract_shots, to_row

SAMPLE = SAMPLES_DIR / "0022600133"


def _shot(event, xfg=0.4, period=1, clock="10:00", shooter=True):
    s = {"period": period, "gameClockTime": clock, "shotClockTime": 12.0, "shotType": "2pt", "homeTeamScore": 2,
         "awayTeamScore": 0, "success": True, "featured": False, "expFieldGoalPct": xfg, "ngssEventId": event,
         "x": 10, "y": 20, "attributes": {"shotDistance": 3.0, "result": "MADE"}}
    if shooter:
        s["shooter"] = {"playerId": 1, "firstName": "A", "lastName": "B", "teamAbbreviation": "XYZ", "teamId": 9}
    return s


def test_extract_plain_and_flight_escaped_records_are_deduplicated():
    plain = json.dumps(_shot(4))
    escaped = json.dumps(json.dumps(_shot(4)))[1:-1]  # React Flight style \" escaping
    shots, audit = extract_shots(f"<html>{plain} junk {escaped}</html>")
    assert len(shots) == 1 and audit["records_rejected"] == 0


def test_extract_reports_rejections_instead_of_silently_dropping():
    text = " ".join(json.dumps(s) for s in [_shot(1), _shot(2, xfg=None), _shot(3, shooter=False), _shot(5, xfg=1.7)])
    shots, audit = extract_shots(text)
    assert [s["ngssEventId"] for s in shots] == [1]
    assert sorted(r["reason"] for r in audit["rejections"]) == ["missing_xfg", "no_shooter", "xfg_out_of_range"]


def test_to_row_expected_points():
    row = to_row("0022600001", "2026-10-20", {**_shot(7, xfg=0.35), "shotType": "3pt"}, "u")
    assert row["xpts_fg_only"] == pytest.approx(1.05) and row["date"] == "2026-10-20"


def test_listing_parser_only_regular_season_ids():
    html = ('<a href="/game/bos-vs-nyk-0022600005">x</a> <a href="/game/lal-vs-gsw-0012600001">pre</a>'
            '<a href="https://www.nba.com/game/den-vs-phx-0022600006/box-score">y</a>')
    games = discover_games_in_html(html, "2026-10-21")
    assert [(g["game_id"], g["away"], g["home"]) for g in games] == [
        ("0022600005", "BOS", "NYK"), ("0022600006", "DEN", "PHX")]


def test_daterange_inclusive():
    assert season.daterange("2026-10-30", "2026-11-02") == ["2026-10-30", "2026-10-31", "2026-11-01", "2026-11-02"]


def test_verify_sample_game_end_to_end():
    rec = season.verify_game("0022600133", "2026-10-28", pages_dir=SAMPLE / "nba_pages", bp_dir=SAMPLE / "blueprint")
    assert rec["status"] == "verified"
    assert rec["official_date"] == "2026-10-28" and rec["listing_date_correct"] is True
    assert rec["pbp_matches_box"] and rec["official_fga"] == rec["pbp_fga"] == 174
    assert rec["blueprint_extracted"] == 169
    assert rec["matched"] + rec["ambiguous"] + rec["unmatched_blueprint"] + rec["rejected_id_conflict"] == 169
    assert rec["matched"] + rec["official_without_blueprint"] == rec["official_fga"]
    assert 0 < rec["coverage_pct"] <= 100


@pytest.mark.parametrize("desc,sub,expected", [
    ("", "Jump Shot", "Jump Shot"), ("pullup", "Jump Shot", "Pullup Jump shot"),
    ("running pullup", "Jump Shot", "Running Pull-Up Jump Shot"), ("cutting", "DUNK", "Cutting Dunk Shot"),
    ("turnaround fadeaway", "Jump Shot", "Turnaround Fadeaway shot"), ("turnaround bank", "Jump Shot", "Turnaround Bank shot"),
    ("running alley-oop", "Layup", "Running Alley Oop Layup Shot"), (None, "Hook", "Hook Shot"),
    ("bank", "Hook Shot", "Hook Bank Shot"), ("bank", "Jump Shot", "Jump Bank Shot"),
    ("pullup bank", "Jump Shot", "Pullup Jump shot"),
])
def test_live_action_type_maps_to_shotchart_vocabulary(desc, sub, expected):
    from pipeline.ingestion.nba_pages import live_action_type
    assert live_action_type(desc, sub) == expected


def test_live_actions_normalized_for_context():
    from pipeline.features.shot_context import pbp_context
    from pipeline.ingestion.nba_pages import normalize_live_actions
    H, A = 1610612761, 1610612741
    acts = [
        {"actionNumber": 1, "orderNumber": 10, "clock": "PT12M00.00S", "period": 1, "actionType": "period", "subType": "start",
         "isFieldGoal": 0, "scoreHome": "0", "scoreAway": "0"},
        {"actionNumber": 3, "orderNumber": 30, "clock": "PT11M40.00S", "period": 1, "teamId": A, "personId": 7, "actionType": "3pt",
         "subType": "Jump Shot", "descriptor": "pullup", "isFieldGoal": 1, "shotResult": "Missed", "scoreHome": "0", "scoreAway": "0"},
        {"actionNumber": 2, "orderNumber": 20, "clock": "PT11M50.00S", "period": 1, "teamId": H, "personId": 5, "actionType": "2pt",
         "subType": "Layup", "descriptor": "driving", "isFieldGoal": 1, "shotResult": "Made", "scoreHome": "2", "scoreAway": "0"},
        {"actionNumber": 4, "orderNumber": 40, "clock": "PT11M38.00S", "period": 1, "teamId": A, "personId": 8, "actionType": "rebound",
         "subType": "offensive", "isFieldGoal": 0, "scoreHome": "2", "scoreAway": "0"},
        {"actionNumber": 5, "orderNumber": 50, "clock": "PT11M36.00S", "period": 1, "teamId": A, "personId": 8, "actionType": "2pt",
         "subType": "Layup", "descriptor": "putback", "isFieldGoal": 1, "shotResult": "Made", "scoreHome": "2", "scoreAway": "2"},
    ]
    a = normalize_live_actions(acts, H)
    assert a["actionNumber"].tolist() == [1, 2, 3, 4, 5]  # sorted by orderNumber
    assert a.loc[a["isFieldGoal"] == 1, "subType"].tolist() == ["Driving Layup Shot", "Pullup Jump shot", "Putback Layup Shot"]
    c = pbp_context(a, H, A).set_index("GAME_EVENT_ID")
    assert c.loc[3, "score_margin_pre"] == -2 and c.loc[3, "poss_start_type"] == "opp_score"
    assert c.loc[5, "second_chance"] == 1 and c.loc[5, "score_margin_pre"] == -2  # pre-shot score, not 2-2


def test_wrong_listing_date_is_flagged():
    rec = season.verify_game("0022600133", "2026-10-27", pages_dir=SAMPLE / "nba_pages", bp_dir=SAMPLE / "blueprint")
    assert rec["listing_date_correct"] is False and rec["official_date"] == "2026-10-28"
