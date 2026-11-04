"""Synthetic matching scenarios: ID corroboration, perturbations, ambiguity, one-to-one, conflicts."""

import numpy as np
import pandas as pd

from pipeline import matching

GID = "0022600999"


def official(rows):
    df = pd.DataFrame(rows, columns=["GAME_EVENT_ID", "PERIOD", "PLAYER_ID", "SHOT_MADE_FLAG", "SHOT_TYPE", "LOC_X", "LOC_Y",
                                     "clock_remaining_s"])
    df.insert(0, "GAME_ID", GID)
    return df


def blueprint(rows):
    df = pd.DataFrame(rows, columns=["ngss_event_id", "period", "shooter_id", "made", "shot_value", "bp_x", "bp_y",
                                     "clock_remaining_s"])
    df.insert(0, "game_id", GID)
    df.insert(0, "shot_uid", [f"{GID}_{i:04d}" for i in df["ngss_event_id"]])
    return df


OFF = official([
    (10, 1, 101, 1, "2PT Field Goal", 0, 10, 700.0),
    (20, 1, 102, 0, "3PT Field Goal", 230, 20, 650.0),
    (30, 2, 101, 1, "2PT Field Goal", -50, 60, 500.0),
    (40, 2, 103, 0, "2PT Field Goal", 10, 5, 300.0),
])


def status_of(res, uid):
    return res.bp_status.set_index("shot_uid").loc[uid]


def test_exact_ids_are_corroborated():
    bp = blueprint([(10, 1, 101, 1, 2, 0, 10, 700.0), (20, 1, 102, 0, 3, 230, 20, 650.0)])
    res = matching.match_shots(bp, OFF)
    assert (res.bp_status["match_method"] == "event_id_verified").all()
    assert res.hypothesis.iloc[0]["id_hypothesis_support"] == 1.0
    assert sorted(res.unmatched_official["GAME_EVENT_ID"]) == [30, 40]


def test_clock_and_location_perturbation_within_tolerance():
    bp = blueprint([(10, 1, 101, 1, 2, 5, 12, 701.5)])
    st = status_of(matching.match_shots(bp, OFF), f"{GID}_0010")
    assert st["match_status"] == "matched"
    assert st["clock_diff_s"] == 1.5


def test_offset_ids_fall_back_to_attributes():
    # Blueprint IDs shifted by +1: the ID pairing fails, attribute matching recovers each shot uniquely.
    bp = blueprint([(11, 1, 101, 1, 2, 0, 10, 700.0), (31, 2, 101, 1, 2, -50, 60, 500.0)])
    res = matching.match_shots(bp, OFF)
    st = res.bp_status.set_index("ngss_event_id")
    assert st.loc[11, "official_event_id"] == 10
    assert st.loc[31, "official_event_id"] == 30
    assert set(st["match_method"]) == {"attributes_unique"}


def test_result_conflict_is_rejected_not_forced():
    bp = blueprint([(10, 1, 101, 0, 2, 0, 10, 700.0)])  # Blueprint says miss, official says make
    st = status_of(matching.match_shots(bp, OFF), f"{GID}_0010")
    assert st["match_status"] == "rejected_id_conflict"
    assert st["id_pair_rejection"] == "result"


def test_out_of_tolerance_is_unmatched():
    bp = blueprint([(99, 1, 101, 1, 2, 0, 10, 690.0)])  # 10 s off, unknown ID
    st = status_of(matching.match_shots(bp, OFF), f"{GID}_0099")
    assert st["match_status"] == "unmatched"


def test_ambiguous_candidates_are_not_guessed():
    off = official([
        (10, 1, 101, 1, "2PT Field Goal", 0, 10, 700.0),
        (11, 1, 101, 1, "2PT Field Goal", 1, 10, 700.0),  # putback, near-identical
    ])
    bp = blueprint([(50, 1, 101, 1, 2, 0, 10, 700.0)])
    st = status_of(matching.match_shots(bp, off), f"{GID}_0050")
    assert st["match_status"] == "ambiguous"
    assert st["n_candidates"] == 2


def test_one_to_one_conflict_leaves_both_ambiguous():
    off = official([(10, 1, 101, 1, "2PT Field Goal", 0, 10, 700.0)])
    bp = blueprint([(50, 1, 101, 1, 2, 0, 10, 700.0), (51, 1, 101, 1, 2, 0, 11, 700.0)])
    res = matching.match_shots(bp, off)
    assert (res.bp_status["match_status"] == "ambiguous").all()
    assert len(res.unmatched_official) == 1


def test_value_disagreement_matches_but_is_flagged():
    bp = blueprint([(20, 1, 102, 0, 2, 230, 20, 650.0)])
    st = status_of(matching.match_shots(bp, OFF), f"{GID}_0020")
    assert st["match_status"] == "matched"
    assert not st["value_agree"]


def test_unified_keeps_every_source_row():
    bp = blueprint([(10, 1, 101, 1, 2, 0, 10, 700.0), (10_000, 1, 999, 1, 2, 0, 0, 1.0)])
    off = OFF.assign(game_date="2026-10-28")
    res = matching.match_shots(bp, off)
    u = matching.build_unified(off, bp, res)
    assert len(u) == len(OFF) + 1
    assert u["shot_uid"].is_unique
    assert u.loc[u["in_blueprint"] & u["in_official"], "shot_uid"].tolist() == [f"{GID}_0010"]
    assert u.loc[~u["in_official"], "shot_uid"].tolist() == [f"{GID}_bp10000"]
    assert not any(c.endswith(("_x", "_y")) and c not in ("bp_x", "bp_y") for c in u.columns)
    assert u["GAME_EVENT_ID"].dtype == "Int64"


def test_duplicate_official_ids_raise():
    off = pd.concat([OFF, OFF.iloc[:1]])
    bp = blueprint([(10, 1, 101, 1, 2, 0, 10, 700.0)])
    try:
        matching.match_shots(bp, off)
    except Exception as e:
        assert "not unique" in str(e)
    else:
        raise AssertionError("duplicate official IDs were accepted")
