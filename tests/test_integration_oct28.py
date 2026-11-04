"""Integration tests on the local 2026-10-28 samples (skipped when the data is absent)."""

import json

import numpy as np
import pytest

from pipeline import config
from pipeline.utils import sha256_file

from .conftest import BLUEPRINT_GAMES, TARGET_DATE

BOX_FGA = {"0022600132": 192, "0022600133": 174, "0022600134": 147, "0022600136": 168}


def test_blueprint_loader_real_sample(oct28_context):
    bp = oct28_context["bp"]
    assert len(bp) == 650
    assert bp["shot_uid"].is_unique
    assert set(bp["game_id"]) == set(BLUEPRINT_GAMES)
    assert bp["nba_xfg"].between(0, 1).all()
    assert (bp["shooter_team_id"] != bp["defender_team_id"]).all()
    assert bp["game_date"].eq(TARGET_DATE).all()


def test_official_pbp_matches_box_score(oct28_context):
    off = oct28_context["official"]
    assert off.groupby("GAME_ID").size().to_dict() == BOX_FGA
    assert oct28_context["box"]["pbp_matches_box"].all()


def test_matching_on_real_games(oct28_context):
    st = oct28_context["match_result"].bp_status
    counts = st["match_status"].value_counts().to_dict()
    assert counts == {"matched": 649, "rejected_id_conflict": 1}
    assert (st.loc[st["match_status"] == "matched", "clock_diff_s"] <= config.MATCH_CLOCK_TOL_S).all()
    rej = st[st["match_status"] == "rejected_id_conflict"].iloc[0]
    assert rej["shot_uid"] == "0022600133_0203" and rej["id_pair_rejection"] == "result"
    u = oct28_context["unified"]
    assert int(u["in_official"].sum()) == sum(BOX_FGA.values())
    assert u["shot_uid"].is_unique


def test_schedule_flags_wrong_date_game(oct28_context):
    sch = oct28_context["schedule"].set_index("game_id")
    assert sch.loc["0022600158", "status"] == "wrong_date"
    assert (sch.loc[BLUEPRINT_GAMES, "status"] == "verified").all()


def test_model_a_assets_use_training_seasons_only(have_model_a):
    meta = json.loads((config.MODEL_A_ASSETS_DIR / "assets_meta.json").read_text())
    assert meta["train_seasons"] == ["2022-23", "2023-24"]
    assert meta["reproduction"]["status"] == "passed"
    for name, info in meta["tables"].items():
        assert (config.MODEL_A_ASSETS_DIR / info["file"]).exists(), name


def test_model_a_scoring_on_real_games(oct28_context, have_model_a):
    from pipeline.legacy import model_a
    original_sha = sha256_file(config.MODEL_A_PATH)
    off = oct28_context["official"]
    pred = model_a.predict(off)
    assert len(pred) == len(off)
    assert pred["model_a_xfg"].between(0, 1).all()
    assert pred["model_a_action_type_known"].all()
    np.testing.assert_allclose(pred["model_a_xpts_fg"],
                               pred["model_a_xfg"] * np.where(off["SHOT_TYPE"].str.startswith("3"), 3, 2), rtol=1e-6)
    assert sha256_file(config.MODEL_A_PATH) == original_sha


@pytest.mark.slow
def test_model_a_reproduces_historical_predictions(have_model_a):
    if not config.HISTORICAL_FEATURES_PATH.exists():
        pytest.skip("historical features artifact not present")
    from pipeline.legacy import model_a
    res = model_a.verify_reproduction()
    assert res["status"] == "passed"
    assert res["prediction_max_abs_diff"] < 1e-6
