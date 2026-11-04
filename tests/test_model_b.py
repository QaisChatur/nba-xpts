import numpy as np
import pandas as pd
import pytest

from pipeline.legacy import model_b


def frame(n=240, games=4, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "GAME_ID": np.repeat([f"G{i}" for i in range(games)], n // games),
        "contest_distance_in": rng.uniform(5, 120, n), "contest_height_in": rng.uniform(30, 120, n),
        "shot_vertical_in": rng.choice([0.0, 10.0, 20.0], n), "shooter_speed_mph": rng.uniform(0, 15, n),
        "release_time_s": rng.choice([0.0, 0.5, 1.0], n), "shot_clock_s": rng.uniform(0, 24, n),
        "model_a_xfg": rng.uniform(0.2, 0.8, n), "SHOT_DISTANCE": rng.integers(0, 28, n),
        "SHOT_TYPE": "2PT Field Goal", "SHOT_ZONE_BASIC": "Mid-Range", "LOC_X": rng.integers(-200, 200, n),
        "nba_xfg": rng.uniform(0, 1, n),
    })
    df["SHOT_MADE_FLAG"] = (rng.random(n) < df["model_a_xfg"]).astype(int)
    return df


def test_feature_sets_exclude_nba_xfg_and_outcomes():
    banned = {"nba_xfg", "nba_xpts_fg", "SHOT_MADE_FLAG", "made", "home_score_after", "away_score_after"}
    for feats in model_b.FEATURE_SETS.values():
        assert not banned & set(feats)


def test_release_time_zero_is_treated_as_unmeasured():
    X = model_b.engineer_tracking_features(frame())
    zero = X["release_time_unmeasured"] == 1
    assert X.loc[zero, "release_time_s_measured"].isna().all()
    assert X.loc[~zero, "release_time_s_measured"].notna().all()


def test_engineered_features_do_not_depend_on_nba_xfg():
    df = frame()
    a = model_b.engineer_tracking_features(df)
    b = model_b.engineer_tracking_features(df.assign(nba_xfg=1 - df["nba_xfg"]))
    pd.testing.assert_frame_equal(a, b)


def test_experiment_predictions_are_out_of_fold_and_deterministic():
    df = frame()
    e1 = model_b.run_experiment(df)
    e2 = model_b.run_experiment(df)
    pd.testing.assert_frame_equal(e1["oof"], e2["oof"])
    held = {f["held_out_game"] for f in e1["folds"]}
    assert held == set(df["GAME_ID"])
    assert all(f["n_train"] + f["n_test"] == len(df) for f in e1["folds"])


def test_required_sample_size():
    assert model_b.required_sample_size(0.26, 0.01) == pytest.approx(5306, rel=0.01)
    assert model_b.required_sample_size(0.26, 0.005) > model_b.required_sample_size(0.26, 0.01)
