"""Master xFG: leakage guards, splits, calibration, priors, inference validation, reproducibility."""

import numpy as np
import pandas as pd
import pytest

from pipeline.features import player_priors as pp
from pipeline.features.dataset import feature_list, in_season_priors, load_feature_config
from pipeline.features.shot_context import action_family, normalize_action, shot_features
from pipeline.training.calibration import Calibrator, select_calibration
from pipeline.training.splits import chronological_split, game_halves
from pipeline.utils import DataValidationError


# --------------------------------------------------------------------------- leakage


def test_no_forbidden_feature_configured():
    cfg = load_feature_config()
    feats = feature_list(cfg)
    assert not set(feats) & set(cfg["forbidden"])
    for bad in ("made", "SHOT_MADE_FLAG", "nba_xfg", "model_a_xfg", "featured", "home_score_after", "pbp_scoreHome"):
        assert bad in cfg["forbidden"] and bad not in feats


def test_feature_list_rejects_forbidden_columns():
    cfg = load_feature_config()
    cfg["groups"] = {**cfg["groups"], "bad": ["nba_xfg"]}
    with pytest.raises(Exception, match="forbidden"):
        feature_list(cfg)


def test_bank_token_removed_from_actions():
    s = pd.Series(["Step Back Bank Jump Shot", "Step Back Jump shot", "Jump Bank Shot", "Driving Floating Bank Jump Shot"])
    n = normalize_action(s)
    assert n[0] == n[1] == "step back jump" and n[2] == "jump" and n[3] == "driving floating jump"
    assert list(action_family(n)) == ["stepback", "stepback", "jump", "floater"]


def _toy_shots():
    rows = []
    for day, made in (("2026-10-20", 0), ("2026-10-21", 1), ("2026-10-22", 1)):
        for k in range(10):
            rows.append({"PLAYER_ID": 1, "game_date": day, "made": made, "resid_baseline": made - 0.4, "is_three": 0,
                         "zone_group": "mid", "action_family": "jump", "in_blueprint": True, "defender_id": 2,
                         "contest_distance_in": 40.0})
    return pd.DataFrame(rows)


def _toy_builder():
    hist = pd.DataFrame(columns=["att", "made", "resid"], index=pd.MultiIndex.from_arrays([[], [], []],
                                                                                         names=["PLAYER_ID", "kind", "cat"]))
    league = {("all", "all"): 0.45, ("type", "two"): 0.5, ("type", "three"): 0.35, ("zone", "mid"): 0.4,
              ("family", "jump"): 0.38}
    dash_range = pd.DataFrame({"FG2M": [10], "FG2A": [20], "FG3M": [5], "FG3A": [15]},
                              index=pd.MultiIndex.from_tuples([(99, "open")], names=["PLAYER_ID", "range"]))
    params = dict(load_feature_config()["priors"], league_three_rate=0.4, league_rim_rate=0.3)
    return pp.PriorBuilder(hist, league, pd.DataFrame(columns=["FG2M", "FG2A", "FG3M", "FG3A"]), dash_range,
                           pd.Series(dtype=float), params), params


def test_in_season_priors_use_only_earlier_dates():
    u = _toy_shots()
    builder, params = _toy_builder()
    pf = in_season_priors(u, builder, load_feature_config())
    by_day = pf.groupby(u["game_date"])
    first = by_day.get_group("2026-10-20")
    assert (first["pl_season_att"] == 0).all()                    # nothing before opening day
    assert np.allclose(first["pl_fg_zone_post"], 0.4)              # pure league prior
    second = by_day.get_group("2026-10-21")
    k = params["k_fg_zone"]
    assert np.allclose(second["pl_fg_zone_post"], (0 + k * 0.4) / (10 + k))   # only day-1 misses, not same-day makes
    third = by_day.get_group("2026-10-22")
    assert np.allclose(third["pl_fg_zone_post"], (10 + k * 0.4) / (20 + k))
    assert (third["def_contests_season"] == 20).all() and (first["def_contests_season"] == 0).all()


def test_shot_features_have_no_outcome_inputs():
    df = pd.DataFrame({"LOC_X": [0, 230], "LOC_Y": [10, 50], "SHOT_TYPE": ["2PT Field Goal", "3PT Field Goal"],
                       "SHOT_ZONE_BASIC": ["Restricted Area", "Right Corner 3"], "ACTION_TYPE": ["Dunk Shot", "Jump Shot"],
                       "PERIOD": [1, 4], "MINUTES_REMAINING": [5, 0], "SECONDS_REMAINING": [3, 1]})
    a = shot_features(df)
    b = shot_features(df.assign(SHOT_MADE_FLAG=[1, 0], EVENT_TYPE=["Made Shot", "Missed Shot"]))
    pd.testing.assert_frame_equal(a, b)
    assert a.loc[1, "is_corner3"] == 1 and a.loc[1, "end_of_period_heave_window"] == 1


# --------------------------------------------------------------------------- splits


def _toy_games(n_dates=20, games_per_date=3):
    rows = []
    for d in range(n_dates):
        for g in range(games_per_date):
            for s in range(5):
                rows.append({"game_date": f"2026-10-{d + 1:02d}", "GAME_ID": f"g{d:02d}{g}", "made": (s + d) % 2})
    return pd.DataFrame(rows)


def test_chronological_split_is_ordered_and_game_disjoint():
    df = _toy_games()
    sp = chronological_split(df, 0.2, 0.15, 4, 10)
    assert max(sp.fit_dates) < min(sp.calib_dates) < min(sp.test_dates)
    parts = [set(df.loc[df["game_date"].isin(x), "GAME_ID"]) for x in (sp.fit_dates, sp.calib_dates, sp.test_dates)]
    assert not (parts[0] & parts[1]) and not (parts[1] & parts[2]) and not (parts[0] & parts[2])
    assert len(sp.test_dates) >= 4  # >= 20% of 20 dates with equal games per date
    for tr, va in sp.folds:
        assert max(tr) < min(va) and set(va) <= set(sp.fit_dates)


def test_game_halves_disjoint():
    df = _toy_games()
    a, b = game_halves(df, 0)
    assert not (set(df.loc[a, "GAME_ID"]) & set(df.loc[b, "GAME_ID"])) and (a | b).all()


# --------------------------------------------------------------------------- calibration


def test_platt_recovers_and_roundtrips():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 20000)
    true = 1 / (1 + np.exp(-(0.3 + 0.6 * np.log(p / (1 - p)))))
    y = rng.binomial(1, true)
    c = Calibrator("platt").fit(p, y)
    assert abs(c.params["b"] - 0.6) < 0.05 and abs(c.params["a"] - 0.3) < 0.05
    c2 = Calibrator.from_dict(c.to_dict())
    assert np.allclose(c.transform(p[:50]), c2.transform(p[:50]))


def test_isotonic_monotone_and_selection_prefers_none_when_calibrated():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, 6000)
    y = rng.binomial(1, p)
    iso = Calibrator("isotonic").fit(p, y)
    q = np.linspace(0.01, 0.99, 200)
    assert np.all(np.diff(iso.transform(q)) >= -1e-12)
    halves = (np.arange(6000) % 2 == 0, np.arange(6000) % 2 == 1)
    best, scores = select_calibration(p, y, halves)
    assert scores["none"] <= scores["isotonic"] and best in ("none", "platt")


# --------------------------------------------------------------------------- inference input validation


def _valid_inputs():
    return pd.DataFrame({
        "GAME_ID": ["0022600001"], "GAME_EVENT_ID": [7], "PLAYER_ID": [1], "game_date": ["2026-10-20"], "LOC_X": [0],
        "LOC_Y": [10], "SHOT_TYPE": ["2PT Field Goal"], "ACTION_TYPE": ["Layup Shot"], "PERIOD": [1],
        "clock_remaining_s": [700.0], "shot_clock_s": [20.0], "contest_distance_in": [30.0],
    })


def test_validate_inputs_accepts_valid_and_rejects_invalid():
    from pipeline.inference import validate_inputs
    req = list(_valid_inputs().columns)
    validate_inputs(_valid_inputs(), req)
    with pytest.raises(DataValidationError, match="missing required"):
        validate_inputs(_valid_inputs().drop(columns=["LOC_X"]), req)
    with pytest.raises(DataValidationError, match="SHOT_TYPE"):
        validate_inputs(_valid_inputs().assign(SHOT_TYPE="FT"), req)
    with pytest.raises(DataValidationError, match="shot_clock_s"):
        validate_inputs(_valid_inputs().assign(shot_clock_s=31.0), req)
    validate_inputs(_valid_inputs().assign(contest_height_in=-1.1), req + ["contest_height_in"])
    with pytest.raises(DataValidationError, match="contest_height_in"):
        validate_inputs(_valid_inputs().assign(contest_height_in=-40.0), req + ["contest_height_in"])
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_inputs(pd.concat([_valid_inputs()] * 2), req)
    with pytest.raises(DataValidationError, match="game_date"):
        validate_inputs(_valid_inputs().assign(game_date="20/10/2026"), req)


# --------------------------------------------------------------------------- reproducibility


@pytest.mark.parametrize("family,params", [
    ("xgboost", {"max_depth": 3, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8}),
    ("lightgbm", {"num_leaves": 8, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8}),
    ("logistic", {"C": 1.0, "n_knots": 4}),
])
def test_training_is_deterministic(family, params):
    from pipeline.training.models import make_model
    rng = np.random.default_rng(3)
    X = pd.DataFrame({"a": rng.normal(size=800), "b": rng.normal(size=800), "zone_code": rng.integers(0, 5, 800)})
    y = (X["a"] + rng.normal(size=800) > 0).astype(int)
    preds = []
    for _ in range(2):
        m = make_model(family, ["a", "b", "zone_code"], params, seed=42, n_estimators=50)
        preds.append(m.fit(X.iloc[:600], y.iloc[:600]).predict(X.iloc[600:]))
    assert np.array_equal(preds[0], preds[1])


# --------------------------------------------------------------------------- serving / V1 preservation


def test_active_model_is_master_v1_and_tracked_metadata_exists():
    from pipeline.inference import active_model_dir
    d = active_model_dir()
    assert d.name == "master_master_v1_20261104T210006"
    assert (d / "model_spec.json").exists() and (d / "metrics.json").exists()


@pytest.mark.slow
def test_master_v1_benchmark_reproduces_exactly():
    from pipeline import config
    from pipeline.inference import active_model_dir
    d = active_model_dir()
    if not (d / "model.cbm").exists() or not (config.MASTER_DATASET_DIR / "master_v1" / "master_shots.parquet").exists():
        pytest.skip("Master V1 binary or dataset not present (outside Git)")
    from pipeline.evaluation.verify import verify_master
    out = verify_master(d)
    assert out["status"] == "passed" and out["prediction_max_abs_diff"] == 0.0
