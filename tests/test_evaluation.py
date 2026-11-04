import numpy as np
import pandas as pd
import pytest

from pipeline import evaluation


def test_log_loss_per_shot_matches_definition():
    ll = evaluation.per_shot_log_loss([1, 0], [0.8, 0.8])
    assert ll == pytest.approx([-np.log(0.8), -np.log(0.2)])


def test_perfectly_calibrated_has_small_ece_and_unit_slope():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 50_000)
    y = (rng.random(p.size) < p).astype(int)
    assert evaluation.expected_calibration_error(y, p) < 0.01
    slope, intercept = evaluation.calibration_slope_intercept(y, p)
    assert slope == pytest.approx(1.0, abs=0.05)
    assert intercept == pytest.approx(0.0, abs=0.05)


def test_overconfident_model_has_slope_below_one():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.2, 0.8, 20_000)
    y = (rng.random(p.size) < p).astype(int)
    logit = np.log(p / (1 - p)) * 2.5
    slope, _ = evaluation.calibration_slope_intercept(y, 1 / (1 + np.exp(-logit)))
    assert slope == pytest.approx(0.4, abs=0.05)


def test_two_valued_predictions_bin_by_value():
    p = np.array([0.35] * 50 + [0.55] * 50)
    y = np.array([1] * 20 + [0] * 30 + [1] * 30 + [0] * 20)
    cb = evaluation.calibration_bins(y, p)
    assert len(cb) == 2
    assert cb["observed"].tolist() == [0.4, 0.6]
    assert evaluation.expected_calibration_error(y, p) == pytest.approx(0.05)


def test_compare_models_uses_identical_rows():
    df = pd.DataFrame({"y": [1, 0, 1, 0], "a": [0.9, 0.2, 0.6, np.nan], "b": [0.5, 0.5, 0.5, 0.5]})
    m = evaluation.compare_models(df, "y", {"A": "a", "B": "b"})
    assert (m["n"] == 3).all()


def test_paired_bootstrap_is_deterministic_and_signed():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.1, 0.9, 400)
    y = (rng.random(400) < p).astype(int)
    df = pd.DataFrame({"y": y, "good": p, "flat": 0.5})
    b1 = evaluation.paired_bootstrap(df, "y", {"good": "good", "flat": "flat"}, [("good", "flat")], n_boot=200)
    b2 = evaluation.paired_bootstrap(df, "y", {"good": "good", "flat": "flat"}, [("good", "flat")], n_boot=200)
    pd.testing.assert_frame_equal(b1, b2)
    d = b1[(b1["quantity"] == "good - flat") & (b1["metric"] == "log_loss")].iloc[0]
    assert d["mean"] < 0 and d["ci_high"] < 0
