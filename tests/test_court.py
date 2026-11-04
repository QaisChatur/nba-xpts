import numpy as np
import pandas as pd
import pytest

from pipeline import config
from pipeline.matching import court


def test_shot_distance_is_floor_of_feet():
    assert court.shot_distance_scd([0, 30, 39, 40, 236], [0, 40, 0, 0, 0]).tolist() == [0, 5, 3, 4, 23]


@pytest.mark.parametrize("x,y,value,zone", [
    (0, 0, 2, "Restricted Area"),
    (30, 20, 2, "Restricted Area"),
    (60, 100, 2, "In The Paint (Non-RA)"),
    (-79, 138, 2, "In The Paint (Non-RA)"),
    (81, 100, 2, "Mid-Range"),
    (0, 160, 2, "Mid-Range"),
    (-225, 50, 3, "Left Corner 3"),
    (225, 50, 3, "Right Corner 3"),
    (225, 100, 3, "Above the Break 3"),
    (0, 250, 3, "Above the Break 3"),
    (0, 450, 3, "Backcourt"),
])
def test_zone_rules(x, y, value, zone):
    assert court.shot_zone_basic([x], [y], [value])[0] == zone


def test_zone_uses_shot_value_not_distance():
    # A long two (official 2PT at 23 ft) must not be labelled a three.
    assert court.shot_zone_basic([0], [230], [2])[0] == "Mid-Range"


def test_court_feet_frame():
    x, y = court.court_feet([0, -250], [-52.5, 0])
    assert x.tolist() == [0.0, -25.0]
    assert y.tolist() == [0.0, 5.25]


@pytest.mark.slow
def test_zone_rules_against_historical_shotchartdetail():
    if not config.RAW_SHOTS_PATH.exists():
        pytest.skip("historical ShotChartDetail shots not present")
    shots = pd.read_csv(config.RAW_SHOTS_PATH, usecols=["LOC_X", "LOC_Y", "SHOT_TYPE", "SHOT_ZONE_BASIC", "SHOT_DISTANCE"],
                        nrows=200_000)
    res = court.validate_zone_rules(shots)
    assert res["shot_distance_agreement"] == 1.0
    assert res["zone_basic_agreement"] > 0.999
