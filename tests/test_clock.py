import math

import pandas as pd
import pytest

from pipeline.matching import clock


@pytest.mark.parametrize("raw,expected", [
    ("11:38", 698.0), ("0:05", 5.0), ("12:00", 720.0), ("0:00.4", 0.4),
    ("PT11M38.00S", 698.0), ("PT00M00.80S", 0.8), ("PT05M00.00S", 300.0), ("PT45.5S", 45.5),
    (698, 698.0), (12.5, 12.5),
])
def test_parse_clock_formats(raw, expected):
    assert clock.parse_clock(raw) == pytest.approx(expected)


def test_parse_clock_missing_is_nan():
    assert math.isnan(clock.parse_clock(None))
    assert math.isnan(clock.parse_clock(float("nan")))


@pytest.mark.parametrize("bad", ["11:61", "abc", "PT", "11-38"])
def test_parse_clock_rejects_invalid(bad):
    with pytest.raises(ValueError):
        clock.parse_clock(bad)


def test_elapsed_regulation_and_overtime():
    assert clock.elapsed_game_seconds(1, 720) == 0
    assert clock.elapsed_game_seconds(1, 0) == 720
    assert clock.elapsed_game_seconds(4, 0) == 2880
    assert clock.elapsed_game_seconds(5, 300) == 2880
    assert clock.elapsed_game_seconds(6, 0) == 2880 + 600


def test_elapsed_rejects_out_of_range_clock():
    with pytest.raises(ValueError):
        clock.elapsed_game_seconds(5, 400)
    with pytest.raises(ValueError):
        clock.elapsed_game_seconds(0, 10)


def test_scd_minutes_seconds_truncates():
    assert clock.to_scd_minutes_seconds(0.5) == (0, 0)
    assert clock.to_scd_minutes_seconds(59.99) == (0, 59)
    assert clock.to_scd_minutes_seconds(698.0) == (11, 38)
    assert clock.to_scd_minutes_seconds(60.0) == (1, 0)


def test_series_helpers_monotone_within_game():
    period = pd.Series([1, 1, 2, 4, 5])
    rem = clock.parse_clock_series(pd.Series(["12:00", "PT00M01.00S", "11:00", "0:00", "PT04M00.00S"]))
    el = clock.elapsed_series(period, rem)
    assert el.is_monotonic_increasing
    assert el.tolist() == [0, 719, 780, 2880, 2940]
