"""Court coordinates, shot distance, and shot-zone classification.

Coordinate system ("legacy" NBA shot-chart frame, used by ShotChartDetail LOC_X/LOC_Y,
play-by-play xLegacy/yLegacy, and Blueprint x/y):
  * units are tenths of a foot,
  * the hoop centre is the origin,
  * +y points from the baseline towards half court (baseline is at y = -52.5),
  * the sign of x follows NBA's convention (LOC_X < 0 <=> SHOT_ZONE_AREA "Left Side").

Zone rules were derived from and validated against 540,627 historical ShotChartDetail
shots (2022-23 .. 2024-25); see `validate_zone_rules` and docs/DATA_MATCHING.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RA_RADIUS = 40.0          # restricted-area radius (tenths of a foot); historical RA max r = 39.96
PAINT_HALF_WIDTH = 80.0   # paint |x| bound
PAINT_MAX_Y = 138.0       # paint far edge (free-throw line)
CORNER_MAX_Y = 87.0       # corner-3 zone upper y bound
CORNER_LEFT_MAX_X = -220.0
CORNER_RIGHT_MIN_X = 221.0
BACKCOURT_MIN_Y = 420.0
BASELINE_Y = -52.5

ZONES_BASIC = [
    "Restricted Area", "In The Paint (Non-RA)", "Mid-Range",
    "Left Corner 3", "Right Corner 3", "Above the Break 3", "Backcourt",
]


def shot_distance_scd(loc_x, loc_y):
    """ShotChartDetail SHOT_DISTANCE: floor of Euclidean distance in feet (100% agreement historically)."""
    return np.floor(np.hypot(np.asarray(loc_x, float), np.asarray(loc_y, float)) / 10.0 + 1e-9).astype(int)


def shot_zone_basic(loc_x, loc_y, shot_value) -> np.ndarray:
    x = np.asarray(loc_x, float)
    y = np.asarray(loc_y, float)
    v = np.asarray(shot_value, int)
    r = np.hypot(x, y)
    three = v == 3
    out = np.where(
        three,
        np.where(
            y >= BACKCOURT_MIN_Y, "Backcourt",
            np.where((y <= CORNER_MAX_Y) & (x <= CORNER_LEFT_MAX_X), "Left Corner 3",
                     np.where((y <= CORNER_MAX_Y) & (x >= CORNER_RIGHT_MIN_X), "Right Corner 3", "Above the Break 3")),
        ),
        np.where(
            r < RA_RADIUS, "Restricted Area",
            np.where((np.abs(x) <= PAINT_HALF_WIDTH) & (y <= PAINT_MAX_Y), "In The Paint (Non-RA)", "Mid-Range"),
        ),
    )
    return out.astype(object)


def shot_zone_range(shot_distance_ft, loc_y=None) -> np.ndarray:
    d = np.asarray(shot_distance_ft, float)
    out = np.select(
        [d < 8, d < 16, d < 24],
        ["Less Than 8 ft.", "8-16 ft.", "16-24 ft."],
        default="24+ ft.",
    ).astype(object)
    if loc_y is not None:
        out = np.where(np.asarray(loc_y, float) >= BACKCOURT_MIN_Y, "Back Court Shot", out).astype(object)
    return out


def court_feet(loc_x, loc_y) -> tuple[np.ndarray, np.ndarray]:
    """Frontend frame: feet, x in [-25, 25] across the court, y measured from the baseline."""
    x = np.asarray(loc_x, float) / 10.0
    y = (np.asarray(loc_y, float) - BASELINE_Y) / 10.0
    return x, y


def validate_zone_rules(shots: pd.DataFrame) -> dict:
    """Agreement of the geometric rules with ShotChartDetail labels on a historical frame."""
    value = np.where(shots["SHOT_TYPE"].astype(str).str.contains("3PT"), 3, 2)
    pred_zone = shot_zone_basic(shots["LOC_X"], shots["LOC_Y"], value)
    pred_dist = shot_distance_scd(shots["LOC_X"], shots["LOC_Y"])
    zone_ok = pred_zone == shots["SHOT_ZONE_BASIC"].to_numpy()
    res = {
        "n": int(len(shots)),
        "zone_basic_agreement": float(zone_ok.mean()),
        "shot_distance_agreement": float((pred_dist == shots["SHOT_DISTANCE"].to_numpy()).mean()),
        "zone_confusions": (
            pd.DataFrame({"truth": shots["SHOT_ZONE_BASIC"].to_numpy()[~zone_ok], "pred": pred_zone[~zone_ok]})
            .value_counts().head(10).rename("count").reset_index().to_dict(orient="records")
        ),
    }
    return res
