"""Game-clock parsing and conversion.

Sources use different clock encodings:
  * Blueprint:            "11:38"            (whole seconds remaining in the period)
  * NBA play-by-play (v3): "PT11M38.00S"     (ISO-8601 duration, centisecond precision)
  * ShotChartDetail:       MINUTES_REMAINING + SECONDS_REMAINING integers

ShotChartDetail integers are treated as the truncation (floor) of the precise clock;
e.g. PT00M00.50S -> 0 min 0 s. That is the convention used to build Model A inputs.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

REGULATION_PERIOD_S = 720
OVERTIME_PERIOD_S = 300

_ISO = re.compile(r"^PT(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?$")
_MMSS = re.compile(r"^(\d{1,2}):(\d{1,2}(?:\.\d+)?)$")


def parse_clock(value) -> float:
    """Seconds remaining in the period from 'MM:SS', 'M:SS.s', 'PTmmMss.ssS', or a number of seconds."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    s = str(value).strip()
    m = _ISO.match(s)
    if m and (m.group(1) or m.group(2)):
        return int(m.group(1) or 0) * 60 + float(m.group(2) or 0)
    m = _MMSS.match(s)
    if m:
        secs = float(m.group(2))
        if secs >= 60:
            raise ValueError(f"invalid clock {value!r}: seconds >= 60")
        return int(m.group(1)) * 60 + secs
    raise ValueError(f"unrecognised clock format {value!r}")


def period_length(period: int) -> int:
    if period < 1:
        raise ValueError(f"invalid period {period}")
    return REGULATION_PERIOD_S if period <= 4 else OVERTIME_PERIOD_S


def elapsed_game_seconds(period: int, remaining_s: float) -> float:
    """Seconds elapsed since tip-off at the given period/clock."""
    length = period_length(period)
    if remaining_s < 0 or remaining_s > length:
        raise ValueError(f"clock {remaining_s}s outside period {period} length {length}s")
    if period <= 4:
        start = (period - 1) * REGULATION_PERIOD_S
    else:
        start = 4 * REGULATION_PERIOD_S + (period - 5) * OVERTIME_PERIOD_S
    return start + (length - remaining_s)


def to_scd_minutes_seconds(remaining_s: float) -> tuple[int, int]:
    """ShotChartDetail-style (MINUTES_REMAINING, SECONDS_REMAINING), truncating fractional seconds."""
    whole = int(math.floor(remaining_s + 1e-9))
    return whole // 60, whole % 60


def parse_clock_series(values: pd.Series) -> pd.Series:
    return values.map(parse_clock).astype(float)


def elapsed_series(period: pd.Series, remaining_s: pd.Series) -> pd.Series:
    return pd.Series(
        [elapsed_game_seconds(int(p), float(r)) for p, r in zip(period, remaining_s)],
        index=period.index, dtype=float,
    )
