"""Chronological, game-grouped splits on whole game dates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pipeline.utils import require


@dataclass
class Split:
    fit_dates: list[str]
    calib_dates: list[str]
    test_dates: list[str]
    folds: list[tuple[list[str], list[str]]]  # rolling-origin (train dates, validation dates) inside fit_dates

    def describe(self, df: pd.DataFrame) -> dict:
        out = {}
        for name, dates in (("model_fit", self.fit_dates), ("calibration", self.calib_dates), ("test", self.test_dates)):
            d = df[df["game_date"].isin(dates)]
            out[name] = {"dates": [min(dates), max(dates)] if dates else None, "games": int(d["GAME_ID"].nunique()),
                         "shots": int(len(d))}
        out["rolling_folds"] = [{"train": [f[0][0], f[0][-1]], "validate": [f[1][0], f[1][-1]],
                                 "train_shots": int(df["game_date"].isin(f[0]).sum()),
                                 "validate_shots": int(df["game_date"].isin(f[1]).sum())} for f in self.folds]
        return out


def _cut_by_games(dates: list[str], games_per_date: pd.Series, fraction: float) -> int:
    """Index i such that dates[i:] hold >= fraction of the games (at least one date)."""
    total = games_per_date.loc[dates].sum()
    acc = 0
    for i in range(len(dates) - 1, -1, -1):
        acc += games_per_date.loc[dates[i]]
        if acc >= fraction * total:
            return i
    return 0


def chronological_split(df: pd.DataFrame, test_fraction: float, calib_fraction: float, n_folds: int,
                        min_train_games: int) -> Split:
    gpd = df.groupby("game_date")["GAME_ID"].nunique().sort_index()
    dates = list(gpd.index)
    require(len(dates) >= 6, f"need at least 6 game dates for a chronological split, have {len(dates)}")
    t = _cut_by_games(dates, gpd, test_fraction)
    test = dates[t:]
    dev = dates[:t]
    c = _cut_by_games(dev, gpd, calib_fraction / (1 - test_fraction))
    calib, fit = dev[c:], dev[:c]
    require(len(fit) >= 3 and len(calib) >= 1, "not enough dates for model-fit / calibration partitions")

    cum = gpd.loc[fit].cumsum()
    start = int(np.searchsorted(cum.to_numpy(), min(min_train_games, cum.iloc[-1] * 0.5)))
    start = max(start, 1)
    rest = fit[start:]
    folds = []
    if rest:
        blocks = np.array_split(np.arange(len(rest)), min(n_folds, len(rest)))
        for b in blocks:
            if len(b) == 0:
                continue
            val = [rest[i] for i in b]
            train = [d for d in fit if d < val[0]]
            folds.append((train, val))
    require(folds, "no rolling-origin folds could be formed")
    return Split(fit, calib, test, folds)


def game_halves(df: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Two disjoint game-grouped halves (boolean masks) for cross-fitted calibration selection."""
    games = np.array(sorted(df["GAME_ID"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(games)
    a = set(games[: len(games) // 2])
    m = df["GAME_ID"].isin(a).to_numpy()
    return m, ~m
