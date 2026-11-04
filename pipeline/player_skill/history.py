"""Historical shots with out-of-fold shot-difficulty expectations.

Context-adjusted skill needs, for every historical shot, an expected make probability that did
not see that shot. Master V1 fitted the baseline on all of 2022-25 and scored the same shots
(in-sample). Here each season is scored by a baseline fitted on the *other* seasons
(leave-one-season-out), so residuals `made - xb_oof` are honest. The in-sample expectation is kept
as `xb_insample` for the audit comparison.

2026-27 shots get `xb` from the full-history baseline (fitted before the season, so already
out-of-sample) and are appended by `pipeline.player_skill.features` as-of each shot date.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import PROCESSED_DIR, RAW_SHOTS_PATH
from pipeline.features import baseline as bl
from pipeline.features.shot_context import shot_features
from pipeline.player_skill import taxonomy as tx
from pipeline.utils import get_logger, sha256_file, write_json

log = get_logger("player_skill.history")

PLAYER_SKILL_DIR = PROCESSED_DIR / "player_skill"
HISTORY_PATH = PLAYER_SKILL_DIR / "history_shots.parquet"
SEASON_END = {"2021-22": "2022-06-30", "2022-23": "2023-06-30", "2023-24": "2024-06-30", "2024-25": "2025-06-30",
              "2025-26": "2026-06-30"}

_COLS = ["GAME_ID", "GAME_DATE", "PLAYER_ID", "PLAYER_NAME", "PERIOD", "MINUTES_REMAINING", "SECONDS_REMAINING",
         "ACTION_TYPE", "SHOT_TYPE", "SHOT_ZONE_BASIC", "LOC_X", "LOC_Y", "SHOT_MADE_FLAG", "SEASON"]


def build_history(path: Path = HISTORY_PATH, insample_model=None) -> Path:
    """Score every historical shot leave-one-season-out. `insample_model` (optional) is the Master V1
    baseline, used only to record the in-sample expectation for the audit."""
    hist = pd.read_csv(RAW_SHOTS_PATH, usecols=_COLS, dtype={"GAME_ID": str})
    feats = shot_features(hist)
    y = hist["SHOT_MADE_FLAG"].astype(int)
    xb = np.full(len(hist), np.nan)
    seasons = sorted(hist["SEASON"].unique())
    for s in seasons:
        test = (hist["SEASON"] == s).to_numpy()
        log.info("OOF baseline for %s: fit on %d shots, score %d", s, int((~test).sum()), int(test.sum()))
        m = bl.fit_baseline(feats[~test].reset_index(drop=True), y[~test].reset_index(drop=True))
        xb[test] = bl.predict_baseline(m, feats[test])
    out = pd.DataFrame({
        "GAME_ID": hist["GAME_ID"], "game_date": pd.to_datetime(hist["GAME_DATE"].astype(str), format="%Y%m%d"),
        "SEASON": hist["SEASON"], "PLAYER_ID": hist["PLAYER_ID"].astype(int), "PLAYER_NAME": hist["PLAYER_NAME"],
        "made": y, "xb": xb,
        "is_three": feats["is_three"], "zone_group": feats["zone_group"], "shot_distance_ft": feats["shot_distance_ft"],
        "action_family": feats["action_family"],
    })
    if insample_model is not None:
        out["xb_insample"] = bl.predict_baseline(insample_model, feats)
    cats = tx.categories(out)
    for lvl in tx.SHOT_LEVELS:
        out[f"cat_{lvl}"] = cats[lvl]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    ll = lambda p: float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))  # noqa: E731
    meta = {"rows": int(len(out)), "seasons": seasons, "logloss_oof": ll(xb),
            "logloss_insample": ll(out["xb_insample"]) if insample_model is not None else None,
            "source_sha256": sha256_file(RAW_SHOTS_PATH)}
    write_json(path.with_suffix(".meta.json"), meta)
    log.info("wrote %s %s", path, meta)
    return path


def load_history(path: Path = HISTORY_PATH) -> pd.DataFrame:
    if not path.exists():
        build_history(path)
    return pd.read_parquet(path)


__all__ = ["HISTORY_PATH", "PLAYER_SKILL_DIR", "SEASON_END", "build_history", "load_history"]
