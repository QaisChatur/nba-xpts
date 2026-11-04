"""Shared paths, feature schema, and logging for the xPTS pipeline scripts."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "historical"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "legacy" / "model_a"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_SHOTS_PATH = RAW_DATA_PATH / "shotchart_shots.csv"
PLAYER_DASH_CLOSEST_PATH = RAW_DATA_PATH / "playerdashptshots_closest_defender.csv"
PLAYER_DASH_CLOSEST_10P_PATH = RAW_DATA_PATH / "playerdashptshots_closest_defender_10ft_plus.csv"
FEATURES_PATH = PROCESSED_DATA_PATH / "xpts_features.pkl"
PREDS_PATH = PROCESSED_DATA_PATH / "xpts_player_specific_predictions.csv"
# The original trained Model A (read-only). Retraining writes a new versioned file instead.
MODEL_PATH = MODELS_DIR / "legacy" / "model_a" / "xpts_player_specific_model.pkl"
METRICS_PATH = MODELS_DIR / "legacy" / "model_a" / "xpts_player_specific_metrics.json"
RETRAIN_DIR = MODELS_DIR / "experiments" / "model_a_retrain"
PROTECTED_PATHS = {MODEL_PATH.resolve(), METRICS_PATH.resolve(), PREDS_PATH.resolve(),
                   (MODELS_DIR / "legacy" / "backend_prep_retrain" / "xpts_player_specific_model.pkl").resolve()}


def default_retrain_paths() -> tuple[Path, Path]:
    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return RETRAIN_DIR / f"model_a_{stamp}.pkl", RETRAIN_DIR / f"model_a_{stamp}_metrics.json"


def guard_output(path: Path, overwrite: bool = False) -> Path:
    """Refuse to write over the frozen Model A artifacts, or over any existing file unless asked."""
    resolved = Path(path).resolve()
    if resolved in PROTECTED_PATHS:
        raise ValueError(f"refusing to overwrite protected Model A artifact {path}")
    if resolved.exists() and not overwrite:
        raise ValueError(f"{path} exists; pass --overwrite to replace it")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved

RANDOM_SEED = 42
TARGET_COL = "SHOT_MADE_FLAG"

CATEGORICAL_FEATURES = ["SHOT_TYPE", "ACTION_TYPE", "SHOT_ZONE_BASIC", "distance_bin"]

NUMERIC_FEATURES = [
    "SHOT_DISTANCE", "LOC_X", "LOC_Y", "shot_angle",
    "seconds_remaining_in_period", "seconds_elapsed_in_game",
    "is_corner_3", "is_restricted_area", "is_midrange", "is_paint_non_ra",
    "is_deep_3", "is_home_team", "is_late_game",
]

PLAYER_PRIOR_NUMERIC_FEATURES = [
    "player_overall_fg_smoothed", "player_2pt_fg_smoothed", "player_3pt_fg_smoothed",
    "player_zone_fg_smoothed", "player_action_fg_smoothed", "player_zone_action_fg_smoothed",
    "player_distance_bin_fg_smoothed", "player_rim_fg_smoothed", "player_midrange_fg_smoothed",
    "player_corner_3_fg_smoothed", "player_above_break_3_fg_smoothed", "player_deep_3_fg_smoothed",
    "player_avg_shot_distance", "player_3pa_rate", "player_rim_frequency", "player_midrange_frequency",
    "player_corner_3_frequency", "player_pullup_frequency",
    "player_all_tight_fg_smoothed", "player_all_open_fg_smoothed", "player_all_wide_open_fg_smoothed",
    "player_10ft_tight_fg_smoothed", "player_10ft_open_fg_smoothed", "player_10ft_wide_open_fg_smoothed",
    "player_all_open_freq", "player_all_wide_open_freq", "player_10ft_open_freq", "player_10ft_wide_open_freq",
]

MODEL_NUMERIC_FEATURES = NUMERIC_FEATURES + PLAYER_PRIOR_NUMERIC_FEATURES
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + PLAYER_PRIOR_NUMERIC_FEATURES

# Bayesian smoothing strengths (pseudo-attempts at league average) per prior family.
K_MAP = {
    "overall": 100,
    "2pt3pt": 150,
    "zone": 100,
    "action": 150,
    "zone_action": 250,
    "defender": 300,
    "rare": 400,
}

XGB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": RANDOM_SEED,
}

PREDICTION_COLS = ["xfg_pct_player_specific", "xpts_player_specific", "points_above_expected"]

EXPORT_COLS = [
    "GAME_ID", "GAME_EVENT_ID", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_NAME", "SEASON",
    "SHOT_TYPE", "ACTION_TYPE", "SHOT_ZONE_BASIC", "SHOT_ZONE_AREA", "SHOT_ZONE_RANGE",
    "SHOT_DISTANCE", "LOC_X", "LOC_Y", "SHOT_MADE_FLAG", "shot_value", "actual_points",
    "xfg_pct_player_specific", "xpts_player_specific", "points_above_expected",
    "player_overall_fg_smoothed", "player_2pt_fg_smoothed", "player_3pt_fg_smoothed",
    "player_zone_fg_smoothed", "player_action_fg_smoothed", "player_zone_action_fg_smoothed",
    "player_distance_bin_fg_smoothed", "player_rim_fg_smoothed", "player_midrange_fg_smoothed",
    "player_corner_3_fg_smoothed", "player_above_break_3_fg_smoothed", "player_deep_3_fg_smoothed",
    "player_avg_shot_distance", "player_3pa_rate", "player_rim_frequency", "player_midrange_frequency",
    "player_corner_3_frequency", "player_pullup_frequency",
    "player_all_very_tight_fg_smoothed", "player_all_tight_fg_smoothed", "player_all_open_fg_smoothed",
    "player_all_wide_open_fg_smoothed", "player_10ft_very_tight_fg_smoothed", "player_10ft_tight_fg_smoothed",
    "player_10ft_open_fg_smoothed", "player_10ft_wide_open_fg_smoothed",
    "player_all_very_tight_freq", "player_all_tight_freq", "player_all_open_freq", "player_all_wide_open_freq",
    "player_10ft_very_tight_freq", "player_10ft_tight_freq", "player_10ft_open_freq", "player_10ft_wide_open_freq",
]


def setup_logging(name: str, verbose: bool = False) -> logging.Logger:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)-14s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    return logging.getLogger(name)


def check(condition: bool, message: str) -> None:
    """Data check that stays active under `python -O` (unlike bare assert)."""
    if not condition:
        raise ValueError(f"Data check failed: {message}")
