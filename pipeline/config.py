"""Project paths and constants. All paths are relative to the repository root.

Tunable experiment settings (feature groups, search spaces, split sizes) live in `configs/*.yaml`;
this module only holds locations and invariants.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = Path(os.environ.get("XPTS_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = DATA_DIR / "reports"
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"  # committed fixtures, always in the repo
MODELS_DIR = PROJECT_ROOT / "models"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
MODEL_A_CODE_DIR = SCRIPTS_DIR / "model_a"  # original Model A training code (feature logic source of truth)

SEASON = "2026-27"

# Raw inputs (immutable once written; ingestion only appends new files).
BLUEPRINT_DIR = RAW_DIR / "blueprint" / SEASON
BLUEPRINT_GAMES_DIR = BLUEPRINT_DIR / "games"
BLUEPRINT_RAW_DIR = BLUEPRINT_DIR / "raw"
BLUEPRINT_MANIFEST = BLUEPRINT_DIR / "legacy_scraper" / "manifest.json"   # legacy scraper manifest (Oct 28 sample)
SEASON_MANIFEST = BLUEPRINT_DIR / "season_manifest.json"       # pipeline.ingestion.season
GAMES_LISTING_DIR = RAW_DIR / "nba_listings" / SEASON          # NBA.com /games?date=D listings (game discovery)
GAME_PAGES_DIR = RAW_DIR / "nba_pages" / SEASON                # NBA.com game pages: box score + play-by-play
HISTORICAL_DIR = RAW_DIR / "historical"
RAW_SHOTS_PATH = HISTORICAL_DIR / "shotchart_shots.csv"
DASH_CLOSEST_PATH = HISTORICAL_DIR / "playerdashptshots_closest_defender.csv"
DASH_CLOSEST_10FT_PATH = HISTORICAL_DIR / "playerdashptshots_closest_defender_10ft_plus.csv"

# Outputs of the original notebook/scripts (kept for reproduction checks).
LEGACY_PROCESSED_DIR = PROCESSED_DIR / "legacy"
HISTORICAL_FEATURES_PATH = LEGACY_PROCESSED_DIR / "model_a" / "xpts_features.pkl"
HISTORICAL_PREDICTIONS_PATH = LEGACY_PROCESSED_DIR / "model_a" / "xpts_player_specific_predictions.csv"

# Model A = the original player-specific XGBoost pipeline (frozen; inference only).
MODEL_A_DIR = MODELS_DIR / "legacy" / "model_a"
MODEL_A_PATH = MODEL_A_DIR / "xpts_player_specific_model.pkl"
MODEL_A_HISTORICAL_METRICS_PATH = MODEL_A_DIR / "xpts_player_specific_metrics.json"
MODEL_A_SHA256 = "28f33625439cc174"  # prefix of sha256 of MODEL_A_PATH observed at audit time
MODEL_A_ASSETS_DIR = MODEL_A_DIR / "assets"
MODEL_B_DIR = MODELS_DIR / "legacy" / "model_b"
BACKEND_PREP_MODEL_PATH = MODELS_DIR / "legacy" / "backend_prep_retrain" / "xpts_player_specific_model.pkl"
PROTECTED_ARTIFACTS = (MODEL_A_PATH, BACKEND_PREP_MODEL_PATH)

# Master xFG model.
MASTER_MODELS_DIR = MODELS_DIR / "master"
EXPERIMENTS_DIR = MODELS_DIR / "experiments"

UNIFIED_DIR = PROCESSED_DIR / "unified"
MASTER_DATASET_DIR = PROCESSED_DIR / "master"
ANALYTICS_DIR = PROCESSED_DIR / "analytics"

RANDOM_SEED = 42

# Matching tolerances (see docs/methodology/DATA_MATCHING.md).
MATCH_CLOCK_TOL_S = 2.0
MATCH_COORD_TOL_FT = 3.0
