"""Run the full xPTS pipeline: build features -> train XGBoost -> predict.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --skip-features   # reuse existing features artifact
    python scripts/run_pipeline.py --skip-train      # reuse existing model
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from build_features import build_features
from common import FEATURES_PATH, check, setup_logging
from predict import predict
from train_model import train_model

log = setup_logging("run_pipeline")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-features", action="store_true", help="reuse data/processed/xpts_features.pkl")
    parser.add_argument("--skip-train", action="store_true", help="score with the original Model A instead of retraining")
    args = parser.parse_args()

    t0 = time.perf_counter()
    if not args.skip_features:
        log.info("=== build_features: start")
        build_features()
    check(FEATURES_PATH.exists(), f"expected features artifact missing: {FEATURES_PATH}")
    if args.skip_train:
        from common import MODEL_PATH, RETRAIN_DIR
        model_path = MODEL_PATH
        preds_path = RETRAIN_DIR / "original_model_a_predictions.csv"
    else:
        log.info("=== train_model: start")
        _, model_path = train_model()
        preds_path = model_path.with_name(model_path.stem + "_predictions.csv")
    log.info("=== predict with %s", model_path)
    pred = predict(model_path=model_path, out_path=preds_path, overwrite=args.skip_train)
    PREDS_PATH = preds_path
    log.info("Pipeline finished in %.1fs", time.perf_counter() - t0)
    log.info("Output %s: %d rows x %d columns", PREDS_PATH, *pred.shape)
    with pd.option_context("display.max_columns", 12, "display.width", 200):
        log.info("Sample:\n%s", pred[[
            "PLAYER_NAME", "SEASON", "SHOT_TYPE", "ACTION_TYPE", "SHOT_DISTANCE", "SHOT_MADE_FLAG",
            "xfg_pct_player_specific", "xpts_player_specific", "points_above_expected",
        ]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
