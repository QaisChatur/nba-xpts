"""Score shots with the trained xPTS model and export shot-level predictions.

xPTS = P(make) * shot_value, where shot_value is 2 or 3.

Usage:
    python scripts/model_a/predict.py
    python scripts/model_a/predict.py --split test --out /tmp/model_a_predictions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from common import EXPORT_COLS, FEATURE_COLS, FEATURES_PATH, MODEL_PATH, RETRAIN_DIR, check, guard_output, setup_logging
from train_model import load_features

log = setup_logging("predict")


def validate_predictions(pred: pd.DataFrame, n_expected: int) -> None:
    check(len(pred) == n_expected, f"expected {n_expected} prediction rows, got {len(pred)}")
    check(list(pred.columns) == EXPORT_COLS, "prediction columns do not match export schema")
    nan_cols = pred.columns[pred.isna().any()].tolist()
    check(not nan_cols, f"NaN values in exported columns {nan_cols}")
    xfg = pred["xfg_pct_player_specific"]
    check(xfg.notna().all(), "NaN xFG predictions")
    check(((xfg >= 0) & (xfg <= 1)).all(), "xFG predictions outside [0, 1]")
    check(pred["shot_value"].isin([2, 3]).all(), "shot_value must be 2 or 3")
    check((pred["xpts_player_specific"] <= pred["shot_value"]).all(), "xPTS exceeds shot value")
    check(np.allclose(pred["actual_points"] - pred["xpts_player_specific"], pred["points_above_expected"]),
          "points_above_expected != actual_points - xpts")
    dupes = pred.duplicated(subset=["GAME_ID", "GAME_EVENT_ID", "PLAYER_ID", "TEAM_ID"]).sum()
    if dupes:
        log.warning("%d duplicate shot keys in predictions", dupes)


def predict(
    features_path: Path = FEATURES_PATH,
    model_path: Path = MODEL_PATH,
    out_path: Path = RETRAIN_DIR / "predictions.csv",
    split: str = "test",
    overwrite: bool = False,
) -> pd.DataFrame:
    artifact = load_features(features_path)
    check(model_path.exists(), f"model not found at {model_path}; run train_model.py first")
    clf = joblib.load(model_path)
    log.info("Loaded model %s from %s", type(clf.named_steps["model"]).__name__, model_path)

    df = artifact[split].copy()
    log.info("Scoring %s split: %d shots (%s)", split, len(df), artifact["meta"]["split_method"])

    df["xfg_pct_player_specific"] = clf.predict_proba(df[FEATURE_COLS])[:, 1]
    df["xpts_player_specific"] = df["xfg_pct_player_specific"] * df["shot_value"]
    df["points_above_expected"] = df["actual_points"] - df["xpts_player_specific"]

    for c in EXPORT_COLS:
        if c not in df.columns:
            df[c] = np.nan
    pred = df[EXPORT_COLS]
    validate_predictions(pred, len(artifact[split]))

    log.info("Totals: actual_points=%.0f expected_points=%.1f points_above_expected=%.1f",
             pred["actual_points"].sum(), pred["xpts_player_specific"].sum(), pred["points_above_expected"].sum())
    log.info("xFG: mean=%.4f min=%.4f max=%.4f | actual FG=%.4f",
             pred["xfg_pct_player_specific"].mean(), pred["xfg_pct_player_specific"].min(),
             pred["xfg_pct_player_specific"].max(), pred["SHOT_MADE_FLAG"].mean())

    out_path = guard_output(out_path, overwrite)
    pred.to_csv(out_path, index=False)
    log.info("Saved predictions -> %s | shape=%s", out_path, pred.shape)
    return pred


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, default=FEATURES_PATH, help="features artifact from build_features.py")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="fitted pipeline from train_model.py")
    parser.add_argument("--out", type=Path, default=RETRAIN_DIR / "predictions.csv",
                        help="output predictions CSV (the original data/processed/legacy predictions are protected)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    parser.add_argument("--split", choices=["test", "train"], default="test",
                        help="which split to score (default: held-out test season)")
    args = parser.parse_args()
    predict(args.features, args.model, args.out, args.split, args.overwrite)


if __name__ == "__main__":
    main()
