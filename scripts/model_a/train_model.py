"""Train the player-specific xPTS make-probability model (XGBoost only).

Fits sklearn Pipeline(preprocessor -> XGBClassifier) on the train split of the
features artifact, evaluates on the held-out test season, and saves the fitted
pipeline with joblib plus a metrics JSON.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --features data/processed/xpts_features.pkl --model-out models/xpts_player_specific_model.pkl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from common import (
    CATEGORICAL_FEATURES,
    FEATURE_COLS,
    FEATURES_PATH,
    MODEL_NUMERIC_FEATURES,
    TARGET_COL,
    XGB_PARAMS,
    check,
    default_retrain_paths,
    guard_output,
    setup_logging,
)

log = setup_logging("train_model")


def build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, MODEL_NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", XGBClassifier(**XGB_PARAMS))])


def evaluate_prob_model(y_true: pd.Series, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
    }


def load_features(features_path: Path) -> dict:
    check(features_path.exists(), f"features artifact not found at {features_path}; run build_features.py first")
    artifact = joblib.load(features_path)
    check({"train", "test", "meta"} <= set(artifact), "features artifact missing train/test/meta keys")
    for split in ["train", "test"]:
        missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in artifact[split].columns]
        check(not missing, f"{split} features missing columns {missing}")
    return artifact


def train_model(
    features_path: Path = FEATURES_PATH,
    model_out: Path | None = None,
    metrics_out: Path | None = None,
    overwrite: bool = False,
) -> tuple[Pipeline, Path]:
    """Fit and save a *new* Model A. The original artifact in models/legacy/model_a is never overwritten."""
    default_model, default_metrics = default_retrain_paths()
    model_out = guard_output(model_out or default_model, overwrite)
    metrics_out = guard_output(metrics_out or default_metrics, overwrite)
    artifact = load_features(features_path)
    train_df, test_df, meta = artifact["train"], artifact["test"], artifact["meta"]

    X_train = train_df[FEATURE_COLS].copy()
    y_train = train_df[TARGET_COL].astype(int).copy()
    X_test = test_df[FEATURE_COLS].copy()
    y_test = test_df[TARGET_COL].astype(int).copy()
    check(y_train.nunique() == 2, "training target has a single class")

    log.info("Split: %s", meta["split_method"])
    log.info("Train X=%s make_rate=%.4f | Test X=%s make_rate=%.4f",
             X_train.shape, y_train.mean(), X_test.shape, y_test.mean())
    log.info("Fitting XGBClassifier %s", XGB_PARAMS)

    clf = build_pipeline()
    clf.fit(X_train, y_train)

    test_prob = clf.predict_proba(X_test)[:, 1]
    check(np.isfinite(test_prob).all(), "model produced non-finite probabilities")
    metrics = evaluate_prob_model(y_test, test_prob)
    metrics["train_log_loss"] = float(log_loss(y_train, clf.predict_proba(X_train)[:, 1]))
    log.info("Test metrics: %s", " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    log.info("Mean predicted xFG=%.4f vs actual FG=%.4f", test_prob.mean(), y_test.mean())

    joblib.dump(clf, model_out)
    log.info("Saved model -> %s", model_out)

    metrics_payload = {
        "model": "XGBClassifier",
        "params": XGB_PARAMS,
        "split_method": meta["split_method"],
        "train_seasons": meta["train_seasons"],
        "test_season": meta["test_season"],
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_metrics": metrics,
    }
    metrics_out.write_text(json.dumps(metrics_payload, indent=2))
    log.info("Saved metrics -> %s", metrics_out)
    return clf, model_out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, default=FEATURES_PATH, help="features artifact from build_features.py")
    parser.add_argument("--model-out", type=Path, default=None,
                        help="where to save the fitted pipeline (default: models/experiments/model_a_retrain/<timestamp>.pkl)")
    parser.add_argument("--metrics-out", type=Path, default=None, help="where to save evaluation metrics (.json)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing (non-protected) output file")
    args = parser.parse_args()
    train_model(args.features, args.model_out, args.metrics_out, args.overwrite)


if __name__ == "__main__":
    main()
