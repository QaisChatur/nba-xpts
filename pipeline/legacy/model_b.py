"""Model B: does shot-level tracking information improve make-probability predictions?

Design
------
* Unit: matched Blueprint/official shots with Model A predictions. Target: official make/miss.
* Candidate models (L2 logistic regression; low capacity is deliberate at this sample size):
    B0  recalibrated Model A        features: logit(model_a_xfg)
    B1  Model A + tracking          features: logit(model_a_xfg) + tracking block
    B2  geometry + tracking, no A   features: official distance/zone geometry + tracking block
  B0 is the control: fitting anything on 2026-27 shots can fix calibration drift, so the
  incremental value of tracking is measured as B1 vs B0, not B1 vs raw Model A.
* Validation: leave-one-game-out outer folds; the L2 strength is chosen by leave-one-game-out
  inner CV on the training games only. Imputers/scalers are fitted inside each training fold.
  Every reported prediction comes from a model that never saw that game.
* Never used as inputs: nba_xfg, made, scores, or anything recorded after the shot.

Tracking-field handling (from the audit in blueprint.audit_fields):
  release_time_s == 0 occurs on 100% of shots within 6 ft and ~2% beyond 10 ft and values are
  capped at 1.0 -> zero is treated as "not measured" (NaN + indicator), 1.0 gets a cap indicator.
  shot_clock_s == 0 is kept with an indicator (may be a legitimate expiry or the clock being off).
  shot_vertical_in == 0 is kept (a set shot can have ~0 vertical) with an indicator.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.config import MODEL_B_DIR, RANDOM_SEED
from pipeline.evaluation.metrics import compare_models, paired_bootstrap, per_shot_log_loss
from pipeline.utils import get_logger, require, require_columns, utc_now, write_json

log = get_logger("model_b")

C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]

TRACKING_FEATURES = [
    "contest_distance_ft", "log_contest_distance_ft", "contest_height_in", "shot_vertical_in",
    "shooter_speed_mph", "release_time_s_measured", "shot_clock_s",
    "release_time_unmeasured", "release_time_at_cap", "shot_clock_zero", "shot_vertical_zero",
]
GEOMETRY_FEATURES = ["SHOT_DISTANCE", "is_three", "is_restricted_area", "is_corner_three", "abs_loc_x"]
FEATURE_SETS = {
    "B0_recalibrated_A": ["logit_model_a"],
    "B1_A_plus_tracking": ["logit_model_a"] + TRACKING_FEATURES,
    "B2_geometry_plus_tracking": GEOMETRY_FEATURES + TRACKING_FEATURES,
}


def engineer_tracking_features(df: pd.DataFrame) -> pd.DataFrame:
    require_columns(df, ["contest_distance_in", "contest_height_in", "shot_vertical_in", "shooter_speed_mph",
                         "release_time_s", "shot_clock_s", "model_a_xfg", "SHOT_DISTANCE", "SHOT_TYPE",
                         "SHOT_ZONE_BASIC", "LOC_X"], "Model B input")
    out = pd.DataFrame(index=df.index)
    p = np.clip(df["model_a_xfg"].astype(float), 1e-6, 1 - 1e-6)
    out["logit_model_a"] = np.log(p / (1 - p))
    out["contest_distance_ft"] = df["contest_distance_in"] / 12.0
    out["log_contest_distance_ft"] = np.log1p(out["contest_distance_ft"])
    out["contest_height_in"] = df["contest_height_in"]
    out["shot_vertical_in"] = df["shot_vertical_in"]
    out["shot_vertical_zero"] = (df["shot_vertical_in"] == 0).astype(int)
    out["shooter_speed_mph"] = df["shooter_speed_mph"]
    out["release_time_unmeasured"] = (df["release_time_s"] == 0).astype(int)
    out["release_time_s_measured"] = df["release_time_s"].where(df["release_time_s"] > 0)
    out["release_time_at_cap"] = (df["release_time_s"] >= 1.0).astype(int)
    out["shot_clock_s"] = df["shot_clock_s"]
    out["shot_clock_zero"] = (df["shot_clock_s"] == 0).astype(int)
    out["SHOT_DISTANCE"] = df["SHOT_DISTANCE"].astype(float)
    out["is_three"] = df["SHOT_TYPE"].astype(str).str.startswith("3").astype(int)
    out["is_restricted_area"] = (df["SHOT_ZONE_BASIC"] == "Restricted Area").astype(int)
    out["is_corner_three"] = df["SHOT_ZONE_BASIC"].isin(["Left Corner 3", "Right Corner 3"]).astype(int)
    out["abs_loc_x"] = df["LOC_X"].abs().astype(float)
    return out


def _pipeline(features: list[str], C: float) -> Pipeline:
    pre = ColumnTransformer([("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                                               ("scale", StandardScaler())]), features)])
    return Pipeline([("pre", pre), ("lr", LogisticRegression(C=C, solver="newton-cholesky", max_iter=5000, random_state=RANDOM_SEED))])


def _select_C(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, features: list[str]) -> tuple[float, dict]:
    scores = {}
    for C in C_GRID:
        losses = []
        for g in np.unique(groups):
            tr, va = groups != g, groups == g
            if len(np.unique(y[tr])) < 2:
                continue
            m = _pipeline(features, C).fit(X.loc[tr, features], y[tr])
            losses.append(per_shot_log_loss(y[va], m.predict_proba(X.loc[va, features])[:, 1]))
        scores[C] = float(np.concatenate(losses).mean())
    best = min(scores, key=scores.get)
    return best, scores


def run_experiment(df: pd.DataFrame, y_col: str = "SHOT_MADE_FLAG", group_col: str = "GAME_ID") -> dict:
    """Nested leave-one-game-out experiment. Returns OOF predictions, metrics and diagnostics."""
    require(df[group_col].nunique() >= 3, "need at least 3 games for nested leave-one-game-out CV")
    X = engineer_tracking_features(df)
    y = df[y_col].astype(int).to_numpy()
    groups = df[group_col].astype(str).to_numpy()
    oof = pd.DataFrame(index=df.index)
    fold_info = []
    for name, feats in FEATURE_SETS.items():
        oof[name] = np.nan
        for g in np.unique(groups):
            tr, te = groups != g, groups == g
            C, inner = _select_C(X[tr], y[tr], groups[tr], feats)
            m = _pipeline(feats, C).fit(X.loc[tr, feats], y[tr])
            oof.loc[te, name] = m.predict_proba(X.loc[te, feats])[:, 1]
            fold_info.append({"model": name, "held_out_game": g, "n_train": int(tr.sum()), "n_test": int(te.sum()),
                              "chosen_C": C, "inner_cv_log_loss": inner})
    require(oof.notna().all().all(), "missing out-of-fold predictions")
    return {"X": X, "oof": oof, "folds": fold_info}


def required_sample_size(sd_diff: float, delta: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Shots needed for a paired test to detect a mean per-shot log-loss improvement `delta`.

    Assumes independent shots; within-game correlation inflates this by a design effect.
    """
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return int(np.ceil((z * sd_diff / delta) ** 2))


def summarize(df: pd.DataFrame, exp: dict, y_col: str = "SHOT_MADE_FLAG") -> dict:
    eval_df = df[[y_col, "GAME_ID", "model_a_xfg", "nba_xfg"]].join(exp["oof"])
    model_cols = {"Model A (raw)": "model_a_xfg", **{k: k for k in FEATURE_SETS}, "NBA xFG": "nba_xfg"}
    metrics = compare_models(eval_df, y_col, model_cols)
    boot = paired_bootstrap(eval_df, y_col, model_cols,
                            pairs=[("B1_A_plus_tracking", "B0_recalibrated_A"), ("B0_recalibrated_A", "Model A (raw)"),
                                   ("B2_geometry_plus_tracking", "B0_recalibrated_A"), ("B1_A_plus_tracking", "NBA xFG")])
    per_game = []
    for g, d in eval_df.groupby("GAME_ID"):
        row = {"GAME_ID": g, "n": len(d)}
        for name, col in model_cols.items():
            row[name] = per_shot_log_loss(d[y_col], d[col]).mean()
        per_game.append(row)
    d_ll = per_shot_log_loss(eval_df[y_col], eval_df["B1_A_plus_tracking"]) - per_shot_log_loss(eval_df[y_col], eval_df["B0_recalibrated_A"])
    sd = float(np.std(d_ll, ddof=1))
    observed = float(-d_ll.mean())
    power = {
        "per_shot_log_loss_diff_sd": sd,
        "observed_improvement_B1_vs_B0": observed,
        "required_shots_80pct_power": {str(dl): required_sample_size(sd, dl) for dl in (0.002, 0.005, 0.01, 0.02)},
    }
    if observed > 0:
        power["required_shots_for_observed_effect"] = required_sample_size(sd, observed)
    return {"metrics": metrics, "bootstrap": boot, "per_game_log_loss": pd.DataFrame(per_game), "power": power}


def fit_and_save_final(df: pd.DataFrame, exp: dict, summary: dict, out_dir: Path = MODEL_B_DIR,
                       tag: str = "exploratory", y_col: str = "SHOT_MADE_FLAG") -> Path:
    """Fit B1 on all available games (for inspection only) and save with full provenance."""
    import joblib
    import sklearn
    X = exp["X"]
    y = df[y_col].astype(int).to_numpy()
    groups = df["GAME_ID"].astype(str).to_numpy()
    feats = FEATURE_SETS["B1_A_plus_tracking"]
    C, inner = _select_C(X, y, groups, feats)
    model = _pipeline(feats, C).fit(X[feats], y)
    out = out_dir / tag
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "model_b1.joblib")
    coefs = dict(zip(feats, model.named_steps["lr"].coef_[0].round(4).tolist()))
    meta = {
        "status": "EXPLORATORY - trained on very few games; not validated for production use",
        "model": "B1_A_plus_tracking (L2 logistic regression on standardized features)",
        "features": feats, "chosen_C": C, "inner_cv_log_loss": inner,
        "standardized_coefficients": coefs, "intercept": float(model.named_steps["lr"].intercept_[0]),
        "training_games": sorted(set(groups)), "training_dates": sorted(df["game_date"].dropna().unique().tolist()),
        "n_training_shots": int(len(y)), "created_utc": utc_now(), "scikit_learn": sklearn.__version__,
        "random_seed": RANDOM_SEED,
        "out_of_fold_metrics": summary["metrics"].reset_index().to_dict(orient="records"),
        "excluded_inputs": ["nba_xfg", "made / SHOT_MADE_FLAG", "scores", "defender identity (no valid historical encoding)"],
    }
    write_json(out / "model_b1_meta.json", meta)
    return out
