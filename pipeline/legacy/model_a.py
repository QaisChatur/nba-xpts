"""Model A: inference with the original player-specific xPTS model, unchanged.

Model A is `models/legacy/model_a/xpts_player_specific_model.pkl`: an sklearn Pipeline
(ColumnTransformer[median-impute + StandardScaler | most-frequent-impute + OneHotEncoder]
 -> XGBClassifier(300 trees, lr 0.05, depth 6)) trained on 2022-23 + 2023-24 ShotChartDetail shots.

Inference inputs are produced by the *same* functions used at training time
(`scripts/build_features.py`: `engineer_features`, `build_smoothed_player_features`,
`build_defender_profile_features`, `merge_player_features`), imported rather than copied so
training and inference cannot drift. Player priors are rebuilt from the training seasons only
and verified against the saved training feature artifact before use.

Model A never receives NBA xFG or Blueprint tracking values.
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import (
    DASH_CLOSEST_10FT_PATH, DASH_CLOSEST_PATH, HISTORICAL_FEATURES_PATH, HISTORICAL_PREDICTIONS_PATH,
    MODEL_A_ASSETS_DIR, MODEL_A_HISTORICAL_METRICS_PATH, MODEL_A_PATH, MODEL_A_SHA256, RAW_SHOTS_PATH,
    MODEL_A_CODE_DIR,
)
from pipeline.utils import DataValidationError, get_logger, rel, require, require_columns, sha256_file, utc_now, write_json

log = get_logger("model_a")

MODEL_A_TRAINED_SKLEARN = "1.4.2"  # reported by sklearn's InconsistentVersionWarning at audit time
# Exact (module, name) allowlist. Module prefixes are deliberately not used: e.g. numpy.testing
# exposes callables that exec strings. A new artifact needing another class fails loudly.
ALLOWED_PICKLE_GLOBALS = {
    "numpy": {"dtype", "ndarray", "float64", "float32", "int64", "int32", "bool_"},
    "numpy._core.multiarray": {"_reconstruct", "scalar"},
    "numpy.core.multiarray": {"_reconstruct", "scalar"},
    "joblib.numpy_pickle": {"NumpyArrayWrapper"},
    "builtins": {"slice"},
    "sklearn.pipeline": {"Pipeline"},
    "sklearn.compose._column_transformer": {"ColumnTransformer"},
    "sklearn.impute._base": {"SimpleImputer"},
    "sklearn.preprocessing._data": {"StandardScaler"},
    "sklearn.preprocessing._encoders": {"OneHotEncoder"},
    "xgboost.core": {"Booster"},
    "xgboost.sklearn": {"XGBClassifier"},
}
# Only for the repository's own historical feature artifact (pandas DataFrames).
PANDAS_FRAME_GLOBALS = {
    "pandas": {"DataFrame", "Index", "RangeIndex", "StringDtype", "CategoricalDtype", "Categorical"},
    "pandas._libs.arrays": {"__pyx_unpickle_NDArrayBacked"},
    "pandas._libs.internals": {"_unpickle_block"},
    "pandas.arrays": {"StringArray"},
    "pandas.core.indexes.base": {"_new_Index"},
    "pandas.core.internals.managers": {"BlockManager"},
}
ASSET_TABLES = ["player_priors", "zone_priors", "action_priors", "zone_action_priors", "distance_bin_priors",
                "defender_all", "defender_10ft"]


def legacy_modules():
    """The training-time modules from scripts/ (single source of truth for feature logic)."""
    if str(MODEL_A_CODE_DIR) not in sys.path:
        sys.path.insert(0, str(MODEL_A_CODE_DIR))
    import build_features  # noqa: E402
    import common  # noqa: E402
    return build_features, common


# --------------------------------------------------------------------------- safe loading


def is_allowed_global(module: str, name: str, extra: dict[str, set[str]] | None = None) -> bool:
    return name in ALLOWED_PICKLE_GLOBALS.get(module, set()) or name in (extra or {}).get(module, set())


def restricted_joblib_load(path: Path, extra: dict[str, set[str]] | None = None):
    """joblib.load with an exact allowlist on every class/function the pickle tries to import.

    Any other global aborts the load before it is resolved, so an artifact cannot reach arbitrary
    callables (os.system, eval, ...) through pickle's global lookup. Combined with the sha256 pin
    in `load_model_a`.
    """
    import joblib
    import joblib.numpy_pickle as jnp

    base = jnp.NumpyUnpickler
    seen: set[str] = set()

    class RestrictedUnpickler(base):
        def find_class(self, module, name):
            seen.add(f"{module}.{name}")
            if not is_allowed_global(module, name, extra):
                raise DataValidationError(f"{rel(path)} references non-allowlisted global {module}.{name}; refusing to unpickle")
            return super().find_class(module, name)

    jnp.NumpyUnpickler = RestrictedUnpickler
    try:
        obj = joblib.load(path)
    finally:
        jnp.NumpyUnpickler = base
    return obj, sorted(seen)


def load_model_a(path: Path = MODEL_A_PATH, expected_sha256_prefix: str | None = MODEL_A_SHA256):
    require(path.exists(), f"Model A artifact not found at {rel(path)}")
    digest = sha256_file(path)
    if expected_sha256_prefix:
        require(digest.startswith(expected_sha256_prefix),
                f"{rel(path)} sha256 {digest[:16]} != pinned {expected_sha256_prefix}; artifact changed since audit")
    import sklearn
    import xgboost
    model, pickle_globals = restricted_joblib_load(path)
    _, common = legacy_modules()
    pre = model.named_steps["preprocessor"]
    require(list(pre.feature_names_in_) == common.FEATURE_COLS, "Model A feature schema differs from scripts/common.py FEATURE_COLS")
    model._xpts_provenance = {
        "path": rel(path), "sha256": digest, "pickle_globals": pickle_globals,
        "trained_with": {"scikit-learn": MODEL_A_TRAINED_SKLEARN},
        "runtime": {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__},
        "compatibility": "cross-version load verified by verify_reproduction() (identical historical predictions)",
    }
    return model


def encoder_categories(model) -> dict[str, list[str]]:
    pre = model.named_steps["preprocessor"]
    cat_cols = pre.transformers_[1][2]
    cats = pre.named_transformers_["cat"].named_steps["onehot"].categories_
    return {c: list(v) for c, v in zip(cat_cols, cats)}


# --------------------------------------------------------------------------- inference assets


def build_inference_assets(out_dir: Path = MODEL_A_ASSETS_DIR) -> dict:
    """Rebuild player priors exactly as at training time (train seasons only) and save them as CSV/JSON."""
    bf, common = legacy_modules()
    for p in (RAW_SHOTS_PATH, DASH_CLOSEST_PATH, DASH_CLOSEST_10FT_PATH):
        require(p.exists(), f"required historical input missing: {rel(p)}")
    shots_raw, dash_all, dash_10 = bf.load_raw_inputs(RAW_SHOTS_PATH, DASH_CLOSEST_PATH, DASH_CLOSEST_10FT_PATH)
    feat = bf.engineer_features(shots_raw)
    train_df, test_df, split_method, test_season = bf.split_train_test(feat)
    require(sorted(train_df["SEASON"].unique()) == ["2022-23", "2023-24"],
            f"unexpected training seasons {sorted(train_df['SEASON'].unique())}")
    priors = bf.build_smoothed_player_features(train_df, common.K_MAP)
    all_def = bf.build_defender_profile_features(dash_all, "player_all", common.K_MAP["defender"], priors["global_fg"])
    ft10_def = bf.build_defender_profile_features(dash_10, "player_10ft", common.K_MAP["defender"], priors["global_fg"])

    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "player_priors": priors["player_priors"], "zone_priors": priors["zone_priors"],
        "action_priors": priors["action_priors"], "zone_action_priors": priors["zone_action_priors"],
        "distance_bin_priors": priors["distance_bin_priors"], "defender_all": all_def, "defender_10ft": ft10_def,
    }
    for name, t in tables.items():
        t.to_csv(out_dir / f"{name}.csv", index=False)
    meta = {
        "built_utc": utc_now(),
        "train_seasons": sorted(train_df["SEASON"].unique()),
        "split_method": split_method,
        "n_train_shots": int(len(train_df)),
        "global_fg": float(priors["global_fg"]),
        "league_defaults": {k: float(v) for k, v in priors["league_defaults"].items()},
        "k_map": common.K_MAP,
        "inputs": {rel(p): sha256_file(p) for p in (RAW_SHOTS_PATH, DASH_CLOSEST_PATH, DASH_CLOSEST_10FT_PATH)},
        "defender_profile_seasons": sorted(dash_all["SEASON"].dropna().unique().tolist()),
        "note": ("Defender-profile priors aggregate every season in the PlayerDashPtShots files (as at training "
                 "time). They include 2024-25, the historical test season, so historical test metrics carry mild "
                 "leakage; no 2026-27 data is included, so 2026-27 evaluation is unaffected."),
        "tables": {name: {"rows": int(len(t)), "file": f"{name}.csv"} for name, t in tables.items()},
    }
    meta["reproduction"] = _verify_assets_against_training_artifact(bf, common, test_df, priors, all_def, ft10_def)
    write_json(out_dir / "assets_meta.json", meta)
    log.info("Model A inference assets written to %s", rel(out_dir))
    return meta


def _verify_assets_against_training_artifact(bf, common, test_df, priors, all_def, ft10_def) -> dict:
    if not HISTORICAL_FEATURES_PATH.exists():
        return {"status": "skipped", "reason": f"{rel(HISTORICAL_FEATURES_PATH)} not present"}
    art, _ = restricted_joblib_load(HISTORICAL_FEATURES_PATH, extra=PANDAS_FRAME_GLOBALS)
    rebuilt = bf.merge_player_features(test_df, priors, all_def, ft10_def)
    saved = art["test"]
    keys = ["GAME_ID", "GAME_EVENT_ID", "PLAYER_ID"]
    require(len(rebuilt) == len(saved), "rebuilt test rows != saved artifact rows")
    a = rebuilt.sort_values(keys).reset_index(drop=True)
    b = saved.sort_values(keys).reset_index(drop=True)
    num = [c for c in common.FEATURE_COLS if c not in common.CATEGORICAL_FEATURES]
    max_abs = float(np.nanmax(np.abs(a[num].to_numpy(float) - b[num].to_numpy(float))))
    cat_equal = bool((a[common.CATEGORICAL_FEATURES].astype(str).to_numpy() == b[common.CATEGORICAL_FEATURES].astype(str).to_numpy()).all())
    res = {"status": "passed" if max_abs < 1e-12 and cat_equal else "failed",
           "compared_rows": int(len(a)), "numeric_max_abs_diff": max_abs, "categoricals_identical": cat_equal}
    require(res["status"] == "passed", f"rebuilt priors do not reproduce the training artifact: {res}")
    return res


@lru_cache(maxsize=2)
def load_inference_assets(assets_dir: Path = MODEL_A_ASSETS_DIR) -> dict:
    meta_path = assets_dir / "assets_meta.json"
    require(meta_path.exists(), f"Model A assets missing at {rel(assets_dir)}; run `python -m pipeline build-model-a-assets`")
    meta = json.loads(meta_path.read_text())
    tables = {name: pd.read_csv(assets_dir / f"{name}.csv") for name in ASSET_TABLES}
    return {
        "meta": meta,
        "priors": {
            "player_priors": tables["player_priors"], "zone_priors": tables["zone_priors"],
            "action_priors": tables["action_priors"], "zone_action_priors": tables["zone_action_priors"],
            "distance_bin_priors": tables["distance_bin_priors"],
            "league_defaults": meta["league_defaults"], "global_fg": meta["global_fg"],
        },
        "defender_all": tables["defender_all"], "defender_10ft": tables["defender_10ft"],
    }


# --------------------------------------------------------------------------- inference


SCD_INPUT_COLUMNS = ["GAME_ID", "GAME_EVENT_ID", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_NAME", "PERIOD",
                     "MINUTES_REMAINING", "SECONDS_REMAINING", "ACTION_TYPE", "SHOT_TYPE", "SHOT_ZONE_BASIC",
                     "SHOT_ZONE_AREA", "SHOT_ZONE_RANGE", "SHOT_DISTANCE", "LOC_X", "LOC_Y", "SHOT_MADE_FLAG",
                     "HTM", "VTM", "SEASON"]


def build_features(shots: pd.DataFrame, assets: dict | None = None) -> pd.DataFrame:
    """ShotChartDetail-schema shots -> Model A feature frame (+ keys and coverage flags)."""
    bf, common = legacy_modules()
    assets = assets or load_inference_assets()
    require_columns(shots, SCD_INPUT_COLUMNS, "Model A input shots")
    unknown_teams = sorted(set(shots["TEAM_NAME"]) - set(bf.TEAM_NAME_TO_ABBR))
    require(not unknown_teams, f"TEAM_NAME values not in training map (is_home_team would be wrong): {unknown_teams}")
    require(not shots.duplicated(["GAME_ID", "GAME_EVENT_ID"]).any(), "duplicate (GAME_ID, GAME_EVENT_ID) in Model A input")

    feat = bf.engineer_features(shots[SCD_INPUT_COLUMNS])
    require(len(feat) == len(shots), "engineer_features dropped rows (non-binary SHOT_MADE_FLAG?)")
    n = len(feat)
    feat = bf.merge_player_features(feat, assets["priors"], assets["defender_all"], assets["defender_10ft"])
    require(len(feat) == n, "prior merge changed row count (join fan-out)")
    missing = feat[common.PLAYER_PRIOR_NUMERIC_FEATURES].isna().sum()
    require(missing.sum() == 0, f"NaN priors after fallback: {missing[missing > 0].to_dict()}")

    pp = assets["priors"]["player_priors"]
    feat["model_a_player_has_priors"] = feat["PLAYER_ID"].isin(pp["PLAYER_ID"])
    feat["model_a_player_has_defender_profile"] = feat["PLAYER_ID"].isin(assets["defender_all"]["PLAYER_ID"])
    return feat


def predict(shots: pd.DataFrame, model=None, assets: dict | None = None) -> pd.DataFrame:
    """Return GAME_ID, GAME_EVENT_ID, model_a_xfg, model_a_xpts_fg and input-coverage flags."""
    _, common = legacy_modules()
    model = model or load_model_a()
    feat = build_features(shots, assets)
    cats = encoder_categories(model)
    prob = model.predict_proba(feat[common.FEATURE_COLS])[:, 1]
    require(np.isfinite(prob).all() and ((prob >= 0) & (prob <= 1)).all(), "Model A produced invalid probabilities")
    out = feat[["GAME_ID", "GAME_EVENT_ID"]].copy()
    out["model_a_xfg"] = prob
    out["model_a_xpts_fg"] = prob * feat["shot_value"].to_numpy()
    out["model_a_player_has_priors"] = feat["model_a_player_has_priors"].to_numpy()
    out["model_a_player_has_defender_profile"] = feat["model_a_player_has_defender_profile"].to_numpy()
    for c in ["ACTION_TYPE", "SHOT_ZONE_BASIC", "distance_bin", "SHOT_TYPE"]:
        out[f"model_a_{c.lower()}_known"] = feat[c].astype(str).isin(cats[c]).to_numpy()
    return out


def verify_reproduction(model=None) -> dict:
    """Score the saved historical test features and compare with the saved predictions and metrics."""
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    _, common = legacy_modules()
    model = model or load_model_a()
    res: dict = {"model": model._xpts_provenance, "checked_utc": utc_now()}
    if not HISTORICAL_FEATURES_PATH.exists():
        res["status"] = "skipped (historical features artifact missing)"
        return res
    art, _ = restricted_joblib_load(HISTORICAL_FEATURES_PATH, extra=PANDAS_FRAME_GLOBALS)
    test = art["test"]
    prob = model.predict_proba(test[common.FEATURE_COLS])[:, 1]
    y = test[common.TARGET_COL].astype(int)
    res["recomputed_metrics"] = {"roc_auc": float(roc_auc_score(y, prob)), "log_loss": float(log_loss(y, prob)),
                                 "brier_score": float(brier_score_loss(y, prob)), "n": int(len(y))}
    if MODEL_A_HISTORICAL_METRICS_PATH.exists():
        rec = json.loads(MODEL_A_HISTORICAL_METRICS_PATH.read_text())["test_metrics"]
        res["recorded_metrics"] = {k: rec[k] for k in ("roc_auc", "log_loss", "brier_score")}
        res["metrics_max_abs_diff"] = max(abs(res["recomputed_metrics"][k] - rec[k]) for k in res["recorded_metrics"])
    if HISTORICAL_PREDICTIONS_PATH.exists():
        saved = pd.read_csv(HISTORICAL_PREDICTIONS_PATH, usecols=["GAME_ID", "GAME_EVENT_ID", "PLAYER_ID", "xfg_pct_player_specific"])
        require(len(saved) == len(test), "saved predictions row count differs from test features")
        same_keys = (saved["GAME_EVENT_ID"].to_numpy() == test["GAME_EVENT_ID"].to_numpy()).all()
        res["prediction_rows_aligned"] = bool(same_keys)
        res["prediction_max_abs_diff"] = float(np.abs(saved["xfg_pct_player_specific"].to_numpy() - prob).max())
    ok = res.get("metrics_max_abs_diff", 0) < 1e-6 and res.get("prediction_max_abs_diff", 0) < 1e-6
    res["status"] = "passed" if ok else "failed"
    return res
