"""Master xFG inference: `predict_shots(shots_df)`.

The artifact directory (models/master/<version>/) holds everything inference needs:
  model file (native XGBoost/LightGBM/CatBoost format, or sha256-pinned joblib for logistic),
  model_spec.json (family, features, params), calibrator.json, metadata.json,
  history_assets/ (pre-season player tables, historical shot-difficulty baseline),
  in_season_state.csv (2026-27 player/defender aggregates through `in_season_state_through`).

Shots dated after the state date get priors from that state. To score shots *inside* the
season window without look-ahead, pass `state=build_prior_state(dataset, before_date=<game date>)`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pipeline.config import CONFIGS_DIR, MASTER_MODELS_DIR
from pipeline.features import player_priors as pp
from pipeline.features.dataset import FAMILIES, POSS_TYPES, load_history_assets, tracking_features
from pipeline.features.shot_context import ZONE_GROUP_ORDER, shot_features
from pipeline.features.baseline import predict_baseline
from pipeline.matching import court
from pipeline.training.calibration import Calibrator
from pipeline.training.models import LoadedModel
from pipeline.utils import DataValidationError, require

BASE_INPUTS = ["GAME_ID", "GAME_EVENT_ID", "PLAYER_ID", "game_date", "LOC_X", "LOC_Y", "SHOT_TYPE", "ACTION_TYPE",
               "PERIOD", "clock_remaining_s"]
GROUP_INPUTS = {
    "shot_context": [],
    "game_context": ["is_home", "score_margin_pre", "poss_start_type", "seconds_since_poss_start", "second_chance",
                     "after_timeout", "shooter_fga_before_game"],
    "tracking": ["shot_clock_s", "contest_distance_in", "contest_height_in", "shot_vertical_in", "shooter_speed_mph",
                 "release_time_s", "bp_shot_distance_ft"],
    "player": [],
    "player_tracking": ["contest_distance_in"],
    "defender": ["defender_id"],
}
RANGES = {"LOC_X": (-260, 260), "LOC_Y": (-60, 900), "PERIOD": (1, 10), "clock_remaining_s": (0, 720),
          "shot_clock_s": (0, 24), "contest_distance_in": (0, 1200), "contest_height_in": (-12, 200),  # published values dip slightly below 0
          "shot_vertical_in": (0, 60), "shooter_speed_mph": (0, 30), "release_time_s": (0, 1.0),
          "bp_shot_distance_ft": (0, 95)}


# --------------------------------------------------------------------------- in-season state


def build_prior_state(dataset: pd.DataFrame, before_date: str | None) -> pd.DataFrame:
    """Long table of in-season player and defender aggregates from shots dated before `before_date`
    (all dataset shots if None). Requires dataset columns produced by build_master_dataset."""
    d = dataset if before_date is None else dataset[dataset["game_date"] < before_date]
    base = pd.DataFrame({"w": 1.0, "made": d["made"].astype(float), "resid": d["resid_baseline"].astype(float),
                         "is_three": d["is_three"], "zone_group": d["zone_group"], "action_family": d["action_family"]})
    players = pp.aggregate(base.assign(PLAYER_ID=d["PLAYER_ID"].astype(int))).reset_index().assign(role="player")
    tracked = d["in_blueprint"].astype(bool) & d["defender_id"].notna()
    db = base[tracked.to_numpy()].assign(PLAYER_ID=pd.to_numeric(d.loc[tracked, "defender_id"]).astype(int))
    defenders = (pp.aggregate(db).reset_index().assign(role="defender") if len(db)
                 else pd.DataFrame(columns=players.columns))
    out = pd.concat([players, defenders], ignore_index=True)
    out["through_date"] = str(d["game_date"].max()) if len(d) else None
    return out


def _state_tables(state: pd.DataFrame):
    if state is None or len(state) == 0:
        return None, None
    idx = ["PLAYER_ID", "kind", "cat"]
    s = state.astype({"PLAYER_ID": int})
    pl = s[s["role"] == "player"].set_index(idx)[["att", "made", "resid"]]
    de = s[s["role"] == "defender"].set_index(idx)[["att", "made", "resid"]]
    return pl, de


# --------------------------------------------------------------------------- artifact


class MasterModel:
    def __init__(self, model_dir: Path | None = None):
        model_dir = Path(model_dir) if model_dir else active_model_dir()
        self.dir = model_dir
        self.spec = json.loads((model_dir / "model_spec.json").read_text())
        self.meta = json.loads((model_dir / "metadata.json").read_text())
        self.model = LoadedModel(model_dir, self.spec)
        self.calibrator = Calibrator.from_dict(json.loads((model_dir / "calibrator.json").read_text()))
        cfg = yaml.safe_load((CONFIGS_DIR / "features.yaml").read_text())
        cfg["priors"] = self.meta.get("priors", cfg["priors"])
        root = model_dir if (model_dir / "history_assets").exists() else model_dir.parent  # fallback sub-model
        self.baseline, self.priors, _ = load_history_assets(root / "history_assets", cfg)
        state_path = root / "in_season_state.csv"
        self.state = pd.read_csv(state_path) if state_path.exists() else None
        self.groups = self.meta.get("feature_groups") or list(GROUP_INPUTS)

    @property
    def required_inputs(self) -> list[str]:
        cols = list(BASE_INPUTS)
        for g in self.groups:
            cols += [c for c in GROUP_INPUTS[g] if c not in cols]
        return cols


def active_model_dir(root: Path = MASTER_MODELS_DIR) -> Path:
    """The promoted model named in configs/serving.yaml."""
    cfg = yaml.safe_load((CONFIGS_DIR / "serving.yaml").read_text())
    d = root / cfg["active_master_model"]
    require((d / "model_spec.json").exists(), f"active model {d} (configs/serving.yaml) not found")
    return d



def validate_inputs(shots: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in shots.columns]
    if missing:
        raise DataValidationError(f"predict_shots: missing required columns {missing}")
    if shots.duplicated(["GAME_ID", "GAME_EVENT_ID"]).any():
        raise DataValidationError("predict_shots: duplicate (GAME_ID, GAME_EVENT_ID)")
    bad_type = ~shots["SHOT_TYPE"].astype(str).isin(["2PT Field Goal", "3PT Field Goal"])
    if bad_type.any():
        raise DataValidationError(f"predict_shots: SHOT_TYPE must be '2PT Field Goal'/'3PT Field Goal' ({int(bad_type.sum())} rows)")
    for c, (lo, hi) in RANGES.items():
        if c in required:
            v = pd.to_numeric(shots[c], errors="coerce")
            if c in ("LOC_X", "LOC_Y", "PERIOD", "clock_remaining_s") and v.isna().any():
                raise DataValidationError(f"predict_shots: {c} has missing/non-numeric values")
            out = v.notna() & ((v < lo) | (v > hi))
            if out.any():
                raise DataValidationError(f"predict_shots: {c} outside [{lo}, {hi}] for {int(out.sum())} rows")
    try:
        pd.to_datetime(shots["game_date"], format="%Y-%m-%d")
    except (ValueError, TypeError) as exc:
        raise DataValidationError(f"predict_shots: game_date must be YYYY-MM-DD ({exc})") from exc


def featurize(shots: pd.DataFrame, mm: MasterModel, state: pd.DataFrame | None = None) -> pd.DataFrame:
    s = shots.reset_index(drop=True).copy()
    value = np.where(s["SHOT_TYPE"].astype(str).str.startswith("3"), 3, 2)
    if "SHOT_ZONE_BASIC" not in s:
        s["SHOT_ZONE_BASIC"] = court.shot_zone_basic(s["LOC_X"], s["LOC_Y"], value)
    rem = pd.to_numeric(s["clock_remaining_s"]).astype(float)
    s["MINUTES_REMAINING"] = (rem // 60).astype(int)
    s["SECONDS_REMAINING"] = np.floor(rem % 60).astype(int)
    sf = shot_features(s)
    s = s.drop(columns=[c for c in sf.columns if c in s.columns]).join(sf)
    s["zone_code"] = s["zone_group"].map({z: i for i, z in enumerate(ZONE_GROUP_ORDER)}).astype(int)
    s["family_code"] = s["action_family"].map({f: i for i, f in enumerate(FAMILIES)}).fillna(-1).astype(int)
    if "poss_start_type" in s:
        s["poss_code"] = s["poss_start_type"].map({p: i for i, p in enumerate(POSS_TYPES)}).fillna(len(POSS_TYPES) - 1).astype(int)
    if "shot_clock_s" in s:
        tf = tracking_features(s)
        s = s.drop(columns=[c for c in tf.columns if c in s.columns]).join(tf)
    s["baseline_hist_xfg"] = predict_baseline(mm.baseline, sf)
    pl, de = _state_tables(mm.state if state is None else state)
    pf = mm.priors.features(s, pl, de)
    s = s.drop(columns=[c for c in pf.columns if c in s.columns]).join(pf)
    for c in mm.spec["features"]:
        s[c] = pd.to_numeric(s[c], errors="coerce").astype(float)
    s["shot_value"] = value
    return s


def predict_shots(shots_df: pd.DataFrame, model_dir: Path | None = None, state: pd.DataFrame | None = None,
                  model: MasterModel | None = None) -> pd.DataFrame:
    """Calibrated make probability and expected field-goal points for each shot.

    Returns GAME_ID, GAME_EVENT_ID, PLAYER_ID, master_xfg, master_xpts_fg, shot_value, model_version,
    feature_schema_version, calibration_version. Raises DataValidationError on invalid inputs.
    """
    mm = model or MasterModel(model_dir)
    validate_inputs(shots_df, mm.required_inputs)
    X = featurize(shots_df, mm, state)
    raw = mm.model.predict(X)
    p = mm.calibrator.transform(raw)
    require(np.isfinite(p).all() and ((p > 0) & (p < 1)).all(), "Master model produced invalid probabilities")
    out = X[["GAME_ID", "GAME_EVENT_ID", "PLAYER_ID"]].copy()
    out["master_xfg"] = p
    out["master_xpts_fg"] = p * X["shot_value"].to_numpy()
    out["shot_value"] = X["shot_value"].to_numpy()
    out["model_version"] = mm.spec["model_version"]
    out["feature_schema_version"] = mm.spec["feature_schema_version"]
    out["calibration_version"] = mm.spec["calibration_version"]
    return out
