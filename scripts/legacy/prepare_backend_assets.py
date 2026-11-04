#!/usr/bin/env python3
"""
Prepare all backend assets needed before frontend implementation.

Outputs (data/processed/legacy/backend_prep/):
- xpts_player_specific_predictions_all_games.csv, game_flow_all_games.csv
- player_zone_shottype_summary.csv, player_action_summary.csv, clip_index_all_games.csv
- db_stage/games.csv (the other DB tables are loaded from the files above)
- models/experiments/backend_prep/xpts_player_specific_model.pkl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise ImportError(
        "This script requires xgboost. Install it with `pip install xgboost` and rerun."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "historical"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "legacy" / "backend_prep"
DB_STAGE_DIR = PROCESSED_DIR / "db_stage"
# Writes go to an experiments folder; the original Model A in models/legacy/model_a is never overwritten.
MODELS_DIR = PROJECT_ROOT / "models" / "experiments" / "backend_prep"

RAW_SHOTS_PATH = RAW_DIR / "shotchart_shots.csv"
RAW_CLOSEST_PATH = RAW_DIR / "playerdashptshots_closest_defender.csv"
RAW_CLOSEST_10P_PATH = RAW_DIR / "playerdashptshots_closest_defender_10ft_plus.csv"

PREDICTIONS_PATH = PROCESSED_DIR / "xpts_player_specific_predictions_all_games.csv"
GAME_FLOW_PATH = PROCESSED_DIR / "game_flow_all_games.csv"
ZONE_SUMMARY_PATH = PROCESSED_DIR / "player_zone_shottype_summary.csv"
ACTION_SUMMARY_PATH = PROCESSED_DIR / "player_action_summary.csv"
CLIP_INDEX_PATH = PROCESSED_DIR / "clip_index_all_games.csv"
MODEL_PATH = MODELS_DIR / "xpts_player_specific_model.pkl"
MODEL_META_PATH = MODELS_DIR / "xpts_player_specific_model_meta.json"


def ensure_dirs() -> None:
    for path in [PROCESSED_DIR, DB_STAGE_DIR, MODELS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_raw_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not RAW_SHOTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {RAW_SHOTS_PATH}. Run notebook/API collection first."
        )

    shots_raw = pd.read_csv(RAW_SHOTS_PATH)
    closest = pd.read_csv(RAW_CLOSEST_PATH) if RAW_CLOSEST_PATH.exists() else pd.DataFrame()
    closest_10p = (
        pd.read_csv(RAW_CLOSEST_10P_PATH) if RAW_CLOSEST_10P_PATH.exists() else pd.DataFrame()
    )
    return shots_raw, closest, closest_10p


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in [
        "SHOT_DISTANCE",
        "LOC_X",
        "LOC_Y",
        "PERIOD",
        "MINUTES_REMAINING",
        "SECONDS_REMAINING",
        "SHOT_MADE_FLAG",
        "PLAYER_ID",
        "TEAM_ID",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out["seconds_remaining_in_period"] = (
        out["MINUTES_REMAINING"].fillna(0) * 60 + out["SECONDS_REMAINING"].fillna(0)
    )
    out["seconds_elapsed_in_period"] = 720 - out["seconds_remaining_in_period"]
    out["seconds_elapsed_in_game"] = np.where(
        out["PERIOD"] <= 4,
        (out["PERIOD"] - 1) * 720 + out["seconds_elapsed_in_period"],
        4 * 720 + (out["PERIOD"] - 5) * 300 + np.minimum(out["seconds_elapsed_in_period"], 300),
    )

    out["is_home_team"] = np.nan
    if all(c in out.columns for c in ["TEAM_NAME", "HTM", "VTM"]):
        team_name_to_abbr = {
            "Atlanta Hawks": "ATL",
            "Boston Celtics": "BOS",
            "Brooklyn Nets": "BKN",
            "Charlotte Hornets": "CHA",
            "Chicago Bulls": "CHI",
            "Cleveland Cavaliers": "CLE",
            "Dallas Mavericks": "DAL",
            "Denver Nuggets": "DEN",
            "Detroit Pistons": "DET",
            "Golden State Warriors": "GSW",
            "Houston Rockets": "HOU",
            "Indiana Pacers": "IND",
            "LA Clippers": "LAC",
            "Los Angeles Lakers": "LAL",
            "Memphis Grizzlies": "MEM",
            "Miami Heat": "MIA",
            "Milwaukee Bucks": "MIL",
            "Minnesota Timberwolves": "MIN",
            "New Orleans Pelicans": "NOP",
            "New York Knicks": "NYK",
            "Oklahoma City Thunder": "OKC",
            "Orlando Magic": "ORL",
            "Philadelphia 76ers": "PHI",
            "Phoenix Suns": "PHX",
            "Portland Trail Blazers": "POR",
            "Sacramento Kings": "SAC",
            "San Antonio Spurs": "SAS",
            "Toronto Raptors": "TOR",
            "Utah Jazz": "UTA",
            "Washington Wizards": "WAS",
        }
        team_abbr = out["TEAM_NAME"].map(team_name_to_abbr)
        out.loc[team_abbr == out["HTM"], "is_home_team"] = 1
        out.loc[team_abbr == out["VTM"], "is_home_team"] = 0
    out["is_home_team"] = out["is_home_team"].fillna(0)

    out["is_late_game"] = (
        ((out["PERIOD"] == 4) & (out["seconds_remaining_in_period"] <= 120))
        | (out["PERIOD"] >= 5)
    ).astype(int)

    out["shot_value"] = np.where(
        out["SHOT_TYPE"].astype(str).str.contains("3PT", case=False, na=False), 3, 2
    )
    out["actual_points"] = out["SHOT_MADE_FLAG"].fillna(0) * out["shot_value"]

    out["abs_loc_x"] = out["LOC_X"].abs()
    out["shot_angle"] = np.degrees(np.arctan2(out["LOC_Y"], out["LOC_X"]))

    shot_zone_basic = out["SHOT_ZONE_BASIC"].astype(str)
    shot_type = out["SHOT_TYPE"].astype(str)
    action_type = out["ACTION_TYPE"].astype(str)

    out["is_corner_3"] = (
        shot_type.str.contains("3PT", case=False, na=False)
        & out["abs_loc_x"].ge(220)
        & out["LOC_Y"].le(100)
    ).astype(int)
    out["is_above_break_3"] = (
        shot_type.str.contains("3PT", case=False, na=False) & (out["is_corner_3"] == 0)
    ).astype(int)
    out["is_restricted_area"] = shot_zone_basic.eq("Restricted Area").astype(int)
    out["is_midrange"] = shot_zone_basic.eq("Mid-Range").astype(int)
    out["is_paint_non_ra"] = shot_zone_basic.eq("In The Paint (Non-RA)").astype(int)

    out["is_left_side"] = out["LOC_X"].lt(0).astype(int)
    out["is_right_side"] = out["LOC_X"].gt(0).astype(int)
    out["is_center"] = out["LOC_X"].eq(0).astype(int)

    out["is_deep_3"] = (
        (out["SHOT_DISTANCE"] >= 28) & shot_type.str.contains("3PT", case=False, na=False)
    ).astype(int)

    out["distance_bin"] = pd.cut(
        out["SHOT_DISTANCE"],
        bins=[-np.inf, 3, 7, 15, 23, 27, 31, np.inf],
        labels=["0-3", "4-7", "8-15", "16-23", "24-27", "28-31", "32+"],
    ).astype(str)

    out["is_pullup"] = action_type.str.contains("pull", case=False, na=False).astype(int)
    out["is_catch_shoot"] = action_type.str.contains("catch|jump shot", case=False, na=False).astype(
        int
    )

    required_cols = [
        "GAME_ID",
        "GAME_EVENT_ID",
        "GAME_DATE",
        "HTM",
        "VTM",
        "PLAYER_ID",
        "PLAYER_NAME",
        "TEAM_ID",
        "TEAM_NAME",
        "SEASON",
        "SHOT_MADE_FLAG",
        "SHOT_TYPE",
        "ACTION_TYPE",
        "SHOT_ZONE_BASIC",
        "SHOT_ZONE_AREA",
        "SHOT_ZONE_RANGE",
        "SHOT_DISTANCE",
        "LOC_X",
        "LOC_Y",
        "PERIOD",
        "MINUTES_REMAINING",
        "SECONDS_REMAINING",
        "seconds_remaining_in_period",
        "seconds_elapsed_in_game",
        "is_home_team",
        "is_late_game",
        "shot_value",
        "actual_points",
        "abs_loc_x",
        "shot_angle",
        "is_corner_3",
        "is_above_break_3",
        "is_restricted_area",
        "is_midrange",
        "is_paint_non_ra",
        "is_left_side",
        "is_right_side",
        "is_center",
        "distance_bin",
        "is_deep_3",
        "is_pullup",
        "is_catch_shoot",
    ]

    for c in required_cols:
        if c not in out.columns:
            out[c] = np.nan

    out = out[required_cols].copy()
    out = out[out["SHOT_MADE_FLAG"].isin([0, 1])].copy()
    return out


def _agg_rate(
    source: pd.DataFrame,
    group_cols: List[str],
    output_col: str,
    k: float,
    global_avg: float,
) -> pd.DataFrame:
    tmp = source.copy()
    tmp["_made"] = tmp["SHOT_MADE_FLAG"].astype(float)
    tmp["_att"] = 1.0
    grouped = tmp.groupby(group_cols, dropna=False)[["_made", "_att"]].sum().reset_index()
    grouped[output_col] = (grouped["_made"] + k * global_avg) / (grouped["_att"] + k)
    return grouped[group_cols + [output_col, "_att"]]


def build_smoothed_player_features(train_df: pd.DataFrame, k_map: Dict[str, int]) -> Dict[str, pd.DataFrame]:
    global_fg = float(train_df["SHOT_MADE_FLAG"].mean())
    is_2pt = train_df["shot_value"] == 2
    is_3pt = train_df["shot_value"] == 3

    player_base = (
        train_df.groupby(["PLAYER_ID"], dropna=False)
        .agg(
            PLAYER_NAME=("PLAYER_NAME", "first"),
            player_shots=("SHOT_MADE_FLAG", "size"),
        )
        .reset_index()
    )

    overall = _agg_rate(train_df, ["PLAYER_ID"], "player_overall_fg_smoothed", k_map["overall"], global_fg)
    p2 = _agg_rate(train_df[is_2pt], ["PLAYER_ID"], "player_2pt_fg_smoothed", k_map["2pt3pt"], global_fg)
    p3 = _agg_rate(train_df[is_3pt], ["PLAYER_ID"], "player_3pt_fg_smoothed", k_map["2pt3pt"], global_fg)
    p_zone = _agg_rate(train_df, ["PLAYER_ID", "SHOT_ZONE_BASIC"], "player_zone_fg_smoothed", k_map["zone"], global_fg)
    p_action = _agg_rate(train_df, ["PLAYER_ID", "ACTION_TYPE"], "player_action_fg_smoothed", k_map["action"], global_fg)
    p_zone_action = _agg_rate(
        train_df,
        ["PLAYER_ID", "SHOT_ZONE_BASIC", "ACTION_TYPE"],
        "player_zone_action_fg_smoothed",
        k_map["zone_action"],
        global_fg,
    )
    p_dist = _agg_rate(
        train_df, ["PLAYER_ID", "distance_bin"], "player_distance_bin_fg_smoothed", k_map["zone_action"], global_fg
    )

    rim = _agg_rate(train_df[train_df["is_restricted_area"] == 1], ["PLAYER_ID"], "player_rim_fg_smoothed", k_map["zone"], global_fg)
    mid = _agg_rate(train_df[train_df["is_midrange"] == 1], ["PLAYER_ID"], "player_midrange_fg_smoothed", k_map["zone"], global_fg)
    c3 = _agg_rate(train_df[train_df["is_corner_3"] == 1], ["PLAYER_ID"], "player_corner_3_fg_smoothed", k_map["zone_action"], global_fg)
    ab3 = _agg_rate(train_df[train_df["is_above_break_3"] == 1], ["PLAYER_ID"], "player_above_break_3_fg_smoothed", k_map["zone_action"], global_fg)
    d3 = _agg_rate(train_df[train_df["is_deep_3"] == 1], ["PLAYER_ID"], "player_deep_3_fg_smoothed", k_map["rare"], global_fg)

    tendencies = (
        train_df.groupby("PLAYER_ID", dropna=False)
        .agg(
            player_avg_shot_distance=("SHOT_DISTANCE", "mean"),
            player_3pa_rate=("shot_value", lambda s: np.mean(s == 3)),
            player_rim_frequency=("is_restricted_area", "mean"),
            player_midrange_frequency=("is_midrange", "mean"),
            player_corner_3_frequency=("is_corner_3", "mean"),
            player_pullup_frequency=("is_pullup", "mean"),
        )
        .reset_index()
    )

    priors = player_base[["PLAYER_ID", "PLAYER_NAME"]].drop_duplicates()
    for tbl in [overall, p2, p3, rim, mid, c3, ab3, d3, tendencies]:
        priors = priors.merge(tbl.drop(columns=[c for c in ["_att"] if c in tbl.columns]), on="PLAYER_ID", how="left")

    league_defaults = {
        "player_overall_fg_smoothed": global_fg,
        "player_2pt_fg_smoothed": float(train_df.loc[is_2pt, "SHOT_MADE_FLAG"].mean()) if is_2pt.any() else global_fg,
        "player_3pt_fg_smoothed": float(train_df.loc[is_3pt, "SHOT_MADE_FLAG"].mean()) if is_3pt.any() else global_fg,
        "player_rim_fg_smoothed": float(train_df.loc[train_df["is_restricted_area"] == 1, "SHOT_MADE_FLAG"].mean())
        if (train_df["is_restricted_area"] == 1).any()
        else global_fg,
        "player_midrange_fg_smoothed": float(train_df.loc[train_df["is_midrange"] == 1, "SHOT_MADE_FLAG"].mean())
        if (train_df["is_midrange"] == 1).any()
        else global_fg,
        "player_corner_3_fg_smoothed": float(train_df.loc[train_df["is_corner_3"] == 1, "SHOT_MADE_FLAG"].mean())
        if (train_df["is_corner_3"] == 1).any()
        else global_fg,
        "player_above_break_3_fg_smoothed": float(train_df.loc[train_df["is_above_break_3"] == 1, "SHOT_MADE_FLAG"].mean())
        if (train_df["is_above_break_3"] == 1).any()
        else global_fg,
        "player_deep_3_fg_smoothed": float(train_df.loc[train_df["is_deep_3"] == 1, "SHOT_MADE_FLAG"].mean())
        if (train_df["is_deep_3"] == 1).any()
        else global_fg,
        "player_avg_shot_distance": float(train_df["SHOT_DISTANCE"].mean()),
        "player_3pa_rate": float(np.mean(train_df["shot_value"] == 3)),
        "player_rim_frequency": float(train_df["is_restricted_area"].mean()),
        "player_midrange_frequency": float(train_df["is_midrange"].mean()),
        "player_corner_3_frequency": float(train_df["is_corner_3"].mean()),
        "player_pullup_frequency": float(train_df["is_pullup"].mean()),
    }

    return {
        "player_priors": priors,
        "zone_priors": p_zone,
        "action_priors": p_action,
        "zone_action_priors": p_zone_action,
        "distance_bin_priors": p_dist,
        "league_defaults": league_defaults,
    }


def _normalize_def_range(val: str) -> str:
    s = str(val).lower()
    if "very tight" in s:
        return "very_tight"
    if "tight" in s:
        return "tight"
    if "wide" in s:
        return "wide_open"
    if "open" in s:
        return "open"
    return "other"


def build_defender_profile_features(df_raw: pd.DataFrame, prefix: str, k: int, global_fg: float) -> pd.DataFrame:
    if df_raw is None or df_raw.empty:
        return pd.DataFrame(columns=["PLAYER_ID"])

    cols_needed = ["PLAYER_ID", "CLOSE_DEF_DIST_RANGE", "FGM", "FGA"]
    if not all(c in df_raw.columns for c in cols_needed):
        return pd.DataFrame(columns=["PLAYER_ID"])

    df = df_raw.copy()
    for c in ["FGM", "FGA"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["def_bucket"] = df["CLOSE_DEF_DIST_RANGE"].map(_normalize_def_range)

    totals = df.groupby("PLAYER_ID", dropna=False)["FGA"].sum().rename("_total_fga")
    wide = df.pivot_table(index="PLAYER_ID", columns="def_bucket", values=["FGM", "FGA"], aggfunc="sum", fill_value=0)
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index().merge(totals.reset_index(), on="PLAYER_ID", how="left")

    for bucket in ["very_tight", "tight", "open", "wide_open"]:
        fgm = wide[f"FGM_{bucket}"] if f"FGM_{bucket}" in wide.columns else 0
        fga = wide[f"FGA_{bucket}"] if f"FGA_{bucket}" in wide.columns else 0
        wide[f"{prefix}_{bucket}_fg_smoothed"] = (fgm + k * global_fg) / (fga + k)
        wide[f"{prefix}_{bucket}_freq"] = np.where(wide["_total_fga"] > 0, fga / wide["_total_fga"], 0)

    keep = ["PLAYER_ID"] + [
        f"{prefix}_tight_fg_smoothed",
        f"{prefix}_open_fg_smoothed",
        f"{prefix}_wide_open_fg_smoothed",
        f"{prefix}_open_freq",
        f"{prefix}_wide_open_freq",
    ]
    for col in keep:
        if col not in wide.columns:
            wide[col] = np.nan
    return wide[keep].copy()


def merge_player_features(
    df_base: pd.DataFrame,
    player_priors: pd.DataFrame,
    zone_priors: pd.DataFrame,
    action_priors: pd.DataFrame,
    zone_action_priors: pd.DataFrame,
    distance_bin_priors: pd.DataFrame,
    all_def: pd.DataFrame,
    ft10_def: pd.DataFrame,
    league_defaults: Dict[str, float],
    global_fg: float,
    player_prior_numeric_features: List[str],
) -> pd.DataFrame:
    df = df_base.copy()

    df = df.merge(player_priors.drop(columns=["PLAYER_NAME"], errors="ignore"), on=["PLAYER_ID"], how="left")
    df = df.merge(zone_priors[["PLAYER_ID", "SHOT_ZONE_BASIC", "player_zone_fg_smoothed"]], on=["PLAYER_ID", "SHOT_ZONE_BASIC"], how="left")
    df = df.merge(action_priors[["PLAYER_ID", "ACTION_TYPE", "player_action_fg_smoothed"]], on=["PLAYER_ID", "ACTION_TYPE"], how="left")
    df = df.merge(
        zone_action_priors[["PLAYER_ID", "SHOT_ZONE_BASIC", "ACTION_TYPE", "player_zone_action_fg_smoothed"]],
        on=["PLAYER_ID", "SHOT_ZONE_BASIC", "ACTION_TYPE"],
        how="left",
    )
    df = df.merge(distance_bin_priors[["PLAYER_ID", "distance_bin", "player_distance_bin_fg_smoothed"]], on=["PLAYER_ID", "distance_bin"], how="left")

    if not all_def.empty:
        df = df.merge(all_def, on="PLAYER_ID", how="left")
    if not ft10_def.empty:
        df = df.merge(ft10_def, on="PLAYER_ID", how="left")

    if "player_zone_action_fg_smoothed" in df.columns:
        df["player_zone_action_fg_smoothed"] = (
            df["player_zone_action_fg_smoothed"]
            .fillna(df.get("player_action_fg_smoothed"))
            .fillna(df.get("player_zone_fg_smoothed"))
            .fillna(league_defaults["player_overall_fg_smoothed"])
        )

    for c in player_prior_numeric_features:
        if c not in df.columns:
            df[c] = np.nan
        if c in league_defaults:
            df[c] = df[c].fillna(league_defaults[c])
        else:
            df[c] = df[c].fillna(global_fg)

    return df


def build_clip_event_url(game_id: str, game_event_id: str, season: str) -> str:
    # Stable nba.com event page URL pattern, can be used for clip access when available.
    return (
        "https://www.nba.com/stats/events"
        f"?CFID=&CFPARAMS=&GameEventID={game_event_id}&GameID={game_id}"
        f"&Season={season}&flag=1&title="
    )


def run_pipeline(seasons: List[str] | None = None, random_seed: int = 42) -> None:
    ensure_dirs()
    shots_raw, player_dash_closest, player_dash_closest_10p = load_raw_inputs()

    if seasons:
        shots_raw = shots_raw[shots_raw["SEASON"].astype(str).isin(seasons)].copy()
        if not player_dash_closest.empty and "SEASON" in player_dash_closest.columns:
            player_dash_closest = player_dash_closest[player_dash_closest["SEASON"].astype(str).isin(seasons)].copy()
        if not player_dash_closest_10p.empty and "SEASON" in player_dash_closest_10p.columns:
            player_dash_closest_10p = player_dash_closest_10p[player_dash_closest_10p["SEASON"].astype(str).isin(seasons)].copy()

    shots_feat = engineer_features(shots_raw)
    if shots_feat.empty:
        raise RuntimeError("No engineered rows found. Check source data.")

    # Prototype strategy: fit on all available historical shots and score all rows.
    train_df = shots_feat.copy()
    score_df = shots_feat.copy()

    k_map = {
        "overall": 100,
        "2pt3pt": 150,
        "zone": 100,
        "action": 150,
        "zone_action": 250,
        "defender": 300,
        "rare": 400,
    }
    prior_payload = build_smoothed_player_features(train_df, k_map)
    player_priors = prior_payload["player_priors"].copy()
    zone_priors = prior_payload["zone_priors"].copy()
    action_priors = prior_payload["action_priors"].copy()
    zone_action_priors = prior_payload["zone_action_priors"].copy()
    distance_bin_priors = prior_payload["distance_bin_priors"].copy()
    league_defaults = prior_payload["league_defaults"]
    global_fg = float(train_df["SHOT_MADE_FLAG"].mean())

    all_def = build_defender_profile_features(player_dash_closest, "player_all", k_map["defender"], global_fg)
    ft10_def = build_defender_profile_features(player_dash_closest_10p, "player_10ft", k_map["defender"], global_fg)

    categorical_features = ["SHOT_TYPE", "ACTION_TYPE", "SHOT_ZONE_BASIC", "distance_bin"]
    numeric_features = [
        "SHOT_DISTANCE",
        "LOC_X",
        "LOC_Y",
        "shot_angle",
        "seconds_remaining_in_period",
        "seconds_elapsed_in_game",
        "is_corner_3",
        "is_restricted_area",
        "is_midrange",
        "is_paint_non_ra",
        "is_deep_3",
        "is_home_team",
        "is_late_game",
    ]
    player_prior_numeric_features = [
        "player_overall_fg_smoothed",
        "player_2pt_fg_smoothed",
        "player_3pt_fg_smoothed",
        "player_zone_fg_smoothed",
        "player_action_fg_smoothed",
        "player_zone_action_fg_smoothed",
        "player_distance_bin_fg_smoothed",
        "player_rim_fg_smoothed",
        "player_midrange_fg_smoothed",
        "player_corner_3_fg_smoothed",
        "player_above_break_3_fg_smoothed",
        "player_deep_3_fg_smoothed",
        "player_avg_shot_distance",
        "player_3pa_rate",
        "player_rim_frequency",
        "player_midrange_frequency",
        "player_corner_3_frequency",
        "player_pullup_frequency",
        "player_all_tight_fg_smoothed",
        "player_all_open_fg_smoothed",
        "player_all_wide_open_fg_smoothed",
        "player_10ft_tight_fg_smoothed",
        "player_10ft_open_fg_smoothed",
        "player_10ft_wide_open_fg_smoothed",
        "player_all_open_freq",
        "player_all_wide_open_freq",
        "player_10ft_open_freq",
        "player_10ft_wide_open_freq",
    ]

    train_df = merge_player_features(
        train_df,
        player_priors,
        zone_priors,
        action_priors,
        zone_action_priors,
        distance_bin_priors,
        all_def,
        ft10_def,
        league_defaults,
        global_fg,
        player_prior_numeric_features,
    )
    score_df = merge_player_features(
        score_df,
        player_priors,
        zone_priors,
        action_priors,
        zone_action_priors,
        distance_bin_priors,
        all_def,
        ft10_def,
        league_defaults,
        global_fg,
        player_prior_numeric_features,
    )

    feature_cols = numeric_features + categorical_features + player_prior_numeric_features
    for col in feature_cols:
        if col not in train_df.columns:
            train_df[col] = np.nan
        if col not in score_df.columns:
            score_df[col] = np.nan

    X_train = train_df[feature_cols].copy()
    y_train = train_df["SHOT_MADE_FLAG"].astype(int).copy()
    X_score = score_df[feature_cols].copy()

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features + player_prior_numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_seed,
    )

    clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    clf.fit(X_train, y_train)

    score_df["xfg_pct_player_specific"] = clf.predict_proba(X_score)[:, 1]
    score_df["xpts_player_specific"] = score_df["xfg_pct_player_specific"] * score_df["shot_value"]
    score_df["points_above_expected"] = score_df["actual_points"] - score_df["xpts_player_specific"]
    score_df["clip_event_url"] = score_df.apply(
        lambda r: build_clip_event_url(str(r["GAME_ID"]), str(r["GAME_EVENT_ID"]), str(r["SEASON"])),
        axis=1,
    )

    export_cols = [
        "GAME_ID",
        "GAME_EVENT_ID",
        "GAME_DATE",
        "PLAYER_ID",
        "PLAYER_NAME",
        "TEAM_ID",
        "TEAM_NAME",
        "SEASON",
        "PERIOD",
        "MINUTES_REMAINING",
        "SECONDS_REMAINING",
        "seconds_elapsed_in_game",
        "SHOT_TYPE",
        "ACTION_TYPE",
        "SHOT_ZONE_BASIC",
        "SHOT_ZONE_AREA",
        "SHOT_ZONE_RANGE",
        "SHOT_DISTANCE",
        "LOC_X",
        "LOC_Y",
        "SHOT_MADE_FLAG",
        "shot_value",
        "actual_points",
        "xfg_pct_player_specific",
        "xpts_player_specific",
        "points_above_expected",
        "distance_bin",
        "is_deep_3",
        "is_corner_3",
        "is_restricted_area",
        "is_midrange",
        "is_paint_non_ra",
        "player_zone_action_fg_smoothed",
        "player_distance_bin_fg_smoothed",
        "player_action_fg_smoothed",
        "player_overall_fg_smoothed",
        "clip_event_url",
    ]
    for c in export_cols:
        if c not in score_df.columns:
            score_df[c] = np.nan

    score_df = score_df[export_cols].copy()
    score_df.to_csv(PREDICTIONS_PATH, index=False)

    # Game flow overlay data
    flow = score_df.copy()
    flow["GAME_EVENT_ID"] = pd.to_numeric(flow["GAME_EVENT_ID"], errors="coerce")
    flow["seconds_elapsed_in_game"] = pd.to_numeric(flow["seconds_elapsed_in_game"], errors="coerce")
    flow = flow.sort_values(["GAME_ID", "PERIOD", "seconds_elapsed_in_game", "GAME_EVENT_ID"], na_position="last")
    flow["actual_cum_points_team"] = flow.groupby(["GAME_ID", "TEAM_NAME"], dropna=False)["actual_points"].cumsum()
    flow["xpts_cum_team"] = flow.groupby(["GAME_ID", "TEAM_NAME"], dropna=False)["xpts_player_specific"].cumsum()
    flow.to_csv(GAME_FLOW_PATH, index=False)

    # Filter tables for zone + shot-type explorer
    zone_summary = (
        score_df.groupby(["SEASON", "PLAYER_ID", "PLAYER_NAME", "TEAM_NAME", "SHOT_TYPE", "SHOT_ZONE_BASIC"], dropna=False)
        .agg(
            shots=("SHOT_MADE_FLAG", "size"),
            makes=("SHOT_MADE_FLAG", "sum"),
            actual_points=("actual_points", "sum"),
            expected_points=("xpts_player_specific", "sum"),
            avg_xfg=("xfg_pct_player_specific", "mean"),
        )
        .reset_index()
    )
    zone_summary["fg_pct"] = zone_summary["makes"] / zone_summary["shots"].clip(lower=1)
    zone_summary["actual_points_per_shot"] = zone_summary["actual_points"] / zone_summary["shots"].clip(lower=1)
    zone_summary["xpts_per_shot"] = zone_summary["expected_points"] / zone_summary["shots"].clip(lower=1)
    zone_summary["points_above_expected"] = zone_summary["actual_points"] - zone_summary["expected_points"]
    zone_summary["points_above_expected_per_shot"] = zone_summary["points_above_expected"] / zone_summary["shots"].clip(lower=1)
    zone_summary.to_csv(ZONE_SUMMARY_PATH, index=False)

    action_summary = (
        score_df.groupby(["SEASON", "PLAYER_ID", "PLAYER_NAME", "TEAM_NAME", "ACTION_TYPE"], dropna=False)
        .agg(
            shots=("SHOT_MADE_FLAG", "size"),
            makes=("SHOT_MADE_FLAG", "sum"),
            actual_points=("actual_points", "sum"),
            expected_points=("xpts_player_specific", "sum"),
            avg_xfg=("xfg_pct_player_specific", "mean"),
        )
        .reset_index()
    )
    action_summary["fg_pct"] = action_summary["makes"] / action_summary["shots"].clip(lower=1)
    action_summary["actual_points_per_shot"] = action_summary["actual_points"] / action_summary["shots"].clip(lower=1)
    action_summary["xpts_per_shot"] = action_summary["expected_points"] / action_summary["shots"].clip(lower=1)
    action_summary["points_above_expected"] = action_summary["actual_points"] - action_summary["expected_points"]
    action_summary["points_above_expected_per_shot"] = action_summary["points_above_expected"] / action_summary["shots"].clip(lower=1)
    action_summary.to_csv(ACTION_SUMMARY_PATH, index=False)

    # Clip index (one row per shot event; use url directly in frontend)
    clip_index = score_df[
        ["GAME_ID", "GAME_EVENT_ID", "SEASON", "PLAYER_NAME", "TEAM_NAME", "SHOT_TYPE", "ACTION_TYPE", "clip_event_url"]
    ].copy()
    clip_index["clip_status"] = "event_url_built"
    clip_index.to_csv(CLIP_INDEX_PATH, index=False)

    # DB staging tables
    games = (
        score_df.groupby("GAME_ID", dropna=False)
        .agg(
            GAME_DATE=("GAME_DATE", "min"),
            SEASON=("SEASON", "first"),
        )
        .reset_index()
    )
    games.to_csv(DB_STAGE_DIR / "games.csv", index=False)

    # Persist model and metadata
    joblib.dump(clf, MODEL_PATH)
    metadata = {
        "model_type": "XGBClassifier",
        "random_seed": random_seed,
        "rows_scored": int(len(score_df)),
        "seasons_scored": sorted(score_df["SEASON"].astype(str).unique().tolist()),
        "feature_count": len(feature_cols),
        "prediction_path": str(PREDICTIONS_PATH.relative_to(PROJECT_ROOT)),
        "game_flow_path": str(GAME_FLOW_PATH.relative_to(PROJECT_ROOT)),
        "zone_summary_path": str(ZONE_SUMMARY_PATH.relative_to(PROJECT_ROOT)),
        "action_summary_path": str(ACTION_SUMMARY_PATH.relative_to(PROJECT_ROOT)),
        "clip_index_path": str(CLIP_INDEX_PATH.relative_to(PROJECT_ROOT)),
    }
    MODEL_META_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved predictions: {PREDICTIONS_PATH} ({len(score_df):,} rows)")
    print(f"Saved game flow: {GAME_FLOW_PATH} ({len(flow):,} rows)")
    print(f"Saved zone summary: {ZONE_SUMMARY_PATH} ({len(zone_summary):,} rows)")
    print(f"Saved action summary: {ACTION_SUMMARY_PATH} ({len(action_summary):,} rows)")
    print(f"Saved clip index: {CLIP_INDEX_PATH} ({len(clip_index):,} rows)")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved db-stage tables in: {DB_STAGE_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare backend data assets for xPTS app.")
    parser.add_argument(
        "--seasons",
        nargs="*",
        default=None,
        help="Optional season filter, e.g. --seasons 2022-23 2023-24 2024-25",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(seasons=args.seasons, random_seed=args.seed)
