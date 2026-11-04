"""Build shot-level xPTS features from raw CSVs.

Reads the raw ShotChartDetail shots and PlayerDashPtShots closest-defender splits,
engineers shot-context features, splits train/test by season (latest season = test),
builds smoothed player priors from TRAIN seasons only, and writes a features artifact
consumed by train_model.py and predict.py.

Usage:
    python scripts/model_a/build_features.py
    python scripts/model_a/build_features.py --shots data/raw/historical/shotchart_shots.csv --out data/processed/legacy/model_a/xpts_features.pkl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from common import (
    CATEGORICAL_FEATURES,
    EXPORT_COLS,
    FEATURE_COLS,
    FEATURES_PATH,
    K_MAP,
    NUMERIC_FEATURES,
    PLAYER_DASH_CLOSEST_10P_PATH,
    PLAYER_DASH_CLOSEST_PATH,
    PLAYER_PRIOR_NUMERIC_FEATURES,
    PREDICTION_COLS,
    RANDOM_SEED,
    RAW_SHOTS_PATH,
    TARGET_COL,
    check,
    setup_logging,
)

log = setup_logging("build_features")

REQUIRED_SHOT_COLS = [
    "GAME_ID", "GAME_EVENT_ID", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_NAME",
    "PERIOD", "MINUTES_REMAINING", "SECONDS_REMAINING", "ACTION_TYPE", "SHOT_TYPE",
    "SHOT_ZONE_BASIC", "SHOT_ZONE_AREA", "SHOT_ZONE_RANGE", "SHOT_DISTANCE",
    "LOC_X", "LOC_Y", "SHOT_MADE_FLAG", "HTM", "VTM", "SEASON",
]
REQUIRED_DASH_COLS = ["PLAYER_ID", "CLOSE_DEF_DIST_RANGE", "FGM", "FGA"]

TEAM_NAME_TO_ABBR = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN", "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE", "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET", "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "LA Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM", "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP", "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA", "Washington Wizards": "WAS",
}

ENGINEERED_COLS = [
    "GAME_ID", "GAME_EVENT_ID", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_NAME", "SEASON",
    "SHOT_MADE_FLAG", "SHOT_TYPE", "ACTION_TYPE", "SHOT_ZONE_BASIC", "SHOT_ZONE_AREA", "SHOT_ZONE_RANGE",
    "SHOT_DISTANCE", "LOC_X", "LOC_Y", "PERIOD", "MINUTES_REMAINING", "SECONDS_REMAINING",
    "seconds_remaining_in_period", "seconds_elapsed_in_game", "is_home_team", "is_late_game",
    "shot_value", "actual_points", "abs_loc_x", "shot_angle",
    "is_corner_3", "is_above_break_3", "is_restricted_area", "is_midrange", "is_paint_non_ra",
    "is_left_side", "is_right_side", "is_center", "distance_bin", "is_deep_3", "is_pullup", "is_catch_shoot",
]

DEFENDER_ALIASES = {
    "player_all_very_tight_2p_smoothed": "player_all_very_tight_fg_smoothed",
    "player_all_tight_2p_smoothed": "player_all_tight_fg_smoothed",
    "player_all_open_2p_smoothed": "player_all_open_fg_smoothed",
    "player_all_wide_open_2p_smoothed": "player_all_wide_open_fg_smoothed",
    "player_all_open_3p_smoothed": "player_all_open_fg_smoothed",
    "player_all_wide_open_3p_smoothed": "player_all_wide_open_fg_smoothed",
    "player_10ft_tight_2p_smoothed": "player_10ft_tight_fg_smoothed",
    "player_10ft_open_2p_smoothed": "player_10ft_open_fg_smoothed",
    "player_10ft_tight_3p_smoothed": "player_10ft_tight_fg_smoothed",
    "player_10ft_open_3p_smoothed": "player_10ft_open_fg_smoothed",
    "player_10ft_wide_open_3p_smoothed": "player_10ft_wide_open_fg_smoothed",
}

DEFENDER_PROFILE_EXPORT_COLS = [
    c for c in EXPORT_COLS
    if c.startswith(("player_all_", "player_10ft_")) and c not in PLAYER_PRIOR_NUMERIC_FEATURES
]


# --------------------------------------------------------------------------- loading


def load_raw_inputs(shots_path: Path, closest_path: Path, closest_10p_path: Path):
    for p in [shots_path, closest_path, closest_10p_path]:
        check(p.exists(), f"missing input file {p}")

    log.info("Loading shots: %s", shots_path)
    shots_raw = pd.read_csv(shots_path)
    log.info("  shots_raw shape=%s", shots_raw.shape)
    missing = [c for c in REQUIRED_SHOT_COLS if c not in shots_raw.columns]
    check(not missing, f"shots CSV missing columns {missing}")
    check(len(shots_raw) > 0, "shots CSV is empty")

    flag_values = set(pd.to_numeric(shots_raw["SHOT_MADE_FLAG"], errors="coerce").dropna().unique())
    check(flag_values <= {0, 1}, f"SHOT_MADE_FLAG has non-binary values {sorted(flag_values)}")
    dupes = shots_raw.duplicated(subset=["GAME_ID", "GAME_EVENT_ID", "PLAYER_ID", "TEAM_ID"]).sum()
    if dupes:
        log.warning("  %d duplicate shot keys (GAME_ID, GAME_EVENT_ID, PLAYER_ID, TEAM_ID)", dupes)
    log.info("  seasons: %s", shots_raw["SEASON"].value_counts().sort_index().to_dict())
    log.info("  make rate: %.4f", shots_raw["SHOT_MADE_FLAG"].mean())

    dash = {}
    for label, p in [("closest_defender", closest_path), ("closest_defender_10ft_plus", closest_10p_path)]:
        log.info("Loading PlayerDashPtShots %s: %s", label, p)
        df = pd.read_csv(p)
        missing = [c for c in REQUIRED_DASH_COLS if c not in df.columns]
        check(not missing, f"{p.name} missing columns {missing}")
        check(len(df) > 0, f"{p.name} is empty")
        log.info("  shape=%s ranges=%s", df.shape, sorted(df["CLOSE_DEF_DIST_RANGE"].dropna().unique()))
        dash[label] = df

    return shots_raw, dash["closest_defender"], dash["closest_defender_10ft_plus"]


# --------------------------------------------------------------------------- shot context features


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in ["SHOT_DISTANCE", "LOC_X", "LOC_Y", "PERIOD", "MINUTES_REMAINING", "SECONDS_REMAINING", "SHOT_MADE_FLAG", "PLAYER_ID", "TEAM_ID"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out["seconds_remaining_in_period"] = out["MINUTES_REMAINING"].fillna(0) * 60 + out["SECONDS_REMAINING"].fillna(0)
    out["seconds_elapsed_in_period"] = 720 - out["seconds_remaining_in_period"]

    out["seconds_elapsed_in_game"] = np.where(
        out["PERIOD"] <= 4,
        (out["PERIOD"] - 1) * 720 + out["seconds_elapsed_in_period"],
        4 * 720 + (out["PERIOD"] - 5) * 300 + np.minimum(out["seconds_elapsed_in_period"], 300),
    )

    out["is_home_team"] = np.nan
    if all(c in out.columns for c in ["TEAM_NAME", "HTM", "VTM"]):
        team_abbr = out["TEAM_NAME"].map(TEAM_NAME_TO_ABBR)
        out.loc[team_abbr == out["HTM"], "is_home_team"] = 1
        out.loc[team_abbr == out["VTM"], "is_home_team"] = 0
    out["is_home_team"] = out["is_home_team"].fillna(0)

    out["is_late_game"] = (
        ((out["PERIOD"] == 4) & (out["seconds_remaining_in_period"] <= 120)) |
        (out["PERIOD"] >= 5)
    ).astype(int)

    out["shot_value"] = np.where(out["SHOT_TYPE"].astype(str).str.contains("3PT", case=False, na=False), 3, 2)
    out["actual_points"] = out["SHOT_MADE_FLAG"].fillna(0) * out["shot_value"]

    out["abs_loc_x"] = out["LOC_X"].abs()
    out["shot_angle"] = np.degrees(np.arctan2(out["LOC_Y"], out["LOC_X"]))

    shot_zone_basic = out["SHOT_ZONE_BASIC"].astype(str)
    shot_type = out["SHOT_TYPE"].astype(str)
    action_type = out["ACTION_TYPE"].astype(str)

    out["is_corner_3"] = (
        shot_type.str.contains("3PT", case=False, na=False) & out["abs_loc_x"].ge(220) & out["LOC_Y"].le(100)
    ).astype(int)
    out["is_above_break_3"] = (shot_type.str.contains("3PT", case=False, na=False) & (out["is_corner_3"] == 0)).astype(int)
    out["is_restricted_area"] = shot_zone_basic.eq("Restricted Area").astype(int)
    out["is_midrange"] = shot_zone_basic.eq("Mid-Range").astype(int)
    out["is_paint_non_ra"] = shot_zone_basic.eq("In The Paint (Non-RA)").astype(int)

    out["is_left_side"] = out["LOC_X"].lt(0).astype(int)
    out["is_right_side"] = out["LOC_X"].gt(0).astype(int)
    out["is_center"] = out["LOC_X"].eq(0).astype(int)

    out["is_deep_3"] = ((out["SHOT_DISTANCE"] >= 28) & shot_type.str.contains("3PT", case=False, na=False)).astype(int)

    out["distance_bin"] = pd.cut(
        out["SHOT_DISTANCE"],
        bins=[-np.inf, 3, 7, 15, 23, 27, 31, np.inf],
        labels=["0-3", "4-7", "8-15", "16-23", "24-27", "28-31", "32+"],
    ).astype(str)

    out["is_pullup"] = action_type.str.contains("pull", case=False, na=False).astype(int)
    out["is_catch_shoot"] = action_type.str.contains("catch|jump shot", case=False, na=False).astype(int)

    for c in ENGINEERED_COLS:
        if c not in out.columns:
            out[c] = np.nan

    out = out[ENGINEERED_COLS].copy()
    out = out[out["SHOT_MADE_FLAG"].isin([0, 1])].copy()
    return out


# --------------------------------------------------------------------------- player priors (train only)


def _agg_rate(source: pd.DataFrame, group_cols, output_col: str, k: float, global_avg: float):
    tmp = source.copy()
    tmp["_made"] = tmp["SHOT_MADE_FLAG"]
    tmp["_att"] = 1.0
    grouped = tmp.groupby(group_cols, dropna=False)[["_made", "_att"]].sum().reset_index()
    grouped[output_col] = (grouped["_made"] + k * global_avg) / (grouped["_att"] + k)
    return grouped[group_cols + [output_col, "_att"]]


def build_smoothed_player_features(train_df: pd.DataFrame, k_map: dict) -> dict:
    global_fg = train_df["SHOT_MADE_FLAG"].mean()

    is_2pt = train_df["shot_value"] == 2
    is_3pt = train_df["shot_value"] == 3

    player_base = train_df.groupby(["PLAYER_ID"], dropna=False).agg(
        PLAYER_NAME=("PLAYER_NAME", "first"), player_shots=("SHOT_MADE_FLAG", "size")
    ).reset_index()

    overall = _agg_rate(train_df, ["PLAYER_ID"], "player_overall_fg_smoothed", k_map["overall"], global_fg)
    p2 = _agg_rate(train_df[is_2pt], ["PLAYER_ID"], "player_2pt_fg_smoothed", k_map["2pt3pt"], global_fg)
    p3 = _agg_rate(train_df[is_3pt], ["PLAYER_ID"], "player_3pt_fg_smoothed", k_map["2pt3pt"], global_fg)

    p_zone = _agg_rate(train_df, ["PLAYER_ID", "SHOT_ZONE_BASIC"], "player_zone_fg_smoothed", k_map["zone"], global_fg)
    p_action = _agg_rate(train_df, ["PLAYER_ID", "ACTION_TYPE"], "player_action_fg_smoothed", k_map["action"], global_fg)
    p_zone_action = _agg_rate(
        train_df, ["PLAYER_ID", "SHOT_ZONE_BASIC", "ACTION_TYPE"], "player_zone_action_fg_smoothed",
        k_map["zone_action"], global_fg,
    )
    p_dist = _agg_rate(train_df, ["PLAYER_ID", "distance_bin"], "player_distance_bin_fg_smoothed", k_map["zone_action"], global_fg)

    rim = _agg_rate(train_df[train_df["is_restricted_area"] == 1], ["PLAYER_ID"], "player_rim_fg_smoothed", k_map["zone"], global_fg)
    mid = _agg_rate(train_df[train_df["is_midrange"] == 1], ["PLAYER_ID"], "player_midrange_fg_smoothed", k_map["zone"], global_fg)
    c3 = _agg_rate(train_df[train_df["is_corner_3"] == 1], ["PLAYER_ID"], "player_corner_3_fg_smoothed", k_map["zone_action"], global_fg)
    ab3 = _agg_rate(train_df[train_df["is_above_break_3"] == 1], ["PLAYER_ID"], "player_above_break_3_fg_smoothed", k_map["zone_action"], global_fg)
    d3 = _agg_rate(train_df[train_df["is_deep_3"] == 1], ["PLAYER_ID"], "player_deep_3_fg_smoothed", k_map["rare"], global_fg)

    tendencies = train_df.groupby("PLAYER_ID", dropna=False).agg(
        player_avg_shot_distance=("SHOT_DISTANCE", "mean"),
        player_3pa_rate=("shot_value", lambda s: np.mean(s == 3)),
        player_rim_frequency=("is_restricted_area", "mean"),
        player_midrange_frequency=("is_midrange", "mean"),
        player_corner_3_frequency=("is_corner_3", "mean"),
        player_pullup_frequency=("is_pullup", "mean"),
        player_catch_shoot_frequency=("is_catch_shoot", "mean"),
    ).reset_index()

    priors = player_base[["PLAYER_ID", "PLAYER_NAME"]].drop_duplicates()
    for tbl in [overall, p2, p3, rim, mid, c3, ab3, d3, tendencies]:
        priors = priors.merge(tbl.drop(columns=[c for c in ["_att"] if c in tbl.columns]), on="PLAYER_ID", how="left")

    def _subset_mean(mask: pd.Series) -> float:
        return train_df.loc[mask, "SHOT_MADE_FLAG"].mean() if mask.any() else global_fg

    league_defaults = {
        "player_overall_fg_smoothed": global_fg,
        "player_2pt_fg_smoothed": _subset_mean(is_2pt),
        "player_3pt_fg_smoothed": _subset_mean(is_3pt),
        "player_rim_fg_smoothed": _subset_mean(train_df["is_restricted_area"] == 1),
        "player_midrange_fg_smoothed": _subset_mean(train_df["is_midrange"] == 1),
        "player_corner_3_fg_smoothed": _subset_mean(train_df["is_corner_3"] == 1),
        "player_above_break_3_fg_smoothed": _subset_mean(train_df["is_above_break_3"] == 1),
        "player_deep_3_fg_smoothed": _subset_mean(train_df["is_deep_3"] == 1),
        "player_avg_shot_distance": train_df["SHOT_DISTANCE"].mean(),
        "player_3pa_rate": np.mean(train_df["shot_value"] == 3),
        "player_rim_frequency": train_df["is_restricted_area"].mean(),
        "player_midrange_frequency": train_df["is_midrange"].mean(),
        "player_corner_3_frequency": train_df["is_corner_3"].mean(),
        "player_pullup_frequency": train_df["is_pullup"].mean(),
        "player_catch_shoot_frequency": train_df["is_catch_shoot"].mean(),
    }

    return {
        "player_priors": priors,
        "zone_priors": p_zone,
        "action_priors": p_action,
        "zone_action_priors": p_zone_action,
        "distance_bin_priors": p_dist,
        "league_defaults": league_defaults,
        "global_fg": global_fg,
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

    df = df_raw.copy()
    if not all(c in df.columns for c in REQUIRED_DASH_COLS):
        return pd.DataFrame(columns=["PLAYER_ID"])

    for c in ["FGM", "FGA"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["def_bucket"] = df["CLOSE_DEF_DIST_RANGE"].map(_normalize_def_range)

    totals = df.groupby("PLAYER_ID", dropna=False)["FGA"].sum().rename("_total_fga")
    wide = df.pivot_table(index="PLAYER_ID", columns="def_bucket", values=["FGM", "FGA"], aggfunc="sum", fill_value=0)
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index().merge(totals.reset_index(), on="PLAYER_ID", how="left")

    for b in ["very_tight", "tight", "open", "wide_open"]:
        fgm = wide[f"FGM_{b}"] if f"FGM_{b}" in wide.columns else 0
        fga = wide[f"FGA_{b}"] if f"FGA_{b}" in wide.columns else 0
        wide[f"{prefix}_{b}_fg_smoothed"] = (fgm + k * global_fg) / (fga + k)
        wide[f"{prefix}_{b}_freq"] = np.where(wide["_total_fga"] > 0, fga / wide["_total_fga"], 0)

    keep = ["PLAYER_ID"] + [
        f"{prefix}_very_tight_fg_smoothed", f"{prefix}_tight_fg_smoothed", f"{prefix}_open_fg_smoothed", f"{prefix}_wide_open_fg_smoothed",
        f"{prefix}_very_tight_freq", f"{prefix}_tight_freq", f"{prefix}_open_freq", f"{prefix}_wide_open_freq",
    ]
    return wide[keep].copy()


def merge_player_features(
    df_base: pd.DataFrame,
    priors: dict,
    all_def: pd.DataFrame,
    ft10_def: pd.DataFrame,
) -> pd.DataFrame:
    league_defaults = priors["league_defaults"]
    global_fg = priors["global_fg"]

    df = df_base.copy()
    df = df.merge(priors["player_priors"].drop(columns=["PLAYER_NAME"], errors="ignore"), on=["PLAYER_ID"], how="left")
    df = df.merge(priors["zone_priors"][["PLAYER_ID", "SHOT_ZONE_BASIC", "player_zone_fg_smoothed"]], on=["PLAYER_ID", "SHOT_ZONE_BASIC"], how="left")
    df = df.merge(priors["action_priors"][["PLAYER_ID", "ACTION_TYPE", "player_action_fg_smoothed"]], on=["PLAYER_ID", "ACTION_TYPE"], how="left")
    df = df.merge(
        priors["zone_action_priors"][["PLAYER_ID", "SHOT_ZONE_BASIC", "ACTION_TYPE", "player_zone_action_fg_smoothed"]],
        on=["PLAYER_ID", "SHOT_ZONE_BASIC", "ACTION_TYPE"], how="left",
    )
    df = df.merge(
        priors["distance_bin_priors"][["PLAYER_ID", "distance_bin", "player_distance_bin_fg_smoothed"]],
        on=["PLAYER_ID", "distance_bin"], how="left",
    )

    if not all_def.empty:
        df = df.merge(all_def, on="PLAYER_ID", how="left")
    if not ft10_def.empty:
        df = df.merge(ft10_def, on="PLAYER_ID", how="left")

    # Hierarchical fallback for the finest-grained prior: zone+action -> action -> zone -> league.
    df["player_zone_action_fg_smoothed"] = (
        df["player_zone_action_fg_smoothed"]
        .fillna(df.get("player_action_fg_smoothed"))
        .fillna(df.get("player_zone_fg_smoothed"))
        .fillna(league_defaults["player_overall_fg_smoothed"])
    )

    for c in PLAYER_PRIOR_NUMERIC_FEATURES:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = df[c].fillna(league_defaults.get(c, global_fg))

    # Export-only defender-profile columns get the same league fallback for players absent from PlayerDashPtShots.
    for c in DEFENDER_PROFILE_EXPORT_COLS:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = df[c].fillna(global_fg)

    for alias_col, base_col in DEFENDER_ALIASES.items():
        if alias_col not in df.columns:
            df[alias_col] = df.get(base_col, global_fg)
        df[alias_col] = df[alias_col].fillna(df.get(base_col, global_fg)).fillna(global_fg)

    return df


# --------------------------------------------------------------------------- split + orchestration


def split_train_test(shots_feat: pd.DataFrame):
    usable_seasons = sorted([s for s in shots_feat["SEASON"].dropna().unique().tolist() if isinstance(s, str)])
    if len(usable_seasons) >= 2:
        test_season = usable_seasons[-1]
        train_df = shots_feat[shots_feat["SEASON"] != test_season].copy()
        test_df = shots_feat[shots_feat["SEASON"] == test_season].copy()
        return train_df, test_df, f"season-based (test={test_season})", test_season

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
    idx_train, idx_test = next(gss.split(shots_feat, groups=shots_feat["GAME_ID"]))
    return shots_feat.iloc[idx_train].copy(), shots_feat.iloc[idx_test].copy(), "grouped random split by GAME_ID", None


def validate_features(train_df: pd.DataFrame, test_df: pd.DataFrame, n_test_before_merge: int, test_season) -> None:
    check(len(train_df) > 0 and len(test_df) > 0, "train/test split produced an empty side")
    check(len(test_df) == n_test_before_merge, "prior merges changed test row count (fan-out in a join)")
    if test_season is not None:
        check(test_season not in set(train_df["SEASON"]), "test season leaked into train")
    for name, df in [("train", train_df), ("test", test_df)]:
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        check(not missing, f"{name} missing feature columns {missing}")
        prior_nans = df[PLAYER_PRIOR_NUMERIC_FEATURES].isna().sum()
        check(prior_nans.sum() == 0, f"{name} has NaN player priors: {prior_nans[prior_nans > 0].to_dict()}")
        rate_cols = [c for c in PLAYER_PRIOR_NUMERIC_FEATURES if c.endswith(("_smoothed", "_freq", "_rate", "_frequency"))]
        out_of_range = ((df[rate_cols] < 0) | (df[rate_cols] > 1)).sum()
        check(out_of_range.sum() == 0, f"{name} has rate priors outside [0, 1]: {out_of_range[out_of_range > 0].to_dict()}")
        check(df[TARGET_COL].isin([0, 1]).all(), f"{name} target is not binary")
        num_nans = df[NUMERIC_FEATURES].isna().sum()
        if num_nans.sum():
            log.warning("%s numeric feature NaNs (median-imputed in model): %s", name, num_nans[num_nans > 0].to_dict())


def build_features(
    shots_path: Path = RAW_SHOTS_PATH,
    closest_path: Path = PLAYER_DASH_CLOSEST_PATH,
    closest_10p_path: Path = PLAYER_DASH_CLOSEST_10P_PATH,
    out_path: Path = FEATURES_PATH,
) -> dict:
    shots_raw, dash_closest, dash_closest_10p = load_raw_inputs(shots_path, closest_path, closest_10p_path)

    log.info("Engineering shot-context features")
    shots_feat = engineer_features(shots_raw)
    log.info("  engineered shape=%s (dropped %d rows with non-binary target)", shots_feat.shape, len(shots_raw) - len(shots_feat))

    for col in CATEGORICAL_FEATURES + NUMERIC_FEATURES:
        if col not in shots_feat.columns:
            shots_feat[col] = np.nan

    train_df, test_df, split_method, test_season = split_train_test(shots_feat)
    log.info("Split: %s | train=%d rows (%s) | test=%d rows", split_method, len(train_df),
             sorted(train_df["SEASON"].unique()), len(test_df))

    log.info("Building smoothed player priors from TRAIN seasons only (k=%s)", K_MAP)
    priors = build_smoothed_player_features(train_df, K_MAP)
    global_fg = priors["global_fg"]
    log.info("  %d players with priors | train global FG=%.4f", len(priors["player_priors"]), global_fg)

    all_def = build_defender_profile_features(dash_closest, "player_all", K_MAP["defender"], global_fg)
    ft10_def = build_defender_profile_features(dash_closest_10p, "player_10ft", K_MAP["defender"], global_fg)
    log.info("  closest-defender profiles: all=%d players, 10ft+=%d players", len(all_def), len(ft10_def))

    n_test = len(test_df)
    train_df = merge_player_features(train_df, priors, all_def, ft10_def)
    test_df = merge_player_features(test_df, priors, all_def, ft10_def)

    validate_features(train_df, test_df, n_test, test_season)
    unseen = (~test_df["PLAYER_ID"].isin(priors["player_priors"]["PLAYER_ID"])).sum()
    log.info("  test shots from players unseen in train (league-default priors): %d (%.1f%%)",
             unseen, 100 * unseen / max(len(test_df), 1))

    keep_cols = list(dict.fromkeys(
        [c for c in EXPORT_COLS if c not in PREDICTION_COLS] + FEATURE_COLS + [TARGET_COL]
    ))
    artifact = {
        "train": train_df[keep_cols].reset_index(drop=True),
        "test": test_df[keep_cols].reset_index(drop=True),
        "meta": {
            "split_method": split_method,
            "test_season": test_season,
            "train_seasons": sorted(train_df["SEASON"].dropna().unique().tolist()),
            "global_fg": float(global_fg),
            "k_map": K_MAP,
            "feature_cols": FEATURE_COLS,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out_path)
    log.info("Saved features -> %s | train=%s test=%s", out_path, artifact["train"].shape, artifact["test"].shape)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shots", type=Path, default=RAW_SHOTS_PATH, help="raw ShotChartDetail CSV")
    parser.add_argument("--closest", type=Path, default=PLAYER_DASH_CLOSEST_PATH, help="PlayerDashPtShots closest-defender CSV")
    parser.add_argument("--closest-10ft", type=Path, default=PLAYER_DASH_CLOSEST_10P_PATH, help="PlayerDashPtShots closest-defender 10ft+ CSV")
    parser.add_argument("--out", type=Path, default=FEATURES_PATH, help="output features artifact (.pkl)")
    args = parser.parse_args()
    build_features(args.shots, args.closest, args.closest_10ft, args.out)


if __name__ == "__main__":
    main()
