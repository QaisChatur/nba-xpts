"""Master Shot Dataset: one row per official 2026-27 field-goal attempt of every verified game.

Columns (see docs/DATA_MATCHING.md and docs/MODEL_METHODOLOGY.md):
  identifiers   shot_uid, GAME_ID, GAME_EVENT_ID, game_date, PLAYER_ID, PLAYER_NAME, TEAM_ID, team, defender_id ...
  label         made (official SHOT_MADE_FLAG; Blueprint `success` must agree on matched rows)
  features      groups in configs/features.yaml (shot_context, game_context, tracking, player, ...)
  benchmarks    nba_xfg (Blueprint expFieldGoalPct), model_a_xfg, baseline_hist_xfg -- never model inputs
  flags         in_blueprint (matched one-to-one to a Blueprint record), match_method

Leakage controls: the score context comes from actions before the shot; player/defender in-season
priors use only games dated strictly before the shot's game; pre-season priors use 2021-22..2025-26
data; the shot-difficulty baseline is fitted on 2022-23..2024-25 only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pipeline.config import (
    CONFIGS_DIR, DASH_CLOSEST_PATH, MASTER_DATASET_DIR, RAW_SHOTS_PATH, SEASON_MANIFEST,
)
from pipeline.features import baseline as bl
from pipeline.features import player_priors as pp
from pipeline.features.shot_context import ZONE_GROUP_ORDER, pbp_context, shot_features
from pipeline.ingestion import blueprint, nba_pages
from pipeline.ingestion.blueprint_scrape import paths as bp_paths
from pipeline.matching import build_unified, match_shots
from pipeline.utils import get_logger, require, sha256_file, utc_now, write_json

log = get_logger("dataset")

POSS_TYPES = ["opp_score", "def_rebound", "off_rebound", "opp_turnover", "period_start", "jump_ball", "unknown"]
FAMILIES = bl.FAMILIES


def load_feature_config(path: Path = CONFIGS_DIR / "features.yaml") -> dict:
    return yaml.safe_load(path.read_text())


def feature_list(cfg: dict, groups: list[str] | None = None) -> list[str]:
    groups = groups or list(cfg["groups"])
    cols = [c for g in groups for c in cfg["groups"][g]]
    bad = set(cols) & set(cfg["forbidden"])
    require(not bad, f"forbidden columns configured as features: {sorted(bad)}")
    return cols


# --------------------------------------------------------------------------- pre-season assets


def build_history_assets(cfg: dict, out_dir: Path) -> dict:
    """Fit the shot-difficulty baseline and pre-season player tables. Saved under out_dir."""
    import joblib
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["GAME_ID", "PLAYER_ID", "PERIOD", "MINUTES_REMAINING", "SECONDS_REMAINING", "ACTION_TYPE", "SHOT_TYPE",
            "SHOT_ZONE_BASIC", "LOC_X", "LOC_Y", "SHOT_MADE_FLAG", "SEASON"]
    hist = pd.read_csv(RAW_SHOTS_PATH, usecols=cols, dtype={"GAME_ID": str})
    feats = shot_features(hist)
    log.info("fitting shot-difficulty baseline on %d historical shots", len(hist))
    model = bl.fit_baseline(feats, hist["SHOT_MADE_FLAG"])
    xb = bl.predict_baseline(model, feats)
    weights = cfg["history"]["season_weights"]
    h = pd.DataFrame({
        "PLAYER_ID": hist["PLAYER_ID"].astype(int), "SEASON": hist["SEASON"], "w": hist["SEASON"].map(weights).fillna(0.0),
        "made": hist["SHOT_MADE_FLAG"].astype(float), "resid": hist["SHOT_MADE_FLAG"] - xb,
        "is_three": feats["is_three"], "zone_group": feats["zone_group"], "action_family": feats["action_family"],
    })
    h = h[h["w"] > 0]
    hist_agg = pp.aggregate(h)
    league = pp.league_rates(h)
    sd = pp.season_smoe_sd(h)
    dash = pd.read_csv(DASH_CLOSEST_PATH)
    dash_last, dash_range = pp.dash_tables(dash, cfg["history"]["last_season_dash"])

    joblib.dump(model, out_dir / "baseline_hgb.joblib")
    hist_agg.reset_index().to_csv(out_dir / "player_hist_agg.csv", index=False)
    pd.Series(league).rename_axis(["kind", "cat"]).rename("fg").reset_index().to_csv(out_dir / "league_rates.csv", index=False)
    sd.rename("smoe_season_sd").reset_index().to_csv(out_dir / "player_smoe_season_sd.csv", index=False)
    dash_last.reset_index().to_csv(out_dir / "dash_last_season.csv", index=False)
    dash_range.reset_index().to_csv(out_dir / "dash_by_range.csv", index=False)
    meta = {
        "built_utc": utc_now(), "n_hist_shots": int(len(hist)), "season_weights": weights,
        "league_three_rate": float(np.average(h["is_three"], weights=h["w"])),
        "league_rim_rate": float(np.average(h["zone_group"] == "rim", weights=h["w"])),
        "baseline_hist_logloss_in_sample": float(-np.mean(hist["SHOT_MADE_FLAG"] * np.log(xb) + (1 - hist["SHOT_MADE_FLAG"]) * np.log(1 - xb))),
        "inputs": {"shotchart": sha256_file(RAW_SHOTS_PATH), "dash_closest": sha256_file(DASH_CLOSEST_PATH)},
        "baseline_features": bl.BASELINE_FEATURES,
    }
    write_json(out_dir / "history_assets_meta.json", meta)
    return meta


def load_history_assets(asset_dir: Path, cfg: dict):
    """Rebuild the PriorBuilder + baseline from saved CSV/joblib assets (no pickles of untrusted origin:
    the baseline joblib is produced by this pipeline)."""
    import joblib
    meta = json.loads((asset_dir / "history_assets_meta.json").read_text())
    model = joblib.load(asset_dir / "baseline_hgb.joblib")
    hist_agg = pd.read_csv(asset_dir / "player_hist_agg.csv").set_index(["PLAYER_ID", "kind", "cat"])
    lr = pd.read_csv(asset_dir / "league_rates.csv")
    league = {(k, c): f for k, c, f in lr.itertuples(index=False)}
    sd = pd.read_csv(asset_dir / "player_smoe_season_sd.csv").set_index("PLAYER_ID")["smoe_season_sd"]
    dash_last = pd.read_csv(asset_dir / "dash_last_season.csv").set_index("PLAYER_ID")
    dash_range = pd.read_csv(asset_dir / "dash_by_range.csv").set_index(["PLAYER_ID", "range"])
    params = dict(cfg["priors"], league_three_rate=meta["league_three_rate"], league_rim_rate=meta["league_rim_rate"])
    return model, pp.PriorBuilder(hist_agg, league, dash_last, dash_range, sd, params), meta


# --------------------------------------------------------------------------- per-game assembly


def game_frame(game_id: str, verified: bool) -> pd.DataFrame:
    """Official shots + pre-shot context (+ matched Blueprint fields if verified) for one game."""
    page = nba_pages.parse_game_page(nba_pages.load_game_page(game_id), game_id)
    h = page.header
    official = nba_pages.pbp_shots(page)
    official["clock_remaining_s"] = official["pbp_clock_remaining_s"]
    ctx = pbp_context(page.actions, int(h["home_team_id"]), int(h["away_team_id"]))
    require(len(ctx) == len(official), f"{game_id}: context rows {len(ctx)} != shots {len(official)}")
    official = official.merge(ctx, on="GAME_EVENT_ID", how="left", validate="one_to_one")
    official["game_date"] = h["game_date_et"]
    if not verified:
        official["in_blueprint"] = False
        return official
    p = bp_paths(game_id)
    bp = blueprint.load_blueprint_game(p["csv"], p["raw"].parent)
    bp["shot_uid"] = bp["game_id"] + "_" + bp["ngss_event_id"].astype(str).str.zfill(4)
    bp = blueprint.fill_dates(bp, {game_id: h["game_date_et"]})
    res = match_shots(bp, official)
    u = build_unified(official, bp, res)
    u = u[u["in_official"]].copy()  # Blueprint-only rows have no official record (reported in coverage)
    return u


def tracking_features(u: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=u.index)
    for c in ["shot_clock_s", "contest_distance_in", "contest_height_in", "shot_vertical_in", "shooter_speed_mph",
              "release_time_s", "bp_shot_distance_ft"]:
        out[c] = pd.to_numeric(u.get(c), errors="coerce")
    out["shot_clock_missing"] = out["shot_clock_s"].isna().astype(int)
    out["release_time_zero"] = (out["release_time_s"] == 0).astype(int)      # not measured (rim attempts)
    out["release_time_capped"] = (out["release_time_s"] >= 1.0).astype(int)  # Blueprint caps at 1.0 s
    out["contest_ratio"] = (out["contest_distance_in"] / 12.0) / np.maximum(out["bp_shot_distance_ft"], 1.0)
    return out


def in_season_priors(u: pd.DataFrame, priors: pp.PriorBuilder, cfg: dict) -> pd.DataFrame:
    """Prior features for every shot using only shots from game dates strictly before its own."""
    hist_like = pd.DataFrame({
        "PLAYER_ID": u["PLAYER_ID"].astype(int), "w": cfg["priors"]["in_season_weight"], "made": u["made"].astype(float),
        "resid": u["resid_baseline"], "is_three": u["is_three"], "zone_group": u["zone_group"],
        "action_family": u["action_family"], "game_date": u["game_date"],
    })
    tracked = u["in_blueprint"].astype(bool).to_numpy() & pd.to_numeric(u["defender_id"], errors="coerce").notna().to_numpy()
    dfr = hist_like[tracked].assign(PLAYER_ID=pd.to_numeric(u.loc[tracked, "defender_id"]).astype(int), w=1.0)
    frames = []
    for day in sorted(u["game_date"].unique()):
        rows = u.index[u["game_date"] == day]
        earlier = hist_like[hist_like["game_date"] < day]
        dearlier = dfr[dfr["game_date"] < day]
        frames.append(priors.features(u.loc[rows], pp.aggregate(earlier) if len(earlier) else None,
                                      pp.aggregate(dearlier) if len(dearlier) else None))
    return pd.concat(frames).loc[u.index]


def build_master_dataset(cfg: dict | None = None, out_dir: Path | None = None, model_a: bool = True,
                         rebuild_history: bool = False) -> Path:
    cfg = cfg or load_feature_config()
    version = cfg["dataset_version"]
    out_dir = out_dir or MASTER_DATASET_DIR / version
    asset_dir = out_dir / "history_assets"
    if rebuild_history or not (asset_dir / "history_assets_meta.json").exists():
        build_history_assets(cfg, asset_dir)
    base_model, priors, hist_meta = load_history_assets(asset_dir, cfg)

    manifest = json.loads(SEASON_MANIFEST.read_text())
    usable = {g: r for g, r in manifest["games"].items()
              if r.get("status") in ("verified", "blueprint_empty") and r.get("pbp_matches_box")}
    require(usable, "no verified games in the season manifest; run `python -m pipeline ingest-season` first")
    frames = []
    for gid, rec in sorted(usable.items()):
        frames.append(game_frame(gid, rec["status"] == "verified"))
    u = pd.concat(frames, ignore_index=True)
    log.info("assembled %d official shots from %d games (%d with Blueprint tracking)",
             len(u), len(usable), int(u["in_blueprint"].sum()))

    sf = shot_features(u)
    u = u.drop(columns=[c for c in sf.columns if c in u.columns]).join(sf)
    u["zone_code"] = u["zone_group"].map({z: i for i, z in enumerate(ZONE_GROUP_ORDER)}).astype(int)
    u["family_code"] = u["action_family"].map({f: i for i, f in enumerate(FAMILIES)}).fillna(-1).astype(int)
    u["poss_code"] = u["poss_start_type"].map({p: i for i, p in enumerate(POSS_TYPES)}).fillna(len(POSS_TYPES) - 1).astype(int)
    u["baseline_hist_xfg"] = bl.predict_baseline(base_model, sf)
    if "made" in u:  # Blueprint `success` on matched rows
        both = u["in_blueprint"].astype(bool)
        require((u.loc[both, "made"].astype(int) == u.loc[both, "SHOT_MADE_FLAG"].astype(int)).all(),
                "Blueprint/official outcome disagreement on matched rows")
    u["made"] = u["SHOT_MADE_FLAG"].astype(int)
    u["resid_baseline"] = u["made"] - u["baseline_hist_xfg"]
    tf = tracking_features(u)
    u = u.drop(columns=[c for c in tf.columns if c in u.columns]).join(tf)

    pf = in_season_priors(u, priors, cfg)
    u = u.drop(columns=[c for c in pf.columns if c in u.columns]).join(pf)

    if model_a:
        from pipeline.legacy import model_a as ma
        pred = ma.predict(u[nba_pages.PBP_SHOT_COLUMNS + ["clock_remaining_s"]])
        u = u.merge(pred[["GAME_ID", "GAME_EVENT_ID", "model_a_xfg", "model_a_player_has_priors"]],
                    on=["GAME_ID", "GAME_EVENT_ID"], how="left", validate="one_to_one")
    u["shot_value"] = np.where(u["is_three"] == 1, 3, 2)
    u["team"] = u["pbp_teamTricode"]
    u = u.loc[:, ~u.columns.duplicated()]
    require(not u["shot_uid"].duplicated().any(), "shot_uid not unique")

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "master_shots.parquet"
    obj_cols = u.select_dtypes(include="object").columns
    u[obj_cols] = u[obj_cols].astype("string")
    u.to_parquet(path, index=False)
    feats = feature_list(cfg)
    meta = {
        "dataset_version": version, "built_utc": utc_now(), "rows": int(len(u)),
        "rows_with_tracking": int(u["in_blueprint"].sum()), "games": int(u["GAME_ID"].nunique()),
        "games_with_tracking": int(u.loc[u["in_blueprint"], "GAME_ID"].nunique()),
        "date_range": [str(u["game_date"].min()), str(u["game_date"].max())],
        "features": feats, "feature_groups": cfg["groups"], "forbidden": cfg["forbidden"],
        "season_manifest_sha256": sha256_file(SEASON_MANIFEST), "history_assets": hist_meta,
        "feature_missing_on_tracked_rows": {c: int(u.loc[u["in_blueprint"], c].isna().sum()) for c in feats if c in u},
    }
    write_json(out_dir / "dataset_meta.json", meta)
    log.info("wrote %s (%d rows)", path, len(u))
    return path
