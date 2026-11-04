"""Season prediction export and product analytics (Game Center, Expected Standings, Shot Explorer,
Nightly Highlights) built on the latest Master model.

Predictions never look ahead: shots on game date D are scored with in-season priors built from
games dated before D. Each shot is labelled with the partition its game belonged to during training
(model_fit = in-sample, calibration, test = held out, after_training = genuinely new).

Official FGA without a Blueprint tracking record are scored by the saved no-tracking fallback model
(context + player features) and flagged `master_source = "fallback_no_tracking"`; tracked shots carry
`master_source = "master"`. NBA xFG is never filled in for shots Blueprint did not publish.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import ANALYTICS_DIR, MASTER_DATASET_DIR
from pipeline.utils import get_logger, records, require, write_json

log = get_logger("exports")

PRED_DIR = MASTER_DATASET_DIR / "predictions"


def _partition(dates: pd.Series, meta: dict) -> pd.Series:
    f0, f1 = meta["trained_on_dates"]
    c0, c1 = meta["calibrated_on_dates"]
    t0, t1 = meta["test_dates"]
    return pd.Series(np.select([dates.between(f0, f1), dates.between(c0, c1), dates.between(t0, t1), dates > t1],
                               ["model_fit", "calibration", "test", "after_training"], default="before_training"),
                     index=dates.index)


def export_season_predictions(model_dir: str | Path | None = None) -> Path:
    from pipeline.inference import MasterModel, build_prior_state, predict_shots
    mm = MasterModel(Path(model_dir) if model_dir else None)
    fb_dir = mm.dir / "fallback_no_tracking"
    fb = MasterModel(fb_dir) if fb_dir.exists() else None
    raw = pd.read_parquet(MASTER_DATASET_DIR / mm.meta["dataset_version"] / "master_shots.parquet")
    raw["in_blueprint"] = raw["in_blueprint"].astype(bool)
    out = []
    for day in sorted(raw["game_date"].unique()):
        state = build_prior_state(raw, before_date=day)
        d = raw[raw["game_date"] == day]
        tr, un = d[d["in_blueprint"]], d[~d["in_blueprint"]]
        if len(tr):
            p = predict_shots(tr[mm.required_inputs], model=mm, state=state)
            out.append(p.assign(master_source="master"))
        if len(un) and fb is not None:
            p = predict_shots(un[fb.required_inputs], model=fb, state=state)
            out.append(p.assign(master_source="fallback_no_tracking"))
    pred = pd.concat(out, ignore_index=True)
    keep = ["shot_uid", "GAME_ID", "GAME_EVENT_ID", "game_date", "team", "TEAM_ID", "PLAYER_ID", "PLAYER_NAME",
            "defender_id", "defender_name", "PERIOD", "pbp_clock", "pbp_clock_remaining_s", "ACTION_TYPE", "SHOT_TYPE",
            "SHOT_ZONE_BASIC", "SHOT_DISTANCE", "bp_shot_distance_ft", "LOC_X", "LOC_Y", "contest_distance_in", "shot_clock_s",
            "made", "nba_xfg", "model_a_xfg", "in_blueprint", "ngss_event_id", "pbp_actionId", "pbp_description",
            "pbp_videoAvailable", "pbp_teamTricode", "SHOT_MADE_FLAG"]
    res = raw[keep].merge(pred.drop(columns=["PLAYER_ID"]), on=["GAME_ID", "GAME_EVENT_ID"], how="left", validate="one_to_one")
    res["partition"] = _partition(res["game_date"], mm.meta)
    res["master_xfg_tracked_only"] = res["master_xfg"].where(res["master_source"] == "master")
    out_dir = PRED_DIR / mm.spec["model_version"]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "shot_predictions.parquet"
    res.to_parquet(path, index=False)
    write_json(out_dir / "export_meta.json", {
        "model_version": mm.spec["model_version"], "rows": int(len(res)),
        "scored_master": int((res["master_source"] == "master").sum()),
        "scored_fallback": int((res["master_source"] == "fallback_no_tracking").sum()),
        "unscored": int(res["master_xfg"].isna().sum()), "partition_counts": res["partition"].value_counts().to_dict(),
        "note": "model_fit rows are in-sample for the final model; use test/after_training rows for honest evaluation.",
    })
    log.info("exported %d shot predictions to %s", len(res), path)
    return path


def latest_predictions() -> Path:
    paths = sorted(PRED_DIR.glob("master_*/shot_predictions.parquet"), key=lambda p: p.stat().st_mtime)
    require(paths, "no exported predictions; run `python -m pipeline export-predictions` first")
    return paths[-1]


def build_season_analytics(pred_path: Path | None = None) -> list[Path]:
    from pipeline import analytics
    from pipeline.ingestion import nba_pages
    pred_path = pred_path or latest_predictions()
    p = pd.read_parquet(pred_path)
    version = pred_path.parent.name
    out_dir = ANALYTICS_DIR / "season" / version
    out_dir.mkdir(parents=True, exist_ok=True)
    u = p.copy()
    u["in_official"] = True
    shots = analytics.prepare_shots(u)
    shots["master_xpts_fg"] = shots["master_xfg"] * shots["shot_value"]
    written = []

    # Game Center: one timeline per game
    gc_dir = out_dir / "game_center"
    gc_dir.mkdir(exist_ok=True)
    boxes = []
    for gid in sorted(shots["GAME_ID"].unique()):
        page = nba_pages.parse_game_page(nba_pages.load_game_page(gid), gid)
        boxes.append(page.team_box)
        tl = analytics.game_timeline(shots, gid, page.header, nba_pages.score_timeline(page), page.team_box)
        write_json(gc_dir / f"{gid}.json", tl)
    written.append(gc_dir)
    boxes = pd.concat(boxes, ignore_index=True)

    # Expected Standings
    st = {
        "master_deterministic": analytics.expected_standings(shots, boxes, "master_xfg"),
        "model_a_deterministic": analytics.expected_standings(shots, boxes, "model_a_xfg"),
        "nba_xfg_deterministic_covered_games_only": analytics.expected_standings(shots, boxes, "nba_xfg"),
        "note": "master_xfg uses the no-tracking fallback for untracked FGA (see master_source); NBA xFG is "
                "unavailable for untracked FGA, so games with any untracked FGA are excluded from its standings.",
    }
    written.append(write_json(out_dir / "expected_standings.json", st))

    # Shot Explorer: full shot table + aggregates
    explorer = shots[[c for c in ["shot_uid", "GAME_ID", "game_date", "team", "PLAYER_ID", "PLAYER_NAME", "defender_id",
                                  "defender_name", "PERIOD", "pbp_clock", "ACTION_TYPE", "SHOT_TYPE", "SHOT_ZONE_BASIC",
                                  "SHOT_DISTANCE", "LOC_X", "LOC_Y", "court_x_ft", "court_y_ft", "made", "shot_value",
                                  "actual_fg_points", "master_xfg", "master_xpts_fg", "master_source", "nba_xfg",
                                  "model_a_xfg", "contest_distance_in", "shot_clock_s", "in_blueprint", "partition"]
                       if c in shots.columns]]
    explorer.to_parquet(out_dir / "explorer_shots.parquet", index=False)
    written.append(out_dir / "explorer_shots.parquet")
    agg = {
        "players_min50": records(analytics.aggregate_shots(shots, ["PLAYER_ID", "PLAYER_NAME"], 50, prob_col="master_xfg")),
        "teams": records(analytics.aggregate_shots(shots, ["team"], 1, prob_col="master_xfg")),
        "zones": records(analytics.aggregate_shots(shots, ["SHOT_ZONE_BASIC"], 1, prob_col="master_xfg")),
        "teams_nba_xfg_covered": records(analytics.aggregate_shots(shots, ["team"], 1, prob_col="nba_xfg")),
    }
    written.append(write_json(out_dir / "explorer_aggregates.json", {"schema_version": "explorer_aggregates.v2",
                                                                     "probability": "master_xfg", **agg}))

    # Nightly Highlights: latest date + every date
    nights = {}
    for day in sorted(shots["game_date"].unique()):
        nights[day] = analytics.shot_rankings(shots[shots["master_source"] == "master"], n=5, date=day, prob_col="master_xfg")
    written.append(write_json(out_dir / "nightly_highlights.json", {"schema_version": "nightly.v1", "by_date": nights,
                                                                    "latest_date": max(nights)}))
    write_json(out_dir / "analytics_meta.json", {"source_predictions": str(pred_path.name), "model_version": version,
                                                 "games": int(shots["GAME_ID"].nunique()), "shots": int(len(shots)),
                                                 "files": [str(w.name) for w in written]})
    log.info("analytics written to %s", out_dir)
    return written
