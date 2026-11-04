"""Command-line interface: `python -m pipeline <command>`."""

from __future__ import annotations

import argparse
import json
import sys

from pipeline.config import (
    BLUEPRINT_GAMES_DIR, BLUEPRINT_RAW_DIR, BLUEPRINT_SCHEDULE_DIR, DASH_CLOSEST_10FT_PATH, DASH_CLOSEST_PATH,
    GAME_PAGES_DIR, HISTORICAL_FEATURES_PATH, HISTORICAL_PREDICTIONS_PATH, MODEL_A_ASSETS_DIR,
    MODEL_A_HISTORICAL_METRICS_PATH, MODEL_A_PATH, RAW_SHOTS_PATH,
)
from pipeline.utils import rel

REQUIRED = {
    "Model A artifact": MODEL_A_PATH,
    "historical ShotChartDetail shots (prior rebuild)": RAW_SHOTS_PATH,
    "PlayerDashPtShots closest defender": DASH_CLOSEST_PATH,
    "PlayerDashPtShots closest defender 10ft+": DASH_CLOSEST_10FT_PATH,
    "Blueprint game CSVs": BLUEPRINT_GAMES_DIR,
    "Blueprint schedule-day cache": BLUEPRINT_SCHEDULE_DIR,
}
OPTIONAL = {
    "Blueprint raw JSON (row cross-check, team IDs)": BLUEPRINT_RAW_DIR,
    "historical features artifact (reproduction checks)": HISTORICAL_FEATURES_PATH,
    "historical predictions (reproduction check)": HISTORICAL_PREDICTIONS_PATH,
    "historical metrics (reference)": MODEL_A_HISTORICAL_METRICS_PATH,
    "Model A inference assets (built by build-model-a-assets)": MODEL_A_ASSETS_DIR / "assets_meta.json",
    "cached NBA.com game pages": GAME_PAGES_DIR,
}


def cmd_audit(_args) -> int:
    ok = True
    for title, group in (("required", REQUIRED), ("optional", OPTIONAL)):
        print(f"\n{title}:")
        for name, path in group.items():
            exists = path.exists() and (path.is_file() or any(path.iterdir()))
            ok &= exists or title == "optional"
            print(f"  [{'x' if exists else ' '}] {name}: {rel(path)}")
    return 0 if ok else 1


def cmd_fetch_pages(args) -> int:
    from pipeline.ingestion import nba_pages
    from pipeline.ingestion import schedule
    sched = schedule.load_schedule_day(args.date)
    have_bp = {p.stem for p in BLUEPRINT_GAMES_DIR.glob("*.csv")}
    games = sched if args.all_scheduled else sched[sched["game_id"].isin(have_bp)]
    status = nba_pages.fetch_many(games[["game_id", "away", "home"]].to_dict(orient="records"),
                                  allow_network=args.allow_network, delay_s=args.delay, refresh=args.refresh)
    print(json.dumps(status, indent=2))
    print(schedule.validate_schedule_day(args.date).to_string())
    return 0


def cmd_build_assets(_args) -> int:
    from pipeline.env import ensure_xgboost_importable
    ensure_xgboost_importable()
    from pipeline.legacy import model_a
    meta = model_a.build_inference_assets()
    print(json.dumps(meta["reproduction"], indent=2))
    return 0


def cmd_verify_model_a(_args) -> int:
    from pipeline.env import ensure_xgboost_importable
    ensure_xgboost_importable()
    from pipeline.legacy import model_a
    res = model_a.verify_reproduction()
    print(json.dumps(res, indent=2))
    return 0 if res["status"] == "passed" else 1


def cmd_run(args) -> int:
    from pipeline.env import ensure_xgboost_importable
    ensure_xgboost_importable()
    from pipeline.legacy import model_a
    from pipeline import run
    if not (MODEL_A_ASSETS_DIR / "assets_meta.json").exists():
        model_a.build_inference_assets()
    run.run_date(args.date, allow_network=args.allow_network)
    return 0


def cmd_ingest_season(args) -> int:
    from pipeline.ingestion import season
    m = season.ingest_season(args.start, args.end, allow_network=args.allow_network, delay_s=args.delay,
                             max_games=args.max_games)
    path = season.write_coverage_report(m)
    print(json.dumps(m["summary"], indent=2))
    print(f"coverage report: {rel(path)}")
    return 1 if m.get("stopped") else 0


def cmd_coverage_report(args) -> int:
    from pipeline.ingestion import season
    m = season.reverify_all() if args.reverify else season.load_manifest()
    print(f"coverage report: {rel(season.write_coverage_report(m))}")
    return 0


def cmd_build_dataset(args) -> int:
    from pipeline.env import ensure_xgboost_importable
    ensure_xgboost_importable()
    from pipeline.features import dataset
    path = dataset.build_master_dataset(model_a=not args.no_model_a, rebuild_history=args.rebuild_history)
    print(f"master dataset: {rel(path)}")
    return 0


def cmd_train_master(args) -> int:
    from pipeline.env import ensure_xgboost_importable
    ensure_xgboost_importable()
    from pipeline.training import train
    run_dir = train.run_training(quick=args.quick, model_version=args.model_version)
    print(f"run: {rel(run_dir)}")
    if not args.no_eval:
        from pipeline.training import evaluate
        evaluate.evaluate_run(run_dir)
        print(f"report: {rel(evaluate.REPORT_PATH)}")
    return 0


def cmd_evaluate_master(args) -> int:
    from pipeline.env import ensure_xgboost_importable
    ensure_xgboost_importable()
    from pathlib import Path
    from pipeline.config import EXPERIMENTS_DIR
    from pipeline.training import evaluate
    runs = sorted((EXPERIMENTS_DIR / "master_runs").glob("2*"))
    run_dir = Path(args.run) if args.run else runs[-1]
    evaluate.evaluate_run(run_dir)
    print(f"report: {rel(evaluate.REPORT_PATH)}")
    return 0


def cmd_export_predictions(args) -> int:
    from pipeline.env import ensure_xgboost_importable
    ensure_xgboost_importable()
    from pipeline import exports
    path = exports.export_season_predictions(model_dir=args.model_dir)
    print(f"predictions: {rel(path)}")
    return 0


def cmd_update_analytics(args) -> int:
    from pipeline import exports
    paths = exports.build_season_analytics()
    for p in paths:
        print(rel(p))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m pipeline", description="xPoints data and modelling pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("audit", help="list required/optional inputs and whether they exist").set_defaults(fn=cmd_audit)
    f = sub.add_parser("fetch-pages", help="cache NBA.com game pages (box score + play-by-play) for a date")
    f.add_argument("--date", required=True)
    f.add_argument("--allow-network", action="store_true", help="permit HTTP requests (otherwise cache only)")
    f.add_argument("--all-scheduled", action="store_true", help="fetch every scheduled game, not only Blueprint games")
    f.add_argument("--refresh", action="store_true")
    f.add_argument("--delay", type=float, default=4.0)
    f.set_defaults(fn=cmd_fetch_pages)
    sub.add_parser("build-model-a-assets", help="rebuild + verify Model A player-prior tables").set_defaults(fn=cmd_build_assets)
    sub.add_parser("verify-model-a", help="reproduce saved historical predictions/metrics").set_defaults(fn=cmd_verify_model_a)
    r = sub.add_parser("run", help="ingest, match, score, compare, Model B, analytics and reports for a date")
    r.add_argument("--date", required=True)
    r.add_argument("--allow-network", action="store_true", help="fetch missing game pages (one request per game)")
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("ingest-season", help="discover, fetch (cached) and verify every completed game in a date range")
    s.add_argument("--start", default="2026-10-18")
    s.add_argument("--end", default=None, help="ET date, default yesterday")
    s.add_argument("--allow-network", action="store_true", help="permit HTTP requests (otherwise cache only)")
    s.add_argument("--delay", type=float, default=3.5, help="seconds between requests")
    s.add_argument("--max-games", type=int, default=None)
    s.set_defaults(fn=cmd_ingest_season)
    c = sub.add_parser("coverage-report", help="write the season coverage report from the manifest")
    c.add_argument("--reverify", action="store_true", help="re-run offline verification for every cached game first")
    c.set_defaults(fn=cmd_coverage_report)
    b = sub.add_parser("build-dataset", help="build the Master Shot Dataset from verified season games")
    b.add_argument("--rebuild-history", action="store_true", help="refit the historical baseline and pre-season priors")
    b.add_argument("--no-model-a", action="store_true", help="skip Model A benchmark scoring")
    b.set_defaults(fn=cmd_build_dataset)
    t = sub.add_parser("train-master", help="tune, select, calibrate and save a new Master xFG model (+ evaluation)")
    t.add_argument("--quick", action="store_true", help="few Optuna trials (smoke test)")
    t.add_argument("--model-version", default=None)
    t.add_argument("--no-eval", action="store_true")
    t.set_defaults(fn=cmd_train_master)
    ev = sub.add_parser("evaluate-master", help="held-out evaluation + docs/reports/MASTER_MODEL_REPORT.md for a run")
    ev.add_argument("--run", default=None, help="run directory (default: latest)")
    ev.set_defaults(fn=cmd_evaluate_master)
    ex = sub.add_parser("export-predictions", help="score every season shot with the latest Master model (no look-ahead)")
    ex.add_argument("--model-dir", default=None)
    ex.set_defaults(fn=cmd_export_predictions)
    ua = sub.add_parser("update-analytics", help="rebuild season analytics tables from exported predictions")
    ua.set_defaults(fn=cmd_update_analytics)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
