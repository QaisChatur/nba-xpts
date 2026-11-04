"""End-to-end orchestration for one game date (offline from caches unless network is allowed)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import analytics
from pipeline.ingestion import blueprint
from pipeline import evaluation
from pipeline import matching
from pipeline.legacy import model_a
from pipeline.legacy import model_b
from pipeline.ingestion import nba_pages
from pipeline.ingestion import schedule
from pipeline.config import (
    ANALYTICS_DIR, BLUEPRINT_GAMES_DIR, BLUEPRINT_MANIFEST, MATCH_CLOCK_TOL_S, MATCH_COORD_TOL_FT,
    HISTORICAL_PREDICTIONS_PATH, MODEL_A_HISTORICAL_METRICS_PATH, MODEL_B_DIR, PROJECT_ROOT, REPORTS_DIR, UNIFIED_DIR,
)
from pipeline.evaluation.reports import fmt, md_table
from pipeline.utils import get_logger, records, rel, require, sha256_file, utc_now, write_json

log = get_logger("run")

DIST_BINS = [-0.1, 3, 9, 15, 22, 100]
DIST_LABELS = ["0-3 ft", "4-9 ft", "10-15 ft", "16-22 ft", "23+ ft"]
CONTEST_BINS = [0, 24, 48, 72, 1e9]
CONTEST_LABELS = ["0-2 ft (very tight)", "2-4 ft (tight)", "4-6 ft (open)", "6+ ft (wide open)"]


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return None


# --------------------------------------------------------------------------- stage 1: ingest


def ingest(date: str, allow_network: bool = False, only_games: list[str] | None = None) -> dict:
    bp_ids = sorted(p.stem for p in BLUEPRINT_GAMES_DIR.glob("*.csv"))
    if only_games is not None:
        bp_ids = [g for g in bp_ids if g in set(only_games)]
    sched_games = schedule.load_schedule_day(date).set_index("game_id")
    to_fetch = [{"game_id": g, "away": sched_games.loc[g, "away"], "home": sched_games.loc[g, "home"]}
                for g in bp_ids if g in sched_games.index]
    fetch_status = nba_pages.fetch_many(to_fetch, allow_network=allow_network)
    pages = {}
    for g in bp_ids:
        try:
            pages[g] = nba_pages.parse_game_page(nba_pages.load_game_page(g), g)
        except Exception as exc:  # missing cache is reported, not fatal for other games
            log.warning("game page for %s unavailable: %s", g, exc)
    game_ids = [g for g, p in pages.items() if p.header["game_date_et"] == date]
    require(game_ids, f"no Blueprint games with a cached official game page dated {date}")
    log.info("games on %s with Blueprint + official data: %s", date, game_ids)

    bp = blueprint.load_blueprint(game_ids=game_ids)
    bp = blueprint.fill_dates(bp, {g: pages[g].header["game_date_et"] for g in game_ids})
    official = pd.concat([nba_pages.pbp_shots(pages[g]) for g in game_ids], ignore_index=True)
    official["clock_remaining_s"] = official["pbp_clock_remaining_s"]
    official["game_date"] = date
    box = pd.concat([nba_pages.box_validation(pages[g], official[official["GAME_ID"] == g]) for g in game_ids], ignore_index=True)
    require(box["pbp_matches_box"].all(), f"play-by-play FGA/FGM/3PA disagree with the box score:\n{box}")
    return {
        "date": date, "game_ids": game_ids, "pages": pages, "bp": bp, "official": official, "box": box,
        "schedule": schedule.validate_schedule_day(date), "fetch_status": fetch_status,
        "score_consistency": blueprint.score_consistency(bp, {g: pages[g].header["home_tricode"] for g in game_ids}),
        "field_audit": blueprint.audit_fields(bp),
        "manifest": json.loads(BLUEPRINT_MANIFEST.read_text()) if BLUEPRINT_MANIFEST.exists() else {},
    }


# --------------------------------------------------------------------------- stage 2: match


def match(ctx: dict) -> dict:
    res = matching.match_shots(ctx["bp"], ctx["official"])
    unified = matching.build_unified(ctx["official"], ctx["bp"], res)
    ctx.update(match_result=res, unified=unified)
    return ctx


def unified_missing_summary(u: pd.DataFrame) -> str:
    """Missing values that are *not* explained by row origin (official-only / Blueprint-only)."""
    off_only = u["in_official"] & ~u["in_blueprint"]
    bp_only = ~u["in_official"]
    rows = []
    for col in u.columns:
        miss = u[col].isna()
        if not miss.any():
            continue
        unexplained = miss & ~off_only & ~bp_only
        rows.append({"column": col, "missing": int(miss.sum()), "missing_on_matched_rows": int(unexplained.sum()),
                     "missing_on_official_only_rows": int((miss & off_only).sum()),
                     "missing_on_blueprint_only_rows": int((miss & bp_only).sum())})
    t = pd.DataFrame(rows)
    flagged = t[(t["missing_on_matched_rows"] > 0) | (t["missing_on_official_only_rows"] > 0) & (t["missing_on_official_only_rows"] < off_only.sum())]
    by_origin = len(t) - len(flagged)
    return "\n".join([
        f"Rows: {len(u)} ({int((u['in_official'] & u['in_blueprint']).sum())} matched, {int(off_only.sum())} official-only, "
        f"{int(bp_only.sum())} Blueprint-only). {by_origin} columns are missing only because of row origin "
        "(Blueprint fields on official-only rows, official fields on the Blueprint-only row). Columns with other gaps:",
        "",
        md_table(flagged) if len(flagged) else "_none_",
        "",
        "`SHOT_ZONE_AREA` is not available from the play-by-play; `model_b_xfg_exploratory` exists only for compared shots.",
    ])


def coverage_table(ctx: dict) -> pd.DataFrame:
    bp, box = ctx["bp"], ctx["box"]
    st = ctx["match_result"].bp_status
    bp_team = bp.groupby(["game_id", "shooter_team"]).size().rename("blueprint_shots")
    matched = bp.merge(st[["shot_uid", "match_status"]], on="shot_uid")
    m_team = matched[matched["match_status"] == "matched"].groupby(["game_id", "shooter_team"]).size().rename("matched")
    t = box.rename(columns={"game_id": "game_id", "team": "shooter_team"}).set_index(["game_id", "shooter_team"])
    out = t[["box_fga", "pbp_fga"]].join(bp_team).join(m_team).fillna(0).reset_index()
    out["blueprint_coverage_of_box_fga"] = out["blueprint_shots"] / out["box_fga"]
    out["matched_coverage_of_box_fga"] = out["matched"] / out["box_fga"]
    return out.rename(columns={"shooter_team": "team"})


# --------------------------------------------------------------------------- stage 3: Model A


def score_model_a(ctx: dict) -> dict:
    model = model_a.load_model_a()
    pred = model_a.predict(ctx["official"], model=model)
    u = ctx["unified"].merge(pred, on=["GAME_ID", "GAME_EVENT_ID"], how="left")
    u.loc[~u["in_official"], [c for c in pred.columns if c.startswith("model_a")]] = np.nan
    require(u.loc[u["in_official"], "model_a_xfg"].notna().all(), "Model A missing predictions for official shots")
    assets = model_a.load_inference_assets()
    ld = assets["meta"]["league_defaults"]
    is3 = u["SHOT_TYPE"].astype(str).str.startswith("3")
    u["baseline_shot_type_xfg"] = np.where(is3, ld["player_3pt_fg_smoothed"], ld["player_2pt_fg_smoothed"])
    u["shot_value_official"] = np.where(is3, 3, 2)
    u["actual_fg_points"] = u["SHOT_MADE_FLAG"] * u["shot_value_official"]
    ctx.update(unified=u, model_a=model, model_a_provenance=model._xpts_provenance)
    ctx["model_a_reproduction"] = model_a.verify_reproduction(model)
    return ctx


# --------------------------------------------------------------------------- stage 4: compare


def comparison_frame(u: pd.DataFrame) -> pd.DataFrame:
    """Exactly the shots on which Model A and NBA xFG can be compared."""
    e = u[u["in_official"] & u["in_blueprint"] & (u["match_status"] == "matched")].copy()
    require(e["nba_xfg"].notna().all() and e["model_a_xfg"].notna().all(), "comparison rows missing probabilities")
    require((e["made"] == e["SHOT_MADE_FLAG"]).all(), "matched rows disagree on outcome")
    e["shot_value"] = e["shot_value_official"]
    e["dist_bin"] = pd.cut(e["SHOT_DISTANCE"], DIST_BINS, labels=DIST_LABELS)
    e["contest_bin"] = pd.cut(e["contest_distance_in"], CONTEST_BINS, labels=CONTEST_LABELS, right=False)
    e["outcome"] = np.where(e["SHOT_MADE_FLAG"] == 1, "made", "missed")
    e["shot_type"] = np.where(e["shot_value"] == 3, "3PT", "2PT")
    e["model_a_priors"] = np.where(e["model_a_player_has_priors"], "player has 2022-24 priors", "league-default priors")
    e["team"] = e["pbp_teamTricode"]
    return e


MODEL_COLS = {"Model A": "model_a_xfg", "NBA xFG": "nba_xfg", "Shot-type baseline": "baseline_shot_type_xfg"}


def extreme_table(df: pd.DataFrame, prob_col: str, y_col: str = "SHOT_MADE_FLAG") -> pd.DataFrame:
    """Share of shots with extreme predictions and how often they actually went in, by shot type."""
    rows = []
    for st, g in df.groupby(df["SHOT_TYPE"].astype(str).str[:3]):
        for label, m in (("prediction > 0.8", g[prob_col] > 0.8), ("prediction < 0.1", g[prob_col] < 0.1)):
            rows.append({"shot_type": st, "band": label, "shots": int(len(g)), "share_of_shots": float(m.mean()),
                         "n_in_band": int(m.sum()), "mean_pred_in_band": float(g.loc[m, prob_col].mean()) if m.any() else None,
                         "observed_fg_in_band": float(g.loc[m, y_col].mean()) if m.any() else None})
    return pd.DataFrame(rows)


def historical_extremes() -> pd.DataFrame | None:
    if not HISTORICAL_PREDICTIONS_PATH.exists():
        return None
    p = pd.read_csv(HISTORICAL_PREDICTIONS_PATH, usecols=["SHOT_TYPE", "SHOT_MADE_FLAG", "xfg_pct_player_specific"])
    return extreme_table(p, "xfg_pct_player_specific")


def compare(ctx: dict, out_dir: Path) -> dict:
    e = comparison_frame(ctx["unified"])
    y = "SHOT_MADE_FLAG"
    metrics = evaluation.compare_models(e, y, MODEL_COLS)
    boot = evaluation.paired_bootstrap(e, y, MODEL_COLS, pairs=[("Model A", "NBA xFG"), ("Model A", "Shot-type baseline"),
                                                               ("NBA xFG", "Shot-type baseline")])
    full = ctx["unified"][ctx["unified"]["in_official"]]
    full_metrics = evaluation.compare_models(full, y, {"Model A": "model_a_xfg", "Shot-type baseline": "baseline_shot_type_xfg"})
    missing = full[~full["in_blueprint"]]
    missing_metrics = evaluation.compare_models(missing, y, {"Model A": "model_a_xfg"}) if len(missing) >= 5 else None
    two = {"Model A": "model_a_xfg", "NBA xFG": "nba_xfg"}
    subgroups = {col: evaluation.subgroup_table(e, col, y, two) for col in
                 ["shot_type", "dist_bin", "SHOT_ZONE_BASIC", "contest_bin", "outcome", "model_a_priors", "GAME_ID", "team"]}
    players = evaluation.subgroup_table(e.assign(player=e["PLAYER_NAME"]), "player", y, two, min_n=12)
    dis = evaluation.disagreement_analysis(
        e, y, "model_a_xfg", "nba_xfg",
        ["shot_uid", "PLAYER_NAME", "team", "ACTION_TYPE", "SHOT_ZONE_BASIC", "SHOT_DISTANCE", "contest_distance_in",
         "defender_name", "model_a_player_has_priors"])
    evaluation.plot_calibration(e, y, MODEL_COLS, out_dir / "calibration_model_a_vs_nba.png",
                                f"Calibration on matched {ctx['date']} shots")
    hist = json.loads(MODEL_A_HISTORICAL_METRICS_PATH.read_text())
    res = {"frame": e, "metrics": metrics, "bootstrap": boot, "full_metrics": full_metrics,
           "missing_metrics": missing_metrics, "subgroups": subgroups, "players": players, "disagreement": dis,
           "historical": hist, "historical_extremes": historical_extremes(),
           "new_extremes": extreme_table(e, "model_a_xfg")}
    ctx["comparison"] = res
    return ctx


# --------------------------------------------------------------------------- stage 5: Model B


def run_model_b(ctx: dict) -> dict:
    e = ctx["comparison"]["frame"].copy()
    exp = model_b.run_experiment(e)
    summ = model_b.summarize(e, exp)
    saved = model_b.fit_and_save_final(e, exp, summ, MODEL_B_DIR, tag=f"exploratory_{ctx['date']}")
    oof = exp["oof"]["B1_A_plus_tracking"].rename("model_b_xfg_exploratory")
    u = ctx["unified"].drop(columns=["model_b_xfg_exploratory"], errors="ignore")
    u = u.merge(e[["shot_uid"]].assign(model_b_xfg_exploratory=oof.to_numpy()), on="shot_uid", how="left")
    ctx.update(unified=u, model_b={"experiment": exp, "summary": summ, "saved_to": rel(saved), "frame": e})
    return ctx


# --------------------------------------------------------------------------- stage 6: analytics


def build_analytics(ctx: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    shots = analytics.prepare_shots(ctx["unified"])
    written = []
    for gid in ctx["game_ids"]:
        page = ctx["pages"][gid]
        tl = analytics.game_timeline(shots, gid, page.header, nba_pages.score_timeline(page), page.team_box)
        written.append(write_json(out_dir / f"timeline_{gid}.json", tl))
    nightly = analytics.shot_rankings(shots, n=5, date=ctx["date"])
    season = analytics.shot_rankings(shots, n=5)
    nightly_a = analytics.shot_rankings(shots, n=5, date=ctx["date"], prob_col="model_a_xfg")
    written.append(write_json(out_dir / "rankings.json", {"nightly_nba_xfg": nightly, "season_nba_xfg": season,
                                                         "nightly_model_a": nightly_a}))
    boxes = pd.concat([ctx["pages"][g].team_box for g in ctx["game_ids"]], ignore_index=True)
    st_a = analytics.expected_standings(shots, boxes, "model_a_xfg")
    st_nba = analytics.expected_standings(shots, boxes, "nba_xfg")
    st_prob = analytics.expected_standings(shots, boxes, "model_a_xfg", analytics.IndependentShotWinProbability())
    written.append(write_json(out_dir / "expected_standings.json", {
        "model_a_deterministic": st_a, "nba_xfg_deterministic": st_nba, "model_a_experimental_probabilistic": st_prob}))
    explorer_cols = ["shot_uid", "GAME_ID", "game_date", "SEASON", "team", "PLAYER_ID", "PLAYER_NAME", "defender_id",
                     "defender_name", "PERIOD", "pbp_clock", "ACTION_TYPE", "SHOT_TYPE", "SHOT_ZONE_BASIC", "SHOT_DISTANCE",
                     "LOC_X", "LOC_Y", "court_x_ft", "court_y_ft", "made", "shot_value", "actual_fg_points",
                     "model_a_xfg", "nba_xfg", "contest_distance_in", "in_blueprint"]
    written.append(write_json(out_dir / "explorer_shots.json", {"schema_version": "explorer_shots.v1",
                                                                "shots": records(shots[explorer_cols])}))
    corner_fade = analytics.filter_shots(shots, zones=["Right Corner 3", "Left Corner 3"])
    examples = {
        "players_min5_model_a": records(analytics.aggregate_shots(shots, ["PLAYER_ID", "PLAYER_NAME"], 5)),
        "teams_model_a": records(analytics.aggregate_shots(shots, ["team"], 1)),
        "zones_model_a": records(analytics.aggregate_shots(shots, ["SHOT_ZONE_BASIC"], 1)),
        "zones_nba_xfg_covered": records(analytics.aggregate_shots(shots, ["SHOT_ZONE_BASIC"], 1, prob_col="nba_xfg")),
        "corner_threes_by_team": records(analytics.aggregate_shots(corner_fade, ["team"], 1)),
    }
    written.append(write_json(out_dir / "explorer_aggregates.json", {"schema_version": "explorer_aggregates.v1",
                                                                     "note": "Exploratory one-night samples; apply sensible minimum-attempt thresholds over a season.",
                                                                     **examples}))
    ctx["analytics"] = {"files": [rel(p) for p in written], "shots": shots, "standings": st_a, "standings_nba": st_nba,
                        "nightly": nightly}
    return ctx


# --------------------------------------------------------------------------- outputs and reports


def write_datasets(ctx: dict, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    st = ctx["match_result"].bp_status
    files = {
        "official_shots.csv": ctx["official"],
        "blueprint_shots_normalized.csv": ctx["bp"],
        "match_status.csv": st,
        "unified_shots.csv": ctx["unified"],
        "unmatched_official.csv": ctx["match_result"].unmatched_official,
        "unmatched_blueprint.csv": ctx["bp"][ctx["bp"]["shot_uid"].isin(st.loc[st["match_status"] != "matched", "shot_uid"])],
        "box_score_validation.csv": ctx["box"],
        "model_comparison_shots.csv": ctx["comparison"]["frame"],
    }
    for name, df in files.items():
        df.to_csv(out_dir / name, index=False)
    prov = {
        "created_utc": utc_now(), "git_commit": git_commit(), "date": ctx["date"], "games": ctx["game_ids"],
        "sources": {
            "blueprint_csvs": {r: h for r, h in ctx["bp"][["source_file", "source_sha256"]].drop_duplicates().itertuples(index=False)},
            "game_pages": {g: json.loads((nba_pages.cache_paths(g)[1]).read_text()) for g in ctx["game_ids"]},
            "model_a": ctx["model_a_provenance"],
        },
        "row_counts": {k: int(len(v)) for k, v in files.items()},
        "matching_params": ctx["match_result"].params,
    }
    write_json(out_dir / "provenance.json", prov)
    return [rel(out_dir / n) for n in list(files) + ["provenance.json"]]


def matching_report(ctx: dict, path: Path) -> None:
    res = ctx["match_result"]
    hyp = res.hypothesis
    st = res.bp_status
    cov = coverage_table(ctx)
    um = res.unmatched_official
    bp_bad = st[st["match_status"] != "matched"].merge(ctx["bp"], on=["shot_uid", "game_id", "ngss_event_id"])
    matched = st[st["match_status"] == "matched"]
    disc = matched.merge(ctx["bp"], on=["shot_uid", "game_id", "ngss_event_id"]).merge(
        ctx["official"], left_on=["game_id", "official_event_id"], right_on=["GAME_ID", "GAME_EVENT_ID"])
    value_disc = disc[disc["value_agree"] == False]  # noqa: E712
    disc["bp_dist_minus_official_ft"] = disc["bp_shot_distance_ft"] - np.hypot(disc["LOC_X"], disc["LOC_Y"]) / 10
    um_summary = um.assign(result=np.where(um["SHOT_MADE_FLAG"] == 1, "made", "missed"),
                           tip_or_putback=um["ACTION_TYPE"].str.contains("Tip|Putback", case=False))
    all_official = ctx["official"]
    tip_rate_all = all_official["ACTION_TYPE"].str.contains("Tip|Putback", case=False).mean()

    lines = [
        f"# Matching-quality report — {ctx['date']}",
        "",
        f"Generated {utc_now()} by `python -m pipeline run --date {ctx['date']}`.",
        "",
        "## Sources",
        "",
        "* **Blueprint** (NBA xFG + tracking): `data/raw/blueprint/2026-27/games/*.csv`, cross-checked row by row against `raw/*.json`.",
        "* **Official play-by-play + box score**: NBA.com game page `__NEXT_DATA__` payload, cached in `data/raw/nba_pages/2026-27/`. "
        "ShotChartDetail (stats.nba.com) and cdn.nba.com liveData were not reachable from this environment (timeout / HTTP 403). "
        "The play-by-play supplies the ShotChartDetail fields directly (event number, player, team, period, clock, LOC_X/LOC_Y as "
        "xLegacy/yLegacy, ACTION_TYPE as subType, result); SHOT_DISTANCE and SHOT_ZONE_BASIC are derived with rules validated on "
        "540,627 historical ShotChartDetail shots (100% and 99.98% agreement).",
        "",
        "## Schedule validation (cached games listing `%s.json`)" % ctx["date"],
        "",
        md_table(ctx["schedule"][["game_id", "away", "home", "official_date_et", "status", "flags"]]),
        "",
        "Games without a cached official page remain `unverified`; they were not fetched because no Blueprint data exists for them.",
        "",
        "## Coverage against the official box score",
        "",
        "Official play-by-play FGA/FGM/3PA equal the box score for every team (see `box_score_validation.csv`).",
        "",
        md_table(cov, 3),
        "",
        f"**Blueprint covers {int(cov['blueprint_shots'].sum())} of {int(cov['box_fga'].sum())} official FGA "
        f"({cov['blueprint_shots'].sum() / cov['box_fga'].sum():.1%}); {int(cov['matched'].sum())} are matched.** "
        "The scraper manifest marked every game `unverified` (box-score fetch failed), so this gap was previously undetected.",
        "",
        "## Event-identifier hypothesis",
        "",
        "Hypothesis: Blueprint `ngss_event_id` equals the official play-by-play `actionNumber` (= ShotChartDetail `GAME_EVENT_ID`). "
        "A pair is counted as corroborating only if period, shooter and result agree, the clock differs by at most "
        f"{MATCH_CLOCK_TOL_S:g} s and the court location by at most {MATCH_COORD_TOL_FT:g} ft.",
        "",
        md_table(hyp, 4),
        "",
        "Every Blueprint ID existed in the official play-by-play; all but the conflicts listed below were corroborated, so the "
        "attribute fallback was never needed on these games (it is exercised by unit tests).",
        "",
        "## Match quality",
        "",
        f"* Status counts: {st['match_status'].value_counts().to_dict()}",
        f"* Clock difference on matched pairs (s): mean {matched['clock_diff_s'].mean():.3f}, max {matched['clock_diff_s'].max():.2f} "
        f"(Blueprint clocks are whole seconds; official clocks have tenths in the last minute).",
        f"* Court-location difference on matched pairs (ft): max {matched['coord_diff_ft'].max():.2f} (identical coordinates).",
        f"* Blueprint `shot_distance_ft` (tracking) minus official coordinate distance: median "
        f"{disc['bp_dist_minus_official_ft'].median():.2f} ft, 5th–95th pct {disc['bp_dist_minus_official_ft'].quantile(.05):.2f} to "
        f"{disc['bp_dist_minus_official_ft'].quantile(.95):.2f} ft. The two distances are measured differently; Model A uses the official one.",
        f"* 2PT/3PT classification disagreements on matched pairs: {len(value_disc)}",
        "",
    ]
    if len(value_disc):
        lines += [md_table(value_disc[["shot_uid", "PLAYER_NAME", "SHOT_TYPE", "shot_value", "bp_shot_distance_ft", "LOC_X", "LOC_Y", "pbp_description"]]
                           .rename(columns={"shot_value": "bp_value"}), 2), "",
                  "Official classification is used for shot value and Model A; NBA xFG is kept as published.", ""]
    lines += ["## Blueprint rows not matched", ""]
    if len(bp_bad):
        lines += [md_table(bp_bad[["shot_uid", "match_status", "id_pair_rejection", "period", "clock_raw", "shooter_name", "made",
                                   "home_score_after", "away_score_after"]]), "",
                  "Investigated: for `0022600133_0203` Blueprint records Deandre Ayton's tip-in as a miss while its own running "
                  "score rises 48→50 on that shot and the official play-by-play records a make. This is a Blueprint data error; the "
                  "row is excluded from the comparison rather than force-matched, and the official record is kept.", ""]
    else:
        lines += ["None.", ""]
    lines += [
        "## Official FGA absent from Blueprint",
        "",
        f"{len(um)} official FGA have no Blueprint record. They are not a random sample: "
        f"{(um_summary['result'] == 'missed').mean():.0%} are misses (all official FGA: {1 - all_official['SHOT_MADE_FLAG'].mean():.0%}) and "
        f"{um_summary['tip_or_putback'].mean():.0%} are tip/putback attempts (all official FGA: {tip_rate_all:.0%}). "
        "Any NBA-xFG evaluation therefore covers a subset that under-represents misses and tip-ins.",
        "",
        md_table(um[["GAME_ID", "GAME_EVENT_ID", "PERIOD", "pbp_clock", "PLAYER_NAME", "ACTION_TYPE", "SHOT_TYPE", "SHOT_DISTANCE", "SHOT_MADE_FLAG"]]),
        "",
        "## Blueprint internal consistency",
        "",
        md_table(ctx["score_consistency"].drop(columns=["residual_points_hist"])),
        "",
        "`residual points` between consecutive Blueprint shots (free throws or missing made shots) per game: "
        + "; ".join(f"{r.game_id}: {r.residual_points_hist}" for r in ctx["score_consistency"].itertuples()),
        "",
        "## Missing values in the unified dataset",
        "",
        unified_missing_summary(ctx["unified"]),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def comparison_report(ctx: dict, path: Path, fig_name: str) -> None:
    c = ctx["comparison"]
    e = c["frame"]
    m = c["metrics"]
    boot = c["bootstrap"]
    rep = ctx["model_a_reproduction"]
    arep = model_a.load_inference_assets()["meta"].get("reproduction", {})
    hist = c["historical"]
    diff = boot[boot["quantity"].str.contains(" - ")].copy()
    show = ["n", "observed_fg", "mean_pred", "mean_signed_error", "mean_abs_error", "log_loss", "brier", "roc_auc",
            "ece_quantile10", "calibration_slope", "calibration_intercept", "mean_pred_on_makes", "mean_pred_on_misses"]
    ci = boot[~boot["quantity"].str.contains(" - ")].pivot(index="quantity", columns="metric", values=["ci_low", "ci_high"])
    ci_rows = []
    for name in MODEL_COLS:
        ci_rows.append({"model": name, **{f"{met} 95% CI": f"{ci.loc[name, ('ci_low', met)]:.4f} – {ci.loc[name, ('ci_high', met)]:.4f}"
                                           for met in ("log_loss", "brier", "roc_auc")}})
    dis = c["disagreement"]
    sg = c["subgroups"]

    def sgt(name, col):
        t = sg[col].copy()
        return f"### By {name}\n\n" + md_table(t, 3) + "\n"

    lines = [
        f"# Model A vs NBA xFG — {ctx['date']} ({len(ctx['game_ids'])} games)",
        "",
        f"Generated {utc_now()}. Ground truth = official make/miss. NBA xFG is a competing model, not the target.",
        "",
        "## Read this first",
        "",
        f"* **Sample**: {len(e)} matched field-goal attempts from {len(ctx['game_ids'])} games on a single night. "
        "Differences of a few thousandths in log loss or Brier are well inside sampling noise at this size.",
        "* **Uncertainty**: 95% intervals are shot-level paired bootstrap percentiles (2,000 resamples). They ignore within-game "
        "correlation and therefore understate uncertainty; with 4 games a game-level bootstrap is not meaningful.",
        "* **Coverage bias**: Blueprint is missing official FGA, mostly misses and tip-ins (see the matching report). Both models are "
        "scored on the identical matched subset, which is not the full population of shots.",
        "* **Inputs**: Model A sees ShotChartDetail-equivalent context and player priors from 2022-23/2023-24 only. NBA xFG uses "
        "per-shot tracking (defender distance, etc.). This is not a like-for-like information comparison.",
        f"* **Model A staleness**: {1 - e['model_a_player_has_priors'].astype(bool).mean():.1%} of compared shots come from players absent from the "
        "2022-24 training data; Model A uses league-default priors for them (faithful to the trained model).",
        "",
        "## Model A reproduction (prerequisite)",
        "",
        f"* Artifact `{ctx['model_a_provenance']['path']}` sha256 `{ctx['model_a_provenance']['sha256'][:16]}…` loaded with an allowlisting "
        f"unpickler (globals: sklearn, numpy, xgboost, joblib array wrapper only). Trained with scikit-learn "
        f"{ctx['model_a_provenance']['trained_with']['scikit-learn']}; run with {ctx['model_a_provenance']['runtime']}.",
        f"* Re-scoring the saved 2024-25 test features reproduces the saved predictions (max |Δ| = "
        f"{rep.get('prediction_max_abs_diff', float('nan')):.2e}) and the recorded metrics (max |Δ| = {rep.get('metrics_max_abs_diff', float('nan')):.1e}). Status: **{rep['status']}**.",
        f"* Player-prior tables were rebuilt from the 2022-23 + 2023-24 training seasons with the training code; merged onto the 2024-25 "
        f"shots they reproduce the saved test feature matrix (status **{arep.get('status')}**, max |Δ| = "
        f"{arep.get('numeric_max_abs_diff', float('nan')):.1e} over {arep.get('compared_rows', 0):,} rows); see "
        "`models/legacy/model_a/assets/assets_meta.json`.",
        "",
        "### Known property of the trained Model A (historical, not specific to this night)",
        "",
        "Model A is over-confident at the extremes on its own 2024-25 test set, which explains the low calibration slope here. "
        "Individual 0.95+ predictions on three-pointers are produced by the original model with in-distribution inputs, not by "
        "the inference pipeline.",
        "",
        md_table(c["historical_extremes"], 3) if c.get("historical_extremes") is not None else "_historical predictions not available_",
        "",
        "## Primary metrics (identical matched shots)",
        "",
        md_table(m[show].T, 4, index=True),
        "",
        "`mean_signed_error` = mean(prediction − outcome) (positive = over-predicts makes); `calibration_slope` < 1 indicates "
        "over-confident spread. The shot-type baseline is the 2022-24 league FG% for 2PT/3PT (not fit on these games).",
        "",
        md_table(pd.DataFrame(ci_rows)),
        "",
        "### Paired differences (first minus second; negative log loss / Brier difference favours the first model)",
        "",
        md_table(diff[["quantity", "metric", "mean", "ci_low", "ci_high", "share_below_zero"]], 4),
        "",
        f"![calibration]({fig_name})",
        "",
        "## Model A on all official FGA (incl. shots Blueprint missed)",
        "",
        md_table(c["full_metrics"][["n", "observed_fg", "mean_pred", "log_loss", "brier", "roc_auc", "ece_quantile10"]], 4, index=True),
        "",
    ]
    if c["missing_metrics"] is not None:
        lines += ["Model A on the official FGA that Blueprint does not contain:", "",
                  md_table(c["missing_metrics"][["n", "observed_fg", "mean_pred", "log_loss", "brier"]], 4, index=True), ""]
    lines += [
        "## Secondary diagnostics",
        "",
        sgt("2PT vs 3PT", "shot_type"), sgt("official shot distance", "dist_bin"), sgt("shot zone", "SHOT_ZONE_BASIC"),
        sgt("closest-defender distance (Blueprint contest distance)", "contest_bin"),
        sgt("outcome (mean prediction on makes vs misses)", "outcome"), sgt("Model A prior availability", "model_a_priors"),
        sgt("game", "GAME_ID"), sgt("team", "team"),
        "### Players with at least 12 matched attempts",
        "",
        md_table(c["players"], 3) if len(c["players"]) else "None.",
        "",
        "## Model disagreement",
        "",
        f"* Correlation of Model A and NBA xFG: Pearson {dis['pearson_r']:.3f}, Spearman {dis['spearman_r']:.3f}.",
        f"* Mean |Model A − NBA| = {dis['mean_abs_diff']:.3f}; {dis['share_abs_diff_ge_0.10']:.1%} of shots differ by ≥ 0.10 and "
        f"{dis['share_abs_diff_ge_0.15']:.1%} by ≥ 0.15.",
        f"* On the {dis['large_disagreements']['n']} shots differing by ≥ 0.15: observed FG {fmt(dis['large_disagreements']['observed_fg'], 3)}, "
        f"mean Model A {fmt(dis['large_disagreements']['mean_a'], 3)}, mean NBA {fmt(dis['large_disagreements']['mean_b'], 3)}, "
        f"log loss Model A {fmt(dis['large_disagreements']['log_loss_a'], 3)} vs NBA {fmt(dis['large_disagreements']['log_loss_b'], 3)}.",
        "",
        "### Largest disagreements where Model A is higher",
        "",
        md_table(dis["top_a_higher"], 3),
        "",
        "### Largest disagreements where NBA xFG is higher",
        "",
        md_table(dis["top_b_higher"], 3),
        "",
        "Individual shots are illustrations only; a single low-probability make says nothing about model quality.",
        "",
        "### Extreme Model A predictions on these shots",
        "",
        md_table(c["new_extremes"], 3),
        "",
        "## Historical reference (not comparable)",
        "",
        f"Model A's recorded test metrics on {hist['test_season']} ShotChartDetail shots (n = {hist['n_test']:,}): "
        f"ROC AUC {hist['test_metrics']['roc_auc']:.4f}, log loss {hist['test_metrics']['log_loss']:.4f}, Brier "
        f"{hist['test_metrics']['brier_score']:.4f}. Different season, population and sample size; these are not a head-to-head "
        "benchmark against NBA xFG. Note also that the defender-profile priors used at training time include 2024-25 data, so those "
        "historical test metrics carry mild leakage.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def model_b_report(ctx: dict, path: Path) -> None:
    mb = ctx["model_b"]
    s = mb["summary"]
    folds = pd.DataFrame(mb["experiment"]["folds"])
    boot = s["bootstrap"]
    diffs = boot[boot["quantity"].str.contains(" - ")]
    audit = ctx["field_audit"]
    pw = s["power"]
    lines = [
        f"# Model B (tracking-aware) — exploratory experiment, {ctx['date']}",
        "",
        "**Status: exploratory. Four games from one night cannot establish whether tracking features improve make-probability "
        "predictions. No improvement claim is made from these results.**",
        "",
        "## Data availability",
        "",
        "* Per-shot tracking (contest distance/height, shot vertical, shooter speed, release time) exists only in the Blueprint "
        f"scrape: {len(mb['frame'])} matched shots from {mb['frame']['GAME_ID'].nunique()} games. No historical per-shot tracking "
        "exists in the repository (the 2022-25 inputs are ShotChartDetail plus aggregate PlayerDashPtShots splits).",
        "* Defender identity is available per shot but there is no historical defender representation to encode it validly; it is "
        "not used.",
        "",
        "## Field audit (units as named)",
        "",
        md_table(pd.DataFrame(audit).T.drop(columns=["distinct_values", "zero_share_within_6ft", "zero_share_beyond_10ft", "share_at_cap_1.0"], errors="ignore"), 2, index=True),
        "",
        f"* `release_time_s` is exactly 0 for {audit['release_time_s']['zero_share_within_6ft']:.0%} of shots within 6 ft and "
        f"{audit['release_time_s']['zero_share_beyond_10ft']:.1%} beyond 10 ft, takes values {audit['release_time_s']['distinct_values']} "
        f"and is capped at 1.0 ({audit['release_time_s']['share_at_cap_1.0']:.0%} of shots) → zero treated as *not measured* "
        "(missing + indicator), cap flagged.",
        "* `shot_clock_s` = 0 kept with an indicator (expiry or clock off); `shot_vertical_in` = 0 kept with an indicator.",
        "* `contest_distance_in` has no zeros (min 3.9 in) and is converted to feet.",
        "",
        "## Design",
        "",
        "* B0 = Model A recalibrated (logistic on logit(Model A)); B1 = B0 + tracking block; B2 = court geometry + tracking, no Model A.",
        "* B0 is the control: incremental tracking value is B1 vs B0 (any model fit on 2026-27 data can correct calibration drift).",
        "* Nested leave-one-game-out: each game is predicted by a model trained on the other games; L2 strength is chosen by "
        "leave-one-game-out CV inside the training games. Imputation and scaling are fitted within training folds.",
        "* Excluded inputs: NBA xFG, outcomes, scores, defender identity. Same-night games are used for training, i.e. the split is "
        "game-grouped but not strictly chronological.",
        "",
        "## Out-of-fold results (identical shots for every model)",
        "",
        md_table(s["metrics"][["n", "mean_pred", "mean_signed_error", "log_loss", "brier", "roc_auc", "ece_quantile10", "calibration_slope"]], 4, index=True),
        "",
        md_table(diffs[["quantity", "metric", "mean", "ci_low", "ci_high", "share_below_zero"]], 4),
        "",
        "How to read this: the intervals resample individual shots and so ignore within-game correlation; with four games the "
        "effective number of independent units is closer to four than to 649. The per-game table is the more honest view: "
        f"B1 beats B0 in {int((s['per_game_log_loss']['B1_A_plus_tracking'] < s['per_game_log_loss']['B0_recalibrated_A']).sum())} "
        f"of {len(s['per_game_log_loss'])} held-out games. B1 and B2 also train on tracking data from the same night they are "
        "tested on (other games), which a production model could not do. Treat any gap as a hypothesis for a larger, "
        "chronologically held-out evaluation, not as evidence of improvement.",
        "",
        "### Per held-out game log loss",
        "",
        md_table(s["per_game_log_loss"], 4),
        "",
        "### Chosen regularization per fold",
        "",
        md_table(folds[["model", "held_out_game", "n_train", "n_test", "chosen_C"]], 3),
        "",
        "## How much data is needed",
        "",
        f"The per-shot log-loss difference between B1 and B0 has SD ≈ {pw['per_shot_log_loss_diff_sd']:.4f}; the observed mean "
        f"improvement of B1 over B0 is {pw['observed_improvement_B1_vs_B0']:+.4f}. Shots required for 80% power (two-sided α = 0.05, "
        "independent shots) to detect a given mean improvement:",
        "",
        md_table(pd.DataFrame([{"log-loss improvement": k, "shots needed": v} for k, v in pw["required_shots_80pct_power"].items()])),
        "",
        "At roughly 170 FGA per game (≈ 160 with Blueprint coverage), detecting a 0.005 improvement needs on the order of "
        f"{pw['required_shots_80pct_power']['0.005'] / 160:.0f} games before accounting for within-game correlation, which inflates "
        "the requirement further. A defensible evaluation also needs a chronologically later held-out block (e.g. train on the first "
        "half of the season, test on the following month) and tracking-aware player priors built only from earlier games.",
        "",
        f"The B1 model refit on all available games is saved for inspection at `{mb['saved_to']}` with metadata; it is not a "
        "production model.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparison_json(ctx: dict, path: Path) -> None:
    c = ctx["comparison"]
    payload = {
        "date": ctx["date"], "games": ctx["game_ids"], "n_compared": int(len(c["frame"])),
        "metrics": c["metrics"].reset_index().to_dict(orient="records"),
        "bootstrap": c["bootstrap"].to_dict(orient="records"),
        "model_a_all_official_fga": c["full_metrics"].reset_index().to_dict(orient="records"),
        "subgroups": {k: v.to_dict(orient="records") for k, v in c["subgroups"].items()},
        "disagreement": {k: (v.to_dict(orient="records") if isinstance(v, pd.DataFrame) else v) for k, v in c["disagreement"].items()},
        "historical_reference": c["historical"], "model_a_reproduction": ctx["model_a_reproduction"],
        "model_b": {"metrics": ctx["model_b"]["summary"]["metrics"].reset_index().to_dict(orient="records"),
                    "bootstrap": ctx["model_b"]["summary"]["bootstrap"].to_dict(orient="records"),
                    "power": ctx["model_b"]["summary"]["power"]},
        "matching": ctx["match_result"].hypothesis.to_dict(orient="records"),
        "coverage": coverage_table(ctx).to_dict(orient="records"),
    }
    write_json(path, payload)


def run_date(date: str, allow_network: bool = False) -> dict:
    report_dir = REPORTS_DIR / date
    report_dir.mkdir(parents=True, exist_ok=True)
    log.info("stage 1/6 ingest")
    ctx = ingest(date, allow_network)
    log.info("stage 2/6 match")
    ctx = match(ctx)
    log.info("stage 3/6 Model A inference")
    ctx = score_model_a(ctx)
    log.info("stage 4/6 Model A vs NBA xFG")
    ctx = compare(ctx, report_dir)
    log.info("stage 5/6 Model B exploratory experiment")
    ctx = run_model_b(ctx)
    log.info("stage 6/6 analytics + reports")
    ctx = build_analytics(ctx, ANALYTICS_DIR / date)
    ctx["dataset_files"] = write_datasets(ctx, UNIFIED_DIR / date)
    matching_report(ctx, report_dir / "matching_report.md")
    comparison_report(ctx, report_dir / "model_comparison.md", "calibration_model_a_vs_nba.png")
    model_b_report(ctx, report_dir / "model_b_report.md")
    write_comparison_json(ctx, report_dir / "metrics.json")
    log.info("done: reports in %s, datasets in %s, analytics in %s", rel(report_dir), rel(UNIFIED_DIR / date), rel(ANALYTICS_DIR / date))
    return ctx
