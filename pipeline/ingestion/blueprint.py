"""Loading, normalization and quality auditing of NBA Blueprint shot records.

Per-game CSVs (`games/<game_id>.csv`) are the primary input; the raw JSON captured by the
scraper (`raw/<game_id>.json`) is used to cross-check every row and to recover team IDs.
`all_shots.csv` is a derived file and is not read.

Normalized columns (one row per Blueprint shot):
  game_id, game_date, date_source, ngss_event_id, period, clock_raw, clock_remaining_s,
  elapsed_s, shot_clock_s, shooter_id, shooter_name, shooter_team, shooter_team_id,
  defender_id, defender_name, defender_team_id, shot_value, made, nba_xfg, nba_xpts_fg,
  bp_shot_distance_ft, contest_distance_in, contest_height_in, shot_vertical_in,
  shooter_speed_mph, release_time_s, bp_x, bp_y, home_score_after, away_score_after,
  featured, source_url, source_file, source_sha256, source_row
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.matching import clock as clk
from pipeline.config import BLUEPRINT_GAMES_DIR, BLUEPRINT_RAW_DIR
from pipeline.utils import get_logger, rel, require, require_columns, sha256_file

log = get_logger("blueprint")

BLUEPRINT_CSV_COLUMNS = [
    "game_id", "date", "ngss_event_id", "period", "clock", "shot_clock", "shooter_id", "shooter", "team",
    "defender_id", "defender", "shot_type", "made", "xfg", "xpts_fg_only", "shot_distance_ft",
    "contest_distance_in", "contest_height_in", "shot_vertical_in", "shooter_speed_mph", "release_time_s",
    "shot_x", "shot_y", "home_score", "away_score", "featured", "source_url",
]
TRACKING_COLUMNS = ["contest_distance_in", "contest_height_in", "shot_vertical_in", "shooter_speed_mph", "release_time_s"]


def _parse_bool(s: pd.Series, name: str) -> pd.Series:
    mapping = {"true": True, "false": False, "1": True, "0": False}
    out = s.astype(str).str.strip().str.lower().map(mapping)
    require(out.notna().all(), f"{name}: non-boolean values {sorted(s[out.isna()].astype(str).unique())[:5]}")
    return out.astype(bool)


def load_blueprint_game(csv_path: Path, raw_dir: Path = BLUEPRINT_RAW_DIR) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"game_id": str, "date": str, "clock": str})
    require_columns(df, BLUEPRINT_CSV_COLUMNS, rel(csv_path))
    require(len(df) > 0, f"{rel(csv_path)} is empty")
    gid = csv_path.stem
    df["game_id"] = df["game_id"].str.zfill(10)
    require((df["game_id"] == gid).all(), f"{rel(csv_path)} contains rows for other games")
    require(not df["ngss_event_id"].duplicated().any(), f"{gid}: duplicate ngss_event_id")
    require(df["shot_type"].isin(["2pt", "3pt"]).all(), f"{gid}: shot_type outside {{2pt,3pt}}")
    require(df["xfg"].between(0, 1).all(), f"{gid}: xfg outside [0, 1]")

    out = pd.DataFrame({
        "game_id": df["game_id"],
        "game_date": df["date"].where(df["date"].notna() & (df["date"].astype(str).str.len() > 0)),
        "ngss_event_id": df["ngss_event_id"].astype(int),
        "period": df["period"].astype(int),
        "clock_raw": df["clock"],
        "shot_clock_s": pd.to_numeric(df["shot_clock"], errors="coerce"),
        "shooter_id": df["shooter_id"].astype(int),
        "shooter_name": df["shooter"],
        "shooter_team": df["team"],
        "defender_id": pd.to_numeric(df["defender_id"], errors="coerce").astype("Int64"),
        "defender_name": df["defender"],
        "shot_value": np.where(df["shot_type"] == "3pt", 3, 2),
        "made": _parse_bool(df["made"], f"{gid}.made").astype(int),
        "nba_xfg": df["xfg"].astype(float),
        "bp_xpts_fg_only": df["xpts_fg_only"].astype(float),
        "bp_shot_distance_ft": df["shot_distance_ft"].astype(float),
        **{c: pd.to_numeric(df[c], errors="coerce") for c in TRACKING_COLUMNS},
        "bp_x": df["shot_x"].astype(int),
        "bp_y": df["shot_y"].astype(int),
        "home_score_after": df["home_score"].astype(int),
        "away_score_after": df["away_score"].astype(int),
        "featured": _parse_bool(df["featured"], f"{gid}.featured"),
        "source_url": df["source_url"],
    })
    out["date_source"] = np.where(out["game_date"].notna(), "blueprint_csv", None)
    out["clock_remaining_s"] = clk.parse_clock_series(out["clock_raw"])
    out["elapsed_s"] = clk.elapsed_series(out["period"], out["clock_remaining_s"])
    out["nba_xpts_fg"] = out["nba_xfg"] * out["shot_value"]
    require(np.allclose(out["nba_xpts_fg"], out["bp_xpts_fg_only"], atol=1e-3),
            f"{gid}: xpts_fg_only != xfg * shot value")
    out["source_file"] = rel(csv_path)
    out["source_sha256"] = sha256_file(csv_path)
    out["source_row"] = np.arange(len(out)) + 2  # 1-based line number in the CSV (header = line 1)

    raw_path = raw_dir / f"{gid}.json"
    out["shooter_team_id"] = pd.NA
    out["defender_team_id"] = pd.NA
    if raw_path.exists():
        out = _cross_check_raw(out, raw_path)
    else:
        log.warning("%s: raw JSON missing (%s); team IDs unavailable", gid, rel(raw_path))
    return out


def _cross_check_raw(out: pd.DataFrame, raw_path: Path) -> pd.DataFrame:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    rows = []
    for s in raw:
        sh, de, at = s.get("shooter") or {}, s.get("defender") or {}, s.get("attributes") or {}
        rows.append({
            "ngss_event_id": s.get("ngssEventId"), "r_period": s.get("period"), "r_clock": s.get("gameClockTime"),
            "r_shooter": sh.get("playerId"), "r_made": bool(s.get("success")), "r_xfg": s.get("expFieldGoalPct"),
            "r_result": at.get("result"), "shooter_team_id_raw": sh.get("teamId"), "defender_team_id_raw": de.get("teamId"),
        })
    r = pd.DataFrame(rows)
    gid = out["game_id"].iloc[0]
    require(not r["ngss_event_id"].duplicated().any(), f"{gid}: duplicate ngssEventId in raw JSON")
    m = out.merge(r, on="ngss_event_id", how="left", validate="one_to_one")
    require(m["r_period"].notna().all(), f"{gid}: {int(m['r_period'].isna().sum())} CSV rows not found in raw JSON")
    require(len(r) == len(out), f"{gid}: raw JSON has {len(r)} shots, CSV has {len(out)}")
    agree = (
        (m["r_period"] == m["period"]) & (m["r_clock"] == m["clock_raw"]) & (m["r_shooter"] == m["shooter_id"])
        & (m["r_made"].astype(int) == m["made"]) & np.isclose(m["r_xfg"], m["nba_xfg"])
        & (m["r_result"].map({"MADE": 1, "MISS": 0}) == m["made"])
    )
    require(agree.all(), f"{gid}: {int((~agree).sum())} CSV rows disagree with raw JSON")
    m["shooter_team_id"] = m["shooter_team_id_raw"].astype("Int64")
    m["defender_team_id"] = m["defender_team_id_raw"].astype("Int64")
    return m.drop(columns=[c for c in m.columns if c.startswith("r_") or (c.endswith("_raw") and c != "clock_raw")])


def load_blueprint(games_dir: Path = BLUEPRINT_GAMES_DIR, raw_dir: Path = BLUEPRINT_RAW_DIR,
                   game_ids: list[str] | None = None) -> pd.DataFrame:
    paths = sorted(games_dir.glob("*.csv"))
    if game_ids is not None:
        paths = [p for p in paths if p.stem in set(game_ids)]
    require(len(paths) > 0, f"no Blueprint game CSVs found in {rel(games_dir)}")
    frames = [load_blueprint_game(p, raw_dir) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df["shot_uid"] = df["game_id"] + "_" + df["ngss_event_id"].astype(str).str.zfill(4)
    log.info("loaded %d Blueprint shots from %d games", len(df), len(paths))
    return df


def fill_dates(bp: pd.DataFrame, official_dates: dict[str, str]) -> pd.DataFrame:
    """Fill or verify game dates from official game headers; conflicting dates raise."""
    out = bp.copy()
    for gid, day in official_dates.items():
        mask = out["game_id"] == gid
        existing = out.loc[mask, "game_date"].dropna().unique()
        require(all(e == day for e in existing), f"{gid}: Blueprint date {existing} != official {day}")
        missing = mask & out["game_date"].isna()
        out.loc[missing, "game_date"] = day
        out.loc[missing, "date_source"] = "nba_game_page_header"
    return out


def audit_fields(bp: pd.DataFrame) -> dict:
    """Distribution / suspicious-value audit of tracking fields (units in column names)."""
    res = {}
    for c in TRACKING_COLUMNS + ["shot_clock_s", "bp_shot_distance_ft"]:
        s = bp[c]
        res[c] = {
            "n": int(s.notna().sum()), "missing": int(s.isna().sum()), "zeros": int((s == 0).sum()),
            "min": float(s.min()), "p01": float(s.quantile(0.01)), "median": float(s.median()),
            "p99": float(s.quantile(0.99)), "max": float(s.max()),
        }
    rt = bp["release_time_s"]
    near = bp["bp_shot_distance_ft"] <= 6
    res["release_time_s"]["zero_share_within_6ft"] = float((rt[near] == 0).mean()) if near.any() else None
    res["release_time_s"]["zero_share_beyond_10ft"] = float((rt[bp["bp_shot_distance_ft"] > 10] == 0).mean())
    res["release_time_s"]["share_at_cap_1.0"] = float((rt == 1.0).mean())
    res["release_time_s"]["distinct_values"] = sorted(map(float, rt.dropna().unique()))
    return res


def score_consistency(bp: pd.DataFrame, home_team: dict[str, str]) -> pd.DataFrame:
    """Check that the running score attached to each shot is consistent with `made` and shot value.

    The score fields are post-shot scores. For a made shot, the shooter's team score must rise by at
    least the shot value since the previous Blueprint shot; the remainder (free throws or shots absent
    from Blueprint) is reported. Outcomes are never inferred from scores.
    """
    rows = []
    for gid, g in bp.sort_values(["game_id", "elapsed_s", "ngss_event_id"]).groupby("game_id"):
        is_home = g["shooter_team"] == home_team[gid]
        dh = g["home_score_after"].diff().fillna(g["home_score_after"])
        da = g["away_score_after"].diff().fillna(g["away_score_after"])
        d_team = np.where(is_home, dh, da)
        residual = d_team - g["made"] * g["shot_value"]
        rows.append({
            "game_id": gid, "n_shots": len(g),
            "event_id_order_is_chronological": bool((g["ngss_event_id"].diff().dropna() > 0).all()),
            "negative_score_steps": int(((dh < 0) | (da < 0)).sum()),
            "made_with_insufficient_score_rise": int(((g["made"] == 1) & (residual < 0)).sum()),
            "residual_points_hist": pd.Series(residual).value_counts().sort_index().to_dict(),
            "last_home_score": int(g["home_score_after"].iloc[-1]), "last_away_score": int(g["away_score_after"].iloc[-1]),
        })
    return pd.DataFrame(rows)
