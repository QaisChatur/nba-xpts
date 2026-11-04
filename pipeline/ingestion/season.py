"""Season-wide ingestion: discovery, date verification, Blueprint + official data, coverage manifest.

Per game the manifest records one of these statuses:
  verified            final game; official date verified from the game page; play-by-play FGA/FGM/3PA
                      equal the box score; Blueprint shots extracted and matched one-to-one.
  not_final           the game page does not report a final game (status != 3); retried next run.
  blueprint_empty     final game but the Blueprint page published no shot records.
  official_mismatch   play-by-play disagrees with the box score (not used for training).
  pending_fetch       required pages are not cached and network access was not allowed.
  error               any other failure (message stored).
Only `verified` games are used downstream. Listing dates are never trusted: the nba.com/games listing
can contain games from neighbouring dates, so every game is dated from its official header (gameEt).

Requests are opt-in (`allow_network`), cached, sequential, delayed, and the run stops at the first
HTTP 403/429 (no retries, no anti-bot workarounds).
"""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import BLUEPRINT_DIR, GAMES_LISTING_DIR, GAME_PAGES_DIR, REPORTS_DIR, SEASON, SEASON_MANIFEST
from pipeline.ingestion import blueprint, nba_pages
from pipeline.ingestion.blueprint_scrape import AccessRefused, fetch_blueprint_game, fetch_games_listing, paths
from pipeline.matching import match_shots
from pipeline.utils import get_logger, rel, utc_now, write_json

log = get_logger("season")

SEASON_SEARCH_START = "2026-10-18"
FINAL_STATUS = 3
TRACKING_FIELDS = ["shot_clock_s", "defender_id", "contest_distance_in", "contest_height_in", "shot_vertical_in",
                   "shooter_speed_mph", "release_time_s", "bp_shot_distance_ft"]


def daterange(start: str, end: str) -> list[str]:
    d0, d1 = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    return [(d0 + dt.timedelta(days=i)).isoformat() for i in range((d1 - d0).days + 1)]


def load_manifest(path: Path = SEASON_MANIFEST) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"season": SEASON, "days": {}, "games": {}}


def save_manifest(manifest: dict, path: Path = SEASON_MANIFEST) -> None:
    games = manifest["games"].values()
    manifest["updated_utc"] = utc_now()
    manifest["summary"] = {
        "games_seen": len(manifest["games"]),
        **{f"status_{s}": sum(g.get("status") == s for g in games)
           for s in ("verified", "not_final", "blueprint_empty", "official_mismatch", "pending_fetch", "error")},
        "verified_official_fga": sum(g.get("official_fga", 0) for g in games if g.get("status") == "verified"),
        "verified_matched": sum(g.get("matched", 0) for g in games if g.get("status") == "verified"),
    }
    write_json(path, manifest)


# --------------------------------------------------------------------------- per-game verification


def verify_game(game_id: str, listing_date: str | None, pages_dir: Path = GAME_PAGES_DIR,
                bp_dir: Path = BLUEPRINT_DIR) -> dict:
    """Offline verification of one game from cached files. Returns its manifest record."""
    page = nba_pages.parse_game_page(nba_pages.load_game_page(game_id, pages_dir), game_id)
    h = page.header
    rec = {
        "game_id": game_id, "official_date": h["game_date_et"], "listing_date": listing_date,
        "listing_date_correct": listing_date == h["game_date_et"] if listing_date else None,
        "matchup": f"{h['away_tricode']}@{h['home_tricode']}", "away": h["away_tricode"], "home": h["home_tricode"],
        "game_status": h["game_status"], "game_status_text": h["game_status_text"],
        "final_score": f"{h['away_score']}-{h['home_score']}", "verified_utc": utc_now(),
    }
    if h["game_status"] != FINAL_STATUS:
        return {**rec, "status": "not_final"}

    official = nba_pages.pbp_shots(page)
    official["clock_remaining_s"] = official["pbp_clock_remaining_s"]
    box = nba_pages.box_validation(page, official)
    rec.update(
        official_fga=int(box["box_fga"].sum()), official_fgm=int(box["box_fgm"].sum()),
        official_fg3a=int(box["box_fg3a"].sum()), pbp_fga=int(len(official)),
        pbp_matches_box=bool(box["pbp_matches_box"].all()),
    )
    if not rec["pbp_matches_box"]:
        return {**rec, "status": "official_mismatch"}

    p = paths(game_id, bp_dir)
    if not p["csv"].exists():
        meta = json.loads(p["meta"].read_text()) if p["meta"].exists() else {}
        return {**rec, "status": "blueprint_empty", "blueprint_records_seen": meta.get("records_kept", 0),
                "blueprint_rejected": meta.get("records_rejected", 0)}

    bp = blueprint.load_blueprint_game(p["csv"], p["raw"].parent)
    bp["shot_uid"] = bp["game_id"] + "_" + bp["ngss_event_id"].astype(str).str.zfill(4)
    bp = blueprint.fill_dates(bp, {game_id: h["game_date_et"]})
    res = match_shots(bp, official)
    st = res.bp_status
    matched = st[st["match_status"] == "matched"].merge(
        official[["GAME_EVENT_ID", "SHOT_MADE_FLAG", "SHOT_TYPE"]],
        left_on="official_event_id", right_on="GAME_EVENT_ID", how="left")
    is3 = official["SHOT_TYPE"].astype(str).str.startswith("3")
    m3 = matched["SHOT_TYPE"].astype(str).str.startswith("3")

    def pct(a, b):
        return round(100.0 * a / b, 2) if b else None

    meta = json.loads(p["meta"].read_text()) if p["meta"].exists() else {}
    rec.update(
        status="verified",
        blueprint_extracted=int(len(bp)), blueprint_rejected=int(meta.get("records_rejected", 0)),
        blueprint_html_sha256=meta.get("html_sha256"), blueprint_fetched_utc=meta.get("fetched_utc"),
        matched=int(len(matched)),
        matched_by_fallback=int(st["match_method"].astype(str).str.startswith("attributes").sum()),
        ambiguous=int((st["match_status"] == "ambiguous").sum()),
        rejected_id_conflict=int((st["match_status"] == "rejected_id_conflict").sum()),
        unmatched_blueprint=int((st["match_status"] == "unmatched").sum()),
        official_without_blueprint=int(len(official) - len(matched)),
        coverage_pct=pct(len(matched), len(official)),
        made_coverage_pct=pct(int(matched["SHOT_MADE_FLAG"].sum()), int(official["SHOT_MADE_FLAG"].sum())),
        missed_coverage_pct=pct(int((matched["SHOT_MADE_FLAG"] == 0).sum()), int((official["SHOT_MADE_FLAG"] == 0).sum())),
        fg2_coverage_pct=pct(int((~m3).sum()), int((~is3).sum())),
        fg3_coverage_pct=pct(int(m3.sum()), int(is3.sum())),
        value_disagreements=int((~st.loc[st["match_status"] == "matched", "value_agree"].astype(bool)).sum())
        if "value_agree" in st else None,
        missing_fields={c: int(bp[c].isna().sum()) for c in TRACKING_FIELDS if c in bp},
    )
    return rec


# --------------------------------------------------------------------------- orchestration


def ingest_season(start: str = SEASON_SEARCH_START, end: str | None = None, allow_network: bool = False,
                  delay_s: float = 3.5, refresh_recent_days: int = 2, max_games: int | None = None,
                  manifest_path: Path = SEASON_MANIFEST) -> dict:
    """Discover, fetch and verify every completed regular-season game between start and end (ET dates).

    Incremental: verified games are skipped; listings for the last `refresh_recent_days` days and
    pages of non-final games are re-fetched.
    """
    end = end or (dt.date.today() - dt.timedelta(days=1)).isoformat()
    manifest = load_manifest(manifest_path)
    manifest["search_range"] = [start, end]
    recent_cutoff = (dt.date.fromisoformat(end) - dt.timedelta(days=refresh_recent_days - 1)).isoformat()
    requests_made = 0

    def pause():
        nonlocal requests_made
        requests_made += 1
        if allow_network:
            time.sleep(delay_s)

    candidates: dict[str, dict] = {}
    try:
        for day in daterange(start, end):
            cached = (GAMES_LISTING_DIR / f"{day}.json").exists()
            refresh = allow_network and day >= recent_cutoff and manifest["days"].get(day, {}).get("fetched_on") != end
            try:
                games = fetch_games_listing(day, allow_network=allow_network, refresh=refresh)
                if refresh or not cached:
                    pause()
                manifest["days"][day] = {"listing_links": len(games), "fetched_on": end if (refresh or not cached)
                                         else manifest["days"].get(day, {}).get("fetched_on")}
            except FileNotFoundError:
                manifest["days"][day] = {"listing_links": None, "note": "not cached (network not allowed)"}
                continue
            for g in games:
                candidates.setdefault(g["game_id"], g)

        todo = []
        for gid, g in sorted(candidates.items()):
            rec = manifest["games"].get(gid)
            if rec and rec.get("status") == "verified" and paths(gid)["csv"].exists():
                continue
            todo.append(g)
        log.info("%d candidate games, %d to process", len(candidates), len(todo))
        if max_games is not None:
            todo = todo[:max_games]

        for i, g in enumerate(todo):
            gid = g["game_id"]
            prev = manifest["games"].get(gid, {})
            try:
                page_cached = nba_pages.cache_paths(gid)[0].exists()
                refresh_page = allow_network and prev.get("status") == "not_final"
                nba_pages.fetch_game_page(gid, g["away"], g["home"], allow_network=allow_network, refresh=refresh_page)
                if refresh_page or not page_cached:
                    pause()
                header = nba_pages.parse_game_page(nba_pages.load_game_page(gid), gid).header
                if header["game_status"] == FINAL_STATUS:
                    bp_cached = paths(gid)["csv"].exists() or paths(gid)["meta"].exists()
                    if not bp_cached:
                        fetch_blueprint_game(gid, g["away"], g["home"], header["game_date_et"], allow_network=allow_network)
                        pause()
                rec = verify_game(gid, g.get("date"))
            except AccessRefused as exc:
                manifest["games"][gid] = {**prev, "game_id": gid, "status": "error", "error": str(exc), "verified_utc": utc_now()}
                manifest["stopped"] = {"reason": f"access refused: {exc}", "utc": utc_now()}
                log.error("access refused (%s); stopping without retrying", exc)
                break
            except FileNotFoundError as exc:
                rec = {"game_id": gid, "listing_date": g.get("date"), "status": "pending_fetch", "error": str(exc)}
            except Exception as exc:  # recorded per game; other games continue
                log.exception("game %s failed", gid)
                rec = {"game_id": gid, "listing_date": g.get("date"), "status": "error", "error": f"{type(exc).__name__}: {exc}"}
            manifest["games"][gid] = rec
            save_manifest(manifest, manifest_path)
            log.info("[%d/%d] %s %s %s coverage=%s", i + 1, len(todo), gid, rec.get("official_date"),
                     rec["status"], rec.get("coverage_pct"))
        else:
            manifest.pop("stopped", None)
    finally:
        manifest["last_run"] = {"utc": utc_now(), "allow_network": allow_network, "requests": requests_made}
        save_manifest(manifest, manifest_path)
    return manifest


def reverify_all(manifest_path: Path = SEASON_MANIFEST) -> dict:
    """Re-run offline verification for every game in the manifest (e.g. after a matching change)."""
    manifest = load_manifest(manifest_path)
    for gid, rec in list(manifest["games"].items()):
        if not nba_pages.cache_paths(gid)[0].exists():
            continue
        try:
            manifest["games"][gid] = verify_game(gid, rec.get("listing_date"))
        except Exception as exc:
            manifest["games"][gid] = {**rec, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
    save_manifest(manifest, manifest_path)
    return manifest


# --------------------------------------------------------------------------- reporting


def coverage_frame(manifest: dict) -> pd.DataFrame:
    rows = []
    for rec in manifest["games"].values():
        r = {k: v for k, v in rec.items() if k != "missing_fields"}
        for k, v in (rec.get("missing_fields") or {}).items():
            r[f"missing_{k}"] = v
        rows.append(r)
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["official_date", "game_id"], na_position="last").reset_index(drop=True)
    return df


def write_coverage_report(manifest: dict, out_dir: Path = REPORTS_DIR / "season") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = coverage_frame(manifest)
    df.to_csv(out_dir / "season_coverage.csv", index=False)
    v = df[df["status"] == "verified"] if len(df) else df
    lines = [f"# {SEASON} Blueprint coverage report", "",
             f"Generated {utc_now()} from `{rel(SEASON_MANIFEST)}`.", ""]
    lines += ["## Game status", "", "| Status | Games |", "|---|---|"]
    for s, n in df["status"].value_counts().items():
        lines.append(f"| {s} | {n} |")
    if len(v):
        tot_off, tot_m = int(v["official_fga"].sum()), int(v["matched"].sum())
        made_off = int(v["official_fgm"].sum())
        lines += ["", "## Verified games", "",
                  f"* Games: {len(v)} ({v['official_date'].min()} to {v['official_date'].max()})",
                  f"* Official FGA: {tot_off:,}; Blueprint records extracted: {int(v['blueprint_extracted'].sum()):,}; "
                  f"matched one-to-one: {tot_m:,} ({100 * tot_m / tot_off:.2f}% of official FGA)",
                  f"* Official FGA without a Blueprint record: {int(v['official_without_blueprint'].sum()):,}",
                  f"* Blueprint records not matched: {int(v['unmatched_blueprint'].sum())} unmatched, "
                  f"{int(v['ambiguous'].sum())} ambiguous, {int(v['rejected_id_conflict'].sum())} ID conflicts",
                  f"* Matched by attribute fallback (event ID not corroborated): {int(v['matched_by_fallback'].sum())}",
                  f"* Shot-value (2 vs 3) disagreements among matches: {int(v['value_disagreements'].fillna(0).sum())}",
                  f"* Extraction rejections (shot-shaped records not kept): {int(v['blueprint_rejected'].fillna(0).sum())}",
                  f"* Per-game coverage: min {v['coverage_pct'].min():.1f}%, median {v['coverage_pct'].median():.1f}%, "
                  f"max {v['coverage_pct'].max():.1f}%; games at 100%: {int((v['coverage_pct'] >= 100).sum())}",
                  f"* Official makes covered: {int(round((v['made_coverage_pct'] * v['official_fgm']).sum() / 100)):,} of {made_off:,}", ""]
        miss_cols = [c for c in v.columns if c.startswith("missing_")]
        if miss_cols:
            lines += ["### Missing Blueprint fields (verified games)", "", "| Field | Missing rows |", "|---|---|"]
            lines += [f"| {c.removeprefix('missing_')} | {int(v[c].fillna(0).sum())} |" for c in miss_cols]
            lines.append("")
        cols = ["game_id", "official_date", "matchup", "official_fga", "blueprint_extracted", "matched",
                "coverage_pct", "made_coverage_pct", "missed_coverage_pct", "fg2_coverage_pct", "fg3_coverage_pct",
                "ambiguous", "unmatched_blueprint"]
        lines += ["## Per-game coverage", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for _, r in v[cols].iterrows():
            lines.append("| " + " | ".join("" if pd.isna(x) else str(x) for x in r) + " |")
    other = df[df["status"] != "verified"] if len(df) else df
    if len(other):
        lines += ["", "## Games not verified", "", "| game_id | official_date | listing_date | status | detail |", "|---|---|---|---|---|"]
        for _, r in other.iterrows():
            detail = r.get("error") if isinstance(r.get("error"), str) else r.get("game_status_text", "")
            lines.append(f"| {r['game_id']} | {r.get('official_date', '')} | {r.get('listing_date', '')} | {r['status']} | {detail} |")
    wrong = df[df.get("listing_date_correct") == False] if "listing_date_correct" in df else df.iloc[:0]  # noqa: E712
    lines += ["", f"Listing pages linked {len(wrong)} game(s) under a date other than their official date; "
              "they were re-dated from the official game header."]
    if manifest.get("stopped"):
        lines += ["", f"**Run stopped:** {manifest['stopped']['reason']}"]
    path = out_dir / "COVERAGE_REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
