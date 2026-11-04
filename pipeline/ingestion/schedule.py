"""Validation of cached schedule-day discovery files (e.g. data/raw/nba_listings/2026-27/2026-10-28.json).

The Blueprint scraper discovers games by regex-matching game links on nba.com/games?date=D.
Such pages can link games from other dates (previews, recaps), so discovered games are
checked for internal conflicts and, where a cached game page exists, against its official date.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from pipeline.config import GAMES_LISTING_DIR, GAME_PAGES_DIR
from pipeline.ingestion.nba_pages import cache_paths, load_game_page, parse_game_page
from pipeline.utils import require


def load_schedule_day(day: str, schedule_dir: Path = GAMES_LISTING_DIR) -> pd.DataFrame:
    path = schedule_dir / f"{day}.json"
    require(path.exists(), f"schedule cache not found: {path}")
    games = json.loads(path.read_text(encoding="utf-8"))
    df = pd.DataFrame(games)
    require({"game_id", "date", "away", "home"} <= set(df.columns), f"{path} missing game_id/date/away/home")
    df["game_id"] = df["game_id"].astype(str)
    return df


def validate_schedule_day(day: str, schedule_dir: Path = GAMES_LISTING_DIR,
                          pages_dir: Path = GAME_PAGES_DIR) -> pd.DataFrame:
    """One row per discovered game with a `status` of verified / conflict / unverified / wrong_date."""
    df = load_schedule_day(day, schedule_dir)
    headers = {}
    for gid in df["game_id"]:
        if cache_paths(gid, pages_dir)[0].exists():
            headers[gid] = parse_game_page(load_game_page(gid, pages_dir), gid).header
    # Games proven to be on another date do not create team conflicts for the remaining games.
    on_day = df[[headers.get(g, {}).get("game_date_et", day) == day for g in df["game_id"]]]
    team_counts = Counter(list(on_day["away"]) + list(on_day["home"]))
    rows = []
    for g in df.itertuples():
        flags = []
        if g.date != day:
            flags.append(f"cached date {g.date} != requested {day}")
        dup_teams = [t for t in (g.away, g.home) if team_counts[t] > 1]
        if dup_teams:
            flags.append(f"team(s) {dup_teams} appear in more than one game on {day}")
        official_date = None
        if g.game_id in headers:
            hdr = headers[g.game_id]
            official_date = hdr["game_date_et"]
            if (hdr["away_tricode"], hdr["home_tricode"]) != (g.away, g.home):
                flags.append(f"teams differ from game page: {hdr['away_tricode']}@{hdr['home_tricode']}")
        if official_date is not None and official_date != day:
            status = "wrong_date"
            flags.append(f"official game date is {official_date}")
        elif official_date == day and not any("differ" in f for f in flags):
            status = "verified"
        elif dup_teams:
            status = "conflict"
        else:
            status = "unverified"
        rows.append({"game_id": g.game_id, "away": g.away, "home": g.home, "cached_date": g.date,
                     "official_date_et": official_date, "status": status, "flags": "; ".join(flags)})
    return pd.DataFrame(rows)
