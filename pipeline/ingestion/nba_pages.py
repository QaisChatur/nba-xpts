"""NBA.com game pages: cached fetch + parsing of the embedded box score and play-by-play.

The public game page (https://www.nba.com/game/<away>-vs-<home>-<game_id>/box-score)
embeds a Next.js `__NEXT_DATA__` payload containing:
  * `game`       – header, final score, team and player box-score statistics,
  * `playByPlay` – the official play-by-play (actionNumber, clock, personId, xLegacy/yLegacy,
                   shotResult, subType = ShotChartDetail ACTION_TYPE vocabulary, ...).

stats.nba.com (ShotChartDetail) and cdn.nba.com (liveData) refused requests from this
environment (timeout / HTTP 403) during the audit, so this page is the source of the
ShotChartDetail-compatible shot table for 2026-27. Requests are opt-in (`allow_network`),
cached, one per game, rate-limited, and never retried aggressively.
"""

from __future__ import annotations

import gzip
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.matching import clock as clk
from pipeline.matching import court
from pipeline.config import GAME_PAGES_DIR, SEASON
from pipeline.utils import DataValidationError, get_logger, require, sha256_bytes, utc_now, write_json

log = get_logger("nba_pages")

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"
NEXT_DATA_RE = re.compile(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.S)


def page_url(game_id: str, away: str, home: str) -> str:
    return f"https://www.nba.com/game/{away.lower()}-vs-{home.lower()}-{game_id}/box-score"


def cache_paths(game_id: str, cache_dir: Path = GAME_PAGES_DIR) -> tuple[Path, Path]:
    return cache_dir / f"{game_id}.next_data.json.gz", cache_dir / f"{game_id}.meta.json"


def fetch_game_page(
    game_id: str, away: str, home: str, cache_dir: Path = GAME_PAGES_DIR,
    allow_network: bool = False, refresh: bool = False, timeout: int = 40,
) -> Path:
    """Download (once) and cache the page's __NEXT_DATA__ JSON. Returns the cached payload path."""
    payload_path, meta_path = cache_paths(game_id, cache_dir)
    if payload_path.exists() and not refresh:
        return payload_path
    if not allow_network:
        raise FileNotFoundError(
            f"no cached game page for {game_id} at {payload_path}; rerun with --allow-network to fetch it once"
        )
    url = page_url(game_id, away, home)
    log.info("fetching %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html", "Accept-Encoding": "gzip"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            if resp.headers.get("Content-Encoding") == "gzip":
                body = gzip.decompress(body)
            status = resp.status
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"NBA.com returned HTTP {exc.code} for {url}; not retrying (respecting access limits)") from exc
    html = body.decode("utf-8", "replace")
    m = NEXT_DATA_RE.search(html)
    if not m:
        raise RuntimeError(f"{url}: page did not contain __NEXT_DATA__ (site change or blocked response)")
    next_data = json.loads(m.group(1))
    cache_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(payload_path, "wt", encoding="utf-8") as f:
        json.dump(next_data, f)
    write_json(meta_path, {
        "game_id": game_id, "url": url, "http_status": status, "fetched_utc": utc_now(),
        "html_bytes": len(body), "html_sha256": sha256_bytes(body),
        "payload_sha256": sha256_bytes(m.group(1).encode("utf-8")),
        "source": "www.nba.com game page __NEXT_DATA__ (props.pageProps)",
    })
    return payload_path


def fetch_many(games: list[dict], cache_dir: Path = GAME_PAGES_DIR, allow_network: bool = False,
               delay_s: float = 4.0, refresh: bool = False) -> dict[str, str]:
    """Fetch pages for [{'game_id','away','home'}, ...]. Stops on the first HTTP refusal."""
    status: dict[str, str] = {}
    for i, g in enumerate(games):
        payload_path, _ = cache_paths(g["game_id"], cache_dir)
        if payload_path.exists() and not refresh:
            status[g["game_id"]] = "cached"
            continue
        try:
            fetch_game_page(g["game_id"], g["away"], g["home"], cache_dir, allow_network, refresh)
            status[g["game_id"]] = "fetched"
        except FileNotFoundError:
            status[g["game_id"]] = "missing (network not allowed)"
        except RuntimeError as exc:
            status[g["game_id"]] = f"error: {exc}"
            if "HTTP 403" in str(exc) or "HTTP 429" in str(exc):
                log.warning("access refused; stopping further requests")
                break
        if allow_network and i < len(games) - 1:
            time.sleep(delay_s)
    return status


def load_game_page(game_id: str, cache_dir: Path = GAME_PAGES_DIR) -> dict:
    payload_path, _ = cache_paths(game_id, cache_dir)
    require(payload_path.exists(), f"cached game page missing: {payload_path}")
    with gzip.open(payload_path, "rt", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class GamePage:
    header: dict
    team_box: pd.DataFrame
    player_box: pd.DataFrame
    actions: pd.DataFrame


# ShotChartDetail ACTION_TYPE vocabulary (2022-23 .. 2024-25). Live-format (NGSS) play-by-play
# describes shots as descriptor + subType ("driving floating" + "Jump Shot"); they are mapped back to
# this vocabulary so both feed formats produce identical ACTION_TYPE values.
HIST_ACTION_TYPES = (
    "Jump Shot", "Pullup Jump shot", "Driving Layup Shot", "Driving Floating Jump Shot", "Step Back Jump shot",
    "Running Layup Shot", "Driving Finger Roll Layup Shot", "Running Jump Shot", "Cutting Layup Shot", "Layup Shot",
    "Tip Layup Shot", "Fadeaway Jump Shot", "Floating Jump shot", "Putback Layup Shot", "Driving Floating Bank Jump Shot",
    "Turnaround Fadeaway shot", "Cutting Dunk Shot", "Running Pull-Up Jump Shot", "Running Dunk Shot",
    "Turnaround Jump Shot", "Turnaround Hook Shot", "Driving Reverse Layup Shot", "Driving Dunk Shot",
    "Alley Oop Dunk Shot", "Running Finger Roll Layup Shot", "Driving Hook Shot", "Hook Shot", "Dunk Shot",
    "Reverse Layup Shot", "Tip Dunk Shot", "Alley Oop Layup shot", "Cutting Finger Roll Layup Shot",
    "Running Reverse Layup Shot", "Running Alley Oop Dunk Shot", "Putback Dunk Shot", "Jump Bank Shot",
    "Finger Roll Layup Shot", "Driving Bank Hook Shot", "Turnaround Bank shot", "Turnaround Bank Hook Shot",
    "Running Alley Oop Layup Shot", "Fadeaway Bank shot", "Hook Bank Shot", "Reverse Dunk Shot",
    "Turnaround Fadeaway Bank Jump Shot", "Step Back Bank Jump Shot", "Driving Reverse Dunk Shot", "Running Reverse Dunk Shot",
)


def _akey(s: str) -> str:
    """Word-order-insensitive key: NGSS splits "Hook Bank Shot" into descriptor "bank" + "Hook Shot"."""
    s = str(s).lower().replace("pull-up", "pullup").replace("step back", "stepback")
    return " ".join(sorted(w for w in re.findall(r"[a-z]+", s) if w != "shot"))


_ACTION_CANON = {_akey(a): a for a in HIST_ACTION_TYPES}
assert len(_ACTION_CANON) == len(HIST_ACTION_TYPES), "historical action types collide under _akey"
LIVE_ACTION_TYPES = {"rebound": "Rebound", "turnover": "Turnover", "freethrow": "Free Throw", "timeout": "Timeout",
                     "jumpball": "Jump Ball", "foul": "Foul", "substitution": "Substitution", "violation": "Violation",
                     "steal": "Steal", "block": "Block", "period": "period", "game": "game"}


def live_action_type(descriptor, sub_type) -> str:
    """Map NGSS (descriptor, subType) to the ShotChartDetail ACTION_TYPE vocabulary."""
    desc = "" if descriptor is None or (isinstance(descriptor, float) and np.isnan(descriptor)) else str(descriptor)
    words = set(_akey(f"{desc} {sub_type}").split())
    # "Bank" is dropped last: it is removed from model features anyway (outcome-leaking scorer tag).
    for drop in ((), ("jump",), ("bank",), ("jump", "bank")):
        key = " ".join(sorted(words - set(drop)))
        if key in _ACTION_CANON:
            return _ACTION_CANON[key]
    return f"{desc} {sub_type}".strip().title() + " Shot"


def normalize_live_actions(actions: list[dict], home_team_id: int) -> pd.DataFrame:
    """Convert NGSS live-format play-by-play to the stats (v3) schema used by pbp_shots/pbp_context."""
    a = pd.DataFrame(actions)
    if "orderNumber" in a:
        a = a.sort_values("orderNumber", kind="stable")
    a = a.reset_index(drop=True)
    fg = a["isFieldGoal"].fillna(0).astype(int) == 1
    raw_type = a["actionType"].astype(str)
    out = a.copy()
    out["actionType"] = raw_type.map(LIVE_ACTION_TYPES).fillna(raw_type)
    out["shotValue"] = np.where(fg, raw_type.map({"2pt": 2, "3pt": 3}), np.nan)
    out.loc[fg, "actionType"] = np.where(a.loc[fg, "shotResult"] == "Made", "Made Shot", "Missed Shot")
    out.loc[fg, "subType"] = [live_action_type(d, t) for d, t in zip(a.loc[fg, "descriptor"], a.loc[fg, "subType"])]
    ft = raw_type == "freethrow"
    out.loc[ft, "subType"] = "Free Throw " + a.loc[ft, "subType"].astype(str)
    for c in ("personId", "teamId"):
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    is_home = (out["teamId"] == home_team_id).fillna(False).to_numpy(bool)
    out["location"] = np.where(is_home, "h", np.where(out["teamId"].notna().to_numpy(), "v", ""))
    if "actionId" not in out:
        out["actionId"] = out["actionNumber"]
    if "videoAvailable" not in out:
        out["videoAvailable"] = 0
    return out


def _team_full_name(team: dict) -> str:
    return f"{team.get('teamCity', '')} {team.get('teamName', '')}".strip()


def parse_game_page(next_data: dict, expected_game_id: str | None = None) -> GamePage:
    try:
        pp = next_data["props"]["pageProps"]
        game = pp["game"]
        pbp = pp["playByPlay"]
    except (KeyError, TypeError) as exc:
        raise DataValidationError(f"game page payload missing expected keys: {exc}") from exc
    gid = str(game.get("gameId", ""))
    if expected_game_id is not None:
        require(gid == expected_game_id, f"game page is for {gid}, expected {expected_game_id}")
    require(str(pbp.get("gameId", gid)) == gid, f"play-by-play gameId {pbp.get('gameId')} != {gid}")

    home, away = game["homeTeam"], game["awayTeam"]
    game_et = str(game.get("gameEt", ""))
    header = {
        "game_id": gid,
        "game_code": game.get("gameCode"),
        "game_date_et": game_et[:10] if game_et else None,
        "game_time_utc": game.get("gameTimeUTC"),
        "game_status": game.get("gameStatus"),
        "game_status_text": game.get("gameStatusText"),
        "final_period": game.get("period"),
        "home_team_id": home["teamId"], "home_tricode": home["teamTricode"], "home_name": _team_full_name(home),
        "away_team_id": away["teamId"], "away_tricode": away["teamTricode"], "away_name": _team_full_name(away),
        "home_score": home.get("score"), "away_score": away.get("score"),
        "video_available": bool(pbp.get("videoAvailable")),
    }

    team_rows, player_rows = [], []
    for side, team in (("home", home), ("away", away)):
        st = team.get("statistics") or {}
        team_rows.append({
            "game_id": gid, "side": side, "team_id": team["teamId"], "team": team["teamTricode"],
            "team_name": _team_full_name(team), "points": team.get("score"),
            **{k: st.get(k) for k in [
                "fieldGoalsMade", "fieldGoalsAttempted", "threePointersMade", "threePointersAttempted",
                "freeThrowsMade", "freeThrowsAttempted", "points",
            ]},
        })
        for p in team.get("players", []) or []:
            ps = p.get("statistics") or {}
            player_rows.append({
                "game_id": gid, "team_id": team["teamId"], "team": team["teamTricode"],
                "person_id": p.get("personId"),
                "player_name": f"{p.get('firstName', '')} {p.get('familyName', '')}".strip(),
                "fga": ps.get("fieldGoalsAttempted"), "fgm": ps.get("fieldGoalsMade"),
                "fg3a": ps.get("threePointersAttempted"), "fg3m": ps.get("threePointersMade"),
                "ftm": ps.get("freeThrowsMade"), "points": ps.get("points"),
            })
    team_box = pd.DataFrame(team_rows).rename(columns={"points": "score"}).loc[:, lambda d: ~d.columns.duplicated()]
    raw_actions = pbp.get("actions", []) or []
    live = pbp.get("source") == "ngss" or any(str(x.get("actionType")) in ("2pt", "3pt") for x in raw_actions)
    actions = normalize_live_actions(raw_actions, home["teamId"]) if live else pd.DataFrame(raw_actions)
    header["pbp_format"] = "ngss_live" if live else "stats_v3"
    require(len(actions) > 0, f"{gid}: play-by-play has no actions")
    actions.insert(0, "game_id", gid)
    return GamePage(header, team_box, pd.DataFrame(player_rows), actions)


PBP_SHOT_COLUMNS = [
    "GAME_ID", "GAME_EVENT_ID", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_NAME", "PERIOD",
    "MINUTES_REMAINING", "SECONDS_REMAINING", "EVENT_TYPE", "ACTION_TYPE", "SHOT_TYPE",
    "SHOT_ZONE_BASIC", "SHOT_ZONE_AREA", "SHOT_ZONE_RANGE", "SHOT_DISTANCE", "LOC_X", "LOC_Y",
    "SHOT_MADE_FLAG", "GAME_DATE", "HTM", "VTM", "SEASON",
]


def pbp_shots(page: GamePage) -> pd.DataFrame:
    """Field-goal attempts from the play-by-play in a ShotChartDetail-compatible schema.

    Direct copies: GAME_EVENT_ID (= actionNumber), PLAYER_ID, TEAM_ID, PERIOD, LOC_X/LOC_Y
    (= xLegacy/yLegacy), ACTION_TYPE (= subType), EVENT_TYPE, SHOT_MADE_FLAG.
    Derived (validated rules, see court.py): SHOT_DISTANCE, SHOT_ZONE_BASIC, SHOT_ZONE_RANGE.
    Not available: SHOT_ZONE_AREA (left NaN; not a Model A feature).
    Raw play-by-play fields are preserved with a `pbp_` prefix.
    """
    a = page.actions
    h = page.header
    fg = a[a["isFieldGoal"] == 1].copy()
    require(fg["actionType"].isin(["Made Shot", "Missed Shot"]).all(),
            f"{h['game_id']}: unexpected actionType among field goals: {sorted(fg['actionType'].unique())}")
    require(fg["shotValue"].isin([2, 3]).all(), f"{h['game_id']}: field goals with shotValue not in {{2,3}}")
    require(not fg["actionNumber"].duplicated().any(), f"{h['game_id']}: duplicate actionNumber among field goals")

    remaining = clk.parse_clock_series(fg["clock"])
    mins_secs = remaining.map(clk.to_scd_minutes_seconds)
    name_map = dict(zip(page.player_box["person_id"], page.player_box["player_name"]))
    team_names = {h["home_team_id"]: h["home_name"], h["away_team_id"]: h["away_name"]}

    out = pd.DataFrame({
        "GAME_ID": h["game_id"],
        "GAME_EVENT_ID": fg["actionNumber"].astype(int).to_numpy(),
        "PLAYER_ID": fg["personId"].astype(int).to_numpy(),
        "PLAYER_NAME": fg["personId"].map(name_map).fillna(fg["playerName"]).to_numpy(),
        "TEAM_ID": fg["teamId"].astype(int).to_numpy(),
        "TEAM_NAME": fg["teamId"].map(team_names).to_numpy(),
        "PERIOD": fg["period"].astype(int).to_numpy(),
        "MINUTES_REMAINING": [m for m, _ in mins_secs],
        "SECONDS_REMAINING": [s for _, s in mins_secs],
        "EVENT_TYPE": fg["actionType"].to_numpy(),
        "ACTION_TYPE": fg["subType"].to_numpy(),
        "SHOT_TYPE": np.where(fg["shotValue"] == 3, "3PT Field Goal", "2PT Field Goal"),
        "LOC_X": fg["xLegacy"].astype(int).to_numpy(),
        "LOC_Y": fg["yLegacy"].astype(int).to_numpy(),
        "SHOT_MADE_FLAG": (fg["shotResult"] == "Made").astype(int).to_numpy(),
        "GAME_DATE": (h["game_date_et"] or "").replace("-", ""),
        "HTM": h["home_tricode"], "VTM": h["away_tricode"], "SEASON": SEASON,
    })
    value = np.where(out["SHOT_TYPE"].str.startswith("3"), 3, 2)
    out["SHOT_DISTANCE"] = court.shot_distance_scd(out["LOC_X"], out["LOC_Y"])
    out["SHOT_ZONE_BASIC"] = court.shot_zone_basic(out["LOC_X"], out["LOC_Y"], value)
    out["SHOT_ZONE_RANGE"] = court.shot_zone_range(out["SHOT_DISTANCE"], out["LOC_Y"])
    out["SHOT_ZONE_AREA"] = np.nan
    out = out[PBP_SHOT_COLUMNS]

    require((out["EVENT_TYPE"] == "Made Shot").eq(out["SHOT_MADE_FLAG"] == 1).all(),
            f"{h['game_id']}: shotResult disagrees with actionType")
    raw = fg[["clock", "shotDistance", "shotResult", "description", "teamTricode", "scoreHome", "scoreAway",
              "videoAvailable", "actionId", "shotValue", "location"]].reset_index(drop=True)
    raw.columns = [f"pbp_{c}" for c in raw.columns]
    out = pd.concat([out.reset_index(drop=True), raw], axis=1)
    out["pbp_clock_remaining_s"] = remaining.to_numpy()
    out["source"] = "nba_game_page_pbp"
    return out


def score_timeline(page: GamePage) -> pd.DataFrame:
    """Actual total-score progression (field goals + free throws) from the play-by-play."""
    a = page.actions.copy()
    a["home_score"] = pd.to_numeric(a["scoreHome"], errors="coerce")
    a["away_score"] = pd.to_numeric(a["scoreAway"], errors="coerce")
    a = a[a["home_score"].notna() & a["away_score"].notna()].copy()
    a["clock_remaining_s"] = clk.parse_clock_series(a["clock"])
    a["elapsed_s"] = clk.elapsed_series(a["period"], a["clock_remaining_s"])
    keep = ["game_id", "actionNumber", "actionId", "period", "clock", "clock_remaining_s", "elapsed_s",
            "teamTricode", "actionType", "home_score", "away_score"]
    a = a[keep].sort_values(["elapsed_s", "actionNumber"]).reset_index(drop=True)
    changed = (a[["home_score", "away_score"]].diff().fillna(a[["home_score", "away_score"]]) != 0).any(axis=1)
    return a[changed].reset_index(drop=True)


def box_validation(page: GamePage, shots: pd.DataFrame) -> pd.DataFrame:
    """Compare play-by-play FGA/FGM/3PA counts per team with the official box score."""
    agg = shots.groupby("TEAM_ID").agg(
        pbp_fga=("SHOT_MADE_FLAG", "size"), pbp_fgm=("SHOT_MADE_FLAG", "sum"),
        pbp_fg3a=("SHOT_TYPE", lambda s: int((s == "3PT Field Goal").sum())),
    )
    tb = page.team_box.set_index("team_id")
    out = tb[["game_id", "team", "fieldGoalsAttempted", "fieldGoalsMade", "threePointersAttempted", "freeThrowsMade", "score"]].join(agg)
    out = out.rename(columns={"fieldGoalsAttempted": "box_fga", "fieldGoalsMade": "box_fgm",
                              "threePointersAttempted": "box_fg3a", "freeThrowsMade": "box_ftm", "score": "box_points"})
    out["pbp_matches_box"] = (out["pbp_fga"] == out["box_fga"]) & (out["pbp_fgm"] == out["box_fgm"]) & (out["pbp_fg3a"] == out["box_fg3a"])
    return out.reset_index()
