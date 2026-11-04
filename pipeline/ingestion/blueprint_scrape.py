"""Blueprint (NBA xFG) page scraping.

Port of the extraction logic in `scripts/legacy/nba_xfg_pipeline.py` (which needs `requests` and
optionally Playwright) to the standard library, with two reliability changes:

* every shot-shaped record found in the page is counted and every rejection is recorded with a
  reason, so parsing loss cannot be silent;
* the page is fetched over plain HTTPS only. No browser automation, no retries on 403/429.

Output files keep the original scraper's formats (`games/<id>.csv` with FIELDS, `raw/<id>.json`
with the verbatim shot objects) so `pipeline.ingestion.blueprint.load_blueprint` reads them unchanged.
"""

from __future__ import annotations

import csv
import gzip
import html
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

from pipeline.config import BLUEPRINT_DIR, GAMES_LISTING_DIR
from pipeline.utils import get_logger, sha256_bytes, utc_now, write_json

log = get_logger("blueprint_scrape")

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"
RECORD_START = re.compile(r'\{\s*"period"\s*:\s*\d+\s*,\s*"gameClockTime"\s*:')
FLIGHT_PUSH = re.compile(r'(?:self\.)?__next_f\.push\(\s*(\[\s*\d+\s*,\s*"(?:\\.|[^"\\])*"\s*\])\s*\)', re.S)
GAME_LINK_RE = re.compile(
    r"(?:https?://(?:www\.)?nba\.com)?/game/([a-z]{2,3})-vs-([a-z]{2,3})-(00226\d{5})(?=[/?#\"'\s<]|$)", re.I)

FIELDS = [
    "game_id", "date", "ngss_event_id", "period", "clock", "shot_clock",
    "shooter_id", "shooter", "team", "defender_id", "defender",
    "shot_type", "made", "xfg", "xpts_fg_only", "shot_distance_ft",
    "contest_distance_in", "contest_height_in", "shot_vertical_in",
    "shooter_speed_mph", "release_time_s", "shot_x", "shot_y",
    "home_score", "away_score", "featured", "source_url",
]


class AccessRefused(RuntimeError):
    """NBA.com refused the request (HTTP 403/429). Callers must stop, not retry."""


def blueprint_url(game_id: str, away: str, home: str) -> str:
    return f"https://www.nba.com/game/{away.lower()}-vs-{home.lower()}-{game_id}/blueprint?watchFullGame=true"


def games_page_url(day: str) -> str:
    return f"https://www.nba.com/games?date={day}"


def http_get(url: str, timeout: int = 45) -> tuple[bytes, int]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html", "Accept-Encoding": "gzip"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            if resp.headers.get("Content-Encoding") == "gzip":
                body = gzip.decompress(body)
            return body, resp.status
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 429):
            raise AccessRefused(f"HTTP {exc.code} for {url}") from exc
        raise RuntimeError(f"HTTP {exc.code} for {url}") from exc


# --------------------------------------------------------------------------- extraction (ported)


def _find_balanced_object(source: str, start: int) -> str | None:
    depth = 0
    inside = escaped = False
    for j in range(start, len(source)):
        c = source[j]
        if inside:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                inside = False
        elif c == '"':
            inside = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return source[start:j + 1]
    return None


def _candidate_sources(text: str) -> list[str]:
    candidates = [text]
    unescaped = html.unescape(text)
    if unescaped != text:
        candidates.append(unescaped)
    for match in FLIGHT_PUSH.finditer(text):
        try:
            value = json.loads(match.group(1))
            if len(value) > 1 and isinstance(value[1], str):
                candidates.append(value[1])
        except (json.JSONDecodeError, IndexError, TypeError):
            pass
    for _ in range(2):  # React Flight may escape quotes more than once
        additions = []
        for source in candidates:
            if '\\"' in source:
                normalized = source.replace('\\"', '"')
                if normalized not in candidates and normalized not in additions:
                    additions.append(normalized)
        candidates.extend(additions)
    return candidates


def _clock_seconds(clock) -> float:
    try:
        minutes, seconds = str(clock).split(":", 1)
        return int(minutes) * 60 + float(seconds)
    except (ValueError, AttributeError):
        return 0.0


def extract_shots(text: str) -> tuple[list[dict], dict]:
    """Shot objects found in the page plus an audit of everything that looked like one.

    Returns (shots, audit). `audit["rejected"]` lists distinct shot-shaped records that were not
    kept and why (unparseable JSON, no shooter, missing/invalid xFG).
    """
    found: dict[tuple, dict] = {}
    rejected: dict[tuple, str] = {}
    for source in _candidate_sources(text):
        for match in RECORD_START.finditer(source):
            blob = _find_balanced_object(source, match.start())
            if not blob:
                rejected[("unbalanced", match.start(), len(source))] = "unbalanced_object"
                continue
            try:
                shot = json.loads(blob)
            except json.JSONDecodeError:
                continue  # the same record also appears in an unescaped candidate source
            shooter = shot.get("shooter")
            key = (shot.get("ngssEventId"), shot.get("period"), shot.get("gameClockTime"),
                   shooter.get("playerId") if isinstance(shooter, dict) else None)
            xfg = shot.get("expFieldGoalPct")
            if not isinstance(shooter, dict):
                rejected[key] = "no_shooter"
            elif not isinstance(xfg, (int, float)) or isinstance(xfg, bool):
                rejected[key] = "missing_xfg"
            elif not 0 <= xfg <= 1:
                rejected[key] = "xfg_out_of_range"
            else:
                found[key] = shot
    rejected = {k: v for k, v in rejected.items() if k not in found}
    shots = sorted(found.values(), key=lambda r: (r.get("period", 0), -_clock_seconds(r.get("gameClockTime", "0:00")),
                                                  r.get("ngssEventId") or 0))
    audit = {"records_kept": len(shots), "records_rejected": len(rejected),
             "rejections": [{"key": [str(x) for x in k], "reason": v} for k, v in rejected.items()],
             "expFieldGoalPct_mentions": text.count("expFieldGoalPct")}
    return shots, audit


def to_row(game_id: str, game_date: str, shot: dict, url: str) -> dict:
    shooter = shot.get("shooter") or {}
    defender = shot.get("defender") or {}
    attrib = shot.get("attributes") or {}
    probability = float(shot["expFieldGoalPct"])
    value = 3 if shot.get("shotType") == "3pt" else 2
    return {
        "game_id": game_id, "date": game_date, "ngss_event_id": shot.get("ngssEventId"),
        "period": shot.get("period"), "clock": shot.get("gameClockTime"), "shot_clock": shot.get("shotClockTime"),
        "shooter_id": shooter.get("playerId"),
        "shooter": f"{shooter.get('firstName', '')} {shooter.get('lastName', '')}".strip(),
        "team": shooter.get("teamAbbreviation"), "defender_id": defender.get("playerId"),
        "defender": f"{defender.get('firstName', '')} {defender.get('lastName', '')}".strip(),
        "shot_type": shot.get("shotType"), "made": shot.get("success"),
        "xfg": probability, "xpts_fg_only": round(probability * value, 5),
        "shot_distance_ft": attrib.get("shotDistance"), "contest_distance_in": attrib.get("contestDistance"),
        "contest_height_in": attrib.get("contestHeight"), "shot_vertical_in": attrib.get("shotVertical"),
        "shooter_speed_mph": attrib.get("shooterSpeed"), "release_time_s": attrib.get("releaseTime"),
        "shot_x": shot.get("x"), "shot_y": shot.get("y"),
        "home_score": shot.get("homeTeamScore"), "away_score": shot.get("awayTeamScore"),
        "featured": shot.get("featured"), "source_url": url,
    }


def discover_games_in_html(source: str, day: str) -> list[dict]:
    found = {}
    for body in (source, html.unescape(source), source.replace("\\/", "/")):
        for away, home, game_id in GAME_LINK_RE.findall(body):
            found[game_id] = {"game_id": game_id, "date": day, "away": away.upper(), "home": home.upper()}
    return sorted(found.values(), key=lambda g: g["game_id"])


# --------------------------------------------------------------------------- cached fetchers


def paths(game_id: str, out_dir: Path = BLUEPRINT_DIR) -> dict[str, Path]:
    return {"csv": out_dir / "games" / f"{game_id}.csv", "raw": out_dir / "raw" / f"{game_id}.json",
            "meta": out_dir / "raw" / f"{game_id}.fetch_meta.json"}


def fetch_blueprint_game(game_id: str, away: str, home: str, game_date: str, out_dir: Path = BLUEPRINT_DIR,
                         allow_network: bool = False, refresh: bool = False) -> dict:
    """Fetch (once) and store one game's Blueprint shots. Returns the fetch metadata."""
    p = paths(game_id, out_dir)
    if p["csv"].exists() and p["raw"].exists() and not refresh:
        meta = json.loads(p["meta"].read_text()) if p["meta"].exists() else {"game_id": game_id, "source": "legacy scraper"}
        meta["cached"] = True
        return meta
    if not allow_network:
        raise FileNotFoundError(f"no cached Blueprint data for {game_id}; rerun with --allow-network")
    url = blueprint_url(game_id, away, home)
    body, status = http_get(url)
    text = body.decode("utf-8", "replace")
    shots, audit = extract_shots(text)
    for s in shots:
        gid = str(s.get("gameId") or game_id)
        if gid != game_id:
            raise RuntimeError(f"{url}: shot record belongs to game {gid}")
    meta = {"game_id": game_id, "url": url, "http_status": status, "fetched_utc": utc_now(),
            "html_bytes": len(body), "html_sha256": sha256_bytes(body), **audit, "cached": False}
    p["csv"].parent.mkdir(parents=True, exist_ok=True)
    p["raw"].parent.mkdir(parents=True, exist_ok=True)
    if shots:
        with p["csv"].open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(to_row(game_id, game_date, s, url) for s in shots)
        p["raw"].write_text(json.dumps(shots, ensure_ascii=False, indent=2), encoding="utf-8")
    write_json(p["meta"], meta)
    return meta


def fetch_games_listing(day: str, listing_dir: Path = GAMES_LISTING_DIR, allow_network: bool = False,
                        refresh: bool = False) -> list[dict]:
    """Game links on nba.com/games?date=DAY (cached). The listing can include games from other
    dates; callers must verify each game's date against its official game page."""
    cache = listing_dir / f"{day}.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text(encoding="utf-8"))
    if not allow_network:
        raise FileNotFoundError(f"no cached games listing for {day}")
    body, _ = http_get(games_page_url(day))
    games = discover_games_in_html(body.decode("utf-8", "replace"), day)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(games, indent=2), encoding="utf-8")
    return games
