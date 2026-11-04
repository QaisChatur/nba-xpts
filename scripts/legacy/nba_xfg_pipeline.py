#!/usr/bin/env python3
"""NBA Blueprint shot-level xFG explorer: first one game, then this season.

USAGE
  pip install requests playwright
  python -m playwright install chromium

  # One game, using the HTML Blueprint page and official boxscore for verification:
  python nba_xfg_pipeline.py one --game-id 0022600136 --away ORL --home TOR --headed

  # Test the exact source fragment you copied from Chrome, entirely offline:
  python nba_xfg_pipeline.py one --game-id 0022600136 --input copied_source.txt

  # First 3 completed games from Oct 28:
  python nba_xfg_pipeline.py season --from-date 2026-10-28 --max-games 3

  # All completed regular-season games available in the 2026-27 schedule:
  python nba_xfg_pipeline.py season --delay 5

Source access may be limited by NBA.com terms or anti-bot measures. The script does
not bypass restrictions. Do not assume a game is complete without a matching FGA
count and further review; ngssEventId has not been proven equal to GAME_EVENT_ID.
"""

import argparse
import csv
import html
import json
import re
import sys
import time
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from urllib.parse import urlparse

try:
    import requests
except ImportError as exc:
    raise SystemExit('Install dependencies: pip install requests playwright') from exc

SCHEDULE_URL = 'https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json'
BOX_URL = 'https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{game_id}.json'
RECORD_START = re.compile(r'\{\s*"period"\s*:\s*\d+\s*,\s*"gameClockTime"\s*:')
FLIGHT_PUSH = re.compile(r'(?:self\.)?__next_f\.push\(\s*(\[\s*\d+\s*,\s*"(?:\\.|[^"\\])*"\s*\])\s*\)', re.S)
USER_AGENT = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
HEADERS = {'User-Agent': USER_AGENT, 'Accept': 'text/html,application/json;q=0.9,*/*;q=0.8'}

FIELDS = [
    'game_id', 'date', 'ngss_event_id', 'period', 'clock', 'shot_clock',
    'shooter_id', 'shooter', 'team', 'defender_id', 'defender',
    'shot_type', 'made', 'xfg', 'xpts_fg_only', 'shot_distance_ft',
    'contest_distance_in', 'contest_height_in', 'shot_vertical_in',
    'shooter_speed_mph', 'release_time_s', 'shot_x', 'shot_y',
    'home_score', 'away_score', 'featured', 'source_url'
]


def find_balanced_object(source, start):
    depth = 0
    inside = escaped = False
    for j in range(start, len(source)):
        c = source[j]
        if inside:
            if escaped:
                escaped = False
            elif c == '\\':
                escaped = True
            elif c == '"':
                inside = False
        elif c == '"':
            inside = True
        elif c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return source[start:j + 1]
    return None


def extract_shots(text):
    """Find shot objects in regular JSON, HTML, or Next.js streaming Flight payloads."""
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

    # React Flight may escape quotes one or more times.
    for _ in range(2):
        additions = []
        for source in candidates:
            if '\\"' in source:
                normalized = source.replace('\\"', '"')
                if normalized not in candidates and normalized not in additions:
                    additions.append(normalized)
        candidates.extend(additions)

    found = {}
    for source in candidates:
        for match in RECORD_START.finditer(source):
            blob = find_balanced_object(source, match.start())
            if not blob:
                continue
            try:
                shot = json.loads(blob)
            except json.JSONDecodeError:
                continue
            if not isinstance(shot.get('shooter'), dict) or not isinstance(shot.get('expFieldGoalPct'), (float, int)):
                continue
            if not (0 <= shot['expFieldGoalPct'] <= 1):
                continue
            key = (shot.get('ngssEventId'), shot.get('period'), shot.get('gameClockTime'), shot['shooter'].get('playerId'))
            found[key] = shot
    return sorted(found.values(), key=lambda r: (r.get('period', 0), -clock_seconds(r.get('gameClockTime', '0:00')), r.get('ngssEventId') or 0))


def clock_seconds(clock):
    try:
        minutes, seconds = str(clock).split(':', 1)
        return int(minutes) * 60 + float(seconds)
    except (ValueError, AttributeError):
        return 0


def game_url(game_id, away, home):
    return f'https://www.nba.com/game/{away.lower()}-vs-{home.lower()}-{game_id}/blueprint?watchFullGame=true'


def request_text(url, timeout=30):
    response = requests.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.text


def get_schedule(outdir, refresh=False):
    """Use cached schedule first; fall back from requests to installed Chrome."""
    cache = outdir / 'schedule.json'
    if cache.exists() and not refresh:
        payload = json.loads(cache.read_text(encoding='utf-8'))
        print(f'Using cached schedule: {cache}')
    else:
        request_error = None
        try:
            response = requests.get(SCHEDULE_URL, headers=HEADERS, timeout=35)
            response.raise_for_status()
            payload = response.json()
            print('Fetched NBA schedule using requests')
        except (requests.RequestException, ValueError) as exc:
            request_error = exc
            print(f'Direct schedule fetch failed ({exc}); trying installed Chrome...')
            try:
                from playwright.sync_api import sync_playwright
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch(channel='chrome', headless=False)
                    try:
                        page = browser.new_page()
                        response = page.goto(SCHEDULE_URL, wait_until='domcontentloaded', timeout=45000)
                        if response is None or not response.ok:
                            status = response.status if response else 'no response'
                            raise RuntimeError(f'Chrome received status {status}')
                        payload = json.loads(page.locator('body').inner_text(timeout=15000))
                        print('Fetched NBA schedule using installed Chrome')
                    finally:
                        browser.close()
            except Exception as chrome_error:
                raise RuntimeError(
                    f'NBA CDN denied both methods. Requests: {request_error}; '
                    f'Chrome: {chrome_error}. '
                    f'If the schedule URL opens in your regular Chrome, save its JSON content '
                    f'as {cache} and run again without --refresh-schedule.'
                ) from chrome_error
        cache.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    schedule = payload.get('leagueSchedule', {})
    year = str(schedule.get('seasonYear', ''))
    if not year.startswith('2026'):
        raise RuntimeError(f'Schedule did not identify the 2026-27 season (reported {year!r}); refuse to process wrong season.')
    games = []
    for group in schedule.get('gameDates', []):
        for game in group.get('games', []):
            gid = str(game.get('gameId', ''))
            if not gid.startswith('00226') or str(game.get('gameStatus')) != '3':
                continue
            code = game.get('gameCode', '')
            try:
                game_date = datetime.strptime(code.split('/')[0], '%Y%m%d').date().isoformat()
            except ValueError:
                game_date = str(group.get('gameDate', ''))[:10]
            away = (game.get('awayTeam') or {}).get('teamTricode', '')
            home = (game.get('homeTeam') or {}).get('teamTricode', '')
            if not away or not home:
                print(f'Skipping game {gid}: missing home/away teams', file=sys.stderr)
                continue
            games.append({'game_id': gid, 'date': game_date, 'away': away, 'home': home})
    return sorted(games, key=lambda g: (g['date'], g['game_id']))


# Alternative to blocked cdn.nba.com schedule: official NBA Games pages contain
# canonical game links such as /game/orl-vs-tor-0022600136.
GAME_LINK_RE = re.compile(
    r"(?:https?://(?:www\.)?nba\.com)?/game/([a-z]{2,3})-vs-([a-z]{2,3})-(00226\d{5})(?=[/?#\"'\s<]|$)",
    re.I,
)


def discover_games_in_html(source, day):
    # Unescape HTML entities and Next.js escaped slashes if necessary.
    alternatives = [source, html.unescape(source), source.replace('\\/', '/')]
    found = {}
    for body in alternatives:
        for away, home, game_id in GAME_LINK_RE.findall(body):
            found[game_id] = {'game_id': game_id, 'date': day, 'away': away.upper(), 'home': home.upper()}
    return sorted(found.values(), key=lambda game: game['game_id'])


def fetch_games_page(day, mode='auto', headed=False):
    url = 'https://www.nba.com/games?date=' + day
    errors = []
    if mode in ('http', 'auto'):
        try:
            source = request_text(url, timeout=35)
            matches = discover_games_in_html(source, day)
            if matches:
                return matches
            errors.append('HTML contained no recognizable NBA game links')
        except requests.RequestException as exc:
            errors.append('requests: ' + str(exc))
    if mode == 'http':
        raise RuntimeError('; '.join(errors))

    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='chrome', headless=not headed)
        try:
            page = browser.new_page(viewport={'width': 1440, 'height': 900})
            response = page.goto(url, wait_until='domcontentloaded', timeout=45000)
            if response and response.status in (403, 429):
                raise RuntimeError(f'NBA Games page returned HTTP {response.status}')
            page.wait_for_timeout(1500)
            sources = [page.content()]
            if response:
                try:
                    sources.append(response.text())
                except Exception:
                    pass
            found = {}
            for body in sources:
                for game in discover_games_in_html(body, day):
                    found[game['game_id']] = game
            if found:
                return sorted(found.values(), key=lambda g: g['game_id'])
            errors.append('Chrome HTML did not contain NBA game links')
        finally:
            browser.close()
    raise RuntimeError('; '.join(errors))


def get_schedule_from_games_pages(outdir, start_day, end_day, mode='auto', headed=False, refresh=False, max_games=None):
    # Discover games directly from NBA Games HTML with one cached JSON per day.
    first = date.fromisoformat(start_day)
    last = date.fromisoformat(end_day)
    if first > last:
        raise RuntimeError('--from-date is after --to-date')
    # Avoid accidental requests spanning multiple seasons or distant future.
    if (last-first).days > 365:
        raise RuntimeError('Date range too large: choose at most 366 calendar days')
    cache_dir = outdir / 'schedule_days'
    cache_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    current = first
    while current <= last:
        day = current.isoformat()
        cache = cache_dir / (day + '.json')
        if cache.exists() and not refresh:
            games = json.loads(cache.read_text(encoding='utf-8'))
            print(f'[{day}] using cached NBA Games listing ({len(games)} games)', flush=True)
        else:
            print(f'[{day}] discovering games from nba.com/games ...', flush=True)
            games = fetch_games_page(day, mode=mode, headed=headed)
            if not games:
                raise RuntimeError(f'No NBA game links discovered for {day}. This may be an off-day or a site change. Use --from-date on a known game day to diagnose.')
            cache.write_text(json.dumps(games, indent=2), encoding='utf-8')
            print(f'[{day}] discovered {len(games)} games', flush=True)
        for game in games:
            result[game['game_id']] = game
        if max_games is not None and len(result) >= max_games:
            break
        current += timedelta(days=1)
    return sorted(result.values(), key=lambda g: (g['date'], g['game_id']))


def fetch_blueprint(url, mode, headed, timeout, keep_html_path=None):
    issues = []
    if mode in ('http', 'auto'):
        try:
            source = request_text(url, timeout)
            shots = extract_shots(source)
            if shots:
                if keep_html_path:
                    keep_html_path.write_text(source, encoding='utf-8')
                return shots, 'http'
            issues.append('HTTP HTML had no recognizable xFG objects')
        except requests.RequestException as exc:
            issues.append(f'HTTP request: {str(exc)[:180]}')

    if mode == 'http':
        raise RuntimeError('; '.join(issues))

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError('Install browser support: pip install playwright && python -m playwright install chromium') from exc

    with sync_playwright() as p:
        browser = p.chromium.launch(channel='chrome', headless=not headed)
        page = browser.new_page(user_agent=USER_AGENT, viewport={'width': 1440, 'height': 900})
        extra_sources = []

        def watch_response(response):
            if 'nba.com' not in response.url:
                return
            content_type = response.headers.get('content-type', '').lower()
            if not any(t in content_type for t in ('html', 'json', 'javascript', 'text/plain')):
                return
            try:
                body = response.text()
                if 'expFieldGoalPct' in body:
                    extra_sources.append(body)
            except Exception:
                pass

        page.on('response', watch_response)
        try:
            response = page.goto(url, wait_until='domcontentloaded', timeout=timeout * 1000)
            page.wait_for_timeout(3000)
            texts = [page.content()]
            if response:
                try:
                    texts.append(response.text())
                except Exception:
                    pass
            texts.extend(extra_sources)
            all_shots = {}
            for text in texts:
                for shot in extract_shots(text):
                    all_shots[(shot.get('ngssEventId'), shot.get('period'), shot.get('gameClockTime'), shot.get('shooter', {}).get('playerId'))] = shot
            shots = list(all_shots.values())
            if keep_html_path:
                keep_html_path.write_text('\n'.join(texts), encoding='utf-8')
        finally:
            browser.close()
    if not shots:
        raise RuntimeError('; '.join(issues + ['Browser HTML/responses contained no recognizable xFG objects']))
    return sorted(shots, key=lambda s: (s.get('period', 0), -clock_seconds(s.get('gameClockTime', '0:00')))), 'browser'


def get_boxscore_fga(game_id):
    url = BOX_URL.format(game_id=game_id)
    try:
        data = requests.get(url, headers=HEADERS, timeout=25)
        data.raise_for_status()
        game = data.json().get('game', {})
        away = (game.get('awayTeam') or {}).get('statistics') or {}
        home = (game.get('homeTeam') or {}).get('statistics') or {}
        if 'fieldGoalsAttempted' not in away or 'fieldGoalsAttempted' not in home:
            return None
        return int(away['fieldGoalsAttempted']) + int(home['fieldGoalsAttempted'])
    except (requests.RequestException, ValueError, TypeError):
        return None


def to_row(game, shot, url):
    shooter = shot.get('shooter') or {}
    defender = shot.get('defender') or {}
    attrib = shot.get('attributes') or {}
    probability = float(shot['expFieldGoalPct'])
    value = 3 if shot.get('shotType') == '3pt' else 2
    return {
        'game_id': game['game_id'], 'date': game.get('date', ''),
        'ngss_event_id': shot.get('ngssEventId'),
        'period': shot.get('period'), 'clock': shot.get('gameClockTime'),
        'shot_clock': shot.get('shotClockTime'), 'shooter_id': shooter.get('playerId'),
        'shooter': f"{shooter.get('firstName','')} {shooter.get('lastName','')}".strip(),
        'team': shooter.get('teamAbbreviation'), 'defender_id': defender.get('playerId'),
        'defender': f"{defender.get('firstName','')} {defender.get('lastName','')}".strip(),
        'shot_type': shot.get('shotType'), 'made': shot.get('success'),
        'xfg': probability, 'xpts_fg_only': round(probability * value, 5),
        'shot_distance_ft': attrib.get('shotDistance'),
        'contest_distance_in': attrib.get('contestDistance'),
        'contest_height_in': attrib.get('contestHeight'),
        'shot_vertical_in': attrib.get('shotVertical'),
        'shooter_speed_mph': attrib.get('shooterSpeed'),
        'release_time_s': attrib.get('releaseTime'),
        'shot_x': shot.get('x'), 'shot_y': shot.get('y'),
        'home_score': shot.get('homeTeamScore'), 'away_score': shot.get('awayTeamScore'),
        'featured': shot.get('featured'), 'source_url': url
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def extract_game(game, args, output_dir):
    gid = game['game_id']
    url = args.url if getattr(args, 'url', None) else game_url(gid, game['away'], game['home'])
    raw = output_dir / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    html_path = raw / f'{gid}.html' if args.keep_html else None
    if getattr(args, 'input', None):
        shots = extract_shots(Path(args.input).read_text(encoding='utf-8'))
        origin = 'local-file'
    else:
        shots, origin = fetch_blueprint(url, args.mode, args.headed, args.timeout, html_path)
    if not shots:
        raise RuntimeError('No shots found — page may have changed, or input may be an incomplete snippet')
    rows = [to_row(game, shot, url) for shot in shots]
    expected = None if args.no_validate else get_boxscore_fga(gid)
    status = 'count_matches' if expected == len(rows) else 'count_mismatch' if expected is not None else 'unverified'
    write_csv(output_dir / 'games' / f'{gid}.csv', rows)
    write_json(raw / f'{gid}.json', shots)
    return {
        'game_id': gid, 'game_date': game.get('date', ''), 'url': url,
        'source_method': origin, 'shots_found': len(shots), 'boxscore_fga': expected,
        'coverage': round(len(shots) / expected, 4) if expected else None,
        'status': status, 'all_featured': all(s.get('featured') is True for s in shots),
        'updated_utc': datetime.now(timezone.utc).isoformat()
    }


def rebuild_master(outdir):
    unique = {}
    for path in sorted((outdir / 'games').glob('*.csv')):
        with path.open(newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                key = (row['game_id'], row['ngss_event_id'], row['period'], row['clock'], row['shooter_id'])
                unique[key] = row
    merged = sorted(unique.values(), key=lambda r: (r['date'], r['game_id'], int(r['period']), -clock_seconds(r['clock'])))
    write_csv(outdir / 'all_shots.csv', merged)
    print(f'MASTER DATASET: {len(merged)} records -> {outdir / "all_shots.csv"}')


def main():
    ap = argparse.ArgumentParser(description='NBA Blueprint xFG extraction pipeline for 2026-27')
    sub = ap.add_subparsers(dest='command', required=True)
    for name in ['one', 'season']:
        parser = sub.add_parser(name)
        parser.add_argument('--outdir', type=Path, default=Path('nba_xfg_2026_27'))
        parser.add_argument('--mode', choices=['auto', 'http', 'browser'], default='auto')
        parser.add_argument('--headed', action='store_true', help='Open Chromium visibly (only for browser mode)')
        parser.add_argument('--timeout', type=int, default=45)
        parser.add_argument('--keep-html', action='store_true')
        parser.add_argument('--no-validate', action='store_true')
        parser.add_argument('--force', action='store_true', help='Recollect files already on disk')
        if name == 'one':
            parser.add_argument('--game-id', required=True)
            parser.add_argument('--away', default='ORL', help='Away team tricode (default ORL for example game)')
            parser.add_argument('--home', default='TOR', help='Home team tricode (default TOR for example game)')
            parser.add_argument('--url', default=None, help='Use exact Blueprint URL instead of generated URL')
            parser.add_argument('--input', type=Path, help='Offline: copied Sources text or saved HTML')
        else:
            parser.add_argument('--from-date', default=None, help='ISO YYYY-MM-DD')
            parser.add_argument('--to-date', default=(date.today()-timedelta(days=1)).isoformat(), help='ISO YYYY-MM-DD, defaults to yesterday to avoid unfinished games')
            parser.add_argument('--max-games', type=int, default=None)
            parser.add_argument('--delay', type=float, default=5.0, help='Seconds between game page requests')
            parser.add_argument('--refresh-schedule', action='store_true')
            parser.add_argument('--schedule-source', choices=['nba-web', 'nba-cdn'], default='nba-web', help='nba-web discovers games directly from NBA Games pages (default); nba-cdn uses schedule JSON')
            parser.add_argument('--retry-partial', action='store_true', help='Recheck games with mismatched attempt counts')
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.outdir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}

    if args.command == 'one':
        if args.game_id != '0022600136' and args.away == 'ORL' and args.home == 'TOR' and not args.url and not args.input:
            ap.error('For a different game, specify --away TEAM --home TEAM (or --url).')
        game = {'game_id': args.game_id, 'away': args.away, 'home': args.home, 'date': ''}
        try:
            result = extract_game(game, args, args.outdir)
            manifest[args.game_id] = result
            print(json.dumps(result, indent=2))
        except Exception as exc:
            manifest[args.game_id] = {'status': 'error', 'error': str(exc)}
            print('FAILED:', exc, file=sys.stderr)
            write_json(manifest_path, manifest)
            sys.exit(1)
    else:
        try:
            if args.schedule_source == 'nba-web':
                games = get_schedule_from_games_pages(
                    args.outdir, args.from_date or '2026-10-20', args.to_date,
                    mode=args.mode, headed=args.headed,
                    refresh=args.refresh_schedule, max_games=args.max_games)
            else:
                games = get_schedule(args.outdir, args.refresh_schedule)
        except Exception as exc:
            ap.error(f'Cannot retrieve season schedule: {exc}')
        games = [g for g in games if (not args.from_date or g['date'] >= args.from_date) and g['date'] <= args.to_date]
        print(f'Found {len(games)} completed 2026-27 regular-season games in selected date range.')
        if args.max_games is not None:
            games = games[:args.max_games]
        for index, game in enumerate(games, 1):
            gid = game['game_id']
            prior = manifest.get(gid, {})
            path = args.outdir / 'games' / f'{gid}.csv'
            if path.exists() and not args.force and not (args.retry_partial and prior.get('status') != 'count_matches'):
                print(f'[{index}/{len(games)}] SKIP {gid} (already saved; --force to refetch)')
                continue
            print(f'[{index}/{len(games)}] {game["date"]} {game["away"]}@{game["home"]} ({gid})')
            try:
                result = extract_game(game, args, args.outdir)
                manifest[gid] = result
                print(f'  {result["shots_found"]} shots / {result["boxscore_fga"]} FGA => {result["status"]}')
            except Exception as exc:
                manifest[gid] = {'status': 'error', 'error': str(exc), 'url': game_url(gid, game['away'], game['home'])}
                print('  FAILED:', exc, file=sys.stderr)
                if '429' in str(exc) or '403' in str(exc):
                    print('Access is limited by the site. Ending this run rather than repeatedly retrying.', file=sys.stderr)
                    write_json(manifest_path, manifest)
                    break
            write_json(manifest_path, manifest)
            if index < len(games):
                time.sleep(max(0, args.delay))
    write_json(manifest_path, manifest)
    rebuild_master(args.outdir)
    ok = sum(v.get('status') == 'count_matches' for v in manifest.values())
    mismatched = sum(v.get('status') == 'count_mismatch' for v in manifest.values())
    errors = sum(v.get('status') == 'error' for v in manifest.values())
    print(f'Validation summary: count matches={ok}, mismatches={mismatched}, errors={errors}; see manifest.json')


if __name__ == '__main__':
    main()
