# NBA Blueprint xFG pipeline (2026–27)

This project extracts the NBA's **publicly displayed shot-level** `expFieldGoalPct` from Blueprint game pages and combines per-game CSVs into one season dataset.

## Install (Mac / Terminal / Cursor)

```bash
python3 -m pip install requests playwright
python3 -m playwright install chromium
```

Save `nba_xfg_pipeline.py` in your project folder.

## 1. Test one game first

```bash
python3 nba_xfg_pipeline.py one --game-id 0022600136 --away ORL --home TOR --headed
```

Blueprint URL: [https://www.nba.com/game/orl-vs-tor-0022600136/blueprint?watchFullGame=true](https://www.nba.com/game/orl-vs-tor-0022600136/blueprint?watchFullGame=true)

The default `--mode auto` tries the raw HTML first, then opens Chromium if it does not find recognizable shots. You can force the browser with `--mode browser`.

If Chrome works but the script cannot fetch the page, you can also test a manually saved source snippet:

```bash
python3 nba_xfg_pipeline.py one --game-id 0022600136 --input copied_source.txt --no-validate
```

You should see:

```
nba_xfg_2026_27/
    games/0022600136.csv
    raw/0022600136.json
    manifest.json
    all_shots.csv
```

Inspect `manifest.json`: `shots_found` and `boxscore_fga` should match. `count_mismatch` means the extracted data is **incomplete or differently defined**, not proof that every shot is available. `unverified` means the NBA boxscore couldn't be accessed. Check the first and last shot clocks too.

## 2. Run a tiny season batch

```bash
python3 nba_xfg_pipeline.py season --from-date 2026-10-28 --max-games 3 --delay 5
```

Review `manifest.json` and `all_shots.csv` before continuing.

## 3. Process all completed games this season

```bash
python3 nba_xfg_pipeline.py season --delay 5
```

The script retrieves game IDs, teams and final statuses from:

[https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json](https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json)

It checks each game's FGA against:

[https://cdn.nba.com/static/json/liveData/boxscore/boxscore_0022600136.json](https://cdn.nba.com/static/json/liveData/boxscore/boxscore_0022600136.json)

The merged output is `nba_xfg_2026_27/all_shots.csv`. Re-running skips games with existing CSVs. Add `--force` to re-fetch all selected games or `--retry-partial` to retry mismatches.

## Data caution and notes

- `ngssEventId` is **not verified** to be the NBA play-by-play `GAME_EVENT_ID` / `actionNumber`. Match on game ID, period, game clock, shooter and shot result, then verify with actual play-by-play.
- `xpts_fg_only = expFieldGoalPct * (2 or 3)`. This does **not** include expected points from free throws, so it is not total possession xPoints.
- `contestDistance` and `contestHeight` are stored using the units shown on the NBA's site (inches); `shotDistance` is feet. Confirm the units against site labels before using them in models.
- Your copied Sources fragment included 112 individual records, all marked `featured: true`. It was not a verified complete-game export. The script may uncover more or fewer records than expected; inspect coverage.
- If the NBA changes its page structure, or restricts automated access, the HTML parser may stop working. Don't bypass access restrictions. Run sequentially, with a delay, and respect NBA website terms.
- The parser was **offline-tested** with representative shot records and embedded Next.js Flight payloads; **live NBA.com retrieval has not been verified** here.

