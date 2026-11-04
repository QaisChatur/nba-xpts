# Frontend Data Contract (v1)

This document defines exactly what the frontend should read and how each page/component maps to backend-prepared data.

Scope:
- Game Flow page (actual vs cumulative xPTS)
- Shot Explorer page (zone + shot type + player filtering)
- Clip linking from shot rows

Primary source files are produced by `scripts/prepare_backend_assets.py`.

---

## 1) Data Sources

### 1.1 Per-shot predictions
File: `data/processed/xpts_player_specific_predictions_all_games.csv`

Required columns:
- `GAME_ID`
- `GAME_EVENT_ID`
- `GAME_DATE`
- `PLAYER_ID`
- `PLAYER_NAME`
- `TEAM_ID`
- `TEAM_NAME`
- `SEASON`
- `PERIOD`
- `MINUTES_REMAINING`
- `SECONDS_REMAINING`
- `seconds_elapsed_in_game`
- `SHOT_TYPE`
- `ACTION_TYPE`
- `SHOT_ZONE_BASIC`
- `SHOT_ZONE_AREA`
- `SHOT_ZONE_RANGE`
- `SHOT_DISTANCE`
- `LOC_X`
- `LOC_Y`
- `SHOT_MADE_FLAG`
- `shot_value`
- `actual_points`
- `xfg_pct_player_specific`
- `xpts_player_specific`
- `points_above_expected`
- `distance_bin`
- `is_deep_3`
- `is_corner_3`
- `is_restricted_area`
- `is_midrange`
- `is_paint_non_ra`
- `player_zone_action_fg_smoothed`
- `player_distance_bin_fg_smoothed`
- `player_action_fg_smoothed`
- `player_overall_fg_smoothed`
- `clip_event_url`

---

### 1.2 Game flow events (precomputed cumulative overlays)
File: `data/processed/game_flow_all_games.csv`

Contains all per-shot fields above plus:
- `actual_cum_points_team`
- `xpts_cum_team`

Sorting contract (must preserve in UI):
1. `PERIOD` ascending
2. `seconds_elapsed_in_game` ascending
3. `GAME_EVENT_ID` ascending

---

### 1.3 Zone/shot-type summary for explorer
File: `data/processed/player_zone_shottype_summary.csv`

Required columns:
- `SEASON`
- `PLAYER_ID`
- `PLAYER_NAME`
- `TEAM_NAME`
- `SHOT_TYPE`
- `SHOT_ZONE_BASIC`
- `shots`
- `makes`
- `actual_points`
- `expected_points`
- `avg_xfg`
- `fg_pct`
- `actual_points_per_shot`
- `xpts_per_shot`
- `points_above_expected`
- `points_above_expected_per_shot`

---

### 1.4 Action summary (optional page/widget)
File: `data/processed/player_action_summary.csv`

Same metric schema as zone summary but grouped by `ACTION_TYPE`.

---

### 1.5 Clip index (optional if using direct shot table URL)
File: `data/processed/clip_index_all_games.csv`

Required columns:
- `GAME_ID`
- `GAME_EVENT_ID`
- `SEASON`
- `PLAYER_NAME`
- `TEAM_NAME`
- `SHOT_TYPE`
- `ACTION_TYPE`
- `clip_event_url`
- `clip_status`

If this file is not explicitly used, `clip_event_url` from prediction rows is sufficient.

---

## 2) Page Contracts

## 2.1 Game Flow Page

### 2.1.1 Controls
- Season selector (`SEASON`)
- Game selector (`GAME_ID`, display text can include teams/date)
- Team visibility toggles (2 teams)
- Optional period filter

### 2.1.2 Main chart
Input: `game_flow_all_games.csv` filtered by `GAME_ID`

Series per team:
- `actual_cum_points_team`
- `xpts_cum_team`

X-axis:
- `seconds_elapsed_in_game` (or derived human-readable clock)

### 2.1.3 Shot table below chart
Input: filtered game rows

Columns to display (minimum):
- `GAME_EVENT_ID`
- `PERIOD`
- `PLAYER_NAME`
- `TEAM_NAME`
- `SHOT_TYPE`
- `ACTION_TYPE`
- `SHOT_ZONE_BASIC`
- `SHOT_DISTANCE`
- `SHOT_MADE_FLAG`
- `xfg_pct_player_specific`
- `xpts_player_specific`
- `actual_points`
- `points_above_expected`
- `clip_event_url`

### 2.1.4 Table interactions
- Clicking row highlights corresponding chart point
- Clip button opens `clip_event_url` in new tab
- If URL missing, show "Clip unavailable"

---

## 2.2 Shot Explorer Page

### 2.2.1 Filters
Mandatory:
- `SEASON` (single or multi)
- `SHOT_ZONE_BASIC` (single + multi select)
- `SHOT_TYPE` (2PT/3PT)
- `PLAYER_NAME` (searchable)

Optional:
- `TEAM_NAME`
- Minimum attempts threshold (`shots >= N`)

### 2.2.2 Primary data source
`player_zone_shottype_summary.csv`

### 2.2.3 Leaderboard columns
- `PLAYER_NAME`
- `TEAM_NAME`
- `SHOT_ZONE_BASIC`
- `SHOT_TYPE`
- `shots`
- `fg_pct`
- `avg_xfg`
- `actual_points_per_shot`
- `xpts_per_shot`
- `points_above_expected_per_shot`
- `points_above_expected`

### 2.2.4 Sorting defaults
Recommended default: `points_above_expected_per_shot` descending
Common alternates: `shots` descending, `avg_xfg` descending

---

## 2.3 Court Zone Filter Component

This component filters explorer results by `SHOT_ZONE_BASIC`.

Zone mapping contract:
- Restricted Area -> `Restricted Area`
- Paint Non-RA -> `In The Paint (Non-RA)`
- Midrange -> `Mid-Range`
- Left Corner 3 -> `Left Corner 3`
- Right Corner 3 -> `Right Corner 3`
- Above Break 3 -> `Above the Break 3`
- Backcourt -> `Backcourt` (optional, rare)

Behavior:
- Single click = select one zone
- Shift/cmd click (or toggle mode) = multi-select
- "Clear zones" resets all

---

## 3) Metric Definitions (frontend text/tooltips)

- `xfg_pct_player_specific`: Predicted make probability for this shot context and player profile.
- `xpts_player_specific`: `xfg_pct_player_specific * shot_value`
- `points_above_expected`: `actual_points - xpts_player_specific`
- `xpts_per_shot`: `expected_points / shots`
- `actual_points_per_shot`: `actual_points / shots`

Note to display in methodology tooltip:
"This model is public-data based and does not include shot-level tracking defender contest data."

---

## 4) Data Types & Parsing Guidance

- IDs (`GAME_ID`, `PLAYER_ID`, `TEAM_ID`) should be treated as strings in frontend state to avoid numeric precision issues.
- Numeric metrics should be parsed as floats.
- `SHOT_MADE_FLAG`, booleans, and indicator flags can be parsed as integers.
- Preserve `GAME_EVENT_ID` as integer-like string for display and clip linking.

---

## 5) Performance Guidance (frontend)

- Load CSV once per page and cache in memory.
- Pre-index rows by `GAME_ID` for fast game switching.
- Precompute dropdown options (`SEASON`, `TEAM_NAME`, `PLAYER_NAME`) on first load.
- For large tables, use pagination or virtualization.

---

## 6) Validation Checklist (frontend handoff)

Before marking UI complete:
- Can pick season and game and render both cumulative lines.
- Shot table rows match selected game and are correctly ordered.
- Clip links open with `GAME_ID` + `GAME_EVENT_ID`.
- Explorer filters update leaderboard correctly.
- Zone filter UI maps to exact `SHOT_ZONE_BASIC` values.
- Methodology caveat is visible on page.

