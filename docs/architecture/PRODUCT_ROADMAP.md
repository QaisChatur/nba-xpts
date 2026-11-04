# Product roadmap

Status: data, modelling and analytics layers exist as files. There is no frontend, Supabase, auth or
deployment yet, by design. The architecture keeps four stages separate so a Next.js + FastAPI +
Postgres product can be added without rewriting the pipeline.

| Stage | Code | Output (today) | Future service |
| --- | --- | --- | --- |
| Ingestion | `pipeline/ingestion/` | `data/raw/...` (immutable, cached) | scheduled job (nightly) |
| Training | `pipeline/features/`, `pipeline/training/` | `models/master/<version>/` | offline job, manual promotion |
| Inference | `pipeline/inference.py`, `pipeline/analytics/season_exports.py` | `data/processed/master/predictions/<version>/` | FastAPI endpoint wrapping `predict_shots` |
| Serving analytics | `pipeline/analytics/`, `pipeline/analytics/season_exports.py` | `data/processed/analytics/season/<version>/` | Postgres tables + read API |

## Analytics outputs (built by `python -m pipeline update-analytics`)

* **Game Center:** `game_center/<game_id>.json` (`timeline.v1`) contains:
  - every FGA with Master xFG, NBA xFG (covered shots only) and Model A;
  - cumulative expected vs actual FG points per team, and the official score progression;
  - video-sync references (`official_event_id`, `pbp_action_id`, `ngss_event_id`).
* **Expected Standings:** `expected_standings.json` (`standings.v1`) gives actual vs expected record
  and points per team. It uses deterministic expected score (expected FG points + actual free throws).
  The probabilistic method is experimental. Limitations are embedded in the file.
* **Shot Explorer:** `explorer_shots.parquet` holds the full season shot table with court coordinates
  in feet. `explorer_aggregates.json` has player (≥50 FGA), team and zone tables with points above
  expected.
* **Nightly Highlights:** `nightly_highlights.json` lists, per date, the hardest makes and easiest
  misses by Master xFG (tracked shots only).

Every record carries `master_source` (`master` = tracking model, `fallback_no_tracking` = official FGA
without a Blueprint record) and the model version. NBA xFG is never filled in for shots Blueprint did
not publish. Rows from games in the model-fit partition are in-sample for the final model; the
`partition` column says so.

## Next steps

1. **Nightly job:**
   - `ingest-season --allow-network` (incremental), `coverage-report`, `build-dataset`;
   - `export-predictions` with the promoted model, then `update-analytics`.
2. **Model promotion:** retrain monthly as data grows. Promote a new version only if it beats the
   current one on the newest held-out dates (same protocol); keep all versions.
3. **Postgres schema:**
   - `games`, `shots` (keyed by `shot_uid`), `shot_predictions` (keyed by `shot_uid` and
     `model_version`), `team_game_expected`, `standings_snapshots`;
   - load from the parquet/JSON exports. `scripts/legacy/supabase_schema.sql` is the legacy starting point.
4. **API (FastAPI):**
   - `GET /games/{id}/timeline`, `GET /standings?model=...`, `GET /shots?player=&zone=&date=`,
     `GET /highlights/{date}`;
   - `POST /predict` validated by `predict_shots`.
5. **Frontend (Next.js):** Game Center, Expected Standings, Shot Explorer (court heat maps), Nightly
   Highlights.
6. **Before any public launch:** NBA data, Blueprint data and video require licensing / permission
   review (see LICENSE). Until then the repository and any deployment stay private.
