# Data files: locations, storage policy, restore steps

All paths are relative to the repository root and defined in `pipeline/config.py` (the data root can be
moved with the `XPTS_DATA_DIR` environment variable). `python -m pipeline audit` lists what is present.

## Storage policy

* **Large data stays outside Git.** `data/raw/`, `data/interim/`, `data/processed/`, model binaries
  (`*.pkl`, `*.joblib`, `*.cbm`, `*.ubj`) and `models/experiments/` are git-ignored. The local data
  directory is ~1.2 GB. Git LFS is deliberately not used: the data is third-party (NBA.com /
  Blueprint) and may not be redistributed, so it must not be pushed to any remote.
* **Committed:** code, configs, docs, reports (Markdown/PNG/JSON), model metadata/metrics/schemas, and
  one small test fixture in `data/samples/0022600133/` (one game, 276 KB) used by unit tests and CI.
* **Raw data is immutable.** Ingestion only adds new files; a refetch writes a new fetch-metadata
  record with the page's sha256. Nothing under `data/raw/` is edited in place.
* **Models are versioned.** Every Master training run writes a new `models/master/<version>/`
  directory and refuses to overwrite an existing one. Model A is frozen (sha256-pinned) and the Model A
  training scripts write only to timestamped files under `models/experiments/model_a_retrain/`.

## Layout

| Path | Contents | Restore |
| --- | --- | --- |
| `data/raw/historical/shotchart_shots.csv` | ShotChartDetail shots 2022-23..2024-25 (540,627) | `notebooks/legacy/model_a_original.ipynb` data-collection cells (`nba_api` ShotChartDetail) |
| `data/raw/historical/playerdashptshots_closest_defender*.csv` | PlayerDashPtShots by closest-defender range, 2021-22..2025-26 | same notebook (`nba_api` PlayerDashPtShots) |
| `data/raw/nba_listings/2026-27/<date>.json` | game links found on nba.com/games?date=D (may include other dates' games) | `python -m pipeline ingest-season --allow-network` |
| `data/raw/blueprint/2026-27/games/<game_id>.csv` | Blueprint shots per game (NBA xFG + tracking) | same |
| `data/raw/blueprint/2026-27/raw/<game_id>.json` | verbatim Blueprint shot objects; `<game_id>.fetch_meta.json` = URL, time, HTML sha256, extraction audit | same |
| `data/raw/blueprint/2026-27/season_manifest.json` | per-game status, coverage, errors, timestamps | same (rebuilt offline with `python -m pipeline coverage-report --reverify`) |
| `data/raw/blueprint/2026-27/legacy_scraper/` | `manifest.json`, `all_shots.csv` from the original scraper (Oct 28 sample; `scripts/legacy/nba_xfg_pipeline.py`) | legacy, kept for provenance |
| `data/raw/nba_pages/2026-27/<game_id>.next_data.json.gz` | NBA.com game page payload: box score + play-by-play (stats v3 or NGSS live format) | `ingest-season` / `fetch-pages` |
| `data/processed/legacy/model_a/` | Model A feature matrix (`xpts_features.pkl`) and 2024-25 test predictions, used by `verify-model-a` | `scripts/model_a/run_pipeline.py` |
| `data/processed/legacy/backend_prep/` | Model A-era backend CSVs (all-games predictions, game flow, summaries, clip index, `db_stage/games.csv`) | `scripts/legacy/prepare_backend_assets.py` |
| `data/processed/master/<dataset_version>/` | `master_shots.parquet`, `dataset_meta.json`, `history_assets/` | `python -m pipeline build-dataset` |
| `data/processed/master/predictions/<model_version>/` | `shot_predictions.parquet`, `export_meta.json` | `python -m pipeline export-predictions` |
| `data/processed/analytics/season/<model_version>/` | Game Center timelines, expected standings, Shot Explorer table + aggregates, nightly highlights | `python -m pipeline update-analytics` |
| `data/processed/unified/<date>/`, `data/processed/analytics/<date>/` | single-date outputs of `python -m pipeline run --date D` | same |
| `data/reports/<date>/`, `data/reports/season/` | single-date reports; season coverage report (`COVERAGE_REPORT.md`, `season_coverage.csv`) | `run`, `ingest-season` / `coverage-report` |
| `models/legacy/model_a/` | Model A (`xpts_player_specific_model.pkl`, sha256 `28f33625439cc174…`), metrics, meta, `assets/` (prior tables rebuilt by `build-model-a-assets`) | the `.pkl` is the original artifact (keep a private backup; it is not in Git) |
| `models/legacy/backend_prep_retrain/` | the all-seasons XGBoost written by the legacy backend-prep script (not Model A; protected from overwrite) | `scripts/legacy/prepare_backend_assets.py` |
| `models/legacy/model_b/exploratory_2026-10-28/` | exploratory tracking model from the Oct 28 study | `python -m pipeline run --date 2026-10-28` |
| `models/master/<model_version>/` | Master artifact (see `docs/methodology/MODEL_METHODOLOGY.md`) | `python -m pipeline train-master` |
| `models/experiments/master_runs/<run_id>/` | Optuna trials, predictions, training summary, evaluation JSON; `experiments.jsonl` run log | `train-master` |

## Restoring a clean checkout

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
# 1. historical inputs: copy your private backup of data/raw/historical/ (or rerun the notebook cells)
# 2. Model A artifact: copy models/legacy/model_a/xpts_player_specific_model.pkl from your backup
python -m pipeline build-model-a-assets
# 3. season data (polite, cached, stops on HTTP 403/429)
python -m pipeline ingest-season --allow-network
python -m pipeline coverage-report
# 4. dataset, model, evaluation, exports
python -m pipeline build-dataset
python -m pipeline train-master
python -m pipeline export-predictions && python -m pipeline update-analytics
```

## Key columns (Master Shot Dataset)

| Column | Meaning |
| --- | --- |
| `shot_uid` | `<game_id>_<GAME_EVENT_ID:04d>` (official event number) |
| `GAME_EVENT_ID`, `pbp_actionId`, `ngss_event_id` | official event number, play-by-play action id, Blueprint id (video sync) |
| `made` / `SHOT_MADE_FLAG` | official result (label); Blueprint `success` must agree on matched rows |
| `in_blueprint`, `match_method` | matched one-to-one to a Blueprint record; how (event ID verified or attribute fallback) |
| `nba_xfg`, `nba_xpts_fg` | NBA's published xFG (benchmark only, never a feature) |
| `model_a_xfg` | Model A probability (benchmark only) |
| `baseline_hist_xfg`, `resid_baseline` | historical shot-difficulty baseline and made − baseline (used for priors, not a feature) |
| feature columns | see `configs/features.yaml` groups |
| tracking | `contest_distance_in`, `contest_height_in`, `shot_vertical_in`, `shooter_speed_mph`, `release_time_s` (0 = not measured, capped at 1.0), `shot_clock_s`, `bp_shot_distance_ft` |
