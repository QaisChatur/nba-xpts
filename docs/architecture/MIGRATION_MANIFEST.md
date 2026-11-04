# Repository migration manifest

Record of the 2026-11-04 repository cleanup: what moved, what was renamed, what was archived or removed, and
why. The baseline is commit `5d1d02e`; any removed tracked file can be recovered with
`git show 5d1d02e:<old path>`.

## Inventory and classification (before cleanup)

| Class | Items |
| --- | --- |
| Pipeline source | `pipeline/**` (ingestion, matching, features, training, analytics, legacy, evaluation, CLI) |
| Active configuration | `configs/features.yaml`, `training.yaml`, `evaluation.yaml` |
| Active model artifact | `models/master/master_master_v1_20261104T210006/` (Master V1) |
| Legacy model / notebook | Model A pickle and assets, a second pickle written by the backend-prep retrain, Model B metadata, `notebooks/original/xpts.ipynb` |
| Raw data | `data/raw/blueprint/2026-27/`, `data/raw/nba_pages/`, `data/raw/historical/` |
| Processed data | `data/processed/{master,unified,analytics,legacy}/` |
| Generated reports | `data/reports/**`, `docs/reports/**` |
| Test fixture | `data/samples/0022600133/` |
| Temporary / obsolete | aborted run `models/experiments/master_runs/20261104T205615/`, `.DS_Store` files |
| Duplicates (sha256-identical) | 3 `db_stage/` CSVs (480 MB) identical to backend-prep CSVs; history assets copied into the model artifact (intentional, see below); data samples identical to raw files (intentional fixture) |
| Misnamed | `data/raw/blueprint/2026-27/schedule_days/` (NBA.com listings, not Blueprint); `*.prepare_backend_assets.bak.pkl` (not a backup of Model A) |

## Tracked files (moved with `git mv`, history preserved)

| Old location | New location | Reason |
| --- | --- | --- |
| `docs/MODEL_METHODOLOGY.md` | `docs/methodology/MODEL_METHODOLOGY.md` | docs grouped by purpose |
| `docs/DATA_MATCHING.md` | `docs/methodology/DATA_MATCHING.md` | 〃 |
| `docs/DATA_FILES.md` | `docs/architecture/DATA_FILES.md` | 〃 |
| `docs/FRONTEND_DATA_CONTRACT.md` | `docs/architecture/FRONTEND_DATA_CONTRACT.md` | 〃 |
| `docs/PRODUCT_ROADMAP.md` | `docs/architecture/PRODUCT_ROADMAP.md` | 〃 |
| `docs/BACKEND_PREP_CHECKLIST.md` | `docs/legacy/BACKEND_PREP_CHECKLIST.md` | describes a legacy script |
| `notebooks/original/xpts.ipynb` | `notebooks/legacy/model_a_original.ipynb` | descriptive name; local absolute paths in saved outputs replaced by `<project_root>` (9 strings, nothing else changed) |
| `sql/supabase_schema.sql` | `scripts/legacy/supabase_schema.sql` | legacy-only; removes a one-file top-level dir |
| `requirements-db.txt` | `scripts/legacy/requirements-db.txt` | only the legacy DB loader needs it |
| `pipeline/training/evaluate.py` | `pipeline/evaluation/master_report.py` | evaluation code lives in `pipeline/evaluation/` |
| `pipeline/run.py` | `pipeline/legacy/nightly.py` | the single-night Model A vs NBA xFG pipeline is legacy |
| `pipeline/exports.py` | `pipeline/analytics/season_exports.py` | season predictions + analytics exports belong to analytics |

## Tracked files removed

| File | Reason | Recovery |
| --- | --- | --- |
| `scripts/legacy/build_game_flow.py` | duplicate: game-flow tables are produced by `scripts/legacy/prepare_backend_assets.py` (`game_flow_all_games.csv`) and superseded by Game Center timelines; its input path no longer existed | `git show 5d1d02e:scripts/legacy/build_game_flow.py` |

## Local (git-ignored) data and models

| Old location | New location / action | Verification |
| --- | --- | --- |
| `data/processed/legacy/db_stage/shots_xpts.csv` | **removed** (duplicate) | sha256 identical to `xpts_player_specific_predictions_all_games.csv` |
| `data/processed/legacy/db_stage/game_flow_events.csv` | **removed** (duplicate) | identical to `game_flow_all_games.csv` |
| `data/processed/legacy/db_stage/player_zone_shottype_summary.csv` | **removed** (duplicate) | identical to `player_zone_shottype_summary.csv` |
| `data/processed/legacy/db_stage/games.csv` | `data/processed/legacy/backend_prep/db_stage/games.csv` | unique file |
| `data/processed/legacy/{clip_index_all_games,game_flow_all_games,player_action_summary,player_zone_shottype_summary,xpts_player_specific_predictions_all_games}.csv` | `data/processed/legacy/backend_prep/` | moved |
| `data/processed/legacy/{xpts_features.pkl,xpts_player_specific_predictions.csv}` | `data/processed/legacy/model_a/` | `verify-model-a` passes from the new path |
| `data/raw/blueprint/2026-27/schedule_days/` | `data/raw/nba_listings/2026-27/` | NBA.com games listings, not Blueprint |
| `data/raw/blueprint/2026-27/{manifest.json,all_shots.csv}` | `data/raw/blueprint/2026-27/legacy_scraper/` | original scraper output (provenance only) |
| `data/raw/nba_pages/<id>.*` | `data/raw/nba_pages/2026-27/<id>.*` | season-scoped like Blueprint |
| `models/legacy/model_a/xpts_player_specific_model.prepare_backend_assets.bak.pkl` | `models/legacy/backend_prep_retrain/xpts_player_specific_model.pkl` | it is the all-seasons model written by `prepare_backend_assets.py`, not a Model A backup; still overwrite-protected |
| `models/experiments/master_runs/20261104T205615/` | **removed** | training run killed after 2 model families when an action-type mapping bug was found; never registered in `experiments.jsonl`; superseded by run `20261104T210006` |
| `**/.DS_Store` | **removed** | OS metadata |

The legacy script `prepare_backend_assets.py` no longer writes duplicate copies, and
`load_db_stage_to_postgres.py` loads each table from its single canonical file.

## Intentional duplicates (kept)

* `models/master/<version>/history_assets/` duplicates `data/processed/master/<dataset>/history_assets/`.
  A model artifact must be self-contained (inference must not depend on a mutable data directory).
* `data/samples/0022600133/` duplicates one game's raw files, so tests and CI run without the data
  directory.
* `models/master/<version>/training_summary.json` duplicates the run directory's copy; the run
  directory is git-ignored, the artifact copy is committed.

## Code and configuration changes that accompany the moves

* `pipeline/config.py`: `GAMES_LISTING_DIR` (replaces `BLUEPRINT_SCHEDULE_DIR`), season-scoped
  `GAME_PAGES_DIR`, new legacy paths, `BACKEND_PREP_MODEL_PATH`.
* `configs/serving.yaml` (new): the active (served) model is named explicitly. Previously the
  alphabetically last directory was used, which would have silently served any new experiment.
* `python -m pipeline verify-master` (new): re-scores the active model's test set from the artifact and
  checks predictions and metrics against the stored run. After the cleanup: prediction difference 0,
  metric difference 0.
* The CLI, tests, docs and `.gitignore` were updated to the new paths. `.gitignore` now also excludes
  XGBoost/LightGBM text model files and parquet under `models/`.

## Not changed (on purpose)

* The Master V1 directory name `master_master_v1_20261104T210006` (the doubled prefix came from the old
  naming code, since fixed). It is the immutable model version recorded in metadata, exported
  predictions and reports, so it is not renamed.
* `models/legacy/model_a/xpts_player_specific_model_meta.json` still lists the backend-prep output paths
  as they were when it was written; it is a historical record of a frozen artifact.
* `data/reports/2026-10-28/` (verified single-night reports) is kept verbatim.
