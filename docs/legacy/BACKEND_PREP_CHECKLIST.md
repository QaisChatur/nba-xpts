# Backend prep checklist (legacy, Model A era)

> **Legacy.** This describes `scripts/legacy/prepare_backend_assets.py`, which retrains a Model A-style
> model on historical ShotChartDetail and stages CSVs for Postgres. The current product data comes from
> `python -m pipeline export-predictions` and `update-analytics` (see
> [`PRODUCT_ROADMAP.md`](../architecture/PRODUCT_ROADMAP.md) and [`FRONTEND_DATA_CONTRACT.md`](../architecture/FRONTEND_DATA_CONTRACT.md)).
> Kept for reference; the script never overwrites the original Model A.

## 1) Install dependencies

```bash
python -m pip install -r requirements.txt -r scripts/legacy/requirements-db.txt
```

## 2) Required raw inputs (see `docs/architecture/DATA_FILES.md`)

- `data/raw/historical/shotchart_shots.csv`
- `data/raw/historical/playerdashptshots_closest_defender.csv` (optional but recommended)
- `data/raw/historical/playerdashptshots_closest_defender_10ft_plus.csv` (optional but recommended)

## 3) Build the assets

```bash
python scripts/legacy/prepare_backend_assets.py
python scripts/legacy/prepare_backend_assets.py --seasons 2022-23 2023-24 2024-25   # optional scope
```

## 4) Outputs

Processed outputs in `data/processed/legacy/backend_prep/`:

- `xpts_player_specific_predictions_all_games.csv`, `game_flow_all_games.csv`;
- `player_zone_shottype_summary.csv`, `player_action_summary.csv`, `clip_index_all_games.csv`.

Model artifacts in `models/experiments/backend_prep/`, never in `models/legacy/model_a/`:

- `xpts_player_specific_model.pkl`, `xpts_player_specific_model_meta.json`.

DB staging output in `data/processed/legacy/backend_prep/db_stage/` (the other tables load from the files above):

- `games.csv` (tables `shots_xpts`, `game_flow_events` and `player_zone_shottype_summary` are loaded from
  the processed files above; earlier versions wrote byte-identical copies here).

## 5) Clip linking

`clip_event_url` is generated per shot from `GAME_ID`, `GAME_EVENT_ID` and `SEASON`, using the nba.com
event URL pattern. Clip availability varies by event, and video use requires a licensing review.

## 6) Optional: load into Postgres

```bash
psql "$DATABASE_URL" -f scripts/legacy/supabase_schema.sql
export DATABASE_URL='postgresql+psycopg://<user>:<password>@<host>:5432/postgres'   # never commit real values
python scripts/legacy/load_db_stage_to_postgres.py --replace
```
