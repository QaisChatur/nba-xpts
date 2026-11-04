#!/usr/bin/env python3
"""
Load DB-staging CSVs into Postgres/Supabase tables.

Usage:
  export DATABASE_URL='postgresql+psycopg://user:pass@host:5432/postgres'
  python scripts/load_db_stage_to_postgres.py --replace
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_STAGE_DIR = PROJECT_ROOT / "data" / "processed" / "legacy" / "db_stage"

TABLE_MAP = {
    "games.csv": "games",
    "shots_xpts.csv": "shots_xpts",
    "game_flow_events.csv": "game_flow_events",
    "player_zone_shottype_summary.csv": "player_zone_shottype_summary",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load db-stage csv files to Postgres/Supabase.")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Truncate target tables before loading.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set.")

    engine = create_engine(db_url)

    with engine.begin() as conn:
        for filename, table in TABLE_MAP.items():
            path = DB_STAGE_DIR / filename
            if not path.exists():
                print(f"Skipping missing file: {path}")
                continue

            df = pd.read_csv(path)
            if args.replace:
                conn.execute(text(f"truncate table {table};"))
            df.to_sql(table, con=conn, if_exists="append", index=False, method="multi", chunksize=5000)
            print(f"Loaded {len(df):,} rows into {table}")


if __name__ == "__main__":
    main()
