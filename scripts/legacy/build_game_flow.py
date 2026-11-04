"""Build app-ready per-game timelines of actual points vs. player-specific xPts.

Reads the shot-level predictions, attaches game clock info from the raw shot
chart, computes per-team running totals, and writes:
  data/processed/game_flow/<GAME_ID>.csv   (one file per game)
  data/processed/game_flow_all.csv         (all games stacked)
"""

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_PATH = ROOT / "data/processed/legacy/xpts_player_specific_predictions.csv"
SHOTCHART_PATH = ROOT / "data/raw/historical/shotchart_shots.csv"
OUT_DIR = ROOT / "data/processed/legacy/game_flow"
OUT_ALL = ROOT / "data/processed/legacy/game_flow_all.csv"

KEY = ["GAME_ID", "GAME_EVENT_ID"]
SORT_COLS = ["GAME_ID", "PERIOD", "seconds_elapsed_in_game", "GAME_EVENT_ID"]

REGULATION_PERIODS = 4
REGULATION_PERIOD_SECONDS = 12 * 60
OT_PERIOD_SECONDS = 5 * 60

OUTPUT_COLUMNS = [
    "GAME_ID",
    "GAME_EVENT_ID",
    "PERIOD",
    "seconds_elapsed_in_game",
    "TEAM_ID",
    "TEAM_NAME",
    "PLAYER_ID",
    "PLAYER_NAME",
    "SHOT_TYPE",
    "SHOT_MADE_FLAG",
    "actual_points",
    "xpts_player_specific",
    "actual_cum_points_team",
    "xpts_cum_team",
]


def period_length(period: pd.Series) -> pd.Series:
    return period.le(REGULATION_PERIODS).map(
        {True: REGULATION_PERIOD_SECONDS, False: OT_PERIOD_SECONDS}
    )


def period_start_seconds(period: pd.Series) -> pd.Series:
    reg_done = period.sub(1).clip(upper=REGULATION_PERIODS)
    ot_done = period.sub(1 + REGULATION_PERIODS).clip(lower=0)
    return reg_done * REGULATION_PERIOD_SECONDS + ot_done * OT_PERIOD_SECONDS


def load_inputs() -> pd.DataFrame:
    preds = pd.read_csv(
        PREDICTIONS_PATH,
        usecols=[
            "GAME_ID", "GAME_EVENT_ID", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID",
            "TEAM_NAME", "SHOT_TYPE", "SHOT_MADE_FLAG", "actual_points",
            "xpts_player_specific",
        ],
    )
    clock = pd.read_csv(
        SHOTCHART_PATH,
        usecols=KEY + ["PLAYER_ID", "PERIOD", "MINUTES_REMAINING", "SECONDS_REMAINING"],
    )

    for name, df in [("predictions", preds), ("shotchart", clock)]:
        dupes = df.duplicated(KEY).sum()
        assert dupes == 0, f"{name}: {dupes} duplicate (GAME_ID, GAME_EVENT_ID) rows"

    merged = preds.merge(
        clock, on=KEY, how="left", suffixes=("", "_clock"), validate="one_to_one"
    )
    missing = merged["PERIOD"].isna().sum()
    assert missing == 0, f"{missing} predictions have no matching shotchart clock row"
    mismatched = (merged["PLAYER_ID"] != merged["PLAYER_ID_clock"]).sum()
    assert mismatched == 0, f"{mismatched} rows matched a different PLAYER_ID in shotchart"
    return merged.drop(columns="PLAYER_ID_clock")


def build_game_flow(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["PERIOD"] = df["PERIOD"].astype(int)
    remaining = df["MINUTES_REMAINING"] * 60 + df["SECONDS_REMAINING"]
    elapsed_in_period = period_length(df["PERIOD"]) - remaining
    df["seconds_elapsed_in_game"] = (period_start_seconds(df["PERIOD"]) + elapsed_in_period).astype(int)

    df = df.sort_values(SORT_COLS, kind="mergesort").reset_index(drop=True)

    by_team = df.groupby(["GAME_ID", "TEAM_ID"], sort=False)
    df["actual_cum_points_team"] = by_team["actual_points"].cumsum()
    df["xpts_cum_team"] = by_team["xpts_player_specific"].cumsum()
    return df[OUTPUT_COLUMNS]


def sanity_check(flow: pd.DataFrame, source: pd.DataFrame) -> None:
    assert len(flow) == len(source), "row count changed during build"

    dupes = flow.duplicated(KEY).sum()
    assert dupes == 0, f"{dupes} duplicate events in game flow"

    assert flow["PERIOD"].between(1, 10).all(), "PERIOD out of expected range"
    period_end = period_start_seconds(flow["PERIOD"]) + period_length(flow["PERIOD"])
    in_period = flow["seconds_elapsed_in_game"].between(period_start_seconds(flow["PERIOD"]), period_end)
    assert in_period.all(), f"{(~in_period).sum()} rows have elapsed time outside their period"

    expected_order = flow.sort_values(SORT_COLS, kind="mergesort").index
    assert (expected_order == flow.index).all(), "rows are not sorted by PERIOD/time/GAME_EVENT_ID"

    g = flow.groupby("GAME_ID", sort=False)
    assert g["seconds_elapsed_in_game"].apply(lambda s: s.is_monotonic_increasing).all(), \
        "seconds_elapsed_in_game decreases within a game"

    team_g = flow.groupby(["GAME_ID", "TEAM_ID"], sort=False)
    assert team_g["actual_cum_points_team"].apply(lambda s: s.is_monotonic_increasing).all(), \
        "actual cumulative points decrease within a team"
    assert team_g["xpts_cum_team"].apply(lambda s: s.is_monotonic_increasing).all(), \
        "xPts cumulative decreases within a team"

    finals = team_g[["actual_cum_points_team", "xpts_cum_team"]].last()
    totals = source.groupby(["GAME_ID", "TEAM_ID"])[["actual_points", "xpts_player_specific"]].sum()
    finals = finals.join(totals)
    assert (finals["actual_cum_points_team"] == finals["actual_points"]).all(), \
        "final actual cumulative != team total"
    assert ((finals["xpts_cum_team"] - finals["xpts_player_specific"]).abs() < 1e-6).all(), \
        "final xPts cumulative != team total"

    teams_per_game = g["TEAM_ID"].nunique()
    if (teams_per_game != 2).any():
        print(f"WARNING: {(teams_per_game != 2).sum()} games do not have exactly 2 teams")

    # Event IDs are not guaranteed to be chronological in NBA data; report, don't fail.
    out_of_order = g["GAME_EVENT_ID"].apply(lambda s: (s.diff() < 0).sum()).sum()
    if out_of_order:
        print(f"NOTE: {out_of_order} rows where GAME_EVENT_ID decreases vs. the previous "
              "shot in clock order (event IDs are not strictly chronological)")

    print(f"Sanity checks passed: {len(flow):,} shots across {flow['GAME_ID'].nunique():,} games")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-per-game", action="store_true", help="skip per-game CSVs")
    parser.add_argument("--no-all", action="store_true", help="skip combined CSV")
    parser.add_argument("--sample-game", help="GAME_ID to print as a sample (default: first game)")
    args = parser.parse_args()

    source = load_inputs()
    flow = build_game_flow(source)
    sanity_check(flow, source)

    if not args.no_all:
        flow.to_csv(OUT_ALL, index=False)
        print(f"Wrote {OUT_ALL.relative_to(ROOT)}")

    if not args.no_per_game:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for game_id, game in flow.groupby("GAME_ID", sort=True):
            game.to_csv(OUT_DIR / f"{game_id}.csv", index=False)
        print(f"Wrote {flow['GAME_ID'].nunique():,} files to {OUT_DIR.relative_to(ROOT)}/")

    sample_id = int(args.sample_game) if args.sample_game else flow["GAME_ID"].iloc[0]
    sample = flow[flow["GAME_ID"] == sample_id]
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(f"\nSample game {sample_id} ({len(sample)} shots) - first 12 rows:")
        print(sample.head(12).to_string(index=False))
        print("\nFinal team totals:")
        print(sample.groupby("TEAM_NAME")[["actual_cum_points_team", "xpts_cum_team"]].last().round(2))


if __name__ == "__main__":
    main()
