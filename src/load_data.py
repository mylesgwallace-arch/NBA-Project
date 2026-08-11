import sqlite3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DB = ROOT / "data" / "database" / "nba.db"


def load_csv(conn, filename, table_name):
    path = RAW / filename

    print(f"\nLoading {filename}...")

    df = pd.read_csv(path, low_memory=False)

    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")

    df.to_sql(
        table_name,
        conn,
        if_exists="replace",
        index=False,
        chunksize=10_000
    )

    print(f"  ✓ Loaded into '{table_name}'")


def main():
    conn = sqlite3.connect(DB)

    files = [
        ("Games.csv", "games"),
        ("Players.csv", "players"),
        ("TeamHistories.csv", "team_histories"),
        ("TeamStatistics.csv", "team_statistics"),
        ("TeamStatisticsExtended.csv", "team_statistics_extended"),
        ("PlayerStatistics.csv", "player_statistics"),
        ("PlayerStatisticsExtended.csv", "player_statistics_extended"),
    ]

    for filename, table_name in files:
        load_csv(conn, filename, table_name)

    conn.close()

    print("\n✓ Database loading complete.")


if __name__ == "__main__":
    main()