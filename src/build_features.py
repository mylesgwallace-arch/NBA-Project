import sqlite3
import pandas as pd
from pathlib import Path

import numpy as np

DB_PATH = Path("data/database/nba.db")
OUTPUT_PATH = Path("data/processed/game_features.csv")

conn = sqlite3.connect(DB_PATH)

df = pd.read_sql_query("""
    SELECT
        team_statistics.gameId,
        team_statistics.gameDateTimeEst,
        team_statistics.teamId,
        team_statistics.opponentTeamId,
        team_statistics.home,
        team_statistics.win,
        team_statistics.teamScore,
        team_statistics.opponentScore,
        team_statistics.assists,
        team_statistics.steals,
        team_statistics.blocks,
        team_statistics.fieldGoalsPercentage,
        team_statistics.threePointersPercentage,
        team_statistics.freeThrowsPercentage,
        team_statistics.reboundsTotal,
        team_statistics.turnovers,
        team_statistics.plusMinusPoints
    FROM team_statistics
    JOIN games ON games.gameId = team_statistics.gameId
    WHERE COALESCE(team_statistics.gameType, games.gameType) = 'Regular Season'
    ORDER BY team_statistics.gameDateTimeEst
""", conn)

conn.close()

print(f"Loaded {len(df):,} team-game rows")

# Make sure dates are properly formatted
df["gameDateTimeEst"] = pd.to_datetime(df["gameDateTimeEst"])

# Label each season by the calendar year in which it starts.
df["season"] = df["gameDateTimeEst"].dt.year - (df["gameDateTimeEst"].dt.month < 10)

# Remove duplicate team-game records if any
df = df.drop_duplicates(subset=["gameId", "teamId"])

# Sort chronologically
df = df.sort_values("gameDateTimeEst")

# Create rolling averages using ONLY previous games
stats = [
    "teamScore",
    "opponentScore",
    "assists",
    "steals",
    "blocks",
    "fieldGoalsPercentage",
    "threePointersPercentage",
    "freeThrowsPercentage",
    "reboundsTotal",
    "turnovers",
    "plusMinusPoints",
]

# Legacy records contain malformed percentage encodings and a few missing metrics.
# Mark those values unusable rather than attempting an unsupported reconstruction.
percentage_stats = [
    "fieldGoalsPercentage",
    "threePointersPercentage",
    "freeThrowsPercentage",
]
for stat in percentage_stats:
    df.loc[~df[stat].between(0, 1), stat] = np.nan

for stat in stats:
    df[f"{stat}_rolling_10"] = (
        df.groupby("teamId")[stat]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=5).mean())
    )

# Keep only rows with complete current-game and rolling predictors.
rolling_columns = [f"{stat}_rolling_10" for stat in stats]
df = df.dropna(subset=stats + rolling_columns)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print(f"Saved {len(df):,} rows")
print(f"Saved to: {OUTPUT_PATH}")
print("\nColumns created:")
print(df.columns.tolist())