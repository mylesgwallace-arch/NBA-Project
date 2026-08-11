import sqlite3
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "database" / "nba.db"
OUTPUT_PATH = ROOT / "data" / "processed" / "game_features.csv"

STATS = [
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
PERCENTAGE_STATS = [
    "fieldGoalsPercentage",
    "threePointersPercentage",
    "freeThrowsPercentage",
]
PLAYER_FEATURE = "active_players_rolling_10"
LAST_GAME_PLAYER_FEATURE = "active_players_last_game"


def load_team_games(connection):
    return pd.read_sql_query(
        """
        SELECT team_statistics.gameId, team_statistics.gameDateTimeEst,
               team_statistics.teamId, team_statistics.opponentTeamId,
               team_statistics.home, team_statistics.win,
               team_statistics.teamScore, team_statistics.opponentScore,
               team_statistics.assists, team_statistics.steals,
               team_statistics.blocks, team_statistics.fieldGoalsPercentage,
               team_statistics.threePointersPercentage,
               team_statistics.freeThrowsPercentage,
               team_statistics.reboundsTotal, team_statistics.turnovers,
               team_statistics.plusMinusPoints
        FROM team_statistics
        JOIN games ON games.gameId = team_statistics.gameId
        WHERE COALESCE(team_statistics.gameType, games.gameType) =
              'Regular Season'
        ORDER BY team_statistics.gameDateTimeEst
        """,
        connection,
    )


def load_player_activity(connection):
    return pd.read_sql_query(
        """
        SELECT DISTINCT player_statistics.gameId,
               player_statistics.playerteamId AS teamId,
               player_statistics.personId
        FROM player_statistics
        JOIN games ON games.gameId = player_statistics.gameId
        WHERE COALESCE(player_statistics.gameType, games.gameType) =
              'Regular Season'
          AND CAST(player_statistics.numMinutes AS REAL) > 0
        """,
        connection,
    )


def add_pregame_player_features(team_games, activity):
    activity_by_game = (
        activity.groupby(["teamId", "gameId"])["personId"].agg(set).to_dict()
    )
    values = {}
    last_game_values = {}
    for team_id, games in team_games.groupby("teamId", sort=False):
        previous_players = deque(maxlen=10)
        for row in games.sort_values(["gameDateTimeEst", "gameId"]).itertuples():
            players = set().union(*previous_players) if previous_players else set()
            values[(row.gameId, team_id)] = len(players)
            last_game_values[(row.gameId, team_id)] = (
                len(previous_players[-1]) if previous_players else 0
            )
            previous_players.append(activity_by_game.get((team_id, row.gameId), set()))

    feature = pd.Series(
        [
            values[(game_id, team_id)]
            for game_id, team_id in zip(team_games["gameId"], team_games["teamId"])
        ],
        index=team_games.index,
        name=PLAYER_FEATURE,
    )
    last_game_feature = pd.Series(
        [
            last_game_values[(game_id, team_id)]
            for game_id, team_id in zip(team_games["gameId"], team_games["teamId"])
        ],
        index=team_games.index,
        name=LAST_GAME_PLAYER_FEATURE,
    )
    return team_games.assign(
        **{
            PLAYER_FEATURE: feature,
            LAST_GAME_PLAYER_FEATURE: last_game_feature,
        }
    )


def build_features():
    with sqlite3.connect(DB_PATH) as connection:
        df = load_team_games(connection)
        activity = load_player_activity(connection)

    print(f"Loaded {len(df):,} team-game rows")
    df["gameDateTimeEst"] = pd.to_datetime(df["gameDateTimeEst"])
    df["season"] = df["gameDateTimeEst"].dt.year - (
        df["gameDateTimeEst"].dt.month < 10
    )
    df = df.drop_duplicates(subset=["gameId", "teamId"])
    df = df.sort_values("gameDateTimeEst")
    df["rest_days"] = (
        df.groupby("teamId")["gameDateTimeEst"]
        .diff()
        .dt.total_seconds()
        .div(86400)
    )

    for stat in PERCENTAGE_STATS:
        df.loc[~df[stat].between(0, 1), stat] = np.nan

    for stat in STATS:
        df[f"{stat}_rolling_10"] = (
            df.groupby("teamId")[stat]
            .transform(lambda values: values.shift(1).rolling(10, min_periods=5).mean())
        )

    df = add_pregame_player_features(df, activity)
    rolling_columns = [f"{stat}_rolling_10" for stat in STATS]
    df = df.dropna(subset=STATS + rolling_columns + ["rest_days"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df):,} rows")
    print(f"Saved to: {OUTPUT_PATH}")
    print("\nColumns created:")
    print(df.columns.tolist())


if __name__ == "__main__":
    build_features()
