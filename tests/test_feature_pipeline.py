import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "database" / "nba.db"
FEATURES_PATH = ROOT / "data" / "processed" / "game_features.csv"

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


def load_expected_features():
    with sqlite3.connect(DB_PATH) as connection:
        source = pd.read_sql_query(
            """
            SELECT gameId, gameDateTimeEst, teamId, opponentTeamId, home, win,
                   teamScore, opponentScore, assists, steals, blocks,
                   fieldGoalsPercentage, threePointersPercentage,
                   freeThrowsPercentage, reboundsTotal, turnovers,
                   plusMinusPoints
            FROM team_statistics
            WHERE gameType = 'Regular Season'
            ORDER BY gameDateTimeEst
            """,
            connection,
        )

    source["gameDateTimeEst"] = pd.to_datetime(source["gameDateTimeEst"])
    source["season"] = source["gameDateTimeEst"].dt.year - (
        source["gameDateTimeEst"].dt.month < 10
    )
    source = source.drop_duplicates(["gameId", "teamId"])
    source = source.sort_values("gameDateTimeEst")

    for stat in PERCENTAGE_STATS:
        source.loc[~source[stat].between(0, 1), stat] = np.nan

    for stat in STATS:
        source[f"{stat}_rolling_10"] = (
            source.groupby("teamId")[stat]
            .transform(lambda values: values.shift(1).rolling(10, min_periods=5).mean())
        )

    rolling_columns = [f"{stat}_rolling_10" for stat in STATS]
    return source.dropna(subset=STATS + rolling_columns)


def test_generated_features_match_source_and_rolling_history():
    expected = load_expected_features()
    actual = pd.read_csv(FEATURES_PATH)

    assert len(expected) == 129_836
    assert len(actual) == len(expected)
    assert not actual.duplicated(["gameId", "teamId"]).any()
    assert not actual[STATS].isna().any().any()
    assert not actual[[f"{stat}_rolling_10" for stat in STATS]].isna().any().any()
    for stat in PERCENTAGE_STATS:
        assert actual[stat].between(0, 1).all()
    np.testing.assert_array_equal(actual["season"].to_numpy(), expected["season"].to_numpy())

    expected = expected.set_index(["gameId", "teamId"])
    actual = actual.set_index(["gameId", "teamId"])
    assert actual.index.equals(expected.index)

    for stat in STATS:
        column = f"{stat}_rolling_10"
        np.testing.assert_allclose(
            actual[column].to_numpy(),
            expected[column].to_numpy(),
            rtol=1e-12,
            atol=1e-12,
        )
