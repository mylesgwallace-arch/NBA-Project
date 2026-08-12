from src.interactive_predict import load_current_teams


def test_load_current_teams_returns_thirty_current_nba_franchises(tmp_path):
    import sqlite3

    db_path = tmp_path / "nba.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE team_histories (
                teamId INTEGER, teamCity TEXT, teamName TEXT,
                teamAbbrev TEXT, seasonFounded INTEGER,
                seasonActiveTill INTEGER, league TEXT
            )
            """
        )
        connection.executemany(
            "INSERT INTO team_histories VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (1610612737, "Atlanta", "Hawks", "ATL", 1968, 2100, "NBA"),
                (1610612738, "Boston", "Celtics", "BOS", 1946, 2100, "NBA"),
                # Historical (no longer active) entry for the same franchise.
                (1610612737, "St. Louis", "Hawks", "STL", 1955, 1967, "NBA"),
                # Non-NBA / international team outside the franchise ID range.
                (9027, "Adelaide", "36ers", "ADL", 1990, 2100, "NBL"),
            ],
        )
        connection.commit()

    teams = load_current_teams(db_path)

    assert teams == [
        (1610612737, "Atlanta Hawks"),
        (1610612738, "Boston Celtics"),
    ]
