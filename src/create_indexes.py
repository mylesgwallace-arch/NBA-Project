import sqlite3
from pathlib import Path

# The database uses lowercase/underscore table names (e.g. `team_statistics`,
# not `TeamStatistics`) -- see PROJECT_CONTEXT.md section 6. Always inspect
# the live schema before adding new index targets here.
ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "database" / "nba.db"

conn = sqlite3.connect(DB_PATH)

indexes = [
    # Games
    ("idx_games_game_id", "games", "gameId"),
    ("idx_games_date", "games", "gameDate"),
    ("idx_games_home", "games", "hometeamId"),
    ("idx_games_away", "games", "awayteamId"),
    ("idx_games_type", "games", "gameType"),

    # Team statistics
    ("idx_team_statistics_game_id", "team_statistics", "gameId"),
    ("idx_team_statistics_team", "team_statistics", "teamId"),
    ("idx_team_statistics_date", "team_statistics", "gameDate"),

    # Player statistics
    ("idx_player_statistics_game", "player_statistics", "gameId"),
    ("idx_player_statistics_person", "player_statistics", "personId"),
    ("idx_player_statistics_team", "player_statistics", "playerteamId"),

    # Extended player statistics
    ("idx_player_statistics_extended_game", "player_statistics_extended", "gameId"),
    ("idx_player_statistics_extended_person", "player_statistics_extended", "personId"),
    ("idx_player_statistics_extended_team", "player_statistics_extended", "playerteamId"),

    # Extended team statistics
    ("idx_team_statistics_extended_game", "team_statistics_extended", "gameId"),
    ("idx_team_statistics_extended_team", "team_statistics_extended", "teamId"),
]

for name, table, column in indexes:
    try:
        conn.execute(
            f'CREATE INDEX IF NOT EXISTS "{name}" '
            f'ON "{table}" ("{column}")'
        )
        print(f"Created/verified: {name}")
    except sqlite3.OperationalError as e:
        print(f"SKIPPED {name}: {e}")

conn.commit()

# Verify indexes
print("\nIndexes in database:")
rows = conn.execute("""
    SELECT name, tbl_name
    FROM sqlite_master
    WHERE type = 'index'
    ORDER BY tbl_name, name
""").fetchall()

for name, table in rows:
    print(f"  {table}: {name}")

conn.close()

print("\nDone.")