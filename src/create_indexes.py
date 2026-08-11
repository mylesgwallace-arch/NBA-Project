import sqlite3

DB_PATH = "data/database/nba.db"

conn = sqlite3.connect(DB_PATH)

indexes = [
    # Games
    ("idx_games_date", "Games", "gameDate"),
    ("idx_games_home", "Games", "hometeamId"),
    ("idx_games_away", "Games", "awayteamId"),
    ("idx_games_type", "Games", "gameType"),

    # Team statistics
    ("idx_teamstats_game", "TeamStatistics", "gameId"),
    ("idx_teamstats_team", "TeamStatistics", "teamId"),
    ("idx_teamstats_date", "TeamStatistics", "gameDate"),

    # Player statistics
    ("idx_playerstats_game", "PlayerStatistics", "gameId"),
    ("idx_playerstats_player", "PlayerStatistics", "personId"),
    ("idx_playerstats_team", "PlayerStatistics", "playerteamId"),

    # Extended player statistics
    ("idx_playerext_game", "PlayerStatisticsExtended", "gameId"),
    ("idx_playerext_player", "PlayerStatisticsExtended", "personId"),
    ("idx_playerext_team", "PlayerStatisticsExtended", "playerteamId"),

    # Extended team statistics
    ("idx_teamext_game", "TeamStatisticsExtended", "gameId"),
    ("idx_teamext_team", "TeamStatisticsExtended", "teamId"),

    # Play-by-play
    ("idx_pbp_game", "PlayByPlay", "gameId"),
    ("idx_pbp_player", "PlayByPlay", "personId"),
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