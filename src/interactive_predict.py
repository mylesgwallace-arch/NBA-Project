"""Minimal interactive interface for the validated NBA game-prediction model.

This module adds no new modeling functionality. It only exposes the existing,
validated ``predict_matchup`` capability from ``src/main.py`` through a simple
team-selection prompt so a user can pick two teams by name and get a
prediction, instead of needing to know numeric team IDs.
"""

import sqlite3
from pathlib import Path

try:
    from src.main import predict_matchup
except ImportError:
    # Support running this file directly (e.g. `python src/interactive_predict.py`).
    from main import predict_matchup


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "database" / "nba.db"

# The database's team_histories table also contains international/exhibition
# rosters (e.g. All-Star teams, EuroLeague clubs) with non-NBA team IDs. Current
# NBA franchise team IDs all fall in this fixed numeric range.
NBA_TEAM_ID_MIN = 1610612737
NBA_TEAM_ID_MAX = 1610612766


def load_current_teams(db_path=DB_PATH):
    """Return the 30 current NBA teams as a list of (teamId, "City Name") tuples."""
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT teamId, teamCity, teamName
            FROM team_histories
            WHERE seasonActiveTill >= 2100
              AND teamId BETWEEN ? AND ?
            ORDER BY teamCity, teamName
            """,
            (NBA_TEAM_ID_MIN, NBA_TEAM_ID_MAX),
        ).fetchall()
    return [(team_id, f"{city} {name}") for team_id, city, name in rows]


def prompt_team_choice(teams, prompt_label):
    """Print a numbered team list and prompt the user to choose one by number."""
    print(f"\n{prompt_label}")
    for index, (_, label) in enumerate(teams, start=1):
        print(f"  {index:2d}. {label}")
    while True:
        choice = input(f"Enter a number for the {prompt_label.lower()}: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(teams):
            return teams[int(choice) - 1]
        print(f"Please enter a number between 1 and {len(teams)}.")


def prompt_optional_game_date():
    game_date = input(
        "\nOptional game date (YYYY-MM-DD), or press Enter to use the latest "
        "available data: "
    ).strip()
    return game_date or None


def run_interactive_session(db_path=DB_PATH):
    teams = load_current_teams(db_path)
    if not teams:
        print("No current NBA teams found in the database.")
        return

    print("NBA Game Prediction (using the validated baseline model)")
    home_team_id, home_label = prompt_team_choice(teams, "Home team")
    away_team_id, away_label = prompt_team_choice(teams, "Away team")

    if home_team_id == away_team_id:
        print("\nHome and away teams must be different. Please restart and choose two different teams.")
        return

    game_date = prompt_optional_game_date()

    try:
        result = predict_matchup(
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            game_date=game_date,
        )
    except ValueError as exc:
        print(f"\nCould not generate a prediction: {exc}")
        return

    print(f"\n{home_label} (home) vs {away_label} (away)")
    if result.get("game_date"):
        print(f"As of: {result['game_date']}")
    print(f"Model used: {result['model']}")
    print(f"{home_label} win probability: {result['home_win_probability']:.1%}")
    print(f"{away_label} win probability: {result['away_win_probability']:.1%}")
    favorite_label = home_label if result["home_team_prediction"] == "favorite" else away_label
    print(f"Predicted favorite: {favorite_label}")


if __name__ == "__main__":
    run_interactive_session()
