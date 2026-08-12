"""Minimal interactive interface for the validated NBA game-prediction model.

This module adds no new modeling functionality. It only exposes the existing,
validated ``predict_matchup`` capability from ``src/main.py`` through a simple
team-selection prompt so a user can pick two teams by name and get a
prediction, instead of needing to know numeric team IDs.
"""

import argparse
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


def summarize_model_features(result):
    features = result.get("feature_importance", [])
    if not features:
        return None
    lines = ["\nTop model features:"]
    for feature in features:
        lines.append(
            f"  {feature['rank']}. {feature['feature']} ({feature['importance']:.1%})"
        )
    return "\n".join(lines)


def summarize_model_report(result):
    summary = result.get("model_summary")
    if not summary:
        return None
    lines = ["\nModel summary:"]
    model_name = summary.get("recommended_model", "unknown")
    metric_name = summary.get("recommendation_metric", "log_loss")
    lines.append(f"  Recommended model: {model_name} (lowest holdout {metric_name})")
    metrics = summary.get("metrics") or {}
    if metrics:
        accuracy = metrics.get("accuracy")
        if accuracy is not None:
            lines.append(f"  Holdout accuracy: {accuracy:.1%}")
        log_loss = metrics.get("log_loss")
        if log_loss is not None:
            lines.append(f"  Holdout log loss: {log_loss:.4f}")
    calibration = summary.get("calibration") or {}
    ece = calibration.get("expected_calibration_error")
    if ece is not None:
        lines.append(f"  Expected calibration error: {ece:.3f}")
    comparison = summary.get("comparison") or {}
    if comparison:
        lines.append("  Model comparison:")
        for name, values in comparison.items():
            log_loss = values.get("log_loss")
            if log_loss is None:
                continue
            ece = values.get("expected_calibration_error")
            ece_text = f", ECE={ece:.3f}" if ece is not None else ""
            lines.append(f"    - {name}: log_loss={log_loss:.4f}{ece_text}")
    top_features = summary.get("top_features") or []
    if top_features:
        lines.append("  Top model features:")
        for feature in top_features:
            lines.append(
                f"    {feature['rank']}. {feature['feature']} ({feature['importance']:.1%})"
            )
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the interactive NBA prediction interface."
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Print the current top model features after the prediction.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print the recommended model, holdout metrics, and calibration diagnostics after the prediction.",
    )
    return parser.parse_args(argv)


def run_interactive_session(db_path=DB_PATH, explain=False, summary=False):
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
    if summary or explain or result.get("feature_importance") or result.get("model_summary"):
        report = summarize_model_report(result) if summary or result.get("model_summary") else None
        if not report:
            report = summarize_model_features(result)
        if report:
            print(report)


if __name__ == "__main__":
    args = parse_args()
    run_interactive_session(explain=args.explain, summary=args.summary)
