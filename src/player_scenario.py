"""Small, descriptive single-player scenario layer for the validated NBA model.

This module intentionally does not add any new predictive model or inject player
impact directly into the production feature set. The current prediction pipeline
uses team-level rolling features and an Elo gap; the descriptive player-impact
estimate remains a separate, assumption-labeled diagnostic.
"""

import argparse
import json
from pathlib import Path

try:
    from src.main import FEATURES_PATH, METRICS_PATH, MODEL_PATH, predict_matchup, resolve_team_name_to_id
    from src.player_impact import summarize_player_impact
except ImportError:  # pragma: no cover - direct-script support
    from main import FEATURES_PATH, METRICS_PATH, MODEL_PATH, predict_matchup, resolve_team_name_to_id
    from player_impact import summarize_player_impact

ROOT = Path(__file__).resolve().parents[1]


def explain_feature_translation_limit():
    """Document why the descriptive player impact estimate is not a model feature."""
    return {
        "status": "unsupported",
        "reason": (
            "The validated production model uses team-level rolling signals and an "
            "Elo delta, not a player-to-team impact feature. The existing "
            "player-impact diagnostic is a minutes-weighted net-rating estimate on a "
            "different scale and with different assumptions; translating it into the "
            "model feature set would require unsupported, ad hoc conversion rules."
        ),
        "model_features_considered": [
            "win_rate_rolling_10",
            "plusMinusPoints_rolling_10",
            "rest_days",
            "active_players_last_game",
            "elo_delta",
        ],
    }


def analyze_single_player_scenario(
    home_team_id,
    away_team_id,
    person_id,
    game_date=None,
    window=10,
    features_path=None,
    model_path=None,
    metrics_path=None,
):
    """Return a descriptive scenario readout without altering the model.

    The model probability remains the production baseline (`elo_boosted_ensemble`),
    while the player-impact estimate is appended as an explanatory diagnostic with
    an explicit note that it is not used to modify the feature matrix.
    """
    base_result = predict_matchup(
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        game_date=game_date,
        features_path=features_path or FEATURES_PATH,
        model_path=model_path or MODEL_PATH,
        metrics_path=metrics_path or METRICS_PATH,
    )
    player_impact = summarize_player_impact(person_id, before=game_date, window=window)
    translation = explain_feature_translation_limit()
    summary = (
        "The current production model remains the validated `elo_boosted_ensemble` "
        "baseline. The player-impact estimate is included only as a descriptive "
        "diagnostic and is not translated into the model's feature matrix or used "
        "to change the score probability."
    )
    return {
        "home_team_id": int(home_team_id),
        "away_team_id": int(away_team_id),
        "game_date": base_result.get("game_date"),
        "model": base_result.get("model"),
        "base_prediction": {
            "home_win_probability": base_result.get("home_win_probability"),
            "away_win_probability": base_result.get("away_win_probability"),
            "favorite": base_result.get("home_team_prediction"),
        },
        "player_id": int(person_id),
        "player_impact": player_impact,
        "feature_translation": translation,
        "scenario_summary": summary,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run a descriptive single-player scenario analysis without altering the "
            "validated production prediction model."
        )
    )
    parser.add_argument("--home-team-id", type=int, help="Numeric NBA teamId for the home team.")
    parser.add_argument("--away-team-id", type=int, help="Numeric NBA teamId for the away team.")
    parser.add_argument("--home-team", type=str, help="Current NBA franchise name or city.")
    parser.add_argument("--away-team", type=str, help="Current NBA franchise name or city.")
    parser.add_argument("--person-id", type=int, required=True, help="Player to evaluate descriptively.")
    parser.add_argument("--game-date", type=str, help="Optional cutoff date in YYYY-MM-DD format.")
    parser.add_argument("--window", type=int, default=10, help="Recent-game window for the player-impact estimate.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    home_team_id = args.home_team_id
    if home_team_id is None:
        home_team_id = resolve_team_name_to_id(args.home_team)
    away_team_id = args.away_team_id
    if away_team_id is None:
        away_team_id = resolve_team_name_to_id(args.away_team)
    result = analyze_single_player_scenario(
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        person_id=args.person_id,
        game_date=args.game_date,
        window=args.window,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
