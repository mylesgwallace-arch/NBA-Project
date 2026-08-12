import argparse
import json
import math
import pickle
from pathlib import Path

import pandas as pd

try:
    from src.train_baseline_model import build_game_dataset, elo_win_probability
except ImportError:
    # Support running this file directly (e.g. `python src/main.py`), where
    # `src` is not importable as a package because the script's own directory
    # is placed on sys.path instead of the repository root.
    from train_baseline_model import build_game_dataset, elo_win_probability


ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "data" / "processed" / "game_features.csv"
MODEL_PATH = ROOT / "models" / "baseline_logistic.pkl"
METRICS_PATH = ROOT / "models" / "baseline_metrics.json"


def load_model(model_path=MODEL_PATH):
    with model_path.open("rb") as handle:
        model_bundle = pickle.load(handle)
    return model_bundle["model"], model_bundle["predictors"]


def load_recommended_model_name(metrics_path=METRICS_PATH, default="boosted_hybrid"):
    """Read which model the validated holdout comparison recommends.

    Falls back to the strongest current candidate if no metrics file is present,
    so the CLI still works before `train_baseline_model.py` has been run.
    """
    if not metrics_path.exists():
        return default
    metadata = json.loads(metrics_path.read_text(encoding="utf-8"))
    return metadata.get("recommended_model", default)


def load_elo_config(metrics_path=METRICS_PATH):
    metadata = json.loads(metrics_path.read_text(encoding="utf-8"))
    return metadata["elo"]


def compute_elo_ratings_as_of(games, cutoff=None, initial_rating=1500.0, k_factor=20.0, home_advantage=65.0):
    """Compute Elo ratings from only the games that occurred before ``cutoff``.

    ``games`` must already be sorted chronologically (as returned by
    ``build_game_dataset``), so the scan can stop at the first game on or
    after the cutoff without looking at any information the CLI caller would
    not have had available before that date.
    """
    ratings = {}
    seen_teams = set()
    for _, game in games.iterrows():
        game_date = game["gameDateTimeEst"]
        if cutoff is not None and game_date >= cutoff:
            break
        home_team = game["homeTeamId"]
        away_team = game["awayTeamId"]
        home_rating = ratings.get(home_team, initial_rating)
        away_rating = ratings.get(away_team, initial_rating)
        probability = elo_win_probability(home_rating, away_rating, home_advantage)
        outcome = game["target"]
        ratings[home_team] = home_rating + k_factor * (outcome - probability)
        ratings[away_team] = away_rating + k_factor * ((1 - outcome) - (1 - probability))
        seen_teams.add(home_team)
        seen_teams.add(away_team)
    return ratings, seen_teams


def lookup_last_team_row(features, team_id, game_date=None):
    team_rows = features[features["teamId"] == team_id].copy()
    if team_rows.empty:
        raise ValueError(f"No feature rows found for teamId={team_id}.")

    team_rows["gameDateTimeEst"] = pd.to_datetime(team_rows["gameDateTimeEst"])
    if game_date is not None:
        cutoff = pd.Timestamp(game_date)
        team_rows = team_rows[team_rows["gameDateTimeEst"] <= cutoff]
        if team_rows.empty:
            raise ValueError(
                f"No feature rows for teamId={team_id} on or before {game_date}."
            )

    return team_rows.sort_values("gameDateTimeEst").iloc[-1].to_dict()


def build_prediction_row(home_row, away_row, predictor_columns):
    row = {}
    for column in predictor_columns:
        home_value = home_row.get(column)
        away_value = away_row.get(column)
        if pd.isna(home_value) or pd.isna(away_value):
            raise ValueError(
                f"Missing feature value for {column} while comparing teamId "
                f"{home_row.get('teamId')} vs {away_row.get('teamId')}"
            )
        row[column] = float(home_value) - float(away_value)
    return pd.DataFrame([row])


def predict_matchup_logistic(
    home_team_id,
    away_team_id,
    game_date,
    features,
    model_path,
):
    model, predictor_columns = load_model(model_path)
    home_row = lookup_last_team_row(features, home_team_id, game_date)
    away_row = lookup_last_team_row(features, away_team_id, game_date)
    prediction_frame = build_prediction_row(home_row, away_row, predictor_columns)
    home_probability = float(model.predict_proba(prediction_frame)[0, 1])

    return home_probability, {
        "home": str(pd.Timestamp(home_row["gameDateTimeEst"]).date()),
        "away": str(pd.Timestamp(away_row["gameDateTimeEst"]).date()),
    }


def predict_matchup_boosted_hybrid(
    home_team_id,
    away_team_id,
    game_date,
    features,
    model_path,
    metrics_path=METRICS_PATH,
):
    model, predictor_columns = load_model(model_path)
    home_row = lookup_last_team_row(features, home_team_id, game_date)
    away_row = lookup_last_team_row(features, away_team_id, game_date)
    non_elo_columns = [column for column in predictor_columns if column != "elo_delta"]
    prediction_frame = build_prediction_row(home_row, away_row, non_elo_columns)

    if "elo_delta" in predictor_columns:
        games, _ = build_game_dataset(features)
        games["gameDateTimeEst"] = pd.to_datetime(games["gameDateTimeEst"])
        cutoff = pd.Timestamp(game_date) if game_date is not None else None
        try:
            elo_config = load_elo_config(metrics_path)
        except FileNotFoundError:
            elo_config = {"initial_rating": 1500.0, "k_factor": 20.0, "home_advantage": 65.0}

        ratings, seen_teams = compute_elo_ratings_as_of(
            games,
            cutoff=cutoff,
            initial_rating=elo_config["initial_rating"],
            k_factor=elo_config["k_factor"],
            home_advantage=elo_config["home_advantage"],
        )
        for team_id in (home_team_id, away_team_id):
            if team_id not in seen_teams:
                cutoff_message = f" on or before {game_date}" if game_date else ""
                raise ValueError(
                    f"No completed games found for teamId={team_id}{cutoff_message}; "
                    "cannot compute an Elo rating."
                )
        prediction_frame["elo_delta"] = float(
            ratings[home_team_id] - ratings[away_team_id] + elo_config["home_advantage"]
        )

    home_probability = float(model.predict_proba(prediction_frame)[0, 1])
    return home_probability, {
        "home": str(pd.Timestamp(home_row["gameDateTimeEst"]).date()),
        "away": str(pd.Timestamp(away_row["gameDateTimeEst"]).date()),
    }


def predict_matchup_ensemble(
    home_team_id,
    away_team_id,
    game_date,
    features,
    model_path,
    metrics_path=METRICS_PATH,
):
    elo_probability, _ = predict_matchup_elo(
        home_team_id, away_team_id, game_date, features, metrics_path
    )
    boosted_probability, feature_snapshot_date = predict_matchup_boosted_hybrid(
        home_team_id, away_team_id, game_date, features, model_path, metrics_path
    )
    return (elo_probability + boosted_probability) / 2.0, feature_snapshot_date


def predict_matchup_elo(
    home_team_id,
    away_team_id,
    game_date,
    features,
    metrics_path,
):
    elo_config = load_elo_config(metrics_path)
    games, _ = build_game_dataset(features)
    games["gameDateTimeEst"] = pd.to_datetime(games["gameDateTimeEst"])
    cutoff = pd.Timestamp(game_date) if game_date is not None else None

    ratings, seen_teams = compute_elo_ratings_as_of(
        games,
        cutoff=cutoff,
        initial_rating=elo_config["initial_rating"],
        k_factor=elo_config["k_factor"],
        home_advantage=elo_config["home_advantage"],
    )
    for team_id in (home_team_id, away_team_id):
        if team_id not in seen_teams:
            cutoff_message = f" on or before {game_date}" if game_date else ""
            raise ValueError(
                f"No completed games found for teamId={team_id}{cutoff_message}; "
                "cannot compute an Elo rating."
            )

    home_probability = elo_win_probability(
        ratings[home_team_id], ratings[away_team_id], elo_config["home_advantage"]
    )
    return home_probability, None


def load_feature_importance(metrics_path=METRICS_PATH, top_n=5):
    if not metrics_path.exists():
        return []
    metadata = json.loads(metrics_path.read_text(encoding="utf-8"))
    feature_importance = metadata.get("feature_importance", {})
    features = feature_importance.get("features", [])
    if not features:
        return []
    return [
        {
            "rank": int(item.get("rank", rank + 1)),
            "feature": item.get("feature"),
            "importance": float(item.get("importance", 0.0)),
        }
        for rank, item in enumerate(features[:top_n])
    ]


def load_model_summary(metrics_path=METRICS_PATH, top_n=5):
    """Return the recommended model, its holdout metrics, and the main comparison signals."""
    if not metrics_path.exists():
        return None
    metadata = json.loads(metrics_path.read_text(encoding="utf-8"))
    recommended_model = metadata.get("recommended_model", "boosted_hybrid")
    model_metrics = metadata.get("metrics", {}).get(recommended_model, {})
    model_calibration = metadata.get("calibration", {}).get(recommended_model, {})
    comparison_candidates = [
        name
        for name in (
            "boosted_hybrid",
            "calibrated_boosted_hybrid",
            "elo",
            "elo_boosted_ensemble",
        )
        if name in metadata.get("metrics", {})
    ]
    comparison = {}
    for name in comparison_candidates:
        metrics = metadata.get("metrics", {}).get(name, {})
        calibration = metadata.get("calibration", {}).get(name, {})
        comparison[name] = {
            "accuracy": metrics.get("accuracy"),
            "log_loss": metrics.get("log_loss"),
            "brier_score": metrics.get("brier_score"),
            "expected_calibration_error": calibration.get("expected_calibration_error"),
        }
    top_features = load_feature_importance(metrics_path, top_n=top_n)
    summary = {
        "recommended_model": recommended_model,
        "recommendation_metric": metadata.get("recommendation_metric", "log_loss"),
        "metrics": model_metrics,
        "calibration": model_calibration,
        "comparison": comparison,
        "top_features": top_features,
    }
    if not model_metrics and not model_calibration and not top_features:
        return None
    return summary


def predict_matchup(
    home_team_id,
    away_team_id,
    game_date=None,
    features_path=FEATURES_PATH,
    model_path=MODEL_PATH,
    metrics_path=METRICS_PATH,
):
    features = pd.read_csv(features_path)
    features["gameDateTimeEst"] = pd.to_datetime(features["gameDateTimeEst"])

    recommended_model = load_recommended_model_name(metrics_path)
    if recommended_model == "elo":
        home_probability, feature_snapshot_date = predict_matchup_elo(
            home_team_id, away_team_id, game_date, features, metrics_path
        )
    elif recommended_model == "elo_boosted_ensemble":
        home_probability, feature_snapshot_date = predict_matchup_ensemble(
            home_team_id, away_team_id, game_date, features, model_path, metrics_path
        )
    elif recommended_model in {"boosted_hybrid", "calibrated_boosted_hybrid"}:
        home_probability, feature_snapshot_date = predict_matchup_boosted_hybrid(
            home_team_id, away_team_id, game_date, features, model_path, metrics_path
        )
    else:
        home_probability, feature_snapshot_date = predict_matchup_logistic(
            home_team_id, away_team_id, game_date, features, model_path
        )
    validate_prediction_probability(home_probability)
    away_probability = 1.0 - home_probability

    result = {
        "home_team_id": int(home_team_id),
        "away_team_id": int(away_team_id),
        "game_date": None if game_date is None else str(pd.Timestamp(game_date).date()),
        "model": recommended_model,
        "home_win_probability": round(home_probability, 6),
        "away_win_probability": round(away_probability, 6),
        "home_team_prediction": "favorite" if home_probability >= 0.5 else "underdog",
    }
    if feature_snapshot_date is not None:
        result["feature_snapshot_date"] = feature_snapshot_date
    feature_importance = load_feature_importance(metrics_path)
    if feature_importance:
        result["feature_importance"] = feature_importance
    model_summary = load_model_summary(metrics_path)
    if model_summary:
        result["model_summary"] = model_summary
    return result


def validate_prediction_probability(home_probability):
    if not math.isfinite(home_probability) or not 0.0 <= home_probability <= 1.0:
        raise ValueError(
            f"Model returned invalid home win probability: {home_probability!r}."
        )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Predict an NBA game outcome using the validated baseline model."
    )
    parser.add_argument("--home-team-id", type=int, required=True)
    parser.add_argument("--away-team-id", type=int, required=True)
    parser.add_argument(
        "--game-date",
        type=str,
        help="Optional cutoff date in YYYY-MM-DD format. Uses the latest available pregame features on or before this date.",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Include the top model features and their normalized importance weights in the JSON output.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Include the recommended model, holdout metrics, calibration diagnostics, and top feature drivers in the JSON output.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        result = predict_matchup(
            home_team_id=args.home_team_id,
            away_team_id=args.away_team_id,
            game_date=args.game_date,
        )
    except ValueError as exc:
        raise SystemExit(str(exc))
    if args.explain:
        result["explanation"] = {
            "model": result.get("model"),
            "top_features": load_feature_importance(METRICS_PATH, top_n=5),
        }
    if args.summary:
        result["model_summary"] = load_model_summary(METRICS_PATH)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
