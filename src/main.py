import argparse
import json
import pickle
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "data" / "processed" / "game_features.csv"
MODEL_PATH = ROOT / "models" / "baseline_logistic.pkl"


def load_model(model_path=MODEL_PATH):
    with model_path.open("rb") as handle:
        model_bundle = pickle.load(handle)
    return model_bundle["model"], model_bundle["predictors"]


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


def predict_matchup(
    home_team_id,
    away_team_id,
    game_date=None,
    features_path=FEATURES_PATH,
    model_path=MODEL_PATH,
):
    features = pd.read_csv(features_path)
    features["gameDateTimeEst"] = pd.to_datetime(features["gameDateTimeEst"])

    model, predictor_columns = load_model(model_path)
    home_row = lookup_last_team_row(features, home_team_id, game_date)
    away_row = lookup_last_team_row(features, away_team_id, game_date)
    prediction_frame = build_prediction_row(home_row, away_row, predictor_columns)
    home_probability = float(model.predict_proba(prediction_frame)[0, 1])
    away_probability = 1.0 - home_probability

    return {
        "home_team_id": int(home_team_id),
        "away_team_id": int(away_team_id),
        "game_date": None if game_date is None else str(pd.Timestamp(game_date).date()),
        "model": "baseline_logistic",
        "home_win_probability": round(home_probability, 6),
        "away_win_probability": round(away_probability, 6),
        "home_team_prediction": "favorite" if home_probability >= 0.5 else "underdog",
        "feature_snapshot_date": {
            "home": str(pd.Timestamp(home_row["gameDateTimeEst"]).date()),
            "away": str(pd.Timestamp(away_row["gameDateTimeEst"]).date()),
        },
    }


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
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
