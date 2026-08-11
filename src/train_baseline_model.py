import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "data" / "processed" / "game_features.csv"
MODEL_PATH = ROOT / "models" / "baseline_logistic.pkl"
METRICS_PATH = ROOT / "models" / "baseline_metrics.json"
TEST_FRACTION = 0.2
ELO_INITIAL_RATING = 1500.0
ELO_K_FACTOR = 20.0
ELO_HOME_ADVANTAGE = 65.0
ELO_PARAMETER_GRID = [
    {"k_factor": 10.0, "home_advantage": 50.0},
    {"k_factor": 20.0, "home_advantage": 65.0},
    {"k_factor": 30.0, "home_advantage": 65.0},
    {"k_factor": 40.0, "home_advantage": 100.0},
]


def build_game_dataset(features):
    rolling_columns = [
        column for column in features.columns if column.endswith("_rolling_10")
    ]
    home = features[features["home"] == 1][
        ["gameId", "gameDateTimeEst", "teamId", "win", *rolling_columns]
    ].rename(
        columns={
            "teamId": "homeTeamId",
            "win": "target",
            **{column: f"{column}_home" for column in rolling_columns},
        }
    )
    away = features[features["home"] == 0][
        ["gameId", "teamId", *rolling_columns]
    ].rename(
        columns={
            "teamId": "awayTeamId",
            **{column: f"{column}_away" for column in rolling_columns},
        }
    )

    games = home.merge(away, on="gameId", how="inner", validate="one_to_one")
    games["target"] = games["target"].astype(int)
    for column in rolling_columns:
        games[column] = games[f"{column}_home"] - games[f"{column}_away"]
    games = games[
        [
            "gameId",
            "gameDateTimeEst",
            "homeTeamId",
            "awayTeamId",
            "target",
            *rolling_columns,
        ]
    ]
    return games.sort_values(["gameDateTimeEst", "gameId"]).reset_index(drop=True), rolling_columns


def elo_win_probability(home_rating, away_rating, home_advantage=ELO_HOME_ADVANTAGE):
    rating_difference = home_rating - away_rating + home_advantage
    return 1 / (1 + 10 ** (-rating_difference / 400))


def evaluate_elo(
    games,
    split,
    initial_rating=ELO_INITIAL_RATING,
    k_factor=ELO_K_FACTOR,
    home_advantage=ELO_HOME_ADVANTAGE,
):
    ratings = {}
    test_probabilities = []
    test_targets = []

    for index, game in games.iterrows():
        home_team = game["homeTeamId"]
        away_team = game["awayTeamId"]
        home_rating = ratings.get(home_team, initial_rating)
        away_rating = ratings.get(away_team, initial_rating)
        probability = elo_win_probability(home_rating, away_rating, home_advantage)

        if index >= split:
            test_probabilities.append(probability)
            test_targets.append(game["target"])

        outcome = game["target"]
        ratings[home_team] = home_rating + k_factor * (outcome - probability)
        ratings[away_team] = away_rating + k_factor * ((1 - outcome) - (1 - probability))

    return evaluate_predictions(pd.Series(test_targets), pd.Series(test_probabilities))


def elo_test_predictions(
    games,
    split,
    initial_rating=ELO_INITIAL_RATING,
    k_factor=ELO_K_FACTOR,
    home_advantage=ELO_HOME_ADVANTAGE,
):
    ratings = {}
    predictions = []

    for index, game in games.iterrows():
        home_team = game["homeTeamId"]
        away_team = game["awayTeamId"]
        home_rating = ratings.get(home_team, initial_rating)
        away_rating = ratings.get(away_team, initial_rating)
        probability = elo_win_probability(home_rating, away_rating, home_advantage)

        if index >= split:
            predictions.append(
                {
                    "index": index,
                    "probability": probability,
                    "target": int(game["target"]),
                    "season": int(game["season"]),
                }
            )

        outcome = game["target"]
        ratings[home_team] = home_rating + k_factor * (outcome - probability)
        ratings[away_team] = away_rating + k_factor * ((1 - outcome) - (1 - probability))

    return pd.DataFrame(predictions)


def evaluate_elo_by_season(
    games,
    split,
    initial_rating=ELO_INITIAL_RATING,
    k_factor=ELO_K_FACTOR,
    home_advantage=ELO_HOME_ADVANTAGE,
):
    predictions = elo_test_predictions(
        games, split, initial_rating, k_factor, home_advantage
    )
    results = {}
    for season, season_predictions in predictions.groupby("season", sort=True):
        metrics = evaluate_predictions(
            season_predictions["target"], season_predictions["probability"]
        )
        metrics["games"] = int(len(season_predictions))
        results[str(season)] = metrics
    return results


def evaluate_elo_parameter_grid(games, split):
    results = []
    for parameters in ELO_PARAMETER_GRID:
        predictions = elo_test_predictions(
            games,
            split,
            k_factor=parameters["k_factor"],
            home_advantage=parameters["home_advantage"],
        )
        metrics = evaluate_predictions(
            predictions["target"], predictions["probability"]
        )
        results.append({**parameters, **metrics})
    return results


def evaluate_predictions_by_season(target, probabilities, seasons):
    values = pd.DataFrame(
        {
            "target": target.to_numpy(),
            "probability": probabilities.to_numpy(),
            "season": seasons.to_numpy(),
        }
    )
    results = {}
    for season, season_values in values.groupby("season", sort=True):
        metrics = evaluate_predictions(
            season_values["target"], season_values["probability"]
        )
        metrics["games"] = int(len(season_values))
        results[str(season)] = metrics
    return results


def evaluate_predictions(target, probabilities):
    return {
        "accuracy": float(accuracy_score(target, probabilities >= 0.5)),
        "log_loss": float(log_loss(target, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(target, probabilities)),
    }


def main():
    features = pd.read_csv(FEATURES_PATH)
    games, predictor_columns = build_game_dataset(features)
    games["season"] = pd.to_datetime(games["gameDateTimeEst"]).dt.year - (
        pd.to_datetime(games["gameDateTimeEst"]).dt.month < 10
    )
    split = int(len(games) * (1 - TEST_FRACTION))
    train = games.iloc[:split]
    test = games.iloc[split:]

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("logistic", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(train[predictor_columns], train["target"])

    home_rate = train["target"].mean()
    predictions = {
        "home_win_rate": pd.Series(home_rate, index=test.index),
        "rolling_logistic": pd.Series(
            model.predict_proba(test[predictor_columns])[:, 1], index=test.index
        ),
    }
    metrics = {
        name: evaluate_predictions(test["target"], probabilities)
        for name, probabilities in predictions.items()
    }
    metrics["elo"] = evaluate_elo(games, split)
    elo_season_metrics = evaluate_elo_by_season(games, split)
    elo_parameter_sensitivity = evaluate_elo_parameter_grid(games, split)
    rolling_season_metrics = evaluate_predictions_by_season(
        test["target"],
        predictions["rolling_logistic"],
        test["season"],
    )
    metadata = {
        "feature_rows": len(features),
        "complete_games": len(games),
        "train_games": len(train),
        "test_games": len(test),
        "test_fraction": TEST_FRACTION,
        "split_start": test["gameDateTimeEst"].iloc[0],
        "predictors": predictor_columns,
        "elo": {
            "initial_rating": ELO_INITIAL_RATING,
            "k_factor": ELO_K_FACTOR,
            "home_advantage": ELO_HOME_ADVANTAGE,
        },
        "metrics": metrics,
        "elo_by_season": elo_season_metrics,
        "rolling_logistic_by_season": rolling_season_metrics,
        "elo_parameter_sensitivity": elo_parameter_sensitivity,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as output:
        pickle.dump({"model": model, "predictors": predictor_columns}, output)
    METRICS_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Complete games: {len(games):,}")
    print(f"Chronological split: {len(train):,} train / {len(test):,} test")
    for name, values in metrics.items():
        print(
            f"{name}: accuracy={values['accuracy']:.5f}, "
            f"log_loss={values['log_loss']:.5f}, "
            f"brier_score={values['brier_score']:.5f}"
        )
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved metrics to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
