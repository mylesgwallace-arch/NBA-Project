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


def build_game_dataset(features):
    rolling_columns = [
        column for column in features.columns if column.endswith("_rolling_10")
    ]
    home = features[features["home"] == 1][
        ["gameId", "gameDateTimeEst", "win", *rolling_columns]
    ].rename(columns={"win": "target", **{column: f"{column}_home" for column in rolling_columns}})
    away = features[features["home"] == 0][
        ["gameId", *rolling_columns]
    ].rename(columns={column: f"{column}_away" for column in rolling_columns})

    games = home.merge(away, on="gameId", how="inner", validate="one_to_one")
    games["target"] = games["target"].astype(int)
    for column in rolling_columns:
        games[column] = games[f"{column}_home"] - games[f"{column}_away"]
    games = games[["gameId", "gameDateTimeEst", "target", *rolling_columns]]
    return games.sort_values(["gameDateTimeEst", "gameId"]).reset_index(drop=True), rolling_columns


def evaluate_predictions(target, probabilities):
    return {
        "accuracy": float(accuracy_score(target, probabilities >= 0.5)),
        "log_loss": float(log_loss(target, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(target, probabilities)),
    }


def main():
    features = pd.read_csv(FEATURES_PATH)
    games, predictor_columns = build_game_dataset(features)
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
    metadata = {
        "feature_rows": len(features),
        "complete_games": len(games),
        "train_games": len(train),
        "test_games": len(test),
        "test_fraction": TEST_FRACTION,
        "split_start": test["gameDateTimeEst"].iloc[0],
        "predictors": predictor_columns,
        "metrics": metrics,
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
