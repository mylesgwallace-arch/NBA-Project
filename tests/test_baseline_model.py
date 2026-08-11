import pandas as pd

from src.train_baseline_model import build_game_dataset


def test_game_dataset_pairs_home_and_away_rows_without_current_game_metrics():
    features = pd.DataFrame(
        [
            {
                "gameId": 1,
                "gameDateTimeEst": "2020-01-01",
                "teamId": 10,
                "home": 1,
                "win": 1,
                "teamScore": 110,
                "teamScore_rolling_10": 100,
            },
            {
                "gameId": 1,
                "gameDateTimeEst": "2020-01-01",
                "teamId": 20,
                "home": 0,
                "win": 0,
                "teamScore": 90,
                "teamScore_rolling_10": 95,
            },
        ]
    )

    games, predictors = build_game_dataset(features)

    assert len(games) == 1
    assert predictors == ["teamScore_rolling_10"]
    assert games.loc[0, "target"] == 1
    assert games.loc[0, "teamScore_rolling_10"] == 5
    assert "teamScore" not in predictors
