import math

import pandas as pd

from src.train_baseline_model import (
    build_game_dataset,
    elo_win_probability,
    evaluate_elo,
    evaluate_elo_by_season,
    select_recommended_model,
)


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


def test_game_dataset_includes_pregame_rest_difference_when_available():
    features = pd.DataFrame(
        [
            {
                "gameId": 1,
                "gameDateTimeEst": "2020-01-03",
                "teamId": 10,
                "home": 1,
                "win": 1,
                "teamScore_rolling_10": 100,
                "rest_days": 2,
            },
            {
                "gameId": 1,
                "gameDateTimeEst": "2020-01-03",
                "teamId": 20,
                "home": 0,
                "win": 0,
                "teamScore_rolling_10": 95,
                "rest_days": 1,
            },
        ]
    )

    games, predictors = build_game_dataset(features)

    assert predictors == ["teamScore_rolling_10", "rest_days"]
    assert games.loc[0, "rest_days"] == 1


def test_game_dataset_includes_pregame_player_availability_differences():
    features = pd.DataFrame(
        [
            {
                "gameId": 1,
                "gameDateTimeEst": "2020-01-03",
                "teamId": 10,
                "home": 1,
                "win": 1,
                "teamScore_rolling_10": 100,
                "active_players_rolling_10": 8,
                "active_players_last_game": 7,
            },
            {
                "gameId": 1,
                "gameDateTimeEst": "2020-01-03",
                "teamId": 20,
                "home": 0,
                "win": 0,
                "teamScore_rolling_10": 95,
                "active_players_rolling_10": 9,
                "active_players_last_game": 9,
            },
        ]
    )

    games, predictors = build_game_dataset(features)

    assert predictors == [
        "teamScore_rolling_10",
        "active_players_rolling_10",
        "active_players_last_game",
    ]
    assert games.loc[0, "active_players_rolling_10"] == -1
    assert games.loc[0, "active_players_last_game"] == -2


def test_elo_updates_after_completed_games_only():
    games = pd.DataFrame(
        [
            {
                "gameId": 1,
                "gameDateTimeEst": "2020-01-01",
                "homeTeamId": 10,
                "awayTeamId": 20,
                "target": 1,
            },
            {
                "gameId": 2,
                "gameDateTimeEst": "2020-01-02",
                "homeTeamId": 10,
                "awayTeamId": 20,
                "target": 0,
            },
        ]
    )

    metrics = evaluate_elo(
        games,
        split=1,
        initial_rating=1500,
        k_factor=20,
        home_advantage=0,
    )

    expected_log_loss = -math.log(
        1 - elo_win_probability(1500 + 10, 1500 - 10, home_advantage=0)
    )
    assert metrics["accuracy"] == 0.0
    assert abs(metrics["log_loss"] - expected_log_loss) < 1e-12


def test_elo_season_metrics_use_pregame_ratings_and_report_games():
    games = pd.DataFrame(
        [
            {
                "gameId": 1,
                "gameDateTimeEst": "2020-01-01",
                "homeTeamId": 10,
                "awayTeamId": 20,
                "target": 1,
                "season": 2019,
            },
            {
                "gameId": 2,
                "gameDateTimeEst": "2020-01-02",
                "homeTeamId": 10,
                "awayTeamId": 20,
                "target": 0,
                "season": 2019,
            },
            {
                "gameId": 3,
                "gameDateTimeEst": "2021-01-01",
                "homeTeamId": 10,
                "awayTeamId": 20,
                "target": 1,
                "season": 2020,
            },
        ]
    )

    metrics = evaluate_elo_by_season(
        games, split=1, initial_rating=1500, k_factor=20, home_advantage=0
    )

    assert metrics["2019"]["games"] == 1
    assert metrics["2020"]["games"] == 1
    assert metrics["2019"]["accuracy"] == 0.0
    assert metrics["2020"]["accuracy"] == 0.0


def test_select_recommended_model_uses_lowest_log_loss():
    metrics = {
        "home_win_rate": {"accuracy": 0.5, "log_loss": 0.69, "brier_score": 0.25},
        "rolling_logistic": {"accuracy": 0.62, "log_loss": 0.65, "brier_score": 0.23},
        "elo": {"accuracy": 0.65, "log_loss": 0.62, "brier_score": 0.21},
    }

    assert select_recommended_model(metrics) == "elo"


def test_select_recommended_model_excludes_trivial_home_win_rate_baseline():
    # home_win_rate should never be "recommended" even if it happens to have
    # the lowest log loss on some slice, because it is a reference baseline,
    # not a deployable model.
    metrics = {
        "home_win_rate": {"accuracy": 0.5, "log_loss": 0.1, "brier_score": 0.05},
        "rolling_logistic": {"accuracy": 0.62, "log_loss": 0.65, "brier_score": 0.23},
    }

    assert select_recommended_model(metrics) == "rolling_logistic"
