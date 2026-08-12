import json

import pandas as pd

from src.main import (
    build_prediction_row,
    compute_elo_ratings_as_of,
    load_recommended_model_name,
    lookup_last_team_row,
    predict_matchup,
    validate_prediction_probability,
)


def test_build_prediction_row_uses_home_minus_away_differences():
    home_row = {
        "teamId": 10,
        "teamScore_rolling_10": 110.0,
        "rest_days": 2.0,
        "active_players_last_game": 8.0,
    }
    away_row = {
        "teamId": 20,
        "teamScore_rolling_10": 104.0,
        "rest_days": 1.0,
        "active_players_last_game": 6.0,
    }

    frame = build_prediction_row(
        home_row,
        away_row,
        ["teamScore_rolling_10", "rest_days", "active_players_last_game"],
    )

    assert frame.loc[0, "teamScore_rolling_10"] == 6.0
    assert frame.loc[0, "rest_days"] == 1.0
    assert frame.loc[0, "active_players_last_game"] == 2.0


def test_validate_prediction_probability_rejects_nonfinite_or_out_of_range_values():
    for probability in (-0.01, 1.01, float("nan"), float("inf")):
        try:
            validate_prediction_probability(probability)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid probability to fail: {probability!r}")


def test_lookup_last_team_row_respects_game_date_cutoff():
    features = pd.DataFrame(
        [
            {
                "teamId": 10,
                "gameDateTimeEst": "2020-01-01 12:00:00",
                "teamScore_rolling_10": 98.0,
            },
            {
                "teamId": 10,
                "gameDateTimeEst": "2020-01-03 12:00:00",
                "teamScore_rolling_10": 102.0,
            },
            {
                "teamId": 10,
                "gameDateTimeEst": "2020-01-05 12:00:00",
                "teamScore_rolling_10": 106.0,
            },
        ]
    )

    row = lookup_last_team_row(features, team_id=10, game_date="2020-01-04")

    assert row["teamScore_rolling_10"] == 102.0


def _sample_games():
    return pd.DataFrame(
        [
            {
                "gameId": 1,
                "gameDateTimeEst": pd.Timestamp("2020-01-01"),
                "homeTeamId": 10,
                "awayTeamId": 20,
                "target": 1,
            },
            {
                "gameId": 2,
                "gameDateTimeEst": pd.Timestamp("2020-01-05"),
                "homeTeamId": 20,
                "awayTeamId": 10,
                "target": 0,
            },
            {
                "gameId": 3,
                "gameDateTimeEst": pd.Timestamp("2020-01-10"),
                "homeTeamId": 10,
                "awayTeamId": 30,
                "target": 1,
            },
        ]
    )


def test_compute_elo_ratings_as_of_only_uses_games_before_cutoff():
    games = _sample_games()

    ratings, seen_teams = compute_elo_ratings_as_of(
        games, cutoff=pd.Timestamp("2020-01-05"), k_factor=20.0, home_advantage=0.0
    )

    # Only the 2020-01-01 game (10 beat 20) should have been applied; the
    # 2020-01-05 game itself is on the cutoff and must be excluded, and
    # team 30 has not appeared yet.
    assert seen_teams == {10, 20}
    assert ratings[10] > 1500.0
    assert ratings[20] < 1500.0
    assert 30 not in ratings


def test_compute_elo_ratings_as_of_uses_all_games_when_cutoff_is_none():
    games = _sample_games()

    ratings, seen_teams = compute_elo_ratings_as_of(games, cutoff=None)

    assert seen_teams == {10, 20, 30}


def test_load_recommended_model_name_falls_back_when_metrics_file_missing(tmp_path):
    missing_path = tmp_path / "does_not_exist.json"

    assert load_recommended_model_name(missing_path) == "boosted_hybrid"


def test_predict_matchup_uses_elo_when_recommended(tmp_path):
    features_path = tmp_path / "game_features.csv"
    metrics_path = tmp_path / "baseline_metrics.json"

    features = pd.DataFrame(
        [
            {
                "gameId": 1,
                "gameDateTimeEst": "2020-01-01",
                "teamId": 10,
                "home": 1,
                "win": 1,
                "teamScore_rolling_10": 100,
            },
            {
                "gameId": 1,
                "gameDateTimeEst": "2020-01-01",
                "teamId": 20,
                "home": 0,
                "win": 0,
                "teamScore_rolling_10": 95,
            },
        ]
    )
    features.to_csv(features_path, index=False)
    metrics_path.write_text(
        json.dumps(
            {
                "recommended_model": "elo",
                "elo": {
                    "initial_rating": 1500.0,
                    "k_factor": 20.0,
                    "home_advantage": 65.0,
                },
            }
        ),
        encoding="utf-8",
    )

    result = predict_matchup(
        home_team_id=10,
        away_team_id=20,
        features_path=features_path,
        metrics_path=metrics_path,
    )

    assert result["model"] == "elo"
    assert "feature_snapshot_date" not in result
    assert result["home_win_probability"] > result["away_win_probability"]


def test_predict_matchup_uses_boosted_hybrid_when_recommended(tmp_path):
    features_path = tmp_path / "game_features.csv"
    metrics_path = tmp_path / "baseline_metrics.json"
    model_path = tmp_path / "baseline_logistic.pkl"

    features = pd.DataFrame(
        [
            {
                "gameId": 1,
                "gameDateTimeEst": "2020-01-01",
                "teamId": 10,
                "home": 1,
                "win": 1,
                "teamScore_rolling_10": 100,
                "rest_days": 2,
            },
            {
                "gameId": 1,
                "gameDateTimeEst": "2020-01-01",
                "teamId": 20,
                "home": 0,
                "win": 0,
                "teamScore_rolling_10": 95,
                "rest_days": 1,
            },
        ]
    )
    features.to_csv(features_path, index=False)
    metrics_path.write_text(
        json.dumps(        {"recommended_model": "calibrated_boosted_hybrid"}),
        encoding="utf-8",
    )
    model_bundle = {
        "model": __import__("sklearn.ensemble", fromlist=["HistGradientBoostingClassifier"]).HistGradientBoostingClassifier(),
        "predictors": ["teamScore_rolling_10", "rest_days"],
    }
    model_bundle["model"].fit(
        pd.DataFrame(
            [{"teamScore_rolling_10": 5.0, "rest_days": 1.0}, {"teamScore_rolling_10": -2.0, "rest_days": -1.0}]
        ),
        [1, 0],
    )
    with model_path.open("wb") as handle:
        import pickle

        pickle.dump(model_bundle, handle)

    result = predict_matchup(
        home_team_id=10,
        away_team_id=20,
        features_path=features_path,
        model_path=model_path,
        metrics_path=metrics_path,
    )

    assert result["model"] == "calibrated_boosted_hybrid"
    assert "feature_snapshot_date" in result
    assert 0.0 <= result["home_win_probability"] <= 1.0
