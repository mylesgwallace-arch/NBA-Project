import json

import pandas as pd

from src.main import (
    build_matchup_summary,
    build_prediction_row,
    compute_elo_ratings_as_of,
    load_feature_importance,
    load_model_summary,
    load_recommended_model_name,
    lookup_last_team_row,
    parse_args,
    predict_matchup,
    resolve_team_name_to_id,
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


def test_resolve_team_name_to_id_handles_current_franchise_names(tmp_path):
    import sqlite3

    db_path = tmp_path / "nba.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE team_histories (
                teamId INTEGER, teamCity TEXT, teamName TEXT,
                teamAbbrev TEXT, seasonFounded INTEGER,
                seasonActiveTill INTEGER, league TEXT
            )
            """
        )
        connection.executemany(
            "INSERT INTO team_histories VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (1610612737, "Atlanta", "Hawks", "ATL", 1968, 2100, "NBA"),
                (1610612738, "Boston", "Celtics", "BOS", 1946, 2100, "NBA"),
                (1610612740, "New Orleans", "Pelicans", "NOP", 2002, 2100, "NBA"),
                (1610612737, "St. Louis", "Hawks", "STL", 1955, 1967, "NBA"),
            ],
        )
        connection.commit()

    assert resolve_team_name_to_id("Atlanta Hawks", db_path) == 1610612737
    assert resolve_team_name_to_id("Boston", db_path) == 1610612738
    assert resolve_team_name_to_id("pelicans", db_path) == 1610612740


def test_parse_args_accepts_human_friendly_team_names():
    args = parse_args(["--home-team", "Boston Celtics", "--away-team", "Los Angeles Lakers"])

    assert args.home_team_id is None
    assert args.away_team_id is None
    assert args.home_team == "Boston Celtics"
    assert args.away_team == "Los Angeles Lakers"


def test_build_matchup_summary_mentions_probability_and_recent_form():
    summary = build_matchup_summary(
        0.7,
        0.3,
        {
            "home": {"win_rate_rolling_10": 0.8, "plusMinusPoints_rolling_10": 11.7},
            "away": {"win_rate_rolling_10": 0.55, "plusMinusPoints_rolling_10": 2.4},
        },
        [{"feature": "elo_delta"}],
    )

    assert "70.0%" in summary
    assert "home team" in summary
    assert "Recent form is 80.0%" in summary
    assert "elo_delta" in summary


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


def test_load_feature_importance_reads_top_features_from_metrics(tmp_path):
    metrics_path = tmp_path / "baseline_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "feature_importance": {
                    "features": [
                        {"rank": 1, "feature": "elo_delta", "importance": 0.7},
                        {"rank": 2, "feature": "win_rate_rolling_10", "importance": 0.2},
                        {"rank": 3, "feature": "rest_days", "importance": 0.1},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    features = load_feature_importance(metrics_path, top_n=2)

    assert len(features) == 2
    assert features[0]["feature"] == "elo_delta"
    assert features[1]["feature"] == "win_rate_rolling_10"
    assert features[0]["importance"] == 0.7


def test_load_model_summary_includes_metrics_and_top_features(tmp_path):
    metrics_path = tmp_path / "baseline_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "recommended_model": "boosted_hybrid",
                "recommendation_metric": "log_loss",
                "metrics": {
                    "boosted_hybrid": {"accuracy": 0.67, "log_loss": 0.62},
                    "calibrated_boosted_hybrid": {"accuracy": 0.65, "log_loss": 0.64},
                    "elo": {"accuracy": 0.66, "log_loss": 0.63},
                },
                "calibration": {
                    "boosted_hybrid": {"expected_calibration_error": 0.05},
                    "calibrated_boosted_hybrid": {"expected_calibration_error": 0.03},
                    "elo": {"expected_calibration_error": 0.04},
                },
                "feature_importance": {
                    "features": [
                        {"rank": 1, "feature": "elo_delta", "importance": 0.7},
                        {"rank": 2, "feature": "win_rate_rolling_10", "importance": 0.2},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    summary = load_model_summary(metrics_path)

    assert summary["recommended_model"] == "boosted_hybrid"
    assert summary["metrics"]["log_loss"] == 0.62
    assert summary["calibration"]["expected_calibration_error"] == 0.05
    assert summary["comparison"]["calibrated_boosted_hybrid"]["log_loss"] == 0.64
    assert summary["top_features"][0]["feature"] == "elo_delta"


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


def test_predict_matchup_uses_ensemble_when_recommended(tmp_path):
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
        json.dumps(
            {
                "recommended_model": "elo_boosted_ensemble",
                "elo": {"initial_rating": 1500.0, "k_factor": 20.0, "home_advantage": 65.0},
            }
        ),
        encoding="utf-8",
    )
    model_bundle = {
        "model": __import__("sklearn.ensemble", fromlist=["HistGradientBoostingClassifier"]).HistGradientBoostingClassifier(),
        "predictors": ["teamScore_rolling_10", "rest_days"],
    }
    model_bundle["model"].fit(
        pd.DataFrame([
            {"teamScore_rolling_10": 5.0, "rest_days": 1.0},
            {"teamScore_rolling_10": -2.0, "rest_days": -1.0},
        ]),
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

    assert result["model"] == "elo_boosted_ensemble"
    assert "feature_snapshot_date" in result
    assert 0.0 <= result["home_win_probability"] <= 1.0
