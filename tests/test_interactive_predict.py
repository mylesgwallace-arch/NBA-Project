from src.interactive_predict import (
    load_current_teams,
    summarize_model_features,
    summarize_model_report,
)


def test_load_current_teams_returns_thirty_current_nba_franchises(tmp_path):
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
                # Historical (no longer active) entry for the same franchise.
                (1610612737, "St. Louis", "Hawks", "STL", 1955, 1967, "NBA"),
                # Non-NBA / international team outside the franchise ID range.
                (9027, "Adelaide", "36ers", "ADL", 1990, 2100, "NBL"),
            ],
        )
        connection.commit()

    teams = load_current_teams(db_path)

    assert teams == [
        (1610612737, "Atlanta Hawks"),
        (1610612738, "Boston Celtics"),
    ]


def test_summarize_model_features_formats_top_features_for_display():
    result = {
        "feature_importance": [
            {"rank": 1, "feature": "elo_delta", "importance": 0.6951892424602657},
            {"rank": 2, "feature": "plusMinusPoints_rolling_10", "importance": 0.14480048677738624},
        ]
    }

    summary = summarize_model_features(result)

    assert "Top model features:" in summary
    assert "elo_delta (69.5%)" in summary
    assert "plusMinusPoints_rolling_10 (14.5%)" in summary


def test_summarize_model_report_formats_current_model_summary():
    result = {
        "model_summary": {
            "recommended_model": "boosted_hybrid",
            "recommendation_metric": "log_loss",
            "metrics": {"accuracy": 0.648, "log_loss": 0.6255},
            "calibration": {"expected_calibration_error": 0.047},
            "comparison": {
                "boosted_hybrid": {"log_loss": 0.6255, "expected_calibration_error": 0.047},
                "calibrated_boosted_hybrid": {"log_loss": 0.6277, "expected_calibration_error": 0.022},
            },
            "top_features": [
                {"rank": 1, "feature": "elo_delta", "importance": 0.6951892424602657},
                {"rank": 2, "feature": "teamScore_rolling_10", "importance": 0.14480048677738624},
            ],
        }
    }

    summary = summarize_model_report(result)

    assert "Recommended model: boosted_hybrid (lowest holdout log_loss)" in summary
    assert "Holdout accuracy: 64.8%" in summary
    assert "Holdout log loss: 0.6255" in summary
    assert "Expected calibration error: 0.047" in summary
    assert "calibrated_boosted_hybrid: log_loss=0.6277, ECE=0.022" in summary
    assert "elo_delta (69.5%)" in summary
