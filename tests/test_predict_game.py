import pandas as pd

from src.main import build_prediction_row, lookup_last_team_row


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
