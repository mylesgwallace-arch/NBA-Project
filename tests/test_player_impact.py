import pandas as pd
import pytest

from src.player_impact import estimate_player_impact, validate_player_impact


def test_player_impact_uses_minutes_weighted_prior_production():
    history = pd.DataFrame(
        [
            {"minutes": 24, "netRating": 10},
            {"minutes": 48, "netRating": -2},
        ]
    )

    result = estimate_player_impact(history)

    assert result["prior_games"] == 2
    assert result["player_net_rating"] == pytest.approx(2.0)
    assert result["expected_minutes"] == pytest.approx(36.0)
    assert result["estimated_net_rating_change"] == pytest.approx(1.5)


def test_player_removal_is_negative_of_addition():
    history = pd.DataFrame([{"minutes": 24, "netRating": 8}])

    addition = estimate_player_impact(history)
    removal = estimate_player_impact(history, direction="removal")

    assert removal["estimated_net_rating_change"] == pytest.approx(
        -addition["estimated_net_rating_change"]
    )


def test_validation_target_uses_only_prior_team_games():
    player_games = pd.DataFrame(
        [
            {"personId": 1, "gameId": i, "teamId": 10,
             "gameDateTimeEst": f"2020-01-{i:02d}", "minutes": 24,
             "netRating": 8, "points": 20, "player_possessions": 20}
            for i in range(1, 13)
        ]
    )
    team_games = pd.DataFrame(
        [
            {"gameId": i, "teamId": 10, "gameDateTimeEst": f"2020-01-{i:02d}",
             "netRating": 0, "team_points": 100, "opponent_points": 100,
             "team_possessions": 100}
            for i in range(1, 13)
        ]
    )
    team_games.loc[team_games["gameId"] == 12, "netRating"] = 20
    team_games.loc[team_games["gameId"] == 11, "netRating"] = 4

    result = validate_player_impact(player_games, team_games)

    assert result["evaluated_player_games"] == 2
    assert result["zero_change_baseline_mae"] > 0
    assert (
        result["association_diagnostic_mae"]
        < result["zero_change_baseline_mae"]
    )
    holdout = result["chronological_holdout"]
    assert holdout["training_games"] == 1
    assert holdout["holdout_games"] == 1
    assert holdout["improves_zero_change_baseline"]
    assert holdout["improves_zero_change_baseline_all_seasons"]
    intervals = holdout["bootstrap_intervals"]
    assert intervals["cluster"] == "gameId"
    assert intervals["replicates"] == 1000
    assert len(intervals["calibrated_minus_baseline_mae"]) == 2
