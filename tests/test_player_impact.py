import pandas as pd
import pytest

from src.player_impact import (
    _add_team_participation_controls,
    estimate_player_impact,
    validate_player_impact,
)


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
    independent = result["independent_pregame_target"]
    assert independent["target"] == "current team net rating"
    assert not independent[
        "target_is_derived_by_removing_current_player_scoring"
    ]
    assert independent["control_predictors"] == [
        "prior_team_net_rating",
        "prior_active_players",
        "prior_rotation_minutes",
    ]
    assert independent["candidate_predictors"] == [
        "prior_team_net_rating",
        "prior_active_players",
        "prior_rotation_minutes",
        "player_signal",
    ]
    assert independent["holdout_player_games"] == 1
    assert independent["bootstrap_intervals"]["cluster"] == "gameId"
    robustness = result["independent_pregame_robustness"]
    assert set(robustness) == {"0.1", "0.2", "0.3"}
    for window in robustness.values():
        assert window["usage_strata_definition"].startswith(
            "Low, middle, and high strata"
        )
        assert set(window["usage_strata"]) <= {"low", "middle", "high"}
        assert window["control_predictors"] == independent["control_predictors"]
    later = result["later_team_game_validation"]
    assert later["validation_start_season"] == 2024
    assert later["status"] == "insufficient_later_season_data"
    assert not later["control_has_player_signal"]
    assert later["candidate_adds_player_signal"]
    assert later["control_predictors"][-2:] == ["rest_days", "home"]
    split_results = result["later_team_game_validation_by_season"]
    assert set(split_results) == {"2022", "2023", "2024", "2025"}
    for split in split_results.values():
        assert split["status"] == "insufficient_later_season_data"
        assert not split["control_has_player_signal"]
    roster_changes = result["roster_change_validation"]
    assert roster_changes["status"] == "insufficient_roster_change_data"


def test_team_participation_controls_use_only_the_prior_team_game():
    players = pd.DataFrame(
        [
            {
                "personId": 1,
                "teamId": 10,
                "gameId": 1,
                "gameDateTimeEst": "2020-01-01",
                "minutes": 30,
            },
            {
                "personId": 2,
                "teamId": 10,
                "gameId": 1,
                "gameDateTimeEst": "2020-01-01",
                "minutes": 20,
            },
            {
                "personId": 1,
                "teamId": 10,
                "gameId": 2,
                "gameDateTimeEst": "2020-01-02",
                "minutes": 35,
            },
        ]
    )

    result = _add_team_participation_controls(players)

    current = result[result["gameId"] == 2].iloc[0]
    assert current["prior_active_players"] == 2
    assert current["prior_rotation_minutes"] == 50
    assert result[result["gameId"] == 1][
        ["prior_active_players", "prior_rotation_minutes"]
    ].isna().all().all()
