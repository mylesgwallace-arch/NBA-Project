"""Leakage-safe, assumption-labeled player impact estimates."""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "database" / "nba.db"
METRICS_PATH = ROOT / "models" / "player_impact_metrics.json"
WINDOW = 10
BASELINE_NET_RATING = 0.0
HOLDOUT_BOOTSTRAP_REPLICATES = 1000
HOLDOUT_BOOTSTRAP_SEED = 42


def load_player_history(connection, person_id, before=None):
    """Load regular-season player appearances available before ``before``."""
    query = """
        SELECT player_statistics_extended.personId,
               player_statistics_extended.gameId,
               player_statistics_extended.gameDateTimeEst,
               player_statistics_extended.playerteamId AS teamId,
               CAST(player_statistics_extended.numMinutes AS REAL) AS minutes,
               player_statistics_extended.netRating
        FROM player_statistics_extended
        JOIN games USING (gameId)
        WHERE personId = ?
          AND COALESCE(player_statistics_extended.gameType, games.gameType) =
                'Regular Season'
          AND player_statistics_extended.playerteamId IS NOT NULL
          AND CAST(player_statistics_extended.numMinutes AS REAL) > 0
          AND player_statistics_extended.netRating IS NOT NULL
    """
    parameters = [person_id]
    if before is not None:
        query += " AND gameDateTimeEst < ?"
        parameters.append(pd.Timestamp(before).strftime("%Y-%m-%d %H:%M:%S"))
    history = pd.read_sql_query(query, connection, params=parameters)
    if not history.empty:
        history["gameDateTimeEst"] = pd.to_datetime(history["gameDateTimeEst"])
        history = history.sort_values(["gameDateTimeEst", "gameId"])
    return history


def estimate_player_impact(
    history, window=WINDOW, baseline_net_rating=BASELINE_NET_RATING,
    direction="addition",
):
    """Estimate team net-rating change from a player's expected minutes.

    This is a descriptive proxy, not a causal estimate: player net rating is
    treated as transferable team impact, replacement minutes are assigned a
    fixed baseline net rating, and expected minutes are the player's prior
    average. The baseline is a reference value, not a replacement-level
    estimate.
    """
    if direction not in {"addition", "removal"}:
        raise ValueError("direction must be 'addition' or 'removal'")
    if window < 1:
        raise ValueError("window must be positive")
    required = {"minutes", "netRating"}
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"history is missing columns: {sorted(missing)}")
    recent = history.tail(window).dropna(subset=["minutes", "netRating"])
    if recent.empty:
        raise ValueError("at least one valid prior player appearance is required")
    weights = recent["minutes"].clip(lower=0)
    if weights.sum() <= 0:
        raise ValueError("prior player minutes must have a positive total")
    player_net_rating = float(np.average(recent["netRating"], weights=weights))
    expected_minutes = float(weights.mean())
    delta = (player_net_rating - baseline_net_rating) * expected_minutes / 48.0
    if direction == "removal":
        delta = -delta
    return {
        "prior_games": int(len(recent)),
        "player_net_rating": player_net_rating,
        "expected_minutes": expected_minutes,
        "baseline_net_rating": float(baseline_net_rating),
        "estimated_net_rating_change": float(delta),
        "direction": direction,
        "assumptions": [
            "Prior player net rating transfers to the new team context.",
            "Expected minutes equal the prior-window average.",
            "Baseline minutes have the configured reference net rating of "
            f"{baseline_net_rating}. This is not a replacement-level estimate.",
            "This estimate is descriptive and is not a causal trade prediction.",
        ],
    }


def _calibrate_signal(training_values):
    """Fit a linear calibration using only targets before the holdout."""
    if len(training_values) < 2 or training_values["player_signal"].nunique() < 2:
        return 1.0, 0.0, "identity"
    signal = training_values["player_signal"].to_numpy(dtype=float)
    target = training_values["leave_one_out_target"].to_numpy(dtype=float)
    slope = float(np.cov(signal, target, ddof=0)[0, 1] / np.var(signal))
    intercept = float(target.mean() - slope * signal.mean())
    return slope, intercept, "linear"


def _mae(target, prediction):
    return float(np.mean(np.abs(target - prediction)))


def _bootstrap_holdout_intervals(
    holdout, prediction_column, replicates=HOLDOUT_BOOTSTRAP_REPLICATES,
    seed=HOLDOUT_BOOTSTRAP_SEED,
):
    """Estimate clustered 95% MAE intervals by resampling complete games."""
    errors = holdout.assign(
        model_error=(
            holdout["leave_one_out_target"]
            - holdout[prediction_column]
        ).abs(),
        baseline_error=holdout["leave_one_out_target"].abs(),
    )
    by_game = errors.groupby("gameId")[["model_error", "baseline_error"]].agg(
        ["sum", "count"]
    )
    rng = np.random.default_rng(seed)
    game_indexes = rng.integers(0, len(by_game), size=(replicates, len(by_game)))
    model_sums = by_game["model_error"]["sum"].to_numpy()[game_indexes].sum(axis=1)
    baseline_sums = by_game["baseline_error"]["sum"].to_numpy()[game_indexes].sum(
        axis=1
    )
    counts = by_game["model_error"]["count"].to_numpy()[game_indexes].sum(axis=1)
    model_mae = model_sums / counts
    baseline_mae = baseline_sums / counts
    difference = model_mae - baseline_mae

    def interval(values):
        lower, upper = np.percentile(values, [2.5, 97.5])
        return [float(lower), float(upper)]

    return {
        "replicates": replicates,
        "seed": seed,
        "cluster": "gameId",
        "confidence_level": 0.95,
        "calibrated_mae": interval(model_mae),
        "zero_change_baseline_mae": interval(baseline_mae),
        "calibrated_minus_baseline_mae": interval(difference),
    }


def validate_player_impact(
    player_games, team_games, window=WINDOW, test_fraction=0.2
):
    """Validate a possession-normalized leave-one-player-out target.

    The target is the change in team scoring efficiency when the player's
    points and estimated possessions are removed from the current game.
    Pregame production is compared with a fixed-zero target baseline. This is
    still an association diagnostic, not causal evidence.
    """
    required_player = {
        "personId", "gameId", "teamId", "gameDateTimeEst", "minutes",
        "netRating", "points", "player_possessions",
    }
    required_team = {
        "gameId", "teamId", "gameDateTimeEst", "netRating", "team_points",
        "opponent_points", "team_possessions",
    }
    if not required_player.issubset(player_games.columns):
        raise ValueError("player_games does not contain the required columns")
    if not required_team.issubset(team_games.columns):
        raise ValueError("team_games does not contain the required columns")

    players = player_games.copy()
    teams = team_games.copy()
    players["gameDateTimeEst"] = pd.to_datetime(players["gameDateTimeEst"])
    teams["gameDateTimeEst"] = pd.to_datetime(teams["gameDateTimeEst"])
    players["teamId"] = players["teamId"].astype("int64")
    teams["teamId"] = teams["teamId"].astype("int64")
    players = players.sort_values(["personId", "gameDateTimeEst", "gameId"])
    teams = teams.sort_values(["teamId", "gameDateTimeEst", "gameId"])
    players["weighted_net_rating"] = players["minutes"] * players["netRating"]
    player_context = teams[["gameId", "teamId", "netRating"]].rename(
        columns={"netRating": "player_team_net_rating"}
    )
    players = players.merge(
        player_context, on=["gameId", "teamId"], how="left", validate="many_to_one"
    )
    player_groups = players.groupby("personId", sort=False)
    players["prior_minutes"] = player_groups["minutes"].transform(
        lambda values: values.shift(1).rolling(window, min_periods=window).sum()
    )
    players["prior_weighted_net_rating"] = player_groups[
        "weighted_net_rating"
    ].transform(
        lambda values: values.shift(1).rolling(window, min_periods=window).sum()
    )
    players["prior_player_team_net_rating"] = player_groups[
        "player_team_net_rating"
    ].transform(
        lambda values: values.shift(1).rolling(window, min_periods=window).mean()
    )
    team_groups = teams.groupby("teamId", sort=False)
    teams["prior_team_net_rating"] = team_groups["netRating"].transform(
        lambda values: values.shift(1).rolling(window, min_periods=window).mean()
    )
    teams["prior_team_possessions"] = team_groups["team_possessions"].transform(
        lambda values: values.shift(1).rolling(window, min_periods=window).mean()
    )
    if "opponentTeamId" in teams:
        opponent_history = teams[["teamId", "gameDateTimeEst", "netRating"]].rename(
            columns={
                "teamId": "opponentTeamId",
                "gameDateTimeEst": "opponent_game_date",
                "netRating": "opponent_net_rating",
            }
        ).sort_values(["opponentTeamId", "opponent_game_date"])
        opponent_groups = opponent_history.groupby("opponentTeamId", sort=False)
        opponent_history["prior_opponent_net_rating"] = opponent_groups[
            "opponent_net_rating"
        ].transform(
            lambda values: values.shift(1).rolling(window, min_periods=window).mean()
        )
        teams = teams.merge(
            opponent_history[
                ["opponentTeamId", "opponent_game_date", "prior_opponent_net_rating"]
            ],
            left_on=["opponentTeamId", "gameDateTimeEst"],
            right_on=["opponentTeamId", "opponent_game_date"],
            how="left",
            validate="many_to_one",
        ).drop(columns=["opponent_game_date"])
    current_team = teams[
        [
            "gameId", "teamId", "netRating", "team_points",
            "opponent_points", "team_possessions", "prior_team_net_rating",
            "prior_team_possessions",
        ]
    ].rename(columns={"netRating": "current_team_net_rating"})
    values = players.merge(
        current_team, on=["gameId", "teamId"], how="inner", validate="many_to_one"
    )
    values = values.dropna(
        subset=[
            "prior_minutes", "prior_weighted_net_rating",
            "prior_team_possessions", "prior_player_team_net_rating",
        ]
    )
    values["player_signal"] = (
        values["prior_weighted_net_rating"] / values["prior_minutes"]
        * (values["prior_minutes"] / window) / 48.0
    )
    player_groups = players.groupby("personId", sort=False)
    values["prior_points"] = player_groups["points"].transform(
        lambda series: series.shift(1).rolling(window, min_periods=window).sum()
    )
    values["prior_player_possessions"] = player_groups["player_possessions"].transform(
        lambda series: series.shift(1).rolling(window, min_periods=window).sum()
    )
    values["player_signal"] = (
        values["prior_points"] / values["prior_player_possessions"]
        * (values["prior_player_possessions"] / window)
        / values["prior_team_possessions"]
        * 100.0
    )
    values["leave_one_out_net_rating"] = (
        (
            values["team_points"] - values["points"] - values["opponent_points"]
        )
        / (values["team_possessions"] - values["player_possessions"])
        * 100.0
    )
    values["leave_one_out_target"] = (
        values["current_team_net_rating"] - values["leave_one_out_net_rating"]
    )
    values = values.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["leave_one_out_target", "player_signal"]
    )
    if values.empty:
        raise ValueError("no player appearances have enough prior history to validate")
    target = values["leave_one_out_target"]
    prediction = values["player_signal"]
    model_mae = _mae(target, prediction)
    baseline_mae = _mae(target, 0.0)
    correlation = (
        prediction.corr(target)
        if prediction.nunique() > 1 and target.nunique() > 1
        else np.nan
    )
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    ordered_games = (
        values[["gameId", "gameDateTimeEst"]]
        .drop_duplicates()
        .sort_values(["gameDateTimeEst", "gameId"])
    )
    split_games = max(
        1,
        min(
            len(ordered_games) - 1,
            int(len(ordered_games) * (1 - test_fraction)),
        ),
    )
    holdout_game_ids = set(ordered_games.iloc[split_games:]["gameId"])
    holdout = values[values["gameId"].isin(holdout_game_ids)].copy()
    training = values[~values["gameId"].isin(holdout_game_ids)]
    slope, intercept, calibration = _calibrate_signal(training)
    holdout["calibrated_prediction"] = (
        slope * holdout["player_signal"] + intercept
    )
    holdout_model_mae = _mae(
        holdout["leave_one_out_target"], holdout["calibrated_prediction"]
    )
    holdout_baseline_mae = _mae(holdout["leave_one_out_target"], 0.0)
    holdout["season"] = holdout["gameDateTimeEst"].dt.year - (
        holdout["gameDateTimeEst"].dt.month < 10
    )
    holdout_seasons = {}
    for season, season_values in holdout.groupby("season", sort=True):
        season_model_mae = _mae(
            season_values["leave_one_out_target"],
            season_values["calibrated_prediction"],
        )
        season_baseline_mae = _mae(
            season_values["leave_one_out_target"], 0.0
        )
        holdout_seasons[str(season)] = {
            "evaluated_player_games": int(len(season_values)),
            "calibrated_mae": season_model_mae,
            "zero_change_baseline_mae": season_baseline_mae,
            "improves_zero_change_baseline": (
                season_model_mae < season_baseline_mae
            ),
        }
    return {
        "evaluated_player_games": int(len(values)),
        "window": window,
        "target": "current team net rating minus leave-one-player-out scoring net rating",
        "association_diagnostic_mae": model_mae,
        "zero_change_baseline_mae": baseline_mae,
        "improves_zero_change_baseline": model_mae < baseline_mae,
        "prediction_target_correlation": (
            None if pd.isna(correlation) else float(correlation)
        ),
        "chronological_holdout": {
            "test_fraction": test_fraction,
            "training_games": int(split_games),
            "holdout_games": int(len(ordered_games) - split_games),
            "holdout_start": holdout["gameDateTimeEst"].min().isoformat(),
            "training_player_games": int(len(training)),
            "holdout_player_games": int(len(holdout)),
            "calibration": calibration,
            "calibration_slope": slope,
            "calibration_intercept": intercept,
            "calibrated_mae": holdout_model_mae,
            "zero_change_baseline_mae": holdout_baseline_mae,
            "improves_zero_change_baseline": (
                holdout_model_mae < holdout_baseline_mae
            ),
            "improves_zero_change_baseline_all_seasons": bool(
                holdout_seasons
                and all(
                    result["improves_zero_change_baseline"]
                    for result in holdout_seasons.values()
                )
            ),
            "bootstrap_intervals": _bootstrap_holdout_intervals(
                holdout, "calibrated_prediction"
            ),
            "by_season": holdout_seasons,
        },
        "note": (
            "The leave-one-out target removes current player points and "
            "possessions from team scoring efficiency. It is an association "
            "diagnostic, not a causal impact estimate. Calibration is fit "
            "only on games before the chronological holdout."
        ),
    }


def load_validation_data(connection):
    player_games = pd.read_sql_query(
        """
        SELECT player_statistics_extended.personId, player_statistics_extended.gameId,
               player_statistics_extended.playerteamId AS teamId,
               player_statistics_extended.gameDateTimeEst,
               CAST(player_statistics_extended.numMinutes AS REAL) AS minutes,
               player_statistics_extended.netRating,
               CAST(player_statistics_extended.points AS REAL) AS points,
               CAST(player_statistics_extended.possessions AS REAL) AS player_possessions
        FROM player_statistics_extended
        JOIN games USING (gameId)
        WHERE COALESCE(player_statistics_extended.gameType, games.gameType) =
              'Regular Season'
          AND player_statistics_extended.playerteamId IS NOT NULL
          AND CAST(player_statistics_extended.numMinutes AS REAL) > 0
          AND player_statistics_extended.netRating IS NOT NULL
        """,
        connection,
    )
    team_games = pd.read_sql_query(
        """
        SELECT teamId, opponentTeamId, gameId, gameDateTimeEst,
               netRating AS teamNetRating, teamScore AS team_points,
               opponentScore AS opponent_points,
               CAST(possessions AS REAL) AS team_possessions
        FROM team_statistics_extended
        WHERE gameType = 'Regular Season' AND netRating IS NOT NULL
        """,
        connection,
    ).rename(columns={"teamNetRating": "netRating"})
    return player_games, team_games


def main():
    with sqlite3.connect(DB_PATH) as connection:
        player_games, team_games = load_validation_data(connection)
    metrics = validate_player_impact(player_games, team_games)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
