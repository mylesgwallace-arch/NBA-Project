"""Validation and ingestion contract for independently sourced roster changes."""

from pathlib import Path

import pandas as pd


ROSTER_CHANGE_COLUMNS = [
    "event_id",
    "event_timestamp",
    "team_id",
    "person_id",
    "change_type",
    "source",
    "source_url",
]
CHANGE_TYPES = {"add", "remove"}


def validate_roster_change_events(events):
    """Return normalized roster events or raise for an unusable source extract."""
    missing = set(ROSTER_CHANGE_COLUMNS).difference(events.columns)
    if missing:
        raise ValueError(
            f"roster change data is missing columns: {sorted(missing)}"
        )
    if events.empty:
        raise ValueError("roster change data must contain at least one event")

    result = events[ROSTER_CHANGE_COLUMNS].copy()
    for column in ["event_id", "source", "source_url"]:
        result[column] = result[column].astype("string").str.strip()
        if result[column].isna().any() or (result[column] == "").any():
            raise ValueError(f"roster change column '{column}' contains blank values")

    if result["event_id"].duplicated().any():
        raise ValueError("roster change event_id values must be unique")

    timestamps = pd.to_datetime(
        result["event_timestamp"], format="mixed", utc=True, errors="coerce"
    )
    if timestamps.isna().any():
        raise ValueError("event_timestamp must contain valid timestamps")
    result["event_timestamp"] = timestamps

    for column in ["team_id", "person_id"]:
        values = pd.to_numeric(result[column], errors="coerce")
        if values.isna().any() or (values < 1).any() or (values % 1 != 0).any():
            raise ValueError(f"{column} must contain positive integer identifiers")
        result[column] = values.astype("int64")

    result["change_type"] = result["change_type"].astype("string").str.strip().str.lower()
    if not result["change_type"].isin(CHANGE_TYPES).all():
        raise ValueError(f"change_type must be one of: {sorted(CHANGE_TYPES)}")
    if ~result["source_url"].str.startswith(("http://", "https://")).all():
        raise ValueError("source_url must use http:// or https://")

    return result.sort_values(["event_timestamp", "event_id"]).reset_index(drop=True)


def load_roster_change_events(path):
    """Load and validate a CSV supplied by an independent roster-change source."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"roster change data not found: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError("roster change data must be a CSV file")
    return validate_roster_change_events(pd.read_csv(path))
