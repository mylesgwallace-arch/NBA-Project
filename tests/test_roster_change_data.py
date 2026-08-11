import pandas as pd
import pytest

from src.roster_change_data import load_roster_change_events, validate_roster_change_events


def _events():
    return pd.DataFrame(
        [
            {
                "event_id": "b",
                "event_timestamp": "2024-10-02T15:00:00-04:00",
                "team_id": 20,
                "person_id": 200,
                "change_type": "REMOVE",
                "source": "independent source",
                "source_url": "https://example.test/event-b",
            },
            {
                "event_id": "a",
                "event_timestamp": "2024-10-01T15:00:00-04:00",
                "team_id": 10,
                "person_id": 100,
                "change_type": "add",
                "source": "independent source",
                "source_url": "https://example.test/event-a",
            },
        ]
    )


def test_roster_events_are_normalized_and_time_ordered():
    result = validate_roster_change_events(_events())

    assert result["event_id"].tolist() == ["a", "b"]
    assert result["change_type"].tolist() == ["add", "remove"]
    assert result["event_timestamp"].dt.tz is not None
    assert result["team_id"].dtype == "int64"


def test_loader_reads_csv_through_the_same_validation_path(tmp_path):
    path = tmp_path / "roster_changes.csv"
    _events().to_csv(path, index=False)

    result = load_roster_change_events(path)

    assert len(result) == 2
    assert result["event_id"].tolist() == ["a", "b"]


@pytest.mark.parametrize(
    "mutator, message",
    [
        (lambda frame: frame.assign(event_id=["a", "a"]), "event_id"),
        (lambda frame: frame.assign(change_type="trade"), "change_type"),
        (lambda frame: frame.assign(source_url="not-a-url"), "source_url"),
        (lambda frame: frame.assign(event_timestamp="not-a-time"), "event_timestamp"),
    ],
)
def test_invalid_source_data_fails_explicitly(mutator, message):
    with pytest.raises(ValueError, match=message):
        validate_roster_change_events(mutator(_events()))


def test_naive_timestamps_are_accepted_and_normalized_to_utc():
    events = _events().assign(event_timestamp="2024-10-01 15:00:00")

    result = validate_roster_change_events(events)

    assert result.loc[0, "event_timestamp"].tzinfo is not None
