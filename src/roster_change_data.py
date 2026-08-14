"""Validation and ingestion contract for independently sourced roster changes."""

import json
import re
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


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
PLAYER_MOVEMENT_COLUMNS = [
    "transaction_type",
    "transaction_date",
    "transaction_description",
    "team_id",
    "team_slug",
    "player_id",
    "player_slug",
    "additional_sort",
    "groupsort",
]
PLAYER_MOVEMENT_SOURCE = "NBA player movement raw CSV"
PLAYER_MOVEMENT_SOURCE_URL = (
    "https://github.com/mylesgwallace-arch/NBA-Project/blob/main/data/raw/"
    "nba_player_movement_raw.csv"
)
PLAYER_MOVEMENT_HIGH_CONFIDENCE = "high"
PLAYER_MOVEMENT_EXCLUDED = "excluded"
PLAYER_MOVEMENT_RECONSTRUCTION_RULES = {
    "signing_add": {
        "transaction_type": "Signing",
        "confidence_level": PLAYER_MOVEMENT_HIGH_CONFIDENCE,
        "description": "Emit one add event when the source row directly identifies a player signing or re-signing with a team.",
    },
    "waive_remove": {
        "transaction_type": "Waive",
        "confidence_level": PLAYER_MOVEMENT_HIGH_CONFIDENCE,
        "description": "Emit one remove event when the source row directly identifies a team waiving a player.",
    },
    "award_on_waivers_add": {
        "transaction_type": "AwardOnWaivers",
        "confidence_level": PLAYER_MOVEMENT_HIGH_CONFIDENCE,
        "description": "Emit one add event when the source row directly identifies a team claiming a player off waivers.",
    },
    "trade_destination_add": {
        "transaction_type": "Trade",
        "confidence_level": PLAYER_MOVEMENT_HIGH_CONFIDENCE,
        "description": "Emit one add event for the receiving team reported in the trade row.",
    },
    "trade_origin_remove": {
        "transaction_type": "Trade",
        "confidence_level": PLAYER_MOVEMENT_HIGH_CONFIDENCE,
        "description": "Emit one remove event for the origin team when the trade row includes a player id and the source-team id in additional_sort.",
    },
    "contract_converted_excluded": {
        "transaction_type": "ContractConverted",
        "confidence_level": PLAYER_MOVEMENT_EXCLUDED,
        "description": "Do not emit downstream roster events because the row proves only a contract-state change, not a roster add/remove.",
    },
    "trade_non_player_excluded": {
        "transaction_type": "Trade",
        "confidence_level": PLAYER_MOVEMENT_EXCLUDED,
        "description": "Do not emit downstream roster events for trade rows without a player id because they describe non-player consideration.",
    },
}
BASKETBALL_REFERENCE_BASE_URL = "https://www.basketball-reference.com"
BASKETBALL_REFERENCE_SEASONS = (2022, 2023, 2024, 2025)
BENCHMARK_SEASONS = BASKETBALL_REFERENCE_SEASONS
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAYERS_PATH = ROOT / "data" / "raw" / "Players.csv"
DEFAULT_TEAM_HISTORIES_PATH = ROOT / "data" / "raw" / "TeamHistories.csv"
DEFAULT_PLAYER_MOVEMENT_PATH = ROOT / "data" / "raw" / "nba_player_movement_raw.csv"
DEFAULT_PLAYER_MOVEMENT_EVENTS_PATH = (
    ROOT / "data" / "processed" / "nba_player_movement_roster_change_events.csv"
)
DEFAULT_PLAYER_MOVEMENT_AUDIT_PATH = (
    ROOT / "data" / "processed" / "nba_player_movement_roster_change_audit.csv"
)


def benchmark_season_for_page(page_season):
    """Normalize BBR page seasons to the repo's benchmark season labels.

    BBR transaction pages are named by the league season they represent (for example,
    ``NBA_2022_transactions`` is the 2021-22 season). These season labels are the
    canonical boundary used by the repo benchmark, even when some transaction dates
    fall in the prior calendar year as offseason moves.
    """
    season = int(page_season)
    if season not in BENCHMARK_SEASONS:
        return season
    return season


def _normalize_text(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def _normalize_team_token(value):
    return re.sub(r"[^a-z0-9]+", "", _normalize_text(value))


def _normalize_player_name(value):
    normalized = _normalize_text(value)
    for suffix in ("jr", "sr"):
        if normalized.endswith(f" {suffix}"):
            normalized = normalized[: -(len(suffix) + 1)]
    return normalized


def _season_team_histories(season, team_histories):
    season = int(season)
    return team_histories[
        (team_histories["seasonFounded"] <= season)
        & (team_histories["seasonActiveTill"] >= season)
    ].copy()


def _positive_int_or_none(value):
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric) or numeric < 1 or numeric % 1 != 0:
        return None
    return int(numeric)


def player_movement_reconstruction_rules():
    """Return the explicit reconstruction rules for nba_player_movement_raw.csv."""
    return {
        key: value.copy()
        for key, value in PLAYER_MOVEMENT_RECONSTRUCTION_RULES.items()
    }


def load_nba_player_movement_source(path=DEFAULT_PLAYER_MOVEMENT_PATH):
    """Load the immutable raw player-movement source CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"player movement data not found: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError("player movement data must be a CSV file")
    frame = pd.read_csv(path)
    missing = set(PLAYER_MOVEMENT_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(
            f"player movement data is missing columns: {sorted(missing)}"
        )
    if frame.empty:
        raise ValueError("player movement data must contain at least one row")
    return frame[PLAYER_MOVEMENT_COLUMNS].copy()


def _player_movement_source_payload(row, source_row_number):
    return {
        "source_row_number": int(source_row_number),
        "source_transaction_type": str(row["transaction_type"]).strip(),
        "source_transaction_date": str(row["transaction_date"]).strip(),
        "source_transaction_description": str(row["transaction_description"]).strip(),
        "source_team_id": _positive_int_or_none(row["team_id"]),
        "source_team_slug": "" if pd.isna(row["team_slug"]) else str(row["team_slug"]).strip(),
        "source_player_id": _positive_int_or_none(row["player_id"]),
        "source_player_slug": "" if pd.isna(row["player_slug"]) else str(row["player_slug"]).strip(),
        "source_additional_sort": "" if pd.isna(row["additional_sort"]) else str(row["additional_sort"]).strip(),
        "source_groupsort": "" if pd.isna(row["groupsort"]) else str(row["groupsort"]).strip(),
    }


def _player_movement_event(event_id, event_timestamp, team_id, person_id, change_type, rule, source_fields):
    event = {
        "event_id": event_id,
        "event_timestamp": event_timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "team_id": int(team_id),
        "person_id": int(person_id),
        "change_type": change_type,
        "source": PLAYER_MOVEMENT_SOURCE,
        "source_url": PLAYER_MOVEMENT_SOURCE_URL,
        "confidence_level": PLAYER_MOVEMENT_HIGH_CONFIDENCE,
        "reconstruction_rule": rule,
    }
    event.update(source_fields)
    return event


def normalize_nba_player_movement_records(records):
    """Normalize nba_player_movement_raw rows into high-confidence roster events plus audit rows."""
    raw = load_nba_player_movement_source(records) if isinstance(records, (str, Path)) else records.copy()
    missing = set(PLAYER_MOVEMENT_COLUMNS).difference(raw.columns)
    if missing:
        raise ValueError(
            f"player movement data is missing columns: {sorted(missing)}"
        )
    if raw.empty:
        raise ValueError("player movement data must contain at least one row")

    raw = raw[PLAYER_MOVEMENT_COLUMNS].copy()
    timestamps = pd.to_datetime(raw["transaction_date"], format="mixed", utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError("transaction_date must contain valid timestamps")

    normalized_events = []
    audit_rows = []
    for position, row in raw.reset_index(drop=True).iterrows():
        source_row_number = position + 1
        source_fields = _player_movement_source_payload(row, source_row_number)
        event_timestamp = timestamps.iloc[position]
        transaction_type = source_fields["source_transaction_type"]
        destination_team_id = source_fields["source_team_id"]
        person_id = source_fields["source_player_id"]
        origin_team_id = _positive_int_or_none(row["additional_sort"])

        row_events = []
        exclusion_reason = ""
        row_rules = []
        highest_confidence_level = ""

        if transaction_type == "Signing":
            if destination_team_id is not None and person_id is not None:
                row_rules = ["signing_add"]
                highest_confidence_level = PLAYER_MOVEMENT_HIGH_CONFIDENCE
                row_events.append(
                    _player_movement_event(
                        event_id=f"nba-player-movement-{source_row_number}-add-{destination_team_id}-{person_id}",
                        event_timestamp=event_timestamp,
                        team_id=destination_team_id,
                        person_id=person_id,
                        change_type="add",
                        rule="signing_add",
                        source_fields=source_fields,
                    )
                )
            else:
                exclusion_reason = "signing row is missing a positive team_id or player_id"
        elif transaction_type == "Waive":
            if destination_team_id is not None and person_id is not None:
                row_rules = ["waive_remove"]
                highest_confidence_level = PLAYER_MOVEMENT_HIGH_CONFIDENCE
                row_events.append(
                    _player_movement_event(
                        event_id=f"nba-player-movement-{source_row_number}-remove-{destination_team_id}-{person_id}",
                        event_timestamp=event_timestamp,
                        team_id=destination_team_id,
                        person_id=person_id,
                        change_type="remove",
                        rule="waive_remove",
                        source_fields=source_fields,
                    )
                )
            else:
                exclusion_reason = "waive row is missing a positive team_id or player_id"
        elif transaction_type == "AwardOnWaivers":
            if destination_team_id is not None and person_id is not None:
                row_rules = ["award_on_waivers_add"]
                highest_confidence_level = PLAYER_MOVEMENT_HIGH_CONFIDENCE
                row_events.append(
                    _player_movement_event(
                        event_id=f"nba-player-movement-{source_row_number}-add-{destination_team_id}-{person_id}",
                        event_timestamp=event_timestamp,
                        team_id=destination_team_id,
                        person_id=person_id,
                        change_type="add",
                        rule="award_on_waivers_add",
                        source_fields=source_fields,
                    )
                )
            else:
                exclusion_reason = "award-on-waivers row is missing a positive team_id or player_id"
        elif transaction_type == "Trade":
            if person_id is None:
                row_rules = ["trade_non_player_excluded"]
                exclusion_reason = "trade row has no player_id, so it only documents non-player consideration"
            elif destination_team_id is None or origin_team_id is None:
                exclusion_reason = "trade row is missing a positive receiving team_id or origin team id in additional_sort"
            else:
                row_rules = ["trade_origin_remove", "trade_destination_add"]
                highest_confidence_level = PLAYER_MOVEMENT_HIGH_CONFIDENCE
                row_events.extend(
                    [
                        _player_movement_event(
                            event_id=f"nba-player-movement-{source_row_number}-remove-{origin_team_id}-{person_id}",
                            event_timestamp=event_timestamp,
                            team_id=origin_team_id,
                            person_id=person_id,
                            change_type="remove",
                            rule="trade_origin_remove",
                            source_fields=source_fields,
                        ),
                        _player_movement_event(
                            event_id=f"nba-player-movement-{source_row_number}-add-{destination_team_id}-{person_id}",
                            event_timestamp=event_timestamp,
                            team_id=destination_team_id,
                            person_id=person_id,
                            change_type="add",
                            rule="trade_destination_add",
                            source_fields=source_fields,
                        ),
                    ]
                )
        elif transaction_type == "ContractConverted":
            row_rules = ["contract_converted_excluded"]
            exclusion_reason = "contract conversion changes contract status but does not prove a roster add/remove transition"
        else:
            raise ValueError(f"unsupported transaction_type for roster normalization: {transaction_type}")

        audit_rows.append(
            {
                **source_fields,
                "normalized_event_count": int(len(row_events)),
                "highest_confidence_level": highest_confidence_level,
                "reconstruction_status": "normalized" if row_events else "excluded",
                "reconstruction_rules": "|".join(row_rules),
                "exclusion_reason": exclusion_reason,
            }
        )
        normalized_events.extend(row_events)

    events_frame = pd.DataFrame(normalized_events)
    if events_frame.empty:
        raise ValueError("player movement normalization did not produce any roster events")
    validate_roster_change_events(events_frame)
    events_frame["event_timestamp"] = pd.to_datetime(
        events_frame["event_timestamp"], format="mixed", utc=True, errors="raise"
    )
    events_frame = events_frame.sort_values(["event_timestamp", "event_id"]).reset_index(drop=True)

    audit_frame = pd.DataFrame(audit_rows)
    audit_frame["source_transaction_timestamp"] = pd.to_datetime(
        audit_frame["source_transaction_date"], format="mixed", utc=True, errors="raise"
    )
    audit_frame = audit_frame.sort_values(
        ["source_transaction_timestamp", "source_row_number"]
    ).reset_index(drop=True)
    return events_frame, audit_frame


def summarize_nba_player_movement_normalization(events, audit_rows):
    """Return normalization counts and date coverage for the immutable player-movement source."""
    validated = validate_roster_change_events(events)
    audit = audit_rows.copy()
    audit_timestamps = pd.to_datetime(
        audit["source_transaction_date"], format="mixed", utc=True, errors="raise"
    )
    raw_dates = set(audit_timestamps.dt.normalize())
    event_dates = set(validated["event_timestamp"].dt.normalize())
    excluded_counts = (
        audit[audit["reconstruction_status"] == "excluded"]["exclusion_reason"]
        .value_counts()
        .to_dict()
    )
    raw_type_counts = audit["source_transaction_type"].value_counts().to_dict()
    return {
        "raw_row_count": int(len(audit)),
        "normalized_source_row_count": int((audit["normalized_event_count"] > 0).sum()),
        "excluded_source_row_count": int((audit["normalized_event_count"] == 0).sum()),
        "raw_transaction_type_counts": {key: int(value) for key, value in raw_type_counts.items()},
        "high_confidence_event_count": int(len(validated)),
        "add_count": int((validated["change_type"] == "add").sum()),
        "remove_count": int((validated["change_type"] == "remove").sum()),
        "first_raw_transaction_timestamp": audit_timestamps.min().isoformat(),
        "last_raw_transaction_timestamp": audit_timestamps.max().isoformat(),
        "first_event_timestamp": validated["event_timestamp"].min().isoformat(),
        "last_event_timestamp": validated["event_timestamp"].max().isoformat(),
        "raw_unique_transaction_dates": int(len(raw_dates)),
        "event_unique_transaction_dates": int(len(event_dates)),
        "missing_raw_transaction_dates_from_events": [
            timestamp.strftime("%Y-%m-%d")
            for timestamp in sorted(raw_dates.difference(event_dates))
        ],
        "excluded_rows_by_reason": {key: int(value) for key, value in excluded_counts.items()},
        "reconstruction_rules": player_movement_reconstruction_rules(),
    }


def load_player_registry(path=DEFAULT_PLAYERS_PATH):
    """Load the canonical player registry and normalize it for roster-event mapping."""
    player_frame = pd.read_csv(path)
    player_frame = player_frame.copy()
    player_frame["firstName"] = player_frame["firstName"].fillna("").astype(str).str.strip()
    player_frame["lastName"] = player_frame["lastName"].fillna("").astype(str).str.strip()
    player_frame["personId"] = pd.to_numeric(player_frame["personId"], errors="coerce")
    player_frame["full_name"] = (
        player_frame["firstName"].fillna("") + " " + player_frame["lastName"].fillna("")
    ).str.replace(r"\s+", " ", regex=True).str.strip()
    player_frame["name_key"] = player_frame["full_name"].map(_normalize_player_name)
    return player_frame


def load_team_registry(path=DEFAULT_TEAM_HISTORIES_PATH):
    """Load the canonical team registry and normalize it for roster-event mapping."""
    team_frame = pd.read_csv(path)
    team_frame = team_frame.copy()
    team_frame["teamId"] = pd.to_numeric(team_frame["teamId"], errors="coerce")
    team_frame["teamCity"] = team_frame["teamCity"].fillna("").astype(str).str.strip()
    team_frame["teamName"] = team_frame["teamName"].fillna("").astype(str).str.strip()
    team_frame["teamAbbrev"] = (
        team_frame["teamAbbrev"].fillna("").astype(str).str.strip().str.upper()
    )
    team_frame["teamAbbrev"] = team_frame["teamAbbrev"].str.replace(r"\s+", "", regex=True)
    team_frame["full_name"] = (
        team_frame["teamCity"] + " " + team_frame["teamName"]
    ).str.replace(r"\s+", " ", regex=True).str.strip()
    team_frame["name_key"] = team_frame["full_name"].map(_normalize_text)
    team_frame["abbr_key"] = team_frame["teamAbbrev"].map(_normalize_team_token)
    return team_frame


def resolve_person_id(player_name, season, players=None):
    """Resolve a player name to the repository personId if the player is in the registry."""
    if players is None:
        players = load_player_registry()
    name_key = _normalize_player_name(player_name)
    if not name_key:
        return None
    matches = players[players["name_key"] == name_key].copy()
    if matches.empty:
        return None
    if not matches["personId"].notna().any():
        return None
    if "fromYear" in matches.columns and "toYear" in matches.columns:
        active = matches[
            matches["fromYear"].fillna(-1).le(season)
            & matches["toYear"].fillna(2100).ge(season)
        ]
        if not active.empty:
            matches = active
    if len(matches) == 1:
        return int(matches.iloc[0]["personId"])
    return int(matches.iloc[0]["personId"])


def resolve_team_id(team_label, season, team_histories=None):
    """Resolve a team label or abbreviation to the repository teamId for the target season."""
    if team_histories is None:
        team_histories = load_team_registry()
    candidate_rows = _season_team_histories(season, team_histories)
    if team_label is None:
        return None
    label = str(team_label).strip()
    if not label:
        return None
    normalized = _normalize_text(label)
    abbreviations = [
        re.sub(r"[^A-Z0-9]+", "", label.upper())
        for label in [label, label.replace(".", "")]
    ]
    if normalized:
        direct_name = candidate_rows[candidate_rows["name_key"] == normalized]
        if not direct_name.empty:
            return int(direct_name.iloc[0]["teamId"])
    for abbrev in abbreviations:
        if not abbrev:
            continue
        direct_abbrev = candidate_rows[candidate_rows["abbr_key"] == _normalize_team_token(abbrev)]
        if not direct_abbrev.empty:
            return int(direct_abbrev.iloc[0]["teamId"])
    short_name = _normalize_text(label.replace("the ", ""))
    if short_name:
        for _, row in candidate_rows.iterrows():
            if short_name in _normalize_text(row["full_name"]) or short_name in _normalize_text(row["teamName"]):
                return int(row["teamId"])
    return None


def _transaction_paragraph_to_events(paragraph, season, players=None, team_histories=None):
    """Convert a single BBR transaction paragraph into one or more add/remove events."""
    if players is None:
        players = load_player_registry()
    if team_histories is None:
        team_histories = load_team_registry()
    text = paragraph.get_text(" ", strip=True)
    if not text:
        return []
    text_lower = text.lower()
    player_links = paragraph.find_all("a", href=True)
    player_links = [
        link for link in player_links if "/players/" in link.get("href", "")
    ]
    if not player_links:
        return []
    team_from = paragraph.find(attrs={"data-attr-from": True})
    team_to = paragraph.find(attrs={"data-attr-to": True})
    from_label = team_from.get_text(" ", strip=True) if team_from else None
    to_label = team_to.get_text(" ", strip=True) if team_to else None
    from_team_id = resolve_team_id(from_label, season, team_histories) if from_label else None
    to_team_id = resolve_team_id(to_label, season, team_histories) if to_label else None

    player_names = [link.get_text(" ", strip=True) for link in player_links]
    events = []
    if any(keyword in text_lower for keyword in ["waived", "released", "buyout", "let go"]):
        if from_team_id is None:
            return []
        for player_name in player_names[:2]:
            person_id = resolve_person_id(player_name, season, players)
            if person_id is None:
                continue
            events.append(
                {
                    "team_id": int(from_team_id),
                    "person_id": int(person_id),
                    "change_type": "remove",
                    "player_name": player_name,
                }
            )
        return events

    if any(keyword in text_lower for keyword in ["signed", "claimed", "selected", "drafted", "acquired", "re-signed"]):
        if to_team_id is None:
            return []
        for player_name in player_names[:2]:
            person_id = resolve_person_id(player_name, season, players)
            if person_id is None:
                continue
            events.append(
                {
                    "team_id": int(to_team_id),
                    "person_id": int(person_id),
                    "change_type": "add",
                    "player_name": player_name,
                }
            )
        return events

    if "traded" in text_lower:
        trade_candidates = []
        for player_name in player_names:
            person_id = resolve_person_id(player_name, season, players)
            if person_id is not None:
                trade_candidates.append((player_name, person_id))
        if not trade_candidates:
            return []
        if from_team_id is not None and to_team_id is not None:
            first_name, first_person = trade_candidates[0]
            events.append(
                {"team_id": int(from_team_id), "person_id": int(first_person), "change_type": "remove", "player_name": first_name}
            )
            events.append(
                {"team_id": int(to_team_id), "person_id": int(first_person), "change_type": "add", "player_name": first_name}
            )
            if len(trade_candidates) > 1:
                second_name, second_person = trade_candidates[1]
                events.append(
                    {"team_id": int(to_team_id), "person_id": int(second_person), "change_type": "remove", "player_name": second_name}
                )
                events.append(
                    {"team_id": int(from_team_id), "person_id": int(second_person), "change_type": "add", "player_name": second_name}
                )
        return events

    return []


def _extract_basketball_reference_season_events(season, players=None, team_histories=None):
    """Fetch and parse a single BBR season transaction page for a target season."""
    if players is None:
        players = load_player_registry()
    if team_histories is None:
        team_histories = load_team_registry()
    url = f"{BASKETBALL_REFERENCE_BASE_URL}/leagues/NBA_{season}_transactions.html"
    response = requests.get(
        url,
        timeout=30,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Upgrade-Insecure-Requests": "1",
            "Referer": "https://www.basketball-reference.com/",
            "Connection": "keep-alive",
        },
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    events = []
    unresolved = []
    for list_item in soup.find_all("li"):
        if list_item.find("span") is None:
            continue
        date_span = list_item.find("span")
        date_text = date_span.get_text(" ", strip=True)
        if not date_text:
            continue
        try:
            event_date = pd.to_datetime(date_text, format="%B %d, %Y")
        except (TypeError, ValueError):
            continue
        for paragraph in list_item.find_all("p"):
            if paragraph.get_text(" ", strip=True) == "":
                continue
            extracted = _transaction_paragraph_to_events(
                paragraph, season, players=players, team_histories=team_histories
            )
            if not extracted:
                unresolved.append(
                    {
                        "season": int(season),
                        "event_date": date_text,
                        "raw_text": paragraph.get_text(" ", strip=True),
                        "source_url": url,
                        "reason": "unresolved transaction text or missing player/team mapping",
                    }
                )
                continue
            for item in extracted:
                person_id = item["person_id"]
                team_id = item["team_id"]
                benchmark_season = benchmark_season_for_page(season)
                event_id = (
                    f"bbr-{benchmark_season}-{event_date.strftime('%Y-%m-%d')}-{team_id}-{person_id}-{item['change_type']}"
                )
                events.append(
                    {
                        "event_id": event_id,
                        "event_timestamp": event_date.strftime("%Y-%m-%dT00:00:00Z"),
                        "team_id": int(team_id),
                        "person_id": int(person_id),
                        "change_type": item["change_type"],
                        "source": "Basketball Reference",
                        "source_url": url,
                    }
                )
    return pd.DataFrame(events), pd.DataFrame(unresolved)


def fetch_basketball_reference_roster_changes(
    seasons=BASKETBALL_REFERENCE_SEASONS,
    players=None,
    team_histories=None,
):
    """Fetch and normalize BBR season transaction pages for roster changes.

    Returns a tuple of (events, unresolved_rows). The unresolved rows are withheld
    rather than inferred when the player or team cannot be mapped to the repository ids.
    """
    if players is None:
        players = load_player_registry()
    if team_histories is None:
        team_histories = load_team_registry()
    events = []
    unresolved = []
    for season in seasons:
        season_events, season_unresolved = _extract_basketball_reference_season_events(
            season, players=players, team_histories=team_histories
        )
        if not season_events.empty:
            events.append(season_events)
        if not season_unresolved.empty:
            unresolved.append(season_unresolved)
    all_events = (
        pd.concat(events, ignore_index=True) if events else pd.DataFrame(columns=ROSTER_CHANGE_COLUMNS)
    )
    all_unresolved = (
        pd.concat(unresolved, ignore_index=True) if unresolved else pd.DataFrame()
    )
    if not all_events.empty:
        all_events = all_events.drop_duplicates(subset=["event_id"]).reset_index(drop=True)
        all_events = validate_roster_change_events(all_events)
    return all_events, all_unresolved


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


def summarize_roster_change_events(events):
    """Return a concise validation summary for an already-normalized roster-change dataset."""
    frame = validate_roster_change_events(events)
    summary = {
        "event_count": int(len(frame)),
        "add_count": int((frame["change_type"] == "add").sum()),
        "remove_count": int((frame["change_type"] == "remove").sum()),
        "team_count": int(frame["team_id"].nunique()),
        "person_count": int(frame["person_id"].nunique()),
        "first_event_timestamp": frame["event_timestamp"].min().isoformat(),
        "last_event_timestamp": frame["event_timestamp"].max().isoformat(),
    }
    return summary


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Roster-change data validation and Basketball-Reference fetch utilities.")
    parser.add_argument(
        "--validate",
        type=Path,
        help="Validate a CSV of timestamped roster-change events and print a summary JSON.",
    )
    parser.add_argument(
        "--normalize-player-movement",
        type=Path,
        help="Normalize data/raw/nba_player_movement_raw.csv into high-confidence roster_change_events and an audit CSV.",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=list(BASKETBALL_REFERENCE_SEASONS),
        help="NBA seasons to fetch from basketball-reference.com (e.g. 2022 2023 2024 2025).",
    )
    parser.add_argument(
        "--output-events",
        type=Path,
        default=ROOT / "data" / "processed" / "bbr_roster_changes_2022_2025.csv",
        help="Where to write valid normalized roster events.",
    )
    parser.add_argument(
        "--output-unresolved",
        type=Path,
        default=ROOT / "data" / "processed" / "bbr_roster_changes_unresolved_2022_2025.csv",
        help="Where to write unresolved transaction rows that could not be mapped to repo IDs.",
    )
    parser.add_argument(
        "--output-player-movement-events",
        type=Path,
        default=DEFAULT_PLAYER_MOVEMENT_EVENTS_PATH,
        help="Where to write normalized high-confidence roster events from nba_player_movement_raw.csv.",
    )
    parser.add_argument(
        "--output-player-movement-audit",
        type=Path,
        default=DEFAULT_PLAYER_MOVEMENT_AUDIT_PATH,
        help="Where to write the source-row audit table for nba_player_movement_raw.csv normalization.",
    )
    args = parser.parse_args(argv)

    if args.validate is not None:
        events = load_roster_change_events(args.validate)
        summary = summarize_roster_change_events(events)
        print(json.dumps(summary, indent=2))
        return 0

    if args.normalize_player_movement is not None:
        events, audit = normalize_nba_player_movement_records(
            args.normalize_player_movement
        )
        summary = summarize_nba_player_movement_normalization(events, audit)
        args.output_player_movement_events.parent.mkdir(parents=True, exist_ok=True)
        args.output_player_movement_audit.parent.mkdir(parents=True, exist_ok=True)
        events.to_csv(args.output_player_movement_events, index=False)
        audit.to_csv(args.output_player_movement_audit, index=False)
        print(json.dumps(summary, indent=2))
        print(f"events_path={args.output_player_movement_events}")
        print(f"audit_path={args.output_player_movement_audit}")
        return 0

    events, unresolved = fetch_basketball_reference_roster_changes(seasons=tuple(args.seasons))
    args.output_events.parent.mkdir(parents=True, exist_ok=True)
    args.output_unresolved.parent.mkdir(parents=True, exist_ok=True)
    if not events.empty:
        events.to_csv(args.output_events, index=False)
    if not unresolved.empty:
        unresolved.to_csv(args.output_unresolved, index=False)
    print(f"valid_events={len(events)}")
    print(f"unresolved_rows={len(unresolved)}")
    print(f"events_path={args.output_events}")
    print(f"unresolved_path={args.output_unresolved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
