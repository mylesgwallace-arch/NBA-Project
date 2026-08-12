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
BASKETBALL_REFERENCE_BASE_URL = "https://www.basketball-reference.com"
BASKETBALL_REFERENCE_SEASONS = (2022, 2023, 2024, 2025)
BENCHMARK_SEASONS = BASKETBALL_REFERENCE_SEASONS
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAYERS_PATH = ROOT / "data" / "raw" / "Players.csv"
DEFAULT_TEAM_HISTORIES_PATH = ROOT / "data" / "raw" / "TeamHistories.csv"


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
    args = parser.parse_args(argv)

    if args.validate is not None:
        events = load_roster_change_events(args.validate)
        summary = summarize_roster_change_events(events)
        print(json.dumps(summary, indent=2))
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
