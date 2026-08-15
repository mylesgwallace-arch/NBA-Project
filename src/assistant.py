"""Deterministic natural-language interface over the validated tool layer.

This module maps a user's plain-language question to a deterministic tool call,
dispatches it through ``src.tools.execute_tool``, and renders the returned
structured envelope as a plain-language answer while surfacing the tool's
assumptions and limitations.

Design rules:

* The assistant never fabricates an answer from its own knowledge. Every value
  in a response is derived from the deterministic tool envelope produced by
  ``execute_tool``.
* Question parsing is pattern-based and deterministic, so the interface is
  testable and requires no language model. A future LLM layer can call the same
  ``execute_tool`` interface directly and reuse the structured envelopes.
* Ambiguous, incomplete, or unsupported questions produce a clear structured
  error instead of a guess.
* Player-impact content stays strictly association-only: the assistant passes
  through the diagnostic's confidence and its "NOT a causal forecast"
  limitation verbatim.
"""

import argparse
import json
import re
import sqlite3
from pathlib import Path

try:
    from src.tools import execute_tool, list_tools
except ImportError:  # pragma: no cover - direct-script support
    from tools import execute_tool, list_tools


ROOT = Path(__file__).resolve().parents[1]
TEAM_DB_PATH = ROOT / "data" / "database" / "nba.db"
NBA_TEAM_ID_MIN = 1610612737
NBA_TEAM_ID_MAX = 1610612766
DEFAULT_SEASON = 2025

SEASON_PATTERN = re.compile(r"\b(20\d{2})\b")

# Small nickname/city alias map in addition to the database city/name labels
# (e.g. "OKC", "Sixers", "Mavs"). Matched against lowercased question text.
TEAM_ALIASES = {
    "okc": 1610612760,
    "oklahoma city": 1610612760,
    "sixers": 1610612755,
    "mavs": 1610612742,
    "blazers": 1610612757,
    "dubs": 1610612744,
    "knicks": 1610612752,
    "clippers": 1610612746,
    "nola": 1610612740,
    "pels": 1610612740,
    "spurs": 1610612759,
    "jazz": 1610612762,
    "grizzlies": 1610612763,
    "wolves": 1610612750,
    "timberwolves": 1610612750,
    "kings": 1610612758,
    "suns": 1610612756,
    "warriors": 1610612744,
    "bulls": 1610612741,
    "cavaliers": 1610612739,
    "cavs": 1610612739,
    "pistons": 1610612765,
    "pacers": 1610612754,
    "bucks": 1610612749,
    "heat": 1610612748,
    "magic": 1610612753,
    "raptors": 1610612761,
    "wizards": 1610612764,
    "hornets": 1610612766,
    "hawks": 1610612737,
    "nets": 1610612751,
    "celtics": 1610612738,
    "lakers": 1610612747,
    "rockets": 1610612745,
    "nuggets": 1610612743,
}


def normalize_text(text):
    return re.sub(r"[^a-z0-9 ]", " ", str(text).lower())


def load_team_labels(db_path=TEAM_DB_PATH):
    """Return {teamId: "City Name"} for the 30 current NBA franchises."""
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT teamId, teamCity, teamName
            FROM team_histories
            WHERE seasonActiveTill >= 2100
              AND teamId BETWEEN ? AND ?
            ORDER BY teamCity, teamName
            """,
            (NBA_TEAM_ID_MIN, NBA_TEAM_ID_MAX),
        ).fetchall()
    return {team_id: f"{city} {name}" for team_id, city, name in rows}


def load_team_mentions(db_path=TEAM_DB_PATH):
    """Return {normalized phrase: [teamId, ...]} from DB labels plus aliases.

    Ambiguous city-only phrases (e.g. "los angeles" matches both the Lakers
    and the Clippers) map to multiple teamIds; extraction skips phrases that
    are ambiguous so a longer, more specific phrase wins.
    """
    mentions = {}
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT teamId, teamCity, teamName
            FROM team_histories
            WHERE seasonActiveTill >= 2100
              AND teamId BETWEEN ? AND ?
            """,
            (NBA_TEAM_ID_MIN, NBA_TEAM_ID_MAX),
        ).fetchall()
    for team_id, city, name in rows:
        for phrase in (
            normalize_text(city),
            normalize_text(name),
            normalize_text(f"{city} {name}"),
        ):
            mentions.setdefault(phrase, set()).add(team_id)
    for phrase, team_id in TEAM_ALIASES.items():
        mentions.setdefault(phrase, set()).add(team_id)
    return {phrase: sorted(team_ids) for phrase, team_ids in mentions.items()}


def extract_team_ids(question, db_path=TEAM_DB_PATH):
    """Return the distinct teamIds mentioned in a question, in mention order."""
    normalized = normalize_text(question)
    mentions = load_team_mentions(db_path)
    # Sort by phrase length so "Los Angeles Lakers" beats "Los Angeles".
    found = []
    remaining = normalized
    for phrase in sorted(mentions, key=len, reverse=True):
        if not phrase or len(mentions[phrase]) != 1:
            continue
        # Match whole phrases within the question, avoiding partial hits.
        pattern = r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])"
        if re.search(pattern, remaining):
            team_id = mentions[phrase][0]
            if team_id not in found:
                found.append(team_id)
            remaining = remaining.replace(phrase, " ")
    return found


def extract_season(question):
    """Return the first 20xx season year mentioned, or the default season."""
    match = SEASON_PATTERN.search(question)
    if match:
        return int(match.group(1))
    return None


def extract_player_ids(question, db_path=TEAM_DB_PATH):
    """Return player personIds matching a full-name mention in the question."""
    normalized = normalize_text(question)
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT personId, firstName, lastName
            FROM players
            WHERE nbaFlag = 1
            ORDER BY lastName, firstName
            """
        ).fetchall()
    matches = []
    for person_id, first_name, last_name in rows:
        full = normalize_text(f"{first_name} {last_name}")
        if not full or " " not in full:
            continue
        pattern = r"(?<![a-z0-9])" + re.escape(full) + r"(?![a-z0-9])"
        if re.search(pattern, normalized):
            matches.append(int(person_id))
    return matches


def _one_team(team_ids, question):
    if not team_ids:
        raise ValueError(
            "No team could be identified in the question. Use a current "
            "franchise name or city, e.g. 'Boston Celtics'."
        )
    if len(team_ids) > 1:
        raise ValueError(
            f"Multiple teams identified ({team_ids}); please ask about a "
            "single team."
        )
    return team_ids[0]


def _two_teams(team_ids, question):
    if len(team_ids) < 2:
        raise ValueError(
            "Two different teams are required for this question. "
            "Use names like 'Boston Celtics vs Los Angeles Lakers'."
        )
    if len(team_ids) > 2:
        raise ValueError(
            f"Too many teams identified ({team_ids}); please ask about two teams."
        )
    return team_ids[0], team_ids[1]


def _one_player(player_ids):
    if not player_ids:
        raise ValueError(
            "No player could be identified in the question. Use a full player "
            "name, e.g. 'Steven Adams'."
        )
    if len(player_ids) > 1:
        raise ValueError(
            f"Multiple players match the name ({player_ids}); please use a "
            "more specific full name."
        )
    return player_ids[0]


def resolve_intent(question):
    """Map a plain-language question to (tool_name, parameters).

    Deterministic pattern routing only. Raises ValueError for ambiguous or
    unsupported questions.
    """
    question = (question or "").strip()
    if not question:
        raise ValueError("The question is empty.")
    normalized = normalize_text(question)
    team_ids = extract_team_ids(question)
    player_ids = extract_player_ids(question)
    season = extract_season(question) or DEFAULT_SEASON

    # Head-to-head record.
    if re.search(r"head.to.head|headtohead|series record|h2h", normalized):
        team_a, team_b = _two_teams(team_ids, question)
        return "head_to_head", {"team_a_id": team_a, "team_b_id": team_b,
                                "season": season}

    # Player impact diagnostic (association-only).
    if re.search(r"player.impact|impact of |diagnostic", normalized) and player_ids:
        person_id = _one_player(player_ids)
        return "player_impact", {"person_id": person_id}

    # Single-team record.
    if re.search(r"record|wins and losses|win.loss", normalized):
        team_id = _one_team(team_ids, question)
        return "team_record", {"team_id": team_id, "season": season}

    # Projected playoff teams / standings.
    if re.search(
        r"projected playoff|playoff team|playoff field|standings|projected standings",
        normalized,
    ):
        return "simulate_season", {"season": season}

    # Team projection: seed probability / projected wins.
    if re.search(
        r"seed|projected wins|how many wins|mean wins|projection", normalized
    ) and team_ids:
        team_id = _one_team(team_ids, question)
        return "team_projection", {"team_id": team_id, "season": season}

    # Player scenario (player + matchup).
    if player_ids and len(team_ids) == 2:
        person_id = _one_player(player_ids)
        home, away = _two_teams(team_ids, question)
        return "player_scenario", {"home_team_id": home, "away_team_id": away,
                                   "person_id": person_id}

    # Single-game matchup prediction (favored / vs / win chance).
    if re.search(r"favored|favorite|vs|versus|win chance|win probability|who wins|beat", normalized):
        home, away = _two_teams(team_ids, question)
        return "predict_matchup", {"home_team_id": home, "away_team_id": away}

    # Team name resolution.
    if re.search(r"team id|teamid|resolve", normalized) and team_ids:
        team_id = _one_team(team_ids, question)
        return "resolve_team_name", {"team": question}

    raise ValueError(
        "I could not map that question to a supported analytical tool. "
        "Try asking about a matchup ('Who is favored in Celtics vs Lakers?'), "
        "projected playoffs, a team's projected wins or seed, a player-impact "
        "diagnostic, a team record, or a head-to-head series."
    )


def _team_label(team_id, labels):
    return labels.get(team_id, str(team_id))


def render_envelope(envelope, labels=None):
    """Render a tool envelope into a plain-language answer (no fabrication)."""
    labels = labels or {}
    if envelope["status"] == "error":
        error = envelope.get("error") or {}
        return (
            f"I could not answer that: {error.get('message', 'unknown error')}"
        )
    if envelope["status"] == "unavailable":
        error = envelope.get("error") or {}
        return (
            f"That analysis is not available: {error.get('message', 'no data')}"
        )

    tool = envelope["tool"]
    data = envelope.get("data") or {}

    if tool == "predict_matchup":
        prediction = data.get("prediction") or {}
        home = _team_label(data.get("home_team_id"), labels)
        away = _team_label(data.get("away_team_id"), labels)
        home_prob = prediction.get("home_win_probability")
        away_prob = prediction.get("away_win_probability")
        favorite = prediction.get("home_team_prediction")
        if home_prob is None or away_prob is None:
            return "The prediction result is missing probabilities."
        favorite_label = home if favorite == "favorite" else away
        return (
            f"Using the frozen production model "
            f"({prediction.get('model', 'elo_boosted_ensemble')}), {favorite_label} "
            f"is favored at {max(home_prob, away_prob):.1%}. {home} is given "
            f"{home_prob:.1%} and {away} {away_prob:.1%}."
        )

    if tool == "simulate_season":
        projection = data.get("projection") or {}
        seedings = projection.get("projected_seedings") or []
        league = projection.get("league_summary") or {}
        season = projection.get("season")
        if not seedings:
            return "The season projection returned no playoff field."
        lines = [
            f"Projected {season} direct-playoff field (top six per conference):"
        ]
        for slot in seedings:
            lines.append(
                f"  {slot['conference']} seed {slot['seed']}: "
                f"{_team_label(slot['teamId'], labels)} ({slot['probability']:.1%})"
            )
        if league:
            lines.append(
                f"League mean projected wins: {league.get('league_mean_wins'):.1f} "
                f"across {league.get('n_teams')} teams."
            )
        return "\n".join(lines)

    if tool == "team_projection":
        projection = data.get("projection") or {}
        team_id = data.get("team_id")
        label = _team_label(team_id, labels)
        mean = projection.get("mean_wins")
        playoff = projection.get("direct_playoff_probability")
        seed_1 = projection.get("p_seed_1")
        seed = projection.get("mean_conference_seed")
        lines = [
            f"{label} is projected for {mean:.1f} mean wins "
            f"(median {projection.get('median_wins'):.1f}, "
            f"[{projection.get('p5_wins'):.0f}-{projection.get('p95_wins'):.0f}])."
        ]
        if playoff is not None:
            lines.append(f"Direct-playoff probability: {playoff:.1%}.")
        if seed is not None:
            lines.append(f"Mean projected conference seed: {seed:.1f}.")
        if seed_1 is not None:
            lines.append(f"Probability of the 1 seed: {seed_1:.1%}.")
        return " ".join(lines)

    if tool == "player_impact":
        diagnostic = data.get("diagnostic") or {}
        person_id = diagnostic.get("person_id")
        confidence = data.get("confidence")
        direction = diagnostic.get("direction", "addition")
        net = diagnostic.get("player_net_rating")
        change = diagnostic.get("estimated_net_rating_change")
        games = diagnostic.get("prior_games")
        line = (
            f"Player-impact diagnostic (personId {person_id}, {confidence} "
            f"confidence, {games} prior games): the estimate for a {direction} "
            f"is a {change:+.2f} net-rating change from a {net:+.2f} player "
            f"net rating."
        )
        return line + " This is an association-only estimate, not a causal forecast."

    if tool == "player_scenario":
        scenario = data.get("scenario") or {}
        base = scenario.get("base_prediction") or {}
        home = _team_label(data.get("home_team_id"), labels)
        away = _team_label(data.get("away_team_id"), labels)
        impact = scenario.get("player_impact") or {}
        model = scenario.get("model", "elo_boosted_ensemble")
        line = (
            f"{model} gives {home} a {base.get('home_win_probability', 0):.1%} "
            f"chance against {away}; the player-impact diagnostic "
            f"(personId {impact.get('person_id')}) is association-only and is "
            f"NOT used to change the model probability."
        )
        return line

    if tool == "team_record":
        label = _team_label(data.get("team_id"), labels)
        season = data.get("season")
        record = (
            f"{label} went {data.get('wins')}-{data.get('losses')} "
            f"({data.get('games')} games)"
        )
        if season:
            record += f" in the {season} season"
        return record + " (regular season, from the repository database)."

    if tool == "head_to_head":
        team_a = _team_label(data.get("team_a_id"), labels)
        team_b = _team_label(data.get("team_b_id"), labels)
        season = data.get("season")
        line = (
            f"In the regular season{(' of ' + str(season)) if season else ''}, "
            f"{team_a} leads {team_b} {data.get('team_a_wins')}-"
            f"{data.get('team_b_wins')} in {data.get('games')} head-to-head games."
        )
        return line

    if tool == "resolve_team_name":
        return f"{data.get('team')} has teamId {data.get('team_id')}."

    return f"Tool '{tool}' produced a result; see the structured envelope."


def answer_question(question):
    """Answer a plain-language question using the deterministic tool layer.

    Returns a structured dict: {question, tool, parameters, envelope, answer}.
    """
    labels = load_team_labels()
    try:
        tool_name, parameters = resolve_intent(question)
    except ValueError as exc:
        return {
            "question": question,
            "tool": None,
            "parameters": None,
            "status": "error",
            "answer": f"I could not answer that: {exc}",
            "envelope": None,
        }
    envelope = execute_tool(tool_name, parameters)
    answer = render_envelope(envelope, labels=labels)
    return {
        "question": question,
        "tool": tool_name,
        "parameters": parameters,
        "status": envelope["status"],
        "answer": answer,
        "envelope": envelope,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Deterministic natural-language interface to the validated NBA "
            "analytics tool layer."
        )
    )
    parser.add_argument(
        "question",
        type=str,
        nargs="?",
        help="The natural-language question to answer.",
    )
    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="Print the available analytical tools and exit.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full structured envelope alongside the plain answer.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.list_tools:
        print(json.dumps(list_tools(), indent=2))
        return 0
    if not args.question:
        raise SystemExit("Provide a question to answer, or use --list-tools.")
    result = answer_question(args.question)
    print(result["answer"])
    if args.json:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())