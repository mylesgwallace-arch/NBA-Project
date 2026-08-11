# Sports AI

## Goal
Build an AI sports analyst capable of answering
statistically-backed questions using historical and
live sports data.

## Initial Scope
NBA only.

## Core capabilities
- Player analysis
- Team analysis
- Game prediction
- Trade simulation
- Statistical modeling
- Eventually live updates

## Planned architecture
Data → Database → Statistical Models → AI Agent → Website

See [PROJECT.md](PROJECT.md) for the full long-term specification and
[PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) for the current, verified state of
the repository (what is built, what is validated, and the exact next step).

## Setup

Requires Python 3.12+ (developed and tested with CPython 3.12).

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

`requirements.txt` pins the exact versions this project has been validated
against (pandas, numpy, scikit-learn, requests, beautifulsoup4, pytest).

## Data provenance

`data/` is not committed to source control (see `.gitignore`) because the raw
files are large historical NBA box-score exports. **Provenance for the raw
CSVs is not currently documented in this repository** — this is a known
reproducibility gap. To rebuild the database, place the following files in
`data/raw/` (exact filenames, as read by `src/load_data.py`):

```text
Games.csv
Players.csv
TeamHistories.csv
TeamStatistics.csv
TeamStatisticsExtended.csv
PlayerStatistics.csv
PlayerStatisticsExtended.csv
```

Each file is expected to contain one row per game / team-game / player-game
respectively, with the columns referenced throughout `src/` (inspect
`src/inspect_data.py` output or query the loaded SQLite tables directly to see
the exact schema — never assume column names without checking).

`data/raw/roster_change_events_valid.csv` is a small, separately curated file
satisfying the roster-change event contract described in
`src/roster_change_data.py` (`event_id`, `event_timestamp`, `team_id`,
`person_id`, `change_type`, `source`, `source_url`). It is optional and only
used by `src/player_impact.py --roster-events PATH`.

## Building the database

Run these from the project root, in order:

```powershell
.\.venv\Scripts\python src\create_database.py   # creates data/database/nba.db if missing
.\.venv\Scripts\python src\load_data.py         # loads the raw CSVs into SQLite tables
.\.venv\Scripts\python src\create_indexes.py    # creates lookup indexes on the large tables
.\.venv\Scripts\python src\check_database.py    # confirms the expected tables exist
```

`load_data.py` replaces existing tables (`if_exists="replace"`), so re-running
it is the standard way to refresh the database from updated raw CSVs.

## Building features and training the baseline model

```powershell
.\.venv\Scripts\python src\build_features.py          # writes data/processed/game_features.csv
.\.venv\Scripts\python -m src.train_baseline_model     # trains/evaluates candidate models, writes models/
```

`train_baseline_model.py` compares several candidate models (a trivial
home-win-rate reference, a rolling-stats logistic model, a player-history
logistic model, and an Elo rating system) on a chronological (not random)
holdout, and records the best-performing candidate (by holdout log loss) as
`recommended_model` in `models/baseline_metrics.json`.

## Predicting a matchup (CLI)

```powershell
.\.venv\Scripts\python src\main.py --home-team-id 1610612744 --away-team-id 1610612743 --game-date 2026-04-12
```

`--game-date` is an optional cutoff: the CLI uses only pregame data available
on or before that date (or the latest available data if omitted), so
predictions remain leakage-safe. The CLI automatically serves whichever model
`train_baseline_model.py` most recently recommended.

## Player-impact diagnostics

```powershell
.\.venv\Scripts\python src\player_impact.py
.\.venv\Scripts\python src\player_impact.py --roster-events data/raw/roster_change_events_valid.csv
```

This produces `models/player_impact_metrics.json`, an association-diagnostic
report (not a causal projection) documented in detail in
`PROJECT_CONTEXT.md`.

## Running tests

```powershell
.\.venv\Scripts\python -m pytest -q
```

Bare `pytest -q` (without `-m`) also works, since `pyproject.toml` configures
`pythonpath = ["."]` for the test run.
