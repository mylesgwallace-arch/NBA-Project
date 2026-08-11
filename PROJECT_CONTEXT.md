# Sports AI — Current Project Context

> **This file describes the current state of the repository.**
>
> `PROJECT.md` describes the long-term project specification.
>
> When deciding what to work on next, prioritize this file and the actual repository/database over the long-term roadmap.

---

# 1. Current Objective

The current objective is to build and validate the **historical NBA analytics foundation**.

The immediate objective was to restore the raw CSV to SQLite to feature-engineering
pipeline after `src/build_features.py` loaded 0 team-game rows.

That blocker is resolved. The database was rebuilt from the confirmed raw CSV files,
and the feature-building pipeline now produces nonzero output. Generated features
and the historical dataset have now been independently validated and covered by
focused regression checks. The baseline prediction model has also been rebuilt and
evaluated with a chronological holdout.

---

# 2. Current Repository

Project root:

```text
C:\Users\myles\Git NBA Proj
```

Python environment:

```text
.venv/
```

General structure:

```text
data/
├── raw/
├── processed/
└── database/

models/
src/
tests/
notebooks/

.gitignore
README.md
PROJECT.md
PROJECT_CONTEXT.md
```

---

# 3. Current Database

SQLite database:

```text
data/database/nba.db
```

The database has been successfully created and reloaded from the raw CSV files.

## Confirmed tables

```text
games
player_statistics
player_statistics_extended
players
team_histories
team_statistics
team_statistics_extended
```

### Important

The actual database uses lowercase/underscore table names.

For example:

```sql
team_statistics
```

not:

```sql
TeamStatistics
```

Do not assume schema names.

Always inspect the database before writing SQL against it.

## Confirmed row counts

```text
games: 73,279
team_statistics: 146,560
team_statistics regular season: 130,014
team_statistics_extended: 79,724
processed model-ready feature rows: 129,836
```

---

# 4. Existing Database Utility

An existing database validation script is:

```text
src/check_database.py
```

**Do not create another equivalent script such as `check_db.py`.**

If database validation is needed, use or improve the existing script.

Avoid creating duplicate utilities when equivalent functionality already exists.

---

# 5. Existing Scripts

Known relevant scripts include:

```text
src/check_database.py
src/create_indexes.py
src/build_features.py
```

Before creating a new script:

1. Search `src/`.
2. Determine whether the required functionality already exists.
3. Reuse or modify the existing script if appropriate.

---

# 6. Database Schema Issues Already Encountered

An earlier implementation attempted to query:

```text
TeamStatistics
```

but the actual table is:

```text
team_statistics
```

Another earlier implementation attempted to reference:

```text
gameDate
```

but the relevant date field is:

```text
gameDateTimeEst
```

These mistakes demonstrate why the actual schema must be inspected rather than assumed.

---

# 7. Feature Pipeline

Feature-building script:

```text
src/build_features.py
```

Output:

```text
data/processed/game_features.csv
```

The script executes successfully. The latest verified run reports:

```text
Loaded 130,014 team-game rows
Saved 129,836 rows
```

The output file is now populated, model-ready, and has the expected
rolling-statistic columns.

The generated CSV has passed the current feature-content validation and is the
validated starting point for baseline model training.

---

# 8. Current Feature Columns

`build_features.py` currently creates the following columns:

```text
gameId
gameDateTimeEst
teamId
opponentTeamId
home
win
teamScore
opponentScore
assists
steals
blocks
fieldGoalsPercentage
threePointersPercentage
freeThrowsPercentage
reboundsTotal
turnovers
plusMinusPoints
season
teamScore_rolling_10
opponentScore_rolling_10
assists_rolling_10
steals_rolling_10
blocks_rolling_10
fieldGoalsPercentage_rolling_10
threePointersPercentage_rolling_10
freeThrowsPercentage_rolling_10
reboundsTotal_rolling_10
turnovers_rolling_10
plusMinusPoints_rolling_10
```

These columns are the intended starting point for game-level feature engineering. The
row count, duplicate handling, and rolling-window values have passed focused
validation. The season definition still needs review before model training.

---

# 9. Resolved Blocker and Remaining Work

## Resolved problem

The SQLite `team_statistics` table existed with the expected schema but contained 0
rows, even though `data/raw/TeamStatistics.csv` contained 146,560 rows. Reloading the
database with `src/load_data.py` restored the table. The `gameType = 'Regular Season'`
filter was valid and returned 130,014 rows after the reload.

The feature pipeline now returns:

```text
Loaded 130,014 team-game rows
Saved 129,836 rows
```

## What is working

- Raw CSV ingestion via `src/load_data.py`.
- SQLite tables and expected lowercase/underscore table names.
- Regular-season team-game extraction in `src/build_features.py`.
- Chronological rolling features using only previous games.
- Model-ready output at `data/processed/game_features.csv` with complete current
  and rolling predictors.
- Basic table validation via `src/check_database.py`.
- Focused feature regression coverage in `tests/test_feature_pipeline.py`.

## What is not complete

- The baseline model uses only team rolling history and does not yet include
  opponent-strength, rest, roster, or player-availability features.
- The feature pipeline now labels `season` by the calendar year in which the NBA
  season starts: October-December use that year, and January-September use the
  preceding year.

## Exact next step

Compare the validated rolling logistic baseline against a second leakage-safe
strength baseline, such as an Elo-style rating calculated chronologically from
completed games. Keep the same chronological holdout and metrics before adding
new feature families.

The earlier investigation confirmed the following facts and should not be repeated as
assumptions:

1. The schema of `team_statistics`.
2. The number of rows in `team_statistics`.
3. The columns available in `team_statistics`.
4. The relevant `gameType` value: `Regular Season`.
5. That the table contains game-level team statistics.
6. That the current SQL filter does not exclude all rows after reload.
7. That the expected feature columns exist in the raw team-statistics data.
8. That `build_features.py` can use `gameDateTimeEst` directly; no join with `games`
	 is required for the current feature set.

---

# 10. Previous NBA Prediction Model

A previous NBA prediction project produced approximately:

```text
Accuracy: 0.62278
Log Loss: 0.66614
Brier Score: 0.23659
```

These numbers are a **baseline**.

The eventual goal is to reproduce the useful parts of the previous model inside this project and then improve it using cleaner data, better features, and better validation.

Do not claim improvement until the models are evaluated on appropriate historical data.

---

# 11. Current Development Priority

Priority order:

```text
1. Fix the 0-row feature pipeline
2. Validate generated game features
3. Validate the historical dataset
4. Build/reproduce the baseline prediction model
5. Establish proper train/test or time-based validation
6. Improve the baseline model
7. Add more advanced player/team features
8. Develop player-impact modeling
9. Develop simulations
10. Build AI/tool layer
11. Add live data
12. Build website
```

This priority is a current development state, not a permanent project roadmap.

Update it as the project progresses.

---

# 12. Rules for AI Coding Assistants

Before modifying the project:

### Read the context

Read:

```text
PROJECT_CONTEXT.md
```

Then read:

```text
PROJECT.md
```

when broader architectural context is necessary.

### Inspect the repository

Do not assume that a file exists.

Search the repository before creating new files.

### Inspect the database

When working with SQLite:

* inspect table names
* inspect columns
* inspect sample rows
* inspect row counts
* inspect distinct categorical values

Do not invent schema names.

### Avoid duplicate scripts

Before creating a utility script, search the repository for existing functionality.

For example, an existing:

```text
src/check_database.py
```

should not be duplicated with:

```text
src/check_db.py
```

without a specific reason.

### Diagnose before changing

If something fails:

1. Reproduce the problem.
2. Inspect the relevant code/data.
3. Identify the root cause.
4. Make the smallest appropriate fix.
5. Run the code again.
6. Verify the result.

Do not blindly modify code until the error disappears.

### Do not jump ahead

If an earlier pipeline stage is broken, fix it before building dependent features.

### Keep the project reproducible

Prefer clear, simple code and documented dependencies.

---

# 13. How to Decide What to Do Next

When asked:

> "What's next?"

the AI should:

1. Read this file.
2. Inspect the repository.
3. Identify the earliest incomplete or broken dependency.
4. Verify the current state rather than relying only on this document.
5. Recommend the next concrete task.
6. If explicitly asked to proceed, implement it.
7. Test the result.
8. Update this file if the project state materially changes.

The AI should **not** simply choose the next item from the long-term roadmap.

---

# 14. Source of Truth

Use this priority when information conflicts:

```text
1. Actual repository/code
2. Actual database/schema/data
3. PROJECT_CONTEXT.md
4. PROJECT.md
```

`PROJECT_CONTEXT.md` should accurately describe the repository, but the repository itself is ultimately authoritative.

If this file says a script exists but it does not exist, inspect the repository and correct this file rather than assuming the script exists.

---

# 15. Updating This File

Update `PROJECT_CONTEXT.md` when there is a meaningful state change, such as:

* a major script is created
* a pipeline stage is completed
* a major bug is fixed
* the database schema changes
* a new blocker appears
* a model is trained
* a model is evaluated
* a major architectural decision is made

Do not turn this file into a detailed chronological diary.

Prefer:

```text
Current state
Current blocker
Current objective
Known important facts
```

over:

```text
August 1:
Did X

August 2:
Did Y

August 3:
Did Z
```

When a blocker is resolved, replace the old blocker with the new current state.

---

# 16. Current Status

```text
Repository:             ✅ Established
Python environment:     ✅ Established
SQLite database:        ✅ Created
Database tables:        ✅ Confirmed
Database validation:    ✅ Existing utility
Feature script:         ✅ Loads regular-season rows and writes features
Game feature CSV:       ✅ Populated and independently validated
Prediction model:         ✅ Leakage-safe rolling logistic baseline rebuilt
Model evaluation:         ✅ Chronological holdout with accuracy, log loss, and Brier score
Player impact model:    ⬜
Simulation engine:      ⬜
AI agent/tool layer:    ⬜
Live data:              ⬜
Website:                ⬜
```

## Diagnosed pipeline blocker

The earlier 0-row result was caused by an empty SQLite `team_statistics` table.
The source `data/raw/TeamStatistics.csv` contained 146,560 rows, but the
database table contained none, so the query in `src/build_features.py` had no
rows to load. The `gameType = 'Regular Season'` predicate was not the cause:
the reloaded table contains 130,014 matching rows.

The current database was reloaded from the confirmed raw CSV files. Running
`src/build_features.py` from the project root now reports:

```text
Loaded 130,014 team-game rows
Saved 129,836 rows
```

The feature script now applies the explicit model-ready policy. The generated
features are validated and covered by a focused test/check. The raw database
remains unchanged.

## Feature validation

`tests/test_feature_pipeline.py` independently reconstructs the regular-season
source rows and all rolling columns, then verifies that:

- the expected 129,836 rows are retained;
- output `(gameId, teamId)` keys match the source-derived rows;
- duplicate keys, null current metrics, and null rolling values are absent;
- percentage predictors are bounded fractions; and
- every rolling value matches the previous-game calculation within a
  floating-point tolerance.

The check passes when invoked directly with the project `.venv`. An independent
validation also confirms 130,014 regular-season source rows, zero duplicate source
keys, 129,836 output rows, zero duplicate output keys, complete current and rolling
predictors, and zero invalid percentage values. The environment does not
currently include `pytest`, so the test was executed by importing and calling its
test function directly; no dependency was added.

## Current status after feature validation

The feature pipeline is reproducible and its rolling-history behavior is verified.
The generated dataset covers 1947-01-10 through 2026-04-12, uses seasons labeled
by the calendar year in which they start, and contains 80 season labels from
1946 through 2025.

## Baseline prediction model

`src/train_baseline_model.py` converts the two team rows for each complete game
into one home-versus-away record. It uses only differences in the 11 rolling
features, so current-game box-score metrics are excluded from prediction. The
final 20% of games by date is held out chronologically.

The reproducible run produced 64,855 complete games: 51,884 training games and
12,971 test games, with the holdout beginning on 2014-11-29. Results are saved
to `models/baseline_metrics.json`, and the fitted pipeline is saved to
`models/baseline_logistic.pkl`.

| Model | Accuracy | Log loss | Brier score |
| --- | ---: | ---: | ---: |
| Training-period home-win rate | 0.56788 | 0.69179 | 0.24914 |
| Rolling-feature logistic model | 0.62971 | 0.64327 | 0.22575 |

The model is a validated first baseline, not evidence that the feature set is
optimal. The next evaluation should add a chronological Elo-style comparison
before attempting feature improvements.

## Historical dataset validation

The regular-season source contains 130,014 rows with dates from
1946-11-26 through 2026-04-12. Game/team keys are structurally consistent:
there are no duplicate `(gameId, teamId)` keys and every regular-season game has
exactly two team rows. `home` is binary (`0`, `1`) and `win` is binary (`0.0`,
`1.0`).

The live audit found three source rows with null required box-score metrics
(two teams in game `26200259` and the three-point percentage for one team in
game `28800195`). The generated CSV therefore contains 17 null metric cells;
the current pipeline retains those source rows.

The percentage fields are not consistently bounded fractions in the legacy
portion of the source: 25 regular-season rows have out-of-range values (4 field
goal percentages and 21 free-throw percentages), all dated from 1950 through
1964. These values contradict the made/attempted columns in the same rows; for
example, some records report more makes than attempts and field-goal attempts
of zero alongside positive makes. This is evidence of a malformed or changing
legacy encoding, not evidence supporting a safe divide-by-attempts repair.

No source values are normalized or imputed. The model-ready policy marks
out-of-range percentages as unusable and excludes rows with any unusable current
metric or incomplete rolling history from the processed output. This preserves
the raw database while preventing malformed values from entering model training.

The season-label rule and model-ready policy are implemented in
`src/build_features.py`, regenerated the feature CSV, and are covered by
`tests/test_feature_pipeline.py`. The focused regression passes with 129,836
rows, zero duplicate keys, complete current and rolling predictors, bounded
percentage values, and the expected rolling values. The output covers
1947-01-10 through 2026-04-12.

The existing `src/check_database.py` validator was also made Windows-console
safe by replacing its non-ASCII status marker with ASCII output. It validates all
seven database tables successfully.

The baseline model and evaluation are now implemented in
`src/train_baseline_model.py`. Model training uses only pregame rolling features,
not current game statistics. The focused baseline pairing test is in
`tests/test_baseline_model.py`.
