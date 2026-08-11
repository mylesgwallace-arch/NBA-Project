# Sports AI — Current Project Context

> **This file describes the current state of the repository.**
>
> `PROJECT.md` describes the long-term project specification.
>
> When deciding what to work on next, prioritize this file and the actual repository/database over the long-term roadmap.

---

# 1. Current Objective

The current objective is to build and validate the **historical NBA analytics foundation**.

The immediate objective was to restore the raw CSV to SQLite feature-engineering
pipeline after `src/build_features.py` loaded 0 team-game rows, establish a
leakage-safe baseline model, and evaluate historical player availability signals.
That quantitative foundation is complete; the current objective is to validate a
simple player/team impact model before exposing it as a user-facing capability.

That blocker is resolved. The database was rebuilt from the confirmed raw CSV files,
and the feature-building pipeline now produces nonzero output. Generated features and
the historical dataset have been independently validated and covered by focused
regression checks. The baseline prediction model has also been rebuilt and evaluated
with a chronological holdout, including an Elo-style strength comparison, a
leakage-safe rest-interval predictor, and player-level prior-production features.
The player-impact association benchmark has now been repeated across four
season-based holdouts.

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
team_statistics regular season (native label): 130,014
effective regular-season rows after Games.csv type fallback: 133,670
team_statistics_extended: 79,724
processed model-ready feature rows: 133,466
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

The script executes successfully. Because `TeamStatistics.csv` has null
`gameType` values for part of late 2021 and 2022, the query uses
`COALESCE(team_statistics.gameType, games.gameType)` after joining on `gameId`.
`Games.csv` supplies the missing game classification without changing raw data.
The latest verified run reports:

```text
Loaded 133,670 team-game rows
Saved 133,466 rows
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
rest_days
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
active_players_rolling_10
active_players_last_game
player_minutes_rolling_10
player_points_rolling_10
player_assists_rolling_10
player_rebounds_rolling_10
```

These columns are the intended starting point for game-level feature engineering. The
row count, duplicate handling, rolling-window values, and season definition have
passed focused validation.

---

# 9. Resolved Blocker and Remaining Work

## Resolved problem

The SQLite `team_statistics` table existed with the expected schema but contained 0
rows, even though `data/raw/TeamStatistics.csv` contained 146,560 rows. Reloading the
database with `src/load_data.py` restored the table. The native
`gameType = 'Regular Season'` filter returned 130,014 rows after the reload, but the
raw team-statistics file has 3,656 rows whose `gameType` is null. Joining those rows
to `games` shows that the game table classifies the regular-season subset correctly.

The feature pipeline now returns:

```text
Loaded 133,670 team-game rows
Saved 133,466 rows
```

## What is working

- Raw CSV ingestion via `src/load_data.py`.
- SQLite tables and expected lowercase/underscore table names.
- Regular-season team-game extraction in `src/build_features.py`.
- Chronological rolling features using only previous games.
- Leakage-safe `rest_days` intervals from each team's previous game date.
- Model-ready output at `data/processed/game_features.csv` with complete current
  and rolling predictors.
- Leakage-safe `active_players_rolling_10` prior-participation feature extraction.
- Leakage-safe `active_players_last_game` feature extraction from the prior team game.
- Leakage-safe prior-ten-team-game player-level summaries of minutes, points,
  assists, and rebounds.
- Basic table validation via `src/check_database.py`.
- Focused feature regression coverage in `tests/test_feature_pipeline.py`.
- Game-type fallback through the `games` table for null team-statistics labels.

## What is not complete

- The player features are historical production/participation proxies, not injury
  forecasts. A trustworthy historical pregame inactive list is not available.
- The player-impact targets are association diagnostics, not causal evidence, and
  remain gated from user-facing projection until stronger controls and independent
  prospective validation are available.

## Latest player-history evaluation

The baseline trainer compares four player-level prior-ten-team-game production
summaries against the prior rolling-plus-rest logistic model and saves the richer
model when it improves the same holdout. The regenerated output still contains
133,466 rows. On the unchanged 13,332-game chronological holdout, the rolling-plus-
rest baseline scored accuracy `0.62669`, log loss `0.64628`, and Brier score
`0.22691`. The player-history model scored accuracy `0.62691`, log loss `0.64599`,
and Brier score `0.22665`; configured chronological Elo remains better at log loss
`0.62616` and Brier score `0.21812`. The saved `baseline_logistic.pkl` now contains
the improved player-history model and its predictor list.

## Player impact prototype

`src/player_impact.py` provides an assumption-labeled addition/removal estimate
based on a player's prior ten regular-season appearances. It uses minutes-weighted
player net rating and prior average minutes, assumes those values transfer to a new
team context, and compares them with a configurable baseline net rating. That
baseline is a reference value of `0.0`, not a claim about replacement level.

The validator also evaluates an independent pregame target: observed current team
net rating, predicted first from prior team/opponent form and prior-game team
participation, then with the prior player-production signal added. The controls
include prior active-player count and prior total rotation minutes. This target is
not constructed by removing the player's current points or possessions. On the
chronological holdout, the pregame-control model scored MAE `11.79552` across
6,697 games and 136,276 player-games; adding the player signal scored `11.79408`.
A game-cluster bootstrap 95% interval for candidate-minus-control MAE was
`[-0.00200, -0.00098]`. The improvement is statistically separated from zero
under this benchmark, but is small and remains an association result rather than
causal evidence.

The validation target is now a possession-normalized leave-one-player-out scoring
target: current team net rating minus team scoring net rating after removing the
player's current points and possessions. Prior ten-game player production predicts
this target for 663,251 player-games with MAE `36.60191`, compared with a
fixed-zero MAE of `43.40858`, correlation `0.43903`, and an improvement over zero.
The chronological 20% holdout calibrates the signal only on earlier games and
scores 136,402 player-games from 6,703 games: calibrated MAE `30.86037` versus
fixed-zero MAE `36.06197`. It improves the fixed-zero baseline in aggregate and
in every available holdout season (2019-2025); the smallest seasonal sample is
19 player-games in 2021. This is a better-defined incremental target, but it
remains an association diagnostic rather than causal evidence. A reproducible
game-cluster bootstrap gives a 95% interval of `[-5.48228, -4.91403]` for
calibrated MAE minus fixed-zero MAE, so the aggregate holdout improvement is
separated from zero under this resampling procedure. This does not establish
causality or quantify uncertainty for a future roster change. The
addition/removal estimator is still not promoted as a reliable projection.

The independent pregame target has now been stress-tested across 10%, 20%, and
30% chronological holdouts. The player signal improves the pregame control MAE
slightly in each window: `12.22946` vs `12.23062`, `11.79408` vs `11.79552`,
and `11.48681` vs `11.48806`, respectively. Usage-stratified results are not
uniform: the low- and middle-prior-minute strata are slightly worse than the
control in all three windows, while the high-minute stratum is better in all
three. These results support robustness as an association diagnostic, but the
small, usage-dependent effect is not evidence for a causal player projection.
The persisted report is `models/player_impact_metrics.json`, and focused tests
cover the window and usage-stratum result shape.

The validator also evaluates the signal once per team-game, rather than weighting
team outcomes by the number of player rows. The control uses only prior team,
opponent, participation, possession, rest, and home-court predictors; the candidate
adds the mean prior player signal. Four untouched season-based holdouts all improve
the control:

```text
validation start | holdout team-games | control MAE | candidate MAE | 95% MAE difference
2022             | 9,694              | 11.30074    | 11.27184      | [-0.04380, -0.01434]
2023             | 7,234              | 11.58557    | 11.56045      | [-0.04391, -0.00723]
2024             | 4,906              | 11.58057    | 11.54684      | [-0.05375, -0.01189]
2025             | 2,460              | 11.70645    | 11.66426      | [-0.06866, -0.01418]
```

The results are persisted in `models/player_impact_metrics.json` under
`later_team_game_validation_by_season`. They remain association diagnostics, not
evidence that the player-impact estimator is causal.

The validator now also evaluates observed roster-change events using the
independent current-team-net-rating target. It selects first player appearances
after a historical team change, aggregates multiple changes in the same
team-game, and compares the player signal with the same pregame team,
opponent, participation, possession, rest, and home controls. The benchmark
contains 5,280 transition player-games and 3,506 evaluable transition events;
the chronological holdout contains 697 events across 614 games. The control
MAE is `12.84272` and the candidate MAE is `12.86279`. The clustered bootstrap
95% interval for candidate-minus-control MAE is `[-0.03576, 0.07496]`, which
includes zero. The roster-change benchmark therefore does not support adding
the player signal to a roster-change projection.

The roster-change result is persisted in `models/player_impact_metrics.json`
under `roster_change_validation`. Rest intervals are computed from each
team's complete game schedule, not only from games containing a roster-change
event. Focused player-impact tests and the full test suite pass.

## Exact next step

Keep the player-impact estimator gated and obtain prospective validation or a
stronger independently sourced roster-change outcome before exposing
addition/removal projections. The historical roster-change benchmark is now
complete but fails to improve its pregame control and does not establish
causal transfer to a new team context. Do not build the simulation or
user-facing projection layer until that validation gap is resolved.

The earlier investigation confirmed the following facts and should not be repeated as
assumptions:

1. The schema of `team_statistics`.
2. The number of rows in `team_statistics`.
3. The columns available in `team_statistics`.
4. The relevant `gameType` value: `Regular Season`.
5. That the table contains game-level team statistics.
6. That the current SQL filter does not exclude all rows after reload.
7. That the expected feature columns exist in the raw team-statistics data.
8. That `build_features.py` uses `gameDateTimeEst` for chronological features and
   joins `games` only to recover missing game-type labels.

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
Prediction model:         ✅ Leakage-safe rolling-plus-rest logistic baseline rebuilt
Model evaluation:         ✅ Chronological holdout with accuracy, log loss, and Brier score
Player impact model:    ⚠️ Historical team-game cutoffs improve slightly, but roster-change validation does not; causal/prospective validation still required
Simulation engine:      ⬜
AI agent/tool layer:    ⬜
Live data:              ⬜
Website:                ⬜
```

## Diagnosed pipeline blocker

The earlier 0-row result was caused by an empty SQLite `team_statistics` table.
The source `data/raw/TeamStatistics.csv` contained 146,560 rows, but the
database table contained none, so the query in `src/build_features.py` had no
rows to load. After reload, the source also exposed a separate coverage issue:
3,656 team-statistics rows have null `gameType` values. The feature query now
falls back to `games.gameType` for those rows.

The current database was reloaded from the confirmed raw CSV files. Running
`src/build_features.py` from the project root now reports:

```text
Loaded 133,670 team-game rows
Saved 133,466 rows
```

The feature script now applies the explicit model-ready policy. The generated
features are validated and covered by a focused test/check. The raw database
remains unchanged.

## Feature validation

`tests/test_feature_pipeline.py` independently reconstructs the regular-season
source rows and all rolling columns, then verifies that:

- the expected 133,466 rows are retained;
- output `(gameId, teamId)` keys match the source-derived rows;
- duplicate keys, null current metrics, and null rolling values are absent;
- percentage predictors are bounded fractions; and
- every rolling value matches the previous-game calculation within a
  floating-point tolerance.

The check passes when invoked directly with the project `.venv`. An independent
validation also confirms 133,670 effective regular-season source rows, zero duplicate
source keys, 133,466 output rows, zero duplicate output keys, complete current and
rolling predictors, and zero invalid percentage values. The environment does not
currently include `pytest`, so the test was executed by importing and calling its
test function directly; no dependency was added.

## Current status after feature validation

The feature pipeline is reproducible and its rolling-history behavior is verified.
The generated dataset covers 1947-01-10 through 2026-04-12, uses seasons labeled
by the calendar year in which they start, and contains 80 season labels from
1946 through 2025. Season label 2021 now contains 2,445 team rows (1,230 games),
rather than the previous two rows.

## Baseline prediction model

`src/train_baseline_model.py` converts the two team rows for each complete game
into one home-versus-away record. It uses differences in the 11 rolling features
and pregame `rest_days`, so current-game box-score metrics are excluded from
prediction. The final 20% of games by date is held out chronologically.

The reproducible run produced 66,658 complete games: 53,326 training games and
13,332 test games, with the holdout beginning on 2015-03-28. Results are saved
to `models/baseline_metrics.json`, and the fitted pipeline is saved to
`models/baseline_logistic.pkl`.

| Model | Accuracy | Log loss | Brier score |
| --- | ---: | ---: | ---: |
| Training-period home-win rate | 0.56481 | 0.69305 | 0.24977 |
| Rolling-plus-rest logistic model | 0.62669 | 0.64628 | 0.22691 |
| Player-history logistic model (saved artifact) | 0.62691 | 0.64599 | 0.22665 |
| Chronological Elo (K=20, home advantage=65) | 0.64971 | 0.62616 | 0.21812 |

The models are validated first baselines, not evidence that the feature set is
optimal. Elo remains the strongest evaluated model, while the player-history
logistic model is the saved artifact because it improves the rolling logistic
baseline. The Elo evaluator initializes teams at 1500,
uses only ratings available before each game, and updates ratings only after that
game's result. Its metrics use the same chronological holdout as the rolling
logistic model.

The Elo comparison is implemented in `src/train_baseline_model.py`, with fixed
parameters recorded in `models/baseline_metrics.json`. The focused baseline tests
also verify that an Elo rating update from one completed game affects the next
pregame probability without using the next game's result.

The trainer now records Elo metrics for each holdout season, rolling-logistic
metrics for the same seasons, and a four-point Elo sensitivity grid. On the
13,332-game holdout, the configured Elo (`K=20`, home advantage `65`) remains
best in aggregate (log loss `0.62616`) versus the player-history logistic model
(`0.64599`). The repaired 2021 comparison now contains 1,215 games and is
interpretable as a seasonal result, subject to the source's remaining
quality limitations. The tested Elo settings are recorded in
`models/baseline_metrics.json`.

## Historical dataset validation

The effective regular-season source contains 133,670 rows with dates from
1946-11-26 through 2026-04-12. Game/team keys are structurally consistent:
there are no duplicate `(gameId, teamId)` keys and every regular-season game has
exactly two team rows. `home` is binary (`0`, `1`) and `win` is binary (`0.0`,
`1.0`).

The live audit found three source rows with null required box-score metrics
(two teams in game `26200259` and the three-point percentage for one team in
game `28800195`). The generated CSV excludes rows with unusable current metrics or incomplete
rolling history; it contains no null current or rolling predictors.

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
`tests/test_feature_pipeline.py`. The focused regression passes with 133,466
rows, zero duplicate keys, complete current and rolling predictors, bounded
percentage values, and the expected rolling values. The output covers
1947-01-10 through 2026-04-12.

The existing `src/check_database.py` validator was also made Windows-console
safe by replacing its non-ASCII status marker with ASCII output. It validates all
seven database tables successfully.

The baseline model and evaluation are now implemented in
`src/train_baseline_model.py`. Model training uses only pregame rolling, rest, and
prior-player-participation and player-production features, not current game
statistics. The focused
baseline pairing test is in `tests/test_baseline_model.py`; the chronology check
for the player feature is in `tests/test_feature_pipeline.py`.

The current quantitative milestone is complete: rolling-history-plus-rest-plus-player
logistic and chronological Elo baselines are evaluated on the same holdout, including
season-level comparison and Elo parameter sensitivity. The 2021 coverage gap
was diagnosed and repaired in feature extraction by using the authoritative
`games` classification for null team-statistics labels. Rest intervals are calculated
from each team's prior game timestamp and verified against an independent source
reconstruction; chronological splits remain required. Player-level prior-production
summaries were evaluated without current-game or future information and improved the
logistic holdout metrics, so they are now in the saved model artifact. Raw CSV and
database contents were not modified.

## Player impact status

The first impact prototype is implemented in `src/player_impact.py` and covered by
`tests/test_player_impact.py`. The validator now calibrates the prior-production
signal using only games before the final 20% chronological holdout and reports
aggregate and season-level MAE against fixed zero. Focused regression tests pass,
the full historical validation completes, and
`models/player_impact_metrics.json` records a holdout improvement in aggregate and
every available season. The estimate remains descriptive rather than causal and
must not yet be wired into a user-facing tool. The holdout's 95% game-cluster
bootstrap interval for model-minus-baseline MAE is `[-5.48228, -4.91403]`;
this quantifies resampling uncertainty for the benchmark comparison only, not
uncertainty around a hypothetical player addition or removal.
The newer team-game evaluation uses one row per team-game and repeats the
untouched holdout from validation starts 2022, 2023, 2024, and 2025. It adds
prior team possessions, rest, and home-court controls; the candidate improves
control MAE at every cutoff. The 2024 candidate MAE is `11.54684` versus
`11.58057` for controls, with a clustered 95% difference interval of
`[-0.05375, -0.01189]`; the complete split report is persisted in
`models/player_impact_metrics.json`. This remains an association diagnostic and
does not justify causal roster projections.
