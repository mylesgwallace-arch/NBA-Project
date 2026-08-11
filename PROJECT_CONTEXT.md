# Project Context

## Inspection

Inspected on 2026-08-10 from branch `agents/nba-data-pipeline-continuation`.

- `PROJECT.md` and this file were absent before this inspection.
- The repository has two commits:
  - `386e1037` - initial `.gitattributes`.
  - `edea2989` - adds `readme.md`, `src/main.py`, and a Python virtual environment.
- `readme.md` contains only a placeholder sentence.
- `src/main.py` only prints `Sports AI is starting...`.
- There are no existing pipeline scripts, dependency manifests, database files, schema definitions, tests, or generated outputs.
- The virtual environment is Python 3.12.10.
- The working tree was clean before this change.

## What changed

- Added `PROJECT.md` to capture the verified bootstrap milestone and scope.
- Added this context file to preserve the repository inspection and decision point.

## What now works

- The existing Python entry point still runs as a startup smoke check:
  `python src/main.py`
- The project now has a durable record of its actual state instead of relying on inferred pipeline assumptions.

## Remaining issues

- No authoritative NBA data source has been selected.
- No initial dataset or persistence target has been defined.
- No database schema, dependencies, ingestion workflow, validation, or output artifacts exist.
- The virtual environment is currently tracked, which should be addressed as repository hygiene after the project requirements are established.

## Exact next step

Choose and document the authoritative NBA data source and the first dataset's persistence schema (including required fields, keys, and update semantics). Then implement one ingestion script against that documented contract; do not add a fetcher before this decision because the repository currently provides no source or schema requirements to validate it against.
