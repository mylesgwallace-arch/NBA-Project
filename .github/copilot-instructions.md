# NBA Project Agent Instructions

## Persistent Workflow

For every task in this repository:

1. Read `PROJECT.md` and `PROJECT_CONTEXT.md` before making changes.
2. Inspect the actual repository, existing scripts, database schema, data, tests, and current outputs before making assumptions.
3. Never assume the project's state. Verify it from the repository.
4. Determine the single most appropriate next step toward the current project milestone.
5. Explain briefly why that is the correct next step.
6. Implement that step when it is safe to do so.
7. Reuse and improve existing scripts rather than creating duplicate scripts.
8. Do not make unrelated changes.
9. Validate every change using the appropriate tests, scripts, database checks, or other available verification.
10. After completing work, update `PROJECT_CONTEXT.md` with:
    - What changed
    - What now works
    - Any remaining issues
    - The exact next step
11. Leave the repository in a clean, understandable state.
12. If something is unclear or potentially destructive, stop and explain the issue rather than guessing.

## Priority

The repository's actual state takes priority over assumptions, previous chat context, or generic recommendations.

`PROJECT.md` defines the project's goals and milestones.

`PROJECT_CONTEXT.md` defines the current verified state and should be kept current after every completed task.