# Golden Run Bundle

This folder freezes deterministic reference outputs for regression checks.

Note: the benchmark result schema has evolved since this bundle was first generated. Treat these files as a legacy reference snapshot until the golden suite is regenerated with the current schema.

## Contents
- `results.csv` and `summary.csv` from the legacy leaderboard schema.
- `events/` and `traces/` sample failure artifacts for recorder/replay checks.
- `current_schema/` contains a tiny fast golden smoke bundle for the current result schema.
- `acceptance/` contains path-independent acceptance report fixtures for generated suites.

## Source Run
- Copied from: `runs_leaderboard_smoke/`
- Methods: `baseline_goal`, `orca_expert` (legacy name; use `orca_heuristic` for new runs)
- Scenarios: `intersection`, `funnel`
- Comm profiles: `ideal_50hz`, `realistic_v2v_50hz`
- N: `10`
- Seeds: `0:2`

## Why this golden is useful
- Small enough to inspect and use as a legacy regression reference.
- Includes both a weak baseline and the legacy ORCA-like heuristic output.
- Exercises safety/success/compute leaderboard columns.
- Includes trace artifacts for recorder/replay regressions.

## Suggested regression workflow
1. Run the same matrix into a fresh out dir.
2. Compare `results.csv`/`summary.csv` against this folder.
3. Spot-check trace rendering with `microbench replay`.
