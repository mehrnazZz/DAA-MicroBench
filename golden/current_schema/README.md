# Current-Schema Golden Smoke Bundle

This folder freezes a tiny deterministic smoke reference for the current leaderboard CSV schema.

## Contents
- `results.csv`: per-episode outputs with explicit collision, communication, negotiation, and planner guardrail semantics.
- `summary.csv`: grouped leaderboard summary, including planner timeout/error/fallback count means.
- `result_schema.json`: explicit schema id/version and ordered CSV field lists.

## Check Or Regenerate
Check the checked-in fixture against a fresh deterministic run:

```bash
python -m microbench.cli golden-current-schema \
  --golden-dir golden/current_schema
```

Regenerate the checked-in fixture after an intentional schema or semantic metric change:

```bash
python -m microbench.cli golden-current-schema \
  --golden-dir golden/current_schema \
  --update
```

The helper runs the source episodes in a temporary directory named `daa_current_schema_m64` so the checked-in `run_id` remains stable, then copies `results.csv`, `summary.csv`, and `result_schema.json` into this folder.

## Review Policy
- Headers and `result_schema.json` must match the declared schema exactly.
- Semantic columns are compared exactly after sorting rows by method/scenario/comm/N/seed.
- Timing columns are only checked for finite nonnegative values because they are machine-dependent:
  - `results.csv`: `planner_ms_per_tick_per_agent_mean`, `planner_ms_per_tick_per_agent_p95`, `episode_runtime_s`
  - `summary.csv`: `planner_ms_mean`, `planner_ms_p95`

## Purpose
- Exercises the current result and summary schemas.
- Freezes the current explicit result schema version (`0.4.0`).
- Exercises both single-method and heterogeneous-agent method labels.
- Exercises zero-traffic communication and negotiation metric columns for schema stability.
- Stays fast enough to regenerate during normal development.
- Timing columns are useful for smoke checks but should not be compared bit-for-bit across machines.
