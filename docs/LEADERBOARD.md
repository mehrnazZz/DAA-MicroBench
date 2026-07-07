# Leaderboard Policy

This document defines the intended public leaderboard policy for DAA Microbench.

For the benchmark scope, planner contract, allowed information, and forbidden information, see [DESIGN_V1.md](DESIGN_V1.md). For planner implementation details, see [PLANNER_API.md](PLANNER_API.md).

## Status

The benchmark is currently pre-v1. Leaderboard fields and official suites may still change. Public results should include the benchmark commit hash and the explicit schema version in `result_schema.json`.

## Result Schema

Every run directory written by the CLI includes:

- `results.csv`: per-episode records.
- `summary.csv`: grouped leaderboard records.
- `result_schema.json`: schema id, schema version, and ordered field lists for both CSV files.

The current result schema version is `0.4.0`. Any change to the ordered CSV fields should update this version and the current-schema golden fixture.

## Official Dimensions

DAA Microbench should report results across five dimensions:

- Safety
- Mission progress
- Efficiency and smoothness
- Robustness to perception/communication degradation
- Compute and communication cost

No single scalar can fully describe DAA behavior. When a scalar ranking is needed, use the v0 score below and always publish the component metrics.

## Required Inputs for a Result

Every submitted result should include:

- benchmark commit hash
- `result_schema.json`
- method name and version
- full command used
- `results.csv`
- `summary.csv`
- hardware and Python version
- any changed config files
- whether the method uses V2V, intent, agent messages, local sensing, or learned weights
- for RL/learned-policy submissions, `learned_submission_bundle.json` or equivalent `rl_contract.json`, `rl_freeze_check.json`, `rl_smoke.json`, `rl_calibration.json`, training-data disclosure, and weight artifact/version

## Safety Metrics

Primary safety fields:

- `collision_episode_rate`
- `unique_collision_pairs_mean`
- `collision_pair_ticks_mean`
- `time_to_first_collision_mean`
- `min_sep_min_mean`
- `min_sep_p05_mean`

Interpretation:

- `collision_episode_rate` answers whether an episode failed at all.
- `unique_collision_pairs_mean` answers how many pairs were involved.
- `collision_pair_ticks_mean` captures duration/severity of overlap.
- `min_sep_*` captures near-collision margins even when no collision happens.

## Mission Metrics

Primary mission fields:

- `completion_rate_mean`
- `mean_time_to_goal_mean`
- `deadlock_time_pct_mean`

Completion should not be optimized by accepting collisions. Safety is the first gate.

## Observation Metrics

Primary observation fields:

- `obs_neighbors_mean`
- `obs_v2v_fraction_mean`
- `obs_sensor_fraction_mean`
- `obs_stale_fraction_mean`
- `obs_empty_fraction_mean`

These explain whether a result is driven by dense observation, stale V2V, sensor-only partial observability, or frequent empty neighborhoods.

## Compute Metrics

Primary compute fields:

- `planner_ms_mean`
- `planner_ms_p95`
- `planner_timeout_count_mean`
- `planner_error_count_mean`
- `planner_fallback_count_mean`

Report hardware and Python version. Timing columns should not be compared bit-for-bit across machines. Any nonzero timeout, error, or fallback count should be disclosed with the relevant trace/debug artifacts.

## v0 Scalar Score

Use this only as a convenience ranking. Publish components next to it.

```text
safety_penalty =
  1000 * collision_episode_rate
  + 50 * unique_collision_pairs_mean
  + 0.1 * collision_pair_ticks_mean
  + 10 * max(0, -min_sep_p05_mean)

mission_penalty =
  100 * (1 - completion_rate_mean)
  + 2 * deadlock_time_pct_mean
  + 0.01 * mean_time_to_goal_mean

compute_penalty =
  0.1 * planner_ms_p95

score_v0 = safety_penalty + mission_penalty + compute_penalty
```

Lower is better. A method with any collisions should rank below a collision-free method unless the collision-free method has near-zero completion and is clearly unusable.

## Result Categories

Results should be grouped, not blended:

- `official_alpha`: generated pre-v1 suite with planar and 3D scenarios
- `official_3d_stress`: generated volumetric/dense/vertical/noncooperative 3D stress suite
- `official_agentic_stress`: generated heterogeneous-priority and multi-intruder noncooperative 3D stress suite
- `official_experimental_baselines`: generated calibration lane for runnable experimental baselines, not a ranking category
- learned fixtures such as `learned_tiny`: useful for submission plumbing and adapter tests, not ranking anchors unless explicitly promoted later
- `primary`: official planar suite
- `three_d`: hand-written 3D development suite
- `perception_stress`: partial observation and fused-observation suite
- custom suites: must be labeled separately

Do not compare methods across different suites as if they share one ranking.

See [SCENARIO_SUITES.md](SCENARIO_SUITES.md) or `python -m microbench.cli list-suites` for the current suite registry.
Use `python -m microbench.cli list-suites --json` to inspect pre-v1 acceptance metadata for generated suites.
See [BASELINES.md](BASELINES.md) for canonical baseline roles, aliases, and limitations.
For learned-policy bundles, use `python -m microbench.cli review-learned-bundle --bundle <bundle> --json` to generate a machine-readable summary of the safety, mission, compute, communication, observation, and v0-score dimensions used during manual review.

## Reproducibility Rules

Submitted results must:

- use unmodified official scenario files unless explicitly marked custom
- include the generated `suite_manifest.yaml` for generated official suites
- pass `python -m microbench.cli validate-scenarios` for any submitted official or custom scenario files
- pass `python -m microbench.cli check-acceptance` for generated-suite acceptance metadata
- use official comm profiles unless explicitly marked custom
- use the same `N`, seeds, and comm matrix for all methods in a comparison
- include failed runs instead of silently dropping them
- disclose learned weights, external dependencies, and runtime services

## Review Policy

Maintainers may reject or mark a result as unofficial if:

- it uses privileged simulator state
- it changes shared benchmark parameters for one method
- it cannot be reproduced from the provided command/config
- it omits failed episodes
- it uses a modified benchmark without disclosure
