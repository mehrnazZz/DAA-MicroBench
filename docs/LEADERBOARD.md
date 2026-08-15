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

The current result schema version is `0.5.0`. Any change to the ordered CSV fields should update this version and the current-schema golden fixture.

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
- `goal_progress_fraction_mean`
- `final_goal_dist_mean_m_mean`
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

For built-in baselines, use:

```bash
python -m microbench.cli baseline-leaderboard \
  --out-dir runs_baseline_leaderboard \
  --suites all \
  --require-pass \
  --require-complete
```

This writes one per-suite `baseline_report.json` plus an aggregate `baseline_leaderboard.json`. The aggregate is a navigation and smoke-comparison artifact; official comparisons should still be read per suite because suite difficulty and purpose differ.

For optimizer-grade NMPC versus EGO-Swarm review, use the narrower wrapper:

```bash
python -m microbench.cli optimizer-suite-review \
  --out-dir runs_optimizer_suite_review \
  --max-runs 4 \
  --require-pass
```

It delegates runs to `baseline-leaderboard`, then writes `optimizer_suite_review.json` with method summaries, review findings, transient guardrail retry evidence, and Foxglove rerun/export commands for the most interesting cases. Capped optimizer reviews use balanced run selection by default, so checkpoints cover scenario/method groups more evenly than a prefix-only cap. Guardrail rows retry once by default only to distinguish transient local runtime spikes from persistent failures; use `--guardrail-retries 0` for a stricter audit. Use `--save-review-traces` to write full `trace_episode.jsonl` artifacts for those cases. Remove `--max-runs` and use `--suites official_alpha,official_3d_stress,official_agentic_stress --require-complete` before making publication-scale optimizer claims.

For fleet-size scaling studies, use the scale benchmark runner:

```bash
python -m microbench.cli scale-benchmark \
  --out-dir runs_scale_benchmark \
  --scenarios config/scenarios/stacked_swap_3d.yaml,config/scenarios/urban_conflict_3d.yaml,config/scenarios/urban_throughput_3d.yaml \
  --n 4,8,16,30 \
  --seeds 2 \
  --comm realistic_v2v_50hz \
  --duration-s 60 \
  --scale-spawn-profile dense \
  --planner-preset scale \
  --run-timeout-s 360
```

This writes raw `results.csv`, standard `summary.csv`, `scale_summary.csv`, `scale_benchmark_progress.json`, and `scale_benchmark.json`. Use `--scale-spawn-profile dense` only when a copied scenario needs a placement profile for large N; it widens four-way spawn lanes in the generated scale copy and leaves the source scenario unchanged. Use `--planner-preset scale` for compute-bounded high-N optimizer sweeps; omit it when auditing the fuller default planner settings. For NMPC/DMPC/BVC/RMADER-style optimizers, the scale preset uses bounded local traffic/constraint budgets plus command-level velocity/yield guarding. For RMADER, it also uses ego-trajectory broadphase filtering, true seed-budget capping, closest-dynamic-hull caps, and kinematic cached-plan validation so the high-N row measures nearby deconfliction pressure instead of distant hyperplane bookkeeping; for `ego_swarm_opt`, it uses receding cached-plan reuse between 5 Hz trajectory optimizations plus a command-level velocity/yield guard. The scale summary includes completion plus final-goal-distance and progress-fraction fields so unfinished high-N rows can be separated into “still far away” versus “near goal but not held yet.” Timeout and partial-timeout rows are first-class evidence, but they are not successful leaderboard rows.

For `urban_throughput_3d`, use `--duration-s 30` for quick high-N diagnostics and `--duration-s 60` or longer for deliberate evidence runs. This scenario is intentionally throughput-focused: a safe row with low progress is still useful evidence, but it should not be described as strong mission performance.

Build a high-volume leaderboard from completed scale summaries:

```bash
python -m microbench.cli high-volume-leaderboard \
  --scale-summary runs_scale_benchmark/scale_summary.csv \
  --out runs_scale_benchmark/high_volume_leaderboard.json
```

The report writes JSON and a sibling CSV. It ranks methods on four axes: safety, scenario-relative mission progress, runtime under a planner-latency budget, and robustness against hard timeouts/soft guardrails. The overall score defaults to safety 30%, progress 40%, runtime 20%, and robustness 10%. Treat it as a high-volume comparison artifact, not a replacement for per-suite acceptance checks.

For qualitative leaderboard review, generate a single Foxglove comparison MCAP from the same scenario:

```bash
python -m microbench.cli advanced-baseline-comparison \
  --out-dir runs_urban_throughput_comparison \
  --scenario config/scenarios/urban_throughput_3d.yaml \
  --methods dmpc_best_response,bvc_tube_dmpc,dynamic_tube_dmpc,mpc_nonlinear,ego_swarm_opt,rmader \
  --n 8 \
  --seed 2 \
  --comm realistic_v2v_50hz \
  --duration-s 20 \
  --export-foxglove-mcap
```

Optionally publish the same run to W&B as dashboard tables:

```bash
python -m microbench.cli baseline-leaderboard \
  --out-dir runs_baseline_leaderboard \
  --suites all \
  --require-pass \
  --require-complete \
  --wandb \
  --wandb-project daa-microbench
```

W&B is a visualization and sharing layer, not the canonical record. The command logs aggregate ranking, suite status, per-suite method summaries, and component rows as W&B Tables, and uploads the local leaderboard JSON/CSV artifacts when `--wandb-upload-results` is enabled. Official submissions should still include the local `baseline_leaderboard.json`, per-suite `baseline_report.json`, `results.csv`, `summary.csv`, `result_schema.json`, and suite manifests.

For long baseline development jobs, `baseline-leaderboard` can checkpoint progress:

```bash
python -m microbench.cli baseline-leaderboard \
  --out-dir runs_baseline_leaderboard \
  --suites all \
  --methods reciprocal_velocity_obstacle \
  --max-wall-time-s 1800 \
  --run-timeout-s 120

python -m microbench.cli baseline-leaderboard \
  --out-dir runs_baseline_leaderboard \
  --suites all \
  --methods reciprocal_velocity_obstacle \
  --resume
```

Checkpointed runs write per-suite `leaderboard_progress.json` files. Official submissions should not be partial: require `complete: true`, `selected_complete: true`, `timeout_run_count: 0`, and no suite-level `stopped_by_wall_time`.
For general `baseline-leaderboard` development caps, use `--max-runs-strategy balanced` when the cap should cover multiple scenario/method groups instead of the default prefix of the planned matrix.

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
