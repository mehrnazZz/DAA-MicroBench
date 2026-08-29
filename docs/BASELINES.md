# Baseline Methods

DAA Microbench ships baselines for calibration, debugging, and reference comparison. They are not all leaderboard anchors. Use this page with:

```bash
python -m microbench.cli list-methods
python -m microbench.cli list-methods --json --include-aliases
python -m microbench.cli baseline-audit
python -m microbench.cli baseline-audit --require-public-alpha-ready --json
python -m microbench.cli baseline-smoke --out-dir runs_baseline_smoke --require-pass
python -m microbench.cli baseline-promotion --out-dir runs_baseline_promotion --require-calibrated
python -m microbench.cli baseline-evidence --out-dir runs_baseline_evidence --require-pass
python -m microbench.cli advanced-baseline-comparison --out-dir runs_advanced_baseline_comparison --require-pass
python -m microbench.cli baseline-validation-matrix --out-dir runs_baseline_validation_matrix --plan-only
python -m microbench.cli baseline-review --out-dir runs_baseline_review --duration-s 20
python -m microbench.cli baseline-leaderboard --out-dir runs_baseline_leaderboard --suites all --require-pass --require-complete
python -m microbench.cli external-reference-bundle --method-family rmader --out-dir runs_external_references/rmader_official_bundle --scenarios urban_conflict_3d,urban_throughput_3d,stacked_swap_3d --n 4,8 --seeds 2 --comm realistic_v2v_50hz --runner-type ros
python -m microbench.cli external-reference-bundle --method-family ego_swarm --out-dir runs_external_references/ego_swarm_official_bundle --scenarios urban_conflict_3d,urban_throughput_3d,stacked_swap_3d --n 4,8 --seeds 2 --comm realistic_v2v_50hz --runner-type ros
python -m microbench.cli validate-external-reference --manifest examples/external_reference_rmader_manifest.yaml --json
python -m microbench.cli validate-external-reference --manifest examples/external_reference_ego_swarm_manifest.yaml --json
```

The public-alpha baseline gate is intentionally stricter than "the code imports":

- required public-alpha reference baselines: `orca_heuristic`, `orca_with_staleness`, `priority_yield`, `negotiation_yield`
- experimental but runnable baselines: `cbf_qp`, `mpc_local`, `mpc_nonlinear`, `dmpc_best_response`, `bvc_tube_dmpc`, `dynamic_tube_dmpc`, `centralized_oracle`, `centralized_mpc_oracle`, `rmader`, `ego_swarm`, `ego_swarm_opt`, `velocity_obstacle`, `reciprocal_velocity_obstacle`, `learned_tiny`, `learned_mlp`
- illustrative or template methods: `baseline_goal`, `intent_dummy`, `template`

Run `baseline-audit --require-public-alpha-ready`, `baseline-smoke --require-pass`, `baseline-promotion --require-calibrated`, and the per-family `baseline-validation-matrix` before inviting external baseline comparisons. Stable v1 still requires promotion work; `baseline-audit --require-stable-v1-ready` and `baseline-promotion --require-stable-v1-ready` are expected to fail while experimental baselines remain experimental.

Every baseline also carries machine-readable fidelity/provenance metadata. See [BASELINE_FIDELITY.md](BASELINE_FIDELITY.md) for the full matrix and the external-reference manifest workflow. In short: `dynamic_tube_dmpc` and `rmader` are clean-room faithful reimplementations, EGO-Swarm-style methods are clean-room inspired implementations, and official dependency-heavy stacks should be compared through validated external manifests rather than vendored into this package.

## Current Methods

| Method | Role | Uses | Dimensions | Intended use |
|---|---|---|---|---|
| `baseline_goal` | illustrative baseline | ego state, goal | 2D, 3D | Lower bound that shows how hard a scenario is without avoidance. |
| `orca_heuristic` | reference baseline | local neighbor tracks, V2V/sensor/fused observations, obstacles | 2D, 3D | Main ORCA-like geometric comparison baseline. |
| `orca_with_staleness` | reference baseline | same as `orca_heuristic`, with stronger stale-track inflation | 2D, 3D | Degraded communication or stale sensor-track comparison baseline. |
| `cbf_qp` | experimental baseline | local neighbor tracks, V2V/sensor/fused observations, obstacles | 2D, 3D | CBF safety-filter baseline with deterministic projection, optional SciPy solver mode, obstacle barriers, and stale-track inflation. |
| `mpc_local` | experimental baseline | local neighbor tracks, V2V/sensor/fused observations, obstacles | 2D, 3D | Deterministic short-horizon predictive sampling baseline with candidate-risk diagnostics and stale-track inflation. |
| `mpc_nonlinear` | experimental baseline | local neighbor tracks, intent trajectories, V2V/sensor/fused observations, obstacles | 2D, 3D | Clean-room nonlinear MPC trajectory-optimization baseline over bounded acceleration controls. |
| `dmpc_best_response` | experimental baseline | local neighbor tracks, intent trajectories, agent-message plan broadcasts, V2V/sensor/fused observations, obstacles | 2D, 3D | Distributed-MPC-style best-response baseline with coupled intent constraints and stale/missing plan fallback. |
| `bvc_tube_dmpc` | experimental baseline | local neighbor tracks, intent trajectories, agent-message plan broadcasts, V2V/sensor/fused observations, obstacles | 2D, 3D | Tube-based DMPC baseline with hard buffered Voronoi-cell halfspace tubes and obstacle halfspace constraints. |
| `dynamic_tube_dmpc` | experimental baseline | local neighbor tracks, intent trajectories, agent-message plan broadcasts, V2V/sensor/fused observations, obstacles | 2D, 3D | Paper-specific Dai/Liao/Chen dynamic tube-DMPC baseline with condensed acceleration QP, elastic tube reconstruction, risk-triggered collision constraints, and local tube halfspaces. |
| `centralized_oracle` | nondeployable upper bound | privileged global truth for all agents and all obstacles | 2D, 3D | Centralized joint controller for estimating the performance gap caused by decentralization, limited sensing, V2V loss, and stale tracks. Not valid as a local DAA policy. |
| `centralized_mpc_oracle` | nondeployable upper bound | privileged global truth, all obstacles, and world bounds | 2D, 3D | Stronger route-aware centralized oracle with coarse global obstacle routing and joint candidate-MPC refinement. Not valid as a local DAA policy. |
| `rmader` | experimental baseline | local neighbor tracks, intent trajectories, agent-message plan broadcasts, V2V/sensor/fused observations, obstacles | 2D, 3D | Clean-room RMADER/MADER-style baseline with cubic B-spline plans, MINVO interval polyhedra, hard separating hyperplanes, and delay-check publication. |
| `ego_swarm` | experimental baseline | local neighbor tracks, intent trajectories, V2V/sensor/fused observations, obstacles | 2D, 3D | Clean-room EGO-Swarm-inspired receding-horizon trajectory-sharing baseline. |
| `ego_swarm_opt` | experimental baseline | local neighbor tracks, intent trajectories, V2V/sensor/fused observations, obstacles | 2D, 3D | Clean-room EGO-Swarm-style optimized control-point trajectory-sharing baseline. |
| `velocity_obstacle` | experimental baseline | local neighbor tracks, V2V/sensor/fused observations, obstacles | 2D, 3D | Deterministic finite-horizon velocity-obstacle cone sampler with candidate-risk diagnostics. |
| `reciprocal_velocity_obstacle` | experimental baseline | local neighbor tracks, V2V/sensor/fused observations, obstacles | 2D, 3D | Hybrid reciprocal/HRVO-style cone sampler with responsibility and apex-shift diagnostics. |
| `learned_tiny` | experimental learned baseline | frozen JSON weights, goal, local neighbor tracks, V2V/sensor/fused observations | 2D, 3D | Tiny learned-model fixture for packaging, disclosure, adapter, and benchmark-result plumbing. |
| `learned_mlp` | experimental learned baseline | frozen JSON MLP weights, goal, local neighbor tracks, V2V/sensor/fused observations | 2D, 3D | Nonlinear learned-model fixture for policy-interface, disclosure, adapter, and benchmark-result plumbing. |
| `learned_policy_spec` | learned submission bridge | trusted JSON/YAML policy spec, RL observation/action contract, local neighbor tracks, V2V/sensor/fused observations | 2D, 3D | Externally configured bridge for evaluating learned policies as standard planner CSV rows; not a reference baseline. |
| `priority_yield` | agentic reference baseline | local tracks, priority, agent messages | 2D, 3D | Simple decentralized right-of-way behavior. |
| `negotiation_yield` | pre-v1 agentic reference | local tracks, proposal/ACK messages, priority, local separation | 2D, 3D | Structured negotiation plumbing and decentralized agentic comparison. |
| `intent_dummy` | illustrative/plumbing baseline | goal, intent-style messages | 2D, 3D | Message and trace plumbing checks, not scoring. |
| `template` | developer template | ego state, goal | 2D, 3D | Minimal example for writing a planner plugin. |

Compatibility aliases:

- `orca_expert` resolves to `orca_heuristic`; use `orca_heuristic` for new runs.
- `template_planner` resolves to `template`.

## Reference Versus Illustrative

Reference baselines are intended to appear in comparison tables:

- `orca_heuristic`
- `orca_with_staleness`
- `priority_yield`
- `negotiation_yield` for structured proposal/ACK agentic comparisons

Experimental baselines are runnable but not leaderboard anchors yet:

- `cbf_qp`
- `mpc_local`
- `mpc_nonlinear`
- `dmpc_best_response`
- `bvc_tube_dmpc`
- `dynamic_tube_dmpc`
- `centralized_oracle` as a nondeployable upper bound, never as a local-policy competitor
- `centralized_mpc_oracle` as a stronger nondeployable route/MPC upper bound, never as a local-policy competitor
- `rmader`
- `ego_swarm`
- `ego_swarm_opt`
- `velocity_obstacle`
- `reciprocal_velocity_obstacle`
- `learned_tiny`
- `learned_mlp`

Illustrative baselines are useful for sanity checks and tutorials but should not be treated as serious DAA competitors:

- `baseline_goal`
- `intent_dummy`
- `template`

`orca_heuristic` and `orca_with_staleness` are ORCA-like heuristics, not formal ORCA proofs and not expert oracles. Dataset labels generated from them are heuristic labels.

## Recommended Sets

Fast smoke:

```bash
python -m microbench.cli canonical-sweep \
  --suite official_smoke_generated \
  --out-dir runs_smoke
```

Baseline behavior gate:

```bash
python -m microbench.cli baseline-smoke \
  --out-dir runs_baseline_smoke \
  --require-pass
```

This runs every non-template built-in baseline except contract-only heavy optimizer probes on one planar and one 3D generated smoke scenario, checks finite key metrics, planner errors, public-alpha guardrails, 2D/3D coverage, agent-message signals for `priority_yield`, proposal/ACK signals for `negotiation_yield`, privileged joint-path signals for `centralized_oracle` and `centralized_mpc_oracle`, and public debug/intent output contracts for `cbf_qp`, `mpc_local`, `mpc_nonlinear`, `dmpc_best_response`, `bvc_tube_dmpc`, `dynamic_tube_dmpc`, `rmader`, `ego_swarm`, `ego_swarm_opt`, `learned_tiny`, `learned_mlp`, and `intent_dummy`. `bvc_tube_dmpc`, `dynamic_tube_dmpc`, and `rmader` are contract-only in this smoke gate because per-tick hard-tube, condensed-QP, and MINVO/hyperplane solves belong in optimizer evidence and leaderboard runs. Experimental `cbf_qp`, `mpc_local`, `mpc_nonlinear`, `dmpc_best_response`, `bvc_tube_dmpc`, `dynamic_tube_dmpc`, and `ego_swarm_opt` soft timeout/fallback counts are reported but do not block public-alpha smoke by themselves; any such counts still block stable-v1 promotion.

Experimental promotion calibration:

```bash
python -m microbench.cli baseline-promotion \
  --out-dir runs_baseline_promotion \
  --require-calibrated
```

This produces `baseline_promotion.json`. It should report `public_alpha_calibrated=true` and `stable_v1_ready=false` for `cbf_qp`, `mpc_local`, and `negotiation_yield` during public alpha. The report records smoke metrics, generated experimental-suite acceptance for CBF/MPC, compact `official_promotion_calibration` 3D/degraded acceptance for all promotion candidates, method-specific signal contracts, and stable-v1 blockers such as non-stable metadata status or non-reference roles.

Geometric comparison under degraded communication:

```bash
python -m microbench.cli canonical-sweep \
  --suite official_3d_stress \
  --methods orca_heuristic,orca_with_staleness \
  --out-dir runs_orca_degraded
```

Agentic comparison:

```bash
python -m microbench.cli canonical-sweep \
  --suite official_agentic_stress \
  --methods priority_yield,negotiation_yield,orca_with_staleness \
  --out-dir runs_agentic_baselines
```

CBF safety-filter smoke:

```bash
python -m microbench.cli run \
  --scenario config/scenarios/stacked_swap_3d.yaml \
  --method cbf_qp \
  --n 2 \
  --seed 0 \
  --comm ideal_50hz \
  --out-dir runs_cbf_qp_smoke
```

MPC-local smoke:

```bash
python -m microbench.cli run \
  --scenario config/scenarios/stacked_swap_3d.yaml \
  --method mpc_local \
  --n 2 \
  --seed 0 \
  --comm ideal_50hz \
  --out-dir runs_mpc_local_smoke
```

Velocity-obstacle smoke:

```bash
python -m microbench.cli run \
  --scenario config/scenarios/stacked_swap_3d.yaml \
  --method velocity_obstacle \
  --n 4 \
  --seed 0 \
  --comm ideal_50hz \
  --out-dir runs_velocity_obstacle_smoke
```

Reciprocal velocity-obstacle smoke:

```bash
python -m microbench.cli run \
  --scenario config/scenarios/stacked_swap_3d.yaml \
  --method reciprocal_velocity_obstacle \
  --n 4 \
  --seed 0 \
  --comm ideal_50hz \
  --out-dir runs_reciprocal_velocity_obstacle_smoke
```

Learned-model baseline smoke:

```bash
python -m microbench.cli run \
  --scenario config/scenarios/stacked_swap_3d.yaml \
  --method learned_tiny \
  --n 4 \
  --seed 0 \
  --comm ideal_50hz \
  --out-dir runs_learned_tiny_smoke
```

MLP learned-model baseline smoke:

```bash
python -m microbench.cli run \
  --scenario config/scenarios/stacked_swap_3d.yaml \
  --method learned_mlp \
  --n 4 \
  --seed 0 \
  --comm ideal_50hz \
  --out-dir runs_learned_mlp_smoke
```

Experimental baseline calibration:

```bash
python -m microbench.cli canonical-sweep \
  --suite official_experimental_baselines \
  --out-dir runs_experimental_baselines
```

Build a compact comparison report from any run:

```bash
python -m microbench.cli baseline-report \
  --summary runs_experimental_baselines/summary.csv \
  --results runs_experimental_baselines/results.csv \
  --suite official_experimental_baselines \
  --out runs_experimental_baselines/baseline_report.json
```

The checked-in example lives at `golden/baseline_comparison/report.json`.

Run a compact shared 3D comparison for advanced local-avoidance baselines:

```bash
python -m microbench.cli advanced-baseline-comparison \
  --out-dir runs_advanced_baseline_comparison \
  --require-pass
```

This runs `orca_heuristic`, `orca_with_staleness`, `cbf_qp`, `mpc_local`, `mpc_nonlinear`, `dmpc_best_response`, `bvc_tube_dmpc`, `dynamic_tube_dmpc`, `rmader`, `ego_swarm`, `ego_swarm_opt`, `velocity_obstacle`, and `reciprocal_velocity_obstacle` on the same `urban_conflict_3d` scenario, with the same seed, agent count, duration override, and communication profile. It writes `advanced_baseline_comparison.json`, `baseline_report.json`, `results.csv`, `summary.csv`, and a copied scenario file under `_comparison_scenario/`. Use it as a quick apples-to-apples advanced-baseline artifact before spending time on the full official leaderboard.

For a panelized Foxglove comparison on the denser urban-throughput map:

```bash
python -m microbench.cli advanced-baseline-comparison \
  --out-dir runs_urban_throughput_comparison \
  --scenario config/scenarios/urban_throughput_3d.yaml \
  --methods dmpc_best_response,bvc_tube_dmpc,dynamic_tube_dmpc,mpc_nonlinear,ego_swarm_opt,rmader \
  --n 8 \
  --seed 2 \
  --comm realistic_v2v_50hz \
  --duration-s 20 \
  --planner-preset scale \
  --export-foxglove-mcap \
  --mcap-trail-frames 1000 \
  --mcap-max-sensing-links 80
```

This writes one `baseline_comparison.mcap` with per-method topics under `/daa/comparison/<method>/...`, suitable for a 2x3 Foxglove dashboard. It also writes `comparison_manifest.json` with the scenario, methods, N, seed, duration, communication profile, planner preset, git commit, guardrail summary, and MCAP path. Use `--planner-preset scale` for visual demos meant to match scale-tuned optimizer behavior; omit it when auditing the fuller default planner settings.

Build an all-official-suite baseline leaderboard:

```bash
python -m microbench.cli baseline-leaderboard \
  --out-dir runs_baseline_leaderboard \
  --suites all \
  --require-pass \
  --require-complete
```

This materializes every generated official suite, runs the serious built-in baselines over each suite's default scenario, N, seed, and communication matrix, writes per-suite `results.csv`, `summary.csv`, `baseline_report.json`, `acceptance.json`, and writes an aggregate `baseline_leaderboard.json`. Use this for serious baseline claims. For quick local plumbing checks, cap each suite:

Add `--wandb --wandb-project daa-microbench` to mirror the generated leaderboard as W&B Tables and a versioned artifact. Treat W&B as a public dashboard/export layer; the generated local JSON/CSV files remain the official benchmark evidence.

```bash
python -m microbench.cli baseline-leaderboard \
  --out-dir runs_baseline_leaderboard_smoke \
  --suites official_smoke_generated \
  --methods baseline_goal,velocity_obstacle \
  --n 4 \
  --seeds 0 \
  --comm ideal_50hz \
  --max-runs 2 \
  --require-pass
```

Run an optimizer-grade suite review for `mpc_nonlinear` versus `ego_swarm_opt`:

```bash
python -m microbench.cli optimizer-suite-review \
  --out-dir runs_optimizer_suite_review \
  --max-runs 4 \
  --require-pass
```

This delegates the actual runs to `baseline-leaderboard`, writes the normal leaderboard artifacts, and adds `optimizer_suite_review.json` with method summaries, guardrail/collision/incomplete-row findings, transient guardrail retry evidence, and Foxglove rerun/export commands for the most interesting review cases. Capped optimizer reviews use `--max-runs-strategy balanced` by default so checkpoint rows spread across scenario/method groups instead of only taking the first suite entries. Guardrail rows are retried once by default to separate transient local runtime spikes from persistent baseline failures; use `--guardrail-retries 0` for a stricter no-retry audit. To materialize full traces for those review cases, add `--save-review-traces`. For publication-scale optimizer claims, use the full generated stress suites without a run cap:

```bash
python -m microbench.cli optimizer-suite-review \
  --out-dir runs_optimizer_suite_review_full \
  --suites official_alpha,official_3d_stress,official_agentic_stress \
  --resume \
  --run-timeout-s 180 \
  --require-pass \
  --require-complete
```

Run a fleet-size scaling ladder when the question is how methods behave as N grows:

```bash
python -m microbench.cli scale-benchmark \
  --out-dir runs_scale_benchmark \
  --scenarios config/scenarios/stacked_swap_3d.yaml,config/scenarios/urban_conflict_3d.yaml,config/scenarios/urban_throughput_3d.yaml \
  --methods mpc_nonlinear,dmpc_best_response,bvc_tube_dmpc,dynamic_tube_dmpc,ego_swarm_opt,rmader \
  --n 4,8,16,30 \
  --seeds 2 \
  --comm realistic_v2v_50hz \
  --duration-s 60 \
  --scale-spawn-profile dense \
  --planner-preset scale \
  --run-timeout-s 360
```

This writes the standard raw result files plus `scale_summary.csv` and `scale_benchmark.json`, grouped by scenario, method, communication profile, and N. Use it to report maximum completed N, timeout pressure, guardrail pressure, collision rate, completion rate, final goal distance, progress fraction, and planner p95 latency. `--planner-preset scale` bounds optimizer effort for high-N studies while preserving the normal full/default planner settings for standard runs. Treat timeout rows and partial-timeout rows as review blockers, not success cases.

For `urban_throughput_3d`, `--duration-s 30` is a practical high-N diagnostic. Use `--duration-s 60` or longer as a deliberate scale-evidence run when runtime is acceptable; those rows are meant to measure mission progress under dense merge pressure, not just whether the planners avoid collision.

Package the default high-volume evidence matrix in one reproducible output folder:

```bash
python -m microbench.cli high-volume-evidence \
  --out-dir runs_high_volume_evidence \
  --require-pass
```

This wraps `scale-benchmark`, then builds `high_volume_leaderboard.json` and a sibling CSV from the generated `scale_summary.csv`. The default matrix is N=30 on the three core 3D scale scenarios with dense spawning, realistic V2V, 30-second episodes, and the scale planner preset. Use `--plan-only` for a dry run and `--resume` to continue partial evidence.

After one or more scale runs, build the high-volume leaderboard artifact:

```bash
python -m microbench.cli high-volume-leaderboard \
  --scale-summary runs_scale_benchmark/scale_summary.csv \
  --out runs_scale_benchmark/high_volume_leaderboard.json
```

The leaderboard separates safety, mission progress, runtime/scale, and robustness scores. This is the preferred way to compare high-volume optimizer baselines because a single hidden score would otherwise blur “safe but slow,” “fast but stalled,” and “progressive but expensive” behaviors.

For `mpc_nonlinear`, `dmpc_best_response`, and `bvc_tube_dmpc`, the scale preset keeps optimizer effort bounded while using dense-fleet local traffic budgets, stronger safety margins, and command-level velocity/yield guarding. This keeps the high-N row focused on deconfliction behavior instead of turning every large-fleet run into a full optimizer-throughput test.

For `rmader`, the scale preset also enables ego-trajectory broadphase filtering for dynamic MINVO hulls, keeps only the closest moving hulls per interval, stops generating topology seeds once the configured seed budget is full, uses kinematic cached-plan validation between replans, and replans slightly less often. This keeps the high-N study focused on nearby deconfliction pressure instead of spending most of the run solving hard hyperplanes against distant traffic.

For `ego_swarm_opt`, the scale preset uses receding cached-plan reuse between 5 Hz trajectory optimizations, a larger dense-fleet local traffic budget, and a command-level velocity/yield guard. This matches a more realistic trajectory-planner cadence than re-solving every simulator tick and keeps high-N rows from becoming pure optimizer-throughput tests.

Long 3D stress runs can be checkpointed. Use `--max-wall-time-s` to stop launching new episodes after a global wall-clock budget, `--resume` to continue from existing per-suite `results.csv` rows, and `--run-timeout-s` to write a failed timeout row instead of letting one episode monopolize the job:

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
  --resume \
  --require-pass
```

Plain `baseline-leaderboard --max-runs` keeps its historical prefix cap by default. Add `--max-runs-strategy balanced` when you want a development checkpoint to spread across scenario/method groups.

Each suite also writes `leaderboard_progress.json`. For publication-grade claims, the aggregate report should have `ok: true`, `complete: true`, `timeout_run_count: 0`, and no `truncated_by_max_runs` or `stopped_by_wall_time` suite entries.

## Stale-Aware ORCA Preset

`orca_with_staleness` uses the same planner implementation as `orca_heuristic`, but its default config is more conservative when neighbor tracks are old:

- larger `stale_age_cap_s`
- larger `stale_inflation_gain`
- larger `responsibility_age_gain`
- slightly larger closing-speed and sidestep buffers

This is useful for comparisons where `obs_stale_fraction_mean`, `obs_sensor_track_stale_fraction_mean`, or communication drop/delay metrics are high. It may reduce collisions at the cost of slower completion or extra path deviation.

## Negotiation-Yield Baseline

`negotiation_yield` is the pre-v1 structured agentic reference for proposal/ACK plumbing. Higher-priority agents send `NEGOTIATION_PROPOSAL` messages, lower-priority or directly threatened agents ACK and slow, and the planner keeps per-agent memory for active yield commitments and already-ACKed proposal correlations.

It also includes a deterministic local separation component for close 3D conflicts. This is intentionally modest: the baseline still demonstrates decentralized negotiation rather than claiming to be an optimized DAA controller, but long-horizon review checks that proposal/ACK behavior does not rely on passive slowing alone in volumetric and heterogeneous-priority scenarios.

Useful diagnostics:

- `comm_negotiation_proposals_mean`
- `comm_negotiation_acks_mean`
- `comm_agent_msg_delivery_fraction_mean`
- `avoidance_active` / `avoidance_weight` in planner debug traces

## CBF-QP Safety Filter

`cbf_qp` is currently an experimental CBF-style safety-filter baseline. The default `solver: projection` mode is quiet and dependency-free. Optional `solver: auto` or `solver: scipy` modes use SciPy SLSQP when available, then fall back to deterministic halfspace projection. It constructs pairwise and obstacle barrier halfspaces, inflates stale tracks to reflect observation uncertainty, clamps speed, and uses a deterministic away-from-risk fallback if constraints remain violated.

Use it for development and API comparison, not as a mature CBF baseline. It is useful because it establishes:

- the public method name and config block
- 2D/3D command shape
- neighbor and obstacle barrier semantics
- bounded fallback behavior
- debug fields in `PlannerOutput.debug_info`
- solver status reporting via `cbf_solver`, `cbf_solver_requested`, and `cbf_solver_status`
- stale-track barrier inflation via `cbf_uncertainty_inflation_max_m`
- constraint accounting via `cbf_neighbor_constraints`, `cbf_obstacle_constraints`, and `cbf_active_constraints`

Requirements before promoting it to a reference baseline:

- collision-free or clearly bounded-collision behavior on calibrated head-on, 3D swap, and obstacle cases
- completion bands on at least one generated 3D suite without relying on privileged state
- broad but explicit compute p95 bands on smoke and 3D stress rows
- zero planner timeout/error/fallback counts in smoke and experimental baseline runs
- stronger solver-backed validation, with documented deterministic projection fallback
- broader tests for infeasible constraints, solver failure, noisy observations, and 2D/3D obstacle barriers

Targeted evidence gate:

```bash
python -m microbench.cli baseline-evidence \
  --out-dir runs_baseline_evidence \
  --require-pass
```

To also generate compact Foxglove-ready trace JSONL artifacts for the optimizer-grade pair:

```bash
python -m microbench.cli baseline-evidence \
  --out-dir runs_optimizer_evidence \
  --save-optimizer-traces \
  --require-pass
```

For `cbf_qp`, this records feasible projection behavior, forced fallback behavior with residual violation reporting, stale-track barrier inflation, and optional solver-path status. Passing it supports continued public-alpha use, but the report intentionally recommends keeping CBF experimental until solver backends and infeasible-constraint behavior are validated beyond these targeted cases.

## MPC-Local Predictive Baseline

`mpc_local` is currently an experimental local predictive baseline. It samples one-step-reachable velocity commands, rolls them forward over a short horizon, and scores goal tracking, progress, smoothness, predicted agent-agent clearance, static obstacle clearance, approach-to-conflict costs, and stale-track risk inflation.

It is intentionally dependency-free and deterministic. Its command is bounded by `a_max * dt` from the current velocity, so the dynamics layer should not need to rescue it through acceleration saturation during normal operation.

Useful debug fields include:

- `mpc_candidates`
- `mpc_horizon_steps`
- `mpc_best_cost`
- `mpc_min_pred_clearance_m`
- `mpc_best_clearance_improvement_m`
- `mpc_current_min_pred_clearance_m`
- `mpc_goal_step_min_pred_clearance_m`
- `mpc_safe_candidate_count`
- `mpc_pred_collision_candidate_count`
- `mpc_collision_penalty`
- `mpc_obstacle_penalty`
- `mpc_approach_penalty`
- `mpc_stale_inflation_max_m`
- `mpc_accel_delta_norm`

Requirements before promoting it to a reference baseline:

- collision-free or clearly bounded-collision behavior on calibrated head-on, 3D swap, and obstacle cases
- completion bands on generated 3D stress slices that remain practical to run locally
- broad but explicit compute p95 bands on smoke, experimental, and small 3D stress rows
- zero planner timeout/error/fallback counts in smoke and experimental baseline runs
- broader tests for degraded observations, dense 3D scenes, candidate capping, candidate-risk accounting, and public `PlannerInput`/`PlannerOutput` behavior
- optional stronger shooting-method or solver-backed variant if needed

`baseline-evidence` exercises dense nonplanar local scenes with nearby traffic, stale tracks, intent trajectories, and obstacles. For `mpc_local`, it checks candidate capping/debug signals, verifies stale-track risk inflation, and records per-call p50/p95 timing for the sampled planner call. For `mpc_nonlinear`, `bvc_tube_dmpc`, `dynamic_tube_dmpc`, `rmader`, and `ego_swarm_opt`, it checks optimizer-grade dense-3D signals, hard BVC tube/cell constraint reporting, dynamic tube reconstruction and risk-triggered collision constraint reporting, RMADER MINVO/hyperplane commit and delay-check fallback reporting, degraded intent/V2V risk inflation where applicable, SciPy-or-fallback solver status for supported solvers, dense-3D timing bands, and optional Foxglove-ready JSONL trace artifacts. Passing it supports public-alpha comparison, but the report intentionally recommends keeping these methods experimental until dense-3D compute bands and stress behavior are calibrated on official suites.

`mpc_nonlinear` is the optimizer-grade MPC counterpart to `mpc_local`. It uses finite-horizon multiple shooting over bounded acceleration controls with a double-integrator translational model, multistart avoidance seeds, warm starts, dynamic obstacle predictions, intent-trajectory penalties, static AABB obstacle penalties, smoothness/jerk costs, terminal tracking, and trajectory intent output. The default solver is deterministic projected-gradient so it runs without optional dependencies; `solver: auto` or `solver: scipy_l_bfgs_b` can use SciPy L-BFGS-B when available.

Useful nonlinear MPC debug fields include:

- `mpc_nonlinear_solver`
- `mpc_nonlinear_solver_status`
- `mpc_nonlinear_horizon_steps`
- `mpc_nonlinear_initializations`
- `mpc_nonlinear_best_seed`
- `mpc_nonlinear_initial_cost`
- `mpc_nonlinear_final_cost`
- `mpc_nonlinear_cost_reduction`
- `mpc_nonlinear_collision_penalty`
- `mpc_nonlinear_obstacle_penalty`
- `mpc_nonlinear_intent_penalty`
- `mpc_nonlinear_min_swarm_clearance_m`
- `mpc_nonlinear_min_obstacle_clearance_m`
- `mpc_nonlinear_intent_points`
- `mpc_nonlinear_velocity_guard_adjusted`
- `mpc_nonlinear_velocity_guard_constraint_count`
- `mpc_nonlinear_velocity_guard_brake_applied`
- `mpc_nonlinear_velocity_guard_min_clearance_m`

Use `mpc_nonlinear` when comparing against `ego_swarm_opt`: both optimize a planned trajectory and publish intent, while `mpc_local` and `ego_swarm` remain faster sampled/scored baselines.

Additional promotion requirements for `mpc_nonlinear`:

- compare projected-gradient and SciPy solver modes on generated 3D stress suites
- calibrate compute p95 bands separately from `mpc_local`
- test infeasible close-range conflicts where predicted single-agent clearance cannot be made positive under acceleration limits
- add richer obstacle-field and degraded-intent evidence before treating it as a reference baseline

`dmpc_best_response` is the distributed-MPC counterpart to `mpc_nonlinear`. It is still an ego-trajectory optimizer per drone, but it treats received neighbor intent trajectories as coupled trajectory constraints, inflates stale or missing plans, emits an `INTENT_TRAJECTORY` agent message in addition to the benchmark intent channel, and republishes its optimized trajectory for the next best-response round. This is an asynchronous one-round-per-simulator-tick distributed best response, not a centralized joint solve and not an ADMM/consensus optimizer.

Useful distributed MPC debug fields include:

- `dmpc_best_response_solver`
- `dmpc_best_response_solver_status`
- `dmpc_best_response_horizon_steps`
- `dmpc_best_response_best_seed`
- `dmpc_best_response_neighbor_intent_count_considered`
- `dmpc_best_response_stale_intent_count`
- `dmpc_best_response_missing_intent_count`
- `dmpc_best_response_intent_primary_predictions`
- `dmpc_best_response_fallback_cv_predictions`
- `dmpc_best_response_coupled_constraints`
- `dmpc_best_response_pairwise_slack_penalty`
- `dmpc_best_response_min_coupled_clearance_m`
- `dmpc_best_response_velocity_guard_adjusted`
- `dmpc_best_response_velocity_guard_constraint_count`
- `dmpc_best_response_velocity_guard_brake_applied`
- `dmpc_best_response_velocity_guard_min_clearance_m`
- `dmpc_best_response_agent_messages`

Use `dmpc_best_response` when you want to compare a reactive local NMPC (`mpc_nonlinear`) against a trajectory-sharing distributed-MPC formulation under the same planner contract. It should be judged on safety and completion metrics plus intent stability, plan staleness, communication load, and compute cost.

Additional promotion requirements for `dmpc_best_response`:

- calibrate dense-3D behavior under realistic V2V rate, delay, and packet loss
- add side-by-side traces against `mpc_nonlinear` and `ego_swarm_opt`
- add explicit plan-staleness and missing-intent ablations
- evaluate whether a synchronous ADMM/consensus DMPC variant is needed as a separate method

`bvc_tube_dmpc` is the hard spatial-partitioning distributed-MPC baseline. It builds time-indexed buffered Voronoi-cell halfspace constraints from local neighbor tracks and received intent trajectories, adds obstacle halfspace boundaries, then optimizes and projects a waypoint tube so the committed trajectory stays inside the assigned moving cell/tube when one is feasible. Because the public simulator interface still asks each local planner for the next velocity command, the planner returns the first bounded command from the committed tube and publishes the full trajectory as intent plus an agent message.

Useful BVC tube-DMPC debug fields include:

- `bvc_tube_dmpc_solver`
- `bvc_tube_dmpc_solver_status`
- `bvc_tube_dmpc_horizon_steps`
- `bvc_tube_dmpc_best_topology`
- `bvc_tube_dmpc_hard_cell_ok`
- `bvc_tube_dmpc_candidate_hard_cell_ok`
- `bvc_tube_dmpc_cell_constraint_count`
- `bvc_tube_dmpc_neighbor_constraint_count`
- `bvc_tube_dmpc_intent_constraint_count`
- `bvc_tube_dmpc_obstacle_constraint_count`
- `bvc_tube_dmpc_max_cell_violation_m`
- `bvc_tube_dmpc_min_cell_slack_m`
- `bvc_tube_dmpc_fallback`
- `bvc_tube_dmpc_velocity_guard_adjusted`
- `bvc_tube_dmpc_velocity_guard_constraint_count`
- `bvc_tube_dmpc_velocity_guard_brake_applied`
- `bvc_tube_dmpc_velocity_guard_min_clearance_m`
- `bvc_tube_dmpc_agent_messages`

Use `bvc_tube_dmpc` when comparing trajectory-sharing DMPC styles: it is more geometry-constrained than `dmpc_best_response`, while `rmader` emphasizes robust delayed trajectory publication and MINVO/hyperplane separation. This implementation is a clean-room benchmark baseline inspired by buffered Voronoi-cell and uncertainty-aware Voronoi-cell formulations; it is not an official port of a BVC/B-UAVC or Schoellig-lab DMPC codebase.

Additional promotion requirements for `bvc_tube_dmpc`:

- calibrate hard-cell feasibility and fallback rates on dense 3D, obstacle, degraded V2V, and heterogeneous-priority suites
- add side-by-side traces against nonlinear MPC, distributed MPC best-response, RMADER, and EGO-Swarm optimized baselines
- characterize whether cell/tube constraints are too conservative in realistic noncooperative intruder and urban obstacle cases
- decide whether to add an optional external QP/SOCP backend for larger cell-constrained horizon programs

`dynamic_tube_dmpc` is the paper-specific dynamic tube-DMPC baseline following Dai, Liao, and Chen, "Safe Swarm Navigation in Constrained Environments: A Dynamic Tube-Based Distributed MPC Approach" (*Drones*, 2026). It implements the paper's double-integrator prediction model, condensed acceleration-QP objective, risk-triggered linearized collision constraints, elastic virtual-tube reconstruction, local tube halfspace constraints, norm-bounded acceleration/velocity projection, and assumed predicted trajectory broadcasting.

Useful dynamic tube-DMPC debug fields include:

- `dynamic_tube_dmpc_solver`
- `dynamic_tube_dmpc_solver_status`
- `dynamic_tube_dmpc_horizon_steps`
- `dynamic_tube_dmpc_qp_variables`
- `dynamic_tube_dmpc_qp_constraint_count`
- `dynamic_tube_dmpc_risk_agent_count`
- `dynamic_tube_dmpc_first_risk_step`
- `dynamic_tube_dmpc_collision_constraint_count`
- `dynamic_tube_dmpc_tube_reconstruction_active`
- `dynamic_tube_dmpc_tube_update_trigger`
- `dynamic_tube_dmpc_tube_connected`
- `dynamic_tube_dmpc_tube_max_shift_m`
- `dynamic_tube_dmpc_tube_constraint_count`
- `dynamic_tube_dmpc_equations`
- `dynamic_tube_dmpc_agent_messages`

Use `dynamic_tube_dmpc` when you want to compare the Dai/Liao/Chen dynamic-tube formulation against the more generic `bvc_tube_dmpc`, `dmpc_best_response`, `rmader`, and `ego_swarm_opt` baselines. It is adapted to DAA Microbench's local velocity-command interface and AABB obstacle representation; it is not a drop-in copy of the authors' MATLAB implementation.

Additional promotion requirements for `dynamic_tube_dmpc`:

- add generated virtual-tube traversal scenarios that match the paper's static and dynamic tube experiments
- calibrate dense-swarm behavior for 10-12 agents with tube radii and dynamic obstacle intrusions close to the paper
- characterize solve time and fallback behavior separately for cached receding-horizon ticks and full QP replans
- add side-by-side Foxglove traces against `bvc_tube_dmpc`, `dmpc_best_response`, `rmader`, and `ego_swarm_opt`

`rmader` is the robust MADER-family trajectory-sharing baseline. It builds a cubic B-spline local plan, converts every interval into continuous MINVO polyhedra, constructs hard separating hyperplanes against dynamic neighbor/intent hulls and static AABB obstacle hulls, smooths the control polygon under velocity/acceleration/jerk diagnostics, and publishes candidate plus committed trajectories through the intent message bus. The delay check gates commitment on the hard MINVO separation recheck; if the candidate is unsafe, the planner keeps a prior committed trajectory when available or falls back to a braking trajectory.

Useful RMADER debug fields include:

- `rmader_solver`
- `rmader_solver_status`
- `rmader_minvo_intervals`
- `rmader_hard_constraint_count`
- `rmader_candidate_hard_constraint_ok`
- `rmader_candidate_max_hyperplane_violation_m`
- `rmader_delay_check_passed`
- `rmader_delay_check_fallback`
- `rmader_kinematic_ok`
- `rmader_candidate_max_accel_violation_mps2`
- `rmader_plan_version`
- `rmader_agent_messages`
- `rmader_cached_reuse_validation`
- `rmader_dynamic_broadphase_margin_m`
- `rmader_max_dynamic_hulls_per_interval`
- `rmader_velocity_guard_adjusted`
- `rmader_velocity_guard_constraint_count`
- `rmader_velocity_guard_brake_applied`
- `rmader_velocity_guard_min_clearance_m`

Use `rmader` when comparing robust trajectory publication and hard convex-separation behavior against `dmpc_best_response`, `mpc_nonlinear`, and `ego_swarm_opt`. It is an original Python implementation adapted to DAA Microbench's local velocity-command contract, not a ROS/Gurobi port of the MIT ACL RMADER/MADER codebase.

Additional promotion requirements for `rmader`:

- validate any official MIT ACL RMADER comparison through `examples/external_reference_rmader_manifest.yaml` or a filled copy of it
- add capped optimizer-suite evidence across dense 3D, delayed/lossy V2V, and noncooperative-intruder scenarios
- characterize delay-check fallback rates separately from planner guardrail fallbacks
- compare against nonlinear MPC, distributed MPC, and EGO-Swarm optimized baselines with side-by-side Foxglove traces
- decide whether to add an optional external solver backend for larger MINVO/hyperplane programs

Observed local calibration on tiny generated suites before public-alpha tuning:

- generated experimental/smoke rows keep `cbf_qp` planner p95 around hundredths of a millisecond per tick per agent
- generated experimental/smoke rows keep `mpc_local` planner p95 in the low single-digit milliseconds per tick per agent on this machine
- a single `official_3d_stress` `mpc_local` row can still take tens of seconds wall-clock locally, so it remains outside default CI smoke
- a 20-second stable-metadata prep review with `baseline-review --methods cbf_qp,mpc_local --duration-s 20` passes the selected 3D/degraded review lanes for both methods, but reports `needs_reference_role_decision` because both remain `experimental_baseline`
- passing that review is evidence for promotion discussion, not promotion by itself; CBF still needs stronger solver/fallback validation, and MPC still needs broader compute and dense-3D stress characterization before either should become a public reference baseline
- `baseline-evidence` adds cheap CBF, MPC, NMPC, BVC tube-DMPC, dynamic tube-DMPC, RMADER, EGO-Swarm optimizer, VO, and RVO targeted checks; passing it is a local evidence point, not a substitute for generated-suite stress characterization

## EGO-Swarm-Inspired Trajectory-Sharing Baselines

`ego_swarm` is a clean-room EGO-Swarm-inspired local planner. The upstream EGO-Swarm project is a decentralized, asynchronous quadrotor swarm system for unknown cluttered environments, and its public repository is GPLv3. DAA Microbench does not vendor or port that code. Instead, this baseline adapts the published idea to the benchmark contract: each agent samples smooth receding-horizon trajectory topologies, scores goal progress, smoothness, dynamic feasibility, static obstacle clearance, and predicted swarm clearance, then publishes the selected trajectory as an intent message.

Useful debug fields include:

- `ego_swarm_algorithm`
- `ego_swarm_best_topology`
- `ego_swarm_candidates`
- `ego_swarm_min_swarm_clearance_m`
- `ego_swarm_min_obstacle_clearance_m`
- `ego_swarm_swarm_penalty`
- `ego_swarm_obstacle_penalty`
- `ego_swarm_intent_count_considered`
- `ego_swarm_intent_points`

Requirements before promoting it to a reference baseline:

- official 3D stress evidence against ORCA, CBF, MPC, VO, and RVO
- official ZJU FAST-Lab EGO-Swarm comparison through `examples/external_reference_ego_swarm_manifest.yaml` or a filled copy of it
- degraded intent/V2V calibration with delayed and stale trajectory sharing
- obstacle-rich scenario evidence beyond AABB proximity penalties
- compute p95 bands on dense 3D scenes
- docs that clearly distinguish the clean-room benchmark baseline from the upstream GPL implementation

`ego_swarm_opt` is the stronger clean-room optimizer variant. It starts from the same decentralized trajectory-sharing idea, but instead of only scoring sampled arcs, it creates topological control-point seeds and optimizes the control points against a continuous cost with smoothness, path length, velocity/acceleration-limit, swarm-clearance, obstacle-clearance, warm-start, and intent-sharing terms. The default solver is deterministic projected-gradient so the baseline remains dependency-free; `solver: auto` or `solver: scipy_l_bfgs_b` can use SciPy L-BFGS-B when available and then fall back to projected-gradient if needed.

Useful optimizer debug fields include:

- `ego_swarm_opt_solver`
- `ego_swarm_opt_solver_status`
- `ego_swarm_opt_control_points`
- `ego_swarm_opt_curve_samples`
- `ego_swarm_opt_initializations`
- `ego_swarm_opt_replanned`
- `ego_swarm_opt_cached_reuse`
- `ego_swarm_opt_replan_period_s`
- `ego_swarm_opt_plan_age_s`
- `ego_swarm_opt_best_topology`
- `ego_swarm_opt_initial_cost`
- `ego_swarm_opt_final_cost`
- `ego_swarm_opt_cost_reduction`
- `ego_swarm_opt_dynamic_penalty`
- `ego_swarm_opt_min_swarm_clearance_m`
- `ego_swarm_opt_min_obstacle_clearance_m`
- `ego_swarm_opt_intent_points`
- `ego_swarm_opt_velocity_guard_adjusted`
- `ego_swarm_opt_velocity_guard_constraint_count`
- `ego_swarm_opt_velocity_guard_brake_applied`
- `ego_swarm_opt_velocity_guard_min_clearance_m`

Use `ego_swarm_opt` when comparing against `mpc_local`: `mpc_local` is a sampled velocity-command predictive baseline, while `ego_swarm_opt` optimizes a planned trajectory and publishes it as intent. That distinction is exactly what the advanced comparison lane is meant to expose.

Additional promotion requirements for `ego_swarm_opt`:

- compare against `mpc_local` on the same 3D conflict lane and full generated suites
- validate any official ZJU FAST-Lab EGO-Swarm comparison through the EGO-Swarm external-reference manifest or bundle workflow
- calibrate compute p95 bands separately for projected-gradient and SciPy solver modes
- verify degraded/stale intent behavior under packet loss and delayed V2V
- add denser obstacle-field evidence once richer maps or obstacle sets are available

## Velocity-Obstacle Baselines

`velocity_obstacle` is an experimental 2D/3D finite-horizon VO-cone sampler. It samples bounded candidate velocity commands, predicts each local neighbor with constant velocity, penalizes candidates that enter the inflated velocity-obstacle cone within the configured horizon, inflates stale tracks using bounded age and velocity uncertainty, and also scores static AABB obstacle clearance.

`reciprocal_velocity_obstacle` is the stronger reciprocal variant. It uses a hybrid VO/RVO apex, deterministic responsibility sharing, stale-track responsibility inflation, HRVO-style apex diagnostics, and tangent-boundary candidate commands. It is meant to be compared against `velocity_obstacle` to show the benefit and limits of reciprocal assumptions under degraded V2V/sensor conditions.

Useful debug fields include:

- `vo_algorithm`
- `vo_candidates`
- `vo_conflict_count`
- `vo_min_ttc_s`
- `vo_min_pred_clearance_m`
- `vo_best_clearance_improvement_m`
- `vo_safe_candidate_count`
- `vo_pred_conflict_candidate_count`
- `vo_stale_inflation_max_m`
- `vo_cone_penalty`
- `vo_obstacle_penalty`
- `vo_planar`
- `vo_reciprocal_mode` for `reciprocal_velocity_obstacle`
- `vo_responsibility_mean` for `reciprocal_velocity_obstacle`
- `vo_responsibility_min` / `vo_responsibility_max` for `reciprocal_velocity_obstacle`
- `vo_stale_responsibility_boost_mean` for `reciprocal_velocity_obstacle`
- `vo_hrvo_apex_shift_mean` / `vo_hrvo_apex_shift_max` for `reciprocal_velocity_obstacle`
- `vo_boundary_candidate_count` for `reciprocal_velocity_obstacle`

Requirements before promoting it to a reference baseline:

- clear distinction from the existing ORCA-like heuristic in docs and calibration reports
- all-suite leaderboard evidence for both `velocity_obstacle` and `reciprocal_velocity_obstacle`
- collision-free or bounded-collision behavior on calibrated head-on, crossing, merge, and 3D swap lanes
- degraded-sensing and stale-track calibration, since VO behavior is sensitive to track uncertainty
- obstacle-field tests beyond single AABB smoke cases
- runtime p95 bands on generated 3D stress rows
- comparison against ORCA-like, CBF-QP, and MPC-local on the same official suites

## Tiny Learned Baseline

`learned_tiny` is a frozen learned-policy fixture. It loads `microbench/bundled_config/learned_baselines/tiny_linear_policy.json`, maps public local planner features to a normalized `(3,)` action through a linear-tanh model, and scales that action into the planner velocity-command contract.

Useful debug fields:

- `learned_model`
- `learned_model_id`
- `learned_weight_artifact`
- `learned_policy_action_norm`
- `learned_policy_threat_scalar`
- `learned_policy_neighbor_count_frac`

The matching RL policy name is `tiny_learned`:

```bash
python -m microbench.cli rl-smoke \
  --out-dir runs_rl_tiny_learned \
  --policy tiny_learned \
  --require-pass
```

The deterministic synthetic training recipe is in `examples/rl_train_tiny_linear_policy.py`. This baseline is included so learned-model submissions have a minimal tested reference path for weight artifacts, disclosure, adapters, and official CSV generation. It should not be treated as a competitive or certified DAA controller.

## MLP Learned Baseline

`learned_mlp` is a stronger frozen learned-policy fixture than `learned_tiny`. It loads `microbench/bundled_config/learned_baselines/mlp_policy.json`, maps the same public local learned-policy features to a normalized `(3,)` action through a two-layer tanh MLP, and scales that action into the planner velocity-command contract.

Additional debug fields:

- `learned_policy_architecture`
- `learned_policy_hidden_dim`
- `learned_policy_action_norm`
- `learned_policy_threat_scalar`
- `learned_policy_neighbor_count_frac`

The matching RL policy name is `mlp_learned`:

```bash
python -m microbench.cli rl-smoke \
  --out-dir runs_rl_mlp_learned \
  --policy mlp_learned \
  --require-pass
```

The deterministic synthetic training recipe is in `examples/rl_train_mlp_policy.py`. This baseline is useful for nonlinear learned-policy adapter and submission-bundle testing, but it is still a benchmark fixture rather than a competitive learned DAA controller.

## External Learned-Policy Bridge

`learned_policy_spec` loads a trusted external policy spec and evaluates it as a normal local planner. It uses the same observation/action contract as the RL wrappers, so a policy that passes `rl-smoke` can also produce benchmark `results.csv` and `summary.csv` rows:

```bash
python -m microbench.cli run \
  --scenario config/scenarios/stacked_swap_3d.yaml \
  --method learned_policy_spec \
  --policy-spec examples/external_policy_spec.json \
  --n 4 \
  --seed 0 \
  --comm ideal_50hz \
  --out-dir runs_external_policy_planner
```

Use this method for learned-policy submissions that should be reviewed as planner sweeps. It requires `--policy-spec` and is intentionally excluded from reference-baseline status.

To create a dependency-free trained spec from benchmark observations, use the built-in behavior-cloning workflow:

```bash
python -m microbench.cli train-learned-bc \
  --out-dir runs_bc_mlp_policy \
  --lanes head_on,crossing,urban_obstacle \
  --eval-lanes head_on,crossing \
  --require-pass
```

This writes `runs_bc_mlp_policy/policy_spec.json`, which can be used with `learned_policy_spec`, `rl-validation-matrix`, and learned-submission bundles. It clones a transparent local DAA teacher from public RL observations and declares a forward-progress/unit-norm action guardrail. Treat it as a starter learned baseline, not a certified DAA controller.

To package that trained policy and compare it against the frozen learned fixtures:

```bash
python -m microbench.cli learned-bc-evidence \
  --out-dir runs_bc_mlp_evidence \
  --lanes head_on,crossing,urban_obstacle \
  --max-runs 1 \
  --require-pass
```

This writes a trained BC bundle, `learned_tiny`/`learned_mlp` comparison bundles, a learned-policy leaderboard JSON/CSV, and a diagnostics JSON/CSV/Markdown report with labels such as `safe_but_slow`, `fast_but_close`, and `balanced`.

For focused learned-policy iteration, `learned-hard-lane-loop` can consume those diagnostics, select the weakest canonical validation lanes, export public observation/action shards, train the BC MLP from the shards, and rerun the bundle/leaderboard/diagnostic evidence. Add `--mix-lanes` to keep broad replay in the training set beside the selected hard lanes, and use `--sample-weighting safety` when collision, near-miss, and low-clearance samples should receive more supervised fit weight. The training-only lane `dense_swarm_hard_negative` can be mixed in explicitly when dense 3D swarm close approaches are the known weak spot; pair it with `--sample-selection hard_negative_windows` to fit only its collision, near-miss, low-clearance, or closest-approach windows.

```bash
python -m microbench.cli learned-hard-lane-loop \
  --out-dir runs_hard_lane_loop \
  --diagnostics runs_bc_mlp_evidence/learned_policy_diagnostics.json \
  --mix-lanes head_on,crossing,urban_obstacle,communication_delay,high_n_dense_merge,dense_swarm_hard_negative \
  --sample-weighting safety \
  --sample-selection hard_negative_windows \
  --max-lanes 3 \
  --max-runs 1 \
  --require-pass
```

For rollout-aware learned-policy iteration, `learned-closed-loop-finetune` starts from an existing portable `mlp_json` policy spec, evaluates perturbed MLP parameters through the PettingZoo-style environment, and accepts only candidates that improve the configured closed-loop objective without collision, clearance, or near-miss guardrail regression.

```bash
python -m microbench.cli learned-closed-loop-finetune \
  --out-dir runs_closed_loop_finetune \
  --base-policy-spec runs_bc_mlp_policy/policy_spec.json \
  --lanes head_on,crossing,urban_obstacle,communication_delay,high_n_dense_merge \
  --trainable-parameters all_layers \
  --search-strategy two_stage \
  --antithetic-sampling \
  --require-per-lane-safety \
  --generations 4 \
  --population-size 12 \
  --train-max-steps 24 \
  --holdout-profile broad_3d_stress \
  --holdout-scenarios sphere_swap_3d_medium,dense_swarm_3d_hard,merge_3d_hard,sensor_volume_3d_hard,noncooperative_intruder_3d_hard \
  --holdout-seeds 0:2 \
  --holdout-comm ideal_50hz,degraded_20hz \
  --require-pass \
  --require-promotion
```

This is benchmark-native closed-loop learned-policy infrastructure. It writes both aggregate candidate results and `candidate_lane_summary.csv`, which shows score, clearance, completion, and lane-level safety regression flags for every candidate. `--search-strategy two_stage` runs an output-head warmup before all-layer refinement, `--antithetic-sampling` evaluates plus/minus perturbation pairs, and `--require-per-lane-safety` rejects aggregate winners that regress an individual lane. `--require-promotion` requires the tuned policy to avoid safety, clearance, and `score_v0` regression against the base policy on the selected broad 3D holdout rows. It is not yet a claim that the shipped learned baseline is SOTA.

When the goal is to reduce validation-lane overfit, use `--lane-profile validation_plus_broad_3d`. That profile trains on the canonical learned validation lanes plus broad 3D stress training lanes for sphere swap, dense swarm, merge, sensor-volume, and noncooperative-intruder geometry.

For a single auditable learned-baseline study, use the wrapper:

```bash
python -m microbench.cli learned-closed-loop-study \
  --out-dir runs_closed_loop_study \
  --base-policy-spec runs_bc_mlp_policy/policy_spec.json \
  --lane-profile validation_plus_broad_3d \
  --trainable-parameters all_layers \
  --search-strategy two_stage \
  --antithetic-sampling \
  --require-per-lane-safety \
  --holdout-profile broad_3d_stress \
  --comparison-bundle runs_bc_mlp_evidence/bc_bundle \
  --bundle-max-runs 1 \
  --require-pass
```

This writes the training report, broad-holdout evidence, learned submission bundle, learned leaderboard, diagnostics, and a top-level recommendation in one folder. A study can be structurally `ok` while still returning `reject_regression`; that is expected when the holdout catches learned-policy overfit. If the holdout passes but no candidate was accepted beyond the base policy, the recommendation is `holdout_passed_no_update`.

## Promotion Calibration

`baseline-promotion --require-calibrated` is the current public-alpha gate for experimental baselines. Passing it means the method imports, has docs/tests coverage, supports 2D and 3D, runs the behavior smoke without planner guardrail failures, emits its expected signal/debug contract, and passes compact promotion-calibration acceptance on an 8-second 3D stress lane plus an 8-second degraded fused-sensing lane. For `cbf_qp` and `mpc_local`, it also runs `official_experimental_baselines` and checks that suite acceptance metadata.

Optional stable-metadata review:

```bash
python -m microbench.cli baseline-review \
  --out-dir runs_baseline_review \
  --duration-s 20
```

This is intentionally outside release readiness. It runs longer selected rows from `official_3d_stress` and `official_agentic_stress`, records `baseline_review.json`, and reports per-method metadata recommendations such as `review_for_pre_v1_metadata`, `needs_reference_role_decision`, or `keep_experimental_until_review_checks_pass`. Use `--plan-only` before running, `--lanes` / `--methods` to narrow scope, and `--full-duration` when you want the official generated scenario durations instead of the default 20-second review override.

Passing this gate does not make a method a stable reference baseline. Stable-v1 promotion still requires:

- method metadata changed out of `experimental` status after review
- reference or agentic-reference role assignment where appropriate
- collision-free or explicitly bounded-collision behavior on calibrated head-on, obstacle, and 3D stress slices
- passing compact promotion-calibration bands plus longer `official_3d_stress` / `official_agentic_stress` review
- degraded communication and sensor calibration beyond the compact 8-second lane when the method is meant to be a leaderboard anchor
- updated docs, fixtures, and leaderboard policy language

## Baseline Comparison Fixture

`golden/baseline_comparison/report.json` is a tiny, reproducible comparison fixture generated from `official_experimental_baselines` with:

- `baseline_goal`
- `orca_heuristic`
- `cbf_qp`
- `mpc_local`

It is not a leaderboard. The suite duration is intentionally short, so the fixture emphasizes safety, compute cost, and guardrail behavior rather than final mission completion. Use it as a compact sanity check and a documentation example for the report schema.

New baselines should include registry metadata, docs, focused tests, and at least one generated-suite smoke run before being recommended in official comparisons.
