# Baseline Methods

DAA Microbench ships baselines for calibration, debugging, and reference comparison. They are not all leaderboard anchors. Use this page with:

```bash
python -m microbench.cli list-methods
python -m microbench.cli list-methods --json --include-aliases
python -m microbench.cli baseline-audit
python -m microbench.cli baseline-audit --require-public-alpha-ready --json
python -m microbench.cli baseline-smoke --out-dir runs_baseline_smoke --require-pass
python -m microbench.cli baseline-promotion --out-dir runs_baseline_promotion --require-calibrated
```

The public-alpha baseline gate is intentionally stricter than "the code imports":

- required public-alpha reference baselines: `orca_heuristic`, `orca_with_staleness`, `priority_yield`
- experimental but runnable baselines: `cbf_qp`, `mpc_local`, `negotiation_yield`
- illustrative or template methods: `baseline_goal`, `intent_dummy`, `template`

Run `baseline-audit --require-public-alpha-ready`, `baseline-smoke --require-pass`, and `baseline-promotion --require-calibrated` before inviting external baseline comparisons. Stable v1 still requires promotion work; `baseline-audit --require-stable-v1-ready` and `baseline-promotion --require-stable-v1-ready` are expected to fail while experimental baselines remain experimental.

## Current Methods

| Method | Role | Uses | Dimensions | Intended use |
|---|---|---|---|---|
| `baseline_goal` | illustrative baseline | ego state, goal | 2D, 3D | Lower bound that shows how hard a scenario is without avoidance. |
| `orca_heuristic` | reference baseline | local neighbor tracks, V2V/sensor/fused observations, obstacles | 2D, 3D | Main ORCA-like geometric comparison baseline. |
| `orca_with_staleness` | reference baseline | same as `orca_heuristic`, with stronger stale-track inflation | 2D, 3D | Degraded communication or stale sensor-track comparison baseline. |
| `cbf_qp` | experimental baseline | local neighbor tracks, V2V/sensor/fused observations, obstacles | 2D, 3D | Dependency-free CBF projection baseline with optional SciPy solver mode. |
| `mpc_local` | experimental baseline | local neighbor tracks, V2V/sensor/fused observations, obstacles | 2D, 3D | Deterministic short-horizon predictive sampling baseline. |
| `priority_yield` | agentic reference baseline | local tracks, priority, agent messages | 2D, 3D | Simple decentralized right-of-way behavior. |
| `negotiation_yield` | experimental agentic baseline | local tracks, proposal/ACK messages, priority | 2D, 3D | Structured negotiation plumbing and early agentic comparison. |
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
- `negotiation_yield` once its behavior is better calibrated

Experimental baselines are runnable but not leaderboard anchors yet:

- `cbf_qp`
- `mpc_local`

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

This runs every non-template built-in baseline on one planar and one 3D generated smoke scenario, checks finite key metrics, zero planner guardrail counts, 2D/3D coverage, agent-message signals for `priority_yield`, proposal/ACK signals for `negotiation_yield`, and public debug/intent output contracts for `cbf_qp`, `mpc_local`, and `intent_dummy`.

Experimental promotion calibration:

```bash
python -m microbench.cli baseline-promotion \
  --out-dir runs_baseline_promotion \
  --require-calibrated
```

This produces `baseline_promotion.json`. It should report `public_alpha_calibrated=true` and `stable_v1_ready=false` for `cbf_qp`, `mpc_local`, and `negotiation_yield` during public alpha. The report records smoke metrics, generated experimental-suite acceptance for CBF/MPC, compact `official_promotion_calibration` 3D/degraded acceptance for all promotion candidates, method-specific signal contracts, and stable-v1 blockers such as experimental metadata status or non-reference roles.

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

CBF skeleton smoke:

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

## Stale-Aware ORCA Preset

`orca_with_staleness` uses the same planner implementation as `orca_heuristic`, but its default config is more conservative when neighbor tracks are old:

- larger `stale_age_cap_s`
- larger `stale_inflation_gain`
- larger `responsibility_age_gain`
- slightly larger closing-speed and sidestep buffers

This is useful for comparisons where `obs_stale_fraction_mean`, `obs_sensor_track_stale_fraction_mean`, or communication drop/delay metrics are high. It may reduce collisions at the cost of slower completion or extra path deviation.

## CBF-QP Skeleton

`cbf_qp` is currently an experimental CBF-style baseline. The default `solver: projection` mode is quiet and dependency-free. Optional `solver: auto` or `solver: scipy` modes use SciPy SLSQP when available, then fall back to deterministic halfspace projection. It constructs pairwise and obstacle barrier halfspaces, clamps speed, and uses a deterministic away-from-risk fallback if constraints remain violated.

Use it for development and API comparison, not as a mature CBF baseline. It is useful because it establishes:

- the public method name and config block
- 2D/3D command shape
- neighbor and obstacle barrier semantics
- bounded fallback behavior
- debug fields in `PlannerOutput.debug_info`
- solver status reporting via `cbf_solver`, `cbf_solver_requested`, and `cbf_solver_status`

Requirements before promoting it to a reference baseline:

- collision-free or clearly bounded-collision behavior on calibrated head-on, 3D swap, and obstacle cases
- completion bands on at least one generated 3D suite without relying on privileged state
- broad but explicit compute p95 bands on smoke and 3D stress rows
- zero planner timeout/error/fallback counts in smoke and experimental baseline runs
- stronger solver-backed validation, with documented deterministic projection fallback
- tests for infeasible constraints, solver failure, stale/noisy observations, and 2D/3D obstacle barriers

## MPC-Local Skeleton

`mpc_local` is currently an experimental local predictive baseline. It samples one-step-reachable velocity commands, rolls them forward over a short horizon, and scores goal tracking, progress, smoothness, predicted agent-agent clearance, static obstacle clearance, and approach-to-conflict costs.

It is intentionally dependency-free and deterministic. Its command is bounded by `a_max * dt` from the current velocity, so the dynamics layer should not need to rescue it through acceleration saturation during normal operation.

Useful debug fields include:

- `mpc_candidates`
- `mpc_horizon_steps`
- `mpc_best_cost`
- `mpc_min_pred_clearance_m`
- `mpc_collision_penalty`
- `mpc_obstacle_penalty`
- `mpc_approach_penalty`
- `mpc_accel_delta_norm`

Requirements before promoting it to a reference baseline:

- collision-free or clearly bounded-collision behavior on calibrated head-on, 3D swap, and obstacle cases
- completion bands on generated 3D stress slices that remain practical to run locally
- broad but explicit compute p95 bands on smoke, experimental, and small 3D stress rows
- zero planner timeout/error/fallback counts in smoke and experimental baseline runs
- tests for degraded observations, dense 3D scenes, candidate capping, and public `PlannerInput`/`PlannerOutput` behavior
- optional stronger shooting-method or solver-backed variant if needed

Observed local calibration on tiny generated suites before public-alpha tuning:

- generated experimental/smoke rows keep `cbf_qp` planner p95 around hundredths of a millisecond per tick per agent
- generated experimental/smoke rows keep `mpc_local` planner p95 in the low single-digit milliseconds per tick per agent on this machine
- a single `official_3d_stress` `mpc_local` row can still take tens of seconds wall-clock locally, so it remains outside default CI smoke

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
