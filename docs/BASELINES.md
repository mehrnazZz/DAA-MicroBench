# Baseline Methods

DAA Microbench ships baselines for calibration, debugging, and reference comparison. They are not all leaderboard anchors. Use this page with:

```bash
python -m microbench.cli list-methods
python -m microbench.cli list-methods --json --include-aliases
```

## Current Methods

| Method | Role | Uses | Dimensions | Intended use |
|---|---|---|---|---|
| `baseline_goal` | illustrative baseline | ego state, goal | 2D, 3D | Lower bound that shows how hard a scenario is without avoidance. |
| `orca_heuristic` | reference baseline | local neighbor tracks, V2V/sensor/fused observations, obstacles | 2D, 3D | Main ORCA-like geometric comparison baseline. |
| `orca_with_staleness` | reference baseline | same as `orca_heuristic`, with stronger stale-track inflation | 2D, 3D | Degraded communication or stale sensor-track comparison baseline. |
| `cbf_qp` | experimental baseline | local neighbor tracks, V2V/sensor/fused observations, obstacles | 2D, 3D | Solver-free CBF projection skeleton for staged development. |
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

## Stale-Aware ORCA Preset

`orca_with_staleness` uses the same planner implementation as `orca_heuristic`, but its default config is more conservative when neighbor tracks are old:

- larger `stale_age_cap_s`
- larger `stale_inflation_gain`
- larger `responsibility_age_gain`
- slightly larger closing-speed and sidestep buffers

This is useful for comparisons where `obs_stale_fraction_mean`, `obs_sensor_track_stale_fraction_mean`, or communication drop/delay metrics are high. It may reduce collisions at the cost of slower completion or extra path deviation.

## CBF-QP Skeleton

`cbf_qp` is currently a deterministic, solver-free CBF-style skeleton. It constructs pairwise and obstacle barrier halfspaces, projects the preferred goal velocity through them for a bounded number of iterations, clamps speed, and uses a deterministic away-from-risk fallback if constraints remain violated.

Use it for development and API comparison, not as a mature CBF baseline. It is useful because it establishes:

- the public method name and config block
- 2D/3D command shape
- neighbor and obstacle barrier semantics
- bounded fallback behavior
- debug fields in `PlannerOutput.debug_info`

Requirements before promoting it to a reference baseline:

- deterministic quadratic-program solver dependency or documented pure-Python fallback
- explicit 2D and 3D barrier constraints for agent-agent and agent-obstacle separation
- bounded solver timeout and deterministic fallback command
- no privileged information beyond `PlannerInput`
- tests for infeasible constraints, solver failure, and stale/noisy observations
- acceptance bands for safety, compute p95, and completion

## Pending Strong Baselines

The following names are intentionally not exposed as planner methods yet. They need implementation and validation before public use.

### `mpc_local`

Requirements before adding:

- fixed-horizon local dynamics model matching benchmark command semantics
- collision, obstacle, smoothness, and progress costs documented in config
- deterministic optimizer settings and bounded runtime
- fallback behavior on timeout or non-convergence
- 2D and 3D support or explicit dimension metadata if staged
- tests for degraded observations, dense 3D scenes, and compute limits

New baselines should include registry metadata, docs, focused tests, and at least one generated-suite smoke run before being recommended in official comparisons.
