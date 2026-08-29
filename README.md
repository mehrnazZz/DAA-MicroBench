# DAA Microbench

[![CI](https://github.com/mehrnazZz/DAA-MicroBench/actions/workflows/ci.yml/badge.svg)](https://github.com/mehrnazZz/DAA-MicroBench/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

DAA Microbench is a deterministic Python harness for evaluating local multi-drone detect-and-avoid planners at 50 Hz under configurable V2V communication impairment (rate, delay/jitter, loss, staleness). It models the local planning contract (velocity commands into a clamped dynamics step) without a full high-fidelity Sim/ROS/PX4 stack. Use it for fast, fair planner implementation and comparisons, agentic multi-drone DAA experiments, and diffusion-training dataset generation from baseline planners.

## Featured Demo: Urban 3D DAA

`urban_conflict_3d` is a packaged 3D scenario with near-coplanar crossing traffic, occluding buildings, a static hazard, fused sensing, intent sharing, and a conflict that straight-line goal seeking does not solve. In the reference seed, `baseline_goal` collides while `reciprocal_velocity_obstacle` completes collision-free.

```bash
python -m microbench.cli run \
  --scenario config/scenarios/urban_conflict_3d.yaml \
  --method reciprocal_velocity_obstacle \
  --n 4 \
  --seed 2 \
  --comm realistic_v2v_50hz \
  --save-trace \
  --out-dir runs_urban_conflict_demo

python -m microbench.cli foxglove-export \
  --trace runs_urban_conflict_demo/episodes/urban_conflict_3d_reciprocal_velocity_obstacle_n4_seed2_comm_realistic_v2v_50hz/trace_episode.jsonl \
  --out runs_urban_conflict_demo/urban_conflict_3d_rvo_avoidance.mcap \
  --trail-frames 2600 \
  --max-sensing-links 24 \
  --compression zstd
```

Open the MCAP in Foxglove Studio for the robotics-grade 3D view.

For a panelized optimizer comparison on the denser urban-throughput map:

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

Open `runs_urban_throughput_comparison/baseline_comparison.mcap` in Foxglove and assign one panel to each `/daa/comparison/<method>/...` topic group. The command also writes `comparison_manifest.json`, which records the scenario, methods, N, seed, duration, communication profile, planner preset, git commit, guardrail summary, and MCAP path.

## 1) What This Is (and Is Not)

- This repo is a fast local-planning benchmark for multi-agent collision avoidance.
- This repo is not a full flight stack and does not model all vehicle, airspace, perception, or system-level effects.
- Primary goals are reproducible sweeps, baseline comparisons, failure traceability, and dataset generation.
- Passing microbench scenarios is a useful gate for local DAA behavior, not a substitute for high-fidelity simulation or flight validation.

## 2) Quickstart (5 Minutes)

Tested from this repository root using `python -m microbench.cli ...`.

1. Create environment and install.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

2. Run a single episode.

```bash
python -m microbench.cli run \
  --scenario config/scenarios/funnel.yaml \
  --method baseline_goal \
  --n 20 \
  --seed 0 \
  --comm ideal_50hz \
  --out-dir runs_readme_quickstart
```

3. Run the generated 2D/3D/agentic smoke sweep.

```bash
python -m microbench.cli canonical-sweep \
  --suite official_smoke_generated \
  --out-dir runs_readme_smoke
```

4. Inspect outputs.

```bash
ls runs_readme_smoke
# results.csv, summary.csv, result_schema.json, episodes/, worst_cases/ (if mined)
```

If your editable install exposed the entrypoint, replace `python -m microbench.cli` with `daa-microbench` or the compatibility alias `microbench`.

Available planners / how to list methods:

```bash
python -m microbench.cli list-methods
python -m microbench.cli list-methods --json --include-aliases
python -m microbench.cli baseline-audit
python -m microbench.cli validate-external-reference --manifest examples/external_reference_rmader_manifest.yaml --json
python -m microbench.cli validate-external-reference --manifest examples/external_reference_ego_swarm_manifest.yaml --json
python -m microbench.cli external-reference-bundle --method-family rmader --out-dir runs_external_references/rmader_official_bundle --scenarios urban_conflict_3d,urban_throughput_3d,stacked_swap_3d --n 4,8 --seeds 2 --comm realistic_v2v_50hz --runner-type ros
python -m microbench.cli external-reference-bundle --method-family ego_swarm --out-dir runs_external_references/ego_swarm_official_bundle --scenarios urban_conflict_3d,urban_throughput_3d,stacked_swap_3d --n 4,8 --seeds 2 --comm realistic_v2v_50hz --runner-type ros
python -m microbench.cli baseline-smoke --out-dir runs_baseline_smoke --require-pass
python -m microbench.cli baseline-promotion --out-dir runs_baseline_promotion --require-calibrated
python -m microbench.cli baseline-evidence --out-dir runs_baseline_evidence --require-pass
python -m microbench.cli baseline-validation-matrix --out-dir runs_baseline_validation_matrix --plan-only
python -m microbench.cli baseline-evidence --out-dir runs_optimizer_evidence --save-optimizer-traces --require-pass
python -m microbench.cli advanced-baseline-comparison --out-dir runs_advanced_baseline_comparison --require-pass
python -m microbench.cli optimizer-suite-review --out-dir runs_optimizer_suite_review --max-runs 4 --require-pass
python -m microbench.cli rl-smoke --out-dir runs_rl_smoke --require-pass
python -m microbench.cli rl-smoke --out-dir runs_external_rl_smoke --policy-spec examples/external_policy_spec.json --require-pass
python -m microbench.cli rl-smoke --out-dir runs_external_model_predict_smoke --policy-spec examples/external_policy_model_predict_spec.json --max-steps 3 --require-pass
python -m microbench.cli run --scenario config/scenarios/stacked_swap_3d.yaml --method learned_policy_spec --policy-spec examples/external_policy_spec.json --n 4 --seed 0 --comm ideal_50hz --out-dir runs_external_policy_planner
python -m microbench.cli rl-smoke --out-dir runs_rl_tiny_learned --policy tiny_learned --require-pass
python -m microbench.cli rl-smoke --out-dir runs_rl_mlp_learned --policy mlp_learned --require-pass
python -m microbench.cli rl-calibration --out-dir runs_rl_calibration --require-pass
python -m microbench.cli rl-validation-matrix --out-dir runs_rl_validation_matrix --policy tiny_learned --require-pass
python -m microbench.cli learned-dataset-export --out-dir runs_learned_dataset --lanes head_on,crossing,urban_obstacle --max-steps 64 --save-replay --require-pass
python -m microbench.cli train-learned-bc --out-dir runs_bc_mlp_policy --lanes head_on,crossing,urban_obstacle --eval-lanes head_on,crossing --require-pass
python -m microbench.cli learned-bc-evidence --out-dir runs_bc_mlp_evidence --lanes head_on,crossing,urban_obstacle --max-runs 1 --require-pass
python -m microbench.cli learned-hard-lane-loop --out-dir runs_hard_lane_loop --diagnostics runs_learned_diagnostics/learned_policy_diagnostics.json --target-policy bc_mlp_learned --target-method learned_policy_spec --fallback-lanes urban_obstacle,communication_delay,high_n_dense_merge --mix-lanes head_on,crossing,urban_obstacle,communication_delay,high_n_dense_merge,dense_swarm_hard_negative --sample-weighting safety --sample-selection hard_negative_windows --max-lanes 3 --max-runs 1 --require-pass
python -m microbench.cli learned-closed-loop-finetune --out-dir runs_closed_loop_finetune --base-policy-spec runs_bc_mlp_policy/policy_spec.json --lanes head_on,crossing,urban_obstacle --trainable-parameters all_layers --generations 2 --population-size 8 --holdout-profile broad_3d_stress --require-pass --require-promotion
python -m microbench.cli learned-closed-loop-study --out-dir runs_closed_loop_study --base-policy-spec runs_bc_mlp_policy/policy_spec.json --lane-profile validation_plus_broad_3d --trainable-parameters all_layers --generations 2 --population-size 8 --holdout-profile broad_3d_stress --bundle-max-runs 1 --require-pass --require-promotion
python -m microbench.cli rl-contract --json
python -m microbench.cli rl-freeze-check --require-pass --json
python -m microbench.cli validate-learned-manifest --manifest examples/learned_submission_manifest_template.json --require-pass
python -m microbench.cli learned-submission-bundle --out-dir runs_learned_bundle --method learned_tiny --policy tiny_learned --require-pass
python -m microbench.cli learned-submission-bundle --out-dir runs_external_learned_bundle --method learned_policy_spec --policy-spec examples/external_policy_spec.json --require-pass
python -m microbench.cli validate-learned-bundle --bundle runs_learned_bundle --require-pass
python -m microbench.cli review-learned-bundle --bundle runs_learned_bundle --require-pass
python -m microbench.cli learned-leaderboard --bundle runs_learned_bundle --bundle runs_external_learned_bundle --out runs_learned_leaderboard/learned_policy_leaderboard.json --require-pass
python -m microbench.cli learned-diagnostics --bundle runs_learned_bundle --bundle runs_external_learned_bundle --out runs_learned_diagnostics/learned_policy_diagnostics.json --require-pass
```

You can also inspect `microbench/planners/` (each planner module maps to a method name in the planner registry).
`orca_heuristic` is the canonical ORCA-like geometric baseline; `orca_expert` remains accepted as a compatibility alias for older scripts and result folders.
See [docs/BASELINES.md](docs/BASELINES.md) for baseline roles, limitations, and recommended comparison sets.
See [docs/BASELINE_FIDELITY.md](docs/BASELINE_FIDELITY.md) for implementation-fidelity tiers, provenance, and external official-reference manifests.
See [docs/DESIGN_V1.md](docs/DESIGN_V1.md) for the benchmark contract and [docs/PLANNER_API.md](docs/PLANNER_API.md) for a planner implementation tutorial.
See [docs/RL_INTERFACE.md](docs/RL_INTERFACE.md) for PettingZoo/Gymnasium-style wrappers and smoke checks, [docs/LEARNED_POLICY_ADOPTION.md](docs/LEARNED_POLICY_ADOPTION.md) for exported learned-policy specs and bundle review, and [docs/RL_STABLE_V1_FREEZE.md](docs/RL_STABLE_V1_FREEZE.md) for RL stable-v1 freeze criteria.
See [docs/README.md](docs/README.md) for the full documentation map.

Run a heterogeneous episode with one explicit planner per drone:

```bash
python -m microbench.cli run \
  --scenario config/scenarios/corridor.yaml \
  --method mixed \
  --agent-methods baseline_goal,template,baseline_goal,template \
  --n 4 \
  --seed 0 \
  --comm ideal_50hz \
  --out-dir runs_heterogeneous_example
```

Run the simple agentic message baseline:

```bash
python -m microbench.cli run \
  --scenario config/scenarios/corridor.yaml \
  --method priority_yield \
  --n 10 \
  --seed 0 \
  --comm ideal_50hz \
  --out-dir runs_priority_yield_example
```

Run the structured proposal/ACK negotiation baseline:

```bash
python -m microbench.cli run \
  --scenario config/scenarios/corridor.yaml \
  --method negotiation_yield \
  --n 10 \
  --seed 0 \
  --comm ideal_50hz \
  --out-dir runs_negotiation_yield_example
```

Set `perception.mode: "sensor"` or `"fused"` in the scenario YAML to switch from V2V-only planner observations.

Optional install tracks (manual extras):
- Core install: `pip install -e .`
- RL wrappers with Gymnasium/PettingZoo classes: `pip install -e ".[rl]"`
- Diffusion/ML extras: `pip install -e ".[ml]"`
- Self-contained Plotly episode reports: `pip install -e ".[viz]"`
- Foxglove/MCAP trace export: `pip install -e ".[foxglove]"`
- Optimization extras: `pip install -e ".[opt]"`

## 3) Core Concepts

- Fixed-step simulation at `dt=0.02` (50 Hz).
- Each drone owns an independent planner instance, so stateful planners can keep per-agent memory.
- Planner outputs a desired `v_cmd`; simulator applies speed and acceleration clamps.
- Agents are spheres for collision/near-miss checks.
- Planner neighbor input is V2V-observed state (possibly stale/noisy), not privileged ground truth.
- Neighbor selection is centralized (`range_m`, `top_k`, threat metric), identical across methods for fairness.
- Planners can use `agent_context.memory` plus delayed/lossy `messages` / `messages_out` for lightweight agentic coordination.
- Planner observations can come from V2V, local sensing, or fused V2V+sensor observations.

### 3.0 Intent Messages (Optional Channel)

Microbench supports an optional intent channel in addition to odometry V2V messages. This is used by trajectory-sharing/negotiation planners.

- Intent delivery is governed by the same comm profile delay/jitter/loss model as odometry.
- Intent send rate is independently throttled by `intent.tx_rate_hz`.
- Most planners can ignore intent entirely.
- Intent `points` are in world frame `(x, y, z)`; in planar mode `y` is fixed.

### 3.0.1 Agentic Messages

In addition to odometry and intent messages, planners can send lightweight agent messages through `PlannerOutput.messages_out`.

- Delivery uses the same delay/loss model as the active comm profile.
- Messages are delivered once via `PlannerInput.messages`; planners should store persistent beliefs in `agent_context.memory`.
- Messages can be directed to `recipient_id` or broadcast when `recipient_id=None`.
- Standard DAA message kinds include negotiation proposals, ACKs, emergency/abort notices, stale-belief notices, priority/yield advisories, and intent trajectories.
- Standard messages carry replay metadata such as `message_id`, `correlation_id`, `seq`, `channel`, `priority`, and `size_bytes`.
- The built-in `priority_yield` planner demonstrates a simple advisory `YIELD` message.
- The built-in `negotiation_yield` planner demonstrates structured `NEGOTIATION_PROPOSAL` and `ACK` exchange.

### 3.0.2 Perception Modes

`perception.mode` controls how planner neighbors are observed before centralized top-k ranking:

- `v2v`: use delivered V2V odometry only. This is the default.
- `sensor`: use local line-of-sight sensor observations only.
- `fused`: prefer fresh sensor observations, otherwise use delivered V2V observations.

Sensor options:

```yaml
perception:
  mode: "sensor"
  sensor:
    range_m: 30.0
    fov_deg: 120.0
    occlusion: true
    occlusion_margin_m: 0.0
    false_negative_p: 0.0
    noise_sigma_pos_m: 0.0
    noise_sigma_vel_mps: 0.0
    track_ttl_s: 0.25
```

Sensor tracks are agent-local. When `track_ttl_s > 0`, a missed detection can remain visible as a stale sensor track until the TTL expires. Stale tracks carry `source: sensor`, `stale: true`, `track_age_sec`, and `last_seen_s`.

Per-agent capability overrides can specialize sensors without changing the global scenario:

```yaml
agents:
  by_id:
    0:
      capabilities:
        sensor:
          range_m: 12.0
          fov_deg: 90.0
          false_negative_p: 0.1
```

When trace logging is enabled, `selected_obs[*].source` is `v2v` or `sensor`; sensor observations also include `stale`, `track_age_sec`, `last_seen_s`, and `occluded` diagnostics.

### 3.1 Coordinate Conventions

- World axes are `(x, y, z)`.
- In planar mode (`world.planar: true`), motion is in the `x-z` plane and `y` is held at `fixed_y_m`.
- In non-planar mode (`world.planar: false`), agents move in full 3D world coordinates.
- `goal_dir` passed to planners is:
  - world-frame
  - unit-norm (`normalize(goal - pos)`), with zero vector fallback if undefined
- Scenario bounds/obstacles are specified in world coordinates using the same axis convention.

### 3.2 PlannerInput and Neighbor Schema

`PlannerInput` fields:
- `ego`: `AgentState` for the controlled agent
- `goal_dir`: `np.ndarray (3,)`, world-frame unit direction to goal
- `neighbors`: `list[NeighborObs]` for selected top-k neighbors
- `obstacles`: `list[AABBObs]` for static world obstacles from scenario geometry
- `neighbor_intents`: optional `list[IntentObs]` for same selected neighbors
- `messages`: optional `list[AgentMessageObs]` delivered to this agent on the current tick
- `agent_context`: optional `AgentContext` with `agent_id`, method name, seed, role, priority, capabilities, mission, failure modes, and persistent `memory`
- `dt`: tick duration in seconds
- `t`: simulation time in seconds

`NeighborObs` fields (all world-frame, SI units):
- `idx: int` neighbor agent id
- `pos: np.ndarray (3,)` absolute world position (m)
- `vel: np.ndarray (3,)` absolute world velocity (m/s)
- `radius: float` collision radius (m)
- `msg_age_sec: float` age of last received odom message (s), clamped by `comm.age_cap_s`
- `valid: bool` whether message passed comm delivery constraints

Note on `valid=False`:
- In the planner path, invalid neighbors are filtered out by centralized neighbor selection and are not passed in `PlannerInput.neighbors`.
- In fixed-size dataset tensors (`cond_nbh`), missing neighbor slots are padded with zeros and `valid=0`.
- `msg_age` for missing/invalid messages is clamped to age cap in comm handling.

`msg_age_sec` semantics for included neighbors:
- computed as `now_s - msg.timestamp_send_s`
- clamped consistently by `comm.age_cap_s`

Neighbor ordering guarantee:
- neighbors are filtered and ranked centrally (shared for all planners)
- order is ascending by threat score (`ttc` or distance, config-dependent)
- ties are stable with deterministic sender-id iteration for fixed seed/config

`AABBObs` fields:
- `center: np.ndarray (3,)` world-frame box center
- `half: np.ndarray (3,)` world-frame half-extents

Obstacle semantics:
- obstacles are static scenario geometry, not sensed V2V data
- they are passed exactly as configured in scenario world coordinates
- planners should treat them as hard constraints for local avoidance

## 4) Repository Layout

```text
microbench/
  core/                 # dynamics, collisions, neighbor selection
  comm/                 # V2V emulator and comm impairment behavior
  scenarios/            # scenario loading, spawn/goal generation, timed events
  planners/             # planner plugins (baseline, ORCA, template)
  metrics/              # episode metrics, CSV writer, failure recorder, ring buffer
  replay/               # Foxglove MCAP export and episode analysis reports
  dataset/              # diffusion dataset generation + shard sanity checks
  tools/                # utilities (hard-case miner, reports, golden checks)
config/
  defaults.yaml         # global defaults
  comm_profiles.yaml    # named V2V profiles
  scenarios/*.yaml      # benchmark scenarios
scripts/
  ci_sanity.sh          # quick CI-style execution sanity check
  package_smoke.sh      # wheel/install smoke check from outside the checkout
  release_readiness.sh  # public-alpha release dry run
docs/
  README.md             # documentation map
  DESIGN_V1.md          # public benchmark contract
  PLANNER_API.md        # planner tutorial and API guide
examples/
  simple_external_planner.py
```

## 5) Configuration Layers

Configuration precedence is:
1. `config/defaults.yaml`
2. `config/scenarios/<scenario>.yaml` overrides defaults
3. CLI flags override run/sweep fields

| Item | Where | Notes |
|---|---|---|
| dt / duration | `defaults.yaml`, scenario YAML | Default `dt_s=0.02`; scenarios can override duration |
| v_max / a_max | defaults + scenario `agent_params` | Official defaults: `v_max=3.0`, `a_max=2.0` unless scenario overrides |
| top_k / range | `defaults.yaml` / scenario overrides | Keep identical across methods for fair comparison |
| comm profile | `config/comm_profiles.yaml` + `--comm` | Used by V2V emulator |
| intent channel | `defaults.yaml -> intent.*` | Optional trajectory-sharing channel |
| min goal distance | scenario `goals.min_goal_distance_m` | Enforced during episode init |
| trace logging | `defaults.yaml -> logging.*` | Controls events and collision-trace output |

Enabling the intent channel:

```yaml
intent:
  enabled: true
  tx_rate_hz: 20
  max_points: 20
```

Note: `intent.tx_rate_hz` is independent from odometry `tx_rate_hz`. Delay/loss still follows the active comm profile.

## 6) Official Evaluation Protocol (Pre-v1)

Generated official suites are the preferred path for new comparisons because they carry scenario metadata, recommended run matrices, and both 2D and 3D coverage.

See [docs/SCENARIO_SUITES.md](docs/SCENARIO_SUITES.md) or run:

```bash
python -m microbench.cli list-suites
python -m microbench.cli list-suites --json
```

Run the fast generated smoke suite:

```bash
python -m microbench.cli canonical-sweep \
  --suite official_smoke_generated \
  --out-dir runs_official_smoke_generated
```

Materialize an official suite without running it:

```bash
python -m microbench.cli materialize-suite \
  --suite official_alpha \
  --out-dir generated_official_alpha \
  --print-plan
```

Run the generated mixed 2D/3D alpha suite:

```bash
python -m microbench.cli canonical-sweep \
  --suite official_alpha \
  --methods baseline_goal,orca_heuristic \
  --out-dir runs_official_alpha \
  --max-runs 6
```

Run the generated 3D stress suite:

```bash
python -m microbench.cli canonical-sweep \
  --suite official_3d_stress \
  --methods orca_heuristic \
  --out-dir runs_official_3d_stress \
  --max-runs 3
```

Compare the ORCA-like baseline with its stale-aware preset:

```bash
python -m microbench.cli canonical-sweep \
  --suite official_3d_stress \
  --methods orca_heuristic,orca_with_staleness \
  --out-dir runs_orca_staleness_comparison \
  --max-runs 6
```

Run the generated experimental-baseline calibration suite:

```bash
python -m microbench.cli canonical-sweep \
  --suite official_experimental_baselines \
  --out-dir runs_experimental_baselines
```

Run the compact promotion-calibration suite for 3D and degraded sensing/communication evidence:

```bash
python -m microbench.cli canonical-sweep \
  --suite official_promotion_calibration \
  --out-dir runs_promotion_calibration
```

Build a compact baseline comparison report:

```bash
python -m microbench.cli baseline-report \
  --summary runs_experimental_baselines/summary.csv \
  --results runs_experimental_baselines/results.csv \
  --suite official_experimental_baselines \
  --out runs_experimental_baselines/baseline_report.json
```

Run a compact shared 3D advanced-baseline comparison:

```bash
python -m microbench.cli advanced-baseline-comparison \
  --out-dir runs_advanced_baseline_comparison \
  --require-pass
```

This runs ORCA, stale-aware ORCA, CBF-QP, MPC-local, nonlinear MPC, distributed MPC, BVC tube-DMPC, dynamic tube-DMPC, RMADER, sampled EGO-Swarm-inspired trajectory sharing, optimized EGO-Swarm-style control-point planning, VO, and RVO on the same `urban_conflict_3d` lane and writes `advanced_baseline_comparison.json`, `baseline_report.json`, `results.csv`, and `summary.csv`. Use it as a quick apples-to-apples advanced-baseline artifact; the all-suite leaderboard remains the publication-grade benchmark evidence.

Add `--export-foxglove-mcap` to save per-method traces and write a single panelized `baseline_comparison.mcap` alongside the CSV/JSON artifacts. For an environment-rich throughput demo, pass `--scenario config/scenarios/urban_throughput_3d.yaml`, the optimizer methods shown in the featured demo command, and `--planner-preset scale` when the visual run should match scale-tuned optimizer settings. Omit the preset when inspecting fuller default optimizer behavior.

Run a fleet-size scaling ladder:

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

This writes raw `results.csv`, standard `summary.csv`, `scale_summary.csv`, and `scale_benchmark.json`. The scale report separates safety from runtime feasibility: a method that times out at high N is recorded as a failed scale row, not silently dropped. It also reports final goal distance and progress fraction, so unfinished dense-fleet rows can still be judged for mission progress. `--planner-preset scale` applies documented compute-budget overrides for large-fleet stress tests; for NMPC/DMPC/BVC/RMADER-style optimizers, this includes bounded local traffic/constraint budgets plus command-level velocity/yield guarding; for RMADER, it also includes nearby dynamic-hull filtering, true seed-budget capping, and kinematic cached-plan validation between replans; for `ego_swarm_opt`, it uses receding cached-plan reuse between 5 Hz trajectory optimizations plus a command-level velocity/yield guard. Omit the preset for the fuller default optimizer settings.

For `urban_throughput_3d`, use `--duration-s 30` for a practical high-N diagnostic and `--duration-s 60` or longer for deliberate scale-evidence runs. Dense urban-throughput rows are intentionally heavier than smoke checks because they measure both safe deconfliction and mission progress under merge pressure.

Run the default high-volume evidence package when you want one reproducible folder with the scale run, scale summary, leaderboard JSON/CSV, and manifest:

```bash
python -m microbench.cli high-volume-evidence \
  --out-dir runs_high_volume_evidence \
  --require-pass
```

The default evidence matrix is N=30 on `stacked_swap_3d`, `urban_conflict_3d`, and `urban_throughput_3d` with the advanced optimizer baseline set, dense spawning, realistic V2V, 30-second episodes, and the scale planner preset. Use `--plan-only` first to inspect the prepared matrix without launching runs, and use `--resume` to continue a partial evidence folder.

Build a multi-axis high-volume leaderboard from one or more scale summaries:

```bash
python -m microbench.cli high-volume-leaderboard \
  --scale-summary runs_scale_benchmark/scale_summary.csv \
  --out runs_scale_benchmark/high_volume_leaderboard.json
```

This writes `high_volume_leaderboard.json` and a sibling CSV with separate safety, mission-progress, runtime/scale, and robustness scores. The overall score is a weighted composite, so publish the component scores with any headline rank. When multiple summaries include the same scenario/method/comm/N cell, rows with mission-progress fields are preferred.

Run the generated agentic stress suite:

```bash
python -m microbench.cli canonical-sweep \
  --suite official_agentic_stress \
  --methods priority_yield,negotiation_yield \
  --out-dir runs_official_agentic_stress \
  --max-runs 3
```

Generated scenario files and `suite_manifest.yaml` are saved under `<out-dir>/_generated_scenarios/<suite>/` so result folders are self-describing.

Generated manifests include an `acceptance` block with pre-v1 baseline rule metadata. Check it against a run with:

```bash
python -m microbench.cli check-acceptance \
  --summary runs_official_smoke_generated/summary.csv \
  --results runs_official_smoke_generated/results.csv \
  --suite-manifest runs_official_smoke_generated/_generated_scenarios/official_smoke_generated/suite_manifest.yaml
```

The generated smoke suite currently has calibrated checks for baseline runtime, ORCA runtime, priority-yield message delivery, and zero planner guardrail events. A path-independent expected report is kept at `golden/acceptance/official_smoke_generated_acceptance.json`. `official_promotion_calibration` adds compact acceptance bands for candidate baseline 3D stress behavior, degraded fused-sensing diagnostics, stale-observation signal, runtime, and guardrail counts; `baseline-promotion` runs it automatically.

If you only ran a subset of methods, filter the rule set:

```bash
python -m microbench.cli check-acceptance \
  --summary runs_official_smoke_generated/summary.csv \
  --suite-manifest runs_official_smoke_generated/_generated_scenarios/official_smoke_generated/suite_manifest.yaml \
  --methods baseline_goal
```

Validate built-in and generated suites:

```bash
python -m microbench.cli validate-scenarios \
  --all-builtins \
  --all-generated-suites
```

Validation checks required scenario keys, supported spawn/perception schemas, AABB obstacle geometry, suite manifests, and 3D-specific constraints such as `world.planar: false`, nonzero vertical volume, and spawn/goal bounds.

Legacy hand-written canonical suites remain available while the official generator matures.

Required scenarios:
- `corridor.yaml`
- `intersection.yaml`
- `funnel.yaml`
- `ring.yaml`
- `crowd_swap.yaml`
- `weather_event.yaml`

Primary suite:
- `N = [10, 20, 50]`
- seeds `0..49`
- comm profiles: `ideal_50hz`, `realistic_v2v_50hz`, `degraded_20hz`

Stretch suite (optional, after primary):
- `N = [100]`
- seeds `0..99`
- optional stress comm profile: `bursty_stress_50hz`

Run primary canonical suite:

```bash
python -m microbench.cli canonical-sweep \
  --suite primary \
  --methods baseline_goal,orca_heuristic \
  --out-dir runs_primary_smoke \
  --max-runs 6
```

Run baseline sanity suite:

```bash
python -m microbench.cli canonical-sweep \
  --suite baseline_sanity \
  --methods baseline_goal,orca_heuristic \
  --out-dir runs_baseline_sanity
```

Expected baseline sanity behavior:
- `baseline_goal`: high collisions in dense scenarios.
- `orca_heuristic`: substantially lower collisions, especially in ideal comm.
- `orca_with_staleness`: more conservative behavior when observations are stale or degraded.
- `cbf_qp`: experimental CBF safety-filter baseline with deterministic projection, optional SciPy solver mode, obstacle barriers, and stale-track inflation; not a calibrated leaderboard anchor.
- `mpc_local`: experimental local predictive sampling baseline with bounded lookahead, smoothness costs, obstacle scoring, candidate-risk diagnostics, and stale-track risk inflation.
- `mpc_nonlinear`: experimental clean-room nonlinear MPC trajectory-optimization baseline with multistart acceleration-control optimization, dynamic obstacle/intent penalties, command-level velocity/yield guarding, and trajectory intent output.
- `dmpc_best_response`: experimental distributed-MPC-style best-response baseline that treats neighbor intent trajectories as coupled constraints, falls back to inflated constant-velocity tracks for stale/missing plans, and republishes its plan for the next coordination round.
- `bvc_tube_dmpc`: experimental clean-room tube-based DMPC baseline with hard buffered Voronoi-cell halfspace tubes, waypoint projection into non-overlapping cells, obstacle halfspaces, and trajectory intent output.
- `dynamic_tube_dmpc`: experimental paper-specific dynamic tube-DMPC baseline after Dai/Liao/Chen 2026, with condensed acceleration QP, elastic virtual-tube reconstruction, risk-triggered collision constraints, local tube halfspaces, and trajectory intent output.
- `rmader`: experimental clean-room RMADER/MADER-style baseline with cubic B-spline plans, continuous MINVO interval polyhedra, hard separating hyperplanes, delay-check publication, and trajectory intent output.
- `ego_swarm`: experimental clean-room EGO-Swarm-inspired trajectory-sharing baseline with smooth receding-horizon topologies and intent output; not a port of the upstream GPL implementation.
- `ego_swarm_opt`: experimental clean-room EGO-Swarm-style optimized control-point trajectory baseline with topological seeds, smoothness/dynamic penalties, obstacle/swarm penalties, and intent output.

Quick acceptance heuristic:
- ORCA collision counts should be at least ~5x lower than baseline in `intersection` and `funnel` under `ideal_50hz`.

Perception stress suite:

```bash
python -m microbench.cli canonical-sweep \
  --suite perception_stress \
  --methods priority_yield \
  --out-dir runs_perception_stress \
  --max-runs 4
```

This suite exercises sensor-only and fused V2V+sensor observations with FOV limits, noise, false negatives, and occlusion.

## 7) 3D Guide

Use this section for non-planar planner work. 3D is first-class in DAA Microbench: `official_alpha` includes 3D cases, and `official_3d_stress` focuses on volumetric, vertical, and partial-sensing stress tests.

Quick navigation:

| If You Want To... | Go Here |
|---|---|
| run one 3D episode | Section `7.2 Quick 3D Run` |
| benchmark a planner in generated 3D stress cases | Section `7.3 Official 3D Stress Suite` |
| inspect a 3D trace | Section `7.4 Foxglove and Episode Reports` |
| profile 3D planner runtime | Section `7.5 3D Profiling Notes` |
| generate 3D diffusion data | Section `7.6 3D Diffusion Data` |

### 7.1 3D Scenarios

Built-in 3D scenarios:
- `config/scenarios/stacked_swap_3d.yaml`
- `config/scenarios/layered_funnel_3d.yaml`
- `config/scenarios/layered_intersection_3d.yaml`
- `config/scenarios/weather_vertical_event_3d.yaml`
- `config/scenarios/vertical_crossing_obstacles_3d.yaml`
- `config/scenarios/urban_airspace_3d.yaml`
- `config/scenarios/urban_conflict_3d.yaml`
- `config/scenarios/urban_throughput_3d.yaml`

Generated 3D family scenarios:
- `sphere_swap_3d_medium`: true volumetric antipodal swap through shared airspace
- `merge_3d_hard`: converging 3D streams into a constrained exit volume
- `overtake_3d_medium`: same-direction 3D corridor traffic with heterogeneous speeds
- `vertical_crossing_3d_hard`: layer-changing crossing around a central obstruction
- `noncooperative_intruder_3d_hard`: sensor-driven encounter with a noncooperative intruder
- `heterogeneous_priority_crossing_3d_medium`: mixed priority/capability crossing with altitude changes
- `sensor_volume_3d_hard`: volumetric fused-perception stress case with stale local tracks

What they are for:
- `stacked_swap_3d`: simple non-planar swap through shared volume
- `layered_funnel_3d`: funnel/gate pressure with mixed target altitude layers
- `layered_intersection_3d`: crossing flows with layered targets
- `weather_vertical_event_3d`: forced climb/descent event
- `vertical_crossing_obstacles_3d`: crossing traffic plus center obstacle
- `urban_airspace_3d`: environment-rich urban airspace with buildings, occlusion, static obstacle proximity, and layered goals
- `urban_conflict_3d`: near-coplanar urban crossing where straight-line goal seeking collides and avoidance baselines must actively deconflict
- `urban_throughput_3d`: asymmetric urban crossing/merge flow for measuring high-density mission progress without perfect four-way gridlock

### 7.2 Quick 3D Run

```bash
python -m microbench.cli run \
  --scenario config/scenarios/stacked_swap_3d.yaml \
  --method orca_heuristic \
  --n 10 \
  --seed 0 \
  --comm ideal_50hz \
  --out-dir runs_3d_example
```

Environment-rich visualization run:

```bash
python -m microbench.cli run \
  --scenario config/scenarios/urban_conflict_3d.yaml \
  --method reciprocal_velocity_obstacle \
  --n 4 \
  --seed 2 \
  --comm realistic_v2v_50hz \
  --save-trace \
  --out-dir runs_urban_conflict
```

### 7.3 Official 3D Stress Suite

Use this generated suite for serious non-planar comparisons:

```bash
python -m microbench.cli canonical-sweep \
  --suite official_3d_stress \
  --methods your_method \
  --out-dir runs_official_3d_stress
```

Defaults:
- scenarios: generated `sphere_swap_3d_medium`, `merge_3d_hard`, `overtake_3d_medium`, `vertical_crossing_3d_hard`, `sensor_volume_3d_hard`, `noncooperative_intruder_3d_hard`, and `heterogeneous_priority_crossing_3d_medium`
- methods: `orca_heuristic`, `orca_with_staleness`
- `N = [6, 10]`
- seeds `0..9`
- comm: `ideal_50hz`, `realistic_v2v_50hz`, `degraded_20hz`

Stretch mode:
- adds `N = 20`
- extends seeds to `0..29`

The older built-in 3D suite is still useful for development:


```bash
python -m microbench.cli canonical-sweep \
  --suite three_d \
  --methods your_method \
  --out-dir runs_three_d
```

Defaults:
- scenarios: all 5 built-in 3D scenarios above
- `N = [6, 10]`
- seeds `0..9`
- comm: `ideal_50hz`

Stretch mode:
- adds `N = 20`
- extends seeds to `0..19`

If `--methods` is omitted for `three_d`, it defaults to `orca_heuristic`.

### 7.4 Foxglove and Episode Reports

- for lab-grade robotics visualization, export traces to Foxglove MCAP and open them in Foxglove Studio
- for deeper debugging, start with the multi-panel episode report: it shows top-down, side/altitude, 3D context, separation, speed, saturation, and sensing freshness in one synchronized HTML artifact

Foxglove MCAP export:

```bash
python -m microbench.cli run \
  --scenario config/scenarios/stacked_swap_3d.yaml \
  --method your_method \
  --n 10 \
  --seed 0 \
  --comm ideal_50hz \
  --save-trace \
  --out-dir runs/<run_id>

python -m microbench.cli foxglove-export \
  --trace runs/<run_id>/episodes/<episode_dir>/trace_episode.jsonl \
  --out runs/<run_id>/episode.mcap \
  --compression zstd
```

Install `daa-microbench[foxglove]` to enable MCAP writing. The default `--compression zstd` keeps longer episode logs practical; pass `--compression none` only when debugging raw MCAP bytes. The export writes Foxglove-recognized channels for `/tf`, `/daa/static`, `/daa/agents`, `/daa/trails`, `/daa/sensing_links`, `/daa/intents`, `/daa/perception`, `/daa/diagnostics`, and `/daa/events`. `/daa/static` includes the operational volume, obstacles/buildings, roads/ground, altitude-layer guides, start/goal markers, and goal tolerance zones when that metadata is available. `/daa/perception` shows optional sensor/range volumes for sensor or fused-perception scenarios. `/daa/trails` shows recent executed history; `/daa/intents` shows future advertised trajectories when the trace contains intent messages. Sensing-link colors encode freshness: green is fresh, orange/yellow is moderately stale, red is stale/expired, and gray means no age was available. DAA Microbench stores altitude on the native `y` axis; the Foxglove export maps coordinates to `x, y=lateral, z=altitude` so the 3D panel is z-up.

For side-by-side baseline review, combine multiple saved traces into one MCAP with method-scoped topics:

```bash
python -m microbench.cli foxglove-comparison-export \
  --trace mpc_nonlinear=runs/<run_id>/episodes/<mpc_episode>/trace_episode.jsonl \
  --trace ego_swarm_opt=runs/<run_id>/episodes/<ego_episode>/trace_episode.jsonl \
  --out runs/<run_id>/baseline_comparison.mcap
```

The comparison MCAP keeps a shared `/tf` stream with namespaced frame ids and writes per-method topics under `/daa/comparison/<method>/...`, so Foxglove panels can subscribe to one method at a time.

The `advanced-baseline-comparison` command can generate the same combined MCAP directly:

```bash
python -m microbench.cli advanced-baseline-comparison \
  --out-dir runs_urban_throughput_comparison \
  --scenario config/scenarios/urban_throughput_3d.yaml \
  --methods dmpc_best_response,bvc_tube_dmpc,dynamic_tube_dmpc,mpc_nonlinear,ego_swarm_opt,rmader \
  --n 8 \
  --duration-s 20 \
  --planner-preset scale \
  --export-foxglove-mcap
```

The command writes `advanced_baseline_comparison.json` and `comparison_manifest.json`; the manifest is the durable record for reproducing or interpreting a copied MCAP.

Episode analysis report:

```bash
python -m microbench.cli episode-report \
  --trace runs/<run_id>/episodes/<episode_dir>/trace_episode.jsonl \
  --out runs/<run_id>/episode_report.html
```

Install `daa-microbench[viz]` and pass `--plotly-source inline` when you want a fully self-contained HTML artifact with Plotly embedded in the file. The default `auto` mode embeds Plotly when it is installed and otherwise falls back to the Plotly CDN.

What the episode report adds:
- synchronized top-down x-z and side x-y projections
- a supporting 3D context panel rather than only a 3D camera view
- nearest-pair and focus-pair separation curves
- speed/command magnitude, control saturation, and V2V/sensing freshness plots
- event markers for collision and near-miss traces when `events.jsonl` is present

### 7.5 3D Profiling Notes

- obstacle-aware `orca_heuristic` is heavier than `baseline_goal`
- `mpc_local` is deliberately bounded but heavier than reactive baselines because it scores short-horizon rollouts
- keep `official_experimental_baselines` separate from CI smoke when profiling slow experimental methods
- use `official_promotion_calibration` for compact candidate-baseline 3D/degraded evidence before attempting longer stress sweeps
- expect a few milliseconds per tick per agent on harder 3D scenes
- use `results.csv` / `summary.csv` to compare:
  - `planner_ms_per_tick_per_agent_mean`
  - `planner_ms_per_tick_per_agent_p95`
  - `planner_timeout_count`
  - `planner_error_count`
  - `planner_fallback_count`
  - `episode_runtime_s`

Quick profiling command:

```bash
python -m microbench.cli run \
  --scenario config/scenarios/layered_intersection_3d.yaml \
  --method orca_heuristic \
  --n 6 \
  --seed 0 \
  --comm ideal_50hz \
  --out-dir runs_profile_orca_3d
```

### 7.6 3D Diffusion Data

- `generate-dataset` works on non-planar scenarios
- non-planar conditioning is stored in full 3D world-frame
- targets remain `U0_raw: (B, T, 3)`
- use `--quality-filter safe_expert` for cleaner expert labels on harder 3D scenes

Example:

```bash
python -m microbench.cli generate-dataset \
  --scenario config/scenarios/layered_intersection_3d.yaml \
  --method orca_heuristic \
  --n 6 \
  --seeds 0:0 \
  --comm ideal_50hz \
  --T 20 \
  --dt-plan-s 0.10 \
  --quality-filter safe_expert \
  --filter-min-sep-m 0.2 \
  --out-dir datasets_3d_orca
```

## 8) Weights & Biases Logging Guide

Use W&B for sweep-level run tracking (not per-tick logging).

Quick start:

```bash
python -m microbench.cli canonical-sweep \
  --suite baseline_sanity \
  --methods baseline_goal,orca_heuristic \
  --out-dir runs_baseline_sanity_wandb \
  --wandb \
  --wandb-project daa-microbench \
  --wandb-mode offline
```

Supported flags (`sweep` and `canonical-sweep`):
- `--wandb`
- `--wandb-project`
- `--wandb-entity`
- `--wandb-group`
- `--wandb-name`
- `--wandb-tags`
- `--wandb-mode {online,offline,disabled}`
- `--wandb-upload-results/--no-wandb-upload-results`
- `--wandb-upload-traces/--no-wandb-upload-traces`
- `--wandb-upload-replays/--no-wandb-upload-replays`
- `--wandb-upload-dataset/--no-wandb-upload-dataset`

`baseline-leaderboard` also supports the same W&B flags:

```bash
python -m microbench.cli baseline-leaderboard \
  --out-dir runs_baseline_leaderboard \
  --suites all \
  --require-pass \
  --require-complete \
  --wandb \
  --wandb-project daa-microbench
```

For leaderboard runs, W&B receives dashboard-friendly Tables for aggregate ranking, suite status, suite-method summaries, and component rows. The local JSON/CSV artifacts remain the canonical benchmark record; W&B is an optional comparison and sharing layer.

What gets logged:
- run config (methods/scenarios/N/seeds/comm, sim params, platform/python, git hash if available)
- overall summary scalars from `summary.csv`
- `leaderboard` table with full `summary.csv`
- optional `top_failures` table from `worst_cases/index.csv` when present

Artifacts:
- default: one results artifact containing `summary.csv`, `results.csv`, and `result_schema.json`
- optional: traces and replays from `worst_cases/` when corresponding upload flags are enabled

Failure safety:
- without `--wandb`, no W&B import or overhead
- W&B import/init/logging errors are warnings only; sweeps continue

## 9) Implementing a Planner (Plugin Guide)

### 9.1 Where to Put Code

- Create `microbench/planners/<name>.py`.
- Register planner in `microbench/planners/__init__.py` factory mapping.

### 9.2 Planner API Contract

- `reset(seed)` called once per planner instance at episode start.
- Each simulated drone receives its own planner object, so instance attributes are safe for per-agent memory.
- Planners may also store persistent per-agent state in `planner_input.agent_context.memory`.
- For heterogeneous episodes, pass `--agent-methods` with either one method or exactly `N` comma-separated methods.
- `compute_cmd(planner_input) -> np.ndarray (3,)` (for v_cmd output only) or `PlannerOutput(v_cmd, intent_out, messages_out)`.
- Inputs include `ego`, `goal_dir`, `neighbors`, `neighbor_intents`, `dt`, `t`.
- Static AABB obstacles are available as `planner_input.obstacles`.
- In planar mode, `y` is effectively fixed by sim, but command is still `(3,)`.
- Returned commands must be finite shape `(3,)`. Exceptions, non-finite commands, and malformed command shapes are counted in planner guardrail metrics and replaced with the deterministic engine fallback.

When intent is enabled, `PlannerInput.neighbor_intents` may contain:
- `sender_id`
- `intent_age_s` (seconds since sent; clamped)
- `valid` (false if missing or expired)
- `kind` (for example `COMMITTED` / `PROPOSED`)
- `expiry_s`
- `tube_radius_m`
- `points` polyline `(M,3)` in world coordinates

Intent selection policy:
- `neighbor_intents` uses the exact same neighbor IDs and ordering as `neighbors` (same top-k odom selection).
- each entry carries its own `valid` flag (for missing/expired intent), so planners can gate behavior without losing alignment.

Planner output intent behavior:
- return `PlannerOutput(v_cmd=..., intent_out=<IntentMsg>)` to publish intent
- set `intent_out=None` (or return just `np.ndarray`) to publish nothing
- harness throttles broadcasting to `intent.tx_rate_hz`

Minimal skeleton:

```python
from __future__ import annotations

import numpy as np
from microbench.planners.base import ILocalPlanner
from microbench.types import PlannerInput

class MyPlanner(ILocalPlanner):
    def reset(self, seed: int) -> None:
        _ = seed

    def compute_cmd(self, planner_input: PlannerInput) -> np.ndarray:
        ego = planner_input.ego
        goal_dir = planner_input.goal_dir
        neighbors = planner_input.neighbors  # use exactly as passed
        obstacles = planner_input.obstacles  # static AABB obstacles in world frame
        _ = neighbors, obstacles
        return np.asarray(goal_dir, dtype=np.float32) * float(ego.v_max)
```

Minimal no-intent explicit output:

```python
from microbench.types import PlannerOutput

def compute_cmd(self, planner_input):
    v_cmd = planner_input.goal_dir * planner_input.ego.v_max
    return PlannerOutput(v_cmd=v_cmd, intent_out=None)
```

### 9.3 Intent Message Schema

`IntentMsg` fields:
- `sender_id`
- `timestamp_send_s`
- `expiry_s`
- `kind` (free string: `COMMITTED` / `PROPOSED` etc.)
- `tube_radius_m`
- `points` polyline `(M,3)` in world coordinates (planar keeps fixed `y`)

Interpretation:
- sender intends to occupy a tube/corridor centered on `points` with radius `tube_radius_m` until `expiry_s`.

### 9.4 What Not to Change

- Do not use ground-truth neighbor state unavailable to planners.
- Do not alter shared neighbor settings to favor one method.
- Do not change global sim timestep for method-specific runs.
- Do not modify collision threshold definitions for comparisons.

## 10) Recorder, Replay, and Debugging Collisions

Per-episode artifacts are written under:
- `runs/<run_id>/episodes/<scenario>_<method>_n<N>_seed<S>[_comm_<profile>]/`

Possible files:
- `events.jsonl`: collision and near-miss events.
- `trace_collision_<i>_<j>_t<time>.jsonl`: ring-buffer trace around collision.
- `trace_episode.jsonl`: full-episode trace when `logging.save_trace: true`

Enable recorder outputs in `config/defaults.yaml` when needed:

```yaml
logging:
  save_trace: true
  trace_save_failures_only: false
  save_events: true
  save_trace_on_collision: true
  trace_window_s: 3.0
```

Render a full-episode analysis report:

```bash
python -m microbench.cli episode-report \
  --trace runs/<run_id>/episodes/<episode_dir>/trace_episode.jsonl \
  --out runs/<run_id>/episode_report.html
```

Mine hard cases from `results.csv`:

```bash
python -m microbench.cli mine-hard-cases \
  --results runs_readme_baseline/results.csv \
  --top-k 20
```

This creates `runs_readme_baseline/worst_cases/` with ranked episode folders and copied trace/event artifacts.

Debugging intent-related failures:
- collision window traces include last received pairwise intent state (presence/age/valid/kind/expiry) for the collision pair
- use this to diagnose stale/missing intents, expired commitments, or mismatched tube radius/horizon

## 11) Diffusion Dataset Generation

`orca_heuristic` can be used to generate heuristic training labels. It is an ORCA-like reference baseline, not an expert oracle.

Generate dataset shards:

```bash
python -m microbench.cli generate-dataset \
  --scenario config/scenarios/intersection.yaml \
  --method orca_heuristic \
  --n 10 \
  --seeds 0:1 \
  --comm ideal_50hz,realistic_v2v_50hz \
  --T 20 \
  --dt-plan-s 0.10 \
  --quality-filter safe_expert \
  --filter-min-sep-m 0.2 \
  --out-dir datasets/orca_heuristic_v0
```

Sanity-check one shard:

```bash
python -m microbench.cli sanity-check-dataset \
  --shard datasets/orca_heuristic_v0/orca_heuristic/intersection/ideal_50hz/shard_00000.npz
```

Stored fields include:
- `cond_ego`, `cond_goal`, `cond_nbh`, optional `cond_evt`, `cond_flat`
- `U0_raw` and normalized `U0`
- metadata arrays (`scenario_id`, `comm_profile_id`, `N_agents`, `seed`, `t_sec`, `ego_id`, etc.)
- dataset-level constants in shard and `dataset_manifest.json`

Dataset quality filtering:
- `--quality-filter none`: keep all rollout samples
- `--quality-filter collision_free`: drop samples with `collision_in_next_H == 1`
- `--quality-filter safe_expert`: drop horizon-collision samples and require `min_sep_next_H >= --filter-min-sep-m`
- manifest records `quality_filter`, `filter_min_sep_m`, `num_samples_raw`, and `num_samples_kept`

Shapes and conventions:
- `cond_ego: (B, 6)`
- `cond_goal: (B, 4)`
- `cond_nbh: (B, k, 9)`
- `cond_evt: (B, evt_dim)` (if enabled)
- `U0_raw: (B, T, 3)` world-frame velocity commands (m/s)
- `U0: (B, T, 3)` normalized action targets

Normalization (current implementation):
- neighbor `rel_pos`: divided by `range_m`
- neighbor `rel_vel` and `ego_vel`: divided by `v_max_ego`
- radius: divided by `r_ref`
- `msg_age`: divided by `age_cap_s` and clamped to `[0,1]`
- `goal_dist`: divided by `goal_dist_cap` and clamped to `[0,1]`
- `U0`: `clip(U0_raw / v_max_ego, -1, 1)`

Dataset schema source of truth:
- implementation: `microbench/dataset/generate.py`
- manifest: `dataset_manifest.json` stores `k`, `T`, `dt_plan_s`, feature dims, and normalization constants for each dataset folder.

## 11.1) Learned Policy Dataset Export

`learned-dataset-export` writes public RL observation/action samples for imitation learning, offline RL experiments, and replayable learned-policy debugging. It uses the same PettingZoo-style observation/action contract as `rl-smoke` and `rl-validation-matrix`.

```bash
python -m microbench.cli learned-dataset-export \
  --out-dir runs_learned_dataset \
  --lanes head_on,crossing,urban_obstacle \
  --max-steps 64 \
  --save-replay \
  --require-pass
```

By default the action source is the transparent `bc_teacher` used by `train-learned-bc`. You can also pass a built-in RL policy with `--policy` or an external exported policy with `--policy-spec`.

The output directory contains:
- `learned_dataset_manifest.json`
- `learned_dataset_episodes.csv`
- compressed `shards/shard_*.npz`
- optional lightweight `replay/*.jsonl` files when `--save-replay` is set

Shard arrays include `observations`, `actions`, `next_observations`, `rewards`, `terminated`, `truncated`, `done`, lane/scenario/seed metadata, collision/near-miss flags, and per-sample clearance.

Recommended scale for training: at least `100k+` samples across multiple scenarios and comm profiles.

## 11.2) Learned Hard-Lane Loop

`learned-hard-lane-loop` closes the development loop from diagnostics to retraining. It reads `learned-diagnostics` rows or diagnoses one or more bundles, selects canonical hard lanes such as `head_on`, `crossing`, `urban_obstacle`, `communication_delay`, or `high_n_dense_merge`, exports public RL observation/action shards, trains the portable BC MLP from the shards, then reruns learned-submission bundle evidence, leaderboard, and diagnostics. Use `--mix-lanes` to add broad replay lanes beside the selected hard lanes so a focused fix does not overfit one scenario family. For dense-swarm learned-policy hard-negative mining, explicitly add the training-only lane `dense_swarm_hard_negative` and use `--sample-selection hard_negative_windows` so only close-approach windows from that lane are fitted.

```bash
python -m microbench.cli learned-hard-lane-loop \
  --out-dir runs_hard_lane_loop \
  --diagnostics runs_learned_diagnostics/learned_policy_diagnostics.json \
  --target-policy bc_mlp_learned \
  --fallback-lanes urban_obstacle,communication_delay,high_n_dense_merge \
  --mix-lanes head_on,crossing,urban_obstacle,communication_delay,high_n_dense_merge,dense_swarm_hard_negative \
  --sample-weighting safety \
  --sample-selection hard_negative_windows \
  --max-lanes 3 \
  --max-runs 1 \
  --require-pass
```

The top-level `learned_hard_lane_loop.json` records the selected lanes, mixed dataset lanes, dataset manifest, training report, policy spec, bundle paths, learned leaderboard, and final diagnostics. New BC artifacts store per-feature mean/std normalization by default; pass `--feature-normalization none` only for ablations. `--sample-weighting safety` gives collision, near-miss, and low-clearance shard samples more influence in the supervised fit while logging the exact weighting recipe in the model artifact and manifest overlay. `--sample-selection hard_negative_windows` keeps all non-hard lanes but filters configured hard-negative lanes down to collision, near-miss, low-clearance, or closest-approach windows; tune `--sample-selection-clearance-threshold-m` and `--sample-selection-context-steps` for dense-swarm replay studies. Use this for development iteration; final learned-policy claims should still keep the uncapped bundle artifacts and training disclosure.

## 11.3) Closed-Loop Learned Fine-Tuning

`learned-closed-loop-finetune` is the first rollout-aware learned-policy optimizer. It starts from an existing portable `mlp_json` policy spec, perturbs either the MLP output head or all MLP layers, evaluates candidates through the PettingZoo-style `DaaParallelEnv`, and accepts candidates according to an explicit acceptance mode. The default `--acceptance-mode strict_feasible` accepts only candidates that improve the closed-loop objective without violating collision, clearance, or near-miss guardrails.

```bash
python -m microbench.cli learned-closed-loop-finetune \
  --out-dir runs_closed_loop_finetune \
  --base-policy-spec runs_bc_mlp_policy/policy_spec.json \
  --lanes head_on,crossing,urban_obstacle,communication_delay,high_n_dense_merge \
  --trainable-parameters all_layers \
  --search-strategy two_stage \
  --antithetic-sampling \
  --require-per-lane-safety \
  --per-lane-clearance-tolerance-m 0.01 \
  --generations 4 \
  --population-size 12 \
  --train-max-steps 24 \
  --eval-lanes crossing,urban_obstacle,communication_delay,high_n_dense_merge \
  --holdout-profile broad_3d_stress \
  --holdout-scenarios sphere_swap_3d_medium,dense_swarm_3d_hard,merge_3d_hard,sensor_volume_3d_hard,noncooperative_intruder_3d_hard \
  --holdout-seeds 0:2 \
  --holdout-comm ideal_50hz,degraded_20hz \
  --require-pass \
  --require-promotion
```

This writes `closed_loop_mlp_policy.json`, `policy_spec.json`, `candidate_summary.csv`, `candidate_lane_summary.csv`, `candidate_episodes.csv`, `closed_loop_training_report.json`, optional RL validation evidence, and, when enabled, `broad_3d_holdout/` with base-versus-tuned `results.csv`, `summary.csv`, `comparison_summary.csv`, and `comparison_report.json`. `--trainable-parameters output_head` is the conservative audit mode; `all_layers` is the stronger search mode for learned-baseline development. `--search-strategy two_stage` warms up the output head before all-layer refinement, `--antithetic-sampling` evaluates plus/minus perturbation pairs, and `--require-per-lane-safety` rejects candidates that hide a lane-level safety regression behind an aggregate score improvement. `--per-lane-clearance-tolerance-m` is recorded in the lane CSV and should stay small; it prevents centimeter-scale numerical jitter from blocking otherwise identical safety behavior. `--acceptance-mode safety_improvement` is intended for curriculum development: it may keep a non-promotable intermediate candidate only when score improves, lane safety does not regress, and aggregate collision, near-miss, or clearance behavior improves relative to the current parent. `--require-pass` checks structural/training gates; `--require-promotion` additionally requires final training feasibility plus broad 3D holdout safety, clearance, and `score_v0` non-regression. Treat it as benchmark-native closed-loop training infrastructure, not as a final SOTA RL algorithm yet.

Use `--lane-profile validation_plus_broad_3d` when you want closed-loop training to include the canonical validation lanes plus broad 3D stress training lanes (`sphere_swap_3d_training`, `dense_swarm_3d_training`, `merge_3d_training`, `sensor_volume_3d_training`, and `noncooperative_intruder_3d_training`). Explicit `--lanes` still overrides the profile.

## 11.4) Closed-Loop Learned Study Workflow

`learned-closed-loop-study` wraps the full learned-policy development loop into one auditable command: closed-loop fine-tuning, broad 3D holdout comparison, learned submission bundle, learned leaderboard, learned diagnostics, and a final promotion recommendation.

```bash
python -m microbench.cli learned-closed-loop-study \
  --out-dir runs_closed_loop_study \
  --base-policy-spec runs_bc_mlp_policy/policy_spec.json \
  --lane-profile validation_plus_broad_3d \
  --trainable-parameters all_layers \
  --search-strategy two_stage \
  --antithetic-sampling \
  --require-per-lane-safety \
  --per-lane-clearance-tolerance-m 0.01 \
  --generations 4 \
  --population-size 12 \
  --train-max-steps 24 \
  --holdout-profile broad_3d_stress \
  --holdout-scenarios sphere_swap_3d_medium,dense_swarm_3d_hard,merge_3d_hard,sensor_volume_3d_hard,noncooperative_intruder_3d_hard \
  --holdout-seeds 0:2 \
  --holdout-comm ideal_50hz,degraded_20hz \
  --comparison-bundle runs_bc_mlp_evidence/bc_bundle \
  --bundle-max-runs 1 \
  --require-pass \
  --require-promotion
```

The study folder contains `study_manifest.json`, `learned_closed_loop_study_report.json`, `training/`, `bundle/`, `learned_leaderboard.json/.csv`, and `learned_diagnostics.json/.csv/.md`. The nested training folder includes `candidate_lane_summary.csv` so reviewers can inspect each candidate by lane before trusting a promotion. `ok` means the workflow produced reviewable artifacts; `promotion_candidate` and `recommendation` (`promote`, `holdout_passed_no_update`, `holdout_review_required`, or `reject_regression`) carry the safety verdict. A no-update result means the base policy passed the holdout but the search did not find a better accepted candidate.

## 12) Performance Expectations and Practical Profiling

- Track planner cost from `results.csv` columns:
  - `planner_ms_per_tick_per_agent_mean`
  - `planner_ms_per_tick_per_agent_p95`
  - `planner_timeout_count`
  - `planner_error_count`
  - `planner_fallback_count`
- `planner_guardrails.timeout_ms` is a soft timeout: an over-budget call is counted and its returned output is replaced with a deterministic fallback after the call returns.
- `planner_guardrails.timeout_clock` defaults to `process_cpu`, so offline smoke tests ignore OS scheduling stalls while wall-clock planner latency is still reported; use `wall` for real-time deadline experiments.
- Planner exceptions and invalid outputs increment `planner_error_count`; timeouts increment `planner_timeout_count`; all guardrail replacements increment `planner_fallback_count`.
- The fallback command moves away from currently observed neighbors/obstacles at `planner_guardrails.fallback_speed_scale * v_max`; if no risk direction is available, it returns zero.
- Track safety using explicit collision semantics:
  - `collision_episode`: whether the episode had any collision
  - `unique_collision_pairs`: number of agent pairs that collided at least once
  - `collision_pair_ticks`: pair-tick count, useful for measuring collision duration/severity
  - `time_to_first_collision_s`: first collision time, `nan` when collision-free
- Track observation quality:
  - `obs_neighbors_mean`: average selected-neighbor count per agent tick
  - `obs_v2v_fraction` / `obs_sensor_fraction`: selected observation source mix
  - `obs_stale_fraction`: fraction of selected observations at the configured age cap
  - `obs_empty_fraction`: fraction of agent ticks with no selected neighbors
- Target is low single-digit milliseconds per tick per agent for scalable sweeps.
- For bottlenecks:
  - keep neighbor logic bounded by `top_k`
  - avoid heavy per-agent Python allocations in the hot loop
  - use NumPy-friendly operations for vector math

Quick local execution sanity (CI-style):

```bash
bash scripts/ci_sanity.sh
```

Public-alpha release dry run:

```bash
bash scripts/release_readiness.sh
```

Baseline behavior smoke:

```bash
python -m microbench.cli baseline-smoke --out-dir runs_baseline_smoke --require-pass
```

RL wrapper smoke:

```bash
python -m microbench.cli rl-smoke --out-dir runs_rl_smoke --require-pass
python -m microbench.cli rl-calibration --out-dir runs_rl_calibration --require-pass
python -m microbench.cli rl-validation-matrix --out-dir runs_rl_validation_matrix --policy tiny_learned --require-pass
python -m microbench.cli rl-smoke --out-dir runs_rl_mlp_learned --policy mlp_learned --require-pass
python -m microbench.cli learned-dataset-export --out-dir runs_learned_dataset --lanes head_on --max-steps 2 --require-pass
python -m microbench.cli train-learned-bc --out-dir runs_bc_mlp_policy --lanes head_on,crossing,urban_obstacle --eval-lanes head_on,crossing --require-pass
python -m microbench.cli learned-bc-evidence --out-dir runs_bc_mlp_evidence --lanes head_on,crossing,urban_obstacle --max-runs 1 --require-pass
python -m microbench.cli rl-contract --json
```

Experimental baseline promotion calibration:

```bash
python -m microbench.cli baseline-promotion --out-dir runs_baseline_promotion --require-calibrated
```

Optional longer stable-metadata review:

```bash
python -m microbench.cli baseline-validation-matrix --out-dir runs_baseline_validation_matrix --plan-only
python -m microbench.cli baseline-review --out-dir runs_baseline_review --plan-only
python -m microbench.cli baseline-review --out-dir runs_baseline_review --duration-s 20
```

Current result-schema golden check:

```bash
python -m microbench.cli golden-current-schema --golden-dir golden/current_schema
```

## 13) Contribution Workflow

Project governance and contribution docs:
- License: `LICENSE` (Apache-2.0)
- Contributing guide: `CONTRIBUTING.md`
- Documentation map: `docs/README.md`
- Public alpha release notes: `docs/PUBLIC_ALPHA_NOTES.md`
- Public alpha release checklist: `docs/RELEASE_CHECKLIST.md`
- Design contract: `docs/DESIGN_V1.md`
- Planner API tutorial: `docs/PLANNER_API.md`
- Result submission template: `docs/RESULT_SUBMISSION.md`
- Leaderboard policy: `docs/LEADERBOARD.md`
- Citation metadata: `CITATION.cff`

When reporting a method result, include:
- method name and commit hash
- run command used
- `result_schema.json`, `summary.csv`, and the relevant slice of `results.csv`
- key failure traces or mined worst-cases
- compute metrics (`planner_ms_*`)

Suggested run naming pattern:
- `runs_<method>_<YYYYMMDD>_<notes>`

## 14) FAQ / Common Gotchas

- Why collisions in `ideal_50hz`?
  - Usually planner logic issue, radius mismatch, or incorrect neighbor use.
- Why low completion with low collisions?
  - Overly conservative behavior or deadlocks.
- Why slower at `N=50+`?
  - Excess per-agent loops/object churn; optimize neighbor and math path.
- Why is my 3D run slower than planar?
  - Non-planar ORCA is obstacle-aware and uses a heavier solve/refinement path.
- Why different results between runs?
  - Seed/config mismatch or nondeterministic code path.
- Why no traces copied by hard-case miner?
  - Trace/event logging disabled (`logging.save_events` / `logging.save_trace_on_collision`).
- Q: Do I need intent messages for my planner?
  - A: Only for trajectory-sharing/negotiation methods. Geometry-only planners and diffusion planners can ignore intent.
- How to add a scenario?
  - Copy an existing YAML in `config/scenarios/`, adjust bounds/spawns/goals/events, include it in your sweep command.

## 15) Appendix Pointers

- Scenario schema examples: `config/scenarios/*.yaml`
- Planner API tutorial: `docs/PLANNER_API.md`
- Simple external planner example: `examples/simple_external_planner.py`
- Diffusion dataset implementation: `microbench/dataset/generate.py`
- Leaderboard CSV schema: `microbench/metrics/io.py`
- Current schema golden policy: `golden/current_schema/README.md`
- Comm profile definitions: `config/comm_profiles.yaml`
- Planner template: `microbench/planners/template.py`
