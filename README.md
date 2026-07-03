# DAA Microbench

DAA Microbench is a deterministic Python harness for evaluating local multi-drone detect-and-avoid planners at 50 Hz under configurable V2V communication impairment (rate, delay/jitter, loss, staleness). It models the local planning contract (velocity commands into a clamped dynamics step) without a full high-fidelity Sim/ROS/PX4 stack. Use it for fast, fair planner implementation and comparisons, agentic multi-drone DAA experiments, and diffusion-training dataset generation from baseline planners.

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

3. Run a small baseline sanity smoke sweep.

```bash
python -m microbench.cli canonical-sweep \
  --suite baseline_sanity \
  --out-dir runs_readme_baseline \
  --max-runs 4
```

4. Inspect outputs.

```bash
ls runs_readme_baseline
# results.csv, summary.csv, episodes/, worst_cases/ (if mined)
```

If your editable install exposed the entrypoint, replace `python -m microbench.cli` with `daa-microbench` or the compatibility alias `microbench`.

Available planners / how to list methods:

```bash
python -m microbench.cli list-methods
```

You can also inspect `microbench/planners/` (each planner module maps to a method name in the planner registry).

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
- Diffusion/ML extras: `pip install -e ".[ml]"`
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
```

When trace logging is enabled, `selected_obs[*].source` is `v2v` or `sensor`.

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
  replay/               # matplotlib replay renderer for trace files
  dataset/              # diffusion dataset generation + shard sanity checks
  tools/                # utilities (hard-case miner)
config/
  defaults.yaml         # global defaults
  comm_profiles.yaml    # named V2V profiles
  scenarios/*.yaml      # benchmark scenarios
scripts/
  ci_sanity.sh          # quick CI-style execution sanity check
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

## 6) Official Evaluation Protocol (Sweep Suite v1.0)

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
  --methods baseline_goal,orca_expert \
  --out-dir runs_primary_smoke \
  --max-runs 6
```

Run baseline sanity suite:

```bash
python -m microbench.cli canonical-sweep \
  --suite baseline_sanity \
  --methods baseline_goal,orca_expert \
  --out-dir runs_baseline_sanity
```

Expected baseline sanity behavior:
- `baseline_goal`: high collisions in dense scenarios.
- `orca_expert`: substantially lower collisions, especially in ideal comm.

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

Use this section for non-planar planner work. The official leaderboard suite above remains planar; the 3D material below is for  development, debugging, and non-planar comparisons.

Quick navigation:

| If You Want To... | Go Here |
|---|---|
| run one 3D episode | Section `7.2 Quick 3D Run` |
| benchmark a planner in 3D | Section `7.3 Canonical 3D Suite` |
| inspect a 3D replay/trace | Section `7.4 3D Replay and Traces` |
| profile 3D planner runtime | Section `7.5 3D Profiling Notes` |
| generate 3D diffusion data | Section `7.6 3D Diffusion Data` |

### 7.1 3D Scenarios

Built-in 3D scenarios:
- `config/scenarios/stacked_swap_3d.yaml`
- `config/scenarios/layered_funnel_3d.yaml`
- `config/scenarios/layered_intersection_3d.yaml`
- `config/scenarios/weather_vertical_event_3d.yaml`
- `config/scenarios/vertical_crossing_obstacles_3d.yaml`

What they are for:
- `stacked_swap_3d`: simple non-planar swap through shared volume
- `layered_funnel_3d`: funnel/gate pressure with mixed target altitude layers
- `layered_intersection_3d`: crossing flows with layered targets
- `weather_vertical_event_3d`: forced climb/descent event
- `vertical_crossing_obstacles_3d`: crossing traffic plus center obstacle

### 7.2 Quick 3D Run

```bash
python -m microbench.cli run \
  --scenario config/scenarios/stacked_swap_3d.yaml \
  --method orca_expert \
  --n 10 \
  --seed 0 \
  --comm ideal_50hz \
  --out-dir runs_3d_example
```

### 7.3 Canonical 3D Suite

Use the built-in 3D suite for comparisons:

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

If `--methods` is omitted for `three_d`, it defaults to `orca_expert`.

### 7.4 3D Replay and Traces

- full-episode replay works the same way as planar replay
- for non-planar episodes, replay automatically switches to a 3D view
- for deeper debugging, use the interactive HTML replay to orbit, zoom, and scrub through the episode with obstacle wireframes visible

Example:

```bash
python -m microbench.cli replay \
  --trace runs/<run_id>/episodes/<episode_dir>/trace_episode.jsonl \
  --out runs/<run_id>/episode.gif
```

Interactive HTML replay:

```bash
python -m microbench.cli replay-interactive \
  --trace runs/<run_id>/episodes/<episode_dir>/trace_episode.jsonl \
  --out runs/<run_id>/episode.html
```

What the interactive replay shows:
- full 3D camera orbit / zoom / pan
- time scrubber and play/pause controls
- collision-pair focus mode
- agent trails
- world bounds wireframe
- obstacle wireframes from scenario AABBs
- received intent tubes
- side plots for neighbor distance and obstacle clearance
- hover tooltips with per-agent position, speed, command speed, and neighbor staleness summary

### 7.5 3D Profiling Notes

- obstacle-aware `orca_expert` is heavier than `baseline_goal`
- expect a few milliseconds per tick per agent on harder 3D scenes
- use `results.csv` / `summary.csv` to compare:
  - `planner_ms_per_tick_per_agent_mean`
  - `planner_ms_per_tick_per_agent_p95`
  - `episode_runtime_s`

Quick profiling command:

```bash
python -m microbench.cli run \
  --scenario config/scenarios/layered_intersection_3d.yaml \
  --method orca_expert \
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
  --method orca_expert \
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
  --methods baseline_goal,orca_expert \
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

What gets logged:
- run config (methods/scenarios/N/seeds/comm, sim params, platform/python, git hash if available)
- overall summary scalars from `summary.csv`
- `leaderboard` table with full `summary.csv`
- optional `top_failures` table from `worst_cases/index.csv` when present

Artifacts:
- default: one results artifact containing `summary.csv` and `results.csv`
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

Render a full-episode replay:

```bash
python -m microbench.cli replay \
  --trace runs/<run_id>/episodes/<episode_dir>/trace_episode.jsonl \
  --out runs/<run_id>/episode.gif
```

For non-planar episodes, replay automatically switches to a 3D view.

Render a replay from a trace:

```bash
python -m microbench.cli replay \
  --trace runs_trace2/episodes/funnel_baseline_goal_n8_seed1/trace_collision_3_7_t16.16.jsonl \
  --out runs_readme_replay.gif \
  --fps 20 \
  --tail 20 \
  --show-sensed
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

`orca_expert` is used as the expert planner to generate training labels.

Generate dataset shards:

```bash
python -m microbench.cli generate-dataset \
  --scenario config/scenarios/intersection.yaml \
  --method orca_expert \
  --n 10 \
  --seeds 0:1 \
  --comm ideal_50hz,realistic_v2v_50hz \
  --T 20 \
  --dt-plan-s 0.10 \
  --quality-filter safe_expert \
  --filter-min-sep-m 0.2 \
  --out-dir datasets/orca_expert_v0
```

Sanity-check one shard:

```bash
python -m microbench.cli sanity-check-dataset \
  --shard datasets/orca_expert_v0/orca_expert/intersection/ideal_50hz/shard_00000.npz
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

Recommended scale for training: at least `100k+` samples across multiple scenarios and comm profiles.

## 12) Performance Expectations and Practical Profiling

- Track planner cost from `results.csv` columns:
  - `planner_ms_per_tick_per_agent_mean`
  - `planner_ms_per_tick_per_agent_p95`
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

## 13) Contribution Workflow

Project governance and contribution docs:
- License: `LICENSE` (Apache-2.0)
- Contributing guide: `CONTRIBUTING.md`
- Result submission template: `docs/RESULT_SUBMISSION.md`
- Leaderboard policy: `docs/LEADERBOARD.md`
- Citation metadata: `CITATION.cff`

When reporting a method result, include:
- method name and commit hash
- run command used
- `summary.csv` and relevant slice of `results.csv`
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
- Diffusion dataset implementation: `microbench/dataset/generate.py`
- Leaderboard CSV schema: `microbench/metrics/io.py`
- Comm profile definitions: `config/comm_profiles.yaml`
- Planner template: `microbench/planners/template.py`
