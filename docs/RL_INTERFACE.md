# RL Interface

DAA Microbench exposes a lightweight PettingZoo/Gymnasium-style interface for learning researchers. The wrappers use the same `EpisodeEngine` as benchmark runs, so scenarios, sensing, V2V impairment, agent messages, heterogeneous agents, collisions, and dynamics stay aligned with official evaluation.

Install the optional RL integrations when you want real Gymnasium/PettingZoo space classes:

```bash
pip install -e ".[rl]"
```

The core package still imports without those extras. In that case DAA Microbench uses a tiny fallback `Box` space with `sample()` and `contains()` for smoke tests and simple scripts.

Current public-alpha RL interface version: `0.1.0`.

## Parallel Multi-Agent

Use `DaaParallelEnv` for decentralized multi-drone control. It follows the PettingZoo `ParallelEnv` shape:

```python
import numpy as np

from microbench.rl import DaaParallelEnv

env = DaaParallelEnv(
    scenario_path="config/scenarios/stacked_swap_3d.yaml",
    n_agents=4,
    seed=0,
    comm_profile="ideal_50hz",
)

observations, infos = env.reset()
while env.agents:
    actions = {
        agent: env.action_space(agent).sample()
        for agent in env.agents
    }
    observations, rewards, terminations, truncations, infos = env.step(actions)

env.close()
```

Actions are normalized desired velocity commands with shape `(3,)` and bounds `[-1, 1]`. The wrapper scales each action by the controlled drone's `v_max_mps`; the simulator still applies speed and acceleration limits.

Observations are fixed-size local vectors:

- ego position, velocity, goal direction, goal distance, done flag, time, normalized agent id, priority, radius, `v_max`, and `a_max`
- padded top-k local neighbor tracks from the benchmark observation pipeline, using relative position, relative velocity, radius, and message age

The vector intentionally uses the same local information surface exposed to planners, not privileged global state for all drones.

Base observation layout:

| Field | Indices | Meaning |
| --- | ---: | --- |
| `ego_pos` | `0:3` | ego position `(x, y, z)` in meters |
| `ego_vel` | `3:6` | ego velocity `(vx, vy, vz)` in m/s |
| `goal_dir` | `6:9` | unit direction from ego to goal |
| `goal_dist` | `9` | distance to goal in meters |
| `done` | `10` | simulator goal-completion flag |
| `time_s` | `11` | episode time in seconds |
| `agent_id_norm` | `12` | agent id normalized to `[0, 1]` |
| `priority` | `13` | scenario priority value |
| `radius_m` | `14` | collision radius |
| `v_max_mps` | `15` | speed limit |
| `a_max_mps2` | `16` | acceleration limit |
| `neighbors` | `17:` | padded top-k neighbor blocks |

Each neighbor block has 9 values: present flag, relative position `(3)`, relative velocity `(3)`, neighbor radius, and message age.

You can inspect the machine-readable contract:

```bash
python -m microbench.cli rl-contract --json
```

The JSON includes schema versions for actions, observations, and rewards:

- action schema: normalized `(3,)` `float32` desired-velocity actions in `[-1, 1]`
- observation schema: fixed `float32` vector with base ego fields plus padded top-k neighbor blocks
- reward schema: default public-alpha training reward weights and term descriptions

## Background Traffic

The default controlled method is `rl_policy`. Scenario-configured agents with another method remain background traffic. This matters for official agentic stress cases such as `multi_intruder_3d_hard`, where noncooperative intruders can be configured as `baseline_goal` while cooperative agents are controlled by the learner.

You can also provide explicit methods:

```python
env = DaaParallelEnv(
    scenario_path="config/scenarios/stacked_swap_3d.yaml",
    n_agents=4,
    agent_methods=["rl_policy", "orca_heuristic", "orca_heuristic", "orca_heuristic"],
    controlled_agents=[0],
)
```

## Single-Agent Gymnasium Style

Use `DaaSingleAgentEnv` for single-ego experiments with benchmark baselines as background traffic:

```python
from microbench.rl import DaaSingleAgentEnv

env = DaaSingleAgentEnv(
    scenario_path="config/scenarios/stacked_swap_3d.yaml",
    n_agents=4,
    ego_agent_id=0,
    background_method="orca_heuristic",
)

obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

## Reward

The default reward is intentionally simple:

- positive progress toward goal
- small time penalty
- collision penalty
- near-miss penalty
- goal-completion bonus

Override weights with `reward_config` on `DaaParallelEnv`. Stable leaderboard comparisons should still use benchmark metrics and official suite reports, not training reward alone.

## Smoke Evaluation

Run a compact 2D/3D wrapper check before handing the environment to a trainer:

```bash
python -m microbench.cli rl-smoke \
  --out-dir runs_rl_smoke \
  --policy goal_direction \
  --require-pass
```

The command materializes `official_smoke_generated`, runs one 2D and one 3D scenario through the RL wrapper, writes `rl_smoke.json`, and writes per-episode rows to `rl_smoke_episodes.csv`. Built-in smoke policies are `zero`, `random`, and `goal_direction`.

The same runner is available from Python:

```python
from microbench.rl import run_rl_policy_smoke

report = run_rl_policy_smoke(
    out_dir="runs_rl_smoke",
    policy="random",
    max_steps=100,
)
assert report["ok"]
```

See `examples/rl_random_policy.py` for a minimal runnable script.

## Compatibility Check

For custom adapters, use the lightweight compatibility checker without installing PettingZoo's optional test helpers:

```python
from microbench.rl import DaaParallelEnv, check_parallel_env_api

env = DaaParallelEnv(
    scenario_path="config/scenarios/stacked_swap_3d.yaml",
    n_agents=4,
)
try:
    report = check_parallel_env_api(env, seed=0, steps=2)
    assert report["ok"]
finally:
    env.close()
```

The checker validates reset/step dictionary keys, observation/action-space shapes, finite rewards, boolean termination/truncation flags, and agent-list consistency.

## Public Alpha Caveats

The RL interface is pre-v1. Observation vector layout, reward defaults, and helper wrappers may still change before stable v1. Official benchmark results should continue to report `results.csv`, `summary.csv`, suite manifests, and result schema sidecars.
