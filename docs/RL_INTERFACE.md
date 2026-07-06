# RL Interface

DAA Microbench exposes a lightweight PettingZoo/Gymnasium-style interface for learning researchers. The wrappers use the same `EpisodeEngine` as benchmark runs, so scenarios, sensing, V2V impairment, agent messages, heterogeneous agents, collisions, and dynamics stay aligned with official evaluation.

Install the optional RL integrations when you want real Gymnasium/PettingZoo space classes:

```bash
pip install -e ".[rl]"
```

The core package still imports without those extras. In that case DAA Microbench uses a tiny fallback `Box` space with `sample()` and `contains()` for smoke tests and simple scripts.

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

## Public Alpha Caveats

The RL interface is pre-v1. Observation vector layout, reward defaults, and helper wrappers may still change before stable v1. Official benchmark results should continue to report `results.csv`, `summary.csv`, suite manifests, and result schema sidecars.
