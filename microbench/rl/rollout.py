from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from microbench.rl.envs import DaaParallelEnv
from microbench.rl.policies import RlPolicy, make_policy
from microbench.rl.schema import RL_INTERFACE_VERSION


RL_ROLLOUT_SCHEMA_VERSION = "0.1"
RL_ROLLOUT_FIELDS = (
    "suite",
    "scenario",
    "dimension",
    "policy",
    "n_agents",
    "seed",
    "comm_profile",
    "steps",
    "controlled_agents",
    "completed_agents",
    "completion_rate",
    "terminated_agents",
    "truncated_agents",
    "total_reward",
    "mean_reward_per_agent",
    "final_min_sep_m",
    "collision_ticks",
    "near_miss_ticks",
    "finite_observations",
    "finite_rewards",
    "api_error",
)

PolicyFactory = Callable[[int], RlPolicy]


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _policy_instance(policy: str | RlPolicy | PolicyFactory, *, seed: int) -> tuple[RlPolicy, str]:
    if isinstance(policy, str):
        return make_policy(policy, seed=int(seed)), str(policy)
    if callable(policy) and not hasattr(policy, "action"):
        created = policy(int(seed))
        if hasattr(created, "reset"):
            created.reset(int(seed))
        return created, str(getattr(created, "policy_name", type(created).__name__))
    if hasattr(policy, "reset"):
        policy.reset(int(seed))
    return policy, str(getattr(policy, "policy_name", type(policy).__name__))


def _policy_label(policy: str | RlPolicy | PolicyFactory, explicit: str | None = None) -> str:
    if explicit:
        return str(explicit)
    if isinstance(policy, str):
        return str(policy)
    return str(getattr(policy, "policy_name", type(policy).__name__))


def rollout_parallel_env(
    env: DaaParallelEnv,
    policy: str | RlPolicy | PolicyFactory,
    *,
    seed: int = 0,
    max_steps: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Roll out a policy in a `DaaParallelEnv` and return one episode row."""

    meta = dict(metadata or {})
    policy_obj, policy_name = _policy_instance(policy, seed=int(seed))
    observations, infos = env.reset(seed=int(seed))
    agent_names = list(env.agents)
    controlled_count = len(agent_names)
    total_rewards = {agent: 0.0 for agent in agent_names}
    completed: set[str] = set()
    terminated: set[str] = set()
    truncated: set[str] = set()
    collision_ticks = 0
    near_miss_ticks = 0
    final_min_sep = math.inf
    finite_observations = all(np.all(np.isfinite(obs)) for obs in observations.values())
    finite_rewards = True
    step_limit = env.episode_step_limit if env.episode_step_limit is not None else 0
    cap = int(max_steps) if max_steps is not None else int(step_limit)
    steps = 0

    while env.agents and steps < cap:
        actions = {
            agent: policy_obj.action(agent, observations[agent], env.action_space(agent), infos.get(agent, {}))
            for agent in env.agents
        }
        observations, rewards, terminations, truncations, infos = env.step(actions)
        steps += 1
        finite_observations = finite_observations and all(np.all(np.isfinite(obs)) for obs in observations.values())
        finite_rewards = finite_rewards and all(_finite(value) for value in rewards.values())
        for agent, reward in rewards.items():
            total_rewards[agent] = total_rewards.get(agent, 0.0) + float(reward)
        for agent, done in terminations.items():
            if done:
                terminated.add(agent)
        for agent, done in truncations.items():
            if done:
                truncated.add(agent)
        for agent, info in infos.items():
            if bool(info.get("done", False)):
                completed.add(agent)
            final_min_sep = min(final_min_sep, float(info.get("min_sep_m", final_min_sep)))
        if any(bool(info.get("collision", False)) for info in infos.values()):
            collision_ticks += 1
        if any(bool(info.get("near_miss", False)) for info in infos.values()):
            near_miss_ticks += 1

    if env.agents and steps >= cap:
        truncated.update(env.agents)
    if not math.isfinite(final_min_sep):
        final_min_sep = float("nan")

    total_reward = float(sum(total_rewards.values()))
    return {
        **meta,
        "interface_version": RL_INTERFACE_VERSION,
        "dimension": "2d" if env.planar else "3d",
        "policy": str(meta.get("policy", policy_name)),
        "n_agents": int(meta.get("n_agents", env.n_agents)),
        "seed": int(seed),
        "comm_profile": str(meta.get("comm_profile", env.comm_profile)),
        "steps": int(steps),
        "controlled_agents": int(controlled_count),
        "completed_agents": int(len(completed)),
        "completion_rate": float(len(completed) / max(1, controlled_count)),
        "terminated_agents": int(len(terminated)),
        "truncated_agents": int(len(truncated)),
        "total_reward": total_reward,
        "mean_reward_per_agent": float(total_reward / max(1, controlled_count)),
        "final_min_sep_m": float(final_min_sep),
        "collision_ticks": int(collision_ticks),
        "near_miss_ticks": int(near_miss_ticks),
        "finite_observations": bool(finite_observations),
        "finite_rewards": bool(finite_rewards),
        "api_error": "",
    }


def run_parallel_policy_rollouts(
    *,
    scenario_paths: dict[str, str | Path],
    policy: str | RlPolicy | PolicyFactory = "goal_direction",
    n_agents: int = 4,
    seeds: tuple[int, ...] | list[int] | None = None,
    comm_profile: str = "ideal_50hz",
    max_steps: int | None = None,
    suite: str = "custom",
    policy_name: str | None = None,
) -> list[dict[str, Any]]:
    """Run a small scenario/seed matrix through the parallel RL wrapper."""

    seed_list = [int(seed) for seed in (seeds if seeds is not None else (0,))]
    rows: list[dict[str, Any]] = []
    for scenario_id, scenario_path in scenario_paths.items():
        for seed in seed_list:
            env = DaaParallelEnv(
                scenario_path=str(scenario_path),
                n_agents=int(n_agents),
                seed=int(seed),
                comm_profile=str(comm_profile),
            )
            try:
                rows.append(
                    rollout_parallel_env(
                        env,
                        policy,
                        seed=int(seed),
                        max_steps=max_steps,
                        metadata={
                            "suite": str(suite),
                            "scenario": str(scenario_id),
                            "policy": _policy_label(policy, policy_name),
                            "n_agents": int(n_agents),
                            "comm_profile": str(comm_profile),
                        },
                    )
                )
            except Exception as exc:  # pragma: no cover - failure reporting path.
                rows.append(
                    {
                        "suite": str(suite),
                        "scenario": str(scenario_id),
                        "dimension": "unknown",
                        "policy": _policy_label(policy, policy_name),
                        "n_agents": int(n_agents),
                        "seed": int(seed),
                        "comm_profile": str(comm_profile),
                        "steps": 0,
                        "controlled_agents": 0,
                        "completed_agents": 0,
                        "completion_rate": 0.0,
                        "terminated_agents": 0,
                        "truncated_agents": 0,
                        "total_reward": 0.0,
                        "mean_reward_per_agent": 0.0,
                        "final_min_sep_m": float("nan"),
                        "collision_ticks": 0,
                        "near_miss_ticks": 0,
                        "finite_observations": False,
                        "finite_rewards": False,
                        "api_error": f"{type(exc).__name__}: {exc}",
                    }
                )
            finally:
                env.close()
    return rows
