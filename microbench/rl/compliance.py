from __future__ import annotations

from typing import Any

import numpy as np

from microbench.rl.envs import DaaParallelEnv
from microbench.rl.schema import RL_INTERFACE_VERSION


def _check(name: str, ok: bool, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "details": details or {}}


def check_parallel_env_api(env: DaaParallelEnv, *, seed: int = 0, steps: int = 2) -> dict[str, Any]:
    """Run lightweight PettingZoo-style API checks against a DAA parallel env.

    This intentionally avoids importing PettingZoo's optional test utilities so
    the core package can run the compatibility check without the `rl` extra.
    """

    checks: list[dict[str, Any]] = []
    obs, infos = env.reset(seed=int(seed))
    initial_agents = list(env.agents)
    possible_agents = list(env.possible_agents)

    checks.append(_check("possible_agents_nonempty", len(possible_agents) > 0, {"possible_agents": possible_agents}))
    checks.append(_check("agents_subset_possible", set(initial_agents).issubset(set(possible_agents))))
    checks.append(_check("reset_obs_keys_match_agents", set(obs) == set(initial_agents)))
    checks.append(_check("reset_info_keys_match_agents", set(infos) == set(initial_agents)))

    reset_space_violations = []
    for agent, value in obs.items():
        if not env.observation_space(agent).contains(value):
            reset_space_violations.append(agent)
    checks.append(_check("reset_observations_in_space", not reset_space_violations, {"violations": reset_space_violations}))

    contract = env.interface_contract()
    obs_shape = tuple(contract["observation"]["shape"])
    act_shape = tuple(contract["action"]["shape"])
    checks.append(
        _check(
            "schema_shapes_match_spaces",
            all(tuple(env.observation_space(agent).shape) == obs_shape for agent in initial_agents)
            and all(tuple(env.action_space(agent).shape) == act_shape for agent in initial_agents),
            {"observation_shape": obs_shape, "action_shape": act_shape},
        )
    )

    step_count = 0
    step_key_violations: list[dict[str, Any]] = []
    type_violations: list[dict[str, Any]] = []
    while env.agents and step_count < int(steps):
        current_agents = list(env.agents)
        actions = {agent: np.zeros(act_shape, dtype=np.float32) for agent in current_agents}
        next_obs, rewards, terminations, truncations, step_infos = env.step(actions)
        step_count += 1
        expected = set(current_agents)
        returned_sets = {
            "observations": set(next_obs),
            "rewards": set(rewards),
            "terminations": set(terminations),
            "truncations": set(truncations),
            "infos": set(step_infos),
        }
        for name, keys in returned_sets.items():
            if keys != expected:
                step_key_violations.append({"step": step_count, "field": name, "expected": sorted(expected), "actual": sorted(keys)})
        for agent, value in next_obs.items():
            if not env.observation_space(agent).contains(value):
                type_violations.append({"step": step_count, "agent": agent, "field": "observation_space"})
        for agent, reward in rewards.items():
            if not isinstance(reward, float) or not np.isfinite(float(reward)):
                type_violations.append({"step": step_count, "agent": agent, "field": "reward", "value": reward})
        for agent, done in terminations.items():
            if not isinstance(done, bool):
                type_violations.append({"step": step_count, "agent": agent, "field": "termination", "value": done})
        for agent, done in truncations.items():
            if not isinstance(done, bool):
                type_violations.append({"step": step_count, "agent": agent, "field": "truncation", "value": done})
        for agent, info in step_infos.items():
            if not isinstance(info, dict):
                type_violations.append({"step": step_count, "agent": agent, "field": "info", "value": type(info).__name__})
        if not set(env.agents).issubset(set(possible_agents)):
            type_violations.append({"step": step_count, "field": "agents_subset_possible", "agents": list(env.agents)})

    checks.append(_check("step_return_keys_match_acting_agents", not step_key_violations, {"violations": step_key_violations[:10]}))
    checks.append(_check("step_values_match_contract", not type_violations, {"violations": type_violations[:10]}))
    checks.append(_check("stepped_at_least_once", step_count > 0, {"steps": step_count}))

    return {
        "interface_version": RL_INTERFACE_VERSION,
        "ok": all(check["ok"] for check in checks),
        "steps": step_count,
        "possible_agents": possible_agents,
        "initial_agents": initial_agents,
        "contract": contract,
        "checks": checks,
    }
