from __future__ import annotations

from pathlib import Path

import numpy as np

from microbench.rl import DaaParallelEnv, DaaSingleAgentEnv, check_parallel_env_api
from microbench.scenarios import materialize_official_suite


ROOT = Path(__file__).resolve().parents[1]


def _generated_scenario(tmp_path: Path, suite: str, scenario_id: str) -> Path:
    generated = materialize_official_suite(suite, tmp_path / suite, overwrite=True)
    return next(path for path in generated["scenario_paths"] if path.stem == scenario_id)


def test_parallel_env_reset_step_and_spaces(tmp_path: Path) -> None:
    scenario_path = _generated_scenario(tmp_path, "official_smoke_generated", "head_on_2d_easy")
    env = DaaParallelEnv(
        scenario_path=str(scenario_path),
        n_agents=4,
        seed=0,
        comm_profile="ideal_50hz",
    )
    try:
        obs, infos = env.reset()
        assert env.agents == ["agent_0", "agent_1", "agent_2", "agent_3"]
        assert set(obs) == set(env.agents)
        assert set(infos) == set(env.agents)
        for agent, value in obs.items():
            assert env.observation_space(agent).contains(value)
            assert env.action_space(agent).contains(np.zeros(3, dtype=np.float32))

        actions = {agent: np.asarray([0.5, 0.0, 0.0], dtype=np.float32) for agent in env.agents}
        next_obs, rewards, terminations, truncations, step_infos = env.step(actions)

        assert set(next_obs) == {"agent_0", "agent_1", "agent_2", "agent_3"}
        assert all(isinstance(value, float) for value in rewards.values())
        assert all(isinstance(value, bool) for value in terminations.values())
        assert all(isinstance(value, bool) for value in truncations.values())
        assert step_infos["agent_0"]["controlled"] is True
        assert env.render()["n_agents"] == 4
        contract = env.interface_contract()
        assert contract["interface_version"] == "0.1.0"
        assert contract["observation"]["shape"] == [env.observation_space("agent_0").shape[0]]
        assert contract["action"]["shape"] == [3]
    finally:
        env.close()


def test_parallel_env_lightweight_api_compatibility_check(tmp_path: Path) -> None:
    scenario_path = _generated_scenario(tmp_path, "official_smoke_generated", "head_on_2d_easy")
    env = DaaParallelEnv(
        scenario_path=str(scenario_path),
        n_agents=4,
        seed=0,
        comm_profile="ideal_50hz",
    )
    try:
        report = check_parallel_env_api(env, seed=0, steps=2)
        assert report["ok"] is True
        assert report["interface_version"] == "0.1.0"
        assert {check["name"] for check in report["checks"]} >= {
            "reset_obs_keys_match_agents",
            "step_return_keys_match_acting_agents",
            "schema_shapes_match_spaces",
        }
    finally:
        env.close()


def test_parallel_env_leaves_scenario_noncooperative_agents_as_background(tmp_path: Path) -> None:
    scenario_path = _generated_scenario(tmp_path, "official_agentic_stress", "multi_intruder_3d_hard")
    env = DaaParallelEnv(
        scenario_path=str(scenario_path),
        n_agents=8,
        seed=0,
        comm_profile="degraded_20hz",
    )
    try:
        obs, infos = env.reset()

        assert env.agents == ["agent_3", "agent_4", "agent_5", "agent_6", "agent_7"]
        assert set(obs) == set(env.agents)
        assert infos["agent_3"]["controlled"] is True
        assert [ctx.method for ctx in env._engine.agent_contexts[:3]] == ["baseline_goal", "baseline_goal", "baseline_goal"]
    finally:
        env.close()


def test_single_agent_env_wraps_one_ego_with_background_baselines(tmp_path: Path) -> None:
    scenario_path = _generated_scenario(tmp_path, "official_smoke_generated", "sphere_swap_3d_medium")
    env = DaaSingleAgentEnv(
        scenario_path=str(scenario_path),
        n_agents=4,
        ego_agent_id=1,
        seed=0,
        comm_profile="ideal_50hz",
        background_method="orca_heuristic",
    )
    try:
        obs, info = env.reset()
        assert env.observation_space.contains(obs)
        assert info["agent_id"] == 1
        assert info["controlled"] is True

        next_obs, reward, terminated, truncated, step_info = env.step(np.asarray([0.2, 0.1, 0.0], dtype=np.float32))
        assert env.observation_space.contains(next_obs)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert step_info["method"] == "rl_policy"
    finally:
        env.close()
