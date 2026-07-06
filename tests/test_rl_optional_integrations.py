from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from microbench.scenarios import materialize_official_suite


def _generated_scenario(tmp_path: Path, scenario_id: str) -> Path:
    generated = materialize_official_suite("official_smoke_generated", tmp_path / "suite", overwrite=True)
    return next(path for path in generated["scenario_paths"] if path.stem == scenario_id)


def test_optional_gymnasium_single_agent_env(tmp_path: Path) -> None:
    gymnasium = pytest.importorskip("gymnasium")

    from microbench.rl import DaaSingleAgentEnv

    env = DaaSingleAgentEnv(
        scenario_path=str(_generated_scenario(tmp_path, "sphere_swap_3d_medium")),
        n_agents=4,
        ego_agent_id=0,
        seed=0,
    )
    try:
        assert isinstance(env, gymnasium.Env)
        obs, info = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        next_obs, reward, terminated, truncated, step_info = env.step(np.zeros(3, dtype=np.float32))
        assert env.observation_space.contains(next_obs)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert step_info["controlled"] is True
        assert info["controlled"] is True
    finally:
        env.close()


def test_optional_pettingzoo_parallel_env(tmp_path: Path) -> None:
    pz_env = pytest.importorskip("pettingzoo.utils.env")

    from microbench.rl import DaaParallelEnv, check_parallel_env_api

    env = DaaParallelEnv(
        scenario_path=str(_generated_scenario(tmp_path, "head_on_2d_easy")),
        n_agents=4,
        seed=0,
    )
    try:
        assert isinstance(env, pz_env.ParallelEnv)
        report = check_parallel_env_api(env, seed=0, steps=2)
        assert report["ok"] is True
    finally:
        env.close()
