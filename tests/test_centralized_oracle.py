from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from microbench.core.episode_engine import EpisodeEngine
from microbench.runner import run_episode
from microbench.types import RunSpec


ROOT = Path(__file__).resolve().parents[1]


def test_centralized_oracle_step_uses_privileged_joint_path() -> None:
    engine = EpisodeEngine(
        scenario_path=str(ROOT / "config" / "scenarios" / "stacked_swap_3d.yaml"),
        method="centralized_oracle",
        n_agents=4,
        seed=0,
        comm_profile="ideal_50hz",
    )
    try:
        step = engine.step()
    finally:
        engine.close()

    assert step is not None
    assert any(np.linalg.norm(cmd) > 0.0 for cmd in step.v_cmds)
    for info in step.planner_debug:
        assert info["centralized_oracle"] is True
        assert info["non_deployable"] is True
        assert info["privileged_global_state"] is True
        assert info["oracle_scope"] == "all_agents_all_obstacles"
        assert info["planner_elapsed_ms"] >= 0.0


def test_centralized_oracle_rejects_mixed_agent_methods() -> None:
    with pytest.raises(ValueError, match="centralized_oracle must control all agents"):
        EpisodeEngine(
            scenario_path=str(ROOT / "config" / "scenarios" / "stacked_swap_3d.yaml"),
            method="baseline_goal",
            agent_methods=["centralized_oracle", "baseline_goal"],
            n_agents=2,
            seed=0,
            comm_profile="ideal_50hz",
        )


def test_centralized_oracle_runs_episode_without_local_planner_errors(tmp_path: Path) -> None:
    row = run_episode(
        RunSpec(
            scenario_path=str(ROOT / "config" / "scenarios" / "urban_conflict_3d.yaml"),
            method="centralized_oracle",
            n_agents=4,
            seed=2,
            comm_profile="ideal_50hz",
            out_dir=str(tmp_path),
            save_trace=False,
        )
    )

    assert row["method"] == "centralized_oracle"
    assert row["planner_error_count"] == 0
    assert row["planner_timeout_count"] == 0
    assert row["planner_fallback_count"] == 0
    assert row["planner_ms_per_tick_per_agent_mean"] >= 0.0
