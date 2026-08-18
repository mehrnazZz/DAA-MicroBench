from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from microbench.core.episode_engine import EpisodeEngine
from microbench.runner import run_episode
from microbench.types import RunSpec


ROOT = Path(__file__).resolve().parents[1]


def _write_compact_gate_scenario(path: Path) -> Path:
    path.write_text(
        """
scenario:
  name: "compact_gate"
  duration_s: 12.0
world:
  planar: true
  fixed_y_m: 0.0
  bounds:
    xmin: -16.0
    xmax: 16.0
    ymin: -1.0
    ymax: 1.0
    zmin: -10.0
    zmax: 10.0
agent_params:
  radius_m: 0.4
  v_max_mps: 3.0
  a_max_mps2: 2.0
  goal_tolerance_m: 1.0
goals:
  min_goal_distance_m: 12.0
spawn:
  type: "rect_to_rect"
  min_start_separation_m: 1.2
  start_region:
    center: [-10.0, 0.0, 0.0]
    half: [1.0, 0.0, 3.0]
  goal_region:
    center: [10.0, 0.0, 0.0]
    half: [1.0, 0.0, 3.0]
obstacles:
  - aabb:
      center: [0.0, 0.0, 7.0]
      half: [1.2, 1.0, 3.0]
  - aabb:
      center: [0.0, 0.0, -7.0]
      half: [1.2, 1.0, 3.0]
""".lstrip(),
        encoding="utf-8",
    )
    return path


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
    with pytest.raises(ValueError, match="privileged centralized oracle methods must control all agents"):
        EpisodeEngine(
            scenario_path=str(ROOT / "config" / "scenarios" / "stacked_swap_3d.yaml"),
            method="baseline_goal",
            agent_methods=["centralized_oracle", "baseline_goal"],
            n_agents=2,
            seed=0,
            comm_profile="ideal_50hz",
        )


def test_centralized_mpc_oracle_step_uses_route_aware_privileged_path() -> None:
    engine = EpisodeEngine(
        scenario_path=str(ROOT / "config" / "scenarios" / "funnel.yaml"),
        method="centralized_mpc_oracle",
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
        assert info["centralized_mpc_oracle"] is True
        assert info["privileged_joint_method"] == "centralized_mpc_oracle"
        assert info["non_deployable"] is True
        assert info["privileged_global_state"] is True
        assert info["oracle_scope"] == "all_agents_all_obstacles_world_bounds"
        assert info["centralized_mpc_oracle_candidates"] > 0
        assert info["route_waypoints"] >= 2
        assert info["planner_elapsed_ms"] >= 0.0


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


def test_centralized_mpc_oracle_runs_gate_episode_without_guardrail_errors(tmp_path: Path) -> None:
    scenario = _write_compact_gate_scenario(tmp_path / "compact_gate.yaml")
    row = run_episode(
        RunSpec(
            scenario_path=str(scenario),
            method="centralized_mpc_oracle",
            n_agents=3,
            seed=0,
            comm_profile="ideal_50hz",
            out_dir=str(tmp_path),
            save_trace=False,
        )
    )

    assert row["method"] == "centralized_mpc_oracle"
    assert row["planner_error_count"] == 0
    assert row["planner_timeout_count"] == 0
    assert row["planner_fallback_count"] == 0
    assert row["collision_episode"] == 0
