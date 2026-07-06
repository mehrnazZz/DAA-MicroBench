from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np

from examples.simple_external_planner import SimpleExternalPlanner
from microbench.core import EpisodeEngine
from microbench.runner import run_episode
from microbench.types import AgentState, PlannerInput, RunSpec


def _write_example_scenario(path: Path) -> None:
    path.write_text(
        """
scenario:
  name: "example_planner_smoke"
  duration_s: 0.08
sim:
  dt_s: 0.02
world:
  planar: false
  bounds:
    x: [-5.0, 5.0]
    y: [-2.0, 2.0]
    z: [-5.0, 5.0]
agent_params:
  radius_m: 0.25
  v_max_mps: 1.0
  a_max_mps2: 2.0
  goal_tolerance_m: 0.1
neighbors:
  range_m: 30.0
  top_k: 2
goals:
  min_goal_distance_m: 1.0
spawn:
  type: "circle_swap"
  center: [0.0, 0.0, 0.0]
  radius_m: 1.0
  jitter_m: 0.0
  start_layers_m: [-1.0, 1.0]
  goal_layers_m: [1.0, -1.0]
logging:
  save_events: false
  save_trace: false
""".strip(),
        encoding="utf-8",
    )


def test_simple_external_planner_returns_finite_public_command() -> None:
    planner = SimpleExternalPlanner()
    planner.reset(agent_id=0, seed=123, config={"priority": 1})
    ego = AgentState(
        idx=0,
        pos=np.asarray([0.0, 0.0, 0.0], dtype=float),
        vel=np.asarray([0.0, 0.0, 0.0], dtype=float),
        goal=np.asarray([1.0, 0.0, 0.0], dtype=float),
        radius=0.25,
        v_max=1.0,
        a_max=2.0,
    )
    out = planner.compute_cmd(
        PlannerInput(
            ego=ego,
            goal_dir=np.asarray([1.0, 0.0, 0.0], dtype=float),
            neighbors=[],
            dt=0.02,
            t=0.0,
            planar=False,
        )
    )

    assert out.v_cmd.shape == (3,)
    assert np.all(np.isfinite(out.v_cmd))
    assert float(np.linalg.norm(out.v_cmd)) <= ego.v_max + 1e-9


def test_simple_external_planner_runs_through_episode_engine(tmp_path: Path) -> None:
    scenario = tmp_path / "example_3d.yaml"
    _write_example_scenario(scenario)

    engine = EpisodeEngine(
        scenario_path=str(scenario),
        method="simple_external",
        n_agents=2,
        seed=0,
        comm_profile="ideal_50hz",
        planner_factory=lambda _: SimpleExternalPlanner(),
    )
    step = engine.step()
    engine.close()

    assert step is not None
    assert step.planner_debug[0]["ticks"] == 1
    assert step.planner_debug[1]["ticks"] == 1
    assert engine.planner_error_count == 0
    assert engine.planner_timeout_count == 0
    assert engine.planner_fallback_count == 0


def test_simple_external_planner_runner_metrics_are_clean(tmp_path: Path) -> None:
    scenario = tmp_path / "example_3d.yaml"
    _write_example_scenario(scenario)

    with patch("microbench.runner.make_planner", side_effect=lambda _: SimpleExternalPlanner()):
        row = run_episode(
            RunSpec(
                scenario_path=str(scenario),
                method="simple_external",
                n_agents=2,
                seed=0,
                comm_profile="ideal_50hz",
                out_dir=str(tmp_path / "runs"),
                save_trace=False,
            )
        )

    assert int(row["planner_error_count"]) == 0
    assert int(row["planner_timeout_count"]) == 0
    assert int(row["planner_fallback_count"]) == 0
