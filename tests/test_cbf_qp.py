from __future__ import annotations

import numpy as np

from microbench.planners.cbf_qp import CbfQpPlanner
from microbench.types import AABBObs, AgentState, NeighborObs, PlannerInput, PlannerOutput


def _agent(pos, vel=(0.0, 0.0, 0.0), goal=(10.0, 0.0, 0.0), radius=0.5, v_max=3.0, a_max=2.0):
    return AgentState(
        idx=0,
        pos=np.asarray(pos, dtype=np.float32),
        vel=np.asarray(vel, dtype=np.float32),
        goal=np.asarray(goal, dtype=np.float32),
        radius=radius,
        v_max=v_max,
        a_max=a_max,
    )


def _planner_input(*, ego: AgentState, neighbors=None, obstacles=None, planar=True) -> PlannerInput:
    return PlannerInput(
        ego=ego,
        goal_dir=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        neighbors=list(neighbors or []),
        obstacles=list(obstacles or []),
        dt=0.02,
        t=0.0,
        planar=planar,
    )


def test_cbf_qp_no_neighbors_tracks_goal() -> None:
    planner = CbfQpPlanner()
    out = planner.compute_cmd(_planner_input(ego=_agent((0.0, 0.0, 0.0))))

    assert isinstance(out, PlannerOutput)
    assert out.v_cmd[0] > 2.9
    assert abs(float(out.v_cmd[1])) < 1e-9
    assert abs(float(out.v_cmd[2])) < 1e-9
    assert out.debug_info["cbf_constraints"] == 0
    assert out.debug_info["cbf_fallback"] is False


def test_cbf_qp_head_on_uses_bounded_avoidance_fallback() -> None:
    ego = _agent((0.0, 0.0, 0.0), vel=(2.0, 0.0, 0.0))
    neighbor = NeighborObs(
        idx=1,
        pos=np.asarray([0.8, 0.0, 0.0], dtype=np.float32),
        vel=np.asarray([-2.0, 0.0, 0.0], dtype=np.float32),
        radius=0.5,
        msg_age_sec=0.0,
        valid=True,
    )

    out = CbfQpPlanner().compute_cmd(_planner_input(ego=ego, neighbors=[neighbor]))

    assert out.v_cmd[0] < 0.0
    assert np.linalg.norm(out.v_cmd) <= ego.v_max + 1e-6
    assert out.debug_info["cbf_constraints"] == 1
    assert "cbf_fallback" in out.debug_info


def test_cbf_qp_obstacle_in_path_slows_or_redirects() -> None:
    ego = _agent((0.0, 0.0, 0.0), vel=(2.0, 0.0, 0.0))
    obstacle = AABBObs(
        center=np.asarray([2.0, 0.0, 0.0], dtype=np.float32),
        half=np.asarray([0.5, 0.5, 0.5], dtype=np.float32),
    )

    out = CbfQpPlanner().compute_cmd(_planner_input(ego=ego, obstacles=[obstacle]))

    assert out.v_cmd[0] < 2.0
    assert np.linalg.norm(out.v_cmd) <= ego.v_max + 1e-6
    assert out.debug_info["cbf_constraints"] == 1


def test_cbf_qp_preserves_3d_command_shape() -> None:
    ego = _agent((0.0, 0.0, 0.0), vel=(2.0, 0.0, 0.0), goal=(10.0, 2.0, 0.0))
    inp = _planner_input(ego=ego, planar=False)
    inp.goal_dir = np.asarray([0.8, 0.6, 0.0], dtype=np.float32)

    out = CbfQpPlanner().compute_cmd(inp)

    assert out.v_cmd.shape == (3,)
    assert out.v_cmd[1] > 0.0
    assert np.linalg.norm(out.v_cmd) <= ego.v_max + 1e-6
