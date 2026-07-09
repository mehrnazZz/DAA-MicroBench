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
    assert out.debug_info["cbf_neighbor_constraints"] == 1
    assert out.debug_info["cbf_obstacle_constraints"] == 0
    assert out.debug_info["cbf_solver"] in {"scipy_slsqp", "deterministic_projection"}


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
    assert out.debug_info["cbf_neighbor_constraints"] == 0
    assert out.debug_info["cbf_obstacle_constraints"] == 1
    assert out.debug_info["cbf_min_clearance_m"] is not None


def test_cbf_qp_preserves_3d_command_shape() -> None:
    ego = _agent((0.0, 0.0, 0.0), vel=(2.0, 0.0, 0.0), goal=(10.0, 2.0, 0.0))
    inp = _planner_input(ego=ego, planar=False)
    inp.goal_dir = np.asarray([0.8, 0.6, 0.0], dtype=np.float32)

    out = CbfQpPlanner().compute_cmd(inp)

    assert out.v_cmd.shape == (3,)
    assert out.v_cmd[1] > 0.0
    assert np.linalg.norm(out.v_cmd) <= ego.v_max + 1e-6


def test_cbf_qp_projection_solver_mode_is_explicit() -> None:
    ego = _agent((0.0, 0.0, 0.0), vel=(2.0, 0.0, 0.0))
    neighbor = NeighborObs(
        idx=1,
        pos=np.asarray([0.8, 0.0, 0.0], dtype=np.float32),
        vel=np.asarray([-2.0, 0.0, 0.0], dtype=np.float32),
        radius=0.5,
        msg_age_sec=0.0,
        valid=True,
    )

    out = CbfQpPlanner(cfg={"solver": "projection"}).compute_cmd(_planner_input(ego=ego, neighbors=[neighbor]))

    assert out.debug_info["cbf_solver"] == "deterministic_projection"
    assert out.debug_info["cbf_solver_requested"] == "projection"
    assert out.debug_info["cbf_solver_status"] in {"projection_converged", "projection_residual_violation"}


def test_cbf_qp_stale_tracks_inflate_barrier_and_slow_more() -> None:
    ego = _agent((0.0, 0.0, 0.0), vel=(1.0, 0.0, 0.0))
    fresh = NeighborObs(
        idx=1,
        pos=np.asarray([2.4, 0.0, 0.0], dtype=np.float32),
        vel=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        radius=0.5,
        msg_age_sec=0.0,
        track_age_sec=0.0,
        valid=True,
    )
    stale = NeighborObs(
        idx=1,
        pos=np.asarray([2.4, 0.0, 0.0], dtype=np.float32),
        vel=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        radius=0.5,
        msg_age_sec=1.0,
        track_age_sec=1.0,
        valid=True,
        stale=True,
    )
    planner = CbfQpPlanner(
        cfg={
            "solver": "projection",
            "stale_inflation_gain": 1.0,
            "track_uncertainty_speed_gain": 0.0,
            "stale_age_cap_s": 2.0,
        }
    )

    fresh_out = planner.compute_cmd(_planner_input(ego=ego, neighbors=[fresh]))
    stale_out = planner.compute_cmd(_planner_input(ego=ego, neighbors=[stale]))

    assert stale_out.v_cmd[0] < fresh_out.v_cmd[0]
    assert fresh_out.debug_info["cbf_uncertainty_inflation_max_m"] == 0.0
    assert stale_out.debug_info["cbf_uncertainty_inflation_max_m"] > 0.9
    assert stale_out.debug_info["cbf_min_clearance_m"] < fresh_out.debug_info["cbf_min_clearance_m"]
