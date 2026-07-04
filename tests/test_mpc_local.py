from __future__ import annotations

import numpy as np

from microbench.planners.mpc_local import MpcLocalPlanner
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


def _planner_input(
    *,
    ego: AgentState,
    neighbors=None,
    obstacles=None,
    planar=True,
    goal_dir=(1.0, 0.0, 0.0),
) -> PlannerInput:
    return PlannerInput(
        ego=ego,
        goal_dir=np.asarray(goal_dir, dtype=np.float32),
        neighbors=list(neighbors or []),
        obstacles=list(obstacles or []),
        dt=0.02,
        t=0.0,
        planar=planar,
    )


def _tiny_mpc() -> MpcLocalPlanner:
    return MpcLocalPlanner(cfg={"candidate_samples_2d": 8, "candidate_samples_3d": 12})


def test_mpc_local_open_space_tracks_goal_and_respects_accel() -> None:
    ego = _agent((0.0, 0.0, 0.0))

    out = _tiny_mpc().compute_cmd(_planner_input(ego=ego))

    assert isinstance(out, PlannerOutput)
    assert out.v_cmd[0] > 0.0
    assert abs(float(out.v_cmd[1])) < 1e-9
    assert abs(float(out.v_cmd[2])) < 1e-9
    assert np.linalg.norm(out.v_cmd - ego.vel) <= ego.a_max * 0.02 + 1e-6
    assert out.debug_info["mpc_candidates"] > 0
    assert out.debug_info["mpc_min_pred_clearance_m"] is None


def test_mpc_local_close_head_on_decelerates() -> None:
    ego = _agent((0.0, 0.0, 0.0), vel=(2.0, 0.0, 0.0))
    neighbor = NeighborObs(
        idx=1,
        pos=np.asarray([2.0, 0.0, 0.0], dtype=np.float32),
        vel=np.asarray([-2.0, 0.0, 0.0], dtype=np.float32),
        radius=0.5,
        msg_age_sec=0.0,
        valid=True,
    )

    out = _tiny_mpc().compute_cmd(_planner_input(ego=ego, neighbors=[neighbor]))

    assert out.v_cmd[0] < ego.vel[0]
    assert np.linalg.norm(out.v_cmd - ego.vel) <= ego.a_max * 0.02 + 1e-6
    assert out.debug_info["mpc_approach_penalty"] > 0.0
    assert out.debug_info["mpc_min_pred_clearance_m"] is not None


def test_mpc_local_obstacle_in_path_decelerates() -> None:
    ego = _agent((0.0, 0.0, 0.0), vel=(1.0, 0.0, 0.0))
    obstacle = AABBObs(
        center=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        half=np.asarray([0.5, 0.5, 0.5], dtype=np.float32),
    )

    out = _tiny_mpc().compute_cmd(_planner_input(ego=ego, obstacles=[obstacle]))

    assert out.v_cmd[0] < ego.vel[0]
    assert np.linalg.norm(out.v_cmd - ego.vel) <= ego.a_max * 0.02 + 1e-6
    assert out.debug_info["mpc_obstacle_penalty"] > 0.0


def test_mpc_local_preserves_3d_command_shape() -> None:
    ego = _agent((0.0, 0.0, 0.0), goal=(10.0, 4.0, 0.0))

    out = _tiny_mpc().compute_cmd(
        _planner_input(
            ego=ego,
            planar=False,
            goal_dir=(0.8, 0.6, 0.0),
        )
    )

    assert out.v_cmd.shape == (3,)
    assert out.v_cmd[0] > 0.0
    assert out.v_cmd[1] > 0.0
    assert np.linalg.norm(out.v_cmd - ego.vel) <= ego.a_max * 0.02 + 1e-6
    assert out.debug_info["mpc_planar"] is False
