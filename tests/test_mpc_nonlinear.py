from __future__ import annotations

import numpy as np

from microbench.planners.mpc_nonlinear import NonlinearMpcPlanner
from microbench.types import AABBObs, AgentState, IntentObs, NeighborObs, PlannerInput, PlannerOutput


def _agent(
    pos,
    vel=(0.0, 0.0, 0.0),
    goal=(10.0, 0.0, 0.0),
    radius=0.5,
    v_max=3.0,
    a_max=2.0,
):
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
    neighbor_intents=None,
    planar=True,
    goal_dir=(1.0, 0.0, 0.0),
) -> PlannerInput:
    return PlannerInput(
        ego=ego,
        goal_dir=np.asarray(goal_dir, dtype=np.float32),
        neighbors=list(neighbors or []),
        obstacles=list(obstacles or []),
        neighbor_intents=list(neighbor_intents or []),
        dt=0.02,
        t=0.0,
        planar=planar,
    )


def _tiny_nmpc() -> NonlinearMpcPlanner:
    return NonlinearMpcPlanner(
        cfg={
            "horizon_s": 2.4,
            "horizon_steps": 5,
            "max_initializations": 4,
            "opt_iterations": 6,
        }
    )


def test_mpc_nonlinear_open_space_tracks_goal_and_emits_intent() -> None:
    ego = _agent((0.0, 0.0, 0.0))

    out = _tiny_nmpc().compute_cmd(_planner_input(ego=ego))

    assert isinstance(out, PlannerOutput)
    assert out.v_cmd[0] > 0.0
    assert abs(float(out.v_cmd[1])) < 1e-9
    assert abs(float(out.v_cmd[2])) < 1e-9
    assert np.linalg.norm(out.v_cmd - ego.vel) <= ego.a_max * 0.02 + 1e-6
    assert out.intent_out is not None
    assert out.intent_out.kind == "MPC_NONLINEAR_TRAJECTORY"
    assert out.intent_out.points.shape[0] >= 2
    assert out.debug_info["mpc_nonlinear_horizon_steps"] >= 2
    assert out.debug_info["mpc_nonlinear_solver"] == "projected_gradient"
    assert out.debug_info["mpc_nonlinear_planar"] is True


def test_mpc_nonlinear_close_head_on_optimizes_avoidance_seed() -> None:
    ego = _agent((0.0, 0.0, 0.0), vel=(2.0, 0.0, 0.0))
    neighbor = NeighborObs(
        idx=1,
        pos=np.asarray([3.2, 0.0, 0.0], dtype=np.float32),
        vel=np.asarray([-2.0, 0.0, 0.0], dtype=np.float32),
        radius=0.5,
        msg_age_sec=0.0,
        valid=True,
    )

    out = _tiny_nmpc().compute_cmd(_planner_input(ego=ego, neighbors=[neighbor]))

    assert out.debug_info["mpc_nonlinear_neighbor_count_considered"] == 1
    assert out.debug_info["mpc_nonlinear_min_swarm_clearance_m"] is not None
    assert out.debug_info["mpc_nonlinear_collision_penalty"] > 0.0
    assert out.debug_info["mpc_nonlinear_best_seed"] != "track_goal"
    assert out.debug_info["mpc_nonlinear_cost_reduction"] > 0.0
    assert abs(float(out.v_cmd[2])) > 1e-6 or out.v_cmd[0] < ego.vel[0]


def test_mpc_nonlinear_obstacle_in_path_optimizes_around_or_slows() -> None:
    ego = _agent((0.0, 0.0, 0.0), vel=(1.0, 0.0, 0.0))
    obstacle = AABBObs(
        center=np.asarray([2.4, 0.0, 0.0], dtype=np.float32),
        half=np.asarray([0.5, 0.5, 0.5], dtype=np.float32),
    )

    out = _tiny_nmpc().compute_cmd(_planner_input(ego=ego, obstacles=[obstacle]))

    assert out.debug_info["mpc_nonlinear_obstacle_count_considered"] == 1
    assert out.debug_info["mpc_nonlinear_min_obstacle_clearance_m"] is not None
    assert out.debug_info["mpc_nonlinear_obstacle_penalty"] > 0.0
    assert out.debug_info["mpc_nonlinear_cost_reduction"] > 0.0
    assert abs(float(out.v_cmd[2])) > 1e-6 or out.v_cmd[0] < ego.vel[0]


def test_mpc_nonlinear_preserves_3d_command_shape() -> None:
    ego = _agent((0.0, 0.0, 0.0), goal=(10.0, 4.0, 0.0))

    out = _tiny_nmpc().compute_cmd(
        _planner_input(
            ego=ego,
            planar=False,
            goal_dir=(0.8, 0.6, 0.0),
        )
    )

    assert out.v_cmd.shape == (3,)
    assert out.v_cmd[0] > 0.0
    assert out.v_cmd[1] > 0.0
    assert out.debug_info["mpc_nonlinear_planar"] is False
    assert np.linalg.norm(out.v_cmd - ego.vel) <= ego.a_max * 0.02 + 1e-6


def test_mpc_nonlinear_neighbor_intent_is_used_in_cost() -> None:
    ego = _agent((0.0, 0.0, 0.0), vel=(1.0, 0.0, 0.0))
    intent = IntentObs(
        sender_id=7,
        points=np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32),
        tube_radius_m=0.8,
        kind="MPC_NONLINEAR_TRAJECTORY",
        expiry_s=1.0,
        intent_age_s=0.2,
        valid=True,
        dt_plan_s=0.4,
    )
    planner = _tiny_nmpc()

    no_intent = planner.compute_cmd(_planner_input(ego=ego))
    with_intent = planner.compute_cmd(_planner_input(ego=ego, neighbor_intents=[intent]))

    assert with_intent.debug_info["mpc_nonlinear_intent_count_considered"] == 1
    assert with_intent.debug_info["mpc_nonlinear_intent_penalty"] > no_intent.debug_info["mpc_nonlinear_intent_penalty"]
