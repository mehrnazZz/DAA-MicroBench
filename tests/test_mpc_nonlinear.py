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
    t: float = 0.0,
) -> PlannerInput:
    return PlannerInput(
        ego=ego,
        goal_dir=np.asarray(goal_dir, dtype=np.float32),
        neighbors=list(neighbors or []),
        obstacles=list(obstacles or []),
        neighbor_intents=list(neighbor_intents or []),
        dt=0.02,
        t=float(t),
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


def test_mpc_nonlinear_reuses_receding_solution_until_replan_period() -> None:
    planner = NonlinearMpcPlanner(
        cfg={
            "horizon_s": 2.4,
            "horizon_steps": 5,
            "max_initializations": 2,
            "opt_iterations": 2,
            "replan_period_s": 0.1,
        }
    )
    ego0 = _agent((0.0, 0.0, 0.0), vel=(0.5, 0.0, 0.0))

    first = planner.compute_cmd(_planner_input(ego=ego0, t=0.0))
    ego1 = _agent((0.01, 0.0, 0.0), vel=np.asarray(first.v_cmd, dtype=np.float32))
    reused = planner.compute_cmd(_planner_input(ego=ego1, t=0.02))
    expired = planner.compute_cmd(_planner_input(ego=ego1, t=0.12))

    assert first.debug_info["mpc_nonlinear_replanned"] is True
    assert reused.debug_info["mpc_nonlinear_replanned"] is False
    assert reused.debug_info["mpc_nonlinear_cached_reuse"] is True
    assert reused.debug_info["mpc_nonlinear_solver"] == "cached_receding_mpc"
    assert expired.debug_info["mpc_nonlinear_replanned"] is True


def test_mpc_nonlinear_cached_reuse_shifts_by_plan_interval() -> None:
    planner = NonlinearMpcPlanner(
        cfg={
            "horizon_s": 2.4,
            "horizon_steps": 4,
            "replan_period_s": 1.0,
        }
    )
    planner._cached_controls = np.asarray(
        [
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    planner._last_controls = planner._cached_controls.copy()
    planner._last_label = "probe"
    planner._last_replan_t = 0.0
    ego = _agent((0.0, 0.0, 0.0), v_max=10.0, a_max=4.0)

    early = planner.compute_cmd(_planner_input(ego=ego, planar=False, t=0.02))

    assert early.debug_info["mpc_nonlinear_replanned"] is False
    assert early.debug_info["mpc_nonlinear_cached_reuse"] is True
    assert float(early.v_cmd[0]) > 0.0
    assert abs(float(early.v_cmd[2])) < 1e-9

    planner._cached_controls = np.asarray(
        [
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    planner._last_controls = planner._cached_controls.copy()
    planner._last_replan_t = 0.0
    shifted = planner.compute_cmd(_planner_input(ego=ego, planar=False, t=planner._dt() + 0.01))

    assert shifted.debug_info["mpc_nonlinear_replanned"] is False
    assert shifted.debug_info["mpc_nonlinear_cached_reuse"] is True
    assert abs(float(shifted.v_cmd[0])) < 1e-9
    assert float(shifted.v_cmd[2]) > 0.0


def test_mpc_nonlinear_preview_command_tracks_planned_position() -> None:
    planner = NonlinearMpcPlanner(
        cfg={
            "horizon_s": 2.4,
            "horizon_steps": 4,
            "command_preview_s": 0.6,
        }
    )
    ego = _agent((0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), v_max=10.0, a_max=4.0)
    result = planner._result_from_breakdown(
        seed=type("Seed", (), {"label": "probe"})(),
        controls=np.zeros((4, 3), dtype=np.float32),
        initial_cost=0.0,
        final={
            "total": 0.0,
            "positions": np.asarray(
                [
                    [0.0, 0.0, 2.0],
                    [0.0, 0.0, 4.0],
                    [0.0, 0.0, 6.0],
                    [0.0, 0.0, 8.0],
                ],
                dtype=np.float32,
            ),
            "velocities": np.zeros((4, 3), dtype=np.float32),
            "tracking_cost": 0.0,
            "terminal_cost": 0.0,
            "control_cost": 0.0,
            "jerk_cost": 0.0,
            "dynamic_penalty": 0.0,
            "collision_penalty": 0.0,
            "obstacle_penalty": 0.0,
            "intent_penalty": 0.0,
            "min_swarm_clearance_m": None,
            "min_obstacle_clearance_m": None,
            "predicted_swarm_conflict": False,
            "predicted_obstacle_conflict": False,
        },
        iterations=0,
        solver="probe",
        status="probe",
    )

    desired, mode, preview_step = planner._command_from_plan(_planner_input(ego=ego, planar=False), result, ego.vel)

    assert mode == "planned_position_preview"
    assert preview_step == 1
    assert abs(float(desired[0])) < 1e-9
    assert float(desired[2]) > 0.0


def test_mpc_nonlinear_goal_capture_lifts_near_goal_progress() -> None:
    planner = NonlinearMpcPlanner(
        cfg={
            "goal_capture_radius_m": 4.5,
            "goal_capture_time_s": 1.0,
        }
    )
    ego = _agent((0.0, 0.0, 0.0), goal=(3.0, 0.0, 0.0), v_max=4.0, a_max=2.0)
    result = planner._result_from_breakdown(
        seed=type("Seed", (), {"label": "probe"})(),
        controls=np.zeros((planner._steps(), 3), dtype=np.float32),
        initial_cost=0.0,
        final={
            "total": 0.0,
            "positions": np.zeros((planner._steps(), 3), dtype=np.float32),
            "velocities": np.zeros((planner._steps(), 3), dtype=np.float32),
            "tracking_cost": 0.0,
            "terminal_cost": 0.0,
            "control_cost": 0.0,
            "jerk_cost": 0.0,
            "dynamic_penalty": 0.0,
            "collision_penalty": 0.0,
            "obstacle_penalty": 0.0,
            "intent_penalty": 0.0,
            "min_swarm_clearance_m": 1.0,
            "min_obstacle_clearance_m": None,
            "predicted_swarm_conflict": False,
            "predicted_obstacle_conflict": False,
        },
        iterations=0,
        solver="probe",
        status="probe",
    )

    capture = planner._goal_capture_command(_planner_input(ego=ego), result, np.zeros(3, dtype=np.float32))

    assert capture is not None
    assert float(capture[0]) > 0.0
