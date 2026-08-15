from __future__ import annotations

import numpy as np

from microbench.planners.ego_swarm_opt import EgoSwarmOptimizingPlanner
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


def _tiny_ego_swarm_opt() -> EgoSwarmOptimizingPlanner:
    return EgoSwarmOptimizingPlanner(
        cfg={
            "horizon_s": 2.4,
            "rollout_dt_s": 0.4,
            "control_points": 6,
            "curve_samples": 9,
            "max_initializations": 5,
            "opt_iterations": 8,
            "offset_scales_m": [0.0, 2.5, 4.0],
            "vertical_offset_scales_m": [2.0],
        }
    )


def test_ego_swarm_opt_open_space_tracks_goal_and_emits_intent() -> None:
    ego = _agent((0.0, 0.0, 0.0))

    out = _tiny_ego_swarm_opt().compute_cmd(_planner_input(ego=ego))

    assert isinstance(out, PlannerOutput)
    assert out.v_cmd[0] > 0.0
    assert abs(float(out.v_cmd[1])) < 1e-9
    assert abs(float(out.v_cmd[2])) < 1e-9
    assert np.linalg.norm(out.v_cmd - ego.vel) <= ego.a_max * 0.02 + 1e-6
    assert out.intent_out is not None
    assert out.intent_out.kind == "EGO_SWARM_OPT_TRAJECTORY"
    assert out.intent_out.points.shape[0] >= 2
    assert out.debug_info["ego_swarm_opt_control_points"] >= 5
    assert out.debug_info["ego_swarm_opt_solver"] == "projected_gradient"
    assert out.debug_info["ego_swarm_opt_planar"] is True


def test_ego_swarm_opt_can_reuse_receding_cached_plan_between_replans() -> None:
    ego = _agent((0.0, 0.0, 0.0))
    planner = EgoSwarmOptimizingPlanner(
        cfg={
            "horizon_s": 2.4,
            "rollout_dt_s": 0.4,
            "control_points": 5,
            "curve_samples": 5,
            "max_initializations": 1,
            "opt_iterations": 1,
            "replan_period_s": 0.35,
        }
    )

    first = planner.compute_cmd(_planner_input(ego=ego, t=0.0))
    moved_ego = _agent((0.02, 0.0, 0.0), vel=first.v_cmd)
    reused = planner.compute_cmd(_planner_input(ego=moved_ego, t=0.02))

    assert first.debug_info["ego_swarm_opt_replanned"] is True
    assert reused.debug_info["ego_swarm_opt_replanned"] is False
    assert reused.debug_info["ego_swarm_opt_cached_reuse"] is True
    assert reused.debug_info["ego_swarm_opt_initializations"] == 0
    assert reused.debug_info["ego_swarm_opt_replan_period_s"] == 0.35
    assert reused.debug_info["ego_swarm_opt_plan_age_s"] > 0.0
    assert reused.intent_out is not None
    assert np.allclose(reused.intent_out.points[0], moved_ego.pos)


def test_ego_swarm_opt_close_head_on_optimizes_deconfliction_topology() -> None:
    ego = _agent((0.0, 0.0, 0.0), vel=(2.0, 0.0, 0.0))
    neighbor = NeighborObs(
        idx=1,
        pos=np.asarray([3.2, 0.0, 0.0], dtype=np.float32),
        vel=np.asarray([-2.0, 0.0, 0.0], dtype=np.float32),
        radius=0.5,
        msg_age_sec=0.0,
        valid=True,
    )

    out = _tiny_ego_swarm_opt().compute_cmd(_planner_input(ego=ego, neighbors=[neighbor]))

    assert out.debug_info["ego_swarm_opt_neighbor_count_considered"] == 1
    assert out.debug_info["ego_swarm_opt_min_swarm_clearance_m"] is not None
    assert out.debug_info["ego_swarm_opt_swarm_penalty"] > 0.0
    assert out.debug_info["ego_swarm_opt_best_topology"] != "direct"
    assert out.debug_info["ego_swarm_opt_cost_reduction"] > 0.0
    assert abs(float(out.v_cmd[2])) > 1e-6 or out.v_cmd[0] < ego.vel[0]


def test_ego_swarm_opt_velocity_guard_projects_immediate_track_conflict() -> None:
    ego = _agent((0.0, 0.0, 0.0), vel=(2.0, 0.0, 0.0))
    neighbor = NeighborObs(
        idx=1,
        pos=np.asarray([1.2, 0.0, 0.0], dtype=np.float32),
        vel=np.asarray([-2.0, 0.0, 0.0], dtype=np.float32),
        radius=0.5,
        msg_age_sec=0.0,
        valid=True,
    )
    planner = EgoSwarmOptimizingPlanner(
        cfg={
            "horizon_s": 2.4,
            "rollout_dt_s": 0.4,
            "control_points": 5,
            "curve_samples": 5,
            "max_initializations": 1,
            "opt_iterations": 1,
            "velocity_guard_enabled": True,
            "velocity_guard_iterations": 3,
        }
    )

    out = planner.compute_cmd(_planner_input(ego=ego, neighbors=[neighbor]))

    assert out.debug_info["ego_swarm_opt_velocity_guard_adjusted"] is True
    assert out.debug_info["ego_swarm_opt_velocity_guard_constraint_count"] > 0
    assert out.debug_info["ego_swarm_opt_velocity_guard_min_clearance_m"] < 0.0
    assert out.debug_info["ego_swarm_opt_velocity_guard_brake_applied"] is True
    assert out.v_cmd[0] < ego.vel[0]
    assert out.intent_out is not None
    assert np.isclose(out.intent_out.points[1, 0], ego.pos[0] + out.v_cmd[0] * out.intent_out.dt_plan_s)


def test_ego_swarm_opt_obstacle_in_path_optimizes_around_or_slows() -> None:
    ego = _agent((0.0, 0.0, 0.0), vel=(1.0, 0.0, 0.0))
    obstacle = AABBObs(
        center=np.asarray([2.4, 0.0, 0.0], dtype=np.float32),
        half=np.asarray([0.5, 0.5, 0.5], dtype=np.float32),
    )

    out = _tiny_ego_swarm_opt().compute_cmd(_planner_input(ego=ego, obstacles=[obstacle]))

    assert out.debug_info["ego_swarm_opt_obstacle_count_considered"] == 1
    assert out.debug_info["ego_swarm_opt_min_obstacle_clearance_m"] is not None
    assert out.debug_info["ego_swarm_opt_obstacle_penalty"] > 0.0
    assert out.debug_info["ego_swarm_opt_cost_reduction"] > 0.0
    assert abs(float(out.v_cmd[2])) > 1e-6 or out.v_cmd[0] < ego.vel[0]


def test_ego_swarm_opt_preserves_3d_command_shape() -> None:
    ego = _agent((0.0, 0.0, 0.0), goal=(10.0, 4.0, 0.0))

    out = _tiny_ego_swarm_opt().compute_cmd(
        _planner_input(
            ego=ego,
            planar=False,
            goal_dir=(0.8, 0.6, 0.0),
        )
    )

    assert out.v_cmd.shape == (3,)
    assert out.v_cmd[0] > 0.0
    assert out.v_cmd[1] > 0.0
    assert out.debug_info["ego_swarm_opt_planar"] is False
    assert np.linalg.norm(out.v_cmd - ego.vel) <= ego.a_max * 0.02 + 1e-6


def test_ego_swarm_opt_neighbor_intent_is_used_in_optimized_cost() -> None:
    ego = _agent((0.0, 0.0, 0.0), vel=(1.0, 0.0, 0.0))
    intent = IntentObs(
        sender_id=7,
        points=np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32),
        tube_radius_m=0.8,
        kind="EGO_SWARM_OPT_TRAJECTORY",
        expiry_s=1.0,
        intent_age_s=0.2,
        valid=True,
        dt_plan_s=0.4,
    )
    planner = _tiny_ego_swarm_opt()

    no_intent = planner.compute_cmd(_planner_input(ego=ego))
    with_intent = planner.compute_cmd(_planner_input(ego=ego, neighbor_intents=[intent]))

    assert with_intent.debug_info["ego_swarm_opt_intent_count_considered"] == 1
    assert with_intent.debug_info["ego_swarm_opt_min_swarm_clearance_m"] is not None
    assert with_intent.debug_info["ego_swarm_opt_swarm_penalty"] > no_intent.debug_info["ego_swarm_opt_swarm_penalty"]
