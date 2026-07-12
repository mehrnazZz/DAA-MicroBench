from __future__ import annotations

import numpy as np

from microbench.planners.dynamic_tube_dmpc import DynamicTubeDmpcPlanner
from microbench.types import AABBObs, AgentState, IntentObs, MSG_INTENT_TRAJECTORY, NeighborObs, PlannerInput, PlannerOutput


def _agent(
    pos,
    vel=(0.0, 0.0, 0.0),
    goal=(10.0, 0.0, 0.0),
    radius=0.2,
    v_max=1.0,
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
    t=0.0,
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


def _neighbor(pos=(1.0, 0.0, 0.0), vel=(-1.0, 0.0, 0.0), radius=0.2) -> NeighborObs:
    return NeighborObs(
        idx=1,
        pos=np.asarray(pos, dtype=np.float32),
        vel=np.asarray(vel, dtype=np.float32),
        radius=radius,
        msg_age_sec=0.0,
        valid=True,
    )


def _obstacle() -> AABBObs:
    return AABBObs(
        center=np.asarray([3.0, 0.0, 0.0], dtype=np.float32),
        half=np.asarray([0.35, 0.5, 0.5], dtype=np.float32),
    )


def _intent() -> IntentObs:
    return IntentObs(
        sender_id=7,
        points=np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.0, 0.0],
                [0.6, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        tube_radius_m=0.25,
        kind="DYNAMIC_TUBE_DMPC_TRAJECTORY",
        expiry_s=1.0,
        intent_age_s=0.1,
        valid=True,
        dt_plan_s=0.2,
    )


def _tiny_dynamic_tube() -> DynamicTubeDmpcPlanner:
    return DynamicTubeDmpcPlanner(
        cfg={
            "horizon_steps": 6,
            "qp_iterations": 18,
            "projection_iterations": 4,
            "tube_waypoints": 11,
            "replan_period_s": 0.2,
        }
    )


def test_dynamic_tube_dmpc_open_space_solves_condensed_qp_and_emits_intent() -> None:
    ego = _agent((0.0, 0.0, 0.0))

    out = _tiny_dynamic_tube().compute_cmd(_planner_input(ego=ego))

    assert isinstance(out, PlannerOutput)
    assert out.v_cmd[0] > 0.0
    assert abs(out.v_cmd[1]) < 1e-9
    assert abs(out.v_cmd[2]) < 1e-9
    assert out.intent_out is not None
    assert out.intent_out.kind == "DYNAMIC_TUBE_DMPC_TRAJECTORY"
    assert len(out.messages_out) == 1
    assert out.messages_out[0].kind == MSG_INTENT_TRAJECTORY
    info = out.debug_info
    assert info["dynamic_tube_dmpc_solver"] == "condensed_qp_projected_gradient"
    assert info["dynamic_tube_dmpc_qp_variables"] == 18
    assert info["dynamic_tube_dmpc_tube_constraint_count"] > 0
    assert info["dynamic_tube_dmpc_collision_constraint_count"] == 0
    assert "28-32" in info["dynamic_tube_dmpc_equations"]


def test_dynamic_tube_dmpc_risk_trigger_activates_collision_constraints() -> None:
    ego = _agent((0.0, 0.0, 0.0), vel=(1.0, 0.0, 0.0))

    out = _tiny_dynamic_tube().compute_cmd(_planner_input(ego=ego, neighbors=[_neighbor()]))

    info = out.debug_info
    assert info["dynamic_tube_dmpc_risk_triggered_activation"] is True
    assert info["dynamic_tube_dmpc_risk_agent_count"] == 1
    assert info["dynamic_tube_dmpc_first_risk_step"] is not None
    assert info["dynamic_tube_dmpc_collision_constraint_count"] > 0
    assert info["dynamic_tube_dmpc_collision_max_violation_m"] is not None


def test_dynamic_tube_dmpc_intent_only_prediction_can_trigger_risk_constraints() -> None:
    ego = _agent((0.0, 0.0, 0.0), vel=(1.0, 0.0, 0.0))

    out = _tiny_dynamic_tube().compute_cmd(_planner_input(ego=ego, neighbor_intents=[_intent()]))

    assert out.debug_info["dynamic_tube_dmpc_risk_agent_count"] == 1
    assert out.debug_info["dynamic_tube_dmpc_collision_constraint_count"] > 0


def test_dynamic_tube_dmpc_elastic_reconstruction_responds_to_obstacle_intrusion() -> None:
    ego = _agent((0.0, 0.0, 0.0))

    out = _tiny_dynamic_tube().compute_cmd(_planner_input(ego=ego, obstacles=[_obstacle()]))

    info = out.debug_info
    assert info["dynamic_tube_dmpc_tube_reconstruction_active"] is True
    assert info["dynamic_tube_dmpc_active_obstacle_count"] > 0
    assert info["dynamic_tube_dmpc_tube_max_shift_m"] > 0.0
    assert info["dynamic_tube_dmpc_tube_connected"] is True


def test_dynamic_tube_dmpc_preserves_3d_command_shape() -> None:
    ego = _agent((0.0, 0.0, 0.0), goal=(10.0, 4.0, 0.0))

    out = _tiny_dynamic_tube().compute_cmd(_planner_input(ego=ego, planar=False, goal_dir=(0.8, 0.6, 0.0)))

    assert out.v_cmd.shape == (3,)
    assert out.v_cmd[0] > 0.0
    assert out.v_cmd[1] > 0.0
    assert out.debug_info["dynamic_tube_dmpc_planar"] is False


def test_dynamic_tube_dmpc_reuses_solution_until_replan_period() -> None:
    planner = _tiny_dynamic_tube()
    ego0 = _agent((0.0, 0.0, 0.0), vel=(0.5, 0.0, 0.0))

    first = planner.compute_cmd(_planner_input(ego=ego0, t=0.0))
    ego1 = _agent((0.01, 0.0, 0.0), vel=np.asarray(first.v_cmd, dtype=np.float32))
    reused = planner.compute_cmd(_planner_input(ego=ego1, t=0.02))

    assert first.debug_info["dynamic_tube_dmpc_replanned"] is True
    assert reused.debug_info["dynamic_tube_dmpc_replanned"] is False
    assert reused.debug_info["dynamic_tube_dmpc_cached_reuse"] is True
    assert reused.debug_info["dynamic_tube_dmpc_solver_status"] == "cached_receding_qp_solution"
