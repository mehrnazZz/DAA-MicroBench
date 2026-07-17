from __future__ import annotations

import numpy as np

from microbench.planners.rmader import RmaderPlanner
from microbench.types import (
    AABBObs,
    AgentContext,
    AgentState,
    IntentObs,
    MSG_INTENT_TRAJECTORY,
    NeighborObs,
    PlannerInput,
    PlannerOutput,
)


def _agent(
    pos,
    vel=(0.0, 0.0, 0.0),
    goal=(10.0, 0.0, 0.0),
    radius=0.5,
    v_max=3.0,
    a_max=2.0,
    idx=0,
):
    return AgentState(
        idx=idx,
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
    neighbor_intents=None,
    planar=True,
    goal_dir=(1.0, 0.0, 0.0),
    context: AgentContext | None = None,
) -> PlannerInput:
    return PlannerInput(
        ego=ego,
        goal_dir=np.asarray(goal_dir, dtype=np.float32),
        neighbors=list(neighbors or []),
        neighbor_intents=list(neighbor_intents or []),
        dt=0.02,
        t=0.0,
        agent_context=context,
        planar=planar,
    )


def _neighbor(pos=(4.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0)) -> NeighborObs:
    return NeighborObs(
        idx=1,
        pos=np.asarray(pos, dtype=np.float32),
        vel=np.asarray(vel, dtype=np.float32),
        radius=0.5,
        msg_age_sec=0.0,
        valid=True,
    )


def _intent() -> IntentObs:
    return IntentObs(
        sender_id=7,
        points=np.asarray(
            [
                [4.0, 0.0, 0.0],
                [3.6, 0.0, 0.0],
                [3.2, 0.0, 0.0],
                [2.8, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        tube_radius_m=0.8,
        kind="RMADER_MINVO_TRAJECTORY",
        expiry_s=1.0,
        intent_age_s=0.1,
        valid=True,
        dt_plan_s=0.4,
    )


def _tiny_rmader() -> RmaderPlanner:
    return RmaderPlanner(
        cfg={
            "horizon_s": 2.4,
            "control_points": 8,
            "samples_per_interval": 3,
            "max_initializations": 4,
            "opt_iterations": 4,
            "hard_projection_iterations": 5,
            "jerk_limit_mps3": 100.0,
        }
    )


def test_rmader_open_space_emits_minvo_intent_and_two_step_publication() -> None:
    ego = _agent((0.0, 0.0, 0.0))
    ctx = AgentContext(agent_id=0, method="rmader", seed=0, priority=0)

    out = _tiny_rmader().compute_cmd(_planner_input(ego=ego, context=ctx))

    assert isinstance(out, PlannerOutput)
    assert out.v_cmd[0] > 0.0
    assert out.intent_out is not None
    assert out.intent_out.kind == "RMADER_MINVO_TRAJECTORY"
    assert out.intent_out.points.shape[0] >= 2
    assert len(out.messages_out) == 2
    assert {msg.payload["publication_stage"] for msg in out.messages_out} == {"candidate", "committed"}
    assert all(msg.kind == MSG_INTENT_TRAJECTORY for msg in out.messages_out)
    assert out.debug_info["rmader_minvo_intervals"] >= 4
    assert out.debug_info["rmader_minvo_control_points_per_interval"] == 4
    assert out.debug_info["rmader_delay_check_passed"] is True
    assert out.debug_info["rmader_plan_version"] == 1
    assert out.debug_info["rmader_planar"] is True


def test_rmader_open_space_uses_meaningful_local_horizon_from_rest() -> None:
    ego = _agent((0.0, 0.0, 0.0), goal=(50.0, 0.0, 0.0))

    out = _tiny_rmader().compute_cmd(_planner_input(ego=ego))

    assert out.debug_info["rmader_path_length_m"] >= 4.0
    assert np.linalg.norm(out.v_cmd - ego.vel) <= ego.a_max * 0.02 + 1e-6


def test_rmader_far_neighbor_uses_hard_minvo_hyperplanes() -> None:
    ego = _agent((0.0, 0.0, 0.0))
    planner = _tiny_rmader()

    out = planner.compute_cmd(_planner_input(ego=ego, neighbors=[_neighbor(pos=(8.0, 0.0, 0.0))]))

    info = out.debug_info
    assert info["rmader_neighbor_count_considered"] == 1
    assert info["rmader_hard_constraint_count"] >= info["rmader_minvo_intervals"]
    assert info["rmader_candidate_hard_constraint_ok"] is True
    assert info["rmader_delay_check_passed"] is True
    assert info["rmader_max_hyperplane_violation_m"] <= 0.08
    assert info["rmader_min_hyperplane_gap_m"] is not None


def test_rmader_close_conflict_delay_check_falls_back_to_braking_plan() -> None:
    ego = _agent((0.0, 0.0, 0.0))
    planner = _tiny_rmader()

    out = planner.compute_cmd(_planner_input(ego=ego, neighbors=[_neighbor(pos=(3.0, 0.0, 0.0))]))

    info = out.debug_info
    assert info["rmader_candidate_hard_constraint_ok"] is False
    assert info["rmader_delay_check_passed"] is False
    assert info["rmader_delay_check_fallback"] == "braking_trajectory"
    assert info["rmader_used_topology"] == "delay_check_brake"
    assert np.linalg.norm(out.v_cmd) <= ego.a_max * 0.02 + 1e-6


def test_rmader_uses_intent_only_hulls() -> None:
    ego = _agent((0.0, 0.0, 0.0))
    planner = _tiny_rmader()

    out = planner.compute_cmd(_planner_input(ego=ego, neighbor_intents=[_intent()]))

    info = out.debug_info
    assert info["rmader_intent_count_considered"] == 1
    assert info["rmader_hard_constraint_count"] >= info["rmader_minvo_intervals"]
    assert info["rmader_candidate_max_hyperplane_violation_m"] is not None


def test_rmader_static_obstacle_broadphase_filters_far_obstacles() -> None:
    planner = _tiny_rmader()
    ego = _agent((0.0, 0.0, 0.0), goal=(50.0, 0.0, 0.0))
    inp = _planner_input(ego=ego, planar=False)
    cp = planner._control_polygon(
        inp, planner._local_target(inp), np.zeros(3, dtype=np.float32), "direct"
    ).control_points
    minvo = planner._minvo_intervals(cp)
    far = AABBObs(center=np.asarray([30.0, 0.0, 0.0], dtype=np.float32), half=np.asarray([1.0, 1.0, 1.0]))
    near = AABBObs(center=np.asarray([7.0, 0.0, 0.0], dtype=np.float32), half=np.asarray([1.0, 1.0, 1.0]))

    far_hulls = planner._build_interval_hulls(
        PlannerInput(
            ego=ego,
            goal_dir=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            neighbors=[],
            obstacles=[far],
            dt=0.02,
            t=0.0,
            planar=False,
        ),
        minvo.shape[0],
        planner._segment_dt(),
        own_minvo=minvo,
    )
    near_hulls = planner._build_interval_hulls(
        PlannerInput(
            ego=ego,
            goal_dir=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            neighbors=[],
            obstacles=[near],
            dt=0.02,
            t=0.0,
            planar=False,
        ),
        minvo.shape[0],
        planner._segment_dt(),
        own_minvo=minvo,
    )

    assert [h.source_kind for h in far_hulls] == []
    assert any(h.source_kind == "obstacle_aabb" for h in near_hulls)


def test_rmader_preserves_3d_command_shape() -> None:
    ego = _agent((0.0, 0.0, 0.0), goal=(10.0, 4.0, 0.0))

    out = _tiny_rmader().compute_cmd(
        _planner_input(
            ego=ego,
            planar=False,
            goal_dir=(0.8, 0.6, 0.0),
        )
    )

    assert out.v_cmd.shape == (3,)
    assert out.v_cmd[0] > 0.0
    assert out.v_cmd[1] > 0.0
    assert out.debug_info["rmader_planar"] is False
    assert out.debug_info["rmader_delay_check_passed"] is True
    assert np.linalg.norm(out.v_cmd - ego.vel) <= ego.a_max * 0.02 + 1e-6


def test_rmader_reuses_committed_plan_until_replan_period() -> None:
    ego = _agent((0.0, 0.0, 0.0), goal=(20.0, 0.0, 0.0))
    ctx = AgentContext(agent_id=0, method="rmader", seed=0, priority=0)
    planner = RmaderPlanner(
        cfg={
            "horizon_s": 2.4,
            "control_points": 8,
            "samples_per_interval": 2,
            "replan_period_s": 0.2,
            "max_initializations": 2,
            "opt_iterations": 2,
            "hard_projection_iterations": 2,
            "jerk_limit_mps3": 100.0,
        }
    )

    first = planner.compute_cmd(_planner_input(ego=ego, context=ctx))
    ego2 = _agent((0.02, 0.0, 0.0), vel=first.v_cmd, goal=(20.0, 0.0, 0.0))
    reused = planner.compute_cmd(
        PlannerInput(
            ego=ego2,
            goal_dir=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            neighbors=[],
            neighbor_intents=[],
            dt=0.02,
            t=0.02,
            agent_context=ctx,
            planar=True,
        )
    )

    assert first.debug_info["rmader_replanned"] is True
    assert reused.debug_info["rmader_replanned"] is False
    assert reused.debug_info["rmader_cached_reuse"] is True
    assert reused.debug_info["rmader_solver_status"] == "cached_committed_minvo_plan"
