from __future__ import annotations

import numpy as np

from microbench.planners.bvc_tube_dmpc import BvcTubeDmpcPlanner
from microbench.types import AABBObs, AgentState, IntentObs, MSG_INTENT_TRAJECTORY, NeighborObs, PlannerInput, PlannerOutput


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
                [3.8, 0.0, 0.0],
                [3.6, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        tube_radius_m=0.75,
        kind="BVC_TUBE_DMPC_TRAJECTORY",
        expiry_s=1.0,
        intent_age_s=0.1,
        valid=True,
        dt_plan_s=0.4,
    )


def _tiny_bvc() -> BvcTubeDmpcPlanner:
    return BvcTubeDmpcPlanner(
        cfg={
            "horizon_steps": 5,
            "max_initializations": 4,
            "opt_iterations": 4,
            "projection_iterations": 5,
        }
    )


def test_bvc_tube_dmpc_open_space_tracks_goal_and_emits_intent() -> None:
    ego = _agent((0.0, 0.0, 0.0))

    out = _tiny_bvc().compute_cmd(_planner_input(ego=ego))

    assert isinstance(out, PlannerOutput)
    assert out.v_cmd[0] > 0.0
    assert out.intent_out is not None
    assert out.intent_out.kind == "BVC_TUBE_DMPC_TRAJECTORY"
    assert out.intent_out.points.shape[0] >= 2
    assert len(out.messages_out) == 1
    assert out.messages_out[0].kind == MSG_INTENT_TRAJECTORY
    assert out.debug_info["bvc_tube_dmpc_hard_cell_ok"] is True
    assert out.debug_info["bvc_tube_dmpc_fallback"] == "none"
    assert out.debug_info["bvc_tube_dmpc_planar"] is True


def test_bvc_tube_dmpc_far_neighbor_uses_buffered_cell_constraints() -> None:
    ego = _agent((0.0, 0.0, 0.0))

    out = _tiny_bvc().compute_cmd(_planner_input(ego=ego, neighbors=[_neighbor()], planar=False))

    info = out.debug_info
    assert info["bvc_tube_dmpc_neighbor_count_considered"] == 1
    assert info["bvc_tube_dmpc_neighbor_constraint_count"] >= info["bvc_tube_dmpc_horizon_steps"]
    assert info["bvc_tube_dmpc_candidate_hard_cell_ok"] is True
    assert info["bvc_tube_dmpc_hard_cell_ok"] is True
    assert info["bvc_tube_dmpc_max_cell_violation_m"] <= 0.04
    assert info["bvc_tube_dmpc_min_cell_slack_m"] is not None


def test_bvc_tube_dmpc_intent_only_neighbor_builds_tube_constraints() -> None:
    ego = _agent((0.0, 0.0, 0.0))

    out = _tiny_bvc().compute_cmd(_planner_input(ego=ego, neighbor_intents=[_intent()], planar=False))

    info = out.debug_info
    assert info["bvc_tube_dmpc_intent_count_considered"] == 1
    assert info["bvc_tube_dmpc_intent_constraint_count"] >= info["bvc_tube_dmpc_horizon_steps"]
    assert info["bvc_tube_dmpc_candidate_max_cell_violation_m"] is not None


def test_bvc_tube_dmpc_obstacle_constraints_are_hard_tube_boundaries() -> None:
    ego = _agent((0.0, 0.0, 0.0), vel=(0.5, 0.0, 0.0))
    obstacle = AABBObs(
        center=np.asarray([3.2, 0.0, 0.0], dtype=np.float32),
        half=np.asarray([0.35, 0.5, 0.5], dtype=np.float32),
    )

    out = _tiny_bvc().compute_cmd(_planner_input(ego=ego, obstacles=[obstacle]))

    info = out.debug_info
    assert info["bvc_tube_dmpc_obstacle_count_considered"] == 1
    assert info["bvc_tube_dmpc_obstacle_constraint_count"] >= info["bvc_tube_dmpc_horizon_steps"]
    assert info["bvc_tube_dmpc_max_cell_violation_m"] <= 0.04
    assert np.linalg.norm(out.v_cmd - ego.vel) <= ego.a_max * 0.02 + 1e-6


def test_bvc_tube_dmpc_preserves_3d_command_shape() -> None:
    ego = _agent((0.0, 0.0, 0.0), goal=(10.0, 4.0, 0.0))

    out = _tiny_bvc().compute_cmd(
        _planner_input(
            ego=ego,
            planar=False,
            goal_dir=(0.8, 0.6, 0.0),
        )
    )

    assert out.v_cmd.shape == (3,)
    assert out.v_cmd[0] > 0.0
    assert out.v_cmd[1] > 0.0
    assert out.debug_info["bvc_tube_dmpc_planar"] is False
    assert np.linalg.norm(out.v_cmd - ego.vel) <= ego.a_max * 0.02 + 1e-6


def test_bvc_tube_dmpc_reuses_committed_tube_until_replan_period() -> None:
    planner = _tiny_bvc()
    ego0 = _agent((0.0, 0.0, 0.0), vel=(0.5, 0.0, 0.0))

    first = planner.compute_cmd(_planner_input(ego=ego0, neighbors=[_neighbor()], planar=False, t=0.0))
    ego1 = _agent((0.01, 0.0, 0.0), vel=np.asarray(first.v_cmd, dtype=np.float32))
    reused = planner.compute_cmd(_planner_input(ego=ego1, neighbors=[_neighbor()], planar=False, t=0.02))

    assert first.debug_info["bvc_tube_dmpc_replanned"] is True
    assert reused.debug_info["bvc_tube_dmpc_replanned"] is False
    assert reused.debug_info["bvc_tube_dmpc_cached_reuse"] is True
    assert reused.debug_info["bvc_tube_dmpc_solver_status"] == "cached_receding_tube"
