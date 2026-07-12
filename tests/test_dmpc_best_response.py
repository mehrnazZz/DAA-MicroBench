from __future__ import annotations

import numpy as np

from microbench.planners.dmpc_best_response import DistributedMpcBestResponsePlanner
from microbench.types import (
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


def _neighbor() -> NeighborObs:
    return NeighborObs(
        idx=1,
        pos=np.asarray([3.0, 0.0, 0.0], dtype=np.float32),
        vel=np.asarray([-2.0, 0.0, 0.0], dtype=np.float32),
        radius=0.5,
        msg_age_sec=0.0,
        valid=True,
    )


def _intent(*, age: float = 0.1, valid: bool = True) -> IntentObs:
    return IntentObs(
        sender_id=1,
        points=np.asarray(
            [
                [3.0, 0.0, 0.0],
                [2.2, 0.0, 0.0],
                [1.4, 0.0, 0.0],
                [0.6, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        tube_radius_m=0.8,
        kind="DMPC_BEST_RESPONSE_TRAJECTORY",
        expiry_s=1.0,
        intent_age_s=age,
        valid=valid,
        dt_plan_s=0.4,
        mode="sequential_best_response:track_goal",
    )


def _tiny_dmpc() -> DistributedMpcBestResponsePlanner:
    return DistributedMpcBestResponsePlanner(
        cfg={
            "horizon_s": 2.4,
            "horizon_steps": 5,
            "max_initializations": 4,
            "opt_iterations": 5,
            "intent_trust_horizon_s": 0.5,
            "emit_agent_messages": True,
        }
    )


def test_dmpc_best_response_open_space_tracks_goal_and_emits_plan() -> None:
    ego = _agent((0.0, 0.0, 0.0))
    ctx = AgentContext(agent_id=0, method="dmpc_best_response", seed=0, priority=0)

    out = _tiny_dmpc().compute_cmd(_planner_input(ego=ego, context=ctx))

    assert isinstance(out, PlannerOutput)
    assert out.v_cmd[0] > 0.0
    assert out.intent_out is not None
    assert out.intent_out.kind == "DMPC_BEST_RESPONSE_TRAJECTORY"
    assert out.intent_out.points.shape[0] >= 2
    assert out.messages_out
    assert out.messages_out[0].kind == MSG_INTENT_TRAJECTORY
    assert out.debug_info["dmpc_best_response_algorithm"] == "distributed_best_response_nonlinear_mpc"
    assert out.debug_info["dmpc_best_response_plan_version"] == 1
    assert out.debug_info["dmpc_best_response_planar"] is True


def test_dmpc_best_response_uses_neighbor_intent_as_coupled_prediction() -> None:
    ego = _agent((0.0, 0.0, 0.0), vel=(2.0, 0.0, 0.0))
    planner = _tiny_dmpc()

    out = planner.compute_cmd(_planner_input(ego=ego, neighbors=[_neighbor()], neighbor_intents=[_intent()]))

    info = out.debug_info
    assert info["dmpc_best_response_neighbor_count_considered"] == 1
    assert info["dmpc_best_response_neighbor_intent_count_considered"] == 1
    assert info["dmpc_best_response_intent_primary_predictions"] > 0
    assert info["dmpc_best_response_coupled_constraints"] > 0
    assert info["dmpc_best_response_min_coupled_clearance_m"] is not None
    assert info["dmpc_best_response_best_seed"] != "track_goal"
    assert abs(float(out.v_cmd[2])) > 1e-6 or out.v_cmd[0] < ego.vel[0]


def test_dmpc_best_response_falls_back_for_stale_intent() -> None:
    ego = _agent((0.0, 0.0, 0.0), vel=(2.0, 0.0, 0.0))

    out = _tiny_dmpc().compute_cmd(
        _planner_input(ego=ego, neighbors=[_neighbor()], neighbor_intents=[_intent(age=1.0)])
    )

    info = out.debug_info
    assert info["dmpc_best_response_neighbor_intent_count_considered"] == 0
    assert info["dmpc_best_response_stale_intent_count"] == 1
    assert info["dmpc_best_response_fallback_cv_predictions"] > 0


def test_dmpc_best_response_preserves_3d_command_shape() -> None:
    ego = _agent((0.0, 0.0, 0.0), goal=(10.0, 4.0, 0.0))

    out = _tiny_dmpc().compute_cmd(
        _planner_input(
            ego=ego,
            planar=False,
            goal_dir=(0.8, 0.6, 0.0),
        )
    )

    assert out.v_cmd.shape == (3,)
    assert out.v_cmd[0] > 0.0
    assert out.v_cmd[1] > 0.0
    assert out.debug_info["dmpc_best_response_planar"] is False
    assert np.linalg.norm(out.v_cmd - ego.vel) <= ego.a_max * 0.02 + 1e-6
