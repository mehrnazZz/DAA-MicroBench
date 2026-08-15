from __future__ import annotations

import numpy as np

from microbench.planners.bvc_tube_dmpc import BvcTubeDmpcPlanner
from microbench.planners.dmpc_best_response import DistributedMpcBestResponsePlanner
from microbench.planners.dynamic_tube_dmpc import DynamicTubeDmpcPlanner
from microbench.planners.ego_swarm import EgoSwarmPlanner
from microbench.planners.ego_swarm_opt import EgoSwarmOptimizingPlanner
from microbench.planners.locality import select_local_traffic
from microbench.planners.mpc_nonlinear import NonlinearMpcPlanner
from microbench.planners.rmader import RmaderPlanner
from microbench.types import AgentContext, AgentState, IntentObs, NeighborObs, PlannerInput


def _agent() -> AgentState:
    return AgentState(
        idx=0,
        pos=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        vel=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        goal=np.asarray([12.0, 0.0, 0.0], dtype=np.float32),
        radius=0.35,
        v_max=2.0,
        a_max=3.0,
    )


def _neighbor(idx: int, x: float) -> NeighborObs:
    return NeighborObs(
        idx=idx,
        pos=np.asarray([x, 0.0, 0.0], dtype=np.float32),
        vel=np.asarray([-0.2, 0.0, 0.0], dtype=np.float32),
        radius=0.35,
        msg_age_sec=0.0,
        valid=True,
    )


def _intent(sender_id: int, x: float, *, age: float = 0.1, valid: bool = True) -> IntentObs:
    return IntentObs(
        sender_id=sender_id,
        points=np.asarray(
            [
                [x, 0.0, 0.0],
                [x + 0.2, 0.0, 0.0],
                [x + 0.4, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        tube_radius_m=0.45,
        kind="TEST_TRAJECTORY",
        expiry_s=1.0,
        intent_age_s=age,
        valid=valid,
        dt_plan_s=0.2,
    )


def _planner_input(*, max_intents_fixture: bool = True) -> PlannerInput:
    intents = [
        _intent(1, 1.0, age=0.4),
        _intent(1, 1.1, age=0.05),
        _intent(2, 2.0),
        _intent(3, 9.0),
        _intent(4, 1.4),
        _intent(5, 3.5),
        _intent(6, 0.8, valid=False),
    ]
    if not max_intents_fixture:
        intents = intents[:3]
    return PlannerInput(
        ego=_agent(),
        goal_dir=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        neighbors=[_neighbor(1, 1.4), _neighbor(2, 2.4), _neighbor(3, 3.4), _neighbor(4, 4.4)],
        neighbor_intents=intents,
        dt=0.05,
        t=0.0,
        planar=False,
    )


def test_select_local_traffic_keeps_neighbors_and_nearest_intent_extras() -> None:
    planner_input = _planner_input()

    traffic = select_local_traffic(planner_input, max_neighbors=2, max_intents=3)

    assert [n.idx for n in traffic.neighbors] == [1, 2]
    assert set(traffic.intents_by_sender) == {1, 2, 4}
    assert traffic.intents_by_sender[1].intent_age_s == 0.05
    assert [intent.sender_id for intent in traffic.intent_only] == [4]
    assert traffic.input_valid_intent_count == 5
    assert traffic.selected_intent_count == 3
    assert traffic.pruned_intent_count == 2


def test_select_local_traffic_includes_associated_intents_even_when_budget_is_smaller() -> None:
    planner_input = _planner_input()

    traffic = select_local_traffic(planner_input, max_neighbors=3, max_intents=1)

    assert [n.idx for n in traffic.neighbors] == [1, 2, 3]
    assert set(traffic.intents_by_sender) == {1, 2, 3}
    assert traffic.intent_only == ()
    assert traffic.selected_intent_count == 3


def test_ego_swarm_debug_reports_pruned_local_intents() -> None:
    planner = EgoSwarmPlanner(
        cfg={
            "max_neighbors": 2,
            "max_intents": 3,
            "horizon_s": 1.2,
            "rollout_dt_s": 0.4,
            "max_candidates": 8,
            "offset_scales_m": [0.0, 2.0],
            "vertical_offset_scales_m": [1.5],
        }
    )

    out = planner.compute_cmd(_planner_input())
    info = out.debug_info

    assert info["ego_swarm_neighbor_count_considered"] == 2
    assert info["ego_swarm_intent_count_available"] == 5
    assert info["ego_swarm_intent_count_considered"] == 3
    assert info["ego_swarm_intent_count_pruned"] == 2


def test_ego_swarm_opt_debug_reports_pruned_local_intents() -> None:
    planner = EgoSwarmOptimizingPlanner(
        cfg={
            "max_neighbors": 2,
            "max_intents": 3,
            "control_points": 5,
            "curve_samples": 5,
            "max_initializations": 1,
            "opt_iterations": 1,
            "offset_scales_m": [0.0],
            "vertical_offset_scales_m": [1.5],
        }
    )

    out = planner.compute_cmd(_planner_input())
    info = out.debug_info

    assert info["ego_swarm_opt_neighbor_count_considered"] == 2
    assert info["ego_swarm_opt_intent_count_available"] == 5
    assert info["ego_swarm_opt_intent_count_considered"] == 3
    assert info["ego_swarm_opt_intent_count_pruned"] == 2


def test_mpc_nonlinear_debug_reports_pruned_local_intents() -> None:
    planner = NonlinearMpcPlanner(
        cfg={
            "max_neighbors": 2,
            "max_intents": 3,
            "horizon_steps": 3,
            "max_initializations": 1,
            "opt_iterations": 1,
        }
    )

    out = planner.compute_cmd(_planner_input())
    info = out.debug_info

    assert info["mpc_nonlinear_neighbor_count_considered"] == 2
    assert info["mpc_nonlinear_intent_count_available"] == 5
    assert info["mpc_nonlinear_intent_count_considered"] == 3
    assert info["mpc_nonlinear_intent_count_pruned"] == 2


def test_dmpc_best_response_debug_reports_pruned_local_intents() -> None:
    planner = DistributedMpcBestResponsePlanner(
        cfg={
            "max_neighbors": 2,
            "max_intents": 3,
            "horizon_steps": 3,
            "max_initializations": 1,
            "opt_iterations": 1,
            "intent_trust_horizon_s": 1.0,
        }
    )
    planner_input = _planner_input()
    planner_input.agent_context = AgentContext(agent_id=0, method="dmpc_best_response", seed=0)

    out = planner.compute_cmd(planner_input)
    info = out.debug_info

    assert info["dmpc_best_response_neighbor_count_considered"] == 2
    assert info["dmpc_best_response_intent_count_available"] == 5
    assert info["dmpc_best_response_intent_count_considered"] == 3
    assert info["dmpc_best_response_intent_count_pruned"] == 2
    assert info["dmpc_best_response_neighbor_intent_count_considered"] == 3


def test_dynamic_tube_dmpc_debug_reports_pruned_local_intents() -> None:
    planner = DynamicTubeDmpcPlanner(
        cfg={
            "max_neighbors": 2,
            "max_intents": 3,
            "horizon_steps": 3,
            "qp_iterations": 2,
            "projection_iterations": 1,
            "tube_waypoints": 5,
            "current_track_guard_iterations": 1,
        }
    )

    out = planner.compute_cmd(_planner_input())
    info = out.debug_info

    assert info["dynamic_tube_dmpc_neighbor_count_considered"] == 2
    assert info["dynamic_tube_dmpc_intent_count_available"] == 5
    assert info["dynamic_tube_dmpc_intent_count_considered"] == 3
    assert info["dynamic_tube_dmpc_intent_count_pruned"] == 2


def test_bvc_tube_dmpc_debug_reports_pruned_local_intents() -> None:
    planner = BvcTubeDmpcPlanner(
        cfg={
            "max_neighbors": 2,
            "max_intents": 3,
            "horizon_steps": 3,
            "max_initializations": 1,
            "opt_iterations": 1,
            "projection_iterations": 1,
        }
    )

    out = planner.compute_cmd(_planner_input())
    info = out.debug_info

    assert info["bvc_tube_dmpc_neighbor_count_considered"] == 2
    assert info["bvc_tube_dmpc_intent_count_available"] == 5
    assert info["bvc_tube_dmpc_intent_count_considered"] == 3
    assert info["bvc_tube_dmpc_intent_count_pruned"] == 2


def test_rmader_debug_reports_pruned_local_intents() -> None:
    planner = RmaderPlanner(
        cfg={
            "max_neighbors": 2,
            "max_intents": 3,
            "control_points": 7,
            "samples_per_interval": 2,
            "max_initializations": 1,
            "opt_iterations": 1,
            "hard_projection_iterations": 1,
            "sampled_delay_check_enabled": False,
            "delay_check_enabled": False,
        }
    )
    planner_input = _planner_input()
    planner_input.agent_context = AgentContext(agent_id=0, method="rmader", seed=0)

    out = planner.compute_cmd(planner_input)
    info = out.debug_info

    assert info["rmader_neighbor_count_considered"] == 2
    assert info["rmader_intent_count_available"] == 5
    assert info["rmader_intent_count_considered"] == 3
    assert info["rmader_intent_count_pruned"] == 2
