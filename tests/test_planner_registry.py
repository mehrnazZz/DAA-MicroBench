from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from microbench.planners import canonical_method, list_methods, make_planner, planner_metadata
from microbench.planners.bvc_tube_dmpc import BvcTubeDmpcPlanner
from microbench.planners.cbf_qp import CbfQpPlanner
from microbench.planners.centralized_oracle import CentralizedMpcOraclePlanner, CentralizedOraclePlanner
from microbench.planners.dmpc_best_response import DistributedMpcBestResponsePlanner
from microbench.planners.dynamic_tube_dmpc import DynamicTubeDmpcPlanner
from microbench.planners.ego_swarm import EgoSwarmPlanner
from microbench.planners.ego_swarm_opt import EgoSwarmOptimizingPlanner
from microbench.planners.learned_tiny import LearnedTinyPlanner
from microbench.planners.learned_policy_spec import LearnedPolicySpecPlanner
from microbench.planners.mpc_local import MpcLocalPlanner
from microbench.planners.mpc_nonlinear import NonlinearMpcPlanner
from microbench.planners.orca_expert import OrcaExpertPlanner
from microbench.planners.rmader import RmaderPlanner
from microbench.planners.velocity_obstacle import ReciprocalVelocityObstaclePlanner, VelocityObstaclePlanner


def test_orca_heuristic_is_canonical_and_orca_expert_is_alias() -> None:
    assert canonical_method("orca_heuristic") == "orca_heuristic"
    assert canonical_method("orca_with_staleness") == "orca_with_staleness"
    assert canonical_method("orca_expert") == "orca_heuristic"
    assert "orca_heuristic" in list_methods()
    assert "orca_with_staleness" in list_methods()
    assert "cbf_qp" in list_methods()
    assert "mpc_local" in list_methods()
    assert "mpc_nonlinear" in list_methods()
    assert "dmpc_best_response" in list_methods()
    assert "bvc_tube_dmpc" in list_methods()
    assert "dynamic_tube_dmpc" in list_methods()
    assert "centralized_oracle" in list_methods()
    assert "centralized_mpc_oracle" in list_methods()
    assert "rmader" in list_methods()
    assert "ego_swarm" in list_methods()
    assert "ego_swarm_opt" in list_methods()
    assert "velocity_obstacle" in list_methods()
    assert "reciprocal_velocity_obstacle" in list_methods()
    assert "learned_tiny" in list_methods()
    assert "learned_policy_spec" in list_methods()
    assert "orca_expert" not in list_methods()
    assert "orca_expert" in list_methods(include_aliases=True)
    assert isinstance(make_planner("orca_heuristic"), OrcaExpertPlanner)
    assert isinstance(make_planner("orca_with_staleness"), OrcaExpertPlanner)
    assert isinstance(make_planner("cbf_qp"), CbfQpPlanner)
    assert isinstance(make_planner("mpc_local"), MpcLocalPlanner)
    assert isinstance(make_planner("mpc_nonlinear"), NonlinearMpcPlanner)
    assert isinstance(make_planner("dmpc_best_response"), DistributedMpcBestResponsePlanner)
    assert isinstance(make_planner("bvc_tube_dmpc"), BvcTubeDmpcPlanner)
    assert isinstance(make_planner("dynamic_tube_dmpc"), DynamicTubeDmpcPlanner)
    assert isinstance(make_planner("centralized_oracle"), CentralizedOraclePlanner)
    assert isinstance(make_planner("centralized_mpc_oracle"), CentralizedMpcOraclePlanner)
    assert isinstance(make_planner("rmader"), RmaderPlanner)
    assert isinstance(make_planner("ego_swarm"), EgoSwarmPlanner)
    assert isinstance(make_planner("ego_swarm_opt"), EgoSwarmOptimizingPlanner)
    assert isinstance(make_planner("velocity_obstacle"), VelocityObstaclePlanner)
    assert isinstance(make_planner("reciprocal_velocity_obstacle"), ReciprocalVelocityObstaclePlanner)
    assert isinstance(make_planner("learned_tiny"), LearnedTinyPlanner)
    assert isinstance(make_planner("learned_policy_spec", policy_spec="examples/external_policy_spec.json"), LearnedPolicySpecPlanner)
    assert isinstance(make_planner("orca_expert"), OrcaExpertPlanner)

    with pytest.raises(ValueError, match="policy-spec"):
        make_planner("learned_policy_spec")


def test_planner_metadata_includes_public_baseline_contract() -> None:
    by_method = {entry["method"]: entry for entry in planner_metadata(include_aliases=True)}

    assert by_method["baseline_goal"]["role"] == "illustrative_baseline"
    assert by_method["orca_heuristic"]["role"] == "reference_baseline"
    assert by_method["orca_heuristic"]["dimensions"] == ("2d", "3d")
    assert by_method["orca_heuristic"]["uses_v2v"] is True
    assert by_method["orca_heuristic"]["uses_local_sensing"] is True
    assert by_method["orca_with_staleness"]["role"] == "reference_baseline"
    assert by_method["orca_with_staleness"]["uses_v2v"] is True
    assert by_method["orca_with_staleness"]["uses_local_sensing"] is True
    assert by_method["cbf_qp"]["role"] == "experimental_baseline"
    assert by_method["cbf_qp"]["status"] == "experimental"
    assert by_method["cbf_qp"]["uses_obstacles"] is True
    assert by_method["mpc_local"]["role"] == "experimental_baseline"
    assert by_method["mpc_local"]["planner_type"] == "predictive_sampling"
    assert by_method["mpc_local"]["uses_obstacles"] is True
    assert by_method["mpc_nonlinear"]["role"] == "experimental_baseline"
    assert by_method["mpc_nonlinear"]["planner_type"] == "nonlinear_mpc_trajectory_optimization"
    assert by_method["mpc_nonlinear"]["uses_intent"] is True
    assert by_method["mpc_nonlinear"]["uses_obstacles"] is True
    assert by_method["dmpc_best_response"]["role"] == "experimental_baseline"
    assert by_method["dmpc_best_response"]["planner_type"] == "distributed_mpc_best_response"
    assert by_method["dmpc_best_response"]["uses_intent"] is True
    assert by_method["dmpc_best_response"]["uses_agent_messages"] is True
    assert by_method["dmpc_best_response"]["uses_obstacles"] is True
    assert by_method["bvc_tube_dmpc"]["role"] == "experimental_baseline"
    assert by_method["bvc_tube_dmpc"]["planner_type"] == "tube_dmpc_buffered_voronoi_cells"
    assert by_method["bvc_tube_dmpc"]["uses_intent"] is True
    assert by_method["bvc_tube_dmpc"]["uses_agent_messages"] is True
    assert by_method["bvc_tube_dmpc"]["uses_obstacles"] is True
    assert by_method["dynamic_tube_dmpc"]["role"] == "experimental_baseline"
    assert by_method["dynamic_tube_dmpc"]["planner_type"] == "dynamic_tube_distributed_mpc_qp"
    assert by_method["dynamic_tube_dmpc"]["uses_intent"] is True
    assert by_method["dynamic_tube_dmpc"]["uses_agent_messages"] is True
    assert by_method["dynamic_tube_dmpc"]["uses_obstacles"] is True
    assert by_method["centralized_oracle"]["role"] == "nondeployable_upper_bound"
    assert by_method["centralized_oracle"]["planner_type"] == "centralized_privileged_oracle"
    assert "global_truth" in by_method["centralized_oracle"]["observation_sources"]
    assert by_method["centralized_oracle"]["uses_v2v"] is False
    assert by_method["centralized_oracle"]["uses_local_sensing"] is False
    assert by_method["centralized_oracle"]["uses_obstacles"] is True
    assert any("privileged global state" in item for item in by_method["centralized_oracle"]["limitations"])
    assert by_method["centralized_mpc_oracle"]["role"] == "nondeployable_upper_bound"
    assert by_method["centralized_mpc_oracle"]["planner_type"] == "centralized_privileged_route_mpc_oracle"
    assert "world_bounds" in by_method["centralized_mpc_oracle"]["observation_sources"]
    assert by_method["centralized_mpc_oracle"]["uses_v2v"] is False
    assert by_method["centralized_mpc_oracle"]["uses_local_sensing"] is False
    assert by_method["centralized_mpc_oracle"]["uses_obstacles"] is True
    assert any("too short" in item for item in by_method["centralized_mpc_oracle"]["limitations"])
    assert by_method["rmader"]["role"] == "experimental_baseline"
    assert by_method["rmader"]["planner_type"] == "rmader_minvo_hyperplane_trajectory_optimization"
    assert by_method["rmader"]["uses_intent"] is True
    assert by_method["rmader"]["uses_agent_messages"] is True
    assert by_method["rmader"]["uses_obstacles"] is True
    assert by_method["ego_swarm"]["role"] == "experimental_baseline"
    assert by_method["ego_swarm"]["planner_type"] == "decentralized_trajectory_optimization"
    assert by_method["ego_swarm"]["uses_intent"] is True
    assert by_method["ego_swarm"]["uses_obstacles"] is True
    assert by_method["ego_swarm_opt"]["role"] == "experimental_baseline"
    assert by_method["ego_swarm_opt"]["planner_type"] == "decentralized_control_point_trajectory_optimization"
    assert by_method["ego_swarm_opt"]["uses_intent"] is True
    assert by_method["ego_swarm_opt"]["uses_obstacles"] is True
    assert by_method["velocity_obstacle"]["role"] == "experimental_baseline"
    assert by_method["velocity_obstacle"]["planner_type"] == "velocity_obstacle_sampling"
    assert by_method["velocity_obstacle"]["uses_obstacles"] is True
    assert by_method["reciprocal_velocity_obstacle"]["role"] == "experimental_baseline"
    assert by_method["reciprocal_velocity_obstacle"]["planner_type"] == "reciprocal_velocity_obstacle_sampling"
    assert by_method["reciprocal_velocity_obstacle"]["uses_obstacles"] is True
    assert by_method["learned_tiny"]["role"] == "experimental_baseline"
    assert by_method["learned_tiny"]["planner_type"] == "learned_policy"
    assert by_method["learned_tiny"]["learned"] is True
    assert by_method["learned_tiny"]["uses_v2v"] is True
    assert by_method["learned_policy_spec"]["role"] == "submission_bridge"
    assert by_method["learned_policy_spec"]["planner_type"] == "learned_policy"
    assert by_method["learned_policy_spec"]["learned"] is True
    assert by_method["learned_policy_spec"]["fidelity"] == "submission_bridge"
    assert by_method["negotiation_yield"]["role"] == "agentic_reference_baseline"
    assert by_method["negotiation_yield"]["status"] == "pre_v1"
    assert by_method["rmader"]["fidelity"] == "faithful_reimplementation"
    assert by_method["rmader"]["external_reference_candidate"] is True
    assert "https://github.com/mit-acl/rmader" in by_method["rmader"]["reference_urls"]
    assert by_method["ego_swarm_opt"]["fidelity"] == "inspired_clean_room"
    assert by_method["ego_swarm_opt"]["external_reference_candidate"] is True
    assert by_method["orca_expert"]["status"] == "alias"
    assert by_method["orca_expert"]["canonical_method"] == "orca_heuristic"


def test_orca_with_staleness_uses_more_conservative_stale_preset() -> None:
    standard = make_planner("orca_heuristic")
    stale_aware = make_planner("orca_with_staleness")

    assert isinstance(standard, OrcaExpertPlanner)
    assert isinstance(stale_aware, OrcaExpertPlanner)
    assert stale_aware.stale_inflation_gain > standard.stale_inflation_gain
    assert stale_aware.stale_age_cap_s > standard.stale_age_cap_s
    assert stale_aware.responsibility_age_gain > standard.responsibility_age_gain


def test_scale_planner_preset_overrides_optimizer_budget(monkeypatch) -> None:
    monkeypatch.setenv("DAA_MICROBENCH_PLANNER_PRESET", "scale")

    mpc = make_planner("mpc_nonlinear")
    dmpc = make_planner("dmpc_best_response")
    bvc = make_planner("bvc_tube_dmpc")
    ego = make_planner("ego_swarm_opt")
    rmader = make_planner("rmader")

    assert isinstance(mpc, NonlinearMpcPlanner)
    assert isinstance(dmpc, DistributedMpcBestResponsePlanner)
    assert isinstance(bvc, BvcTubeDmpcPlanner)
    assert isinstance(ego, EgoSwarmOptimizingPlanner)
    assert isinstance(rmader, RmaderPlanner)
    assert mpc.max_neighbors == 6
    assert mpc.max_intents == 8
    assert mpc.horizon_steps == 4
    assert mpc.safety_margin_m == 0.5
    assert mpc.velocity_guard_margin_m == 0.65
    assert mpc.velocity_guard_alpha == 0.7
    assert mpc.velocity_guard_brake_scale == 0.3
    assert dmpc.max_neighbors == 8
    assert dmpc.max_intents == 10
    assert dmpc.max_initializations == 2
    assert dmpc.safety_margin_m == 0.5
    assert dmpc.velocity_guard_margin_m == 0.65
    assert dmpc.velocity_guard_alpha == 0.7
    assert dmpc.velocity_guard_brake_scale == 0.3
    assert bvc.max_neighbors == 8
    assert bvc.max_intents == 10
    assert bvc.projection_iterations == 2
    assert bvc.safety_margin_m == 0.6
    assert bvc.velocity_guard_margin_m == 0.7
    assert bvc.velocity_guard_alpha == 0.7
    assert bvc.velocity_guard_brake_scale == 0.25
    assert ego.max_neighbors == 8
    assert ego.max_intents == 10
    assert ego.max_initializations == 4
    assert ego.opt_iterations == 2
    assert ego.replan_period_s == 0.2
    assert ego.safety_margin_m == 0.55
    assert ego.velocity_guard_margin_m == 0.55
    assert ego.velocity_guard_brake_clearance_m == 1.0
    assert ego.velocity_guard_brake_scale == 0.35
    assert rmader.max_neighbors == 6
    assert rmader.max_dynamic_hulls_per_interval == 5
    assert rmader.dynamic_broadphase_margin_m == 6.0
    assert rmader.cached_reuse_validation == "kinematic"
    assert rmader.replan_period_s == 0.3
    assert rmader.sampled_delay_check_enabled is False
    assert rmader.velocity_guard_margin_m == 0.65
    assert rmader.velocity_guard_alpha == 0.7
    assert rmader.velocity_guard_brake_scale == 0.3


def test_list_methods_cli_can_emit_metadata_json() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "microbench.cli",
            "list-methods",
            "--json",
            "--include-aliases",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )

    entries = json.loads(proc.stdout)
    by_method = {entry["method"]: entry for entry in entries}
    assert by_method["orca_heuristic"]["status"] == "pre_v1"
    assert by_method["orca_with_staleness"]["role"] == "reference_baseline"
    assert by_method["cbf_qp"]["status"] == "experimental"
    assert by_method["mpc_local"]["status"] == "experimental"
    assert by_method["mpc_nonlinear"]["status"] == "experimental"
    assert by_method["dmpc_best_response"]["status"] == "experimental"
    assert by_method["bvc_tube_dmpc"]["status"] == "experimental"
    assert by_method["dynamic_tube_dmpc"]["status"] == "experimental"
    assert by_method["centralized_oracle"]["status"] == "experimental"
    assert by_method["centralized_mpc_oracle"]["status"] == "experimental"
    assert by_method["rmader"]["status"] == "experimental"
    assert by_method["ego_swarm"]["status"] == "experimental"
    assert by_method["ego_swarm_opt"]["status"] == "experimental"
    assert by_method["velocity_obstacle"]["status"] == "experimental"
    assert by_method["reciprocal_velocity_obstacle"]["status"] == "experimental"
    assert by_method["learned_tiny"]["learned"] is True
    assert by_method["learned_policy_spec"]["role"] == "submission_bridge"
    assert by_method["learned_policy_spec"]["fidelity"] == "submission_bridge"
    assert by_method["dynamic_tube_dmpc"]["fidelity"] == "faithful_reimplementation"
    assert by_method["ego_swarm"]["fidelity"] == "inspired_clean_room"
    assert by_method["negotiation_yield"]["status"] == "pre_v1"
    assert by_method["orca_expert"]["canonical_method"] == "orca_heuristic"
