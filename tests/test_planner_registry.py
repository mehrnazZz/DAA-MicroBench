from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from microbench.planners import canonical_method, list_methods, make_planner, planner_metadata
from microbench.planners.cbf_qp import CbfQpPlanner
from microbench.planners.ego_swarm import EgoSwarmPlanner
from microbench.planners.learned_tiny import LearnedTinyPlanner
from microbench.planners.learned_policy_spec import LearnedPolicySpecPlanner
from microbench.planners.mpc_local import MpcLocalPlanner
from microbench.planners.orca_expert import OrcaExpertPlanner
from microbench.planners.velocity_obstacle import ReciprocalVelocityObstaclePlanner, VelocityObstaclePlanner


def test_orca_heuristic_is_canonical_and_orca_expert_is_alias() -> None:
    assert canonical_method("orca_heuristic") == "orca_heuristic"
    assert canonical_method("orca_with_staleness") == "orca_with_staleness"
    assert canonical_method("orca_expert") == "orca_heuristic"
    assert "orca_heuristic" in list_methods()
    assert "orca_with_staleness" in list_methods()
    assert "cbf_qp" in list_methods()
    assert "mpc_local" in list_methods()
    assert "ego_swarm" in list_methods()
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
    assert isinstance(make_planner("ego_swarm"), EgoSwarmPlanner)
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
    assert by_method["ego_swarm"]["role"] == "experimental_baseline"
    assert by_method["ego_swarm"]["planner_type"] == "decentralized_trajectory_optimization"
    assert by_method["ego_swarm"]["uses_intent"] is True
    assert by_method["ego_swarm"]["uses_obstacles"] is True
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
    assert by_method["negotiation_yield"]["role"] == "agentic_reference_baseline"
    assert by_method["negotiation_yield"]["status"] == "pre_v1"
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
    assert by_method["ego_swarm"]["status"] == "experimental"
    assert by_method["velocity_obstacle"]["status"] == "experimental"
    assert by_method["reciprocal_velocity_obstacle"]["status"] == "experimental"
    assert by_method["learned_tiny"]["learned"] is True
    assert by_method["learned_policy_spec"]["role"] == "submission_bridge"
    assert by_method["negotiation_yield"]["status"] == "pre_v1"
    assert by_method["orca_expert"]["canonical_method"] == "orca_heuristic"
