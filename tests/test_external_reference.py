from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

import yaml

from microbench.tools.external_reference import build_external_reference_bundle, validate_external_reference_manifest


ROOT = Path(__file__).resolve().parents[1]


def _write_results_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "scenario",
                "N",
                "seed",
                "collision_episode",
                "completion_rate",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "method": "rmader_official_ros_noetic",
                "scenario": "urban_conflict_3d",
                "N": 4,
                "seed": 2,
                "collision_episode": 0,
                "completion_rate": 1.0,
            }
        )


def _manifest(results_csv: str = "artifacts/results.csv") -> dict:
    return {
        "schema_version": "0.1",
        "reference_id": "rmader_official_ros_noetic",
        "method_family": "RMADER",
        "related_microbench_method": "rmader",
        "fidelity": "official_implementation",
        "implementation": {
            "name": "MIT ACL RMADER",
            "source_url": "https://github.com/mit-acl/rmader",
            "license": "BSD-3-Clause",
            "commit": "abc123",
            "publication_urls": [
                "https://arxiv.org/abs/2303.06222",
                "https://doi.org/10.1109/ICRA48891.2023.10161244",
            ],
        },
        "method_claims": {
            "decentralized_asynchronous_planning": True,
            "communication_delay_robustness": True,
            "delay_check": True,
            "two_step_trajectory_publication": True,
            "trajectory_storing_and_checking": True,
            "minvo_interval_polyhedra": True,
            "hard_separating_hyperplanes": True,
            "dynamic_obstacles": True,
            "static_obstacles": True,
            "solver_backend": "Gurobi",
        },
        "runner": {
            "type": "ros",
            "command": "roslaunch rmader external_microbench_adapter.launch",
        },
        "contract": {
            "input_format": "daa_microbench_scenario_yaml",
            "output_format": "daa_microbench_results_csv",
            "uses_microbench_scenarios": True,
            "privileged_information": False,
            "decentralized_authority": True,
        },
        "adapter": {
            "scenario_mapping": "scenario yaml to ROS starts/goals/obstacles",
            "agent_authority_mapping": "one RMADER planner per drone",
            "observation_mapping": "only Microbench-allowed local tracks/intents",
            "communication_mapping": "Microbench comm profile to ROS publication impairment",
            "obstacle_mapping": "AABB/static/dynamic obstacle conversion",
            "output_mapping": "external results to Microbench results.csv",
        },
        "artifacts": {
            "results_csv": results_csv,
            "summary_csv": "artifacts/summary.csv",
        },
    }


def _ego_swarm_manifest(results_csv: str = "artifacts/results.csv") -> dict:
    return {
        "schema_version": "0.1",
        "reference_id": "ego_swarm_official_ros",
        "method_family": "EGO-Swarm",
        "related_microbench_method": "ego_swarm_opt",
        "fidelity": "official_implementation",
        "implementation": {
            "name": "ZJU FAST-Lab EGO-Planner-Swarm",
            "source_url": "https://github.com/ZJU-FAST-Lab/ego-planner-swarm",
            "license": "GPL-3.0",
            "commit": "abc123",
            "publication_urls": [
                "https://github.com/ZJU-FAST-Lab/ego-planner-swarm",
                "https://github.com/ZJU-FAST-Lab/ego-planner",
            ],
        },
        "method_claims": {
            "decentralized_swarm_planning": True,
            "asynchronous_planning": True,
            "onboard_sensing_and_compute": True,
            "unknown_cluttered_environment_navigation": True,
            "trajectory_sharing": True,
            "b_spline_trajectory_representation": True,
            "gradient_based_optimization": True,
            "esdf_free_local_planning": True,
            "static_obstacle_avoidance": True,
            "inter_agent_collision_avoidance": True,
            "simulator_mode_disclosed": True,
            "local_sensing_mode_disclosed": True,
            "license_gpl3_disclosed": True,
            "simulator_mode": "fake_drone",
            "local_sensing_backend": "CPU local_sensing",
        },
        "runner": {
            "type": "ros",
            "command": "roslaunch ego_planner swarm_external_microbench_adapter.launch",
        },
        "contract": {
            "input_format": "daa_microbench_scenario_yaml",
            "output_format": "daa_microbench_results_csv",
            "uses_microbench_scenarios": True,
            "privileged_information": False,
            "decentralized_authority": True,
        },
        "adapter": {
            "scenario_mapping": "scenario yaml to ROS starts/goals/obstacles",
            "agent_authority_mapping": "one EGO-Swarm planner per drone",
            "observation_mapping": "only Microbench-allowed local tracks/intents",
            "communication_mapping": "Microbench comm profile to trajectory broadcast impairment",
            "obstacle_mapping": "AABB/static/dynamic obstacle conversion",
            "output_mapping": "external results to Microbench results.csv",
            "map_sensing_mapping": "Microbench obstacles to local_sensing point cloud",
            "trajectory_message_mapping": "upstream B-spline broadcasts to Microbench intents",
            "dynamics_simulator_mapping": "fake_drone timing and limits to Microbench fields",
            "license_boundary": "GPL code stays outside the core package",
        },
        "artifacts": {
            "results_csv": results_csv,
            "summary_csv": "artifacts/summary.csv",
        },
    }


def test_external_reference_manifest_validates_with_required_artifacts(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    data = _manifest()
    manifest_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    _write_results_csv(tmp_path / "artifacts" / "results.csv")
    (tmp_path / "artifacts" / "summary.csv").write_text("method,episodes\nrmader,1\n", encoding="utf-8")

    report = validate_external_reference_manifest(manifest=manifest_path, require_artifacts=True)

    assert report["ok"] is True
    assert report["related_microbench_method"] == "rmader"
    assert report["fidelity"] == "official_implementation"
    assert report["artifact_status"]["results_csv"]["exists"] is True
    assert "completion_rate" in report["artifact_status"]["results_csv"]["fields"]
    assert report["method_claim_status"]["delay_check"] is True
    assert report["method_claim_status"]["adapter_fields"]["communication_mapping"] is True


def test_external_reference_manifest_blocks_privileged_information(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    data = _manifest()
    data["contract"]["privileged_information"] = True
    manifest_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    report = validate_external_reference_manifest(manifest=manifest_path)

    assert report["ok"] is False
    assert "contract_must_declare_no_privileged_information" in report["errors"]


def test_external_reference_manifest_requires_rmader_claim_contract(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    data = _manifest()
    del data["method_claims"]["delay_check"]
    data["method_claims"]["two_step_trajectory_publication"] = False
    del data["adapter"]["communication_mapping"]
    manifest_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    report = validate_external_reference_manifest(manifest=manifest_path)

    assert report["ok"] is False
    assert "rmader_required_claim_missing_or_false:delay_check" in report["errors"]
    assert "rmader_required_claim_missing_or_false:two_step_trajectory_publication" in report["errors"]
    assert "rmader_adapter_field_missing:communication_mapping" in report["errors"]


def test_external_reference_manifest_requires_ego_swarm_claim_contract(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    data = _ego_swarm_manifest()
    del data["method_claims"]["trajectory_sharing"]
    data["method_claims"]["gradient_based_optimization"] = False
    data["implementation"]["license"] = "proprietary"
    del data["adapter"]["map_sensing_mapping"]
    manifest_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    report = validate_external_reference_manifest(manifest=manifest_path)

    assert report["ok"] is False
    assert "ego_swarm_required_claim_missing_or_false:trajectory_sharing" in report["errors"]
    assert "ego_swarm_required_claim_missing_or_false:gradient_based_optimization" in report["errors"]
    assert "ego_swarm_adapter_field_missing:map_sensing_mapping" in report["errors"]
    assert "ego_swarm_official_license_must_disclose_gpl3" in report["errors"]


def test_external_reference_example_manifest_is_structurally_valid() -> None:
    report = validate_external_reference_manifest(
        manifest=ROOT / "examples" / "external_reference_rmader_manifest.yaml",
        require_artifacts=False,
    )

    assert report["ok"] is True
    assert report["method_claim_status"]["delay_check"] is True
    assert report["method_claim_status"]["solver_backend"] == "gurobi"
    assert report["warnings"]
    assert any(warning.startswith("artifact_not_found") for warning in report["warnings"])


def test_external_reference_ego_swarm_example_manifest_is_structurally_valid() -> None:
    report = validate_external_reference_manifest(
        manifest=ROOT / "examples" / "external_reference_ego_swarm_manifest.yaml",
        require_artifacts=False,
    )

    assert report["ok"] is True
    assert report["related_microbench_method"] == "ego_swarm_opt"
    assert report["method_claim_status"]["trajectory_sharing"] is True
    assert report["method_claim_status"]["gradient_based_optimization"] is True
    assert report["method_claim_status"]["adapter_fields"]["map_sensing_mapping"] is True
    assert report["warnings"]
    assert any(warning.startswith("artifact_not_found") for warning in report["warnings"])


def test_external_reference_cli_json(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(_manifest(), sort_keys=False), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "microbench.cli",
            "validate-external-reference",
            "--manifest",
            str(manifest_path),
            "--json",
            "--require-pass",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    report = json.loads(proc.stdout)
    assert report["ok"] is True
    assert report["reference_id"] == "rmader_official_ros_noetic"


def test_external_reference_bundle_writes_rmader_capture_template(tmp_path: Path) -> None:
    out_dir = tmp_path / "rmader_bundle"

    report = build_external_reference_bundle(
        method_family="rmader",
        out_dir=out_dir,
        scenarios=["urban_conflict_3d", "stacked_swap_3d"],
        n_agents=[4, 8],
        seeds=[2],
        comm_profiles=["ideal_50hz"],
        runner_type="ros",
    )

    assert report["ok"] is True
    assert report["run_count"] == 4
    assert report["scenario_count"] == 2
    assert (out_dir / "manifest.yaml").exists()
    assert (out_dir / "run_matrix.csv").exists()
    assert (out_dir / "run_matrix.json").exists()
    assert (out_dir / "results_template.csv").exists()
    assert (out_dir / "summary_template.csv").exists()
    assert (out_dir / "result_schema.json").exists()
    assert (out_dir / "RUN_NOTES.md").exists()
    assert (out_dir / "ADAPTER_PLAN.md").exists()
    assert (out_dir / "adapter_plan.json").exists()
    assert (out_dir / "checksums.json").exists()
    assert (out_dir / "scenarios" / "urban_conflict_3d.yaml").exists()
    assert (out_dir / "scenarios" / "stacked_swap_3d.yaml").exists()

    with (out_dir / "run_matrix.csv").open("r", newline="", encoding="utf-8") as f:
        matrix_rows = list(csv.DictReader(f))
    assert len(matrix_rows) == 4
    assert {row["scenario"] for row in matrix_rows} == {"urban_conflict_3d", "stacked_swap_3d"}
    assert {row["N"] for row in matrix_rows} == {"4", "8"}
    assert {row["expected_results_method"] for row in matrix_rows} == {"rmader_official_ros_noetic"}

    with (out_dir / "results_template.csv").open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        template_rows = list(reader)
        fields = list(reader.fieldnames or [])
    assert len(template_rows) == 4
    assert "collision_episode" in fields
    assert {row["method"] for row in template_rows} == {"rmader_official_ros_noetic"}

    manifest = yaml.safe_load((out_dir / "manifest.yaml").read_text(encoding="utf-8"))
    assert manifest["method_claims"]["delay_check"] is True
    assert manifest["adapter"]["communication_mapping"]

    adapter_plan = json.loads((out_dir / "adapter_plan.json").read_text(encoding="utf-8"))
    assert adapter_plan["reference_id"] == "rmader_official_ros_noetic"
    assert adapter_plan["run_count"] == 4
    assert adapter_plan["adapter_fields"]["communication_mapping"]
    assert {task["id"] for task in adapter_plan["method_specific_tasks"]} == {
        "rmader_delay_check",
        "rmader_minvo_hyperplanes",
    }
    adapter_plan_md = (out_dir / "ADAPTER_PLAN.md").read_text(encoding="utf-8")
    assert "Preserve RMADER publication semantics" in adapter_plan_md
    assert "Final Validation" in adapter_plan_md

    checksums = json.loads((out_dir / "checksums.json").read_text(encoding="utf-8"))
    checksum_paths = {entry["path"] for entry in checksums["files"]}
    assert "ADAPTER_PLAN.md" in checksum_paths
    assert "adapter_plan.json" in checksum_paths

    validation = validate_external_reference_manifest(manifest=out_dir / "manifest.yaml")
    assert validation["ok"] is True
    assert validation["method_claim_status"]["hard_separating_hyperplanes"] is True


def test_external_reference_bundle_writes_ego_swarm_capture_template(tmp_path: Path) -> None:
    out_dir = tmp_path / "ego_swarm_bundle"

    report = build_external_reference_bundle(
        method_family="ego_swarm",
        out_dir=out_dir,
        scenarios=["urban_conflict_3d"],
        n_agents=[4],
        seeds=[2],
        comm_profiles=["ideal_50hz"],
        runner_type="ros",
    )

    assert report["ok"] is True
    assert report["run_count"] == 1
    assert report["scenario_count"] == 1
    assert report["reference_id"] == "ego_swarm_official_ros"
    assert report["method_family"] == "EGO-Swarm"
    assert report["related_microbench_method"] == "ego_swarm_opt"
    assert (out_dir / "manifest.yaml").exists()
    assert (out_dir / "run_matrix.csv").exists()
    assert (out_dir / "ADAPTER_PLAN.md").exists()
    assert (out_dir / "adapter_plan.json").exists()
    assert (out_dir / "scenarios" / "urban_conflict_3d.yaml").exists()

    manifest = yaml.safe_load((out_dir / "manifest.yaml").read_text(encoding="utf-8"))
    assert manifest["implementation"]["license"] == "GPL-3.0"
    assert manifest["method_claims"]["trajectory_sharing"] is True
    assert manifest["method_claims"]["b_spline_trajectory_representation"] is True
    assert manifest["adapter"]["map_sensing_mapping"]
    assert manifest["adapter"]["license_boundary"]

    adapter_plan = json.loads((out_dir / "adapter_plan.json").read_text(encoding="utf-8"))
    assert adapter_plan["method_family"] == "EGO-Swarm"
    assert adapter_plan["adapter_fields"]["map_sensing_mapping"]
    assert adapter_plan["adapter_fields"]["license_boundary"]
    assert {task["id"] for task in adapter_plan["method_specific_tasks"]} == {
        "ego_swarm_sensing_and_map",
        "ego_swarm_trajectory_broadcast",
        "ego_swarm_gpl_boundary",
    }
    adapter_plan_md = (out_dir / "ADAPTER_PLAN.md").read_text(encoding="utf-8")
    assert "Preserve the GPL boundary" in adapter_plan_md
    assert "trajectory broadcast" in adapter_plan_md

    validation = validate_external_reference_manifest(manifest=out_dir / "manifest.yaml")
    assert validation["ok"] is True
    assert validation["method_claim_status"]["trajectory_sharing"] is True
    assert validation["method_claim_status"]["adapter_fields"]["trajectory_message_mapping"] is True


def test_external_reference_bundle_cli_json(tmp_path: Path) -> None:
    out_dir = tmp_path / "cli_bundle"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "microbench.cli",
            "external-reference-bundle",
            "--method-family",
            "rmader",
            "--out-dir",
            str(out_dir),
            "--scenarios",
            "urban_conflict_3d",
            "--n",
            "4",
            "--seeds",
            "2",
            "--comm",
            "ideal_50hz",
            "--runner-type",
            "ros",
            "--require-pass",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    report = json.loads(proc.stdout)
    assert report["ok"] is True
    assert report["run_count"] == 1
    assert report["reference_id"] == "rmader_official_ros_noetic"
    assert (out_dir / Path(report["adapter_plan_md"]).name).exists()
    assert (out_dir / Path(report["adapter_plan_json"]).name).exists()
    assert (out_dir / "external_reference_bundle.json").exists()
    assert (out_dir / "manifest.yaml").exists()
    assert (out_dir / "run_matrix.csv").exists()
