from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

import yaml

from microbench.tools.external_reference import validate_external_reference_manifest


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
