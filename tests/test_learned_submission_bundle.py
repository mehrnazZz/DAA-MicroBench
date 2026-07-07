from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from microbench.learned import tiny_learned_model_path
from microbench.rl.policy_spec import RL_POLICY_SPEC_SCHEMA_VERSION
from microbench.rl.submission_bundle import (
    review_learned_policy_submission_bundle,
    run_learned_policy_submission_bundle,
    validate_learned_policy_submission_bundle,
)


ROOT = Path(__file__).resolve().parents[1]


def _check(report: dict, name: str) -> dict:
    return next(check for check in report["checks"] if check["name"] == name)


def _tiny_policy_spec(tmp_path: Path) -> Path:
    path = tmp_path / "external_policy_spec.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": RL_POLICY_SPEC_SCHEMA_VERSION,
                "policy_name": "external_tiny_fixture",
                "adapter": "tiny_linear_json",
                "artifact_path": tiny_learned_model_path(),
                "deterministic": True,
                "clip": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_learned_policy_submission_bundle_helper_writes_expected_artifacts(tmp_path: Path) -> None:
    report = run_learned_policy_submission_bundle(
        out_dir=tmp_path / "bundle",
        method="learned_tiny",
        policy="tiny_learned",
        max_runs=1,
        max_steps=3,
    )

    assert report["schema_version"] == "0.1"
    assert report["ok"] is True
    assert report["method"] == "learned_tiny"
    assert report["policy"] == "tiny_learned"
    assert report["planner_sweep"]["run_count"] == 1
    assert report["acceptance"]["ok"] is True
    assert report["artifacts"]["learned_submission_manifest"] == "learned_submission_manifest.json"
    assert report["submission_manifest"]["schema_version"] == "0.1"
    assert _check(report, "method_marked_learned")["ok"] is True
    assert _check(report, "expected_artifacts_present")["ok"] is True
    assert _check(report, "learned_submission_manifest_written")["ok"] is True

    bundle_root = tmp_path / "bundle"
    for path in report["artifacts"].values():
        assert not Path(path).is_absolute()
        assert (bundle_root / path).exists(), path
    assert (bundle_root / "learned_submission_bundle.json").exists()

    validation = validate_learned_policy_submission_bundle(bundle=bundle_root)
    assert validation["ok"] is True
    assert _check(validation, "required_artifacts_present")["ok"] is True
    assert _check(validation, "csv_artifacts_nonempty")["ok"] is True
    assert _check(validation, "learned_submission_manifest_schema_supported")["ok"] is True
    assert _check(validation, "learned_submission_manifest_hashes_match")["ok"] is True
    assert validation["submission_manifest"]["schema_version"] == "0.1"

    validation_from_json = validate_learned_policy_submission_bundle(bundle=bundle_root / "learned_submission_bundle.json")
    assert validation_from_json["ok"] is True
    assert validation_from_json["bundle_json"].endswith("learned_submission_bundle.json")

    review = review_learned_policy_submission_bundle(bundle=bundle_root)
    assert review["schema_version"] == "0.1"
    assert review["ok"] is True
    assert review["method"] == "learned_tiny"
    assert review["recommendation"] == "manual_review_limited_sweep"
    assert "limited_planner_sweep" in review["limitations"]
    assert "submission_disclosure_incomplete" in review["limitations"]
    assert review["submission_manifest"]["schema_version"] == "0.1"
    assert review["score_v0"]["mean"] is not None
    assert review["dimensions"]["safety"]["collision_episode_count"] == 0


def test_learned_submission_bundle_cli_json_and_gate(tmp_path: Path) -> None:
    out_dir = tmp_path / "cli_bundle"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "microbench.cli",
            "learned-submission-bundle",
            "--out-dir",
            str(out_dir),
            "--method",
            "learned_tiny",
            "--policy",
            "tiny_learned",
            "--max-runs",
            "1",
            "--max-steps",
            "3",
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
    assert report["planner_sweep"]["run_count"] == 1
    assert (out_dir / "learned_submission_bundle.json").exists()
    assert (out_dir / "rl_contract.json").exists()
    assert (out_dir / "rl_freeze_check.json").exists()
    assert (out_dir / "rl_smoke.json").exists()
    assert (out_dir / "rl_calibration.json").exists()
    assert (out_dir / "learned_submission_manifest.json").exists()
    assert (out_dir / "planner_sweep" / "results.csv").exists()
    assert (out_dir / "planner_sweep" / "summary.csv").exists()

    validation_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "microbench.cli",
            "validate-learned-bundle",
            "--bundle",
            str(out_dir),
            "--require-pass",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    validation = json.loads(validation_proc.stdout)
    assert validation["ok"] is True
    assert validation["method"] == "learned_tiny"
    assert validation["submission_manifest"]["artifact_count"] >= 10

    review_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "microbench.cli",
            "review-learned-bundle",
            "--bundle",
            str(out_dir),
            "--require-pass",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    review = json.loads(review_proc.stdout)
    assert review["ok"] is True
    assert review["method"] == "learned_tiny"
    assert review["recommendation"] == "manual_review_limited_sweep"
    assert review["dimensions"]["compute"]["planner_error_count"] == 0
    assert review["submission_manifest"]["policy"]["name"] == "tiny_learned"


def test_validate_learned_bundle_reports_missing_artifacts(tmp_path: Path) -> None:
    bundle_root = tmp_path / "bundle"
    report = run_learned_policy_submission_bundle(
        out_dir=bundle_root,
        method="learned_tiny",
        policy="tiny_learned",
        max_runs=1,
        max_steps=3,
    )
    (bundle_root / report["artifacts"]["rl_smoke"]).unlink()

    validation = validate_learned_policy_submission_bundle(bundle=bundle_root)

    assert validation["ok"] is False
    missing_check = _check(validation, "required_artifacts_present")
    assert missing_check["ok"] is False
    assert "rl_smoke" in missing_check["details"]["missing"]


def test_learned_bundle_policy_spec_artifacts_are_portable(tmp_path: Path) -> None:
    spec_path = _tiny_policy_spec(tmp_path)
    bundle_root = tmp_path / "bundle_with_spec"
    report = run_learned_policy_submission_bundle(
        out_dir=bundle_root,
        method="learned_tiny",
        policy_spec=spec_path,
        max_runs=1,
        max_steps=3,
    )

    assert report["ok"] is True
    assert report["policy"] == "external_tiny_fixture"
    assert report["policy_spec"]["policy_name"] == "external_tiny_fixture"
    assert report["artifacts"]["policy_spec"] == "policy_spec.json"
    assert report["artifacts"]["policy_artifact"].startswith("policy_artifacts/")
    assert report["artifacts"]["learned_submission_manifest"] == "learned_submission_manifest.json"
    assert (bundle_root / report["artifacts"]["policy_spec"]).exists()
    assert (bundle_root / report["artifacts"]["policy_artifact"]).exists()

    copied_spec = json.loads((bundle_root / "policy_spec.json").read_text(encoding="utf-8"))
    assert copied_spec["artifact_path"] == report["artifacts"]["policy_artifact"]
    assert copied_spec["source_spec_path"] == str(spec_path)

    validation = validate_learned_policy_submission_bundle(bundle=bundle_root)
    assert validation["ok"] is True
    assert _check(validation, "optional_artifacts_present")["ok"] is True
    assert _check(validation, "learned_submission_manifest_policy_spec_provenance")["ok"] is True

    manifest = json.loads((bundle_root / "learned_submission_manifest.json").read_text(encoding="utf-8"))
    assert manifest["policy"]["policy_spec"]["policy_name"] == "external_tiny_fixture"
    assert any(item["name"] == "policy_artifact" and item["sha256"] for item in manifest["artifacts"])


def test_learned_bundle_policy_spec_can_drive_planner_sweep(tmp_path: Path) -> None:
    spec_path = _tiny_policy_spec(tmp_path)
    bundle_root = tmp_path / "bundle_with_spec_planner"
    report = run_learned_policy_submission_bundle(
        out_dir=bundle_root,
        method="learned_policy_spec",
        policy_spec=spec_path,
        max_runs=1,
        max_steps=3,
    )

    assert report["ok"] is True
    assert report["method"] == "learned_policy_spec"
    assert report["policy"] == "external_tiny_fixture"
    assert report["planner_sweep"]["policy_spec"] == str(spec_path)
    assert report["planner_sweep"]["run_count"] == 1
    assert _check(report, "method_marked_learned")["ok"] is True

    results_text = (bundle_root / "planner_sweep" / "results.csv").read_text(encoding="utf-8")
    assert "learned_policy_spec" in results_text


def test_learned_bundle_submission_manifest_overrides(tmp_path: Path) -> None:
    spec_path = _tiny_policy_spec(tmp_path)
    override_path = tmp_path / "submission_manifest_overrides.json"
    override_path.write_text(
        json.dumps(
            {
                "training_disclosure": {
                    "training_suites": ["custom_training_suite"],
                    "environment_steps": 12345,
                    "observation_normalization": "none",
                    "reward_configuration": {"progress": 1.0, "collision": -10.0},
                    "external_data": "none",
                    "pretrained_models": "none",
                    "hardware": "cpu",
                },
                "inference_disclosure": {
                    "uses_external_services": False,
                    "runtime_notes": "dependency-free fixture",
                },
                "dependencies": {
                    "inference_packages": [{"name": "numpy", "version": "test"}],
                },
                "review_notes": {
                    "privileged_information": "none",
                    "intended_category": "external_submission",
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    bundle_root = tmp_path / "bundle_with_manifest"
    report = run_learned_policy_submission_bundle(
        out_dir=bundle_root,
        method="learned_policy_spec",
        policy_spec=spec_path,
        submission_manifest=override_path,
        max_runs=1,
        max_steps=3,
    )

    assert report["ok"] is True
    manifest = json.loads((bundle_root / "learned_submission_manifest.json").read_text(encoding="utf-8"))
    assert manifest["training_disclosure"]["training_suites"] == ["custom_training_suite"]
    assert manifest["training_disclosure"]["environment_steps"] == 12345
    assert manifest["inference_disclosure"]["uses_external_services"] is False
    assert manifest["dependencies"]["inference_packages"][0]["name"] == "numpy"

    validation = validate_learned_policy_submission_bundle(bundle=bundle_root)
    assert validation["ok"] is True
    assert validation["submission_manifest"]["unknown_fields"] == []

    review = review_learned_policy_submission_bundle(bundle=bundle_root)
    assert "submission_disclosure_incomplete" not in review["limitations"]
    assert review["submission_manifest"]["training_disclosure"]["environment_steps"] == 12345
