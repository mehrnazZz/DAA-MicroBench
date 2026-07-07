from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from microbench.rl.submission_bundle import run_learned_policy_submission_bundle, validate_learned_policy_submission_bundle


ROOT = Path(__file__).resolve().parents[1]


def _check(report: dict, name: str) -> dict:
    return next(check for check in report["checks"] if check["name"] == name)


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
    assert _check(report, "method_marked_learned")["ok"] is True
    assert _check(report, "expected_artifacts_present")["ok"] is True

    bundle_root = tmp_path / "bundle"
    for path in report["artifacts"].values():
        assert not Path(path).is_absolute()
        assert (bundle_root / path).exists(), path
    assert (bundle_root / "learned_submission_bundle.json").exists()

    validation = validate_learned_policy_submission_bundle(bundle=bundle_root)
    assert validation["ok"] is True
    assert _check(validation, "required_artifacts_present")["ok"] is True
    assert _check(validation, "csv_artifacts_nonempty")["ok"] is True

    validation_from_json = validate_learned_policy_submission_bundle(bundle=bundle_root / "learned_submission_bundle.json")
    assert validation_from_json["ok"] is True
    assert validation_from_json["bundle_json"].endswith("learned_submission_bundle.json")


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
