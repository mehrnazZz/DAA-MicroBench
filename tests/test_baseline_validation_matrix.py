from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from microbench.tools.baseline_validation_matrix import (
    BASELINE_VALIDATION_MATRIX_SCHEMA_VERSION,
    run_baseline_validation_matrix,
)


REQUIRED_LANES = {
    "head_on",
    "crossing",
    "urban_obstacle",
    "communication_delay",
    "high_n_dense_merge",
}


def test_baseline_validation_matrix_plan_only_covers_required_lanes(tmp_path: Path) -> None:
    report = run_baseline_validation_matrix(
        out_dir=tmp_path / "plan",
        methods=["cbf_qp", "learned_tiny"],
        plan_only=True,
    )

    assert report["schema_version"] == BASELINE_VALIDATION_MATRIX_SCHEMA_VERSION
    assert report["plan_only"] is True
    assert report["run_count"] == 0
    assert {lane["lane_id"] for lane in report["lanes"]} == REQUIRED_LANES
    assert report["planned_run_count"] == len(REQUIRED_LANES) * 2
    assert report["selected_run_count"] == report["planned_run_count"]
    assert any(entry["method"] == "learned_tiny" and entry["learned"] for entry in report["matrix"])
    assert any(
        entry["lane_id"] == "communication_delay"
        and "stale" in " ".join(entry["expected_behavior"]).lower()
        for entry in report["matrix"]
    )


def test_baseline_validation_matrix_runs_single_head_on_lane(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    report = run_baseline_validation_matrix(
        out_dir=out_dir,
        methods=["baseline_goal"],
        lanes=["head_on"],
        duration_s=1.0,
    )

    assert report["ok"] is True
    assert report["run_count"] == 1
    assert Path(report["results_csv"]).exists()
    assert Path(report["summary_csv"]).exists()
    assert any(check["name"] == "finite_core_metrics" and check["ok"] for check in report["checks"])
    assert report["methods_detail"][0]["gate_pass"] is True


def test_baseline_validation_matrix_cli_plan_only(tmp_path: Path) -> None:
    out_dir = tmp_path / "cli_plan"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "microbench.cli",
            "baseline-validation-matrix",
            "--out-dir",
            str(out_dir),
            "--methods",
            "learned_tiny",
            "--plan-only",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(proc.stdout)
    assert report["plan_only"] is True
    assert report["methods"] == ["learned_tiny"]
    assert (out_dir / "baseline_validation_matrix.json").exists()
