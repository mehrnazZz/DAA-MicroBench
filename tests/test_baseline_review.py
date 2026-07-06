from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from microbench.tools.baseline_review import run_baseline_stable_review


ROOT = Path(__file__).resolve().parents[1]


def test_baseline_review_plan_only_covers_3d_and_agentic_official_lanes(tmp_path: Path) -> None:
    report = run_baseline_stable_review(
        out_dir=tmp_path / "plan",
        root=ROOT,
        methods=["negotiation_yield"],
        plan_only=True,
    )

    assert report["schema_version"] == "0.1"
    assert report["plan_only"] is True
    assert report["run_count"] == 0
    assert report["planned_run_count"] == 3
    assert report["methods"] == ["negotiation_yield"]
    assert {lane["suite"] for lane in report["lanes"]} == {"official_3d_stress", "official_agentic_stress"}
    assert {lane["category"] for lane in report["lanes"]} == {
        "3d_stress",
        "degraded_sensing_comm",
        "agentic_stress",
    }


def test_baseline_review_runs_short_3d_lane(tmp_path: Path) -> None:
    out_dir = tmp_path / "review"
    report = run_baseline_stable_review(
        out_dir=out_dir,
        root=ROOT,
        methods=["cbf_qp"],
        lanes=["3d_sphere_swap"],
        duration_s=2.0,
    )

    assert report["plan_only"] is False
    assert report["run_count"] == 1
    assert report["review_checks_pass"] is True
    assert (out_dir / "results.csv").exists()
    assert (out_dir / "summary.csv").exists()
    detail = report["methods_detail"][0]
    assert detail["method"] == "cbf_qp"
    assert detail["review_checks_pass"] is True
    assert detail["metadata_recommendation"] == "needs_reference_role_decision"
    assert detail["failed_checks"] == []


def test_baseline_review_negotiation_yield_passes_longer_review_lanes(tmp_path: Path) -> None:
    report = run_baseline_stable_review(
        out_dir=tmp_path / "negotiation_review",
        root=ROOT,
        methods=["negotiation_yield"],
        duration_s=20.0,
    )

    assert report["review_checks_pass"] is True
    assert report["run_count"] == 3
    detail = report["methods_detail"][0]
    assert detail["review_checks_pass"] is True
    assert detail["metadata_recommendation"] == "review_for_stable_metadata"
    assert detail["failed_checks"] == []
    assert {row["lane_id"] for row in report["rows"]} == {
        "3d_sphere_swap",
        "3d_sensor_degraded",
        "agentic_priority_degraded",
    }
    assert all(float(row["min_sep_min_m"]) >= 0.0 for row in report["rows"])


def test_baseline_review_cli_json_plan_only(tmp_path: Path) -> None:
    out_dir = tmp_path / "cli_plan"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "microbench.cli",
            "baseline-review",
            "--out-dir",
            str(out_dir),
            "--methods",
            "negotiation_yield",
            "--plan-only",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    report = json.loads(proc.stdout)
    assert report["plan_only"] is True
    assert report["planned_run_count"] == 3
    assert (out_dir / "baseline_review.json").exists()
