from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from microbench.tools.baseline_promotion import run_baseline_promotion_calibration


ROOT = Path(__file__).resolve().parents[1]


def test_baseline_promotion_calibrates_experimentals_but_blocks_stable_v1(tmp_path: Path) -> None:
    report = run_baseline_promotion_calibration(out_dir=tmp_path / "promotion", root=ROOT)
    by_method = {entry["method"]: entry for entry in report["methods_detail"]}

    assert report["schema_version"] == "0.1"
    assert report["public_alpha_calibrated"] is True
    assert report["stable_v1_ready"] is False
    assert report["summary"]["calibration_ready_count"] == 3
    assert report["summary"]["stable_v1_ready_count"] == 0
    assert report["experimental_suite"]["status"] == "PASS"
    assert report["experimental_suite"]["run_count"] == 4

    for method in ("cbf_qp", "mpc_local", "negotiation_yield"):
        entry = by_method[method]
        assert entry["calibration_ready"] is True
        assert entry["stable_v1_ready"] is False
        assert entry["calibration_blockers"] == []
        assert entry["behavior_metrics"]["guardrail_total"] == 0
        assert entry["behavior_metrics"]["collision_episode_count"] == 0
        assert "metadata_status_not_stable" in entry["stable_v1_blockers"]
        assert "stable_3d_stress_acceptance_bands_missing" in entry["stable_v1_blockers"]
        assert "degraded_comm_or_sensor_calibration_missing" in entry["stable_v1_blockers"]
        assert "smoke_collision_episode_present" not in entry["stable_v1_blockers"]

    assert "role_not_reference_baseline" in by_method["cbf_qp"]["stable_v1_blockers"]
    assert "role_not_reference_baseline" in by_method["mpc_local"]["stable_v1_blockers"]
    assert by_method["negotiation_yield"]["experimental_acceptance_status"] is None
    assert by_method["cbf_qp"]["experimental_acceptance_status"] == "PASS"
    assert by_method["mpc_local"]["experimental_acceptance_status"] == "PASS"


def test_baseline_promotion_cli_json_and_calibrated_gate(tmp_path: Path) -> None:
    out_dir = tmp_path / "cli_promotion"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "microbench.cli",
            "baseline-promotion",
            "--out-dir",
            str(out_dir),
            "--methods",
            "negotiation_yield",
            "--skip-experimental-suite",
            "--require-calibrated",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    report = json.loads(proc.stdout)
    assert report["public_alpha_calibrated"] is True
    assert report["stable_v1_ready"] is False
    assert report["methods"] == ["negotiation_yield"]
    assert (out_dir / "baseline_promotion.json").exists()


def test_baseline_promotion_stable_v1_gate_fails_while_experimental(tmp_path: Path) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "microbench.cli",
            "baseline-promotion",
            "--out-dir",
            str(tmp_path / "stable_fail"),
            "--methods",
            "negotiation_yield",
            "--skip-experimental-suite",
            "--require-stable-v1-ready",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert proc.returncode != 0
    assert "stable-v1 blockers present" in proc.stderr
