from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from microbench.tools.baseline_evidence import run_baseline_reference_evidence


ROOT = Path(__file__).resolve().parents[1]


def test_baseline_evidence_cbf_mpc_checks_pass_with_promotion_blockers() -> None:
    report = run_baseline_reference_evidence(mpc_profile_iters=3, mpc_p95_max_ms=250.0)

    assert report["schema_version"] == "0.1"
    assert report["ok"] is True
    assert report["methods"] == ["cbf_qp", "mpc_local"]
    assert report["summary"]["failed_count"] == 0
    names = {(check["method"], check["name"]) for check in report["checks"]}
    assert ("cbf_qp", "cbf_projection_feasible_constraint") in names
    assert ("cbf_qp", "cbf_forced_fallback_is_bounded_and_reported") in names
    assert ("cbf_qp", "cbf_auto_solver_path_reports_status") in names
    assert ("mpc_local", "mpc_dense_3d_candidate_cap_and_signals") in names
    assert ("mpc_local", "mpc_dense_3d_profile_p95_bounded") in names
    assert "keep_experimental" in report["promotion_recommendations"]["cbf_qp"]
    assert "keep_experimental" in report["promotion_recommendations"]["mpc_local"]


def test_baseline_evidence_cli_json_writes_report(tmp_path: Path) -> None:
    out_dir = tmp_path / "evidence"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "microbench.cli",
            "baseline-evidence",
            "--out-dir",
            str(out_dir),
            "--mpc-profile-iters",
            "3",
            "--max-mpc-p95-ms",
            "250",
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
    assert (out_dir / "baseline_evidence.json").exists()
