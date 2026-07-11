from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from microbench.tools.baseline_evidence import run_baseline_reference_evidence


ROOT = Path(__file__).resolve().parents[1]


def test_baseline_evidence_cbf_mpc_checks_pass_with_promotion_blockers() -> None:
    report = run_baseline_reference_evidence(
        mpc_profile_iters=3,
        mpc_p95_max_ms=250.0,
        optimizer_profile_iters=2,
        optimizer_p95_max_ms=250.0,
    )

    assert report["schema_version"] == "0.3"
    assert report["ok"] is True
    assert report["methods"] == [
        "cbf_qp",
        "mpc_local",
        "mpc_nonlinear",
        "ego_swarm_opt",
        "velocity_obstacle",
        "reciprocal_velocity_obstacle",
    ]
    assert report["summary"]["failed_count"] == 0
    names = {(check["method"], check["name"]) for check in report["checks"]}
    assert ("cbf_qp", "cbf_projection_feasible_constraint") in names
    assert ("cbf_qp", "cbf_forced_fallback_is_bounded_and_reported") in names
    assert ("cbf_qp", "cbf_stale_track_inflates_barrier") in names
    assert ("cbf_qp", "cbf_auto_solver_path_reports_status") in names
    assert ("mpc_local", "mpc_dense_3d_candidate_cap_and_signals") in names
    assert ("mpc_local", "mpc_stale_track_inflates_rollout_risk") in names
    assert ("mpc_local", "mpc_dense_3d_profile_p95_bounded") in names
    assert ("mpc_nonlinear", "mpc_nonlinear_dense_3d_optimizer_signals") in names
    assert ("mpc_nonlinear", "mpc_nonlinear_degraded_intent_and_v2v_inflate_risk") in names
    assert ("mpc_nonlinear", "mpc_nonlinear_scipy_or_fallback_solver_reports_status") in names
    assert ("mpc_nonlinear", "mpc_nonlinear_dense_3d_profile_p95_bounded") in names
    assert ("ego_swarm_opt", "ego_swarm_opt_dense_3d_optimizer_signals") in names
    assert ("ego_swarm_opt", "ego_swarm_opt_degraded_intent_and_v2v_inflate_risk") in names
    assert ("ego_swarm_opt", "ego_swarm_opt_scipy_or_fallback_solver_reports_status") in names
    assert ("ego_swarm_opt", "ego_swarm_opt_dense_3d_profile_p95_bounded") in names
    assert ("velocity_obstacle", "vo_finite_horizon_cone_signals") in names
    assert ("reciprocal_velocity_obstacle", "rvo_hrvo_apex_and_candidate_signals") in names
    assert ("reciprocal_velocity_obstacle", "rvo_priority_and_stale_responsibility") in names
    assert "keep_experimental" in report["promotion_recommendations"]["cbf_qp"]
    assert "keep_experimental" in report["promotion_recommendations"]["mpc_local"]
    assert "keep_experimental" in report["promotion_recommendations"]["mpc_nonlinear"]
    assert "keep_experimental" in report["promotion_recommendations"]["ego_swarm_opt"]
    assert "keep_experimental" in report["promotion_recommendations"]["velocity_obstacle"]
    assert "keep_experimental" in report["promotion_recommendations"]["reciprocal_velocity_obstacle"]


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
            "--optimizer-profile-iters",
            "2",
            "--max-optimizer-p95-ms",
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


def test_baseline_evidence_can_write_optimizer_trace_artifacts(tmp_path: Path) -> None:
    out_dir = tmp_path / "evidence_traces"
    report = run_baseline_reference_evidence(
        mpc_profile_iters=1,
        mpc_p95_max_ms=250.0,
        optimizer_profile_iters=1,
        optimizer_p95_max_ms=300.0,
        artifact_dir=out_dir,
        save_optimizer_traces=True,
    )

    assert report["ok"] is True
    trace_checks = [
        check for check in report["checks"] if check["name"].endswith("_foxglove_trace_jsonl_written")
    ]
    assert {check["method"] for check in trace_checks} == {"mpc_nonlinear", "ego_swarm_opt"}
    for check in trace_checks:
        trace_path = Path(check["details"]["trace_path"])
        assert trace_path.exists()
        assert trace_path.read_text(encoding="utf-8").splitlines()[0].startswith('{"kind": "meta"')
        assert "foxglove-export" in check["details"]["foxglove_export_command"]
