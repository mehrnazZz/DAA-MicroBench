from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from microbench.tools.baseline_audit import build_baseline_audit


ROOT = Path(__file__).resolve().parents[1]


def test_baseline_audit_public_alpha_contract() -> None:
    report = build_baseline_audit(root=ROOT)
    by_method = {entry["method"]: entry for entry in report["methods"]}

    assert report["schema_version"] == "0.1"
    assert report["public_alpha_ready"] is True
    assert report["stable_v1_ready"] is False
    assert report["required_public_alpha_reference_methods"] == [
        "orca_heuristic",
        "orca_with_staleness",
        "priority_yield",
        "negotiation_yield",
    ]
    assert report["summary"]["public_alpha_blockers"] == []

    for method in report["required_public_alpha_reference_methods"]:
        entry = by_method[method]
        assert entry["readiness"] == "public_alpha_reference_ready"
        assert entry["blockers"] == []
        assert entry["checks"]["factory_constructible"] is True
        assert entry["checks"]["docs_mentioned"] is True
        assert entry["checks"]["tests_mentioned"] is True
        assert entry["checks"]["supports_2d"] is True
        assert entry["checks"]["supports_3d"] is True
        assert entry["checks"]["in_official_suite_defaults"] is True
        assert entry["checks"]["has_acceptance_coverage"] is True

    assert by_method["cbf_qp"]["readiness"] == "experimental_runnable"
    assert by_method["mpc_local"]["readiness"] == "experimental_runnable"
    assert by_method["mpc_nonlinear"]["readiness"] == "experimental_runnable"
    assert by_method["mpc_nonlinear"]["checks"]["docs_mentioned"] is True
    assert by_method["mpc_nonlinear"]["checks"]["tests_mentioned"] is True
    assert by_method["ego_swarm_opt"]["readiness"] == "experimental_runnable"
    assert by_method["ego_swarm_opt"]["checks"]["docs_mentioned"] is True
    assert by_method["ego_swarm_opt"]["checks"]["tests_mentioned"] is True
    assert by_method["velocity_obstacle"]["readiness"] == "experimental_runnable"
    assert by_method["velocity_obstacle"]["checks"]["docs_mentioned"] is True
    assert by_method["velocity_obstacle"]["checks"]["tests_mentioned"] is True
    assert by_method["reciprocal_velocity_obstacle"]["readiness"] == "experimental_runnable"
    assert by_method["reciprocal_velocity_obstacle"]["checks"]["docs_mentioned"] is True
    assert by_method["reciprocal_velocity_obstacle"]["checks"]["tests_mentioned"] is True
    assert by_method["learned_tiny"]["readiness"] == "experimental_runnable"
    assert by_method["learned_tiny"]["checks"]["docs_mentioned"] is True
    assert by_method["learned_tiny"]["checks"]["tests_mentioned"] is True
    assert by_method["learned_policy_spec"]["readiness"] == "externally_configured_bridge"
    assert by_method["learned_policy_spec"]["checks"]["factory_constructible"] is True
    assert by_method["learned_policy_spec"]["checks"]["docs_mentioned"] is True
    assert by_method["learned_policy_spec"]["checks"]["tests_mentioned"] is True
    assert by_method["negotiation_yield"]["readiness"] == "public_alpha_reference_ready"
    assert by_method["baseline_goal"]["readiness"] == "illustrative_or_template"


def test_baseline_audit_cli_json_and_public_alpha_gate() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "microbench.cli",
            "baseline-audit",
            "--root",
            str(ROOT),
            "--require-public-alpha-ready",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    report = json.loads(proc.stdout)
    assert report["public_alpha_ready"] is True
    assert any(entry["method"] == "orca_heuristic" for entry in report["methods"])


def test_baseline_audit_stable_v1_gate_fails_while_experimental_baselines_remain() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "microbench.cli",
            "baseline-audit",
            "--root",
            str(ROOT),
            "--require-stable-v1-ready",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert proc.returncode != 0
    assert "stable-v1 baseline blockers" in proc.stderr
