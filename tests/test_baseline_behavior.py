from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from microbench.tools.baseline_behavior import run_baseline_behavior_smoke


ROOT = Path(__file__).resolve().parents[1]


def _check(report: dict, name: str) -> dict:
    return next(check for check in report["checks"] if check["name"] == name)


def test_baseline_behavior_smoke_agentic_message_contract(tmp_path: Path) -> None:
    report = run_baseline_behavior_smoke(
        out_dir=tmp_path / "baseline_smoke",
        methods=("priority_yield", "negotiation_yield"),
    )

    assert report["schema_version"] == "0.1"
    assert report["ok"] is True
    assert report["run_count"] == 4
    assert report["scenario_ids"] == ["head_on_2d_easy", "sphere_swap_3d_medium"]
    assert _check(report, "run_count")["ok"] is True
    assert _check(report, "finite_key_metrics")["ok"] is True
    assert _check(report, "planner_errors_clear")["ok"] is True
    assert _check(report, "public_alpha_guardrails_clear")["ok"] is True
    assert _check(report, "zero_planner_guardrails")["ok"] is True
    assert _check(report, "two_d_and_three_d_coverage")["ok"] is True

    priority_signal = _check(report, "priority_yield_message_signal")
    assert priority_signal["ok"] is True
    assert priority_signal["details"]["attempted"] > 0
    assert priority_signal["details"]["delivered"] > 0

    negotiation_signal = _check(report, "negotiation_yield_signal")
    assert negotiation_signal["ok"] is True
    assert negotiation_signal["details"]["proposals"] > 0
    assert negotiation_signal["details"]["acks"] > 0

    assert Path(report["results_csv"]).exists()
    assert Path(report["summary_csv"]).exists()
    assert Path(report["suite_manifest"]).exists()


def test_baseline_behavior_smoke_output_contracts(tmp_path: Path) -> None:
    report = run_baseline_behavior_smoke(
        out_dir=tmp_path / "baseline_smoke",
        methods=(
            "cbf_qp",
            "mpc_local",
            "mpc_nonlinear",
            "dmpc_best_response",
            "bvc_tube_dmpc",
            "dynamic_tube_dmpc",
            "rmader",
            "ego_swarm",
            "ego_swarm_opt",
            "velocity_obstacle",
            "reciprocal_velocity_obstacle",
            "learned_tiny",
            "intent_dummy",
        ),
    )

    assert report["ok"] is True
    assert report["run_count"] == 20
    assert report["contract_only_methods"] == ["bvc_tube_dmpc", "dynamic_tube_dmpc", "rmader"]
    assert _check(report, "cbf_qp_debug_contract")["ok"] is True
    assert _check(report, "mpc_local_debug_contract")["ok"] is True
    assert _check(report, "mpc_nonlinear_debug_contract")["ok"] is True
    assert _check(report, "dmpc_best_response_debug_contract")["ok"] is True
    assert _check(report, "bvc_tube_dmpc_debug_contract")["ok"] is True
    assert _check(report, "dynamic_tube_dmpc_debug_contract")["ok"] is True
    assert _check(report, "rmader_debug_contract")["ok"] is True
    assert _check(report, "ego_swarm_debug_contract")["ok"] is True
    assert _check(report, "ego_swarm_opt_debug_contract")["ok"] is True
    assert _check(report, "velocity_obstacle_debug_contract")["ok"] is True
    assert _check(report, "reciprocal_velocity_obstacle_debug_contract")["ok"] is True
    assert _check(report, "learned_tiny_model_contract")["ok"] is True
    assert _check(report, "intent_dummy_intent_contract")["ok"] is True


def test_baseline_smoke_cli_json_and_gate(tmp_path: Path) -> None:
    out_dir = tmp_path / "cli_smoke"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "microbench.cli",
            "baseline-smoke",
            "--out-dir",
            str(out_dir),
            "--methods",
            "baseline_goal",
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
    assert report["methods"] == ["baseline_goal"]
    assert report["run_count"] == 2
    assert (out_dir / "baseline_smoke.json").exists()
