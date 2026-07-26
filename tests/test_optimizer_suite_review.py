from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from microbench.tools.optimizer_suite_review import run_optimizer_suite_review


ROOT = Path(__file__).resolve().parents[1]


def _write_fake_optimizer_leaderboard_fixture(out_dir: Path, row: dict[str, Any]) -> dict[str, Any]:
    suite = "fake_suite"
    generated = out_dir / suite / "_generated_scenarios" / suite
    generated.mkdir(parents=True)
    (generated / "guardrail_case.yaml").write_text(
        """
scenario:
  name: guardrail_case
benchmark:
  dimension: 3d
world:
  planar: false
""".lstrip(),
        encoding="utf-8",
    )
    (generated / "manifest.yaml").write_text("suite: fake_suite\n", encoding="utf-8")
    run_dir = out_dir / suite / "runs"
    run_dir.mkdir(parents=True)
    results_csv = run_dir / "results.csv"
    fieldnames = [
        "scenario",
        "method",
        "comm_profile",
        "N",
        "seed",
        "duration_s",
        "collision_episode",
        "min_sep_min_m",
        "completion_rate",
        "planner_ms_per_tick_per_agent_p95",
        "planner_timeout_count",
        "planner_error_count",
        "planner_fallback_count",
    ]
    with results_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
    leaderboard_path = out_dir / "baseline_leaderboard.json"
    leaderboard = {
        "ok": True,
        "complete": False,
        "selected_complete": True,
        "timeout_run_count": 1,
        "leaderboard_path": str(leaderboard_path),
        "suites": [
            {
                "suite": suite,
                "ok": True,
                "results_csv": str(results_csv.relative_to(out_dir)),
                "summary_csv": str((run_dir / "summary.csv").relative_to(out_dir)),
                "suite_manifest": str((generated / "manifest.yaml").relative_to(out_dir)),
                "selected_complete": True,
                "complete": False,
            }
        ],
    }
    leaderboard_path.write_text(json.dumps(leaderboard), encoding="utf-8")
    return leaderboard


def _guardrail_row(**overrides: Any) -> dict[str, Any]:
    row = {
        "scenario": "guardrail_case",
        "method": "mpc_nonlinear",
        "comm_profile": "ideal_50hz",
        "N": 4,
        "seed": 0,
        "duration_s": "",
        "collision_episode": 0,
        "min_sep_min_m": 0.25,
        "completion_rate": 1.0,
        "planner_ms_per_tick_per_agent_p95": 2500.0,
        "planner_timeout_count": 1,
        "planner_error_count": 1,
        "planner_fallback_count": 0,
    }
    row.update(overrides)
    return row


def test_optimizer_suite_review_runs_capped_suite_and_writes_report(tmp_path: Path) -> None:
    out_dir = tmp_path / "optimizer_review"
    report = run_optimizer_suite_review(
        out_dir=out_dir,
        suites=["official_smoke_generated"],
        methods=["baseline_goal"],
        n_agents=[4],
        seeds=[0],
        comm_profiles=["ideal_50hz"],
        max_runs=1,
        max_trace_cases=1,
    )

    assert report["schema_version"] == "0.2"
    assert report["review_type"] == "optimizer_suite_review"
    assert report["ok"] is True
    assert report["official_acceptance_ok"] is True
    assert report["selected_complete"] is True
    assert report["publication_complete"] is False
    assert report["max_runs_strategy"] == "balanced"
    assert report["baseline_leaderboard"]["max_runs_strategy"] == "balanced"
    assert report["guardrail_retries"] == 1
    assert report["guardrail_retry_evidence"] == []
    assert report["methods"] == ["baseline_goal"]
    assert report["suites"] == ["official_smoke_generated"]
    assert report["method_summaries"][0]["run_count"] == 1
    assert report["review_cases"]
    assert "foxglove-export" in report["review_cases"][0]["foxglove_export_command"]
    assert " --save-trace" in report["review_cases"][0]["rerun_trace_command"]
    assert Path(report["review_cases"][0]["trace_scenario_path"]).exists()
    assert str(report["review_cases"][0]["trace_scenario_path"]) in report["review_cases"][0]["rerun_trace_command"]
    assert (out_dir / "baseline_leaderboard.json").exists()
    assert (out_dir / "optimizer_suite_review.json").exists()


def test_optimizer_suite_review_retries_transient_guardrails(monkeypatch, tmp_path: Path) -> None:
    out_dir = tmp_path / "optimizer_review_transient"
    fake_leaderboard = _write_fake_optimizer_leaderboard_fixture(out_dir, _guardrail_row())

    def fake_leaderboard_runner(**kwargs):
        assert kwargs["out_dir"] == out_dir
        return fake_leaderboard

    def fake_retry_runner(spec, *, run_timeout_s):
        assert spec.method == "mpc_nonlinear"
        assert run_timeout_s == 5.0
        return _guardrail_row(
            duration_s=10.0,
            planner_ms_per_tick_per_agent_p95=25.0,
            planner_timeout_count=0,
            planner_error_count=0,
            planner_fallback_count=0,
        ), False

    monkeypatch.setattr("microbench.tools.optimizer_suite_review.run_baseline_leaderboard", fake_leaderboard_runner)
    monkeypatch.setattr("microbench.tools.optimizer_suite_review._run_episode_checked", fake_retry_runner)

    report = run_optimizer_suite_review(
        out_dir=out_dir,
        suites=["fake_suite"],
        methods=["mpc_nonlinear"],
        max_trace_cases=0,
        run_timeout_s=5.0,
    )

    assert report["ok"] is True
    assert report["findings"]["guardrail_rows"] == 1
    assert report["findings"]["transient_guardrail_rows"] == 1
    assert report["findings"]["persistent_guardrail_rows"] == 0
    assert report["findings"]["persistent_hard_timeout_rows"] == 0
    assert report["guardrail_retry_resolved_rows"] == 1
    assert report["guardrail_retry_evidence"][0]["status"] == "resolved_on_retry"
    assert report["guardrail_retry_evidence"][0]["attempts"][0]["status"] == "clear"
    assert (
        out_dir
        / "guardrail_retries"
        / "fake_suite"
        / "guardrail_case_mpc_nonlinear_n4_seed0_ideal_50hz"
        / "attempt_1"
        / "result_row.json"
    ).exists()


def test_optimizer_suite_review_fails_persistent_guardrails(monkeypatch, tmp_path: Path) -> None:
    out_dir = tmp_path / "optimizer_review_persistent"
    fake_leaderboard = _write_fake_optimizer_leaderboard_fixture(out_dir, _guardrail_row())

    monkeypatch.setattr("microbench.tools.optimizer_suite_review.run_baseline_leaderboard", lambda **kwargs: fake_leaderboard)
    monkeypatch.setattr(
        "microbench.tools.optimizer_suite_review._run_episode_checked",
        lambda spec, run_timeout_s: (
            _guardrail_row(duration_s=10.0, planner_timeout_count=0, planner_error_count=0, planner_fallback_count=1),
            False,
        ),
    )

    report = run_optimizer_suite_review(
        out_dir=out_dir,
        suites=["fake_suite"],
        methods=["mpc_nonlinear"],
        max_trace_cases=0,
    )

    assert report["ok"] is False
    assert report["findings"]["guardrail_rows"] == 1
    assert report["findings"]["transient_guardrail_rows"] == 0
    assert report["findings"]["persistent_guardrail_rows"] == 1
    assert report["guardrail_retry_resolved_rows"] == 0
    assert report["guardrail_retry_evidence"][0]["status"] == "persistent_guardrail"
    assert report["guardrail_retry_evidence"][0]["attempts"][0]["status"] == "guardrail_present"


def test_optimizer_suite_review_can_write_full_trace_for_review_case(tmp_path: Path) -> None:
    out_dir = tmp_path / "optimizer_review_traces"
    report = run_optimizer_suite_review(
        out_dir=out_dir,
        suites=["official_smoke_generated"],
        methods=["baseline_goal"],
        n_agents=[4],
        seeds=[0],
        comm_profiles=["ideal_50hz"],
        max_runs=1,
        max_trace_cases=1,
        save_review_traces=True,
        trace_max_steps=500,
    )

    case = report["review_cases"][0]
    trace_path = Path(case["trace_path"])
    assert case["trace_status"] == "written"
    assert trace_path.exists()
    assert trace_path.read_text(encoding="utf-8").splitlines()[0].startswith('{"kind": "meta"')


def test_optimizer_suite_review_cli_json(tmp_path: Path) -> None:
    out_dir = tmp_path / "cli_optimizer_review"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "microbench.cli",
            "optimizer-suite-review",
            "--out-dir",
            str(out_dir),
            "--suites",
            "official_smoke_generated",
            "--methods",
            "baseline_goal",
            "--n",
            "4",
            "--seeds",
            "0",
            "--comm",
            "ideal_50hz",
            "--max-runs",
            "1",
            "--max-trace-cases",
            "1",
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
    assert report["baseline_leaderboard"]["suites"][0]["selected_completed_count"] == 1
    assert (out_dir / "optimizer_suite_review.json").exists()
