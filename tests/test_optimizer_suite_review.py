from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from microbench.tools.optimizer_suite_review import run_optimizer_suite_review


ROOT = Path(__file__).resolve().parents[1]


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

    assert report["schema_version"] == "0.1"
    assert report["review_type"] == "optimizer_suite_review"
    assert report["ok"] is True
    assert report["official_acceptance_ok"] is True
    assert report["selected_complete"] is True
    assert report["publication_complete"] is False
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
