from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

from microbench.tools import build_baseline_report


FIELDS = [
    "method",
    "scenario",
    "comm_profile",
    "N",
    "episodes",
    "collision_episode_rate",
    "collisions_mean",
    "unique_collision_pairs_mean",
    "min_sep_min_mean",
    "min_sep_p05_mean",
    "completion_rate_mean",
    "mean_time_to_goal_mean",
    "deadlock_time_pct_mean",
    "planner_ms_p95",
    "planner_timeout_count_mean",
    "planner_error_count_mean",
    "planner_fallback_count_mean",
]


def _write_summary(path: Path) -> None:
    rows = [
        {
            "method": "mpc_local",
            "scenario": "sphere_swap_3d_medium",
            "comm_profile": "ideal_50hz",
            "N": 4,
            "episodes": 1,
            "collision_episode_rate": 0.0,
            "collisions_mean": 0.0,
            "unique_collision_pairs_mean": 0.0,
            "min_sep_min_mean": 5.0,
            "min_sep_p05_mean": 5.5,
            "completion_rate_mean": 1.0,
            "mean_time_to_goal_mean": 27.5,
            "deadlock_time_pct_mean": 0.0,
            "planner_ms_p95": 2.1234567,
            "planner_timeout_count_mean": 0.0,
            "planner_error_count_mean": 0.0,
            "planner_fallback_count_mean": 0.0,
        },
        {
            "method": "cbf_qp",
            "scenario": "head_on_2d_easy",
            "comm_profile": "ideal_50hz",
            "N": 4,
            "episodes": 1,
            "collision_episode_rate": 1.0,
            "collisions_mean": 17.0,
            "unique_collision_pairs_mean": 1.0,
            "min_sep_min_mean": -0.2,
            "min_sep_p05_mean": 0.1,
            "completion_rate_mean": 0.0,
            "mean_time_to_goal_mean": "nan",
            "deadlock_time_pct_mean": 0.01,
            "planner_ms_p95": 0.02,
            "planner_timeout_count_mean": 0.0,
            "planner_error_count_mean": 0.0,
            "planner_fallback_count_mean": 0.0,
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_results(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method"])
        writer.writeheader()
        writer.writerow({"method": "cbf_qp"})
        writer.writerow({"method": "mpc_local"})


def test_build_baseline_report_is_path_independent_and_sorted(tmp_path: Path) -> None:
    summary = tmp_path / "summary.csv"
    results = tmp_path / "results.csv"
    _write_summary(summary)
    _write_results(results)

    report = build_baseline_report(
        summary_csv=summary,
        results_csv=results,
        suite="official_experimental_baselines",
        generated_by="unit-test",
    )

    assert report["schema_version"] == "0.1"
    assert report["suite"] == "official_experimental_baselines"
    assert report["generated_by"] == "unit-test"
    assert report["methods"] == ["cbf_qp", "mpc_local"]
    assert report["scenarios"] == ["head_on_2d_easy", "sphere_swap_3d_medium"]
    assert report["n_agents"] == [4]
    assert report["run_count"] == 2
    assert report["rows"][0]["method"] == "cbf_qp"
    assert report["rows"][0]["mean_time_to_goal_mean"] is None
    assert report["rows"][1]["planner_ms_p95"] == 2.123457
    assert report["method_summaries"][0]["method"] == "cbf_qp"
    assert report["method_summaries"][0]["min_sep_min_worst"] == -0.2


def test_baseline_report_cli_writes_json(tmp_path: Path) -> None:
    summary = tmp_path / "summary.csv"
    out = tmp_path / "report.json"
    _write_summary(summary)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "microbench.cli",
            "baseline-report",
            "--summary",
            str(summary),
            "--suite",
            "official_experimental_baselines",
            "--out",
            str(out),
            "--generated-by",
            "pytest",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )

    assert "baseline comparison report saved" in proc.stdout
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["generated_by"] == "pytest"
    assert len(report["rows"]) == 2


def test_golden_baseline_comparison_fixture_schema() -> None:
    fixture = Path(__file__).resolve().parents[1] / "golden" / "baseline_comparison" / "report.json"
    report = json.loads(fixture.read_text(encoding="utf-8"))

    assert report["schema_version"] == "0.1"
    assert report["suite"] == "official_experimental_baselines"
    assert report["methods"] == ["baseline_goal", "cbf_qp", "mpc_local", "orca_heuristic"]
    assert report["scenarios"] == ["head_on_2d_easy", "sphere_swap_3d_medium"]
    assert report["n_agents"] == [4]
    assert report["run_count"] == 8
    assert len(report["rows"]) == 8
    assert len(report["method_summaries"]) == 4
    for row in report["rows"]:
        assert row["planner_timeout_count_mean"] == 0
        assert row["planner_error_count_mean"] == 0
        assert row["planner_fallback_count_mean"] == 0
