from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from microbench.tools.advanced_baseline_comparison import run_advanced_baseline_comparison


ROOT = Path(__file__).resolve().parents[1]


def test_advanced_baseline_comparison_runs_compact_shared_lane(tmp_path: Path) -> None:
    out_dir = tmp_path / "comparison"
    report = run_advanced_baseline_comparison(
        out_dir=out_dir,
        methods=["orca_heuristic", "reciprocal_velocity_obstacle"],
        duration_s=1.0,
        n_agents=4,
        seed=2,
        comm_profile="ideal_50hz",
    )

    assert report["schema_version"] == "0.1"
    assert report["comparison_type"] == "advanced_baseline_3d_conflict"
    assert report["ok"] is True
    assert report["methods"] == ["orca_heuristic", "reciprocal_velocity_obstacle"]
    assert report["run_count"] == 2
    assert report["planned_run_count"] == 2
    json.dumps(report, allow_nan=False)
    assert len(report["ranking"]) == 2
    assert {row["method"] for row in report["ranking"]} == set(report["methods"])
    assert report["checks"] == {
        "complete_matrix": True,
        "guardrails_clear": True,
        "critical_metrics_finite": True,
    }
    assert (out_dir / "results.csv").exists()
    assert (out_dir / "summary.csv").exists()
    assert (out_dir / "baseline_report.json").exists()
    assert (out_dir / "advanced_baseline_comparison.json").exists()
    assert (out_dir / "result_schema.json").exists()
    assert Path(report["scenario_path"]).exists()


def test_advanced_baseline_comparison_save_traces_writes_full_episode_trace(tmp_path: Path) -> None:
    out_dir = tmp_path / "comparison_traces"
    report = run_advanced_baseline_comparison(
        out_dir=out_dir,
        methods=["reciprocal_velocity_obstacle"],
        duration_s=0.4,
        n_agents=4,
        seed=2,
        comm_profile="ideal_50hz",
        save_traces=True,
    )

    assert report["ok"] is True
    traces = sorted((out_dir / "episodes").glob("*/trace_episode.jsonl"))
    assert len(traces) == 1
    assert traces[0].stat().st_size > 0


def test_advanced_baseline_comparison_cli_json(tmp_path: Path) -> None:
    out_dir = tmp_path / "cli_comparison"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "microbench.cli",
            "advanced-baseline-comparison",
            "--out-dir",
            str(out_dir),
            "--methods",
            "orca_heuristic,reciprocal_velocity_obstacle",
            "--duration-s",
            "1.0",
            "--comm",
            "ideal_50hz",
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
    assert report["methods"] == ["orca_heuristic", "reciprocal_velocity_obstacle"]
    assert report["run_count"] == 2
    assert (out_dir / "advanced_baseline_comparison.json").exists()
