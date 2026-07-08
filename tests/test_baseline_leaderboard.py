from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from microbench.tools.baseline_leaderboard import run_baseline_leaderboard


ROOT = Path(__file__).resolve().parents[1]


def test_baseline_leaderboard_runs_capped_official_suite_and_ranks(tmp_path: Path) -> None:
    report = run_baseline_leaderboard(
        out_dir=tmp_path / "leaderboard",
        suites=["official_smoke_generated"],
        methods=["baseline_goal", "velocity_obstacle"],
        n_agents=[4],
        seeds=[0],
        comm_profiles=["ideal_50hz"],
        max_runs=2,
    )

    assert report["schema_version"] == "0.1"
    assert report["ok"] is True
    assert report["methods"] == ["baseline_goal", "velocity_obstacle"]
    assert len(report["suites"]) == 1
    suite = report["suites"][0]
    assert suite["suite"] == "official_smoke_generated"
    assert suite["run_count"] == 2
    assert suite["truncated_by_max_runs"] is True
    assert (tmp_path / "leaderboard" / "baseline_leaderboard.json").exists()
    assert (tmp_path / "leaderboard" / suite["report_path"]).exists()
    assert (tmp_path / "leaderboard" / suite["acceptance_path"]).exists()

    ranking = report["aggregate_ranking"]
    assert {entry["method"] for entry in ranking} == {"baseline_goal", "velocity_obstacle"}
    assert all(entry["score_v0_mean"] is not None for entry in ranking)
    assert [entry["rank"] for entry in ranking] == [1, 2]


def test_baseline_leaderboard_cli_json_and_gate(tmp_path: Path) -> None:
    out_dir = tmp_path / "cli_leaderboard"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "microbench.cli",
            "baseline-leaderboard",
            "--out-dir",
            str(out_dir),
            "--suites",
            "official_smoke_generated",
            "--methods",
            "baseline_goal,velocity_obstacle",
            "--n",
            "4",
            "--seeds",
            "0",
            "--comm",
            "ideal_50hz",
            "--max-runs",
            "2",
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
    assert report["suites"][0]["run_count"] == 2
    assert (out_dir / "baseline_leaderboard.json").exists()
