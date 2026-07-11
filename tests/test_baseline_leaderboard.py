from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

import microbench.tools.baseline_leaderboard as baseline_leaderboard
from microbench.tools.baseline_leaderboard import run_baseline_leaderboard
from microbench.types import RunSpec


ROOT = Path(__file__).resolve().parents[1]


def _dummy_spec(scenario: str, method: str, seed: int) -> RunSpec:
    return RunSpec(
        scenario_path=f"{scenario}.yaml",
        method=method,
        n_agents=4,
        seed=seed,
        comm_profile="ideal_50hz",
        out_dir="unused",
        save_trace=False,
    )


def test_balanced_max_runs_strategy_spreads_across_scenarios_and_methods() -> None:
    specs = [
        _dummy_spec(scenario, method, seed)
        for scenario in ("s0", "s1", "s2")
        for method in ("m0", "m1")
        for seed in (0, 1)
    ]

    prefix = baseline_leaderboard._select_specs(specs, max_runs=4, strategy="prefix")
    balanced = baseline_leaderboard._select_specs(specs, max_runs=4, strategy="balanced")

    assert [(Path(s.scenario_path).stem, s.method, s.seed) for s in prefix] == [
        ("s0", "m0", 0),
        ("s0", "m0", 1),
        ("s0", "m1", 0),
        ("s0", "m1", 1),
    ]
    assert [(Path(s.scenario_path).stem, s.method, s.seed) for s in balanced] == [
        ("s0", "m0", 0),
        ("s0", "m1", 0),
        ("s1", "m0", 0),
        ("s1", "m1", 0),
    ]


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

    assert report["schema_version"] == "0.2"
    assert report["max_runs_strategy"] == "prefix"
    assert report["ok"] is True
    assert report["complete"] is False
    assert report["selected_complete"] is True
    assert report["methods"] == ["baseline_goal", "velocity_obstacle"]
    assert len(report["suites"]) == 1
    suite = report["suites"][0]
    assert suite["suite"] == "official_smoke_generated"
    assert suite["run_count"] == 2
    assert suite["selected_run_count"] == 2
    assert suite["selected_completed_count"] == 2
    assert suite["truncated_by_max_runs"] is True
    assert (tmp_path / "leaderboard" / "baseline_leaderboard.json").exists()
    assert (tmp_path / "leaderboard" / suite["report_path"]).exists()
    assert (tmp_path / "leaderboard" / suite["acceptance_path"]).exists()
    assert (tmp_path / "leaderboard" / suite["progress_path"]).exists()

    ranking = report["aggregate_ranking"]
    assert {entry["method"] for entry in ranking} == {"baseline_goal", "velocity_obstacle"}
    assert all(entry["score_v0_mean"] is not None for entry in ranking)
    assert [entry["rank"] for entry in ranking] == [1, 2]


def test_baseline_leaderboard_resume_skips_existing_specs(tmp_path: Path) -> None:
    out_dir = tmp_path / "leaderboard"
    first = run_baseline_leaderboard(
        out_dir=out_dir,
        suites=["official_smoke_generated"],
        methods=["baseline_goal", "velocity_obstacle"],
        n_agents=[4],
        seeds=[0],
        comm_profiles=["ideal_50hz"],
        max_runs=1,
    )
    assert first["suites"][0]["run_count"] == 1

    second = run_baseline_leaderboard(
        out_dir=out_dir,
        suites=["official_smoke_generated"],
        methods=["baseline_goal", "velocity_obstacle"],
        n_agents=[4],
        seeds=[0],
        comm_profiles=["ideal_50hz"],
        max_runs=2,
        resume=True,
    )

    suite = second["suites"][0]
    assert suite["run_count"] == 2
    assert suite["selected_completed_count"] == 2
    assert suite["skipped_existing_count"] == 1
    assert suite["new_run_count"] == 1
    assert suite["selected_complete"] is True
    progress = json.loads((out_dir / suite["progress_path"]).read_text(encoding="utf-8"))
    assert progress["resume"] is True
    assert progress["skipped_existing_count"] == 1


def test_baseline_leaderboard_rejects_existing_results_without_resume(tmp_path: Path) -> None:
    out_dir = tmp_path / "leaderboard"
    run_baseline_leaderboard(
        out_dir=out_dir,
        suites=["official_smoke_generated"],
        methods=["baseline_goal"],
        n_agents=[4],
        seeds=[0],
        comm_profiles=["ideal_50hz"],
        max_runs=1,
    )

    with pytest.raises(RuntimeError, match="Use --resume"):
        run_baseline_leaderboard(
            out_dir=out_dir,
            suites=["official_smoke_generated"],
            methods=["baseline_goal"],
            n_agents=[4],
            seeds=[0],
            comm_profiles=["ideal_50hz"],
            max_runs=1,
        )


def test_baseline_leaderboard_wall_time_budget_writes_partial_progress(tmp_path: Path) -> None:
    out_dir = tmp_path / "leaderboard"
    report = run_baseline_leaderboard(
        out_dir=out_dir,
        suites=["official_smoke_generated"],
        methods=["baseline_goal"],
        n_agents=[4],
        seeds=[0],
        comm_profiles=["ideal_50hz"],
        max_runs=1,
        max_wall_time_s=0.0,
    )

    suite = report["suites"][0]
    assert report["ok"] is False
    assert report["complete"] is False
    assert report["stopped_by_wall_time"] is True
    assert suite["run_count"] == 0
    assert suite["selected_complete"] is False
    assert suite["stopped_by_wall_time"] is True
    assert (out_dir / suite["progress_path"]).exists()
    assert (out_dir / suite["summary_csv"]).exists()


def test_baseline_leaderboard_episode_timeout_writes_failed_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    if not baseline_leaderboard._hard_timeout_supported():
        pytest.skip("hard episode timeout requires SIGALRM in the main thread")

    def _slow_run_episode(spec):
        _ = spec
        time.sleep(2.0)
        raise AssertionError("timeout did not interrupt slow run")

    monkeypatch.setattr(baseline_leaderboard, "run_episode", _slow_run_episode)

    out_dir = tmp_path / "leaderboard"
    report = run_baseline_leaderboard(
        out_dir=out_dir,
        suites=["official_smoke_generated"],
        methods=["baseline_goal"],
        n_agents=[4],
        seeds=[0],
        comm_profiles=["ideal_50hz"],
        max_runs=1,
        run_timeout_s=0.01,
    )

    suite = report["suites"][0]
    assert report["ok"] is False
    assert report["timeout_run_count"] == 1
    assert report["complete"] is False
    assert suite["run_count"] == 1
    assert suite["timeout_run_count"] == 1
    assert suite["selected_complete"] is True

    with (out_dir / suite["results_csv"]).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["method"] == "baseline_goal"
    assert int(float(rows[0]["planner_timeout_count"])) == 1
    assert int(float(rows[0]["planner_error_count"])) == 1


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
    assert report["suites"][0]["selected_completed_count"] == 2
    assert (out_dir / "baseline_leaderboard.json").exists()
