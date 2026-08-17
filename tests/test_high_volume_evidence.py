from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import microbench.tools.scale_benchmark as scale_benchmark
from microbench.tools.high_volume_evidence import run_high_volume_evidence


ROOT = Path(__file__).resolve().parents[1]


def _evidence_row(
    spec,
    *,
    progress: float,
    completion: float,
    planner_p95_ms: float,
    min_sep_m: float = 2.0,
) -> dict[str, Any]:
    final_goal_dist = 0.4 if completion >= 1.0 else 10.0 * (1.0 - completion)
    return {
        "run_id": Path(spec.out_dir).name,
        "method": spec.method,
        "scenario": Path(spec.scenario_path).stem,
        "comm_profile": spec.comm_profile,
        "N": spec.n_agents,
        "seed": spec.seed,
        "dt_s": 0.1,
        "duration_s": 2.0,
        "v_max_mps": 3.0,
        "a_max_mps2": 2.0,
        "range_m": 30.0,
        "top_k": 8,
        "collisions": 0,
        "collision_pair_ticks": 0,
        "unique_collision_pairs": 0,
        "collision_episode": 0,
        "near_miss_episode": 0,
        "min_sep_min_m": min_sep_m,
        "completion_rate": completion,
        "final_goal_dist_mean_m": final_goal_dist,
        "final_goal_dist_p95_m": final_goal_dist + 0.5,
        "goal_progress_mean_m": 20.0 * progress,
        "goal_progress_fraction_mean": progress,
        "goal_progress_fraction_p05": max(0.0, progress - 0.05),
        "planner_ms_per_tick_per_agent_p95": planner_p95_ms,
        "planner_timeout_count": 0,
        "planner_error_count": 0,
        "planner_fallback_count": 0,
        "episode_runtime_s": 0.8,
    }


def test_high_volume_evidence_packages_scale_and_leaderboard(tmp_path: Path, monkeypatch) -> None:
    def _fake_run(spec, *, run_timeout_s):
        _ = run_timeout_s
        if spec.method == "ego_swarm_opt":
            return _evidence_row(spec, progress=0.96, completion=1.0, planner_p95_ms=6.0), False
        return _evidence_row(spec, progress=0.55, completion=0.4, planner_p95_ms=7.0), False

    monkeypatch.setattr(scale_benchmark, "_run_episode_checked", _fake_run)

    report = run_high_volume_evidence(
        out_dir=tmp_path / "evidence",
        scenarios=["config/scenarios/urban_throughput_3d.yaml"],
        methods=["dynamic_tube_dmpc", "ego_swarm_opt"],
        n_agents=[4],
        seeds=[0],
        comm_profiles=["ideal_50hz"],
        duration_s=1.0,
        run_timeout_s=5.0,
    )

    assert report["ok"] is True
    assert report["complete"] is True
    assert report["planned_run_count"] == 2
    assert report["selected_completed_count"] == 2
    assert report["overall_ranking"][0]["method"] == "ego_swarm_opt"
    assert Path(report["report_path"]).exists()
    assert Path(report["scale_report_path"]).exists()
    assert Path(report["scale_summary_csv"]).exists()
    assert Path(report["high_volume_leaderboard_path"]).exists()
    assert Path(report["high_volume_leaderboard_csv"]).exists()


def test_high_volume_evidence_cli_plan_only_json(tmp_path: Path) -> None:
    out_dir = tmp_path / "cli_evidence"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "microbench.cli",
            "high-volume-evidence",
            "--out-dir",
            str(out_dir),
            "--scenarios",
            "config/scenarios/urban_throughput_3d.yaml",
            "--methods",
            "baseline_goal",
            "--n",
            "4",
            "--seeds",
            "0",
            "--comm",
            "ideal_50hz",
            "--duration-s",
            "0.2",
            "--plan-only",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    report = json.loads(proc.stdout)
    assert report["ok"] is True
    assert report["plan_only"] is True
    assert report["planned_run_count"] == 1
    assert report["high_volume_leaderboard_path"] is None
    assert Path(report["report_path"]).exists()
    assert Path(report["scale_report_path"]).exists()
