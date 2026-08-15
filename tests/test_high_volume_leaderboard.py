from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from microbench.tools.high_volume_leaderboard import build_high_volume_leaderboard, write_high_volume_leaderboard


ROOT = Path(__file__).resolve().parents[1]


FIELDS = [
    "scenario",
    "method",
    "comm_profile",
    "N",
    "status",
    "run_count",
    "completed_run_count",
    "timeout_run_count",
    "timeout_rate",
    "guardrail_count_mean",
    "collision_episode_rate",
    "collision_pair_ticks_mean",
    "unique_collision_pairs_mean",
    "completion_rate_mean",
    "final_goal_dist_mean_m_mean",
    "final_goal_dist_p95_m_mean",
    "goal_progress_mean_m_mean",
    "goal_progress_fraction_mean",
    "goal_progress_fraction_p05_mean",
    "min_sep_min_worst_m",
    "planner_ms_p95_max",
    "episode_runtime_s_mean",
]


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            base = {
                "comm_profile": "realistic_v2v_50hz",
                "N": 30,
                "status": "completed",
                "run_count": 1,
                "completed_run_count": 1,
                "timeout_run_count": 0,
                "timeout_rate": 0.0,
                "guardrail_count_mean": 0.0,
                "collision_episode_rate": 0.0,
                "collision_pair_ticks_mean": 0.0,
                "unique_collision_pairs_mean": 0.0,
                "completion_rate_mean": 0.0,
                "final_goal_dist_mean_m_mean": 50.0,
                "final_goal_dist_p95_m_mean": 60.0,
                "goal_progress_mean_m_mean": "",
                "goal_progress_fraction_mean": "",
                "goal_progress_fraction_p05_mean": "",
                "min_sep_min_worst_m": 0.8,
                "planner_ms_p95_max": 5.0,
                "episode_runtime_s_mean": 10.0,
            }
            base.update(row)
            writer.writerow({field: base.get(field, "") for field in FIELDS})
    return path


def test_high_volume_leaderboard_scores_axes_and_deduplicates_progress_rows(tmp_path: Path) -> None:
    old = _write_summary(
        tmp_path / "old" / "scale_summary.csv",
        [
            {
                "scenario": "00_urban_conflict_3d_scale_dense",
                "method": "ego_swarm_opt",
                "completion_rate_mean": 0.0,
                "planner_ms_p95_max": 15.0,
            },
            {
                "scenario": "00_urban_conflict_3d_scale_dense",
                "method": "dynamic_tube_dmpc",
                "completion_rate_mean": 0.0,
                "planner_ms_p95_max": 1.0,
            },
        ],
    )
    current = _write_summary(
        tmp_path / "current" / "scale_summary.csv",
        [
            {
                "scenario": "00_stacked_swap_3d_scale_dense",
                "method": "ego_swarm_opt",
                "completion_rate_mean": 0.4,
                "planner_ms_p95_max": 14.0,
            },
            {
                "scenario": "00_stacked_swap_3d_scale_dense",
                "method": "dynamic_tube_dmpc",
                "completion_rate_mean": 0.0,
                "planner_ms_p95_max": 1.0,
            },
            {
                "scenario": "00_stacked_swap_3d_scale_dense",
                "method": "unsafe_fast",
                "collision_episode_rate": 1.0,
                "unique_collision_pairs_mean": 2.0,
                "planner_ms_p95_max": 0.5,
            },
            {
                "scenario": "00_urban_conflict_3d_scale_dense",
                "method": "ego_swarm_opt",
                "goal_progress_fraction_mean": 0.25,
                "goal_progress_fraction_p05_mean": 0.02,
                "planner_ms_p95_max": 15.0,
            },
            {
                "scenario": "00_urban_conflict_3d_scale_dense",
                "method": "dynamic_tube_dmpc",
                "goal_progress_fraction_mean": 0.03,
                "goal_progress_fraction_p05_mean": 0.01,
                "planner_ms_p95_max": 1.2,
            },
            {
                "scenario": "00_urban_conflict_3d_scale_dense",
                "method": "unsafe_fast",
                "collision_episode_rate": 1.0,
                "goal_progress_fraction_mean": 0.25,
                "planner_ms_p95_max": 0.5,
            },
            {
                "scenario": "00_urban_throughput_3d_scale_dense",
                "method": "ego_swarm_opt",
                "completion_rate_mean": 0.066667,
                "goal_progress_fraction_mean": 0.17,
                "goal_progress_fraction_p05_mean": 0.005,
                "planner_ms_p95_max": 16.0,
            },
            {
                "scenario": "00_urban_throughput_3d_scale_dense",
                "method": "dynamic_tube_dmpc",
                "goal_progress_fraction_mean": 0.04,
                "goal_progress_fraction_p05_mean": 0.01,
                "planner_ms_p95_max": 2.0,
            },
            {
                "scenario": "00_urban_throughput_3d_scale_dense",
                "method": "unsafe_fast",
                "collision_episode_rate": 1.0,
                "goal_progress_fraction_mean": 0.17,
                "planner_ms_p95_max": 0.5,
            },
        ],
    )

    report = build_high_volume_leaderboard(scale_summaries=[old, current])

    assert report["duplicate_row_count"] == 2
    assert report["overall_ranking"][0]["method"] == "ego_swarm_opt"
    assert report["axis_rankings"]["progress"][0]["method"] == "ego_swarm_opt"
    assert report["axis_rankings"]["runtime"][0]["method"] == "unsafe_fast"
    unsafe = next(row for row in report["method_summaries"] if row["method"] == "unsafe_fast")
    assert unsafe["safety_score"] < 100.0
    dynamic = next(row for row in report["method_summaries"] if row["method"] == "dynamic_tube_dmpc")
    assert dynamic["classification"] == "runtime_leader"


def test_high_volume_leaderboard_write_and_cli_json(tmp_path: Path) -> None:
    summary = _write_summary(
        tmp_path / "scale_summary.csv",
        [
            {
                "scenario": "00_urban_throughput_3d_scale_dense",
                "method": "ego_swarm_opt",
                "goal_progress_fraction_mean": 0.2,
                "planner_ms_p95_max": 15.0,
            },
            {
                "scenario": "00_urban_throughput_3d_scale_dense",
                "method": "dynamic_tube_dmpc",
                "goal_progress_fraction_mean": 0.05,
                "planner_ms_p95_max": 1.0,
            },
        ],
    )
    out = tmp_path / "leaderboard.json"
    report = write_high_volume_leaderboard(scale_summaries=[summary], out=out)

    assert out.exists()
    assert Path(report["leaderboard_csv"]).exists()

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "microbench.cli",
            "high-volume-leaderboard",
            "--scale-summary",
            str(summary),
            "--out",
            str(tmp_path / "cli_leaderboard.json"),
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    cli_report = json.loads(proc.stdout)
    assert cli_report["schema_version"] == "0.1"
    assert cli_report["overall_ranking"][0]["method"] == "ego_swarm_opt"
