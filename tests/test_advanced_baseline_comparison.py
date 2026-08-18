from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import microbench.tools.advanced_baseline_comparison as advanced_baseline_comparison
from microbench.tools.advanced_baseline_comparison import run_advanced_baseline_comparison


ROOT = Path(__file__).resolve().parents[1]


def test_advanced_baseline_comparison_runs_compact_shared_lane(tmp_path: Path) -> None:
    out_dir = tmp_path / "comparison"
    report = run_advanced_baseline_comparison(
        out_dir=out_dir,
        methods=["orca_heuristic"],
        duration_s=1.0,
        n_agents=4,
        seed=2,
        comm_profile="ideal_50hz",
    )

    assert report["schema_version"] == "0.1"
    assert report["comparison_type"] == "advanced_baseline_3d_conflict"
    assert report["ok"] is True
    assert report["methods"] == ["orca_heuristic"]
    assert report["planner_preset"] == "default"
    assert report["run_count"] == 1
    assert report["planned_run_count"] == 1
    json.dumps(report, allow_nan=False)
    assert len(report["ranking"]) == 1
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
    assert (out_dir / "comparison_manifest.json").exists()
    manifest = json.loads((out_dir / "comparison_manifest.json").read_text(encoding="utf-8"))
    assert manifest["planner_preset"] == "default"
    assert manifest["advanced_baseline_comparison_path"] == report["report_path"]
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
    assert report["foxglove_mcap"] is None


def test_advanced_baseline_comparison_can_export_foxglove_mcap(tmp_path: Path) -> None:
    if importlib.util.find_spec("mcap") is None:
        return
    out_dir = tmp_path / "comparison_mcap"
    report = run_advanced_baseline_comparison(
        out_dir=out_dir,
        scenario=ROOT / "config/scenarios/urban_throughput_3d.yaml",
        methods=["reciprocal_velocity_obstacle"],
        duration_s=0.2,
        n_agents=4,
        seed=2,
        comm_profile="ideal_50hz",
        export_foxglove_mcap=True,
        mcap_trail_frames=4,
        mcap_max_sensing_links=4,
    )

    assert report["ok"] is True
    assert report["comparison_type"] == "advanced_baseline_3d_urban_throughput"
    assert report["save_traces"] is True
    assert report["foxglove_mcap"]["labels"] == ["reciprocal_velocity_obstacle"]
    assert report["comparison_manifest_path"] == str(out_dir / "comparison_manifest.json")
    assert Path(report["foxglove_mcap"]["path"]).exists()
    assert Path(report["foxglove_mcap"]["trace_paths"]["reciprocal_velocity_obstacle"]).exists()


def _minimal_row(spec) -> dict[str, Any]:
    return {
        "run_id": Path(spec.out_dir).name,
        "method": spec.method,
        "scenario": Path(spec.scenario_path).stem,
        "comm_profile": spec.comm_profile,
        "N": spec.n_agents,
        "seed": spec.seed,
        "dt_s": 0.1,
        "duration_s": 0.1,
        "v_max_mps": 3.0,
        "a_max_mps2": 2.0,
        "range_m": 30.0,
        "top_k": 8,
        "collisions": 0,
        "near_misses": 0,
        "collision_pair_ticks": 0,
        "near_miss_pair_ticks": 0,
        "unique_collision_pairs": 0,
        "unique_near_miss_pairs": 0,
        "collision_episode": 0,
        "near_miss_episode": 0,
        "time_to_first_collision_s": float("nan"),
        "min_sep_min_m": 2.0,
        "min_sep_p05_m": 2.0,
        "completion_rate": 1.0,
        "final_goal_dist_mean_m": 0.5,
        "final_goal_dist_p95_m": 0.5,
        "goal_progress_mean_m": 10.0,
        "goal_progress_fraction_mean": 1.0,
        "goal_progress_fraction_p05": 1.0,
        "mean_time_to_goal_s": 0.1,
        "p95_time_to_goal_s": 0.1,
        "deadlock_time_pct": 0.0,
        "jerk_mean": 0.0,
        "planner_ms_per_tick_per_agent_mean": 0.1,
        "planner_ms_per_tick_per_agent_p95": 0.1,
        "obs_neighbors_mean": 0.0,
        "obs_v2v_fraction": 1.0,
        "obs_sensor_fraction": 0.0,
        "obs_stale_fraction": 0.0,
        "planner_timeout_count": 0,
        "planner_error_count": 0,
        "planner_fallback_count": 0,
        "episode_runtime_s": 0.01,
    }


def test_advanced_baseline_comparison_applies_and_restores_planner_preset(tmp_path: Path, monkeypatch) -> None:
    seen_presets: list[str | None] = []
    monkeypatch.setenv("DAA_MICROBENCH_PLANNER_PRESET", "preexisting")

    def _fake_run_episode(spec):
        seen_presets.append(os.environ.get("DAA_MICROBENCH_PLANNER_PRESET"))
        return _minimal_row(spec)

    monkeypatch.setattr(advanced_baseline_comparison, "run_episode", _fake_run_episode)

    out_dir = tmp_path / "comparison_preset"
    report = run_advanced_baseline_comparison(
        out_dir=out_dir,
        methods=["baseline_goal"],
        duration_s=0.1,
        n_agents=2,
        seed=0,
        comm_profile="ideal_50hz",
        planner_preset="scale",
    )

    assert seen_presets == ["scale"]
    assert os.environ.get("DAA_MICROBENCH_PLANNER_PRESET") == "preexisting"
    assert report["planner_preset"] == "scale"
    manifest = json.loads((out_dir / "comparison_manifest.json").read_text(encoding="utf-8"))
    assert manifest["planner_preset"] == "scale"
    assert manifest["ok"] is True


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
            "orca_heuristic",
            "--duration-s",
            "1.0",
            "--comm",
            "ideal_50hz",
            "--planner-preset",
            "scale",
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
    assert report["methods"] == ["orca_heuristic"]
    assert report["planner_preset"] == "scale"
    assert report["run_count"] == 1
    assert (out_dir / "advanced_baseline_comparison.json").exists()
    assert (out_dir / "comparison_manifest.json").exists()
