from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml

import microbench.tools.scale_benchmark as scale_benchmark
from microbench.tools.scale_benchmark import run_scale_benchmark


ROOT = Path(__file__).resolve().parents[1]


def _complete_row(spec, *, collision_episode: int = 0, completion_rate: float = 1.0) -> dict[str, Any]:
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
        "collisions": collision_episode,
        "collision_pair_ticks": 5 if collision_episode else 0,
        "unique_collision_pairs": 1 if collision_episode else 0,
        "collision_episode": collision_episode,
        "near_miss_episode": 0,
        "min_sep_min_m": -0.1 if collision_episode else 1.5,
        "completion_rate": completion_rate,
        "planner_ms_per_tick_per_agent_p95": 3.0 + spec.n_agents,
        "planner_timeout_count": 0,
        "planner_error_count": 0,
        "planner_fallback_count": 0,
        "episode_runtime_s": 0.5 + 0.1 * spec.n_agents,
    }


def _timeout_row(spec) -> dict[str, Any]:
    row = _complete_row(spec)
    row.update(
        {
            "dt_s": float("nan"),
            "duration_s": float("nan"),
            "v_max_mps": float("nan"),
            "a_max_mps2": float("nan"),
            "range_m": float("nan"),
            "top_k": float("nan"),
            "collisions": float("nan"),
            "collision_pair_ticks": float("nan"),
            "unique_collision_pairs": float("nan"),
            "collision_episode": float("nan"),
            "near_miss_episode": float("nan"),
            "min_sep_min_m": float("nan"),
            "completion_rate": float("nan"),
            "planner_ms_per_tick_per_agent_p95": 1000.0,
            "planner_timeout_count": 1,
            "planner_error_count": 1,
            "planner_fallback_count": 0,
            "episode_runtime_s": 1.0,
        }
    )
    return row


def test_scale_benchmark_plan_only_dense_spawn_prepares_scale_copy(tmp_path: Path) -> None:
    report = run_scale_benchmark(
        out_dir=tmp_path / "scale",
        scenarios=["config/scenarios/urban_conflict_3d.yaml"],
        methods=["baseline_goal"],
        n_agents=[30],
        seeds=[2],
        comm_profiles=["realistic_v2v_50hz"],
        scale_spawn_profile="dense",
        duration_s=60.0,
        plan_only=True,
    )

    assert report["plan_only"] is True
    assert report["planned_run_count"] == 1
    assert report["selected_run_count"] == 1
    prepared = Path(report["scenarios"][0]["prepared_path"])
    cfg = yaml.safe_load(prepared.read_text(encoding="utf-8"))
    assert cfg["scenario"]["duration_s"] == 60.0
    assert cfg["spawn"]["scale_profile"] == "dense"
    assert cfg["spawn"]["lane_half_width_m"] >= 12.0
    assert cfg["spawn"]["min_start_separation_m"] == 2.0
    assert (tmp_path / "scale" / "scale_benchmark.json").exists()


def test_scale_benchmark_aggregates_across_n_ladder(tmp_path: Path, monkeypatch) -> None:
    def _fake_run(spec, *, run_timeout_s):
        _ = run_timeout_s
        return _complete_row(spec, collision_episode=int(spec.n_agents >= 8)), False

    monkeypatch.setattr(scale_benchmark, "_run_episode_checked", _fake_run)

    report = run_scale_benchmark(
        out_dir=tmp_path / "scale",
        scenarios=["config/scenarios/stacked_swap_3d.yaml"],
        methods=["method_a"],
        n_agents=[4, 8],
        seeds=[0],
        comm_profiles=["ideal_50hz"],
        max_runs_strategy="balanced",
    )

    assert report["ok"] is True
    assert report["selected_completed_count"] == 2
    assert report["timeout_run_count"] == 0
    assert report["method_summaries"][0]["method"] == "method_a"
    assert report["method_summaries"][0]["max_completed_N"] == 8
    assert report["method_summaries"][0]["max_clean_N"] == 4
    scale_rows = {int(row["N"]): row for row in report["scale_rows"]}
    assert scale_rows[4]["collision_episode_rate"] == 0.0
    assert scale_rows[8]["collision_episode_rate"] == 1.0

    with Path(report["scale_summary_csv"]).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert {row["N"] for row in rows} == {"4", "8"}


def test_scale_benchmark_records_timeout_rows(tmp_path: Path, monkeypatch) -> None:
    def _fake_run(spec, *, run_timeout_s):
        _ = run_timeout_s
        return _timeout_row(spec), True

    monkeypatch.setattr(scale_benchmark, "_run_episode_checked", _fake_run)

    report = run_scale_benchmark(
        out_dir=tmp_path / "scale",
        scenarios=["config/scenarios/stacked_swap_3d.yaml"],
        methods=["slow_method"],
        n_agents=[30],
        seeds=[0],
        comm_profiles=["ideal_50hz"],
        run_timeout_s=1.0,
    )

    assert report["ok"] is False
    assert report["complete"] is False
    assert report["timeout_run_count"] == 1
    assert report["scale_rows"][0]["status"] == "timeout"
    assert report["scale_rows"][0]["timeout_rate"] == 1.0
    assert report["method_summaries"][0]["max_completed_N"] is None


def test_scale_benchmark_sets_planner_preset_only_during_runs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("DAA_MICROBENCH_PLANNER_PRESET", raising=False)
    seen_presets: list[str | None] = []

    def _fake_run(spec, *, run_timeout_s):
        _ = spec
        _ = run_timeout_s
        seen_presets.append(os.environ.get("DAA_MICROBENCH_PLANNER_PRESET"))
        return _complete_row(spec), False

    monkeypatch.setattr(scale_benchmark, "_run_episode_checked", _fake_run)

    report = run_scale_benchmark(
        out_dir=tmp_path / "scale",
        scenarios=["config/scenarios/stacked_swap_3d.yaml"],
        methods=["method_a"],
        n_agents=[30],
        seeds=[0],
        comm_profiles=["ideal_50hz"],
        planner_preset="scale",
    )

    assert seen_presets == ["scale"]
    assert os.environ.get("DAA_MICROBENCH_PLANNER_PRESET") is None
    assert report["planner_preset"] == "scale"


def test_scale_benchmark_cli_plan_only_json(tmp_path: Path) -> None:
    out_dir = tmp_path / "cli_scale"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "microbench.cli",
            "scale-benchmark",
            "--out-dir",
            str(out_dir),
            "--scenarios",
            "config/scenarios/urban_conflict_3d.yaml",
            "--methods",
            "baseline_goal",
            "--n",
            "30",
            "--seeds",
            "2",
            "--comm",
            "realistic_v2v_50hz",
            "--scale-spawn-profile",
            "dense",
            "--duration-s",
            "60",
            "--plan-only",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    report = json.loads(proc.stdout)
    assert report["plan_only"] is True
    assert report["planned_run_count"] == 1
    assert report["scale_spawn_profile"] == "dense"
    assert Path(report["scenarios"][0]["prepared_path"]).exists()
