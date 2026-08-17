from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import microbench.tools.high_volume_leaderboard as high_volume_leaderboard
import microbench.tools.scale_benchmark as scale_benchmark


HIGH_VOLUME_EVIDENCE_SCHEMA_VERSION = "0.1"
DEFAULT_HIGH_VOLUME_SCENARIOS = (
    "config/scenarios/stacked_swap_3d.yaml",
    "config/scenarios/urban_conflict_3d.yaml",
    "config/scenarios/urban_throughput_3d.yaml",
)
DEFAULT_HIGH_VOLUME_N_AGENTS = (30,)
DEFAULT_HIGH_VOLUME_SEEDS = (2,)
DEFAULT_HIGH_VOLUME_COMM_PROFILES = ("realistic_v2v_50hz",)
DEFAULT_HIGH_VOLUME_DURATION_S = 30.0
DEFAULT_HIGH_VOLUME_RUN_TIMEOUT_S = 360.0
DEFAULT_HIGH_VOLUME_SCALE_SPAWN_PROFILE = "dense"
DEFAULT_HIGH_VOLUME_PLANNER_PRESET = "scale"


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _as_str_list(values: tuple[str, ...] | list[str] | None, default: tuple[str, ...]) -> list[str]:
    return [str(v).strip() for v in (values if values is not None else default) if str(v).strip()]


def _as_int_list(values: tuple[int, ...] | list[int] | None, default: tuple[int, ...]) -> list[int]:
    return [int(v) for v in (values if values is not None else default)]


def _compact_scale_report(report: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "schema_version",
        "ok",
        "complete",
        "selected_complete",
        "stopped_by_wall_time",
        "timeout_run_count",
        "resume",
        "plan_only",
        "out_dir",
        "methods",
        "n_agents",
        "seeds",
        "comm_profiles",
        "scenario_count",
        "planned_run_count",
        "selected_run_count",
        "selected_completed_count",
        "new_run_count",
        "skipped_existing_count",
        "new_timeout_run_count",
        "truncated_by_max_runs",
        "max_runs_strategy",
        "max_wall_time_s",
        "run_timeout_s",
        "duration_s_override",
        "scale_spawn_profile",
        "planner_preset",
        "save_traces",
        "wall_runtime_s",
        "run_timeout_supported",
        "results_csv",
        "summary_csv",
        "scale_summary_csv",
        "progress_path",
        "report_path",
        "method_summaries",
    )
    return {key: report.get(key) for key in keys if key in report}


def run_high_volume_evidence(
    *,
    out_dir: str | Path,
    scenarios: tuple[str, ...] | list[str] | None = None,
    methods: tuple[str, ...] | list[str] | None = None,
    n_agents: tuple[int, ...] | list[int] | None = None,
    seeds: tuple[int, ...] | list[int] | None = None,
    comm_profiles: tuple[str, ...] | list[str] | None = None,
    max_runs: int | None = None,
    max_runs_strategy: str = "balanced",
    resume: bool = False,
    max_wall_time_s: float | None = None,
    run_timeout_s: float | None = DEFAULT_HIGH_VOLUME_RUN_TIMEOUT_S,
    duration_s: float | None = DEFAULT_HIGH_VOLUME_DURATION_S,
    save_traces: bool = False,
    scale_spawn_profile: str = DEFAULT_HIGH_VOLUME_SCALE_SPAWN_PROFILE,
    planner_preset: str = DEFAULT_HIGH_VOLUME_PLANNER_PRESET,
    latency_budget_ms: float = high_volume_leaderboard.DEFAULT_LATENCY_BUDGET_MS,
    plan_only: bool = False,
) -> dict[str, Any]:
    """Run the high-volume scale suite and package ranking evidence artifacts."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    scenario_values = _as_str_list(scenarios, DEFAULT_HIGH_VOLUME_SCENARIOS)
    method_values = _as_str_list(methods, scale_benchmark.DEFAULT_SCALE_BENCHMARK_METHODS)
    n_values = _as_int_list(n_agents, DEFAULT_HIGH_VOLUME_N_AGENTS)
    seed_values = _as_int_list(seeds, DEFAULT_HIGH_VOLUME_SEEDS)
    comm_values = _as_str_list(comm_profiles, DEFAULT_HIGH_VOLUME_COMM_PROFILES)

    scale_dir = out / "scale"
    scale_report = scale_benchmark.run_scale_benchmark(
        out_dir=scale_dir,
        scenarios=scenario_values,
        methods=method_values,
        n_agents=n_values,
        seeds=seed_values,
        comm_profiles=comm_values,
        max_runs=max_runs,
        max_runs_strategy=max_runs_strategy,
        resume=resume,
        max_wall_time_s=max_wall_time_s,
        run_timeout_s=run_timeout_s,
        duration_s=duration_s,
        save_traces=save_traces,
        scale_spawn_profile=scale_spawn_profile,
        planner_preset=planner_preset,
        plan_only=plan_only,
    )

    leaderboard_report: dict[str, Any] | None = None
    leaderboard_path = out / "high_volume_leaderboard.json"
    if not plan_only:
        scale_summary_csv = scale_report.get("scale_summary_csv")
        if scale_summary_csv:
            leaderboard_report = high_volume_leaderboard.write_high_volume_leaderboard(
                scale_summaries=[str(scale_summary_csv)],
                out=leaderboard_path,
                latency_budget_ms=float(latency_budget_ms),
                generated_by="python -m microbench.cli high-volume-evidence",
            )

    report_path = out / "high_volume_evidence.json"
    ok = bool(scale_report.get("ok")) if not plan_only else True
    if not plan_only:
        ok = ok and leaderboard_report is not None
    report = {
        "schema_version": HIGH_VOLUME_EVIDENCE_SCHEMA_VERSION,
        "generated_at": _now_utc(),
        "generated_by": "python -m microbench.cli high-volume-evidence",
        "ok": ok,
        "complete": bool(scale_report.get("complete")) if not plan_only else False,
        "selected_complete": bool(scale_report.get("selected_complete")),
        "plan_only": bool(plan_only),
        "out_dir": str(out),
        "scale_dir": str(scale_dir),
        "scale_report_path": scale_report.get("report_path"),
        "scale_progress_path": scale_report.get("progress_path"),
        "scale_summary_csv": scale_report.get("scale_summary_csv"),
        "results_csv": scale_report.get("results_csv"),
        "summary_csv": scale_report.get("summary_csv"),
        "high_volume_leaderboard_path": str(leaderboard_path) if leaderboard_report is not None else None,
        "high_volume_leaderboard_csv": (
            leaderboard_report.get("leaderboard_csv") if leaderboard_report is not None else None
        ),
        "overall_ranking": leaderboard_report.get("overall_ranking", []) if leaderboard_report is not None else [],
        "axis_rankings": leaderboard_report.get("axis_rankings", {}) if leaderboard_report is not None else {},
        "method_count": leaderboard_report.get("method_count", 0) if leaderboard_report is not None else 0,
        "scale_cell_count": leaderboard_report.get("scale_cell_count", 0) if leaderboard_report is not None else 0,
        "latency_budget_ms": float(latency_budget_ms),
        "scenarios": scenario_values,
        "methods": method_values,
        "n_agents": n_values,
        "seeds": seed_values,
        "comm_profiles": comm_values,
        "planned_run_count": scale_report.get("planned_run_count"),
        "selected_run_count": scale_report.get("selected_run_count"),
        "selected_completed_count": scale_report.get("selected_completed_count"),
        "new_run_count": scale_report.get("new_run_count"),
        "timeout_run_count": scale_report.get("timeout_run_count"),
        "truncated_by_max_runs": scale_report.get("truncated_by_max_runs"),
        "stopped_by_wall_time": scale_report.get("stopped_by_wall_time"),
        "resume": bool(resume),
        "max_runs_strategy": max_runs_strategy,
        "max_wall_time_s": None if max_wall_time_s is None else float(max_wall_time_s),
        "run_timeout_s": None if run_timeout_s is None else float(run_timeout_s),
        "duration_s_override": None if duration_s is None else float(duration_s),
        "scale_spawn_profile": scale_spawn_profile,
        "planner_preset": planner_preset,
        "save_traces": bool(save_traces),
        "scale_benchmark": _compact_scale_report(scale_report),
        "report_path": str(report_path),
        "note": (
            "This artifact ties the timeout-aware scale benchmark to the multi-axis high-volume "
            "leaderboard. Use --resume with the same output directory to continue partial evidence runs."
        ),
    }
    report_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return _json_safe(report)
