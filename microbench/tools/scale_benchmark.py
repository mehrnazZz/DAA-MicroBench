from __future__ import annotations

import csv
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import yaml

from microbench.config import load_yaml, resolve_config_path
from microbench.metrics import append_result, write_summary
from microbench.tools.baseline_leaderboard import (
    _existing_keys,
    _hard_timeout_supported,
    _read_result_rows,
    _run_episode_checked,
    _select_specs,
    _spec_key,
    _write_empty_summary,
    MAX_RUNS_STRATEGIES,
)
from microbench.types import RunSpec


SCALE_BENCHMARK_SCHEMA_VERSION = "0.1"
DEFAULT_SCALE_BENCHMARK_METHODS = (
    "mpc_nonlinear",
    "dmpc_best_response",
    "bvc_tube_dmpc",
    "dynamic_tube_dmpc",
    "ego_swarm_opt",
    "rmader",
)
SCALE_SPAWN_PROFILES = ("none", "dense")
GUARDRAIL_FIELDS = ("planner_timeout_count", "planner_error_count", "planner_fallback_count")
PLANNER_PRESET_ENV = "DAA_MICROBENCH_PLANNER_PRESET"


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _rel(path: str | Path, root: Path) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)


def _as_list(values: tuple[str, ...] | list[str] | None, default: tuple[str, ...]) -> list[str]:
    return [str(v).strip() for v in (values if values is not None else default) if str(v).strip()]


@contextmanager
def _planner_preset_env(preset: str):
    previous = os.environ.get(PLANNER_PRESET_ENV)
    clean = str(preset or "default").strip() or "default"
    if clean == "default":
        os.environ.pop(PLANNER_PRESET_ENV, None)
    else:
        os.environ[PLANNER_PRESET_ENV] = clean
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(PLANNER_PRESET_ENV, None)
        else:
            os.environ[PLANNER_PRESET_ENV] = previous


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _max(values: list[float]) -> float | None:
    return max(values) if values else None


def _min(values: list[float]) -> float | None:
    return min(values) if values else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _guardrail_count(row: dict[str, Any]) -> int:
    return sum(int(_num(row.get(field)) or 0) for field in GUARDRAIL_FIELDS)


def _is_hard_timeout_row(row: dict[str, Any]) -> bool:
    duration = _num(row.get("duration_s"))
    return (
        int(_num(row.get("planner_timeout_count")) or 0) > 0
        and int(_num(row.get("planner_error_count")) or 0) > 0
        and duration is None
    )


def _is_completed_episode_row(row: dict[str, Any]) -> bool:
    return not _is_hard_timeout_row(row) and _num(row.get("duration_s")) is not None


def _scenario_dimension(cfg: dict[str, Any]) -> str:
    benchmark = cfg.get("benchmark", {}) if isinstance(cfg, dict) else {}
    if isinstance(benchmark, dict) and benchmark.get("dimension"):
        return str(benchmark["dimension"])
    world = cfg.get("world", {}) if isinstance(cfg, dict) else {}
    if isinstance(world, dict):
        return "2d" if bool(world.get("planar", True)) else "3d"
    return "unknown"


def _apply_dense_spawn_profile(cfg: dict[str, Any], *, max_n_agents: int) -> dict[str, Any]:
    out = dict(cfg)
    agent_params = out.get("agent_params", {}) if isinstance(out.get("agent_params", {}), dict) else {}
    radius = float(agent_params.get("radius_m", 0.6))
    spawn = out.setdefault("spawn", {})
    if not isinstance(spawn, dict):
        return out

    spawn["scale_profile"] = "dense"
    spawn["scale_profile_max_n_agents"] = int(max_n_agents)
    min_sep = max(2.0, 2.0 * radius + 0.2)
    current_sep = _num(spawn.get("min_start_separation_m"))
    if current_sep is None or current_sep > min_sep:
        spawn["min_start_separation_m"] = float(min_sep)

    if str(spawn.get("type", "")).lower() == "four_way":
        current_width = _num(spawn.get("lane_half_width_m")) or 0.0
        spawn["lane_half_width_m"] = float(max(current_width, 0.4 * float(max_n_agents)))
        goals = out.setdefault("goals", {})
        if isinstance(goals, dict):
            goals["max_attempts"] = int(max(int(goals.get("max_attempts", 0) or 0), 20000))

    return out


def _prepare_scenario(
    *,
    source: str | Path,
    out_dir: Path,
    duration_s: float | None,
    save_traces: bool,
    scale_spawn_profile: str,
    max_n_agents: int,
    index: int,
) -> dict[str, Any]:
    source_path = Path(resolve_config_path(str(source)))
    cfg = load_yaml(source_path)
    if duration_s is not None:
        cfg.setdefault("scenario", {})["duration_s"] = float(duration_s)
    if scale_spawn_profile == "dense":
        cfg = _apply_dense_spawn_profile(cfg, max_n_agents=max_n_agents)
    logging_cfg = cfg.setdefault("logging", {})
    logging_cfg["save_trace"] = bool(save_traces)
    if save_traces:
        logging_cfg["trace_save_failures_only"] = False
        logging_cfg["trace_agents_mode"] = "all"
    else:
        logging_cfg.setdefault("save_events", True)

    prepared_dir = out_dir / "_scale_scenarios"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    profile_suffix = "" if scale_spawn_profile == "none" else f"_{scale_spawn_profile}"
    dest = prepared_dir / f"{index:02d}_{source_path.stem}_scale{profile_suffix}.yaml"
    dest.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    scenario_cfg = cfg.get("scenario", {}) if isinstance(cfg.get("scenario", {}), dict) else {}
    return {
        "source_path": str(source_path),
        "prepared_path": str(dest),
        "source_stem": source_path.stem,
        "prepared_stem": dest.stem,
        "name": str(scenario_cfg.get("name") or source_path.stem),
        "duration_s": _num(scenario_cfg.get("duration_s")),
        "dimension": _scenario_dimension(cfg),
        "scale_spawn_profile": scale_spawn_profile,
    }


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _group_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("scenario", "")),
        str(row.get("method", "")),
        str(row.get("comm_profile", "")),
        str(row.get("N", "")),
    )


def _scale_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_group_key(row), []).append(row)

    summaries: list[dict[str, Any]] = []
    for (scenario, method, comm, n_agents), items in sorted(groups.items()):
        timeout_rows = [row for row in items if _is_hard_timeout_row(row)]
        completed_rows = [row for row in items if _is_completed_episode_row(row)]
        guardrails = [_guardrail_count(row) for row in items]
        collision_episodes = [
            _num(row.get("collision_episode"))
            for row in completed_rows
            if _num(row.get("collision_episode")) is not None
        ]
        collision_ticks = [
            _num(row.get("collision_pair_ticks"))
            for row in completed_rows
            if _num(row.get("collision_pair_ticks")) is not None
        ]
        unique_pairs = [
            _num(row.get("unique_collision_pairs"))
            for row in completed_rows
            if _num(row.get("unique_collision_pairs")) is not None
        ]
        completion_rates = [
            _num(row.get("completion_rate"))
            for row in completed_rows
            if _num(row.get("completion_rate")) is not None
        ]
        final_goal_dist_mean = [
            _num(row.get("final_goal_dist_mean_m"))
            for row in completed_rows
            if _num(row.get("final_goal_dist_mean_m")) is not None
        ]
        final_goal_dist_p95 = [
            _num(row.get("final_goal_dist_p95_m"))
            for row in completed_rows
            if _num(row.get("final_goal_dist_p95_m")) is not None
        ]
        goal_progress_mean = [
            _num(row.get("goal_progress_mean_m"))
            for row in completed_rows
            if _num(row.get("goal_progress_mean_m")) is not None
        ]
        goal_progress_fraction_mean = [
            _num(row.get("goal_progress_fraction_mean"))
            for row in completed_rows
            if _num(row.get("goal_progress_fraction_mean")) is not None
        ]
        goal_progress_fraction_p05 = [
            _num(row.get("goal_progress_fraction_p05"))
            for row in completed_rows
            if _num(row.get("goal_progress_fraction_p05")) is not None
        ]
        min_seps = [
            _num(row.get("min_sep_min_m"))
            for row in completed_rows
            if _num(row.get("min_sep_min_m")) is not None
        ]
        planner_p95 = [
            _num(row.get("planner_ms_per_tick_per_agent_p95"))
            for row in completed_rows
            if _num(row.get("planner_ms_per_tick_per_agent_p95")) is not None
        ]
        runtimes = [_num(row.get("episode_runtime_s")) for row in items if _num(row.get("episode_runtime_s")) is not None]
        status = "completed"
        if len(timeout_rows) == len(items):
            status = "timeout"
        elif timeout_rows:
            status = "partial_timeout"

        summaries.append(
            {
                "scenario": scenario,
                "method": method,
                "comm_profile": comm,
                "N": int(float(n_agents)),
                "status": status,
                "run_count": len(items),
                "completed_run_count": len(completed_rows),
                "timeout_run_count": len(timeout_rows),
                "timeout_rate": round(len(timeout_rows) / len(items), 6) if items else None,
                "guardrail_count_mean": _mean([float(v) for v in guardrails]),
                "collision_episode_rate": _mean(collision_episodes),
                "collision_pair_ticks_mean": _mean(collision_ticks),
                "unique_collision_pairs_mean": _mean(unique_pairs),
                "completion_rate_mean": _mean(completion_rates),
                "final_goal_dist_mean_m_mean": _mean(final_goal_dist_mean),
                "final_goal_dist_p95_m_mean": _mean(final_goal_dist_p95),
                "goal_progress_mean_m_mean": _mean(goal_progress_mean),
                "goal_progress_fraction_mean": _mean(goal_progress_fraction_mean),
                "goal_progress_fraction_p05_mean": _mean(goal_progress_fraction_p05),
                "min_sep_min_worst_m": _min(min_seps),
                "planner_ms_p95_max": _max(planner_p95),
                "episode_runtime_s_mean": _mean([float(v) for v in runtimes]),
            }
        )
    return summaries


def _write_scale_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _method_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)

    out: list[dict[str, Any]] = []
    for method, items in sorted(by_method.items()):
        completed = [row for row in items if row["status"] != "timeout" and int(row["completed_run_count"]) > 0]
        clean = [
            row
            for row in completed
            if (_num(row.get("collision_episode_rate")) or 0.0) == 0.0
            and (_num(row.get("timeout_rate")) or 0.0) == 0.0
            and (_num(row.get("guardrail_count_mean")) or 0.0) == 0.0
            and (_num(row.get("completion_rate_mean")) or 0.0) >= 1.0
        ]
        timeout_rates = [_num(row.get("timeout_rate")) for row in items if _num(row.get("timeout_rate")) is not None]
        collision_rates = [
            _num(row.get("collision_episode_rate"))
            for row in items
            if _num(row.get("collision_episode_rate")) is not None
        ]
        completion_rates = [
            _num(row.get("completion_rate_mean"))
            for row in items
            if _num(row.get("completion_rate_mean")) is not None
        ]
        goal_progress_fractions = [
            _num(row.get("goal_progress_fraction_mean"))
            for row in items
            if _num(row.get("goal_progress_fraction_mean")) is not None
        ]
        planner_p95 = [_num(row.get("planner_ms_p95_max")) for row in items if _num(row.get("planner_ms_p95_max")) is not None]
        out.append(
            {
                "method": method,
                "scale_cells": len(items),
                "timeout_rate_mean": _mean([float(v) for v in timeout_rates]),
                "collision_episode_rate_mean": _mean([float(v) for v in collision_rates]),
                "completion_rate_mean": _mean([float(v) for v in completion_rates]),
                "goal_progress_fraction_mean": _mean([float(v) for v in goal_progress_fractions]),
                "max_completed_N": max([int(row["N"]) for row in completed], default=None),
                "max_clean_N": max([int(row["N"]) for row in clean], default=None),
                "planner_ms_p95_max": _max([float(v) for v in planner_p95]),
            }
        )
    out.sort(
        key=lambda row: (
            float("inf") if row["timeout_rate_mean"] is None else float(row["timeout_rate_mean"]),
            -1 if row["max_clean_N"] is None else -int(row["max_clean_N"]),
            float("inf") if row["collision_episode_rate_mean"] is None else float(row["collision_episode_rate_mean"]),
            float("inf") if row["planner_ms_p95_max"] is None else float(row["planner_ms_p95_max"]),
            str(row["method"]),
        )
    )
    for rank, row in enumerate(out, start=1):
        row["rank"] = rank
    return out


def run_scale_benchmark(
    *,
    out_dir: str | Path,
    scenarios: tuple[str, ...] | list[str],
    methods: tuple[str, ...] | list[str] | None = None,
    n_agents: tuple[int, ...] | list[int] | None = None,
    seeds: tuple[int, ...] | list[int] | None = None,
    comm_profiles: tuple[str, ...] | list[str] | None = None,
    max_runs: int | None = None,
    max_runs_strategy: str = "balanced",
    resume: bool = False,
    max_wall_time_s: float | None = None,
    run_timeout_s: float | None = None,
    duration_s: float | None = None,
    save_traces: bool = False,
    scale_spawn_profile: str = "none",
    planner_preset: str = "default",
    plan_only: bool = False,
) -> dict[str, Any]:
    if not scenarios:
        raise ValueError("scale benchmark requires at least one scenario")
    if max_runs_strategy not in MAX_RUNS_STRATEGIES:
        raise ValueError(f"unknown max-runs strategy: {max_runs_strategy!r}")
    if scale_spawn_profile not in SCALE_SPAWN_PROFILES:
        raise ValueError(f"unknown scale spawn profile: {scale_spawn_profile!r}")
    clean_planner_preset = str(planner_preset or "default").strip() or "default"

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    method_values = _as_list(methods, DEFAULT_SCALE_BENCHMARK_METHODS)
    n_values = [int(v) for v in (n_agents if n_agents is not None else [4, 8, 16, 30])]
    seed_values = [int(v) for v in (seeds if seeds is not None else [0])]
    comm_values = [str(v) for v in (comm_profiles if comm_profiles is not None else ["realistic_v2v_50hz"])]
    prepared = [
        _prepare_scenario(
            source=scenario,
            out_dir=out,
            duration_s=duration_s,
            save_traces=save_traces,
            scale_spawn_profile=scale_spawn_profile,
            max_n_agents=max(n_values),
            index=i,
        )
        for i, scenario in enumerate(scenarios)
    ]

    run_dir = out / "runs"
    results_csv = run_dir / "results.csv"
    summary_csv = run_dir / "summary.csv"
    progress_path = out / "scale_benchmark_progress.json"
    scale_summary_csv = out / "scale_summary.csv"
    report_path = out / "scale_benchmark.json"

    specs: list[RunSpec] = []
    for scenario in prepared:
        for method in method_values:
            for comm in comm_values:
                for n in n_values:
                    for seed in seed_values:
                        specs.append(
                            RunSpec(
                                scenario_path=str(scenario["prepared_path"]),
                                method=str(method),
                                n_agents=int(n),
                                seed=int(seed),
                                comm_profile=str(comm),
                                out_dir=str(run_dir),
                                save_trace=bool(save_traces),
                            )
                        )

    planned_run_count = len(specs)
    selected_specs = _select_specs(specs, max_runs=max_runs, strategy=max_runs_strategy)
    selected_run_count = len(selected_specs)

    if results_csv.exists() and not resume and not plan_only:
        raise RuntimeError(
            f"{results_csv} already exists. Use --resume to continue this scale benchmark "
            "or choose a fresh --out-dir."
        )

    started = time.perf_counter()
    deadline_at = started + float(max_wall_time_s) if max_wall_time_s is not None else None
    completed_keys = _existing_keys(results_csv) if resume else set()
    skipped_existing = 0
    newly_run = 0
    newly_timed_out = 0
    stopped_by_wall_time = False

    if not plan_only:
        with _planner_preset_env(clean_planner_preset):
            for spec in selected_specs:
                if _spec_key(spec) in completed_keys:
                    skipped_existing += 1
                    continue
                if deadline_at is not None and time.perf_counter() >= deadline_at:
                    stopped_by_wall_time = True
                    break
                row, timed_out = _run_episode_checked(spec, run_timeout_s=run_timeout_s)
                append_result(run_dir, row)
                completed_keys.add(_spec_key(spec))
                newly_run += 1
                newly_timed_out += int(timed_out)
        if results_csv.exists():
            write_summary(run_dir)
        else:
            _write_empty_summary(run_dir)
    else:
        _write_empty_summary(run_dir)

    rows = _read_result_rows(results_csv)
    existing_after = {_spec_key(spec) for spec in selected_specs if _spec_key(spec) in _existing_keys(results_csv)}
    selected_completed_count = len(existing_after)
    timeout_run_count = sum(1 for row in rows if _is_hard_timeout_row(row))
    summary_rows = _scale_summary_rows(rows)
    method_summaries = _method_summary_rows(summary_rows)
    _write_scale_summary_csv(scale_summary_csv, summary_rows)

    truncated_by_max_runs = max_runs is not None and selected_run_count < planned_run_count
    selected_complete = selected_completed_count == selected_run_count
    complete = selected_completed_count == planned_run_count and max_runs is None and not stopped_by_wall_time
    if timeout_run_count:
        complete = False

    progress = {
        "schema_version": SCALE_BENCHMARK_SCHEMA_VERSION,
        "updated_at": _now_utc(),
        "planned_run_count": planned_run_count,
        "selected_run_count": selected_run_count,
        "selected_completed_count": selected_completed_count,
        "new_run_count": newly_run,
        "skipped_existing_count": skipped_existing,
        "timeout_run_count": timeout_run_count,
        "new_timeout_run_count": newly_timed_out,
        "truncated_by_max_runs": truncated_by_max_runs,
        "max_runs_strategy": max_runs_strategy,
        "stopped_by_wall_time": stopped_by_wall_time,
        "selected_complete": selected_complete,
        "complete": complete,
        "resume": bool(resume),
        "plan_only": bool(plan_only),
        "max_wall_time_s": None if max_wall_time_s is None else float(max_wall_time_s),
        "run_timeout_s": None if run_timeout_s is None else float(run_timeout_s),
        "run_timeout_supported": _hard_timeout_supported(),
        "results_csv": _rel(results_csv, out),
        "summary_csv": _rel(summary_csv, out),
        "scale_summary_csv": _rel(scale_summary_csv, out),
        "planner_preset": clean_planner_preset,
    }
    progress_path.write_text(json.dumps(_json_safe(progress), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = {
        "schema_version": SCALE_BENCHMARK_SCHEMA_VERSION,
        "ok": selected_complete and timeout_run_count == 0 and not stopped_by_wall_time,
        "complete": complete,
        "selected_complete": selected_complete,
        "stopped_by_wall_time": stopped_by_wall_time,
        "timeout_run_count": timeout_run_count,
        "resume": bool(resume),
        "plan_only": bool(plan_only),
        "out_dir": str(out),
        "methods": method_values,
        "n_agents": n_values,
        "seeds": seed_values,
        "comm_profiles": comm_values,
        "scenario_count": len(prepared),
        "scenarios": prepared,
        "planned_run_count": planned_run_count,
        "selected_run_count": selected_run_count,
        "selected_completed_count": selected_completed_count,
        "new_run_count": newly_run,
        "skipped_existing_count": skipped_existing,
        "new_timeout_run_count": newly_timed_out,
        "truncated_by_max_runs": truncated_by_max_runs,
        "max_runs_strategy": max_runs_strategy,
        "max_wall_time_s": None if max_wall_time_s is None else float(max_wall_time_s),
        "run_timeout_s": None if run_timeout_s is None else float(run_timeout_s),
        "duration_s_override": None if duration_s is None else float(duration_s),
        "scale_spawn_profile": scale_spawn_profile,
        "planner_preset": clean_planner_preset,
        "save_traces": bool(save_traces),
        "wall_runtime_s": round(time.perf_counter() - started, 6),
        "run_timeout_supported": _hard_timeout_supported(),
        "results_csv": str(results_csv),
        "summary_csv": str(summary_csv),
        "scale_summary_csv": str(scale_summary_csv),
        "progress_path": str(progress_path),
        "method_summaries": method_summaries,
        "scale_rows": summary_rows,
        "score_note": (
            "Scale rows separate completion/runtime feasibility from safety. "
            "Treat timeout or partial-timeout rows as non-publication-grade evidence."
        ),
    }
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return _json_safe(report)


def write_scale_benchmark(report: dict[str, Any], out: str | Path) -> Path:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
