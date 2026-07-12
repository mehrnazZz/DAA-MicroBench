from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import yaml

from microbench.config import load_yaml, resolve_config_path
from microbench.metrics import append_result, write_summary
from microbench.metrics.io import RESULT_SCHEMA_FILENAME
from microbench.runner import run_episode
from microbench.tools.baseline_report import build_baseline_report
from microbench.types import RunSpec


ADVANCED_BASELINE_COMPARISON_SCHEMA_VERSION = "0.1"
DEFAULT_ADVANCED_COMPARISON_SCENARIO = "config/scenarios/urban_conflict_3d.yaml"
DEFAULT_ADVANCED_COMPARISON_METHODS = (
    "orca_heuristic",
    "orca_with_staleness",
    "cbf_qp",
    "mpc_local",
    "mpc_nonlinear",
    "dmpc_best_response",
    "bvc_tube_dmpc",
    "dynamic_tube_dmpc",
    "rmader",
    "ego_swarm",
    "ego_swarm_opt",
    "velocity_obstacle",
    "reciprocal_velocity_obstacle",
)
DEFAULT_ADVANCED_COMPARISON_N_AGENTS = 4
DEFAULT_ADVANCED_COMPARISON_SEED = 2
DEFAULT_ADVANCED_COMPARISON_COMM_PROFILE = "realistic_v2v_50hz"
DEFAULT_ADVANCED_COMPARISON_DURATION_S = 18.0


def _as_list(values: tuple[str, ...] | list[str] | None, default: tuple[str, ...]) -> list[str]:
    return [str(v).strip() for v in (values if values is not None else default) if str(v).strip()]


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _guardrail_total(row: dict[str, Any]) -> int:
    return sum(
        int(_to_float(row.get(field)) or 0)
        for field in ("planner_timeout_count", "planner_error_count", "planner_fallback_count")
    )


def _critical_metrics_finite(row: dict[str, Any]) -> bool:
    fields = (
        "collision_episode",
        "min_sep_min_m",
        "completion_rate",
        "planner_ms_per_tick_per_agent_p95",
        "episode_runtime_s",
    )
    return all(_to_float(row.get(field)) is not None for field in fields)


def _prepare_scenario(
    *,
    scenario: str | Path,
    out_dir: Path,
    duration_s: float | None,
) -> tuple[Path, Path, float | None]:
    source = resolve_config_path(str(scenario))
    cfg = load_yaml(source)
    if duration_s is not None:
        cfg.setdefault("scenario", {})["duration_s"] = float(duration_s)
    effective_duration = _to_float(cfg.get("scenario", {}).get("duration_s"))
    scenario_dir = out_dir / "_comparison_scenario"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    dest = scenario_dir / Path(source).name
    dest.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return Path(source), dest, effective_duration


def _rank_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in report.get("rows", []) if isinstance(row, dict)]
    rows.sort(
        key=lambda row: (
            float("inf") if row.get("score_v0") is None else float(row["score_v0"]),
            str(row.get("method", "")),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def _project_episode_row(row: dict[str, Any]) -> dict[str, Any]:
    return _json_safe({
        "method": row.get("method"),
        "scenario": row.get("scenario"),
        "comm_profile": row.get("comm_profile"),
        "N": row.get("N"),
        "seed": row.get("seed"),
        "collision_episode": row.get("collision_episode"),
        "collisions": row.get("collisions"),
        "unique_collision_pairs": row.get("unique_collision_pairs"),
        "min_sep_min_m": row.get("min_sep_min_m"),
        "min_sep_p05_m": row.get("min_sep_p05_m"),
        "completion_rate": row.get("completion_rate"),
        "mean_time_to_goal_s": row.get("mean_time_to_goal_s"),
        "deadlock_time_pct": row.get("deadlock_time_pct"),
        "planner_ms_per_tick_per_agent_p95": row.get("planner_ms_per_tick_per_agent_p95"),
        "obs_v2v_fraction": row.get("obs_v2v_fraction"),
        "obs_sensor_fraction": row.get("obs_sensor_fraction"),
        "obs_stale_fraction": row.get("obs_stale_fraction"),
        "planner_timeout_count": row.get("planner_timeout_count"),
        "planner_error_count": row.get("planner_error_count"),
        "planner_fallback_count": row.get("planner_fallback_count"),
        "episode_runtime_s": row.get("episode_runtime_s"),
    })


def run_advanced_baseline_comparison(
    *,
    out_dir: str | Path,
    scenario: str | Path = DEFAULT_ADVANCED_COMPARISON_SCENARIO,
    methods: tuple[str, ...] | list[str] | None = None,
    n_agents: int = DEFAULT_ADVANCED_COMPARISON_N_AGENTS,
    seed: int = DEFAULT_ADVANCED_COMPARISON_SEED,
    comm_profile: str = DEFAULT_ADVANCED_COMPARISON_COMM_PROFILE,
    duration_s: float | None = DEFAULT_ADVANCED_COMPARISON_DURATION_S,
    save_traces: bool = False,
) -> dict[str, Any]:
    out = Path(out_dir)
    if (out / "results.csv").exists():
        raise RuntimeError(f"advanced baseline comparison output already exists: {out / 'results.csv'}")
    out.mkdir(parents=True, exist_ok=True)

    method_values = _as_list(methods, DEFAULT_ADVANCED_COMPARISON_METHODS)
    source_scenario, scenario_path, effective_duration_s = _prepare_scenario(
        scenario=scenario,
        out_dir=out,
        duration_s=duration_s,
    )

    rows: list[dict[str, Any]] = []
    for method in method_values:
        spec = RunSpec(
            scenario_path=str(scenario_path),
            method=str(method),
            n_agents=int(n_agents),
            seed=int(seed),
            comm_profile=str(comm_profile),
            out_dir=str(out),
            save_trace=bool(save_traces),
        )
        row = run_episode(spec)
        append_result(out, row)
        rows.append(row)

    summary_csv = write_summary(out)
    baseline_report = build_baseline_report(
        summary_csv=summary_csv,
        results_csv=out / "results.csv",
        suite="advanced_baseline_comparison",
        generated_by="python -m microbench.cli advanced-baseline-comparison",
    )
    baseline_report_path = out / "baseline_report.json"
    baseline_report_path.write_text(json.dumps(baseline_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    complete = len(rows) == len(method_values)
    guardrail_failures = {
        str(row.get("method")): _guardrail_total(row)
        for row in rows
        if _guardrail_total(row) > 0
    }
    nonfinite_methods = [
        str(row.get("method"))
        for row in rows
        if not _critical_metrics_finite(row)
    ]
    ranking = _rank_rows(baseline_report)
    report = _json_safe({
        "schema_version": ADVANCED_BASELINE_COMPARISON_SCHEMA_VERSION,
        "comparison_type": "advanced_baseline_3d_conflict",
        "ok": bool(complete and not guardrail_failures and not nonfinite_methods),
        "complete": bool(complete),
        "methods": method_values,
        "scenario_source": str(source_scenario),
        "scenario_path": str(scenario_path),
        "scenario": Path(scenario_path).stem,
        "duration_s": effective_duration_s,
        "n_agents": int(n_agents),
        "seed": int(seed),
        "comm_profile": str(comm_profile),
        "save_traces": bool(save_traces),
        "planned_run_count": len(method_values),
        "run_count": len(rows),
        "results_csv": str(out / "results.csv"),
        "summary_csv": str(summary_csv),
        "result_schema": str(out / RESULT_SCHEMA_FILENAME),
        "baseline_report_path": str(baseline_report_path),
        "ranking": ranking,
        "episode_rows": [_project_episode_row(row) for row in rows],
        "method_summaries": baseline_report.get("method_summaries", []),
        "checks": {
            "complete_matrix": bool(complete),
            "guardrails_clear": not guardrail_failures,
            "critical_metrics_finite": not nonfinite_methods,
        },
        "guardrail_failures": guardrail_failures,
        "nonfinite_methods": nonfinite_methods,
        "score_note": "score_v0 follows docs/LEADERBOARD.md; use component metrics, not only rank.",
    })
    report_path = out / "advanced_baseline_comparison.json"
    report["report_path"] = str(report_path)
    report_path.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def write_advanced_baseline_comparison(*, out_dir: str | Path, **kwargs: Any) -> Path:
    run_advanced_baseline_comparison(out_dir=out_dir, **kwargs)
    return Path(out_dir) / "advanced_baseline_comparison.json"
