from __future__ import annotations

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any

import yaml

from microbench.config import load_yaml, resolve_config_path
from microbench.metrics import append_result, write_summary
from microbench.metrics.io import RESULT_SCHEMA_FILENAME
from microbench.metrics.recorder import episode_dir_name
from microbench.replay import export_foxglove_comparison_mcap
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
DEFAULT_ADVANCED_COMPARISON_MCAP = "baseline_comparison.mcap"
DEFAULT_ADVANCED_COMPARISON_PLANNER_PRESET = "default"
PLANNER_PRESET_ENV = "DAA_MICROBENCH_PLANNER_PRESET"


@contextmanager
def _planner_preset_env(preset: str):
    previous = os.environ.get(PLANNER_PRESET_ENV)
    clean = _clean_planner_preset(preset)
    if clean == DEFAULT_ADVANCED_COMPARISON_PLANNER_PRESET:
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


def _git_commit() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            capture_output=True,
            check=True,
        )
    except Exception:
        return None
    commit = proc.stdout.strip()
    return commit or None


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


def _clean_planner_preset(preset: str | None) -> str:
    return str(preset or DEFAULT_ADVANCED_COMPARISON_PLANNER_PRESET).strip() or DEFAULT_ADVANCED_COMPARISON_PLANNER_PRESET


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
    save_traces: bool,
) -> tuple[Path, Path, float | None]:
    source = resolve_config_path(str(scenario))
    cfg = load_yaml(source)
    if duration_s is not None:
        cfg.setdefault("scenario", {})["duration_s"] = float(duration_s)
    if save_traces:
        logging_cfg = cfg.setdefault("logging", {})
        logging_cfg["save_trace"] = True
        logging_cfg["trace_save_failures_only"] = False
        logging_cfg["trace_agents_mode"] = "all"
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


def _comparison_type(scenario_stem: str) -> str:
    if scenario_stem == "urban_conflict_3d":
        return "advanced_baseline_3d_conflict"
    if scenario_stem == "urban_throughput_3d":
        return "advanced_baseline_3d_urban_throughput"
    return "advanced_baseline_3d_custom"


def _trace_path_for_row(out_dir: Path, row: dict[str, Any]) -> Path:
    return out_dir / "episodes" / episode_dir_name(
        scenario=str(row.get("scenario")),
        method=str(row.get("method")),
        n_agents=int(row.get("N")),
        seed=int(row.get("seed")),
        comm_profile=str(row.get("comm_profile") or ""),
    ) / "trace_episode.jsonl"


def _export_comparison_mcap(
    *,
    out_dir: Path,
    rows: list[dict[str, Any]],
    mcap_path: str | Path | None,
    trail_frames: int,
    max_sensing_links: int,
    compression: str,
) -> dict[str, Any]:
    labels: list[str] = []
    traces: list[str] = []
    missing: list[str] = []
    trace_paths: dict[str, str] = {}

    for row in rows:
        method = str(row.get("method"))
        trace_path = _trace_path_for_row(out_dir, row)
        trace_paths[method] = str(trace_path)
        if not trace_path.exists():
            missing.append(str(trace_path))
            continue
        labels.append(method)
        traces.append(str(trace_path))

    if missing:
        raise RuntimeError("cannot export Foxglove comparison MCAP; missing trace(s): " + ", ".join(missing))

    out_path = Path(mcap_path) if mcap_path is not None else out_dir / DEFAULT_ADVANCED_COMPARISON_MCAP
    exported = export_foxglove_comparison_mcap(
        traces,
        str(out_path),
        labels=labels,
        trail_frames=int(trail_frames),
        max_sensing_links=int(max_sensing_links),
        compression=str(compression),
    )
    return {
        "path": str(exported),
        "labels": labels,
        "trace_paths": trace_paths,
        "topic_root": "/daa/comparison",
        "trail_frames": int(trail_frames),
        "max_sensing_links": int(max_sensing_links),
        "compression": str(compression),
    }


def _comparison_manifest(report: dict[str, Any], *, report_path: Path) -> dict[str, Any]:
    foxglove_mcap = report.get("foxglove_mcap") if isinstance(report.get("foxglove_mcap"), dict) else None
    return _json_safe({
        "schema_version": ADVANCED_BASELINE_COMPARISON_SCHEMA_VERSION,
        "generated_by": "python -m microbench.cli advanced-baseline-comparison",
        "git_commit": report.get("git_commit"),
        "ok": bool(report.get("ok")),
        "complete": bool(report.get("complete")),
        "comparison_type": report.get("comparison_type"),
        "scenario": report.get("scenario"),
        "scenario_source": report.get("scenario_source"),
        "scenario_path": report.get("scenario_path"),
        "methods": report.get("methods", []),
        "n_agents": report.get("n_agents"),
        "seed": report.get("seed"),
        "duration_s": report.get("duration_s"),
        "comm_profile": report.get("comm_profile"),
        "planner_preset": report.get("planner_preset"),
        "save_traces": bool(report.get("save_traces")),
        "foxglove_mcap_path": foxglove_mcap.get("path") if foxglove_mcap else None,
        "checks": report.get("checks", {}),
        "guardrail_failures": report.get("guardrail_failures", {}),
        "nonfinite_methods": report.get("nonfinite_methods", []),
        "results_csv": report.get("results_csv"),
        "summary_csv": report.get("summary_csv"),
        "result_schema": report.get("result_schema"),
        "baseline_report_path": report.get("baseline_report_path"),
        "advanced_baseline_comparison_path": str(report_path),
        "note": (
            "Self-contained manifest for visual/qualitative baseline comparison runs. "
            "Use planner_preset='scale' when the MCAP should match scale-tuned optimizer settings."
        ),
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
    export_foxglove_mcap: bool = False,
    foxglove_mcap_path: str | Path | None = None,
    mcap_trail_frames: int = 200,
    mcap_max_sensing_links: int = 200,
    mcap_compression: str = "zstd",
    planner_preset: str = DEFAULT_ADVANCED_COMPARISON_PLANNER_PRESET,
) -> dict[str, Any]:
    out = Path(out_dir)
    if (out / "results.csv").exists():
        raise RuntimeError(f"advanced baseline comparison output already exists: {out / 'results.csv'}")
    out.mkdir(parents=True, exist_ok=True)

    effective_save_traces = bool(save_traces or export_foxglove_mcap)
    clean_planner_preset = _clean_planner_preset(planner_preset)
    method_values = _as_list(methods, DEFAULT_ADVANCED_COMPARISON_METHODS)
    source_scenario, scenario_path, effective_duration_s = _prepare_scenario(
        scenario=scenario,
        out_dir=out,
        duration_s=duration_s,
        save_traces=effective_save_traces,
    )

    rows: list[dict[str, Any]] = []
    with _planner_preset_env(clean_planner_preset):
        for method in method_values:
            spec = RunSpec(
                scenario_path=str(scenario_path),
                method=str(method),
                n_agents=int(n_agents),
                seed=int(seed),
                comm_profile=str(comm_profile),
                out_dir=str(out),
                save_trace=effective_save_traces,
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
    scenario_stem = Path(scenario_path).stem
    foxglove_mcap = None
    if export_foxglove_mcap:
        foxglove_mcap = _export_comparison_mcap(
            out_dir=out,
            rows=rows,
            mcap_path=foxglove_mcap_path,
            trail_frames=mcap_trail_frames,
            max_sensing_links=mcap_max_sensing_links,
            compression=mcap_compression,
        )
    report_path = out / "advanced_baseline_comparison.json"
    manifest_path = out / "comparison_manifest.json"
    report = _json_safe({
        "schema_version": ADVANCED_BASELINE_COMPARISON_SCHEMA_VERSION,
        "comparison_type": _comparison_type(scenario_stem),
        "ok": bool(complete and not guardrail_failures and not nonfinite_methods),
        "complete": bool(complete),
        "git_commit": _git_commit(),
        "methods": method_values,
        "scenario_source": str(source_scenario),
        "scenario_path": str(scenario_path),
        "scenario": scenario_stem,
        "duration_s": effective_duration_s,
        "n_agents": int(n_agents),
        "seed": int(seed),
        "comm_profile": str(comm_profile),
        "planner_preset": clean_planner_preset,
        "save_traces": effective_save_traces,
        "foxglove_mcap": foxglove_mcap,
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
        "comparison_manifest_path": str(manifest_path),
        "score_note": "score_v0 follows docs/LEADERBOARD.md; use component metrics, not only rank.",
    })
    report["report_path"] = str(report_path)
    report_path.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            _comparison_manifest(report, report_path=report_path),
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def write_advanced_baseline_comparison(*, out_dir: str | Path, **kwargs: Any) -> Path:
    run_advanced_baseline_comparison(out_dir=out_dir, **kwargs)
    return Path(out_dir) / "advanced_baseline_comparison.json"
