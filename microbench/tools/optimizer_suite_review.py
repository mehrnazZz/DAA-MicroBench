from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import yaml

from microbench.metrics import episode_dir_name
from microbench.runner import run_episode
from microbench.scenarios import list_official_suites
from microbench.tools.baseline_leaderboard import _run_episode_checked, run_baseline_leaderboard
from microbench.types import RunSpec


OPTIMIZER_SUITE_REVIEW_SCHEMA_VERSION = "0.2"
OPTIMIZER_REVIEW_METHODS = ("mpc_nonlinear", "ego_swarm_opt")
DEFAULT_OPTIMIZER_REVIEW_SUITES = ("official_smoke_generated", "official_promotion_calibration")
GUARDRAIL_FIELDS = ("planner_timeout_count", "planner_error_count", "planner_fallback_count")


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _int(value: Any) -> int | None:
    number = _num(value)
    return int(number) if number is not None else None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _suite_manifest_dir(out_dir: Path, suite_entry: dict[str, Any]) -> Path:
    manifest_path = Path(str(suite_entry["suite_manifest"]))
    if not manifest_path.is_absolute():
        manifest_path = out_dir / manifest_path
    return manifest_path.parent


def _scenario_path(out_dir: Path, suite_entry: dict[str, Any], scenario: str) -> Path:
    return _suite_manifest_dir(out_dir, suite_entry) / f"{scenario}.yaml"


def _scenario_dimension(path: Path) -> str | None:
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return None
    benchmark = cfg.get("benchmark", {}) if isinstance(cfg, dict) else {}
    dimension = benchmark.get("dimension") if isinstance(benchmark, dict) else None
    if dimension:
        return str(dimension)
    world = cfg.get("world", {}) if isinstance(cfg, dict) else {}
    if isinstance(world, dict):
        return "2d" if bool(world.get("planar", True)) else "3d"
    return None


def _row_key(row: dict[str, Any]) -> tuple[str, str, str, int, int]:
    return (
        str(row.get("scenario", "")),
        str(row.get("method", "")),
        str(row.get("comm_profile", "")),
        int(_int(row.get("N")) or 0),
        int(_int(row.get("seed")) or 0),
    )


def _guardrail_count(row: dict[str, Any]) -> int:
    return sum(int(_num(row.get(field)) or 0) for field in GUARDRAIL_FIELDS)


def _is_hard_timeout_row(row: dict[str, Any]) -> bool:
    duration = _num(row.get("duration_s"))
    return (
        int(_num(row.get("planner_timeout_count")) or 0) > 0
        and int(_num(row.get("planner_error_count")) or 0) > 0
        and duration is None
    )


def _all_result_rows(*, out_dir: Path, leaderboard: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for suite_entry in leaderboard.get("suites", []):
        results_path = out_dir / str(suite_entry["results_csv"])
        for row in _read_csv(results_path):
            rows.append((suite_entry, row))
    return rows


def _row_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "scenario": str(row.get("scenario", "")),
        "method": str(row.get("method", "")),
        "comm_profile": str(row.get("comm_profile", "")),
        "N": int(_int(row.get("N")) or 0),
        "seed": int(_int(row.get("seed")) or 0),
        "collision_episode": _num(row.get("collision_episode")),
        "min_sep_min_m": _num(row.get("min_sep_min_m")),
        "completion_rate": _num(row.get("completion_rate")),
        "planner_ms_per_tick_per_agent_p95": _num(row.get("planner_ms_per_tick_per_agent_p95")),
        "planner_timeout_count": _num(row.get("planner_timeout_count")),
        "planner_error_count": _num(row.get("planner_error_count")),
        "planner_fallback_count": _num(row.get("planner_fallback_count")),
        "guardrail_count": _guardrail_count(row),
        "hard_timeout": _is_hard_timeout_row(row),
    }


def _case_slug(row: dict[str, Any]) -> str:
    scenario, method, comm_profile, n_agents, seed = _row_key(row)
    raw = f"{scenario}_{method}_n{n_agents}_seed{seed}_{comm_profile}"
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in raw)


def _guardrail_retry_attempt(
    *,
    out_dir: Path,
    suite_entry: dict[str, Any],
    row: dict[str, Any],
    attempt: int,
    run_timeout_s: float | None,
) -> dict[str, Any]:
    scenario = str(row.get("scenario", ""))
    method = str(row.get("method", ""))
    comm_profile = str(row.get("comm_profile", ""))
    n_agents = int(_int(row.get("N")) or 0)
    seed = int(_int(row.get("seed")) or 0)
    attempt_dir = out_dir / "guardrail_retries" / str(suite_entry["suite"]) / _case_slug(row) / f"attempt_{attempt}"
    spec = RunSpec(
        scenario_path=str(_scenario_path(out_dir, suite_entry, scenario)),
        method=method,
        n_agents=n_agents,
        seed=seed,
        comm_profile=comm_profile,
        out_dir=str(attempt_dir),
        save_trace=False,
    )
    retry_row, hard_timed_out = _run_episode_checked(spec, run_timeout_s=run_timeout_s)
    snapshot = _row_snapshot(retry_row)
    snapshot.update(
        {
            "attempt": int(attempt),
            "hard_timed_out": bool(hard_timed_out),
            "out_dir": str(attempt_dir),
            "status": "clear" if _guardrail_count(retry_row) == 0 and not hard_timed_out else "guardrail_present",
        }
    )
    attempt_dir.mkdir(parents=True, exist_ok=True)
    (attempt_dir / "result_row.json").write_text(
        json.dumps(_json_safe(snapshot), allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return snapshot


def _guardrail_retry_evidence(
    *,
    out_dir: Path,
    leaderboard: dict[str, Any],
    retry_count: int,
    run_timeout_s: float | None,
) -> tuple[list[dict[str, Any]], set[tuple[str, str, str, int, int]]]:
    max_attempts = max(0, int(retry_count))
    evidence: list[dict[str, Any]] = []
    resolved_keys: set[tuple[str, str, str, int, int]] = set()
    if max_attempts == 0:
        return evidence, resolved_keys

    for suite_entry, row in _all_result_rows(out_dir=out_dir, leaderboard=leaderboard):
        if _guardrail_count(row) == 0:
            continue
        attempts: list[dict[str, Any]] = []
        status = "persistent_guardrail"
        for attempt in range(1, max_attempts + 1):
            retry = _guardrail_retry_attempt(
                out_dir=out_dir,
                suite_entry=suite_entry,
                row=row,
                attempt=attempt,
                run_timeout_s=run_timeout_s,
            )
            attempts.append(retry)
            if retry["status"] == "clear":
                status = "resolved_on_retry"
                resolved_keys.add(_row_key(row))
                break
        evidence.append(
            {
                "suite": str(suite_entry["suite"]),
                "key": list(_row_key(row)),
                "status": status,
                "original": _row_snapshot(row),
                "attempts": attempts,
            }
        )
    return evidence, resolved_keys


def _case_severity(row: dict[str, Any]) -> tuple[float, float, float, float, str]:
    collision = _num(row.get("collision_episode")) or 0.0
    guardrails = float(_guardrail_count(row))
    min_sep = _num(row.get("min_sep_min_m"))
    completion = _num(row.get("completion_rate"))
    return (
        -collision,
        -guardrails,
        float("inf") if min_sep is None else min_sep,
        float("inf") if completion is None else completion,
        str(row.get("method", "")) + str(row.get("scenario", "")) + str(row.get("seed", "")),
    )


def _case_record(
    *,
    out_dir: Path,
    suite_entry: dict[str, Any],
    row: dict[str, Any],
    trace_root: Path,
) -> dict[str, Any]:
    scenario = str(row.get("scenario", ""))
    method = str(row.get("method", ""))
    comm_profile = str(row.get("comm_profile", ""))
    n_agents = int(_int(row.get("N")) or 0)
    seed = int(_int(row.get("seed")) or 0)
    scenario_path = _scenario_path(out_dir, suite_entry, scenario)
    trace_run_dir = trace_root / str(suite_entry["suite"]) / f"{scenario}_{method}_n{n_agents}_seed{seed}_{comm_profile}"
    trace_scenario_path = trace_run_dir / "_trace_scenario" / scenario_path.name
    episode_dir = trace_run_dir / "episodes" / episode_dir_name(
        scenario=scenario,
        method=method,
        n_agents=n_agents,
        seed=seed,
        comm_profile=comm_profile,
    )
    trace_path = episode_dir / "trace_episode.jsonl"
    mcap_path = episode_dir / "trace_episode.mcap"
    return {
        "suite": suite_entry["suite"],
        "scenario": scenario,
        "scenario_path": str(scenario_path),
        "trace_scenario_path": str(trace_scenario_path),
        "dimension": _scenario_dimension(scenario_path),
        "method": method,
        "comm_profile": comm_profile,
        "N": n_agents,
        "seed": seed,
        "collision_episode": _num(row.get("collision_episode")),
        "min_sep_min_m": _num(row.get("min_sep_min_m")),
        "completion_rate": _num(row.get("completion_rate")),
        "planner_ms_per_tick_per_agent_p95": _num(row.get("planner_ms_per_tick_per_agent_p95")),
        "planner_timeout_count": _num(row.get("planner_timeout_count")),
        "planner_error_count": _num(row.get("planner_error_count")),
        "planner_fallback_count": _num(row.get("planner_fallback_count")),
        "trace_run_dir": str(trace_run_dir),
        "trace_path": str(trace_path),
        "foxglove_export_command": f"python -m microbench.cli foxglove-export --trace {trace_path} --out {mcap_path}",
        "rerun_trace_command": (
            "python -m microbench.cli run "
            f"--scenario {trace_scenario_path} --method {method} --n {n_agents} --seed {seed} "
            f"--comm {comm_profile} --out-dir {trace_run_dir} --save-trace"
        ),
    }


def _interesting_cases(
    *,
    out_dir: Path,
    leaderboard: dict[str, Any],
    max_cases: int,
    resolved_guardrail_keys: set[tuple[str, str, str, int, int]] | None = None,
) -> list[dict[str, Any]]:
    rows = _all_result_rows(out_dir=out_dir, leaderboard=leaderboard)
    resolved = resolved_guardrail_keys or set()

    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    seen: set[tuple[str, str, str, int, int]] = set()
    for predicate in (
        lambda r: (_num(r.get("collision_episode")) or 0.0) > 0.0,
        lambda r: _guardrail_count(r) > 0 and _row_key(r) not in resolved,
        lambda r: (_num(r.get("min_sep_min_m")) is not None and float(_num(r.get("min_sep_min_m")) or 0.0) < 0.0),
        lambda r: (_num(r.get("completion_rate")) is not None and float(_num(r.get("completion_rate")) or 0.0) < 1.0),
    ):
        candidates = sorted((item for item in rows if predicate(item[1])), key=lambda item: _case_severity(item[1]))
        for item in candidates:
            key = _row_key(item[1])
            if key not in seen:
                selected.append(item)
                seen.add(key)
            if len(selected) >= max_cases:
                break
        if len(selected) >= max_cases:
            break

    if len(selected) < max_cases:
        remaining = sorted((item for item in rows if _row_key(item[1]) not in seen), key=lambda item: _case_severity(item[1]))
        for item in remaining:
            selected.append(item)
            seen.add(_row_key(item[1]))
            if len(selected) >= max_cases:
                break

    trace_root = out_dir / "review_traces"
    return [
        _case_record(out_dir=out_dir, suite_entry=suite_entry, row=row, trace_root=trace_root)
        for suite_entry, row in selected[:max(0, int(max_cases))]
    ]


def _prepare_trace_scenario(source: Path, dest_dir: Path, *, trace_max_steps: int) -> Path:
    cfg = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    logging_cfg = cfg.setdefault("logging", {})
    logging_cfg["save_trace"] = True
    logging_cfg["trace_save_failures_only"] = False
    logging_cfg["trace_max_steps"] = int(trace_max_steps)
    logging_cfg["save_events"] = True
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / source.name
    dest.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return dest


def _prepare_trace_scenarios(cases: list[dict[str, Any]], *, trace_max_steps: int) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for case in cases:
        case = dict(case)
        trace_scenario_path = _prepare_trace_scenario(
            Path(str(case["scenario_path"])),
            Path(str(case["trace_scenario_path"])).parent,
            trace_max_steps=trace_max_steps,
        )
        case["trace_scenario_path"] = str(trace_scenario_path)
        prepared.append(case)
    return prepared


def _write_review_traces(cases: list[dict[str, Any]], *, trace_max_steps: int) -> list[dict[str, Any]]:
    written: list[dict[str, Any]] = []
    for case in cases:
        trace_run_dir = Path(str(case["trace_run_dir"]))
        trace_scenario = Path(str(case["trace_scenario_path"]))
        if not trace_scenario.exists():
            trace_scenario = _prepare_trace_scenario(
                Path(str(case["scenario_path"])),
                trace_run_dir / "_trace_scenario",
                trace_max_steps=trace_max_steps,
            )
        spec = RunSpec(
            scenario_path=str(trace_scenario),
            method=str(case["method"]),
            n_agents=int(case["N"]),
            seed=int(case["seed"]),
            comm_profile=str(case["comm_profile"]),
            out_dir=str(trace_run_dir),
            save_trace=True,
        )
        row = run_episode(spec)
        trace_path = Path(str(case["trace_path"]))
        case = dict(case)
        case["trace_status"] = "written" if trace_path.exists() else "missing"
        case["trace_row"] = {
            "collision_episode": _num(row.get("collision_episode")),
            "min_sep_min_m": _num(row.get("min_sep_min_m")),
            "completion_rate": _num(row.get("completion_rate")),
            "planner_ms_per_tick_per_agent_p95": _num(row.get("planner_ms_per_tick_per_agent_p95")),
        }
        written.append(case)
    return written


def _method_summaries(*, out_dir: Path, leaderboard: dict[str, Any], methods: list[str]) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in methods}
    suite_by_method: dict[str, set[str]] = {method: set() for method in methods}
    scenarios_by_method: dict[str, set[str]] = {method: set() for method in methods}
    dimensions_by_method: dict[str, set[str]] = {method: set() for method in methods}

    for suite_entry in leaderboard.get("suites", []):
        results_path = out_dir / str(suite_entry["results_csv"])
        for row in _read_csv(results_path):
            method = str(row.get("method", ""))
            if method not in by_method:
                continue
            by_method[method].append(row)
            suite_by_method[method].add(str(suite_entry["suite"]))
            scenario = str(row.get("scenario", ""))
            scenarios_by_method[method].add(scenario)
            dimension = _scenario_dimension(_scenario_path(out_dir, suite_entry, scenario))
            if dimension:
                dimensions_by_method[method].add(dimension)

    summaries: list[dict[str, Any]] = []
    for method in methods:
        rows = by_method[method]
        completion = [_num(row.get("completion_rate")) for row in rows]
        min_sep = [_num(row.get("min_sep_min_m")) for row in rows]
        planner = [_num(row.get("planner_ms_per_tick_per_agent_p95")) for row in rows]
        collision_eps = sum(1 for row in rows if (_num(row.get("collision_episode")) or 0.0) > 0.0)
        guardrail_rows = sum(1 for row in rows if _guardrail_count(row) > 0)
        summaries.append(
            {
                "method": method,
                "run_count": len(rows),
                "suite_count": len(suite_by_method[method]),
                "scenario_count": len(scenarios_by_method[method]),
                "dimensions": sorted(dimensions_by_method[method]),
                "collision_episode_rate": round(collision_eps / len(rows), 6) if rows else None,
                "completion_rate_mean": _mean([float(x) for x in completion if x is not None]),
                "min_sep_min_worst_m": round(min(float(x) for x in min_sep if x is not None), 6)
                if any(x is not None for x in min_sep)
                else None,
                "planner_ms_p95_max": round(max(float(x) for x in planner if x is not None), 6)
                if any(x is not None for x in planner)
                else None,
                "guardrail_row_count": guardrail_rows,
            }
        )
    return summaries


def _findings(
    *,
    out_dir: Path,
    leaderboard: dict[str, Any],
    resolved_guardrail_keys: set[tuple[str, str, str, int, int]] | None = None,
) -> dict[str, Any]:
    collision_rows = 0
    negative_clearance_rows = 0
    incomplete_rows = 0
    guardrail_rows = 0
    hard_timeout_rows = 0
    resolved = resolved_guardrail_keys or set()
    dimensions: set[str] = set()
    for suite_entry, row in _all_result_rows(out_dir=out_dir, leaderboard=leaderboard):
        if (_num(row.get("collision_episode")) or 0.0) > 0.0:
            collision_rows += 1
        min_sep = _num(row.get("min_sep_min_m"))
        if min_sep is not None and min_sep < 0.0:
            negative_clearance_rows += 1
        completion = _num(row.get("completion_rate"))
        if completion is not None and completion < 1.0:
            incomplete_rows += 1
        if _guardrail_count(row) > 0:
            guardrail_rows += 1
        if _is_hard_timeout_row(row):
            hard_timeout_rows += 1
        dimension = _scenario_dimension(_scenario_path(out_dir, suite_entry, str(row.get("scenario", ""))))
        if dimension:
            dimensions.add(dimension)
    persistent_guardrail_rows = sum(
        1
        for _, row in _all_result_rows(out_dir=out_dir, leaderboard=leaderboard)
        if _guardrail_count(row) > 0 and _row_key(row) not in resolved
    )
    persistent_hard_timeout_rows = sum(
        1
        for _, row in _all_result_rows(out_dir=out_dir, leaderboard=leaderboard)
        if _is_hard_timeout_row(row) and _row_key(row) not in resolved
    )
    return {
        "collision_episode_rows": collision_rows,
        "negative_clearance_rows": negative_clearance_rows,
        "incomplete_episode_rows": incomplete_rows,
        "guardrail_rows": guardrail_rows,
        "transient_guardrail_rows": guardrail_rows - persistent_guardrail_rows,
        "persistent_guardrail_rows": persistent_guardrail_rows,
        "hard_timeout_rows": hard_timeout_rows,
        "transient_hard_timeout_rows": hard_timeout_rows - persistent_hard_timeout_rows,
        "persistent_hard_timeout_rows": persistent_hard_timeout_rows,
        "dimensions_covered": sorted(dimensions),
    }


def run_optimizer_suite_review(
    *,
    out_dir: str | Path,
    suites: tuple[str, ...] | list[str] | None = None,
    methods: tuple[str, ...] | list[str] | None = None,
    n_agents: tuple[int, ...] | list[int] | None = None,
    seeds: tuple[int, ...] | list[int] | None = None,
    comm_profiles: tuple[str, ...] | list[str] | None = None,
    max_runs: int | None = None,
    max_runs_strategy: str = "balanced",
    stretch: bool = False,
    resume: bool = False,
    max_wall_time_s: float | None = None,
    run_timeout_s: float | None = None,
    max_trace_cases: int = 4,
    save_review_traces: bool = False,
    trace_max_steps: int = 4000,
    guardrail_retries: int = 1,
) -> dict[str, Any]:
    out = Path(out_dir)
    suite_values = list(suites) if suites is not None else list(DEFAULT_OPTIMIZER_REVIEW_SUITES)
    if suite_values == ["all"]:
        suite_values = list_official_suites()
    method_values = [str(method) for method in (methods if methods is not None else OPTIMIZER_REVIEW_METHODS)]

    leaderboard = run_baseline_leaderboard(
        out_dir=out,
        suites=suite_values,
        methods=method_values,
        n_agents=n_agents,
        seeds=seeds,
        comm_profiles=comm_profiles,
        max_runs=max_runs,
        max_runs_strategy=max_runs_strategy,
        stretch=stretch,
        resume=resume,
        max_wall_time_s=max_wall_time_s,
        run_timeout_s=run_timeout_s,
    )
    guardrail_retry_evidence, resolved_guardrail_keys = _guardrail_retry_evidence(
        out_dir=out,
        leaderboard=leaderboard,
        retry_count=guardrail_retries,
        run_timeout_s=run_timeout_s,
    )
    findings = _findings(out_dir=out, leaderboard=leaderboard, resolved_guardrail_keys=resolved_guardrail_keys)
    cases = _interesting_cases(
        out_dir=out,
        leaderboard=leaderboard,
        max_cases=max_trace_cases,
        resolved_guardrail_keys=resolved_guardrail_keys,
    )
    cases = _prepare_trace_scenarios(cases, trace_max_steps=trace_max_steps)
    trace_cases = _write_review_traces(cases, trace_max_steps=trace_max_steps) if save_review_traces else cases
    official_acceptance_ok = bool(leaderboard.get("ok", False))
    selected_complete = bool(leaderboard.get("selected_complete", False))
    no_persistent_guardrails = int(findings["persistent_guardrail_rows"]) == 0 and int(
        findings["persistent_hard_timeout_rows"]
    ) == 0

    report = _json_safe(
        {
            "schema_version": OPTIMIZER_SUITE_REVIEW_SCHEMA_VERSION,
            "review_type": "optimizer_suite_review",
            "ok": bool(official_acceptance_ok and selected_complete and no_persistent_guardrails),
            "official_acceptance_ok": official_acceptance_ok,
            "selected_complete": selected_complete,
            "publication_complete": bool(leaderboard.get("complete", False)),
            "methods": method_values,
            "suites": suite_values,
            "out_dir": str(out),
            "leaderboard_path": str(leaderboard.get("leaderboard_path")),
            "baseline_leaderboard": leaderboard,
            "method_summaries": _method_summaries(out_dir=out, leaderboard=leaderboard, methods=method_values),
            "findings": findings,
            "guardrail_retries": int(max(0, int(guardrail_retries))),
            "guardrail_retry_evidence": guardrail_retry_evidence,
            "guardrail_retry_resolved_rows": len(resolved_guardrail_keys),
            "guardrail_retry_note": (
                "Rows with planner guardrails are retried only to distinguish transient machine/runtime spikes "
                "from persistent benchmark failures. Official acceptance remains the original leaderboard result."
            ),
            "review_cases": trace_cases,
            "save_review_traces": bool(save_review_traces),
            "max_runs_strategy": str(max_runs_strategy),
            "score_note": "Uses baseline-leaderboard score_v0; inspect per-suite metrics and Foxglove traces before promotion claims.",
            "next_steps": [
                "Run without --max-runs for complete publication evidence when local runtime budget allows.",
                "Inspect review_cases in Foxglove, especially collision, negative-clearance, guardrail, and incomplete episodes.",
                "Tune optimizer defaults only after comparing safety, completion, compute, and communication metrics per suite.",
            ],
        }
    )
    report_path = out / "optimizer_suite_review.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def write_optimizer_suite_review(*, out_dir: str | Path, **kwargs: Any) -> Path:
    run_optimizer_suite_review(out_dir=out_dir, **kwargs)
    return Path(out_dir) / "optimizer_suite_review.json"
