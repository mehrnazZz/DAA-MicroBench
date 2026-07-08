from __future__ import annotations

import csv
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import signal
import threading
import time
from typing import Any

from microbench.acceptance import check_acceptance
from microbench.metrics import append_result, write_summary
from microbench.metrics.io import RESULT_FIELDS, SUMMARY_FIELDS, write_result_schema_manifest
from microbench.runner import run_episode
from microbench.scenarios import list_official_suites, materialize_official_suite, suite_defaults
from microbench.tools.baseline_report import build_baseline_report
from microbench.types import RunSpec


BASELINE_LEADERBOARD_SCHEMA_VERSION = "0.2"

SERIOUS_BASELINE_METHODS = (
    "baseline_goal",
    "orca_heuristic",
    "orca_with_staleness",
    "velocity_obstacle",
    "reciprocal_velocity_obstacle",
    "cbf_qp",
    "mpc_local",
    "priority_yield",
    "negotiation_yield",
    "learned_tiny",
)


def _as_list(values: tuple[str, ...] | list[str] | None, default: tuple[str, ...]) -> list[str]:
    return [str(v).strip() for v in (values if values is not None else default) if str(v).strip()]


def _rel(path: str | Path, root: Path) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _spec_key(spec: RunSpec) -> tuple[str, str, str, str, str]:
    return (
        Path(spec.scenario_path).stem,
        str(spec.method),
        str(spec.comm_profile),
        str(int(spec.n_agents)),
        str(int(spec.seed)),
    )


def _row_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("scenario", "")),
        str(row.get("method", "")),
        str(row.get("comm_profile", "")),
        str(row.get("N", "")),
        str(row.get("seed", "")),
    )


def _read_result_rows(results_csv: Path) -> list[dict[str, str]]:
    if not results_csv.exists():
        return []
    with results_csv.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _existing_keys(results_csv: Path) -> set[tuple[str, str, str, str, str]]:
    return {_row_key(row) for row in _read_result_rows(results_csv)}


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _hard_timeout_row_count(results_csv: Path) -> int:
    count = 0
    for row in _read_result_rows(results_csv):
        planner_timeout = _float_or_none(row.get("planner_timeout_count")) or 0.0
        planner_error = _float_or_none(row.get("planner_error_count")) or 0.0
        duration = _float_or_none(row.get("duration_s"))
        if planner_timeout > 0.0 and planner_error > 0.0 and duration is None:
            count += 1
    return count


def _write_empty_summary(run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    write_result_schema_manifest(run_dir)
    results_path = run_dir / "results.csv"
    if not results_path.exists():
        with results_path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=RESULT_FIELDS).writeheader()
    summary_path = run_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=SUMMARY_FIELDS).writeheader()
    return summary_path


class LeaderboardRunTimeout(TimeoutError):
    pass


def _hard_timeout_supported() -> bool:
    return (
        hasattr(signal, "SIGALRM")
        and hasattr(signal, "ITIMER_REAL")
        and threading.current_thread() is threading.main_thread()
    )


@contextmanager
def _episode_time_limit(timeout_s: float | None):
    if timeout_s is None:
        yield
        return
    timeout = float(timeout_s)
    if timeout <= 0.0:
        raise LeaderboardRunTimeout("episode exceeded run timeout before launch")
    if not _hard_timeout_supported():
        yield
        return

    def _raise_timeout(signum, frame):
        _ = signum, frame
        raise LeaderboardRunTimeout(f"episode exceeded run timeout of {timeout:.3f}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _timeout_row(spec: RunSpec, *, elapsed_s: float, timeout_s: float | None) -> dict[str, Any]:
    row = {field: float("nan") for field in RESULT_FIELDS}
    row.update(
        {
            "run_id": Path(spec.out_dir).name,
            "method": str(spec.method),
            "scenario": Path(spec.scenario_path).stem,
            "comm_profile": str(spec.comm_profile),
            "N": int(spec.n_agents),
            "seed": int(spec.seed),
            "planner_timeout_count": 1,
            "planner_error_count": 1,
            "planner_fallback_count": 0,
            "episode_runtime_s": float(elapsed_s),
        }
    )
    if timeout_s is not None:
        row["planner_ms_per_tick_per_agent_p95"] = float(timeout_s) * 1000.0
    return row


def _run_episode_checked(spec: RunSpec, *, run_timeout_s: float | None) -> tuple[dict[str, Any], bool]:
    started = time.perf_counter()
    try:
        with _episode_time_limit(run_timeout_s):
            return run_episode(spec), False
    except LeaderboardRunTimeout:
        elapsed = time.perf_counter() - started
        return _timeout_row(spec, elapsed_s=elapsed, timeout_s=run_timeout_s), True


def _aggregate_method_scores(suite_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[str, list[float]] = {}
    episodes: dict[str, int] = {}
    suites: dict[str, set[str]] = {}
    for suite_report in suite_reports:
        suite_id = str(suite_report["suite"])
        report = suite_report["report"]
        for entry in report.get("method_summaries", []):
            method = str(entry.get("method"))
            score = entry.get("score_v0_mean")
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                by_method.setdefault(method, []).append(float(score))
            episodes[method] = episodes.get(method, 0) + int(entry.get("episodes") or 0)
            suites.setdefault(method, set()).add(suite_id)

    ranking = []
    for method, scores in sorted(by_method.items()):
        ranking.append(
            {
                "method": method,
                "score_v0_mean": _mean(scores),
                "score_v0_best_suite": round(min(scores), 6) if scores else None,
                "score_v0_worst_suite": round(max(scores), 6) if scores else None,
                "suite_count": len(suites.get(method, set())),
                "episodes": int(episodes.get(method, 0)),
            }
        )
    ranking.sort(key=lambda row: (float("inf") if row["score_v0_mean"] is None else float(row["score_v0_mean"]), row["method"]))
    for rank, row in enumerate(ranking, start=1):
        row["rank"] = rank
    return ranking


def _run_suite(
    *,
    out_dir: Path,
    suite: str,
    methods: list[str],
    n_agents: list[int] | None,
    seeds: list[int] | None,
    comm_profiles: list[str] | None,
    max_runs: int | None,
    stretch: bool,
    resume: bool,
    deadline_at: float | None,
    max_wall_time_s: float | None,
    run_timeout_s: float | None,
) -> dict[str, Any]:
    suite_dir = out_dir / suite
    run_dir = suite_dir / "runs"
    generated_dir = suite_dir / "_generated_scenarios" / suite
    generated = materialize_official_suite(suite, generated_dir, overwrite=True, stretch=stretch)
    defaults = suite_defaults(suite, stretch=stretch)
    scenario_paths = [Path(path) for path in generated["scenario_paths"]]
    n_values = list(n_agents if n_agents is not None else [int(x) for x in defaults["n_agents"]])
    seed_values = list(seeds if seeds is not None else [int(x) for x in defaults["seeds"]])
    comm_values = list(comm_profiles if comm_profiles is not None else [str(x) for x in defaults["comm_profiles"]])

    specs: list[RunSpec] = []
    for scenario_path in scenario_paths:
        for method in methods:
            for comm in comm_values:
                for n in n_values:
                    for seed in seed_values:
                        specs.append(
                            RunSpec(
                                scenario_path=str(scenario_path),
                                method=str(method),
                                n_agents=int(n),
                                seed=int(seed),
                                comm_profile=str(comm),
                                out_dir=str(run_dir),
                                save_trace=False,
                            )
                        )

    planned_run_count = len(specs)
    if max_runs is not None:
        specs = specs[: max(0, int(max_runs))]
    selected_run_count = len(specs)

    results_csv = run_dir / "results.csv"
    if results_csv.exists() and not resume:
        raise RuntimeError(
            f"{results_csv} already exists. Use --resume to continue this leaderboard run "
            "or choose a fresh --out-dir."
        )

    completed_keys = _existing_keys(results_csv) if resume else set()
    skipped_existing = 0
    newly_run = 0
    newly_timed_out = 0
    stopped_by_wall_time = False
    wall_started = time.perf_counter()

    for spec in specs:
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

    existing_after = _existing_keys(results_csv) if results_csv.exists() else set()
    selected_keys = {_spec_key(spec) for spec in specs}
    selected_completed_count = len(selected_keys & existing_after)
    result_row_count = len(_read_result_rows(results_csv))

    summary_csv = write_summary(run_dir)
    if not summary_csv.exists():
        summary_csv = _write_empty_summary(run_dir)
    report = build_baseline_report(
        summary_csv=summary_csv,
        results_csv=results_csv if results_csv.exists() else None,
        suite=suite,
        generated_by="python -m microbench.cli baseline-leaderboard",
    )
    report_path = suite_dir / "baseline_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    acceptance = check_acceptance(
        summary_csv=summary_csv,
        results_csv=results_csv if results_csv.exists() else None,
        suite_manifest=generated["manifest_path"],
        methods=methods,
    )
    acceptance_path = suite_dir / "acceptance.json"
    acceptance_path.write_text(json.dumps(acceptance, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    hard_timeout_count = _hard_timeout_row_count(results_csv) if results_csv.exists() else 0
    truncated_by_max_runs = max_runs is not None and selected_run_count < planned_run_count
    selected_complete = selected_completed_count == selected_run_count
    complete = selected_completed_count == planned_run_count and max_runs is None and not stopped_by_wall_time and hard_timeout_count == 0
    suite_result = {
        "suite": suite,
        "ok": bool(acceptance.get("ok", False)),
        "planned_run_count": planned_run_count,
        "selected_run_count": selected_run_count,
        "run_count": result_row_count,
        "selected_completed_count": selected_completed_count,
        "new_run_count": newly_run,
        "skipped_existing_count": skipped_existing,
        "timeout_run_count": hard_timeout_count,
        "new_timeout_run_count": newly_timed_out,
        "methods": methods,
        "scenario_count": len(scenario_paths),
        "n_agents": n_values,
        "seeds": seed_values,
        "comm_profiles": comm_values,
        "report": report,
        "report_path": str(report_path),
        "acceptance": acceptance,
        "acceptance_path": str(acceptance_path),
        "results_csv": str(results_csv),
        "summary_csv": str(summary_csv),
        "suite_manifest": str(generated["manifest_path"]),
        "truncated_by_max_runs": truncated_by_max_runs,
        "stopped_by_wall_time": stopped_by_wall_time,
        "resume": bool(resume),
        "run_timeout_s": None if run_timeout_s is None else float(run_timeout_s),
        "max_wall_time_s": None if max_wall_time_s is None else float(max_wall_time_s),
        "run_timeout_supported": _hard_timeout_supported(),
        "selected_complete": selected_complete,
        "complete": complete,
        "wall_runtime_s": round(time.perf_counter() - wall_started, 6),
    }
    progress_path = suite_dir / "leaderboard_progress.json"
    progress = {
        "schema_version": BASELINE_LEADERBOARD_SCHEMA_VERSION,
        "updated_at": _now_utc(),
        "suite": suite,
        "planned_run_count": planned_run_count,
        "selected_run_count": selected_run_count,
        "selected_completed_count": selected_completed_count,
        "new_run_count": newly_run,
        "skipped_existing_count": skipped_existing,
        "timeout_run_count": hard_timeout_count,
        "new_timeout_run_count": newly_timed_out,
        "truncated_by_max_runs": truncated_by_max_runs,
        "stopped_by_wall_time": stopped_by_wall_time,
        "selected_complete": selected_complete,
        "complete": complete,
        "resume": bool(resume),
        "max_wall_time_s": None if max_wall_time_s is None else float(max_wall_time_s),
        "run_timeout_s": None if run_timeout_s is None else float(run_timeout_s),
        "run_timeout_supported": _hard_timeout_supported(),
        "results_csv": _rel(results_csv, suite_dir),
        "summary_csv": _rel(summary_csv, suite_dir),
    }
    progress_path.write_text(json.dumps(progress, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    suite_result["progress_path"] = str(progress_path)
    return suite_result


def run_baseline_leaderboard(
    *,
    out_dir: str | Path,
    suites: tuple[str, ...] | list[str] | None = None,
    methods: tuple[str, ...] | list[str] | None = None,
    n_agents: tuple[int, ...] | list[int] | None = None,
    seeds: tuple[int, ...] | list[int] | None = None,
    comm_profiles: tuple[str, ...] | list[str] | None = None,
    max_runs: int | None = None,
    stretch: bool = False,
    resume: bool = False,
    max_wall_time_s: float | None = None,
    run_timeout_s: float | None = None,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    suite_values = _as_list(suites, tuple(list_official_suites()))
    method_values = _as_list(methods, SERIOUS_BASELINE_METHODS)
    official = set(list_official_suites())
    unknown = sorted(set(suite_values) - official)
    if unknown:
        raise ValueError(f"unknown official suite(s): {','.join(unknown)}")

    started = time.perf_counter()
    deadline_at = started + float(max_wall_time_s) if max_wall_time_s is not None else None
    suite_reports = [
        _run_suite(
            out_dir=out,
            suite=suite,
            methods=method_values,
            n_agents=list(n_agents) if n_agents is not None else None,
            seeds=list(seeds) if seeds is not None else None,
            comm_profiles=list(comm_profiles) if comm_profiles is not None else None,
            max_runs=max_runs,
            stretch=bool(stretch),
            resume=bool(resume),
            deadline_at=deadline_at,
            max_wall_time_s=max_wall_time_s,
            run_timeout_s=run_timeout_s,
        )
        for suite in suite_values
    ]

    aggregate_ranking = _aggregate_method_scores(suite_reports)
    complete = all(entry["complete"] for entry in suite_reports)
    selected_complete = all(entry["selected_complete"] for entry in suite_reports)
    timeout_run_count = sum(int(entry["timeout_run_count"]) for entry in suite_reports)
    stopped_by_wall_time = any(bool(entry["stopped_by_wall_time"]) for entry in suite_reports)
    report = {
        "schema_version": BASELINE_LEADERBOARD_SCHEMA_VERSION,
        "ok": all(entry["ok"] for entry in suite_reports),
        "complete": complete,
        "selected_complete": selected_complete,
        "stopped_by_wall_time": stopped_by_wall_time,
        "timeout_run_count": timeout_run_count,
        "resume": bool(resume),
        "max_wall_time_s": None if max_wall_time_s is None else float(max_wall_time_s),
        "run_timeout_s": None if run_timeout_s is None else float(run_timeout_s),
        "wall_runtime_s": round(time.perf_counter() - started, 6),
        "out_dir": str(out),
        "suites": [
            {
                "suite": entry["suite"],
                "ok": entry["ok"],
                "planned_run_count": entry["planned_run_count"],
                "selected_run_count": entry["selected_run_count"],
                "run_count": entry["run_count"],
                "selected_completed_count": entry["selected_completed_count"],
                "new_run_count": entry["new_run_count"],
                "skipped_existing_count": entry["skipped_existing_count"],
                "timeout_run_count": entry["timeout_run_count"],
                "scenario_count": entry["scenario_count"],
                "report_path": _rel(entry["report_path"], out),
                "acceptance_path": _rel(entry["acceptance_path"], out),
                "progress_path": _rel(entry["progress_path"], out),
                "results_csv": _rel(entry["results_csv"], out),
                "summary_csv": _rel(entry["summary_csv"], out),
                "suite_manifest": _rel(entry["suite_manifest"], out),
                "truncated_by_max_runs": entry["truncated_by_max_runs"],
                "stopped_by_wall_time": entry["stopped_by_wall_time"],
                "selected_complete": entry["selected_complete"],
                "complete": entry["complete"],
            }
            for entry in suite_reports
        ],
        "methods": method_values,
        "aggregate_ranking": aggregate_ranking,
        "score_note": "score_v0 follows docs/LEADERBOARD.md; use per-suite rankings for official comparisons.",
    }
    leaderboard_path = out / "baseline_leaderboard.json"
    report["leaderboard_path"] = str(leaderboard_path)
    leaderboard_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
