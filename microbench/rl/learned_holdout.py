from __future__ import annotations

import csv
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
from pathlib import Path
import signal
import shutil
import threading
import time
from typing import Any

from microbench.metrics import append_result, write_result_schema_manifest, write_summary
from microbench.metrics.io import RESULT_FIELDS, SUMMARY_FIELDS
from microbench.rl.closed_loop_training import CLOSED_LOOP_BROAD_3D_HOLDOUT_SCENARIOS
from microbench.rl.policy_spec import load_policy_spec
from microbench.runner import run_episode
from microbench.scenarios import materialize_official_suite
from microbench.tools.baseline_report import build_baseline_report, score_v0
from microbench.types import RunSpec


LEARNED_HOLDOUT_EVAL_SCHEMA_VERSION = "0.1"
DEFAULT_LEARNED_HOLDOUT_REFERENCE_METHODS = ("dynamic_tube_dmpc", "ego_swarm_opt")
LEARNED_HOLDOUT_TABLE_FIELDS = (
    "kind",
    "label",
    "method",
    "policy_name",
    "adapter",
    "policy_spec",
    "run_count",
    "summary_row_count",
    "scenario_count",
    "collision_episodes",
    "near_miss_episodes",
    "collision_episode_rate_mean",
    "completion_rate_mean",
    "min_sep_min_row_m",
    "min_sep_p05_row_min_m",
    "score_v0_mean",
    "score_v0_worst",
    "planner_ms_p95_max",
    "planner_timeout_count",
    "planner_error_count",
    "planner_fallback_count",
    "results_csv",
    "summary_csv",
)
LEARNED_HOLDOUT_DELTA_FIELDS = (
    "learned_label",
    "reference_label",
    "learned_policy_name",
    "reference_method",
    "run_count_delta",
    "collision_episodes_delta",
    "near_miss_episodes_delta",
    "completion_rate_mean_delta",
    "min_sep_min_row_m_delta",
    "min_sep_p05_row_min_m_delta",
    "score_v0_mean_delta",
    "planner_ms_p95_max_delta",
    "planner_timeout_count_delta",
    "planner_error_count_delta",
    "planner_fallback_count_delta",
)


@dataclass(frozen=True)
class LearnedPolicyEntry:
    label: str
    policy_spec: Path
    policy_name: str
    adapter: str
    summary: dict[str, Any]


class LearnedHoldoutRunTimeout(BaseException):
    pass


def _round_or_none(value: Any, *, digits: int = 6) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return round(out, int(digits))


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _finite_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _finite_float(row.get(field))
        if value is not None:
            values.append(value)
    return values


def _row_timeout_count(row: dict[str, Any]) -> int:
    return int(_finite_float(row.get("planner_timeout_count")) or 0)


def _row_error_count(row: dict[str, Any]) -> int:
    return int(_finite_float(row.get("planner_error_count")) or 0)


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return round(float(sum(values) / len(values)), 6)


def _min_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return round(float(min(values)), 6)


def _max_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return round(float(max(values)), 6)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _spec_key(spec: RunSpec) -> tuple[str, str, str, str, str]:
    method = str(spec.method)
    if spec.agent_methods:
        method = "mixed[" + "+".join(str(m) for m in spec.agent_methods) + "]"
    return (
        Path(spec.scenario_path).stem,
        method,
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


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    return path


def _rel(path: str | Path, root: Path) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)


def _entry_label(raw: str | Path, *, used: set[str]) -> str:
    text = str(raw)
    if "=" in text:
        label, _ = text.split("=", 1)
        label = label.strip()
    else:
        label = Path(text).stem
    label = label or "learned_policy"
    base = label
    suffix = 2
    while label in used:
        label = f"{base}_{suffix}"
        suffix += 1
    used.add(label)
    return label


def parse_learned_policy_spec_entries(values: tuple[str, ...] | list[str]) -> list[LearnedPolicyEntry]:
    """Parse CLI-style learned policy entries in path or label=path form."""

    entries: list[LearnedPolicyEntry] = []
    used_labels: set[str] = set()
    for raw in values:
        text = str(raw).strip()
        if not text:
            continue
        if "=" in text:
            _, path_text = text.split("=", 1)
        else:
            path_text = text
        path = Path(path_text.strip())
        spec = load_policy_spec(path)
        label = _entry_label(text, used=used_labels)
        entries.append(
            LearnedPolicyEntry(
                label=label,
                policy_spec=path,
                policy_name=str(spec.get("policy_name")),
                adapter=str(spec.get("adapter")),
                summary={
                    "schema_version": spec.get("schema_version"),
                    "policy_name": spec.get("policy_name"),
                    "adapter": spec.get("adapter"),
                    "spec_path": str(path),
                    "artifact_path": spec.get("artifact_path"),
                    "deterministic": bool(spec.get("deterministic", True)),
                    "clip": bool(spec.get("clip", True)),
                    "description": spec.get("description"),
                },
            )
        )
    if not entries:
        raise ValueError("learned holdout requires at least one policy spec")
    return entries


def _scenario_paths(out_dir: Path, scenarios: list[str]) -> dict[str, Path]:
    generated = materialize_official_suite("official_3d_stress", out_dir, overwrite=True)
    by_id = {path.stem: path for path in generated["scenario_paths"]}
    missing = sorted(set(scenarios) - set(by_id))
    if missing:
        raise ValueError(f"unknown official_3d_stress holdout scenario(s): {','.join(missing)}")
    return {scenario_id: by_id[scenario_id] for scenario_id in scenarios}


def _agent_methods_for(scenario_id: str, method: str, n_agents: int) -> list[str] | None:
    if scenario_id != "noncooperative_intruder_3d_hard":
        return None
    if method == "learned_policy_spec":
        return ["baseline_goal"] + ["learned_policy_spec"] * max(0, int(n_agents) - 1)
    return ["baseline_goal"] + [str(method)] * max(0, int(n_agents) - 1)


def _run_specs(
    *,
    scenario_paths: dict[str, Path],
    scenario_ids: list[str],
    method: str,
    run_dir: Path,
    n_agents: int,
    seeds: list[int],
    comm_profiles: list[str],
    max_runs: int | None,
    policy_spec: str | Path | None = None,
) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for scenario_id in scenario_ids:
        for comm in comm_profiles:
            for seed in seeds:
                specs.append(
                    RunSpec(
                        scenario_path=str(scenario_paths[scenario_id]),
                        method=str(method),
                        n_agents=int(n_agents),
                        seed=int(seed),
                        comm_profile=str(comm),
                        out_dir=str(run_dir),
                        save_trace=False,
                        agent_methods=_agent_methods_for(scenario_id, str(method), int(n_agents)),
                        policy_spec=None if policy_spec is None else str(policy_spec),
                    )
                )
    if max_runs is None:
        return specs
    return specs[: max(0, int(max_runs))]


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
        raise LearnedHoldoutRunTimeout("episode exceeded run timeout before launch")
    if not _hard_timeout_supported():
        yield
        return

    def _raise_timeout(signum, frame):
        _ = signum, frame
        raise LearnedHoldoutRunTimeout(f"episode exceeded run timeout of {timeout:.3f}s")

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
    except LearnedHoldoutRunTimeout:
        elapsed = time.perf_counter() - started
        return _timeout_row(spec, elapsed_s=elapsed, timeout_s=run_timeout_s), True


def _run_specs_to_csv_checked(
    specs: list[RunSpec],
    run_dir: Path,
    *,
    resume: bool = False,
    run_timeout_s: float | None = None,
    max_timeouts_per_entry: int | None = None,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    results_csv = run_dir / "results.csv"
    if not specs:
        if not results_csv.exists():
            write_result_schema_manifest(run_dir)
            with (run_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=RESULT_FIELDS).writeheader()
            with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=SUMMARY_FIELDS).writeheader()
        return {
            "selected_run_count": 0,
            "new_run_count": 0,
            "skipped_existing_count": 0,
            "timeout_run_count": 0,
        }
    existing_rows = _read_csv_rows(results_csv) if resume else []
    completed_keys = {_row_key(row) for row in existing_rows} if resume else set()
    existing_timeout_count = sum(_row_timeout_count(row) for row in existing_rows)
    if max_timeouts_per_entry is not None and existing_timeout_count >= int(max_timeouts_per_entry):
        write_summary(run_dir)
        selected_keys = {_spec_key(spec) for spec in specs}
        return {
            "selected_run_count": int(len(specs)),
            "new_run_count": 0,
            "skipped_existing_count": int(len(selected_keys & completed_keys)),
            "timeout_run_count": int(existing_timeout_count),
            "new_timeout_run_count": 0,
            "stopped_by_timeout_budget": True,
        }
    skipped_existing = 0
    newly_run = 0
    newly_timed_out = 0
    stopped_by_timeout_budget = False
    for spec in specs:
        if _spec_key(spec) in completed_keys:
            skipped_existing += 1
            continue
        row, timed_out = _run_episode_checked(spec, run_timeout_s=run_timeout_s)
        append_result(run_dir, row)
        completed_keys.add(_spec_key(spec))
        newly_run += 1
        newly_timed_out += int(timed_out)
        if max_timeouts_per_entry is not None and existing_timeout_count + newly_timed_out >= int(max_timeouts_per_entry):
            stopped_by_timeout_budget = True
            break
    write_summary(run_dir)
    timeout_count = sum(
        int(_finite_float(row.get("planner_timeout_count")) or 0)
        for row in _read_csv_rows(results_csv)
    )
    return {
        "selected_run_count": int(len(specs)),
        "new_run_count": int(newly_run),
        "skipped_existing_count": int(skipped_existing),
        "timeout_run_count": int(timeout_count),
        "new_timeout_run_count": int(newly_timed_out),
        "stopped_by_timeout_budget": bool(stopped_by_timeout_budget),
    }


def _summary_from_run_dir(
    *,
    kind: str,
    label: str,
    method: str,
    run_dir: Path,
    out_dir: Path,
    policy_name: str | None = None,
    adapter: str | None = None,
    policy_spec: str | Path | None = None,
) -> dict[str, Any]:
    results_csv = run_dir / "results.csv"
    summary_csv = run_dir / "summary.csv"
    result_rows = _read_csv_rows(results_csv)
    summary_rows = _read_csv_rows(summary_csv)
    scored_rows: list[dict[str, Any]] = []
    scores: list[float] = []
    for row in summary_rows:
        projected = dict(row)
        score = score_v0(projected)
        projected["score_v0"] = score
        scored_rows.append(projected)
        if score is not None:
            scores.append(float(score))

    collision_episodes = sum(int(_finite_float(row.get("collision_episode")) or 0) for row in result_rows)
    near_miss_episodes = sum(int(_finite_float(row.get("near_miss_episode")) or 0) for row in result_rows)
    planner_timeout_count = sum(_row_timeout_count(row) for row in result_rows)
    planner_error_count = sum(_row_error_count(row) for row in result_rows)
    planner_fallback_count = sum(int(_finite_float(row.get("planner_fallback_count")) or 0) for row in result_rows)
    completed_rows = [
        row
        for row in result_rows
        if _row_timeout_count(row) <= 0 and _row_error_count(row) <= 0
    ]
    collision_rates = _finite_values(summary_rows, "collision_episode_rate")
    completion_rates = _finite_values(result_rows, "completion_rate")
    return {
        "kind": str(kind),
        "label": str(label),
        "method": str(method),
        "policy_name": None if policy_name is None else str(policy_name),
        "adapter": None if adapter is None else str(adapter),
        "policy_spec": None if policy_spec is None else str(policy_spec),
        "run_count": int(len(result_rows)),
        "summary_row_count": int(len(summary_rows)),
        "scenario_count": len({str(row.get("scenario", "")) for row in result_rows if str(row.get("scenario", ""))}),
        "collision_episodes": int(collision_episodes),
        "near_miss_episodes": int(near_miss_episodes),
        "collision_episode_rate_mean": _mean_or_none(collision_rates),
        "completion_rate_mean": _mean_or_none(completion_rates),
        "min_sep_min_row_m": _min_or_none(_finite_values(result_rows, "min_sep_min_m")),
        "min_sep_p05_row_min_m": _min_or_none(_finite_values(result_rows, "min_sep_p05_m")),
        "score_v0_mean": _mean_or_none(scores),
        "score_v0_worst": _max_or_none(scores),
        "planner_ms_p95_max": _max_or_none(_finite_values(completed_rows, "planner_ms_per_tick_per_agent_p95")),
        "planner_timeout_count": int(planner_timeout_count),
        "planner_error_count": int(planner_error_count),
        "planner_fallback_count": int(planner_fallback_count),
        "results_csv": _rel(results_csv, out_dir),
        "summary_csv": _rel(summary_csv, out_dir),
        "scored_summary_rows": scored_rows,
    }


def _delta(learned: dict[str, Any], reference: dict[str, Any], field: str) -> float | None:
    learned_value = _finite_float(learned.get(field))
    reference_value = _finite_float(reference.get(field))
    if learned_value is None or reference_value is None:
        return None
    return round(float(learned_value - reference_value), 6)


def _pairwise_deltas(
    *,
    learned_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for learned in learned_rows:
        for reference in reference_rows:
            rows.append(
                {
                    "learned_label": learned["label"],
                    "reference_label": reference["label"],
                    "learned_policy_name": learned.get("policy_name"),
                    "reference_method": reference.get("method"),
                    "run_count_delta": _delta(learned, reference, "run_count"),
                    "collision_episodes_delta": _delta(learned, reference, "collision_episodes"),
                    "near_miss_episodes_delta": _delta(learned, reference, "near_miss_episodes"),
                    "completion_rate_mean_delta": _delta(learned, reference, "completion_rate_mean"),
                    "min_sep_min_row_m_delta": _delta(learned, reference, "min_sep_min_row_m"),
                    "min_sep_p05_row_min_m_delta": _delta(learned, reference, "min_sep_p05_row_min_m"),
                    "score_v0_mean_delta": _delta(learned, reference, "score_v0_mean"),
                    "planner_ms_p95_max_delta": _delta(learned, reference, "planner_ms_p95_max"),
                    "planner_timeout_count_delta": _delta(learned, reference, "planner_timeout_count"),
                    "planner_error_count_delta": _delta(learned, reference, "planner_error_count"),
                    "planner_fallback_count_delta": _delta(learned, reference, "planner_fallback_count"),
                }
            )
    return rows


def _checks(
    *,
    expected_runs: int,
    learned_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = [*learned_rows, *reference_rows]
    return [
        {
            "name": "learned_policy_specs_present",
            "ok": bool(learned_rows),
            "details": {"learned_policy_count": len(learned_rows)},
        },
        {
            "name": "reference_methods_present",
            "ok": bool(reference_rows),
            "details": {"reference_method_count": len(reference_rows)},
        },
        {
            "name": "all_requested_runs_completed",
            "ok": all(int(row.get("run_count") or 0) == int(expected_runs) for row in rows),
            "details": {
                "expected_runs_per_entry": int(expected_runs),
                "run_counts": {str(row["label"]): int(row.get("run_count") or 0) for row in rows},
            },
        },
        {
            "name": "all_entries_scored",
            "ok": all(_finite_float(row.get("score_v0_mean")) is not None for row in rows),
            "details": {"labels": [str(row["label"]) for row in rows]},
        },
        {
            "name": "no_planner_timeouts",
            "ok": all(int(row.get("planner_timeout_count") or 0) == 0 for row in rows),
            "details": {str(row["label"]): int(row.get("planner_timeout_count") or 0) for row in rows},
        },
        {
            "name": "learned_metrics_finite",
            "ok": all(
                _finite_float(row.get("completion_rate_mean")) is not None
                and _finite_float(row.get("min_sep_min_row_m")) is not None
                for row in learned_rows
            ),
            "details": {"labels": [str(row["label"]) for row in learned_rows]},
        },
    ]


def run_learned_holdout_eval(
    *,
    out_dir: str | Path,
    policy_specs: tuple[str, ...] | list[str],
    reference_methods: tuple[str, ...] | list[str] | None = None,
    scenarios: tuple[str, ...] | list[str] | None = None,
    seeds: tuple[int, ...] | list[int] | None = None,
    comm_profiles: tuple[str, ...] | list[str] | None = None,
    n_agents: int = 6,
    max_runs: int | None = None,
    resume: bool = False,
    run_timeout_s: float | None = None,
    max_timeouts_per_entry: int | None = None,
    require_no_collision: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Evaluate learned policy specs against optimizer references on broad 3D holdout rows."""

    out = Path(out_dir)
    if out.exists():
        if bool(overwrite):
            shutil.rmtree(out)
        elif any(out.iterdir()) and not bool(resume):
            raise RuntimeError(f"learned holdout output directory already exists and is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    learned_entries = parse_learned_policy_spec_entries(policy_specs)
    references = [str(method).strip() for method in (reference_methods or DEFAULT_LEARNED_HOLDOUT_REFERENCE_METHODS) if str(method).strip()]
    if not references:
        raise ValueError("learned holdout requires at least one reference method")
    scenario_ids = [str(value).strip() for value in (scenarios or CLOSED_LOOP_BROAD_3D_HOLDOUT_SCENARIOS) if str(value).strip()]
    seed_values = [int(seed) for seed in (seeds if seeds is not None else [0, 1, 2])]
    comm_values = [str(comm).strip() for comm in (comm_profiles if comm_profiles is not None else ["ideal_50hz", "degraded_20hz"]) if str(comm).strip()]
    if int(n_agents) < 2:
        raise ValueError("learned holdout n_agents must be >= 2")
    if max_timeouts_per_entry is not None and int(max_timeouts_per_entry) < 1:
        raise ValueError("max_timeouts_per_entry must be >= 1 when provided")
    if not scenario_ids:
        raise ValueError("learned holdout scenarios must not be empty")
    if not seed_values:
        raise ValueError("learned holdout seeds must not be empty")
    if not comm_values:
        raise ValueError("learned holdout comm_profiles must not be empty")

    scenario_dir = out / "_holdout_scenarios" / "official_3d_stress"
    paths = _scenario_paths(scenario_dir, scenario_ids)
    expected_specs = _run_specs(
        scenario_paths=paths,
        scenario_ids=scenario_ids,
        method="learned_policy_spec",
        run_dir=out / "_expected",
        n_agents=int(n_agents),
        seeds=seed_values,
        comm_profiles=comm_values,
        max_runs=max_runs,
    )
    expected_runs = len(expected_specs)

    learned_rows: list[dict[str, Any]] = []
    execution: dict[str, Any] = {}
    for entry in learned_entries:
        run_dir = out / "learned" / entry.label
        specs = _run_specs(
            scenario_paths=paths,
            scenario_ids=scenario_ids,
            method="learned_policy_spec",
            run_dir=run_dir,
            n_agents=int(n_agents),
            seeds=seed_values,
            comm_profiles=comm_values,
            max_runs=max_runs,
            policy_spec=entry.policy_spec,
        )
        execution[entry.label] = _run_specs_to_csv_checked(
            specs,
            run_dir,
            resume=bool(resume),
            run_timeout_s=run_timeout_s,
            max_timeouts_per_entry=max_timeouts_per_entry,
        )
        report_path = run_dir / "baseline_report.json"
        baseline_report = build_baseline_report(
            summary_csv=run_dir / "summary.csv",
            results_csv=run_dir / "results.csv",
            suite="learned_holdout_eval",
            generated_by="python -m microbench.cli learned-holdout-eval",
        )
        report_path.write_text(json.dumps(baseline_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        learned_rows.append(
            _summary_from_run_dir(
                kind="learned_policy",
                label=entry.label,
                method="learned_policy_spec",
                run_dir=run_dir,
                out_dir=out,
                policy_name=entry.policy_name,
                adapter=entry.adapter,
                policy_spec=entry.policy_spec,
            )
        )

    reference_rows: list[dict[str, Any]] = []
    for method in references:
        run_dir = out / "references" / method
        specs = _run_specs(
            scenario_paths=paths,
            scenario_ids=scenario_ids,
            method=method,
            run_dir=run_dir,
            n_agents=int(n_agents),
            seeds=seed_values,
            comm_profiles=comm_values,
            max_runs=max_runs,
        )
        execution[method] = _run_specs_to_csv_checked(
            specs,
            run_dir,
            resume=bool(resume),
            run_timeout_s=run_timeout_s,
            max_timeouts_per_entry=max_timeouts_per_entry,
        )
        report_path = run_dir / "baseline_report.json"
        baseline_report = build_baseline_report(
            summary_csv=run_dir / "summary.csv",
            results_csv=run_dir / "results.csv",
            suite="learned_holdout_eval",
            generated_by="python -m microbench.cli learned-holdout-eval",
        )
        report_path.write_text(json.dumps(baseline_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        reference_rows.append(
            _summary_from_run_dir(
                kind="reference_optimizer",
                label=method,
                method=method,
                run_dir=run_dir,
                out_dir=out,
            )
        )

    all_rows = [*learned_rows, *reference_rows]
    all_rows.sort(
        key=lambda row: (
            float("inf") if _finite_float(row.get("score_v0_mean")) is None else float(row["score_v0_mean"]),
            str(row["kind"]),
            str(row["label"]),
        )
    )
    for rank, row in enumerate(all_rows, start=1):
        row["rank"] = rank
    deltas = _pairwise_deltas(learned_rows=learned_rows, reference_rows=reference_rows)
    table_csv = _write_csv(out / "learned_holdout_table.csv", all_rows, ("rank", *LEARNED_HOLDOUT_TABLE_FIELDS))
    delta_csv = _write_csv(out / "learned_holdout_deltas.csv", deltas, LEARNED_HOLDOUT_DELTA_FIELDS)
    checks = _checks(expected_runs=expected_runs, learned_rows=learned_rows, reference_rows=reference_rows)
    if require_no_collision:
        checks.append(
            {
                "name": "learned_policies_collision_free",
                "ok": all(int(row.get("collision_episodes") or 0) == 0 for row in learned_rows),
                "details": {str(row["label"]): int(row.get("collision_episodes") or 0) for row in learned_rows},
                "severity": "behavior",
            }
        )

    report = {
        "schema_version": LEARNED_HOLDOUT_EVAL_SCHEMA_VERSION,
        "ok": all(bool(check["ok"]) for check in checks),
        "out_dir": str(out),
        "suite": "official_3d_stress",
        "profile": "broad_3d_holdout",
        "scenarios": scenario_ids,
        "seeds": seed_values,
        "comm_profiles": comm_values,
        "n_agents": int(n_agents),
        "max_runs": None if max_runs is None else int(max_runs),
        "resume": bool(resume),
        "run_timeout_s": None if run_timeout_s is None else float(run_timeout_s),
        "max_timeouts_per_entry": None if max_timeouts_per_entry is None else int(max_timeouts_per_entry),
        "run_timeout_supported": _hard_timeout_supported(),
        "expected_runs_per_entry": int(expected_runs),
        "execution": execution,
        "learned_policies": [entry.summary for entry in learned_entries],
        "reference_methods": references,
        "rows": all_rows,
        "learned_rows": learned_rows,
        "reference_rows": reference_rows,
        "pairwise_deltas": deltas,
        "table_csv": str(table_csv),
        "delta_csv": str(delta_csv),
        "checks": checks,
        "score_note": "score_v0 follows docs/LEADERBOARD.md; lower is better. Pairwise deltas are learned minus reference, so negative score deltas are better.",
        "promotion_note": "This evaluator is a holdout comparison aid, not a release or promotion gate by itself.",
    }
    report_path = out / "learned_holdout_eval.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
