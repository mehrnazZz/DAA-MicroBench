from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


BASELINE_REPORT_SCHEMA_VERSION = "0.1"

REPORT_METRICS = (
    "collision_episode_rate",
    "collisions_mean",
    "unique_collision_pairs_mean",
    "min_sep_min_mean",
    "min_sep_p05_mean",
    "completion_rate_mean",
    "mean_time_to_goal_mean",
    "deadlock_time_pct_mean",
    "planner_ms_p95",
    "planner_timeout_count_mean",
    "planner_error_count_mean",
    "planner_fallback_count_mean",
)


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(value: Any, *, digits: int = 6) -> float | int | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    rounded = round(out, digits)
    if abs(rounded - int(rounded)) < 1e-12:
        return int(rounded)
    return rounded


def _str_key(row: dict[str, Any], key: str) -> str:
    return str(row.get(key, ""))


def _int_key(row: dict[str, Any], key: str) -> int | None:
    value = _num(row.get(key), digits=0)
    return int(value) if value is not None else None


def _row_projection(row: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "method": _str_key(row, "method"),
        "scenario": _str_key(row, "scenario"),
        "comm_profile": _str_key(row, "comm_profile"),
        "N": _int_key(row, "N"),
        "episodes": _int_key(row, "episodes"),
    }
    for metric in REPORT_METRICS:
        out[metric] = _num(row.get(metric))
    return out


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _max(values: list[float]) -> float | None:
    if not values:
        return None
    return round(max(values), 6)


def _metric_values(rows: list[dict[str, Any]], metric: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        value = row.get(metric)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out.append(float(value))
    return out


def _method_projection(method: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "method": method,
        "scenario_count": len({str(row.get("scenario")) for row in rows}),
        "episodes": int(sum(int(row.get("episodes") or 0) for row in rows)),
        "collision_episode_rate_mean": _mean(_metric_values(rows, "collision_episode_rate")),
        "completion_rate_mean": _mean(_metric_values(rows, "completion_rate_mean")),
        "min_sep_min_worst": _min(_metric_values(rows, "min_sep_min_mean")),
        "planner_ms_p95_max": _max(_metric_values(rows, "planner_ms_p95")),
        "planner_timeout_count_mean": _mean(_metric_values(rows, "planner_timeout_count_mean")),
        "planner_error_count_mean": _mean(_metric_values(rows, "planner_error_count_mean")),
        "planner_fallback_count_mean": _mean(_metric_values(rows, "planner_fallback_count_mean")),
    }


def _min(values: list[float]) -> float | None:
    if not values:
        return None
    return round(min(values), 6)


def _result_run_count(results_csv: str | Path | None) -> int | None:
    if results_csv is None:
        return None
    return len(_read_csv(results_csv))


def build_baseline_report(
    *,
    summary_csv: str | Path,
    suite: str,
    results_csv: str | Path | None = None,
    generated_by: str | None = None,
) -> dict[str, Any]:
    rows = [_row_projection(row) for row in _read_csv(summary_csv)]
    rows.sort(key=lambda row: (str(row["method"]), str(row["scenario"]), str(row["comm_profile"]), int(row["N"] or 0)))

    methods = sorted({str(row["method"]) for row in rows})
    scenarios = sorted({str(row["scenario"]) for row in rows})
    comm_profiles = sorted({str(row["comm_profile"]) for row in rows})
    n_agents = sorted({int(row["N"]) for row in rows if row["N"] is not None})
    method_summaries = [_method_projection(method, [row for row in rows if row["method"] == method]) for method in methods]

    return {
        "schema_version": BASELINE_REPORT_SCHEMA_VERSION,
        "suite": suite,
        "generated_by": generated_by,
        "methods": methods,
        "scenarios": scenarios,
        "comm_profiles": comm_profiles,
        "n_agents": n_agents,
        "run_count": _result_run_count(results_csv),
        "metric_note": "Timing fields are machine-dependent; use broad bands and guardrail counts for regression checks.",
        "rows": rows,
        "method_summaries": method_summaries,
    }


def write_baseline_report(
    *,
    summary_csv: str | Path,
    suite: str,
    out: str | Path,
    results_csv: str | Path | None = None,
    generated_by: str | None = None,
) -> Path:
    report = build_baseline_report(
        summary_csv=summary_csv,
        results_csv=results_csv,
        suite=suite,
        generated_by=generated_by,
    )
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path
