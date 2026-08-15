from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any


HIGH_VOLUME_LEADERBOARD_SCHEMA_VERSION = "0.1"
DEFAULT_LATENCY_BUDGET_MS = 20.0
DEFAULT_COMPONENT_WEIGHTS = {
    "safety": 0.30,
    "progress": 0.40,
    "runtime": 0.20,
    "robustness": 0.10,
}


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    p = Path(path)
    with p.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _mean(values: list[float]) -> float | None:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return None
    return round(sum(clean) / len(clean), 6)


def _max(values: list[float]) -> float | None:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return round(max(clean), 6) if clean else None


def _canonical_scenario(name: str) -> str:
    out = re.sub(r"^\d+_", "", str(name))
    for suffix in ("_scale_dense", "_scale"):
        if out.endswith(suffix):
            out = out[: -len(suffix)]
    return out


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, float(value)))


def _progress_raw(row: dict[str, Any]) -> float:
    completion = _num(row.get("completion_rate_mean")) or 0.0
    mean_progress = _num(row.get("goal_progress_fraction_mean"))
    p05_progress = _num(row.get("goal_progress_fraction_p05_mean"))
    if mean_progress is None:
        mean_progress = completion
    if p05_progress is None:
        p05_progress = min(mean_progress, completion)
    return _clamp(0.65 * mean_progress + 0.25 * completion + 0.10 * p05_progress, 0.0, 1.0)


def _row_quality(row: dict[str, Any]) -> tuple[int, int, float, int]:
    has_progress = int(_num(row.get("goal_progress_fraction_mean")) is not None)
    completed = int(_num(row.get("completed_run_count")) or 0)
    timeout_rate = _num(row.get("timeout_rate")) or 0.0
    return (has_progress, completed, -timeout_rate, int(row.get("source_index") or 0))


def _project_row(row: dict[str, str], *, source_path: Path, source_index: int, row_index: int) -> dict[str, Any]:
    scenario = str(row.get("scenario", ""))
    projected: dict[str, Any] = {
        "source": str(source_path),
        "source_index": int(source_index),
        "row_index": int(row_index),
        "scenario": scenario,
        "canonical_scenario": _canonical_scenario(scenario),
        "method": str(row.get("method", "")),
        "comm_profile": str(row.get("comm_profile", "")),
        "N": int(_num(row.get("N")) or 0),
        "status": str(row.get("status", "")),
    }
    metric_fields = (
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
    )
    for field in metric_fields:
        projected[field] = _num(row.get(field))
    projected["goal_progress_raw"] = round(_progress_raw(projected), 6)
    return projected


def _dedupe_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    duplicates: list[dict[str, Any]] = []
    for row in rows:
        key = (
            str(row["canonical_scenario"]),
            str(row["method"]),
            str(row["comm_profile"]),
            int(row["N"]),
        )
        existing = selected.get(key)
        if existing is None:
            selected[key] = row
            continue
        if _row_quality(row) >= _row_quality(existing):
            duplicates.append(existing)
            selected[key] = row
        else:
            duplicates.append(row)
    out = list(selected.values())
    out.sort(key=lambda r: (str(r["canonical_scenario"]), str(r["method"]), str(r["comm_profile"]), int(r["N"])))
    return out, duplicates


def _safety_score(row: dict[str, Any]) -> float:
    collision_rate = _num(row.get("collision_episode_rate")) or 0.0
    unique_pairs = _num(row.get("unique_collision_pairs_mean")) or 0.0
    pair_ticks = _num(row.get("collision_pair_ticks_mean")) or 0.0
    min_sep = _num(row.get("min_sep_min_worst_m")) or 0.0
    penalty = 100.0 * collision_rate + 25.0 * unique_pairs + 0.5 * pair_ticks + 20.0 * max(0.0, -min_sep)
    return round(_clamp(100.0 - penalty), 6)


def _runtime_score(row: dict[str, Any], *, latency_budget_ms: float) -> float:
    p95 = _num(row.get("planner_ms_p95_max"))
    timeout_rate = _num(row.get("timeout_rate")) or 0.0
    if p95 is None:
        return 0.0
    latency_factor = _clamp(1.0 - (float(p95) / float(latency_budget_ms)), 0.0, 1.0)
    return round(_clamp(100.0 * latency_factor * (1.0 - timeout_rate)), 6)


def _robustness_score(row: dict[str, Any]) -> float:
    timeout_rate = _num(row.get("timeout_rate")) or 0.0
    guardrails = _num(row.get("guardrail_count_mean")) or 0.0
    run_count = _num(row.get("run_count")) or 0.0
    completed = _num(row.get("completed_run_count")) or 0.0
    completed_fraction = completed / run_count if run_count > 0.0 else 0.0
    penalty = 100.0 * timeout_rate + 15.0 * guardrails + 50.0 * (1.0 - completed_fraction)
    return round(_clamp(100.0 - penalty), 6)


def _score_rows(rows: list[dict[str, Any]], *, latency_budget_ms: float) -> list[dict[str, Any]]:
    by_cell: dict[tuple[str, str, int], float] = {}
    for row in rows:
        key = (str(row["canonical_scenario"]), str(row["comm_profile"]), int(row["N"]))
        by_cell[key] = max(by_cell.get(key, 0.0), float(row["goal_progress_raw"]))

    scored: list[dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        key = (str(row["canonical_scenario"]), str(row["comm_profile"]), int(row["N"]))
        best_progress = by_cell.get(key, 0.0)
        raw_progress = float(out["goal_progress_raw"])
        relative_progress = raw_progress / best_progress if best_progress > 0.0 else 0.0
        progress_score = 100.0 * (0.70 * relative_progress + 0.30 * raw_progress)
        axis_scores = {
            "safety": _safety_score(out),
            "progress": round(_clamp(progress_score), 6),
            "runtime": _runtime_score(out, latency_budget_ms=latency_budget_ms),
            "robustness": _robustness_score(out),
        }
        out["axis_scores"] = axis_scores
        scored.append(out)
    return scored


def _classification(summary: dict[str, Any]) -> str:
    if (summary.get("safety_score") or 0.0) < 100.0:
        return "safety_review"
    if (summary.get("timeout_rate_mean") or 0.0) > 0.0 or (summary.get("guardrail_count_mean") or 0.0) > 0.0:
        return "robustness_review"
    if (summary.get("progress_score") or 0.0) >= 70.0:
        return "throughput_leader"
    if (summary.get("runtime_score") or 0.0) >= 80.0:
        return "runtime_leader"
    return "safe_conservative"


def _method_summaries(rows: list[dict[str, Any]], *, weights: dict[str, float]) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)

    summaries: list[dict[str, Any]] = []
    for method, items in sorted(by_method.items()):
        axis = {
            name: _mean([float(row["axis_scores"][name]) for row in items]) or 0.0
            for name in ("safety", "progress", "runtime", "robustness")
        }
        overall = sum(float(weights[name]) * float(axis[name]) for name in weights)
        summary = {
            "method": method,
            "scale_cells": len(items),
            "scenario_count": len({str(row["canonical_scenario"]) for row in items}),
            "max_N": max(int(row["N"]) for row in items),
            "overall_score": round(overall, 6),
            "safety_score": round(axis["safety"], 6),
            "progress_score": round(axis["progress"], 6),
            "runtime_score": round(axis["runtime"], 6),
            "robustness_score": round(axis["robustness"], 6),
            "collision_episode_rate_mean": _mean([
                float(row.get("collision_episode_rate") or 0.0) for row in items
            ]),
            "completion_rate_mean": _mean([
                float(row.get("completion_rate_mean") or 0.0) for row in items
            ]),
            "goal_progress_raw_mean": _mean([float(row["goal_progress_raw"]) for row in items]),
            "planner_ms_p95_max": _max([
                float(row.get("planner_ms_p95_max") or 0.0) for row in items
            ]),
            "guardrail_count_mean": _mean([
                float(row.get("guardrail_count_mean") or 0.0) for row in items
            ]),
            "timeout_rate_mean": _mean([
                float(row.get("timeout_rate") or 0.0) for row in items
            ]),
        }
        summary["classification"] = _classification(summary)
        summaries.append(summary)

    summaries.sort(
        key=lambda row: (
            -float(row["overall_score"]),
            -float(row["safety_score"]),
            -float(row["progress_score"]),
            -float(row["runtime_score"]),
            str(row["method"]),
        )
    )
    for rank, row in enumerate(summaries, start=1):
        row["overall_rank"] = rank
    return summaries


def _axis_rankings(summaries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    rankings: dict[str, list[dict[str, Any]]] = {}
    for axis in ("safety", "progress", "runtime", "robustness"):
        score_key = f"{axis}_score"
        rows = [
            {
                "rank": 0,
                "method": str(summary["method"]),
                "score": summary[score_key],
                "overall_rank": summary["overall_rank"],
            }
            for summary in summaries
        ]
        rows.sort(key=lambda row: (-float(row["score"]), int(row["overall_rank"]), str(row["method"])))
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        rankings[axis] = rows
    return rankings


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def build_high_volume_leaderboard(
    *,
    scale_summaries: list[str | Path] | tuple[str | Path, ...],
    latency_budget_ms: float = DEFAULT_LATENCY_BUDGET_MS,
    weights: dict[str, float] | None = None,
    generated_by: str | None = None,
) -> dict[str, Any]:
    if not scale_summaries:
        raise ValueError("at least one scale_summary.csv path is required")
    clean_weights = dict(DEFAULT_COMPONENT_WEIGHTS if weights is None else weights)
    weight_sum = sum(float(v) for v in clean_weights.values())
    if weight_sum <= 0.0:
        raise ValueError("component weights must sum to a positive value")
    clean_weights = {key: float(value) / weight_sum for key, value in clean_weights.items()}

    raw_rows: list[dict[str, Any]] = []
    source_paths = [Path(path) for path in scale_summaries]
    for source_index, path in enumerate(source_paths):
        for row_index, row in enumerate(_read_csv(path)):
            raw_rows.append(_project_row(row, source_path=path, source_index=source_index, row_index=row_index))
    if not raw_rows:
        raise ValueError("no scale summary rows found")

    selected_rows, duplicate_rows = _dedupe_rows(raw_rows)
    scored_rows = _score_rows(selected_rows, latency_budget_ms=float(latency_budget_ms))
    summaries = _method_summaries(scored_rows, weights=clean_weights)
    axis_rankings = _axis_rankings(summaries)

    report = {
        "schema_version": HIGH_VOLUME_LEADERBOARD_SCHEMA_VERSION,
        "generated_by": generated_by,
        "scale_summaries": [str(path) for path in source_paths],
        "latency_budget_ms": float(latency_budget_ms),
        "component_weights": clean_weights,
        "method_count": len(summaries),
        "scale_cell_count": len(scored_rows),
        "duplicate_row_count": len(duplicate_rows),
        "duplicate_note": (
            "Rows with the same canonical scenario/method/comm/N are deduplicated by preferring rows "
            "with mission-progress fields, more completed runs, fewer timeouts, and later input order."
        ),
        "score_note": (
            "Higher is better. Overall score is a weighted composite of safety, scenario-relative mission "
            "progress, runtime under the latency budget, and timeout/guardrail robustness. Publish component "
            "scores with the overall rank."
        ),
        "rows": scored_rows,
        "method_summaries": summaries,
        "overall_ranking": summaries,
        "axis_rankings": axis_rankings,
    }
    return _json_safe(report)


def _write_method_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    fields = [
        "overall_rank",
        "method",
        "overall_score",
        "safety_score",
        "progress_score",
        "runtime_score",
        "robustness_score",
        "classification",
        "scale_cells",
        "scenario_count",
        "max_N",
        "collision_episode_rate_mean",
        "completion_rate_mean",
        "goal_progress_raw_mean",
        "planner_ms_p95_max",
        "guardrail_count_mean",
        "timeout_rate_mean",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            writer.writerow({field: row.get(field) for field in fields})


def write_high_volume_leaderboard(
    *,
    scale_summaries: list[str | Path] | tuple[str | Path, ...],
    out: str | Path,
    latency_budget_ms: float = DEFAULT_LATENCY_BUDGET_MS,
    weights: dict[str, float] | None = None,
    generated_by: str | None = None,
) -> dict[str, Any]:
    report = build_high_volume_leaderboard(
        scale_summaries=scale_summaries,
        latency_budget_ms=latency_budget_ms,
        weights=weights,
        generated_by=generated_by,
    )
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_path.with_suffix(".csv")
    report["leaderboard_csv"] = str(csv_path)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_method_csv(csv_path, list(report["method_summaries"]))
    return report
