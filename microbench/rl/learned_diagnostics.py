from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from microbench.rl.learned_leaderboard import LEARNED_POLICY_LEADERBOARD_SCHEMA_VERSION
from microbench.rl.learned_lineage import LEARNED_POLICY_LINEAGE_FIELDS, classify_learned_policy_lineage
from microbench.rl.submission_bundle import (
    LEARNED_SUBMISSION_BUNDLE_REVIEW_SCHEMA_VERSION,
    review_learned_policy_submission_bundle,
)


LEARNED_POLICY_DIAGNOSTICS_SCHEMA_VERSION = "0.1"

LEARNED_POLICY_DIAGNOSTIC_FIELDS = (
    "diagnostic_rank",
    "safety_rank",
    "mission_rank",
    "balanced_rank",
    "bundle",
    "method",
    "policy",
    *LEARNED_POLICY_LINEAGE_FIELDS,
    "suite",
    "diagnostic_label",
    "primary_failure",
    "next_action",
    "recommendation",
    "ok",
    "score_v0_mean",
    "safety_score",
    "mission_score",
    "balanced_score",
    "collision_episode_count",
    "rl_collision_ticks",
    "near_miss_signals",
    "min_sep_min_m",
    "min_sep_p05_min_m",
    "min_sep_min_row_m",
    "min_sep_p05_row_min_m",
    "min_sep_min_summary_mean_min_m",
    "min_sep_p05_summary_mean_min_m",
    "completion_rate_mean",
    "completion_rate_min",
    "final_goal_dist_mean_max_m",
    "worst_scenario",
    "worst_seed",
    "worst_completion_rate",
    "worst_final_goal_dist_mean_m",
    "worst_min_sep_m",
    "worst_rl_lane",
    "worst_rl_completion_rate",
    "worst_rl_final_min_sep_m",
    "planner_ms_p95_max",
    "planner_guardrail_count",
    "rl_ok",
    "rl_behavior_pass",
    "rl_completion_rate_mean",
    "rl_final_min_sep_min_m",
    "limitation_count",
    "limitations",
    "error",
)


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _to_int(value: Any) -> int:
    number = _to_float(value)
    return int(number) if number is not None else 0


def _round_or_none(value: Any, ndigits: int = 6) -> float | None:
    number = _to_float(value)
    return None if number is None else round(number, ndigits)


def _get(mapping: dict[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _read_csv_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _artifact_path(review: dict[str, Any], name: str) -> Path | None:
    artifacts = _get(review, "validation", "artifacts")
    if not isinstance(artifacts, dict):
        return None
    raw = artifacts.get(name)
    if not raw:
        return None
    return Path(str(raw))


def _numeric_values(rows: list[dict[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _to_float(row.get(field))
        if value is not None:
            values.append(value)
    return values


def _worst_planner_row(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {}

    def key(row: dict[str, str]) -> tuple[float, float, float, float, float, float]:
        completion = _to_float(row.get("completion_rate"))
        min_sep = _to_float(row.get("min_sep_min_m"))
        final_goal = _to_float(row.get("final_goal_dist_mean_m"))
        guardrails = (
            _to_int(row.get("planner_timeout_count"))
            + _to_int(row.get("planner_error_count"))
            + _to_int(row.get("planner_fallback_count"))
        )
        return (
            float(guardrails),
            float(_to_int(row.get("collision_episode"))),
            -float(completion if completion is not None else 1.0),
            float(final_goal if final_goal is not None else 0.0),
            -float(min_sep if min_sep is not None else 1e9),
            float(_to_int(row.get("near_misses"))),
        )

    worst = max(rows, key=key)
    completions = _numeric_values(rows, "completion_rate")
    final_goal_distances = _numeric_values(rows, "final_goal_dist_mean_m")
    min_separations = _numeric_values(rows, "min_sep_min_m")
    guardrail_count = sum(
        _to_int(row.get("planner_timeout_count"))
        + _to_int(row.get("planner_error_count"))
        + _to_int(row.get("planner_fallback_count"))
        for row in rows
    )
    return {
        "row_count": len(rows),
        "final_goal_dist_mean_max_m": _round_or_none(max(final_goal_distances) if final_goal_distances else None),
        "completion_rate_min": _round_or_none(min(completions) if completions else None),
        "min_sep_min_m": _round_or_none(min(min_separations) if min_separations else None),
        "planner_guardrail_count": int(guardrail_count),
        "worst_scenario": worst.get("scenario"),
        "worst_seed": _to_int(worst.get("seed")),
        "worst_completion_rate": _round_or_none(worst.get("completion_rate")),
        "worst_final_goal_dist_mean_m": _round_or_none(worst.get("final_goal_dist_mean_m")),
        "worst_min_sep_m": _round_or_none(worst.get("min_sep_min_m")),
    }


def _worst_rl_row(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {}

    def key(row: dict[str, str]) -> tuple[float, float, float, float]:
        completion = _to_float(row.get("completion_rate"))
        clearance = _to_float(row.get("final_min_sep_m"))
        return (
            float(_to_int(row.get("collision_ticks"))),
            float(_to_int(row.get("near_miss_ticks"))),
            -float(completion if completion is not None else 1.0),
            -float(clearance if clearance is not None else 1e9),
        )

    worst = max(rows, key=key)
    return {
        "worst_rl_lane": worst.get("lane_id") or worst.get("scenario"),
        "worst_rl_completion_rate": _round_or_none(worst.get("completion_rate")),
        "worst_rl_final_min_sep_m": _round_or_none(worst.get("final_min_sep_m")),
    }


def _primary_failure(
    *,
    ok: bool,
    error: str | None,
    collision_count: int,
    rl_collision_ticks: int,
    planner_guardrail_count: int,
    completion_rate_mean: float | None,
    completion_rate_min: float | None,
    min_sep_min_m: float | None,
    min_sep_p05_min_m: float | None,
    near_miss_signals: int,
    limitations: list[str],
) -> str:
    if error or not ok:
        return "artifact_or_contract"
    if planner_guardrail_count > 0:
        return "planner_guardrails"
    if collision_count > 0 or rl_collision_ticks > 0:
        return "collision"
    if completion_rate_mean is not None and completion_rate_mean < 0.95:
        return "incomplete_missions"
    if completion_rate_min is not None and completion_rate_min < 0.80:
        return "incomplete_missions"
    if (min_sep_min_m is not None and min_sep_min_m < 0.75) or (
        min_sep_p05_min_m is not None and min_sep_p05_min_m < 1.25
    ):
        return "low_clearance"
    if near_miss_signals > 0:
        return "near_miss"
    if "limited_planner_sweep" in limitations:
        return "limited_evidence"
    return "none"


def _diagnostic_label(
    *,
    primary_failure: str,
    completion_rate_mean: float | None,
    min_sep_min_m: float | None,
    min_sep_p05_min_m: float | None,
) -> str:
    if primary_failure == "artifact_or_contract":
        return "artifact_issue"
    if primary_failure in {"collision", "planner_guardrails"}:
        return "unsafe"
    if primary_failure == "incomplete_missions":
        if (min_sep_min_m is not None and min_sep_min_m >= 1.5) or (min_sep_p05_min_m is not None and min_sep_p05_min_m >= 2.0):
            return "safe_but_slow"
        return "needs_training"
    if primary_failure in {"low_clearance", "near_miss"}:
        return "fast_but_close" if completion_rate_mean is not None and completion_rate_mean >= 0.95 else "needs_training"
    if primary_failure == "limited_evidence":
        return "balanced_limited_evidence"
    return "balanced"


def _next_action(label: str, primary_failure: str, worst_scenario: Any, worst_rl_lane: Any) -> str:
    target = str(worst_scenario or worst_rl_lane or "the weakest validation lane")
    if label == "artifact_issue":
        return "Fix bundle artifacts, manifest disclosures, or RL validation gates before comparing policy behavior."
    if primary_failure == "planner_guardrails":
        return "Reduce planner errors/timeouts/fallbacks before using the learned-policy score."
    if label == "unsafe":
        return f"Add collision-focused training and stress replay around {target}; do not promote until collision ticks clear."
    if label == "safe_but_slow":
        return f"Increase horizon/progress weighting or add completion demonstrations around {target} while preserving clearance."
    if label == "fast_but_close":
        return f"Strengthen separation-margin shaping around {target} and rerun near-miss validation."
    if label == "needs_training":
        return f"Expand demonstrations and reward shaping around {target}; current behavior is neither clearly safe nor complete."
    if label == "balanced_limited_evidence":
        return "Rerun the uncapped official suite before treating this as leaderboard-ready."
    return "Promote to larger seeds, high-N stress, and held-out urban obstacle suites."


def _score_values(row: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    if row.get("error"):
        return None, None, None
    collision_count = int(row.get("collision_episode_count", 0) or 0)
    rl_collision_ticks = int(row.get("rl_collision_ticks", 0) or 0)
    near_miss_signals = int(row.get("near_miss_signals", 0) or 0)
    min_sep_min_m = _to_float(row.get("min_sep_min_m"))
    min_sep_p05_min_m = _to_float(row.get("min_sep_p05_min_m"))
    completion = _to_float(row.get("completion_rate_mean"))
    completion_min = _to_float(row.get("completion_rate_min"))
    final_goal = _to_float(row.get("final_goal_dist_mean_max_m")) or 0.0
    deadlock = _to_float(row.get("deadlock_time_pct_mean")) or 0.0
    guardrails = int(row.get("planner_guardrail_count", 0) or 0)
    score_v0 = _to_float(row.get("score_v0_mean"))

    safety_score = (
        100000.0 * collision_count
        + 1000.0 * rl_collision_ticks
        + 5.0 * near_miss_signals
        + 100.0 * max(0.0, 0.75 - (min_sep_min_m if min_sep_min_m is not None else 0.0))
        + 50.0 * max(0.0, 1.25 - (min_sep_p05_min_m if min_sep_p05_min_m is not None else 0.0))
        + 10.0 * guardrails
    )
    mission_score = (
        100.0 * max(0.0, 1.0 - (completion if completion is not None else 0.0))
        + 50.0 * max(0.0, 1.0 - (completion_min if completion_min is not None else 0.0))
        + final_goal
        + 2.0 * deadlock
    )
    balanced_score = score_v0 if score_v0 is not None else safety_score + mission_score
    if not row.get("ok"):
        safety_score += 1e6
        mission_score += 1e6
        balanced_score += 1e6
    return safety_score, mission_score, balanced_score


def _rank(rows: list[dict[str, Any]], *, score_field: str, rank_field: str) -> None:
    ranked = sorted(
        [row for row in rows if _to_float(row.get(score_field)) is not None],
        key=lambda row: (float(row[score_field]), str(row.get("policy") or ""), str(row.get("method") or "")),
    )
    for idx, row in enumerate(ranked, start=1):
        row[rank_field] = idx


def _review_to_diagnostic_row(*, bundle: str | Path, review: dict[str, Any], error: str | None = None) -> dict[str, Any]:
    safety = _get(review, "dimensions", "safety") or {}
    mission = _get(review, "dimensions", "mission") or {}
    compute = _get(review, "dimensions", "compute") or {}
    rl_validation = _get(review, "dimensions", "rl_validation_matrix") or {}
    score = review.get("score_v0") if isinstance(review.get("score_v0"), dict) else {}
    limitations = list(review.get("limitations", [])) if isinstance(review.get("limitations"), list) else []

    planner_rows = _read_csv_rows(_artifact_path(review, "planner_results"))
    rl_rows = _read_csv_rows(_artifact_path(review, "rl_validation_matrix_episodes"))
    planner_diag = _worst_planner_row(planner_rows)
    rl_diag = _worst_rl_row(rl_rows)
    lineage = classify_learned_policy_lineage(bundle=bundle, review=review)

    collision_count = int(safety.get("collision_episode_count", 0) or 0)
    rl_collision_ticks = int(rl_validation.get("collision_ticks", 0) or 0)
    near_miss_signals = int(rl_validation.get("near_miss_ticks", 0) or 0)
    near_miss_rate = _to_float(safety.get("near_miss_episode_rate_mean"))
    if near_miss_rate is not None and near_miss_rate > 0.0:
        near_miss_signals += 1
    min_sep_min_m = _to_float(safety.get("min_sep_min_m"))
    min_sep_p05_min_m = _to_float(safety.get("min_sep_p05_min_m"))
    completion_rate_mean = _to_float(mission.get("completion_rate_mean"))
    completion_rate_min = _to_float(mission.get("completion_rate_min"))
    planner_guardrail_count = (
        int(compute.get("planner_timeout_count", 0) or 0)
        + int(compute.get("planner_error_count", 0) or 0)
        + int(compute.get("planner_fallback_count", 0) or 0)
        + int(planner_diag.get("planner_guardrail_count", 0) or 0)
    )

    primary_failure = _primary_failure(
        ok=bool(review.get("ok")) and error is None,
        error=error,
        collision_count=collision_count,
        rl_collision_ticks=rl_collision_ticks,
        planner_guardrail_count=planner_guardrail_count,
        completion_rate_mean=completion_rate_mean,
        completion_rate_min=completion_rate_min,
        min_sep_min_m=min_sep_min_m,
        min_sep_p05_min_m=min_sep_p05_min_m,
        near_miss_signals=near_miss_signals,
        limitations=limitations,
    )
    label = _diagnostic_label(
        primary_failure=primary_failure,
        completion_rate_mean=completion_rate_mean,
        min_sep_min_m=min_sep_min_m,
        min_sep_p05_min_m=min_sep_p05_min_m,
    )

    row: dict[str, Any] = {
        "diagnostic_rank": None,
        "safety_rank": None,
        "mission_rank": None,
        "balanced_rank": None,
        "bundle": str(bundle),
        "method": review.get("method"),
        "policy": review.get("policy"),
        **{field: lineage.get(field) for field in LEARNED_POLICY_LINEAGE_FIELDS},
        "suite": review.get("suite"),
        "diagnostic_label": label,
        "primary_failure": primary_failure,
        "next_action": _next_action(label, primary_failure, planner_diag.get("worst_scenario"), rl_diag.get("worst_rl_lane")),
        "recommendation": review.get("recommendation") or "fix_artifacts",
        "ok": bool(review.get("ok")) and error is None,
        "score_v0_mean": _round_or_none(score.get("mean")),
        "collision_episode_count": collision_count,
        "rl_collision_ticks": rl_collision_ticks,
        "near_miss_signals": near_miss_signals,
        "min_sep_min_m": _round_or_none(min_sep_min_m),
        "min_sep_p05_min_m": _round_or_none(min_sep_p05_min_m),
        "min_sep_min_row_m": _round_or_none(safety.get("min_sep_min_row_m")),
        "min_sep_p05_row_min_m": _round_or_none(safety.get("min_sep_p05_row_min_m")),
        "min_sep_min_summary_mean_min_m": _round_or_none(safety.get("min_sep_min_summary_mean_min_m")),
        "min_sep_p05_summary_mean_min_m": _round_or_none(safety.get("min_sep_p05_summary_mean_min_m")),
        "completion_rate_mean": _round_or_none(completion_rate_mean),
        "completion_rate_min": _round_or_none(completion_rate_min),
        "final_goal_dist_mean_max_m": planner_diag.get("final_goal_dist_mean_max_m"),
        "worst_scenario": planner_diag.get("worst_scenario"),
        "worst_seed": planner_diag.get("worst_seed"),
        "worst_completion_rate": planner_diag.get("worst_completion_rate"),
        "worst_final_goal_dist_mean_m": planner_diag.get("worst_final_goal_dist_mean_m"),
        "worst_min_sep_m": planner_diag.get("worst_min_sep_m"),
        "worst_rl_lane": rl_diag.get("worst_rl_lane"),
        "worst_rl_completion_rate": rl_diag.get("worst_rl_completion_rate"),
        "worst_rl_final_min_sep_m": rl_diag.get("worst_rl_final_min_sep_m"),
        "planner_ms_p95_max": _round_or_none(compute.get("planner_ms_p95_max")),
        "planner_guardrail_count": planner_guardrail_count,
        "rl_ok": bool(rl_validation.get("ok")),
        "rl_behavior_pass": bool(rl_validation.get("behavior_pass")),
        "rl_completion_rate_mean": _round_or_none(rl_validation.get("completion_rate_mean")),
        "rl_final_min_sep_min_m": _round_or_none(rl_validation.get("final_min_sep_min_m")),
        "limitation_count": len(limitations),
        "limitations": ",".join(str(item) for item in limitations),
        "error": error,
    }
    safety_score, mission_score, balanced_score = _score_values(row)
    row["safety_score"] = _round_or_none(safety_score)
    row["mission_score"] = _round_or_none(mission_score)
    row["balanced_score"] = _round_or_none(balanced_score)
    return row


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get("diagnostic_label") or "unknown")
        label_counts[label] = label_counts.get(label, 0) + 1
    by_balanced = sorted(
        [row for row in rows if _to_float(row.get("balanced_score")) is not None],
        key=lambda row: float(row["balanced_score"]),
    )
    by_safety = sorted(
        [row for row in rows if _to_float(row.get("safety_score")) is not None],
        key=lambda row: float(row["safety_score"]),
    )
    by_mission = sorted(
        [row for row in rows if _to_float(row.get("mission_score")) is not None],
        key=lambda row: float(row["mission_score"]),
    )
    return {
        "diagnosable_count": sum(1 for row in rows if not row.get("error")),
        "label_counts": label_counts,
        "balanced_leader": by_balanced[0].get("policy") if by_balanced else None,
        "safety_leader": by_safety[0].get("policy") if by_safety else None,
        "mission_leader": by_mission[0].get("policy") if by_mission else None,
        "unsafe_count": sum(1 for row in rows if row.get("diagnostic_label") == "unsafe"),
        "safe_but_slow_count": sum(1 for row in rows if row.get("diagnostic_label") == "safe_but_slow"),
        "fast_but_close_count": sum(1 for row in rows if row.get("diagnostic_label") == "fast_but_close"),
        "needs_training_count": sum(1 for row in rows if row.get("diagnostic_label") == "needs_training"),
    }


def _findings(rows: list[dict[str, Any]]) -> list[str]:
    findings: list[str] = []
    for row in rows:
        policy = row.get("policy") or row.get("method") or row.get("bundle")
        label = row.get("diagnostic_label")
        if label in {"safe_but_slow", "fast_but_close", "needs_training", "unsafe"}:
            findings.append(f"{policy}: {label}; {row.get('next_action')}")
    if not findings:
        leader = min(
            [row for row in rows if _to_float(row.get("balanced_score")) is not None],
            key=lambda row: float(row["balanced_score"]),
            default=None,
        )
        if leader is not None:
            findings.append(f"{leader.get('policy')}: strongest balanced evidence in this bundle set.")
    return findings[:8]


def build_learned_policy_diagnostics(*, bundles: list[str | Path] | tuple[str | Path, ...]) -> dict[str, Any]:
    """Diagnose learned-policy bundle behavior beyond leaderboard ranking."""

    rows: list[dict[str, Any]] = []
    for bundle in bundles:
        try:
            review = review_learned_policy_submission_bundle(bundle=bundle)
            rows.append(_review_to_diagnostic_row(bundle=bundle, review=review))
        except Exception as exc:
            rows.append(
                _review_to_diagnostic_row(
                    bundle=bundle,
                    review={"recommendation": "fix_artifacts"},
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    _rank(rows, score_field="safety_score", rank_field="safety_rank")
    _rank(rows, score_field="mission_score", rank_field="mission_rank")
    _rank(rows, score_field="balanced_score", rank_field="balanced_rank")
    for row in rows:
        row["diagnostic_rank"] = row.get("balanced_rank")
    rows = sorted(
        rows,
        key=lambda row: (
            1 if row.get("balanced_rank") is None else 0,
            int(row.get("balanced_rank") or 10**9),
            str(row.get("policy") or ""),
        ),
    )

    failed_rows = [row for row in rows if row.get("error")]
    return {
        "schema_version": LEARNED_POLICY_DIAGNOSTICS_SCHEMA_VERSION,
        "leaderboard_schema_version": LEARNED_POLICY_LEADERBOARD_SCHEMA_VERSION,
        "review_schema_version": LEARNED_SUBMISSION_BUNDLE_REVIEW_SCHEMA_VERSION,
        "ok": not failed_rows,
        "bundle_count": len(rows),
        "failed_count": len(failed_rows),
        "summary": _summary(rows),
        "findings": _findings(rows),
        "score_note": (
            "Diagnostics are explanatory development evidence. Safety, mission, and balanced ranks are separate; "
            "balanced_score uses learned-bundle score_v0_mean when available, and lower is better."
        ),
        "label_note": (
            "Labels flag policy behavior patterns: unsafe, safe_but_slow, fast_but_close, "
            "needs_training, balanced_limited_evidence, balanced, or artifact_issue. "
            "Lineage labels separately identify BC-only, hard-lane BC, and closed-loop holdout-passed artifacts."
        ),
        "columns": list(LEARNED_POLICY_DIAGNOSTIC_FIELDS),
        "rows": rows,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(LEARNED_POLICY_DIAGNOSTIC_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in LEARNED_POLICY_DIAGNOSTIC_FIELDS})


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Learned Policy Diagnostics",
        "",
        report["score_note"],
        "",
        "## Summary",
        "",
    ]
    summary = report.get("summary", {})
    lines.extend(
        [
            f"- Bundles: {report.get('bundle_count')}",
            f"- Balanced leader: {summary.get('balanced_leader')}",
            f"- Safety leader: {summary.get('safety_leader')}",
            f"- Mission leader: {summary.get('mission_leader')}",
            f"- Labels: {json.dumps(summary.get('label_counts', {}), sort_keys=True)}",
            "",
        ]
    )
    if report.get("findings"):
        lines.extend(["## Findings", ""])
        lines.extend(f"- {finding}" for finding in report["findings"])
        lines.append("")
    lines.extend(
        [
            "## Rows",
            "",
            "| rank | policy | lineage | label | primary failure | completion | min sep | next action |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in report.get("rows", []):
        lines.append(
            "| {rank} | {policy} | {lineage} | {label} | {failure} | {completion} | {min_sep} | {next_action} |".format(
                rank=row.get("diagnostic_rank"),
                policy=row.get("policy"),
                lineage=row.get("lineage_label"),
                label=row.get("diagnostic_label"),
                failure=row.get("primary_failure"),
                completion=row.get("completion_rate_mean"),
                min_sep=row.get("min_sep_min_m"),
                next_action=str(row.get("next_action", "")).replace("|", "/"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_learned_policy_diagnostics(
    *,
    bundles: list[str | Path] | tuple[str | Path, ...],
    out: str | Path,
    csv_out: str | Path | None = None,
    markdown_out: str | Path | None = None,
) -> dict[str, Any]:
    report = build_learned_policy_diagnostics(bundles=bundles)
    out_path = Path(out)
    csv_path = Path(csv_out) if csv_out is not None else out_path.with_suffix(".csv")
    md_path = Path(markdown_out) if markdown_out is not None else out_path.with_suffix(".md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(csv_path, list(report["rows"]))
    _write_markdown(md_path, report)
    report["diagnostics_path"] = str(out_path)
    report["diagnostics_csv"] = str(csv_path)
    report["diagnostics_markdown"] = str(md_path)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
