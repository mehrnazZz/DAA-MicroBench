from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from microbench.rl.submission_bundle import (
    LEARNED_SUBMISSION_BUNDLE_REVIEW_SCHEMA_VERSION,
    review_learned_policy_submission_bundle,
)


LEARNED_POLICY_LEADERBOARD_SCHEMA_VERSION = "0.1"

LEARNED_POLICY_LEADERBOARD_FIELDS = (
    "development_rank",
    "bundle",
    "ok",
    "leaderboard_candidate",
    "recommendation",
    "method",
    "policy",
    "suite",
    "run_count",
    "planned_run_count",
    "score_v0_mean",
    "score_v0_best",
    "score_v0_worst",
    "collision_episode_count",
    "collision_episode_rate_mean",
    "min_sep_min_m",
    "min_sep_p05_min_m",
    "completion_rate_mean",
    "completion_rate_min",
    "deadlock_time_pct_mean",
    "planner_ms_p95_max",
    "planner_timeout_count",
    "planner_error_count",
    "planner_fallback_count",
    "rl_ok",
    "rl_behavior_pass",
    "rl_run_count",
    "rl_lane_count",
    "rl_collision_ticks",
    "rl_near_miss_ticks",
    "rl_completion_rate_mean",
    "rl_final_min_sep_min_m",
    "failed_rl_gate_checks",
    "failed_rl_behavior_checks",
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


def _round_or_none(value: Any, ndigits: int = 6) -> float | None:
    out = _to_float(value)
    return None if out is None else round(out, ndigits)


def _get(mapping: dict[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _list_len(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _review_to_row(*, bundle: str | Path, review: dict[str, Any], error: str | None = None) -> dict[str, Any]:
    safety = _get(review, "dimensions", "safety") or {}
    mission = _get(review, "dimensions", "mission") or {}
    compute = _get(review, "dimensions", "compute") or {}
    rl_validation = _get(review, "dimensions", "rl_validation_matrix") or {}
    score = review.get("score_v0") if isinstance(review.get("score_v0"), dict) else {}
    limitations = list(review.get("limitations", [])) if isinstance(review.get("limitations"), list) else []
    recommendation = str(review.get("recommendation") or "fix_artifacts")

    return {
        "development_rank": None,
        "bundle": str(bundle),
        "ok": bool(review.get("ok")) and error is None,
        "leaderboard_candidate": bool(review.get("ok")) and error is None and recommendation == "leaderboard_candidate",
        "recommendation": recommendation,
        "method": review.get("method"),
        "policy": review.get("policy"),
        "suite": review.get("suite"),
        "run_count": int(review.get("run_count", 0) or 0),
        "planned_run_count": int(review.get("planned_run_count", 0) or 0),
        "score_v0_mean": _round_or_none(score.get("mean")),
        "score_v0_best": _round_or_none(score.get("best")),
        "score_v0_worst": _round_or_none(score.get("worst")),
        "collision_episode_count": int(safety.get("collision_episode_count", 0) or 0),
        "collision_episode_rate_mean": _round_or_none(safety.get("collision_episode_rate_mean")),
        "min_sep_min_m": _round_or_none(safety.get("min_sep_min_m")),
        "min_sep_p05_min_m": _round_or_none(safety.get("min_sep_p05_min_m")),
        "completion_rate_mean": _round_or_none(mission.get("completion_rate_mean")),
        "completion_rate_min": _round_or_none(mission.get("completion_rate_min")),
        "deadlock_time_pct_mean": _round_or_none(mission.get("deadlock_time_pct_mean")),
        "planner_ms_p95_max": _round_or_none(compute.get("planner_ms_p95_max")),
        "planner_timeout_count": int(compute.get("planner_timeout_count", 0) or 0),
        "planner_error_count": int(compute.get("planner_error_count", 0) or 0),
        "planner_fallback_count": int(compute.get("planner_fallback_count", 0) or 0),
        "rl_ok": bool(rl_validation.get("ok")),
        "rl_behavior_pass": bool(rl_validation.get("behavior_pass")),
        "rl_run_count": int(rl_validation.get("run_count", 0) or 0),
        "rl_lane_count": int(rl_validation.get("lane_count", 0) or 0),
        "rl_collision_ticks": int(rl_validation.get("collision_ticks", 0) or 0),
        "rl_near_miss_ticks": int(rl_validation.get("near_miss_ticks", 0) or 0),
        "rl_completion_rate_mean": _round_or_none(rl_validation.get("completion_rate_mean")),
        "rl_final_min_sep_min_m": _round_or_none(rl_validation.get("final_min_sep_min_m")),
        "failed_rl_gate_checks": _list_len(rl_validation.get("failed_gate_checks")),
        "failed_rl_behavior_checks": _list_len(rl_validation.get("failed_behavior_checks")),
        "limitation_count": len(limitations),
        "limitations": ",".join(str(item) for item in limitations),
        "error": error,
    }


def _sort_key(row: dict[str, Any]) -> tuple[int, int, float, str, str]:
    score = _to_float(row.get("score_v0_mean"))
    return (
        0 if row.get("ok") else 1,
        0 if score is not None else 1,
        float(score if score is not None else 1e18),
        str(row.get("method") or ""),
        str(row.get("policy") or ""),
    )


def _rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = [dict(row) for row in sorted(rows, key=_sort_key)]
    rank = 1
    for row in ranked:
        if row.get("ok") and row.get("score_v0_mean") is not None:
            row["development_rank"] = rank
            rank += 1
    return ranked


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(LEARNED_POLICY_LEADERBOARD_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in LEARNED_POLICY_LEADERBOARD_FIELDS})


def build_learned_policy_leaderboard(*, bundles: list[str | Path] | tuple[str | Path, ...]) -> dict[str, Any]:
    """Build a development leaderboard from existing learned-policy submission bundles."""

    rows: list[dict[str, Any]] = []
    for bundle in bundles:
        try:
            review = review_learned_policy_submission_bundle(bundle=bundle)
            rows.append(_review_to_row(bundle=bundle, review=review))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            rows.append(_review_to_row(bundle=bundle, review={"recommendation": "fix_artifacts"}, error=error))

    ranked_rows = _rank_rows(rows)
    candidate_rows = [row for row in ranked_rows if row.get("leaderboard_candidate")]
    reviewable_rows = [row for row in ranked_rows if row.get("ok")]
    failed_rows = [row for row in ranked_rows if not row.get("ok")]

    return {
        "schema_version": LEARNED_POLICY_LEADERBOARD_SCHEMA_VERSION,
        "review_schema_version": LEARNED_SUBMISSION_BUNDLE_REVIEW_SCHEMA_VERSION,
        "ok": not failed_rows,
        "bundle_count": len(rows),
        "reviewable_count": len(reviewable_rows),
        "leaderboard_candidate_count": len(candidate_rows),
        "failed_count": len(failed_rows),
        "score_note": (
            "Development rows are sorted by learned bundle review score_v0_mean; lower is better. "
            "Only rows with recommendation=leaderboard_candidate should be treated as candidate leaderboard evidence."
        ),
        "source_note": "Rows are derived from review-learned-bundle summaries; keep the underlying bundles for audit.",
        "columns": list(LEARNED_POLICY_LEADERBOARD_FIELDS),
        "rows": ranked_rows,
    }


def write_learned_policy_leaderboard(
    *,
    bundles: list[str | Path] | tuple[str | Path, ...],
    out: str | Path,
    csv_out: str | Path | None = None,
) -> dict[str, Any]:
    report = build_learned_policy_leaderboard(bundles=bundles)
    out_path = Path(out)
    csv_path = Path(csv_out) if csv_out is not None else out_path.with_suffix(".csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(csv_path, list(report["rows"]))
    report["leaderboard_path"] = str(out_path)
    report["leaderboard_csv"] = str(csv_path)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
