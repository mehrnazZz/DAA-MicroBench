from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import yaml

from microbench.config import load_yaml
from microbench.metrics import append_result, write_summary
from microbench.runner import run_episode
from microbench.scenarios import materialize_official_suite
from microbench.tools.baseline_audit import build_baseline_audit
from microbench.tools.baseline_promotion import PROMOTION_METHODS
from microbench.types import RunSpec


BASELINE_REVIEW_SCHEMA_VERSION = "0.1"
DEFAULT_REVIEW_DURATION_S = 20.0
REVIEW_RUNTIME_P95_MAX_MS = 100.0
DEGRADED_SENSOR_FRACTION_MIN = 0.05
DEGRADED_STALE_FRACTION_MIN = 0.005


@dataclass(frozen=True)
class ReviewLane:
    lane_id: str
    suite: str
    scenario: str
    comm_profile: str
    n_agents: int
    seed: int
    category: str
    purpose: str


REVIEW_LANES: tuple[ReviewLane, ...] = (
    ReviewLane(
        lane_id="3d_sphere_swap",
        suite="official_3d_stress",
        scenario="sphere_swap_3d_medium",
        comm_profile="ideal_50hz",
        n_agents=6,
        seed=0,
        category="3d_stress",
        purpose="Longer volumetric 3D stress row for candidate baseline metadata review.",
    ),
    ReviewLane(
        lane_id="3d_sensor_degraded",
        suite="official_3d_stress",
        scenario="sensor_volume_3d_hard",
        comm_profile="degraded_20hz",
        n_agents=6,
        seed=0,
        category="degraded_sensing_comm",
        purpose="Longer 3D fused-sensing row with degraded V2V and stale observation diagnostics.",
    ),
    ReviewLane(
        lane_id="agentic_priority_degraded",
        suite="official_agentic_stress",
        scenario="heterogeneous_priority_crossing_3d_medium",
        comm_profile="degraded_20hz",
        n_agents=6,
        seed=0,
        category="agentic_stress",
        purpose="Longer heterogeneous-priority 3D row for agentic coordination review.",
    ),
)


def _as_list(values: tuple[str, ...] | list[str] | None, default: tuple[str, ...]) -> list[str]:
    return [str(v).strip() for v in (values if values is not None else default) if str(v).strip()]


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _check(name: str, ok: bool, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "details": details or {}}


def _selected_lanes(lanes: tuple[str, ...] | list[str] | None) -> list[ReviewLane]:
    lane_ids = _as_list(lanes, tuple(lane.lane_id for lane in REVIEW_LANES))
    by_id = {lane.lane_id: lane for lane in REVIEW_LANES}
    unknown = sorted(set(lane_ids) - set(by_id))
    if unknown:
        raise ValueError(f"Unknown review lane(s): {','.join(unknown)}")
    return [by_id[lane_id] for lane_id in lane_ids]


def _prepare_lane_scenarios(*, out_dir: Path, lanes: list[ReviewLane], duration_s: float | None) -> dict[tuple[str, str], Path]:
    paths: dict[tuple[str, str], Path] = {}
    for suite in sorted({lane.suite for lane in lanes}):
        generated = materialize_official_suite(
            suite,
            out_dir / "_generated_scenarios" / suite,
            overwrite=True,
        )
        by_stem = {Path(path).stem: Path(path) for path in generated["scenario_paths"]}
        for lane in lanes:
            if lane.suite != suite:
                continue
            scenario_path = by_stem[lane.scenario]
            if duration_s is not None:
                cfg = load_yaml(scenario_path)
                cfg.setdefault("scenario", {})["duration_s"] = float(duration_s)
                scenario_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            paths[(lane.suite, lane.scenario)] = scenario_path
    return paths


def _project_row(row: dict[str, Any], lane: ReviewLane) -> dict[str, Any]:
    return {
        "lane_id": lane.lane_id,
        "lane_category": lane.category,
        "suite": lane.suite,
        "method": row.get("method"),
        "scenario": row.get("scenario"),
        "comm_profile": row.get("comm_profile"),
        "N": row.get("N"),
        "seed": row.get("seed"),
        "collision_episode": row.get("collision_episode"),
        "min_sep_min_m": row.get("min_sep_min_m"),
        "completion_rate": row.get("completion_rate"),
        "planner_ms_per_tick_per_agent_p95": row.get("planner_ms_per_tick_per_agent_p95"),
        "obs_v2v_fraction": row.get("obs_v2v_fraction"),
        "obs_sensor_fraction": row.get("obs_sensor_fraction"),
        "obs_stale_fraction": row.get("obs_stale_fraction"),
        "comm_agent_msg_delivery_fraction": row.get("comm_agent_msg_delivery_fraction"),
        "comm_negotiation_proposals": row.get("comm_negotiation_proposals"),
        "comm_negotiation_acks": row.get("comm_negotiation_acks"),
        "planner_timeout_count": row.get("planner_timeout_count"),
        "planner_error_count": row.get("planner_error_count"),
        "planner_fallback_count": row.get("planner_fallback_count"),
        "episode_runtime_s": row.get("episode_runtime_s"),
    }


def _row_review_checks(row: dict[str, Any]) -> list[dict[str, Any]]:
    guardrails = sum(
        int(_to_float(row.get(field)) or 0)
        for field in ("planner_timeout_count", "planner_error_count", "planner_fallback_count")
    )
    checks = [
        _check(
            "collision_free",
            (_to_float(row.get("collision_episode")) or 0.0) <= 0.0,
            {"collision_episode": row.get("collision_episode")},
        ),
        _check(
            "min_clearance_nonnegative",
            (_to_float(row.get("min_sep_min_m")) or -1.0) >= 0.0,
            {"min_sep_min_m": row.get("min_sep_min_m")},
        ),
        _check(
            "runtime_p95_bounded",
            (_to_float(row.get("planner_ms_per_tick_per_agent_p95")) or 0.0) <= REVIEW_RUNTIME_P95_MAX_MS,
            {
                "planner_ms_per_tick_per_agent_p95": row.get("planner_ms_per_tick_per_agent_p95"),
                "threshold_ms": REVIEW_RUNTIME_P95_MAX_MS,
            },
        ),
        _check("planner_guardrails_clear", guardrails == 0, {"guardrail_total": guardrails}),
    ]
    if row.get("lane_category") == "degraded_sensing_comm":
        checks.extend(
            [
                _check(
                    "degraded_sensor_signal_present",
                    (_to_float(row.get("obs_sensor_fraction")) or 0.0) >= DEGRADED_SENSOR_FRACTION_MIN,
                    {
                        "obs_sensor_fraction": row.get("obs_sensor_fraction"),
                        "threshold": DEGRADED_SENSOR_FRACTION_MIN,
                    },
                ),
                _check(
                    "degraded_stale_signal_present",
                    (_to_float(row.get("obs_stale_fraction")) or 0.0) >= DEGRADED_STALE_FRACTION_MIN,
                    {
                        "obs_stale_fraction": row.get("obs_stale_fraction"),
                        "threshold": DEGRADED_STALE_FRACTION_MIN,
                    },
                ),
            ]
        )
    return checks


def _method_recommendation(*, method: str, audit_entry: dict[str, Any], rows: list[dict[str, Any]], checks: list[dict[str, Any]]) -> str:
    if not rows:
        return "not_run"
    if any(not check["ok"] for check in checks):
        return "keep_experimental_until_review_checks_pass"
    if str(audit_entry.get("role")) not in {"reference_baseline", "agentic_reference_baseline"}:
        return "needs_reference_role_decision"
    if str(audit_entry.get("status")) == "experimental":
        return "review_for_pre_v1_metadata"
    if str(audit_entry.get("status")) == "pre_v1":
        return "review_for_stable_metadata"
    return "already_metadata_ready"


def run_baseline_stable_review(
    *,
    out_dir: str | Path,
    root: str | Path = ".",
    methods: tuple[str, ...] | list[str] | None = None,
    lanes: tuple[str, ...] | list[str] | None = None,
    duration_s: float | None = DEFAULT_REVIEW_DURATION_S,
    max_runs: int | None = None,
    plan_only: bool = False,
) -> dict[str, Any]:
    out = Path(out_dir)
    methods_list = _as_list(methods, PROMOTION_METHODS)
    unknown_methods = sorted(set(methods_list) - set(PROMOTION_METHODS))
    if unknown_methods:
        raise ValueError(f"Unknown review baseline(s): {','.join(unknown_methods)}")
    selected_lanes = _selected_lanes(lanes)

    planned = [(lane, method) for lane in selected_lanes for method in methods_list]
    if max_runs is not None:
        planned = planned[: max(0, int(max_runs))]

    lane_plan = [asdict(lane) for lane in selected_lanes]
    if plan_only:
        return {
            "schema_version": BASELINE_REVIEW_SCHEMA_VERSION,
            "review_type": "stable_metadata_prep",
            "plan_only": True,
            "duration_s": duration_s,
            "methods": methods_list,
            "lanes": lane_plan,
            "planned_run_count": len(planned),
            "run_count": 0,
            "review_checks_pass": False,
            "results_csv": None,
            "summary_csv": None,
            "rows": [],
            "methods_detail": [],
        }

    if (out / "results.csv").exists():
        raise RuntimeError(f"baseline review output already exists: {out / 'results.csv'}")
    scenario_paths = _prepare_lane_scenarios(out_dir=out, lanes=selected_lanes, duration_s=duration_s)

    rows: list[dict[str, Any]] = []
    for lane, method in planned:
        spec = RunSpec(
            scenario_path=str(scenario_paths[(lane.suite, lane.scenario)]),
            method=method,
            n_agents=int(lane.n_agents),
            seed=int(lane.seed),
            comm_profile=str(lane.comm_profile),
            out_dir=str(out),
            save_trace=False,
        )
        row = run_episode(spec)
        append_result(out, row)
        rows.append(_project_row(row, lane))
    summary_csv = write_summary(out)

    row_checks: list[dict[str, Any]] = []
    for row in rows:
        for check in _row_review_checks(row):
            row_checks.append(
                {
                    **check,
                    "method": row["method"],
                    "lane_id": row["lane_id"],
                    "lane_category": row["lane_category"],
                    "scenario": row["scenario"],
                    "comm_profile": row["comm_profile"],
                }
            )

    audit = build_baseline_audit(root=root)
    audit_by_method = {entry["method"]: entry for entry in audit["methods"]}
    methods_detail: list[dict[str, Any]] = []
    for method in methods_list:
        method_rows = [row for row in rows if row["method"] == method]
        method_checks = [check for check in row_checks if check["method"] == method]
        if method == "negotiation_yield" and method_rows:
            proposals = sum(int(_to_float(row.get("comm_negotiation_proposals")) or 0) for row in method_rows)
            acks = sum(int(_to_float(row.get("comm_negotiation_acks")) or 0) for row in method_rows)
            method_checks.append(
                {
                    **_check(
                        "negotiation_signal_present",
                        proposals > 0 and acks > 0,
                        {"proposals": proposals, "acks": acks},
                    ),
                    "method": method,
                    "lane_id": "*",
                    "lane_category": "agentic_signal",
                    "scenario": "*",
                    "comm_profile": "*",
                }
            )
        audit_entry = audit_by_method.get(method, {})
        checks_pass = bool(method_rows) and all(check["ok"] for check in method_checks)
        methods_detail.append(
            {
                "method": method,
                "role": audit_entry.get("role"),
                "status": audit_entry.get("status"),
                "lanes_run": sorted({row["lane_id"] for row in method_rows}),
                "run_count": len(method_rows),
                "review_checks_pass": checks_pass,
                "failed_checks": [
                    {
                        "name": check["name"],
                        "lane_id": check["lane_id"],
                        "details": check["details"],
                    }
                    for check in method_checks
                    if not check["ok"]
                ],
                "metadata_recommendation": _method_recommendation(
                    method=method,
                    audit_entry=audit_entry,
                    rows=method_rows,
                    checks=method_checks,
                ),
            }
        )

    return {
        "schema_version": BASELINE_REVIEW_SCHEMA_VERSION,
        "review_type": "stable_metadata_prep",
        "plan_only": False,
        "duration_s": duration_s,
        "methods": methods_list,
        "lanes": lane_plan,
        "planned_run_count": len(planned),
        "run_count": len(rows),
        "review_checks_pass": all(entry["review_checks_pass"] for entry in methods_detail),
        "results_csv": str(out / "results.csv"),
        "summary_csv": str(summary_csv),
        "rows": rows,
        "checks": row_checks,
        "methods_detail": methods_detail,
    }


def write_baseline_stable_review(*, out_dir: str | Path, **kwargs: Any) -> Path:
    report = run_baseline_stable_review(out_dir=out_dir, **kwargs)
    path = Path(out_dir) / "baseline_review.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
