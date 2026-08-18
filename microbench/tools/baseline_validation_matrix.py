from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import yaml

from microbench.config import load_yaml
from microbench.metrics import append_result, write_summary
from microbench.planners import planner_metadata
from microbench.runner import run_episode
from microbench.scenarios import materialize_official_suite
from microbench.tools.baseline_leaderboard import SERIOUS_BASELINE_METHODS
from microbench.types import RunSpec


BASELINE_VALIDATION_MATRIX_SCHEMA_VERSION = "0.1"


@dataclass(frozen=True)
class ValidationLane:
    lane_id: str
    category: str
    suite: str
    scenario: str
    comm_profile: str
    n_agents: int
    seed: int
    duration_s: float
    purpose: str
    expected_failure_modes: tuple[str, ...]


VALIDATION_LANES: tuple[ValidationLane, ...] = (
    ValidationLane(
        lane_id="head_on",
        category="head_on",
        suite="official_smoke_generated",
        scenario="head_on_2d_easy",
        comm_profile="ideal_50hz",
        n_agents=4,
        seed=0,
        duration_s=8.0,
        purpose="Reciprocal planar encounter; catches basic pairwise deconfliction and symmetry failures.",
        expected_failure_modes=("reciprocal_deadlock", "centerline_collision", "late_yield"),
    ),
    ValidationLane(
        lane_id="crossing",
        category="crossing",
        suite="official_alpha",
        scenario="crossing_2d_medium",
        comm_profile="ideal_50hz",
        n_agents=6,
        seed=0,
        duration_s=10.0,
        purpose="Four-way planar crossing; catches priority inversion, deadlock, and center conflict.",
        expected_failure_modes=("priority_inversion", "deadlock", "dense_center_conflict"),
    ),
    ValidationLane(
        lane_id="urban_obstacle",
        category="urban_obstacle",
        suite="config",
        scenario="config/scenarios/urban_conflict_3d.yaml",
        comm_profile="realistic_v2v_50hz",
        n_agents=4,
        seed=2,
        duration_s=14.0,
        purpose="Hand-authored 3D urban obstacle conflict; catches building cut-corners and vertical deconfliction gaps.",
        expected_failure_modes=("static_obstacle_cut_corner", "building_shadow_late_detection", "altitude_layer_conflict"),
    ),
    ValidationLane(
        lane_id="communication_delay",
        category="communication_delay",
        suite="official_promotion_calibration",
        scenario="sensor_volume_3d_hard",
        comm_profile="degraded_20hz",
        n_agents=4,
        seed=0,
        duration_s=8.0,
        purpose="Degraded fused-sensing lane; catches stale-track and delayed-message fragility.",
        expected_failure_modes=("stale_track_collision", "fov_blind_spot", "message_sensor_disagreement"),
    ),
    ValidationLane(
        lane_id="high_n_dense_merge",
        category="high_n_dense_merge",
        suite="official_3d_stress",
        scenario="merge_3d_hard",
        comm_profile="degraded_20hz",
        n_agents=12,
        seed=0,
        duration_s=12.0,
        purpose="Higher-N 3D merge bottleneck; catches scaling, throughput, and late-merge failures.",
        expected_failure_modes=("late_merge", "vertical_squeeze", "bottleneck_deadlock", "stale_intent"),
    ),
)


FINITE_RESULT_FIELDS = (
    "collision_episode",
    "min_sep_min_m",
    "completion_rate",
    "final_goal_dist_mean_m",
    "planner_ms_per_tick_per_agent_p95",
    "planner_timeout_count",
    "planner_error_count",
    "planner_fallback_count",
    "episode_runtime_s",
)


def _as_list(values: tuple[str, ...] | list[str] | None, default: tuple[str, ...]) -> list[str]:
    return [str(v).strip() for v in (values if values is not None else default) if str(v).strip()]


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _is_finite(value: Any) -> bool:
    return _to_float(value) is not None


def _check(name: str, ok: bool, details: dict[str, Any] | None = None, *, severity: str = "gate") -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "severity": severity, "details": details or {}}


def _lane_by_id() -> dict[str, ValidationLane]:
    return {lane.lane_id: lane for lane in VALIDATION_LANES}


def _selected_lanes(lanes: tuple[str, ...] | list[str] | None) -> list[ValidationLane]:
    lane_ids = _as_list(lanes, tuple(lane.lane_id for lane in VALIDATION_LANES))
    by_id = _lane_by_id()
    unknown = sorted(set(lane_ids) - set(by_id))
    if unknown:
        raise ValueError(f"Unknown validation lane(s): {','.join(unknown)}")
    return [by_id[lane_id] for lane_id in lane_ids]


def _metadata_by_method() -> dict[str, dict[str, Any]]:
    return {entry["method"]: entry for entry in planner_metadata(include_aliases=False)}


def _method_validation_role(metadata: dict[str, Any]) -> str:
    role = str(metadata.get("role", ""))
    if role == "nondeployable_upper_bound":
        return "upper_bound_non_deployable"
    if bool(metadata.get("learned", False)):
        return "learned_policy"
    if role in {"illustrative_baseline", "agentic_example", "developer_template"}:
        return "illustrative_or_plumbing"
    if role in {"reference_baseline", "agentic_reference_baseline"}:
        return "reference_candidate"
    return "experimental_candidate"


def _expected_behavior(metadata: dict[str, Any], lane: ValidationLane) -> list[str]:
    method = str(metadata["method"])
    validation_role = _method_validation_role(metadata)
    expectations = [
        "emit finite schema-v0.5 metrics",
        "avoid planner exceptions, soft timeouts, and engine fallback commands",
    ]
    if validation_role == "upper_bound_non_deployable":
        expectations.append("act as a privileged upper-bound row, not a deployable local planner")
    elif validation_role == "learned_policy":
        expectations.append("exercise learned-policy observation/action plumbing and disclose learned-model status")
    elif validation_role == "illustrative_or_plumbing":
        expectations.append("serve as a lower-bound or plumbing row; collisions may be expected")
    else:
        expectations.append("provide deployable-style local-planner evidence under allowed observations")

    if lane.category == "head_on":
        expectations.append("resolve or expose reciprocal head-on conflict behavior")
    elif lane.category == "crossing":
        expectations.append("resolve or expose crossing-priority/deadlock behavior")
    elif lane.category == "urban_obstacle":
        expectations.append("show static-obstacle and 3D route-clearance behavior")
    elif lane.category == "communication_delay":
        expectations.append("show behavior under stale/degraded V2V and fused sensing")
        if metadata.get("uses_intent"):
            expectations.append("interpret delayed/stale intent trajectories conservatively")
        if metadata.get("uses_local_sensing"):
            expectations.append("consume fused local sensing when available")
    elif lane.category == "high_n_dense_merge":
        expectations.append("show scaling, merge throughput, and dense-conflict behavior")

    if method == "baseline_goal":
        expectations.append("expected lower bound: ignores traffic and obstacles")
    if method in {"centralized_oracle", "centralized_mpc_oracle"}:
        expectations.append("must remain clearly labeled privileged/nondeployable in reports")
    return expectations


def _planned_entry(method: str, lane: ValidationLane, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": method,
        "display_name": metadata.get("display_name"),
        "validation_role": _method_validation_role(metadata),
        "lane_id": lane.lane_id,
        "category": lane.category,
        "suite": lane.suite,
        "scenario": lane.scenario,
        "comm_profile": lane.comm_profile,
        "n_agents": int(lane.n_agents),
        "seed": int(lane.seed),
        "duration_s": float(lane.duration_s),
        "purpose": lane.purpose,
        "expected_failure_modes": list(lane.expected_failure_modes),
        "expected_behavior": _expected_behavior(metadata, lane),
        "uses_v2v": bool(metadata.get("uses_v2v", False)),
        "uses_local_sensing": bool(metadata.get("uses_local_sensing", False)),
        "uses_intent": bool(metadata.get("uses_intent", False)),
        "uses_agent_messages": bool(metadata.get("uses_agent_messages", False)),
        "uses_obstacles": bool(metadata.get("uses_obstacles", False)),
        "learned": bool(metadata.get("learned", False)),
        "nondeployable_upper_bound": str(metadata.get("role")) == "nondeployable_upper_bound",
    }


def _write_duration_override(path: Path, duration_s: float) -> None:
    cfg = load_yaml(path)
    cfg.setdefault("scenario", {})["duration_s"] = float(duration_s)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def _prepare_lane_scenarios(*, out_dir: Path, lanes: list[ValidationLane]) -> dict[str, Path]:
    by_lane: dict[str, Path] = {}
    generated_cache: dict[str, dict[str, Any]] = {}
    for lane in lanes:
        if lane.suite == "config":
            src = Path(lane.scenario)
            dst = out_dir / "_validation_scenarios" / "config" / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            cfg = load_yaml(src)
            cfg.setdefault("scenario", {})["duration_s"] = float(lane.duration_s)
            dst.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            by_lane[lane.lane_id] = dst
            continue

        if lane.suite not in generated_cache:
            generated_cache[lane.suite] = materialize_official_suite(
                lane.suite,
                out_dir / "_validation_scenarios" / lane.suite,
                overwrite=True,
            )
        paths = {Path(path).stem: Path(path) for path in generated_cache[lane.suite]["scenario_paths"]}
        scenario_path = paths[lane.scenario]
        _write_duration_override(scenario_path, lane.duration_s)
        by_lane[lane.lane_id] = scenario_path
    return by_lane


def _project_row(row: dict[str, Any], lane: ValidationLane) -> dict[str, Any]:
    return {
        "method": row.get("method"),
        "lane_id": lane.lane_id,
        "category": lane.category,
        "suite": lane.suite,
        "scenario": row.get("scenario"),
        "comm_profile": row.get("comm_profile"),
        "N": row.get("N"),
        "seed": row.get("seed"),
        "duration_s": row.get("duration_s"),
        "collision_episode": row.get("collision_episode"),
        "collisions": row.get("collisions"),
        "near_misses": row.get("near_misses"),
        "min_sep_min_m": row.get("min_sep_min_m"),
        "completion_rate": row.get("completion_rate"),
        "goal_progress_fraction": row.get("goal_progress_fraction"),
        "final_goal_dist_mean_m": row.get("final_goal_dist_mean_m"),
        "deadlock_time_pct": row.get("deadlock_time_pct"),
        "planner_ms_per_tick_per_agent_p95": row.get("planner_ms_per_tick_per_agent_p95"),
        "planner_timeout_count": row.get("planner_timeout_count"),
        "planner_error_count": row.get("planner_error_count"),
        "planner_fallback_count": row.get("planner_fallback_count"),
        "obs_v2v_fraction": row.get("obs_v2v_fraction"),
        "obs_sensor_fraction": row.get("obs_sensor_fraction"),
        "obs_stale_fraction": row.get("obs_stale_fraction"),
        "comm_agent_msg_delivery_fraction": row.get("comm_agent_msg_delivery_fraction"),
        "episode_runtime_s": row.get("episode_runtime_s"),
    }


def _row_checks(row: dict[str, Any], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    missing_or_nonfinite = [field for field in FINITE_RESULT_FIELDS if not _is_finite(row.get(field))]
    timeout = int(_to_float(row.get("planner_timeout_count")) or 0)
    error = int(_to_float(row.get("planner_error_count")) or 0)
    fallback = int(_to_float(row.get("planner_fallback_count")) or 0)
    collision_episode = _to_float(row.get("collision_episode")) or 0.0
    completion = _to_float(row.get("completion_rate")) or 0.0
    min_sep = _to_float(row.get("min_sep_min_m"))
    lane_category = str(row.get("category"))
    validation_role = _method_validation_role(metadata)

    checks = [
        _check("finite_core_metrics", not missing_or_nonfinite, {"missing_or_nonfinite": missing_or_nonfinite}),
        _check("planner_timeouts_clear", timeout == 0, {"planner_timeout_count": timeout}),
        _check("planner_errors_clear", error == 0, {"planner_error_count": error}),
        _check("planner_fallbacks_clear", fallback == 0, {"planner_fallback_count": fallback}),
        _check(
            "collision_free_observed",
            collision_episode <= 0.0,
            {"collision_episode": collision_episode},
            severity="behavior",
        ),
        _check(
            "nonnegative_min_clearance_observed",
            min_sep is not None and min_sep >= 0.0,
            {"min_sep_min_m": min_sep},
            severity="behavior",
        ),
        _check(
            "mission_progress_observed",
            completion > 0.0 or (_to_float(row.get("final_goal_dist_mean_m")) or float("inf")) < 50.0,
            {"completion_rate": completion, "final_goal_dist_mean_m": row.get("final_goal_dist_mean_m")},
            severity="behavior",
        ),
    ]
    if lane_category == "communication_delay":
        checks.append(
            _check(
                "degraded_observation_signal_present",
                (_to_float(row.get("obs_sensor_fraction")) or 0.0) > 0.0
                or (_to_float(row.get("obs_stale_fraction")) or 0.0) > 0.0,
                {
                    "obs_sensor_fraction": row.get("obs_sensor_fraction"),
                    "obs_stale_fraction": row.get("obs_stale_fraction"),
                },
                severity="behavior",
            )
        )
    if validation_role == "upper_bound_non_deployable":
        checks.append(
            _check(
                "upper_bound_role_disclosed",
                True,
                {"role": "nondeployable_upper_bound"},
                severity="metadata",
            )
        )
    return checks


def run_baseline_validation_matrix(
    *,
    out_dir: str | Path,
    methods: tuple[str, ...] | list[str] | None = None,
    lanes: tuple[str, ...] | list[str] | None = None,
    duration_s: float | None = None,
    max_runs: int | None = None,
    plan_only: bool = False,
) -> dict[str, Any]:
    out = Path(out_dir)
    methods_list = _as_list(methods, SERIOUS_BASELINE_METHODS)
    selected_lanes = _selected_lanes(lanes)
    metadata = _metadata_by_method()
    unknown_methods = sorted(set(methods_list) - set(metadata))
    if unknown_methods:
        raise ValueError(f"Unknown validation baseline(s): {','.join(unknown_methods)}")

    if duration_s is not None:
        selected_lanes = [
            ValidationLane(**{**asdict(lane), "duration_s": float(duration_s)})
            for lane in selected_lanes
        ]

    planned = [
        _planned_entry(method, lane, metadata[method])
        for lane in selected_lanes
        for method in methods_list
    ]
    selected_plan = planned[: max(0, int(max_runs))] if max_runs is not None else planned

    if plan_only:
        return {
            "schema_version": BASELINE_VALIDATION_MATRIX_SCHEMA_VERSION,
            "plan_only": True,
            "ok": False,
            "behavior_pass": False,
            "methods": methods_list,
            "lanes": [asdict(lane) for lane in selected_lanes],
            "planned_run_count": len(planned),
            "selected_run_count": len(selected_plan),
            "run_count": 0,
            "matrix": planned,
            "rows": [],
            "checks": [],
            "methods_detail": [],
            "results_csv": None,
            "summary_csv": None,
        }

    if (out / "results.csv").exists():
        raise RuntimeError(f"baseline validation matrix output already exists: {out / 'results.csv'}")
    out.mkdir(parents=True, exist_ok=True)
    scenario_paths = _prepare_lane_scenarios(out_dir=out, lanes=selected_lanes)
    lane_by_id = {lane.lane_id: lane for lane in selected_lanes}

    rows: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for entry in selected_plan:
        lane = lane_by_id[str(entry["lane_id"])]
        spec = RunSpec(
            scenario_path=str(scenario_paths[lane.lane_id]),
            method=str(entry["method"]),
            n_agents=int(lane.n_agents),
            seed=int(lane.seed),
            comm_profile=str(lane.comm_profile),
            out_dir=str(out),
            save_trace=False,
        )
        row = run_episode(spec)
        append_result(out, row)
        projected = _project_row(row, lane)
        rows.append(projected)
        for check in _row_checks(projected, metadata[str(entry["method"])]):
            checks.append(
                {
                    **check,
                    "method": str(entry["method"]),
                    "lane_id": lane.lane_id,
                    "category": lane.category,
                }
            )
    summary_csv = write_summary(out)

    methods_detail = []
    for method in methods_list:
        method_rows = [row for row in rows if row["method"] == method]
        method_checks = [check for check in checks if check["method"] == method]
        gate_failed = [check for check in method_checks if check["severity"] == "gate" and not check["ok"]]
        behavior_failed = [check for check in method_checks if check["severity"] == "behavior" and not check["ok"]]
        methods_detail.append(
            {
                "method": method,
                "display_name": metadata[method].get("display_name"),
                "validation_role": _method_validation_role(metadata[method]),
                "run_count": len(method_rows),
                "gate_pass": not gate_failed if method_rows else False,
                "behavior_pass": not behavior_failed if method_rows else False,
                "failed_gate_checks": gate_failed,
                "failed_behavior_checks": behavior_failed,
            }
        )

    gate_pass = all(check["ok"] for check in checks if check["severity"] == "gate")
    behavior_pass = all(check["ok"] for check in checks if check["severity"] == "behavior")
    report = {
        "schema_version": BASELINE_VALIDATION_MATRIX_SCHEMA_VERSION,
        "plan_only": False,
        "ok": bool(gate_pass),
        "behavior_pass": bool(behavior_pass),
        "methods": methods_list,
        "lanes": [asdict(lane) for lane in selected_lanes],
        "planned_run_count": len(planned),
        "selected_run_count": len(selected_plan),
        "run_count": len(rows),
        "matrix": planned,
        "rows": rows,
        "checks": checks,
        "methods_detail": methods_detail,
        "results_csv": str(out / "results.csv"),
        "summary_csv": str(summary_csv),
    }
    return report


def write_baseline_validation_matrix(*, out_dir: str | Path, **kwargs: Any) -> Path:
    report = run_baseline_validation_matrix(out_dir=out_dir, **kwargs)
    path = Path(out_dir) / "baseline_validation_matrix.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
