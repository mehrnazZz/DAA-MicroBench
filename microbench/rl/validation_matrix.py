from __future__ import annotations

import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any

from microbench.rl.envs import DaaParallelEnv
from microbench.rl.policy_spec import policy_factory_from_spec
from microbench.rl.rollout import RL_ROLLOUT_FIELDS, RL_ROLLOUT_SCHEMA_VERSION, rollout_parallel_env
from microbench.rl.schema import (
    RL_ACTION_SCHEMA_VERSION,
    RL_INTERFACE_VERSION,
    RL_OBSERVATION_SCHEMA_VERSION,
    RL_REWARD_SCHEMA_VERSION,
    interface_contract,
)
from microbench.tools.baseline_validation_matrix import (
    ValidationLane,
    prepare_validation_lane_scenarios,
    selected_validation_lanes,
)


RL_VALIDATION_MATRIX_SCHEMA_VERSION = "0.1"
RL_VALIDATION_MATRIX_FIELDS = (
    "lane_id",
    "category",
    "purpose",
    "expected_failure_modes",
    "duration_s",
    *RL_ROLLOUT_FIELDS,
)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _check(name: str, ok: bool, details: dict[str, Any] | None = None, *, severity: str = "gate") -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "severity": str(severity), "details": details or {}}


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return ";".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return value


def _write_episode_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(RL_VALIDATION_MATRIX_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in RL_VALIDATION_MATRIX_FIELDS})


def _with_overrides(
    lanes: list[ValidationLane],
    *,
    duration_s: float | None,
    n_agents: int | None,
) -> list[ValidationLane]:
    out: list[ValidationLane] = []
    for lane in lanes:
        payload = asdict(lane)
        if duration_s is not None:
            payload["duration_s"] = float(duration_s)
        if n_agents is not None:
            payload["n_agents"] = int(n_agents)
        out.append(ValidationLane(**payload))
    return out


def _expected_behavior(lane: ValidationLane) -> list[str]:
    expectations = [
        "exercise the PettingZoo-style ParallelEnv reset/step/action-space contract",
        "emit finite rollout metrics through the public RL observation/action interface",
    ]
    if lane.category == "head_on":
        expectations.append("expose reciprocal head-on learned-policy behavior")
    elif lane.category == "crossing":
        expectations.append("expose crossing-priority/deadlock learned-policy behavior")
    elif lane.category == "urban_obstacle":
        expectations.append("expose 3D urban obstacle and route-clearance behavior")
    elif lane.category == "communication_delay":
        expectations.append("expose stale/degraded V2V and fused-sensing behavior")
    elif lane.category == "high_n_dense_merge":
        expectations.append("expose high-N merge scaling behavior")
    return expectations


def _planned_entry(lane: ValidationLane, *, seed: int) -> dict[str, Any]:
    return {
        "lane_id": lane.lane_id,
        "category": lane.category,
        "suite": lane.suite,
        "scenario": lane.scenario,
        "comm_profile": lane.comm_profile,
        "n_agents": int(lane.n_agents),
        "seed": int(seed),
        "duration_s": float(lane.duration_s),
        "purpose": lane.purpose,
        "expected_failure_modes": list(lane.expected_failure_modes),
        "expected_behavior": _expected_behavior(lane),
    }


def _row_checks(row: dict[str, Any]) -> list[dict[str, Any]]:
    api_error = str(row.get("api_error") or "")
    finite_metrics = (
        bool(row.get("finite_observations"))
        and bool(row.get("finite_rewards"))
        and _finite(row.get("total_reward"))
        and _finite(row.get("completion_rate"))
        and _finite(row.get("final_min_sep_m"))
    )
    steps = int(float(row.get("steps", 0) or 0))
    controlled_agents = int(float(row.get("controlled_agents", 0) or 0))
    collision_ticks = int(float(row.get("collision_ticks", 0) or 0))
    completion = float(row.get("completion_rate", 0.0) or 0.0)
    min_sep = float(row.get("final_min_sep_m", float("nan")) or float("nan"))

    checks = [
        _check("api_error_clear", not api_error, {"api_error": api_error}),
        _check(
            "finite_rollout_metrics",
            finite_metrics,
            {
                "finite_observations": row.get("finite_observations"),
                "finite_rewards": row.get("finite_rewards"),
                "total_reward": row.get("total_reward"),
                "completion_rate": row.get("completion_rate"),
                "final_min_sep_m": row.get("final_min_sep_m"),
            },
        ),
        _check("controlled_agents_present", controlled_agents > 0, {"controlled_agents": controlled_agents}),
        _check("episode_progressed", steps > 0, {"steps": steps}),
        _check(
            "collision_free_observed",
            collision_ticks <= 0,
            {"collision_ticks": collision_ticks},
            severity="behavior",
        ),
        _check(
            "nonnegative_final_clearance_observed",
            math.isfinite(min_sep) and min_sep >= 0.0,
            {"final_min_sep_m": row.get("final_min_sep_m")},
            severity="behavior",
        ),
        _check(
            "completion_observed",
            completion > 0.0,
            {"completion_rate": completion},
            severity="behavior",
        ),
    ]
    if row.get("lane_id") == "communication_delay":
        checks.append(
            _check(
                "degraded_comm_profile_applied",
                str(row.get("comm_profile")) == "degraded_20hz",
                {"comm_profile": row.get("comm_profile")},
            )
        )
    if row.get("lane_id") == "high_n_dense_merge":
        checks.append(
            _check(
                "high_n_agent_count_applied",
                int(float(row.get("n_agents", 0) or 0)) >= 10,
                {"n_agents": row.get("n_agents")},
            )
        )
    return checks


def run_rl_validation_matrix(
    *,
    out_dir: str | Path,
    policy: str = "goal_direction",
    policy_spec: str | Path | None = None,
    lanes: tuple[str, ...] | list[str] | None = None,
    seeds: tuple[int, ...] | list[int] | None = None,
    duration_s: float | None = None,
    n_agents: int | None = None,
    max_steps: int | None = None,
    plan_only: bool = False,
) -> dict[str, Any]:
    """Run learned-policy wrapper validation on the canonical baseline matrix lanes."""

    out = Path(out_dir)
    episode_csv = out / "rl_validation_matrix_episodes.csv"
    if not plan_only and episode_csv.exists():
        raise RuntimeError(f"RL validation matrix output already exists: {episode_csv}")

    selected = _with_overrides(
        selected_validation_lanes(lanes),
        duration_s=duration_s,
        n_agents=n_agents,
    )
    seed_override = None if seeds is None else [int(seed) for seed in seeds]
    planned = [
        _planned_entry(lane, seed=int(seed))
        for lane in selected
        for seed in (seed_override if seed_override is not None else [int(lane.seed)])
    ]

    policy_for_rollout: Any = str(policy)
    policy_name = str(policy)
    policy_spec_summary = None
    if policy_spec is not None:
        policy_for_rollout, policy_spec_summary = policy_factory_from_spec(policy_spec)
        policy_name = str(policy_spec_summary["policy_name"])

    if plan_only:
        return {
            "schema_version": RL_VALIDATION_MATRIX_SCHEMA_VERSION,
            "rollout_schema_version": RL_ROLLOUT_SCHEMA_VERSION,
            "interface_version": RL_INTERFACE_VERSION,
            "action_schema_version": RL_ACTION_SCHEMA_VERSION,
            "observation_schema_version": RL_OBSERVATION_SCHEMA_VERSION,
            "reward_schema_version": RL_REWARD_SCHEMA_VERSION,
            "plan_only": True,
            "ok": False,
            "behavior_pass": False,
            "policy": policy_name,
            "policy_spec": policy_spec_summary,
            "lanes": [asdict(lane) for lane in selected],
            "planned_run_count": len(planned),
            "run_count": 0,
            "max_steps": None if max_steps is None else int(max_steps),
            "matrix": planned,
            "episodes": [],
            "checks": [],
            "episode_csv": None,
            "interface_contract": interface_contract(top_k=8),
        }

    scenario_paths = prepare_validation_lane_scenarios(out_dir=out, lanes=selected)
    lane_by_id = {lane.lane_id: lane for lane in selected}
    rows: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for entry in planned:
        lane = lane_by_id[str(entry["lane_id"])]
        env = DaaParallelEnv(
            scenario_path=str(scenario_paths[lane.lane_id]),
            n_agents=int(lane.n_agents),
            seed=int(entry["seed"]),
            comm_profile=str(lane.comm_profile),
        )
        try:
            row = rollout_parallel_env(
                env,
                policy_for_rollout,
                seed=int(entry["seed"]),
                max_steps=max_steps,
                metadata={
                    "suite": str(lane.suite),
                    "scenario": Path(str(scenario_paths[lane.lane_id])).stem,
                    "lane_id": lane.lane_id,
                    "category": lane.category,
                    "purpose": lane.purpose,
                    "expected_failure_modes": list(lane.expected_failure_modes),
                    "duration_s": float(lane.duration_s),
                    "policy": policy_name,
                    "n_agents": int(lane.n_agents),
                    "comm_profile": str(lane.comm_profile),
                },
            )
        except Exception as exc:  # pragma: no cover - failure reporting path.
            row = {
                "suite": str(lane.suite),
                "scenario": Path(str(scenario_paths[lane.lane_id])).stem,
                "lane_id": lane.lane_id,
                "category": lane.category,
                "purpose": lane.purpose,
                "expected_failure_modes": list(lane.expected_failure_modes),
                "duration_s": float(lane.duration_s),
                "dimension": "unknown",
                "policy": policy_name,
                "n_agents": int(lane.n_agents),
                "seed": int(entry["seed"]),
                "comm_profile": str(lane.comm_profile),
                "steps": 0,
                "controlled_agents": 0,
                "completed_agents": 0,
                "completion_rate": 0.0,
                "terminated_agents": 0,
                "truncated_agents": 0,
                "total_reward": 0.0,
                "mean_reward_per_agent": 0.0,
                "final_min_sep_m": float("nan"),
                "collision_ticks": 0,
                "near_miss_ticks": 0,
                "finite_observations": False,
                "finite_rewards": False,
                "api_error": f"{type(exc).__name__}: {exc}",
            }
        finally:
            env.close()
        rows.append(row)
        for check in _row_checks(row):
            checks.append({**check, "lane_id": lane.lane_id, "category": lane.category})

    _write_episode_csv(episode_csv, rows)

    lane_ids = sorted({str(row.get("lane_id")) for row in rows})
    dimensions = sorted({str(row.get("dimension")) for row in rows})
    selected_lane_ids = [lane.lane_id for lane in selected]
    expects_2d = any(lane.category in {"head_on", "crossing"} and "_2d_" in lane.scenario for lane in selected)
    expects_3d = any(lane.category not in {"head_on", "crossing"} or "_3d_" in lane.scenario for lane in selected)
    dimension_coverage_ok = (not expects_2d or "2d" in dimensions) and (not expects_3d or "3d" in dimensions)
    global_checks = [
        _check(
            "run_count",
            len(rows) == len(planned),
            {"expected": len(planned), "actual": len(rows)},
        ),
        _check(
            "validation_lane_coverage",
            set(selected_lane_ids).issubset(set(lane_ids)),
            {"expected": selected_lane_ids, "actual": lane_ids},
        ),
        _check(
            "dimensions_known",
            "unknown" not in set(dimensions),
            {"dimensions": dimensions},
        ),
        _check(
            "selected_dimension_coverage",
            dimension_coverage_ok,
            {"dimensions": dimensions, "expects_2d": expects_2d, "expects_3d": expects_3d},
        ),
        _check(
            "schema_versions_present",
            bool(RL_INTERFACE_VERSION)
            and bool(RL_ACTION_SCHEMA_VERSION)
            and bool(RL_OBSERVATION_SCHEMA_VERSION)
            and bool(RL_REWARD_SCHEMA_VERSION)
            and bool(RL_ROLLOUT_SCHEMA_VERSION),
        ),
    ]
    checks = [{**check, "lane_id": "*", "category": "matrix"} for check in global_checks] + checks
    gate_pass = all(check["ok"] for check in checks if check["severity"] == "gate")
    behavior_pass = all(check["ok"] for check in checks if check["severity"] == "behavior")

    return {
        "schema_version": RL_VALIDATION_MATRIX_SCHEMA_VERSION,
        "rollout_schema_version": RL_ROLLOUT_SCHEMA_VERSION,
        "interface_version": RL_INTERFACE_VERSION,
        "action_schema_version": RL_ACTION_SCHEMA_VERSION,
        "observation_schema_version": RL_OBSERVATION_SCHEMA_VERSION,
        "reward_schema_version": RL_REWARD_SCHEMA_VERSION,
        "plan_only": False,
        "ok": bool(gate_pass),
        "behavior_pass": bool(behavior_pass),
        "policy": policy_name,
        "policy_spec": policy_spec_summary,
        "lanes": [asdict(lane) for lane in selected],
        "planned_run_count": len(planned),
        "run_count": len(rows),
        "max_steps": None if max_steps is None else int(max_steps),
        "matrix": planned,
        "dimensions": dimensions,
        "episodes": rows,
        "checks": checks,
        "episode_csv": str(episode_csv),
        "interface_contract": interface_contract(top_k=8),
    }
