from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from microbench.metrics import append_result, write_summary
from microbench.planners import make_planner
from microbench.scenarios import materialize_official_suite
from microbench.types import AgentState, NeighborObs, PlannerInput, RunSpec
from microbench.runner import run_episode


BASELINE_BEHAVIOR_SCHEMA_VERSION = "0.1"
BASELINE_BEHAVIOR_SUITE = "official_smoke_generated"
BASELINE_BEHAVIOR_METHODS = (
    "baseline_goal",
    "orca_heuristic",
    "orca_with_staleness",
    "cbf_qp",
    "mpc_local",
    "mpc_nonlinear",
    "dmpc_best_response",
    "rmader",
    "ego_swarm",
    "ego_swarm_opt",
    "velocity_obstacle",
    "reciprocal_velocity_obstacle",
    "learned_tiny",
    "intent_dummy",
    "priority_yield",
    "negotiation_yield",
)
BASELINE_BEHAVIOR_SCENARIOS = (
    "head_on_2d_easy",
    "sphere_swap_3d_medium",
)
FINITE_RESULT_FIELDS = (
    "collisions",
    "near_misses",
    "collision_episode",
    "near_miss_episode",
    "min_sep_min_m",
    "min_sep_p05_m",
    "completion_rate",
    "deadlock_time_pct",
    "jerk_mean",
    "planner_ms_per_tick_per_agent_mean",
    "planner_ms_per_tick_per_agent_p95",
    "obs_neighbors_mean",
    "obs_v2v_fraction",
    "comm_agent_msg_delivery_fraction",
    "episode_runtime_s",
)
GUARDRAIL_FIELDS = (
    "planner_timeout_count",
    "planner_error_count",
    "planner_fallback_count",
)
EXPERIMENTAL_SOFT_GUARDRAIL_METHODS = {
    "cbf_qp",
    "mpc_local",
    "mpc_nonlinear",
    "dmpc_best_response",
    "rmader",
    "ego_swarm_opt",
}
SOFT_GUARDRAIL_FIELDS = {"planner_timeout_count", "planner_fallback_count"}
CONTRACT_ONLY_METHODS = {
    # RMADER is an optimizer-grade MINVO/hyperplane baseline. A direct contract
    # probe catches API regressions cheaply; full episode evidence belongs in a
    # capped optimizer review lane rather than the public-alpha smoke loop.
    "rmader",
}


def _as_list(values: tuple[str, ...] | list[str] | None, default: tuple[str, ...]) -> list[str]:
    return [str(v).strip() for v in (values if values is not None else default) if str(v).strip()]


def _float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def _is_finite(value: Any) -> bool:
    out = _float(value)
    return out is not None and math.isfinite(out)


def _sum(rows: list[dict[str, Any]], field: str) -> float:
    total = 0.0
    for row in rows:
        value = _float(row.get(field))
        if value is not None and math.isfinite(value):
            total += value
    return total


def _check(name: str, ok: bool, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "details": details or {}}


def _agent(pos: tuple[float, float, float], vel: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> AgentState:
    return AgentState(
        idx=0,
        pos=np.asarray(pos, dtype=np.float32),
        vel=np.asarray(vel, dtype=np.float32),
        goal=np.asarray([10.0, 0.0, 0.0], dtype=np.float32),
        radius=0.5,
        v_max=3.0,
        a_max=2.0,
    )


def _planner_input(*, neighbors: list[NeighborObs] | None = None, planar: bool = True) -> PlannerInput:
    return PlannerInput(
        ego=_agent((0.0, 0.0, 0.0), vel=(2.0, 0.0, 0.0)),
        goal_dir=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        neighbors=list(neighbors or []),
        dt=0.02,
        t=0.0,
        planar=planar,
    )


def _neighbor() -> NeighborObs:
    return NeighborObs(
        idx=1,
        pos=np.asarray([0.8, 0.0, 0.0], dtype=np.float32),
        vel=np.asarray([-2.0, 0.0, 0.0], dtype=np.float32),
        radius=0.5,
        msg_age_sec=0.0,
        valid=True,
    )


def _planner_output_contracts(methods: list[str]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    if "cbf_qp" in methods:
        try:
            planner = make_planner("cbf_qp")
            planner.reset(0)
            out = planner.compute_cmd(_planner_input(neighbors=[_neighbor()]))
            info = getattr(out, "debug_info", {})
            checks.append(
                _check(
                    "cbf_qp_debug_contract",
                    int(info.get("cbf_constraints", 0)) >= 1
                    and str(info.get("cbf_solver_status", ""))
                    and "cbf_fallback" in info,
                    {
                        "cbf_constraints": info.get("cbf_constraints"),
                        "cbf_solver": info.get("cbf_solver"),
                        "cbf_solver_status": info.get("cbf_solver_status"),
                        "cbf_fallback": info.get("cbf_fallback"),
                    },
                )
            )
        except Exception as exc:
            checks.append(_check("cbf_qp_debug_contract", False, {"error": f"{type(exc).__name__}: {exc}"}))

    if "mpc_local" in methods:
        try:
            planner = make_planner("mpc_local")
            planner.reset(0)
            out = planner.compute_cmd(_planner_input(neighbors=[_neighbor()], planar=False))
            info = getattr(out, "debug_info", {})
            checks.append(
                _check(
                    "mpc_local_debug_contract",
                    int(info.get("mpc_candidates", 0)) > 0
                    and info.get("mpc_min_pred_clearance_m") is not None
                    and info.get("mpc_planar") is False,
                    {
                        "mpc_candidates": info.get("mpc_candidates"),
                        "mpc_min_pred_clearance_m": info.get("mpc_min_pred_clearance_m"),
                        "mpc_planar": info.get("mpc_planar"),
                    },
                )
            )
        except Exception as exc:
            checks.append(_check("mpc_local_debug_contract", False, {"error": f"{type(exc).__name__}: {exc}"}))

    if "mpc_nonlinear" in methods:
        try:
            planner = make_planner("mpc_nonlinear")
            planner.reset(0)
            out = planner.compute_cmd(_planner_input(neighbors=[_neighbor()], planar=False))
            info = getattr(out, "debug_info", {})
            intent = getattr(out, "intent_out", None)
            checks.append(
                _check(
                    "mpc_nonlinear_debug_contract",
                    int(info.get("mpc_nonlinear_horizon_steps", 0)) >= 2
                    and str(info.get("mpc_nonlinear_solver_status", ""))
                    and info.get("mpc_nonlinear_min_swarm_clearance_m") is not None
                    and float(info.get("mpc_nonlinear_cost_reduction", 0.0)) >= -1e-6
                    and info.get("mpc_nonlinear_planar") is False
                    and intent is not None
                    and getattr(intent, "kind", "") == "MPC_NONLINEAR_TRAJECTORY",
                    {
                        "mpc_nonlinear_horizon_steps": info.get("mpc_nonlinear_horizon_steps"),
                        "mpc_nonlinear_solver": info.get("mpc_nonlinear_solver"),
                        "mpc_nonlinear_solver_status": info.get("mpc_nonlinear_solver_status"),
                        "mpc_nonlinear_cost_reduction": info.get("mpc_nonlinear_cost_reduction"),
                        "mpc_nonlinear_min_swarm_clearance_m": info.get("mpc_nonlinear_min_swarm_clearance_m"),
                        "mpc_nonlinear_planar": info.get("mpc_nonlinear_planar"),
                        "intent_kind": getattr(intent, "kind", None),
                    },
                )
            )
        except Exception as exc:
            checks.append(_check("mpc_nonlinear_debug_contract", False, {"error": f"{type(exc).__name__}: {exc}"}))

    if "dmpc_best_response" in methods:
        try:
            planner = make_planner("dmpc_best_response")
            planner.reset(0)
            out = planner.compute_cmd(_planner_input(neighbors=[_neighbor()], planar=False))
            info = getattr(out, "debug_info", {})
            intent = getattr(out, "intent_out", None)
            checks.append(
                _check(
                    "dmpc_best_response_debug_contract",
                    int(info.get("dmpc_best_response_horizon_steps", 0)) >= 2
                    and str(info.get("dmpc_best_response_solver_status", ""))
                    and info.get("dmpc_best_response_min_coupled_clearance_m") is not None
                    and int(info.get("dmpc_best_response_coupled_constraints", 0)) > 0
                    and info.get("dmpc_best_response_planar") is False
                    and intent is not None
                    and getattr(intent, "kind", "") == "DMPC_BEST_RESPONSE_TRAJECTORY"
                    and int(info.get("dmpc_best_response_agent_messages", 0)) > 0,
                    {
                        "dmpc_best_response_horizon_steps": info.get("dmpc_best_response_horizon_steps"),
                        "dmpc_best_response_solver": info.get("dmpc_best_response_solver"),
                        "dmpc_best_response_solver_status": info.get("dmpc_best_response_solver_status"),
                        "dmpc_best_response_coupled_constraints": info.get("dmpc_best_response_coupled_constraints"),
                        "dmpc_best_response_min_coupled_clearance_m": info.get(
                            "dmpc_best_response_min_coupled_clearance_m"
                        ),
                        "dmpc_best_response_planar": info.get("dmpc_best_response_planar"),
                        "dmpc_best_response_agent_messages": info.get("dmpc_best_response_agent_messages"),
                        "intent_kind": getattr(intent, "kind", None),
                    },
                )
            )
        except Exception as exc:
            checks.append(
                _check("dmpc_best_response_debug_contract", False, {"error": f"{type(exc).__name__}: {exc}"})
            )

    if "rmader" in methods:
        try:
            planner = make_planner("rmader")
            planner.reset(0)
            out = planner.compute_cmd(_planner_input(neighbors=[_neighbor()], planar=False))
            info = getattr(out, "debug_info", {})
            intent = getattr(out, "intent_out", None)
            checks.append(
                _check(
                    "rmader_debug_contract",
                    int(info.get("rmader_minvo_intervals", 0)) >= 4
                    and int(info.get("rmader_minvo_control_points_per_interval", 0)) == 4
                    and int(info.get("rmader_hard_constraint_count", 0)) > 0
                    and str(info.get("rmader_solver_status", ""))
                    and info.get("rmader_delay_check_fallback") in {"none", "previous_committed", "braking_trajectory"}
                    and info.get("rmader_planar") is False
                    and intent is not None
                    and getattr(intent, "kind", "") == "RMADER_MINVO_TRAJECTORY"
                    and int(info.get("rmader_agent_messages", 0)) >= 2,
                    {
                        "rmader_minvo_intervals": info.get("rmader_minvo_intervals"),
                        "rmader_solver": info.get("rmader_solver"),
                        "rmader_solver_status": info.get("rmader_solver_status"),
                        "rmader_hard_constraint_count": info.get("rmader_hard_constraint_count"),
                        "rmader_max_hyperplane_violation_m": info.get("rmader_max_hyperplane_violation_m"),
                        "rmader_delay_check_passed": info.get("rmader_delay_check_passed"),
                        "rmader_planar": info.get("rmader_planar"),
                        "rmader_agent_messages": info.get("rmader_agent_messages"),
                        "intent_kind": getattr(intent, "kind", None),
                    },
                )
            )
        except Exception as exc:
            checks.append(_check("rmader_debug_contract", False, {"error": f"{type(exc).__name__}: {exc}"}))

    if "ego_swarm" in methods:
        try:
            planner = make_planner("ego_swarm")
            planner.reset(0)
            out = planner.compute_cmd(_planner_input(neighbors=[_neighbor()], planar=False))
            info = getattr(out, "debug_info", {})
            intent = getattr(out, "intent_out", None)
            checks.append(
                _check(
                    "ego_swarm_debug_contract",
                    int(info.get("ego_swarm_candidates", 0)) > 0
                    and str(info.get("ego_swarm_best_topology", ""))
                    and info.get("ego_swarm_min_swarm_clearance_m") is not None
                    and info.get("ego_swarm_planar") is False
                    and intent is not None
                    and getattr(intent, "kind", "") == "EGO_SWARM_TRAJECTORY",
                    {
                        "ego_swarm_candidates": info.get("ego_swarm_candidates"),
                        "ego_swarm_best_topology": info.get("ego_swarm_best_topology"),
                        "ego_swarm_min_swarm_clearance_m": info.get("ego_swarm_min_swarm_clearance_m"),
                        "ego_swarm_planar": info.get("ego_swarm_planar"),
                        "intent_kind": getattr(intent, "kind", None),
                    },
                )
            )
        except Exception as exc:
            checks.append(_check("ego_swarm_debug_contract", False, {"error": f"{type(exc).__name__}: {exc}"}))

    if "ego_swarm_opt" in methods:
        try:
            planner = make_planner("ego_swarm_opt")
            planner.reset(0)
            out = planner.compute_cmd(_planner_input(neighbors=[_neighbor()], planar=False))
            info = getattr(out, "debug_info", {})
            intent = getattr(out, "intent_out", None)
            checks.append(
                _check(
                    "ego_swarm_opt_debug_contract",
                    int(info.get("ego_swarm_opt_control_points", 0)) >= 5
                    and str(info.get("ego_swarm_opt_solver_status", ""))
                    and info.get("ego_swarm_opt_min_swarm_clearance_m") is not None
                    and float(info.get("ego_swarm_opt_cost_reduction", 0.0)) >= -1e-6
                    and info.get("ego_swarm_opt_planar") is False
                    and intent is not None
                    and getattr(intent, "kind", "") == "EGO_SWARM_OPT_TRAJECTORY",
                    {
                        "ego_swarm_opt_control_points": info.get("ego_swarm_opt_control_points"),
                        "ego_swarm_opt_solver": info.get("ego_swarm_opt_solver"),
                        "ego_swarm_opt_solver_status": info.get("ego_swarm_opt_solver_status"),
                        "ego_swarm_opt_cost_reduction": info.get("ego_swarm_opt_cost_reduction"),
                        "ego_swarm_opt_min_swarm_clearance_m": info.get("ego_swarm_opt_min_swarm_clearance_m"),
                        "ego_swarm_opt_planar": info.get("ego_swarm_opt_planar"),
                        "intent_kind": getattr(intent, "kind", None),
                    },
                )
            )
        except Exception as exc:
            checks.append(_check("ego_swarm_opt_debug_contract", False, {"error": f"{type(exc).__name__}: {exc}"}))

    if "velocity_obstacle" in methods:
        try:
            planner = make_planner("velocity_obstacle")
            planner.reset(0)
            out = planner.compute_cmd(_planner_input(neighbors=[_neighbor()], planar=False))
            info = getattr(out, "debug_info", {})
            checks.append(
                _check(
                    "velocity_obstacle_debug_contract",
                    int(info.get("vo_candidates", 0)) > 0
                    and int(info.get("vo_conflict_count", 0)) >= 1
                    and info.get("vo_min_pred_clearance_m") is not None
                    and info.get("vo_best_clearance_improvement_m") is not None
                    and int(info.get("vo_pred_conflict_candidate_count", 0)) >= 1
                    and info.get("vo_planar") is False,
                    {
                        "vo_candidates": info.get("vo_candidates"),
                        "vo_conflict_count": info.get("vo_conflict_count"),
                        "vo_min_pred_clearance_m": info.get("vo_min_pred_clearance_m"),
                        "vo_best_clearance_improvement_m": info.get("vo_best_clearance_improvement_m"),
                        "vo_pred_conflict_candidate_count": info.get("vo_pred_conflict_candidate_count"),
                        "vo_planar": info.get("vo_planar"),
                    },
                )
            )
        except Exception as exc:
            checks.append(_check("velocity_obstacle_debug_contract", False, {"error": f"{type(exc).__name__}: {exc}"}))

    if "reciprocal_velocity_obstacle" in methods:
        try:
            planner = make_planner("reciprocal_velocity_obstacle")
            planner.reset(0)
            out = planner.compute_cmd(_planner_input(neighbors=[_neighbor()], planar=False))
            info = getattr(out, "debug_info", {})
            checks.append(
                _check(
                    "reciprocal_velocity_obstacle_debug_contract",
                    int(info.get("vo_candidates", 0)) > 0
                    and int(info.get("vo_conflict_count", 0)) >= 1
                    and info.get("vo_min_pred_clearance_m") is not None
                    and info.get("vo_reciprocal_mode") == "hrvo"
                    and info.get("vo_responsibility_mean") is not None
                    and info.get("vo_hrvo_apex_shift_mean") is not None
                    and int(info.get("vo_boundary_candidate_count", 0)) > 0,
                    {
                        "vo_candidates": info.get("vo_candidates"),
                        "vo_conflict_count": info.get("vo_conflict_count"),
                        "vo_min_pred_clearance_m": info.get("vo_min_pred_clearance_m"),
                        "vo_reciprocal_mode": info.get("vo_reciprocal_mode"),
                        "vo_responsibility_mean": info.get("vo_responsibility_mean"),
                        "vo_hrvo_apex_shift_mean": info.get("vo_hrvo_apex_shift_mean"),
                        "vo_boundary_candidate_count": info.get("vo_boundary_candidate_count"),
                    },
                )
            )
        except Exception as exc:
            checks.append(_check("reciprocal_velocity_obstacle_debug_contract", False, {"error": f"{type(exc).__name__}: {exc}"}))

    if "intent_dummy" in methods:
        try:
            planner = make_planner("intent_dummy")
            planner.reset(0)
            out = planner.compute_cmd(_planner_input())
            intent = getattr(out, "intent_out", None)
            checks.append(
                _check(
                    "intent_dummy_intent_contract",
                    intent is not None and getattr(intent, "points", np.empty((0,))).shape[0] >= 2,
                    {
                        "kind": getattr(intent, "kind", None),
                        "num_points": int(getattr(intent, "points", np.empty((0,))).shape[0]) if intent else 0,
                    },
                )
            )
        except Exception as exc:
            checks.append(_check("intent_dummy_intent_contract", False, {"error": f"{type(exc).__name__}: {exc}"}))

    if "learned_tiny" in methods:
        try:
            planner = make_planner("learned_tiny")
            planner.reset(0)
            out = planner.compute_cmd(_planner_input(neighbors=[_neighbor()], planar=False))
            info = getattr(out, "debug_info", {})
            checks.append(
                _check(
                    "learned_tiny_model_contract",
                    bool(info.get("learned_model"))
                    and str(info.get("learned_model_id", ""))
                    and float(info.get("learned_policy_threat_scalar", 0.0)) > 0.0,
                    {
                        "learned_model_id": info.get("learned_model_id"),
                        "learned_policy_action_norm": info.get("learned_policy_action_norm"),
                        "learned_policy_threat_scalar": info.get("learned_policy_threat_scalar"),
                    },
                )
            )
        except Exception as exc:
            checks.append(_check("learned_tiny_model_contract", False, {"error": f"{type(exc).__name__}: {exc}"}))

    return checks


def run_baseline_behavior_smoke(
    *,
    out_dir: str | Path,
    methods: tuple[str, ...] | list[str] | None = None,
    scenario_ids: tuple[str, ...] | list[str] | None = None,
    n_agents: int = 4,
    seed: int = 0,
    comm_profile: str = "ideal_50hz",
) -> dict[str, Any]:
    out = Path(out_dir)
    methods_list = _as_list(methods, BASELINE_BEHAVIOR_METHODS)
    episode_methods = [method for method in methods_list if method not in CONTRACT_ONLY_METHODS]
    contract_only_methods = [method for method in methods_list if method in CONTRACT_ONLY_METHODS]
    scenario_id_list = _as_list(scenario_ids, BASELINE_BEHAVIOR_SCENARIOS)

    results_csv = out / "results.csv"
    if results_csv.exists():
        raise RuntimeError(f"baseline smoke output already exists: {results_csv}")

    generated_dir = out / "_generated_scenarios" / BASELINE_BEHAVIOR_SUITE
    generated = materialize_official_suite(BASELINE_BEHAVIOR_SUITE, generated_dir, overwrite=True)
    manifest = generated["manifest"]
    scenario_meta = {str(entry["id"]): entry for entry in manifest["scenarios"]}
    scenario_paths = {
        Path(path).stem: Path(path)
        for path in generated["scenario_paths"]
    }

    unknown = sorted(set(scenario_id_list) - set(scenario_paths))
    if unknown:
        raise ValueError(f"Unknown scenario(s) for {BASELINE_BEHAVIOR_SUITE}: {','.join(unknown)}")

    rows: list[dict[str, Any]] = []
    for scenario_id in scenario_id_list:
        for method in episode_methods:
            spec = RunSpec(
                scenario_path=str(scenario_paths[scenario_id]),
                method=method,
                n_agents=int(n_agents),
                seed=int(seed),
                comm_profile=str(comm_profile),
                out_dir=str(out),
                save_trace=False,
            )
            row = run_episode(spec)
            append_result(out, row)
            rows.append(row)

    summary_csv = write_summary(out)

    checks: list[dict[str, Any]] = []
    expected_runs = len(episode_methods) * len(scenario_id_list)
    checks.append(
        _check(
            "run_count",
            len(rows) == expected_runs,
            {
                "expected": expected_runs,
                "actual": len(rows),
                "episode_methods": episode_methods,
                "contract_only_methods": contract_only_methods,
            },
        )
    )

    missing_finite: list[dict[str, Any]] = []
    for row in rows:
        for field in FINITE_RESULT_FIELDS:
            if not _is_finite(row.get(field)):
                missing_finite.append(
                    {
                        "method": row.get("method"),
                        "scenario": row.get("scenario"),
                        "field": field,
                        "value": row.get(field),
                    }
                )
    checks.append(_check("finite_key_metrics", not missing_finite, {"violations": missing_finite[:20]}))

    planner_error_violations: list[dict[str, Any]] = []
    strict_guardrail_violations: list[dict[str, Any]] = []
    soft_allowed_guardrail_violations: list[dict[str, Any]] = []
    for row in rows:
        for field in GUARDRAIL_FIELDS:
            value = _float(row.get(field))
            if value is None or value != 0.0:
                violation = {
                    "method": row.get("method"),
                    "scenario": row.get("scenario"),
                    "field": field,
                    "value": row.get(field),
                }
                if field == "planner_error_count":
                    planner_error_violations.append(violation)
                    strict_guardrail_violations.append(violation)
                elif row.get("method") in EXPERIMENTAL_SOFT_GUARDRAIL_METHODS and field in SOFT_GUARDRAIL_FIELDS:
                    soft_allowed_guardrail_violations.append(violation)
                else:
                    strict_guardrail_violations.append(violation)
    checks.append(_check("planner_errors_clear", not planner_error_violations, {"violations": planner_error_violations}))
    checks.append(
        _check(
            "public_alpha_guardrails_clear",
            not strict_guardrail_violations,
            {
                "violations": strict_guardrail_violations,
                "soft_allowed_violations": soft_allowed_guardrail_violations,
            },
        )
    )
    checks.append(
        _check(
            "zero_planner_guardrails",
            not strict_guardrail_violations,
            {
                "violations": strict_guardrail_violations,
                "soft_allowed_violations": soft_allowed_guardrail_violations,
            },
        )
    )

    dims_by_method: dict[str, list[str]] = {}
    for method in methods_list:
        dims = sorted(
            {
                str(scenario_meta[scenario_id]["dimension"])
                for scenario_id in scenario_id_list
                if scenario_id in scenario_meta
            }
        )
        dims_by_method[method] = dims
    missing_dims = {method: dims for method, dims in dims_by_method.items() if not {"2d", "3d"}.issubset(set(dims))}
    checks.append(_check("two_d_and_three_d_coverage", not missing_dims, {"dimensions_by_method": dims_by_method}))

    by_method = {method: [row for row in rows if row.get("method") == method] for method in methods_list}
    if "priority_yield" in methods_list:
        priority_rows = by_method.get("priority_yield", [])
        attempted = _sum(priority_rows, "comm_agent_msg_attempted")
        delivered = _sum(priority_rows, "comm_agent_msg_delivered")
        checks.append(
            _check(
                "priority_yield_message_signal",
                attempted > 0.0 and delivered > 0.0,
                {"attempted": attempted, "delivered": delivered},
            )
        )

    if "negotiation_yield" in methods_list:
        negotiation_rows = by_method.get("negotiation_yield", [])
        proposals = _sum(negotiation_rows, "comm_negotiation_proposals")
        acks = _sum(negotiation_rows, "comm_negotiation_acks")
        checks.append(
            _check(
                "negotiation_yield_signal",
                proposals > 0.0 and acks > 0.0,
                {"proposals": proposals, "acks": acks},
            )
        )

    checks.extend(_planner_output_contracts(methods_list))

    projected_rows = [
        {
            "method": row.get("method"),
            "scenario": row.get("scenario"),
            "comm_profile": row.get("comm_profile"),
            "N": row.get("N"),
            "seed": row.get("seed"),
            "collision_episode": row.get("collision_episode"),
            "min_sep_min_m": row.get("min_sep_min_m"),
            "completion_rate": row.get("completion_rate"),
            "planner_ms_per_tick_per_agent_p95": row.get("planner_ms_per_tick_per_agent_p95"),
            "comm_agent_msg_attempted": row.get("comm_agent_msg_attempted"),
            "comm_agent_msg_delivered": row.get("comm_agent_msg_delivered"),
            "comm_negotiation_proposals": row.get("comm_negotiation_proposals"),
            "comm_negotiation_acks": row.get("comm_negotiation_acks"),
            "planner_timeout_count": row.get("planner_timeout_count"),
            "planner_error_count": row.get("planner_error_count"),
            "planner_fallback_count": row.get("planner_fallback_count"),
        }
        for row in rows
    ]

    report = {
        "schema_version": BASELINE_BEHAVIOR_SCHEMA_VERSION,
        "suite": BASELINE_BEHAVIOR_SUITE,
        "methods": methods_list,
        "episode_methods": episode_methods,
        "contract_only_methods": contract_only_methods,
        "scenario_ids": scenario_id_list,
        "n_agents": int(n_agents),
        "seed": int(seed),
        "comm_profile": str(comm_profile),
        "run_count": len(rows),
        "ok": all(check["ok"] for check in checks),
        "checks": checks,
        "rows": projected_rows,
        "results_csv": str(results_csv),
        "summary_csv": str(summary_csv),
        "suite_manifest": str(generated["manifest_path"]),
    }
    return report


def write_baseline_behavior_smoke(*, out_dir: str | Path, **kwargs: Any) -> Path:
    report = run_baseline_behavior_smoke(out_dir=out_dir, **kwargs)
    path = Path(out_dir) / "baseline_smoke.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
