from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from microbench.planners.cbf_qp import CbfQpPlanner
from microbench.planners.mpc_local import MpcLocalPlanner
from microbench.types import AABBObs, AgentState, NeighborObs, PlannerInput


BASELINE_EVIDENCE_SCHEMA_VERSION = "0.1"
DEFAULT_MPC_PROFILE_ITERS = 20
DEFAULT_MPC_P95_MAX_MS = 50.0


def _agent(
    *,
    idx: int = 0,
    pos: tuple[float, float, float],
    vel: tuple[float, float, float] = (0.0, 0.0, 0.0),
    goal: tuple[float, float, float] = (10.0, 0.0, 0.0),
    radius: float = 0.5,
    v_max: float = 3.0,
    a_max: float = 2.0,
) -> AgentState:
    return AgentState(
        idx=idx,
        pos=np.asarray(pos, dtype=np.float32),
        vel=np.asarray(vel, dtype=np.float32),
        goal=np.asarray(goal, dtype=np.float32),
        radius=float(radius),
        v_max=float(v_max),
        a_max=float(a_max),
    )


def _neighbor(
    *,
    idx: int,
    pos: tuple[float, float, float],
    vel: tuple[float, float, float] = (0.0, 0.0, 0.0),
    radius: float = 0.5,
    msg_age_sec: float = 0.0,
    source: str = "v2v",
) -> NeighborObs:
    return NeighborObs(
        idx=idx,
        pos=np.asarray(pos, dtype=np.float32),
        vel=np.asarray(vel, dtype=np.float32),
        radius=float(radius),
        msg_age_sec=float(msg_age_sec),
        valid=True,
        source=source,
        track_age_sec=float(msg_age_sec),
        stale=bool(msg_age_sec > 0.0),
    )


def _planner_input(
    *,
    ego: AgentState,
    goal_dir: tuple[float, float, float] = (1.0, 0.0, 0.0),
    neighbors: list[NeighborObs] | None = None,
    obstacles: list[AABBObs] | None = None,
    planar: bool = True,
) -> PlannerInput:
    return PlannerInput(
        ego=ego,
        goal_dir=np.asarray(goal_dir, dtype=np.float32),
        neighbors=list(neighbors or []),
        obstacles=list(obstacles or []),
        dt=0.02,
        t=0.0,
        planar=bool(planar),
    )


def _check(method: str, name: str, ok: bool, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "method": method,
        "name": name,
        "ok": bool(ok),
        "details": details or {},
    }


def _finite_vec(v: np.ndarray | list[float] | tuple[float, ...]) -> bool:
    arr = np.asarray(v, dtype=float)
    return arr.shape == (3,) and bool(np.all(np.isfinite(arr)))


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(math.ceil(0.95 * len(ordered))) - 1
    return float(ordered[max(0, min(len(ordered) - 1, idx))])


def _cbf_evidence_checks() -> list[dict[str, Any]]:
    method = "cbf_qp"
    checks: list[dict[str, Any]] = []

    feasible_ego = _agent(pos=(0.0, 0.0, 0.0))
    feasible_neighbor = _neighbor(idx=1, pos=(3.0, 0.0, 0.0))
    feasible_out = CbfQpPlanner(cfg={"solver": "projection"}).compute_cmd(
        _planner_input(ego=feasible_ego, neighbors=[feasible_neighbor])
    )
    checks.append(
        _check(
            method,
            "cbf_projection_feasible_constraint",
            bool(
                feasible_out.debug_info.get("cbf_solver") == "projection_skeleton"
                and feasible_out.debug_info.get("cbf_fallback") is False
                and float(feasible_out.debug_info.get("cbf_max_violation", 1.0)) <= 1e-4
                and 0.0 < float(feasible_out.v_cmd[0]) < float(feasible_ego.v_max)
                and abs(float(feasible_out.v_cmd[1])) <= 1e-9
                and _finite_vec(feasible_out.v_cmd)
            ),
            {
                "v_cmd": [float(x) for x in feasible_out.v_cmd],
                "debug_info": feasible_out.debug_info,
            },
        )
    )

    fallback_ego = _agent(pos=(0.0, 0.0, 0.0), vel=(2.0, 0.0, 0.0))
    fallback_neighbor = _neighbor(idx=1, pos=(0.8, 0.0, 0.0), vel=(-2.0, 0.0, 0.0))
    fallback_out = CbfQpPlanner(cfg={"solver": "projection", "max_projection_iters": 0}).compute_cmd(
        _planner_input(ego=fallback_ego, neighbors=[fallback_neighbor])
    )
    checks.append(
        _check(
            method,
            "cbf_forced_fallback_is_bounded_and_reported",
            bool(
                fallback_out.debug_info.get("cbf_solver") == "projection_skeleton"
                and fallback_out.debug_info.get("cbf_fallback") is True
                and float(fallback_out.debug_info.get("cbf_max_violation", 0.0)) > 0.0
                and float(np.linalg.norm(fallback_out.v_cmd)) <= float(fallback_ego.v_max) + 1e-6
                and _finite_vec(fallback_out.v_cmd)
            ),
            {
                "v_cmd": [float(x) for x in fallback_out.v_cmd],
                "debug_info": fallback_out.debug_info,
            },
        )
    )

    auto_out = CbfQpPlanner(cfg={"solver": "auto"}).compute_cmd(
        _planner_input(ego=feasible_ego, neighbors=[feasible_neighbor])
    )
    checks.append(
        _check(
            method,
            "cbf_auto_solver_path_reports_status",
            bool(
                auto_out.debug_info.get("cbf_solver") in {"scipy_slsqp", "projection_skeleton"}
                and auto_out.debug_info.get("cbf_solver_requested") == "auto"
                and str(auto_out.debug_info.get("cbf_solver_status", ""))
                and _finite_vec(auto_out.v_cmd)
            ),
            {
                "v_cmd": [float(x) for x in auto_out.v_cmd],
                "debug_info": auto_out.debug_info,
            },
        )
    )

    return checks


def _dense_mpc_input() -> PlannerInput:
    ego = _agent(
        pos=(0.0, 0.0, 0.0),
        vel=(1.0, 0.0, 0.0),
        goal=(12.0, 2.0, 0.0),
        v_max=3.0,
        a_max=2.0,
    )
    neighbors: list[NeighborObs] = []
    for i in range(12):
        angle = 2.0 * math.pi * i / 12.0
        radius = 1.0 + 0.1 * (i % 3)
        x = 1.8 + 0.2 * (i % 4)
        y = math.sin(angle) * radius
        z = math.cos(angle) * radius
        neighbors.append(
            _neighbor(
                idx=i + 1,
                pos=(x, y, z),
                vel=(-0.35 + 0.05 * (i % 2), -0.05 * math.sin(angle), -0.05 * math.cos(angle)),
                msg_age_sec=0.08 * (i % 4),
                source="fused" if i % 2 else "sensor",
            )
        )
    obstacles = [
        AABBObs(
            center=np.asarray([2.4, 0.45, 0.0], dtype=np.float32),
            half=np.asarray([0.25, 0.35, 0.45], dtype=np.float32),
        )
    ]
    return _planner_input(
        ego=ego,
        goal_dir=(0.98, 0.16, 0.0),
        neighbors=neighbors,
        obstacles=obstacles,
        planar=False,
    )


def _mpc_evidence_checks(*, profile_iters: int, p95_max_ms: float) -> list[dict[str, Any]]:
    method = "mpc_local"
    checks: list[dict[str, Any]] = []
    planner = MpcLocalPlanner(
        cfg={
            "candidate_samples_3d": 64,
            "max_candidates": 24,
            "horizon_s": 1.5,
            "rollout_dt_s": 0.5,
            "direction_scales": (1.0, 0.5),
        }
    )
    inp = _dense_mpc_input()
    out = planner.compute_cmd(inp)
    debug = out.debug_info
    accel_limit = float(inp.ego.a_max) * float(inp.dt)

    checks.append(
        _check(
            method,
            "mpc_dense_3d_candidate_cap_and_signals",
            bool(
                debug.get("mpc_planar") is False
                and int(debug.get("mpc_candidates", 9999)) <= 24
                and int(debug.get("mpc_candidates_raw", 0)) >= int(debug.get("mpc_candidates", 0))
                and float(debug.get("mpc_accel_delta_norm", 9999.0)) <= accel_limit + 1e-6
                and debug.get("mpc_min_pred_clearance_m") is not None
                and _finite_vec(out.v_cmd)
            ),
            {
                "v_cmd": [float(x) for x in out.v_cmd],
                "debug_info": debug,
                "neighbor_count": len(inp.neighbors),
                "obstacle_count": len(inp.obstacles),
            },
        )
    )

    for _ in range(2):
        planner.compute_cmd(inp)

    samples_ms: list[float] = []
    for _ in range(max(1, int(profile_iters))):
        start = time.perf_counter()
        planner.compute_cmd(inp)
        samples_ms.append((time.perf_counter() - start) * 1000.0)
    p95_ms = _p95(samples_ms)
    checks.append(
        _check(
            method,
            "mpc_dense_3d_profile_p95_bounded",
            bool(p95_ms <= float(p95_max_ms)),
            {
                "profile_iters": int(max(1, int(profile_iters))),
                "p50_ms": float(np.median(samples_ms)) if samples_ms else 0.0,
                "p95_ms": p95_ms,
                "max_ms": max(samples_ms) if samples_ms else 0.0,
                "threshold_ms": float(p95_max_ms),
            },
        )
    )

    return checks


def run_baseline_reference_evidence(
    *,
    mpc_profile_iters: int = DEFAULT_MPC_PROFILE_ITERS,
    mpc_p95_max_ms: float = DEFAULT_MPC_P95_MAX_MS,
) -> dict[str, Any]:
    checks = [
        *_cbf_evidence_checks(),
        *_mpc_evidence_checks(profile_iters=mpc_profile_iters, p95_max_ms=mpc_p95_max_ms),
    ]
    failed = [check for check in checks if not check["ok"]]
    return {
        "schema_version": BASELINE_EVIDENCE_SCHEMA_VERSION,
        "evidence_type": "cbf_mpc_reference_evidence",
        "methods": ["cbf_qp", "mpc_local"],
        "ok": not failed,
        "summary": {
            "check_count": len(checks),
            "failed_count": len(failed),
            "cbf_check_count": sum(1 for check in checks if check["method"] == "cbf_qp"),
            "mpc_check_count": sum(1 for check in checks if check["method"] == "mpc_local"),
        },
        "checks": checks,
        "promotion_recommendations": {
            "cbf_qp": "keep_experimental_until_solver_backends_and_infeasible_constraint_behavior_are_validated_beyond_skeleton_cases",
            "mpc_local": "keep_experimental_until_dense_3d_compute_bands_and_stress_behavior_are_calibrated_on_official_suites",
        },
    }


def write_baseline_reference_evidence(*, out_dir: str | Path, **kwargs: Any) -> Path:
    report = run_baseline_reference_evidence(**kwargs)
    path = Path(out_dir) / "baseline_evidence.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
