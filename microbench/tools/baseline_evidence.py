from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from microbench.planners.cbf_qp import CbfQpPlanner
from microbench.planners.ego_swarm_opt import EgoSwarmOptimizingPlanner
from microbench.planners.mpc_local import MpcLocalPlanner
from microbench.planners.mpc_nonlinear import NonlinearMpcPlanner
from microbench.planners.rmader import RmaderPlanner
from microbench.planners.velocity_obstacle import ReciprocalVelocityObstaclePlanner, VelocityObstaclePlanner
from microbench.runner import run_episode
from microbench.types import AABBObs, AgentContext, AgentState, IntentObs, NeighborObs, PlannerInput, RunSpec


BASELINE_EVIDENCE_SCHEMA_VERSION = "0.3"
DEFAULT_MPC_PROFILE_ITERS = 20
DEFAULT_MPC_P95_MAX_MS = 50.0
DEFAULT_OPT_PROFILE_ITERS = 8
DEFAULT_OPT_P95_MAX_MS = 80.0


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
    neighbor_intents: list[IntentObs] | None = None,
    planar: bool = True,
) -> PlannerInput:
    return PlannerInput(
        ego=ego,
        goal_dir=np.asarray(goal_dir, dtype=np.float32),
        neighbors=list(neighbors or []),
        obstacles=list(obstacles or []),
        neighbor_intents=list(neighbor_intents or []),
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
                feasible_out.debug_info.get("cbf_solver") == "deterministic_projection"
                and feasible_out.debug_info.get("cbf_fallback") is False
                and float(feasible_out.debug_info.get("cbf_max_violation", 1.0)) <= 1e-4
                and int(feasible_out.debug_info.get("cbf_neighbor_constraints", 0)) == 1
                and int(feasible_out.debug_info.get("cbf_obstacle_constraints", -1)) == 0
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
                fallback_out.debug_info.get("cbf_solver") == "deterministic_projection"
                and fallback_out.debug_info.get("cbf_fallback") is True
                and float(fallback_out.debug_info.get("cbf_max_violation", 0.0)) > 0.0
                and float(fallback_out.debug_info.get("cbf_pre_fallback_max_violation", 0.0)) > 0.0
                and float(np.linalg.norm(fallback_out.v_cmd)) <= float(fallback_ego.v_max) + 1e-6
                and _finite_vec(fallback_out.v_cmd)
            ),
            {
                "v_cmd": [float(x) for x in fallback_out.v_cmd],
                "debug_info": fallback_out.debug_info,
            },
        )
    )

    stale_planner = CbfQpPlanner(
        cfg={
            "solver": "projection",
            "stale_inflation_gain": 1.0,
            "track_uncertainty_speed_gain": 0.0,
            "stale_age_cap_s": 2.0,
        }
    )
    stale_ego = _agent(pos=(0.0, 0.0, 0.0), vel=(1.0, 0.0, 0.0))
    fresh_track = _neighbor(idx=1, pos=(2.4, 0.0, 0.0), msg_age_sec=0.0)
    stale_track = _neighbor(idx=1, pos=(2.4, 0.0, 0.0), msg_age_sec=1.0)
    fresh_out = stale_planner.compute_cmd(_planner_input(ego=stale_ego, neighbors=[fresh_track]))
    stale_out = stale_planner.compute_cmd(_planner_input(ego=stale_ego, neighbors=[stale_track]))
    checks.append(
        _check(
            method,
            "cbf_stale_track_inflates_barrier",
            bool(
                float(stale_out.v_cmd[0]) < float(fresh_out.v_cmd[0])
                and float(stale_out.debug_info.get("cbf_uncertainty_inflation_max_m", 0.0)) > 0.9
                and float(fresh_out.debug_info.get("cbf_uncertainty_inflation_max_m", 1.0)) == 0.0
                and stale_out.debug_info.get("cbf_min_clearance_m") is not None
                and fresh_out.debug_info.get("cbf_min_clearance_m") is not None
                and float(stale_out.debug_info["cbf_min_clearance_m"]) < float(fresh_out.debug_info["cbf_min_clearance_m"])
            ),
            {
                "fresh_v_cmd": [float(x) for x in fresh_out.v_cmd],
                "stale_v_cmd": [float(x) for x in stale_out.v_cmd],
                "fresh_debug_info": fresh_out.debug_info,
                "stale_debug_info": stale_out.debug_info,
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
                auto_out.debug_info.get("cbf_solver") in {"scipy_slsqp", "deterministic_projection"}
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
                and int(debug.get("mpc_neighbor_count_considered", 0)) == 8
                and int(debug.get("mpc_obstacle_count_considered", 0)) == 1
                and int(debug.get("mpc_pred_collision_candidate_count", 0)) > 0
                and debug.get("mpc_best_clearance_improvement_m") is not None
                and float(debug.get("mpc_best_clearance_improvement_m", -9999.0)) > 0.0
                and float(debug.get("mpc_stale_inflation_max_m", 0.0)) > 0.0
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

    stale_planner = MpcLocalPlanner(
        cfg={
            "candidate_samples_2d": 8,
            "stale_inflation_gain": 1.0,
            "track_uncertainty_speed_gain": 0.0,
            "stale_age_cap_s": 2.0,
        }
    )
    stale_ego = _agent(pos=(0.0, 0.0, 0.0), vel=(1.0, 0.0, 0.0))
    fresh_track = _neighbor(idx=1, pos=(2.4, 0.0, 0.0), msg_age_sec=0.0)
    stale_track = _neighbor(idx=1, pos=(2.4, 0.0, 0.0), msg_age_sec=1.0)
    fresh_out = stale_planner.compute_cmd(_planner_input(ego=stale_ego, neighbors=[fresh_track]))
    stale_out = stale_planner.compute_cmd(_planner_input(ego=stale_ego, neighbors=[stale_track]))
    checks.append(
        _check(
            method,
            "mpc_stale_track_inflates_rollout_risk",
            bool(
                float(fresh_out.debug_info.get("mpc_stale_inflation_max_m", 1.0)) == 0.0
                and float(stale_out.debug_info.get("mpc_stale_inflation_max_m", 0.0)) > 0.9
                and stale_out.debug_info.get("mpc_min_pred_clearance_m") is not None
                and fresh_out.debug_info.get("mpc_min_pred_clearance_m") is not None
                and float(stale_out.debug_info["mpc_min_pred_clearance_m"])
                < float(fresh_out.debug_info["mpc_min_pred_clearance_m"])
                and float(stale_out.debug_info.get("mpc_collision_penalty", 0.0))
                > float(fresh_out.debug_info.get("mpc_collision_penalty", 0.0))
            ),
            {
                "fresh_v_cmd": [float(x) for x in fresh_out.v_cmd],
                "stale_v_cmd": [float(x) for x in stale_out.v_cmd],
                "fresh_debug_info": fresh_out.debug_info,
                "stale_debug_info": stale_out.debug_info,
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


def _optimizer_dense_input() -> PlannerInput:
    ego = _agent(
        pos=(0.0, 0.0, 0.0),
        vel=(1.4, 0.15, 0.0),
        goal=(11.0, 2.5, 1.0),
        v_max=3.2,
        a_max=2.4,
    )
    neighbors = [
        _neighbor(idx=1, pos=(3.6, 0.2, 0.0), vel=(-1.1, 0.0, 0.05), msg_age_sec=0.1, source="fused"),
        _neighbor(idx=2, pos=(4.0, 1.1, 0.7), vel=(-0.8, -0.1, -0.1), msg_age_sec=0.35, source="v2v"),
        _neighbor(idx=3, pos=(3.2, -0.8, -0.8), vel=(-0.7, 0.2, 0.2), msg_age_sec=0.6, source="sensor"),
        _neighbor(idx=4, pos=(4.6, 0.6, -0.9), vel=(-0.9, -0.1, 0.15), msg_age_sec=0.2, source="fused"),
    ]
    obstacles = [
        AABBObs(
            center=np.asarray([3.2, 0.7, 0.25], dtype=np.float32),
            half=np.asarray([0.35, 0.45, 0.55], dtype=np.float32),
        ),
        AABBObs(
            center=np.asarray([4.8, 1.8, 0.85], dtype=np.float32),
            half=np.asarray([0.4, 0.6, 0.35], dtype=np.float32),
        ),
    ]
    intent = IntentObs(
        sender_id=8,
        points=np.asarray(
            [
                [1.2, 0.1, 0.0],
                [2.1, 0.4, 0.2],
                [3.0, 0.8, 0.4],
                [4.0, 1.1, 0.6],
            ],
            dtype=np.float32,
        ),
        tube_radius_m=0.8,
        kind="OPTIMIZER_EVIDENCE_TRAJECTORY",
        expiry_s=1.0,
        intent_age_s=0.25,
        valid=True,
        dt_plan_s=0.4,
    )
    return _planner_input(
        ego=ego,
        goal_dir=(0.95, 0.24, 0.09),
        neighbors=neighbors,
        obstacles=obstacles,
        neighbor_intents=[intent],
        planar=False,
    )


def _optimizer_degraded_input() -> PlannerInput:
    inp = _optimizer_dense_input()
    degraded_neighbors = []
    for nobs in inp.neighbors:
        degraded_neighbors.append(
            _neighbor(
                idx=int(nobs.idx),
                pos=tuple(float(x) for x in nobs.pos),
                vel=tuple(float(x) for x in nobs.vel),
                radius=float(nobs.radius),
                msg_age_sec=max(float(nobs.msg_age_sec), 0.9),
                source=str(nobs.source),
            )
        )
    degraded_intents = [
        IntentObs(
            sender_id=int(intent.sender_id),
            points=np.asarray(intent.points, dtype=np.float32),
            tube_radius_m=float(intent.tube_radius_m),
            kind=str(intent.kind),
            expiry_s=float(intent.expiry_s),
            intent_age_s=1.2,
            valid=bool(intent.valid),
            dt_plan_s=intent.dt_plan_s,
            mode=intent.mode,
        )
        for intent in inp.neighbor_intents
    ]
    return PlannerInput(
        ego=inp.ego,
        goal_dir=inp.goal_dir,
        neighbors=degraded_neighbors,
        obstacles=inp.obstacles,
        neighbor_intents=degraded_intents,
        dt=inp.dt,
        t=inp.t,
        planar=inp.planar,
    )


def _profile_planner(planner, inp: PlannerInput, *, iters: int) -> dict[str, float]:
    samples_ms: list[float] = []
    for _ in range(2):
        planner.compute_cmd(inp)
    for _ in range(max(1, int(iters))):
        start = time.perf_counter()
        planner.compute_cmd(inp)
        samples_ms.append((time.perf_counter() - start) * 1000.0)
    return {
        "profile_iters": int(max(1, int(iters))),
        "p50_ms": float(np.median(samples_ms)) if samples_ms else 0.0,
        "p95_ms": _p95(samples_ms),
        "max_ms": max(samples_ms) if samples_ms else 0.0,
    }


def _optimizer_solver_mode_check(method: str, planner_cls, debug_prefix: str, intent_kind: str) -> dict[str, Any]:
    inp = _optimizer_dense_input()
    planner = planner_cls(cfg={"solver": "scipy_l_bfgs_b", "scipy_maxiter": 6, "opt_iterations": 4})
    planner.reset(0)
    out = planner.compute_cmd(inp)
    debug = out.debug_info
    solver = str(debug.get(f"{debug_prefix}_solver", ""))
    status = str(debug.get(f"{debug_prefix}_solver_status", ""))
    reduction = float(debug.get(f"{debug_prefix}_cost_reduction", 0.0))
    return _check(
        method,
        f"{method}_scipy_or_fallback_solver_reports_status",
        bool(
            solver in {"scipy_l_bfgs_b", "projected_gradient"}
            and status
            and reduction >= -1e-6
            and out.intent_out is not None
            and getattr(out.intent_out, "kind", "") == intent_kind
            and _finite_vec(out.v_cmd)
        ),
        {
            "solver": solver,
            "solver_status": status,
            "cost_reduction": reduction,
            "intent_kind": getattr(out.intent_out, "kind", None),
            "v_cmd": [float(x) for x in out.v_cmd],
            "debug_info": debug,
        },
    )


def _optimizer_episode_trace_checks(*, artifact_dir: str | Path | None, save_traces: bool) -> list[dict[str, Any]]:
    if not save_traces:
        return []
    if artifact_dir is None:
        return [
            _check(
                "optimizer_grade",
                "optimizer_trace_artifact_dir_configured",
                False,
                {"error": "save_traces requested without artifact_dir"},
            )
        ]

    out_root = Path(artifact_dir) / "optimizer_traces"
    scenario_path = out_root / "optimizer_trace_urban_conflict_short.yaml"
    scenario_path.parent.mkdir(parents=True, exist_ok=True)
    scenario_path.write_text(
        """
scenario:
  name: "optimizer_trace_urban_conflict_short"
  duration_s: 0.6
world:
  planar: false
  bounds:
    xmin: -55.0
    xmax: 55.0
    ymin: -8.0
    ymax: 24.0
    zmin: -55.0
    zmax: 55.0
agent_params:
  radius_m: 0.6
  v_max_mps: 3.2
  a_max_mps2: 2.2
  goal_tolerance_m: 1.2
goals:
  min_goal_distance_m: 70.0
spawn:
  type: "four_way"
  extent_m: 46.0
  lane_half_width_m: 2.2
  y_m: 6.0
  min_start_separation_m: 6.0
  start_layers_m: [6.0, 6.0, 6.5, 5.5]
  goal_layers_m: [6.0, 6.0, 5.5, 6.5]
obstacles:
  - kind: "building"
    label: "tower A"
    aabb:
      center: [-14.0, 5.0, -12.0]
      half: [4.0, 10.0, 5.0]
  - kind: "building"
    label: "tower B"
    aabb:
      center: [14.0, 5.0, 12.0]
      half: [5.0, 11.0, 4.0]
  - kind: "no_fly_volume"
    label: "offset crane"
    aabb:
      center: [4.5, 6.0, -4.5]
      half: [1.8, 12.0, 1.8]
perception:
  mode: "fused"
  sensor:
    range_m: 30.0
    fov_deg: 150.0
    occlusion: true
    occlusion_margin_m: 0.4
    false_negative_p: 0.01
    noise_sigma_pos_m: 0.04
    noise_sigma_vel_mps: 0.04
    track_ttl_s: 0.35
intent:
  enabled: true
  tx_rate_hz: 10.0
  max_points: 10
  age_cap_s: 0.75
visual:
  environment: "urban_airspace"
  ground_y_m: -4.0
  road_width_m: 8.0
  show_sensor_ranges: false
logging:
  save_trace: true
  trace_save_failures_only: false
  trace_max_steps: 200
  save_events: false
  save_trace_on_collision: false
benchmark:
  family: "urban_airspace"
  dimension: "3d"
  difficulty: "evidence_smoke"
  purpose: "Compact optimizer-grade trace artifact for Foxglove export review."
""".strip()
        + "\n",
        encoding="utf-8",
    )
    methods = ("mpc_nonlinear", "ego_swarm_opt")
    checks: list[dict[str, Any]] = []
    for method in methods:
        run_dir = out_root / method
        row = run_episode(
            RunSpec(
                scenario_path=str(scenario_path),
                method=method,
                n_agents=4,
                seed=2,
                comm_profile="realistic_v2v_50hz",
                out_dir=str(run_dir),
                save_trace=True,
            )
        )
        trace_paths = sorted((run_dir / "episodes").glob("*/trace_episode.jsonl"))
        trace_path = trace_paths[0] if trace_paths else None
        checks.append(
            _check(
                method,
                f"{method}_foxglove_trace_jsonl_written",
                bool(
                    trace_path is not None
                    and trace_path.exists()
                    and trace_path.stat().st_size > 0
                    and float(row.get("planner_error_count", 1.0) or 0.0) == 0.0
                    and float(row.get("planner_timeout_count", 1.0) or 0.0) == 0.0
                ),
                {
                    "trace_path": str(trace_path) if trace_path is not None else None,
                    "foxglove_export_command": (
                        f"python -m microbench.cli foxglove-export --trace {trace_path} --out "
                        f"{Path(trace_path).with_suffix('.mcap') if trace_path is not None else '<trace>.mcap'}"
                    )
                    if trace_path is not None
                    else None,
                    "row": row,
                },
            )
        )
    return checks


def _optimizer_evidence_checks(
    *,
    profile_iters: int,
    p95_max_ms: float,
    artifact_dir: str | Path | None,
    save_traces: bool,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    dense = _optimizer_dense_input()
    degraded = _optimizer_degraded_input()

    nmpc = NonlinearMpcPlanner(cfg={"max_initializations": 4, "opt_iterations": 8})
    nmpc.reset(0)
    nmpc_out = nmpc.compute_cmd(dense)
    nmpc_debug = nmpc_out.debug_info
    checks.append(
        _check(
            "mpc_nonlinear",
            "mpc_nonlinear_dense_3d_optimizer_signals",
            bool(
                nmpc_debug.get("mpc_nonlinear_planar") is False
                and int(nmpc_debug.get("mpc_nonlinear_horizon_steps", 0)) >= 2
                and int(nmpc_debug.get("mpc_nonlinear_initializations", 0)) >= 2
                and float(nmpc_debug.get("mpc_nonlinear_cost_reduction", 0.0)) > 0.0
                and nmpc_debug.get("mpc_nonlinear_min_swarm_clearance_m") is not None
                and nmpc_debug.get("mpc_nonlinear_min_obstacle_clearance_m") is not None
                and float(nmpc_debug.get("mpc_nonlinear_intent_penalty", 0.0)) > 0.0
                and nmpc_out.intent_out is not None
                and nmpc_out.intent_out.kind == "MPC_NONLINEAR_TRAJECTORY"
                and _finite_vec(nmpc_out.v_cmd)
            ),
            {"v_cmd": [float(x) for x in nmpc_out.v_cmd], "debug_info": nmpc_debug},
        )
    )

    nmpc_fresh = NonlinearMpcPlanner(cfg={"max_initializations": 3, "opt_iterations": 4}).compute_cmd(dense)
    nmpc_degraded = NonlinearMpcPlanner(cfg={"max_initializations": 3, "opt_iterations": 4}).compute_cmd(degraded)
    checks.append(
        _check(
            "mpc_nonlinear",
            "mpc_nonlinear_degraded_intent_and_v2v_inflate_risk",
            bool(
                float(nmpc_degraded.debug_info.get("mpc_nonlinear_collision_penalty", 0.0))
                >= float(nmpc_fresh.debug_info.get("mpc_nonlinear_collision_penalty", 0.0))
                and float(nmpc_degraded.debug_info.get("mpc_nonlinear_intent_penalty", 0.0))
                >= float(nmpc_fresh.debug_info.get("mpc_nonlinear_intent_penalty", 0.0))
            ),
            {
                "fresh_debug_info": nmpc_fresh.debug_info,
                "degraded_debug_info": nmpc_degraded.debug_info,
            },
        )
    )

    ego = EgoSwarmOptimizingPlanner(cfg={"max_initializations": 4, "opt_iterations": 6})
    ego.reset(0)
    ego_out = ego.compute_cmd(dense)
    ego_debug = ego_out.debug_info
    checks.append(
        _check(
            "ego_swarm_opt",
            "ego_swarm_opt_dense_3d_optimizer_signals",
            bool(
                ego_debug.get("ego_swarm_opt_planar") is False
                and int(ego_debug.get("ego_swarm_opt_control_points", 0)) >= 5
                and int(ego_debug.get("ego_swarm_opt_initializations", 0)) >= 2
                and float(ego_debug.get("ego_swarm_opt_cost_reduction", 0.0)) >= -1e-6
                and ego_debug.get("ego_swarm_opt_min_swarm_clearance_m") is not None
                and ego_debug.get("ego_swarm_opt_min_obstacle_clearance_m") is not None
                and ego_out.intent_out is not None
                and ego_out.intent_out.kind == "EGO_SWARM_OPT_TRAJECTORY"
                and _finite_vec(ego_out.v_cmd)
            ),
            {"v_cmd": [float(x) for x in ego_out.v_cmd], "debug_info": ego_debug},
        )
    )

    ego_fresh = EgoSwarmOptimizingPlanner(cfg={"max_initializations": 3, "opt_iterations": 4}).compute_cmd(dense)
    ego_degraded = EgoSwarmOptimizingPlanner(cfg={"max_initializations": 3, "opt_iterations": 4}).compute_cmd(degraded)
    checks.append(
        _check(
            "ego_swarm_opt",
            "ego_swarm_opt_degraded_intent_and_v2v_inflate_risk",
            bool(
                float(ego_degraded.debug_info.get("ego_swarm_opt_swarm_penalty", 0.0))
                >= float(ego_fresh.debug_info.get("ego_swarm_opt_swarm_penalty", 0.0))
            ),
            {
                "fresh_debug_info": ego_fresh.debug_info,
                "degraded_debug_info": ego_degraded.debug_info,
            },
        )
    )

    checks.append(
        _optimizer_solver_mode_check(
            "mpc_nonlinear",
            NonlinearMpcPlanner,
            "mpc_nonlinear",
            "MPC_NONLINEAR_TRAJECTORY",
        )
    )
    checks.append(
        _optimizer_solver_mode_check(
            "ego_swarm_opt",
            EgoSwarmOptimizingPlanner,
            "ego_swarm_opt",
            "EGO_SWARM_OPT_TRAJECTORY",
        )
    )

    nmpc_profile = _profile_planner(NonlinearMpcPlanner(cfg={"max_initializations": 3, "opt_iterations": 5}), dense, iters=profile_iters)
    ego_profile = _profile_planner(EgoSwarmOptimizingPlanner(cfg={"max_initializations": 3, "opt_iterations": 4}), dense, iters=profile_iters)
    checks.append(
        _check(
            "mpc_nonlinear",
            "mpc_nonlinear_dense_3d_profile_p95_bounded",
            bool(nmpc_profile["p95_ms"] <= float(p95_max_ms)),
            {**nmpc_profile, "threshold_ms": float(p95_max_ms)},
        )
    )
    checks.append(
        _check(
            "ego_swarm_opt",
            "ego_swarm_opt_dense_3d_profile_p95_bounded",
            bool(ego_profile["p95_ms"] <= float(p95_max_ms)),
            {**ego_profile, "threshold_ms": float(p95_max_ms)},
        )
    )

    checks.extend(_optimizer_episode_trace_checks(artifact_dir=artifact_dir, save_traces=save_traces))
    return checks


def _rmader_evidence_checks() -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    accepted = _planner_input(
        ego=_agent(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), goal=(10.0, 0.0, 0.0)),
        goal_dir=(1.0, 0.0, 0.0),
        neighbors=[_neighbor(idx=1, pos=(4.0, 0.0, 0.0))],
        planar=False,
    )
    accepted_planner = RmaderPlanner(
        cfg={
            "horizon_s": 2.4,
            "control_points": 8,
            "samples_per_interval": 2,
            "max_initializations": 1,
            "opt_iterations": 3,
            "hard_projection_iterations": 3,
            "jerk_limit_mps3": 100.0,
        }
    )
    accepted_planner.reset(0)
    accepted_out = accepted_planner.compute_cmd(accepted)
    accepted_debug = accepted_out.debug_info
    checks.append(
        _check(
            "rmader",
            "rmader_minvo_hyperplane_commit_signals",
            bool(
                accepted_debug.get("rmader_planar") is False
                and int(accepted_debug.get("rmader_minvo_intervals", 0)) >= 4
                and int(accepted_debug.get("rmader_minvo_control_points_per_interval", 0)) == 4
                and int(accepted_debug.get("rmader_hard_constraint_count", 0)) > 0
                and accepted_debug.get("rmader_candidate_hard_constraint_ok") is True
                and accepted_debug.get("rmader_delay_check_passed") is True
                and accepted_out.intent_out is not None
                and accepted_out.intent_out.kind == "RMADER_MINVO_TRAJECTORY"
                and len(accepted_out.messages_out) >= 2
                and _finite_vec(accepted_out.v_cmd)
            ),
            {"v_cmd": [float(x) for x in accepted_out.v_cmd], "debug_info": accepted_debug},
        )
    )

    dense_planner = RmaderPlanner(
        cfg={
            "horizon_s": 2.4,
            "control_points": 8,
            "samples_per_interval": 2,
            "max_initializations": 3,
            "opt_iterations": 3,
            "hard_projection_iterations": 3,
            "jerk_limit_mps3": 100.0,
        }
    )
    dense_planner.reset(0)
    dense_out = dense_planner.compute_cmd(_optimizer_dense_input())
    dense_debug = dense_out.debug_info
    checks.append(
        _check(
            "rmader",
            "rmader_dense_delay_check_fallback_reported",
            bool(
                int(dense_debug.get("rmader_hard_constraint_count", 0)) > 0
                and dense_debug.get("rmader_delay_check_fallback") in {"none", "previous_committed", "braking_trajectory"}
                and dense_debug.get("rmader_candidate_max_hyperplane_violation_m") is not None
                and dense_out.intent_out is not None
                and dense_out.intent_out.kind == "RMADER_MINVO_TRAJECTORY"
                and len(dense_out.messages_out) >= 2
                and _finite_vec(dense_out.v_cmd)
            ),
            {"v_cmd": [float(x) for x in dense_out.v_cmd], "debug_info": dense_debug},
        )
    )
    return checks


def _vo_conflict_input(*, stale_age_s: float = 1.0, planar: bool = True, priority: int | None = None) -> PlannerInput:
    ego = _agent(
        idx=2,
        pos=(0.0, 0.0, 0.0),
        vel=(2.0, 0.0, 0.0),
        goal=(12.0, 0.0, 0.0),
        v_max=3.0,
        a_max=2.0,
    )
    neighbor = _neighbor(
        idx=1,
        pos=(5.0, 0.0, 0.0),
        vel=(-2.0, 0.0, 0.0),
        msg_age_sec=float(stale_age_s),
    )
    ctx = None
    if priority is not None:
        ctx = AgentContext(agent_id=2, method="reciprocal_velocity_obstacle", seed=0, priority=int(priority))
    return PlannerInput(
        ego=ego,
        goal_dir=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        neighbors=[neighbor],
        dt=0.02,
        t=0.0,
        obstacles=[],
        planar=bool(planar),
        agent_context=ctx,
    )


def _vo_evidence_checks() -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    vo_method = "velocity_obstacle"
    rvo_method = "reciprocal_velocity_obstacle"

    vo = VelocityObstaclePlanner(cfg={"candidate_samples_2d": 24, "candidate_samples_3d": 36})
    inp = _vo_conflict_input(stale_age_s=1.0, planar=True)
    vo_out = vo.compute_cmd(inp)
    vo_debug = vo_out.debug_info
    checks.append(
        _check(
            vo_method,
            "vo_finite_horizon_cone_signals",
            bool(
                vo_debug.get("vo_algorithm") == "velocity_obstacle_cone_sampling"
                and int(vo_debug.get("vo_candidates", 0)) > 0
                and int(vo_debug.get("vo_pred_conflict_candidate_count", 0)) > 0
                and int(vo_debug.get("vo_safe_candidate_count", 0)) > 0
                and vo_debug.get("vo_min_ttc_s") is not None
                and vo_debug.get("vo_best_clearance_improvement_m") is not None
                and float(vo_debug.get("vo_best_clearance_improvement_m", -9999.0)) > 0.0
                and float(vo_debug.get("vo_stale_inflation_max_m", 0.0)) > 0.0
                and _finite_vec(vo_out.v_cmd)
            ),
            {
                "v_cmd": [float(x) for x in vo_out.v_cmd],
                "debug_info": vo_debug,
            },
        )
    )

    rvo = ReciprocalVelocityObstaclePlanner(cfg={"candidate_samples_2d": 24, "candidate_samples_3d": 36})
    rvo_out = rvo.compute_cmd(inp)
    rvo_debug = rvo_out.debug_info
    checks.append(
        _check(
            rvo_method,
            "rvo_hrvo_apex_and_candidate_signals",
            bool(
                rvo_debug.get("vo_algorithm") == "hybrid_reciprocal_velocity_obstacle"
                and rvo_debug.get("vo_reciprocal_mode") == "hrvo"
                and float(rvo_debug.get("vo_hrvo_apex_shift_mean", 0.0)) > 0.0
                and int(rvo_debug.get("vo_boundary_candidate_count", 0)) > 0
                and rvo_debug.get("vo_responsibility_mean") is not None
                and float(rvo_debug.get("vo_responsibility_mean", 0.0)) > 0.5
                and int(rvo_debug.get("vo_safe_candidate_count", 0)) >= int(vo_debug.get("vo_safe_candidate_count", 0))
                and int(rvo_debug.get("vo_pred_conflict_candidate_count", 9999))
                <= int(vo_debug.get("vo_pred_conflict_candidate_count", 0))
                and _finite_vec(rvo_out.v_cmd)
            ),
            {
                "vo_v_cmd": [float(x) for x in vo_out.v_cmd],
                "rvo_v_cmd": [float(x) for x in rvo_out.v_cmd],
                "vo_debug_info": vo_debug,
                "rvo_debug_info": rvo_debug,
            },
        )
    )

    fresh = rvo.compute_cmd(_vo_conflict_input(stale_age_s=0.0, planar=True))
    stale = rvo.compute_cmd(_vo_conflict_input(stale_age_s=1.0, planar=True))
    low_priority = rvo.compute_cmd(_vo_conflict_input(stale_age_s=0.0, planar=True, priority=10))
    high_priority = rvo.compute_cmd(_vo_conflict_input(stale_age_s=0.0, planar=True, priority=0))
    checks.append(
        _check(
            rvo_method,
            "rvo_priority_and_stale_responsibility",
            bool(
                float(stale.debug_info.get("vo_responsibility_mean", 0.0))
                > float(fresh.debug_info.get("vo_responsibility_mean", 0.0))
                and float(stale.debug_info.get("vo_stale_responsibility_boost_mean", 0.0)) > 0.0
                and float(low_priority.debug_info.get("vo_responsibility_mean", 0.0))
                > float(high_priority.debug_info.get("vo_responsibility_mean", 1.0))
                and float(high_priority.debug_info.get("vo_responsibility_mean", 1.0)) >= 0.45
            ),
            {
                "fresh_debug_info": fresh.debug_info,
                "stale_debug_info": stale.debug_info,
                "low_priority_debug_info": low_priority.debug_info,
                "high_priority_debug_info": high_priority.debug_info,
            },
        )
    )
    return checks


def run_baseline_reference_evidence(
    *,
    mpc_profile_iters: int = DEFAULT_MPC_PROFILE_ITERS,
    mpc_p95_max_ms: float = DEFAULT_MPC_P95_MAX_MS,
    optimizer_profile_iters: int = DEFAULT_OPT_PROFILE_ITERS,
    optimizer_p95_max_ms: float = DEFAULT_OPT_P95_MAX_MS,
    artifact_dir: str | Path | None = None,
    save_optimizer_traces: bool = False,
) -> dict[str, Any]:
    checks = [
        *_cbf_evidence_checks(),
        *_mpc_evidence_checks(profile_iters=mpc_profile_iters, p95_max_ms=mpc_p95_max_ms),
        *_optimizer_evidence_checks(
            profile_iters=optimizer_profile_iters,
            p95_max_ms=optimizer_p95_max_ms,
            artifact_dir=artifact_dir,
            save_traces=save_optimizer_traces,
        ),
        *_rmader_evidence_checks(),
        *_vo_evidence_checks(),
    ]
    failed = [check for check in checks if not check["ok"]]
    return {
        "schema_version": BASELINE_EVIDENCE_SCHEMA_VERSION,
        "evidence_type": "advanced_baseline_reference_evidence",
        "methods": [
            "cbf_qp",
            "mpc_local",
            "mpc_nonlinear",
            "rmader",
            "ego_swarm_opt",
            "velocity_obstacle",
            "reciprocal_velocity_obstacle",
        ],
        "ok": not failed,
        "summary": {
            "check_count": len(checks),
            "failed_count": len(failed),
            "cbf_check_count": sum(1 for check in checks if check["method"] == "cbf_qp"),
            "mpc_check_count": sum(1 for check in checks if check["method"] == "mpc_local"),
            "mpc_nonlinear_check_count": sum(1 for check in checks if check["method"] == "mpc_nonlinear"),
            "rmader_check_count": sum(1 for check in checks if check["method"] == "rmader"),
            "ego_swarm_opt_check_count": sum(1 for check in checks if check["method"] == "ego_swarm_opt"),
            "vo_check_count": sum(1 for check in checks if check["method"] == "velocity_obstacle"),
            "rvo_check_count": sum(1 for check in checks if check["method"] == "reciprocal_velocity_obstacle"),
        },
        "artifacts": {
            "artifact_dir": str(artifact_dir) if artifact_dir is not None else None,
            "optimizer_traces_requested": bool(save_optimizer_traces),
        },
        "checks": checks,
        "promotion_recommendations": {
            "cbf_qp": "keep_experimental_until_solver_backends_and_infeasible_constraint_behavior_are_validated_beyond_targeted_cases",
            "mpc_local": "keep_experimental_until_dense_3d_compute_bands_and_stress_behavior_are_calibrated_on_official_suites",
            "mpc_nonlinear": "keep_experimental_until_optimizer_grade_dense_3d_degraded_intent_and_solver_mode_evidence_is_calibrated_on_official_suites",
            "rmader": "keep_experimental_until_minvo_hyperplane_delay_check_behavior_is_calibrated_on_official_dense_3d_and_degraded_v2v_suites",
            "ego_swarm_opt": "keep_experimental_until_optimizer_grade_dense_3d_degraded_intent_and_solver_mode_evidence_is_calibrated_on_official_suites",
            "velocity_obstacle": "keep_experimental_until_all_suite_vo_evidence_is_calibrated_against_orca_and_rvo",
            "reciprocal_velocity_obstacle": "keep_experimental_until_hrvo_responsibility_and_degraded_track_behavior_are_calibrated_on_official_suites",
        },
    }


def write_baseline_reference_evidence(*, out_dir: str | Path, **kwargs: Any) -> Path:
    report = run_baseline_reference_evidence(**kwargs)
    path = Path(out_dir) / "baseline_evidence.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
