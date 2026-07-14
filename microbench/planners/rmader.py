from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from microbench.comm.messages import make_intent_trajectory
from microbench.planners.base import ILocalPlanner
from microbench.types import AABBObs, IntentMsg, IntentObs, NeighborObs, PlannerInput, PlannerOutput


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _normalize(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32)
    n = _norm(arr)
    if n < 1e-9:
        return np.zeros(3, dtype=np.float32)
    return (arr / n).astype(np.float32)


def _clamp_speed(v: np.ndarray, v_max: float) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32)
    n = _norm(arr)
    if n <= float(v_max) or n < 1e-9:
        return arr
    return (arr / n * float(v_max)).astype(np.float32)


def _limit_delta(v: np.ndarray, current: np.ndarray, max_delta: float) -> np.ndarray:
    target = np.asarray(v, dtype=np.float32)
    base = np.asarray(current, dtype=np.float32)
    delta = target - base
    n = _norm(delta)
    if n <= float(max_delta) or n < 1e-9:
        return target
    return (base + delta / n * float(max_delta)).astype(np.float32)


def _perp_xz(v: np.ndarray, sign: float = 1.0) -> np.ndarray:
    return np.asarray([sign * float(v[2]), 0.0, -sign * float(v[0])], dtype=np.float32)


def _aabb_vertices(center: np.ndarray, half: np.ndarray) -> np.ndarray:
    c = np.asarray(center, dtype=np.float32)
    h = np.maximum(np.asarray(half, dtype=np.float32), 0.0)
    vertices = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                vertices.append(c + h * np.asarray([sx, sy, sz], dtype=np.float32))
    return np.asarray(vertices, dtype=np.float32)


def _closest_point_on_aabb(point: np.ndarray, obs: AABBObs) -> np.ndarray:
    center = np.asarray(obs.center, dtype=np.float32)
    half = np.asarray(obs.half, dtype=np.float32)
    return np.minimum(np.maximum(np.asarray(point, dtype=np.float32), center - half), center + half)


def _signed_distance_and_grad_to_aabb(point: np.ndarray, obs: AABBObs) -> tuple[float, np.ndarray]:
    point = np.asarray(point, dtype=np.float32)
    center = np.asarray(obs.center, dtype=np.float32)
    half = np.asarray(obs.half, dtype=np.float32)
    q = np.abs(point - center) - half
    outside = np.maximum(q, 0.0)
    outside_dist = _norm(outside)
    if outside_dist > 1e-9:
        closest = _closest_point_on_aabb(point, obs)
        return outside_dist, _normalize(point - closest)

    axis = int(np.argmax(q))
    grad = np.zeros(3, dtype=np.float32)
    grad[axis] = 1.0 if float(point[axis] - center[axis]) >= 0.0 else -1.0
    return float(q[axis]), grad


# Cubic B-spline interval polynomial converters from the public MADER/MINVO formulation.
# For a 3x4 control block Qbs, interval samples are Qbs @ A @ [u^3, u^2, u, 1],
# and continuous MINVO interval polyhedra are the columns of Qbs @ M.
_A_POS_BS_SEG0 = np.asarray(
    [
        [-1.0, 3.0, -3.0, 1.0],
        [1.75, -4.5, 3.0, 0.0],
        [-0.9167, 1.5, 0.0, 0.0],
        [0.1667, 0.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)
_A_POS_BS_SEG1 = np.asarray(
    [
        [-0.25, 0.75, -0.75, 0.25],
        [0.5833, -1.25, 0.25, 0.5833],
        [-0.5, 0.5, 0.5, 0.1667],
        [0.1667, 0.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)
_A_POS_BS_REST = np.asarray(
    [
        [-0.1667, 0.5, -0.5, 0.1667],
        [0.5, -1.0, 0.0, 0.6667],
        [-0.5, 0.5, 0.5, 0.1667],
        [0.1667, 0.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)
_A_POS_BS_LAST2 = np.asarray(
    [
        [-0.1667, 0.5, -0.5, 0.1667],
        [0.5, -1.0, 0.0, 0.6667],
        [-0.5833, 0.5, 0.5, 0.1667],
        [0.25, 0.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)
_A_POS_BS_LAST = np.asarray(
    [
        [-0.1667, 0.5, -0.5, 0.1667],
        [0.9167, -1.25, -0.25, 0.5833],
        [-1.75, 0.75, 0.75, 0.25],
        [1.0, 0.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)

_M_POS_BS2MV_SEG0 = np.asarray(
    [
        [1.1023313949144333, 0.3420572455666697, -0.09273093424558287, -0.03203276669713062],
        [-0.04968355625374918, 0.6578034732467718, 0.530538637601869, 0.21181027098212013],
        [-0.04730904421116235, 0.015594436894155586, 0.505182755715935, 0.6365005965626043],
        [-0.005338794449521744, -0.015455155707597083, 0.0570095409277783, 0.18372189915240558],
    ],
    dtype=np.float32,
)
_M_POS_BS2MV_SEG1 = np.asarray(
    [
        [0.27558284872860833, 0.08551431139166743, -0.02318273356139572, -0.008008191674282655],
        [0.6099042761975866, 0.6380690420784051, 0.2995993800913226, 0.12252106674808683],
        [0.11985166952332682, 0.29187180223752446, 0.6665738125422942, 0.7017652257737893],
        [-0.005338794449521744, -0.015455155707597083, 0.0570095409277783, 0.18372189915240558],
    ],
    dtype=np.float32,
)
_M_POS_BS2MV_REST = np.asarray(
    [
        [0.18372189915240555, 0.05700954092777831, -0.015455155707597118, -0.005338794449521816],
        [0.7017652257737892, 0.6665738125422942, 0.29187180223752385, 0.11985166952332582],
        [0.11985166952332682, 0.29187180223752446, 0.6665738125422942, 0.7017652257737893],
        [-0.005338794449521744, -0.015455155707597083, 0.0570095409277783, 0.18372189915240558],
    ],
    dtype=np.float32,
)
_M_POS_BS2MV_LAST2 = np.asarray(
    [
        [0.1837218991524057, 0.05700954092777831, -0.015455155707597146, -0.005338794449521816],
        [0.7017652257737895, 0.6665738125422945, 0.2918718022375241, 0.11985166952332593],
        [0.12252106674808753, 0.2995993800913228, 0.638069042078405, 0.6099042761975862],
        [-0.008008191674282615, -0.02318273356139562, 0.08551431139166744, 0.27558284872860833],
    ],
    dtype=np.float32,
)
_M_POS_BS2MV_LAST = np.asarray(
    [
        [0.18372189915240555, 0.05700954092777831, -0.015455155707597118, -0.005338794449521816],
        [0.6365005965626042, 0.505182755715935, 0.015594436894155295, -0.04730904421116289],
        [0.21181027098212069, 0.5305386376018691, 0.6578034732467715, -0.04968355625374962],
        [-0.03203276669713046, -0.09273093424558249, 0.3420572455666698, 1.1023313949144333],
    ],
    dtype=np.float32,
)

_M_VEL_BS2MV_SEG0 = np.asarray(
    [
        [1.077349059083916, 0.1666702138890985, -0.07735049175615138],
        [-0.03867488648729411, 0.7499977187062712, 0.5386802643920123],
        [-0.03867417280506149, 0.08333206631563977, 0.538670227146185],
    ],
    dtype=np.float32,
)
_M_VEL_BS2MV_REST = np.asarray(
    [
        [0.538674529541958, 0.08333510694454926, -0.03867524587807569],
        [0.4999996430546639, 0.8333328256508203, 0.5000050185139366],
        [-0.03867417280506149, 0.08333206631563977, 0.538670227146185],
    ],
    dtype=np.float32,
)
_M_VEL_BS2MV_LAST = np.asarray(
    [
        [0.538674529541958, 0.08333510694454926, -0.03867524587807569],
        [0.5386738158597254, 0.7500007593351806, -0.03866520863224832],
        [-0.07734834561012298, 0.1666641326312795, 1.07734045429237],
    ],
    dtype=np.float32,
)


def _interval_matrices(num_segments: int, *, kind: str) -> list[np.ndarray]:
    if num_segments < 4:
        raise ValueError("RMADER needs at least four cubic B-spline intervals")
    if kind == "a_pos_bs":
        return [_A_POS_BS_SEG0, _A_POS_BS_SEG1, *([_A_POS_BS_REST] * (num_segments - 4)), _A_POS_BS_LAST2, _A_POS_BS_LAST]
    if kind == "m_pos_bs2mv":
        return [
            _M_POS_BS2MV_SEG0,
            _M_POS_BS2MV_SEG1,
            *([_M_POS_BS2MV_REST] * (num_segments - 4)),
            _M_POS_BS2MV_LAST2,
            _M_POS_BS2MV_LAST,
        ]
    if kind == "m_vel_bs2mv":
        return [_M_VEL_BS2MV_SEG0, *([_M_VEL_BS2MV_REST] * (num_segments - 2)), _M_VEL_BS2MV_LAST]
    raise ValueError(f"unknown RMADER matrix kind: {kind}")


@dataclass(frozen=True)
class _Seed:
    label: str
    control_points: np.ndarray
    offset_norm_m: float


@dataclass(frozen=True)
class _IntervalHull:
    interval_idx: int
    source_kind: str
    source_id: int
    vertices: np.ndarray
    inflation_m: float


@dataclass(frozen=True)
class _PlaneConstraint:
    interval_idx: int
    source_kind: str
    source_id: int
    normal: np.ndarray
    d: float
    margin_m: float
    gap_m: float
    max_violation_m: float
    feasible: bool


@dataclass(frozen=True)
class _ConstraintReport:
    planes: list[_PlaneConstraint]
    max_violation_m: float
    sum_violation_m: float
    min_gap_m: float | None
    hard_ok: bool
    hull_count: int


@dataclass(frozen=True)
class _KinematicReport:
    max_speed_violation_mps: float
    max_accel_violation_mps2: float
    max_jerk_violation_mps3: float
    ok: bool


@dataclass(frozen=True)
class _PlanResult:
    label: str
    control_points: np.ndarray
    samples: np.ndarray
    minvo_intervals: np.ndarray
    initial_cost: float
    final_cost: float
    iterations: int
    solver_status: str
    constraint_report: _ConstraintReport
    kinematic_report: _KinematicReport
    smoothness_cost: float
    path_length_m: float


class RmaderPlanner(ILocalPlanner):
    """Clean-room RMADER-style decentralized trajectory optimizer.

    The planner follows RMADER/MADER's algorithmic contract inside DAA
    Microbench: cubic B-spline trajectory optimization, continuous MINVO
    interval polyhedra, hard separating hyperplane constraints against dynamic
    and static hulls, kinematic smoothing, and robust publish/check/commit
    behavior. It is not a ROS/Gurobi port; it is a deterministic Python
    baseline that uses only the public ``PlannerInput`` data available to a
    local DAA planner.
    """

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.horizon_s = float(cfg.get("horizon_s", 3.2))
        self.control_points = int(cfg.get("control_points", 8))
        self.samples_per_interval = int(cfg.get("samples_per_interval", 3))
        self.max_neighbors = int(cfg.get("max_neighbors", 8))
        self.max_initializations = int(cfg.get("max_initializations", 6))
        self.opt_iterations = int(cfg.get("opt_iterations", 8))
        self.hard_projection_iterations = int(cfg.get("hard_projection_iterations", 7))
        self.gradient_step_m = float(cfg.get("gradient_step_m", 0.16))
        self.line_search_shrink = float(cfg.get("line_search_shrink", 0.55))
        self.hard_projection_relaxation = float(cfg.get("hard_projection_relaxation", 0.92))
        self.safety_margin_m = float(cfg.get("safety_margin_m", 0.35))
        self.obstacle_margin_m = float(cfg.get("obstacle_margin_m", 0.3))
        self.minvo_epsilon_m = float(cfg.get("minvo_epsilon_m", 0.12))
        self.hard_safety_tolerance_m = float(cfg.get("hard_safety_tolerance_m", 0.05))
        self.near_clearance_m = float(cfg.get("near_clearance_m", 1.7))
        self.goal_slowdown_radius_m = float(cfg.get("goal_slowdown_radius_m", 6.0))
        self.intent_ttl_s = float(cfg.get("intent_ttl_s", 1.0))
        self.intent_tube_margin_m = float(cfg.get("intent_tube_margin_m", 0.35))
        self.coordination_message_ttl_s = float(cfg.get("coordination_message_ttl_s", 0.75))
        self.max_intent_points = int(cfg.get("max_intent_points", 14))
        self.delay_check_enabled = bool(cfg.get("delay_check_enabled", True))
        self.delay_check_tolerance_m = float(cfg.get("delay_check_tolerance_m", 0.08))
        self.stale_age_cap_s = float(cfg.get("stale_age_cap_s", 1.5))
        self.stale_inflation_gain = float(cfg.get("stale_inflation_gain", 0.75))
        self.intent_age_inflation_gain = float(cfg.get("intent_age_inflation_gain", 0.35))
        self.track_uncertainty_speed_gain = float(cfg.get("track_uncertainty_speed_gain", 0.12))
        self.jerk_limit_mps3 = float(cfg.get("jerk_limit_mps3", 8.0))
        self.offset_scales_m = tuple(float(x) for x in cfg.get("offset_scales_m", (0.0, 2.5, 5.0)))
        self.vertical_offset_scales_m = tuple(float(x) for x in cfg.get("vertical_offset_scales_m", (2.0, 4.0)))

        self.reference_weight = float(cfg.get("reference_weight", 0.08))
        self.warm_start_weight = float(cfg.get("warm_start_weight", 0.16))
        self.smoothness_weight = float(cfg.get("smoothness_weight", 2.7))
        self.path_length_weight = float(cfg.get("path_length_weight", 0.05))
        self.terminal_weight = float(cfg.get("terminal_weight", 4.0))
        self.soft_clearance_weight = float(cfg.get("soft_clearance_weight", 120.0))
        self.hard_violation_weight = float(cfg.get("hard_violation_weight", 18000.0))
        self.kinematic_violation_weight = float(cfg.get("kinematic_violation_weight", 4000.0))

        self.seed = 0
        self._last_control_points: np.ndarray | None = None
        self._last_label: str | None = None
        self._local_memory: dict[str, object] = {}

    def reset(self, seed: int) -> None:
        self.seed = int(seed)
        self._last_control_points = None
        self._last_label = None
        self._local_memory.clear()

    def compute_cmd(self, planner_input: PlannerInput) -> PlannerOutput:
        ego = planner_input.ego
        current = np.asarray(ego.vel, dtype=np.float32).copy()
        if planner_input.planar:
            current[1] = 0.0

        seeds = self._initializations(planner_input)
        results = [self._optimize_seed(planner_input, seed) for seed in seeds]
        best = min(
            results,
            key=lambda r: (
                not r.constraint_report.hard_ok,
                not r.kinematic_report.ok,
                r.constraint_report.max_violation_m,
                r.final_cost,
            ),
        )

        memory = self._memory(planner_input)
        accepted = self._delay_check(best)
        plan_used = best
        delay_fallback = "none"
        if not accepted:
            fallback = self._committed_fallback(planner_input, memory)
            if fallback is not None:
                plan_used = fallback
                delay_fallback = "previous_committed"
            else:
                plan_used = self._braking_plan(planner_input)
                delay_fallback = "braking_trajectory"

        if accepted or delay_fallback == "previous_committed":
            plan_version = int(memory.get("rmader_plan_version", 0)) + (1 if accepted else 0)
            if accepted:
                memory["rmader_plan_version"] = plan_version
                memory["rmader_committed_control_points"] = best.control_points.copy()
                memory["rmader_committed_label"] = str(best.label)
                memory["rmader_committed_until_s"] = float(planner_input.t) + self.intent_ttl_s
            else:
                plan_version = int(memory.get("rmader_plan_version", 0))
        else:
            plan_version = int(memory.get("rmader_plan_version", 0))

        sample_dt = self.horizon_s / max(1, plan_used.samples.shape[0] - 1)
        next_idx = 1 if plan_used.samples.shape[0] > 1 else 0
        desired_v = (plan_used.samples[next_idx] - np.asarray(ego.pos, dtype=np.float32)) / max(1e-6, sample_dt)
        v_cmd = _limit_delta(desired_v, current, float(ego.a_max) * float(planner_input.dt))
        v_cmd = _clamp_speed(v_cmd, float(ego.v_max))
        if planner_input.planar:
            v_cmd[1] = 0.0

        intent_points = self._intent_points(planner_input, plan_used.samples)
        intent = IntentMsg(
            sender_id=int(ego.idx),
            timestamp_send_s=float(planner_input.t),
            expiry_s=float(planner_input.t) + self.intent_ttl_s,
            kind="RMADER_MINVO_TRAJECTORY",
            tube_radius_m=float(ego.radius) + self.intent_tube_margin_m,
            points=intent_points.astype(float),
            dt_plan_s=float(sample_dt),
            mode=f"{plan_used.label}:commit:{plan_version}",
        )

        candidate_msg = make_intent_trajectory(
            sender_id=int(ego.idx),
            recipient_id=None,
            now_s=float(planner_input.t),
            trajectory=best.samples,
            dt_plan_s=float(sample_dt),
            expiry_s=float(planner_input.t) + self.intent_ttl_s,
            tube_radius_m=float(ego.radius) + self.intent_tube_margin_m,
            ttl_s=self.coordination_message_ttl_s,
        )
        candidate_msg.payload.update(
            {
                "algorithm": "rmader",
                "publication_stage": "candidate",
                "delay_check_passed": bool(accepted),
                "plan_version": int(plan_version),
            }
        )
        committed_msg = make_intent_trajectory(
            sender_id=int(ego.idx),
            recipient_id=None,
            now_s=float(planner_input.t),
            trajectory=plan_used.samples,
            dt_plan_s=float(sample_dt),
            expiry_s=float(planner_input.t) + self.intent_ttl_s,
            tube_radius_m=float(ego.radius) + self.intent_tube_margin_m,
            ttl_s=self.coordination_message_ttl_s,
        )
        committed_msg.payload.update(
            {
                "algorithm": "rmader",
                "publication_stage": "committed",
                "delay_check_passed": bool(accepted),
                "plan_version": int(plan_version),
                "fallback": delay_fallback,
            }
        )

        prior_label = self._last_label
        if accepted:
            self._last_control_points = best.control_points.copy()
            self._last_label = best.label

        chosen_report = plan_used.constraint_report
        chosen_kin = plan_used.kinematic_report
        best_report = best.constraint_report
        debug = {
            "rmader_algorithm": "robust_mader_minvo_hyperplane_trajectory_optimization",
            "rmader_reference": "RMADER/MADER-style clean-room baseline; not a ROS/Gurobi port",
            "rmader_solver": "projected_minvo_hyperplane_sqp",
            "rmader_solver_status": str(best.solver_status),
            "rmader_horizon_s": float(self.horizon_s),
            "rmader_control_points": int(plan_used.control_points.shape[0]),
            "rmader_segments": int(max(0, plan_used.control_points.shape[0] - 3)),
            "rmader_minvo_intervals": int(plan_used.minvo_intervals.shape[0]),
            "rmader_minvo_control_points_per_interval": int(plan_used.minvo_intervals.shape[1])
            if plan_used.minvo_intervals.ndim == 3
            else 0,
            "rmader_initializations": int(len(seeds)),
            "rmader_iterations": int(best.iterations),
            "rmader_best_topology": str(best.label),
            "rmader_used_topology": str(plan_used.label),
            "rmader_initial_cost": float(best.initial_cost),
            "rmader_final_cost": float(best.final_cost),
            "rmader_cost_reduction": float(best.initial_cost - best.final_cost),
            "rmader_path_length_m": float(plan_used.path_length_m),
            "rmader_smoothness_cost": float(plan_used.smoothness_cost),
            "rmader_hard_constraint_ok": bool(chosen_report.hard_ok),
            "rmader_hard_constraint_count": int(len(chosen_report.planes)),
            "rmader_hard_hull_count": int(chosen_report.hull_count),
            "rmader_max_hyperplane_violation_m": float(chosen_report.max_violation_m),
            "rmader_sum_hyperplane_violation_m": float(chosen_report.sum_violation_m),
            "rmader_min_hyperplane_gap_m": chosen_report.min_gap_m,
            "rmader_candidate_max_hyperplane_violation_m": float(best_report.max_violation_m),
            "rmader_candidate_hard_constraint_ok": bool(best_report.hard_ok),
            "rmader_candidate_kinematic_ok": bool(best.kinematic_report.ok),
            "rmader_candidate_max_speed_violation_mps": float(best.kinematic_report.max_speed_violation_mps),
            "rmader_candidate_max_accel_violation_mps2": float(best.kinematic_report.max_accel_violation_mps2),
            "rmader_candidate_max_jerk_violation_mps3": float(best.kinematic_report.max_jerk_violation_mps3),
            "rmader_kinematic_ok": bool(chosen_kin.ok),
            "rmader_max_speed_violation_mps": float(chosen_kin.max_speed_violation_mps),
            "rmader_max_accel_violation_mps2": float(chosen_kin.max_accel_violation_mps2),
            "rmader_max_jerk_violation_mps3": float(chosen_kin.max_jerk_violation_mps3),
            "rmader_delay_check_enabled": bool(self.delay_check_enabled),
            "rmader_delay_check_passed": bool(accepted),
            "rmader_delay_check_fallback": str(delay_fallback),
            "rmader_two_step_publication": True,
            "rmader_plan_version": int(plan_version),
            "rmader_agent_messages": 2,
            "rmader_neighbor_count_considered": int(min(len(planner_input.neighbors), self.max_neighbors)),
            "rmader_intent_count_considered": int(sum(1 for intent_obs in planner_input.neighbor_intents if intent_obs.valid)),
            "rmader_obstacle_count_considered": int(len(planner_input.obstacles)),
            "rmader_planar": bool(planner_input.planar),
            "rmader_intent_points": int(intent_points.shape[0]),
            "rmader_prior_label": prior_label,
            "rmader_accel_delta_norm": float(_norm(v_cmd - current)),
            "rmader_accel_delta_limit": float(float(ego.a_max) * float(planner_input.dt)),
        }
        return PlannerOutput(
            v_cmd=v_cmd.astype(float),
            intent_out=intent,
            messages_out=[candidate_msg, committed_msg],
            debug_info=debug,
        )

    def _memory(self, planner_input: PlannerInput) -> dict[str, object]:
        if planner_input.agent_context is not None:
            return planner_input.agent_context.memory
        return self._local_memory

    def _control_point_count(self) -> int:
        return max(7, int(self.control_points))

    def _segment_count(self) -> int:
        return self._control_point_count() - 3

    def _segment_dt(self) -> float:
        return self.horizon_s / max(1, self._segment_count())

    def _preferred_velocity(self, planner_input: PlannerInput) -> np.ndarray:
        ego = planner_input.ego
        goal_dir = np.asarray(planner_input.goal_dir, dtype=np.float32).copy()
        if planner_input.planar:
            goal_dir[1] = 0.0
        direction = _normalize(goal_dir)
        if _norm(direction) < 1e-9:
            direction = _normalize(np.asarray(ego.goal, dtype=np.float32) - np.asarray(ego.pos, dtype=np.float32))
        goal_dist = _norm(np.asarray(ego.goal, dtype=np.float32) - np.asarray(ego.pos, dtype=np.float32))
        soft_speed = float(ego.v_max) * min(1.0, goal_dist / max(1e-6, self.goal_slowdown_radius_m))
        stop_speed = math.sqrt(max(0.0, 2.0 * max(1e-6, float(ego.a_max)) * goal_dist))
        return (direction * min(float(ego.v_max), max(soft_speed, min(float(ego.v_max), stop_speed)))).astype(np.float32)

    def _local_target(self, planner_input: PlannerInput) -> np.ndarray:
        ego = planner_input.ego
        p0 = np.asarray(ego.pos, dtype=np.float32)
        goal = np.asarray(ego.goal, dtype=np.float32)
        v_pref = self._preferred_velocity(planner_input)
        if _norm(v_pref) < 1e-9:
            v_pref = _normalize(goal - p0) * float(ego.v_max)
        if planner_input.planar:
            v_pref[1] = 0.0
        to_goal = goal - p0
        if planner_input.planar:
            to_goal[1] = 0.0
        direction = _normalize(v_pref if _norm(v_pref) > 1e-9 else to_goal)
        goal_dist = _norm(to_goal)
        current_speed = _norm(np.asarray(ego.vel, dtype=np.float32))
        accel_reachable = current_speed * self.horizon_s + 0.5 * float(ego.a_max) * self.horizon_s * self.horizon_s
        speed_reachable = float(ego.v_max) * self.horizon_s
        horizon_dist = min(goal_dist, max(0.5, 0.85 * min(speed_reachable, accel_reachable)))
        target = goal.copy() if goal_dist <= horizon_dist else p0 + direction * horizon_dist
        if planner_input.planar:
            target[1] = p0[1]
        return target.astype(np.float32)

    def _initializations(self, planner_input: PlannerInput) -> list[_Seed]:
        target = self._local_target(planner_input)
        seeds = [self._control_polygon(planner_input, target, np.zeros(3, dtype=np.float32), "direct")]
        v_pref = self._preferred_velocity(planner_input)
        directions = self._topology_directions(planner_input, v_pref)
        for label, direction in directions:
            d = _normalize(direction)
            if _norm(d) < 1e-9:
                continue
            scales = self.vertical_offset_scales_m if label.startswith("vertical") else self.offset_scales_m
            for scale in scales:
                if scale <= 0.0:
                    continue
                offset = d * float(scale)
                if planner_input.planar:
                    offset[1] = 0.0
                seeds.append(self._control_polygon(planner_input, target, offset, f"{label}:{scale:g}m"))

        if self._last_control_points is not None and self._last_control_points.shape[0] == self._control_point_count():
            warm = self._last_control_points.copy()
            warm = self._apply_boundary_conditions(planner_input, warm, target)
            seeds.append(_Seed("warm_start", warm.astype(np.float32), 0.0))

        return self._dedupe_seeds(seeds)[: max(1, self.max_initializations)]

    def _topology_directions(self, planner_input: PlannerInput, v_pref: np.ndarray) -> list[tuple[str, np.ndarray]]:
        p0 = np.asarray(planner_input.ego.pos, dtype=np.float32)
        directions: list[tuple[str, np.ndarray]] = []
        if _norm(v_pref) > 1e-9:
            directions.append(("left", _perp_xz(v_pref, 1.0)))
            directions.append(("right", _perp_xz(v_pref, -1.0)))
            if not planner_input.planar:
                directions.append(("vertical_up", np.asarray([0.0, 1.0, 0.0], dtype=np.float32)))
                directions.append(("vertical_down", np.asarray([0.0, -1.0, 0.0], dtype=np.float32)))

        for nobs in planner_input.neighbors[: self.max_neighbors]:
            rel = p0 - np.asarray(nobs.pos, dtype=np.float32)
            if planner_input.planar:
                rel[1] = 0.0
            if _norm(rel) > 1e-9:
                sign = -1.0 if int(planner_input.ego.idx) % 2 == 0 else 1.0
                directions.append((f"agent_{int(nobs.idx)}_away", rel))
                directions.append((f"agent_{int(nobs.idx)}_side", rel + _perp_xz(rel, sign)))

        for intent in planner_input.neighbor_intents:
            if not intent.valid:
                continue
            points = np.asarray(intent.points, dtype=np.float32)
            if points.ndim != 2 or points.shape[0] == 0:
                continue
            rel = p0 - points[min(1, points.shape[0] - 1)]
            if planner_input.planar:
                rel[1] = 0.0
            if _norm(rel) > 1e-9:
                directions.append((f"intent_{int(intent.sender_id)}_away", rel))

        for obs_idx, obs in enumerate(planner_input.obstacles):
            rel = p0 - _closest_point_on_aabb(p0, obs)
            if _norm(rel) <= 1e-9:
                rel = p0 - np.asarray(obs.center, dtype=np.float32)
            if planner_input.planar:
                rel[1] = 0.0
            if _norm(rel) > 1e-9:
                directions.append((f"obstacle_{obs_idx}_away", rel))
                directions.append((f"obstacle_{obs_idx}_side", rel + _perp_xz(rel, 1.0)))
        return directions

    def _control_polygon(self, planner_input: PlannerInput, target: np.ndarray, offset: np.ndarray, label: str) -> _Seed:
        count = self._control_point_count()
        cp = np.zeros((count, 3), dtype=np.float32)
        cp = self._apply_boundary_conditions(planner_input, cp, target)
        first_free = 3
        last_free = count - 4
        if last_free >= first_free:
            start = cp[2]
            end = target
            for idx in range(first_free, last_free + 1):
                tau = (idx - first_free + 1) / max(1, last_free - first_free + 2)
                point = start + (end - start) * tau + np.sin(np.pi * tau) * offset
                if planner_input.planar:
                    point[1] = float(planner_input.ego.pos[1])
                cp[idx] = point.astype(np.float32)
        cp = self._project_kinematic(planner_input, cp)
        return _Seed(label=label, control_points=cp.astype(np.float32), offset_norm_m=_norm(offset))

    def _apply_boundary_conditions(self, planner_input: PlannerInput, cp: np.ndarray, target: np.ndarray | None = None) -> np.ndarray:
        cp = np.asarray(cp, dtype=np.float32).copy()
        ego = planner_input.ego
        p0 = np.asarray(ego.pos, dtype=np.float32)
        v0 = _clamp_speed(np.asarray(ego.vel, dtype=np.float32), float(ego.v_max))
        if planner_input.planar:
            v0[1] = 0.0
        dt = self._segment_dt()
        pf = np.asarray(self._local_target(planner_input) if target is None else target, dtype=np.float32)
        cp[0] = p0
        cp[1] = p0 + v0 * dt / 3.0
        cp[2] = p0 + 2.0 * v0 * dt / 3.0
        cp[-3] = pf
        cp[-2] = pf
        cp[-1] = pf
        if planner_input.planar:
            cp[:, 1] = float(p0[1])
        return cp.astype(np.float32)

    def _dedupe_seeds(self, seeds: list[_Seed]) -> list[_Seed]:
        out: list[_Seed] = []
        seen: set[tuple[int, ...]] = set()
        for seed in seeds:
            cp = np.asarray(seed.control_points, dtype=np.float32)
            if cp.ndim != 2 or cp.shape[0] < 7:
                continue
            free = cp[3 : max(4, cp.shape[0] - 3)]
            key_values = free.reshape(-1)[:9] if free.size else cp.reshape(-1)[:9]
            key = tuple(int(round(float(x) * 1000.0)) for x in key_values)
            if key in seen:
                continue
            seen.add(key)
            out.append(seed)
        return out

    def _optimize_seed(self, planner_input: PlannerInput, seed: _Seed) -> _PlanResult:
        cp = self._project_kinematic(planner_input, np.asarray(seed.control_points, dtype=np.float32).copy())
        cp, _ = self._project_hard_separation(planner_input, cp)
        initial = self._objective(planner_input, cp, seed.control_points)
        previous = initial
        iterations = 0
        status = "projected_minvo_hyperplane_converged"

        for _ in range(max(0, self.opt_iterations)):
            iterations += 1
            grad = self._objective_gradient(planner_input, cp, seed.control_points)
            fixed = self._fixed_mask(cp.shape[0])
            grad[fixed] = 0.0
            if planner_input.planar:
                grad[:, 1] = 0.0
            grad_norm = _norm(grad)
            if grad_norm < 1e-8:
                status = "projected_minvo_hyperplane_stationary"
                break
            step = self.gradient_step_m / max(1.0, grad_norm)
            accepted = False
            for _ls in range(8):
                candidate = cp - grad * step
                candidate = self._project_kinematic(planner_input, candidate)
                candidate, _ = self._project_hard_separation(planner_input, candidate)
                current = self._objective(planner_input, candidate, seed.control_points)
                if current["total"] <= previous["total"] + 1e-6:
                    cp = candidate
                    previous = current
                    accepted = True
                    break
                step *= max(0.1, min(0.95, self.line_search_shrink))
            if not accepted:
                status = "projected_minvo_hyperplane_line_search_stalled"
                break

        final = self._objective(planner_input, cp, seed.control_points)
        return self._plan_result(
            label=seed.label,
            cp=cp,
            initial_cost=float(initial["total"]),
            final=final,
            iterations=iterations,
            status=status,
        )

    def _objective(self, planner_input: PlannerInput, cp: np.ndarray, reference_cp: np.ndarray) -> dict[str, Any]:
        samples = self._bspline_samples(planner_input, cp)
        target = self._local_target(planner_input)
        terminal = self.terminal_weight * float(np.dot(samples[-1] - target, samples[-1] - target))
        ref_delta = cp - np.asarray(reference_cp, dtype=np.float32)
        reference = self.reference_weight * float(np.sum(ref_delta[3:-3] ** 2))
        warm = 0.0
        if self._last_control_points is not None and self._last_control_points.shape == cp.shape:
            warm = self.warm_start_weight * float(np.sum((cp[3:-3] - self._last_control_points[3:-3]) ** 2))
        smoothness, _ = self._smoothness_cost_and_grad(cp)
        path_length = self._path_length(samples)
        clearance, _ = self._sample_clearance_cost_and_grad(planner_input, samples)
        constraints = self._constraint_report(planner_input, cp)
        kin = self._kinematic_report(planner_input, cp)
        hard = self.hard_violation_weight * (
            constraints.sum_violation_m + 10.0 * constraints.max_violation_m * constraints.max_violation_m
        )
        dyn = self.kinematic_violation_weight * (
            kin.max_speed_violation_mps * kin.max_speed_violation_mps
            + kin.max_accel_violation_mps2 * kin.max_accel_violation_mps2
            + kin.max_jerk_violation_mps3 * kin.max_jerk_violation_mps3
        )
        total = terminal + reference + warm + smoothness + self.path_length_weight * path_length + clearance + hard + dyn
        return {
            "total": float(total),
            "samples": samples,
            "smoothness_cost": float(smoothness),
            "path_length_m": float(path_length),
            "constraint_report": constraints,
            "kinematic_report": kin,
        }

    def _objective_gradient(self, planner_input: PlannerInput, cp: np.ndarray, reference_cp: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(cp, dtype=np.float32)
        _, smooth_grad = self._smoothness_cost_and_grad(cp)
        grad += smooth_grad
        grad[3:-3] += (2.0 * self.reference_weight * (cp[3:-3] - np.asarray(reference_cp, dtype=np.float32)[3:-3])).astype(
            np.float32
        )
        if self._last_control_points is not None and self._last_control_points.shape == cp.shape:
            grad[3:-3] += (2.0 * self.warm_start_weight * (cp[3:-3] - self._last_control_points[3:-3])).astype(
                np.float32
            )
        sample_grad = self._control_point_clearance_gradient(planner_input, cp)
        grad += sample_grad
        return grad.astype(np.float32)

    def _smoothness_cost_and_grad(self, cp: np.ndarray) -> tuple[float, np.ndarray]:
        grad = np.zeros_like(cp, dtype=np.float32)
        cost = 0.0
        for i in range(1, cp.shape[0] - 1):
            second = cp[i - 1] - 2.0 * cp[i] + cp[i + 1]
            cost += self.smoothness_weight * float(np.dot(second, second))
            g = 2.0 * self.smoothness_weight * second
            grad[i - 1] += g
            grad[i] += -2.0 * g
            grad[i + 1] += g
        return float(cost), grad.astype(np.float32)

    def _sample_clearance_cost_and_grad(self, planner_input: PlannerInput, samples: np.ndarray) -> tuple[float, np.ndarray]:
        grad = np.zeros_like(samples, dtype=np.float32)
        cost = 0.0
        if samples.size == 0:
            return 0.0, grad
        dt = self.horizon_s / max(1, samples.shape[0] - 1)
        intent_by_sender = {
            int(intent.sender_id): intent
            for intent in planner_input.neighbor_intents
            if intent.valid and np.asarray(intent.points).size > 0
        }
        for step_idx, point in enumerate(samples[1:], start=1):
            t = step_idx * dt
            sample_idx = step_idx
            for nobs in planner_input.neighbors[: self.max_neighbors]:
                intent = intent_by_sender.get(int(nobs.idx))
                other = self._neighbor_prediction(nobs, intent, t)
                radius = float(planner_input.ego.radius) + float(nobs.radius) + self.safety_margin_m
                radius += self._neighbor_inflation(nobs) + (self._intent_inflation(intent) if intent is not None else 0.0)
                p, g = self._point_clearance_penalty(point, other, radius)
                cost += p
                grad[sample_idx] += g
            for obs in planner_input.obstacles:
                signed_dist, dist_grad = _signed_distance_and_grad_to_aabb(point, obs)
                clearance = signed_dist - float(planner_input.ego.radius) - self.obstacle_margin_m
                if clearance < self.near_clearance_m:
                    p = self.soft_clearance_weight * (self.near_clearance_m - clearance) ** 2
                    g = -2.0 * self.soft_clearance_weight * (self.near_clearance_m - clearance) * dist_grad
                    cost += float(p)
                    grad[sample_idx] += g.astype(np.float32)
        return float(cost), grad.astype(np.float32)

    def _control_point_clearance_gradient(self, planner_input: PlannerInput, cp: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(cp, dtype=np.float32)
        hulls = self._build_interval_hulls(planner_input, self._segment_count(), self._segment_dt())
        if not hulls:
            return grad
        fixed = self._fixed_mask(cp.shape[0])
        for idx in range(cp.shape[0]):
            if fixed[idx]:
                continue
            point = cp[idx]
            for hull in hulls:
                center = np.mean(hull.vertices, axis=0)
                half = np.maximum(np.max(np.abs(hull.vertices - center), axis=0), 0.1)
                closest = np.minimum(np.maximum(point, center - half), center + half)
                rel = point - closest
                dist = _norm(rel)
                if dist <= 1e-9:
                    rel = point - center
                    dist = _norm(rel)
                direction = _normalize(rel if dist > 1e-9 else point - center + np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
                clearance = dist - self.minvo_epsilon_m
                if clearance < self.near_clearance_m:
                    grad[idx] += (-2.0 * self.soft_clearance_weight * (self.near_clearance_m - clearance) * direction).astype(
                        np.float32
                    )
        if planner_input.planar:
            grad[:, 1] = 0.0
        return grad

    def _point_clearance_penalty(self, point: np.ndarray, other: np.ndarray, radius: float) -> tuple[float, np.ndarray]:
        rel = np.asarray(point, dtype=np.float32) - np.asarray(other, dtype=np.float32)
        dist = _norm(rel)
        direction = _normalize(rel if dist > 1e-9 else np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
        clearance = dist - float(radius)
        if clearance < self.near_clearance_m:
            penalty = self.soft_clearance_weight * (self.near_clearance_m - clearance) ** 2
            grad = -2.0 * self.soft_clearance_weight * (self.near_clearance_m - clearance) * direction
            return float(penalty), grad.astype(np.float32)
        return 0.0, np.zeros(3, dtype=np.float32)

    def _project_kinematic(self, planner_input: PlannerInput, cp: np.ndarray) -> np.ndarray:
        cp = self._apply_boundary_conditions(planner_input, cp)
        fixed = self._fixed_mask(cp.shape[0])
        dt = self._segment_dt()
        max_step = float(planner_input.ego.v_max) * max(1e-6, dt)
        for _ in range(2):
            for i in range(1, cp.shape[0]):
                if fixed[i]:
                    continue
                delta = cp[i] - cp[i - 1]
                n = _norm(delta)
                if n > max_step:
                    cp[i] = cp[i - 1] + delta / n * max_step
            for i in range(cp.shape[0] - 2, -1, -1):
                if fixed[i]:
                    continue
                delta = cp[i] - cp[i + 1]
                n = _norm(delta)
                if n > max_step:
                    cp[i] = cp[i + 1] + delta / n * max_step

        max_second = float(planner_input.ego.a_max) * dt * dt
        for i in range(1, cp.shape[0] - 1):
            if fixed[i]:
                continue
            second = cp[i - 1] - 2.0 * cp[i] + cp[i + 1]
            n = _norm(second)
            if n > max_second:
                desired = second / n * max_second
                cp[i] = 0.5 * (cp[i - 1] + cp[i + 1] - desired)

        if planner_input.planar:
            cp[:, 1] = float(planner_input.ego.pos[1])
        return self._apply_boundary_conditions(planner_input, cp).astype(np.float32)

    def _project_hard_separation(self, planner_input: PlannerInput, cp: np.ndarray) -> tuple[np.ndarray, _ConstraintReport]:
        cp = np.asarray(cp, dtype=np.float32).copy()
        matrices = _interval_matrices(cp.shape[0] - 3, kind="m_pos_bs2mv")
        fixed = self._fixed_mask(cp.shape[0])
        report = self._constraint_report(planner_input, cp)
        if not report.planes:
            return cp, report

        for _ in range(max(0, self.hard_projection_iterations)):
            adjusted = False
            minvo = self._minvo_intervals(cp)
            report = self._constraint_report(planner_input, cp, minvo=minvo)
            if report.max_violation_m <= self.hard_safety_tolerance_m:
                break
            for plane in report.planes:
                if plane.max_violation_m <= 0.0:
                    continue
                segment = int(plane.interval_idx)
                normal = np.asarray(plane.normal, dtype=np.float32)
                d = float(plane.d)
                margin = float(plane.margin_m)
                own_points = minvo[segment]
                matrix = matrices[segment]
                for u, q in enumerate(own_points):
                    h = float(np.dot(normal, q) + d + margin)
                    if h <= 0.0:
                        continue
                    free_indices = [segment + k for k in range(4) if not fixed[segment + k]]
                    if not free_indices:
                        continue
                    denom = sum(float(matrix[k, u] * matrix[k, u]) for k in range(4) if not fixed[segment + k])
                    if denom <= 1e-9:
                        continue
                    for k in range(4):
                        idx = segment + k
                        if fixed[idx]:
                            continue
                        cp[idx] -= self.hard_projection_relaxation * h * float(matrix[k, u]) / denom * normal
                        adjusted = True
            cp = self._project_kinematic(planner_input, cp)
            if planner_input.planar:
                cp[:, 1] = float(planner_input.ego.pos[1])
            if not adjusted:
                break
        report = self._constraint_report(planner_input, cp)
        return cp.astype(np.float32), report

    def _constraint_report(
        self,
        planner_input: PlannerInput,
        cp: np.ndarray,
        *,
        minvo: np.ndarray | None = None,
    ) -> _ConstraintReport:
        minvo = self._minvo_intervals(cp) if minvo is None else np.asarray(minvo, dtype=np.float32)
        hulls = self._build_interval_hulls(planner_input, minvo.shape[0], self._segment_dt())
        planes: list[_PlaneConstraint] = []
        max_violation = 0.0
        sum_violation = 0.0
        min_gap: float | None = None
        for hull in hulls:
            own = minvo[int(hull.interval_idx)]
            plane = self._separating_plane(own, hull.vertices, hull.interval_idx, hull.source_kind, hull.source_id)
            own_violation = float(np.max(own @ plane.normal + plane.d + plane.margin_m))
            other_violation = float(np.max(-(hull.vertices @ plane.normal + plane.d - plane.margin_m)))
            violation = max(0.0, own_violation, other_violation)
            plane = _PlaneConstraint(
                interval_idx=plane.interval_idx,
                source_kind=plane.source_kind,
                source_id=plane.source_id,
                normal=plane.normal,
                d=plane.d,
                margin_m=plane.margin_m,
                gap_m=plane.gap_m,
                max_violation_m=float(violation),
                feasible=bool(violation <= self.hard_safety_tolerance_m),
            )
            planes.append(plane)
            max_violation = max(max_violation, violation)
            sum_violation += violation
            min_gap = plane.gap_m if min_gap is None else min(min_gap, plane.gap_m)
        return _ConstraintReport(
            planes=planes,
            max_violation_m=float(max_violation),
            sum_violation_m=float(sum_violation),
            min_gap_m=None if min_gap is None else float(min_gap),
            hard_ok=bool(max_violation <= self.hard_safety_tolerance_m),
            hull_count=int(len(hulls)),
        )

    def _separating_plane(
        self,
        own_vertices: np.ndarray,
        other_vertices: np.ndarray,
        interval_idx: int,
        source_kind: str,
        source_id: int,
    ) -> _PlaneConstraint:
        own = np.asarray(own_vertices, dtype=np.float32)
        other = np.asarray(other_vertices, dtype=np.float32)
        own_centroid = np.mean(own, axis=0)
        other_centroid = np.mean(other, axis=0)
        candidates = [other_centroid - own_centroid]
        pair_delta = self._closest_vertex_delta(own, other)
        candidates.append(pair_delta)
        candidates.extend(
            [
                np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
                np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
                np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
            ]
        )
        best: tuple[float, np.ndarray, float, float] | None = None
        for cand in candidates:
            n = _normalize(cand)
            if _norm(n) < 1e-9:
                continue
            if float(np.dot(other_centroid - own_centroid, n)) < 0.0:
                n = -n
            own_max = float(np.max(own @ n))
            other_min = float(np.min(other @ n))
            gap = other_min - own_max
            if best is None or gap > best[0]:
                best = (gap, n, own_max, other_min)
        if best is None:
            n = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
            own_max = float(np.max(own @ n))
            other_min = float(np.min(other @ n))
            best = (other_min - own_max, n, own_max, other_min)
        gap, normal, own_max, other_min = best
        d = -0.5 * (own_max + other_min)
        violation = max(0.0, 2.0 * self.minvo_epsilon_m - gap)
        return _PlaneConstraint(
            interval_idx=int(interval_idx),
            source_kind=str(source_kind),
            source_id=int(source_id),
            normal=np.asarray(normal, dtype=np.float32),
            d=float(d),
            margin_m=float(self.minvo_epsilon_m),
            gap_m=float(gap - 2.0 * self.minvo_epsilon_m),
            max_violation_m=float(violation),
            feasible=bool(violation <= self.hard_safety_tolerance_m),
        )

    def _closest_vertex_delta(self, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        best_delta = np.asarray(other[0] - own[0], dtype=np.float32)
        best_dist = float("inf")
        for a in own:
            deltas = other - a
            dists = np.sum(deltas * deltas, axis=1)
            idx = int(np.argmin(dists))
            dist = float(dists[idx])
            if dist < best_dist:
                best_dist = dist
                best_delta = deltas[idx]
        return best_delta.astype(np.float32)

    def _build_interval_hulls(self, planner_input: PlannerInput, num_segments: int, dt_segment: float) -> list[_IntervalHull]:
        hulls: list[_IntervalHull] = []
        intent_by_sender = {
            int(intent.sender_id): intent
            for intent in planner_input.neighbor_intents
            if intent.valid and np.asarray(intent.points).size > 0
        }
        for i in range(num_segments):
            t0 = i * dt_segment
            t1 = (i + 1) * dt_segment
            tm = 0.5 * (t0 + t1)
            for nobs in planner_input.neighbors[: self.max_neighbors]:
                intent = intent_by_sender.get(int(nobs.idx))
                points = np.asarray(
                    [
                        self._neighbor_prediction(nobs, intent, t0),
                        self._neighbor_prediction(nobs, intent, tm),
                        self._neighbor_prediction(nobs, intent, t1),
                    ],
                    dtype=np.float32,
                )
                inflation = (
                    float(planner_input.ego.radius)
                    + float(nobs.radius)
                    + self.safety_margin_m
                    + self._neighbor_inflation(nobs)
                    + (self._intent_inflation(intent) if intent is not None else 0.0)
                )
                center = 0.5 * (np.min(points, axis=0) + np.max(points, axis=0))
                half = 0.5 * (np.max(points, axis=0) - np.min(points, axis=0)) + inflation
                hulls.append(
                    _IntervalHull(
                        interval_idx=i,
                        source_kind="neighbor_intent" if intent is not None else "neighbor_cv",
                        source_id=int(nobs.idx),
                        vertices=_aabb_vertices(center, half),
                        inflation_m=float(inflation),
                    )
                )
            seen_neighbors = {int(n.idx) for n in planner_input.neighbors[: self.max_neighbors]}
            for sender_id, intent in intent_by_sender.items():
                if sender_id in seen_neighbors:
                    continue
                points = np.asarray(
                    [
                        self._intent_prediction(intent, t0),
                        self._intent_prediction(intent, tm),
                        self._intent_prediction(intent, t1),
                    ],
                    dtype=np.float32,
                )
                inflation = float(planner_input.ego.radius) + float(intent.tube_radius_m) + self.safety_margin_m
                inflation += self._intent_inflation(intent)
                center = 0.5 * (np.min(points, axis=0) + np.max(points, axis=0))
                half = 0.5 * (np.max(points, axis=0) - np.min(points, axis=0)) + inflation
                hulls.append(
                    _IntervalHull(
                        interval_idx=i,
                        source_kind="intent_only",
                        source_id=int(sender_id),
                        vertices=_aabb_vertices(center, half),
                        inflation_m=float(inflation),
                    )
                )
            for obs_idx, obs in enumerate(planner_input.obstacles):
                half = np.asarray(obs.half, dtype=np.float32) + float(planner_input.ego.radius) + self.obstacle_margin_m
                hulls.append(
                    _IntervalHull(
                        interval_idx=i,
                        source_kind="obstacle_aabb",
                        source_id=int(obs_idx),
                        vertices=_aabb_vertices(np.asarray(obs.center, dtype=np.float32), half),
                        inflation_m=float(planner_input.ego.radius) + self.obstacle_margin_m,
                    )
                )
        return hulls

    def _neighbor_prediction(self, nobs: NeighborObs, intent: IntentObs | None, t: float) -> np.ndarray:
        if intent is not None and intent.valid:
            return self._intent_prediction(intent, t)
        return (np.asarray(nobs.pos, dtype=np.float32) + np.asarray(nobs.vel, dtype=np.float32) * float(t)).astype(np.float32)

    def _intent_prediction(self, intent: IntentObs, t: float) -> np.ndarray:
        points = np.asarray(intent.points, dtype=np.float32)
        if points.ndim != 2 or points.shape[0] == 0:
            return np.zeros(3, dtype=np.float32)
        if points.shape[0] == 1:
            return points[0].astype(np.float32)
        dt = float(intent.dt_plan_s) if intent.dt_plan_s is not None and intent.dt_plan_s > 1e-9 else self._segment_dt()
        tau = max(0.0, float(t)) / max(1e-6, dt)
        lo = int(math.floor(tau))
        hi = min(points.shape[0] - 1, lo + 1)
        lo = min(points.shape[0] - 1, max(0, lo))
        alpha = min(1.0, max(0.0, tau - lo))
        return ((1.0 - alpha) * points[lo] + alpha * points[hi]).astype(np.float32)

    def _neighbor_inflation(self, nobs: NeighborObs) -> float:
        age = min(max(float(nobs.track_age_sec), float(nobs.msg_age_sec), 0.0), self.stale_age_cap_s)
        speed = _norm(np.asarray(nobs.vel, dtype=np.float32))
        stale_factor = 1.0 if bool(nobs.stale) else 0.5
        return float(stale_factor * self.stale_inflation_gain * age + self.track_uncertainty_speed_gain * speed * age)

    def _intent_inflation(self, intent: IntentObs | None) -> float:
        if intent is None:
            return 0.0
        return float(self.intent_age_inflation_gain * min(max(0.0, float(intent.intent_age_s)), self.stale_age_cap_s))

    def _bspline_samples(self, planner_input: PlannerInput, cp: np.ndarray) -> np.ndarray:
        num_segments = cp.shape[0] - 3
        matrices = _interval_matrices(num_segments, kind="a_pos_bs")
        samples: list[np.ndarray] = []
        per_interval = max(2, self.samples_per_interval)
        for i, matrix in enumerate(matrices):
            qbs = np.asarray(cp[i : i + 4], dtype=np.float32).T
            us = np.linspace(0.0, 1.0, per_interval, dtype=np.float32)
            if i > 0:
                us = us[1:]
            coeff = qbs @ matrix
            for u in us:
                powers = np.asarray([u**3, u**2, u, 1.0], dtype=np.float32)
                samples.append((coeff @ powers).astype(np.float32))
        out = np.asarray(samples, dtype=np.float32)
        if out.size == 0:
            out = np.asarray(planner_input.ego.pos, dtype=np.float32).reshape(1, 3)
        out[0] = np.asarray(planner_input.ego.pos, dtype=np.float32)
        out[-1] = self._local_target(planner_input)
        if planner_input.planar:
            out[:, 1] = float(planner_input.ego.pos[1])
        return out.astype(np.float32)

    def _minvo_intervals(self, cp: np.ndarray) -> np.ndarray:
        num_segments = cp.shape[0] - 3
        matrices = _interval_matrices(num_segments, kind="m_pos_bs2mv")
        intervals = []
        for i, matrix in enumerate(matrices):
            qbs = np.asarray(cp[i : i + 4], dtype=np.float32).T
            qmv = qbs @ matrix
            intervals.append(qmv.T.astype(np.float32))
        return np.asarray(intervals, dtype=np.float32)

    def _kinematic_report(self, planner_input: PlannerInput, cp: np.ndarray) -> _KinematicReport:
        dt = self._segment_dt()
        v_max = float(planner_input.ego.v_max)
        a_max = float(planner_input.ego.a_max)
        speed_violation = 0.0
        accel_violation = 0.0
        jerk_violation = 0.0

        vel_mats = _interval_matrices(cp.shape[0] - 3, kind="m_vel_bs2mv")
        for i, matrix in enumerate(vel_mats, start=2):
            if i + 1 >= cp.shape[0]:
                continue
            qbs = np.asarray(
                [
                    (cp[i - 1] - cp[i - 2]) / max(1e-6, dt),
                    (cp[i] - cp[i - 1]) / max(1e-6, dt),
                    (cp[i + 1] - cp[i]) / max(1e-6, dt),
                ],
                dtype=np.float32,
            ).T
            qmv = qbs @ matrix
            speed_violation = max(speed_violation, float(np.max(np.abs(qmv))) - v_max)

        for i in range(1, cp.shape[0] - 2):
            acc = (cp[i + 2] - 2.0 * cp[i + 1] + cp[i]) / max(1e-6, dt * dt)
            accel_violation = max(accel_violation, float(np.max(np.abs(acc))) - a_max)

        a_mats = _interval_matrices(cp.shape[0] - 3, kind="a_pos_bs")
        tmp = np.asarray([6.0, 0.0, 0.0, 0.0], dtype=np.float32) / max(1e-6, dt**3)
        for i, matrix in enumerate(a_mats):
            qbs = np.asarray(cp[i : i + 4], dtype=np.float32).T
            jerk = qbs @ matrix @ tmp
            jerk_violation = max(jerk_violation, float(np.max(np.abs(jerk))) - self.jerk_limit_mps3)

        speed_violation = max(0.0, speed_violation)
        accel_violation = max(0.0, accel_violation)
        jerk_violation = max(0.0, jerk_violation)
        return _KinematicReport(
            max_speed_violation_mps=float(speed_violation),
            max_accel_violation_mps2=float(accel_violation),
            max_jerk_violation_mps3=float(jerk_violation),
            ok=bool(max(speed_violation, accel_violation, jerk_violation) <= 1e-5),
        )

    def _fixed_mask(self, count: int) -> np.ndarray:
        fixed = np.zeros(count, dtype=bool)
        fixed[:3] = True
        fixed[-3:] = True
        return fixed

    def _delay_check(self, result: _PlanResult) -> bool:
        if not self.delay_check_enabled:
            return True
        return bool(result.constraint_report.max_violation_m <= self.delay_check_tolerance_m)

    def _committed_fallback(self, planner_input: PlannerInput, memory: dict[str, object]) -> _PlanResult | None:
        cp = memory.get("rmader_committed_control_points")
        if cp is None:
            return None
        until = float(memory.get("rmader_committed_until_s", -float("inf")))
        if until < float(planner_input.t):
            return None
        arr = np.asarray(cp, dtype=np.float32)
        if arr.shape != (self._control_point_count(), 3):
            return None
        arr = self._apply_boundary_conditions(planner_input, arr)
        final = self._objective(planner_input, arr, arr)
        return self._plan_result(
            label=str(memory.get("rmader_committed_label", "previous_committed")),
            cp=arr,
            initial_cost=float(final["total"]),
            final=final,
            iterations=0,
            status="delay_check_previous_committed_fallback",
        )

    def _braking_plan(self, planner_input: PlannerInput) -> _PlanResult:
        ego = planner_input.ego
        target = np.asarray(ego.pos, dtype=np.float32)
        cp = np.repeat(target.reshape(1, 3), self._control_point_count(), axis=0).astype(np.float32)
        cp = self._apply_boundary_conditions(planner_input, cp, target)
        cp[3:] = target
        if planner_input.planar:
            cp[:, 1] = float(ego.pos[1])
        final = self._objective(planner_input, cp, cp)
        return self._plan_result(
            label="delay_check_brake",
            cp=cp,
            initial_cost=float(final["total"]),
            final=final,
            iterations=0,
            status="delay_check_braking_fallback",
        )

    def _plan_result(
        self,
        *,
        label: str,
        cp: np.ndarray,
        initial_cost: float,
        final: dict[str, Any],
        iterations: int,
        status: str,
    ) -> _PlanResult:
        samples = np.asarray(final["samples"], dtype=np.float32)
        return _PlanResult(
            label=str(label),
            control_points=np.asarray(cp, dtype=np.float32),
            samples=samples,
            minvo_intervals=self._minvo_intervals(cp),
            initial_cost=float(initial_cost),
            final_cost=float(final["total"]),
            iterations=int(iterations),
            solver_status=str(status),
            constraint_report=final["constraint_report"],
            kinematic_report=final["kinematic_report"],
            smoothness_cost=float(final["smoothness_cost"]),
            path_length_m=float(final["path_length_m"]),
        )

    def _path_length(self, points: np.ndarray) -> float:
        if points.size == 0:
            return 0.0
        total = 0.0
        for a, b in zip(points[:-1], points[1:]):
            total += _norm(b - a)
        return float(total)

    def _intent_points(self, planner_input: PlannerInput, samples: np.ndarray) -> np.ndarray:
        ego_pos = np.asarray(planner_input.ego.pos, dtype=np.float32)
        out = samples.copy() if samples.size else ego_pos.reshape(1, 3)
        if out.shape[0] == 0 or _norm(out[0] - ego_pos) > 1e-5:
            out = np.vstack([ego_pos, out])
        if self.max_intent_points > 0 and out.shape[0] > self.max_intent_points:
            idx = np.linspace(0, out.shape[0] - 1, self.max_intent_points).round().astype(int)
            out = out[idx]
        return out.astype(np.float32)
