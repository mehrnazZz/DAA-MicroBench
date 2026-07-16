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


def _closest_point_on_aabb(point: np.ndarray, obs: AABBObs) -> np.ndarray:
    center = np.asarray(obs.center, dtype=np.float32)
    half = np.asarray(obs.half, dtype=np.float32)
    return np.minimum(np.maximum(np.asarray(point, dtype=np.float32), center - half), center + half)


def _aabb_gap_sq(point: np.ndarray, center: np.ndarray, half: np.ndarray) -> float:
    dx = max(abs(float(point[0]) - float(center[0])) - float(half[0]), 0.0)
    dy = max(abs(float(point[1]) - float(center[1])) - float(half[1]), 0.0)
    dz = max(abs(float(point[2]) - float(center[2])) - float(half[2]), 0.0)
    return dx * dx + dy * dy + dz * dz


@dataclass(frozen=True)
class _Seed:
    label: str
    positions: np.ndarray
    offset_norm_m: float


@dataclass(frozen=True)
class _TubeConstraint:
    step_idx: int
    source_kind: str
    source_id: int
    normal: np.ndarray
    b: float
    buffer_m: float


@dataclass(frozen=True)
class _ObstacleData:
    source_id: int
    center: np.ndarray
    half: np.ndarray
    inflated_half: np.ndarray
    buffer_m: float


@dataclass(frozen=True)
class _TubeReport:
    constraints: list[_TubeConstraint]
    max_violation_m: float
    sum_violation_m: float
    min_slack_m: float | None
    hard_ok: bool
    neighbor_constraint_count: int
    intent_constraint_count: int
    obstacle_constraint_count: int


@dataclass(frozen=True)
class _KinematicReport:
    max_speed_violation_mps: float
    max_accel_violation_mps2: float
    ok: bool


@dataclass(frozen=True)
class _PlanResult:
    label: str
    positions: np.ndarray
    initial_cost: float
    final_cost: float
    iterations: int
    solver_status: str
    tube_report: _TubeReport
    kinematic_report: _KinematicReport
    smoothness_cost: float
    path_length_m: float
    fallback: str


class BvcTubeDmpcPlanner(ILocalPlanner):
    """Tube-based distributed MPC with hard buffered Voronoi-cell constraints.

    This is a clean-room planner for the DAA Microbench velocity-command
    contract. Each agent builds a time-indexed local tube from buffered
    Voronoi-cell halfspaces against neighbor predictions and static obstacle
    separating halfspaces, then optimizes a short double-integrator trajectory
    whose waypoints are projected back into that convex tube.
    """

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.horizon_s = float(cfg.get("horizon_s", 2.8))
        self.step_dt_s = float(cfg.get("step_dt_s", 0.4))
        self.horizon_steps = int(cfg.get("horizon_steps", 7))
        self.replan_period_s = float(cfg.get("replan_period_s", 0.2))
        self.max_neighbors = int(cfg.get("max_neighbors", 10))
        self.max_initializations = int(cfg.get("max_initializations", 4))
        self.opt_iterations = int(cfg.get("opt_iterations", 5))
        self.projection_iterations = int(cfg.get("projection_iterations", 5))
        self.gradient_step_m = float(cfg.get("gradient_step_m", 0.18))
        self.line_search_shrink = float(cfg.get("line_search_shrink", 0.55))
        self.safety_margin_m = float(cfg.get("safety_margin_m", 0.35))
        self.obstacle_margin_m = float(cfg.get("obstacle_margin_m", 0.3))
        self.obstacle_broadphase_margin_m = float(cfg.get("obstacle_broadphase_margin_m", 6.0))
        self.hard_tolerance_m = float(cfg.get("hard_tolerance_m", 0.04))
        self.near_clearance_m = float(cfg.get("near_clearance_m", 1.5))
        self.goal_slowdown_radius_m = float(cfg.get("goal_slowdown_radius_m", 6.0))
        self.intent_ttl_s = float(cfg.get("intent_ttl_s", 1.0))
        self.intent_tube_margin_m = float(cfg.get("intent_tube_margin_m", 0.35))
        self.coordination_message_ttl_s = float(cfg.get("coordination_message_ttl_s", 0.75))
        self.max_intent_points = int(cfg.get("max_intent_points", 12))
        self.stale_age_cap_s = float(cfg.get("stale_age_cap_s", 1.5))
        self.stale_inflation_gain = float(cfg.get("stale_inflation_gain", 0.75))
        self.intent_age_inflation_gain = float(cfg.get("intent_age_inflation_gain", 0.35))
        self.track_uncertainty_speed_gain = float(cfg.get("track_uncertainty_speed_gain", 0.12))
        self.offset_scales_m = tuple(float(x) for x in cfg.get("offset_scales_m", (0.0, 2.0, 4.0)))
        self.vertical_offset_scales_m = tuple(float(x) for x in cfg.get("vertical_offset_scales_m", (1.8, 3.2)))

        self.reference_weight = float(cfg.get("reference_weight", 0.08))
        self.warm_start_weight = float(cfg.get("warm_start_weight", 0.12))
        self.terminal_weight = float(cfg.get("terminal_weight", 5.0))
        self.progress_weight = float(cfg.get("progress_weight", 0.6))
        self.smoothness_weight = float(cfg.get("smoothness_weight", 2.4))
        self.path_length_weight = float(cfg.get("path_length_weight", 0.04))
        self.tube_violation_weight = float(cfg.get("tube_violation_weight", 25000.0))
        self.kinematic_violation_weight = float(cfg.get("kinematic_violation_weight", 3500.0))

        self.seed = 0
        self._last_positions: np.ndarray | None = None
        self._last_label: str | None = None
        self._last_plan_t: float | None = None
        self._last_replan_t: float | None = None

    def reset(self, seed: int) -> None:
        self.seed = int(seed)
        self._last_positions = None
        self._last_label = None
        self._last_plan_t = None
        self._last_replan_t = None

    def compute_cmd(self, planner_input: PlannerInput) -> PlannerOutput:
        ego = planner_input.ego
        current = np.asarray(ego.vel, dtype=np.float32).copy()
        if planner_input.planar:
            current[1] = 0.0

        prior_label = self._last_label
        cached = self._maybe_reuse_plan(planner_input)
        replanned = cached is None
        seeds: list[_Seed] = []
        if cached is not None:
            best = cached
            used = cached
        else:
            seeds = self._initializations(planner_input)
            results = [self._optimize_seed(planner_input, seed) for seed in seeds]
            best = min(
                results,
                key=lambda result: (
                    not result.tube_report.hard_ok,
                    result.tube_report.max_violation_m,
                    not result.kinematic_report.ok,
                    result.final_cost,
                ),
            )

            used = best
            if not best.tube_report.hard_ok:
                used = self._braking_plan(planner_input, status="tube_projection_braking_fallback")

        dt = self._dt()
        desired_v = (used.positions[1] - np.asarray(ego.pos, dtype=np.float32)) / max(1e-6, dt)
        v_cmd = _limit_delta(desired_v, current, float(ego.a_max) * float(planner_input.dt))
        v_cmd = _clamp_speed(v_cmd, float(ego.v_max))
        if planner_input.planar:
            v_cmd[1] = 0.0

        intent_points = self._intent_points(planner_input, used.positions)
        intent = IntentMsg(
            sender_id=int(ego.idx),
            timestamp_send_s=float(planner_input.t),
            expiry_s=float(planner_input.t) + self.intent_ttl_s,
            kind="BVC_TUBE_DMPC_TRAJECTORY",
            tube_radius_m=float(ego.radius) + self.intent_tube_margin_m,
            points=intent_points.astype(float),
            dt_plan_s=float(dt),
            mode=f"{used.label}:{used.fallback}",
        )
        msg = make_intent_trajectory(
            sender_id=int(ego.idx),
            recipient_id=None,
            now_s=float(planner_input.t),
            trajectory=used.positions,
            dt_plan_s=float(dt),
            expiry_s=float(planner_input.t) + self.intent_ttl_s,
            tube_radius_m=float(ego.radius) + self.intent_tube_margin_m,
            ttl_s=self.coordination_message_ttl_s,
        )
        msg.payload.update(
            {
                "algorithm": "bvc_tube_dmpc",
                "tube_constraint_count": int(used.tube_report.neighbor_constraint_count),
                "obstacle_constraint_count": int(used.tube_report.obstacle_constraint_count),
                "fallback": str(used.fallback),
            }
        )

        if used.tube_report.hard_ok:
            self._last_positions = used.positions.copy()
            self._last_label = used.label
            self._last_plan_t = float(planner_input.t)
            if replanned:
                self._last_replan_t = float(planner_input.t)

        report = used.tube_report
        kin = used.kinematic_report
        best_report = best.tube_report
        debug = {
            "bvc_tube_dmpc_algorithm": "tube_based_distributed_mpc_buffered_voronoi_cells",
            "bvc_tube_dmpc_reference": "clean-room BVC/B-UAVC-style hard spatial partitioning baseline",
            "bvc_tube_dmpc_solver": "projected_convex_tube_position_qp",
            "bvc_tube_dmpc_solver_status": str(best.solver_status),
            "bvc_tube_dmpc_horizon_steps": int(used.positions.shape[0] - 1),
            "bvc_tube_dmpc_step_dt_s": float(dt),
            "bvc_tube_dmpc_initializations": int(len(seeds)),
            "bvc_tube_dmpc_iterations": int(best.iterations),
            "bvc_tube_dmpc_replanned": bool(replanned),
            "bvc_tube_dmpc_cached_reuse": bool(not replanned),
            "bvc_tube_dmpc_replan_period_s": float(self.replan_period_s),
            "bvc_tube_dmpc_best_topology": str(best.label),
            "bvc_tube_dmpc_used_topology": str(used.label),
            "bvc_tube_dmpc_initial_cost": float(best.initial_cost),
            "bvc_tube_dmpc_final_cost": float(best.final_cost),
            "bvc_tube_dmpc_cost_reduction": float(best.initial_cost - best.final_cost),
            "bvc_tube_dmpc_path_length_m": float(used.path_length_m),
            "bvc_tube_dmpc_smoothness_cost": float(used.smoothness_cost),
            "bvc_tube_dmpc_hard_cell_ok": bool(report.hard_ok),
            "bvc_tube_dmpc_candidate_hard_cell_ok": bool(best_report.hard_ok),
            "bvc_tube_dmpc_cell_constraint_count": int(len(report.constraints)),
            "bvc_tube_dmpc_neighbor_constraint_count": int(report.neighbor_constraint_count),
            "bvc_tube_dmpc_intent_constraint_count": int(report.intent_constraint_count),
            "bvc_tube_dmpc_obstacle_constraint_count": int(report.obstacle_constraint_count),
            "bvc_tube_dmpc_max_cell_violation_m": float(report.max_violation_m),
            "bvc_tube_dmpc_candidate_max_cell_violation_m": float(best_report.max_violation_m),
            "bvc_tube_dmpc_sum_cell_violation_m": float(report.sum_violation_m),
            "bvc_tube_dmpc_min_cell_slack_m": report.min_slack_m,
            "bvc_tube_dmpc_kinematic_ok": bool(kin.ok),
            "bvc_tube_dmpc_max_speed_violation_mps": float(kin.max_speed_violation_mps),
            "bvc_tube_dmpc_max_accel_violation_mps2": float(kin.max_accel_violation_mps2),
            "bvc_tube_dmpc_fallback": str(used.fallback),
            "bvc_tube_dmpc_neighbor_count_considered": int(min(len(planner_input.neighbors), self.max_neighbors)),
            "bvc_tube_dmpc_intent_count_considered": int(sum(1 for intent_obs in planner_input.neighbor_intents if intent_obs.valid)),
            "bvc_tube_dmpc_obstacle_count_considered": int(len(planner_input.obstacles)),
            "bvc_tube_dmpc_agent_messages": 1,
            "bvc_tube_dmpc_planar": bool(planner_input.planar),
            "bvc_tube_dmpc_intent_points": int(intent_points.shape[0]),
            "bvc_tube_dmpc_prior_label": prior_label,
            "bvc_tube_dmpc_accel_delta_norm": float(_norm(v_cmd - current)),
            "bvc_tube_dmpc_accel_delta_limit": float(float(ego.a_max) * float(planner_input.dt)),
        }
        return PlannerOutput(v_cmd=v_cmd.astype(float), intent_out=intent, messages_out=[msg], debug_info=debug)

    def _maybe_reuse_plan(self, planner_input: PlannerInput) -> _PlanResult | None:
        if self.replan_period_s <= 0.0:
            return None
        if self._last_positions is None or self._last_plan_t is None or self._last_replan_t is None:
            return None
        if float(planner_input.t) - float(self._last_replan_t) >= self.replan_period_s:
            return None

        shifted = self._shift_cached_positions(planner_input)
        report = self._tube_report(planner_input, shifted)
        if not report.hard_ok:
            return None
        kin = self._kinematic_report(planner_input, shifted)
        smoothness, _ = self._smoothness_cost_and_grad(shifted)
        final = {
            "total": float(smoothness + self.path_length_weight * self._path_length(shifted)),
            "tube_report": report,
            "kinematic_report": kin,
            "smoothness_cost": float(smoothness),
            "path_length_m": float(self._path_length(shifted)),
        }
        label = self._last_label or "cached"
        return self._plan_result(
            f"{label}:reuse",
            shifted,
            final["total"],
            final,
            0,
            "cached_receding_tube",
            fallback="none",
        )

    def _shift_cached_positions(self, planner_input: PlannerInput) -> np.ndarray:
        assert self._last_positions is not None
        assert self._last_plan_t is not None
        old = np.asarray(self._last_positions, dtype=np.float32)
        dt = self._dt()
        elapsed = max(0.0, float(planner_input.t) - float(self._last_plan_t))
        times = np.arange(old.shape[0], dtype=np.float32) * float(dt)
        shifted = np.zeros_like(old, dtype=np.float32)
        for k in range(old.shape[0]):
            shifted[k] = self._sample_cached_position(old, times, elapsed + k * dt)
        shifted[0] = np.asarray(planner_input.ego.pos, dtype=np.float32)
        if planner_input.planar:
            shifted[:, 1] = float(planner_input.ego.pos[1])
        shifted = self._project_kinematic(planner_input, shifted)
        return shifted.astype(np.float32)

    def _sample_cached_position(self, points: np.ndarray, times: np.ndarray, sample_t: float) -> np.ndarray:
        if points.shape[0] == 0:
            return np.zeros(3, dtype=np.float32)
        if points.shape[0] == 1 or sample_t <= 0.0:
            return points[0].astype(np.float32)
        if sample_t >= float(times[-1]):
            return points[-1].astype(np.float32)
        hi = int(np.searchsorted(times, float(sample_t), side="right"))
        lo = max(0, hi - 1)
        hi = min(points.shape[0] - 1, hi)
        denom = max(1e-6, float(times[hi] - times[lo]))
        alpha = min(1.0, max(0.0, (float(sample_t) - float(times[lo])) / denom))
        return ((1.0 - alpha) * points[lo] + alpha * points[hi]).astype(np.float32)

    def _dt(self) -> float:
        if self.step_dt_s > 1e-6:
            return float(self.step_dt_s)
        return float(self.horizon_s) / max(1, int(self.horizon_steps))

    def _steps(self) -> int:
        return max(2, int(self.horizon_steps))

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
        direction = _normalize(v_pref if _norm(v_pref) > 1e-9 else goal - p0)
        goal_delta = goal - p0
        if planner_input.planar:
            goal_delta[1] = 0.0
        goal_dist = _norm(goal_delta)
        reach = float(ego.v_max) * self._dt() * self._steps()
        target = goal.copy() if goal_dist <= reach else p0 + direction * reach
        if planner_input.planar:
            target[1] = p0[1]
        return target.astype(np.float32)

    def _initializations(self, planner_input: PlannerInput) -> list[_Seed]:
        p0 = np.asarray(planner_input.ego.pos, dtype=np.float32)
        target = self._local_target(planner_input)
        seeds = [self._seed_positions(planner_input, target, np.zeros(3, dtype=np.float32), "direct")]
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
                seeds.append(self._seed_positions(planner_input, target, offset, f"{label}:{scale:g}m"))
        if self._last_positions is not None and self._last_positions.shape[0] == self._steps() + 1:
            warm = self._last_positions.copy()
            warm[0] = p0
            if planner_input.planar:
                warm[:, 1] = p0[1]
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

    def _seed_positions(self, planner_input: PlannerInput, target: np.ndarray, offset: np.ndarray, label: str) -> _Seed:
        p0 = np.asarray(planner_input.ego.pos, dtype=np.float32)
        steps = self._steps()
        positions = np.zeros((steps + 1, 3), dtype=np.float32)
        positions[0] = p0
        for k in range(1, steps + 1):
            tau = k / max(1, steps)
            point = p0 + (target - p0) * tau + np.sin(np.pi * tau) * offset
            if planner_input.planar:
                point[1] = p0[1]
            positions[k] = point.astype(np.float32)
        positions = self._project_kinematic(planner_input, positions)
        return _Seed(label=label, positions=positions.astype(np.float32), offset_norm_m=_norm(offset))

    def _dedupe_seeds(self, seeds: list[_Seed]) -> list[_Seed]:
        out: list[_Seed] = []
        seen: set[tuple[int, ...]] = set()
        for seed in seeds:
            pos = np.asarray(seed.positions, dtype=np.float32)
            key_values = pos[min(2, pos.shape[0] - 1) : min(5, pos.shape[0])].reshape(-1)
            key = tuple(int(round(float(x) * 1000.0)) for x in key_values)
            if key in seen:
                continue
            seen.add(key)
            out.append(seed)
        return out

    def _optimize_seed(self, planner_input: PlannerInput, seed: _Seed) -> _PlanResult:
        positions = np.asarray(seed.positions, dtype=np.float32).copy()
        positions, _ = self._project_into_tube(planner_input, positions)
        initial = self._objective(planner_input, positions, seed.positions)
        previous = initial
        status = "projected_tube_qp_converged"
        iterations = 0
        for _ in range(max(0, self.opt_iterations)):
            iterations += 1
            grad = self._objective_gradient(planner_input, positions, seed.positions)
            grad[0] = 0.0
            if planner_input.planar:
                grad[:, 1] = 0.0
            grad_norm = _norm(grad)
            if grad_norm < 1e-8:
                status = "projected_tube_qp_stationary"
                break
            step = self.gradient_step_m / max(1.0, grad_norm)
            accepted = False
            for _ls in range(8):
                candidate = positions - grad * step
                candidate[0] = np.asarray(planner_input.ego.pos, dtype=np.float32)
                candidate = self._project_kinematic(planner_input, candidate)
                candidate, _ = self._project_into_tube(planner_input, candidate)
                current = self._objective(planner_input, candidate, seed.positions)
                if current["total"] <= previous["total"] + 1e-6:
                    positions = candidate
                    previous = current
                    accepted = True
                    break
                step *= max(0.1, min(0.95, self.line_search_shrink))
            if not accepted:
                status = "projected_tube_qp_line_search_stalled"
                break
        final = self._objective(planner_input, positions, seed.positions)
        return self._plan_result(seed.label, positions, initial["total"], final, iterations, status, fallback="none")

    def _objective(self, planner_input: PlannerInput, positions: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
        positions = np.asarray(positions, dtype=np.float32)
        target = self._local_target(planner_input)
        terminal_delta = positions[-1] - target
        terminal = self.terminal_weight * float(np.dot(terminal_delta, terminal_delta))
        reference_cost = self.reference_weight * float(np.sum((positions[1:] - np.asarray(reference, dtype=np.float32)[1:]) ** 2))
        warm = 0.0
        if self._last_positions is not None and self._last_positions.shape == positions.shape:
            warm = self.warm_start_weight * float(np.sum((positions[1:] - self._last_positions[1:]) ** 2))
        smoothness, _ = self._smoothness_cost_and_grad(positions)
        path_length = self._path_length(positions)
        progress = -self.progress_weight * float(np.dot(positions[-1] - positions[0], _normalize(target - positions[0])))
        report = self._tube_report(planner_input, positions)
        kin = self._kinematic_report(planner_input, positions)
        tube_penalty = self.tube_violation_weight * (
            report.sum_violation_m + 10.0 * report.max_violation_m * report.max_violation_m
        )
        kin_penalty = self.kinematic_violation_weight * (
            kin.max_speed_violation_mps * kin.max_speed_violation_mps
            + kin.max_accel_violation_mps2 * kin.max_accel_violation_mps2
        )
        total = terminal + reference_cost + warm + smoothness + self.path_length_weight * path_length + progress
        total += tube_penalty + kin_penalty
        return {
            "total": float(total),
            "tube_report": report,
            "kinematic_report": kin,
            "smoothness_cost": float(smoothness),
            "path_length_m": float(path_length),
        }

    def _objective_gradient(self, planner_input: PlannerInput, positions: np.ndarray, reference: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(positions, dtype=np.float32)
        target = self._local_target(planner_input)
        grad[-1] += (2.0 * self.terminal_weight * (positions[-1] - target)).astype(np.float32)
        grad[1:] += (2.0 * self.reference_weight * (positions[1:] - np.asarray(reference, dtype=np.float32)[1:])).astype(
            np.float32
        )
        if self._last_positions is not None and self._last_positions.shape == positions.shape:
            grad[1:] += (2.0 * self.warm_start_weight * (positions[1:] - self._last_positions[1:])).astype(np.float32)
        _, smooth_grad = self._smoothness_cost_and_grad(positions)
        grad += smooth_grad
        direction = _normalize(target - positions[0])
        grad[-1] += (-self.progress_weight * direction).astype(np.float32)
        if planner_input.planar:
            grad[:, 1] = 0.0
        return grad.astype(np.float32)

    def _smoothness_cost_and_grad(self, positions: np.ndarray) -> tuple[float, np.ndarray]:
        grad = np.zeros_like(positions, dtype=np.float32)
        cost = 0.0
        for k in range(1, positions.shape[0] - 1):
            second = positions[k - 1] - 2.0 * positions[k] + positions[k + 1]
            cost += self.smoothness_weight * float(np.dot(second, second))
            g = 2.0 * self.smoothness_weight * second
            grad[k - 1] += g
            grad[k] += -2.0 * g
            grad[k + 1] += g
        return float(cost), grad.astype(np.float32)

    def _project_into_tube(self, planner_input: PlannerInput, positions: np.ndarray) -> tuple[np.ndarray, _TubeReport]:
        pos = np.asarray(positions, dtype=np.float32).copy()
        pos[0] = np.asarray(planner_input.ego.pos, dtype=np.float32)
        if planner_input.planar:
            pos[:, 1] = float(planner_input.ego.pos[1])
        for _ in range(max(1, self.projection_iterations)):
            constraints = self._build_tube_constraints(planner_input, pos)
            for constraint in constraints:
                k = int(constraint.step_idx)
                h = float(np.dot(constraint.normal, pos[k]) - constraint.b)
                if h > 0.0:
                    pos[k] -= h * np.asarray(constraint.normal, dtype=np.float32)
            pos = self._project_kinematic(planner_input, pos)
            if planner_input.planar:
                pos[:, 1] = float(planner_input.ego.pos[1])
        return pos.astype(np.float32), self._tube_report(planner_input, pos)

    def _project_kinematic(self, planner_input: PlannerInput, positions: np.ndarray) -> np.ndarray:
        pos = np.asarray(positions, dtype=np.float32).copy()
        pos[0] = np.asarray(planner_input.ego.pos, dtype=np.float32)
        dt = self._dt()
        max_step = float(planner_input.ego.v_max) * dt
        for _ in range(2):
            for k in range(1, pos.shape[0]):
                delta = pos[k] - pos[k - 1]
                n = _norm(delta)
                if n > max_step:
                    pos[k] = pos[k - 1] + delta / n * max_step
            for k in range(pos.shape[0] - 2, 0, -1):
                delta = pos[k] - pos[k + 1]
                n = _norm(delta)
                if n > max_step:
                    pos[k] = pos[k + 1] + delta / n * max_step
        max_second = float(planner_input.ego.a_max) * dt * dt
        for k in range(1, pos.shape[0] - 1):
            second = pos[k - 1] - 2.0 * pos[k] + pos[k + 1]
            n = _norm(second)
            if n > max_second:
                desired = second / n * max_second
                pos[k] = 0.5 * (pos[k - 1] + pos[k + 1] - desired)
        if planner_input.planar:
            pos[:, 1] = float(planner_input.ego.pos[1])
        pos[0] = np.asarray(planner_input.ego.pos, dtype=np.float32)
        return pos.astype(np.float32)

    def _build_tube_constraints(self, planner_input: PlannerInput, positions: np.ndarray) -> list[_TubeConstraint]:
        constraints: list[_TubeConstraint] = []
        dt = self._dt()
        obstacles = self._obstacle_data(planner_input)
        intent_by_sender = {
            int(intent.sender_id): intent
            for intent in planner_input.neighbor_intents
            if intent.valid and np.asarray(intent.points).size > 0
        }
        seen_neighbors: set[int] = set()
        for k in range(1, positions.shape[0]):
            anchor = np.asarray(positions[k], dtype=np.float32)
            t = k * dt
            for nobs in planner_input.neighbors[: self.max_neighbors]:
                seen_neighbors.add(int(nobs.idx))
                intent = intent_by_sender.get(int(nobs.idx))
                other = self._neighbor_prediction(nobs, intent, t)
                inflation = self._neighbor_inflation(nobs) + (self._intent_inflation(intent) if intent is not None else 0.0)
                buffer_m = 0.5 * (float(planner_input.ego.radius) + float(nobs.radius) + self.safety_margin_m + inflation)
                constraint = self._bvc_constraint(
                    step_idx=k,
                    source_kind="neighbor_intent" if intent is not None else "neighbor_cv",
                    source_id=int(nobs.idx),
                    ego_anchor=anchor,
                    other_anchor=other,
                    buffer_m=buffer_m,
                    planar=planner_input.planar,
                )
                if constraint is not None:
                    constraints.append(constraint)
            for sender_id, intent in intent_by_sender.items():
                if sender_id in seen_neighbors:
                    continue
                other = self._intent_prediction(intent, t)
                inflation = self._intent_inflation(intent)
                buffer_m = 0.5 * (float(planner_input.ego.radius) + float(intent.tube_radius_m) + self.safety_margin_m + inflation)
                constraint = self._bvc_constraint(
                    step_idx=k,
                    source_kind="intent_only",
                    source_id=int(sender_id),
                    ego_anchor=anchor,
                    other_anchor=other,
                    buffer_m=buffer_m,
                    planar=planner_input.planar,
                )
                if constraint is not None:
                    constraints.append(constraint)
            for obs in obstacles:
                if not self._obstacle_relevant_to_anchor(anchor, obs):
                    continue
                constraint = self._obstacle_constraint(k, anchor, obs, planner_input.planar)
                if constraint is not None:
                    constraints.append(constraint)
        return constraints

    def _bvc_constraint(
        self,
        *,
        step_idx: int,
        source_kind: str,
        source_id: int,
        ego_anchor: np.ndarray,
        other_anchor: np.ndarray,
        buffer_m: float,
        planar: bool,
    ) -> _TubeConstraint | None:
        rel = np.asarray(other_anchor, dtype=np.float32) - np.asarray(ego_anchor, dtype=np.float32)
        if planar:
            rel[1] = 0.0
        dist = _norm(rel)
        if dist < 1e-6:
            return None
        normal = rel / dist
        midpoint = 0.5 * (np.asarray(ego_anchor, dtype=np.float32) + np.asarray(other_anchor, dtype=np.float32))
        b = float(np.dot(normal, midpoint) - float(buffer_m))
        return _TubeConstraint(
            step_idx=int(step_idx),
            source_kind=str(source_kind),
            source_id=int(source_id),
            normal=normal.astype(np.float32),
            b=b,
            buffer_m=float(buffer_m),
        )

    def _obstacle_constraint(
        self,
        step_idx: int,
        anchor: np.ndarray,
        obs: _ObstacleData,
        planar: bool,
    ) -> _TubeConstraint | None:
        closest = np.minimum(np.maximum(np.asarray(anchor, dtype=np.float32), obs.center - obs.half), obs.center + obs.half)
        rel = closest - np.asarray(anchor, dtype=np.float32)
        if _norm(rel) < 1e-6:
            rel = obs.center - np.asarray(anchor, dtype=np.float32)
        if planar:
            rel[1] = 0.0
        dist = _norm(rel)
        if dist < 1e-6:
            return None
        normal = rel / dist
        b = float(np.dot(normal, closest) - float(obs.buffer_m))
        return _TubeConstraint(
            step_idx=int(step_idx),
            source_kind="obstacle_aabb",
            source_id=int(obs.source_id),
            normal=normal.astype(np.float32),
            b=b,
            buffer_m=float(obs.buffer_m),
        )

    def _obstacle_data(self, planner_input: PlannerInput) -> tuple[_ObstacleData, ...]:
        buffer_m = float(planner_input.ego.radius) + self.obstacle_margin_m
        out: list[_ObstacleData] = []
        for obs_idx, obs in enumerate(planner_input.obstacles):
            center = np.asarray(obs.center, dtype=np.float32)
            half = np.asarray(obs.half, dtype=np.float32)
            inflated_half = (half + buffer_m).astype(np.float32)
            out.append(
                _ObstacleData(
                    source_id=int(obs_idx),
                    center=center,
                    half=half,
                    inflated_half=inflated_half,
                    buffer_m=float(buffer_m),
                )
            )
        return tuple(out)

    def _obstacle_relevant_to_anchor(self, anchor: np.ndarray, obs: _ObstacleData) -> bool:
        broadphase_sq = max(0.0, float(self.obstacle_broadphase_margin_m))
        broadphase_sq *= broadphase_sq
        return bool(_aabb_gap_sq(anchor, obs.center, obs.inflated_half) <= broadphase_sq)

    def _tube_report(self, planner_input: PlannerInput, positions: np.ndarray) -> _TubeReport:
        constraints = self._build_tube_constraints(planner_input, positions)
        max_violation = 0.0
        sum_violation = 0.0
        min_slack: float | None = None
        neighbor_count = 0
        intent_count = 0
        obstacle_count = 0
        for c in constraints:
            value = float(np.dot(c.normal, positions[int(c.step_idx)]) - c.b)
            violation = max(0.0, value)
            slack = -value
            max_violation = max(max_violation, violation)
            sum_violation += violation
            min_slack = slack if min_slack is None else min(min_slack, slack)
            if c.source_kind == "obstacle_aabb":
                obstacle_count += 1
            elif c.source_kind == "intent_only":
                intent_count += 1
            else:
                neighbor_count += 1
        return _TubeReport(
            constraints=constraints,
            max_violation_m=float(max_violation),
            sum_violation_m=float(sum_violation),
            min_slack_m=None if min_slack is None else float(min_slack),
            hard_ok=bool(max_violation <= self.hard_tolerance_m),
            neighbor_constraint_count=int(neighbor_count),
            intent_constraint_count=int(intent_count),
            obstacle_constraint_count=int(obstacle_count),
        )

    def _kinematic_report(self, planner_input: PlannerInput, positions: np.ndarray) -> _KinematicReport:
        dt = self._dt()
        speed_v = 0.0
        accel_v = 0.0
        for k in range(1, positions.shape[0]):
            speed_v = max(speed_v, _norm(positions[k] - positions[k - 1]) / max(1e-6, dt) - float(planner_input.ego.v_max))
        for k in range(1, positions.shape[0] - 1):
            accel = _norm(positions[k + 1] - 2.0 * positions[k] + positions[k - 1]) / max(1e-6, dt * dt)
            accel_v = max(accel_v, accel - float(planner_input.ego.a_max))
        speed_v = max(0.0, speed_v)
        accel_v = max(0.0, accel_v)
        return _KinematicReport(
            max_speed_violation_mps=float(speed_v),
            max_accel_violation_mps2=float(accel_v),
            ok=bool(max(speed_v, accel_v) <= 1e-5),
        )

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
        dt = float(intent.dt_plan_s) if intent.dt_plan_s is not None and intent.dt_plan_s > 1e-9 else self._dt()
        tau = max(0.0, float(t)) / max(1e-6, dt)
        lo = min(points.shape[0] - 1, max(0, int(math.floor(tau))))
        hi = min(points.shape[0] - 1, lo + 1)
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

    def _braking_plan(self, planner_input: PlannerInput, *, status: str) -> _PlanResult:
        ego = planner_input.ego
        dt = self._dt()
        positions = np.zeros((self._steps() + 1, 3), dtype=np.float32)
        positions[0] = np.asarray(ego.pos, dtype=np.float32)
        vel = np.asarray(ego.vel, dtype=np.float32).copy()
        if planner_input.planar:
            vel[1] = 0.0
        for k in range(1, positions.shape[0]):
            speed = _norm(vel)
            if speed > 1e-9:
                dv = min(speed, float(ego.a_max) * dt)
                vel = vel - vel / speed * dv
            positions[k] = positions[k - 1] + vel * dt
        positions = self._project_kinematic(planner_input, positions)
        positions, _ = self._project_into_tube(planner_input, positions)
        final = self._objective(planner_input, positions, positions)
        return self._plan_result("tube_brake", positions, final["total"], final, 0, status, fallback="braking_trajectory")

    def _plan_result(
        self,
        label: str,
        positions: np.ndarray,
        initial_cost: float,
        final: dict[str, Any],
        iterations: int,
        status: str,
        *,
        fallback: str,
    ) -> _PlanResult:
        return _PlanResult(
            label=str(label),
            positions=np.asarray(positions, dtype=np.float32),
            initial_cost=float(initial_cost),
            final_cost=float(final["total"]),
            iterations=int(iterations),
            solver_status=str(status),
            tube_report=final["tube_report"],
            kinematic_report=final["kinematic_report"],
            smoothness_cost=float(final["smoothness_cost"]),
            path_length_m=float(final["path_length_m"]),
            fallback=str(fallback),
        )

    def _path_length(self, points: np.ndarray) -> float:
        total = 0.0
        for a, b in zip(points[:-1], points[1:]):
            total += _norm(b - a)
        return float(total)

    def _intent_points(self, planner_input: PlannerInput, points: np.ndarray) -> np.ndarray:
        ego_pos = np.asarray(planner_input.ego.pos, dtype=np.float32)
        out = np.asarray(points, dtype=np.float32).copy() if points.size else ego_pos.reshape(1, 3)
        if out.shape[0] == 0 or _norm(out[0] - ego_pos) > 1e-5:
            out = np.vstack([ego_pos, out])
        if self.max_intent_points > 0 and out.shape[0] > self.max_intent_points:
            idx = np.linspace(0, out.shape[0] - 1, self.max_intent_points).round().astype(int)
            out = out[idx]
        return out.astype(np.float32)
