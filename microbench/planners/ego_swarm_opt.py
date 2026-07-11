from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from microbench.planners.base import ILocalPlanner
from microbench.types import AABBObs, IntentMsg, IntentObs, NeighborObs, PlannerInput, PlannerOutput


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _normalize(v: np.ndarray) -> np.ndarray:
    n = _norm(v)
    if n < 1e-9:
        return np.zeros(3, dtype=np.float32)
    return (np.asarray(v, dtype=np.float32) / n).astype(np.float32)


def _clamp_speed(v: np.ndarray, v_max: float) -> np.ndarray:
    n = _norm(v)
    if n <= v_max or n < 1e-9:
        return np.asarray(v, dtype=np.float32)
    return (np.asarray(v, dtype=np.float32) / n * v_max).astype(np.float32)


def _limit_delta(v: np.ndarray, current: np.ndarray, max_delta: float) -> np.ndarray:
    target = np.asarray(v, dtype=np.float32)
    base = np.asarray(current, dtype=np.float32)
    delta = target - base
    n = _norm(delta)
    if n <= max_delta or n < 1e-9:
        return target
    return (base + delta / n * max_delta).astype(np.float32)


def _perp_xz(v: np.ndarray, sign: float = 1.0) -> np.ndarray:
    return np.asarray([sign * v[2], 0.0, -sign * v[0]], dtype=np.float32)


def _closest_point_on_aabb(point: np.ndarray, obs: AABBObs) -> np.ndarray:
    center = np.asarray(obs.center, dtype=np.float32)
    half = np.asarray(obs.half, dtype=np.float32)
    return np.minimum(np.maximum(point, center - half), center + half)


def _signed_distance_and_grad_to_aabb(point: np.ndarray, obs: AABBObs) -> tuple[float, np.ndarray]:
    point = np.asarray(point, dtype=np.float32)
    center = np.asarray(obs.center, dtype=np.float32)
    half = np.asarray(obs.half, dtype=np.float32)
    q = np.abs(point - center) - half
    outside = np.maximum(q, 0.0)
    outside_dist = _norm(outside)
    if outside_dist > 1e-9:
        closest = _closest_point_on_aabb(point, obs)
        grad = _normalize(point - closest)
        return outside_dist, grad

    axis = int(np.argmax(q))
    grad = np.zeros(3, dtype=np.float32)
    sign = 1.0 if float(point[axis] - center[axis]) >= 0.0 else -1.0
    grad[axis] = sign
    return float(q[axis]), grad


def _basis_value(i: int, degree: int, t: float, knots: np.ndarray) -> float:
    if degree == 0:
        if knots[i] <= t < knots[i + 1]:
            return 1.0
        if abs(t - 1.0) <= 1e-12 and knots[i] <= t <= knots[i + 1] and abs(knots[i + 1] - 1.0) <= 1e-12:
            return 1.0
        return 0.0

    left_den = float(knots[i + degree] - knots[i])
    right_den = float(knots[i + degree + 1] - knots[i + 1])
    left = 0.0 if left_den <= 1e-12 else (t - float(knots[i])) / left_den * _basis_value(i, degree - 1, t, knots)
    right = (
        0.0
        if right_den <= 1e-12
        else (float(knots[i + degree + 1]) - t) / right_den * _basis_value(i + 1, degree - 1, t, knots)
    )
    return float(left + right)


def _clamped_basis_matrix(n_ctrl: int, n_samples: int, degree: int = 3) -> np.ndarray:
    degree = min(int(degree), max(0, int(n_ctrl) - 1))
    if degree <= 0:
        return np.eye(n_ctrl, dtype=np.float32)

    interior_count = max(0, n_ctrl - degree - 1)
    knots = [0.0] * (degree + 1)
    if interior_count:
        knots.extend((i + 1) / (interior_count + 1) for i in range(interior_count))
    knots.extend([1.0] * (degree + 1))
    knot_arr = np.asarray(knots, dtype=np.float64)
    ts = np.linspace(0.0, 1.0, max(2, int(n_samples)), dtype=np.float64)
    basis = np.zeros((len(ts), n_ctrl), dtype=np.float32)
    for row, t in enumerate(ts):
        for i in range(n_ctrl):
            basis[row, i] = float(_basis_value(i, degree, float(t), knot_arr))
    basis[0, :] = 0.0
    basis[0, 0] = 1.0
    basis[-1, :] = 0.0
    basis[-1, -1] = 1.0
    return basis


@dataclass(frozen=True)
class _Seed:
    label: str
    control_points: np.ndarray
    offset_norm_m: float


@dataclass(frozen=True)
class _OptimizationResult:
    label: str
    control_points: np.ndarray
    samples: np.ndarray
    initial_cost: float
    final_cost: float
    iterations: int
    solver: str
    solver_status: str
    swarm_penalty: float
    obstacle_penalty: float
    dynamic_penalty: float
    smoothness_cost: float
    min_swarm_clearance_m: float | None
    min_obstacle_clearance_m: float | None
    predicted_swarm_conflict: bool
    predicted_obstacle_conflict: bool


class EgoSwarmOptimizingPlanner(ILocalPlanner):
    """Clean-room EGO-Swarm-style control-point trajectory optimizer.

    The upstream EGO-Swarm system optimizes decentralized smooth trajectories
    and shares planned trajectories among vehicles. This class follows that
    algorithmic shape inside DAA Microbench's local velocity-command contract,
    but it is an original Python implementation and does not vendor or port the
    upstream GPL ROS/C++ code.
    """

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.horizon_s = float(cfg.get("horizon_s", 3.2))
        self.rollout_dt_s = float(cfg.get("rollout_dt_s", 0.4))
        self.control_points = int(cfg.get("control_points", 5))
        self.curve_samples = int(cfg.get("curve_samples", 7))
        self.max_neighbors = int(cfg.get("max_neighbors", 8))
        self.max_initializations = int(cfg.get("max_initializations", 3))
        self.solver = str(cfg.get("solver", "projected_gradient")).strip().lower()
        self.opt_iterations = int(cfg.get("opt_iterations", 3))
        self.scipy_maxiter = int(cfg.get("scipy_maxiter", 35))
        self.gradient_step_m = float(cfg.get("gradient_step_m", 0.08))
        self.line_search_shrink = float(cfg.get("line_search_shrink", 0.55))
        self.offset_scales_m = tuple(float(x) for x in cfg.get("offset_scales_m", (0.0, 2.5, 5.0)))
        self.vertical_offset_scales_m = tuple(float(x) for x in cfg.get("vertical_offset_scales_m", (2.0, 4.0)))
        self.safety_margin_m = float(cfg.get("safety_margin_m", 0.35))
        self.obstacle_margin_m = float(cfg.get("obstacle_margin_m", 0.25))
        self.near_clearance_m = float(cfg.get("near_clearance_m", 1.85))
        self.goal_slowdown_radius_m = float(cfg.get("goal_slowdown_radius_m", 6.0))
        self.intent_ttl_s = float(cfg.get("intent_ttl_s", 1.0))
        self.intent_tube_margin_m = float(cfg.get("intent_tube_margin_m", 0.35))
        self.max_intent_points = int(cfg.get("max_intent_points", 12))
        self.stale_inflation_gain = float(cfg.get("stale_inflation_gain", 0.7))
        self.stale_age_cap_s = float(cfg.get("stale_age_cap_s", 1.5))
        self.intent_age_inflation_gain = float(cfg.get("intent_age_inflation_gain", 0.35))
        self.track_uncertainty_speed_gain = float(cfg.get("track_uncertainty_speed_gain", 0.1))

        self.goal_weight = float(cfg.get("goal_weight", 3.0))
        self.reference_weight = float(cfg.get("reference_weight", 0.08))
        self.smoothness_weight = float(cfg.get("smoothness_weight", 2.2))
        self.path_length_weight = float(cfg.get("path_length_weight", 0.05))
        self.velocity_limit_weight = float(cfg.get("velocity_limit_weight", 35.0))
        self.acceleration_limit_weight = float(cfg.get("acceleration_limit_weight", 18.0))
        self.swarm_collision_weight = float(cfg.get("swarm_collision_weight", 5200.0))
        self.swarm_clearance_weight = float(cfg.get("swarm_clearance_weight", 120.0))
        self.obstacle_collision_weight = float(cfg.get("obstacle_collision_weight", 6800.0))
        self.obstacle_clearance_weight = float(cfg.get("obstacle_clearance_weight", 150.0))
        self.warm_start_weight = float(cfg.get("warm_start_weight", 0.12))

        self._last_control_points: np.ndarray | None = None
        self._last_label: str | None = None
        self.seed = 0

    def reset(self, seed: int) -> None:
        self.seed = int(seed)
        self._last_control_points = None
        self._last_label = None

    def compute_cmd(self, planner_input: PlannerInput) -> PlannerOutput:
        ego = planner_input.ego
        current = np.asarray(ego.vel, dtype=np.float32).copy()
        if planner_input.planar:
            current[1] = 0.0

        basis = _clamped_basis_matrix(self._control_point_count(), self._sample_count())
        seeds = self._initializations(planner_input)
        results = [self._optimize_seed(planner_input, seed, basis) for seed in seeds]
        best = min(results, key=lambda result: result.final_cost)

        samples = best.samples
        next_idx = 1 if samples.shape[0] > 1 else 0
        sample_dt = self.horizon_s / max(1, samples.shape[0] - 1)
        desired_v = (samples[next_idx] - np.asarray(ego.pos, dtype=np.float32)) / max(1e-6, sample_dt)
        v_cmd = _limit_delta(desired_v, current, float(ego.a_max) * float(planner_input.dt))
        v_cmd = _clamp_speed(v_cmd, float(ego.v_max))
        if planner_input.planar:
            v_cmd[1] = 0.0

        intent_points = self._intent_points(planner_input, samples)
        intent = IntentMsg(
            sender_id=int(ego.idx),
            timestamp_send_s=float(planner_input.t),
            expiry_s=float(planner_input.t) + self.intent_ttl_s,
            kind="EGO_SWARM_OPT_TRAJECTORY",
            tube_radius_m=float(ego.radius) + self.intent_tube_margin_m,
            points=intent_points.astype(float),
            dt_plan_s=float(sample_dt),
            mode=str(best.label),
        )

        prior_label = self._last_label
        self._last_control_points = best.control_points.copy()
        self._last_label = best.label

        final_goal_dist = _norm(np.asarray(ego.goal, dtype=np.float32) - samples[-1])
        initial_goal_dist = _norm(np.asarray(ego.goal, dtype=np.float32) - np.asarray(ego.pos, dtype=np.float32))
        return PlannerOutput(
            v_cmd=v_cmd.astype(float),
            intent_out=intent,
            debug_info={
                "ego_swarm_opt_algorithm": "clean_room_control_point_trajectory_optimization",
                "ego_swarm_opt_reference": "EGO-Swarm-style; not a port of the GPL ROS/C++ implementation",
                "ego_swarm_opt_solver": best.solver,
                "ego_swarm_opt_solver_status": best.solver_status,
                "ego_swarm_opt_iterations": int(best.iterations),
                "ego_swarm_opt_horizon_s": float(self.horizon_s),
                "ego_swarm_opt_rollout_dt_s": float(self.rollout_dt_s),
                "ego_swarm_opt_control_points": int(best.control_points.shape[0]),
                "ego_swarm_opt_curve_samples": int(samples.shape[0]),
                "ego_swarm_opt_initializations": int(len(seeds)),
                "ego_swarm_opt_best_topology": str(best.label),
                "ego_swarm_opt_initial_cost": float(best.initial_cost),
                "ego_swarm_opt_final_cost": float(best.final_cost),
                "ego_swarm_opt_cost_reduction": float(best.initial_cost - best.final_cost),
                "ego_swarm_opt_final_goal_dist_m": float(final_goal_dist),
                "ego_swarm_opt_progress_m": float(initial_goal_dist - final_goal_dist),
                "ego_swarm_opt_path_length_m": float(self._path_length(samples)),
                "ego_swarm_opt_smoothness_cost": float(best.smoothness_cost),
                "ego_swarm_opt_dynamic_penalty": float(best.dynamic_penalty),
                "ego_swarm_opt_swarm_penalty": float(best.swarm_penalty),
                "ego_swarm_opt_obstacle_penalty": float(best.obstacle_penalty),
                "ego_swarm_opt_min_swarm_clearance_m": best.min_swarm_clearance_m,
                "ego_swarm_opt_min_obstacle_clearance_m": best.min_obstacle_clearance_m,
                "ego_swarm_opt_predicted_swarm_conflict": bool(best.predicted_swarm_conflict),
                "ego_swarm_opt_predicted_obstacle_conflict": bool(best.predicted_obstacle_conflict),
                "ego_swarm_opt_neighbor_count_considered": int(min(len(planner_input.neighbors), self.max_neighbors)),
                "ego_swarm_opt_intent_count_considered": int(sum(1 for intent_obs in planner_input.neighbor_intents if intent_obs.valid)),
                "ego_swarm_opt_obstacle_count_considered": int(len(planner_input.obstacles)),
                "ego_swarm_opt_planar": bool(planner_input.planar),
                "ego_swarm_opt_intent_points": int(intent_points.shape[0]),
                "ego_swarm_opt_accel_delta_norm": float(_norm(v_cmd - current)),
                "ego_swarm_opt_accel_delta_limit": float(float(ego.a_max) * float(planner_input.dt)),
                "ego_swarm_opt_prior_label": prior_label,
            },
        )

    def _control_point_count(self) -> int:
        return max(5, int(self.control_points))

    def _sample_count(self) -> int:
        steps = max(2, int(math.ceil(max(1e-6, self.horizon_s) / max(1e-6, self.rollout_dt_s))))
        return max(steps + 1, int(self.curve_samples))

    def _preferred_velocity(self, planner_input: PlannerInput) -> np.ndarray:
        ego = planner_input.ego
        goal_dir = np.asarray(planner_input.goal_dir, dtype=np.float32).copy()
        if planner_input.planar:
            goal_dir[1] = 0.0
        goal_dir = _normalize(goal_dir)
        goal_dist = _norm(np.asarray(ego.goal, dtype=np.float32) - np.asarray(ego.pos, dtype=np.float32))
        soft_speed = float(ego.v_max) * min(1.0, goal_dist / max(1e-6, self.goal_slowdown_radius_m))
        stop_speed = math.sqrt(max(0.0, 2.0 * max(1e-6, float(ego.a_max)) * goal_dist))
        speed = min(float(ego.v_max), max(soft_speed, min(float(ego.v_max), stop_speed)))
        return (goal_dir * speed).astype(np.float32)

    def _local_target(self, planner_input: PlannerInput) -> np.ndarray:
        ego = planner_input.ego
        p0 = np.asarray(ego.pos, dtype=np.float32)
        goal = np.asarray(ego.goal, dtype=np.float32)
        v_pref = self._preferred_velocity(planner_input)
        if _norm(v_pref) < 1e-9:
            v_pref = _normalize(goal - p0) * float(ego.v_max)
        if planner_input.planar:
            v_pref[1] = 0.0
        horizon_reach = v_pref * self.horizon_s
        to_goal = goal - p0
        if planner_input.planar:
            to_goal[1] = 0.0
        target = goal.copy() if _norm(to_goal) <= _norm(horizon_reach) else p0 + horizon_reach
        if planner_input.planar:
            target[1] = p0[1]
        return target.astype(np.float32)

    def _initializations(self, planner_input: PlannerInput) -> list[_Seed]:
        ego = planner_input.ego
        p0 = np.asarray(ego.pos, dtype=np.float32)
        target = self._local_target(planner_input)
        base = self._arc_control_polygon(planner_input, target, np.zeros(3, dtype=np.float32), "direct")
        seeds = [base]

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
                seeds.append(self._arc_control_polygon(planner_input, target, offset, f"{label}:{scale:g}m"))

        if self._last_control_points is not None and self._last_control_points.shape == base.control_points.shape:
            warm = self._last_control_points.copy()
            warm[0] = p0
            warm[-1] = target
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
            if _norm(rel) <= 1e-9:
                continue
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
            closest = _closest_point_on_aabb(p0, obs)
            rel = p0 - closest
            if _norm(rel) <= 1e-9:
                rel = p0 - np.asarray(obs.center, dtype=np.float32)
            if planner_input.planar:
                rel[1] = 0.0
            if _norm(rel) > 1e-9:
                directions.append((f"obstacle_{obs_idx}_away", rel))
                directions.append((f"obstacle_{obs_idx}_side", rel + _perp_xz(rel, 1.0)))
        return directions

    def _arc_control_polygon(self, planner_input: PlannerInput, target: np.ndarray, offset: np.ndarray, label: str) -> _Seed:
        p0 = np.asarray(planner_input.ego.pos, dtype=np.float32)
        count = self._control_point_count()
        points = []
        for i in range(count):
            tau = i / max(1, count - 1)
            point = p0 + (target - p0) * tau + np.sin(np.pi * tau) * offset
            if planner_input.planar:
                point[1] = p0[1]
            points.append(point.astype(np.float32))
        return _Seed(label=label, control_points=np.asarray(points, dtype=np.float32), offset_norm_m=_norm(offset))

    def _dedupe_seeds(self, seeds: list[_Seed]) -> list[_Seed]:
        out: list[_Seed] = []
        seen: set[tuple[int, ...]] = set()
        for seed in seeds:
            cp = np.asarray(seed.control_points, dtype=np.float32)
            if cp.ndim != 2 or cp.shape[0] < 2:
                continue
            key_values = [*cp[1], *cp[-2]] if cp.shape[0] > 2 else [*cp[0], *cp[-1]]
            key = tuple(int(round(float(x) * 1000.0)) for x in key_values)
            if key in seen:
                continue
            seen.add(key)
            out.append(seed)
        return out

    def _optimize_seed(self, planner_input: PlannerInput, seed: _Seed, basis: np.ndarray) -> _OptimizationResult:
        cp0 = self._project_control_points(planner_input, np.asarray(seed.control_points, dtype=np.float32).copy())
        initial_breakdown = self._cost_and_gradient(planner_input, cp0, basis, seed.control_points)

        if self.solver in {"auto", "scipy", "scipy_l_bfgs_b", "lbfgsb", "l_bfgs_b"}:
            result = self._optimize_scipy(planner_input, cp0, basis, seed.control_points)
            if result is not None:
                cp, iterations, status = result
                final = self._cost_and_gradient(planner_input, cp, basis, seed.control_points)
                return self._result_from_breakdown(
                    seed=seed,
                    cp=cp,
                    basis=basis,
                    initial_cost=initial_breakdown["total"],
                    final=final,
                    iterations=iterations,
                    solver="scipy_l_bfgs_b",
                    status=status,
                )
            if self.solver not in {"auto"}:
                status = "scipy_unavailable_projected_gradient_fallback"
            else:
                status = "auto_projected_gradient_fallback"
        else:
            status = "projected_gradient_converged"

        cp, iterations, status = self._optimize_projected_gradient(planner_input, cp0, basis, seed.control_points, status)
        final = self._cost_and_gradient(planner_input, cp, basis, seed.control_points)
        return self._result_from_breakdown(
            seed=seed,
            cp=cp,
            basis=basis,
            initial_cost=initial_breakdown["total"],
            final=final,
            iterations=iterations,
            solver="projected_gradient",
            status=status,
        )

    def _optimize_scipy(
        self,
        planner_input: PlannerInput,
        cp0: np.ndarray,
        basis: np.ndarray,
        reference_cp: np.ndarray,
    ) -> tuple[np.ndarray, int, str] | None:
        try:
            from scipy.optimize import minimize
        except Exception:
            return None

        interior0 = cp0[1:-1].reshape(-1).astype(float)
        if interior0.size == 0:
            return cp0, 0, "trivial_no_interior"

        def unpack(x: np.ndarray) -> np.ndarray:
            cp = cp0.copy()
            cp[1:-1] = np.asarray(x, dtype=np.float32).reshape(cp[1:-1].shape)
            return self._project_control_points(planner_input, cp)

        def objective(x: np.ndarray) -> float:
            cp = unpack(x)
            return float(self._cost_and_gradient(planner_input, cp, basis, reference_cp, need_grad=False)["total"])

        def jac(x: np.ndarray) -> np.ndarray:
            cp = unpack(x)
            grad = np.asarray(self._cost_and_gradient(planner_input, cp, basis, reference_cp, need_grad=True)["grad"], dtype=float)
            if planner_input.planar:
                grad[:, 1] = 0.0
            return grad[1:-1].reshape(-1)

        try:
            result = minimize(
                objective,
                interior0,
                jac=jac,
                method="L-BFGS-B",
                options={"maxiter": max(1, self.scipy_maxiter), "ftol": 1e-5, "maxls": 12},
            )
        except Exception:
            return None
        cp = unpack(np.asarray(result.x, dtype=float))
        status = "converged" if bool(result.success) else f"status_{int(getattr(result, 'status', -1))}"
        return cp.astype(np.float32), int(getattr(result, "nit", 0) or 0), status

    def _optimize_projected_gradient(
        self,
        planner_input: PlannerInput,
        cp0: np.ndarray,
        basis: np.ndarray,
        reference_cp: np.ndarray,
        initial_status: str,
    ) -> tuple[np.ndarray, int, str]:
        cp = cp0.copy()
        previous = self._cost_and_gradient(planner_input, cp, basis, reference_cp)
        iterations = 0
        status = initial_status
        for _ in range(max(0, self.opt_iterations)):
            iterations += 1
            grad = np.asarray(previous["grad"], dtype=np.float32)
            grad[0] = 0.0
            grad[-1] = 0.0
            if planner_input.planar:
                grad[:, 1] = 0.0
            grad_norm = _norm(grad[1:-1])
            if grad_norm < 1e-7:
                status = "projected_gradient_stationary"
                break

            step = self.gradient_step_m / max(1.0, grad_norm)
            accepted = False
            for _ls in range(8):
                candidate = cp - grad * step
                candidate = self._project_control_points(planner_input, candidate)
                current = self._cost_and_gradient(planner_input, candidate, basis, reference_cp)
                if current["total"] <= previous["total"] + 1e-6:
                    cp = candidate
                    previous = current
                    accepted = True
                    break
                step *= max(0.1, min(0.95, self.line_search_shrink))
            if not accepted:
                status = "projected_gradient_line_search_stalled"
                break
        return cp.astype(np.float32), iterations, status

    def _project_control_points(self, planner_input: PlannerInput, cp: np.ndarray) -> np.ndarray:
        cp = np.asarray(cp, dtype=np.float32).copy()
        ego = planner_input.ego
        cp[0] = np.asarray(ego.pos, dtype=np.float32)
        target = self._local_target(planner_input)
        cp[-1] = target
        if planner_input.planar:
            cp[:, 1] = float(ego.pos[1])

        knot_dt = self.horizon_s / max(1, cp.shape[0] - 1)
        max_step = float(ego.v_max) * max(1e-6, knot_dt) * 1.2
        for i in range(1, cp.shape[0]):
            delta = cp[i] - cp[i - 1]
            n = _norm(delta)
            if n > max_step:
                cp[i] = cp[i - 1] + delta / n * max_step
        cp[-1] = target
        for i in range(cp.shape[0] - 2, -1, -1):
            delta = cp[i] - cp[i + 1]
            n = _norm(delta)
            if n > max_step:
                cp[i] = cp[i + 1] + delta / n * max_step
        cp[0] = np.asarray(ego.pos, dtype=np.float32)
        cp[-1] = target
        if planner_input.planar:
            cp[:, 1] = float(ego.pos[1])
        return cp.astype(np.float32)

    def _cost_and_gradient(
        self,
        planner_input: PlannerInput,
        cp: np.ndarray,
        basis: np.ndarray,
        reference_cp: np.ndarray,
        *,
        need_grad: bool = True,
    ) -> dict[str, Any]:
        cp = np.asarray(cp, dtype=np.float32)
        samples = np.asarray(basis @ cp, dtype=np.float32)
        if planner_input.planar:
            samples[:, 1] = float(planner_input.ego.pos[1])

        sample_grad = np.zeros_like(samples, dtype=np.float32)
        cp_grad = np.zeros_like(cp, dtype=np.float32)
        total = 0.0

        goal_cost, goal_grad = self._goal_cost(planner_input, samples)
        total += goal_cost
        if need_grad:
            sample_grad += goal_grad

        ref_delta = cp - np.asarray(reference_cp, dtype=np.float32)
        ref_cost = self.reference_weight * float(np.sum(ref_delta[1:-1] ** 2))
        total += ref_cost
        if need_grad:
            cp_grad[1:-1] += (2.0 * self.reference_weight * ref_delta[1:-1]).astype(np.float32)

        smoothness_cost, smoothness_grad = self._smoothness_cost_and_grad(cp)
        total += smoothness_cost
        if need_grad:
            cp_grad += smoothness_grad

        path_cost, path_grad = self._path_length_cost_and_grad(samples, basis)
        total += path_cost
        if need_grad:
            cp_grad += path_grad

        dynamic_penalty, dynamic_sample_grad = self._dynamic_penalty_and_grad(planner_input, samples)
        total += dynamic_penalty
        if need_grad:
            sample_grad += dynamic_sample_grad

        swarm_penalty, swarm_grad, min_swarm, swarm_conflict = self._swarm_penalty_and_grad(planner_input, samples)
        total += swarm_penalty
        if need_grad:
            sample_grad += swarm_grad

        obstacle_penalty, obstacle_grad, min_obstacle, obstacle_conflict = self._obstacle_penalty_and_grad(planner_input, samples)
        total += obstacle_penalty
        if need_grad:
            sample_grad += obstacle_grad

        warm_start_cost = self._warm_start_cost(cp)
        total += warm_start_cost
        if need_grad and self._last_control_points is not None and self._last_control_points.shape == cp.shape:
            cp_grad[1:-1] += (2.0 * self.warm_start_weight * (cp[1:-1] - self._last_control_points[1:-1])).astype(np.float32)

        if need_grad:
            cp_grad += np.asarray(basis.T @ sample_grad, dtype=np.float32)
            cp_grad[0] = 0.0
            cp_grad[-1] = 0.0
            if planner_input.planar:
                cp_grad[:, 1] = 0.0
        return {
            "total": float(total),
            "grad": cp_grad,
            "samples": samples,
            "smoothness_cost": float(smoothness_cost),
            "dynamic_penalty": float(dynamic_penalty),
            "swarm_penalty": float(swarm_penalty),
            "obstacle_penalty": float(obstacle_penalty),
            "min_swarm_clearance_m": min_swarm,
            "min_obstacle_clearance_m": min_obstacle,
            "predicted_swarm_conflict": bool(swarm_conflict),
            "predicted_obstacle_conflict": bool(obstacle_conflict),
        }

    def _goal_cost(self, planner_input: PlannerInput, samples: np.ndarray) -> tuple[float, np.ndarray]:
        target = self._local_target(planner_input)
        final_delta = samples[-1] - target
        grad = np.zeros_like(samples, dtype=np.float32)
        cost = self.goal_weight * float(np.dot(final_delta, final_delta))
        grad[-1] = (2.0 * self.goal_weight * final_delta).astype(np.float32)
        return cost, grad

    def _smoothness_cost_and_grad(self, cp: np.ndarray) -> tuple[float, np.ndarray]:
        grad = np.zeros_like(cp, dtype=np.float32)
        cost = 0.0
        if cp.shape[0] < 3:
            return 0.0, grad
        for i in range(1, cp.shape[0] - 1):
            second = cp[i - 1] - 2.0 * cp[i] + cp[i + 1]
            cost += self.smoothness_weight * float(np.dot(second, second))
            g = 2.0 * self.smoothness_weight * second
            grad[i - 1] += g
            grad[i] += -2.0 * g
            grad[i + 1] += g
        return float(cost), grad.astype(np.float32)

    def _path_length_cost_and_grad(self, samples: np.ndarray, basis: np.ndarray) -> tuple[float, np.ndarray]:
        sample_grad = np.zeros_like(samples, dtype=np.float32)
        cost = 0.0
        for i in range(1, samples.shape[0]):
            delta = samples[i] - samples[i - 1]
            n = _norm(delta)
            if n <= 1e-9:
                continue
            cost += self.path_length_weight * n
            g = self.path_length_weight * delta / n
            sample_grad[i] += g
            sample_grad[i - 1] -= g
        return float(cost), np.asarray(basis.T @ sample_grad, dtype=np.float32)

    def _dynamic_penalty_and_grad(self, planner_input: PlannerInput, samples: np.ndarray) -> tuple[float, np.ndarray]:
        grad = np.zeros_like(samples, dtype=np.float32)
        cost = 0.0
        ego = planner_input.ego
        dt = self.horizon_s / max(1, samples.shape[0] - 1)
        for i in range(1, samples.shape[0]):
            delta = samples[i] - samples[i - 1]
            dist = _norm(delta)
            speed = dist / max(1e-6, dt)
            excess = speed - float(ego.v_max)
            if excess > 0.0 and dist > 1e-9:
                cost += self.velocity_limit_weight * excess * excess
                g = 2.0 * self.velocity_limit_weight * excess * delta / (dist * max(1e-6, dt))
                grad[i] += g
                grad[i - 1] -= g
        for i in range(1, samples.shape[0] - 1):
            second = samples[i - 1] - 2.0 * samples[i] + samples[i + 1]
            acc = _norm(second) / max(1e-6, dt * dt)
            excess = acc - float(ego.a_max)
            if excess > 0.0 and _norm(second) > 1e-9:
                cost += self.acceleration_limit_weight * excess * excess
                g = 2.0 * self.acceleration_limit_weight * excess * second / (_norm(second) * max(1e-6, dt * dt))
                grad[i - 1] += g
                grad[i] += -2.0 * g
                grad[i + 1] += g
        return float(cost), grad.astype(np.float32)

    def _swarm_penalty_and_grad(self, planner_input: PlannerInput, samples: np.ndarray) -> tuple[float, np.ndarray, float | None, bool]:
        grad = np.zeros_like(samples, dtype=np.float32)
        if samples.size == 0:
            return 0.0, grad, None, False
        intent_by_sender = {
            int(intent.sender_id): intent
            for intent in planner_input.neighbor_intents
            if intent.valid and np.asarray(intent.points).size > 0
        }
        sample_dt = self.horizon_s / max(1, samples.shape[0] - 1)
        seen_ids: set[int] = set()
        penalty = 0.0
        min_clearance: float | None = None
        conflict = False
        for step_idx, point in enumerate(samples[1:], start=1):
            t = step_idx * sample_dt
            sample_index = step_idx
            for nobs in planner_input.neighbors[: self.max_neighbors]:
                seen_ids.add(int(nobs.idx))
                other_pos = self._neighbor_prediction(nobs, intent_by_sender.get(int(nobs.idx)), step_idx, t)
                inflation = self._neighbor_inflation(nobs)
                if int(nobs.idx) in intent_by_sender:
                    inflation += self._intent_inflation(intent_by_sender[int(nobs.idx)])
                safe_radius = float(planner_input.ego.radius) + float(nobs.radius) + self.safety_margin_m + inflation
                p, g, clearance = self._clearance_penalty_and_grad(
                    point,
                    other_pos,
                    safe_radius=safe_radius,
                    collision_weight=self.swarm_collision_weight,
                    clearance_weight=self.swarm_clearance_weight,
                )
                penalty += p
                grad[sample_index] += g
                min_clearance = clearance if min_clearance is None else min(min_clearance, clearance)
                conflict = conflict or clearance < 0.0
            for sender_id, intent in intent_by_sender.items():
                if sender_id in seen_ids:
                    continue
                other_pos = self._intent_prediction(intent, step_idx)
                safe_radius = float(planner_input.ego.radius) + float(intent.tube_radius_m) + self.safety_margin_m + self._intent_inflation(intent)
                p, g, clearance = self._clearance_penalty_and_grad(
                    point,
                    other_pos,
                    safe_radius=safe_radius,
                    collision_weight=self.swarm_collision_weight,
                    clearance_weight=self.swarm_clearance_weight,
                )
                penalty += p
                grad[sample_index] += g
                min_clearance = clearance if min_clearance is None else min(min_clearance, clearance)
                conflict = conflict or clearance < 0.0
        return float(penalty), grad.astype(np.float32), min_clearance, bool(conflict)

    def _obstacle_penalty_and_grad(self, planner_input: PlannerInput, samples: np.ndarray) -> tuple[float, np.ndarray, float | None, bool]:
        grad = np.zeros_like(samples, dtype=np.float32)
        if not planner_input.obstacles or samples.size == 0:
            return 0.0, grad, None, False
        penalty = 0.0
        min_clearance: float | None = None
        conflict = False
        for sample_index, point in enumerate(samples[1:], start=1):
            for obs in planner_input.obstacles:
                signed_dist, dist_grad = _signed_distance_and_grad_to_aabb(point, obs)
                clearance = signed_dist - float(planner_input.ego.radius) - self.obstacle_margin_m
                if clearance < 0.0:
                    term = self.obstacle_collision_weight * (1.0 - clearance) ** 2
                    term_grad = -2.0 * self.obstacle_collision_weight * (1.0 - clearance) * dist_grad
                elif clearance < self.near_clearance_m:
                    term = self.obstacle_clearance_weight * (self.near_clearance_m - clearance) ** 2
                    term_grad = -2.0 * self.obstacle_clearance_weight * (self.near_clearance_m - clearance) * dist_grad
                else:
                    term = 0.0
                    term_grad = np.zeros(3, dtype=np.float32)
                penalty += float(term)
                grad[sample_index] += term_grad.astype(np.float32)
                min_clearance = clearance if min_clearance is None else min(min_clearance, clearance)
                conflict = conflict or clearance < 0.0
        return float(penalty), grad.astype(np.float32), min_clearance, bool(conflict)

    def _clearance_penalty_and_grad(
        self,
        point: np.ndarray,
        other_pos: np.ndarray,
        *,
        safe_radius: float,
        collision_weight: float,
        clearance_weight: float,
    ) -> tuple[float, np.ndarray, float]:
        rel = np.asarray(point, dtype=np.float32) - np.asarray(other_pos, dtype=np.float32)
        dist = _norm(rel)
        if dist <= 1e-9:
            direction = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            direction = rel / dist
        clearance = dist - float(safe_radius)
        if clearance < 0.0:
            penalty = collision_weight * (1.0 - clearance) ** 2
            grad = -2.0 * collision_weight * (1.0 - clearance) * direction
        elif clearance < self.near_clearance_m:
            penalty = clearance_weight * (self.near_clearance_m - clearance) ** 2
            grad = -2.0 * clearance_weight * (self.near_clearance_m - clearance) * direction
        else:
            penalty = 0.0
            grad = np.zeros(3, dtype=np.float32)
        return float(penalty), np.asarray(grad, dtype=np.float32), float(clearance)

    def _neighbor_prediction(
        self,
        nobs: NeighborObs,
        intent: IntentObs | None,
        step_idx: int,
        t: float,
    ) -> np.ndarray:
        if intent is not None and intent.valid:
            return self._intent_prediction(intent, step_idx)
        return (np.asarray(nobs.pos, dtype=np.float32) + np.asarray(nobs.vel, dtype=np.float32) * float(t)).astype(np.float32)

    def _intent_prediction(self, intent: IntentObs, step_idx: int) -> np.ndarray:
        points = np.asarray(intent.points, dtype=np.float32)
        if points.ndim != 2 or points.shape[0] == 0:
            return np.zeros(3, dtype=np.float32)
        sample_dt = self.horizon_s / max(1, self._sample_count() - 1)
        if intent.dt_plan_s is not None and intent.dt_plan_s > 1e-9:
            idx = min(points.shape[0] - 1, max(0, int(round((step_idx * sample_dt) / float(intent.dt_plan_s)))))
        else:
            idx = min(points.shape[0] - 1, step_idx)
        return points[idx].astype(np.float32)

    def _neighbor_inflation(self, nobs: NeighborObs) -> float:
        age = max(float(nobs.track_age_sec), float(nobs.msg_age_sec), 0.0)
        age = min(age, self.stale_age_cap_s)
        speed = _norm(np.asarray(nobs.vel, dtype=np.float32))
        stale_factor = 1.0 if bool(nobs.stale) else 0.5
        return float(stale_factor * self.stale_inflation_gain * age + self.track_uncertainty_speed_gain * speed * age)

    def _intent_inflation(self, intent: IntentObs) -> float:
        age = max(0.0, float(intent.intent_age_s))
        return float(self.intent_age_inflation_gain * min(age, self.stale_age_cap_s))

    def _warm_start_cost(self, cp: np.ndarray) -> float:
        if self._last_control_points is None or self._last_control_points.shape != cp.shape:
            return 0.0
        delta = cp[1:-1] - self._last_control_points[1:-1]
        return float(self.warm_start_weight * np.sum(delta * delta))

    def _result_from_breakdown(
        self,
        *,
        seed: _Seed,
        cp: np.ndarray,
        basis: np.ndarray,
        initial_cost: float,
        final: dict[str, Any],
        iterations: int,
        solver: str,
        status: str,
    ) -> _OptimizationResult:
        samples = np.asarray(final["samples"], dtype=np.float32)
        return _OptimizationResult(
            label=seed.label,
            control_points=np.asarray(cp, dtype=np.float32),
            samples=samples,
            initial_cost=float(initial_cost),
            final_cost=float(final["total"]),
            iterations=int(iterations),
            solver=solver,
            solver_status=status,
            swarm_penalty=float(final["swarm_penalty"]),
            obstacle_penalty=float(final["obstacle_penalty"]),
            dynamic_penalty=float(final["dynamic_penalty"]),
            smoothness_cost=float(final["smoothness_cost"]),
            min_swarm_clearance_m=final["min_swarm_clearance_m"],
            min_obstacle_clearance_m=final["min_obstacle_clearance_m"],
            predicted_swarm_conflict=bool(final["predicted_swarm_conflict"]),
            predicted_obstacle_conflict=bool(final["predicted_obstacle_conflict"]),
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
