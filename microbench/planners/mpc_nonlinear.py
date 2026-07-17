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
        return outside_dist, _normalize(point - closest)
    axis = int(np.argmax(q))
    grad = np.zeros(3, dtype=np.float32)
    grad[axis] = 1.0 if float(point[axis] - center[axis]) >= 0.0 else -1.0
    return float(q[axis]), grad


@dataclass(frozen=True)
class _Seed:
    label: str
    controls: np.ndarray


@dataclass(frozen=True)
class _OptimizationResult:
    label: str
    controls: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    initial_cost: float
    final_cost: float
    iterations: int
    solver: str
    solver_status: str
    tracking_cost: float
    terminal_cost: float
    control_cost: float
    jerk_cost: float
    dynamic_penalty: float
    collision_penalty: float
    obstacle_penalty: float
    intent_penalty: float
    min_swarm_clearance_m: float | None
    min_obstacle_clearance_m: float | None
    predicted_swarm_conflict: bool
    predicted_obstacle_conflict: bool


class NonlinearMpcPlanner(ILocalPlanner):
    """Clean-room nonlinear MPC baseline for local multi-drone DAA.

    This baseline optimizes a finite-horizon double-integrator trajectory using
    multiple shooting over acceleration controls. It is deliberately scoped to
    the DAA Microbench planner contract: public local tracks, optional intent
    trajectories, static AABB obstacles, bounded speed/acceleration, and one
    velocity command returned per simulator tick.
    """

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.horizon_s = float(cfg.get("horizon_s", 2.4))
        self.step_dt_s = float(cfg.get("step_dt_s", 0.4))
        self.horizon_steps = int(cfg.get("horizon_steps", 6))
        self.replan_period_s = float(cfg.get("replan_period_s", 0.0))
        self.max_neighbors = int(cfg.get("max_neighbors", 8))
        self.max_initializations = int(cfg.get("max_initializations", 4))
        self.solver = str(cfg.get("solver", "projected_gradient")).strip().lower()
        self.opt_iterations = int(cfg.get("opt_iterations", 8))
        self.scipy_maxiter = int(cfg.get("scipy_maxiter", 35))
        self.gradient_step_accel = float(cfg.get("gradient_step_accel", 0.18))
        self.line_search_shrink = float(cfg.get("line_search_shrink", 0.55))

        self.safety_margin_m = float(cfg.get("safety_margin_m", 0.3))
        self.obstacle_margin_m = float(cfg.get("obstacle_margin_m", 0.25))
        self.near_clearance_m = float(cfg.get("near_clearance_m", 1.75))
        self.goal_slowdown_radius_m = float(cfg.get("goal_slowdown_radius_m", 6.0))
        self.intent_ttl_s = float(cfg.get("intent_ttl_s", 1.0))
        self.intent_tube_margin_m = float(cfg.get("intent_tube_margin_m", 0.3))
        self.max_intent_points = int(cfg.get("max_intent_points", 10))
        self.stale_inflation_gain = float(cfg.get("stale_inflation_gain", 0.65))
        self.stale_age_cap_s = float(cfg.get("stale_age_cap_s", 1.5))
        self.intent_age_inflation_gain = float(cfg.get("intent_age_inflation_gain", 0.3))
        self.track_uncertainty_speed_gain = float(cfg.get("track_uncertainty_speed_gain", 0.1))

        self.tracking_weight = float(cfg.get("tracking_weight", 0.35))
        self.velocity_tracking_weight = float(cfg.get("velocity_tracking_weight", 0.45))
        self.terminal_weight = float(cfg.get("terminal_weight", 6.0))
        self.progress_weight = float(cfg.get("progress_weight", 0.8))
        self.control_weight = float(cfg.get("control_weight", 0.04))
        self.jerk_weight = float(cfg.get("jerk_weight", 0.2))
        self.speed_limit_weight = float(cfg.get("speed_limit_weight", 45.0))
        self.accel_limit_weight = float(cfg.get("accel_limit_weight", 30.0))
        self.collision_weight = float(cfg.get("collision_weight", 6000.0))
        self.clearance_weight = float(cfg.get("clearance_weight", 120.0))
        self.obstacle_collision_weight = float(cfg.get("obstacle_collision_weight", 7500.0))
        self.obstacle_clearance_weight = float(cfg.get("obstacle_clearance_weight", 150.0))
        self.intent_collision_weight = float(cfg.get("intent_collision_weight", 5200.0))
        self.intent_clearance_weight = float(cfg.get("intent_clearance_weight", 110.0))
        self.warm_start_weight = float(cfg.get("warm_start_weight", 0.05))

        self._last_controls: np.ndarray | None = None
        self._last_label: str | None = None
        self._cached_controls: np.ndarray | None = None
        self._last_replan_t: float | None = None
        self.seed = 0

    def reset(self, seed: int) -> None:
        self.seed = int(seed)
        self._last_controls = None
        self._last_label = None
        self._cached_controls = None
        self._last_replan_t = None

    def compute_cmd(self, planner_input: PlannerInput) -> PlannerOutput:
        ego = planner_input.ego
        current = np.asarray(ego.vel, dtype=np.float32).copy()
        if planner_input.planar:
            current[1] = 0.0

        cached = self._maybe_reuse_controls(planner_input)
        replanned = cached is None
        if cached is None:
            seeds = self._initializations(planner_input)
            results = [self._optimize_seed(planner_input, seed) for seed in seeds]
            best = min(results, key=lambda result: result.final_cost)
            self._last_replan_t = float(planner_input.t)
        else:
            seeds = []
            best = cached

        first_accel = best.controls[0] if best.controls.size else np.zeros(3, dtype=np.float32)
        v_cmd = current + first_accel * float(planner_input.dt)
        v_cmd = _limit_delta(v_cmd, current, float(ego.a_max) * float(planner_input.dt))
        v_cmd = _clamp_speed(v_cmd, float(ego.v_max))
        if planner_input.planar:
            v_cmd[1] = 0.0

        intent_points = self._intent_points(planner_input, best.positions)
        intent = IntentMsg(
            sender_id=int(ego.idx),
            timestamp_send_s=float(planner_input.t),
            expiry_s=float(planner_input.t) + self.intent_ttl_s,
            kind="MPC_NONLINEAR_TRAJECTORY",
            tube_radius_m=float(ego.radius) + self.intent_tube_margin_m,
            points=intent_points.astype(float),
            dt_plan_s=float(self._dt()),
            mode=str(best.label),
        )

        prior_label = self._last_label
        self._last_controls = best.controls.copy()
        self._last_label = best.label
        self._cached_controls = best.controls.copy()

        return PlannerOutput(
            v_cmd=v_cmd.astype(float),
            intent_out=intent,
            debug_info={
                "mpc_nonlinear_algorithm": "clean_room_multiple_shooting_nonlinear_mpc",
                "mpc_nonlinear_solver": best.solver,
                "mpc_nonlinear_solver_status": best.solver_status,
                "mpc_nonlinear_iterations": int(best.iterations),
                "mpc_nonlinear_horizon_s": float(self.horizon_s),
                "mpc_nonlinear_step_dt_s": float(self._dt()),
                "mpc_nonlinear_horizon_steps": int(best.controls.shape[0]),
                "mpc_nonlinear_initializations": int(len(seeds)),
                "mpc_nonlinear_replanned": bool(replanned),
                "mpc_nonlinear_cached_reuse": bool(not replanned),
                "mpc_nonlinear_replan_period_s": float(self.replan_period_s),
                "mpc_nonlinear_best_seed": str(best.label),
                "mpc_nonlinear_initial_cost": float(best.initial_cost),
                "mpc_nonlinear_final_cost": float(best.final_cost),
                "mpc_nonlinear_cost_reduction": float(best.initial_cost - best.final_cost),
                "mpc_nonlinear_tracking_cost": float(best.tracking_cost),
                "mpc_nonlinear_terminal_cost": float(best.terminal_cost),
                "mpc_nonlinear_control_cost": float(best.control_cost),
                "mpc_nonlinear_jerk_cost": float(best.jerk_cost),
                "mpc_nonlinear_dynamic_penalty": float(best.dynamic_penalty),
                "mpc_nonlinear_collision_penalty": float(best.collision_penalty),
                "mpc_nonlinear_obstacle_penalty": float(best.obstacle_penalty),
                "mpc_nonlinear_intent_penalty": float(best.intent_penalty),
                "mpc_nonlinear_min_swarm_clearance_m": best.min_swarm_clearance_m,
                "mpc_nonlinear_min_obstacle_clearance_m": best.min_obstacle_clearance_m,
                "mpc_nonlinear_predicted_swarm_conflict": bool(best.predicted_swarm_conflict),
                "mpc_nonlinear_predicted_obstacle_conflict": bool(best.predicted_obstacle_conflict),
                "mpc_nonlinear_neighbor_count_considered": int(min(len(planner_input.neighbors), self.max_neighbors)),
                "mpc_nonlinear_intent_count_considered": int(sum(1 for intent_obs in planner_input.neighbor_intents if intent_obs.valid)),
                "mpc_nonlinear_obstacle_count_considered": int(len(planner_input.obstacles)),
                "mpc_nonlinear_planar": bool(planner_input.planar),
                "mpc_nonlinear_intent_points": int(intent_points.shape[0]),
                "mpc_nonlinear_accel_delta_norm": float(_norm(v_cmd - current)),
                "mpc_nonlinear_accel_delta_limit": float(float(ego.a_max) * float(planner_input.dt)),
                "mpc_nonlinear_prior_seed": prior_label,
            },
        )

    def _steps(self) -> int:
        if self.horizon_steps > 0:
            return max(2, int(self.horizon_steps))
        return max(2, int(math.ceil(max(1e-6, self.horizon_s) / max(1e-6, self.step_dt_s))))

    def _dt(self) -> float:
        return max(1e-6, self.horizon_s / self._steps())

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
        horizon_reach = v_pref * self.horizon_s
        to_goal = goal - p0
        if planner_input.planar:
            to_goal[1] = 0.0
        target = goal.copy() if _norm(to_goal) <= _norm(horizon_reach) else p0 + horizon_reach
        if planner_input.planar:
            target[1] = p0[1]
        return target.astype(np.float32)

    def _initializations(self, planner_input: PlannerInput) -> list[_Seed]:
        steps = self._steps()
        ego = planner_input.ego
        current = np.asarray(ego.vel, dtype=np.float32)
        if planner_input.planar:
            current[1] = 0.0
        v_pref = self._preferred_velocity(planner_input)
        dt = self._dt()
        base_accel = _limit_delta((v_pref - current) / dt, np.zeros(3, dtype=np.float32), float(ego.a_max))
        if planner_input.planar:
            base_accel[1] = 0.0
        seeds = [_Seed("track_goal", np.tile(base_accel, (steps, 1)).astype(np.float32))]
        seeds.append(_Seed("brake", np.tile(_limit_delta(-current / dt, np.zeros(3, dtype=np.float32), float(ego.a_max)), (steps, 1)).astype(np.float32)))

        directions = self._avoidance_directions(planner_input, v_pref)
        for label, direction in directions:
            d = _normalize(direction)
            if _norm(d) < 1e-9:
                continue
            accel = base_accel + d * float(ego.a_max) * 0.75
            accel = _limit_delta(accel, np.zeros(3, dtype=np.float32), float(ego.a_max))
            if planner_input.planar:
                accel[1] = 0.0
            seeds.append(_Seed(label, np.tile(accel, (steps, 1)).astype(np.float32)))

        if self._last_controls is not None and self._last_controls.shape == (steps, 3):
            shifted = np.vstack([self._last_controls[1:], self._last_controls[-1:]])
            if planner_input.planar:
                shifted[:, 1] = 0.0
            seeds.append(_Seed("warm_start", shifted.astype(np.float32)))

        return self._dedupe_seeds(seeds)[: max(1, self.max_initializations)]

    def _avoidance_directions(self, planner_input: PlannerInput, v_pref: np.ndarray) -> list[tuple[str, np.ndarray]]:
        ego = planner_input.ego
        p0 = np.asarray(ego.pos, dtype=np.float32)
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
                directions.append((f"agent_{int(nobs.idx)}_away", rel))
        for obs_idx, obs in enumerate(planner_input.obstacles):
            rel = p0 - _closest_point_on_aabb(p0, obs)
            if _norm(rel) <= 1e-9:
                rel = p0 - np.asarray(obs.center, dtype=np.float32)
            if planner_input.planar:
                rel[1] = 0.0
            if _norm(rel) > 1e-9:
                directions.append((f"obstacle_{obs_idx}_away", rel))
        return directions

    def _dedupe_seeds(self, seeds: list[_Seed]) -> list[_Seed]:
        out: list[_Seed] = []
        seen: set[tuple[int, ...]] = set()
        for seed in seeds:
            controls = self._project_controls(seed.controls, None)
            key = tuple(int(round(float(x) * 1000.0)) for x in controls.reshape(-1)[: min(9, controls.size)])
            if key in seen:
                continue
            seen.add(key)
            out.append(_Seed(seed.label, controls.astype(np.float32)))
        return out

    def _optimize_seed(self, planner_input: PlannerInput, seed: _Seed) -> _OptimizationResult:
        u0 = self._project_controls(seed.controls, planner_input)
        initial = self._cost_and_gradient(planner_input, u0)
        if self.solver in {"auto", "scipy", "scipy_l_bfgs_b", "lbfgsb", "l_bfgs_b"}:
            result = self._optimize_scipy(planner_input, u0)
            if result is not None:
                controls, iterations, status = result
                final = self._cost_and_gradient(planner_input, controls)
                return self._result_from_breakdown(seed, controls, initial["total"], final, iterations, "scipy_l_bfgs_b", status)
            status = "auto_projected_gradient_fallback" if self.solver == "auto" else "scipy_unavailable_projected_gradient_fallback"
        else:
            status = "projected_gradient_converged"
        controls, iterations, status = self._optimize_projected_gradient(planner_input, u0, status)
        final = self._cost_and_gradient(planner_input, controls)
        return self._result_from_breakdown(seed, controls, initial["total"], final, iterations, "projected_gradient", status)

    def _maybe_reuse_controls(self, planner_input: PlannerInput) -> _OptimizationResult | None:
        if self.replan_period_s <= 0.0:
            return None
        if self._cached_controls is None or self._last_replan_t is None:
            return None
        if float(planner_input.t) - float(self._last_replan_t) >= self.replan_period_s:
            return None
        if self._cached_controls.shape != (self._steps(), 3):
            return None
        controls = self._project_controls(self._cached_controls.copy(), planner_input)
        final = self._cost_and_gradient(planner_input, controls, need_grad=False)
        return self._result_from_breakdown(
            _Seed("cached_receding", controls),
            controls,
            float(final["total"]),
            final,
            0,
            "cached_receding_mpc",
            "cached_receding_mpc_solution",
        )

    def _optimize_scipy(self, planner_input: PlannerInput, u0: np.ndarray) -> tuple[np.ndarray, int, str] | None:
        try:
            from scipy.optimize import minimize
        except Exception:
            return None

        flat0 = u0.reshape(-1).astype(float)
        a_max = float(planner_input.ego.a_max)
        bounds: list[tuple[float, float]] = []
        for _step in range(u0.shape[0]):
            bounds.append((-a_max, a_max))
            bounds.append((0.0, 0.0) if planner_input.planar else (-a_max, a_max))
            bounds.append((-a_max, a_max))

        def unpack(x: np.ndarray) -> np.ndarray:
            return self._project_controls(np.asarray(x, dtype=np.float32).reshape(u0.shape), planner_input)

        def objective(x: np.ndarray) -> float:
            return float(self._cost_and_gradient(planner_input, unpack(x), need_grad=False)["total"])

        def jac(x: np.ndarray) -> np.ndarray:
            grad = np.asarray(self._cost_and_gradient(planner_input, unpack(x), need_grad=True)["grad"], dtype=float)
            if planner_input.planar:
                grad[:, 1] = 0.0
            return grad.reshape(-1)

        try:
            result = minimize(
                objective,
                flat0,
                jac=jac,
                bounds=bounds,
                method="L-BFGS-B",
                options={"maxiter": max(1, self.scipy_maxiter), "ftol": 1e-5, "maxls": 12},
            )
        except Exception:
            return None
        status = "converged" if bool(result.success) else f"status_{int(getattr(result, 'status', -1))}"
        return unpack(np.asarray(result.x, dtype=float)), int(getattr(result, "nit", 0) or 0), status

    def _optimize_projected_gradient(
        self,
        planner_input: PlannerInput,
        u0: np.ndarray,
        initial_status: str,
    ) -> tuple[np.ndarray, int, str]:
        controls = u0.copy()
        current = self._cost_and_gradient(planner_input, controls)
        iterations = 0
        status = initial_status
        for _ in range(max(0, self.opt_iterations)):
            iterations += 1
            grad = np.asarray(current["grad"], dtype=np.float32)
            if planner_input.planar:
                grad[:, 1] = 0.0
            grad_norm = _norm(grad)
            if grad_norm < 1e-7:
                status = "projected_gradient_stationary"
                break
            step = self.gradient_step_accel / max(1.0, grad_norm)
            accepted = False
            for _ls in range(8):
                candidate = self._project_controls(controls - grad * step, planner_input)
                candidate_cost = self._cost_and_gradient(planner_input, candidate)
                if candidate_cost["total"] <= current["total"] + 1e-6:
                    controls = candidate
                    current = candidate_cost
                    accepted = True
                    break
                step *= max(0.1, min(0.95, self.line_search_shrink))
            if not accepted:
                status = "projected_gradient_line_search_stalled"
                break
        return controls.astype(np.float32), iterations, status

    def _project_controls(self, controls: np.ndarray, planner_input: PlannerInput | None) -> np.ndarray:
        out = np.asarray(controls, dtype=np.float32).copy()
        a_max = float(planner_input.ego.a_max) if planner_input is not None else 1.0e9
        if planner_input is not None and planner_input.planar:
            out[:, 1] = 0.0
        for i in range(out.shape[0]):
            out[i] = _limit_delta(out[i], np.zeros(3, dtype=np.float32), a_max)
            if planner_input is not None and planner_input.planar:
                out[i, 1] = 0.0
        return out.astype(np.float32)

    def _rollout(self, planner_input: PlannerInput, controls: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ego = planner_input.ego
        dt = self._dt()
        pos = np.asarray(ego.pos, dtype=np.float32).copy()
        vel = np.asarray(ego.vel, dtype=np.float32).copy()
        if planner_input.planar:
            vel[1] = 0.0
        positions = []
        velocities = []
        for accel in controls:
            a = np.asarray(accel, dtype=np.float32)
            if planner_input.planar:
                a[1] = 0.0
            pos = pos + vel * dt + 0.5 * a * dt * dt
            vel = vel + a * dt
            if planner_input.planar:
                pos[1] = float(ego.pos[1])
                vel[1] = 0.0
            positions.append(pos.copy())
            velocities.append(vel.copy())
        return np.asarray(positions, dtype=np.float32), np.asarray(velocities, dtype=np.float32)

    def _cost_and_gradient(self, planner_input: PlannerInput, controls: np.ndarray, *, need_grad: bool = True) -> dict[str, Any]:
        controls = self._project_controls(controls, planner_input)
        positions, velocities = self._rollout(planner_input, controls)
        pos_grad = np.zeros_like(positions, dtype=np.float32)
        vel_grad = np.zeros_like(velocities, dtype=np.float32)
        control_grad = np.zeros_like(controls, dtype=np.float32)
        total = 0.0

        tracking, tracking_pg, tracking_vg = self._tracking_cost_and_grad(planner_input, positions, velocities)
        total += tracking
        pos_grad += tracking_pg
        vel_grad += tracking_vg

        terminal, terminal_pg = self._terminal_cost_and_grad(planner_input, positions)
        total += terminal
        pos_grad += terminal_pg

        progress, progress_pg = self._progress_reward_and_grad(planner_input, positions)
        total += progress
        pos_grad += progress_pg

        control_cost, control_cost_grad = self._control_cost_and_grad(controls)
        total += control_cost
        control_grad += control_cost_grad

        jerk_cost, jerk_grad = self._jerk_cost_and_grad(controls)
        total += jerk_cost
        control_grad += jerk_grad

        dynamic, dynamic_vg, dynamic_ug = self._dynamic_penalty_and_grad(planner_input, velocities, controls)
        total += dynamic
        vel_grad += dynamic_vg
        control_grad += dynamic_ug

        collision, collision_pg, min_swarm, swarm_conflict = self._collision_penalty_and_grad(planner_input, positions)
        total += collision
        pos_grad += collision_pg

        obstacle, obstacle_pg, min_obstacle, obstacle_conflict = self._obstacle_penalty_and_grad(planner_input, positions)
        total += obstacle
        pos_grad += obstacle_pg

        intent, intent_pg = self._intent_penalty_and_grad(planner_input, positions)
        total += intent
        pos_grad += intent_pg

        warm_start = self._warm_start_cost(controls)
        total += warm_start
        if self._last_controls is not None and self._last_controls.shape == controls.shape:
            control_grad += (2.0 * self.warm_start_weight * (controls - self._last_controls)).astype(np.float32)

        if need_grad:
            control_grad += self._backprop_rollout(planner_input, pos_grad, vel_grad)
            if planner_input.planar:
                control_grad[:, 1] = 0.0
        return {
            "total": float(total),
            "grad": control_grad,
            "positions": positions,
            "velocities": velocities,
            "tracking_cost": float(tracking),
            "terminal_cost": float(terminal),
            "control_cost": float(control_cost),
            "jerk_cost": float(jerk_cost),
            "dynamic_penalty": float(dynamic),
            "collision_penalty": float(collision),
            "obstacle_penalty": float(obstacle),
            "intent_penalty": float(intent),
            "min_swarm_clearance_m": min_swarm,
            "min_obstacle_clearance_m": min_obstacle,
            "predicted_swarm_conflict": bool(swarm_conflict),
            "predicted_obstacle_conflict": bool(obstacle_conflict),
        }

    def _tracking_cost_and_grad(
        self,
        planner_input: PlannerInput,
        positions: np.ndarray,
        velocities: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        p0 = np.asarray(planner_input.ego.pos, dtype=np.float32)
        target = self._local_target(planner_input)
        v_pref = self._preferred_velocity(planner_input)
        pos_grad = np.zeros_like(positions, dtype=np.float32)
        vel_grad = np.zeros_like(velocities, dtype=np.float32)
        cost = 0.0
        for k in range(positions.shape[0]):
            tau = (k + 1) / max(1, positions.shape[0])
            ref = p0 + (target - p0) * tau
            pos_delta = positions[k] - ref
            vel_delta = velocities[k] - v_pref
            cost += self.tracking_weight * float(np.dot(pos_delta, pos_delta))
            cost += self.velocity_tracking_weight * float(np.dot(vel_delta, vel_delta))
            pos_grad[k] += (2.0 * self.tracking_weight * pos_delta).astype(np.float32)
            vel_grad[k] += (2.0 * self.velocity_tracking_weight * vel_delta).astype(np.float32)
        return float(cost), pos_grad, vel_grad

    def _terminal_cost_and_grad(self, planner_input: PlannerInput, positions: np.ndarray) -> tuple[float, np.ndarray]:
        grad = np.zeros_like(positions, dtype=np.float32)
        if positions.size == 0:
            return 0.0, grad
        target = self._local_target(planner_input)
        delta = positions[-1] - target
        cost = self.terminal_weight * float(np.dot(delta, delta))
        grad[-1] += (2.0 * self.terminal_weight * delta).astype(np.float32)
        return float(cost), grad

    def _progress_reward_and_grad(self, planner_input: PlannerInput, positions: np.ndarray) -> tuple[float, np.ndarray]:
        grad = np.zeros_like(positions, dtype=np.float32)
        if positions.size == 0:
            return 0.0, grad
        goal_dir = _normalize(np.asarray(planner_input.goal_dir, dtype=np.float32))
        if planner_input.planar:
            goal_dir[1] = 0.0
            goal_dir = _normalize(goal_dir)
        cost = -self.progress_weight * float(np.dot(positions[-1] - np.asarray(planner_input.ego.pos, dtype=np.float32), goal_dir))
        grad[-1] += (-self.progress_weight * goal_dir).astype(np.float32)
        return float(cost), grad

    def _control_cost_and_grad(self, controls: np.ndarray) -> tuple[float, np.ndarray]:
        cost = self.control_weight * float(np.sum(controls * controls))
        return cost, (2.0 * self.control_weight * controls).astype(np.float32)

    def _jerk_cost_and_grad(self, controls: np.ndarray) -> tuple[float, np.ndarray]:
        grad = np.zeros_like(controls, dtype=np.float32)
        cost = 0.0
        prev = np.zeros(3, dtype=np.float32)
        for i, control in enumerate(controls):
            delta = control - prev
            cost += self.jerk_weight * float(np.dot(delta, delta))
            g = 2.0 * self.jerk_weight * delta
            grad[i] += g
            if i > 0:
                grad[i - 1] -= g
            prev = control
        return float(cost), grad

    def _dynamic_penalty_and_grad(
        self,
        planner_input: PlannerInput,
        velocities: np.ndarray,
        controls: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        vel_grad = np.zeros_like(velocities, dtype=np.float32)
        control_grad = np.zeros_like(controls, dtype=np.float32)
        cost = 0.0
        v_max = float(planner_input.ego.v_max)
        a_max = float(planner_input.ego.a_max)
        for i, vel in enumerate(velocities):
            speed = _norm(vel)
            excess = speed - v_max
            if excess > 0.0 and speed > 1e-9:
                cost += self.speed_limit_weight * excess * excess
                vel_grad[i] += (2.0 * self.speed_limit_weight * excess * vel / speed).astype(np.float32)
        for i, accel in enumerate(controls):
            mag = _norm(accel)
            excess = mag - a_max
            if excess > 0.0 and mag > 1e-9:
                cost += self.accel_limit_weight * excess * excess
                control_grad[i] += (2.0 * self.accel_limit_weight * excess * accel / mag).astype(np.float32)
        return float(cost), vel_grad, control_grad

    def _collision_penalty_and_grad(self, planner_input: PlannerInput, positions: np.ndarray) -> tuple[float, np.ndarray, float | None, bool]:
        grad = np.zeros_like(positions, dtype=np.float32)
        penalty = 0.0
        min_clearance: float | None = None
        conflict = False
        dt = self._dt()
        for step_idx, pos in enumerate(positions, start=1):
            t = step_idx * dt
            for nobs in planner_input.neighbors[: self.max_neighbors]:
                other = np.asarray(nobs.pos, dtype=np.float32) + np.asarray(nobs.vel, dtype=np.float32) * t
                safe_radius = (
                    float(planner_input.ego.radius)
                    + float(nobs.radius)
                    + self.safety_margin_m
                    + self._neighbor_inflation(nobs)
                )
                p, g, clearance = self._sphere_clearance_penalty_and_grad(
                    pos,
                    other,
                    safe_radius=safe_radius,
                    collision_weight=self.collision_weight,
                    clearance_weight=self.clearance_weight,
                )
                penalty += p
                grad[step_idx - 1] += g
                min_clearance = clearance if min_clearance is None else min(min_clearance, clearance)
                conflict = conflict or clearance < 0.0
        return float(penalty), grad, min_clearance, bool(conflict)

    def _intent_penalty_and_grad(self, planner_input: PlannerInput, positions: np.ndarray) -> tuple[float, np.ndarray]:
        grad = np.zeros_like(positions, dtype=np.float32)
        penalty = 0.0
        for step_idx, pos in enumerate(positions, start=1):
            for intent in planner_input.neighbor_intents:
                if not intent.valid or np.asarray(intent.points).size == 0:
                    continue
                other = self._intent_prediction(intent, step_idx)
                safe_radius = float(planner_input.ego.radius) + float(intent.tube_radius_m) + self.safety_margin_m + self._intent_inflation(intent)
                p, g, _clearance = self._sphere_clearance_penalty_and_grad(
                    pos,
                    other,
                    safe_radius=safe_radius,
                    collision_weight=self.intent_collision_weight,
                    clearance_weight=self.intent_clearance_weight,
                )
                penalty += p
                grad[step_idx - 1] += g
        return float(penalty), grad

    def _obstacle_penalty_and_grad(self, planner_input: PlannerInput, positions: np.ndarray) -> tuple[float, np.ndarray, float | None, bool]:
        grad = np.zeros_like(positions, dtype=np.float32)
        if not planner_input.obstacles:
            return 0.0, grad, None, False
        penalty = 0.0
        min_clearance: float | None = None
        conflict = False
        for step_idx, pos in enumerate(positions):
            for obs in planner_input.obstacles:
                signed_dist, dist_grad = _signed_distance_and_grad_to_aabb(pos, obs)
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
                grad[step_idx] += term_grad.astype(np.float32)
                min_clearance = clearance if min_clearance is None else min(min_clearance, clearance)
                conflict = conflict or clearance < 0.0
        return float(penalty), grad, min_clearance, bool(conflict)

    def _sphere_clearance_penalty_and_grad(
        self,
        point: np.ndarray,
        other: np.ndarray,
        *,
        safe_radius: float,
        collision_weight: float,
        clearance_weight: float,
    ) -> tuple[float, np.ndarray, float]:
        rel = np.asarray(point, dtype=np.float32) - np.asarray(other, dtype=np.float32)
        dist = _norm(rel)
        direction = np.asarray([1.0, 0.0, 0.0], dtype=np.float32) if dist <= 1e-9 else rel / dist
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

    def _backprop_rollout(self, planner_input: PlannerInput, pos_grad: np.ndarray, vel_grad: np.ndarray) -> np.ndarray:
        controls = np.zeros_like(pos_grad, dtype=np.float32)
        dt = self._dt()
        steps = pos_grad.shape[0]
        for i in range(steps):
            g = np.zeros(3, dtype=np.float32)
            for k in range(i, steps):
                g += vel_grad[k] * dt
                g += pos_grad[k] * (dt * dt * (float(k - i) + 0.5))
            controls[i] = g
        if planner_input.planar:
            controls[:, 1] = 0.0
        return controls.astype(np.float32)

    def _neighbor_inflation(self, nobs: NeighborObs) -> float:
        age = max(float(nobs.track_age_sec), float(nobs.msg_age_sec), 0.0)
        age = min(age, self.stale_age_cap_s)
        speed = _norm(np.asarray(nobs.vel, dtype=np.float32))
        stale_factor = 1.0 if bool(nobs.stale) else 0.5
        return float(stale_factor * self.stale_inflation_gain * age + self.track_uncertainty_speed_gain * speed * age)

    def _intent_inflation(self, intent: IntentObs) -> float:
        age = max(0.0, float(intent.intent_age_s))
        return float(self.intent_age_inflation_gain * min(age, self.stale_age_cap_s))

    def _intent_prediction(self, intent: IntentObs, step_idx: int) -> np.ndarray:
        points = np.asarray(intent.points, dtype=np.float32)
        if points.ndim != 2 or points.shape[0] == 0:
            return np.zeros(3, dtype=np.float32)
        if intent.dt_plan_s is not None and intent.dt_plan_s > 1e-9:
            idx = min(points.shape[0] - 1, max(0, int(round((step_idx * self._dt()) / float(intent.dt_plan_s)))))
        else:
            idx = min(points.shape[0] - 1, step_idx)
        return points[idx].astype(np.float32)

    def _warm_start_cost(self, controls: np.ndarray) -> float:
        if self._last_controls is None or self._last_controls.shape != controls.shape:
            return 0.0
        delta = controls - self._last_controls
        return float(self.warm_start_weight * np.sum(delta * delta))

    def _result_from_breakdown(
        self,
        seed: _Seed,
        controls: np.ndarray,
        initial_cost: float,
        final: dict[str, Any],
        iterations: int,
        solver: str,
        status: str,
    ) -> _OptimizationResult:
        return _OptimizationResult(
            label=seed.label,
            controls=np.asarray(controls, dtype=np.float32),
            positions=np.asarray(final["positions"], dtype=np.float32),
            velocities=np.asarray(final["velocities"], dtype=np.float32),
            initial_cost=float(initial_cost),
            final_cost=float(final["total"]),
            iterations=int(iterations),
            solver=solver,
            solver_status=status,
            tracking_cost=float(final["tracking_cost"]),
            terminal_cost=float(final["terminal_cost"]),
            control_cost=float(final["control_cost"]),
            jerk_cost=float(final["jerk_cost"]),
            dynamic_penalty=float(final["dynamic_penalty"]),
            collision_penalty=float(final["collision_penalty"]),
            obstacle_penalty=float(final["obstacle_penalty"]),
            intent_penalty=float(final["intent_penalty"]),
            min_swarm_clearance_m=final["min_swarm_clearance_m"],
            min_obstacle_clearance_m=final["min_obstacle_clearance_m"],
            predicted_swarm_conflict=bool(final["predicted_swarm_conflict"]),
            predicted_obstacle_conflict=bool(final["predicted_obstacle_conflict"]),
        )

    def _intent_points(self, planner_input: PlannerInput, positions: np.ndarray) -> np.ndarray:
        ego_pos = np.asarray(planner_input.ego.pos, dtype=np.float32)
        out = np.vstack([ego_pos, positions]) if positions.size else ego_pos.reshape(1, 3)
        if self.max_intent_points > 0 and out.shape[0] > self.max_intent_points:
            idx = np.linspace(0, out.shape[0] - 1, self.max_intent_points).round().astype(int)
            out = out[idx]
        return out.astype(np.float32)
