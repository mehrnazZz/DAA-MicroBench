from __future__ import annotations

import math
import numpy as np

from microbench.planners.base import ILocalPlanner
from microbench.types import AABBObs, NeighborObs, PlannerInput, PlannerOutput


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _normalize(v: np.ndarray) -> np.ndarray:
    n = _norm(v)
    if n < 1e-9:
        return np.zeros(3, dtype=np.float32)
    return (v / n).astype(np.float32)


def _clamp_speed(v: np.ndarray, v_max: float) -> np.ndarray:
    n = _norm(v)
    if n <= v_max or n < 1e-9:
        return v.astype(np.float32)
    return (v / n * v_max).astype(np.float32)


def _limit_delta(v: np.ndarray, current: np.ndarray, max_delta: float) -> np.ndarray:
    delta = np.asarray(v, dtype=np.float32) - np.asarray(current, dtype=np.float32)
    n = _norm(delta)
    if n <= max_delta or n < 1e-9:
        return np.asarray(v, dtype=np.float32)
    return (np.asarray(current, dtype=np.float32) + delta / n * max_delta).astype(np.float32)


def _closest_point_on_aabb(point: np.ndarray, obs: AABBObs) -> np.ndarray:
    center = np.asarray(obs.center, dtype=np.float32)
    half = np.asarray(obs.half, dtype=np.float32)
    return np.minimum(np.maximum(point, center - half), center + half)


def _signed_distance_to_aabb(point: np.ndarray, obs: AABBObs) -> float:
    center = np.asarray(obs.center, dtype=np.float32)
    half = np.asarray(obs.half, dtype=np.float32)
    q = np.abs(np.asarray(point, dtype=np.float32) - center) - half
    outside = np.maximum(q, 0.0)
    outside_dist = _norm(outside)
    if outside_dist > 1e-9:
        return outside_dist
    return float(np.max(q))


def _perp_xz(v: np.ndarray) -> np.ndarray:
    return np.asarray([v[2], 0.0, -v[0]], dtype=np.float32)


class MpcLocalPlanner(ILocalPlanner):
    """Deterministic local MPC-style baseline.

    This is a lightweight predictive baseline rather than a full nonlinear MPC
    solver. It samples one-step-reachable velocity commands, rolls each command
    forward against constant-velocity neighbor predictions and static AABBs,
    then selects the lowest-cost command.
    """

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.horizon_s = float(cfg.get("horizon_s", 2.0))
        self.rollout_dt_s = float(cfg.get("rollout_dt_s", 0.2))
        self.max_neighbors = int(cfg.get("max_neighbors", 8))
        self.candidate_samples_2d = int(cfg.get("candidate_samples_2d", 24))
        self.candidate_samples_3d = int(cfg.get("candidate_samples_3d", 42))
        self.safety_margin_m = float(cfg.get("safety_margin_m", 0.25))
        self.obstacle_margin_m = float(cfg.get("obstacle_margin_m", 0.2))
        self.stale_inflation_gain = float(cfg.get("stale_inflation_gain", 0.6))
        self.goal_slowdown_radius_m = float(cfg.get("goal_slowdown_radius_m", 6.0))
        self.tracking_weight = float(cfg.get("tracking_weight", 1.0))
        self.progress_weight = float(cfg.get("progress_weight", 0.2))
        self.smoothness_weight = float(cfg.get("smoothness_weight", 0.15))
        self.collision_weight = float(cfg.get("collision_weight", 3000.0))
        self.clearance_weight = float(cfg.get("clearance_weight", 40.0))
        self.obstacle_weight = float(cfg.get("obstacle_weight", 80.0))
        self.approach_weight = float(cfg.get("approach_weight", 20000.0))
        self.low_speed_weight = float(cfg.get("low_speed_weight", 0.05))
        self.collision_clearance_m = float(cfg.get("collision_clearance_m", 0.0))
        self.near_clearance_m = float(cfg.get("near_clearance_m", 1.5))
        self.direction_scales = tuple(float(x) for x in cfg.get("direction_scales", (1.0, 0.5)))

    def reset(self, seed: int) -> None:
        _ = seed

    def compute_cmd(self, planner_input: PlannerInput) -> PlannerOutput:
        ego = planner_input.ego
        current = np.asarray(ego.vel, dtype=np.float32)
        if planner_input.planar:
            current[1] = 0.0
        v_pref = self._preferred_velocity(planner_input)
        max_delta = max(0.0, float(ego.a_max) * float(planner_input.dt))
        candidates = self._candidates(planner_input, current, v_pref, max_delta)

        best_idx = 0
        best_cost = float("inf")
        best_breakdown: dict[str, float | None] = {}
        for idx, candidate in enumerate(candidates):
            cost, breakdown = self._score_candidate(planner_input, candidate, v_pref, current, max_delta)
            if cost < best_cost:
                best_idx = idx
                best_cost = cost
                best_breakdown = breakdown

        v_cmd = candidates[best_idx] if candidates else _limit_delta(v_pref, current, max_delta)
        v_cmd = _clamp_speed(v_cmd, float(ego.v_max))
        if planner_input.planar:
            v_cmd[1] = 0.0

        min_clearance = best_breakdown.get("min_pred_clearance_m")
        return PlannerOutput(
            v_cmd=v_cmd.astype(float),
            debug_info={
                "mpc_horizon_s": float(self.horizon_s),
                "mpc_horizon_steps": int(self._horizon_steps()),
                "mpc_candidates": int(len(candidates)),
                "mpc_best_index": int(best_idx),
                "mpc_best_cost": float(best_cost),
                "mpc_min_pred_clearance_m": min_clearance,
                "mpc_collision_penalty": float(best_breakdown.get("collision_penalty", 0.0) or 0.0),
                "mpc_obstacle_penalty": float(best_breakdown.get("obstacle_penalty", 0.0) or 0.0),
                "mpc_approach_penalty": float(best_breakdown.get("approach_penalty", 0.0) or 0.0),
                "mpc_accel_delta_norm": float(_norm(v_cmd - current)),
                "mpc_accel_delta_limit": float(max_delta),
                "mpc_planar": bool(planner_input.planar),
            },
        )

    def _preferred_velocity(self, planner_input: PlannerInput) -> np.ndarray:
        ego = planner_input.ego
        goal_dir = np.asarray(planner_input.goal_dir, dtype=np.float32)
        if planner_input.planar:
            goal_dir[1] = 0.0
        goal_dir = _normalize(goal_dir)
        goal_dist = _norm(np.asarray(ego.goal, dtype=np.float32) - np.asarray(ego.pos, dtype=np.float32))
        soft_speed = float(ego.v_max) * min(1.0, goal_dist / max(1e-6, self.goal_slowdown_radius_m))
        stop_speed = math.sqrt(max(0.0, 2.0 * max(1e-6, float(ego.a_max)) * goal_dist))
        speed = min(float(ego.v_max), max(soft_speed, min(float(ego.v_max), stop_speed)))
        return (goal_dir * speed).astype(np.float32)

    def _horizon_steps(self) -> int:
        return max(1, int(math.ceil(max(1e-6, self.horizon_s) / max(1e-6, self.rollout_dt_s))))

    def _candidates(
        self,
        planner_input: PlannerInput,
        current: np.ndarray,
        v_pref: np.ndarray,
        max_delta: float,
    ) -> list[np.ndarray]:
        ego = planner_input.ego
        directions = self._candidate_directions(planner_input, current, v_pref)
        candidates: list[np.ndarray] = []
        candidates.append(_clamp_speed(current, float(ego.v_max)))
        candidates.append(_limit_delta(v_pref, current, max_delta))
        for direction in directions:
            d = _normalize(direction)
            if _norm(d) < 1e-9:
                continue
            for scale in self.direction_scales:
                candidate = current + d * max_delta * max(0.0, scale)
                candidate = _clamp_speed(candidate, float(ego.v_max))
                if planner_input.planar:
                    candidate[1] = 0.0
                candidates.append(candidate.astype(np.float32))
        return self._dedupe(candidates, planner_input.planar)

    def _candidate_directions(self, planner_input: PlannerInput, current: np.ndarray, v_pref: np.ndarray) -> list[np.ndarray]:
        directions: list[np.ndarray] = []
        goal_delta = v_pref - current
        if _norm(goal_delta) > 1e-9:
            directions.append(goal_delta)
        if _norm(v_pref) > 1e-9:
            directions.append(v_pref)
            directions.append(-v_pref)
            lateral = _perp_xz(v_pref)
            directions.extend([lateral, -lateral])

        p_i = np.asarray(planner_input.ego.pos, dtype=np.float32)
        for nobs in planner_input.neighbors[: self.max_neighbors]:
            rel = p_i - np.asarray(nobs.pos, dtype=np.float32)
            if planner_input.planar:
                rel[1] = 0.0
            if _norm(rel) > 1e-9:
                directions.append(rel)
                directions.append(rel + _perp_xz(rel) * (1.0 if int(planner_input.ego.idx) % 2 else -1.0))
        for obs in planner_input.obstacles:
            rel = p_i - _closest_point_on_aabb(p_i, obs)
            if planner_input.planar:
                rel[1] = 0.0
            if _norm(rel) > 1e-9:
                directions.append(rel)

        directions.extend(self._uniform_directions(planner_input.planar))
        if planner_input.planar:
            for direction in directions:
                direction[1] = 0.0
        return directions

    def _uniform_directions(self, planar: bool) -> list[np.ndarray]:
        if planar:
            samples = max(4, self.candidate_samples_2d)
            return [
                np.asarray([math.cos(2.0 * math.pi * i / samples), 0.0, math.sin(2.0 * math.pi * i / samples)], dtype=np.float32)
                for i in range(samples)
            ]

        samples = max(6, self.candidate_samples_3d)
        out: list[np.ndarray] = []
        phi = math.pi * (3.0 - math.sqrt(5.0))
        for i in range(samples):
            y = 1.0 - (2.0 * i + 1.0) / samples
            r = math.sqrt(max(0.0, 1.0 - y * y))
            theta = phi * i
            out.append(np.asarray([math.cos(theta) * r, y, math.sin(theta) * r], dtype=np.float32))
        return out

    def _dedupe(self, candidates: list[np.ndarray], planar: bool) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        seen: set[tuple[int, int, int]] = set()
        for candidate in candidates:
            v = np.asarray(candidate, dtype=np.float32).copy()
            if planar:
                v[1] = 0.0
            key = tuple(int(round(float(x) * 10000.0)) for x in v)
            if key in seen:
                continue
            seen.add(key)
            out.append(v)
        return out

    def _score_candidate(
        self,
        planner_input: PlannerInput,
        candidate: np.ndarray,
        v_pref: np.ndarray,
        current: np.ndarray,
        max_delta: float,
    ) -> tuple[float, dict[str, float | None]]:
        goal_dir = _normalize(np.asarray(planner_input.goal_dir, dtype=np.float32))
        if planner_input.planar:
            goal_dir[1] = 0.0
            goal_dir = _normalize(goal_dir)
        tracking = self.tracking_weight * _norm(candidate - v_pref) ** 2
        progress = -self.progress_weight * float(np.dot(candidate, goal_dir))
        smoothness = self.smoothness_weight * (_norm(candidate - current) / max(1e-6, max_delta)) ** 2
        low_speed = self.low_speed_weight * max(0.0, float(planner_input.ego.v_max) - _norm(candidate))

        collision_penalty, obstacle_penalty, approach_penalty, min_clearance = self._rollout_risk(planner_input, candidate)
        total = tracking + progress + smoothness + low_speed + collision_penalty + obstacle_penalty + approach_penalty
        return float(total), {
            "collision_penalty": float(collision_penalty),
            "obstacle_penalty": float(obstacle_penalty),
            "approach_penalty": float(approach_penalty),
            "min_pred_clearance_m": min_clearance,
        }

    def _rollout_risk(self, planner_input: PlannerInput, candidate: np.ndarray) -> tuple[float, float, float, float | None]:
        ego = planner_input.ego
        p_i0 = np.asarray(ego.pos, dtype=np.float32)
        dt = max(1e-6, self.rollout_dt_s)
        collision_penalty = 0.0
        obstacle_penalty = 0.0
        approach_penalty = 0.0
        min_clearance = float("inf")
        has_risk_object = bool(planner_input.neighbors[: self.max_neighbors] or planner_input.obstacles)
        steps = self._horizon_steps()

        for nobs in planner_input.neighbors[: self.max_neighbors]:
            approach_penalty += self._approach_penalty(planner_input, nobs, candidate)
        for obs in planner_input.obstacles:
            approach_penalty += self._obstacle_approach_penalty(planner_input, obs, candidate)

        for step in range(1, steps + 1):
            tau = step * dt
            p_i = p_i0 + np.asarray(candidate, dtype=np.float32) * tau
            for nobs in planner_input.neighbors[: self.max_neighbors]:
                clearance = self._neighbor_clearance(planner_input, nobs, p_i, tau)
                min_clearance = min(min_clearance, clearance)
                early = 1.0 + max(0.0, (steps - step + 1) / max(1, steps))
                collision_penalty += early * self._clearance_penalty(clearance, self.collision_weight, self.clearance_weight)
            for obs in planner_input.obstacles:
                clearance = _signed_distance_to_aabb(p_i, obs) - float(ego.radius) - self.obstacle_margin_m
                min_clearance = min(min_clearance, clearance)
                early = 1.0 + max(0.0, (steps - step + 1) / max(1, steps))
                obstacle_penalty += early * self._clearance_penalty(clearance, self.collision_weight, self.obstacle_weight)

        if not has_risk_object:
            return float(collision_penalty), float(obstacle_penalty), float(approach_penalty), None
        return float(collision_penalty), float(obstacle_penalty), float(approach_penalty), float(min_clearance)

    def _neighbor_clearance(
        self,
        planner_input: PlannerInput,
        nobs: NeighborObs,
        ego_pos: np.ndarray,
        tau: float,
    ) -> float:
        ego = planner_input.ego
        p_j = np.asarray(nobs.pos, dtype=np.float32) + np.asarray(nobs.vel, dtype=np.float32) * tau
        rel = np.asarray(ego_pos, dtype=np.float32) - p_j
        if planner_input.planar:
            rel[1] = 0.0
        age_inflation = self.stale_inflation_gain * max(0.0, float(nobs.msg_age_sec))
        radius = float(ego.radius) + float(nobs.radius) + self.safety_margin_m + age_inflation
        return _norm(rel) - radius

    def _approach_penalty(self, planner_input: PlannerInput, nobs: NeighborObs, candidate: np.ndarray) -> float:
        ego = planner_input.ego
        rel = np.asarray(nobs.pos, dtype=np.float32) - np.asarray(ego.pos, dtype=np.float32)
        if planner_input.planar:
            rel[1] = 0.0
        dist = _norm(rel)
        if dist < 1e-9:
            return self.approach_weight
        age_inflation = self.stale_inflation_gain * max(0.0, float(nobs.msg_age_sec))
        radius = float(ego.radius) + float(nobs.radius) + self.safety_margin_m + age_inflation
        clearance = dist - radius
        rel_hat = rel / dist
        rel_vel = np.asarray(candidate, dtype=np.float32) - np.asarray(nobs.vel, dtype=np.float32)
        if planner_input.planar:
            rel_vel[1] = 0.0
        closing_speed = max(0.0, float(np.dot(rel_vel, rel_hat)))
        if closing_speed <= 1e-9:
            return 0.0
        ttc = clearance / max(1e-6, closing_speed)
        if ttc > self.horizon_s:
            return 0.0
        urgency = min(1.0, max(0.0, (self.horizon_s - max(0.0, ttc)) / max(1e-6, self.horizon_s)))
        return float(self.approach_weight * urgency * closing_speed * closing_speed)

    def _obstacle_approach_penalty(self, planner_input: PlannerInput, obs: AABBObs, candidate: np.ndarray) -> float:
        ego = planner_input.ego
        p_i = np.asarray(ego.pos, dtype=np.float32)
        closest = _closest_point_on_aabb(p_i, obs)
        rel = closest - p_i
        if planner_input.planar:
            rel[1] = 0.0
        dist = _norm(rel)
        if dist < 1e-9:
            return self.approach_weight
        clearance = dist - float(ego.radius) - self.obstacle_margin_m
        rel_hat = rel / dist
        velocity = np.asarray(candidate, dtype=np.float32)
        if planner_input.planar:
            velocity[1] = 0.0
        closing_speed = max(0.0, float(np.dot(velocity, rel_hat)))
        if closing_speed <= 1e-9:
            return 0.0
        ttc = clearance / max(1e-6, closing_speed)
        if ttc > self.horizon_s:
            return 0.0
        urgency = min(1.0, max(0.0, (self.horizon_s - max(0.0, ttc)) / max(1e-6, self.horizon_s)))
        return float(self.approach_weight * urgency * closing_speed * closing_speed)

    def _clearance_penalty(self, clearance: float, collision_weight: float, near_weight: float) -> float:
        if clearance <= self.collision_clearance_m:
            return collision_weight * (1.0 + (self.collision_clearance_m - clearance) ** 2)
        if clearance < self.near_clearance_m:
            return near_weight * (self.near_clearance_m - clearance) ** 2
        return 0.0
