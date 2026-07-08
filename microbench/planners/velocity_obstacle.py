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
        return np.asarray(v, dtype=np.float32)
    return (np.asarray(v, dtype=np.float32) / n * v_max).astype(np.float32)


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


def _perp_xz(v: np.ndarray, sign: float = 1.0) -> np.ndarray:
    return np.asarray([sign * v[2], 0.0, -sign * v[0]], dtype=np.float32)


class VelocityObstaclePlanner(ILocalPlanner):
    """Deterministic 2D/3D velocity-obstacle cone baseline.

    The planner samples bounded velocity commands, predicts constant-velocity
    neighbor motion, and penalizes candidates that enter a finite-horizon
    velocity-obstacle cone. It is intentionally dependency-free and remains an
    experimental baseline until calibrated on official stress suites.
    """

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.time_horizon_s = float(cfg.get("time_horizon_s", 3.0))
        self.obstacle_time_horizon_s = float(cfg.get("obstacle_time_horizon_s", self.time_horizon_s))
        self.rollout_dt_s = float(cfg.get("rollout_dt_s", 0.35))
        self.max_neighbors = int(cfg.get("max_neighbors", 8))
        self.safety_margin_m = float(cfg.get("safety_margin_m", 0.25))
        self.obstacle_margin_m = float(cfg.get("obstacle_margin_m", 0.2))
        self.near_clearance_m = float(cfg.get("near_clearance_m", 1.25))
        self.stale_inflation_gain = float(cfg.get("stale_inflation_gain", 0.7))
        self.goal_slowdown_radius_m = float(cfg.get("goal_slowdown_radius_m", 6.0))
        self.candidate_samples_2d = int(cfg.get("candidate_samples_2d", 24))
        self.candidate_samples_3d = int(cfg.get("candidate_samples_3d", 36))
        self.max_candidates = int(cfg.get("max_candidates", 96))
        self.speed_scales = tuple(float(x) for x in cfg.get("speed_scales", (1.0, 0.75, 0.5, 0.25, 0.0)))
        self.tracking_weight = float(cfg.get("tracking_weight", 1.0))
        self.progress_weight = float(cfg.get("progress_weight", 0.25))
        self.smoothness_weight = float(cfg.get("smoothness_weight", 0.08))
        self.cone_weight = float(cfg.get("cone_weight", 1200.0))
        self.collision_weight = float(cfg.get("collision_weight", 5000.0))
        self.clearance_weight = float(cfg.get("clearance_weight", 80.0))
        self.obstacle_weight = float(cfg.get("obstacle_weight", 160.0))
        self.algorithm_name = str(cfg.get("algorithm_name", "velocity_obstacle_cone_sampling"))

    def reset(self, seed: int) -> None:
        self.seed = int(seed)

    def compute_cmd(self, planner_input: PlannerInput) -> PlannerOutput:
        ego = planner_input.ego
        current = np.asarray(ego.vel, dtype=np.float32).copy()
        if planner_input.planar:
            current[1] = 0.0

        v_pref = self._preferred_velocity(planner_input)
        candidates, raw_count = self._candidates(planner_input, current, v_pref)

        best_idx = 0
        best_cost = float("inf")
        best_breakdown: dict[str, float | int | None] = {}
        for idx, candidate in enumerate(candidates):
            cost, breakdown = self._score_candidate(planner_input, candidate, v_pref, current)
            if cost < best_cost:
                best_idx = idx
                best_cost = cost
                best_breakdown = breakdown

        v_cmd = candidates[best_idx] if candidates else v_pref
        v_cmd = _clamp_speed(v_cmd, float(ego.v_max))
        if planner_input.planar:
            v_cmd[1] = 0.0

        debug_info = {
            "vo_algorithm": "velocity_obstacle_cone_sampling",
            "vo_horizon_s": float(self.time_horizon_s),
            "vo_candidates": int(len(candidates)),
            "vo_candidates_raw": int(raw_count),
            "vo_candidate_limit": int(self.max_candidates),
            "vo_best_index": int(best_idx),
            "vo_best_cost": float(best_cost),
            "vo_conflict_count": int(best_breakdown.get("conflict_count", 0) or 0),
            "vo_min_ttc_s": best_breakdown.get("min_ttc_s"),
            "vo_min_pred_clearance_m": best_breakdown.get("min_pred_clearance_m"),
            "vo_cone_penalty": float(best_breakdown.get("cone_penalty", 0.0) or 0.0),
            "vo_obstacle_penalty": float(best_breakdown.get("obstacle_penalty", 0.0) or 0.0),
            "vo_planar": bool(planner_input.planar),
        }
        debug_info["vo_algorithm"] = self.algorithm_name
        debug_info.update(self._extra_debug_info(planner_input))
        return PlannerOutput(
            v_cmd=v_cmd.astype(float),
            debug_info=debug_info,
        )

    def _extra_debug_info(self, planner_input: PlannerInput) -> dict[str, object]:
        _ = planner_input
        return {}

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

    def _candidates(
        self,
        planner_input: PlannerInput,
        current: np.ndarray,
        v_pref: np.ndarray,
    ) -> tuple[list[np.ndarray], int]:
        ego = planner_input.ego
        directions = self._candidate_directions(planner_input, current, v_pref)
        candidates: list[np.ndarray] = [
            _clamp_speed(v_pref, float(ego.v_max)),
            _clamp_speed(current, float(ego.v_max)),
            np.zeros(3, dtype=np.float32),
        ]
        for direction in directions:
            d = _normalize(direction)
            if _norm(d) < 1e-9:
                continue
            for scale in self.speed_scales:
                speed = max(0.0, float(scale)) * float(ego.v_max)
                candidate = _clamp_speed(d * speed, float(ego.v_max))
                if planner_input.planar:
                    candidate[1] = 0.0
                candidates.append(candidate.astype(np.float32))

        deduped = self._dedupe(candidates, planner_input.planar)
        if self.max_candidates > 0 and len(deduped) > self.max_candidates:
            return deduped[: self.max_candidates], len(deduped)
        return deduped, len(deduped)

    def _candidate_directions(
        self,
        planner_input: PlannerInput,
        current: np.ndarray,
        v_pref: np.ndarray,
    ) -> list[np.ndarray]:
        directions: list[np.ndarray] = []
        if _norm(v_pref) > 1e-9:
            directions.append(v_pref)
            directions.append(-v_pref)
            directions.append(_perp_xz(v_pref, 1.0))
            directions.append(_perp_xz(v_pref, -1.0))
        if _norm(current) > 1e-9:
            directions.append(current)

        p_i = np.asarray(planner_input.ego.pos, dtype=np.float32)
        side_sign = -1.0 if int(planner_input.ego.idx) % 2 == 0 else 1.0
        for nobs in planner_input.neighbors[: self.max_neighbors]:
            rel = np.asarray(nobs.pos, dtype=np.float32) - p_i
            if planner_input.planar:
                rel[1] = 0.0
            if _norm(rel) <= 1e-9:
                continue
            away = -rel
            directions.append(away)
            directions.append(away + _perp_xz(rel, side_sign))
            directions.append(away - _perp_xz(rel, side_sign))
            if not planner_input.planar:
                vertical = np.asarray([0.0, side_sign, 0.0], dtype=np.float32)
                directions.extend([vertical, -vertical])

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
            samples = max(8, self.candidate_samples_2d)
            return [
                np.asarray([math.cos(2.0 * math.pi * i / samples), 0.0, math.sin(2.0 * math.pi * i / samples)], dtype=np.float32)
                for i in range(samples)
            ]

        samples = max(8, self.candidate_samples_3d)
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
    ) -> tuple[float, dict[str, float | int | None]]:
        goal_dir = np.asarray(planner_input.goal_dir, dtype=np.float32).copy()
        if planner_input.planar:
            goal_dir[1] = 0.0
        goal_dir = _normalize(goal_dir)
        tracking = self.tracking_weight * _norm(candidate - v_pref) ** 2
        progress = -self.progress_weight * float(np.dot(candidate, goal_dir))
        smoothness = self.smoothness_weight * _norm(candidate - current) ** 2

        cone_penalty = 0.0
        min_clearance: float | None = None
        min_ttc: float | None = None
        conflict_count = 0
        for nobs in planner_input.neighbors[: self.max_neighbors]:
            penalty, clearance, ttc, conflict = self._neighbor_vo_penalty(planner_input, nobs, candidate)
            cone_penalty += penalty
            conflict_count += int(conflict)
            min_clearance = clearance if min_clearance is None else min(min_clearance, clearance)
            if ttc is not None:
                min_ttc = ttc if min_ttc is None else min(min_ttc, ttc)

        obstacle_penalty, obstacle_clearance = self._obstacle_penalty(planner_input, candidate)
        if obstacle_clearance is not None:
            min_clearance = obstacle_clearance if min_clearance is None else min(min_clearance, obstacle_clearance)

        total = tracking + progress + smoothness + cone_penalty + obstacle_penalty
        return float(total), {
            "cone_penalty": float(cone_penalty),
            "obstacle_penalty": float(obstacle_penalty),
            "min_pred_clearance_m": min_clearance,
            "min_ttc_s": min_ttc,
            "conflict_count": int(conflict_count),
        }

    def _neighbor_vo_penalty(
        self,
        planner_input: PlannerInput,
        nobs: NeighborObs,
        candidate: np.ndarray,
    ) -> tuple[float, float, float | None, bool]:
        ego = planner_input.ego
        rel = np.asarray(nobs.pos, dtype=np.float32) - np.asarray(ego.pos, dtype=np.float32)
        rel_vel = np.asarray(candidate, dtype=np.float32) - np.asarray(nobs.vel, dtype=np.float32)
        if planner_input.planar:
            rel[1] = 0.0
            rel_vel[1] = 0.0

        dist = _norm(rel)
        age_inflation = self.stale_inflation_gain * max(0.0, float(nobs.msg_age_sec))
        radius = float(ego.radius) + float(nobs.radius) + self.safety_margin_m + age_inflation
        current_clearance = dist - radius
        penalty = 0.0
        ttc: float | None = None
        conflict = False

        if current_clearance <= 0.0:
            return self.collision_weight * (1.0 + current_clearance * current_clearance), current_clearance, 0.0, True

        rel_speed_sq = float(np.dot(rel_vel, rel_vel))
        if rel_speed_sq <= 1e-12:
            return 0.0, current_clearance, None, False

        closing_metric = float(np.dot(rel, rel_vel))
        if closing_metric <= 0.0:
            if current_clearance < self.near_clearance_m:
                gap = self.near_clearance_m - current_clearance
                penalty += self.clearance_weight * gap * gap
            return float(penalty), current_clearance, None, False

        t_cpa = max(0.0, min(self.time_horizon_s, closing_metric / rel_speed_sq))
        cpa = rel - rel_vel * t_cpa
        cpa_clearance = _norm(cpa) - radius
        ttc = current_clearance / max(1e-6, closing_metric / max(dist, 1e-6))
        horizon_hit = ttc <= self.time_horizon_s
        cone_hit = cpa_clearance < self.near_clearance_m and horizon_hit
        if cone_hit:
            conflict = True
            time_weight = 1.0 + max(0.0, (self.time_horizon_s - ttc) / max(1e-6, self.time_horizon_s))
            cone_depth = max(0.0, self.near_clearance_m - cpa_clearance)
            collision_depth = max(0.0, -cpa_clearance)
            penalty += self.cone_weight * time_weight * (cone_depth / max(1e-6, self.near_clearance_m)) ** 2
            penalty += self.collision_weight * time_weight * collision_depth * collision_depth
        elif current_clearance < self.near_clearance_m:
            gap = self.near_clearance_m - current_clearance
            penalty += self.clearance_weight * gap * gap

        return float(penalty), float(min(current_clearance, cpa_clearance)), float(ttc), bool(conflict)

    def _obstacle_penalty(self, planner_input: PlannerInput, candidate: np.ndarray) -> tuple[float, float | None]:
        if not planner_input.obstacles:
            return 0.0, None

        ego = planner_input.ego
        dt = max(1e-6, self.rollout_dt_s)
        steps = max(1, int(math.ceil(max(1e-6, self.obstacle_time_horizon_s) / dt)))
        p0 = np.asarray(ego.pos, dtype=np.float32)
        penalty = 0.0
        min_clearance = float("inf")
        for step in range(1, steps + 1):
            tau = step * dt
            p_i = p0 + np.asarray(candidate, dtype=np.float32) * tau
            for obs in planner_input.obstacles:
                clearance = _signed_distance_to_aabb(p_i, obs) - float(ego.radius) - self.obstacle_margin_m
                min_clearance = min(min_clearance, clearance)
                if clearance <= 0.0:
                    penalty += self.collision_weight * (1.0 + clearance * clearance)
                elif clearance < self.near_clearance_m:
                    gap = self.near_clearance_m - clearance
                    penalty += self.obstacle_weight * gap * gap
        return float(penalty), float(min_clearance)


class ReciprocalVelocityObstaclePlanner(VelocityObstaclePlanner):
    """Hybrid reciprocal velocity-obstacle baseline.

    This variant keeps the dependency-free candidate-search design, but scores
    candidates against a reciprocal/hybrid VO apex instead of treating every
    neighbor as fully noncooperative. Responsibility is deterministic and
    increases for stale tracks or lower-priority/lower-right-of-way agents.
    """

    def __init__(self, cfg: dict | None = None):
        cfg = dict(cfg or {})
        cfg.setdefault("algorithm_name", "hybrid_reciprocal_velocity_obstacle")
        super().__init__(cfg=cfg)
        self.responsibility_min = float(cfg.get("responsibility_min", 0.45))
        self.responsibility_max = float(cfg.get("responsibility_max", 0.9))
        self.priority_responsibility_gain = float(cfg.get("priority_responsibility_gain", 0.18))
        self.stale_responsibility_gain = float(cfg.get("stale_responsibility_gain", 0.25))
        self.tangent_margin_rad = float(cfg.get("tangent_margin_rad", 0.08))
        self.hrvo_apex_blend = float(cfg.get("hrvo_apex_blend", 0.65))
        self._reciprocal_cache: dict[int, tuple[np.ndarray, float]] = {}

    def compute_cmd(self, planner_input: PlannerInput) -> PlannerOutput:
        self._reciprocal_cache = self._build_reciprocal_cache(planner_input)
        try:
            return super().compute_cmd(planner_input)
        finally:
            self._reciprocal_cache = {}

    def _build_reciprocal_cache(self, planner_input: PlannerInput) -> dict[int, tuple[np.ndarray, float]]:
        cache: dict[int, tuple[np.ndarray, float]] = {}
        for nobs in planner_input.neighbors[: self.max_neighbors]:
            responsibility = self._compute_responsibility(planner_input, nobs)
            cache[int(nobs.idx)] = (
                self._compute_reciprocal_apex(planner_input, nobs, responsibility),
                responsibility,
            )
        return cache

    def _extra_debug_info(self, planner_input: PlannerInput) -> dict[str, object]:
        responsibilities = [
            self._responsibility(planner_input, nobs)
            for nobs in planner_input.neighbors[: self.max_neighbors]
        ]
        return {
            "vo_reciprocal_mode": "hrvo",
            "vo_responsibility_mean": float(sum(responsibilities) / len(responsibilities)) if responsibilities else None,
            "vo_responsibility_max": float(max(responsibilities)) if responsibilities else None,
        }

    def _responsibility(self, planner_input: PlannerInput, nobs: NeighborObs) -> float:
        cached = self._reciprocal_cache.get(int(nobs.idx))
        if cached is not None:
            return float(cached[1])
        return self._compute_responsibility(planner_input, nobs)

    def _compute_responsibility(self, planner_input: PlannerInput, nobs: NeighborObs) -> float:
        ego = planner_input.ego
        base = 0.5
        ego_priority = None
        if planner_input.agent_context is not None:
            ego_priority = int(planner_input.agent_context.priority)

        # Lower numeric priority means higher right-of-way in the existing
        # agentic baselines. Without explicit priority, use stable agent ids.
        if ego_priority is not None:
            neighbor_priority = int(nobs.idx)
            if ego_priority > neighbor_priority:
                base += self.priority_responsibility_gain
            elif ego_priority < neighbor_priority:
                base -= 0.5 * self.priority_responsibility_gain
        elif int(ego.idx) > int(nobs.idx):
            base += self.priority_responsibility_gain
        elif int(ego.idx) < int(nobs.idx):
            base -= 0.5 * self.priority_responsibility_gain

        stale_frac = min(1.0, max(0.0, float(nobs.msg_age_sec)) / max(1e-6, self.time_horizon_s))
        base += self.stale_responsibility_gain * stale_frac
        return float(min(self.responsibility_max, max(self.responsibility_min, base)))

    def _reciprocal_apex(self, planner_input: PlannerInput, nobs: NeighborObs) -> np.ndarray:
        cached = self._reciprocal_cache.get(int(nobs.idx))
        if cached is not None:
            return cached[0]
        return self._compute_reciprocal_apex(planner_input, nobs, self._responsibility(planner_input, nobs))

    def _compute_reciprocal_apex(
        self,
        planner_input: PlannerInput,
        nobs: NeighborObs,
        responsibility: float,
    ) -> np.ndarray:
        ego_vel = np.asarray(planner_input.ego.vel, dtype=np.float32).copy()
        neighbor_vel = np.asarray(nobs.vel, dtype=np.float32).copy()
        if planner_input.planar:
            ego_vel[1] = 0.0
            neighbor_vel[1] = 0.0
        vo_apex = neighbor_vel
        rvo_apex = (1.0 - responsibility) * ego_vel + responsibility * neighbor_vel
        return ((1.0 - self.hrvo_apex_blend) * vo_apex + self.hrvo_apex_blend * rvo_apex).astype(np.float32)

    def _candidates(
        self,
        planner_input: PlannerInput,
        current: np.ndarray,
        v_pref: np.ndarray,
    ) -> tuple[list[np.ndarray], int]:
        candidates, raw_count = super()._candidates(planner_input, current, v_pref)
        candidates.extend(self._reciprocal_boundary_candidates(planner_input))
        deduped = self._dedupe(candidates, planner_input.planar)
        raw_count = max(raw_count, len(deduped))
        if self.max_candidates > 0 and len(deduped) > self.max_candidates:
            return deduped[: self.max_candidates], raw_count
        return deduped, raw_count

    def _reciprocal_boundary_candidates(self, planner_input: PlannerInput) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        ego = planner_input.ego
        p_i = np.asarray(ego.pos, dtype=np.float32)
        for nobs in planner_input.neighbors[: self.max_neighbors]:
            rel = np.asarray(nobs.pos, dtype=np.float32) - p_i
            if planner_input.planar:
                rel[1] = 0.0
            dist = _norm(rel)
            if dist <= 1e-6:
                continue
            age_inflation = self.stale_inflation_gain * max(0.0, float(nobs.msg_age_sec))
            radius = float(ego.radius) + float(nobs.radius) + self.safety_margin_m + age_inflation
            theta = min(math.pi * 0.48, math.asin(min(0.98, radius / max(dist, radius + 1e-6))) + self.tangent_margin_rad)
            axis = _normalize(rel)
            apex = self._reciprocal_apex(planner_input, nobs)
            for direction in self._boundary_directions(axis, theta, planner_input.planar, int(ego.idx), int(nobs.idx)):
                for scale in self.speed_scales[:3]:
                    candidate = apex + _normalize(direction) * float(ego.v_max) * max(0.0, float(scale))
                    candidate = _clamp_speed(candidate, float(ego.v_max))
                    if planner_input.planar:
                        candidate[1] = 0.0
                    out.append(candidate.astype(np.float32))
        return out

    def _boundary_directions(
        self,
        axis: np.ndarray,
        theta: float,
        planar: bool,
        ego_idx: int,
        neighbor_idx: int,
    ) -> list[np.ndarray]:
        side = -1.0 if ego_idx < neighbor_idx else 1.0
        if planar:
            horiz = np.asarray([axis[0], 0.0, axis[2]], dtype=np.float32)
            if _norm(horiz) <= 1e-9:
                horiz = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
            horiz = _normalize(horiz)
            lateral = _normalize(_perp_xz(horiz, side))
            return [
                _normalize(horiz * math.cos(theta) + lateral * math.sin(theta)),
                _normalize(horiz * math.cos(theta) - lateral * math.sin(theta)),
                -horiz,
            ]

        helper = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(helper, axis))) > 0.92:
            helper = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        u = _normalize(np.cross(axis, helper))
        v = _normalize(np.cross(axis, u))
        dirs: list[np.ndarray] = [-axis]
        for phi in (0.0, math.pi * 0.5, math.pi, math.pi * 1.5):
            tangent = _normalize(u * math.cos(phi) + v * math.sin(phi))
            dirs.append(_normalize(axis * math.cos(theta) + tangent * math.sin(theta)))
        return dirs

    def _neighbor_vo_penalty(
        self,
        planner_input: PlannerInput,
        nobs: NeighborObs,
        candidate: np.ndarray,
    ) -> tuple[float, float, float | None, bool]:
        ego = planner_input.ego
        rel = np.asarray(nobs.pos, dtype=np.float32) - np.asarray(ego.pos, dtype=np.float32)
        apex = self._reciprocal_apex(planner_input, nobs)
        rel_vel = np.asarray(candidate, dtype=np.float32) - apex
        if planner_input.planar:
            rel[1] = 0.0
            rel_vel[1] = 0.0

        dist = _norm(rel)
        age_inflation = self.stale_inflation_gain * max(0.0, float(nobs.msg_age_sec))
        radius = float(ego.radius) + float(nobs.radius) + self.safety_margin_m + age_inflation
        current_clearance = dist - radius
        if current_clearance <= 0.0:
            return self.collision_weight * (1.0 + current_clearance * current_clearance), current_clearance, 0.0, True

        rel_speed_sq = float(np.dot(rel_vel, rel_vel))
        if rel_speed_sq <= 1e-12:
            return 0.0, current_clearance, None, False

        closing_metric = float(np.dot(rel, rel_vel))
        if closing_metric <= 0.0:
            return 0.0, current_clearance, None, False

        t_cpa = max(0.0, min(self.time_horizon_s, closing_metric / rel_speed_sq))
        cpa = rel - rel_vel * t_cpa
        cpa_clearance = _norm(cpa) - radius
        ttc = current_clearance / max(1e-6, closing_metric / max(dist, 1e-6))
        horizon_hit = ttc <= self.time_horizon_s
        conflict = cpa_clearance < self.near_clearance_m and horizon_hit
        if not conflict:
            return 0.0, float(min(current_clearance, cpa_clearance)), float(ttc), False

        responsibility = self._responsibility(planner_input, nobs)
        time_weight = 1.0 + max(0.0, (self.time_horizon_s - ttc) / max(1e-6, self.time_horizon_s))
        cone_depth = max(0.0, self.near_clearance_m - cpa_clearance)
        collision_depth = max(0.0, -cpa_clearance)
        penalty = responsibility * (
            self.cone_weight * time_weight * (cone_depth / max(1e-6, self.near_clearance_m)) ** 2
            + self.collision_weight * time_weight * collision_depth * collision_depth
        )
        return float(penalty), float(min(current_clearance, cpa_clearance)), float(ttc), True
