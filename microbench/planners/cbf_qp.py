from __future__ import annotations

import numpy as np

from microbench.planners.base import ILocalPlanner
from microbench.types import AABBObs, NeighborObs, PlannerInput, PlannerOutput


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _normalize(v: np.ndarray) -> np.ndarray:
    n = _norm(v)
    if n < 1e-9:
        return np.zeros_like(v, dtype=np.float32)
    return (v / n).astype(np.float32)


def _clamp_speed(v: np.ndarray, v_max: float) -> np.ndarray:
    n = _norm(v)
    if n <= v_max or n < 1e-9:
        return v.astype(np.float32)
    return (v / n * v_max).astype(np.float32)


def _closest_point_on_aabb(point: np.ndarray, obs: AABBObs) -> np.ndarray:
    center = np.asarray(obs.center, dtype=np.float32)
    half = np.asarray(obs.half, dtype=np.float32)
    return np.minimum(np.maximum(point, center - half), center + half)


class CbfQpPlanner(ILocalPlanner):
    """Deterministic CBF-QP skeleton with a projection fallback.

    This is intentionally solver-free for now. It builds linearized control
    barrier constraints and projects a preferred velocity onto the resulting
    halfspaces. The public method is experimental until a solver-backed QP
    variant and calibrated acceptance bands are added.
    """

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.alpha = float(cfg.get("alpha", 2.0))
        self.safety_margin_m = float(cfg.get("safety_margin_m", 0.25))
        self.obstacle_margin_m = float(cfg.get("obstacle_margin_m", 0.2))
        self.max_neighbors = int(cfg.get("max_neighbors", 8))
        self.max_projection_iters = int(cfg.get("max_projection_iters", 8))
        self.violation_tol = float(cfg.get("violation_tol", 1e-5))
        self.fallback_speed_scale = float(cfg.get("fallback_speed_scale", 0.5))

    def reset(self, seed: int) -> None:
        _ = seed

    def compute_cmd(self, planner_input: PlannerInput) -> PlannerOutput:
        ego = planner_input.ego
        v_pref = np.asarray(planner_input.goal_dir, dtype=np.float32) * float(ego.v_max)
        if planner_input.planar:
            v_pref[1] = 0.0

        constraints = self._constraints(planner_input)
        v_cmd, iterations = self._project(v_pref, constraints, float(ego.v_max), bool(planner_input.planar))
        max_violation = self._max_violation(v_cmd, constraints)
        fallback = bool(max_violation > self.violation_tol)
        if fallback:
            v_cmd = self._fallback(planner_input, v_pref)
            max_violation = self._max_violation(v_cmd, constraints)

        return PlannerOutput(
            v_cmd=v_cmd.astype(float),
            debug_info={
                "cbf_constraints": len(constraints),
                "cbf_projection_iters": int(iterations),
                "cbf_max_violation": float(max_violation),
                "cbf_fallback": fallback,
                "cbf_solver": "projection_skeleton",
            },
        )

    def _constraints(self, planner_input: PlannerInput) -> list[tuple[np.ndarray, float]]:
        constraints: list[tuple[np.ndarray, float]] = []
        ego = planner_input.ego
        p_i = np.asarray(ego.pos, dtype=np.float32)
        for nobs in planner_input.neighbors[: self.max_neighbors]:
            constraints.append(self._neighbor_constraint(planner_input, nobs))
        for obs in planner_input.obstacles:
            constraints.append(self._obstacle_constraint(planner_input, obs, p_i))
        if planner_input.planar:
            cleaned = []
            for a, b in constraints:
                aa = np.asarray(a, dtype=np.float32)
                aa[1] = 0.0
                if _norm(aa) > 1e-9:
                    cleaned.append((aa, float(b)))
            return cleaned
        return [(np.asarray(a, dtype=np.float32), float(b)) for a, b in constraints if _norm(np.asarray(a)) > 1e-9]

    def _neighbor_constraint(self, planner_input: PlannerInput, nobs: NeighborObs) -> tuple[np.ndarray, float]:
        ego = planner_input.ego
        p_i = np.asarray(ego.pos, dtype=np.float32)
        p_j = np.asarray(nobs.pos, dtype=np.float32)
        v_j = np.asarray(nobs.vel, dtype=np.float32)
        rel = p_i - p_j
        if _norm(rel) < 1e-6:
            rel = _normalize(p_i - np.asarray(ego.goal, dtype=np.float32))
            if _norm(rel) < 1e-6:
                rel = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        radius = float(ego.radius) + float(nobs.radius) + self.safety_margin_m
        h = float(np.dot(rel, rel) - radius * radius)
        a = 2.0 * rel
        b = float(2.0 * np.dot(rel, v_j) - self.alpha * h)
        return a.astype(np.float32), b

    def _obstacle_constraint(self, planner_input: PlannerInput, obs: AABBObs, p_i: np.ndarray) -> tuple[np.ndarray, float]:
        ego = planner_input.ego
        closest = _closest_point_on_aabb(p_i, obs)
        rel = p_i - closest
        if _norm(rel) < 1e-6:
            center = np.asarray(obs.center, dtype=np.float32)
            half = np.asarray(obs.half, dtype=np.float32)
            local = p_i - center
            slack = half - np.abs(local)
            axis = int(np.argmin(slack))
            rel = np.zeros(3, dtype=np.float32)
            rel[axis] = 1.0 if local[axis] >= 0.0 else -1.0
        radius = float(ego.radius) + self.obstacle_margin_m
        h = float(np.dot(rel, rel) - radius * radius)
        a = 2.0 * rel
        b = float(-self.alpha * h)
        return a.astype(np.float32), b

    def _project(
        self,
        v_pref: np.ndarray,
        constraints: list[tuple[np.ndarray, float]],
        v_max: float,
        planar: bool,
    ) -> tuple[np.ndarray, int]:
        v = np.asarray(v_pref, dtype=np.float32).copy()
        iterations = 0
        for iteration in range(max(0, self.max_projection_iters)):
            changed = False
            for a, b in constraints:
                aa = np.asarray(a, dtype=np.float32)
                denom = float(np.dot(aa, aa))
                if denom < 1e-12:
                    continue
                violation = float(b - np.dot(aa, v))
                if violation <= self.violation_tol:
                    continue
                v = v + (violation / denom) * aa
                if planar:
                    v[1] = 0.0
                v = _clamp_speed(v, v_max)
                changed = True
            iterations = iteration + 1
            if not changed:
                break
        return _clamp_speed(v, v_max), iterations

    def _fallback(self, planner_input: PlannerInput, v_pref: np.ndarray) -> np.ndarray:
        ego = planner_input.ego
        away = np.zeros(3, dtype=np.float32)
        p_i = np.asarray(ego.pos, dtype=np.float32)
        for nobs in planner_input.neighbors[: self.max_neighbors]:
            rel = p_i - np.asarray(nobs.pos, dtype=np.float32)
            dist = max(1e-6, _norm(rel))
            away += rel / (dist * dist)
        for obs in planner_input.obstacles:
            rel = p_i - _closest_point_on_aabb(p_i, obs)
            dist = max(1e-6, _norm(rel))
            away += rel / (dist * dist)
        if planner_input.planar:
            away[1] = 0.0
        if _norm(away) < 1e-9:
            return _clamp_speed(v_pref * self.fallback_speed_scale, float(ego.v_max))
        return _clamp_speed(_normalize(away) * float(ego.v_max) * self.fallback_speed_scale, float(ego.v_max))

    def _max_violation(self, v_cmd: np.ndarray, constraints: list[tuple[np.ndarray, float]]) -> float:
        if not constraints:
            return 0.0
        return max(0.0, max(float(b - np.dot(a, v_cmd)) for a, b in constraints))
