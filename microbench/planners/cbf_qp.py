from __future__ import annotations

import contextlib
from dataclasses import dataclass
import io
import warnings
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


@dataclass(frozen=True)
class _CbfConstraint:
    kind: str
    a: np.ndarray
    b: float
    h: float
    clearance_m: float
    uncertainty_inflation_m: float = 0.0


class CbfQpPlanner(ILocalPlanner):
    """Deterministic CBF-QP baseline with an optional solver path.

    The default mode is deterministic halfspace projection. Optional `auto` or
    `scipy` modes use SciPy SLSQP when available, then fall back to projection.
    The public method remains experimental until solver behavior and acceptance
    bands are calibrated.
    """

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.solver = str(cfg.get("solver", "projection")).lower()
        self.alpha = float(cfg.get("alpha", 2.0))
        self.safety_margin_m = float(cfg.get("safety_margin_m", 0.25))
        self.obstacle_margin_m = float(cfg.get("obstacle_margin_m", 0.2))
        self.max_neighbors = int(cfg.get("max_neighbors", 8))
        self.max_projection_iters = int(cfg.get("max_projection_iters", 8))
        self.max_solver_iters = int(cfg.get("max_solver_iters", 40))
        self.violation_tol = float(cfg.get("violation_tol", 1e-5))
        self.fallback_speed_scale = float(cfg.get("fallback_speed_scale", 0.5))
        self.stale_age_cap_s = float(cfg.get("stale_age_cap_s", 1.5))
        self.stale_inflation_gain = float(cfg.get("stale_inflation_gain", 0.7))
        self.track_uncertainty_speed_gain = float(cfg.get("track_uncertainty_speed_gain", 0.15))

    def reset(self, seed: int) -> None:
        _ = seed

    def compute_cmd(self, planner_input: PlannerInput) -> PlannerOutput:
        ego = planner_input.ego
        v_pref = np.asarray(planner_input.goal_dir, dtype=np.float32) * float(ego.v_max)
        if planner_input.planar:
            v_pref[1] = 0.0

        constraints = self._constraints(planner_input)
        v_cmd, iterations, solver_used, solver_status = self._solve(
            v_pref,
            constraints,
            float(ego.v_max),
            bool(planner_input.planar),
        )
        max_violation = self._max_violation(v_cmd, constraints)
        pre_fallback_max_violation = max_violation
        fallback = bool(max_violation > self.violation_tol)
        if fallback:
            v_cmd = self._fallback(planner_input, v_pref)
            max_violation = self._max_violation(v_cmd, constraints)

        neighbor_constraints = sum(1 for c in constraints if c.kind == "neighbor")
        obstacle_constraints = sum(1 for c in constraints if c.kind == "obstacle")
        active_constraints = sum(1 for c in constraints if self._constraint_violation(v_cmd, c) > self.violation_tol)
        min_h = min((float(c.h) for c in constraints), default=None)
        min_clearance = min((float(c.clearance_m) for c in constraints), default=None)
        max_uncertainty = max((float(c.uncertainty_inflation_m) for c in constraints), default=0.0)

        return PlannerOutput(
            v_cmd=v_cmd.astype(float),
            debug_info={
                "cbf_constraints": len(constraints),
                "cbf_neighbor_constraints": int(neighbor_constraints),
                "cbf_obstacle_constraints": int(obstacle_constraints),
                "cbf_active_constraints": int(active_constraints),
                "cbf_projection_iters": int(iterations),
                "cbf_max_violation": float(max_violation),
                "cbf_pre_fallback_max_violation": float(pre_fallback_max_violation),
                "cbf_min_h": None if min_h is None else float(min_h),
                "cbf_min_clearance_m": None if min_clearance is None else float(min_clearance),
                "cbf_uncertainty_inflation_max_m": float(max_uncertainty),
                "cbf_fallback": fallback,
                "cbf_solver": solver_used,
                "cbf_solver_requested": self.solver,
                "cbf_solver_status": solver_status,
            },
        )

    def _constraints(self, planner_input: PlannerInput) -> list[_CbfConstraint]:
        constraints: list[_CbfConstraint] = []
        ego = planner_input.ego
        p_i = np.asarray(ego.pos, dtype=np.float32)
        for nobs in planner_input.neighbors[: self.max_neighbors]:
            if not bool(getattr(nobs, "valid", True)):
                continue
            constraints.append(self._neighbor_constraint(planner_input, nobs))
        for obs in planner_input.obstacles:
            constraints.append(self._obstacle_constraint(planner_input, obs, p_i))
        if planner_input.planar:
            cleaned = []
            for constraint in constraints:
                aa = np.asarray(constraint.a, dtype=np.float32)
                aa[1] = 0.0
                if _norm(aa) > 1e-9:
                    cleaned.append(
                        _CbfConstraint(
                            kind=constraint.kind,
                            a=aa,
                            b=float(constraint.b),
                            h=float(constraint.h),
                            clearance_m=float(constraint.clearance_m),
                            uncertainty_inflation_m=float(constraint.uncertainty_inflation_m),
                        )
                    )
            return cleaned
        return [
            _CbfConstraint(
                kind=c.kind,
                a=np.asarray(c.a, dtype=np.float32),
                b=float(c.b),
                h=float(c.h),
                clearance_m=float(c.clearance_m),
                uncertainty_inflation_m=float(c.uncertainty_inflation_m),
            )
            for c in constraints
            if _norm(np.asarray(c.a)) > 1e-9
        ]

    def _neighbor_constraint(self, planner_input: PlannerInput, nobs: NeighborObs) -> _CbfConstraint:
        ego = planner_input.ego
        p_i = np.asarray(ego.pos, dtype=np.float32)
        p_j = np.asarray(nobs.pos, dtype=np.float32)
        v_j = np.asarray(nobs.vel, dtype=np.float32)
        rel = p_i - p_j
        if _norm(rel) < 1e-6:
            rel = _normalize(p_i - np.asarray(ego.goal, dtype=np.float32))
            if _norm(rel) < 1e-6:
                rel = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        age_s = self._track_age_s(nobs)
        uncertainty_inflation = self.stale_inflation_gain * age_s + self.track_uncertainty_speed_gain * age_s * _norm(v_j)
        radius = float(ego.radius) + float(nobs.radius) + self.safety_margin_m + uncertainty_inflation
        dist = _norm(rel)
        h = float(np.dot(rel, rel) - radius * radius)
        a = 2.0 * rel
        b = float(2.0 * np.dot(rel, v_j) - self.alpha * h)
        return _CbfConstraint(
            kind="neighbor",
            a=a.astype(np.float32),
            b=b,
            h=h,
            clearance_m=float(dist - radius),
            uncertainty_inflation_m=float(uncertainty_inflation),
        )

    def _obstacle_constraint(self, planner_input: PlannerInput, obs: AABBObs, p_i: np.ndarray) -> _CbfConstraint:
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
        dist = _norm(rel)
        h = float(np.dot(rel, rel) - radius * radius)
        a = 2.0 * rel
        b = float(-self.alpha * h)
        return _CbfConstraint(
            kind="obstacle",
            a=a.astype(np.float32),
            b=b,
            h=h,
            clearance_m=float(dist - radius),
        )

    def _track_age_s(self, nobs: NeighborObs) -> float:
        msg_age = max(0.0, float(getattr(nobs, "msg_age_sec", 0.0) or 0.0))
        track_age = max(0.0, float(getattr(nobs, "track_age_sec", 0.0) or 0.0))
        return min(max(msg_age, track_age), max(0.0, self.stale_age_cap_s))

    def _project(
        self,
        v_pref: np.ndarray,
        constraints: list[_CbfConstraint],
        v_max: float,
        planar: bool,
    ) -> tuple[np.ndarray, int, bool]:
        v = np.asarray(v_pref, dtype=np.float32).copy()
        iterations = 0
        for iteration in range(max(0, self.max_projection_iters)):
            changed = False
            for constraint in constraints:
                aa = np.asarray(constraint.a, dtype=np.float32)
                denom = float(np.dot(aa, aa))
                if denom < 1e-12:
                    continue
                violation = float(constraint.b - np.dot(aa, v))
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
        v = _clamp_speed(v, v_max)
        if planar:
            v[1] = 0.0
        return v, iterations, self._max_violation(v, constraints) <= self.violation_tol

    def _solve(
        self,
        v_pref: np.ndarray,
        constraints: list[_CbfConstraint],
        v_max: float,
        planar: bool,
    ) -> tuple[np.ndarray, int, str, str]:
        if not constraints:
            return _clamp_speed(v_pref, v_max), 0, "none", "no_constraints"

        if self.solver in {"auto", "scipy", "scipy_slsqp"}:
            solved = self._solve_scipy(v_pref, constraints, v_max, planar)
            if solved is not None:
                return solved[0], 0, "scipy_slsqp", solved[1]
            if self.solver in {"scipy", "scipy_slsqp"}:
                v, iters, converged = self._project(v_pref, constraints, v_max, planar)
                status = "scipy_unavailable_projection_converged" if converged else "scipy_unavailable_projection_residual_violation"
                return v, iters, "deterministic_projection", status

        v, iters, converged = self._project(v_pref, constraints, v_max, planar)
        if self.solver == "auto":
            status = "projection_fallback_converged" if converged else "projection_fallback_residual_violation"
        else:
            status = "projection_converged" if converged else "projection_residual_violation"
        return v, iters, "deterministic_projection", status

    def _solve_scipy(
        self,
        v_pref: np.ndarray,
        constraints: list[_CbfConstraint],
        v_max: float,
        planar: bool,
    ) -> tuple[np.ndarray, str] | None:
        try:
            with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
                warnings.simplefilter("ignore")
                from scipy.optimize import minimize
        except Exception:
            return None

        x0 = _clamp_speed(np.asarray(v_pref, dtype=float), v_max).astype(float)
        if planar:
            x0[1] = 0.0

        cons = []
        for constraint in constraints:
            aa = np.asarray(constraint.a, dtype=float)
            bb = float(constraint.b)
            cons.append(
                {
                    "type": "ineq",
                    "fun": lambda x, aa=aa, bb=bb: float(np.dot(aa, x) - bb),
                    "jac": lambda x, aa=aa, bb=bb: aa,
                }
            )
        cons.append(
            {
                "type": "ineq",
                "fun": lambda x: float(v_max * v_max - np.dot(x, x)),
                "jac": lambda x: -2.0 * np.asarray(x, dtype=float),
            }
        )
        if planar:
            cons.append(
                {
                    "type": "eq",
                    "fun": lambda x: float(x[1]),
                    "jac": lambda x: np.asarray([0.0, 1.0, 0.0], dtype=float),
                }
            )

        def objective(x):
            d = np.asarray(x, dtype=float) - np.asarray(v_pref, dtype=float)
            return 0.5 * float(np.dot(d, d))

        def jac(x):
            return np.asarray(x, dtype=float) - np.asarray(v_pref, dtype=float)

        try:
            result = minimize(
                objective,
                x0,
                jac=jac,
                constraints=cons,
                method="SLSQP",
                options={
                    "maxiter": max(1, self.max_solver_iters),
                    "ftol": max(1e-12, self.violation_tol * 0.1),
                    "disp": False,
                },
            )
        except Exception:
            return None
        if not bool(getattr(result, "success", False)):
            return None
        v = _clamp_speed(np.asarray(result.x, dtype=np.float32), v_max)
        if planar:
            v[1] = 0.0
        return v, str(getattr(result, "message", "success"))

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

    def _constraint_violation(self, v_cmd: np.ndarray, constraint: _CbfConstraint) -> float:
        return max(0.0, float(constraint.b - np.dot(constraint.a, v_cmd)))

    def _max_violation(self, v_cmd: np.ndarray, constraints: list[_CbfConstraint]) -> float:
        if not constraints:
            return 0.0
        return max(self._constraint_violation(v_cmd, constraint) for constraint in constraints)
