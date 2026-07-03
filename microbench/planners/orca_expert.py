from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from microbench.planners.base import ILocalPlanner
from microbench.types import AABBObs, PlannerInput


@dataclass
class OrcaLine:
    point: np.ndarray
    direction: np.ndarray


@dataclass
class OrcaPlane:
    point: np.ndarray
    normal: np.ndarray


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _normalize(v: np.ndarray) -> np.ndarray:
    n = _norm(v)
    if n < 1e-9:
        return np.zeros_like(v)
    return v / n


def _dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def _det(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _perp(v: np.ndarray) -> np.ndarray:
    return np.asarray([v[1], -v[0]], dtype=np.float32)


def _clamp_speed(v: np.ndarray, v_max: float) -> np.ndarray:
    n = _norm(v)
    if n <= v_max or n < 1e-9:
        return v.astype(np.float32)
    return (v / n * v_max).astype(np.float32)


def _closest_point_on_aabb(point: np.ndarray, center: np.ndarray, half: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(point, center - half), center + half)


def _outside_normal(point: np.ndarray, center: np.ndarray, half: np.ndarray) -> np.ndarray:
    local = point - center
    q = np.abs(local) - half
    if np.any(q > 0.0):
        normal = point - _closest_point_on_aabb(point, center, half)
        if _norm(normal) > 1e-9:
            return _normalize(normal)
    slack = half - np.abs(local)
    axis = int(np.argmin(slack))
    sign = 1.0 if local[axis] >= 0.0 else -1.0
    out = np.zeros(3, dtype=np.float32)
    out[axis] = sign
    return out


class OrcaExpertPlanner(ILocalPlanner):
    def __init__(self, cfg: dict | None = None, age_cap_s: float = 0.75):
        cfg = cfg or {}
        self.time_horizon_s = float(cfg.get("time_horizon_s", 3.0))
        self.obstacle_time_horizon_s = float(cfg.get("obstacle_time_horizon_s", self.time_horizon_s))
        self.safety_margin_m = float(cfg.get("safety_margin_m", 0.2))
        self.obstacle_margin_m = float(cfg.get("obstacle_margin_m", 0.15))
        self.stale_inflation_gain = float(cfg.get("stale_inflation_gain", 0.8))
        self.stale_age_cap_s = float(cfg.get("stale_age_cap_s", age_cap_s))
        self.max_neighbors = int(cfg.get("max_neighbors", 8))
        self.goal_slowdown_radius_m = float(cfg.get("goal_slowdown_radius_m", 6.0))
        self.closing_speed_buffer_s = float(cfg.get("closing_speed_buffer_s", 0.1))
        self.responsibility_age_gain = float(cfg.get("responsibility_age_gain", 0.35))
        self.sidestep_bias_gain = float(cfg.get("sidestep_bias_gain", 0.18))
        self.candidate_samples_2d = int(cfg.get("candidate_samples_2d", 32))
        self.candidate_samples_3d = int(cfg.get("candidate_samples_3d", 48))
        self.violation_weight = float(cfg.get("violation_weight", 500.0))
        self.risk_weight = float(cfg.get("risk_weight", 40.0))
        self.obstacle_weight = float(cfg.get("obstacle_weight", 80.0))
        self.progress_weight = float(cfg.get("progress_weight", 0.05))

    def reset(self, seed: int) -> None:
        _ = seed

    def compute_cmd(self, planner_input: PlannerInput) -> np.ndarray:
        ego = planner_input.ego
        goal_dist = float(np.linalg.norm(ego.goal - ego.pos))
        v_pref_mag = self._preferred_speed(float(ego.v_max), float(ego.a_max), goal_dist)

        if planner_input.planar:
            return self._compute_cmd_2d(planner_input, v_pref_mag)
        return self._compute_cmd_3d(planner_input, v_pref_mag)

    def _compute_cmd_2d(self, planner_input: PlannerInput, v_pref_mag: float) -> np.ndarray:
        ego = planner_input.ego
        p_i = np.asarray([ego.pos[0], ego.pos[2]], dtype=np.float32)
        v_i = np.asarray([ego.vel[0], ego.vel[2]], dtype=np.float32)
        g = np.asarray([planner_input.goal_dir[0], planner_input.goal_dir[2]], dtype=np.float32)
        v_pref = _normalize(g) * v_pref_mag if _norm(g) > 1e-9 else np.zeros(2, dtype=np.float32)
        v_pref = _clamp_speed(v_pref + self._sidestep_bias_2d(planner_input, g, v_pref_mag), float(ego.v_max))

        lines = self._compute_neighbor_lines_2d(planner_input, p_i, v_i)
        lines.extend(self._compute_obstacle_lines_2d(planner_input, p_i, v_i))
        v_lp = self._solve_orca_lp(lines, v_pref, float(ego.v_max))
        v_best = self._refine_candidates_2d(planner_input, lines, v_pref, v_lp, float(ego.v_max))

        out = np.zeros(3, dtype=np.float32)
        out[0] = float(v_best[0])
        out[2] = float(v_best[1])
        return out

    def _compute_cmd_3d(self, planner_input: PlannerInput, v_pref_mag: float) -> np.ndarray:
        ego = planner_input.ego
        p_i = np.asarray(ego.pos, dtype=np.float32)
        v_i = np.asarray(ego.vel, dtype=np.float32)
        g3 = np.asarray(planner_input.goal_dir, dtype=np.float32)
        v_pref = _normalize(g3) * v_pref_mag if _norm(g3) > 1e-9 else np.zeros(3, dtype=np.float32)
        v_pref = _clamp_speed(v_pref + self._sidestep_bias_3d(planner_input, g3, v_pref_mag), float(ego.v_max))

        planes = self._compute_neighbor_planes_3d(planner_input, p_i, v_i)
        planes.extend(self._compute_obstacle_planes_3d(planner_input, p_i, v_i))
        v_proj = self._solve_planes_projection(planes, v_pref, float(ego.v_max))
        return self._refine_candidates_3d(planner_input, planes, v_pref, v_proj, float(ego.v_max))

    def _preferred_speed(self, v_max: float, a_max: float, goal_dist: float) -> float:
        stop_speed = math.sqrt(max(0.0, 2.0 * max(1e-6, a_max) * max(0.0, goal_dist)))
        soft_speed = v_max * min(1.0, goal_dist / max(1e-6, self.goal_slowdown_radius_m))
        return float(min(v_max, max(soft_speed, min(v_max, stop_speed))))

    def _combined_radius(self, planner_input: PlannerInput, nobs) -> float:
        age = min(max(0.0, float(nobs.msg_age_sec)), self.stale_age_cap_s)
        rel_pos = np.asarray(nobs.pos, dtype=np.float32) - np.asarray(planner_input.ego.pos, dtype=np.float32)
        rel_vel = np.asarray(planner_input.ego.vel, dtype=np.float32) - np.asarray(nobs.vel, dtype=np.float32)
        rel_hat = _normalize(rel_pos)
        closing_speed = max(0.0, _dot(rel_vel, rel_hat))
        return (
            float(planner_input.ego.radius)
            + float(nobs.radius)
            + self.safety_margin_m
            + self.stale_inflation_gain * age
            + self.closing_speed_buffer_s * closing_speed
        )

    def _responsibility(self, planner_input: PlannerInput, nobs) -> float:
        age = min(max(0.0, float(nobs.msg_age_sec)), self.stale_age_cap_s)
        frac = age / max(1e-6, self.stale_age_cap_s)
        return float(min(0.9, max(0.5, 0.5 + self.responsibility_age_gain * frac)))

    def _min_time_gap(self, planner_input: PlannerInput) -> float:
        min_t_gap = float("inf")
        for nobs in planner_input.neighbors[: self.max_neighbors]:
            rel_pos = np.asarray(nobs.pos, dtype=np.float32) - np.asarray(planner_input.ego.pos, dtype=np.float32)
            rel_vel = np.asarray(planner_input.ego.vel, dtype=np.float32) - np.asarray(nobs.vel, dtype=np.float32)
            clearance = _norm(rel_pos) - self._combined_radius(planner_input, nobs)
            if clearance <= 0.0:
                return 0.0
            rel_hat = _normalize(rel_pos)
            closing_speed = _dot(rel_vel, rel_hat)
            if closing_speed <= 1e-6:
                continue
            min_t_gap = min(min_t_gap, clearance / closing_speed)
        return min_t_gap

    def _risk_weight(self, planner_input: PlannerInput) -> float:
        tau = max(1e-6, self.time_horizon_s)
        t_gap = self._min_time_gap(planner_input)
        if not np.isfinite(t_gap):
            return 0.0
        return float(min(1.0, max(0.0, 1.0 - t_gap / tau)))

    def _sidestep_sign(self, planner_input: PlannerInput) -> float:
        return -1.0 if int(planner_input.ego.idx) % 2 == 0 else 1.0

    def _sidestep_bias_2d(self, planner_input: PlannerInput, goal_2d: np.ndarray, v_pref_mag: float) -> np.ndarray:
        weight = self._risk_weight(planner_input)
        if weight <= 1e-6:
            return np.zeros(2, dtype=np.float32)
        base = goal_2d if _norm(goal_2d) > 1e-9 else np.asarray([1.0, 0.0], dtype=np.float32)
        return _normalize(_perp(base)) * self._sidestep_sign(planner_input) * (self.sidestep_bias_gain * v_pref_mag * weight)

    def _sidestep_bias_3d(self, planner_input: PlannerInput, goal_3d: np.ndarray, v_pref_mag: float) -> np.ndarray:
        weight = self._risk_weight(planner_input)
        if weight <= 1e-6:
            return np.zeros(3, dtype=np.float32)
        horiz = np.asarray([goal_3d[0], goal_3d[2]], dtype=np.float32)
        if _norm(horiz) <= 1e-9:
            horiz = np.asarray([1.0, 0.0], dtype=np.float32)
        lateral = _normalize(_perp(horiz)) * self._sidestep_sign(planner_input) * (self.sidestep_bias_gain * v_pref_mag * weight)
        out = np.zeros(3, dtype=np.float32)
        out[0] = lateral[0]
        out[2] = lateral[1]
        return out

    def _compute_neighbor_lines_2d(self, planner_input: PlannerInput, p_i: np.ndarray, v_i: np.ndarray) -> list[OrcaLine]:
        dt = max(1e-6, float(planner_input.dt))
        inv_tau = 1.0 / max(1e-6, self.time_horizon_s)
        lines: list[OrcaLine] = []
        for nobs in planner_input.neighbors[: self.max_neighbors]:
            p_j = np.asarray([nobs.pos[0], nobs.pos[2]], dtype=np.float32)
            v_j = np.asarray([nobs.vel[0], nobs.vel[2]], dtype=np.float32)
            combined_radius = self._combined_radius(planner_input, nobs)
            responsibility = self._responsibility(planner_input, nobs)

            rel_pos = p_j - p_i
            rel_vel = v_i - v_j
            dist_sq = _dot(rel_pos, rel_pos)
            combined_radius_sq = combined_radius * combined_radius
            line_dir = np.zeros(2, dtype=np.float32)
            u = np.zeros(2, dtype=np.float32)

            if dist_sq > combined_radius_sq:
                w = rel_vel - inv_tau * rel_pos
                w_len_sq = _dot(w, w)
                dot1 = _dot(w, rel_pos)
                if dot1 < 0.0 and dot1 * dot1 > combined_radius_sq * w_len_sq:
                    w_len = max(1e-9, math.sqrt(w_len_sq))
                    unit_w = w / w_len
                    line_dir = _perp(unit_w)
                    u = (combined_radius * inv_tau - w_len) * unit_w
                else:
                    leg = math.sqrt(max(0.0, dist_sq - combined_radius_sq))
                    if _det(rel_pos, w) > 0.0:
                        line_dir = np.asarray(
                            [rel_pos[0] * leg - rel_pos[1] * combined_radius, rel_pos[0] * combined_radius + rel_pos[1] * leg],
                            dtype=np.float32,
                        ) / max(1e-9, dist_sq)
                    else:
                        line_dir = -np.asarray(
                            [rel_pos[0] * leg + rel_pos[1] * combined_radius, -rel_pos[0] * combined_radius + rel_pos[1] * leg],
                            dtype=np.float32,
                        ) / max(1e-9, dist_sq)
                    line_dir = _normalize(line_dir)
                    u = _dot(rel_vel, line_dir) * line_dir - rel_vel
            else:
                inv_dt = 1.0 / dt
                w = rel_vel - inv_dt * rel_pos
                w_len = _norm(w)
                unit_w = w / w_len if w_len > 1e-9 else _normalize(_perp(rel_pos + np.asarray([1e-3, 0.0], dtype=np.float32)))
                line_dir = _perp(unit_w)
                u = (combined_radius * inv_dt - w_len) * unit_w

            lines.append(OrcaLine(point=v_i + responsibility * u, direction=_normalize(line_dir)))
        return lines

    def _compute_neighbor_planes_3d(self, planner_input: PlannerInput, p_i: np.ndarray, v_i: np.ndarray) -> list[OrcaPlane]:
        dt = max(1e-6, float(planner_input.dt))
        tau = max(1e-6, self.time_horizon_s)
        planes: list[OrcaPlane] = []
        for nobs in planner_input.neighbors[: self.max_neighbors]:
            p_j = np.asarray(nobs.pos, dtype=np.float32)
            v_j = np.asarray(nobs.vel, dtype=np.float32)
            combined_radius = self._combined_radius(planner_input, nobs)
            responsibility = self._responsibility(planner_input, nobs)

            rel_pos = p_j - p_i
            rel_vel = v_i - v_j
            dist = _norm(rel_pos)
            rel_speed_sq = max(1e-9, _dot(rel_vel, rel_vel))
            t_cpa = max(0.0, min(tau, _dot(rel_pos, rel_vel) / rel_speed_sq))
            cpa = rel_pos - t_cpa * rel_vel
            cpa_dist = _norm(cpa)
            approaching = _dot(rel_pos, rel_vel) > 0.0

            if dist > combined_radius and (not approaching or cpa_dist > combined_radius or t_cpa <= 0.0):
                continue

            if dist <= combined_radius:
                n = _normalize(rel_pos if dist > 1e-6 else np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
                horizon = dt
                projected_sep = dist
            else:
                n = _normalize(cpa if cpa_dist > 1e-6 else rel_pos)
                horizon = max(dt, t_cpa)
                projected_sep = _dot(rel_pos, n)

            current_speed_n = _dot(rel_vel, n)
            target_speed_n = (projected_sep - combined_radius) / max(horizon, 1e-6)
            delta_speed_n = target_speed_n - current_speed_n
            if delta_speed_n >= 0.0:
                continue
            u = delta_speed_n * n
            planes.append(OrcaPlane(point=v_i + responsibility * u, normal=_normalize(u)))
        return planes

    def _compute_obstacle_lines_2d(self, planner_input: PlannerInput, p_i: np.ndarray, v_i: np.ndarray) -> list[OrcaLine]:
        lines: list[OrcaLine] = []
        tau = max(1e-6, self.obstacle_time_horizon_s)
        for obs in planner_input.obstacles:
            center = np.asarray([obs.center[0], obs.center[2]], dtype=np.float32)
            half = np.asarray([obs.half[0], obs.half[2]], dtype=np.float32) + float(planner_input.ego.radius + self.safety_margin_m + self.obstacle_margin_m)
            closest = np.minimum(np.maximum(p_i, center - half), center + half)
            rel = closest - p_i
            dist = _norm(rel)
            influence = max(half[0], half[1]) + self.time_horizon_s * float(planner_input.ego.v_max) + 1.0
            if dist > influence:
                continue
            if dist > 1e-6:
                normal = _normalize(-rel)
                speed_n = _dot(v_i, normal)
                target_speed_n = max(0.0, (self.obstacle_margin_m - dist) / tau)
            else:
                local = p_i - center
                slack = half - np.abs(local)
                axis = int(np.argmin(slack))
                sign = 1.0 if local[axis] >= 0.0 else -1.0
                normal = np.zeros(2, dtype=np.float32)
                normal[axis] = sign
                speed_n = _dot(v_i, normal)
                target_speed_n = float(np.max(half)) / max(1e-6, planner_input.dt)
            if speed_n >= target_speed_n:
                continue
            u = (target_speed_n - speed_n) * normal
            lines.append(OrcaLine(point=v_i + u, direction=_normalize(_perp(normal))))
        return lines

    def _compute_obstacle_planes_3d(self, planner_input: PlannerInput, p_i: np.ndarray, v_i: np.ndarray) -> list[OrcaPlane]:
        planes: list[OrcaPlane] = []
        tau = max(1e-6, self.obstacle_time_horizon_s)
        for obs in planner_input.obstacles:
            center = np.asarray(obs.center, dtype=np.float32)
            half = np.asarray(obs.half, dtype=np.float32) + float(planner_input.ego.radius + self.safety_margin_m + self.obstacle_margin_m)
            closest = _closest_point_on_aabb(p_i, center, half)
            rel = closest - p_i
            dist = _norm(rel)
            influence = float(np.max(half)) + self.time_horizon_s * float(planner_input.ego.v_max) + 1.0
            if dist > influence:
                continue
            if dist > 1e-6:
                normal = _normalize(-rel)
                speed_n = _dot(v_i, normal)
                target_speed_n = max(0.0, (self.obstacle_margin_m - dist) / tau)
            else:
                normal = _outside_normal(p_i, center, half)
                speed_n = _dot(v_i, normal)
                target_speed_n = float(np.max(half)) / max(1e-6, planner_input.dt)
            if speed_n >= target_speed_n:
                continue
            u = (target_speed_n - speed_n) * normal
            planes.append(OrcaPlane(point=v_i + u, normal=normal))
        return planes

    def _linear_program1(self, lines: list[OrcaLine], line_no: int, radius: float, opt_velocity: np.ndarray, direction_opt: bool, result: np.ndarray) -> tuple[bool, np.ndarray]:
        line = lines[line_no]
        dotp = _dot(line.point, line.direction)
        disc = dotp * dotp + radius * radius - _dot(line.point, line.point)
        if disc < 0.0:
            return False, result
        sqrt_disc = math.sqrt(max(0.0, disc))
        t_left = -dotp - sqrt_disc
        t_right = -dotp + sqrt_disc
        for i in range(line_no):
            denom = _det(line.direction, lines[i].direction)
            numer = _det(lines[i].direction, line.point - lines[i].point)
            if abs(denom) <= 1e-9:
                if numer < 0.0:
                    return False, result
                continue
            t = numer / denom
            if denom >= 0.0:
                t_right = min(t_right, t)
            else:
                t_left = max(t_left, t)
            if t_left > t_right:
                return False, result
        if direction_opt:
            result = line.point + (t_right if _dot(opt_velocity, line.direction) > 0.0 else t_left) * line.direction
        else:
            t = _dot(line.direction, opt_velocity - line.point)
            if t < t_left:
                result = line.point + t_left * line.direction
            elif t > t_right:
                result = line.point + t_right * line.direction
            else:
                result = line.point + t * line.direction
        return True, result

    def _linear_program2(self, lines: list[OrcaLine], radius: float, opt_velocity: np.ndarray, direction_opt: bool) -> tuple[int, np.ndarray]:
        result = opt_velocity * radius if direction_opt else (_normalize(opt_velocity) * radius if _dot(opt_velocity, opt_velocity) > radius * radius else opt_velocity.copy())
        for i in range(len(lines)):
            if _det(lines[i].direction, lines[i].point - result) > 0.0:
                temp = result.copy()
                ok, result = self._linear_program1(lines, i, radius, opt_velocity, direction_opt, result)
                if not ok:
                    return i, temp
        return len(lines), result

    def _linear_program3(self, lines: list[OrcaLine], begin_line: int, radius: float, result: np.ndarray) -> np.ndarray:
        distance = 0.0
        for i in range(begin_line, len(lines)):
            if _det(lines[i].direction, lines[i].point - result) > distance:
                proj_lines: list[OrcaLine] = []
                for j in range(i):
                    denom = _det(lines[i].direction, lines[j].direction)
                    if abs(denom) <= 1e-9:
                        if _dot(lines[i].direction, lines[j].direction) > 0.0:
                            continue
                        point = 0.5 * (lines[i].point + lines[j].point)
                    else:
                        point = lines[i].point + (_det(lines[j].direction, lines[i].point - lines[j].point) / denom) * lines[i].direction
                    proj_lines.append(OrcaLine(point=point.astype(np.float32), direction=_normalize(lines[j].direction - lines[i].direction)))
                temp = result.copy()
                idx, result = self._linear_program2(proj_lines, radius, _perp(lines[i].direction), True)
                if idx < len(proj_lines):
                    result = temp
                distance = _det(lines[i].direction, lines[i].point - result)
        return result

    def _solve_orca_lp(self, lines: list[OrcaLine], v_pref: np.ndarray, v_max: float) -> np.ndarray:
        if not lines:
            return _clamp_speed(v_pref, v_max)
        line_fail, result = self._linear_program2(lines, v_max, v_pref.astype(np.float32), False)
        if line_fail < len(lines):
            result = self._linear_program3(lines, line_fail, v_max, result)
        for _ in range(8):
            worst = 0.0
            for line in lines:
                violation = _det(line.direction, line.point - result)
                if violation > 0.0:
                    result = result - violation * _perp(line.direction)
                    worst = max(worst, float(violation))
            result = _clamp_speed(result, v_max)
            if worst < 1e-4:
                break
        return _clamp_speed(result, v_max)

    def _solve_planes_projection(self, planes: list[OrcaPlane], v_pref: np.ndarray, v_max: float) -> np.ndarray:
        result = _clamp_speed(v_pref, v_max)
        for _ in range(16):
            worst = 0.0
            for plane in planes:
                violation = _dot(plane.normal, plane.point - result)
                if violation > 0.0:
                    result = result + violation * plane.normal
                    result = _clamp_speed(result, v_max)
                    worst = max(worst, float(violation))
            if worst < 1e-4:
                break
        return _clamp_speed(result, v_max)

    def _candidate_cost_2d(self, planner_input: PlannerInput, lines: list[OrcaLine], v: np.ndarray, v_pref: np.ndarray) -> float:
        violation = sum(max(0.0, _det(line.direction, line.point - v)) ** 2 for line in lines)
        risk = 0.0
        p_i = np.asarray([planner_input.ego.pos[0], planner_input.ego.pos[2]], dtype=np.float32)
        tau = max(1e-6, self.time_horizon_s)
        for nobs in planner_input.neighbors[: self.max_neighbors]:
            rel_pos = np.asarray([nobs.pos[0], nobs.pos[2]], dtype=np.float32) - p_i
            rel_vel = v - np.asarray([nobs.vel[0], nobs.vel[2]], dtype=np.float32)
            vv = max(1e-9, _dot(rel_vel, rel_vel))
            t_cpa = max(0.0, min(tau, _dot(rel_pos, rel_vel) / vv))
            cpa = rel_pos - t_cpa * rel_vel
            miss = _norm(cpa) - self._combined_radius(planner_input, nobs)
            if miss < 0.0:
                risk += miss * miss * (1.0 + float(nobs.msg_age_sec))
        obs_risk = 0.0
        full_v = np.asarray([v[0], 0.0, v[1]], dtype=np.float32)
        obs_risk = self._obstacle_path_risk(planner_input, full_v)
        return (
            float(np.sum((v - v_pref) ** 2))
            + self.violation_weight * violation
            + self.risk_weight * risk
            + self.obstacle_weight * obs_risk
            - self.progress_weight * _norm(v)
        )

    def _candidate_cost_3d(self, planner_input: PlannerInput, planes: list[OrcaPlane], v: np.ndarray, v_pref: np.ndarray) -> float:
        violation = sum(max(0.0, _dot(plane.normal, plane.point - v)) ** 2 for plane in planes)
        risk = 0.0
        p_i = np.asarray(planner_input.ego.pos, dtype=np.float32)
        tau = max(1e-6, self.time_horizon_s)
        for nobs in planner_input.neighbors[: self.max_neighbors]:
            rel_pos = np.asarray(nobs.pos, dtype=np.float32) - p_i
            rel_vel = v - np.asarray(nobs.vel, dtype=np.float32)
            vv = max(1e-9, _dot(rel_vel, rel_vel))
            t_cpa = max(0.0, min(tau, _dot(rel_pos, rel_vel) / vv))
            cpa = rel_pos - t_cpa * rel_vel
            miss = _norm(cpa) - self._combined_radius(planner_input, nobs)
            if miss < 0.0:
                risk += miss * miss * (1.0 + float(nobs.msg_age_sec))
        obs_risk = self._obstacle_path_risk(planner_input, v)
        return (
            float(np.sum((v - v_pref) ** 2))
            + self.violation_weight * violation
            + self.risk_weight * risk
            + self.obstacle_weight * obs_risk
            - self.progress_weight * _norm(v)
        )

    def _obstacle_path_risk(self, planner_input: PlannerInput, v: np.ndarray) -> float:
        if not planner_input.obstacles:
            return 0.0
        p_i = np.asarray(planner_input.ego.pos, dtype=np.float32)
        total = 0.0
        for obs in planner_input.obstacles:
            center = np.asarray(obs.center, dtype=np.float32)
            half = np.asarray(obs.half, dtype=np.float32) + float(planner_input.ego.radius + self.safety_margin_m + self.obstacle_margin_m)
            worst = 0.0
            for frac in (0.33, 0.66, 1.0):
                probe = p_i + frac * self.obstacle_time_horizon_s * v
                closest = _closest_point_on_aabb(probe, center, half)
                miss = _norm(probe - closest)
                if miss < self.obstacle_margin_m:
                    worst = max(worst, (self.obstacle_margin_m - miss) ** 2)
            total += worst
        return total

    def _refine_candidates_2d(self, planner_input: PlannerInput, lines: list[OrcaLine], v_pref: np.ndarray, v_seed: np.ndarray, v_max: float) -> np.ndarray:
        candidates: list[np.ndarray] = [v_seed.astype(np.float32), v_pref.astype(np.float32), np.zeros(2, dtype=np.float32)]
        for line in lines:
            t = _dot(line.direction, v_pref - line.point)
            candidates.append(_clamp_speed(line.point + t * line.direction, v_max))
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                denom = _det(lines[i].direction, lines[j].direction)
                if abs(denom) <= 1e-9:
                    continue
                point = lines[i].point + (_det(lines[j].direction, lines[i].point - lines[j].point) / denom) * lines[i].direction
                candidates.append(_clamp_speed(point, v_max))
        for m in range(max(8, self.candidate_samples_2d)):
            ang = 2.0 * math.pi * m / max(8, self.candidate_samples_2d)
            d = np.asarray([math.cos(ang), math.sin(ang)], dtype=np.float32)
            for scale in (0.3, 0.6, 1.0):
                candidates.append((d * (scale * v_max)).astype(np.float32))
        best = v_seed.astype(np.float32)
        best_cost = self._candidate_cost_2d(planner_input, lines, best, v_pref)
        for cand in candidates:
            cand = _clamp_speed(cand, v_max)
            cost = self._candidate_cost_2d(planner_input, lines, cand, v_pref)
            if cost < best_cost:
                best_cost = cost
                best = cand
        return _clamp_speed(best, v_max)

    def _sample_sphere_dirs(self, n: int) -> list[np.ndarray]:
        dirs: list[np.ndarray] = []
        golden = math.pi * (3.0 - math.sqrt(5.0))
        count = max(12, n)
        for i in range(count):
            y = 1.0 - 2.0 * (i + 0.5) / count
            r = math.sqrt(max(0.0, 1.0 - y * y))
            th = golden * i
            dirs.append(np.asarray([r * math.cos(th), y, r * math.sin(th)], dtype=np.float32))
        return dirs

    def _refine_candidates_3d(self, planner_input: PlannerInput, planes: list[OrcaPlane], v_pref: np.ndarray, v_seed: np.ndarray, v_max: float) -> np.ndarray:
        candidates: list[np.ndarray] = [v_seed.astype(np.float32), v_pref.astype(np.float32), np.zeros(3, dtype=np.float32)]
        for plane in planes:
            proj = v_pref + max(0.0, _dot(plane.normal, plane.point - v_pref)) * plane.normal
            candidates.append(_clamp_speed(proj, v_max))
        for d in self._sample_sphere_dirs(self.candidate_samples_3d):
            for scale in (0.25, 0.5, 0.75, 1.0):
                candidates.append((d * (scale * v_max)).astype(np.float32))
        best = v_seed.astype(np.float32)
        best_cost = self._candidate_cost_3d(planner_input, planes, best, v_pref)
        for cand in candidates:
            cand = _clamp_speed(cand, v_max)
            cost = self._candidate_cost_3d(planner_input, planes, cand, v_pref)
            if cost < best_cost:
                best_cost = cost
                best = cand
        return _clamp_speed(best, v_max)
