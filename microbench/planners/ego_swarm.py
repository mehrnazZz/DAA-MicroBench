from __future__ import annotations

from dataclasses import dataclass
import math

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


def _signed_distance_to_aabb(point: np.ndarray, obs: AABBObs) -> float:
    center = np.asarray(obs.center, dtype=np.float32)
    half = np.asarray(obs.half, dtype=np.float32)
    q = np.abs(np.asarray(point, dtype=np.float32) - center) - half
    outside = np.maximum(q, 0.0)
    outside_dist = _norm(outside)
    if outside_dist > 1e-9:
        return outside_dist
    return float(np.max(q))


@dataclass(frozen=True)
class _Candidate:
    label: str
    desired_points: np.ndarray
    offset_norm_m: float


@dataclass(frozen=True)
class _Score:
    total: float
    final_goal_dist_m: float
    progress_m: float
    path_length_m: float
    smoothness_cost: float
    swarm_penalty: float
    obstacle_penalty: float
    min_swarm_clearance_m: float | None
    min_obstacle_clearance_m: float | None
    predicted_swarm_conflict: bool
    predicted_obstacle_conflict: bool


class EgoSwarmPlanner(ILocalPlanner):
    """Clean-room EGO-Swarm-inspired trajectory-sharing baseline.

    This planner adapts EGO-Swarm's main ideas to DAA Microbench's local
    velocity-command contract: each agent independently generates smooth local
    trajectory candidates, scores them against predicted swarm/obstacle
    clearance and dynamic feasibility, then advertises the selected trajectory
    as an intent message for neighboring agents.

    It is not a port of the ROS/C++ GPL implementation.
    """

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.horizon_s = float(cfg.get("horizon_s", 3.2))
        self.rollout_dt_s = float(cfg.get("rollout_dt_s", 0.4))
        self.max_neighbors = int(cfg.get("max_neighbors", 8))
        self.max_candidates = int(cfg.get("max_candidates", 48))
        self.offset_scales_m = tuple(float(x) for x in cfg.get("offset_scales_m", (0.0, 2.0, 4.0, 7.0)))
        self.vertical_offset_scales_m = tuple(float(x) for x in cfg.get("vertical_offset_scales_m", (2.0, 4.0)))
        self.safety_margin_m = float(cfg.get("safety_margin_m", 0.35))
        self.obstacle_margin_m = float(cfg.get("obstacle_margin_m", 0.25))
        self.near_clearance_m = float(cfg.get("near_clearance_m", 1.75))
        self.goal_slowdown_radius_m = float(cfg.get("goal_slowdown_radius_m", 6.0))
        self.stale_inflation_gain = float(cfg.get("stale_inflation_gain", 0.7))
        self.stale_age_cap_s = float(cfg.get("stale_age_cap_s", 1.5))
        self.intent_age_inflation_gain = float(cfg.get("intent_age_inflation_gain", 0.35))
        self.track_uncertainty_speed_gain = float(cfg.get("track_uncertainty_speed_gain", 0.1))
        self.goal_weight = float(cfg.get("goal_weight", 1.15))
        self.progress_weight = float(cfg.get("progress_weight", 0.45))
        self.path_length_weight = float(cfg.get("path_length_weight", 0.04))
        self.smoothness_weight = float(cfg.get("smoothness_weight", 0.22))
        self.offset_weight = float(cfg.get("offset_weight", 0.04))
        self.swarm_collision_weight = float(cfg.get("swarm_collision_weight", 4500.0))
        self.swarm_clearance_weight = float(cfg.get("swarm_clearance_weight", 85.0))
        self.obstacle_collision_weight = float(cfg.get("obstacle_collision_weight", 6000.0))
        self.obstacle_clearance_weight = float(cfg.get("obstacle_clearance_weight", 120.0))
        self.warm_start_weight = float(cfg.get("warm_start_weight", 0.08))
        self.intent_ttl_s = float(cfg.get("intent_ttl_s", 1.0))
        self.intent_tube_margin_m = float(cfg.get("intent_tube_margin_m", 0.35))
        self.max_intent_points = int(cfg.get("max_intent_points", 10))
        self._last_plan: np.ndarray | None = None
        self._last_label: str | None = None
        self.seed = 0

    def reset(self, seed: int) -> None:
        self.seed = int(seed)
        self._last_plan = None
        self._last_label = None

    def compute_cmd(self, planner_input: PlannerInput) -> PlannerOutput:
        ego = planner_input.ego
        current = np.asarray(ego.vel, dtype=np.float32).copy()
        if planner_input.planar:
            current[1] = 0.0

        candidates = self._candidate_trajectories(planner_input)
        best_candidate = candidates[0]
        best_score: _Score | None = None
        best_rollout = np.empty((0, 3), dtype=np.float32)
        raw_scores: list[_Score] = []
        prior_label = self._last_label
        for candidate in candidates:
            rollout, _velocities, _first_cmd = self._rollout(planner_input, candidate.desired_points, current)
            score = self._score(planner_input, candidate, rollout, _velocities)
            raw_scores.append(score)
            if best_score is None or score.total < best_score.total:
                best_candidate = candidate
                best_score = score
                best_rollout = rollout

        assert best_score is not None
        first_target = best_rollout[0] if best_rollout.size else np.asarray(ego.pos, dtype=np.float32)
        desired_v = (first_target - np.asarray(ego.pos, dtype=np.float32)) / max(1e-6, self.rollout_dt_s)
        v_cmd = _limit_delta(desired_v, current, float(ego.a_max) * float(planner_input.dt))
        v_cmd = _clamp_speed(v_cmd, float(ego.v_max))
        if planner_input.planar:
            v_cmd[1] = 0.0

        intent_points = self._intent_points(planner_input, best_rollout)
        intent = IntentMsg(
            sender_id=int(ego.idx),
            timestamp_send_s=float(planner_input.t),
            expiry_s=float(planner_input.t) + self.intent_ttl_s,
            kind="EGO_SWARM_TRAJECTORY",
            tube_radius_m=float(ego.radius) + self.intent_tube_margin_m,
            points=intent_points.astype(float),
            dt_plan_s=float(self.rollout_dt_s),
            mode=str(best_candidate.label),
        )
        self._last_plan = best_rollout.copy()
        self._last_label = best_candidate.label

        return PlannerOutput(
            v_cmd=v_cmd.astype(float),
            intent_out=intent,
            debug_info={
                "ego_swarm_algorithm": "clean_room_receding_horizon_trajectory_sharing",
                "ego_swarm_reference": "EGO-Swarm-inspired; not a port of the GPL ROS/C++ implementation",
                "ego_swarm_horizon_s": float(self.horizon_s),
                "ego_swarm_rollout_dt_s": float(self.rollout_dt_s),
                "ego_swarm_horizon_steps": int(self._horizon_steps()),
                "ego_swarm_candidates": int(len(candidates)),
                "ego_swarm_best_topology": str(best_candidate.label),
                "ego_swarm_best_cost": float(best_score.total),
                "ego_swarm_final_goal_dist_m": float(best_score.final_goal_dist_m),
                "ego_swarm_progress_m": float(best_score.progress_m),
                "ego_swarm_path_length_m": float(best_score.path_length_m),
                "ego_swarm_smoothness_cost": float(best_score.smoothness_cost),
                "ego_swarm_swarm_penalty": float(best_score.swarm_penalty),
                "ego_swarm_obstacle_penalty": float(best_score.obstacle_penalty),
                "ego_swarm_min_swarm_clearance_m": best_score.min_swarm_clearance_m,
                "ego_swarm_min_obstacle_clearance_m": best_score.min_obstacle_clearance_m,
                "ego_swarm_predicted_swarm_conflict": bool(best_score.predicted_swarm_conflict),
                "ego_swarm_predicted_obstacle_conflict": bool(best_score.predicted_obstacle_conflict),
                "ego_swarm_neighbor_count_considered": int(min(len(planner_input.neighbors), self.max_neighbors)),
                "ego_swarm_intent_count_considered": int(sum(1 for intent_obs in planner_input.neighbor_intents if intent_obs.valid)),
                "ego_swarm_obstacle_count_considered": int(len(planner_input.obstacles)),
                "ego_swarm_planar": bool(planner_input.planar),
                "ego_swarm_intent_points": int(intent_points.shape[0]),
                "ego_swarm_accel_delta_norm": float(_norm(v_cmd - current)),
                "ego_swarm_accel_delta_limit": float(float(ego.a_max) * float(planner_input.dt)),
                "ego_swarm_prior_label": prior_label,
                "ego_swarm_candidate_cost_min": float(min(score.total for score in raw_scores)),
                "ego_swarm_candidate_cost_max": float(max(score.total for score in raw_scores)),
            },
        )

    def _horizon_steps(self) -> int:
        return max(2, int(math.ceil(max(1e-6, self.horizon_s) / max(1e-6, self.rollout_dt_s))))

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

    def _candidate_trajectories(self, planner_input: PlannerInput) -> list[_Candidate]:
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

        directions = self._topology_directions(planner_input, v_pref)
        candidates = [self._arc_candidate(planner_input, target, np.zeros(3, dtype=np.float32), "direct")]
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
                candidates.append(self._arc_candidate(planner_input, target, offset, f"{label}:{scale:g}m"))

        if self._last_plan is not None and self._last_plan.size:
            shifted = self._last_plan.copy()
            if planner_input.planar:
                shifted[:, 1] = p0[1]
            candidates.append(_Candidate("warm_start", shifted, 0.0))

        return self._dedupe_candidates(candidates)[: max(1, self.max_candidates)]

    def _topology_directions(self, planner_input: PlannerInput, v_pref: np.ndarray) -> list[tuple[str, np.ndarray]]:
        p0 = np.asarray(planner_input.ego.pos, dtype=np.float32)
        directions: list[tuple[str, np.ndarray]] = []
        if _norm(v_pref) > 1e-9:
            directions.extend(
                [
                    ("left", _perp_xz(v_pref, 1.0)),
                    ("right", _perp_xz(v_pref, -1.0)),
                ]
            )
            if not planner_input.planar:
                directions.extend(
                    [
                        ("vertical_up", np.asarray([0.0, 1.0, 0.0], dtype=np.float32)),
                        ("vertical_down", np.asarray([0.0, -1.0, 0.0], dtype=np.float32)),
                    ]
                )

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
            if points.size == 0:
                continue
            rel = p0 - points[min(1, points.shape[0] - 1)]
            if planner_input.planar:
                rel[1] = 0.0
            if _norm(rel) > 1e-9:
                directions.append((f"intent_{int(intent.sender_id)}_away", rel))

        for obs_idx, obs in enumerate(planner_input.obstacles):
            rel = p0 - _closest_point_on_aabb(p0, obs)
            if planner_input.planar:
                rel[1] = 0.0
            if _norm(rel) > 1e-9:
                directions.append((f"obstacle_{obs_idx}_away", rel))
                directions.append((f"obstacle_{obs_idx}_side", rel + _perp_xz(rel, 1.0)))

        return directions

    def _arc_candidate(self, planner_input: PlannerInput, target: np.ndarray, offset: np.ndarray, label: str) -> _Candidate:
        p0 = np.asarray(planner_input.ego.pos, dtype=np.float32)
        steps = self._horizon_steps()
        points = []
        for k in range(1, steps + 1):
            tau = k / steps
            point = p0 + (target - p0) * tau + np.sin(np.pi * tau) * offset
            if planner_input.planar:
                point[1] = p0[1]
            points.append(point.astype(np.float32))
        return _Candidate(label=label, desired_points=np.asarray(points, dtype=np.float32), offset_norm_m=_norm(offset))

    def _dedupe_candidates(self, candidates: list[_Candidate]) -> list[_Candidate]:
        out: list[_Candidate] = []
        seen: set[tuple[int, int, int, int, int, int]] = set()
        for candidate in candidates:
            if candidate.desired_points.size == 0:
                continue
            first = candidate.desired_points[0]
            last = candidate.desired_points[-1]
            key = tuple(int(round(float(x) * 1000.0)) for x in (*first, *last))
            if key in seen:
                continue
            seen.add(key)
            out.append(candidate)
        return out

    def _rollout(
        self,
        planner_input: PlannerInput,
        desired_points: np.ndarray,
        current_velocity: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ego = planner_input.ego
        pos = np.asarray(ego.pos, dtype=np.float32).copy()
        vel = np.asarray(current_velocity, dtype=np.float32).copy()
        points: list[np.ndarray] = []
        velocities: list[np.ndarray] = []
        first_cmd: np.ndarray | None = None
        max_delta = float(ego.a_max) * self.rollout_dt_s
        for desired in desired_points:
            target_vel = (np.asarray(desired, dtype=np.float32) - pos) / max(1e-6, self.rollout_dt_s)
            target_vel = _clamp_speed(target_vel, float(ego.v_max))
            vel = _limit_delta(target_vel, vel, max_delta)
            vel = _clamp_speed(vel, float(ego.v_max))
            if planner_input.planar:
                vel[1] = 0.0
            if first_cmd is None:
                first_cmd = vel.copy()
            pos = pos + vel * self.rollout_dt_s
            if planner_input.planar:
                pos[1] = float(ego.pos[1])
            points.append(pos.copy())
            velocities.append(vel.copy())
        if first_cmd is None:
            first_cmd = np.zeros(3, dtype=np.float32)
        return np.asarray(points, dtype=np.float32), np.asarray(velocities, dtype=np.float32), first_cmd

    def _score(
        self,
        planner_input: PlannerInput,
        candidate: _Candidate,
        points: np.ndarray,
        velocities: np.ndarray,
    ) -> _Score:
        ego = planner_input.ego
        p0 = np.asarray(ego.pos, dtype=np.float32)
        goal = np.asarray(ego.goal, dtype=np.float32)
        initial_goal_dist = _norm(goal - p0)
        final_goal_dist = _norm(goal - points[-1]) if points.size else initial_goal_dist
        progress = initial_goal_dist - final_goal_dist
        path_length = self._path_length(p0, points)
        smoothness = self._smoothness_cost(np.asarray(ego.vel, dtype=np.float32), velocities)
        swarm_penalty, min_swarm, swarm_conflict = self._swarm_penalty(planner_input, points)
        obstacle_penalty, min_obstacle, obstacle_conflict = self._obstacle_penalty(planner_input, points)
        warm_start = self._warm_start_cost(points)

        total = (
            self.goal_weight * final_goal_dist
            - self.progress_weight * progress
            + self.path_length_weight * path_length
            + self.smoothness_weight * smoothness
            + self.offset_weight * candidate.offset_norm_m
            + swarm_penalty
            + obstacle_penalty
            + self.warm_start_weight * warm_start
        )
        return _Score(
            total=float(total),
            final_goal_dist_m=float(final_goal_dist),
            progress_m=float(progress),
            path_length_m=float(path_length),
            smoothness_cost=float(smoothness),
            swarm_penalty=float(swarm_penalty),
            obstacle_penalty=float(obstacle_penalty),
            min_swarm_clearance_m=min_swarm,
            min_obstacle_clearance_m=min_obstacle,
            predicted_swarm_conflict=bool(swarm_conflict),
            predicted_obstacle_conflict=bool(obstacle_conflict),
        )

    def _path_length(self, p0: np.ndarray, points: np.ndarray) -> float:
        if points.size == 0:
            return 0.0
        total = _norm(points[0] - p0)
        for a, b in zip(points[:-1], points[1:]):
            total += _norm(b - a)
        return float(total)

    def _smoothness_cost(self, current_velocity: np.ndarray, velocities: np.ndarray) -> float:
        if velocities.size == 0:
            return 0.0
        prev = np.asarray(current_velocity, dtype=np.float32)
        cost = 0.0
        for vel in velocities:
            cost += _norm(vel - prev) ** 2
            prev = vel
        return float(cost)

    def _warm_start_cost(self, points: np.ndarray) -> float:
        if self._last_plan is None or self._last_plan.size == 0 or points.size == 0:
            return 0.0
        count = min(points.shape[0], self._last_plan.shape[0])
        return float(np.mean(np.linalg.norm(points[:count] - self._last_plan[:count], axis=1)))

    def _swarm_penalty(self, planner_input: PlannerInput, points: np.ndarray) -> tuple[float, float | None, bool]:
        if points.size == 0:
            return 0.0, None, False
        intent_by_sender = {
            int(intent.sender_id): intent
            for intent in planner_input.neighbor_intents
            if intent.valid and np.asarray(intent.points).size > 0
        }
        seen_ids: set[int] = set()
        penalty = 0.0
        min_clearance: float | None = None
        conflict = False
        for step_idx, point in enumerate(points, start=1):
            t = step_idx * self.rollout_dt_s
            for nobs in planner_input.neighbors[: self.max_neighbors]:
                seen_ids.add(int(nobs.idx))
                other_pos = self._neighbor_prediction(nobs, intent_by_sender.get(int(nobs.idx)), step_idx, t)
                inflation = self._neighbor_inflation(nobs)
                if int(nobs.idx) in intent_by_sender:
                    inflation += self._intent_inflation(intent_by_sender[int(nobs.idx)])
                clearance = _norm(np.asarray(point, dtype=np.float32) - other_pos) - (
                    float(planner_input.ego.radius) + float(nobs.radius) + self.safety_margin_m + inflation
                )
                min_clearance = clearance if min_clearance is None else min(min_clearance, clearance)
                penalty += self._clearance_penalty(clearance, self.swarm_collision_weight, self.swarm_clearance_weight)
                conflict = conflict or clearance < 0.0
            for sender_id, intent in intent_by_sender.items():
                if sender_id in seen_ids:
                    continue
                other_pos = self._intent_prediction(intent, step_idx)
                clearance = _norm(np.asarray(point, dtype=np.float32) - other_pos) - (
                    float(planner_input.ego.radius) + float(intent.tube_radius_m) + self.safety_margin_m + self._intent_inflation(intent)
                )
                min_clearance = clearance if min_clearance is None else min(min_clearance, clearance)
                penalty += self._clearance_penalty(clearance, self.swarm_collision_weight, self.swarm_clearance_weight)
                conflict = conflict or clearance < 0.0
        return float(penalty), min_clearance, bool(conflict)

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
        if intent.dt_plan_s is not None and intent.dt_plan_s > 1e-9:
            idx = min(points.shape[0] - 1, max(0, int(round((step_idx * self.rollout_dt_s) / float(intent.dt_plan_s)))))
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

    def _obstacle_penalty(self, planner_input: PlannerInput, points: np.ndarray) -> tuple[float, float | None, bool]:
        if not planner_input.obstacles or points.size == 0:
            return 0.0, None, False
        penalty = 0.0
        min_clearance: float | None = None
        conflict = False
        for point in points:
            for obs in planner_input.obstacles:
                clearance = _signed_distance_to_aabb(point, obs) - float(planner_input.ego.radius) - self.obstacle_margin_m
                min_clearance = clearance if min_clearance is None else min(min_clearance, clearance)
                penalty += self._clearance_penalty(clearance, self.obstacle_collision_weight, self.obstacle_clearance_weight)
                conflict = conflict or clearance < 0.0
        return float(penalty), min_clearance, bool(conflict)

    def _clearance_penalty(self, clearance: float, collision_weight: float, clearance_weight: float) -> float:
        if clearance < 0.0:
            return float(collision_weight * (-clearance + 1.0) ** 2)
        if clearance < self.near_clearance_m:
            return float(clearance_weight * (self.near_clearance_m - clearance) ** 2)
        return 0.0

    def _intent_points(self, planner_input: PlannerInput, points: np.ndarray) -> np.ndarray:
        ego_pos = np.asarray(planner_input.ego.pos, dtype=np.float32)
        if points.size == 0:
            out = ego_pos.reshape(1, 3)
        else:
            out = np.vstack([ego_pos, points])
        if self.max_intent_points > 0 and out.shape[0] > self.max_intent_points:
            idx = np.linspace(0, out.shape[0] - 1, self.max_intent_points).round().astype(int)
            out = out[idx]
        return out.astype(np.float32)
