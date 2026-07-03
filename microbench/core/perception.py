from __future__ import annotations

import math
import numpy as np

from microbench.types import AgentState, NeighborObs


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.zeros_like(v)
    return v / n


def _segment_intersects_aabb(p0: np.ndarray, p1: np.ndarray, center: np.ndarray, half: np.ndarray) -> bool:
    lo = center - half
    hi = center + half
    d = p1 - p0
    t_min = 0.0
    t_max = 1.0
    for axis in range(3):
        if abs(float(d[axis])) < 1e-12:
            if p0[axis] < lo[axis] or p0[axis] > hi[axis]:
                return False
            continue
        inv = 1.0 / float(d[axis])
        t1 = float((lo[axis] - p0[axis]) * inv)
        t2 = float((hi[axis] - p0[axis]) * inv)
        t_enter = min(t1, t2)
        t_exit = max(t1, t2)
        t_min = max(t_min, t_enter)
        t_max = min(t_max, t_exit)
        if t_min > t_max:
            return False
    return t_max >= 0.0 and t_min <= 1.0


def _line_of_sight_clear(p0: np.ndarray, p1: np.ndarray, obstacles: list[dict], occlusion_margin_m: float) -> bool:
    for ob in obstacles:
        if "aabb" not in ob:
            continue
        aabb = ob["aabb"]
        center = np.asarray(aabb.get("center", [0.0, 0.0, 0.0]), dtype=float)
        half = np.asarray(aabb.get("half", [0.0, 0.0, 0.0]), dtype=float) + float(occlusion_margin_m)
        if _segment_intersects_aabb(p0, p1, center, half):
            return False
    return True


def _in_fov(rel: np.ndarray, forward: np.ndarray, fov_deg: float, planar: bool) -> bool:
    if fov_deg >= 359.999:
        return True
    if planar:
        rel_v = np.asarray([rel[0], rel[2]], dtype=float)
        fwd_v = np.asarray([forward[0], forward[2]], dtype=float)
    else:
        rel_v = np.asarray(rel, dtype=float)
        fwd_v = np.asarray(forward, dtype=float)
    rel_n = float(np.linalg.norm(rel_v))
    fwd_n = float(np.linalg.norm(fwd_v))
    if rel_n < 1e-9:
        return True
    if fwd_n < 1e-9:
        return True
    cosang = float(np.dot(rel_v, fwd_v) / max(1e-9, rel_n * fwd_n))
    cosang = max(-1.0, min(1.0, cosang))
    angle_deg = math.degrees(math.acos(cosang))
    return angle_deg <= 0.5 * float(fov_deg)


def sense_neighbors(
    *,
    ego: AgentState,
    states: list[AgentState],
    goal_dir: np.ndarray,
    obstacles: list[dict],
    perception_cfg: dict,
    planar: bool,
    rng: np.random.Generator,
) -> list[NeighborObs]:
    sensor_cfg = perception_cfg.get("sensor", perception_cfg)
    range_m = float(sensor_cfg.get("range_m", perception_cfg.get("range_m", 30.0)))
    fov_deg = float(sensor_cfg.get("fov_deg", perception_cfg.get("fov_deg", 360.0)))
    occlusion_enabled = bool(sensor_cfg.get("occlusion", perception_cfg.get("occlusion", False)))
    occlusion_margin_m = float(sensor_cfg.get("occlusion_margin_m", perception_cfg.get("occlusion_margin_m", 0.0)))
    false_negative_p = float(sensor_cfg.get("false_negative_p", perception_cfg.get("false_negative_p", 0.0)))
    sigma_pos = float(sensor_cfg.get("noise_sigma_pos_m", perception_cfg.get("noise_sigma_pos_m", 0.0)))
    sigma_vel = float(sensor_cfg.get("noise_sigma_vel_mps", perception_cfg.get("noise_sigma_vel_mps", 0.0)))

    forward = np.asarray(ego.vel, dtype=float)
    if float(np.linalg.norm(forward)) < 1e-6:
        forward = np.asarray(goal_dir, dtype=float)
    forward = _normalize(forward)

    out: list[NeighborObs] = []
    for other in states:
        if other.idx == ego.idx:
            continue
        rel = np.asarray(other.pos, dtype=float) - np.asarray(ego.pos, dtype=float)
        d = float(np.linalg.norm(rel))
        if d > range_m:
            continue
        if not _in_fov(rel, forward, fov_deg, planar):
            continue
        if occlusion_enabled and not _line_of_sight_clear(
            np.asarray(ego.pos, dtype=float),
            np.asarray(other.pos, dtype=float),
            obstacles,
            occlusion_margin_m,
        ):
            continue
        if false_negative_p > 0.0 and rng.random() < false_negative_p:
            continue

        pos = np.asarray(other.pos, dtype=float).copy()
        vel = np.asarray(other.vel, dtype=float).copy()
        if sigma_pos > 0.0:
            pos += rng.normal(0.0, sigma_pos, size=3)
        if sigma_vel > 0.0:
            vel += rng.normal(0.0, sigma_vel, size=3)
        out.append(
            NeighborObs(
                idx=int(other.idx),
                pos=pos,
                vel=vel,
                radius=float(other.radius),
                msg_age_sec=0.0,
                valid=True,
                source="sensor",
            )
        )
    return out


def fuse_observations(v2v_obs: list[NeighborObs], sensor_obs: list[NeighborObs]) -> list[NeighborObs]:
    by_id: dict[int, NeighborObs] = {n.idx: n for n in v2v_obs if n.valid}
    for obs in sensor_obs:
        prev = by_id.get(obs.idx)
        if prev is None or obs.msg_age_sec <= prev.msg_age_sec:
            by_id[obs.idx] = obs
    return [by_id[k] for k in sorted(by_id)]
