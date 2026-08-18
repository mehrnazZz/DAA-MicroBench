from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from microbench.planners.base import ILocalPlanner
from microbench.types import AABBObs, AgentState, PlannerInput, PlannerOutput


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.zeros(3, dtype=float)
    return np.asarray(v, dtype=float) / n


def _clip_norm(v: np.ndarray, limit: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= float(limit) + 1e-12 or n < 1e-12:
        return np.asarray(v, dtype=float)
    return np.asarray(v, dtype=float) / n * float(limit)


def _closest_point_on_aabb(pos: np.ndarray, obstacle: AABBObs) -> np.ndarray:
    center = np.asarray(obstacle.center, dtype=float)
    half = np.asarray(obstacle.half, dtype=float)
    return np.minimum(np.maximum(pos, center - half), center + half)


def _lateral_axis(axis: np.ndarray, *, planar: bool, agent_idx: int) -> np.ndarray:
    axis = _normalize(axis)
    if planar:
        lateral = np.asarray([-axis[2], 0.0, axis[0]], dtype=float)
    else:
        ref = np.asarray([0.0, 1.0, 0.0], dtype=float)
        if abs(float(np.dot(axis, ref))) > 0.9:
            ref = np.asarray([1.0, 0.0, 0.0], dtype=float)
        lateral = np.cross(axis, ref)
    lateral = _normalize(lateral)
    if np.linalg.norm(lateral) < 1e-9:
        lateral = np.asarray([0.0, 0.0, 1.0], dtype=float)
    if int(agent_idx) % 2:
        lateral = -lateral
    return lateral


@dataclass
class CentralizedOracleOutput:
    v_cmds: list[np.ndarray]
    debug: list[dict[str, Any]]


class CentralizedOraclePlanner(ILocalPlanner):
    """Privileged joint controller used only as a non-deployable upper bound.

    The normal local planner API intentionally does not expose global truth. The
    episode engine calls ``compute_joint_cmds`` only when every agent uses this
    method, and planner metadata labels it as non-deployable.
    """

    def __init__(self, cfg: dict[str, Any] | None = None):
        cfg = dict(cfg or {})
        self.horizon_s = float(cfg.get("horizon_s", 4.0))
        self.samples = max(2, int(cfg.get("trajectory_samples", 6)))
        self.pair_trigger_margin_m = float(cfg.get("pair_trigger_margin_m", 2.0))
        self.pair_hard_margin_m = float(cfg.get("pair_hard_margin_m", 0.25))
        self.obstacle_trigger_margin_m = float(cfg.get("obstacle_trigger_margin_m", 1.5))
        self.goal_weight = float(cfg.get("goal_weight", 1.0))
        self.pair_weight = float(cfg.get("pair_weight", 1.1))
        self.lateral_weight = float(cfg.get("lateral_weight", 0.85))
        self.obstacle_weight = float(cfg.get("obstacle_weight", 1.0))
        self.max_pair_corrections = max(0, int(cfg.get("max_pair_corrections", 256)))
        self.agent_id = 0

    def reset(self, agent_id: int = 0, seed: int = 0, config: dict[str, Any] | None = None) -> None:
        _ = seed, config
        self.agent_id = int(agent_id)

    def compute_cmd(self, inp: PlannerInput) -> PlannerOutput:
        # Fallback for direct contract probes. Benchmark runs use the privileged
        # joint path in EpisodeEngine, not this local API.
        v_cmd = np.asarray(inp.goal_dir, dtype=float) * float(inp.ego.v_max)
        if inp.planar:
            v_cmd[1] = 0.0
        return PlannerOutput(
            v_cmd=_clip_norm(v_cmd, inp.ego.v_max),
            debug_info={
                "centralized_oracle": True,
                "centralized_oracle_local_fallback": True,
                "non_deployable": True,
                "privileged_global_state": False,
            },
        )

    def compute_joint_cmds(
        self,
        *,
        states: list[AgentState],
        obstacles: list[AABBObs],
        planar: bool,
        t: float,
        dt: float,
    ) -> CentralizedOracleOutput:
        _ = dt
        n_agents = len(states)
        v_cmds: list[np.ndarray] = []
        debug: list[dict[str, Any]] = []
        min_pair_clearance = [float("inf") for _ in states]
        min_obstacle_clearance = [float("inf") for _ in states]
        pair_corrections = [0 for _ in states]
        obstacle_corrections = [0 for _ in states]

        for s in states:
            if s.done:
                v_cmds.append(np.zeros(3, dtype=float))
                continue
            goal = np.asarray(s.goal, dtype=float) - np.asarray(s.pos, dtype=float)
            cmd = _normalize(goal) * float(s.v_max) * self.goal_weight
            if planar:
                cmd[1] = 0.0
            v_cmds.append(_clip_norm(cmd, s.v_max))

        times = np.linspace(0.0, self.horizon_s, self.samples)
        for _pass in range(2):
            corrections = [np.zeros(3, dtype=float) for _ in states]
            inspected = 0
            for i in range(n_agents):
                si = states[i]
                if si.done:
                    continue
                for j in range(i + 1, n_agents):
                    if inspected >= self.max_pair_corrections:
                        break
                    sj = states[j]
                    if sj.done:
                        continue
                    inspected += 1
                    min_clearance, ttc_s = self._pair_clearance(si, sj, v_cmds[i], v_cmds[j], times)
                    min_pair_clearance[i] = min(min_pair_clearance[i], min_clearance)
                    min_pair_clearance[j] = min(min_pair_clearance[j], min_clearance)
                    required = float(si.radius + sj.radius + self.pair_hard_margin_m)
                    trigger = required + self.pair_trigger_margin_m
                    if min_clearance >= trigger:
                        continue

                    pi = np.asarray(si.pos, dtype=float) + v_cmds[i] * ttc_s
                    pj = np.asarray(sj.pos, dtype=float) + v_cmds[j] * ttc_s
                    axis = _normalize(pi - pj)
                    if np.linalg.norm(axis) < 1e-9:
                        axis = _normalize(np.asarray(si.pos, dtype=float) - np.asarray(sj.pos, dtype=float))
                    if np.linalg.norm(axis) < 1e-9:
                        axis = np.asarray([1.0, 0.0, 0.0], dtype=float)

                    closing = max(0.0, float(np.dot(v_cmds[j] - v_cmds[i], axis)))
                    deficit = max(0.0, trigger - min_clearance)
                    lateral_i = _lateral_axis(axis, planar=planar, agent_idx=i)
                    lateral_j = _lateral_axis(axis, planar=planar, agent_idx=j)
                    strength = self.pair_weight * (0.35 + deficit + 0.2 * closing)
                    corrections[i] += (axis * strength) + (lateral_i * self.lateral_weight * deficit)
                    corrections[j] -= (axis * strength) - (lateral_j * self.lateral_weight * deficit)
                    pair_corrections[i] += 1
                    pair_corrections[j] += 1
                if inspected >= self.max_pair_corrections:
                    break

            for i, s in enumerate(states):
                if s.done:
                    continue
                obstacle_cmd = self._obstacle_correction(s, v_cmds[i], obstacles, times)
                if obstacle_cmd is not None:
                    correction, clearance = obstacle_cmd
                    corrections[i] += correction
                    min_obstacle_clearance[i] = min(min_obstacle_clearance[i], clearance)
                    obstacle_corrections[i] += 1

            for i, s in enumerate(states):
                if s.done:
                    continue
                cmd = v_cmds[i] + corrections[i]
                if planar:
                    cmd[1] = 0.0
                v_cmds[i] = _clip_norm(cmd, s.v_max)

        for i, s in enumerate(states):
            if not np.isfinite(min_pair_clearance[i]):
                min_pair_clearance[i] = float("inf")
            if not np.isfinite(min_obstacle_clearance[i]):
                min_obstacle_clearance[i] = float("inf")
            debug.append(
                {
                    "centralized_oracle": True,
                    "non_deployable": True,
                    "privileged_global_state": True,
                    "oracle_scope": "all_agents_all_obstacles",
                    "oracle_horizon_s": float(self.horizon_s),
                    "oracle_samples": int(self.samples),
                    "oracle_planar": bool(planar),
                    "oracle_time_s": float(t),
                    "oracle_pair_corrections": int(pair_corrections[i]),
                    "oracle_obstacle_corrections": int(obstacle_corrections[i]),
                    "oracle_min_pair_clearance_m": float(min_pair_clearance[i]),
                    "oracle_min_obstacle_clearance_m": float(min_obstacle_clearance[i]),
                    "oracle_active_agents": int(sum(0 if st.done else 1 for st in states)),
                    "oracle_done": bool(s.done),
                }
            )

        return CentralizedOracleOutput(v_cmds=v_cmds, debug=debug)

    @staticmethod
    def _pair_clearance(
        si: AgentState,
        sj: AgentState,
        vi: np.ndarray,
        vj: np.ndarray,
        times: np.ndarray,
    ) -> tuple[float, float]:
        pi0 = np.asarray(si.pos, dtype=float)
        pj0 = np.asarray(sj.pos, dtype=float)
        rel_p = pi0 - pj0
        rel_v = np.asarray(vi, dtype=float) - np.asarray(vj, dtype=float)
        denom = float(np.dot(rel_v, rel_v))
        if denom > 1e-12:
            ttc = float(np.clip(-float(np.dot(rel_p, rel_v)) / denom, 0.0, float(times[-1])))
            query = np.unique(np.concatenate([times, np.asarray([ttc], dtype=float)]))
        else:
            query = times
        clearances = [
            float(np.linalg.norm((pi0 + vi * tau) - (pj0 + vj * tau)) - si.radius - sj.radius)
            for tau in query
        ]
        idx = int(np.argmin(clearances))
        return float(clearances[idx]), float(query[idx])

    def _obstacle_correction(
        self,
        s: AgentState,
        v_cmd: np.ndarray,
        obstacles: list[AABBObs],
        times: np.ndarray,
    ) -> tuple[np.ndarray, float] | None:
        best_clearance = float("inf")
        best_axis = np.zeros(3, dtype=float)
        for obstacle in obstacles:
            for tau in times:
                p = np.asarray(s.pos, dtype=float) + np.asarray(v_cmd, dtype=float) * float(tau)
                closest = _closest_point_on_aabb(p, obstacle)
                rel = p - closest
                dist = float(np.linalg.norm(rel))
                clearance = dist - float(s.radius)
                if clearance < best_clearance:
                    best_clearance = clearance
                    if dist > 1e-9:
                        best_axis = rel / dist
                    else:
                        center = np.asarray(obstacle.center, dtype=float)
                        best_axis = _normalize(p - center)
        trigger = float(s.radius + self.obstacle_trigger_margin_m)
        if best_clearance >= trigger:
            return None
        if np.linalg.norm(best_axis) < 1e-9:
            best_axis = np.asarray([0.0, 1.0, 0.0], dtype=float)
        strength = self.obstacle_weight * max(0.0, trigger - best_clearance + 0.25)
        return best_axis * strength, float(best_clearance)
