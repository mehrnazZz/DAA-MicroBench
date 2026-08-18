from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
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
        **_: Any,
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


class CentralizedMpcOraclePlanner(CentralizedOraclePlanner):
    """Privileged route-aware centralized MPC oracle.

    This controller is intentionally non-deployable: it uses all current agent
    states, all static obstacles, and world bounds. It combines coarse global
    route planning with deterministic joint candidate-MPC refinement.
    """

    def __init__(self, cfg: dict[str, Any] | None = None):
        super().__init__(cfg)
        cfg = dict(cfg or {})
        self.horizon_s = float(cfg.get("horizon_s", 5.0))
        self.samples = max(3, int(cfg.get("trajectory_samples", 9)))
        self.coord_iterations = max(1, int(cfg.get("coord_iterations", 3)))
        self.route_resolution_m = float(cfg.get("route_resolution_m", 3.0))
        self.route_resolution_3d_m = float(cfg.get("route_resolution_3d_m", 4.0))
        self.max_grid_cells = max(1000, int(cfg.get("max_grid_cells", 180_000)))
        self.max_astar_expansions = max(1000, int(cfg.get("max_astar_expansions", 25_000)))
        self.route_margin_m = float(cfg.get("route_margin_m", 0.35))
        self.route_lookahead_m = float(cfg.get("route_lookahead_m", 9.0))
        self.replan_goal_shift_m = float(cfg.get("replan_goal_shift_m", 1.0))
        self.mpc_safety_margin_m = float(cfg.get("mpc_safety_margin_m", 0.6))
        self.mpc_obstacle_margin_m = float(cfg.get("mpc_obstacle_margin_m", 0.45))
        self.collision_weight = float(cfg.get("collision_weight", 18_000.0))
        self.clearance_weight = float(cfg.get("clearance_weight", 160.0))
        self.obstacle_mpc_weight = float(cfg.get("obstacle_mpc_weight", 20_000.0))
        self.route_tracking_weight = float(cfg.get("route_tracking_weight", 1.8))
        self.progress_weight = float(cfg.get("progress_weight", 1.0))
        self.smoothness_weight = float(cfg.get("smoothness_weight", 0.25))
        self.stop_penalty_weight = float(cfg.get("stop_penalty_weight", 0.05))
        self.command_update_period_s = float(cfg.get("command_update_period_s", 0.1))
        self.route_cache: dict[int, dict[str, Any]] = {}
        self._last_joint_t: float | None = None
        self._last_joint_output: CentralizedOracleOutput | None = None

    def reset(self, agent_id: int = 0, seed: int = 0, config: dict[str, Any] | None = None) -> None:
        super().reset(agent_id=agent_id, seed=seed, config=config)
        if int(agent_id) == 0:
            self.route_cache.clear()
            self._last_joint_t = None
            self._last_joint_output = None

    def compute_cmd(self, inp: PlannerInput) -> PlannerOutput:
        out = super().compute_cmd(inp)
        info = dict(out.debug_info)
        info["centralized_mpc_oracle"] = True
        info["centralized_mpc_oracle_local_fallback"] = True
        return PlannerOutput(v_cmd=out.v_cmd, intent_out=out.intent_out, messages_out=out.messages_out, debug_info=info)

    def compute_joint_cmds(
        self,
        *,
        states: list[AgentState],
        obstacles: list[AABBObs],
        planar: bool,
        t: float,
        dt: float,
        world_bounds: dict[str, float] | None = None,
        **_: Any,
    ) -> CentralizedOracleOutput:
        _ = dt
        if (
            self._last_joint_output is not None
            and self._last_joint_t is not None
            and self.command_update_period_s > 0.0
            and t - self._last_joint_t < self.command_update_period_s - 1e-12
            and len(self._last_joint_output.v_cmds) == len(states)
        ):
            return CentralizedOracleOutput(
                v_cmds=[cmd.copy() for cmd in self._last_joint_output.v_cmds],
                debug=[
                    {
                        **dict(info),
                        "centralized_mpc_oracle_replanned": False,
                        "centralized_mpc_oracle_cached_reuse": True,
                        "oracle_time_s": float(t),
                    }
                    for info in self._last_joint_output.debug
                ],
            )

        times = np.linspace(0.0, self.horizon_s, self.samples)
        routes: dict[int, list[np.ndarray]] = {}
        route_targets: dict[int, np.ndarray] = {}
        route_debug: dict[int, dict[str, Any]] = {}
        preferred: list[np.ndarray] = []

        for s in states:
            if s.done:
                routes[s.idx] = [np.asarray(s.pos, dtype=float)]
                route_targets[s.idx] = np.asarray(s.pos, dtype=float)
                preferred.append(np.zeros(3, dtype=float))
                route_debug[s.idx] = {"route_status": "done", "route_waypoints": 1}
                continue
            route, info = self._route_for_agent(
                state=s,
                obstacles=obstacles,
                planar=planar,
                world_bounds=world_bounds or {},
            )
            target = self._route_target(np.asarray(s.pos, dtype=float), route, lookahead_m=self.route_lookahead_m)
            cmd = _normalize(target - np.asarray(s.pos, dtype=float)) * float(s.v_max)
            if np.linalg.norm(target - s.goal) < self.route_lookahead_m:
                remaining = float(np.linalg.norm(s.goal - s.pos))
                cmd *= min(1.0, max(0.25, remaining / max(1e-6, self.route_lookahead_m)))
            if planar:
                cmd[1] = 0.0
            routes[s.idx] = route
            route_targets[s.idx] = target
            route_debug[s.idx] = info
            preferred.append(_clip_norm(cmd, s.v_max))

        v_cmds = [cmd.copy() for cmd in preferred]
        candidate_counts = [0 for _ in states]
        min_pair_clearance = [float("inf") for _ in states]
        min_obstacle_clearance = [float("inf") for _ in states]

        for _iteration in range(self.coord_iterations):
            order = self._planning_order(states, route_targets, v_cmds)
            for i in order:
                s = states[i]
                if s.done:
                    continue
                candidates = self._candidate_commands(s, preferred[i], route_targets[s.idx], planar)
                candidate_counts[i] = max(candidate_counts[i], len(candidates))
                best_cmd = v_cmds[i]
                best_cost = float("inf")
                for cand in candidates:
                    cmds = list(v_cmds)
                    cmds[i] = cand
                    cost, pair_clearance, obstacle_clearance = self._joint_candidate_cost(
                        agent_id=i,
                        states=states,
                        cmds=cmds,
                        preferred=preferred,
                        route_target=route_targets[s.idx],
                        obstacles=obstacles,
                        times=times,
                    )
                    if cost < best_cost:
                        best_cost = cost
                        best_cmd = cand
                        min_pair_clearance[i] = min(min_pair_clearance[i], pair_clearance)
                        min_obstacle_clearance[i] = min(min_obstacle_clearance[i], obstacle_clearance)
                v_cmds[i] = _clip_norm(best_cmd, s.v_max)
                if planar:
                    v_cmds[i][1] = 0.0

        debug: list[dict[str, Any]] = []
        for i, s in enumerate(states):
            pair_clearance = self._min_pair_clearance_for_agent(i, states, v_cmds, times)
            obstacle_clearance = self._min_obstacle_clearance_for_agent(i, states, v_cmds, obstacles, times)
            if np.isfinite(pair_clearance):
                min_pair_clearance[i] = min(min_pair_clearance[i], pair_clearance)
            if np.isfinite(obstacle_clearance):
                min_obstacle_clearance[i] = min(min_obstacle_clearance[i], obstacle_clearance)
            debug.append(
                {
                    "centralized_oracle": True,
                    "centralized_mpc_oracle": True,
                    "non_deployable": True,
                    "privileged_global_state": True,
                    "oracle_scope": "all_agents_all_obstacles_world_bounds",
                    "oracle_horizon_s": float(self.horizon_s),
                    "oracle_samples": int(self.samples),
                    "oracle_planar": bool(planar),
                    "oracle_time_s": float(t),
                    "centralized_mpc_oracle_coord_iterations": int(self.coord_iterations),
                    "centralized_mpc_oracle_replanned": True,
                    "centralized_mpc_oracle_cached_reuse": False,
                    "centralized_mpc_oracle_candidates": int(candidate_counts[i]),
                    "centralized_mpc_oracle_route_target": route_targets[s.idx].tolist(),
                    "centralized_mpc_oracle_min_pair_clearance_m": float(min_pair_clearance[i]),
                    "centralized_mpc_oracle_min_obstacle_clearance_m": float(min_obstacle_clearance[i]),
                    "centralized_mpc_oracle_active_agents": int(sum(0 if st.done else 1 for st in states)),
                    "centralized_mpc_oracle_done": bool(s.done),
                    **route_debug.get(s.idx, {}),
                }
            )

        out = CentralizedOracleOutput(v_cmds=[cmd.copy() for cmd in v_cmds], debug=[dict(info) for info in debug])
        self._last_joint_t = float(t)
        self._last_joint_output = out
        return CentralizedOracleOutput(v_cmds=[cmd.copy() for cmd in out.v_cmds], debug=[dict(info) for info in out.debug])

    def _route_for_agent(
        self,
        *,
        state: AgentState,
        obstacles: list[AABBObs],
        planar: bool,
        world_bounds: dict[str, float],
    ) -> tuple[list[np.ndarray], dict[str, Any]]:
        goal_key = tuple(np.round(np.asarray(state.goal, dtype=float), 3).tolist())
        cache = self.route_cache.get(state.idx)
        if cache and cache.get("goal_key") == goal_key:
            route = [np.asarray(p, dtype=float) for p in cache["route"]]
            return route, {
                "route_status": str(cache.get("status", "cached")),
                "route_waypoints": int(len(route)),
                "route_cached": True,
            }

        route, status = self._astar_route(
            start=np.asarray(state.pos, dtype=float),
            goal=np.asarray(state.goal, dtype=float),
            radius=float(state.radius),
            obstacles=obstacles,
            planar=planar,
            world_bounds=world_bounds,
        )
        self.route_cache[state.idx] = {
            "goal_key": goal_key,
            "route": [p.copy() for p in route],
            "status": status,
        }
        return route, {"route_status": status, "route_waypoints": int(len(route)), "route_cached": False}

    def _astar_route(
        self,
        *,
        start: np.ndarray,
        goal: np.ndarray,
        radius: float,
        obstacles: list[AABBObs],
        planar: bool,
        world_bounds: dict[str, float],
    ) -> tuple[list[np.ndarray], str]:
        axes = (0, 2) if planar else (0, 1, 2)
        resolution = self.route_resolution_m if planar else self.route_resolution_3d_m
        mins, maxs = self._route_bounds(start, goal, obstacles, world_bounds, axes, margin=8.0)
        spans = np.maximum(np.asarray(maxs) - np.asarray(mins), resolution)
        dims = np.floor(spans / resolution).astype(int) + 1
        total_cells = int(np.prod(dims))
        if total_cells > self.max_grid_cells:
            scale = (total_cells / self.max_grid_cells) ** (1.0 / len(axes))
            resolution *= float(scale) * 1.05
            dims = np.floor(spans / resolution).astype(int) + 1

        mins_arr = np.asarray(mins, dtype=float)
        dims_t = tuple(int(d) for d in dims)

        def to_idx(p: np.ndarray) -> tuple[int, ...]:
            vals = np.round((np.asarray(p, dtype=float)[list(axes)] - mins_arr) / resolution).astype(int)
            vals = np.minimum(np.maximum(vals, 0), dims - 1)
            return tuple(int(v) for v in vals)

        def to_point(idx: tuple[int, ...]) -> np.ndarray:
            p = np.asarray(start, dtype=float).copy()
            coords = mins_arr + np.asarray(idx, dtype=float) * resolution
            for axis_i, axis in enumerate(axes):
                p[axis] = coords[axis_i]
            if planar:
                p[1] = start[1]
            return p

        start_idx = to_idx(start)
        goal_idx = to_idx(goal)
        blocked_cache: dict[tuple[int, ...], bool] = {}

        def blocked(idx: tuple[int, ...]) -> bool:
            if idx in {start_idx, goal_idx}:
                return False
            if idx not in blocked_cache:
                p = to_point(idx)
                blocked_cache[idx] = self._point_blocked(p, obstacles, radius + self.route_margin_m, planar)
            return blocked_cache[idx]

        if blocked(start_idx) or blocked(goal_idx):
            return [start.copy(), goal.copy()], "direct_endpoint_blocked"

        offsets = [
            offset
            for offset in np.ndindex(*(3 for _ in axes))
            if any(v != 1 for v in offset)
        ]
        offsets = [tuple(int(v - 1) for v in offset) for offset in offsets]
        open_heap: list[tuple[float, int, tuple[int, ...]]] = []
        heappush(open_heap, (0.0, 0, start_idx))
        came_from: dict[tuple[int, ...], tuple[int, ...]] = {}
        g_score = {start_idx: 0.0}
        closed: set[tuple[int, ...]] = set()
        counter = 0

        def heuristic(idx: tuple[int, ...]) -> float:
            return float(np.linalg.norm(to_point(idx)[list(axes)] - goal[list(axes)]))

        while open_heap:
            _, _, current = heappop(open_heap)
            if current in closed:
                continue
            if len(closed) >= self.max_astar_expansions:
                return [start.copy(), goal.copy()], "direct_astar_expansion_cap"
            if current == goal_idx:
                route = self._reconstruct_route(came_from, current, to_point, start, goal)
                return self._shortcut_route(route, obstacles, radius + self.route_margin_m, planar), "astar"
            closed.add(current)
            for offset in offsets:
                nxt = tuple(current[d] + offset[d] for d in range(len(axes)))
                if any(nxt[d] < 0 or nxt[d] >= dims_t[d] for d in range(len(axes))):
                    continue
                if blocked(nxt):
                    continue
                step = float(np.linalg.norm((np.asarray(nxt) - np.asarray(current)) * resolution))
                tentative = g_score[current] + step
                if tentative >= g_score.get(nxt, float("inf")):
                    continue
                came_from[nxt] = current
                g_score[nxt] = tentative
                counter += 1
                heappush(open_heap, (tentative + heuristic(nxt), counter, nxt))

        return [start.copy(), goal.copy()], "direct_astar_failed"

    @staticmethod
    def _route_bounds(
        start: np.ndarray,
        goal: np.ndarray,
        obstacles: list[AABBObs],
        world_bounds: dict[str, float],
        axes: tuple[int, ...],
        margin: float,
    ) -> tuple[list[float], list[float]]:
        axis_names = {0: ("xmin", "xmax"), 1: ("ymin", "ymax"), 2: ("zmin", "zmax")}
        mins = []
        maxs = []
        for axis in axes:
            lo_key, hi_key = axis_names[axis]
            if lo_key in world_bounds and hi_key in world_bounds:
                mins.append(float(world_bounds[lo_key]))
                maxs.append(float(world_bounds[hi_key]))
                continue
            values = [float(start[axis]), float(goal[axis])]
            for obstacle in obstacles:
                center = np.asarray(obstacle.center, dtype=float)
                half = np.asarray(obstacle.half, dtype=float)
                values.extend([float(center[axis] - half[axis]), float(center[axis] + half[axis])])
            mins.append(min(values) - margin)
            maxs.append(max(values) + margin)
        return mins, maxs

    @staticmethod
    def _point_blocked(p: np.ndarray, obstacles: list[AABBObs], margin: float, planar: bool) -> bool:
        for obstacle in obstacles:
            center = np.asarray(obstacle.center, dtype=float)
            half = np.asarray(obstacle.half, dtype=float) + float(margin)
            if planar:
                if abs(float(p[0] - center[0])) <= half[0] and abs(float(p[2] - center[2])) <= half[2]:
                    return True
            elif np.all(np.abs(p - center) <= half):
                return True
        return False

    @staticmethod
    def _segment_clear(p0: np.ndarray, p1: np.ndarray, obstacles: list[AABBObs], margin: float, planar: bool) -> bool:
        dist = float(np.linalg.norm(p1 - p0))
        samples = max(2, int(np.ceil(dist / max(0.5, margin))))
        for alpha in np.linspace(0.0, 1.0, samples):
            p = p0 * (1.0 - alpha) + p1 * alpha
            if CentralizedMpcOraclePlanner._point_blocked(p, obstacles, margin, planar):
                return False
        return True

    @staticmethod
    def _reconstruct_route(
        came_from: dict[tuple[int, ...], tuple[int, ...]],
        current: tuple[int, ...],
        to_point,
        start: np.ndarray,
        goal: np.ndarray,
    ) -> list[np.ndarray]:
        cells = [current]
        while current in came_from:
            current = came_from[current]
            cells.append(current)
        cells.reverse()
        route = [np.asarray(start, dtype=float)]
        route.extend(to_point(cell) for cell in cells[1:-1])
        route.append(np.asarray(goal, dtype=float))
        return route

    def _shortcut_route(
        self,
        route: list[np.ndarray],
        obstacles: list[AABBObs],
        margin: float,
        planar: bool,
    ) -> list[np.ndarray]:
        if len(route) <= 2:
            return route
        out = [route[0]]
        i = 0
        while i < len(route) - 1:
            j = len(route) - 1
            while j > i + 1 and not self._segment_clear(route[i], route[j], obstacles, margin, planar):
                j -= 1
            out.append(route[j])
            i = j
        return out

    @staticmethod
    def _route_target(pos: np.ndarray, route: list[np.ndarray], *, lookahead_m: float) -> np.ndarray:
        if len(route) <= 1:
            return np.asarray(route[-1], dtype=float)
        closest_i = 0
        closest_d = float("inf")
        for i, point in enumerate(route):
            d = float(np.linalg.norm(np.asarray(point, dtype=float) - pos))
            if d < closest_d:
                closest_i = i
                closest_d = d
        remaining = float(lookahead_m)
        current = pos.copy()
        for point in route[closest_i + 1 :]:
            point = np.asarray(point, dtype=float)
            seg = point - current
            seg_len = float(np.linalg.norm(seg))
            if seg_len >= remaining:
                return current + _normalize(seg) * remaining
            remaining -= seg_len
            current = point
        return np.asarray(route[-1], dtype=float)

    @staticmethod
    def _planning_order(
        states: list[AgentState],
        route_targets: dict[int, np.ndarray],
        v_cmds: list[np.ndarray],
    ) -> list[int]:
        entries = []
        for s in states:
            if s.done:
                continue
            target = route_targets.get(s.idx, s.goal)
            progress = float(np.dot(_normalize(target - s.pos), v_cmds[s.idx]))
            dist = float(np.linalg.norm(s.goal - s.pos))
            entries.append((-progress, dist, int(s.idx)))
        return [idx for _, _, idx in sorted(entries)]

    def _candidate_commands(
        self,
        s: AgentState,
        preferred: np.ndarray,
        route_target: np.ndarray,
        planar: bool,
    ) -> list[np.ndarray]:
        direction = _normalize(preferred)
        if np.linalg.norm(direction) < 1e-9:
            direction = _normalize(route_target - s.pos)
        if np.linalg.norm(direction) < 1e-9:
            direction = np.asarray([1.0, 0.0, 0.0], dtype=float)
        lateral = _lateral_axis(direction, planar=planar, agent_idx=s.idx)
        axes = [direction, lateral, -lateral]
        if not planar:
            up = np.asarray([0.0, 1.0, 0.0], dtype=float)
            side2 = _normalize(np.cross(direction, lateral))
            axes.extend([up, -up, side2, -side2])
        candidates: list[np.ndarray] = []
        for scale in (1.0, 0.8, 0.6, 0.4, 0.2):
            candidates.append(_clip_norm(direction * s.v_max * scale, s.v_max))
        candidates.append(np.zeros(3, dtype=float))
        for axis in axes[1:]:
            for mix in (0.35, 0.65, 1.0):
                cmd = _normalize(direction + axis * mix) * s.v_max * (0.65 if mix >= 0.65 else 0.85)
                candidates.append(_clip_norm(cmd, s.v_max))
        if planar:
            for cmd in candidates:
                cmd[1] = 0.0
        unique: list[np.ndarray] = []
        seen: set[tuple[float, float, float]] = set()
        for cmd in candidates:
            key = tuple(np.round(cmd, 3).tolist())
            if key in seen:
                continue
            seen.add(key)
            unique.append(cmd)
        return unique

    def _joint_candidate_cost(
        self,
        *,
        agent_id: int,
        states: list[AgentState],
        cmds: list[np.ndarray],
        preferred: list[np.ndarray],
        route_target: np.ndarray,
        obstacles: list[AABBObs],
        times: np.ndarray,
    ) -> tuple[float, float, float]:
        s = states[agent_id]
        cmd = cmds[agent_id]
        cost = self.route_tracking_weight * float(np.linalg.norm(cmd - preferred[agent_id]) ** 2)
        cost += self.smoothness_weight * float(np.linalg.norm(cmd - s.vel) ** 2)
        cost += self.stop_penalty_weight * max(0.0, float(s.v_max) - float(np.linalg.norm(cmd)))
        cost -= self.progress_weight * float(np.dot(cmd, _normalize(route_target - s.pos)))
        min_pair = self._min_pair_clearance_for_agent(agent_id, states, cmds, times)
        min_obstacle = self._min_obstacle_clearance_for_agent(agent_id, states, cmds, obstacles, times)
        required_pair = float(s.radius + self.mpc_safety_margin_m)
        if min_pair < required_pair:
            cost += self.collision_weight * (required_pair - min_pair) ** 2
        elif min_pair < required_pair + 2.0:
            cost += self.clearance_weight / max(0.1, min_pair - required_pair + 0.1)
        required_obstacle = float(s.radius + self.mpc_obstacle_margin_m)
        if min_obstacle < required_obstacle:
            cost += self.obstacle_mpc_weight * (required_obstacle - min_obstacle) ** 2
        return float(cost), float(min_pair), float(min_obstacle)

    @staticmethod
    def _min_pair_clearance_for_agent(
        agent_id: int,
        states: list[AgentState],
        cmds: list[np.ndarray],
        times: np.ndarray,
    ) -> float:
        s = states[agent_id]
        if s.done:
            return float("inf")
        min_clearance = float("inf")
        for j, other in enumerate(states):
            if j == agent_id or other.done:
                continue
            clearance, _ = CentralizedOraclePlanner._pair_clearance(s, other, cmds[agent_id], cmds[j], times)
            min_clearance = min(min_clearance, clearance)
        return min_clearance

    @staticmethod
    def _min_obstacle_clearance_for_agent(
        agent_id: int,
        states: list[AgentState],
        cmds: list[np.ndarray],
        obstacles: list[AABBObs],
        times: np.ndarray,
    ) -> float:
        s = states[agent_id]
        if s.done or not obstacles:
            return float("inf")
        min_clearance = float("inf")
        for obstacle in obstacles:
            for tau in times:
                p = np.asarray(s.pos, dtype=float) + np.asarray(cmds[agent_id], dtype=float) * float(tau)
                closest = _closest_point_on_aabb(p, obstacle)
                min_clearance = min(min_clearance, float(np.linalg.norm(p - closest) - s.radius))
        return min_clearance
