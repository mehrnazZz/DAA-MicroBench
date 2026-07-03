from __future__ import annotations

from dataclasses import dataclass
import time
from pathlib import Path
from typing import Callable

import numpy as np

from microbench.comm.v2v import V2VEmulator
from microbench.config import load_comm_profiles, load_defaults
from microbench.core.collision import pairwise_stats
from microbench.core.dynamics import apply_dynamics
from microbench.core.neighbors import select_neighbors
from microbench.core.perception import fuse_observations, sense_neighbors
from microbench.planners import make_planner as default_make_planner
from microbench.scenarios import EventEngine, generate_spawns_goals, load_scenario
from microbench.types import (
    AABBObs,
    AgentContext,
    AgentState,
    IntentMsg,
    IntentObs,
    NeighborObs,
    PlannerInput,
    PlannerOutput,
)


@dataclass
class EpisodeStep:
    k: int
    t: float
    states: list[AgentState]
    planner_states: list[AgentState]
    v_cmds: list[np.ndarray]
    goal_dirs: list[np.ndarray]
    selected_neighbors: list[list[int]]
    selected_neighbor_obs: list[list[NeighborObs]]
    selected_obs: list[list[dict]]
    selected_intents: list[list[dict]]
    selected_messages: list[list[dict]]
    msg_age_matrix: np.ndarray
    speed_saturated: list[bool]
    accel_saturated: list[bool]
    pos: np.ndarray
    vel: np.ndarray
    radii: np.ndarray
    done: np.ndarray
    active_for_sampling: np.ndarray
    collisions: int
    near_misses: int
    min_sep: float
    collision_pairs: set[tuple[int, int]]
    near_miss_pairs: set[tuple[int, int]]

    def trace_frame(self) -> dict:
        return {
            "t": float(self.t),
            "n_agents": len(self.states),
            "positions": self.pos.tolist(),
            "velocities": self.vel.tolist(),
            "v_cmd": [v.tolist() for v in self.v_cmds],
            "goal_dirs": [g.tolist() for g in self.goal_dirs],
            "selected_neighbors": self.selected_neighbors,
            "selected_obs": self.selected_obs,
            "selected_intents": self.selected_intents,
            "selected_messages": self.selected_messages,
            "speed_saturated": [bool(x) for x in self.speed_saturated],
            "accel_saturated": [bool(x) for x in self.accel_saturated],
        }


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.zeros(3, dtype=float)
    return v / n


def _agent_seed(base_seed: int, agent_idx: int) -> int:
    return int((int(base_seed) * 1_000_003 + int(agent_idx)) % (2**32 - 1))


def resolve_agent_methods(method: str, n_agents: int, agent_methods: list[str] | None) -> tuple[list[str], str]:
    if not agent_methods:
        return [method for _ in range(n_agents)], method

    cleaned = [m.strip() for m in agent_methods if m.strip()]
    if not cleaned:
        return [method for _ in range(n_agents)], method
    if len(cleaned) == 1:
        return [cleaned[0] for _ in range(n_agents)], cleaned[0]
    if len(cleaned) != n_agents:
        raise ValueError(
            f"agent_methods must contain either 1 method or exactly N={n_agents} methods; got {len(cleaned)}"
        )
    return cleaned, "mixed[" + "+".join(cleaned) + "]"


def _in_aabb(pos: np.ndarray, center: np.ndarray, half: np.ndarray, radius: float) -> bool:
    d = np.abs(pos - center)
    return bool(np.all(d <= (half + radius)))


def _saturation_flags(vel: np.ndarray, v_cmd: np.ndarray, v_max: float, a_max: float, dt: float) -> tuple[bool, bool]:
    speed_sat = bool(np.linalg.norm(v_cmd) > v_max + 1e-9)
    if np.linalg.norm(v_cmd) > v_max + 1e-12:
        v_cmd_eff = v_cmd / np.linalg.norm(v_cmd) * v_max
    else:
        v_cmd_eff = v_cmd
    dv = v_cmd_eff - vel
    accel_sat = bool(np.linalg.norm(dv) > (a_max * dt) + 1e-9)
    return speed_sat, accel_sat


def _copy_state(s: AgentState) -> AgentState:
    return AgentState(
        idx=int(s.idx),
        pos=s.pos.copy(),
        vel=s.vel.copy(),
        goal=s.goal.copy(),
        radius=float(s.radius),
        v_max=float(s.v_max),
        a_max=float(s.a_max),
        done=bool(s.done),
        done_time_s=float(s.done_time_s) if s.done_time_s is not None else None,
        path_length_m=float(s.path_length_m),
    )


class EpisodeEngine:
    """Shared closed-loop simulation engine for benchmark, trace, and dataset runs."""

    def __init__(
        self,
        *,
        scenario_path: str,
        method: str,
        n_agents: int,
        seed: int,
        comm_profile: str,
        agent_methods: list[str] | None = None,
        planner_factory: Callable[[str], object] | None = None,
    ):
        self.scenario_path = scenario_path
        self.scenario_stem = Path(scenario_path).stem
        self.method = method
        self.n_agents = int(n_agents)
        self.seed = int(seed)
        self.comm_profile = comm_profile

        self.defaults = load_defaults()
        self.profiles = load_comm_profiles()
        if comm_profile not in self.profiles:
            raise ValueError(f"Unknown comm profile: {comm_profile}")

        self.cfg = load_scenario(self.defaults, scenario_path)
        self.rng = np.random.default_rng(seed)

        self.sim_cfg = self.cfg.get("sim", {})
        self.world_cfg = self.cfg.get("world", {})
        self.agent_cfg = self.cfg.get("agent_params", {})
        self.dyn_cfg = self.cfg.get("dynamics", {})
        self.neighbor_cfg = self.cfg.get("neighbors", {})
        self.perception_cfg = self.cfg.get("perception", {})
        self.comm_cfg = self.cfg.get("comm", {})
        self.intent_cfg = self.cfg.get("intent", {})
        self.events_cfg = self.cfg.get("events", [])
        self.obstacles = self.cfg.get("obstacles", [])

        self.dt = float(self.sim_cfg.get("dt_s", 0.02))
        self.duration_s = float(self.cfg.get("scenario", {}).get("duration_s", self.sim_cfg.get("duration_s", 60.0)))
        self.steps = int(round(self.duration_s / self.dt))
        self.planar = bool(self.world_cfg.get("planar", self.sim_cfg.get("planar", True)))
        self.fixed_y = float(self.world_cfg.get("fixed_y_m", self.sim_cfg.get("fixed_y_m", 0.0)))
        self.goal_tol = float(self.agent_cfg.get("goal_tolerance_m", self.sim_cfg.get("goal_tolerance_m", 1.0)))
        self.goal_hold_time_s = float(self.sim_cfg.get("goal_hold_time_s", 0.5))
        self.near_margin = float(self.sim_cfg.get("near_miss_margin_m", 0.5))

        self.v_max = float(self.agent_cfg.get("v_max_mps", self.dyn_cfg.get("v_max_mps", 3.0)))
        self.a_max = float(self.agent_cfg.get("a_max_mps2", self.dyn_cfg.get("a_max_mps2", 2.0)))
        self.radius = float(self.agent_cfg.get("radius_m", 0.5))
        self.age_cap_s = float(self.comm_cfg.get("age_cap_s", 0.75))

        self.spawns, self.goals = generate_spawns_goals(self.cfg, self.n_agents, self.rng)
        if self.planar:
            self.spawns[:, 1] = self.fixed_y
            self.goals[:, 1] = self.fixed_y

        self.states = [
            AgentState(
                idx=i,
                pos=self.spawns[i].copy(),
                vel=np.zeros(3, dtype=float),
                goal=self.goals[i].copy(),
                radius=self.radius,
                v_max=self.v_max,
                a_max=self.a_max,
            )
            for i in range(self.n_agents)
        ]

        self.agent_methods, self.method_label = resolve_agent_methods(method, self.n_agents, agent_methods)
        make_planner = planner_factory or default_make_planner
        self.planners = [make_planner(agent_method) for agent_method in self.agent_methods]
        self.agent_contexts: list[AgentContext] = []
        for i, planner in enumerate(self.planners):
            seed_i = _agent_seed(seed, i)
            planner.reset(seed_i)
            self.agent_contexts.append(
                AgentContext(
                    agent_id=i,
                    method=self.agent_methods[i],
                    seed=seed_i,
                    priority=i,
                )
            )

        cprof = self.profiles[comm_profile].copy()
        if self.comm_cfg.get("noise_sigma_pos_m") is not None:
            cprof.setdefault("noise", {})["sigma_pos_m"] = float(
                self.comm_cfg.get("noise_sigma_pos_m", cprof.get("noise", {}).get("sigma_pos_m", 0.0))
            )
        if self.comm_cfg.get("noise_sigma_vel_mps") is not None:
            cprof.setdefault("noise", {})["sigma_vel_mps"] = float(
                self.comm_cfg.get("noise_sigma_vel_mps", cprof.get("noise", {}).get("sigma_vel_mps", 0.0))
            )

        self.intent_enabled = bool(self.intent_cfg.get("enabled", False))
        self.intent_tx_rate_hz = float(self.intent_cfg.get("tx_rate_hz", cprof.get("tx_rate_hz", 10.0)))
        self.intent_max_points = int(self.intent_cfg.get("max_points", 12))
        self.intent_age_cap_s = float(self.intent_cfg.get("age_cap_s", self.age_cap_s))
        self.v2v = V2VEmulator(
            cprof,
            age_cap_s=self.age_cap_s,
            rng=self.rng,
            intent_cfg={
                "enabled": self.intent_enabled,
                "tx_rate_hz": self.intent_tx_rate_hz,
                "age_cap_s": self.intent_age_cap_s,
            },
        )
        self.v2v.reset(self.n_agents)

        self.events = EventEngine(self.events_cfg, self.rng)
        self.events.reset()

        self.planner_obstacles = [
            AABBObs(
                center=np.asarray(ob["aabb"].get("center", [0.0, 0.0, 0.0]), dtype=float),
                half=np.asarray(ob["aabb"].get("half", [0.0, 0.0, 0.0]), dtype=float),
            )
            for ob in self.obstacles
            if "aabb" in ob
        ]

        self.done_times = np.full(self.n_agents, np.inf, dtype=float)
        self.goal_hold_elapsed = np.zeros(self.n_agents, dtype=float)
        self.spawn_goal_dists = np.linalg.norm(self.goals - self.spawns, axis=1)
        self.planner_ms_samples: list[float] = []
        self.k = 0

    def done(self) -> bool:
        return self.k >= self.steps or bool(np.all([s.done for s in self.states]))

    def step(self) -> EpisodeStep | None:
        if self.k >= self.steps:
            return None

        k = self.k
        t = k * self.dt
        self._update_goal_completion(t)
        self.v2v.step(t, self.states)
        planner_states = [_copy_state(s) for s in self.states]

        v_cmds: list[np.ndarray] = [np.zeros(3, dtype=float) for _ in self.states]
        goal_dirs = [np.zeros(3, dtype=float) for _ in self.states]
        selected_neighbors: list[list[int]] = [[] for _ in self.states]
        selected_neighbor_obs: list[list[NeighborObs]] = [[] for _ in self.states]
        selected_obs: list[list[dict]] = [[] for _ in self.states]
        selected_intents: list[list[dict]] = [[] for _ in self.states]
        selected_messages: list[list[dict]] = [[] for _ in self.states]
        msg_age_matrix = np.full((self.n_agents, self.n_agents), self.age_cap_s, dtype=float)
        pending_intent_out: list[IntentMsg | None] = [None for _ in self.states]
        pending_messages_out: list[list] = [[] for _ in self.states]
        active_for_sampling = np.zeros(self.n_agents, dtype=bool)

        for i, s in enumerate(self.states):
            if s.done:
                continue

            goal_delta = s.goal - s.pos
            goal_dist = float(np.linalg.norm(goal_delta))
            goal_dir = _normalize(goal_delta)
            goal_dirs[i] = goal_dir
            active_for_sampling[i] = goal_dist >= self.goal_tol

            v2v_obs: list[NeighborObs] = []
            for j in range(self.n_agents):
                if j == i:
                    continue
                m = self.v2v.get_last(i, j)
                valid, age = self.v2v.message_age(t, m)
                msg_age_matrix[i, j] = age
                if m is not None and valid:
                    v2v_obs.append(
                        NeighborObs(
                            idx=j,
                            pos=m.pos.copy(),
                            vel=m.vel.copy(),
                            radius=m.radius,
                            msg_age_sec=age,
                            valid=True,
                            source="v2v",
                        )
                    )

            perception_mode = str(self.perception_cfg.get("mode", "v2v")).lower()
            sensor_obs: list[NeighborObs] = []
            if perception_mode in {"sensor", "fused"}:
                sensor_obs = sense_neighbors(
                    ego=s,
                    states=self.states,
                    goal_dir=goal_dir,
                    obstacles=self.obstacles,
                    perception_cfg=self.perception_cfg,
                    planar=self.planar,
                    rng=self.rng,
                )
            if perception_mode == "sensor":
                all_obs = sensor_obs
            elif perception_mode == "fused":
                all_obs = fuse_observations(v2v_obs, sensor_obs)
            else:
                all_obs = v2v_obs

            selected = select_neighbors(
                ego_idx=i,
                ego_pos=s.pos,
                ego_vel=s.vel,
                obs=all_obs,
                range_m=float(self.neighbor_cfg.get("range_m", 30.0)),
                top_k=int(self.neighbor_cfg.get("top_k", 8)),
                threat_metric=str(self.neighbor_cfg.get("threat_metric", "ttc")),
                ttc_horizon_s=float(self.neighbor_cfg.get("ttc_horizon_s", 6.0)),
            )

            selected_neighbors[i] = [n.idx for n in selected]
            selected_neighbor_obs[i] = selected
            selected_obs[i] = [
                {
                    "idx": n.idx,
                    "msg_age_sec": float(n.msg_age_sec),
                    "valid": bool(n.valid),
                    "source": str(n.source),
                    "radius": float(n.radius),
                    "pos": n.pos.tolist(),
                    "vel": n.vel.tolist(),
                }
                for n in selected
            ]

            selected_intent_obs: list[IntentObs] = []
            for n in selected:
                im = self.v2v.get_last_intent(i, n.idx)
                ivalid, iage = self.v2v.intent_status(t, im)
                if im is not None:
                    points = np.asarray(im.points, dtype=float).copy()
                    tube_radius_m = float(im.tube_radius_m)
                    kind = str(im.kind)
                    expiry_s = float(im.expiry_s)
                    dt_plan_s = float(im.dt_plan_s) if im.dt_plan_s is not None else None
                    mode = im.mode
                else:
                    points = np.zeros((0, 3), dtype=float)
                    tube_radius_m = 0.0
                    kind = ""
                    expiry_s = float(t)
                    dt_plan_s = None
                    mode = None
                selected_intent_obs.append(
                    IntentObs(
                        sender_id=n.idx,
                        points=points,
                        tube_radius_m=tube_radius_m,
                        kind=kind,
                        expiry_s=expiry_s,
                        intent_age_s=float(iage),
                        valid=bool(ivalid and im is not None),
                        dt_plan_s=dt_plan_s,
                        mode=mode,
                    )
                )
                selected_intents[i].append(
                    {
                        "idx": n.idx,
                        "valid": bool(ivalid and im is not None),
                        "intent_age_s": float(iage),
                        "kind": kind,
                        "expiry_s": expiry_s,
                        "tube_radius_m": tube_radius_m,
                        "points": points.tolist(),
                    }
                )

            agent_messages = self.v2v.drain_agent_messages(i, t)
            selected_messages[i] = [
                {
                    "sender_id": int(m.sender_id),
                    "recipient_id": m.recipient_id,
                    "kind": str(m.kind),
                    "msg_age_s": float(m.msg_age_s),
                    "valid": bool(m.valid),
                    "ttl_s": float(m.ttl_s),
                    "payload": dict(m.payload),
                }
                for m in agent_messages
            ]

            p_input = PlannerInput(
                ego=s,
                goal_dir=goal_dir,
                neighbors=selected,
                dt=self.dt,
                t=t,
                obstacles=self.planner_obstacles,
                neighbor_intents=selected_intent_obs,
                messages=agent_messages,
                agent_context=self.agent_contexts[i],
                planar=self.planar,
            )
            c0 = time.perf_counter()
            planner_out = self.planners[i].compute_cmd(p_input)
            c1 = time.perf_counter()
            self.planner_ms_samples.append((c1 - c0) * 1000.0)
            if isinstance(planner_out, PlannerOutput):
                v_cmds[i] = np.asarray(planner_out.v_cmd, dtype=float)
                pending_intent_out[i] = planner_out.intent_out
                pending_messages_out[i] = list(planner_out.messages_out or [])
            else:
                v_cmds[i] = np.asarray(planner_out, dtype=float)

        self._publish_intents(t, pending_intent_out)
        self._publish_messages(t, pending_messages_out)

        v_cmds = self.events.apply_overrides(t, self.states, v_cmds)
        speed_sat, accel_sat = self._apply_motion(v_cmds)

        pos = np.array([s.pos for s in self.states], dtype=float)
        vel = np.array([s.vel for s in self.states], dtype=float)
        radii = np.array([s.radius for s in self.states], dtype=float)
        done = np.array([s.done for s in self.states], dtype=bool)
        collisions, near_misses, min_sep = pairwise_stats(pos, radii, self.near_margin)
        collision_pairs_step, near_miss_pairs_step = self._proximity_pairs(pos, radii)
        post_states = [_copy_state(s) for s in self.states]

        self.k += 1
        return EpisodeStep(
            k=k,
            t=t,
            states=post_states,
            planner_states=planner_states,
            v_cmds=v_cmds,
            goal_dirs=goal_dirs,
            selected_neighbors=selected_neighbors,
            selected_neighbor_obs=selected_neighbor_obs,
            selected_obs=selected_obs,
            selected_intents=selected_intents,
            selected_messages=selected_messages,
            msg_age_matrix=msg_age_matrix,
            speed_saturated=speed_sat,
            accel_saturated=accel_sat,
            pos=pos,
            vel=vel,
            radii=radii,
            done=done,
            active_for_sampling=active_for_sampling,
            collisions=collisions,
            near_misses=near_misses,
            min_sep=min_sep,
            collision_pairs=collision_pairs_step,
            near_miss_pairs=near_miss_pairs_step,
        )

    def _update_goal_completion(self, t: float) -> None:
        for s in self.states:
            if s.done:
                continue
            if np.linalg.norm(s.goal - s.pos) <= self.goal_tol:
                self.goal_hold_elapsed[s.idx] += self.dt
                if self.goal_hold_elapsed[s.idx] + 1e-12 >= self.goal_hold_time_s:
                    s.done = True
                    s.done_time_s = t
                    self.done_times[s.idx] = t
                    s.vel = np.zeros(3, dtype=float)
            else:
                self.goal_hold_elapsed[s.idx] = 0.0

    def _publish_intents(self, t: float, pending_intent_out: list[IntentMsg | None]) -> None:
        if not self.intent_enabled:
            return
        for i, out_msg in enumerate(pending_intent_out):
            if self.states[i].done or out_msg is None:
                continue
            points = np.asarray(out_msg.points, dtype=float)
            if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 1:
                continue
            self.v2v.publish_intent(
                sender=i,
                intent=IntentMsg(
                    sender_id=i,
                    timestamp_send_s=float(out_msg.timestamp_send_s),
                    expiry_s=float(out_msg.expiry_s),
                    kind=str(out_msg.kind),
                    tube_radius_m=float(out_msg.tube_radius_m),
                    points=points,
                    dt_plan_s=float(out_msg.dt_plan_s) if out_msg.dt_plan_s is not None else None,
                    mode=out_msg.mode,
                ),
                now_s=t,
                max_points=self.intent_max_points,
            )

    def _publish_messages(self, t: float, pending_messages_out: list[list]) -> None:
        for i, out_msgs in enumerate(pending_messages_out):
            if self.states[i].done:
                continue
            for out_msg in out_msgs:
                self.v2v.publish_agent_message(
                    sender=i,
                    msg=out_msg,
                    now_s=t,
                    n_agents=self.n_agents,
                )

    def _apply_motion(self, v_cmds: list[np.ndarray]) -> tuple[list[bool], list[bool]]:
        speed_sat = [False for _ in self.states]
        accel_sat = [False for _ in self.states]
        for i, s in enumerate(self.states):
            if s.done:
                continue
            speed_sat[i], accel_sat[i] = _saturation_flags(s.vel, v_cmds[i], s.v_max, s.a_max, self.dt)
            p_next, v_next = apply_dynamics(s.pos, s.vel, v_cmds[i], s.v_max, s.a_max, self.dt)
            if self.planar:
                p_next[1] = self.fixed_y
                v_next[1] = 0.0

            blocked = False
            for ob in self.obstacles:
                if "aabb" not in ob:
                    continue
                aabb = ob["aabb"]
                center = np.asarray(aabb.get("center", [0.0, 0.0, 0.0]), dtype=float)
                half = np.asarray(aabb.get("half", [0.0, 0.0, 0.0]), dtype=float)
                if _in_aabb(p_next, center, half, s.radius):
                    blocked = True
                    break

            if blocked:
                s.vel = np.zeros(3, dtype=float)
            else:
                s.path_length_m += float(np.linalg.norm(p_next - s.pos))
                s.pos = p_next
                s.vel = v_next
        return speed_sat, accel_sat

    def _proximity_pairs(self, pos: np.ndarray, radii: np.ndarray) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
        collision_pairs_step: set[tuple[int, int]] = set()
        near_miss_pairs_step: set[tuple[int, int]] = set()
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                dist = float(np.linalg.norm(pos[i] - pos[j]))
                collision_threshold = float(radii[i] + radii[j])
                near_threshold = collision_threshold + self.near_margin
                if dist < collision_threshold:
                    collision_pairs_step.add((i, j))
                elif dist < near_threshold:
                    near_miss_pairs_step.add((i, j))
        return collision_pairs_step, near_miss_pairs_step
