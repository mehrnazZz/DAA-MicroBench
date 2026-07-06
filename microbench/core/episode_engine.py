from __future__ import annotations

from dataclasses import dataclass
import inspect
import time
from pathlib import Path
from typing import Callable

import numpy as np

from microbench.comm.v2v import V2VEmulator
from microbench.config import deep_merge, load_comm_profiles, load_defaults
from microbench.core.collision import pairwise_stats
from microbench.core.dynamics import apply_dynamics
from microbench.core.neighbors import select_neighbors
from microbench.core.perception import fuse_observations, sense_neighbors
from microbench.planners import make_planner as default_make_planner
from microbench.scenarios import EventEngine, generate_spawns_goals, load_scenario
from microbench.types import (
    AABBObs,
    AgentContext,
    AgentMemory,
    AgentProfile,
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
    planner_debug: list[dict]
    perception_debug: list[dict]
    agent_failures: list[list[str]]
    message_events: list[dict]
    comm_stats: dict[str, int]

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
            "planner_debug": self.planner_debug,
            "perception_debug": self.perception_debug,
            "agent_failures": self.agent_failures,
            "message_events": self.message_events,
            "comm_stats": self.comm_stats,
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


def _copy_neighbor_obs(
    n: NeighborObs,
    *,
    msg_age_sec: float | None = None,
    valid: bool | None = None,
    source: str | None = None,
    track_age_sec: float | None = None,
    last_seen_s: float | None = None,
    stale: bool | None = None,
    occluded: bool | None = None,
) -> NeighborObs:
    return NeighborObs(
        idx=int(n.idx),
        pos=np.asarray(n.pos, dtype=float).copy(),
        vel=np.asarray(n.vel, dtype=float).copy(),
        radius=float(n.radius),
        msg_age_sec=float(n.msg_age_sec if msg_age_sec is None else msg_age_sec),
        valid=bool(n.valid if valid is None else valid),
        source=str(n.source if source is None else source),
        track_age_sec=float(n.track_age_sec if track_age_sec is None else track_age_sec),
        last_seen_s=n.last_seen_s if last_seen_s is None else float(last_seen_s),
        stale=bool(n.stale if stale is None else stale),
        occluded=bool(n.occluded if occluded is None else occluded),
    )


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _public_config(profile: AgentProfile) -> dict:
    return {
        "agent_id": int(profile.agent_id),
        "method": profile.method,
        "role": profile.role,
        "priority": int(profile.priority),
        "capabilities": _json_safe(profile.capabilities),
        "mission": _json_safe(profile.mission),
        "failure_modes": _json_safe(profile.failure_modes),
        "tags": list(profile.tags),
    }


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
        self.planner_guardrails_cfg = self.cfg.get("planner_guardrails", {})
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

        self.agent_profiles = [self._build_agent_profile(i) for i in range(self.n_agents)]
        self.agent_rngs = [np.random.default_rng(_agent_seed(seed + 17_171, i)) for i in range(self.n_agents)]
        self.command_delay_buffers: list[list[np.ndarray]] = [[] for _ in range(self.n_agents)]
        self.sensor_tracks: list[dict[int, NeighborObs]] = [dict() for _ in range(self.n_agents)]
        self.sensor_track_last_seen_s: list[dict[int, float]] = [dict() for _ in range(self.n_agents)]

        self.states = []
        for i, profile in enumerate(self.agent_profiles):
            caps = profile.capabilities
            self.states.append(
                AgentState(
                    idx=i,
                    pos=self.spawns[i].copy(),
                    vel=np.zeros(3, dtype=float),
                    goal=self.goals[i].copy(),
                    radius=float(caps.get("radius_m", self.radius)),
                    v_max=float(caps.get("v_max_mps", self.v_max)),
                    a_max=float(caps.get("a_max_mps2", self.a_max)),
                )
            )

        profile_methods = [p.method or method for p in self.agent_profiles]
        resolved_agent_methods = agent_methods
        if resolved_agent_methods is None and any(p.method for p in self.agent_profiles):
            resolved_agent_methods = profile_methods
        self.agent_methods, self.method_label = resolve_agent_methods(method, self.n_agents, resolved_agent_methods)
        make_planner = planner_factory or default_make_planner
        self.planners = [make_planner(agent_method) for agent_method in self.agent_methods]
        self.agent_contexts: list[AgentContext] = []
        for i, planner in enumerate(self.planners):
            seed_i = _agent_seed(seed, i)
            profile = self.agent_profiles[i]
            profile.method = self.agent_methods[i]
            config = _public_config(profile)
            self._reset_planner(planner, agent_id=i, seed=seed_i, config=config)
            self.agent_contexts.append(
                AgentContext(
                    agent_id=i,
                    method=self.agent_methods[i],
                    seed=seed_i,
                    memory=AgentMemory(),
                    role=profile.role,
                    priority=int(profile.priority),
                    capabilities=dict(profile.capabilities),
                    mission=dict(profile.mission),
                    failure_modes=dict(profile.failure_modes),
                    profile=profile,
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
        if self.comm_cfg.get("agent_messages") is not None:
            cprof["agent_messages"] = deep_merge(cprof.get("agent_messages", {}), self.comm_cfg.get("agent_messages", {}))
        if self.comm_cfg.get("message_bus") is not None:
            cprof["agent_messages"] = deep_merge(cprof.get("agent_messages", {}), self.comm_cfg.get("message_bus", {}))

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
        self.planner_timeout_count = 0
        self.planner_error_count = 0
        self.planner_fallback_count = 0
        self.planner_timeout_ms = float(self.planner_guardrails_cfg.get("timeout_ms", 100.0))
        self.planner_fallback_speed_scale = float(self.planner_guardrails_cfg.get("fallback_speed_scale", 0.5))
        self.k = 0
        self._closed = False

    def _build_agent_profile(self, agent_id: int) -> AgentProfile:
        agents_cfg = self.cfg.get("agents", {})
        default_cfg: dict = {}
        override_cfg: dict = {}

        if isinstance(agents_cfg, dict):
            default_cfg = agents_cfg.get("defaults", {}) or {}
            by_id = agents_cfg.get("by_id", agents_cfg.get("overrides", {})) or {}
            if isinstance(by_id, dict):
                override_cfg = by_id.get(agent_id, by_id.get(str(agent_id), {})) or {}
            profiles = agents_cfg.get("profiles", agents_cfg.get("list", [])) or []
            if isinstance(profiles, list):
                for idx, item in enumerate(profiles):
                    if not isinstance(item, dict):
                        continue
                    item_id = item.get("id", item.get("agent_id", idx))
                    if int(item_id) == int(agent_id):
                        override_cfg = deep_merge(override_cfg, item)
                        break
        elif isinstance(agents_cfg, list):
            if agent_id < len(agents_cfg) and isinstance(agents_cfg[agent_id], dict):
                override_cfg = agents_cfg[agent_id]
            for idx, item in enumerate(agents_cfg):
                if not isinstance(item, dict):
                    continue
                item_id = item.get("id", item.get("agent_id", idx))
                if int(item_id) == int(agent_id):
                    override_cfg = deep_merge(override_cfg, item)
                    break

        merged = deep_merge(default_cfg, override_cfg)
        capabilities = dict(merged.get("capabilities", {}) or {})
        for key in (
            "radius_m",
            "v_max_mps",
            "a_max_mps2",
            "sensor_range_m",
            "sensor_fov_deg",
            "sensor_false_negative_p",
            "sensor_noise_sigma_pos_m",
            "sensor_noise_sigma_vel_mps",
            "sensor_occlusion",
            "sensor_occlusion_margin_m",
            "sensor_track_ttl_s",
            "comm_range_m",
        ):
            if key in merged:
                capabilities.setdefault(key, merged[key])

        failure_modes = merged.get("failure_modes", merged.get("failures", {})) or {}
        mission = merged.get("mission", {}) or {}
        tags = merged.get("tags", []) or []
        if isinstance(tags, str):
            tags = [tags]

        return AgentProfile(
            agent_id=int(agent_id),
            method=merged.get("method"),
            role=merged.get("role"),
            priority=int(merged.get("priority", merged.get("mission_priority", agent_id))),
            capabilities=capabilities,
            mission=dict(mission),
            failure_modes=dict(failure_modes),
            tags=[str(t) for t in tags],
        )

    @staticmethod
    def _reset_planner(planner, *, agent_id: int, seed: int, config: dict) -> None:
        reset = getattr(planner, "reset", None)
        if reset is None:
            return
        try:
            sig = inspect.signature(reset)
        except (TypeError, ValueError):
            reset(seed)
            return

        params = list(sig.parameters.values())
        names = {p.name for p in params}
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
        if accepts_kwargs or {"agent_id", "seed", "config"} & names:
            kwargs = {}
            if accepts_kwargs or "agent_id" in names:
                kwargs["agent_id"] = int(agent_id)
            if accepts_kwargs or "seed" in names:
                kwargs["seed"] = int(seed)
            if accepts_kwargs or "config" in names:
                kwargs["config"] = dict(config)
            try:
                reset(**kwargs)
                return
            except TypeError:
                pass

        positional = [
            p
            for p in params
            if p.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
        ]
        if len(positional) >= 3:
            reset(agent_id, seed, config)
        elif len(positional) >= 2:
            reset(seed, config)
        else:
            reset(seed)

    @staticmethod
    def _finalize_planner(planner, *, context: AgentContext, config: dict) -> None:
        finalize = getattr(planner, "finalize", None)
        if finalize is None:
            return
        try:
            sig = inspect.signature(finalize)
        except (TypeError, ValueError):
            finalize()
            return

        params = list(sig.parameters.values())
        names = {p.name for p in params}
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
        if accepts_kwargs or {"agent_context", "context", "config"} & names:
            kwargs = {}
            if accepts_kwargs or "agent_context" in names:
                kwargs["agent_context"] = context
            if accepts_kwargs or "context" in names:
                kwargs["context"] = context
            if accepts_kwargs or "config" in names:
                kwargs["config"] = dict(config)
            try:
                finalize(**kwargs)
                return
            except TypeError:
                pass

        positional = [
            p
            for p in params
            if p.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
        ]
        if len(positional) >= 2:
            finalize(context, config)
        elif len(positional) == 1:
            finalize(context)
        else:
            finalize()

    def _perception_config_for_agent(self, agent_id: int) -> dict:
        cfg = deep_merge({}, self.perception_cfg)
        caps = self.agent_profiles[agent_id].capabilities

        perception_caps = caps.get("perception")
        if isinstance(perception_caps, dict):
            cfg = deep_merge(cfg, perception_caps)

        sensor_override: dict = {}
        sensor_caps = caps.get("sensor")
        if isinstance(sensor_caps, dict):
            sensor_override = deep_merge(sensor_override, sensor_caps)

        cap_to_sensor_key = {
            "sensor_range_m": "range_m",
            "sensor_fov_deg": "fov_deg",
            "sensor_false_negative_p": "false_negative_p",
            "sensor_noise_sigma_pos_m": "noise_sigma_pos_m",
            "sensor_noise_sigma_vel_mps": "noise_sigma_vel_mps",
            "sensor_occlusion": "occlusion",
            "sensor_occlusion_margin_m": "occlusion_margin_m",
            "sensor_track_ttl_s": "track_ttl_s",
        }
        for cap_key, sensor_key in cap_to_sensor_key.items():
            if cap_key in caps:
                sensor_override[sensor_key] = caps[cap_key]

        if sensor_override:
            cfg["sensor"] = deep_merge(cfg.get("sensor", {}), sensor_override)
        return cfg

    @staticmethod
    def _sensor_track_ttl_s(perception_cfg: dict) -> float:
        sensor_cfg = perception_cfg.get("sensor", perception_cfg)
        value = sensor_cfg.get(
            "track_ttl_s",
            sensor_cfg.get("stale_track_ttl_s", perception_cfg.get("track_ttl_s", 0.0)),
        )
        return max(0.0, float(value))

    def _sensor_observations_with_tracks(
        self,
        *,
        agent_id: int,
        t: float,
        detected_obs: list[NeighborObs],
        perception_cfg: dict,
    ) -> list[NeighborObs]:
        ttl_s = self._sensor_track_ttl_s(perception_cfg)
        tracks = self.sensor_tracks[agent_id]
        last_seen = self.sensor_track_last_seen_s[agent_id]

        current: dict[int, NeighborObs] = {}
        for obs in detected_obs:
            live = _copy_neighbor_obs(
                obs,
                msg_age_sec=0.0,
                track_age_sec=0.0,
                last_seen_s=t,
                stale=False,
                source="sensor",
            )
            current[live.idx] = live
            tracks[live.idx] = _copy_neighbor_obs(live)
            last_seen[live.idx] = float(t)

        if ttl_s <= 0.0:
            return [current[k] for k in sorted(current)]

        expired: list[int] = []
        for idx, cached in list(tracks.items()):
            if idx in current:
                continue
            seen_s = last_seen.get(idx)
            if seen_s is None:
                expired.append(idx)
                continue
            age_s = max(0.0, float(t) - float(seen_s))
            if age_s <= ttl_s + 1e-12:
                current[idx] = _copy_neighbor_obs(
                    cached,
                    msg_age_sec=age_s,
                    track_age_sec=age_s,
                    last_seen_s=seen_s,
                    stale=True,
                    valid=True,
                    source="sensor",
                )
            else:
                expired.append(idx)

        for idx in expired:
            tracks.pop(idx, None)
            last_seen.pop(idx, None)

        return [current[k] for k in sorted(current)]

    @staticmethod
    def _neighbor_obs_trace(n: NeighborObs) -> dict:
        return {
            "idx": int(n.idx),
            "msg_age_sec": float(n.msg_age_sec),
            "valid": bool(n.valid),
            "source": str(n.source),
            "radius": float(n.radius),
            "pos": np.asarray(n.pos, dtype=float).tolist(),
            "vel": np.asarray(n.vel, dtype=float).tolist(),
            "track_age_sec": float(n.track_age_sec),
            "last_seen_s": float(n.last_seen_s) if n.last_seen_s is not None else None,
            "stale": bool(n.stale),
            "occluded": bool(n.occluded),
        }

    def _planner_fallback_cmd(self, planner_input: PlannerInput) -> np.ndarray:
        ego = planner_input.ego
        away = np.zeros(3, dtype=float)
        p_i = np.asarray(ego.pos, dtype=float)
        for obs in planner_input.neighbors:
            rel = p_i - np.asarray(obs.pos, dtype=float)
            dist = max(1e-6, float(np.linalg.norm(rel)))
            away += rel / (dist * dist)
        for obs in planner_input.obstacles:
            center = np.asarray(obs.center, dtype=float)
            half = np.asarray(obs.half, dtype=float)
            closest = np.minimum(np.maximum(p_i, center - half), center + half)
            rel = p_i - closest
            dist = max(1e-6, float(np.linalg.norm(rel)))
            away += rel / (dist * dist)
        if planner_input.planar:
            away[1] = 0.0
        if np.linalg.norm(away) < 1e-9:
            return np.zeros(3, dtype=float)
        return _normalize(away) * float(ego.v_max) * min(1.0, max(0.0, self.planner_fallback_speed_scale))

    @staticmethod
    def _coerce_planner_cmd(v_cmd) -> np.ndarray:
        arr = np.asarray(v_cmd, dtype=float)
        if arr.shape != (3,):
            raise ValueError(f"planner v_cmd must have shape (3,), got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("planner v_cmd must contain only finite values")
        return arr

    def _record_planner_fallback(
        self,
        *,
        agent_id: int,
        planner_input: PlannerInput,
        planner_debug: list[dict],
        reason: str,
        elapsed_ms: float,
        error: Exception | None = None,
    ) -> np.ndarray:
        self.planner_fallback_count += 1
        if reason == "timeout":
            self.planner_timeout_count += 1
        elif reason in {"error", "invalid_output"}:
            self.planner_error_count += 1
        debug = {
            "engine_guardrail": reason,
            "planner_elapsed_ms": float(elapsed_ms),
            "planner_timeout_ms": float(self.planner_timeout_ms),
            "fallback_cmd": "away_from_risk",
        }
        if error is not None:
            debug["error_type"] = type(error).__name__
            debug["error"] = str(error)
        planner_debug[agent_id] = debug
        return self._planner_fallback_cmd(planner_input)

    def close(self) -> None:
        if self._closed:
            return
        for planner, context, profile in zip(self.planners, self.agent_contexts, self.agent_profiles):
            self._finalize_planner(planner, context=context, config=_public_config(profile))
        self._closed = True

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
        planner_debug: list[dict] = [{} for _ in self.states]
        perception_debug: list[dict] = [{} for _ in self.states]
        agent_failures: list[list[str]] = [[] for _ in self.states]
        command_delay_steps = [0 for _ in self.states]

        for i, s in enumerate(self.states):
            active_failures, delay_steps = self._resolve_agent_failures(i, t)
            agent_failures[i] = active_failures
            command_delay_steps[i] = delay_steps
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
            if self._has_failure(active_failures, "comm_dropout", "communication_dropout", "noncooperative"):
                v2v_obs = []

            agent_perception_cfg = self._perception_config_for_agent(i)
            perception_mode = str(agent_perception_cfg.get("mode", "v2v")).lower()
            sensor_obs: list[NeighborObs] = []
            sensor_detections: list[NeighborObs] = []
            if perception_mode in {"sensor", "fused"}:
                sensor_detections = sense_neighbors(
                    ego=s,
                    states=self.states,
                    goal_dir=goal_dir,
                    obstacles=self.obstacles,
                    perception_cfg=agent_perception_cfg,
                    planar=self.planar,
                    rng=self.rng,
                )
            if self._has_failure(active_failures, "sensor_dropout", "perception_dropout"):
                sensor_detections = []
            if perception_mode in {"sensor", "fused"}:
                sensor_obs = self._sensor_observations_with_tracks(
                    agent_id=i,
                    t=t,
                    detected_obs=sensor_detections,
                    perception_cfg=agent_perception_cfg,
                )
            if perception_mode == "sensor":
                all_obs = sensor_obs
            elif perception_mode == "fused":
                all_obs = fuse_observations(v2v_obs, sensor_obs)
            else:
                all_obs = v2v_obs
            if self._has_failure(active_failures, "observation_dropout"):
                all_obs = []

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
            selected_obs[i] = [self._neighbor_obs_trace(n) for n in selected]
            perception_debug[i] = {
                "mode": perception_mode,
                "v2v_obs": int(len(v2v_obs)),
                "sensor_detections": int(len(sensor_detections)),
                "sensor_tracks": int(len(sensor_obs)),
                "sensor_stale_tracks": int(sum(1 for obs in sensor_obs if obs.stale)),
                "candidate_obs": int(len(all_obs)),
                "selected_obs": int(len(selected)),
                "track_ttl_s": float(self._sensor_track_ttl_s(agent_perception_cfg))
                if perception_mode in {"sensor", "fused"}
                else 0.0,
            }

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

            raw_agent_messages = self.v2v.drain_agent_messages(i, t)
            agent_messages = [] if self._has_failure(active_failures, "comm_dropout", "communication_dropout") else raw_agent_messages
            selected_messages[i] = [
                {
                    "sender_id": int(m.sender_id),
                    "recipient_id": m.recipient_id,
                    "kind": str(m.kind),
                    "msg_age_s": float(m.msg_age_s),
                    "valid": bool(m.valid),
                    "ttl_s": float(m.ttl_s),
                    "payload": dict(m.payload),
                    "message_id": m.message_id,
                    "correlation_id": m.correlation_id,
                    "seq": m.seq,
                    "channel": str(m.channel),
                    "priority": int(m.priority),
                    "size_bytes": int(m.size_bytes),
                }
                for m in agent_messages
            ]

            if self._has_failure(active_failures, "frozen", "frozen_planner"):
                planner_debug[i] = {"engine_failure": "frozen_planner"}
                continue

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
            try:
                planner_out = self.planners[i].compute_cmd(p_input)
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - c0) * 1000.0
                self.planner_ms_samples.append(elapsed_ms)
                v_cmds[i] = self._record_planner_fallback(
                    agent_id=i,
                    planner_input=p_input,
                    planner_debug=planner_debug,
                    reason="error",
                    elapsed_ms=elapsed_ms,
                    error=exc,
                )
                continue

            elapsed_ms = (time.perf_counter() - c0) * 1000.0
            self.planner_ms_samples.append(elapsed_ms)
            if self.planner_timeout_ms >= 0.0 and elapsed_ms > self.planner_timeout_ms:
                v_cmds[i] = self._record_planner_fallback(
                    agent_id=i,
                    planner_input=p_input,
                    planner_debug=planner_debug,
                    reason="timeout",
                    elapsed_ms=elapsed_ms,
                )
                continue

            if isinstance(planner_out, PlannerOutput):
                raw_v_cmd = planner_out.v_cmd
                intent_out = planner_out.intent_out
                messages_out = list(planner_out.messages_out or [])
                debug_info = _json_safe(planner_out.debug_info)
            else:
                raw_v_cmd = planner_out
                intent_out = None
                messages_out = []
                debug_info = {}

            try:
                v_cmds[i] = self._coerce_planner_cmd(raw_v_cmd)
            except (TypeError, ValueError) as exc:
                v_cmds[i] = self._record_planner_fallback(
                    agent_id=i,
                    planner_input=p_input,
                    planner_debug=planner_debug,
                    reason="invalid_output",
                    elapsed_ms=elapsed_ms,
                    error=exc,
                )
                continue

            pending_intent_out[i] = intent_out
            pending_messages_out[i] = messages_out
            planner_debug[i] = debug_info

        self._publish_intents(t, pending_intent_out, agent_failures)
        self._publish_messages(t, pending_messages_out, agent_failures)
        message_events = self.v2v.drain_agent_message_events()
        comm_stats = self.v2v.agent_message_stats_snapshot()

        v_cmds = self.events.apply_overrides(t, self.states, v_cmds)
        v_cmds = self._apply_command_failures(v_cmds, agent_failures, command_delay_steps)
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
            planner_debug=planner_debug,
            perception_debug=perception_debug,
            agent_failures=agent_failures,
            message_events=message_events,
            comm_stats=comm_stats,
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

    @staticmethod
    def _has_failure(active_failures: list[str], *names: str) -> bool:
        active = {str(name) for name in active_failures}
        return any(name in active for name in names)

    def _mode_active(self, cfg, t: float, rng: np.random.Generator) -> bool:
        if isinstance(cfg, bool):
            return bool(cfg)
        if isinstance(cfg, (int, float)):
            return float(cfg) != 0.0
        if not isinstance(cfg, dict):
            return False

        if not bool(cfg.get("enabled", True)):
            return False
        start_s = float(cfg.get("start_s", cfg.get("t_start_s", 0.0)))
        end_s = cfg.get("end_s")
        if end_s is None and cfg.get("duration_s") is not None:
            end_s = start_s + float(cfg.get("duration_s", 0.0))
        end = float(end_s) if end_s is not None else float("inf")
        if not (start_s <= t <= end):
            return False

        p = float(cfg.get("p", cfg.get("probability", cfg.get("drop_probability", 1.0))))
        p = max(0.0, min(1.0, p))
        if p >= 1.0:
            return True
        if p <= 0.0:
            return False
        return bool(rng.random() < p)

    def _active_command_delay_steps(self, agent_id: int, t: float) -> int:
        modes = self.agent_profiles[agent_id].failure_modes
        rng = self.agent_rngs[agent_id]
        for name in ("actuation_delay", "command_delay", "actuation_delay_steps", "command_delay_steps"):
            if name not in modes:
                continue
            cfg = modes[name]
            if isinstance(cfg, (int, float)):
                return max(0, int(cfg))
            if isinstance(cfg, dict) and self._mode_active(cfg, t, rng):
                return max(0, int(cfg.get("steps", cfg.get("delay_steps", cfg.get("value", 0)))))
        return 0

    def _resolve_agent_failures(self, agent_id: int, t: float) -> tuple[list[str], int]:
        modes = self.agent_profiles[agent_id].failure_modes
        rng = self.agent_rngs[agent_id]
        active: list[str] = []
        delay_names = {"actuation_delay", "command_delay", "actuation_delay_steps", "command_delay_steps"}
        for name, cfg in modes.items():
            if str(name) in delay_names:
                continue
            if self._mode_active(cfg, t, rng):
                active.append(str(name))

        delay_steps = self._active_command_delay_steps(agent_id, t)
        if delay_steps > 0:
            active.append("actuation_delay")
        return active, delay_steps

    def _apply_command_failures(
        self,
        v_cmds: list[np.ndarray],
        agent_failures: list[list[str]],
        command_delay_steps: list[int],
    ) -> list[np.ndarray]:
        out = [np.asarray(v, dtype=float).copy() for v in v_cmds]
        for i, cmd in enumerate(out):
            delay_steps = int(command_delay_steps[i])
            if delay_steps > 0:
                buf = self.command_delay_buffers[i]
                buf.append(cmd.copy())
                if len(buf) <= delay_steps:
                    cmd = np.zeros(3, dtype=float)
                else:
                    cmd = buf.pop(0)
                while len(buf) > delay_steps:
                    buf.pop(0)
            else:
                self.command_delay_buffers[i].clear()

            if self._has_failure(agent_failures[i], "frozen", "frozen_planner"):
                cmd = np.zeros(3, dtype=float)
            out[i] = cmd
        return out

    def _publish_intents(
        self,
        t: float,
        pending_intent_out: list[IntentMsg | None],
        agent_failures: list[list[str]],
    ) -> None:
        if not self.intent_enabled:
            return
        for i, out_msg in enumerate(pending_intent_out):
            if self.states[i].done or out_msg is None:
                continue
            if self._has_failure(agent_failures[i], "comm_dropout", "communication_dropout", "noncooperative"):
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

    def _publish_messages(
        self,
        t: float,
        pending_messages_out: list[list],
        agent_failures: list[list[str]],
    ) -> None:
        for i, out_msgs in enumerate(pending_messages_out):
            if self.states[i].done:
                continue
            if self._has_failure(agent_failures[i], "comm_dropout", "communication_dropout", "noncooperative"):
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
