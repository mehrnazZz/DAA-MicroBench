from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import glob
import json
import math
import hashlib
import numpy as np

from microbench.types import AABBObs, AgentContext, AgentState, NeighborObs, PlannerInput, PlannerOutput
from microbench.config import load_defaults, load_comm_profiles
from microbench.scenarios import load_scenario, generate_spawns_goals, EventEngine
from microbench.planners import make_planner
from microbench.comm.v2v import V2VEmulator
from microbench.core import apply_dynamics, select_neighbors
from microbench.core.perception import fuse_observations, sense_neighbors


@dataclass
class DatasetGenSpec:
    scenario_path: str
    method: str
    n_agents: int
    seed: int
    comm_profile: str
    dt_plan_s: float
    T: int
    goal_dist_cap: float
    scenario_id: int
    comm_profile_id: int
    quality_filter: str = "none"
    filter_min_sep_m: float = 0.0


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.zeros_like(v)
    return v / n


def _stable_id(name: str) -> int:
    h = hashlib.md5(name.encode("utf-8")).hexdigest()[:8]
    return int(h, 16)


def _agent_seed(base_seed: int, agent_idx: int) -> int:
    return int((int(base_seed) * 1_000_003 + int(agent_idx)) % (2**32 - 1))


def _in_aabb(pos: np.ndarray, center: np.ndarray, half: np.ndarray, radius: float) -> bool:
    d = np.abs(pos - center)
    return bool(np.all(d <= (half + radius)))


def _frame_angle(goal_dir: np.ndarray, ego_vel: np.ndarray) -> float:
    gx, gz = float(goal_dir[0]), float(goal_dir[2])
    if math.hypot(gx, gz) > 1e-9:
        return math.atan2(gz, gx)
    vx, vz = float(ego_vel[0]), float(ego_vel[2])
    if math.hypot(vx, vz) > 1e-9:
        return math.atan2(vz, vx)
    return 0.0


def _rot_goal_frame(vec: np.ndarray, angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    x, y, z = float(vec[0]), float(vec[1]), float(vec[2])
    # Rotate world vector by -angle around Y to align goal-dir to +X.
    return np.asarray([c * x + s * z, y, -s * x + c * z], dtype=np.float32)


def _event_features(events_cfg: list[dict], t: float, duration_s: float) -> np.ndarray:
    weather_active = 0.0
    for ev in events_cfg:
        if not ev.get("enabled", True):
            continue
        if ev.get("type") != "weather_maneuver":
            continue
        t0 = float(ev.get("t_start_s", 0.0))
        dur = float(ev.get("duration_s", 0.0))
        if t0 <= t <= t0 + dur:
            weather_active = 1.0
            break
    time_remaining = max(0.0, min(1.0, (duration_s - t) / max(1e-6, duration_s)))
    return np.asarray([weather_active, time_remaining], dtype=np.float32)


def _encode_cond_structured(
    ego: AgentState,
    goal_dir_world: np.ndarray,
    neighbors: list[NeighborObs],
    top_k: int,
    range_m: float,
    age_cap_s: float,
    r_ref: float,
    goal_dist_cap: float,
    evt_feat: np.ndarray,
    v_ref: float,
    a_ref: float,
    planar: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if planar:
        angle = _frame_angle(goal_dir_world, ego.vel)
        ego_vel_f = _rot_goal_frame(ego.vel, angle)
        goal_dir_f = _rot_goal_frame(goal_dir_world, angle)
    else:
        angle = 0.0
        ego_vel_f = np.asarray(ego.vel, dtype=np.float32)
        goal_dir_f = np.asarray(goal_dir_world, dtype=np.float32)

    v_max_ego = max(1e-6, float(ego.v_max))
    goal_dist = float(np.linalg.norm(ego.goal - ego.pos))

    cond_ego = np.asarray(
        [
            float(ego_vel_f[0] / v_max_ego),
            float(ego_vel_f[1] / v_max_ego),
            float(ego_vel_f[2] / v_max_ego),
            float(ego.radius / max(1e-6, r_ref)),
            float(ego.v_max / max(1e-6, v_ref)),
            float(ego.a_max / max(1e-6, a_ref)),
        ],
        dtype=np.float32,
    )

    cond_goal = np.asarray(
        [
            float(goal_dir_f[0]),
            float(goal_dir_f[1]),
            float(goal_dir_f[2]),
            float(min(max(goal_dist / max(1e-6, goal_dist_cap), 0.0), 1.0)),
        ],
        dtype=np.float32,
    )

    cond_nbh = np.zeros((top_k, 9), dtype=np.float32)
    for i in range(min(top_k, len(neighbors))):
        n = neighbors[i]
        rel_pos = n.pos - ego.pos
        rel_vel = n.vel - ego.vel
        if planar:
            rel_pos_f = _rot_goal_frame(rel_pos, angle)
            rel_vel_f = _rot_goal_frame(rel_vel, angle)
        else:
            rel_pos_f = np.asarray(rel_pos, dtype=np.float32)
            rel_vel_f = np.asarray(rel_vel, dtype=np.float32)
        cond_nbh[i, 0:3] = rel_pos_f / max(1e-6, range_m)
        cond_nbh[i, 3:6] = rel_vel_f / v_max_ego
        cond_nbh[i, 6] = float(n.radius / max(1e-6, r_ref))
        cond_nbh[i, 7] = float(min(max(n.msg_age_sec / max(1e-6, age_cap_s), 0.0), 1.0))
        cond_nbh[i, 8] = 1.0 if n.valid else 0.0

    return cond_ego, cond_goal, cond_nbh, evt_feat.astype(np.float32)


def _episode_collect(spec: DatasetGenSpec) -> dict[str, np.ndarray]:
    defaults = load_defaults()
    profiles = load_comm_profiles()
    if spec.comm_profile not in profiles:
        raise ValueError(f"Unknown comm profile: {spec.comm_profile}")

    cfg = load_scenario(defaults, spec.scenario_path)
    rng = np.random.default_rng(spec.seed)

    sim_cfg = cfg.get("sim", {})
    world_cfg = cfg.get("world", {})
    agent_cfg = cfg.get("agent_params", {})
    dyn_cfg = cfg.get("dynamics", {})
    ncfg = cfg.get("neighbors", {})
    perception_cfg = cfg.get("perception", {})
    comm_cfg = cfg.get("comm", {})

    dt = float(sim_cfg.get("dt_s", 0.02))
    stride = int(round(spec.dt_plan_s / dt))
    if stride < 1:
        raise ValueError("dt_plan_s must be >= sim dt")
    dt_plan_eff = stride * dt

    duration_s = float(cfg.get("scenario", {}).get("duration_s", sim_cfg.get("duration_s", 60.0)))
    steps = int(round(duration_s / dt))
    top_k = int(ncfg.get("top_k", 8))

    planar = bool(world_cfg.get("planar", sim_cfg.get("planar", True)))
    fixed_y = float(world_cfg.get("fixed_y_m", sim_cfg.get("fixed_y_m", 0.0)))
    goal_tol = float(agent_cfg.get("goal_tolerance_m", sim_cfg.get("goal_tolerance_m", 1.0)))

    v_max = float(agent_cfg.get("v_max_mps", dyn_cfg.get("v_max_mps", 3.0)))
    a_max = float(agent_cfg.get("a_max_mps2", dyn_cfg.get("a_max_mps2", 2.0)))
    radius = float(agent_cfg.get("radius_m", 0.5))

    v_ref = float(defaults.get("dynamics", {}).get("v_max_mps", 3.0))
    a_ref = float(defaults.get("dynamics", {}).get("a_max_mps2", 2.0))
    r_ref = float(agent_cfg.get("radius_m", radius))
    range_m = float(ncfg.get("range_m", 30.0))

    spawns, goals = generate_spawns_goals(cfg, spec.n_agents, rng)
    if planar:
        spawns[:, 1] = fixed_y
        goals[:, 1] = fixed_y

    states: list[AgentState] = []
    for i in range(spec.n_agents):
        states.append(
            AgentState(
                idx=i,
                pos=spawns[i].copy(),
                vel=np.zeros(3, dtype=np.float32),
                goal=goals[i].copy(),
                radius=radius,
                v_max=v_max,
                a_max=a_max,
            )
        )

    planners = [make_planner(spec.method) for _ in range(spec.n_agents)]
    agent_contexts: list[AgentContext] = []
    for i, planner in enumerate(planners):
        seed_i = _agent_seed(spec.seed, i)
        planner.reset(seed_i)
        agent_contexts.append(
            AgentContext(
                agent_id=i,
                method=spec.method,
                seed=seed_i,
                priority=i,
            )
        )

    cprof = profiles[spec.comm_profile].copy()
    if comm_cfg.get("noise_sigma_pos_m") is not None:
        cprof.setdefault("noise", {})["sigma_pos_m"] = float(comm_cfg.get("noise_sigma_pos_m", cprof.get("noise", {}).get("sigma_pos_m", 0.0)))
    if comm_cfg.get("noise_sigma_vel_mps") is not None:
        cprof.setdefault("noise", {})["sigma_vel_mps"] = float(comm_cfg.get("noise_sigma_vel_mps", cprof.get("noise", {}).get("sigma_vel_mps", 0.0)))
    age_cap_s = float(comm_cfg.get("age_cap_s", 0.75))

    v2v = V2VEmulator(cprof, age_cap_s=age_cap_s, rng=rng)
    v2v.reset(spec.n_agents)

    events_cfg = cfg.get("events", [])
    events = EventEngine(events_cfg, rng)
    events.reset()

    obstacles = cfg.get("obstacles", [])
    planner_obstacles = [
        AABBObs(
            center=np.asarray(ob["aabb"].get("center", [0.0, 0.0, 0.0]), dtype=float),
            half=np.asarray(ob["aabb"].get("half", [0.0, 0.0, 0.0]), dtype=float),
        )
        for ob in obstacles
        if "aabb" in ob
    ]

    cond_ego_steps: list[np.ndarray] = []
    cond_goal_steps: list[np.ndarray] = []
    cond_nbh_steps: list[np.ndarray] = []
    cond_evt_steps: list[np.ndarray] = []
    cmd_steps: list[np.ndarray] = []
    active_steps: list[np.ndarray] = []
    pos_steps: list[np.ndarray] = []
    rad_steps: list[np.ndarray] = []

    for k_tick in range(steps):
        t = k_tick * dt

        for s in states:
            if s.done:
                continue
            if np.linalg.norm(s.goal - s.pos) <= goal_tol:
                s.done = True
                s.vel = np.zeros(3, dtype=np.float32)

        v2v.step(t, states)

        v_cmds: list[np.ndarray] = [np.zeros(3, dtype=np.float32) for _ in states]
        pending_messages_out: list[list] = [[] for _ in states]
        ego_batch = np.zeros((spec.n_agents, 6), dtype=np.float32)
        goal_batch = np.zeros((spec.n_agents, 4), dtype=np.float32)
        nbh_batch = np.zeros((spec.n_agents, top_k, 9), dtype=np.float32)
        evt_batch = np.zeros((spec.n_agents, 2), dtype=np.float32)
        active_batch = np.zeros(spec.n_agents, dtype=bool)

        evt_feat = _event_features(events_cfg, t, duration_s)

        for i, s in enumerate(states):
            evt_batch[i] = evt_feat
            if s.done:
                continue
            goal_dir = _normalize(s.goal - s.pos).astype(np.float32)
            goal_dist = float(np.linalg.norm(s.goal - s.pos))
            if goal_dist < goal_tol:
                continue
            active_batch[i] = True

            v2v_obs: list[NeighborObs] = []
            for j in range(spec.n_agents):
                if j == i:
                    continue
                m = v2v.get_last(i, j)
                valid, age = v2v.message_age(t, m)
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

            perception_mode = str(perception_cfg.get("mode", "v2v")).lower()
            sensor_obs: list[NeighborObs] = []
            if perception_mode in {"sensor", "fused"}:
                sensor_obs = sense_neighbors(
                    ego=s,
                    states=states,
                    goal_dir=goal_dir,
                    obstacles=obstacles,
                    perception_cfg=perception_cfg,
                    planar=planar,
                    rng=rng,
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
                range_m=range_m,
                top_k=top_k,
                threat_metric=str(ncfg.get("threat_metric", "ttc")),
                ttc_horizon_s=float(ncfg.get("ttc_horizon_s", 6.0)),
            )

            cond_ego, cond_goal, cond_nbh, cond_evt = _encode_cond_structured(
                ego=s,
                goal_dir_world=goal_dir,
                neighbors=selected,
                top_k=top_k,
                range_m=range_m,
                age_cap_s=age_cap_s,
                r_ref=r_ref,
                goal_dist_cap=spec.goal_dist_cap,
                evt_feat=evt_feat,
                v_ref=v_ref,
                a_ref=a_ref,
                planar=planar,
            )
            ego_batch[i] = cond_ego
            goal_batch[i] = cond_goal
            nbh_batch[i] = cond_nbh
            evt_batch[i] = cond_evt

            p_input = PlannerInput(
                ego=s,
                goal_dir=goal_dir,
                neighbors=selected,
                dt=dt,
                t=t,
                obstacles=planner_obstacles,
                messages=v2v.drain_agent_messages(i, t),
                agent_context=agent_contexts[i],
                planar=planar,
            )
            pout = planners[i].compute_cmd(p_input)
            if isinstance(pout, PlannerOutput):
                v_cmds[i] = np.asarray(pout.v_cmd, dtype=np.float32)
                pending_messages_out[i] = list(pout.messages_out or [])
            else:
                v_cmds[i] = np.asarray(pout, dtype=np.float32)

        for i, out_msgs in enumerate(pending_messages_out):
            if states[i].done:
                continue
            for out_msg in out_msgs:
                v2v.publish_agent_message(sender=i, msg=out_msg, now_s=t, n_agents=spec.n_agents)

        v_cmds = events.apply_overrides(t, states, v_cmds)

        for i, s in enumerate(states):
            if s.done:
                continue
            p_next, v_next = apply_dynamics(s.pos, s.vel, v_cmds[i], s.v_max, s.a_max, dt)
            if planar:
                p_next[1] = fixed_y
                v_next[1] = 0.0

            blocked = False
            for ob in obstacles:
                if "aabb" not in ob:
                    continue
                aabb = ob["aabb"]
                center = np.asarray(aabb.get("center", [0.0, 0.0, 0.0]), dtype=float)
                half = np.asarray(aabb.get("half", [0.0, 0.0, 0.0]), dtype=float)
                if _in_aabb(p_next, center, half, s.radius):
                    blocked = True
                    break

            if blocked:
                s.vel = np.zeros(3, dtype=np.float32)
            else:
                s.pos = p_next
                s.vel = v_next

        cond_ego_steps.append(ego_batch)
        cond_goal_steps.append(goal_batch)
        cond_nbh_steps.append(nbh_batch)
        cond_evt_steps.append(evt_batch)
        cmd_steps.append(np.stack(v_cmds, axis=0).astype(np.float32))
        active_steps.append(active_batch)
        pos_steps.append(np.stack([s.pos for s in states], axis=0).astype(np.float32))
        rad_steps.append(np.asarray([s.radius for s in states], dtype=np.float32))

        if all(s.done for s in states):
            break

    cond_ego_arr = np.stack(cond_ego_steps, axis=0)
    cond_goal_arr = np.stack(cond_goal_steps, axis=0)
    cond_nbh_arr = np.stack(cond_nbh_steps, axis=0)
    cond_evt_arr = np.stack(cond_evt_steps, axis=0)
    cmd_arr = np.stack(cmd_steps, axis=0)
    active_arr = np.stack(active_steps, axis=0)
    pos_arr = np.stack(pos_steps, axis=0)
    rad_arr = np.stack(rad_steps, axis=0)

    S = cmd_arr.shape[0]
    last_start = S - (spec.T - 1) * stride

    out_cond_ego = []
    out_cond_goal = []
    out_cond_nbh = []
    out_cond_evt = []
    out_U0_raw = []
    out_scenario_id = []
    out_comm_profile = []
    out_comm_profile_id = []
    out_N_agents = []
    out_seed = []
    out_t_sec = []
    out_ego_id = []
    out_min_sep = []
    out_collision = []
    out_vmax = []
    out_amax = []

    for t0 in range(max(0, last_start)):
        sample_idx = [t0 + m * stride for m in range(spec.T)]
        horizon_end = sample_idx[-1]
        for ego_id in range(spec.n_agents):
            if not bool(active_arr[t0, ego_id]):
                continue

            U0_raw = cmd_arr[sample_idx, ego_id, :]
            pos_h = pos_arr[t0 : horizon_end + 1, ego_id, :]
            rad_h = rad_arr[t0 : horizon_end + 1, ego_id]

            min_sep = float("inf")
            collision = 0
            for tt in range(pos_h.shape[0]):
                p = pos_h[tt]
                r = float(rad_h[tt])
                others = pos_arr[t0 + tt]
                other_r = rad_arr[t0 + tt]
                d = np.linalg.norm(others - p[None, :], axis=1)
                d[ego_id] = np.inf
                sep = d - (r + other_r)
                this_min = float(np.min(sep))
                if this_min < min_sep:
                    min_sep = this_min
                if np.any(sep < 0.0):
                    collision = 1

            out_cond_ego.append(cond_ego_arr[t0, ego_id])
            out_cond_goal.append(cond_goal_arr[t0, ego_id])
            out_cond_nbh.append(cond_nbh_arr[t0, ego_id])
            out_cond_evt.append(cond_evt_arr[t0, ego_id])
            out_U0_raw.append(U0_raw)
            out_scenario_id.append(int(spec.scenario_id))
            out_comm_profile.append(spec.comm_profile)
            out_comm_profile_id.append(int(spec.comm_profile_id))
            out_N_agents.append(spec.n_agents)
            out_seed.append(spec.seed)
            out_t_sec.append(float(t0 * dt))
            out_ego_id.append(ego_id)
            out_min_sep.append(min_sep)
            out_collision.append(collision)
            out_vmax.append(float(states[ego_id].v_max))
            out_amax.append(float(states[ego_id].a_max))

    if not out_cond_ego:
        return {
            "cond_ego": np.zeros((0, 6), dtype=np.float32),
            "cond_goal": np.zeros((0, 4), dtype=np.float32),
            "cond_nbh": np.zeros((0, top_k, 9), dtype=np.float32),
            "cond_evt": np.zeros((0, 2), dtype=np.float32),
            "cond_flat": np.zeros((0, 6 + 4 + top_k * 9 + 2), dtype=np.float32),
            "U0_raw": np.zeros((0, spec.T, 3), dtype=np.float32),
            "U0": np.zeros((0, spec.T, 3), dtype=np.float32),
            "scenario_id": np.zeros((0,), dtype=np.int64),
            "comm_profile": np.zeros((0,), dtype="U1"),
            "comm_profile_id": np.zeros((0,), dtype=np.int64),
            "N_agents": np.zeros((0,), dtype=np.int32),
            "seed": np.zeros((0,), dtype=np.int32),
            "t_sec": np.zeros((0,), dtype=np.float32),
            "ego_id": np.zeros((0,), dtype=np.int32),
            "min_sep_next_H": np.zeros((0,), dtype=np.float32),
            "collision_in_next_H": np.zeros((0,), dtype=np.int8),
            "v_max_ego": np.zeros((0,), dtype=np.float32),
            "a_max_ego": np.zeros((0,), dtype=np.float32),
            "dt_plan_s": np.asarray(dt_plan_eff, dtype=np.float32),
            "T": np.asarray(spec.T, dtype=np.int32),
            "k": np.asarray(top_k, dtype=np.int32),
            "norm_R_sense": np.asarray(range_m, dtype=np.float32),
            "norm_age_cap_s": np.asarray(age_cap_s, dtype=np.float32),
            "norm_goal_dist_cap": np.asarray(spec.goal_dist_cap, dtype=np.float32),
            "norm_r_ref": np.asarray(r_ref, dtype=np.float32),
            "norm_v_ref": np.asarray(v_ref, dtype=np.float32),
            "norm_a_ref": np.asarray(a_ref, dtype=np.float32),
            "sim_dt_s": np.asarray(dt, dtype=np.float32),
        }

    cond_ego = np.stack(out_cond_ego, axis=0).astype(np.float32)
    cond_goal = np.stack(out_cond_goal, axis=0).astype(np.float32)
    cond_nbh = np.stack(out_cond_nbh, axis=0).astype(np.float32)
    cond_evt = np.stack(out_cond_evt, axis=0).astype(np.float32)
    U0_raw = np.stack(out_U0_raw, axis=0).astype(np.float32)
    v_max_ego = np.asarray(out_vmax, dtype=np.float32)
    U0 = U0_raw / np.maximum(v_max_ego[:, None, None], 1e-6)
    U0 = np.clip(U0, -1.0, 1.0)

    cond_flat = np.concatenate(
        [
            cond_ego,
            cond_goal,
            cond_nbh.reshape(cond_nbh.shape[0], -1),
            cond_evt,
        ],
        axis=1,
    ).astype(np.float32)

    rec = {
        "cond_ego": cond_ego,
        "cond_goal": cond_goal,
        "cond_nbh": cond_nbh,
        "cond_evt": cond_evt,
        "cond_flat": cond_flat,
        "U0_raw": U0_raw,
        "U0": U0,
        "scenario_id": np.asarray(out_scenario_id, dtype=np.int64),
        "comm_profile": np.asarray(out_comm_profile, dtype="U64"),
        "comm_profile_id": np.asarray(out_comm_profile_id, dtype=np.int64),
        "N_agents": np.asarray(out_N_agents, dtype=np.int32),
        "seed": np.asarray(out_seed, dtype=np.int32),
        "t_sec": np.asarray(out_t_sec, dtype=np.float32),
        "ego_id": np.asarray(out_ego_id, dtype=np.int32),
        "min_sep_next_H": np.asarray(out_min_sep, dtype=np.float32),
        "collision_in_next_H": np.asarray(out_collision, dtype=np.int8),
        "v_max_ego": v_max_ego,
        "a_max_ego": np.asarray(out_amax, dtype=np.float32),
        "dt_plan_s": np.asarray(dt_plan_eff, dtype=np.float32),
        "T": np.asarray(spec.T, dtype=np.int32),
        "k": np.asarray(top_k, dtype=np.int32),
        "norm_R_sense": np.asarray(range_m, dtype=np.float32),
        "norm_age_cap_s": np.asarray(age_cap_s, dtype=np.float32),
        "norm_goal_dist_cap": np.asarray(spec.goal_dist_cap, dtype=np.float32),
        "norm_r_ref": np.asarray(r_ref, dtype=np.float32),
        "norm_v_ref": np.asarray(v_ref, dtype=np.float32),
        "norm_a_ref": np.asarray(a_ref, dtype=np.float32),
        "sim_dt_s": np.asarray(dt, dtype=np.float32),
    }
    if spec.quality_filter != "none":
        min_sep = rec["min_sep_next_H"]
        collision = rec["collision_in_next_H"]
        keep = np.ones(min_sep.shape[0], dtype=bool)
        if spec.quality_filter in {"safe_expert", "collision_free"}:
            keep &= collision == 0
        if spec.quality_filter == "safe_expert":
            keep &= min_sep >= float(spec.filter_min_sep_m)
        rec["num_samples_raw"] = np.asarray(int(min_sep.shape[0]), dtype=np.int64)
        rec["num_samples_kept"] = np.asarray(int(np.sum(keep)), dtype=np.int64)
        for k, v in list(rec.items()):
            if np.isscalar(v) or (isinstance(v, np.ndarray) and v.ndim == 0):
                continue
            rec[k] = v[keep]
    else:
        n0 = int(rec["cond_ego"].shape[0])
        rec["num_samples_raw"] = np.asarray(n0, dtype=np.int64)
        rec["num_samples_kept"] = np.asarray(n0, dtype=np.int64)
    return rec


def _concat_records(records: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = records[0].keys()
    out: dict[str, np.ndarray] = {}
    for k in keys:
        v0 = records[0][k]
        if np.isscalar(v0) or (isinstance(v0, np.ndarray) and v0.ndim == 0):
            out[k] = np.asarray(v0)
            continue
        out[k] = np.concatenate([r[k] for r in records], axis=0)
    return out


def _iter_chunks(n: int, chunk: int) -> list[tuple[int, int]]:
    out = []
    i = 0
    while i < n:
        j = min(n, i + chunk)
        out.append((i, j))
        i = j
    return out


def generate_dataset(
    scenarios: list[str],
    method: str,
    n_agents_list: list[int],
    seeds: list[int],
    T: int,
    dt_plan_s: float,
    out_dir: str,
    comm_profiles: list[str],
    shard_size: int = 50000,
    goal_dist_cap: float = 60.0,
    quality_filter: str = "none",
    filter_min_sep_m: float = 0.0,
) -> list[Path]:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)

    shard_paths: list[Path] = []

    for scenario_path in scenarios:
        scenario_name = Path(scenario_path).stem
        scenario_id = _stable_id(scenario_name)
        scenario_cfg = load_scenario(load_defaults(), scenario_path)
        scenario_world = scenario_cfg.get("world", {})
        scenario_sim = scenario_cfg.get("sim", {})
        scenario_planar = bool(scenario_world.get("planar", scenario_sim.get("planar", True)))
        for comm in comm_profiles:
            comm_profile_id = _stable_id(comm)
            combo_records: list[dict[str, np.ndarray]] = []
            for n_agents in n_agents_list:
                for seed in seeds:
                    spec = DatasetGenSpec(
                        scenario_path=scenario_path,
                        method=method,
                        n_agents=n_agents,
                        seed=seed,
                        comm_profile=comm,
                        dt_plan_s=dt_plan_s,
                        T=T,
                        goal_dist_cap=goal_dist_cap,
                        scenario_id=scenario_id,
                        comm_profile_id=comm_profile_id,
                        quality_filter=quality_filter,
                        filter_min_sep_m=filter_min_sep_m,
                    )
                    rec = _episode_collect(spec)
                    if rec["cond_ego"].shape[0] == 0:
                        continue
                    combo_records.append(rec)

            combo_dir = root / method / scenario_name / comm
            combo_dir.mkdir(parents=True, exist_ok=True)

            if not combo_records:
                (combo_dir / "dataset_manifest.json").write_text(
                    json.dumps(
                        {
                            "num_shards": 0,
                            "method": method,
                            "scenario": scenario_name,
                            "scenario_path": str(Path(scenario_path)),
                            "planar": scenario_planar,
                            "scenario_id": scenario_id,
                            "comm_profile": comm,
                            "comm_profile_id": comm_profile_id,
                            "quality_filter": quality_filter,
                            "filter_min_sep_m": filter_min_sep_m,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                continue

            merged = _concat_records(combo_records)
            B = int(merged["cond_ego"].shape[0])
            shard_ranges = _iter_chunks(B, shard_size)
            local_paths: list[str] = []

            for shard_idx, (a, b) in enumerate(shard_ranges):
                shard_path = combo_dir / f"shard_{shard_idx:05d}.npz"
                payload = {}
                for k, v in merged.items():
                    if np.isscalar(v) or (isinstance(v, np.ndarray) and v.ndim == 0):
                        payload[k] = v
                    else:
                        payload[k] = v[a:b]
                np.savez_compressed(shard_path, **payload)
                shard_paths.append(shard_path)
                local_paths.append(str(shard_path))

            manifest = {
                "num_samples": B,
                "num_shards": len(local_paths),
                "shards": local_paths,
                "method": method,
                "scenario": scenario_name,
                "scenario_path": str(Path(scenario_path)),
                "planar": scenario_planar,
                "scenario_id": scenario_id,
                "comm_profile": comm,
                "comm_profile_id": comm_profile_id,
                "quality_filter": quality_filter,
                "filter_min_sep_m": filter_min_sep_m,
                "num_samples_raw": int(np.sum([int(r["num_samples_raw"]) for r in combo_records])),
                "num_samples_kept": int(np.sum([int(r["num_samples_kept"]) for r in combo_records])),
                "dt_plan_s": float(merged["dt_plan_s"]),
                "T": int(merged["T"]),
                "k": int(merged["k"]),
                "cond_shapes": {
                    "cond_ego": [6],
                    "cond_goal": [4],
                    "cond_nbh": [int(merged["k"]), 9],
                    "cond_evt": [2],
                    "cond_flat": [int(merged["cond_flat"].shape[1])],
                },
                "normalization": {
                    "R_sense": float(merged["norm_R_sense"]),
                    "age_cap_s": float(merged["norm_age_cap_s"]),
                    "goal_dist_cap": float(merged["norm_goal_dist_cap"]),
                    "r_ref": float(merged["norm_r_ref"]),
                    "v_ref": float(merged["norm_v_ref"]),
                    "a_ref": float(merged["norm_a_ref"]),
                },
            }
            (combo_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return shard_paths


def expand_scenarios(spec: str) -> list[str]:
    out: list[str] = []
    for token in [s.strip() for s in spec.split(",") if s.strip()]:
        matches = sorted(glob.glob(token))
        if matches:
            out.extend(matches)
        else:
            out.append(token)
    return out


def expand_list(spec: str) -> list[str]:
    return [s.strip() for s in spec.split(",") if s.strip()]
