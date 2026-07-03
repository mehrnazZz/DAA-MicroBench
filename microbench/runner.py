from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time
import numpy as np

from microbench.types import AABBObs, AgentContext, AgentState, IntentMsg, IntentObs, NeighborObs, PlannerInput, PlannerOutput, RunSpec
from microbench.config import load_defaults, load_comm_profiles
from microbench.scenarios import load_scenario, generate_spawns_goals, EventEngine
from microbench.planners import make_planner
from microbench.comm.v2v import V2VEmulator
from microbench.core import apply_dynamics, pairwise_stats, select_neighbors
from microbench.core.perception import fuse_observations, sense_neighbors
from microbench.metrics import EpisodeRecorder, EpisodeRingBuffer, FailureRecorder


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.zeros(3, dtype=float)
    return v / n


def _agent_seed(base_seed: int, agent_idx: int) -> int:
    return int((int(base_seed) * 1_000_003 + int(agent_idx)) % (2**32 - 1))


def _resolve_agent_methods(method: str, n_agents: int, agent_methods: list[str] | None) -> tuple[list[str], str]:
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


def _ttc_rel(pos_i: np.ndarray, vel_i: np.ndarray, pos_j: np.ndarray, vel_j: np.ndarray) -> float | None:
    rel_p = pos_j - pos_i
    rel_v = vel_j - vel_i
    vv = float(np.dot(rel_v, rel_v))
    if vv < 1e-9:
        return None
    t = -float(np.dot(rel_p, rel_v)) / vv
    return float(t) if t >= 0.0 else None


def _saturation_flags(vel: np.ndarray, v_cmd: np.ndarray, v_max: float, a_max: float, dt: float) -> tuple[bool, bool]:
    speed_sat = bool(np.linalg.norm(v_cmd) > v_max + 1e-9)
    if np.linalg.norm(v_cmd) > v_max + 1e-12:
        v_cmd_eff = v_cmd / np.linalg.norm(v_cmd) * v_max
    else:
        v_cmd_eff = v_cmd
    dv = v_cmd_eff - vel
    accel_sat = bool(np.linalg.norm(dv) > (a_max * dt) + 1e-9)
    return speed_sat, accel_sat


def _intent_event_snapshot(now_s: float, m, valid: bool, age_s: float) -> dict:
    if m is None:
        return {"present": False, "valid": False, "intent_age_s": float(age_s)}
    return {
        "present": True,
        "valid": bool(valid),
        "intent_age_s": float(age_s),
        "kind": str(m.kind),
        "expiry_s": float(m.expiry_s),
        "tube_radius_m": float(m.tube_radius_m),
        "num_points": int(np.asarray(m.points).shape[0]),
        "timestamp_send_s": float(m.timestamp_send_s),
        "recv_stale_at_t": float(now_s),
    }


def run_episode(spec: RunSpec) -> dict:
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
    intent_cfg = cfg.get("intent", {})
    log_cfg = cfg.get("logging", {})

    dt = float(sim_cfg.get("dt_s", 0.02))
    duration_s = float(cfg.get("scenario", {}).get("duration_s", sim_cfg.get("duration_s", 60.0)))
    steps = int(round(duration_s / dt))
    planar = bool(world_cfg.get("planar", sim_cfg.get("planar", True)))
    fixed_y = float(world_cfg.get("fixed_y_m", sim_cfg.get("fixed_y_m", 0.0)))
    goal_tol = float(agent_cfg.get("goal_tolerance_m", sim_cfg.get("goal_tolerance_m", 1.0)))
    goal_hold_time_s = float(sim_cfg.get("goal_hold_time_s", 0.5))
    near_margin = float(sim_cfg.get("near_miss_margin_m", 0.5))

    v_max = float(agent_cfg.get("v_max_mps", dyn_cfg.get("v_max_mps", 3.0)))
    a_max = float(agent_cfg.get("a_max_mps2", dyn_cfg.get("a_max_mps2", 2.0)))
    radius = float(agent_cfg.get("radius_m", 0.5))

    save_events = bool(log_cfg.get("save_events", True))
    save_trace = bool(spec.save_trace or log_cfg.get("save_trace", False))
    trace_max_steps = int(log_cfg.get("trace_max_steps", 4000))
    trace_save_failures_only = bool(log_cfg.get("trace_save_failures_only", True))
    save_trace_on_collision = bool(log_cfg.get("save_trace_on_collision", False))
    trace_window_s = float(log_cfg.get("trace_window_s", 3.0))
    trace_agents_mode = str(log_cfg.get("trace_agents_mode", "collision_pair_plus_neighbors"))
    near_miss_record_threshold = float(log_cfg.get("near_miss_record_threshold_m", 0.2))

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
                vel=np.zeros(3, dtype=float),
                goal=goals[i].copy(),
                radius=radius,
                v_max=v_max,
                a_max=a_max,
            )
        )

    agent_methods, method_label = _resolve_agent_methods(spec.method, spec.n_agents, spec.agent_methods)

    planners = [make_planner(agent_method) for agent_method in agent_methods]
    agent_contexts: list[AgentContext] = []
    for i, planner in enumerate(planners):
        seed_i = _agent_seed(spec.seed, i)
        planner.reset(seed_i)
        agent_contexts.append(
            AgentContext(
                agent_id=i,
                method=agent_methods[i],
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

    intent_enabled = bool(intent_cfg.get("enabled", False))
    intent_tx_rate_hz = float(intent_cfg.get("tx_rate_hz", cprof.get("tx_rate_hz", 10.0)))
    intent_max_points = int(intent_cfg.get("max_points", 12))
    intent_age_cap_s = float(intent_cfg.get("age_cap_s", age_cap_s))
    v2v = V2VEmulator(
        cprof,
        age_cap_s=age_cap_s,
        rng=rng,
        intent_cfg={
            "enabled": intent_enabled,
            "tx_rate_hz": intent_tx_rate_hz,
            "age_cap_s": intent_age_cap_s,
        },
    )
    v2v.reset(spec.n_agents)

    events = EventEngine(cfg.get("events", []), rng)
    events.reset()

    recorder = EpisodeRecorder(spec.n_agents, dt)
    ring = EpisodeRingBuffer(max_frames=max(1, int(round(trace_window_s / dt))))
    obstacles = cfg.get("obstacles", [])
    planner_obstacles = [
        AABBObs(
            center=np.asarray(ob["aabb"].get("center", [0.0, 0.0, 0.0]), dtype=float),
            half=np.asarray(ob["aabb"].get("half", [0.0, 0.0, 0.0]), dtype=float),
        )
        for ob in obstacles
        if "aabb" in ob
    ]
    failure = FailureRecorder(
        out_dir=spec.out_dir,
        scenario=Path(spec.scenario_path).stem,
        method=method_label,
        n_agents=spec.n_agents,
        seed=spec.seed,
        comm_profile=spec.comm_profile,
        save_trace=save_trace,
        trace_max_steps=trace_max_steps,
        trace_save_failures_only=trace_save_failures_only,
        save_events=save_events,
        save_trace_on_collision=save_trace_on_collision,
        trace_agents_mode=trace_agents_mode,
    )
    failure.set_episode_meta(
        {
            "scenario": Path(spec.scenario_path).stem,
            "scenario_name": cfg.get("scenario", {}).get("name", Path(spec.scenario_path).stem),
            "scenario_path": spec.scenario_path,
            "method": method_label,
            "agent_methods": agent_methods,
            "comm_profile": spec.comm_profile,
            "seed": int(spec.seed),
            "N": int(spec.n_agents),
            "planar": bool(planar),
            "fixed_y_m": float(fixed_y),
            "world_bounds": world_cfg.get("bounds", {}),
            "obstacles": obstacles,
        }
    )

    done_times = np.full(spec.n_agents, np.inf, dtype=float)
    goal_hold_elapsed = np.zeros(spec.n_agents, dtype=float)
    spawn_goal_dists = np.linalg.norm(goals - spawns, axis=1)
    planner_ms_samples: list[float] = []

    t_wall0 = time.perf_counter()
    try:
        for k in range(steps):
            t = k * dt

            for s in states:
                if s.done:
                    continue
                if np.linalg.norm(s.goal - s.pos) <= goal_tol:
                    goal_hold_elapsed[s.idx] += dt
                    if goal_hold_elapsed[s.idx] + 1e-12 >= goal_hold_time_s:
                        s.done = True
                        s.done_time_s = t
                        done_times[s.idx] = t
                        s.vel = np.zeros(3, dtype=float)
                else:
                    goal_hold_elapsed[s.idx] = 0.0

            v2v.step(t, states)

            v_cmds: list[np.ndarray] = [np.zeros(3, dtype=float) for _ in states]
            goal_dirs = [np.zeros(3, dtype=float) for _ in states]
            selected_neighbors: list[list[int]] = [[] for _ in states]
            selected_obs: list[list[dict]] = [[] for _ in states]
            selected_intents: list[list[dict]] = [[] for _ in states]
            selected_messages: list[list[dict]] = [[] for _ in states]
            msg_age_matrix = np.full((spec.n_agents, spec.n_agents), age_cap_s, dtype=float)
            pending_intent_out: list[IntentMsg | None] = [None for _ in states]
            pending_messages_out: list[list] = [[] for _ in states]

            for i, s in enumerate(states):
                if s.done:
                    continue
                goal_dir = _normalize(s.goal - s.pos)
                goal_dirs[i] = goal_dir

                v2v_obs: list[NeighborObs] = []
                for j in range(spec.n_agents):
                    if j == i:
                        continue
                    m = v2v.get_last(i, j)
                    valid, age = v2v.message_age(t, m)
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
                    range_m=float(ncfg.get("range_m", 30.0)),
                    top_k=int(ncfg.get("top_k", 8)),
                    threat_metric=str(ncfg.get("threat_metric", "ttc")),
                    ttc_horizon_s=float(ncfg.get("ttc_horizon_s", 6.0)),
                )

                selected_neighbors[i] = [n.idx for n in selected]
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
                    im = v2v.get_last_intent(i, n.idx)
                    ivalid, iage = v2v.intent_status(t, im)
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

                agent_messages = v2v.drain_agent_messages(i, t)
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
                    dt=dt,
                    t=t,
                    obstacles=planner_obstacles,
                    neighbor_intents=selected_intent_obs,
                    messages=agent_messages,
                    agent_context=agent_contexts[i],
                    planar=planar,
                )
                c0 = time.perf_counter()
                planner_out = planners[i].compute_cmd(p_input)
                c1 = time.perf_counter()
                planner_ms_samples.append((c1 - c0) * 1000.0)
                if isinstance(planner_out, PlannerOutput):
                    v_cmds[i] = np.asarray(planner_out.v_cmd, dtype=float)
                    pending_intent_out[i] = planner_out.intent_out
                    pending_messages_out[i] = list(planner_out.messages_out or [])
                else:
                    v_cmds[i] = np.asarray(planner_out, dtype=float)

            if intent_enabled:
                for i, out_msg in enumerate(pending_intent_out):
                    if states[i].done or out_msg is None:
                        continue
                    msg = out_msg
                    points = np.asarray(msg.points, dtype=float)
                    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 1:
                        continue
                    v2v.publish_intent(
                        sender=i,
                        intent=IntentMsg(
                            sender_id=i,
                            timestamp_send_s=float(msg.timestamp_send_s),
                            expiry_s=float(msg.expiry_s),
                            kind=str(msg.kind),
                            tube_radius_m=float(msg.tube_radius_m),
                            points=points,
                            dt_plan_s=float(msg.dt_plan_s) if msg.dt_plan_s is not None else None,
                            mode=msg.mode,
                        ),
                        now_s=t,
                        max_points=intent_max_points,
                    )

            for i, out_msgs in enumerate(pending_messages_out):
                if states[i].done:
                    continue
                for out_msg in out_msgs:
                    v2v.publish_agent_message(
                        sender=i,
                        msg=out_msg,
                        now_s=t,
                        n_agents=spec.n_agents,
                    )

            v_cmds = events.apply_overrides(t, states, v_cmds)

            speed_sat = [False for _ in states]
            accel_sat = [False for _ in states]
            for i, s in enumerate(states):
                if s.done:
                    continue
                speed_sat[i], accel_sat[i] = _saturation_flags(s.vel, v_cmds[i], s.v_max, s.a_max, dt)
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
                    s.vel = np.zeros(3, dtype=float)
                else:
                    s.path_length_m += float(np.linalg.norm(p_next - s.pos))
                    s.pos = p_next
                    s.vel = v_next

            pos = np.array([s.pos for s in states], dtype=float)
            vel = np.array([s.vel for s in states], dtype=float)
            rad = np.array([s.radius for s in states], dtype=float)
            done = np.array([s.done for s in states], dtype=bool)

            ring.push(
                frame := {
                    "t": float(t),
                    "n_agents": spec.n_agents,
                    "positions": pos.tolist(),
                    "velocities": vel.tolist(),
                    "v_cmd": [v.tolist() for v in v_cmds],
                    "goal_dirs": [g.tolist() for g in goal_dirs],
                    "selected_neighbors": selected_neighbors,
                    "selected_obs": selected_obs,
                    "selected_intents": selected_intents,
                    "selected_messages": selected_messages,
                    "speed_saturated": [bool(x) for x in speed_sat],
                    "accel_saturated": [bool(x) for x in accel_sat],
                }
            )
            failure.push_episode_frame(frame)
            recorder.record_observations(selected_obs, stale_age_s=age_cap_s)

            collisions, near_misses, min_sep = pairwise_stats(pos, rad, near_margin)
            collision_pairs_step: set[tuple[int, int]] = set()
            near_miss_pairs_step: set[tuple[int, int]] = set()

            for i in range(spec.n_agents):
                for j in range(i + 1, spec.n_agents):
                    dist = float(np.linalg.norm(pos[i] - pos[j]))
                    collision_threshold = float(rad[i] + rad[j])
                    near_threshold = collision_threshold + near_margin
                    is_collision = dist < collision_threshold
                    is_near = (dist < near_threshold) and not is_collision
                    if not is_collision and not is_near:
                        continue

                    if is_collision:
                        collision_pairs_step.add((i, j))
                    else:
                        near_miss_pairs_step.add((i, j))

                    clearance = dist - collision_threshold
                    if is_near and clearance > near_miss_record_threshold:
                        continue

                    max_age_i = max((o["msg_age_sec"] for o in selected_obs[i]), default=age_cap_s)
                    max_age_j = max((o["msg_age_sec"] for o in selected_obs[j]), default=age_cap_s)
                    intent_i_of_j = v2v.get_last_intent(i, j)
                    intent_j_of_i = v2v.get_last_intent(j, i)
                    intent_valid_i, intent_age_i = v2v.intent_status(t, intent_i_of_j)
                    intent_valid_j, intent_age_j = v2v.intent_status(t, intent_j_of_i)
                    event = {
                        "t": float(t),
                        "type": "collision" if is_collision else "near_miss",
                        "i": i,
                        "j": j,
                        "pos_i": pos[i].tolist(),
                        "pos_j": pos[j].tolist(),
                        "vel_i": vel[i].tolist(),
                        "vel_j": vel[j].tolist(),
                        "dist": dist,
                        "threshold": collision_threshold if is_collision else near_threshold,
                        "msg_age_i_of_j": float(msg_age_matrix[i, j]),
                        "msg_age_j_of_i": float(msg_age_matrix[j, i]),
                        "topk_snapshot": {
                            "i": int(len(selected_neighbors[i])),
                            "j": int(len(selected_neighbors[j])),
                        },
                        "ttc_s": _ttc_rel(pos[i], vel[i], pos[j], vel[j]),
                        "control_saturation": {
                            "i": {"speed": bool(speed_sat[i]), "accel": bool(accel_sat[i])},
                            "j": {"speed": bool(speed_sat[j]), "accel": bool(accel_sat[j])},
                        },
                        "staleness_blame": {
                            "max_topk_msg_age_i": float(max_age_i),
                            "max_topk_msg_age_j": float(max_age_j),
                        },
                        "intent_i_of_j": _intent_event_snapshot(t, intent_i_of_j, intent_valid_i, intent_age_i),
                        "intent_j_of_i": _intent_event_snapshot(t, intent_j_of_i, intent_valid_j, intent_age_j),
                    }
                    failure.record_proximity_event(event)

                    if is_collision:
                        failure.maybe_dump_collision_trace(
                            pair=(i, j),
                            t=t,
                            ring_snapshot=ring.snapshot(),
                            collision_meta=event,
                        )

            recorder.record_step(
                vel,
                done,
                collisions,
                near_misses,
                min_sep,
                t=t,
                collision_pairs=collision_pairs_step,
                near_miss_pairs=near_miss_pairs_step,
            )

            if bool(np.all(done)):
                break
    finally:
        failure.close()

    episode_runtime_s = time.perf_counter() - t_wall0

    metrics = recorder.finalize(
        done_times=done_times,
        spawn_goal_dists=spawn_goal_dists,
        planner_ms_samples=np.asarray(planner_ms_samples, dtype=float),
        episode_runtime_s=episode_runtime_s,
    )

    run_id = Path(spec.out_dir).name
    row = {
        "run_id": run_id,
        "method": method_label,
        "scenario": Path(spec.scenario_path).stem,
        "comm_profile": spec.comm_profile,
        "N": spec.n_agents,
        "seed": spec.seed,
        "dt_s": dt,
        "duration_s": duration_s,
        "v_max_mps": v_max,
        "a_max_mps2": a_max,
        "range_m": float(ncfg.get("range_m", 30.0)),
        "top_k": int(ncfg.get("top_k", 8)),
    }
    row.update(asdict(metrics))
    return row
