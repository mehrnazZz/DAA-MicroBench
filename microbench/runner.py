from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time

import numpy as np

from microbench.core import EpisodeEngine
from microbench.metrics import EpisodeRecorder, EpisodeRingBuffer, FailureRecorder
from microbench.planners import make_planner
from microbench.types import RunSpec


def _ttc_rel(pos_i: np.ndarray, vel_i: np.ndarray, pos_j: np.ndarray, vel_j: np.ndarray) -> float | None:
    rel_p = pos_j - pos_i
    rel_v = vel_j - vel_i
    vv = float(np.dot(rel_v, rel_v))
    if vv < 1e-9:
        return None
    t = -float(np.dot(rel_p, rel_v)) / vv
    return float(t) if t >= 0.0 else None


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
    engine = EpisodeEngine(
        scenario_path=spec.scenario_path,
        method=spec.method,
        n_agents=spec.n_agents,
        seed=spec.seed,
        comm_profile=spec.comm_profile,
        agent_methods=spec.agent_methods,
        policy_spec=spec.policy_spec,
        planner_factory=make_planner,
    )

    log_cfg = engine.cfg.get("logging", {})
    save_events = bool(log_cfg.get("save_events", True))
    save_trace = bool(spec.save_trace or log_cfg.get("save_trace", False))
    trace_max_steps = int(log_cfg.get("trace_max_steps", 4000))
    trace_save_failures_only = bool(log_cfg.get("trace_save_failures_only", True))
    save_trace_on_collision = bool(log_cfg.get("save_trace_on_collision", False))
    trace_window_s = float(log_cfg.get("trace_window_s", 3.0))
    trace_agents_mode = str(log_cfg.get("trace_agents_mode", "collision_pair_plus_neighbors"))
    near_miss_record_threshold = float(log_cfg.get("near_miss_record_threshold_m", 0.2))

    recorder = EpisodeRecorder(spec.n_agents, engine.dt)
    ring = EpisodeRingBuffer(max_frames=max(1, int(round(trace_window_s / engine.dt))))
    failure = FailureRecorder(
        out_dir=spec.out_dir,
        scenario=Path(spec.scenario_path).stem,
        method=engine.method_label,
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
            "scenario_name": engine.cfg.get("scenario", {}).get("name", Path(spec.scenario_path).stem),
            "scenario_path": spec.scenario_path,
            "method": engine.method_label,
            "agent_methods": engine.agent_methods,
            "comm_profile": spec.comm_profile,
            "seed": int(spec.seed),
            "N": int(spec.n_agents),
            "planar": bool(engine.planar),
            "fixed_y_m": float(engine.fixed_y),
            "world_bounds": engine.world_cfg.get("bounds", {}),
            "obstacles": engine.obstacles,
            "agent_profiles": [
                {
                    "agent_id": int(ctx.agent_id),
                    "method": str(ctx.method),
                    "role": ctx.role,
                    "priority": int(ctx.priority),
                    "capabilities": dict(ctx.capabilities),
                    "mission": dict(ctx.mission),
                    "failure_modes": dict(ctx.failure_modes),
                }
                for ctx in engine.agent_contexts
            ],
        }
    )

    t_wall0 = time.perf_counter()
    try:
        while True:
            step = engine.step()
            if step is None:
                break

            frame = step.trace_frame()
            ring.push(frame)
            failure.push_episode_frame(frame)
            recorder.record_observations(step.selected_obs, stale_age_s=engine.age_cap_s)

            for i in range(spec.n_agents):
                for j in range(i + 1, spec.n_agents):
                    dist = float(np.linalg.norm(step.pos[i] - step.pos[j]))
                    collision_threshold = float(step.radii[i] + step.radii[j])
                    near_threshold = collision_threshold + engine.near_margin
                    is_collision = (i, j) in step.collision_pairs
                    is_near = (i, j) in step.near_miss_pairs
                    if not is_collision and not is_near:
                        continue

                    clearance = dist - collision_threshold
                    if is_near and clearance > near_miss_record_threshold:
                        continue

                    max_age_i = max((o["msg_age_sec"] for o in step.selected_obs[i]), default=engine.age_cap_s)
                    max_age_j = max((o["msg_age_sec"] for o in step.selected_obs[j]), default=engine.age_cap_s)
                    intent_i_of_j = engine.v2v.get_last_intent(i, j)
                    intent_j_of_i = engine.v2v.get_last_intent(j, i)
                    intent_valid_i, intent_age_i = engine.v2v.intent_status(step.t, intent_i_of_j)
                    intent_valid_j, intent_age_j = engine.v2v.intent_status(step.t, intent_j_of_i)
                    event = {
                        "t": float(step.t),
                        "type": "collision" if is_collision else "near_miss",
                        "i": i,
                        "j": j,
                        "pos_i": step.pos[i].tolist(),
                        "pos_j": step.pos[j].tolist(),
                        "vel_i": step.vel[i].tolist(),
                        "vel_j": step.vel[j].tolist(),
                        "dist": dist,
                        "threshold": collision_threshold if is_collision else near_threshold,
                        "msg_age_i_of_j": float(step.msg_age_matrix[i, j]),
                        "msg_age_j_of_i": float(step.msg_age_matrix[j, i]),
                        "topk_snapshot": {
                            "i": int(len(step.selected_neighbors[i])),
                            "j": int(len(step.selected_neighbors[j])),
                        },
                        "ttc_s": _ttc_rel(step.pos[i], step.vel[i], step.pos[j], step.vel[j]),
                        "control_saturation": {
                            "i": {
                                "speed": bool(step.speed_saturated[i]),
                                "accel": bool(step.accel_saturated[i]),
                            },
                            "j": {
                                "speed": bool(step.speed_saturated[j]),
                                "accel": bool(step.accel_saturated[j]),
                            },
                        },
                        "staleness_blame": {
                            "max_topk_msg_age_i": float(max_age_i),
                            "max_topk_msg_age_j": float(max_age_j),
                        },
                        "intent_i_of_j": _intent_event_snapshot(step.t, intent_i_of_j, intent_valid_i, intent_age_i),
                        "intent_j_of_i": _intent_event_snapshot(step.t, intent_j_of_i, intent_valid_j, intent_age_j),
                    }
                    failure.record_proximity_event(event)

                    if is_collision:
                        failure.maybe_dump_collision_trace(
                            pair=(i, j),
                            t=step.t,
                            ring_snapshot=ring.snapshot(),
                            collision_meta=event,
                        )

            recorder.record_step(
                step.vel,
                step.done,
                step.collisions,
                step.near_misses,
                step.min_sep,
                t=step.t,
                collision_pairs=step.collision_pairs,
                near_miss_pairs=step.near_miss_pairs,
            )

            if bool(np.all(step.done)):
                break
    finally:
        engine.close()
        failure.close()

    episode_runtime_s = time.perf_counter() - t_wall0
    metrics = recorder.finalize(
        done_times=engine.done_times,
        spawn_goal_dists=engine.spawn_goal_dists,
        planner_ms_samples=np.asarray(engine.planner_ms_samples, dtype=float),
        episode_runtime_s=episode_runtime_s,
        comm_stats=engine.v2v.agent_message_stats_snapshot(),
        planner_guardrail_stats={
            "planner_timeout_count": engine.planner_timeout_count,
            "planner_error_count": engine.planner_error_count,
            "planner_fallback_count": engine.planner_fallback_count,
        },
    )

    run_id = Path(spec.out_dir).name
    row = {
        "run_id": run_id,
        "method": engine.method_label,
        "scenario": Path(spec.scenario_path).stem,
        "comm_profile": spec.comm_profile,
        "N": spec.n_agents,
        "seed": spec.seed,
        "dt_s": engine.dt,
        "duration_s": engine.duration_s,
        "v_max_mps": engine.v_max,
        "a_max_mps2": engine.a_max,
        "range_m": float(engine.neighbor_cfg.get("range_m", 30.0)),
        "top_k": int(engine.neighbor_cfg.get("top_k", 8)),
    }
    row.update(asdict(metrics))
    return row
