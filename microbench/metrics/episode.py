from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class EpisodeMetrics:
    # Backward-compatible aliases for collision_pair_ticks and near_miss_pair_ticks.
    collisions: int
    near_misses: int
    collision_pair_ticks: int
    near_miss_pair_ticks: int
    unique_collision_pairs: int
    unique_near_miss_pairs: int
    collision_episode: int
    near_miss_episode: int
    time_to_first_collision_s: float
    min_sep_min_m: float
    min_sep_p05_m: float
    completion_rate: float
    mean_time_to_goal_s: float
    p95_time_to_goal_s: float
    deadlock_time_pct: float
    jerk_mean: float
    spawn_goal_dist_min: float
    spawn_goal_dist_mean: float
    planner_ms_per_tick_per_agent_mean: float
    planner_ms_per_tick_per_agent_p95: float
    obs_neighbors_mean: float
    obs_v2v_fraction: float
    obs_sensor_fraction: float
    obs_stale_fraction: float
    obs_empty_fraction: float
    comm_agent_msg_attempted: int
    comm_agent_msg_scheduled: int
    comm_agent_msg_delivered: int
    comm_agent_msg_dropped: int
    comm_agent_msg_expired: int
    comm_agent_msg_bytes_scheduled: int
    comm_agent_msg_bytes_delivered: int
    comm_agent_msg_bandwidth_Bps: float
    comm_agent_msg_drop_fraction: float
    comm_agent_msg_delivery_fraction: float
    comm_negotiation_proposals: int
    comm_negotiation_acks: int
    comm_negotiation_correlations_acked: int
    comm_negotiation_rejections: int
    episode_runtime_s: float


class EpisodeRecorder:
    def __init__(self, n_agents: int, dt: float):
        self.n_agents = n_agents
        self.dt = dt
        self.collision_pair_ticks = 0
        self.near_miss_pair_ticks = 0
        self.collision_pairs_seen: set[tuple[int, int]] = set()
        self.near_miss_pairs_seen: set[tuple[int, int]] = set()
        self.first_collision_time_s: float | None = None
        self.min_seps: list[float] = []
        self.deadlock_ticks = 0
        self.total_ticks = 0
        self.prev_vel: np.ndarray | None = None
        self.jerk_values: list[float] = []
        self.obs_neighbor_counts: list[int] = []
        self.obs_v2v_count = 0
        self.obs_sensor_count = 0
        self.obs_total_count = 0
        self.obs_stale_count = 0
        self.obs_empty_agent_ticks = 0
        self.obs_agent_ticks = 0

    def record_step(
        self,
        velocities: np.ndarray,
        done: np.ndarray,
        collisions: int,
        near_misses: int,
        min_sep: float,
        *,
        t: float | None = None,
        collision_pairs: set[tuple[int, int]] | None = None,
        near_miss_pairs: set[tuple[int, int]] | None = None,
    ) -> None:
        self.collision_pair_ticks += collisions
        self.near_miss_pair_ticks += near_misses
        if collision_pairs:
            self.collision_pairs_seen.update(tuple(sorted(p)) for p in collision_pairs)
            if self.first_collision_time_s is None and t is not None:
                self.first_collision_time_s = float(t)
        elif collisions > 0 and self.first_collision_time_s is None and t is not None:
            self.first_collision_time_s = float(t)
        if near_miss_pairs:
            self.near_miss_pairs_seen.update(tuple(sorted(p)) for p in near_miss_pairs)
        self.min_seps.append(float(min_sep))

        speeds = np.linalg.norm(velocities, axis=1)
        self.deadlock_ticks += int(np.sum((speeds < 0.15) & (~done)))
        self.total_ticks += self.n_agents

        if self.prev_vel is not None:
            dv = velocities - self.prev_vel
            jerk = np.linalg.norm(dv, axis=1) / self.dt
            self.jerk_values.extend(jerk.tolist())

        self.prev_vel = velocities.copy()

    def record_observations(self, selected_obs_by_agent: list[list[dict]], stale_age_s: float) -> None:
        for obs_list in selected_obs_by_agent:
            count = len(obs_list)
            self.obs_neighbor_counts.append(count)
            self.obs_agent_ticks += 1
            if count == 0:
                self.obs_empty_agent_ticks += 1
            for obs in obs_list:
                self.obs_total_count += 1
                source = str(obs.get("source", "v2v"))
                if source == "sensor":
                    self.obs_sensor_count += 1
                elif source == "v2v":
                    self.obs_v2v_count += 1
                if float(obs.get("msg_age_sec", 0.0)) >= float(stale_age_s):
                    self.obs_stale_count += 1

    def finalize(
        self,
        done_times: np.ndarray,
        spawn_goal_dists: np.ndarray,
        planner_ms_samples: np.ndarray,
        episode_runtime_s: float,
        comm_stats: dict[str, int] | None = None,
    ) -> EpisodeMetrics:
        finished = np.isfinite(done_times)
        completion_rate = float(np.mean(finished)) if len(finished) else 0.0
        if np.any(finished):
            t = done_times[finished]
            mean_time = float(np.mean(t))
            p95_time = float(np.percentile(t, 95))
        else:
            mean_time = float("nan")
            p95_time = float("nan")

        min_seps = np.asarray(self.min_seps, dtype=float)
        min_sep_min = float(np.min(min_seps)) if len(min_seps) else 0.0
        min_sep_p05 = float(np.percentile(min_seps, 5)) if len(min_seps) else 0.0
        deadlock_time_pct = float(self.deadlock_ticks / max(1, self.total_ticks))
        jerk_mean = float(np.mean(self.jerk_values)) if self.jerk_values else 0.0

        if planner_ms_samples.size:
            planner_mean = float(np.mean(planner_ms_samples))
            planner_p95 = float(np.percentile(planner_ms_samples, 95))
        else:
            planner_mean = 0.0
            planner_p95 = 0.0

        obs_neighbors_mean = float(np.mean(self.obs_neighbor_counts)) if self.obs_neighbor_counts else 0.0
        obs_total = max(1, self.obs_total_count)
        obs_agent_ticks = max(1, self.obs_agent_ticks)
        comm = comm_stats or {}
        msg_attempted = int(comm.get("agent_msg_attempted", 0))
        msg_scheduled = int(comm.get("agent_msg_scheduled", 0))
        msg_delivered = int(comm.get("agent_msg_delivered", 0))
        msg_dropped = int(comm.get("agent_msg_dropped", 0))
        msg_expired = int(comm.get("agent_msg_expired", 0))
        bytes_scheduled = int(comm.get("agent_msg_bytes_scheduled", 0))
        bytes_delivered = int(comm.get("agent_msg_bytes_delivered", 0))
        negotiation_proposals = int(comm.get("agent_msg_negotiation_proposals", 0))
        negotiation_acks = int(comm.get("agent_msg_negotiation_acks", 0))
        negotiation_correlations_acked = int(comm.get("agent_msg_negotiation_correlations_acked", 0))
        negotiation_rejections = int(comm.get("agent_msg_negotiation_rejections", 0))
        sim_time_s = max(self.dt, (self.total_ticks / max(1, self.n_agents)) * self.dt)

        return EpisodeMetrics(
            collisions=int(self.collision_pair_ticks),
            near_misses=int(self.near_miss_pair_ticks),
            collision_pair_ticks=int(self.collision_pair_ticks),
            near_miss_pair_ticks=int(self.near_miss_pair_ticks),
            unique_collision_pairs=int(len(self.collision_pairs_seen)),
            unique_near_miss_pairs=int(len(self.near_miss_pairs_seen)),
            collision_episode=int(self.collision_pair_ticks > 0),
            near_miss_episode=int(self.near_miss_pair_ticks > 0),
            time_to_first_collision_s=float(self.first_collision_time_s)
            if self.first_collision_time_s is not None
            else float("nan"),
            min_sep_min_m=min_sep_min,
            min_sep_p05_m=min_sep_p05,
            completion_rate=completion_rate,
            mean_time_to_goal_s=mean_time,
            p95_time_to_goal_s=p95_time,
            deadlock_time_pct=deadlock_time_pct,
            jerk_mean=jerk_mean,
            spawn_goal_dist_min=float(np.min(spawn_goal_dists)) if len(spawn_goal_dists) else 0.0,
            spawn_goal_dist_mean=float(np.mean(spawn_goal_dists)) if len(spawn_goal_dists) else 0.0,
            planner_ms_per_tick_per_agent_mean=planner_mean,
            planner_ms_per_tick_per_agent_p95=planner_p95,
            obs_neighbors_mean=obs_neighbors_mean,
            obs_v2v_fraction=float(self.obs_v2v_count / obs_total),
            obs_sensor_fraction=float(self.obs_sensor_count / obs_total),
            obs_stale_fraction=float(self.obs_stale_count / obs_total),
            obs_empty_fraction=float(self.obs_empty_agent_ticks / obs_agent_ticks),
            comm_agent_msg_attempted=msg_attempted,
            comm_agent_msg_scheduled=msg_scheduled,
            comm_agent_msg_delivered=msg_delivered,
            comm_agent_msg_dropped=msg_dropped,
            comm_agent_msg_expired=msg_expired,
            comm_agent_msg_bytes_scheduled=bytes_scheduled,
            comm_agent_msg_bytes_delivered=bytes_delivered,
            comm_agent_msg_bandwidth_Bps=float(bytes_scheduled / sim_time_s),
            comm_agent_msg_drop_fraction=float(msg_dropped / max(1, msg_attempted)),
            comm_agent_msg_delivery_fraction=float(msg_delivered / max(1, msg_scheduled)),
            comm_negotiation_proposals=negotiation_proposals,
            comm_negotiation_acks=negotiation_acks,
            comm_negotiation_correlations_acked=negotiation_correlations_acked,
            comm_negotiation_rejections=negotiation_rejections,
            episode_runtime_s=float(episode_runtime_s),
        )
