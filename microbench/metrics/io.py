from __future__ import annotations

from pathlib import Path
import csv
import math
from collections import defaultdict


RESULT_FIELDS = [
    "run_id",
    "method",
    "scenario",
    "comm_profile",
    "N",
    "seed",
    "dt_s",
    "duration_s",
    "v_max_mps",
    "a_max_mps2",
    "range_m",
    "top_k",
    "spawn_goal_dist_min",
    "spawn_goal_dist_mean",
    "collisions",
    "near_misses",
    "collision_pair_ticks",
    "near_miss_pair_ticks",
    "unique_collision_pairs",
    "unique_near_miss_pairs",
    "collision_episode",
    "near_miss_episode",
    "time_to_first_collision_s",
    "min_sep_min_m",
    "min_sep_p05_m",
    "completion_rate",
    "mean_time_to_goal_s",
    "p95_time_to_goal_s",
    "deadlock_time_pct",
    "jerk_mean",
    "planner_ms_per_tick_per_agent_mean",
    "planner_ms_per_tick_per_agent_p95",
    "obs_neighbors_mean",
    "obs_v2v_fraction",
    "obs_sensor_fraction",
    "obs_stale_fraction",
    "obs_empty_fraction",
    "comm_agent_msg_attempted",
    "comm_agent_msg_scheduled",
    "comm_agent_msg_delivered",
    "comm_agent_msg_dropped",
    "comm_agent_msg_expired",
    "comm_agent_msg_bytes_scheduled",
    "comm_agent_msg_bytes_delivered",
    "comm_agent_msg_bandwidth_Bps",
    "comm_agent_msg_drop_fraction",
    "comm_agent_msg_delivery_fraction",
    "comm_negotiation_proposals",
    "comm_negotiation_acks",
    "comm_negotiation_correlations_acked",
    "comm_negotiation_rejections",
    "episode_runtime_s",
]

SUMMARY_FIELDS = [
    "method",
    "scenario",
    "comm_profile",
    "N",
    "episodes",
    "collision_rate",
    "collision_episode_rate",
    "collisions_mean",
    "collisions_p95",
    "collision_pair_ticks_mean",
    "unique_collision_pairs_mean",
    "unique_collision_pairs_p95",
    "time_to_first_collision_mean",
    "near_miss_episode_rate",
    "near_miss_pair_ticks_mean",
    "unique_near_miss_pairs_mean",
    "min_sep_p05_mean",
    "min_sep_min_mean",
    "completion_rate_mean",
    "mean_time_to_goal_mean",
    "deadlock_time_pct_mean",
    "planner_ms_mean",
    "planner_ms_p95",
    "obs_neighbors_mean",
    "obs_v2v_fraction_mean",
    "obs_sensor_fraction_mean",
    "obs_stale_fraction_mean",
    "obs_empty_fraction_mean",
    "comm_agent_msg_attempted_mean",
    "comm_agent_msg_scheduled_mean",
    "comm_agent_msg_delivered_mean",
    "comm_agent_msg_dropped_mean",
    "comm_agent_msg_expired_mean",
    "comm_agent_msg_bandwidth_Bps_mean",
    "comm_agent_msg_drop_fraction_mean",
    "comm_agent_msg_delivery_fraction_mean",
    "comm_negotiation_proposals_mean",
    "comm_negotiation_acks_mean",
    "comm_negotiation_correlations_acked_mean",
    "comm_negotiation_rejections_mean",
]


def _to_float(x) -> float | None:
    try:
        v = float(x)
    except (ValueError, TypeError):
        return None
    if math.isnan(v):
        return None
    return v


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def _p95(vals: list[float]) -> float:
    if not vals:
        return float("nan")
    vals = sorted(vals)
    idx = int(round(0.95 * (len(vals) - 1)))
    return float(vals[max(0, min(idx, len(vals) - 1))])


def append_result(out_dir: str | Path, row: dict) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "results.csv"
    exists = path.exists()
    if exists:
        with path.open("r", encoding="utf-8") as f:
            first = f.readline().strip()
        existing = first.split(",") if first else []
        if existing and existing != RESULT_FIELDS:
            raise RuntimeError(
                f"Existing results.csv schema mismatch in {path}. "
                "Use a new out_dir or remove the old results.csv."
            )
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in RESULT_FIELDS})
    return path


def write_summary(out_dir: str | Path) -> Path:
    out = Path(out_dir)
    results_path = out / "results.csv"
    summary_path = out / "summary.csv"
    if not results_path.exists():
        return summary_path

    with results_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    groups: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["method"], r["scenario"], r["comm_profile"], r["N"])].append(r)

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()

        for (method, scenario, comm_profile, N), items in sorted(groups.items()):
            coll = [_to_float(x.get("collisions")) for x in items]
            coll = [x for x in coll if x is not None]
            collision_pair_ticks = [_to_float(x.get("collision_pair_ticks")) for x in items]
            collision_pair_ticks = [x for x in collision_pair_ticks if x is not None]
            collision_episode = [_to_float(x.get("collision_episode")) for x in items]
            collision_episode = [x for x in collision_episode if x is not None]
            unique_collision_pairs = [_to_float(x.get("unique_collision_pairs")) for x in items]
            unique_collision_pairs = [x for x in unique_collision_pairs if x is not None]
            first_collision = [_to_float(x.get("time_to_first_collision_s")) for x in items]
            first_collision = [x for x in first_collision if x is not None]
            near_miss_pair_ticks = [_to_float(x.get("near_miss_pair_ticks")) for x in items]
            near_miss_pair_ticks = [x for x in near_miss_pair_ticks if x is not None]
            near_miss_episode = [_to_float(x.get("near_miss_episode")) for x in items]
            near_miss_episode = [x for x in near_miss_episode if x is not None]
            unique_near_miss_pairs = [_to_float(x.get("unique_near_miss_pairs")) for x in items]
            unique_near_miss_pairs = [x for x in unique_near_miss_pairs if x is not None]
            min_sep_p05 = [_to_float(x.get("min_sep_p05_m")) for x in items]
            min_sep_p05 = [x for x in min_sep_p05 if x is not None]
            min_sep_min = [_to_float(x.get("min_sep_min_m")) for x in items]
            min_sep_min = [x for x in min_sep_min if x is not None]
            comp = [_to_float(x.get("completion_rate")) for x in items]
            comp = [x for x in comp if x is not None]
            ttg = [_to_float(x.get("mean_time_to_goal_s")) for x in items]
            ttg = [x for x in ttg if x is not None]
            dead = [_to_float(x.get("deadlock_time_pct")) for x in items]
            dead = [x for x in dead if x is not None]
            pms = [_to_float(x.get("planner_ms_per_tick_per_agent_mean")) for x in items]
            pms = [x for x in pms if x is not None]
            obs_neighbors = [_to_float(x.get("obs_neighbors_mean")) for x in items]
            obs_neighbors = [x for x in obs_neighbors if x is not None]
            obs_v2v = [_to_float(x.get("obs_v2v_fraction")) for x in items]
            obs_v2v = [x for x in obs_v2v if x is not None]
            obs_sensor = [_to_float(x.get("obs_sensor_fraction")) for x in items]
            obs_sensor = [x for x in obs_sensor if x is not None]
            obs_stale = [_to_float(x.get("obs_stale_fraction")) for x in items]
            obs_stale = [x for x in obs_stale if x is not None]
            obs_empty = [_to_float(x.get("obs_empty_fraction")) for x in items]
            obs_empty = [x for x in obs_empty if x is not None]
            msg_attempted = [_to_float(x.get("comm_agent_msg_attempted")) for x in items]
            msg_attempted = [x for x in msg_attempted if x is not None]
            msg_scheduled = [_to_float(x.get("comm_agent_msg_scheduled")) for x in items]
            msg_scheduled = [x for x in msg_scheduled if x is not None]
            msg_delivered = [_to_float(x.get("comm_agent_msg_delivered")) for x in items]
            msg_delivered = [x for x in msg_delivered if x is not None]
            msg_dropped = [_to_float(x.get("comm_agent_msg_dropped")) for x in items]
            msg_dropped = [x for x in msg_dropped if x is not None]
            msg_expired = [_to_float(x.get("comm_agent_msg_expired")) for x in items]
            msg_expired = [x for x in msg_expired if x is not None]
            msg_bandwidth = [_to_float(x.get("comm_agent_msg_bandwidth_Bps")) for x in items]
            msg_bandwidth = [x for x in msg_bandwidth if x is not None]
            msg_drop_fraction = [_to_float(x.get("comm_agent_msg_drop_fraction")) for x in items]
            msg_drop_fraction = [x for x in msg_drop_fraction if x is not None]
            msg_delivery_fraction = [_to_float(x.get("comm_agent_msg_delivery_fraction")) for x in items]
            msg_delivery_fraction = [x for x in msg_delivery_fraction if x is not None]
            negotiation_proposals = [_to_float(x.get("comm_negotiation_proposals")) for x in items]
            negotiation_proposals = [x for x in negotiation_proposals if x is not None]
            negotiation_acks = [_to_float(x.get("comm_negotiation_acks")) for x in items]
            negotiation_acks = [x for x in negotiation_acks if x is not None]
            negotiation_correlations_acked = [_to_float(x.get("comm_negotiation_correlations_acked")) for x in items]
            negotiation_correlations_acked = [x for x in negotiation_correlations_acked if x is not None]
            negotiation_rejections = [_to_float(x.get("comm_negotiation_rejections")) for x in items]
            negotiation_rejections = [x for x in negotiation_rejections if x is not None]

            collision_rate = (
                sum(1 for x in coll if x > 0.0) / len(coll)
                if coll
                else float("nan")
            )

            row = {
                "method": method,
                "scenario": scenario,
                "comm_profile": comm_profile,
                "N": N,
                "episodes": len(items),
                "collision_rate": collision_rate,
                "collision_episode_rate": _mean(collision_episode) if collision_episode else collision_rate,
                "collisions_mean": _mean(coll),
                "collisions_p95": _p95(coll),
                "collision_pair_ticks_mean": _mean(collision_pair_ticks) if collision_pair_ticks else _mean(coll),
                "unique_collision_pairs_mean": _mean(unique_collision_pairs),
                "unique_collision_pairs_p95": _p95(unique_collision_pairs),
                "time_to_first_collision_mean": _mean(first_collision),
                "near_miss_episode_rate": _mean(near_miss_episode),
                "near_miss_pair_ticks_mean": _mean(near_miss_pair_ticks),
                "unique_near_miss_pairs_mean": _mean(unique_near_miss_pairs),
                "min_sep_p05_mean": _mean(min_sep_p05),
                "min_sep_min_mean": _mean(min_sep_min),
                "completion_rate_mean": _mean(comp),
                "mean_time_to_goal_mean": _mean(ttg),
                "deadlock_time_pct_mean": _mean(dead),
                "planner_ms_mean": _mean(pms),
                "planner_ms_p95": _p95(pms),
                "obs_neighbors_mean": _mean(obs_neighbors),
                "obs_v2v_fraction_mean": _mean(obs_v2v),
                "obs_sensor_fraction_mean": _mean(obs_sensor),
                "obs_stale_fraction_mean": _mean(obs_stale),
                "obs_empty_fraction_mean": _mean(obs_empty),
                "comm_agent_msg_attempted_mean": _mean(msg_attempted),
                "comm_agent_msg_scheduled_mean": _mean(msg_scheduled),
                "comm_agent_msg_delivered_mean": _mean(msg_delivered),
                "comm_agent_msg_dropped_mean": _mean(msg_dropped),
                "comm_agent_msg_expired_mean": _mean(msg_expired),
                "comm_agent_msg_bandwidth_Bps_mean": _mean(msg_bandwidth),
                "comm_agent_msg_drop_fraction_mean": _mean(msg_drop_fraction),
                "comm_agent_msg_delivery_fraction_mean": _mean(msg_delivery_fraction),
                "comm_negotiation_proposals_mean": _mean(negotiation_proposals),
                "comm_negotiation_acks_mean": _mean(negotiation_acks),
                "comm_negotiation_correlations_acked_mean": _mean(negotiation_correlations_acked),
                "comm_negotiation_rejections_mean": _mean(negotiation_rejections),
            }
            writer.writerow(row)

    return summary_path
