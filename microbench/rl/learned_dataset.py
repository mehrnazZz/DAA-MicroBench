from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path
import shutil
from typing import Any, Callable

import numpy as np

from microbench.core import EpisodeEngine
from microbench.core.episode_engine import PRIVILEGED_JOINT_METHODS
from microbench.planners import canonical_method, list_methods
from microbench.rl.bc_training import BC_TEACHER_NAME, _teacher_action_from_observation
from microbench.rl.envs import DaaParallelEnv, agent_id_from_name, observation_from_public_snapshot
from microbench.rl.policies import POLICY_NAMES, RlPolicy, make_policy
from microbench.rl.policy_spec import policy_factory_from_spec
from microbench.rl.schema import (
    DEFAULT_REWARD_WEIGHTS,
    RL_ACTION_SCHEMA_VERSION,
    RL_INTERFACE_VERSION,
    RL_OBSERVATION_SCHEMA_VERSION,
    RL_REWARD_SCHEMA_VERSION,
    interface_contract,
)
from microbench.tools.baseline_validation_matrix import (
    ValidationLane,
    prepare_validation_lane_scenarios,
    selected_validation_lanes,
)


LEARNED_DATASET_SCHEMA_VERSION = "0.1"
LEARNED_DATASET_TEACHER_POLICY = "bc_teacher"
LEARNED_DATASET_PLANNER_EXPERT_SOURCE = "planner_expert"
LEARNED_DATASET_POLICY_CHOICES = (LEARNED_DATASET_TEACHER_POLICY, *POLICY_NAMES)
LEARNED_DENSE_SWARM_HARD_NEGATIVE_LANE_ID = "dense_swarm_hard_negative"
LEARNED_DATASET_EXTRA_LANES = (
    ValidationLane(
        lane_id=LEARNED_DENSE_SWARM_HARD_NEGATIVE_LANE_ID,
        category="high_n_dense_merge",
        suite="official_3d_stress",
        scenario="dense_swarm_3d_hard",
        comm_profile="degraded_20hz",
        n_agents=12,
        seed=0,
        duration_s=18.0,
        purpose=(
            "Learned-policy hard-negative training lane for dense 3D swarm interactions; "
            "not part of the default validation matrix."
        ),
        expected_failure_modes=("dense_center_conflict", "late_lateral_yield", "stale_intent", "throughput_collapse"),
    ),
)
LEARNED_DATASET_EXTRA_LANE_IDS = tuple(lane.lane_id for lane in LEARNED_DATASET_EXTRA_LANES)

LEARNED_DATASET_EPISODE_FIELDS = (
    "episode_id",
    "lane_id",
    "category",
    "suite",
    "scenario",
    "scenario_path",
    "dimension",
    "action_source",
    "policy",
    "n_agents",
    "seed",
    "comm_profile",
    "steps",
    "sample_count",
    "controlled_agents",
    "completion_rate",
    "collision_ticks",
    "near_miss_ticks",
    "final_min_sep_m",
    "replay_path",
    "api_error",
)


class _BcTeacherPolicy:
    policy_name = BC_TEACHER_NAME

    def reset(self, seed: int) -> None:
        _ = seed

    def action(self, agent: str, observation: np.ndarray, action_space: Any, info: dict[str, Any]) -> np.ndarray:
        _ = agent, action_space, info
        _features, action = _teacher_action_from_observation(observation)
        return np.asarray(action, dtype=np.float32)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    return path


def _with_overrides(
    lanes: list[ValidationLane],
    *,
    duration_s: float | None,
    n_agents: int | None,
) -> list[ValidationLane]:
    out: list[ValidationLane] = []
    for lane in lanes:
        payload = asdict(lane)
        if duration_s is not None:
            payload["duration_s"] = float(duration_s)
        if n_agents is not None:
            payload["n_agents"] = int(n_agents)
        out.append(ValidationLane(**payload))
    return out


def _learned_dataset_lane_map() -> dict[str, ValidationLane]:
    lanes = [*selected_validation_lanes(None), *LEARNED_DATASET_EXTRA_LANES]
    return {lane.lane_id: lane for lane in lanes}


def selected_learned_dataset_lanes(lanes: tuple[str, ...] | list[str] | None = None) -> list[ValidationLane]:
    """Return learned-dataset lanes, including explicit hard-negative training lanes.

    The default remains the canonical validation-matrix lanes. Extra lanes must
    be requested explicitly so normal wrapper validation stays compact.
    """

    if lanes is None:
        return selected_validation_lanes(None)
    lane_ids = [str(lane_id).strip() for lane_id in lanes if str(lane_id).strip()]
    by_id = _learned_dataset_lane_map()
    unknown = sorted(set(lane_ids) - set(by_id))
    if unknown:
        raise ValueError(
            "Unknown learned dataset lane(s): "
            + ",".join(unknown)
            + "; expected validation lanes or "
            + ",".join(LEARNED_DATASET_EXTRA_LANE_IDS)
        )
    return [by_id[lane_id] for lane_id in lane_ids]


def _seed_list_for_lane(lane: ValidationLane, seeds: list[int] | None) -> list[int]:
    return [int(lane.seed)] if seeds is None else [int(seed) for seed in seeds]


def _policy_factory(
    *,
    policy: str,
    policy_spec: str | Path | None,
) -> tuple[Callable[[int], RlPolicy], str, str, dict[str, Any] | None]:
    if policy_spec is not None:
        factory, summary = policy_factory_from_spec(policy_spec)
        return factory, "policy_spec", str(summary.get("policy_name")), summary

    key = str(policy).strip()
    if key == LEARNED_DATASET_TEACHER_POLICY:
        return lambda seed: _BcTeacherPolicy(), "bc_teacher", BC_TEACHER_NAME, None
    if key not in POLICY_NAMES:
        raise ValueError(
            f"Unknown learned dataset policy {policy!r}; expected one of "
            f"{','.join(LEARNED_DATASET_POLICY_CHOICES)} or pass --policy-spec"
        )
    return lambda seed: make_policy(key, seed=int(seed)), "builtin_policy", key, None


def _safe_action(policy_obj: RlPolicy, agent: str, observation: np.ndarray, action_space: Any, info: dict[str, Any]) -> np.ndarray:
    action = np.asarray(policy_obj.action(agent, observation, action_space, info), dtype=np.float32)
    if action.shape != (3,):
        raise ValueError(f"policy action for {agent} must have shape (3,), got {action.shape}")
    if not np.all(np.isfinite(action)):
        raise ValueError(f"policy action for {agent} must be finite")
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def _planner_expert_name(planner_expert: str | None) -> str | None:
    if planner_expert is None:
        return None
    key = str(planner_expert).strip()
    if not key:
        raise ValueError("planner_expert must be a non-empty planner method name")
    canonical = canonical_method(key)
    if canonical not in set(list_methods()):
        raise ValueError(f"Unknown planner expert {planner_expert!r}; expected one of {','.join(list_methods())}")
    if canonical == "learned_policy_spec":
        raise ValueError("planner_expert cannot be learned_policy_spec; pass it as --policy-spec instead")
    return canonical


def _planner_expert_uses_privileged_state(planner_expert: str | None) -> bool:
    if planner_expert is None:
        return False
    return canonical_method(str(planner_expert)) in PRIVILEGED_JOINT_METHODS


def _action_from_planner_v_cmd(v_cmd: np.ndarray, *, v_max: float, planar: bool) -> np.ndarray:
    action = np.asarray(v_cmd, dtype=np.float32).reshape(3) / max(1e-6, float(v_max))
    if bool(planar):
        action[1] = 0.0
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def _empty_arrays() -> dict[str, np.ndarray]:
    return {
        "observations": np.zeros((0, 0), dtype=np.float32),
        "next_observations": np.zeros((0, 0), dtype=np.float32),
        "observation_valid_dim": np.zeros((0,), dtype=np.int32),
        "next_observation_valid_dim": np.zeros((0,), dtype=np.int32),
        "actions": np.zeros((0, 3), dtype=np.float32),
        "rewards": np.zeros((0,), dtype=np.float32),
        "terminated": np.zeros((0,), dtype=bool),
        "truncated": np.zeros((0,), dtype=bool),
        "done": np.zeros((0,), dtype=bool),
        "episode_id": np.zeros((0,), dtype=np.int32),
        "step": np.zeros((0,), dtype=np.int32),
        "agent_id": np.zeros((0,), dtype=np.int32),
        "lane_id": np.zeros((0,), dtype="U1"),
        "category": np.zeros((0,), dtype="U1"),
        "scenario": np.zeros((0,), dtype="U1"),
        "seed": np.zeros((0,), dtype=np.int32),
        "n_agents": np.zeros((0,), dtype=np.int32),
        "comm_profile": np.zeros((0,), dtype="U1"),
        "t_sec": np.zeros((0,), dtype=np.float32),
        "next_t_sec": np.zeros((0,), dtype=np.float32),
        "goal_dist_m": np.zeros((0,), dtype=np.float32),
        "next_goal_dist_m": np.zeros((0,), dtype=np.float32),
        "collision": np.zeros((0,), dtype=bool),
        "near_miss": np.zeros((0,), dtype=bool),
        "min_sep_m": np.zeros((0,), dtype=np.float32),
    }


def _samples_to_arrays(samples: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    if not samples:
        return _empty_arrays()

    max_obs_dim = max(int(np.asarray(sample["observation"]).shape[0]) for sample in samples)
    max_next_dim = max(int(np.asarray(sample["next_observation"]).shape[0]) for sample in samples)
    obs_dim = max(max_obs_dim, max_next_dim)
    observations = np.zeros((len(samples), obs_dim), dtype=np.float32)
    next_observations = np.zeros((len(samples), obs_dim), dtype=np.float32)
    obs_valid_dim = np.zeros((len(samples),), dtype=np.int32)
    next_obs_valid_dim = np.zeros((len(samples),), dtype=np.int32)
    for idx, sample in enumerate(samples):
        obs = np.asarray(sample["observation"], dtype=np.float32).reshape(-1)
        next_obs = np.asarray(sample["next_observation"], dtype=np.float32).reshape(-1)
        observations[idx, : obs.shape[0]] = obs
        next_observations[idx, : next_obs.shape[0]] = next_obs
        obs_valid_dim[idx] = int(obs.shape[0])
        next_obs_valid_dim[idx] = int(next_obs.shape[0])

    return {
        "observations": observations,
        "next_observations": next_observations,
        "observation_valid_dim": obs_valid_dim,
        "next_observation_valid_dim": next_obs_valid_dim,
        "actions": np.stack([np.asarray(sample["action"], dtype=np.float32) for sample in samples], axis=0),
        "rewards": np.asarray([float(sample["reward"]) for sample in samples], dtype=np.float32),
        "terminated": np.asarray([bool(sample["terminated"]) for sample in samples], dtype=bool),
        "truncated": np.asarray([bool(sample["truncated"]) for sample in samples], dtype=bool),
        "done": np.asarray([bool(sample["done"]) for sample in samples], dtype=bool),
        "episode_id": np.asarray([int(sample["episode_id"]) for sample in samples], dtype=np.int32),
        "step": np.asarray([int(sample["step"]) for sample in samples], dtype=np.int32),
        "agent_id": np.asarray([int(sample["agent_id"]) for sample in samples], dtype=np.int32),
        "lane_id": np.asarray([str(sample["lane_id"]) for sample in samples], dtype="U64"),
        "category": np.asarray([str(sample["category"]) for sample in samples], dtype="U64"),
        "scenario": np.asarray([str(sample["scenario"]) for sample in samples], dtype="U128"),
        "seed": np.asarray([int(sample["seed"]) for sample in samples], dtype=np.int32),
        "n_agents": np.asarray([int(sample["n_agents"]) for sample in samples], dtype=np.int32),
        "comm_profile": np.asarray([str(sample["comm_profile"]) for sample in samples], dtype="U64"),
        "t_sec": np.asarray([float(sample["t_sec"]) for sample in samples], dtype=np.float32),
        "next_t_sec": np.asarray([float(sample["next_t_sec"]) for sample in samples], dtype=np.float32),
        "goal_dist_m": np.asarray([float(sample["goal_dist_m"]) for sample in samples], dtype=np.float32),
        "next_goal_dist_m": np.asarray([float(sample["next_goal_dist_m"]) for sample in samples], dtype=np.float32),
        "collision": np.asarray([bool(sample["collision"]) for sample in samples], dtype=bool),
        "near_miss": np.asarray([bool(sample["near_miss"]) for sample in samples], dtype=bool),
        "min_sep_m": np.asarray([float(sample["min_sep_m"]) for sample in samples], dtype=np.float32),
    }


def _write_episode_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(LEARNED_DATASET_EPISODE_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in LEARNED_DATASET_EPISODE_FIELDS})
    return path


def _write_shards(*, out_dir: Path, arrays: dict[str, np.ndarray], shard_size: int) -> list[str]:
    shard_root = out_dir / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    sample_count = int(arrays["actions"].shape[0])
    if sample_count <= 0:
        return []
    size = max(1, int(shard_size))
    shards: list[str] = []
    for shard_idx, start in enumerate(range(0, sample_count, size)):
        end = min(sample_count, start + size)
        path = shard_root / f"shard_{shard_idx:05d}.npz"
        payload = {key: value[start:end] for key, value in arrays.items()}
        np.savez_compressed(path, **payload)
        shards.append(str(path))
    return shards


def _collect_episode(
    *,
    episode_id: int,
    lane: ValidationLane,
    scenario_path: Path,
    seed: int,
    policy_factory: Callable[[int], RlPolicy],
    action_source: str,
    policy_name: str,
    max_steps: int | None,
    save_replay: bool,
    replay_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    env = DaaParallelEnv(
        scenario_path=str(scenario_path),
        n_agents=int(lane.n_agents),
        seed=int(seed),
        comm_profile=str(lane.comm_profile),
    )
    policy_obj = policy_factory(int(seed))
    if hasattr(policy_obj, "reset"):
        policy_obj.reset(int(seed))

    samples: list[dict[str, Any]] = []
    replay_path: Path | None = None
    replay_fh = None
    steps = 0
    collision_ticks = 0
    near_miss_ticks = 0
    final_min_sep = float("nan")
    api_error = ""
    completed_agents = 0
    controlled_agents = 0
    dimension = "unknown"

    try:
        observations, infos = env.reset(seed=int(seed))
        controlled_agents = len(env.agents)
        dimension = "2d" if env.planar else "3d"
        if save_replay:
            replay_path = replay_dir / f"episode_{episode_id:05d}_{lane.lane_id}_seed{int(seed)}.jsonl"
            replay_path.parent.mkdir(parents=True, exist_ok=True)
            replay_fh = replay_path.open("w", encoding="utf-8")
            replay_fh.write(
                json.dumps(
                    {
                        "kind": "meta",
                        "trace_type": "learned_dataset_replay",
                        "episode_id": int(episode_id),
                        "lane_id": lane.lane_id,
                        "scenario": Path(str(scenario_path)).stem,
                        "seed": int(seed),
                        "n_agents": int(lane.n_agents),
                        "comm_profile": lane.comm_profile,
                        "action_source": action_source,
                        "policy": policy_name,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

        step_limit = env.episode_step_limit if env.episode_step_limit is not None else 0
        cap = int(max_steps) if max_steps is not None else int(step_limit)
        while env.agents and steps < cap:
            current_agents = list(env.agents)
            prev_observations = {agent: np.asarray(observations[agent], dtype=np.float32).copy() for agent in current_agents}
            prev_infos = {agent: dict(infos.get(agent, {})) for agent in current_agents}
            actions = {
                agent: _safe_action(policy_obj, agent, prev_observations[agent], env.action_space(agent), prev_infos[agent])
                for agent in current_agents
            }
            next_observations, rewards, terminations, truncations, infos = env.step(actions)
            frame = env.render()
            if replay_fh is not None and frame is not None:
                replay_fh.write(
                    json.dumps(
                        {"kind": "frame", "episode_id": int(episode_id), "lane_id": lane.lane_id, "step": int(steps), **frame},
                        default=_json_default,
                    )
                    + "\n"
                )

            if any(bool(info.get("collision", False)) for info in infos.values()):
                collision_ticks += 1
            if any(bool(info.get("near_miss", False)) for info in infos.values()):
                near_miss_ticks += 1
            min_sep_values = [float(info.get("min_sep_m")) for info in infos.values() if info.get("min_sep_m") is not None]
            if min_sep_values:
                final_min_sep = min(min_sep_values)

            for agent in current_agents:
                idx = agent_id_from_name(agent)
                next_obs = np.asarray(next_observations[agent], dtype=np.float32).copy()
                info = dict(infos.get(agent, {}))
                terminated = bool(terminations.get(agent, False))
                truncated = bool(truncations.get(agent, False))
                completed_agents += 1 if bool(info.get("done", False)) and terminated else 0
                samples.append(
                    {
                        "episode_id": int(episode_id),
                        "step": int(steps),
                        "agent_id": int(idx),
                        "lane_id": lane.lane_id,
                        "category": lane.category,
                        "scenario": Path(str(scenario_path)).stem,
                        "seed": int(seed),
                        "n_agents": int(lane.n_agents),
                        "comm_profile": lane.comm_profile,
                        "observation": prev_observations[agent],
                        "action": actions[agent],
                        "next_observation": next_obs,
                        "reward": float(rewards.get(agent, 0.0)),
                        "terminated": terminated,
                        "truncated": truncated,
                        "done": bool(terminated or truncated),
                        "t_sec": float(prev_observations[agent][11]) if prev_observations[agent].shape[0] > 11 else float(steps),
                        "next_t_sec": float(next_obs[11]) if next_obs.shape[0] > 11 else float(steps + 1),
                        "goal_dist_m": float(prev_observations[agent][9]) if prev_observations[agent].shape[0] > 9 else 0.0,
                        "next_goal_dist_m": float(next_obs[9]) if next_obs.shape[0] > 9 else 0.0,
                        "collision": bool(info.get("collision", False)),
                        "near_miss": bool(info.get("near_miss", False)),
                        "min_sep_m": float(info.get("min_sep_m", final_min_sep)),
                    }
                )
            observations = next_observations
            steps += 1
    except Exception as exc:  # pragma: no cover - exercised by CLI failure paths.
        api_error = f"{type(exc).__name__}: {exc}"
    finally:
        if replay_fh is not None:
            replay_fh.close()
        env.close()

    sample_count = len(samples)
    episode_row = {
        "episode_id": int(episode_id),
        "lane_id": lane.lane_id,
        "category": lane.category,
        "suite": lane.suite,
        "scenario": Path(str(scenario_path)).stem,
        "scenario_path": str(scenario_path),
        "dimension": dimension,
        "action_source": action_source,
        "policy": policy_name,
        "n_agents": int(lane.n_agents),
        "seed": int(seed),
        "comm_profile": lane.comm_profile,
        "steps": int(steps),
        "sample_count": int(sample_count),
        "controlled_agents": int(controlled_agents),
        "completion_rate": float(completed_agents / max(1, controlled_agents)),
        "collision_ticks": int(collision_ticks),
        "near_miss_ticks": int(near_miss_ticks),
        "final_min_sep_m": final_min_sep,
        "replay_path": "" if replay_path is None else str(replay_path),
        "api_error": api_error,
    }
    return samples, episode_row


def _collect_planner_expert_episode(
    *,
    episode_id: int,
    lane: ValidationLane,
    scenario_path: Path,
    seed: int,
    planner_expert: str,
    max_steps: int | None,
    save_replay: bool,
    replay_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    engine = EpisodeEngine(
        scenario_path=str(scenario_path),
        method=str(planner_expert),
        n_agents=int(lane.n_agents),
        seed=int(seed),
        comm_profile=str(lane.comm_profile),
    )

    samples: list[dict[str, Any]] = []
    replay_path: Path | None = None
    replay_fh = None
    steps = 0
    collision_ticks = 0
    near_miss_ticks = 0
    final_min_sep = float("nan")
    api_error = ""
    completed_agents = 0
    controlled_agents = int(engine.n_agents)
    dimension = "2d" if engine.planar else "3d"

    try:
        if save_replay:
            replay_path = replay_dir / f"episode_{episode_id:05d}_{lane.lane_id}_seed{int(seed)}.jsonl"
            replay_path.parent.mkdir(parents=True, exist_ok=True)
            replay_fh = replay_path.open("w", encoding="utf-8")
            replay_fh.write(
                json.dumps(
                    {
                        "kind": "meta",
                        "trace_type": "learned_dataset_replay",
                        "episode_id": int(episode_id),
                        "lane_id": lane.lane_id,
                        "scenario": Path(str(scenario_path)).stem,
                        "seed": int(seed),
                        "n_agents": int(lane.n_agents),
                        "comm_profile": lane.comm_profile,
                        "action_source": LEARNED_DATASET_PLANNER_EXPERT_SOURCE,
                        "policy": planner_expert,
                        "planner_expert": planner_expert,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

        cap = int(max_steps) if max_steps is not None else int(engine.steps)
        top_k = int(engine.neighbor_cfg.get("top_k", 8))
        while not engine.done() and steps < cap:
            step = engine.step()
            if step is None:
                break
            if replay_fh is not None:
                replay_fh.write(
                    json.dumps(
                        {"kind": "frame", "episode_id": int(episode_id), "lane_id": lane.lane_id, "step": int(steps), **step.trace_frame()},
                        default=_json_default,
                    )
                    + "\n"
                )

            if step.collisions > 0:
                collision_ticks += 1
            if step.near_misses > 0:
                near_miss_ticks += 1
            final_min_sep = float(step.min_sep)
            collision_agents = {idx for pair in step.collision_pairs for idx in pair}
            near_agents = {idx for pair in step.near_miss_pairs for idx in pair}
            horizon_truncated = bool(engine.k >= engine.steps)

            for idx, state in enumerate(step.planner_states):
                if bool(state.done):
                    continue
                post_state = step.states[idx]
                context = engine.agent_contexts[idx]
                selected_obs = list(step.selected_obs[idx]) if idx < len(step.selected_obs) else []
                obs = observation_from_public_snapshot(
                    state=state,
                    selected_obs=selected_obs,
                    agent_id=int(idx),
                    n_agents=int(engine.n_agents),
                    top_k=top_k,
                    t=float(step.t),
                    priority=int(context.priority),
                )
                next_obs = observation_from_public_snapshot(
                    state=post_state,
                    selected_obs=selected_obs,
                    agent_id=int(idx),
                    n_agents=int(engine.n_agents),
                    top_k=top_k,
                    t=float(step.t + engine.dt),
                    priority=int(context.priority),
                )
                action = _action_from_planner_v_cmd(
                    np.asarray(step.v_cmds[idx], dtype=np.float32),
                    v_max=float(state.v_max),
                    planar=bool(engine.planar),
                )
                previous_goal_dist = float(np.linalg.norm(np.asarray(state.goal) - np.asarray(state.pos)))
                next_goal_dist = float(np.linalg.norm(np.asarray(post_state.goal) - np.asarray(post_state.pos)))
                newly_done = bool(post_state.done and not state.done)
                collision = int(idx) in collision_agents
                near_miss = int(idx) in near_agents
                reward = (
                    float(DEFAULT_REWARD_WEIGHTS["progress"]) * (previous_goal_dist - next_goal_dist)
                    + float(DEFAULT_REWARD_WEIGHTS["time"])
                    + (float(DEFAULT_REWARD_WEIGHTS["goal"]) if newly_done else 0.0)
                    + (float(DEFAULT_REWARD_WEIGHTS["collision"]) if collision else 0.0)
                    + (float(DEFAULT_REWARD_WEIGHTS["near_miss"]) if near_miss and not collision else 0.0)
                )
                terminated = bool(post_state.done)
                truncated = bool(horizon_truncated and not terminated)
                completed_agents += 1 if newly_done and terminated else 0
                samples.append(
                    {
                        "episode_id": int(episode_id),
                        "step": int(steps),
                        "agent_id": int(idx),
                        "lane_id": lane.lane_id,
                        "category": lane.category,
                        "scenario": Path(str(scenario_path)).stem,
                        "seed": int(seed),
                        "n_agents": int(lane.n_agents),
                        "comm_profile": lane.comm_profile,
                        "observation": obs,
                        "action": action,
                        "next_observation": next_obs,
                        "reward": float(reward),
                        "terminated": terminated,
                        "truncated": truncated,
                        "done": bool(terminated or truncated),
                        "t_sec": float(step.t),
                        "next_t_sec": float(step.t + engine.dt),
                        "goal_dist_m": previous_goal_dist,
                        "next_goal_dist_m": next_goal_dist,
                        "collision": collision,
                        "near_miss": near_miss,
                        "min_sep_m": float(step.min_sep),
                    }
                )
            steps += 1
    except Exception as exc:  # pragma: no cover - exercised by CLI failure paths.
        api_error = f"{type(exc).__name__}: {exc}"
    finally:
        if replay_fh is not None:
            replay_fh.close()
        engine.close()

    sample_count = len(samples)
    episode_row = {
        "episode_id": int(episode_id),
        "lane_id": lane.lane_id,
        "category": lane.category,
        "suite": lane.suite,
        "scenario": Path(str(scenario_path)).stem,
        "scenario_path": str(scenario_path),
        "dimension": dimension,
        "action_source": LEARNED_DATASET_PLANNER_EXPERT_SOURCE,
        "policy": planner_expert,
        "n_agents": int(lane.n_agents),
        "seed": int(seed),
        "comm_profile": lane.comm_profile,
        "steps": int(steps),
        "sample_count": int(sample_count),
        "controlled_agents": int(controlled_agents),
        "completion_rate": float(completed_agents / max(1, controlled_agents)),
        "collision_ticks": int(collision_ticks),
        "near_miss_ticks": int(near_miss_ticks),
        "final_min_sep_m": final_min_sep,
        "replay_path": "" if replay_path is None else str(replay_path),
        "api_error": api_error,
    }
    return samples, episode_row


def export_learned_policy_dataset(
    *,
    out_dir: str | Path,
    policy: str = LEARNED_DATASET_TEACHER_POLICY,
    policy_spec: str | Path | None = None,
    planner_expert: str | None = None,
    lanes: tuple[str, ...] | list[str] | None = None,
    seeds: tuple[int, ...] | list[int] | None = None,
    duration_s: float | None = None,
    n_agents: int | None = None,
    max_steps: int | None = None,
    shard_size: int = 50000,
    save_replay: bool = False,
    overwrite: bool = False,
    plan_only: bool = False,
) -> dict[str, Any]:
    """Export public RL observation/action samples for learned-policy training."""

    out = Path(out_dir)
    manifest_path = out / "learned_dataset_manifest.json"
    if manifest_path.exists() and not bool(overwrite):
        raise RuntimeError(f"learned dataset output already exists: {manifest_path}")
    if bool(overwrite) and out.exists():
        for name in ("shards", "replay", "learned_dataset_manifest.json", "learned_dataset_episodes.csv"):
            path = out / name
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
    out.mkdir(parents=True, exist_ok=True)

    selected = _with_overrides(
        selected_learned_dataset_lanes(lanes),
        duration_s=duration_s,
        n_agents=n_agents,
    )
    seed_override = None if seeds is None else [int(seed) for seed in seeds]
    planned = [
        {
            "episode_id": idx,
            "lane_id": lane.lane_id,
            "category": lane.category,
            "suite": lane.suite,
            "scenario": lane.scenario,
            "comm_profile": lane.comm_profile,
            "n_agents": int(lane.n_agents),
            "seed": int(seed),
            "duration_s": float(lane.duration_s),
        }
        for idx, (lane, seed) in enumerate(
            (lane, seed)
            for lane in selected
            for seed in _seed_list_for_lane(lane, seed_override)
        )
    ]
    planner_expert_name = _planner_expert_name(planner_expert)
    if planner_expert_name is not None:
        if policy_spec is not None:
            raise ValueError("planner_expert cannot be combined with policy_spec")
        policy_factory = None
        action_source = LEARNED_DATASET_PLANNER_EXPERT_SOURCE
        policy_name = planner_expert_name
        policy_spec_summary = None
    else:
        policy_factory, action_source, policy_name, policy_spec_summary = _policy_factory(policy=policy, policy_spec=policy_spec)
    privileged_label_source = _planner_expert_uses_privileged_state(planner_expert_name)

    if plan_only:
        report = {
            "schema_version": LEARNED_DATASET_SCHEMA_VERSION,
            "interface_version": RL_INTERFACE_VERSION,
            "action_schema_version": RL_ACTION_SCHEMA_VERSION,
            "observation_schema_version": RL_OBSERVATION_SCHEMA_VERSION,
            "reward_schema_version": RL_REWARD_SCHEMA_VERSION,
            "plan_only": True,
            "ok": False,
            "out_dir": str(out),
            "action_source": action_source,
            "policy": policy_name,
            "planner_expert": planner_expert_name,
            "privileged_label_source": privileged_label_source,
            "policy_spec": policy_spec_summary,
            "lanes": [asdict(lane) for lane in selected],
            "extra_lane_ids": list(LEARNED_DATASET_EXTRA_LANE_IDS),
            "planned_episode_count": len(planned),
            "episode_count": 0,
            "sample_count": 0,
            "matrix": planned,
            "checks": [],
            "shards": [],
            "episodes_csv": None,
        }
        _write_json(manifest_path, report)
        return report

    scenario_paths = prepare_validation_lane_scenarios(out_dir=out, lanes=selected)
    lane_by_id = {lane.lane_id: lane for lane in selected}
    all_samples: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    for entry in planned:
        lane = lane_by_id[str(entry["lane_id"])]
        if planner_expert_name is not None:
            samples, episode_row = _collect_planner_expert_episode(
                episode_id=int(entry["episode_id"]),
                lane=lane,
                scenario_path=scenario_paths[lane.lane_id],
                seed=int(entry["seed"]),
                planner_expert=planner_expert_name,
                max_steps=max_steps,
                save_replay=bool(save_replay),
                replay_dir=out / "replay",
            )
        else:
            if policy_factory is None:
                raise RuntimeError("dataset policy factory was not initialized")
            samples, episode_row = _collect_episode(
                episode_id=int(entry["episode_id"]),
                lane=lane,
                scenario_path=scenario_paths[lane.lane_id],
                seed=int(entry["seed"]),
                policy_factory=policy_factory,
                action_source=action_source,
                policy_name=policy_name,
                max_steps=max_steps,
                save_replay=bool(save_replay),
                replay_dir=out / "replay",
            )
        all_samples.extend(samples)
        episode_rows.append(episode_row)

    arrays = _samples_to_arrays(all_samples)
    shards = _write_shards(out_dir=out, arrays=arrays, shard_size=int(shard_size))
    episodes_csv = _write_episode_csv(out / "learned_dataset_episodes.csv", episode_rows)
    sample_count = int(arrays["actions"].shape[0])
    obs_dim = int(arrays["observations"].shape[1]) if arrays["observations"].ndim == 2 else 0
    finite_arrays = bool(
        np.all(np.isfinite(arrays["observations"]))
        and np.all(np.isfinite(arrays["next_observations"]))
        and np.all(np.isfinite(arrays["actions"]))
        and np.all(np.isfinite(arrays["rewards"]))
    )
    episode_errors = [row for row in episode_rows if str(row.get("api_error") or "")]
    checks = [
        {"name": "episodes_ran", "ok": len(episode_rows) == len(planned), "details": {"episodes": len(episode_rows), "planned": len(planned)}},
        {"name": "samples_collected", "ok": sample_count > 0, "details": {"sample_count": sample_count}},
        {"name": "finite_arrays", "ok": finite_arrays, "details": {"observation_dim": obs_dim}},
        {"name": "shards_written", "ok": bool(shards) or sample_count == 0, "details": {"shard_count": len(shards)}},
        {"name": "episode_errors_clear", "ok": not episode_errors, "details": {"errors": episode_errors[:5]}},
    ]

    report = {
        "schema_version": LEARNED_DATASET_SCHEMA_VERSION,
        "interface_version": RL_INTERFACE_VERSION,
        "action_schema_version": RL_ACTION_SCHEMA_VERSION,
        "observation_schema_version": RL_OBSERVATION_SCHEMA_VERSION,
        "reward_schema_version": RL_REWARD_SCHEMA_VERSION,
        "plan_only": False,
        "ok": all(check["ok"] for check in checks),
        "out_dir": str(out),
        "manifest": str(manifest_path),
        "action_source": action_source,
        "policy": policy_name,
        "planner_expert": planner_expert_name,
        "privileged_label_source": privileged_label_source,
        "policy_spec": policy_spec_summary,
        "teacher_policy": BC_TEACHER_NAME if action_source == "bc_teacher" else None,
        "public_observations_only": True,
        "privileged_global_state": False,
        "lanes": [asdict(lane) for lane in selected],
        "extra_lane_ids": list(LEARNED_DATASET_EXTRA_LANE_IDS),
        "planned_episode_count": len(planned),
        "episode_count": len(episode_rows),
        "sample_count": sample_count,
        "observation_dim": obs_dim,
        "action_dim": 3,
        "max_steps": None if max_steps is None else int(max_steps),
        "shard_size": int(shard_size),
        "shards": shards,
        "episodes_csv": str(episodes_csv),
        "replay_dir": str(out / "replay") if save_replay else None,
        "matrix": planned,
        "episodes": episode_rows,
        "dataset_arrays": sorted(arrays.keys()),
        "interface_contract": interface_contract(top_k=max(0, (obs_dim - 17) // 9)),
        "checks": checks,
    }
    _write_json(manifest_path, report)
    return report
