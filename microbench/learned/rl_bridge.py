from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from microbench.learned.tiny_linear import OBS_BASE_DIM, OBS_NEIGHBOR_DIM
from microbench.types import PlannerInput


AGENT_NAME_PREFIX = "agent_"


def agent_name(agent_id: int) -> str:
    return f"{AGENT_NAME_PREFIX}{int(agent_id)}"


def _normalize(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-9:
        return np.zeros(3, dtype=np.float32)
    return (arr / norm).astype(np.float32)


def _coerce_action(action: Any, action_space: Any | None = None, *, clip: bool = True) -> np.ndarray:
    if isinstance(action, tuple):
        action = action[0]
    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"RL policy action must have shape (3,), got {arr.shape}")
    if not bool(np.all(np.isfinite(arr))):
        raise ValueError("RL policy action must contain only finite values")
    if clip:
        low = getattr(action_space, "low", np.full((3,), -1.0, dtype=np.float32))
        high = getattr(action_space, "high", np.full((3,), 1.0, dtype=np.float32))
        arr = np.clip(arr, np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32))
    return arr.astype(np.float32, copy=False)


def clamp_normalized_velocity_action(action: Any, action_space: Any | None = None) -> np.ndarray:
    return _coerce_action(action, action_space=action_space, clip=True)


def clamp_velocity(v_cmd: np.ndarray, v_max: float) -> np.ndarray:
    arr = np.asarray(v_cmd, dtype=np.float32)
    speed = float(np.linalg.norm(arr))
    if speed <= float(v_max) or speed < 1e-9:
        return arr
    return (arr / speed * float(v_max)).astype(np.float32)


def planner_input_to_rl_observation(
    planner_input: PlannerInput,
    *,
    top_k: int = 8,
    n_agents: int | None = None,
) -> np.ndarray:
    """Convert public PlannerInput into the stable RL local observation vector."""

    ego = planner_input.ego
    top = max(0, int(top_k))
    pos = np.asarray(ego.pos, dtype=float)
    vel = np.asarray(ego.vel, dtype=float)
    goal_delta = np.asarray(ego.goal, dtype=float) - pos
    goal_dist = float(np.linalg.norm(goal_delta))
    goal_dir = _normalize(goal_delta)
    context = planner_input.agent_context
    if n_agents is None:
        agent_id_norm = 0.0
    else:
        agent_id_norm = float(ego.idx) / max(1.0, float(int(n_agents) - 1))
    priority = int(context.priority) if context is not None else 0

    base = [
        *pos.tolist(),
        *vel.tolist(),
        *goal_dir.tolist(),
        goal_dist,
        1.0 if ego.done else 0.0,
        float(planner_input.t),
        agent_id_norm,
        float(priority),
        float(ego.radius),
        float(ego.v_max),
        float(ego.a_max),
    ]
    if len(base) != OBS_BASE_DIM:
        raise ValueError(f"RL observation base must have length {OBS_BASE_DIM}, got {len(base)}")

    features: list[float] = []
    for neighbor in list(planner_input.neighbors)[:top]:
        rel_pos = np.asarray(neighbor.pos, dtype=float) - pos
        rel_vel = np.asarray(neighbor.vel, dtype=float) - vel
        features.extend(
            [
                1.0,
                *rel_pos.tolist(),
                *rel_vel.tolist(),
                float(neighbor.radius),
                float(neighbor.msg_age_sec),
            ]
        )
    while len(features) < top * OBS_NEIGHBOR_DIM:
        features.extend([0.0] * OBS_NEIGHBOR_DIM)

    return np.asarray([*base, *features], dtype=np.float32)


def planner_input_to_rl_info(
    planner_input: PlannerInput,
    *,
    top_k: int = 8,
    n_agents: int | None = None,
    policy_name: str | None = None,
) -> dict[str, Any]:
    context = planner_input.agent_context
    return {
        "agent_id": int(planner_input.ego.idx),
        "method": None if context is None else str(context.method),
        "role": None if context is None else context.role,
        "priority": 0 if context is None else int(context.priority),
        "t": float(planner_input.t),
        "dt": float(planner_input.dt),
        "planar": bool(planner_input.planar),
        "neighbor_count": len(planner_input.neighbors),
        "observation_top_k": max(0, int(top_k)),
        "n_agents": None if n_agents is None else int(n_agents),
        "policy_name": policy_name,
    }


@dataclass(frozen=True)
class NormalizedVelocityActionSpace:
    low: np.ndarray
    high: np.ndarray
    shape: tuple[int, ...] = (3,)
    dtype: np.dtype = np.dtype(np.float32)


def normalized_velocity_action_space() -> NormalizedVelocityActionSpace:
    return NormalizedVelocityActionSpace(
        low=np.full((3,), -1.0, dtype=np.float32),
        high=np.full((3,), 1.0, dtype=np.float32),
    )
