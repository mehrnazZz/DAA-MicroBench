from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from microbench.learned import TINY_LEARNED_POLICY_NAME, TinyLinearPolicyModel
from microbench.rl.schema import OBS_GOAL_DIR_SLICE


class RlPolicy(Protocol):
    def reset(self, seed: int) -> None:
        ...

    def action(self, agent: str, observation: np.ndarray, action_space: Any, info: dict[str, Any]) -> np.ndarray:
        ...


@dataclass
class ZeroPolicy:
    """Hold-position policy for API smoke checks."""

    def reset(self, seed: int) -> None:
        _ = seed

    def action(self, agent: str, observation: np.ndarray, action_space: Any, info: dict[str, Any]) -> np.ndarray:
        _ = agent, observation, action_space, info
        return np.zeros(3, dtype=np.float32)


@dataclass
class RandomPolicy:
    """Deterministic random-action policy seeded through `reset`."""

    scale: float = 1.0
    _rng: np.random.Generator | None = None

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng(int(seed))

    def action(self, agent: str, observation: np.ndarray, action_space: Any, info: dict[str, Any]) -> np.ndarray:
        _ = agent, observation, action_space, info
        rng = self._rng if self._rng is not None else np.random.default_rng(0)
        scale = float(np.clip(self.scale, 0.0, 1.0))
        return rng.uniform(-scale, scale, size=3).astype(np.float32)


@dataclass
class GoalDirectionPolicy:
    """Drive along the local goal-direction features in the observation."""

    speed_fraction: float = 1.0

    def reset(self, seed: int) -> None:
        _ = seed

    def action(self, agent: str, observation: np.ndarray, action_space: Any, info: dict[str, Any]) -> np.ndarray:
        _ = agent, action_space, info
        goal_dir = np.asarray(observation[OBS_GOAL_DIR_SLICE], dtype=np.float32)
        norm = float(np.linalg.norm(goal_dir))
        if norm < 1e-9:
            return np.zeros(3, dtype=np.float32)
        scale = float(np.clip(self.speed_fraction, 0.0, 1.0))
        return (goal_dir / norm * scale).astype(np.float32)


@dataclass
class TinyLearnedPolicy:
    """Frozen tiny learned-policy fixture for adapter and baseline tests."""

    model: TinyLinearPolicyModel | None = None

    def __post_init__(self) -> None:
        if self.model is None:
            self.model = TinyLinearPolicyModel.from_path()

    def reset(self, seed: int) -> None:
        _ = seed

    def action(self, agent: str, observation: np.ndarray, action_space: Any, info: dict[str, Any]) -> np.ndarray:
        _ = agent, action_space, info
        assert self.model is not None
        action, _state = self.model.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float32)


POLICY_NAMES = ("zero", "random", "goal_direction", TINY_LEARNED_POLICY_NAME)


def make_policy(name: str, *, seed: int = 0) -> RlPolicy:
    key = str(name).strip().lower()
    if key == "zero":
        policy: RlPolicy = ZeroPolicy()
    elif key == "random":
        policy = RandomPolicy()
    elif key == "goal_direction":
        policy = GoalDirectionPolicy()
    elif key == TINY_LEARNED_POLICY_NAME:
        policy = TinyLearnedPolicy()
    else:
        raise ValueError(f"Unknown RL policy {name!r}; expected one of {','.join(POLICY_NAMES)}")
    policy.reset(int(seed))
    return policy
