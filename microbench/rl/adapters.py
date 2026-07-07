from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np


CallablePolicySignature = Literal["full", "observation", "observation_info"]


def normalize_action(action: Any, action_space: Any | None = None, *, clip: bool = True) -> np.ndarray:
    """Convert external model output into a finite normalized DAA action."""

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


@dataclass
class CallablePolicyAdapter:
    """Adapter for dependency-free Python callables.

    Use `signature="observation"` for functions shaped like
    `f(observation) -> action`, `signature="observation_info"` for
    `f(observation, info) -> action`, and `signature="full"` for
    `f(agent, observation, action_space, info) -> action`.
    """

    predict: Callable[..., Any]
    signature: CallablePolicySignature = "full"
    clip: bool = True

    def reset(self, seed: int) -> None:
        reset = getattr(self.predict, "reset", None)
        if callable(reset):
            reset(int(seed))

    def action(self, agent: str, observation: np.ndarray, action_space: Any, info: dict[str, Any]) -> np.ndarray:
        if self.signature == "full":
            raw = self.predict(agent, observation, action_space, info)
        elif self.signature == "observation":
            raw = self.predict(observation)
        elif self.signature == "observation_info":
            raw = self.predict(observation, info)
        else:
            raise ValueError(f"Unknown callable policy signature {self.signature!r}")
        return normalize_action(raw, action_space, clip=self.clip)


@dataclass
class ModelPredictPolicyAdapter:
    """Adapter for common trained-policy objects.

    The adapter accepts model objects that expose one of these interfaces:
    `compute_single_action(observation)`, `predict(observation, deterministic=...)`,
    `predict(observation)`, or a direct callable `model(observation)`.
    Tuple returns such as Stable-Baselines-style `(action, state)` are accepted.
    """

    model: Any
    deterministic: bool = True
    clip: bool = True

    def reset(self, seed: int) -> None:
        for method_name in ("reset", "set_seed", "seed"):
            method = getattr(self.model, method_name, None)
            if callable(method):
                method(int(seed))
                return

    def action(self, agent: str, observation: np.ndarray, action_space: Any, info: dict[str, Any]) -> np.ndarray:
        _ = agent, info
        if hasattr(self.model, "compute_single_action"):
            raw = self.model.compute_single_action(observation)
        elif hasattr(self.model, "predict"):
            try:
                raw = self.model.predict(observation, deterministic=bool(self.deterministic))
            except TypeError:
                raw = self.model.predict(observation)
        elif callable(self.model):
            raw = self.model(observation)
        else:
            raise TypeError("model must expose compute_single_action, predict, or be callable")
        return normalize_action(raw, action_space, clip=self.clip)
