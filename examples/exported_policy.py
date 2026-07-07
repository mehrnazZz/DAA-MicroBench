from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from microbench.rl.schema import (
    OBS_BASE_DIM,
    OBS_GOAL_DIR_SLICE,
    OBS_NEIGHBOR_DIM,
    OBS_NEIGHBOR_START,
)


DEFAULT_ARTIFACT = Path(__file__).with_name("exported_linear_policy.json")


def _normalize(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-9:
        return np.zeros(3, dtype=np.float32)
    return (arr / norm).astype(np.float32)


def _load_artifact(path: str | Path | None) -> dict[str, Any]:
    artifact_path = DEFAULT_ARTIFACT if path is None else Path(path)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "example-exported-linear-v0":
        raise ValueError(f"Unsupported example policy artifact schema: {payload.get('schema_version')!r}")
    return payload


class ExportedLinearPolicyModel:
    """Tiny dependency-free stand-in for an exported learned policy.

    Real adopters can replace this class with a wrapper around Torch,
    Stable-Baselines, RLlib, CleanRL, or a custom inference runtime. The public
    contract is only that `predict(...)` returns a finite normalized action with
    shape `(3,)`.
    """

    def __init__(
        self,
        artifact_path: str | Path | None = None,
        *,
        speed_fraction: float | None = None,
        avoid_gain: float | None = None,
    ):
        self.artifact_path = DEFAULT_ARTIFACT if artifact_path is None else Path(artifact_path)
        self.spec = _load_artifact(self.artifact_path)
        self.speed_fraction = float(speed_fraction if speed_fraction is not None else self.spec.get("speed_fraction", 0.8))
        self.avoid_gain = float(avoid_gain if avoid_gain is not None else self.spec.get("avoid_gain", 0.4))
        self.influence_radius_m = float(self.spec.get("neighbor_influence_radius_m", 4.0))
        self.seed = 0

    @property
    def policy_name(self) -> str:
        return str(self.spec.get("model_id", "example_exported_linear_policy"))

    def reset(self, seed: int) -> None:
        self.seed = int(seed)

    def set_seed(self, seed: int) -> None:
        self.reset(seed)

    def _avoidance(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        if obs.shape[0] < OBS_BASE_DIM:
            return np.zeros(3, dtype=np.float32)
        top_k = max(0, (obs.shape[0] - OBS_BASE_DIM) // OBS_NEIGHBOR_DIM)
        avoid = np.zeros(3, dtype=np.float32)
        for i in range(top_k):
            start = OBS_NEIGHBOR_START + i * OBS_NEIGHBOR_DIM
            block = obs[start : start + OBS_NEIGHBOR_DIM]
            if float(block[0]) <= 0.0:
                continue
            rel_pos = np.asarray(block[1:4], dtype=np.float32)
            distance = float(np.linalg.norm(rel_pos))
            if distance < 1e-6:
                avoid += np.asarray([-1.0, 0.0, 0.0], dtype=np.float32)
                continue
            pressure = float(np.clip((self.influence_radius_m - distance) / max(1e-6, self.influence_radius_m), 0.0, 1.0))
            avoid += (-rel_pos / distance * pressure).astype(np.float32)
        return avoid

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        _ = deterministic
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        goal = _normalize(obs[OBS_GOAL_DIR_SLICE])
        action = goal * float(np.clip(self.speed_fraction, 0.0, 1.0))
        action += _normalize(self._avoidance(obs)) * float(np.clip(self.avoid_gain, 0.0, 1.0))
        norm = float(np.linalg.norm(action))
        if norm > 1.0:
            action = action / norm
        return np.clip(action, -1.0, 1.0).astype(np.float32), None


def make_model(
    artifact_path: str | Path | None = None,
    *,
    speed_fraction: float | None = None,
    avoid_gain: float | None = None,
) -> ExportedLinearPolicyModel:
    return ExportedLinearPolicyModel(
        artifact_path=artifact_path,
        speed_fraction=speed_fraction,
        avoid_gain=avoid_gain,
    )


_CALLABLE_MODEL: ExportedLinearPolicyModel | None = None


def callable_policy(observation: np.ndarray, info: dict[str, Any] | None = None) -> np.ndarray:
    """Callable-adapter entry point shaped as f(observation, info) -> action."""

    _ = info
    global _CALLABLE_MODEL
    if _CALLABLE_MODEL is None:
        _CALLABLE_MODEL = ExportedLinearPolicyModel()
    action, _state = _CALLABLE_MODEL.predict(observation, deterministic=True)
    return action
