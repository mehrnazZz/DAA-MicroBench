from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
import json
from pathlib import Path
from typing import Any

import numpy as np

from microbench.learned.tiny_linear import (
    LEARNED_BASELINE_SCHEMA_VERSION,
    OBS_BASE_DIM,
    OBS_NEIGHBOR_DIM,
    TINY_LEARNED_FEATURE_NAMES,
    observation_to_tiny_features,
    planner_input_to_tiny_features,
)
from microbench.types import PlannerInput


MLP_LEARNED_MODEL_ID = "mlp_goal_avoidance_v0"
MLP_LEARNED_POLICY_NAME = "mlp_learned"
MLP_LEARNED_FEATURE_NAMES = TINY_LEARNED_FEATURE_NAMES


def _mlp_model_resource():
    return resources.files("microbench").joinpath("bundled_config", "learned_baselines", "mlp_policy.json")


def mlp_learned_model_path() -> str:
    return str(_mlp_model_resource())


def load_mlp_learned_spec(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        text = _mlp_model_resource().read_text(encoding="utf-8")
        spec = json.loads(text)
    else:
        spec = json.loads(Path(path).read_text(encoding="utf-8"))
    if spec.get("schema_version") != LEARNED_BASELINE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported learned baseline schema {spec.get('schema_version')!r}; "
            f"expected {LEARNED_BASELINE_SCHEMA_VERSION!r}"
        )
    if spec.get("model_id") != MLP_LEARNED_MODEL_ID:
        raise ValueError(f"Unsupported MLP learned model id {spec.get('model_id')!r}")
    if tuple(spec.get("input_features", ())) != MLP_LEARNED_FEATURE_NAMES:
        raise ValueError("MLP learned baseline feature list does not match the public learned-policy contract")
    return spec


def planner_input_to_mlp_features(planner_input: PlannerInput, *, max_neighbors: int = 8) -> np.ndarray:
    return planner_input_to_tiny_features(planner_input, max_neighbors=max_neighbors)


def observation_to_mlp_features(observation: np.ndarray, *, top_k: int = 8) -> np.ndarray:
    return observation_to_tiny_features(observation, top_k=top_k)


def _normalize(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-9:
        return np.zeros(3, dtype=np.float32)
    return (arr / norm).astype(np.float32)


def _normalized_model_features(features: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    payload = spec.get("feature_normalization", {})
    if not isinstance(payload, dict) or str(payload.get("mode", "none")) == "none":
        return np.asarray(features, dtype=np.float32)
    if str(payload.get("mode")) != "standard":
        raise ValueError(f"Unsupported MLP feature normalization mode {payload.get('mode')!r}")
    x = np.asarray(features, dtype=np.float32).reshape(-1)
    mean = np.asarray(payload.get("mean"), dtype=np.float32).reshape(-1)
    scale = np.asarray(payload.get("scale"), dtype=np.float32).reshape(-1)
    if mean.shape != x.shape or scale.shape != x.shape:
        raise ValueError(
            f"MLP feature normalization shape mismatch: mean={mean.shape}, scale={scale.shape}, features={x.shape}"
        )
    normalized = (x - mean) / np.maximum(scale, 1e-6)
    clip_value = payload.get("clip")
    if clip_value is not None:
        normalized = np.clip(normalized, -float(clip_value), float(clip_value))
    return normalized.astype(np.float32)


@dataclass
class FrozenMlpPolicyModel:
    spec: dict[str, Any]

    @classmethod
    def from_path(cls, path: str | Path | None = None) -> "FrozenMlpPolicyModel":
        return cls(load_mlp_learned_spec(path))

    @property
    def model_id(self) -> str:
        return str(self.spec["model_id"])

    @property
    def training_metadata(self) -> dict[str, Any]:
        return dict(self.spec.get("training", {}))

    @property
    def hidden_dim(self) -> int:
        return int(self.spec["hidden_dim"])

    def _postprocess_action(self, action: np.ndarray, features: np.ndarray) -> np.ndarray:
        postprocess = self.spec.get("postprocess", {})
        if not isinstance(postprocess, dict):
            return np.asarray(action, dtype=np.float32)

        out = np.asarray(action, dtype=np.float32).reshape(3)
        goal = _normalize(np.asarray(features[0:3], dtype=np.float32))
        if bool(postprocess.get("goal_forward_floor", False)) and float(np.linalg.norm(goal)) > 1e-9:
            threat = float(np.clip(features[-2], 0.0, 1.0))
            base = float(postprocess.get("min_forward_base", 0.2))
            free_boost = float(postprocess.get("min_forward_free_boost", 0.25))
            min_forward = base + free_boost * (1.0 - threat)
            forward = float(np.dot(out, goal))
            if forward < min_forward:
                out += (min_forward - forward) * goal

        if bool(postprocess.get("normalize_max_norm", False)):
            norm = float(np.linalg.norm(out))
            if norm > 1.0:
                out = out / norm
        return np.clip(out, -1.0, 1.0).astype(np.float32)

    def action_from_features(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float32).reshape(-1)
        x_model = _normalized_model_features(x, self.spec)
        w1 = np.asarray(self.spec["layer1_weights"], dtype=np.float32)
        b1 = np.asarray(self.spec["layer1_bias"], dtype=np.float32)
        w2 = np.asarray(self.spec["layer2_weights"], dtype=np.float32)
        b2 = np.asarray(self.spec["layer2_bias"], dtype=np.float32)
        if x.shape != (len(MLP_LEARNED_FEATURE_NAMES),):
            raise ValueError(f"MLP learned features must have shape {(len(MLP_LEARNED_FEATURE_NAMES),)}, got {x.shape}")
        if w1.shape != (self.hidden_dim, len(MLP_LEARNED_FEATURE_NAMES)):
            raise ValueError(
                f"MLP layer1 weights have shape {w1.shape}, "
                f"expected {(self.hidden_dim, len(MLP_LEARNED_FEATURE_NAMES))}"
            )
        if b1.shape != (self.hidden_dim,):
            raise ValueError(f"MLP layer1 bias has shape {b1.shape}, expected {(self.hidden_dim,)}")
        if w2.shape != (3, self.hidden_dim):
            raise ValueError(f"MLP layer2 weights have shape {w2.shape}, expected {(3, self.hidden_dim)}")
        if b2.shape != (3,):
            raise ValueError(f"MLP layer2 bias has shape {b2.shape}, expected (3,)")
        hidden = np.tanh(w1 @ x_model + b1)
        raw = w2 @ hidden + b2
        return self._postprocess_action(np.tanh(raw), x)

    def action_from_planner_input(self, planner_input: PlannerInput, *, max_neighbors: int = 8) -> np.ndarray:
        return self.action_from_features(planner_input_to_mlp_features(planner_input, max_neighbors=max_neighbors))

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        _ = deterministic
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        top_k = max(0, (obs.shape[0] - OBS_BASE_DIM) // OBS_NEIGHBOR_DIM)
        return self.action_from_features(observation_to_mlp_features(obs, top_k=top_k)), None
