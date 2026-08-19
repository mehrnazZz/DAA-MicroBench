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

    def action_from_features(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float32).reshape(-1)
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
        hidden = np.tanh(w1 @ x + b1)
        raw = w2 @ hidden + b2
        return np.tanh(raw).astype(np.float32)

    def action_from_planner_input(self, planner_input: PlannerInput, *, max_neighbors: int = 8) -> np.ndarray:
        return self.action_from_features(planner_input_to_mlp_features(planner_input, max_neighbors=max_neighbors))

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        _ = deterministic
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        top_k = max(0, (obs.shape[0] - OBS_BASE_DIM) // OBS_NEIGHBOR_DIM)
        return self.action_from_features(observation_to_mlp_features(obs, top_k=top_k)), None
