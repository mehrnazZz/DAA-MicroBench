from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np

from microbench.learned.mlp_policy import (
    MLP_LEARNED_PUBLIC_OBS_FEATURE_SET,
    MLP_LEARNED_PUBLIC_OBS_TOP_K,
    mlp_feature_names,
    observation_to_mlp_features,
    planner_input_to_mlp_features,
)
from microbench.learned.tiny_linear import (
    LEARNED_BASELINE_SCHEMA_VERSION,
    observation_to_tiny_features,
)
from microbench.types import PlannerInput


TEMPORAL_MLP_LEARNED_MODEL_ID = "temporal_mlp_goal_avoidance_public_obs_v1"
TEMPORAL_MLP_POLICY_ADAPTER = "temporal_mlp_json"
TEMPORAL_MLP_MODEL_TYPE = "temporal_mlp_tanh_policy"
DEFAULT_TEMPORAL_MLP_HISTORY_LEN = 3


def _history_len(value: int | None) -> int:
    history_len = int(DEFAULT_TEMPORAL_MLP_HISTORY_LEN if value is None else value)
    if history_len < 1:
        raise ValueError("temporal MLP history_len must be at least 1")
    return history_len


def temporal_mlp_feature_names(
    *,
    base_feature_set: str = MLP_LEARNED_PUBLIC_OBS_FEATURE_SET,
    top_k: int = MLP_LEARNED_PUBLIC_OBS_TOP_K,
    history_len: int = DEFAULT_TEMPORAL_MLP_HISTORY_LEN,
) -> tuple[str, ...]:
    if str(base_feature_set) != MLP_LEARNED_PUBLIC_OBS_FEATURE_SET:
        raise ValueError("temporal MLP currently supports only public_obs_v1 base features")
    base_names = mlp_feature_names(base_feature_set, top_k=int(top_k))
    names: list[str] = []
    for offset in range(_history_len(history_len)):
        prefix = "t0" if offset == 0 else f"t_minus_{offset}"
        names.extend(f"{prefix}_{name}" for name in base_names)
    return tuple(names)


def stack_temporal_feature_rows(
    features: np.ndarray,
    *,
    episode_ids: np.ndarray,
    agent_ids: np.ndarray,
    steps: np.ndarray,
    history_len: int = DEFAULT_TEMPORAL_MLP_HISTORY_LEN,
) -> np.ndarray:
    """Build current-plus-history feature rows keyed by episode, agent, and step."""

    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"temporal feature stacking expects a 2D matrix, got shape {x.shape}")
    count = int(x.shape[0])
    episodes = np.asarray(episode_ids, dtype=np.int64).reshape(-1)
    agents = np.asarray(agent_ids, dtype=np.int64).reshape(-1)
    step_values = np.asarray(steps, dtype=np.int64).reshape(-1)
    if episodes.shape != (count,) or agents.shape != (count,) or step_values.shape != (count,):
        raise ValueError("temporal feature metadata arrays must match the feature row count")

    hist_len = _history_len(history_len)
    lookup: dict[tuple[int, int, int], np.ndarray] = {}
    for idx in range(count):
        lookup[(int(episodes[idx]), int(agents[idx]), int(step_values[idx]))] = x[idx]

    stacked = np.zeros((count, x.shape[1] * hist_len), dtype=np.float32)
    for idx in range(count):
        current = x[idx]
        chunks: list[np.ndarray] = []
        episode = int(episodes[idx])
        agent = int(agents[idx])
        step = int(step_values[idx])
        for offset in range(hist_len):
            chunks.append(lookup.get((episode, agent, step - offset), current))
        stacked[idx] = np.concatenate(chunks, axis=0)
    return stacked


def _normalize(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-9:
        return np.zeros(3, dtype=np.float32)
    return (arr / norm).astype(np.float32)


def _normalized_features(features: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
    payload = spec.get("feature_normalization", {})
    if not isinstance(payload, dict) or str(payload.get("mode", "none")) == "none":
        return np.asarray(features, dtype=np.float32)
    if str(payload.get("mode")) != "standard":
        raise ValueError(f"Unsupported temporal MLP feature normalization mode {payload.get('mode')!r}")
    x = np.asarray(features, dtype=np.float32).reshape(-1)
    mean = np.asarray(payload.get("mean"), dtype=np.float32).reshape(-1)
    scale = np.asarray(payload.get("scale"), dtype=np.float32).reshape(-1)
    if mean.shape != x.shape or scale.shape != x.shape:
        raise ValueError(
            f"Temporal MLP feature normalization shape mismatch: mean={mean.shape}, "
            f"scale={scale.shape}, features={x.shape}"
        )
    normalized = (x - mean) / np.maximum(scale, 1e-6)
    clip_value = payload.get("clip")
    if clip_value is not None:
        normalized = np.clip(normalized, -float(clip_value), float(clip_value))
    return normalized.astype(np.float32)


def load_temporal_mlp_learned_spec(path: str | Path) -> dict[str, Any]:
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    if spec.get("schema_version") != LEARNED_BASELINE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported learned baseline schema {spec.get('schema_version')!r}; "
            f"expected {LEARNED_BASELINE_SCHEMA_VERSION!r}"
        )
    if spec.get("model_id") != TEMPORAL_MLP_LEARNED_MODEL_ID:
        raise ValueError(
            f"Unsupported temporal MLP learned model id {spec.get('model_id')!r}; "
            f"expected {TEMPORAL_MLP_LEARNED_MODEL_ID!r}"
        )
    if str(spec.get("model_type")) != TEMPORAL_MLP_MODEL_TYPE:
        raise ValueError(f"Unsupported temporal MLP model type {spec.get('model_type')!r}")
    base_feature_set = str(spec.get("base_feature_set", MLP_LEARNED_PUBLIC_OBS_FEATURE_SET))
    feature_top_k = int(spec.get("feature_top_k", MLP_LEARNED_PUBLIC_OBS_TOP_K))
    history_len = _history_len(int(spec.get("history_len", DEFAULT_TEMPORAL_MLP_HISTORY_LEN)))
    expected = temporal_mlp_feature_names(
        base_feature_set=base_feature_set,
        top_k=feature_top_k,
        history_len=history_len,
    )
    if tuple(spec.get("input_features", ())) != expected:
        raise ValueError("Temporal MLP input_features do not match the declared public history contract")
    return spec


@dataclass
class FrozenTemporalMlpPolicyModel:
    spec: dict[str, Any]
    _history: dict[str, list[np.ndarray]] = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def from_path(cls, path: str | Path) -> "FrozenTemporalMlpPolicyModel":
        return cls(load_temporal_mlp_learned_spec(path))

    @property
    def model_id(self) -> str:
        return str(self.spec["model_id"])

    @property
    def base_feature_set(self) -> str:
        return str(self.spec.get("base_feature_set", MLP_LEARNED_PUBLIC_OBS_FEATURE_SET))

    @property
    def feature_top_k(self) -> int:
        return int(self.spec.get("feature_top_k", MLP_LEARNED_PUBLIC_OBS_TOP_K))

    @property
    def history_len(self) -> int:
        return _history_len(int(self.spec.get("history_len", DEFAULT_TEMPORAL_MLP_HISTORY_LEN)))

    @property
    def base_feature_dim(self) -> int:
        return len(mlp_feature_names(self.base_feature_set, top_k=self.feature_top_k))

    @property
    def feature_names(self) -> tuple[str, ...]:
        return temporal_mlp_feature_names(
            base_feature_set=self.base_feature_set,
            top_k=self.feature_top_k,
            history_len=self.history_len,
        )

    @property
    def training_metadata(self) -> dict[str, Any]:
        return dict(self.spec.get("training", {}))

    @property
    def hidden_dim(self) -> int:
        return int(self.spec["hidden_dim"])

    def reset(self, seed: int) -> None:
        _ = seed
        self._history.clear()

    def _base_features_from_observation(self, observation: np.ndarray) -> np.ndarray:
        return observation_to_mlp_features(
            np.asarray(observation, dtype=np.float32).reshape(-1),
            top_k=self.feature_top_k,
            feature_set=self.base_feature_set,
        )

    def _stack_for_agent(self, agent: str, current: np.ndarray) -> np.ndarray:
        key = str(agent or "__default__")
        previous = list(self._history.get(key, []))
        chunks = [np.asarray(current, dtype=np.float32)]
        chunks.extend(previous[: self.history_len - 1])
        while len(chunks) < self.history_len:
            chunks.append(np.asarray(current, dtype=np.float32))
        self._history[key] = [np.asarray(current, dtype=np.float32)] + previous[: self.history_len - 1]
        return np.concatenate(chunks, axis=0).astype(np.float32)

    def _postprocess_action(self, action: np.ndarray, stacked_features: np.ndarray) -> np.ndarray:
        postprocess = self.spec.get("postprocess", {})
        if not isinstance(postprocess, dict):
            return np.asarray(action, dtype=np.float32)

        out = np.asarray(action, dtype=np.float32).reshape(3)
        current = np.asarray(stacked_features[: self.base_feature_dim], dtype=np.float32)
        goal = _normalize(current[6:9])
        threat_features = observation_to_tiny_features(current, top_k=self.feature_top_k)
        threat = float(np.clip(threat_features[-2], 0.0, 1.0))

        if bool(postprocess.get("goal_forward_floor", False)) and float(np.linalg.norm(goal)) > 1e-9:
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
        x_model = _normalized_features(x, self.spec)
        w1 = np.asarray(self.spec["layer1_weights"], dtype=np.float32)
        b1 = np.asarray(self.spec["layer1_bias"], dtype=np.float32)
        w2 = np.asarray(self.spec["layer2_weights"], dtype=np.float32)
        b2 = np.asarray(self.spec["layer2_bias"], dtype=np.float32)
        feature_dim = len(self.feature_names)
        if x.shape != (feature_dim,):
            raise ValueError(f"Temporal MLP features must have shape {(feature_dim,)}, got {x.shape}")
        if w1.shape != (self.hidden_dim, feature_dim):
            raise ValueError(
                f"Temporal MLP layer1 weights have shape {w1.shape}, "
                f"expected {(self.hidden_dim, feature_dim)}"
            )
        if b1.shape != (self.hidden_dim,):
            raise ValueError(f"Temporal MLP layer1 bias has shape {b1.shape}, expected {(self.hidden_dim,)}")
        if w2.shape != (3, self.hidden_dim):
            raise ValueError(f"Temporal MLP layer2 weights have shape {w2.shape}, expected {(3, self.hidden_dim)}")
        if b2.shape != (3,):
            raise ValueError(f"Temporal MLP layer2 bias has shape {b2.shape}, expected (3,)")
        hidden = np.tanh(w1 @ x_model + b1)
        raw = w2 @ hidden + b2
        return self._postprocess_action(np.tanh(raw), x)

    def predict_for_agent(self, agent: str, observation: np.ndarray, deterministic: bool = True):
        _ = deterministic
        current = self._base_features_from_observation(observation)
        stacked = self._stack_for_agent(agent, current)
        return self.action_from_features(stacked), None

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        return self.predict_for_agent("__default__", observation, deterministic=deterministic)

    def action_from_planner_input(self, planner_input: PlannerInput, *, max_neighbors: int = 8) -> np.ndarray:
        _ = max_neighbors
        current = planner_input_to_mlp_features(
            planner_input,
            max_neighbors=self.feature_top_k,
            feature_set=self.base_feature_set,
        )
        stacked = self._stack_for_agent(f"agent_{int(planner_input.ego.idx)}", current)
        return self.action_from_features(stacked)
