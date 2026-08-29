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
from microbench.learned.rl_bridge import planner_input_to_rl_observation
from microbench.types import PlannerInput


MLP_LEARNED_MODEL_ID = "mlp_goal_avoidance_v0"
MLP_LEARNED_PUBLIC_OBS_MODEL_ID = "mlp_goal_avoidance_public_obs_v1"
MLP_LEARNED_MODEL_IDS = (MLP_LEARNED_MODEL_ID, MLP_LEARNED_PUBLIC_OBS_MODEL_ID)
MLP_LEARNED_POLICY_NAME = "mlp_learned"
MLP_LEARNED_COMPACT_FEATURE_SET = "compact_v0"
MLP_LEARNED_PUBLIC_OBS_FEATURE_SET = "public_obs_v1"
MLP_LEARNED_FEATURE_SET_CHOICES = (MLP_LEARNED_COMPACT_FEATURE_SET, MLP_LEARNED_PUBLIC_OBS_FEATURE_SET)
MLP_LEARNED_FEATURE_NAMES = TINY_LEARNED_FEATURE_NAMES
MLP_LEARNED_PUBLIC_OBS_TOP_K = 8


def _public_obs_feature_names(top_k: int = MLP_LEARNED_PUBLIC_OBS_TOP_K) -> tuple[str, ...]:
    base = (
        "ego_pos_x",
        "ego_pos_y",
        "ego_pos_z",
        "ego_vel_x",
        "ego_vel_y",
        "ego_vel_z",
        "goal_dir_x",
        "goal_dir_y",
        "goal_dir_z",
        "goal_dist_m",
        "done",
        "time_s",
        "agent_id_norm",
        "priority",
        "radius_m",
        "v_max_mps",
        "a_max_mps2",
    )
    neighbor_fields = (
        "present",
        "rel_pos_x",
        "rel_pos_y",
        "rel_pos_z",
        "rel_vel_x",
        "rel_vel_y",
        "rel_vel_z",
        "radius_m",
        "msg_age_s",
    )
    names = list(base)
    for idx in range(max(0, int(top_k))):
        names.extend(f"neighbor_{idx}_{field}" for field in neighbor_fields)
    return tuple(names)


MLP_LEARNED_PUBLIC_OBS_FEATURE_NAMES = _public_obs_feature_names(MLP_LEARNED_PUBLIC_OBS_TOP_K)


def _mlp_model_resource():
    return resources.files("microbench").joinpath("bundled_config", "learned_baselines", "mlp_policy.json")


def mlp_learned_model_path() -> str:
    return str(_mlp_model_resource())


def _mlp_feature_set(value: str | None) -> str:
    normalized = str(value or MLP_LEARNED_COMPACT_FEATURE_SET).strip()
    if normalized not in MLP_LEARNED_FEATURE_SET_CHOICES:
        raise ValueError(
            f"Unsupported MLP learned feature set {value!r}; "
            f"expected one of {','.join(MLP_LEARNED_FEATURE_SET_CHOICES)}"
        )
    return normalized


def mlp_feature_names(feature_set: str = MLP_LEARNED_COMPACT_FEATURE_SET, *, top_k: int = MLP_LEARNED_PUBLIC_OBS_TOP_K) -> tuple[str, ...]:
    mode = _mlp_feature_set(feature_set)
    if mode == MLP_LEARNED_COMPACT_FEATURE_SET:
        return MLP_LEARNED_FEATURE_NAMES
    return _public_obs_feature_names(top_k)


def mlp_model_id_for_feature_set(feature_set: str = MLP_LEARNED_COMPACT_FEATURE_SET) -> str:
    mode = _mlp_feature_set(feature_set)
    if mode == MLP_LEARNED_PUBLIC_OBS_FEATURE_SET:
        return MLP_LEARNED_PUBLIC_OBS_MODEL_ID
    return MLP_LEARNED_MODEL_ID


def mlp_feature_set_from_spec(spec: dict[str, Any]) -> str:
    declared = spec.get("feature_set")
    if declared is not None:
        return _mlp_feature_set(str(declared))
    features = tuple(spec.get("input_features", ()))
    if features == MLP_LEARNED_FEATURE_NAMES:
        return MLP_LEARNED_COMPACT_FEATURE_SET
    if features == MLP_LEARNED_PUBLIC_OBS_FEATURE_NAMES:
        return MLP_LEARNED_PUBLIC_OBS_FEATURE_SET
    raise ValueError("MLP learned baseline feature list does not match a supported public feature contract")


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
    feature_set = mlp_feature_set_from_spec(spec)
    expected_model_id = mlp_model_id_for_feature_set(feature_set)
    if spec.get("model_id") != expected_model_id:
        raise ValueError(f"Unsupported MLP learned model id {spec.get('model_id')!r}; expected {expected_model_id!r}")
    feature_top_k = int(spec.get("feature_top_k", MLP_LEARNED_PUBLIC_OBS_TOP_K))
    expected_features = mlp_feature_names(feature_set, top_k=feature_top_k)
    if tuple(spec.get("input_features", ())) != expected_features:
        raise ValueError("MLP learned baseline feature list does not match the public learned-policy contract")
    return spec


def planner_input_to_mlp_features(
    planner_input: PlannerInput,
    *,
    max_neighbors: int = 8,
    feature_set: str = MLP_LEARNED_COMPACT_FEATURE_SET,
) -> np.ndarray:
    mode = _mlp_feature_set(feature_set)
    if mode == MLP_LEARNED_COMPACT_FEATURE_SET:
        return planner_input_to_tiny_features(planner_input, max_neighbors=max_neighbors)
    return planner_input_to_rl_observation(planner_input, top_k=max_neighbors)


def _observation_to_public_obs_features(observation: np.ndarray, *, top_k: int = MLP_LEARNED_PUBLIC_OBS_TOP_K) -> np.ndarray:
    obs = np.asarray(observation, dtype=np.float32).reshape(-1)
    expected = OBS_BASE_DIM + max(0, int(top_k)) * OBS_NEIGHBOR_DIM
    if obs.shape[0] < OBS_BASE_DIM:
        raise ValueError(f"RL observation has length {obs.shape[0]}, expected at least {OBS_BASE_DIM}")
    if obs.shape[0] >= expected:
        return obs[:expected].astype(np.float32, copy=False)
    out = np.zeros((expected,), dtype=np.float32)
    out[: obs.shape[0]] = obs
    return out


def observation_to_mlp_features(
    observation: np.ndarray,
    *,
    top_k: int = 8,
    feature_set: str = MLP_LEARNED_COMPACT_FEATURE_SET,
) -> np.ndarray:
    mode = _mlp_feature_set(feature_set)
    if mode == MLP_LEARNED_COMPACT_FEATURE_SET:
        return observation_to_tiny_features(observation, top_k=top_k)
    return _observation_to_public_obs_features(observation, top_k=top_k)


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
    def feature_set(self) -> str:
        return mlp_feature_set_from_spec(self.spec)

    @property
    def feature_top_k(self) -> int:
        if self.feature_set == MLP_LEARNED_PUBLIC_OBS_FEATURE_SET:
            return int(self.spec.get("feature_top_k", MLP_LEARNED_PUBLIC_OBS_TOP_K))
        return MLP_LEARNED_PUBLIC_OBS_TOP_K

    @property
    def feature_names(self) -> tuple[str, ...]:
        return mlp_feature_names(self.feature_set, top_k=self.feature_top_k)

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
        if self.feature_set == MLP_LEARNED_PUBLIC_OBS_FEATURE_SET:
            goal = _normalize(np.asarray(features[6:9], dtype=np.float32))
            threat_features = observation_to_tiny_features(features, top_k=self.feature_top_k)
            threat = float(np.clip(threat_features[-2], 0.0, 1.0))
        else:
            goal = _normalize(np.asarray(features[0:3], dtype=np.float32))
            threat = float(np.clip(features[-2], 0.0, 1.0))
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
        x_model = _normalized_model_features(x, self.spec)
        w1 = np.asarray(self.spec["layer1_weights"], dtype=np.float32)
        b1 = np.asarray(self.spec["layer1_bias"], dtype=np.float32)
        w2 = np.asarray(self.spec["layer2_weights"], dtype=np.float32)
        b2 = np.asarray(self.spec["layer2_bias"], dtype=np.float32)
        feature_dim = len(self.feature_names)
        if x.shape != (feature_dim,):
            raise ValueError(f"MLP learned features must have shape {(feature_dim,)}, got {x.shape}")
        if w1.shape != (self.hidden_dim, feature_dim):
            raise ValueError(
                f"MLP layer1 weights have shape {w1.shape}, "
                f"expected {(self.hidden_dim, feature_dim)}"
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
        top_k = self.feature_top_k if self.feature_set == MLP_LEARNED_PUBLIC_OBS_FEATURE_SET else int(max_neighbors)
        return self.action_from_features(
            planner_input_to_mlp_features(
                planner_input,
                max_neighbors=top_k,
                feature_set=self.feature_set,
            )
        )

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        _ = deterministic
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        top_k = self.feature_top_k if self.feature_set == MLP_LEARNED_PUBLIC_OBS_FEATURE_SET else max(0, (obs.shape[0] - OBS_BASE_DIM) // OBS_NEIGHBOR_DIM)
        return self.action_from_features(
            observation_to_mlp_features(obs, top_k=top_k, feature_set=self.feature_set)
        ), None
