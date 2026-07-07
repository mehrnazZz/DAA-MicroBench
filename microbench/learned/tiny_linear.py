from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
import json
from pathlib import Path
from typing import Any

import numpy as np

from microbench.types import PlannerInput


LEARNED_BASELINE_SCHEMA_VERSION = "0.1"
TINY_LEARNED_MODEL_ID = "tiny_linear_goal_avoidance_v0"
TINY_LEARNED_POLICY_NAME = "tiny_learned"
TINY_LEARNED_FEATURE_NAMES = (
    "goal_dir_x",
    "goal_dir_y",
    "goal_dir_z",
    "ego_vel_frac_x",
    "ego_vel_frac_y",
    "ego_vel_frac_z",
    "avoid_vec_x",
    "avoid_vec_y",
    "avoid_vec_z",
    "closest_rel_vel_frac_x",
    "closest_rel_vel_frac_y",
    "closest_rel_vel_frac_z",
    "threat_scalar",
    "neighbor_count_frac",
)

# Keep these in sync with microbench.rl.schema without importing microbench.rl
# from planner modules. Importing the RL package also imports the environment,
# which imports the planner registry.
OBS_BASE_DIM = 17
OBS_NEIGHBOR_DIM = 9
OBS_GOAL_DIR_SLICE = slice(6, 9)
OBS_VEL_SLICE = slice(3, 6)
OBS_RADIUS_INDEX = 14
OBS_V_MAX_INDEX = 15
OBS_A_MAX_INDEX = 16
OBS_NEIGHBOR_START = OBS_BASE_DIM


def _normalize(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-9:
        return np.zeros(3, dtype=np.float32)
    return (arr / norm).astype(np.float32)


def _safe_div(v: np.ndarray, denom: float) -> np.ndarray:
    return np.asarray(v, dtype=np.float32) / max(1e-6, float(denom))


def _tiny_model_resource():
    return resources.files("microbench").joinpath("bundled_config", "learned_baselines", "tiny_linear_policy.json")


def tiny_learned_model_path() -> str:
    return str(_tiny_model_resource())


def load_tiny_learned_spec(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        text = _tiny_model_resource().read_text(encoding="utf-8")
        spec = json.loads(text)
    else:
        spec = json.loads(Path(path).read_text(encoding="utf-8"))
    if spec.get("schema_version") != LEARNED_BASELINE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported learned baseline schema {spec.get('schema_version')!r}; "
            f"expected {LEARNED_BASELINE_SCHEMA_VERSION!r}"
        )
    if tuple(spec.get("input_features", ())) != TINY_LEARNED_FEATURE_NAMES:
        raise ValueError("Learned baseline feature list does not match the public tiny policy contract")
    return spec


def _neighbor_threat(
    *,
    rel_pos_neighbor_minus_ego: np.ndarray,
    rel_vel_neighbor_minus_ego: np.ndarray,
    ego_radius: float,
    neighbor_radius: float,
    safety_horizon_m: float,
    v_ref: float,
) -> tuple[np.ndarray, float, float]:
    rel_pos = np.asarray(rel_pos_neighbor_minus_ego, dtype=np.float32)
    rel_vel = np.asarray(rel_vel_neighbor_minus_ego, dtype=np.float32)
    distance = float(np.linalg.norm(rel_pos))
    if distance < 1e-9:
        away = np.asarray([-1.0, 0.0, 0.0], dtype=np.float32)
    else:
        away = (-rel_pos / distance).astype(np.float32)
    clearance = distance - float(ego_radius) - float(neighbor_radius)
    proximity = float(np.clip((float(safety_horizon_m) - clearance) / max(1e-6, float(safety_horizon_m)), 0.0, 1.0))
    closing_speed = 0.0
    if distance >= 1e-9:
        closing_speed = float(np.clip(-float(np.dot(rel_pos / distance, rel_vel)) / max(1e-6, float(v_ref)), 0.0, 1.0))
    threat = float(np.clip(proximity * (0.5 + 0.5 * closing_speed), 0.0, 1.0))
    return away, threat, clearance


def _assemble_features(
    *,
    goal_dir: np.ndarray,
    ego_vel: np.ndarray,
    v_max: float,
    avoid_vec: np.ndarray,
    closest_rel_vel: np.ndarray,
    threat_scalar: float,
    neighbor_count: int,
    max_neighbors: int,
) -> np.ndarray:
    features = np.concatenate(
        [
            _normalize(goal_dir),
            np.clip(_safe_div(ego_vel, v_max), -1.0, 1.0),
            np.clip(np.asarray(avoid_vec, dtype=np.float32), -1.0, 1.0),
            np.clip(_safe_div(closest_rel_vel, v_max), -1.0, 1.0),
            np.asarray([float(np.clip(threat_scalar, 0.0, 1.0))], dtype=np.float32),
            np.asarray([float(np.clip(neighbor_count / max(1, max_neighbors), 0.0, 1.0))], dtype=np.float32),
        ]
    ).astype(np.float32)
    if features.shape != (len(TINY_LEARNED_FEATURE_NAMES),):
        raise ValueError(f"tiny learned features must have shape {(len(TINY_LEARNED_FEATURE_NAMES),)}, got {features.shape}")
    return features


def planner_input_to_tiny_features(planner_input: PlannerInput, *, max_neighbors: int = 8) -> np.ndarray:
    ego = planner_input.ego
    v_max = float(max(1e-6, ego.v_max))
    safety_horizon_m = max(2.5, 4.0 * float(ego.radius))
    avoid_vec = np.zeros(3, dtype=np.float32)
    closest_rel_vel = np.zeros(3, dtype=np.float32)
    threat_scalar = 0.0
    closest_distance = float("inf")
    neighbor_count = 0

    for neighbor in list(planner_input.neighbors)[:max_neighbors]:
        if not bool(neighbor.valid):
            continue
        neighbor_count += 1
        rel_pos = np.asarray(neighbor.pos, dtype=np.float32) - np.asarray(ego.pos, dtype=np.float32)
        rel_vel = np.asarray(neighbor.vel, dtype=np.float32) - np.asarray(ego.vel, dtype=np.float32)
        away, threat, _ = _neighbor_threat(
            rel_pos_neighbor_minus_ego=rel_pos,
            rel_vel_neighbor_minus_ego=rel_vel,
            ego_radius=float(ego.radius),
            neighbor_radius=float(neighbor.radius),
            safety_horizon_m=safety_horizon_m,
            v_ref=v_max,
        )
        avoid_vec += away * threat
        threat_scalar = max(threat_scalar, threat)
        distance = float(np.linalg.norm(rel_pos))
        if distance < closest_distance:
            closest_distance = distance
            closest_rel_vel = rel_vel.astype(np.float32)

    return _assemble_features(
        goal_dir=np.asarray(planner_input.goal_dir, dtype=np.float32),
        ego_vel=np.asarray(ego.vel, dtype=np.float32),
        v_max=v_max,
        avoid_vec=avoid_vec,
        closest_rel_vel=closest_rel_vel,
        threat_scalar=threat_scalar,
        neighbor_count=neighbor_count,
        max_neighbors=max_neighbors,
    )


def observation_to_tiny_features(observation: np.ndarray, *, top_k: int = 8) -> np.ndarray:
    obs = np.asarray(observation, dtype=np.float32).reshape(-1)
    expected_min = OBS_BASE_DIM + int(top_k) * OBS_NEIGHBOR_DIM
    if obs.shape[0] < expected_min:
        raise ValueError(f"RL observation has length {obs.shape[0]}, expected at least {expected_min}")

    goal_dir = obs[OBS_GOAL_DIR_SLICE]
    ego_vel = obs[OBS_VEL_SLICE]
    ego_radius = float(obs[OBS_RADIUS_INDEX])
    v_max = float(max(1e-6, obs[OBS_V_MAX_INDEX]))
    a_max = float(max(1e-6, obs[OBS_A_MAX_INDEX]))
    safety_horizon_m = max(2.5, 4.0 * ego_radius, 2.0 * a_max)
    avoid_vec = np.zeros(3, dtype=np.float32)
    closest_rel_vel = np.zeros(3, dtype=np.float32)
    threat_scalar = 0.0
    closest_distance = float("inf")
    neighbor_count = 0

    for i in range(int(top_k)):
        start = OBS_NEIGHBOR_START + i * OBS_NEIGHBOR_DIM
        block = obs[start : start + OBS_NEIGHBOR_DIM]
        if float(block[0]) <= 0.0:
            continue
        neighbor_count += 1
        rel_pos = block[1:4]
        rel_vel = block[4:7]
        away, threat, _ = _neighbor_threat(
            rel_pos_neighbor_minus_ego=rel_pos,
            rel_vel_neighbor_minus_ego=rel_vel,
            ego_radius=ego_radius,
            neighbor_radius=float(block[7]),
            safety_horizon_m=safety_horizon_m,
            v_ref=v_max,
        )
        avoid_vec += away * threat
        threat_scalar = max(threat_scalar, threat)
        distance = float(np.linalg.norm(rel_pos))
        if distance < closest_distance:
            closest_distance = distance
            closest_rel_vel = rel_vel.astype(np.float32)

    return _assemble_features(
        goal_dir=goal_dir,
        ego_vel=ego_vel,
        v_max=v_max,
        avoid_vec=avoid_vec,
        closest_rel_vel=closest_rel_vel,
        threat_scalar=threat_scalar,
        neighbor_count=neighbor_count,
        max_neighbors=int(top_k),
    )


@dataclass
class TinyLinearPolicyModel:
    spec: dict[str, Any]

    @classmethod
    def from_path(cls, path: str | Path | None = None) -> "TinyLinearPolicyModel":
        return cls(load_tiny_learned_spec(path))

    @property
    def model_id(self) -> str:
        return str(self.spec["model_id"])

    @property
    def training_metadata(self) -> dict[str, Any]:
        return dict(self.spec.get("training", {}))

    def action_from_features(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float32).reshape(-1)
        weights = np.asarray(self.spec["weights"], dtype=np.float32)
        bias = np.asarray(self.spec["bias"], dtype=np.float32)
        if weights.shape != (3, len(TINY_LEARNED_FEATURE_NAMES)):
            raise ValueError(f"tiny learned weights must have shape {(3, len(TINY_LEARNED_FEATURE_NAMES))}, got {weights.shape}")
        if bias.shape != (3,):
            raise ValueError(f"tiny learned bias must have shape (3,), got {bias.shape}")
        raw = weights @ x + bias
        return np.tanh(raw).astype(np.float32)

    def action_from_planner_input(self, planner_input: PlannerInput, *, max_neighbors: int = 8) -> np.ndarray:
        return self.action_from_features(planner_input_to_tiny_features(planner_input, max_neighbors=max_neighbors))

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        _ = deterministic
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        top_k = max(0, (obs.shape[0] - OBS_BASE_DIM) // OBS_NEIGHBOR_DIM)
        return self.action_from_features(observation_to_tiny_features(obs, top_k=top_k)), None
