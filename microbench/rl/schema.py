from __future__ import annotations

from typing import Any


RL_INTERFACE_VERSION = "0.1.0"
RL_ACTION_SCHEMA_VERSION = "0.1.0"
RL_OBSERVATION_SCHEMA_VERSION = "0.1.0"
RL_REWARD_SCHEMA_VERSION = "0.1.0"

RL_POLICY_METHOD = "rl_policy"
AGENT_NAME_PREFIX = "agent_"

OBS_BASE_DIM = 17
OBS_NEIGHBOR_DIM = 9
OBS_POS_SLICE = slice(0, 3)
OBS_VEL_SLICE = slice(3, 6)
OBS_GOAL_DIR_SLICE = slice(6, 9)
OBS_GOAL_DIST_INDEX = 9
OBS_DONE_INDEX = 10
OBS_TIME_INDEX = 11
OBS_AGENT_ID_NORM_INDEX = 12
OBS_PRIORITY_INDEX = 13
OBS_RADIUS_INDEX = 14
OBS_V_MAX_INDEX = 15
OBS_A_MAX_INDEX = 16
OBS_NEIGHBOR_START = OBS_BASE_DIM
OBSERVATION_LAYOUT = {
    "ego_pos": (0, 3),
    "ego_vel": (3, 6),
    "goal_dir": (6, 9),
    "goal_dist": (9, 10),
    "done": (10, 11),
    "time_s": (11, 12),
    "agent_id_norm": (12, 13),
    "priority": (13, 14),
    "radius_m": (14, 15),
    "v_max_mps": (15, 16),
    "a_max_mps2": (16, 17),
    "neighbors": (OBS_NEIGHBOR_START, None),
}

BASE_OBSERVATION_FIELDS = (
    {"name": "ego_pos", "start": 0, "end": 3, "units": "m", "description": "ego position x,y,z"},
    {"name": "ego_vel", "start": 3, "end": 6, "units": "m/s", "description": "ego velocity vx,vy,vz"},
    {"name": "goal_dir", "start": 6, "end": 9, "units": "unitless", "description": "unit direction to goal"},
    {"name": "goal_dist", "start": 9, "end": 10, "units": "m", "description": "distance to goal"},
    {"name": "done", "start": 10, "end": 11, "units": "bool", "description": "goal-completion flag"},
    {"name": "time_s", "start": 11, "end": 12, "units": "s", "description": "episode time"},
    {"name": "agent_id_norm", "start": 12, "end": 13, "units": "unitless", "description": "agent id normalized to [0, 1]"},
    {"name": "priority", "start": 13, "end": 14, "units": "unitless", "description": "scenario priority"},
    {"name": "radius_m", "start": 14, "end": 15, "units": "m", "description": "collision radius"},
    {"name": "v_max_mps", "start": 15, "end": 16, "units": "m/s", "description": "speed limit"},
    {"name": "a_max_mps2", "start": 16, "end": 17, "units": "m/s^2", "description": "acceleration limit"},
)

NEIGHBOR_OBSERVATION_FIELDS = (
    {"name": "present", "offset": 0, "width": 1, "units": "bool", "description": "neighbor slot is populated"},
    {"name": "rel_pos", "offset": 1, "width": 3, "units": "m", "description": "neighbor position minus ego position"},
    {"name": "rel_vel", "offset": 4, "width": 3, "units": "m/s", "description": "neighbor velocity minus ego velocity"},
    {"name": "radius_m", "offset": 7, "width": 1, "units": "m", "description": "neighbor collision radius"},
    {"name": "msg_age_s", "offset": 8, "width": 1, "units": "s", "description": "selected track/message age"},
)

DEFAULT_REWARD_WEIGHTS = {
    "progress": 1.0,
    "time": -0.01,
    "collision": -10.0,
    "near_miss": -1.0,
    "goal": 10.0,
}

REWARD_TERMS = (
    {"name": "progress", "description": "change in distance-to-goal since the previous step"},
    {"name": "time", "description": "constant per-step time term"},
    {"name": "collision", "description": "penalty when the controlled agent is in a collision pair"},
    {"name": "near_miss", "description": "penalty when the controlled agent is in a near-miss pair and not colliding"},
    {"name": "goal", "description": "bonus when the controlled agent newly reaches its goal"},
)


def action_schema() -> dict[str, Any]:
    return {
        "schema_version": RL_ACTION_SCHEMA_VERSION,
        "shape": [3],
        "dtype": "float32",
        "low": -1.0,
        "high": 1.0,
        "semantics": "normalized desired world-frame velocity; scaled by agent v_max and clamped by simulator dynamics",
        "planar_note": "for planar scenarios, the y component is forced to zero before command execution",
    }


def observation_schema(*, top_k: int = 8) -> dict[str, Any]:
    top = max(0, int(top_k))
    return {
        "schema_version": RL_OBSERVATION_SCHEMA_VERSION,
        "shape": [OBS_BASE_DIM + top * OBS_NEIGHBOR_DIM],
        "dtype": "float32",
        "base_dim": OBS_BASE_DIM,
        "neighbor_dim": OBS_NEIGHBOR_DIM,
        "top_k": top,
        "layout": dict(OBSERVATION_LAYOUT),
        "base_fields": [dict(field) for field in BASE_OBSERVATION_FIELDS],
        "neighbor_fields": [dict(field) for field in NEIGHBOR_OBSERVATION_FIELDS],
        "privileged_global_state": False,
    }


def reward_schema(reward_config: dict[str, float] | None = None) -> dict[str, Any]:
    weights = {**DEFAULT_REWARD_WEIGHTS, **(reward_config or {})}
    return {
        "schema_version": RL_REWARD_SCHEMA_VERSION,
        "weights": {name: float(value) for name, value in weights.items()},
        "terms": [dict(term) for term in REWARD_TERMS],
        "leaderboard_note": "training reward is not a leaderboard score; use official benchmark metrics for comparisons",
    }


def interface_contract(*, top_k: int = 8, reward_config: dict[str, float] | None = None) -> dict[str, Any]:
    return {
        "interface_version": RL_INTERFACE_VERSION,
        "action": action_schema(),
        "observation": observation_schema(top_k=top_k),
        "reward": reward_schema(reward_config),
    }
