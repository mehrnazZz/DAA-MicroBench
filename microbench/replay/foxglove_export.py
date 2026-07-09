from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from microbench.replay.replay_interactive import _load_trace


WORLD_FRAME = "daa_world"
SCHEMA_ENCODING = "jsonschema"
MESSAGE_ENCODING = "json"

COLORS = (
    (0.145, 0.388, 0.922, 0.92),
    (0.863, 0.149, 0.149, 0.92),
    (0.020, 0.588, 0.412, 0.92),
    (0.851, 0.467, 0.024, 0.92),
    (0.486, 0.227, 0.929, 0.92),
    (0.031, 0.569, 0.698, 0.92),
    (0.745, 0.094, 0.365, 0.92),
    (0.302, 0.486, 0.059, 0.92),
)


def _schema_object(title: str, properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "title": title,
        "type": "object",
        "properties": properties,
        "required": required if required is not None else list(properties),
    }


def _schema_array(items: dict[str, Any]) -> dict[str, Any]:
    return {"type": "array", "items": items}


TIME_SCHEMA = _schema_object(
    "time",
    {
        "sec": {"type": "integer", "minimum": 0},
        "nsec": {"type": "integer", "minimum": 0, "maximum": 999999999},
    },
)

DURATION_SCHEMA = _schema_object(
    "duration",
    {
        "sec": {"type": "integer"},
        "nsec": {"type": "integer", "minimum": 0, "maximum": 999999999},
    },
)

VECTOR3_SCHEMA = _schema_object(
    "foxglove.Vector3",
    {
        "x": {"type": "number"},
        "y": {"type": "number"},
        "z": {"type": "number"},
    },
)

QUATERNION_SCHEMA = _schema_object(
    "foxglove.Quaternion",
    {
        "x": {"type": "number"},
        "y": {"type": "number"},
        "z": {"type": "number"},
        "w": {"type": "number"},
    },
)

POSE_SCHEMA = _schema_object(
    "foxglove.Pose",
    {
        "position": VECTOR3_SCHEMA,
        "orientation": QUATERNION_SCHEMA,
    },
)

COLOR_SCHEMA = _schema_object(
    "foxglove.Color",
    {
        "r": {"type": "number"},
        "g": {"type": "number"},
        "b": {"type": "number"},
        "a": {"type": "number"},
    },
)

POINT3_SCHEMA = _schema_object(
    "foxglove.Point3",
    {
        "x": {"type": "number"},
        "y": {"type": "number"},
        "z": {"type": "number"},
    },
)

KEY_VALUE_SCHEMA = _schema_object(
    "foxglove.KeyValuePair",
    {
        "key": {"type": "string"},
        "value": {"type": "string"},
    },
)

ARROW_PRIMITIVE_SCHEMA = _schema_object(
    "foxglove.ArrowPrimitive",
    {
        "pose": POSE_SCHEMA,
        "shaft_length": {"type": "number"},
        "shaft_diameter": {"type": "number"},
        "head_length": {"type": "number"},
        "head_diameter": {"type": "number"},
        "color": COLOR_SCHEMA,
    },
)

CUBE_PRIMITIVE_SCHEMA = _schema_object(
    "foxglove.CubePrimitive",
    {
        "pose": POSE_SCHEMA,
        "size": VECTOR3_SCHEMA,
        "color": COLOR_SCHEMA,
    },
)

SPHERE_PRIMITIVE_SCHEMA = _schema_object(
    "foxglove.SpherePrimitive",
    {
        "pose": POSE_SCHEMA,
        "size": VECTOR3_SCHEMA,
        "color": COLOR_SCHEMA,
    },
)

CYLINDER_PRIMITIVE_SCHEMA = _schema_object(
    "foxglove.CylinderPrimitive",
    {
        "pose": POSE_SCHEMA,
        "size": VECTOR3_SCHEMA,
        "bottom_scale": {"type": "number"},
        "top_scale": {"type": "number"},
        "color": COLOR_SCHEMA,
    },
)

LINE_TYPE_SCHEMA = {
    "title": "foxglove.LineType",
    "oneOf": [
        {"title": "LINE_STRIP", "const": 0},
        {"title": "LINE_LOOP", "const": 1},
        {"title": "LINE_LIST", "const": 2},
    ],
}

LINE_PRIMITIVE_SCHEMA = _schema_object(
    "foxglove.LinePrimitive",
    {
        "type": LINE_TYPE_SCHEMA,
        "pose": POSE_SCHEMA,
        "thickness": {"type": "number"},
        "scale_invariant": {"type": "boolean"},
        "points": _schema_array(POINT3_SCHEMA),
        "color": COLOR_SCHEMA,
        "colors": _schema_array(COLOR_SCHEMA),
        "indices": _schema_array({"type": "integer", "minimum": 0}),
    },
)

TRIANGLE_LIST_PRIMITIVE_SCHEMA = _schema_object(
    "foxglove.TriangleListPrimitive",
    {
        "pose": POSE_SCHEMA,
        "points": _schema_array(POINT3_SCHEMA),
        "color": COLOR_SCHEMA,
        "colors": _schema_array(COLOR_SCHEMA),
        "indices": _schema_array({"type": "integer", "minimum": 0}),
    },
)

TEXT_PRIMITIVE_SCHEMA = _schema_object(
    "foxglove.TextPrimitive",
    {
        "pose": POSE_SCHEMA,
        "billboard": {"type": "boolean"},
        "font_size": {"type": "number"},
        "scale_invariant": {"type": "boolean"},
        "color": COLOR_SCHEMA,
        "text": {"type": "string"},
    },
)

MODEL_PRIMITIVE_SCHEMA = _schema_object(
    "foxglove.ModelPrimitive",
    {
        "pose": POSE_SCHEMA,
        "scale": VECTOR3_SCHEMA,
        "color": COLOR_SCHEMA,
        "override_color": {"type": "boolean"},
        "url": {"type": "string"},
        "media_type": {"type": "string"},
        "data": {"type": "string", "contentEncoding": "base64"},
    },
)

SCENE_ENTITY_DELETION_SCHEMA = _schema_object(
    "foxglove.SceneEntityDeletion",
    {
        "timestamp": TIME_SCHEMA,
        "type": {
            "title": "foxglove.SceneEntityDeletionType",
            "oneOf": [
                {"title": "MATCHING_ID", "const": 0},
                {"title": "ALL", "const": 1},
            ],
        },
        "id": {"type": "string"},
    },
)

SCENE_ENTITY_SCHEMA = _schema_object(
    "foxglove.SceneEntity",
    {
        "timestamp": TIME_SCHEMA,
        "frame_id": {"type": "string"},
        "id": {"type": "string"},
        "lifetime": DURATION_SCHEMA,
        "frame_locked": {"type": "boolean"},
        "metadata": _schema_array(KEY_VALUE_SCHEMA),
        "arrows": _schema_array(ARROW_PRIMITIVE_SCHEMA),
        "cubes": _schema_array(CUBE_PRIMITIVE_SCHEMA),
        "spheres": _schema_array(SPHERE_PRIMITIVE_SCHEMA),
        "cylinders": _schema_array(CYLINDER_PRIMITIVE_SCHEMA),
        "lines": _schema_array(LINE_PRIMITIVE_SCHEMA),
        "triangles": _schema_array(TRIANGLE_LIST_PRIMITIVE_SCHEMA),
        "texts": _schema_array(TEXT_PRIMITIVE_SCHEMA),
        "models": _schema_array(MODEL_PRIMITIVE_SCHEMA),
    },
)

SCENE_UPDATE_SCHEMA = _schema_object(
    "foxglove.SceneUpdate",
    {
        "deletions": _schema_array(SCENE_ENTITY_DELETION_SCHEMA),
        "entities": _schema_array(SCENE_ENTITY_SCHEMA),
    },
)

FRAME_TRANSFORM_SCHEMA = _schema_object(
    "foxglove.FrameTransform",
    {
        "timestamp": TIME_SCHEMA,
        "parent_frame_id": {"type": "string"},
        "child_frame_id": {"type": "string"},
        "translation": VECTOR3_SCHEMA,
        "rotation": QUATERNION_SCHEMA,
    },
)

FRAME_TRANSFORMS_SCHEMA = _schema_object(
    "foxglove.FrameTransforms",
    {"transforms": _schema_array(FRAME_TRANSFORM_SCHEMA)},
)

FRAME_DIAGNOSTICS_SCHEMA = {
    "title": "daa.FrameDiagnostics",
    "type": "object",
    "properties": {
        "timestamp": {"type": "object"},
        "frame": {"type": "integer"},
        "t_sec": {"type": "number"},
        "n_agents": {"type": "integer"},
        "min_center_distance_m": {"type": ["number", "null"]},
        "speed_saturated_count": {"type": "integer"},
        "accel_saturated_count": {"type": "integer"},
        "selected_obs_count": {"type": "integer"},
        "max_msg_age_sec": {"type": ["number", "null"]},
        "mean_speed_mps": {"type": ["number", "null"]},
        "mean_cmd_mps": {"type": ["number", "null"]},
    },
    "required": ["timestamp", "frame", "t_sec", "n_agents"],
    "additionalProperties": True,
}

EVENT_SCHEMA = {
    "title": "daa.Event",
    "type": "object",
    "properties": {
        "timestamp": {"type": "object"},
        "type": {"type": "string"},
        "t_sec": {"type": "number"},
    },
    "required": ["timestamp", "type", "t_sec"],
    "additionalProperties": True,
}


def _require_mcap() -> tuple[type[Any], Any]:
    try:
        from mcap.writer import CompressionType, Writer
    except ImportError as exc:
        raise RuntimeError(
            "Foxglove export requires the optional MCAP dependency. "
            'Install it with `pip install -e ".[foxglove]"` or `pip install daa-microbench[foxglove]`.'
        ) from exc
    return Writer, CompressionType


def _timestamp_from_ns(t_ns: int) -> dict[str, int]:
    sec, nsec = divmod(max(0, int(t_ns)), 1_000_000_000)
    return {"sec": sec, "nsec": nsec}


def _relative_ns(t_sec: float, start_t_sec: float) -> int:
    return max(0, int(round((float(t_sec) - float(start_t_sec)) * 1_000_000_000)))


def _duration(sec: int = 0, nsec: int = 0) -> dict[str, int]:
    return {"sec": int(sec), "nsec": int(nsec)}


def _color(values: tuple[float, float, float, float]) -> dict[str, float]:
    return {"r": values[0], "g": values[1], "b": values[2], "a": values[3]}


def _age_color(age_sec: float | None) -> dict[str, float]:
    if age_sec is None:
        return _color((0.35, 0.35, 0.35, 0.35))
    if age_sec < 0.05:
        return _color((0.18, 0.63, 0.28, 0.70))
    if age_sec < 0.20:
        return _color((0.94, 0.55, 0.12, 0.75))
    return _color((0.86, 0.15, 0.15, 0.80))


def _intent_color(age_sec: float | None, valid: bool) -> dict[str, float]:
    if not valid:
        return _color((0.86, 0.15, 0.15, 0.55))
    if age_sec is None:
        return _color((0.12, 0.47, 0.71, 0.50))
    if age_sec < 0.10:
        return _color((0.12, 0.47, 0.95, 0.72))
    if age_sec < 0.50:
        return _color((0.58, 0.30, 0.92, 0.68))
    return _color((0.86, 0.15, 0.15, 0.58))


def _vec3_native_to_foxglove(values: list[float] | tuple[float, float, float]) -> dict[str, float]:
    # DAA Microbench stores altitude on y. Foxglove's 3D panel convention is z-up.
    return {"x": float(values[0]), "y": float(values[2]), "z": float(values[1])}


def _identity_quat() -> dict[str, float]:
    return {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}


def _quat_from_x_axis(values: list[float] | tuple[float, float, float]) -> dict[str, float]:
    v = _vec3_native_to_foxglove(values)
    x, y, z = v["x"], v["y"], v["z"]
    norm = math.sqrt(x * x + y * y + z * z)
    if norm <= 1e-9:
        return _identity_quat()
    x, y, z = x / norm, y / norm, z / norm
    if x < -0.999999:
        return {"x": 0.0, "y": 0.0, "z": 1.0, "w": 0.0}
    cross_y = -z
    cross_z = y
    w = 1.0 + x
    qnorm = math.sqrt(cross_y * cross_y + cross_z * cross_z + w * w)
    if qnorm <= 1e-9:
        return _identity_quat()
    return {"x": 0.0, "y": cross_y / qnorm, "z": cross_z / qnorm, "w": w / qnorm}


def _pose(
    position: dict[str, float] | None = None,
    orientation: dict[str, float] | None = None,
) -> dict[str, dict[str, float]]:
    return {
        "position": position or {"x": 0.0, "y": 0.0, "z": 0.0},
        "orientation": orientation or _identity_quat(),
    }


def _empty_entity(
    *,
    timestamp: dict[str, int],
    frame_id: str,
    entity_id: str,
    frame_locked: bool = False,
    metadata: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    return {
        "timestamp": timestamp,
        "frame_id": frame_id,
        "id": entity_id,
        "lifetime": _duration(),
        "frame_locked": frame_locked,
        "metadata": metadata or [],
        "arrows": [],
        "cubes": [],
        "spheres": [],
        "cylinders": [],
        "lines": [],
        "triangles": [],
        "texts": [],
        "models": [],
    }


def _line_primitive(
    points: list[dict[str, float]],
    *,
    color: dict[str, float],
    colors: list[dict[str, float]] | None = None,
    line_type: int = 0,
    thickness: float = 0.08,
    scale_invariant: bool = False,
) -> dict[str, Any]:
    return {
        "type": int(line_type),
        "pose": _pose(),
        "thickness": float(thickness),
        "scale_invariant": bool(scale_invariant),
        "points": points,
        "color": color,
        "colors": colors or [],
        "indices": [],
    }


def _agent_radius(meta: dict[str, Any], frame: dict[str, Any], local_idx: int, default: float = 0.5) -> float:
    params = meta.get("agent_params", {})
    if isinstance(params, dict) and "radius_m" in params:
        return float(params["radius_m"])
    agent_ids = frame.get("agent_ids", list(range(len(frame.get("positions", [])))))
    if local_idx < len(agent_ids):
        ego_id = int(agent_ids[local_idx])
        for obs in _frame_obs_list(frame, ego_id, local_idx):
            if int(obs.get("idx", -1)) == ego_id and "radius" in obs:
                return float(obs["radius"])
    return default


def _frame_obs_list(frame: dict[str, Any], ego_id: int, ego_local_idx: int) -> list[dict[str, Any]]:
    selected_obs = frame.get("selected_obs", {})
    if isinstance(selected_obs, dict):
        return selected_obs.get(str(ego_id), [])
    if isinstance(selected_obs, list) and ego_local_idx < len(selected_obs):
        return selected_obs[ego_local_idx]
    return []


def _frame_intent_list(frame: dict[str, Any], ego_id: int, ego_local_idx: int) -> list[dict[str, Any]]:
    selected_intents = frame.get("selected_intents", {})
    if isinstance(selected_intents, dict):
        return selected_intents.get(str(ego_id), [])
    if isinstance(selected_intents, list) and ego_local_idx < len(selected_intents):
        return selected_intents[ego_local_idx]
    return []


def _speed(values: list[float] | tuple[float, float, float]) -> float:
    return math.sqrt(float(values[0]) ** 2 + float(values[1]) ** 2 + float(values[2]) ** 2)


def _min_center_distance(positions: list[list[float]]) -> float | None:
    if len(positions) < 2:
        return None
    best: float | None = None
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dx = float(positions[i][0]) - float(positions[j][0])
            dy = float(positions[i][1]) - float(positions[j][1])
            dz = float(positions[i][2]) - float(positions[j][2])
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if best is None or dist < best:
                best = dist
    return best


def _message_age_stats(frame: dict[str, Any], agent_ids: list[int]) -> tuple[int, float | None]:
    ages: list[float] = []
    count = 0
    for local_idx, agent_id in enumerate(agent_ids):
        for obs in _frame_obs_list(frame, int(agent_id), local_idx):
            count += 1
            if "msg_age_sec" in obs:
                ages.append(float(obs["msg_age_sec"]))
    return count, max(ages) if ages else None


def _aabb_edge_points(lo: list[float], hi: list[float]) -> list[dict[str, float]]:
    corners = [
        [lo[0], lo[1], lo[2]],
        [hi[0], lo[1], lo[2]],
        [hi[0], hi[1], lo[2]],
        [lo[0], hi[1], lo[2]],
        [lo[0], lo[1], hi[2]],
        [hi[0], lo[1], hi[2]],
        [hi[0], hi[1], hi[2]],
        [lo[0], hi[1], hi[2]],
    ]
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    out: list[dict[str, float]] = []
    for a, b in edges:
        out.append(_vec3_native_to_foxglove(corners[a]))
        out.append(_vec3_native_to_foxglove(corners[b]))
    return out


def build_foxglove_static_scene(meta: dict[str, Any], timestamp_ns: int = 0) -> dict[str, Any]:
    timestamp = _timestamp_from_ns(timestamp_ns)
    entities: list[dict[str, Any]] = []
    bounds = meta.get("world_bounds", {}) or {}
    if bounds:
        lo = [float(bounds.get("xmin", 0.0)), float(bounds.get("ymin", 0.0)), float(bounds.get("zmin", 0.0))]
        hi = [float(bounds.get("xmax", 0.0)), float(bounds.get("ymax", 0.0)), float(bounds.get("zmax", 0.0))]
        entity = _empty_entity(timestamp=timestamp, frame_id=WORLD_FRAME, entity_id="world_bounds")
        entity["lines"].append(
            _line_primitive(
                _aabb_edge_points(lo, hi),
                color=_color((0.25, 0.28, 0.34, 0.80)),
                line_type=2,
                thickness=0.06,
            )
        )
        entities.append(entity)

    for idx, obstacle in enumerate(meta.get("obstacles", []) or []):
        aabb = obstacle.get("aabb")
        if not aabb:
            continue
        center = [float(v) for v in aabb.get("center", [0.0, 0.0, 0.0])]
        half = [float(v) for v in aabb.get("half", [0.0, 0.0, 0.0])]
        entity = _empty_entity(
            timestamp=timestamp,
            frame_id=WORLD_FRAME,
            entity_id=f"obstacle_{idx}",
            metadata=[{"key": "kind", "value": "obstacle"}],
        )
        entity["cubes"].append(
            {
                "pose": _pose(_vec3_native_to_foxglove(center)),
                "size": _vec3_native_to_foxglove([2.0 * half[0], 2.0 * half[1], 2.0 * half[2]]),
                "color": _color((0.42, 0.45, 0.50, 0.22)),
            }
        )
        entities.append(entity)

    return {"deletions": [], "entities": entities}


def build_foxglove_frame_messages(
    *,
    meta: dict[str, Any],
    frames: list[dict[str, Any]],
    frame_idx: int,
    start_t_sec: float | None = None,
    trail_frames: int = 200,
    max_sensing_links: int = 200,
) -> dict[str, dict[str, Any]]:
    frame = frames[frame_idx]
    agent_ids = [int(v) for v in frame.get("agent_ids", meta.get("agent_ids", []))]
    if not agent_ids:
        agent_ids = list(range(len(frame.get("positions", []))))
    if start_t_sec is None:
        start_t_sec = float(frames[0].get("t", 0.0))
    timestamp_ns = _relative_ns(float(frame.get("t", 0.0)), start_t_sec)
    timestamp = _timestamp_from_ns(timestamp_ns)
    positions = frame.get("positions", [])
    velocities = frame.get("velocities", [[0.0, 0.0, 0.0] for _ in positions])
    commands = frame.get("v_cmd", [[0.0, 0.0, 0.0] for _ in positions])

    transforms: list[dict[str, Any]] = []
    agent_entities: list[dict[str, Any]] = []
    trail_entities: list[dict[str, Any]] = []
    intent_by_sender: dict[int, dict[str, Any]] = {}

    for local_idx, agent_id in enumerate(agent_ids):
        pos = positions[local_idx]
        cmd = commands[local_idx] if local_idx < len(commands) else [0.0, 0.0, 0.0]
        vel = velocities[local_idx] if local_idx < len(velocities) else [0.0, 0.0, 0.0]
        radius = _agent_radius(meta, frame, local_idx)
        color = _color(COLORS[local_idx % len(COLORS)])
        direction = cmd if _speed(cmd) > 1e-6 else vel
        transforms.append(
            {
                "timestamp": timestamp,
                "parent_frame_id": WORLD_FRAME,
                "child_frame_id": f"drone_{agent_id}",
                "translation": _vec3_native_to_foxglove(pos),
                "rotation": _quat_from_x_axis(direction),
            }
        )

        entity = _empty_entity(
            timestamp=timestamp,
            frame_id=f"drone_{agent_id}",
            entity_id=f"agent_{agent_id}",
            frame_locked=True,
            metadata=[
                {"key": "agent_id", "value": str(agent_id)},
                {"key": "speed_mps", "value": f"{_speed(vel):.3f}"},
                {"key": "cmd_mps", "value": f"{_speed(cmd):.3f}"},
            ],
        )
        entity["spheres"].append(
            {
                "pose": _pose(),
                "size": {"x": 2.0 * radius, "y": 2.0 * radius, "z": 2.0 * radius},
                "color": color,
            }
        )
        cmd_speed = _speed(cmd)
        if cmd_speed > 1e-6:
            entity["arrows"].append(
                {
                    "pose": _pose(),
                    "shaft_length": max(0.05, 0.45 * cmd_speed),
                    "shaft_diameter": max(0.03, 0.10 * radius),
                    "head_length": max(0.08, 0.18 * cmd_speed),
                    "head_diameter": max(0.06, 0.25 * radius),
                    "color": _color((color["r"], color["g"], color["b"], 0.78)),
                }
            )
        entity["texts"].append(
            {
                "pose": _pose({"x": 0.0, "y": 0.0, "z": radius + 0.35}),
                "billboard": True,
                "font_size": 13.0,
                "scale_invariant": True,
                "color": _color((0.05, 0.07, 0.12, 1.0)),
                "text": f"agent {agent_id}",
            }
        )
        agent_entities.append(entity)

        trail_start = max(0, frame_idx - max(0, int(trail_frames)))
        trail_points = [_vec3_native_to_foxglove(frames[k]["positions"][local_idx]) for k in range(trail_start, frame_idx + 1)]
        trail = _empty_entity(timestamp=timestamp, frame_id=WORLD_FRAME, entity_id=f"trail_{agent_id}")
        trail["lines"].append(
            _line_primitive(
                trail_points,
                color=_color((color["r"], color["g"], color["b"], 0.55)),
                line_type=0,
                thickness=0.045,
            )
        )
        trail_entities.append(trail)

        for intent in _frame_intent_list(frame, int(agent_id), local_idx):
            points = intent.get("points", [])
            if not isinstance(points, list) or len(points) < 2:
                continue
            sender_id = int(intent.get("idx", intent.get("sender_id", -1)))
            if sender_id < 0:
                continue
            valid = bool(intent.get("valid", False))
            age = float(intent["intent_age_s"]) if "intent_age_s" in intent else None
            existing = intent_by_sender.get(sender_id)
            existing_valid = bool(existing.get("valid", False)) if existing else False
            existing_age = float(existing.get("intent_age_s", float("inf"))) if existing else float("inf")
            should_replace = existing is None or (valid and not existing_valid) or (
                valid == existing_valid and age is not None and age < existing_age
            )
            if should_replace:
                receiver_ids = list(existing.get("receiver_ids", [])) if existing else []
                receiver_ids.append(int(agent_id))
                intent_by_sender[sender_id] = {
                    "sender_id": sender_id,
                    "receiver_ids": receiver_ids,
                    "valid": valid,
                    "intent_age_s": age,
                    "kind": str(intent.get("kind", "")),
                    "expiry_s": float(intent.get("expiry_s", frame.get("t", 0.0))),
                    "tube_radius_m": float(intent.get("tube_radius_m", 0.0)),
                    "points": points,
                }
            elif existing is not None:
                existing.setdefault("receiver_ids", []).append(int(agent_id))

    link_points: list[dict[str, float]] = []
    link_colors: list[dict[str, float]] = []
    for local_idx, agent_id in enumerate(agent_ids):
        if len(link_points) // 2 >= max_sensing_links:
            break
        ego = positions[local_idx]
        for obs in _frame_obs_list(frame, agent_id, local_idx):
            if len(link_points) // 2 >= max_sensing_links:
                break
            if not bool(obs.get("valid", True)) or "pos" not in obs:
                continue
            c = _age_color(float(obs.get("msg_age_sec", 0.0)) if "msg_age_sec" in obs else None)
            link_points.extend([_vec3_native_to_foxglove(ego), _vec3_native_to_foxglove(obs["pos"])])
            link_colors.extend([c, c])
    link_entity = _empty_entity(
        timestamp=timestamp,
        frame_id=WORLD_FRAME,
        entity_id="sensing_links",
        metadata=[{"key": "link_count", "value": str(len(link_points) // 2)}],
    )
    link_entity["lines"].append(
        _line_primitive(
            link_points,
            color=_color((0.45, 0.45, 0.45, 0.25)),
            colors=link_colors,
            line_type=2,
            thickness=2.0,
            scale_invariant=True,
        )
    )

    intent_entities: list[dict[str, Any]] = []
    for sender_id, intent in sorted(intent_by_sender.items()):
        points = [_vec3_native_to_foxglove([float(v[0]), float(v[1]), float(v[2])]) for v in intent["points"]]
        if len(points) < 2:
            continue
        tube_radius = float(intent.get("tube_radius_m", 0.0))
        age = intent.get("intent_age_s")
        valid = bool(intent.get("valid", False))
        entity = _empty_entity(
            timestamp=timestamp,
            frame_id=WORLD_FRAME,
            entity_id=f"intent_{sender_id}",
            metadata=[
                {"key": "sender_id", "value": str(sender_id)},
                {"key": "kind", "value": str(intent.get("kind", ""))},
                {"key": "valid", "value": str(valid).lower()},
                {"key": "intent_age_s", "value": "" if age is None else f"{float(age):.3f}"},
                {"key": "expiry_s", "value": f"{float(intent.get('expiry_s', 0.0)):.3f}"},
                {"key": "tube_radius_m", "value": f"{tube_radius:.3f}"},
                {"key": "receiver_count", "value": str(len(set(intent.get("receiver_ids", []))))},
            ],
        )
        entity["lines"].append(
            _line_primitive(
                points,
                color=_intent_color(float(age) if age is not None else None, valid),
                line_type=0,
                thickness=max(0.06, min(0.28, tube_radius * 0.20 if tube_radius > 0.0 else 0.10)),
            )
        )
        intent_entities.append(entity)

    obs_count, max_msg_age = _message_age_stats(frame, agent_ids)
    speeds = [_speed(v) for v in velocities]
    cmd_speeds = [_speed(v) for v in commands]
    diagnostics = {
        "timestamp": timestamp,
        "frame": int(frame_idx),
        "t_sec": float(frame.get("t", 0.0)),
        "n_agents": len(agent_ids),
        "min_center_distance_m": _min_center_distance(positions),
        "speed_saturated_count": sum(1 for v in frame.get("speed_saturated", []) if bool(v)),
        "accel_saturated_count": sum(1 for v in frame.get("accel_saturated", []) if bool(v)),
        "selected_obs_count": int(obs_count),
        "max_msg_age_sec": max_msg_age,
        "mean_speed_mps": sum(speeds) / len(speeds) if speeds else None,
        "mean_cmd_mps": sum(cmd_speeds) / len(cmd_speeds) if cmd_speeds else None,
    }

    return {
        "transforms": {"transforms": transforms},
        "agents": {"deletions": [], "entities": agent_entities},
        "trails": {"deletions": [], "entities": trail_entities},
        "sensing_links": {"deletions": [], "entities": [link_entity]},
        "intents": {"deletions": [], "entities": intent_entities},
        "diagnostics": diagnostics,
    }


def _event_rows(trace_path: str, meta: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if meta.get("type"):
        rows.append({k: v for k, v in meta.items() if k != "kind"})
    events_path = Path(trace_path).with_name("events.jsonl")
    if events_path.exists():
        with events_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    return rows


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0.0 else "-Infinity"
    if isinstance(value, dict):
        return {str(k): _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, tuple):
        return [_json_sanitize(v) for v in value]
    return value


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(_json_sanitize(payload), separators=(",", ":"), allow_nan=False).encode("utf-8")


def export_foxglove_mcap(
    trace_path: str,
    out_path: str,
    *,
    trail_frames: int = 200,
    max_sensing_links: int = 200,
) -> Path:
    Writer, CompressionType = _require_mcap()
    meta, frames = _load_trace(trace_path)
    opath = Path(out_path)
    opath.parent.mkdir(parents=True, exist_ok=True)
    start_t_sec = float(frames[0].get("t", 0.0))

    with opath.open("wb") as stream:
        writer = Writer(stream, compression=CompressionType.NONE)
        writer.start(profile="foxglove", library="daa-microbench")
        scene_schema = writer.register_schema("foxglove.SceneUpdate", SCHEMA_ENCODING, _json_bytes(SCENE_UPDATE_SCHEMA))
        tf_schema = writer.register_schema("foxglove.FrameTransforms", SCHEMA_ENCODING, _json_bytes(FRAME_TRANSFORMS_SCHEMA))
        diagnostics_schema = writer.register_schema("daa.FrameDiagnostics", SCHEMA_ENCODING, _json_bytes(FRAME_DIAGNOSTICS_SCHEMA))
        event_schema = writer.register_schema("daa.Event", SCHEMA_ENCODING, _json_bytes(EVENT_SCHEMA))

        static_ch = writer.register_channel("/daa/static", MESSAGE_ENCODING, scene_schema)
        agents_ch = writer.register_channel("/daa/agents", MESSAGE_ENCODING, scene_schema)
        trails_ch = writer.register_channel("/daa/trails", MESSAGE_ENCODING, scene_schema)
        sensing_ch = writer.register_channel("/daa/sensing_links", MESSAGE_ENCODING, scene_schema)
        intents_ch = writer.register_channel("/daa/intents", MESSAGE_ENCODING, scene_schema)
        tf_ch = writer.register_channel("/tf", MESSAGE_ENCODING, tf_schema)
        diagnostics_ch = writer.register_channel("/daa/diagnostics", MESSAGE_ENCODING, diagnostics_schema)
        events_ch = writer.register_channel("/daa/events", MESSAGE_ENCODING, event_schema)

        writer.add_metadata(
            "daa_microbench",
            {
                "trace_path": str(trace_path),
                "scenario": str(meta.get("scenario_name", meta.get("scenario", "unknown"))),
                "method": str(meta.get("method", "unknown")),
                "comm_profile": str(meta.get("comm_profile", "unknown")),
                "coordinate_mapping": "foxglove=(x, daa_z, daa_y_altitude)",
                "start_t_sec": f"{start_t_sec:.9f}",
                "frame_count": str(len(frames)),
            },
        )

        writer.add_message(static_ch, 0, _json_bytes(build_foxglove_static_scene(meta)), 0)
        for frame_idx, frame in enumerate(frames):
            t_ns = _relative_ns(float(frame.get("t", 0.0)), start_t_sec)
            messages = build_foxglove_frame_messages(
                meta=meta,
                frames=frames,
                frame_idx=frame_idx,
                start_t_sec=start_t_sec,
                trail_frames=trail_frames,
                max_sensing_links=max_sensing_links,
            )
            writer.add_message(tf_ch, t_ns, _json_bytes(messages["transforms"]), t_ns)
            writer.add_message(agents_ch, t_ns, _json_bytes(messages["agents"]), t_ns)
            writer.add_message(trails_ch, t_ns, _json_bytes(messages["trails"]), t_ns)
            writer.add_message(sensing_ch, t_ns, _json_bytes(messages["sensing_links"]), t_ns)
            writer.add_message(intents_ch, t_ns, _json_bytes(messages["intents"]), t_ns)
            writer.add_message(diagnostics_ch, t_ns, _json_bytes(messages["diagnostics"]), t_ns)

        for event in _event_rows(trace_path, meta):
            if "t" not in event:
                continue
            t_ns = _relative_ns(float(event.get("t", start_t_sec)), start_t_sec)
            payload = {
                "timestamp": _timestamp_from_ns(t_ns),
                "type": str(event.get("type", "event")),
                "t_sec": float(event.get("t", 0.0)),
                "data": event,
            }
            writer.add_message(events_ch, t_ns, _json_bytes(payload), t_ns)

        writer.finish()
    return opath
