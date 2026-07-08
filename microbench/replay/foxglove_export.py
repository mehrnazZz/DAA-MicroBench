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

SCENE_UPDATE_SCHEMA = {
    "title": "foxglove.SceneUpdate",
    "type": "object",
    "properties": {
        "deletions": {"type": "array"},
        "entities": {"type": "array"},
    },
    "required": ["deletions", "entities"],
    "additionalProperties": True,
}

FRAME_TRANSFORMS_SCHEMA = {
    "title": "foxglove.FrameTransforms",
    "type": "object",
    "properties": {"transforms": {"type": "array"}},
    "required": ["transforms"],
    "additionalProperties": True,
}

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
