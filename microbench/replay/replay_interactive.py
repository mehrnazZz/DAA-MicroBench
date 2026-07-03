from __future__ import annotations

from html import escape
import json
from pathlib import Path
from typing import Any


def _age_color(age: float) -> str:
    if age < 0.05:
        return "#2CA02C"
    if age < 0.20:
        return "#F2A104"
    return "#D62728"


def _load_trace(trace_path: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    meta: dict[str, Any] | None = None
    frames: list[dict[str, Any]] = []
    with Path(trace_path).open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("kind") == "meta":
                meta = rec
            elif rec.get("kind") == "frame":
                frames.append(rec)
    if not frames:
        raise ValueError(f"Trace has no frames: {trace_path}")
    return meta or {}, frames


def _infer_collision_pair(meta: dict[str, Any], trace_path: str, agent_ids: list[int]) -> tuple[int, int] | None:
    if "i" in meta and "j" in meta:
        id_to_local = {aid: i for i, aid in enumerate(agent_ids)}
        if meta["i"] in id_to_local and meta["j"] in id_to_local:
            return (id_to_local[meta["i"]], id_to_local[meta["j"]])

    events_path = Path(trace_path).with_name("events.jsonl")
    if not events_path.exists():
        return None
    id_to_local = {aid: i for i, aid in enumerate(agent_ids)}
    with events_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("type") != "collision":
                continue
            i = rec.get("i")
            j = rec.get("j")
            if i in id_to_local and j in id_to_local:
                return (id_to_local[i], id_to_local[j])
    return None


def _frame_obs_list(frm: dict[str, Any], ego_id: int, ego_local_idx: int) -> list[dict[str, Any]]:
    selected_obs = frm.get("selected_obs", {})
    if isinstance(selected_obs, dict):
        return selected_obs.get(str(ego_id), [])
    if isinstance(selected_obs, list) and ego_local_idx < len(selected_obs):
        return selected_obs[ego_local_idx]
    return []


def _frame_intent_list(frm: dict[str, Any], ego_id: int, ego_local_idx: int) -> list[dict[str, Any]]:
    selected_intents = frm.get("selected_intents", {})
    if isinstance(selected_intents, dict):
        return selected_intents.get(str(ego_id), [])
    if isinstance(selected_intents, list) and ego_local_idx < len(selected_intents):
        return selected_intents[ego_local_idx]
    return []


def _wireframe_segments_from_bounds(bounds: dict[str, Any]) -> tuple[list[float | None], list[float | None], list[float | None]]:
    if not bounds:
        return [], [], []
    xmin = float(bounds.get("xmin", 0.0))
    xmax = float(bounds.get("xmax", 0.0))
    ymin = float(bounds.get("ymin", 0.0))
    ymax = float(bounds.get("ymax", 0.0))
    zmin = float(bounds.get("zmin", 0.0))
    zmax = float(bounds.get("zmax", 0.0))
    return _wireframe_segments_from_aabb([xmin, ymin, zmin], [xmax, ymax, zmax])


def _wireframe_segments_from_aabb(lo: list[float], hi: list[float]) -> tuple[list[float | None], list[float | None], list[float | None]]:
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    corners = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for a, b in edges:
        pa = corners[a]
        pb = corners[b]
        xs.extend([pa[0], pb[0], None])
        ys.extend([pa[1], pb[1], None])
        zs.extend([pa[2], pb[2], None])
    return xs, ys, zs


def _trace_segments(meta: dict[str, Any]) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    bounds = meta.get("world_bounds", {}) or {}
    bx, by, bz = _wireframe_segments_from_bounds(bounds)
    if bx:
        traces.append(
            {
                "type": "scatter3d",
                "mode": "lines",
                "name": "world_bounds",
                "x": bx,
                "y": by,
                "z": bz,
                "line": {"color": "rgba(90,90,90,0.6)", "width": 3},
                "hoverinfo": "skip",
            }
        )
    for idx, ob in enumerate(meta.get("obstacles", []) or []):
        aabb = ob.get("aabb")
        if not aabb:
            continue
        center = aabb.get("center", [0.0, 0.0, 0.0])
        half = aabb.get("half", [0.0, 0.0, 0.0])
        lo = [float(c) - float(h) for c, h in zip(center, half)]
        hi = [float(c) + float(h) for c, h in zip(center, half)]
        ox, oy, oz = _wireframe_segments_from_aabb(lo, hi)
        traces.append(
            {
                "type": "scatter3d",
                "mode": "lines",
                "name": f"obstacle_{idx}",
                "x": ox,
                "y": oy,
                "z": oz,
                "line": {"color": "rgba(140,140,140,0.9)", "width": 5},
                "hovertemplate": f"Obstacle {idx}<extra></extra>",
            }
        )
    return traces


def _trail_xyz(frames: list[dict[str, Any]], frame_idx: int, agent_local_idx: int, tail: int) -> tuple[list[float], list[float], list[float]]:
    start = max(0, frame_idx - tail)
    xs = [float(frames[k]["positions"][agent_local_idx][0]) for k in range(start, frame_idx + 1)]
    ys = [float(frames[k]["positions"][agent_local_idx][1]) for k in range(start, frame_idx + 1)]
    zs = [float(frames[k]["positions"][agent_local_idx][2]) for k in range(start, frame_idx + 1)]
    return xs, ys, zs


def _pair_distance(positions: list[list[float]], i: int, j: int) -> float:
    dx = float(positions[i][0]) - float(positions[j][0])
    dy = float(positions[i][1]) - float(positions[j][1])
    dz = float(positions[i][2]) - float(positions[j][2])
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def _min_pair_distance(frm: dict[str, Any]) -> float | None:
    pos = frm.get("positions", [])
    if len(pos) < 2:
        return None
    best: float | None = None
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            d = _pair_distance(pos, i, j)
            if best is None or d < best:
                best = d
    return best


def _signed_distance_to_aabb(pos: list[float], center: list[float], half: list[float]) -> float:
    dx = abs(float(pos[0]) - float(center[0])) - float(half[0])
    dy = abs(float(pos[1]) - float(center[1])) - float(half[1])
    dz = abs(float(pos[2]) - float(center[2])) - float(half[2])
    ox = max(dx, 0.0)
    oy = max(dy, 0.0)
    oz = max(dz, 0.0)
    outside = (ox * ox + oy * oy + oz * oz) ** 0.5
    if outside > 0.0:
        return outside
    return max(dx, dy, dz)


def _min_obstacle_clearance(frm: dict[str, Any], obstacles: list[dict[str, Any]]) -> float | None:
    if not obstacles:
        return None
    best: float | None = None
    for pos in frm.get("positions", []):
        for ob in obstacles:
            aabb = ob.get("aabb")
            if not aabb:
                continue
            d = _signed_distance_to_aabb(pos, aabb.get("center", [0.0, 0.0, 0.0]), aabb.get("half", [0.0, 0.0, 0.0]))
            if best is None or d < best:
                best = d
    return best


def _agent_hover_text(frm: dict[str, Any], agent_ids: list[int]) -> list[str]:
    out: list[str] = []
    positions = frm.get("positions", [])
    velocities = frm.get("velocities", [])
    v_cmd = frm.get("v_cmd", [])
    goal_dirs = frm.get("goal_dirs", [])
    speed_sat = frm.get("speed_saturated", [])
    accel_sat = frm.get("accel_saturated", [])
    for local_idx, agent_id in enumerate(agent_ids):
        px, py, pz = [float(x) for x in positions[local_idx]]
        vx, vy, vz = [float(x) for x in velocities[local_idx]]
        cx, cy, cz = [float(x) for x in v_cmd[local_idx]]
        gx, gy, gz = [float(x) for x in goal_dirs[local_idx]]
        speed = (vx * vx + vy * vy + vz * vz) ** 0.5
        cmd_speed = (cx * cx + cy * cy + cz * cz) ** 0.5
        obs_list = _frame_obs_list(frm, agent_id, local_idx)
        min_obs_age = min((float(o.get("msg_age_sec", 0.0)) for o in obs_list), default=0.0)
        intent_list = _frame_intent_list(frm, agent_id, local_idx)
        valid_intents = sum(1 for o in intent_list if bool(o.get("valid", False)))
        out.append(
            "<br>".join(
                [
                    f"agent={agent_id}",
                    f"pos=({px:.2f}, {py:.2f}, {pz:.2f})",
                    f"speed={speed:.2f} m/s",
                    f"v_cmd={cmd_speed:.2f} m/s",
                    f"goal_dir=({gx:.2f}, {gy:.2f}, {gz:.2f})",
                    f"min_msg_age={min_obs_age:.2f} s",
                    f"valid_intents={valid_intents}",
                    f"speed_sat={bool(speed_sat[local_idx])}",
                    f"accel_sat={bool(accel_sat[local_idx])}",
                ]
            )
        )
    return out


def _neighbor_segments(
    frm: dict[str, Any],
    agent_ids: list[int],
    focus_local_idxs: list[int] | None,
    max_sensed_per_agent: int,
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    pos = frm["positions"]
    id_to_local = {aid: i for i, aid in enumerate(agent_ids)}
    focus = focus_local_idxs if focus_local_idxs is not None else list(range(len(agent_ids)))
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for ego_local_idx in focus:
        ego_id = agent_ids[ego_local_idx]
        obs_list = _frame_obs_list(frm, ego_id, ego_local_idx)
        ex, ey, ez = [float(x) for x in pos[ego_local_idx]]
        for obs in obs_list[:max_sensed_per_agent]:
            nbr_id = int(obs["idx"])
            nbr_local_idx = id_to_local.get(nbr_id)
            if nbr_local_idx is None:
                continue
            nx, ny, nz = [float(x) for x in pos[nbr_local_idx]]
            xs.extend([ex, nx, None])
            ys.extend([ey, ny, None])
            zs.extend([ez, nz, None])
    return xs, ys, zs


def _collect_intent_tubes(
    frm: dict[str, Any],
    agent_ids: list[int],
    focus_sender_ids: set[int] | None = None,
) -> list[dict[str, Any]]:
    best_by_sender: dict[int, dict[str, Any]] = {}
    for ego_local_idx, ego_id in enumerate(agent_ids):
        for obs in _frame_intent_list(frm, ego_id, ego_local_idx):
            sender_id = int(obs.get("idx", -1))
            if sender_id < 0:
                continue
            if focus_sender_ids is not None and sender_id not in focus_sender_ids:
                continue
            if not bool(obs.get("valid", False)):
                continue
            points = obs.get("points") or []
            if len(points) < 2:
                continue
            age = float(obs.get("intent_age_s", 0.0))
            prev = best_by_sender.get(sender_id)
            if prev is None or age < float(prev.get("intent_age_s", 1e9)):
                best_by_sender[sender_id] = {
                    "sender_id": sender_id,
                    "intent_age_s": age,
                    "kind": str(obs.get("kind", "")),
                    "tube_radius_m": float(obs.get("tube_radius_m", 0.0)),
                    "points": points,
                }
    return [best_by_sender[k] for k in sorted(best_by_sender)]


def _tube_segments(tubes: list[dict[str, Any]]) -> tuple[list[float | None], list[float | None], list[float | None]]:
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for tube in tubes:
        pts = tube.get("points", [])
        for p in pts:
            xs.append(float(p[0]))
            ys.append(float(p[1]))
            zs.append(float(p[2]))
        xs.append(None)
        ys.append(None)
        zs.append(None)
    return xs, ys, zs


def render_interactive_trace(
    trace_path: str,
    out_path: str,
    tail: int = 40,
    show_sensed: bool = True,
    max_sensed_per_agent: int = 8,
) -> Path:
    meta, frames = _load_trace(trace_path)
    opath = Path(out_path)

    agent_ids = frames[0].get("agent_ids")
    if agent_ids is None:
        agent_ids = meta.get("agent_ids")
    if agent_ids is None:
        agent_ids = list(range(len(frames[0]["positions"])))

    collision_pair = _infer_collision_pair(meta, trace_path, agent_ids)

    all_pos = [p for frm in frames for p in frm["positions"]]
    xs = [float(p[0]) for p in all_pos]
    ys = [float(p[1]) for p in all_pos]
    zs = [float(p[2]) for p in all_pos]
    pad = 2.0
    x_range = [min(xs) - pad, max(xs) + pad]
    y_range = [min(ys) - pad, max(ys) + pad]
    z_range = [min(zs) - pad, max(zs) + pad]

    base_colors = [
        "#E45756" if collision_pair and i in collision_pair else "#4C78A8"
        for i in range(len(agent_ids))
    ]
    dim_colors = [
        "rgba(228,87,86,0.18)" if collision_pair and i in collision_pair else "rgba(76,120,168,0.18)"
        for i in range(len(agent_ids))
    ]

    dynamic_traces: list[dict[str, Any]] = [
        {
            "type": "scatter3d",
            "mode": "markers",
            "name": "agents",
            "x": [],
            "y": [],
            "z": [],
            "text": [],
            "hovertemplate": "%{text}<extra></extra>",
            "marker": {
                "size": 7,
                "color": base_colors,
                "line": {"width": 1, "color": "#1F1F1F"},
                "opacity": 0.95,
            },
        }
    ]
    for i, agent_id in enumerate(agent_ids):
        dynamic_traces.append(
            {
                "type": "scatter3d",
                "mode": "lines",
                "name": f"trail_{agent_id}",
                "x": [],
                "y": [],
                "z": [],
                "line": {"color": base_colors[i], "width": 4},
                "opacity": 0.45,
                "hoverinfo": "skip",
                "showlegend": False,
            }
        )
    dynamic_traces.extend(
        [
            {
                "type": "scatter3d",
                "mode": "lines",
                "name": "focus_pair",
                "x": [],
                "y": [],
                "z": [],
                "line": {"color": "#E45756", "width": 6},
                "hoverinfo": "skip",
                "showlegend": True,
            },
            {
                "type": "scatter3d",
                "mode": "lines",
                "name": "sensed_neighbors",
                "x": [],
                "y": [],
                "z": [],
                "line": {"color": "rgba(242,161,4,0.50)", "width": 2, "dash": "dot"},
                "hoverinfo": "skip",
                "showlegend": bool(show_sensed),
                "visible": bool(show_sensed),
            },
            {
                "type": "scatter3d",
                "mode": "lines",
                "name": "intent_tubes",
                "x": [],
                "y": [],
                "z": [],
                "line": {"color": "rgba(33,158,188,0.90)", "width": 5},
                "hoverinfo": "skip",
                "showlegend": True,
            },
        ]
    )

    static_traces = _trace_segments(meta)
    obstacles = meta.get("obstacles", []) or []
    focus_local_idxs = list(collision_pair) if collision_pair else None
    focus_sender_ids = {agent_ids[i] for i in focus_local_idxs} if focus_local_idxs else None

    frame_payloads: list[dict[str, Any]] = []
    ts: list[float] = []
    min_pair_series: list[float | None] = []
    focus_pair_series: list[float | None] = []
    obstacle_clearance_series: list[float | None] = []
    valid_intent_series: list[int] = []

    for frame_idx, frm in enumerate(frames):
        positions = frm["positions"]
        pair_dist = _pair_distance(positions, *collision_pair) if collision_pair else None
        min_pair = _min_pair_distance(frm)
        obstacle_clearance = _min_obstacle_clearance(frm, obstacles)
        all_tubes = _collect_intent_tubes(frm, agent_ids)
        focus_tubes = _collect_intent_tubes(frm, agent_ids, focus_sender_ids)
        all_neighbors = _neighbor_segments(frm, agent_ids, None, max_sensed_per_agent)
        focus_neighbors = _neighbor_segments(frm, agent_ids, focus_local_idxs, max_sensed_per_agent)
        pair_line = {"x": [], "y": [], "z": []}
        if collision_pair:
            i, j = collision_pair
            pair_line = {
                "x": [float(positions[i][0]), float(positions[j][0])],
                "y": [float(positions[i][1]), float(positions[j][1])],
                "z": [float(positions[i][2]), float(positions[j][2])],
            }
        frame_payloads.append(
            {
                "t": float(frm["t"]),
                "positions": positions,
                "hover": _agent_hover_text(frm, agent_ids),
                "trails": [
                    {"x": tx, "y": ty, "z": tz}
                    for tx, ty, tz in (_trail_xyz(frames, frame_idx, aidx, tail) for aidx in range(len(agent_ids)))
                ],
                "neighbors_all": {"x": all_neighbors[0], "y": all_neighbors[1], "z": all_neighbors[2]},
                "neighbors_focus": {"x": focus_neighbors[0], "y": focus_neighbors[1], "z": focus_neighbors[2]},
                "intent_all": {"x": _tube_segments(all_tubes)[0], "y": _tube_segments(all_tubes)[1], "z": _tube_segments(all_tubes)[2]},
                "intent_focus": {"x": _tube_segments(focus_tubes)[0], "y": _tube_segments(focus_tubes)[1], "z": _tube_segments(focus_tubes)[2]},
                "pair_line": pair_line,
                "summary": {
                    "min_pair_distance": min_pair,
                    "focus_pair_distance": pair_dist,
                    "min_obstacle_clearance": obstacle_clearance,
                    "valid_intents": len(all_tubes),
                },
            }
        )
        ts.append(float(frm["t"]))
        min_pair_series.append(min_pair)
        focus_pair_series.append(pair_dist)
        obstacle_clearance_series.append(obstacle_clearance)
        valid_intent_series.append(len(all_tubes))

    replay = {
        "trace_name": Path(trace_path).name,
        "scenario": meta.get("scenario_name", meta.get("scenario", "unknown")),
        "method": meta.get("method", "unknown"),
        "comm_profile": meta.get("comm_profile", "unknown"),
        "trace_type": meta.get("trace_type", "window"),
        "agent_ids": agent_ids,
        "collision_pair": collision_pair,
        "base_colors": base_colors,
        "dim_colors": dim_colors,
        "frame_count": len(frame_payloads),
        "frames": frame_payloads,
        "series": {
            "t": ts,
            "min_pair_distance": min_pair_series,
            "focus_pair_distance": focus_pair_series,
            "min_obstacle_clearance": obstacle_clearance_series,
            "valid_intents": valid_intent_series,
        },
        "main_scene": {
            "x_range": x_range,
            "y_range": y_range,
            "z_range": z_range,
            "show_sensed": bool(show_sensed),
        },
    }

    initial_frame = frame_payloads[0]
    title = f"{escape(Path(trace_path).name)} | t={initial_frame['t']:.2f}s | agents={len(agent_ids)}"
    if initial_frame["summary"]["min_pair_distance"] is not None:
        title += f" | min_pair={initial_frame['summary']['min_pair_distance']:.2f}m"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Microbench Interactive Replay</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, sans-serif; background: linear-gradient(180deg, #f5f7fb 0%, #edf2f7 100%); color: #17212b; }}
    .wrap {{ max-width: 1680px; margin: 0 auto; padding: 18px; }}
    .meta, .controls {{ display: grid; gap: 12px; margin-bottom: 12px; }}
    .meta {{ grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); }}
    .controls {{ grid-template-columns: 160px 160px 1fr 220px 220px; align-items: center; }}
    .card, .panel {{ background: rgba(255,255,255,0.9); border: 1px solid #d8dfeb; border-radius: 14px; box-shadow: 0 8px 28px rgba(17,24,39,0.06); }}
    .card {{ padding: 10px 12px; }}
    .label {{ font-size: 12px; color: #667085; margin-bottom: 4px; }}
    .value {{ font-size: 14px; font-weight: 700; }}
    .controls .panel {{ padding: 12px; display: flex; gap: 10px; align-items: center; }}
    .controls button {{ background: #0f766e; color: white; border: none; border-radius: 10px; padding: 8px 14px; font-weight: 700; cursor: pointer; }}
    .controls button.secondary {{ background: #475467; }}
    .controls label {{ font-size: 13px; font-weight: 600; color: #344054; }}
    .slider-wrap {{ display: flex; gap: 10px; align-items: center; }}
    input[type=range] {{ width: 100%; }}
    #main-plot {{ width: 100%; height: 72vh; }}
    .charts {{ display: grid; grid-template-columns: 1.2fr 1fr; gap: 12px; margin-top: 12px; }}
    #distance-plot, #clearance-plot {{ width: 100%; height: 32vh; }}
    .hint {{ margin-top: 10px; color: #475467; font-size: 13px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="meta">
      <div class="card"><div class="label">Scenario</div><div class="value">{escape(str(meta.get("scenario_name", meta.get("scenario", "unknown"))))}</div></div>
      <div class="card"><div class="label">Method</div><div class="value">{escape(str(meta.get("method", "unknown")))}</div></div>
      <div class="card"><div class="label">Comm</div><div class="value">{escape(str(meta.get("comm_profile", "unknown")))}</div></div>
      <div class="card"><div class="label">Agents</div><div class="value">{len(agent_ids)}</div></div>
      <div class="card"><div class="label">Trace Type</div><div class="value">{escape(str(meta.get("trace_type", "window")))}</div></div>
      <div class="card"><div class="label">Time</div><div class="value" id="metric-time">0.00s</div></div>
      <div class="card"><div class="label">Min Pair Dist</div><div class="value" id="metric-minpair">--</div></div>
      <div class="card"><div class="label">Focus Pair Dist</div><div class="value" id="metric-focuspair">--</div></div>
      <div class="card"><div class="label">Obstacle Clearance</div><div class="value" id="metric-clearance">--</div></div>
      <div class="card"><div class="label">Visible Intent Tubes</div><div class="value" id="metric-intents">0</div></div>
    </div>

    <div class="controls">
      <div class="panel">
        <button id="play">Play</button>
        <button id="pause" class="secondary">Pause</button>
      </div>
      <div class="panel">
        <button id="prev" class="secondary">Prev</button>
        <button id="next" class="secondary">Next</button>
      </div>
      <div class="panel slider-wrap">
        <label for="frame-slider">Frame</label>
        <input id="frame-slider" type="range" min="0" max="{len(frame_payloads) - 1}" value="0" step="1" />
        <span id="frame-label">0 / {len(frame_payloads) - 1}</span>
      </div>
      <div class="panel">
        <label><input id="focus-toggle" type="checkbox" {'checked' if collision_pair else ''} {' ' if collision_pair else 'disabled '} /> Focus collision pair</label>
      </div>
      <div class="panel">
        <label><input id="sensed-toggle" type="checkbox" {'checked' if show_sensed else ''} /> Show sensed links</label>
      </div>
    </div>

    <div class="panel"><div id="main-plot"></div></div>
    <div class="charts">
      <div class="panel"><div id="distance-plot"></div></div>
      <div class="panel"><div id="clearance-plot"></div></div>
    </div>
    <div class="hint">Drag to orbit, scroll to zoom, scrub with the slider, and use focus mode to isolate the collision pair. Gray wireframes are world bounds and obstacles; cyan polylines are received intent tubes.</div>
  </div>

  <script>
    const replay = {json.dumps(replay)};
    const staticTraces = {json.dumps(static_traces)};
    const baseColors = replay.base_colors;
    const dimColors = replay.dim_colors;
    const agentCount = replay.agent_ids.length;
    const hasCollisionPair = Array.isArray(replay.collision_pair) && replay.collision_pair.length === 2;

    const MAIN_TRACE = {{ agents: 0, trailsStart: 1, pair: 1 + agentCount, sensed: 2 + agentCount, intents: 3 + agentCount }};
    const focusToggle = document.getElementById('focus-toggle');
    const sensedToggle = document.getElementById('sensed-toggle');
    const frameSlider = document.getElementById('frame-slider');
    const frameLabel = document.getElementById('frame-label');
    let currentFrame = 0;
    let playHandle = null;

    function metricFmt(v) {{
      if (v === null || v === undefined || Number.isNaN(v)) return '--';
      return `${{v.toFixed(2)}} m`;
    }}

    function buildMainData() {{
      const traces = [];
      traces.push({{
        type: 'scatter3d', mode: 'markers', name: 'agents', x: [], y: [], z: [], text: [],
        hovertemplate: '%{{text}}<extra></extra>',
        marker: {{ size: 7, color: baseColors.slice(), line: {{ width: 1, color: '#1F1F1F' }}, opacity: 0.95 }}
      }});
      for (let i = 0; i < agentCount; i += 1) {{
        traces.push({{
          type: 'scatter3d', mode: 'lines', name: `trail_${{replay.agent_ids[i]}}`, x: [], y: [], z: [],
          line: {{ color: baseColors[i], width: 4 }}, opacity: 0.45, hoverinfo: 'skip', showlegend: false
        }});
      }}
      traces.push({{ type: 'scatter3d', mode: 'lines', name: 'focus_pair', x: [], y: [], z: [], line: {{ color: '#E45756', width: 6 }}, hoverinfo: 'skip', showlegend: true }});
      traces.push({{ type: 'scatter3d', mode: 'lines', name: 'sensed_neighbors', x: [], y: [], z: [], line: {{ color: 'rgba(242,161,4,0.50)', width: 2, dash: 'dot' }}, hoverinfo: 'skip', showlegend: true, visible: replay.main_scene.show_sensed }});
      traces.push({{ type: 'scatter3d', mode: 'lines', name: 'intent_tubes', x: [], y: [], z: [], line: {{ color: 'rgba(33,158,188,0.90)', width: 5 }}, hoverinfo: 'skip', showlegend: true }});
      return traces.concat(staticTraces);
    }}

    function buildTitle(frame) {{
      let title = `${{replay.trace_name}} | t=${{frame.t.toFixed(2)}}s | agents=${{agentCount}}`;
      if (frame.summary.min_pair_distance !== null && frame.summary.min_pair_distance !== undefined) {{
        title += ` | min_pair=${{frame.summary.min_pair_distance.toFixed(2)}}m`;
      }}
      if (frame.summary.focus_pair_distance !== null && frame.summary.focus_pair_distance !== undefined && focusToggle.checked) {{
        title += ` | focus_pair=${{frame.summary.focus_pair_distance.toFixed(2)}}m`;
      }}
      return title;
    }}

    function updateMetricCards(frame) {{
      document.getElementById('metric-time').textContent = `${{frame.t.toFixed(2)}}s`;
      document.getElementById('metric-minpair').textContent = metricFmt(frame.summary.min_pair_distance);
      document.getElementById('metric-focuspair').textContent = metricFmt(frame.summary.focus_pair_distance);
      document.getElementById('metric-clearance').textContent = metricFmt(frame.summary.min_obstacle_clearance);
      document.getElementById('metric-intents').textContent = String(frame.summary.valid_intents);
    }}

    function updateMainPlot(frameIndex) {{
      const frame = replay.frames[frameIndex];
      const focusOn = hasCollisionPair && focusToggle.checked;
      const focusIdxs = focusOn ? replay.collision_pair : replay.agent_ids.map((_, idx) => idx);
      const focusSet = new Set(focusIdxs);
      const agentColors = replay.agent_ids.map((_, idx) => focusOn ? (focusSet.has(idx) ? baseColors[idx] : dimColors[idx]) : baseColors[idx]);
      const agentSizes = replay.agent_ids.map((_, idx) => focusOn ? (focusSet.has(idx) ? 10 : 5) : 7);
      const markerOpacity = replay.agent_ids.map((_, idx) => focusOn ? (focusSet.has(idx) ? 0.98 : 0.25) : 0.95);
      Plotly.restyle('main-plot', {{
        x: [[...frame.positions.map(p => p[0])]],
        y: [[...frame.positions.map(p => p[1])]],
        z: [[...frame.positions.map(p => p[2])]],
        text: [[...frame.hover]],
        'marker.color': [agentColors],
        'marker.size': [agentSizes],
        'marker.opacity': [markerOpacity],
      }}, [MAIN_TRACE.agents]);

      for (let i = 0; i < agentCount; i += 1) {{
        const trail = frame.trails[i];
        Plotly.restyle('main-plot', {{
          x: [[...trail.x]],
          y: [[...trail.y]],
          z: [[...trail.z]],
          opacity: [focusOn ? (focusSet.has(i) ? 0.85 : 0.12) : 0.45],
          'line.color': [focusOn ? (focusSet.has(i) ? baseColors[i] : dimColors[i]) : baseColors[i]],
        }}, [MAIN_TRACE.trailsStart + i]);
      }}

      const pairLine = (focusOn && frame.pair_line.x.length) ? frame.pair_line : {{ x: [], y: [], z: [] }};
      Plotly.restyle('main-plot', {{ x: [[...pairLine.x]], y: [[...pairLine.y]], z: [[...pairLine.z]] }}, [MAIN_TRACE.pair]);

      const sensedVisible = sensedToggle.checked;
      const sensed = focusOn ? frame.neighbors_focus : frame.neighbors_all;
      Plotly.restyle('main-plot', {{
        x: [[...(sensedVisible ? sensed.x : [])]],
        y: [[...(sensedVisible ? sensed.y : [])]],
        z: [[...(sensedVisible ? sensed.z : [])]],
        visible: [sensedVisible],
      }}, [MAIN_TRACE.sensed]);

      const intents = focusOn ? frame.intent_focus : frame.intent_all;
      Plotly.restyle('main-plot', {{ x: [[...intents.x]], y: [[...intents.y]], z: [[...intents.z]] }}, [MAIN_TRACE.intents]);
      Plotly.relayout('main-plot', {{ title: {{ text: buildTitle(frame) }} }});
      updateMetricCards(frame);
    }}

    function cursorShape(x) {{
      return [{{ type: 'line', x0: x, x1: x, y0: 0, y1: 1, yref: 'paper', line: {{ color: '#111827', width: 2, dash: 'dot' }} }}];
    }}

    function updateTimeSeries(frameIndex) {{
      const t = replay.frames[frameIndex].t;
      Plotly.relayout('distance-plot', {{ shapes: cursorShape(t) }});
      Plotly.relayout('clearance-plot', {{ shapes: cursorShape(t) }});
    }}

    function applyFrame(frameIndex) {{
      currentFrame = Math.max(0, Math.min(frameIndex, replay.frame_count - 1));
      frameSlider.value = String(currentFrame);
      frameLabel.textContent = `${{currentFrame}} / ${{replay.frame_count - 1}}`;
      updateMainPlot(currentFrame);
      updateTimeSeries(currentFrame);
    }}

    function stopPlay() {{
      if (playHandle !== null) {{
        clearInterval(playHandle);
        playHandle = null;
      }}
    }}

    function startPlay() {{
      stopPlay();
      playHandle = setInterval(() => {{
        const next = currentFrame + 1;
        if (next >= replay.frame_count) {{
          stopPlay();
          return;
        }}
        applyFrame(next);
      }}, 60);
    }}

    const mainLayout = {{
      title: {{ text: {json.dumps(title)} }},
      template: 'plotly_white',
      legend: {{ orientation: 'h' }},
      margin: {{ l: 0, r: 0, b: 0, t: 42 }},
      scene: {{
        xaxis: {{ title: 'x', range: replay.main_scene.x_range }},
        yaxis: {{ title: 'y', range: replay.main_scene.y_range }},
        zaxis: {{ title: 'z', range: replay.main_scene.z_range }},
        aspectmode: 'data',
        camera: {{ eye: {{ x: 1.5, y: 1.2, z: 1.1 }} }},
      }},
    }};

    const distanceData = [
      {{ type: 'scatter', mode: 'lines', name: 'nearest neighbor', x: replay.series.t, y: replay.series.min_pair_distance, line: {{ color: '#4C78A8', width: 3 }} }},
      {{ type: 'scatter', mode: 'lines', name: 'focus pair', x: replay.series.t, y: replay.series.focus_pair_distance, line: {{ color: '#E45756', width: 3, dash: 'dash' }}, visible: hasCollisionPair }}
    ];
    const clearanceData = [
      {{ type: 'scatter', mode: 'lines', name: 'obstacle clearance', x: replay.series.t, y: replay.series.min_obstacle_clearance, line: {{ color: '#6D5EF0', width: 3 }} }},
      {{ type: 'scatter', mode: 'lines', name: 'visible intents', x: replay.series.t, y: replay.series.valid_intents, yaxis: 'y2', line: {{ color: '#219EBC', width: 2 }} }}
    ];
    const distanceLayout = {{
      title: {{ text: 'Neighbor Distances' }},
      template: 'plotly_white',
      margin: {{ l: 50, r: 20, b: 40, t: 40 }},
      xaxis: {{ title: 't (s)' }},
      yaxis: {{ title: 'distance (m)' }},
      legend: {{ orientation: 'h' }},
      shapes: cursorShape(replay.frames[0].t),
    }};
    const clearanceLayout = {{
      title: {{ text: 'Obstacle Clearance and Intent Visibility' }},
      template: 'plotly_white',
      margin: {{ l: 50, r: 50, b: 40, t: 40 }},
      xaxis: {{ title: 't (s)' }},
      yaxis: {{ title: 'clearance (m)' }},
      yaxis2: {{ title: 'intent tubes', overlaying: 'y', side: 'right', rangemode: 'tozero' }},
      legend: {{ orientation: 'h' }},
      shapes: cursorShape(replay.frames[0].t),
    }};

    Plotly.newPlot('main-plot', buildMainData(), mainLayout, {{ responsive: true }});
    Plotly.newPlot('distance-plot', distanceData, distanceLayout, {{ responsive: true }});
    Plotly.newPlot('clearance-plot', clearanceData, clearanceLayout, {{ responsive: true }});
    applyFrame(0);

    document.getElementById('play').addEventListener('click', startPlay);
    document.getElementById('pause').addEventListener('click', stopPlay);
    document.getElementById('prev').addEventListener('click', () => {{ stopPlay(); applyFrame(currentFrame - 1); }});
    document.getElementById('next').addEventListener('click', () => {{ stopPlay(); applyFrame(currentFrame + 1); }});
    frameSlider.addEventListener('input', (ev) => {{ stopPlay(); applyFrame(Number(ev.target.value)); }});
    focusToggle.addEventListener('change', () => applyFrame(currentFrame));
    sensedToggle.addEventListener('change', () => applyFrame(currentFrame));
  </script>
</body>
</html>
"""
    opath.write_text(html, encoding="utf-8")
    return opath
