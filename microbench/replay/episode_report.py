from __future__ import annotations

from html import escape
import json
import math
from pathlib import Path
from typing import Any

from microbench.replay.trace_io import _load_trace, _trace_segments


COLORS = (
    "#2563eb",
    "#dc2626",
    "#059669",
    "#d97706",
    "#7c3aed",
    "#0891b2",
    "#be185d",
    "#4d7c0f",
    "#9333ea",
    "#0f766e",
    "#b45309",
    "#1d4ed8",
)


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _dist(a: list[float], b: list[float]) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    dz = float(a[2]) - float(b[2])
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _speed(v: list[float]) -> float:
    return math.sqrt(float(v[0]) ** 2 + float(v[1]) ** 2 + float(v[2]) ** 2)


def _pair_distances(positions: list[list[float]]) -> list[tuple[tuple[int, int], float]]:
    out: list[tuple[tuple[int, int], float]] = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            out.append(((i, j), _dist(positions[i], positions[j])))
    return out


def _nearest_pair(positions: list[list[float]]) -> tuple[tuple[int, int] | None, float | None]:
    pairs = _pair_distances(positions)
    if not pairs:
        return None, None
    pair, distance = min(pairs, key=lambda item: item[1])
    return pair, distance


def _frame_obs_list(frm: dict[str, Any], ego_id: int, ego_local_idx: int) -> list[dict[str, Any]]:
    selected_obs = frm.get("selected_obs", {})
    if isinstance(selected_obs, dict):
        return selected_obs.get(str(ego_id), [])
    if isinstance(selected_obs, list) and ego_local_idx < len(selected_obs):
        return selected_obs[ego_local_idx]
    return []


def _obs_age_stats(frm: dict[str, Any], agent_ids: list[int]) -> tuple[float | None, float | None, int]:
    ages: list[float] = []
    count = 0
    for local_idx, agent_id in enumerate(agent_ids):
        for obs in _frame_obs_list(frm, int(agent_id), local_idx):
            count += 1
            age = _finite(obs.get("msg_age_sec"))
            if age is not None:
                ages.append(age)
    if not ages:
        return None, None, count
    return sum(ages) / len(ages), max(ages), count


def _downsample_frames(frames: list[dict[str, Any]], max_frames: int | None) -> list[dict[str, Any]]:
    if max_frames is None or max_frames <= 0 or len(frames) <= max_frames:
        return frames
    if max_frames == 1:
        return [frames[0]]
    step = (len(frames) - 1) / float(max_frames - 1)
    idxs = sorted({min(len(frames) - 1, round(i * step)) for i in range(max_frames)})
    return [frames[i] for i in idxs]


def _event_rows(trace_path: str) -> list[dict[str, Any]]:
    events_path = Path(trace_path).with_name("events.jsonl")
    if not events_path.exists():
        return []
    rows = []
    with events_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _agent_label(meta: dict[str, Any], agent_id: int) -> str:
    profiles = meta.get("agent_profiles", [])
    if isinstance(profiles, list):
        for profile in profiles:
            if int(profile.get("agent_id", -1)) != int(agent_id):
                continue
            role = profile.get("role")
            priority = profile.get("priority")
            if role is not None:
                return f"{agent_id} ({role}, p={priority})"
    return str(agent_id)


def _axis_range(vals: list[float], pad: float = 2.0) -> list[float]:
    if not vals:
        return [-1.0, 1.0]
    lo = min(vals)
    hi = max(vals)
    if math.isclose(lo, hi):
        return [lo - pad, hi + pad]
    return [lo - pad, hi + pad]


def _null_safe(values: list[float | None]) -> list[float | None]:
    return [None if v is None or not math.isfinite(float(v)) else float(v) for v in values]


def _plotly_script_tag(source: str) -> str:
    if source not in {"auto", "cdn", "inline"}:
        raise ValueError(f"unknown Plotly source: {source}")
    if source in {"auto", "inline"}:
        try:
            from plotly.offline import get_plotlyjs  # type: ignore[import-untyped]
        except ImportError:
            if source == "inline":
                raise RuntimeError("Plotly is required for inline episode reports; install daa-microbench[viz]") from None
        else:
            return f"<script>{get_plotlyjs()}</script>"
    return '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>'


def _build_payload(
    *,
    trace_path: str,
    meta: dict[str, Any],
    frames: list[dict[str, Any]],
    source_frame_count: int,
) -> dict[str, Any]:
    agent_ids = frames[0].get("agent_ids") or meta.get("agent_ids") or list(range(len(frames[0].get("positions", []))))
    agent_ids = [int(x) for x in agent_ids]
    n_agents = len(agent_ids)
    colors = [COLORS[i % len(COLORS)] for i in range(n_agents)]

    positions_by_agent = [[] for _ in range(n_agents)]
    altitude_by_agent = [[] for _ in range(n_agents)]
    hover_by_agent = [[] for _ in range(n_agents)]
    speeds_by_agent = [[] for _ in range(n_agents)]
    cmd_speeds_by_agent = [[] for _ in range(n_agents)]
    t_values: list[float] = []
    nearest_pairs: list[tuple[int, int] | None] = []
    nearest_distances: list[float | None] = []
    mean_speeds: list[float | None] = []
    max_speeds: list[float | None] = []
    mean_cmd_speeds: list[float | None] = []
    speed_sat_counts: list[int] = []
    accel_sat_counts: list[int] = []
    mean_msg_ages: list[float | None] = []
    max_msg_ages: list[float | None] = []
    obs_counts: list[int] = []

    global_nearest_pair: tuple[int, int] | None = None
    global_nearest_distance: float | None = None
    global_nearest_frame = 0

    for frame_idx, frm in enumerate(frames):
        t = float(frm.get("t", frame_idx))
        t_values.append(t)
        positions = frm.get("positions", [])
        velocities = frm.get("velocities", [[0.0, 0.0, 0.0] for _ in positions])
        commands = frm.get("v_cmd", [[0.0, 0.0, 0.0] for _ in positions])
        pair, distance = _nearest_pair(positions)
        nearest_pairs.append(pair)
        nearest_distances.append(distance)
        if distance is not None and (global_nearest_distance is None or distance < global_nearest_distance):
            global_nearest_distance = distance
            global_nearest_pair = pair
            global_nearest_frame = frame_idx

        speeds = [_speed(v) for v in velocities]
        cmd_speeds = [_speed(v) for v in commands]
        mean_speeds.append(sum(speeds) / len(speeds) if speeds else None)
        max_speeds.append(max(speeds) if speeds else None)
        mean_cmd_speeds.append(sum(cmd_speeds) / len(cmd_speeds) if cmd_speeds else None)
        speed_sat_counts.append(sum(1 for x in frm.get("speed_saturated", []) if bool(x)))
        accel_sat_counts.append(sum(1 for x in frm.get("accel_saturated", []) if bool(x)))
        mean_age, max_age, obs_count = _obs_age_stats(frm, agent_ids)
        mean_msg_ages.append(mean_age)
        max_msg_ages.append(max_age)
        obs_counts.append(obs_count)

        for i in range(n_agents):
            pos = positions[i]
            positions_by_agent[i].append([float(pos[0]), float(pos[1]), float(pos[2])])
            altitude_by_agent[i].append(float(pos[1]))
            speed_value = speeds[i] if i < len(speeds) else 0.0
            cmd_value = cmd_speeds[i] if i < len(cmd_speeds) else 0.0
            speeds_by_agent[i].append(speed_value)
            cmd_speeds_by_agent[i].append(cmd_value)
            label = _agent_label(meta, agent_ids[i])
            hover_by_agent[i].append(
                "<br>".join(
                    [
                        f"agent={escape(label)}",
                        f"t={t:.2f}s",
                        f"pos=({float(pos[0]):.2f}, {float(pos[1]):.2f}, {float(pos[2]):.2f})",
                        f"speed={speed_value:.2f} m/s",
                        f"cmd={cmd_value:.2f} m/s",
                    ]
                )
            )

    focus_pair = None
    if "i" in meta and "j" in meta:
        id_to_local = {agent_id: idx for idx, agent_id in enumerate(agent_ids)}
        if int(meta["i"]) in id_to_local and int(meta["j"]) in id_to_local:
            focus_pair = [id_to_local[int(meta["i"])], id_to_local[int(meta["j"])]]
    if focus_pair is None and global_nearest_pair is not None:
        focus_pair = [global_nearest_pair[0], global_nearest_pair[1]]

    focus_distance: list[float | None] = []
    if focus_pair is not None:
        i, j = focus_pair
        for frm in frames:
            positions = frm.get("positions", [])
            focus_distance.append(_dist(positions[i], positions[j]))
    else:
        focus_distance = [None for _ in frames]

    all_positions = [p for agent in positions_by_agent for p in agent]
    xs = [p[0] for p in all_positions]
    ys = [p[1] for p in all_positions]
    zs = [p[2] for p in all_positions]

    events = [
        {
            "t": _finite(row.get("t")),
            "type": str(row.get("type", "")),
            "i": row.get("i"),
            "j": row.get("j"),
            "dist": _finite(row.get("dist")),
        }
        for row in _event_rows(trace_path)
        if _finite(row.get("t")) is not None
    ]

    return {
        "schema_version": "0.1",
        "trace_path": str(trace_path),
        "trace_name": Path(trace_path).name,
        "scenario": meta.get("scenario_name", meta.get("scenario", "unknown")),
        "method": meta.get("method", "unknown"),
        "comm_profile": meta.get("comm_profile", "unknown"),
        "trace_type": meta.get("trace_type", "episode"),
        "world_bounds": meta.get("world_bounds", {}),
        "agent_ids": agent_ids,
        "agent_labels": [_agent_label(meta, agent_id) for agent_id in agent_ids],
        "colors": colors,
        "t": t_values,
        "source_frame_count": int(source_frame_count),
        "frame_count": len(frames),
        "positions_by_agent": positions_by_agent,
        "altitude_by_agent": altitude_by_agent,
        "hover_by_agent": hover_by_agent,
        "speeds_by_agent": speeds_by_agent,
        "cmd_speeds_by_agent": cmd_speeds_by_agent,
        "nearest_pairs": [[p[0], p[1]] if p is not None else None for p in nearest_pairs],
        "nearest_distances": _null_safe(nearest_distances),
        "focus_pair": focus_pair,
        "focus_distance": _null_safe(focus_distance),
        "global_nearest_distance": global_nearest_distance,
        "global_nearest_frame": global_nearest_frame,
        "mean_speeds": _null_safe(mean_speeds),
        "max_speeds": _null_safe(max_speeds),
        "mean_cmd_speeds": _null_safe(mean_cmd_speeds),
        "speed_sat_counts": speed_sat_counts,
        "accel_sat_counts": accel_sat_counts,
        "mean_msg_ages": _null_safe(mean_msg_ages),
        "max_msg_ages": _null_safe(max_msg_ages),
        "obs_counts": obs_counts,
        "events": events,
        "obstacle_traces": _trace_segments(meta),
        "ranges": {
            "x": _axis_range(xs),
            "y": _axis_range(ys),
            "z": _axis_range(zs),
        },
    }


def render_episode_report(
    trace_path: str,
    out_path: str,
    *,
    max_frames: int | None = 800,
    plotly_source: str = "auto",
) -> Path:
    meta, source_frames = _load_trace(trace_path)
    frames = _downsample_frames(source_frames, max_frames)
    payload = _build_payload(
        trace_path=trace_path,
        meta=meta,
        frames=frames,
        source_frame_count=len(source_frames),
    )
    opath = Path(out_path)
    opath.parent.mkdir(parents=True, exist_ok=True)

    title = f"{payload['scenario']} | {payload['method']} | episode report"
    plotly_script = _plotly_script_tag(plotly_source)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{escape(title)}</title>
  {plotly_script}
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f7fa;
      --panel: #ffffff;
      --ink: #172033;
      --muted: #667085;
      --line: #d9e1ec;
      --accent: #0f766e;
      --danger: #dc2626;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--ink); font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .wrap {{ max-width: 1760px; margin: 0 auto; padding: 18px; }}
    .title-row {{ display: flex; justify-content: space-between; align-items: flex-end; gap: 18px; margin-bottom: 12px; }}
    h1 {{ font-size: 22px; margin: 0; letter-spacing: 0; }}
    .subtitle {{ color: var(--muted); font-size: 13px; margin-top: 4px; }}
    .cards {{ display: grid; grid-template-columns: repeat(8, minmax(120px, 1fr)); gap: 10px; margin-bottom: 12px; }}
    .card, .panel, .controls {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06); }}
    .card {{ padding: 10px 12px; min-width: 0; }}
    .label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; }}
    .value {{ font-size: 15px; font-weight: 700; margin-top: 3px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .controls {{ display: grid; grid-template-columns: auto auto minmax(220px, 1fr) auto auto auto; align-items: center; gap: 10px; padding: 10px; margin-bottom: 12px; }}
    button {{ border: 0; border-radius: 8px; padding: 8px 12px; color: #fff; background: var(--accent); font-weight: 700; cursor: pointer; }}
    button.secondary {{ background: #475467; }}
    input[type=range] {{ width: 100%; }}
    label {{ color: #344054; font-size: 13px; font-weight: 650; }}
    .grid-main {{ display: grid; grid-template-columns: minmax(0, 1.08fr) minmax(420px, 0.92fr); gap: 12px; }}
    .projection-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    .plot {{ width: 100%; }}
    #topdown, #side, #distance, #speed, #comm {{ height: 340px; }}
    #scene3d {{ height: 692px; }}
    .bottom-grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-top: 12px; }}
    .note {{ color: var(--muted); font-size: 13px; margin-top: 12px; line-height: 1.45; }}
    @media (max-width: 1200px) {{
      .cards {{ grid-template-columns: repeat(2, minmax(120px, 1fr)); }}
      .controls {{ grid-template-columns: 1fr 1fr; }}
      .grid-main, .projection-grid, .bottom-grid {{ grid-template-columns: 1fr; }}
      #scene3d {{ height: 520px; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="title-row">
      <div>
        <h1>{escape(str(payload["scenario"]))}</h1>
        <div class="subtitle">DAA Microbench episode report: synchronized projections, 3D context, safety, compute, and sensing diagnostics.</div>
      </div>
      <div class="subtitle">{escape(Path(trace_path).name)}</div>
    </div>
    <div class="cards">
      <div class="card"><div class="label">method</div><div class="value">{escape(str(payload["method"]))}</div></div>
      <div class="card"><div class="label">comm</div><div class="value">{escape(str(payload["comm_profile"]))}</div></div>
      <div class="card"><div class="label">agents</div><div class="value">{len(payload["agent_ids"])}</div></div>
      <div class="card"><div class="label">frames</div><div class="value">{payload["frame_count"]}/{payload["source_frame_count"]}</div></div>
      <div class="card"><div class="label">t</div><div class="value" id="card-time">0.00 s</div></div>
      <div class="card"><div class="label">nearest</div><div class="value" id="card-nearest">--</div></div>
      <div class="card"><div class="label">focus pair</div><div class="value" id="card-focus">--</div></div>
      <div class="card"><div class="label">msg age max</div><div class="value" id="card-msg">--</div></div>
    </div>
    <div class="controls">
      <button id="play">Play</button>
      <button id="pause" class="secondary">Pause</button>
      <input id="frame" type="range" min="0" max="{payload["frame_count"] - 1}" step="1" value="0" />
      <div id="frame-label">0 / {payload["frame_count"] - 1}</div>
      <label><input type="checkbox" id="focus" checked /> emphasize focus pair</label>
      <label><input type="checkbox" id="ghost" checked /> show full trajectories</label>
    </div>
    <div class="grid-main">
      <div class="projection-grid">
        <div class="panel"><div id="topdown" class="plot"></div></div>
        <div class="panel"><div id="side" class="plot"></div></div>
        <div class="panel"><div id="distance" class="plot"></div></div>
        <div class="panel"><div id="speed" class="plot"></div></div>
      </div>
      <div class="panel"><div id="scene3d" class="plot"></div></div>
    </div>
    <div class="bottom-grid">
      <div class="panel"><div id="comm" class="plot"></div></div>
      <div class="panel"><div id="sat" class="plot"></div></div>
      <div class="panel"><div id="events" class="plot"></div></div>
    </div>
    <div class="note">
      Top-down uses x-z, side view uses x-altitude. The focus pair defaults to the collision pair when available, otherwise the globally closest pair. Use this report for diagnostics and papers; use Foxglove MCAP export for robotics-grade 3D playback.
    </div>
  </div>
  <script>
    const report = {json.dumps(payload)};
    const colors = report.colors;
    const n = report.agent_ids.length;
    const t = report.t;
    const focusPair = report.focus_pair;
    let current = report.global_nearest_frame || 0;
    let playing = null;

    const topTraceStart = 0;
    const sideTraceStart = 0;
    const sceneTraceStart = 0;
    const cursorColor = '#111827';

    function fmtMeters(v) {{
      return v === null || v === undefined ? '--' : `${{Number(v).toFixed(2)}} m`;
    }}
    function fmtSecs(v) {{
      return v === null || v === undefined ? '--' : `${{Number(v).toFixed(2)}} s`;
    }}
    function focusSet() {{
      return Array.isArray(focusPair) ? new Set(focusPair) : new Set();
    }}
    function fadedColor(hex, alpha) {{
      const value = hex.replace('#', '');
      const r = parseInt(value.slice(0, 2), 16);
      const g = parseInt(value.slice(2, 4), 16);
      const b = parseInt(value.slice(4, 6), 16);
      return `rgba(${{r}},${{g}},${{b}},${{alpha}})`;
    }}
    function markerColors() {{
      const emphasize = document.getElementById('focus').checked && focusPair;
      const fs = focusSet();
      return colors.map((c, i) => emphasize && !fs.has(i) ? fadedColor(c, 0.22) : c);
    }}
    function markerSizes() {{
      const emphasize = document.getElementById('focus').checked && focusPair;
      const fs = focusSet();
      return colors.map((_, i) => emphasize ? (fs.has(i) ? 13 : 6) : 9);
    }}
    function showTrajectories() {{
      return document.getElementById('ghost').checked;
    }}
    function cursorShape(x) {{
      return [{{ type: 'line', x0: x, x1: x, y0: 0, y1: 1, yref: 'paper', line: {{ color: cursorColor, width: 2, dash: 'dot' }} }}];
    }}
    function agentHover(i, frameIdx) {{
      return report.hover_by_agent[i][frameIdx];
    }}
    function currentPositions(frameIdx) {{
      return report.positions_by_agent.map(agent => agent[frameIdx]);
    }}
    function pairLine2d(frameIdx, axes) {{
      if (!focusPair) return {{ x: [], y: [] }};
      const p = currentPositions(frameIdx);
      const a = p[focusPair[0]];
      const b = p[focusPair[1]];
      if (axes === 'top') return {{ x: [a[0], b[0]], y: [a[2], b[2]] }};
      return {{ x: [a[0], b[0]], y: [a[1], b[1]] }};
    }}
    function pairLine3d(frameIdx) {{
      if (!focusPair) return {{ x: [], y: [], z: [] }};
      const p = currentPositions(frameIdx);
      const a = p[focusPair[0]];
      const b = p[focusPair[1]];
      return {{ x: [a[0], b[0]], y: [a[1], b[1]], z: [a[2], b[2]] }};
    }}

    function buildProjectionData(kind) {{
      const data = [];
      for (let i = 0; i < n; i += 1) {{
        const pts = report.positions_by_agent[i];
        data.push({{
          type: 'scatter',
          mode: 'lines',
          name: `agent ${{report.agent_labels[i]}} path`,
          x: pts.map(p => p[0]),
          y: kind === 'top' ? pts.map(p => p[2]) : pts.map(p => p[1]),
          line: {{ color: colors[i], width: 2 }},
          opacity: 0.38,
          hoverinfo: 'skip',
          showlegend: kind === 'top',
        }});
      }}
      const p = currentPositions(current);
      data.push({{
        type: 'scatter',
        mode: 'markers+text',
        name: 'current agents',
        x: p.map(v => v[0]),
        y: kind === 'top' ? p.map(v => v[2]) : p.map(v => v[1]),
        text: report.agent_ids.map(String),
        textposition: 'top center',
        customdata: report.agent_ids.map((_, i) => agentHover(i, current)),
        hovertemplate: '%{{customdata}}<extra></extra>',
        marker: {{ color: markerColors(), size: markerSizes(), line: {{ color: '#111827', width: 1 }} }},
        showlegend: false,
      }});
      const pair = pairLine2d(current, kind);
      data.push({{
        type: 'scatter',
        mode: 'lines',
        name: 'focus pair',
        x: pair.x,
        y: pair.y,
        line: {{ color: '#dc2626', width: 4 }},
        hoverinfo: 'skip',
        showlegend: false,
      }});
      return data;
    }}

    function build3dData() {{
      const data = [];
      for (let i = 0; i < n; i += 1) {{
        const pts = report.positions_by_agent[i];
        data.push({{
          type: 'scatter3d',
          mode: 'lines',
          name: `agent ${{report.agent_labels[i]}} path`,
          x: pts.map(p => p[0]),
          y: pts.map(p => p[1]),
          z: pts.map(p => p[2]),
          line: {{ color: colors[i], width: 4 }},
          opacity: 0.30,
          hoverinfo: 'skip',
          showlegend: false,
        }});
      }}
      const p = currentPositions(current);
      data.push({{
        type: 'scatter3d',
        mode: 'markers+text',
        name: 'current agents',
        x: p.map(v => v[0]),
        y: p.map(v => v[1]),
        z: p.map(v => v[2]),
        text: report.agent_ids.map(String),
        customdata: report.agent_ids.map((_, i) => agentHover(i, current)),
        hovertemplate: '%{{customdata}}<extra></extra>',
        marker: {{ color: markerColors(), size: markerSizes(), line: {{ color: '#111827', width: 1 }} }},
        showlegend: false,
      }});
      const pair = pairLine3d(current);
      data.push({{
        type: 'scatter3d',
        mode: 'lines',
        name: 'focus pair',
        x: pair.x,
        y: pair.y,
        z: pair.z,
        line: {{ color: '#dc2626', width: 7 }},
        hoverinfo: 'skip',
        showlegend: false,
      }});
      return data.concat(report.obstacle_traces || []);
    }}

    function layout2d(title, yTitle, yRange) {{
      return {{
        title: {{ text: title }},
        template: 'plotly_white',
        margin: {{ l: 48, r: 14, t: 42, b: 42 }},
        xaxis: {{ title: 'x (m)', range: report.ranges.x, zeroline: false }},
        yaxis: {{ title: yTitle, range: yRange, scaleanchor: title.includes('Top') ? 'x' : undefined, zeroline: false }},
        legend: {{ orientation: 'h', y: -0.2 }},
      }};
    }}

    function buildSeries() {{
      Plotly.newPlot('distance', [
        {{ type: 'scatter', mode: 'lines', name: 'nearest pair', x: t, y: report.nearest_distances, line: {{ color: '#2563eb', width: 3 }} }},
        {{ type: 'scatter', mode: 'lines', name: 'focus pair', x: t, y: report.focus_distance, line: {{ color: '#dc2626', width: 3, dash: 'dash' }} }},
      ], {{
        title: {{ text: 'Separation Over Time' }},
        template: 'plotly_white',
        margin: {{ l: 54, r: 18, t: 42, b: 42 }},
        xaxis: {{ title: 't (s)' }},
        yaxis: {{ title: 'center distance (m)' }},
        shapes: cursorShape(t[current]),
        legend: {{ orientation: 'h' }},
      }}, {{ responsive: true }});
      Plotly.newPlot('speed', [
        {{ type: 'scatter', mode: 'lines', name: 'mean speed', x: t, y: report.mean_speeds, line: {{ color: '#059669', width: 3 }} }},
        {{ type: 'scatter', mode: 'lines', name: 'max speed', x: t, y: report.max_speeds, line: {{ color: '#d97706', width: 3 }} }},
        {{ type: 'scatter', mode: 'lines', name: 'mean cmd', x: t, y: report.mean_cmd_speeds, line: {{ color: '#7c3aed', width: 2, dash: 'dot' }} }},
      ], {{
        title: {{ text: 'Speed and Command Magnitude' }},
        template: 'plotly_white',
        margin: {{ l: 54, r: 18, t: 42, b: 42 }},
        xaxis: {{ title: 't (s)' }},
        yaxis: {{ title: 'm/s' }},
        shapes: cursorShape(t[current]),
        legend: {{ orientation: 'h' }},
      }}, {{ responsive: true }});
      Plotly.newPlot('comm', [
        {{ type: 'scatter', mode: 'lines', name: 'mean msg age', x: t, y: report.mean_msg_ages, line: {{ color: '#0891b2', width: 3 }} }},
        {{ type: 'scatter', mode: 'lines', name: 'max msg age', x: t, y: report.max_msg_ages, line: {{ color: '#be185d', width: 3 }} }},
        {{ type: 'scatter', mode: 'lines', name: 'selected obs', x: t, y: report.obs_counts, yaxis: 'y2', line: {{ color: '#475467', width: 2 }} }},
      ], {{
        title: {{ text: 'Sensing / V2V Track Freshness' }},
        template: 'plotly_white',
        margin: {{ l: 54, r: 50, t: 42, b: 42 }},
        xaxis: {{ title: 't (s)' }},
        yaxis: {{ title: 'message age (s)' }},
        yaxis2: {{ title: 'obs count', overlaying: 'y', side: 'right', rangemode: 'tozero' }},
        shapes: cursorShape(t[current]),
        legend: {{ orientation: 'h' }},
      }}, {{ responsive: true }});
      Plotly.newPlot('sat', [
        {{ type: 'scatter', mode: 'lines', name: 'speed saturated', x: t, y: report.speed_sat_counts, line: {{ color: '#dc2626', width: 3 }} }},
        {{ type: 'scatter', mode: 'lines', name: 'accel saturated', x: t, y: report.accel_sat_counts, line: {{ color: '#d97706', width: 3 }} }},
      ], {{
        title: {{ text: 'Control Saturation Count' }},
        template: 'plotly_white',
        margin: {{ l: 54, r: 18, t: 42, b: 42 }},
        xaxis: {{ title: 't (s)' }},
        yaxis: {{ title: 'agents' }},
        shapes: cursorShape(t[current]),
        legend: {{ orientation: 'h' }},
      }}, {{ responsive: true }});
      const eventY = report.events.map(e => e.type === 'collision' ? 2 : 1);
      Plotly.newPlot('events', [
        {{ type: 'scatter', mode: 'markers', name: 'events', x: report.events.map(e => e.t), y: eventY, text: report.events.map(e => `${{e.type}} ${{e.i}}-${{e.j}} dist=${{e.dist}}`), marker: {{ color: eventY.map(v => v === 2 ? '#dc2626' : '#d97706'), size: 11 }}, hovertemplate: '%{{text}}<extra></extra>' }}
      ], {{
        title: {{ text: 'Near-Miss / Collision Events' }},
        template: 'plotly_white',
        margin: {{ l: 54, r: 18, t: 42, b: 42 }},
        xaxis: {{ title: 't (s)' }},
        yaxis: {{ tickmode: 'array', tickvals: [1, 2], ticktext: ['near miss', 'collision'], range: [0.5, 2.5] }},
        shapes: cursorShape(t[current]),
      }}, {{ responsive: true }});
    }}

    function updateCards() {{
      const nearest = report.nearest_distances[current];
      const focus = report.focus_distance[current];
      const maxAge = report.max_msg_ages[current];
      const pair = report.nearest_pairs[current];
      document.getElementById('card-time').textContent = `${{t[current].toFixed(2)}} s`;
      document.getElementById('card-nearest').textContent = pair ? `${{report.agent_ids[pair[0]]}}-${{report.agent_ids[pair[1]]}} / ${{fmtMeters(nearest)}}` : '--';
      document.getElementById('card-focus').textContent = focusPair ? `${{report.agent_ids[focusPair[0]]}}-${{report.agent_ids[focusPair[1]]}} / ${{fmtMeters(focus)}}` : '--';
      document.getElementById('card-msg').textContent = maxAge === null ? '--' : fmtSecs(maxAge);
      document.getElementById('frame-label').textContent = `${{current}} / ${{report.frame_count - 1}}`;
      document.getElementById('frame').value = current;
    }}

    function updatePlots(frameIdx) {{
      current = Math.max(0, Math.min(report.frame_count - 1, frameIdx));
      const p = currentPositions(current);
      const x = p.map(v => v[0]);
      const y = p.map(v => v[1]);
      const z = p.map(v => v[2]);
      const colorsNow = markerColors();
      const sizesNow = markerSizes();
      const topPair = pairLine2d(current, 'top');
      const sidePair = pairLine2d(current, 'side');
      const threePair = pairLine3d(current);

      Plotly.restyle('topdown', {{ x: [x], y: [z], customdata: [report.agent_ids.map((_, i) => agentHover(i, current))], 'marker.color': [colorsNow], 'marker.size': [sizesNow] }}, [n]);
      Plotly.restyle('topdown', {{ x: [topPair.x], y: [topPair.y] }}, [n + 1]);
      Plotly.restyle('side', {{ x: [x], y: [y], customdata: [report.agent_ids.map((_, i) => agentHover(i, current))], 'marker.color': [colorsNow], 'marker.size': [sizesNow] }}, [n]);
      Plotly.restyle('side', {{ x: [sidePair.x], y: [sidePair.y] }}, [n + 1]);
      Plotly.restyle('scene3d', {{ x: [x], y: [y], z: [z], customdata: [report.agent_ids.map((_, i) => agentHover(i, current))], 'marker.color': [colorsNow], 'marker.size': [sizesNow] }}, [n]);
      Plotly.restyle('scene3d', {{ x: [threePair.x], y: [threePair.y], z: [threePair.z] }}, [n + 1]);
      const opacity = showTrajectories() ? 0.38 : 0.04;
      for (let i = 0; i < n; i += 1) {{
        Plotly.restyle('topdown', {{ opacity: [opacity] }}, [i]);
        Plotly.restyle('side', {{ opacity: [opacity] }}, [i]);
        Plotly.restyle('scene3d', {{ opacity: [showTrajectories() ? 0.30 : 0.03] }}, [i]);
      }}
      for (const id of ['distance', 'speed', 'comm', 'sat', 'events']) {{
        Plotly.relayout(id, {{ shapes: cursorShape(t[current]) }});
      }}
      updateCards();
    }}

    function init() {{
      Plotly.newPlot('topdown', buildProjectionData('top'), layout2d('Top-Down View (x-z)', 'z (m)', report.ranges.z), {{ responsive: true }});
      Plotly.newPlot('side', buildProjectionData('side'), layout2d('Side / Altitude View (x-y)', 'altitude y (m)', report.ranges.y), {{ responsive: true }});
      Plotly.newPlot('scene3d', build3dData(), {{
        title: {{ text: '3D Context View' }},
        template: 'plotly_white',
        margin: {{ l: 0, r: 0, t: 42, b: 0 }},
        scene: {{
          xaxis: {{ title: 'x (m)', range: report.ranges.x }},
          yaxis: {{ title: 'altitude y (m)', range: report.ranges.y }},
          zaxis: {{ title: 'z (m)', range: report.ranges.z }},
          aspectmode: 'data',
          camera: {{ eye: {{ x: 1.7, y: 1.2, z: 1.1 }} }},
        }},
      }}, {{ responsive: true }});
      buildSeries();
      updatePlots(current);
    }}
    function stop() {{
      if (playing) clearInterval(playing);
      playing = null;
    }}
    function play() {{
      stop();
      playing = setInterval(() => {{
        if (current >= report.frame_count - 1) {{
          stop();
          return;
        }}
        updatePlots(current + 1);
      }}, 70);
    }}
    document.getElementById('play').addEventListener('click', play);
    document.getElementById('pause').addEventListener('click', stop);
    document.getElementById('frame').addEventListener('input', ev => {{ stop(); updatePlots(Number(ev.target.value)); }});
    document.getElementById('focus').addEventListener('change', () => updatePlots(current));
    document.getElementById('ghost').addEventListener('change', () => updatePlots(current));
    init();
  </script>
</body>
</html>
"""
    opath.write_text(html, encoding="utf-8")
    return opath
