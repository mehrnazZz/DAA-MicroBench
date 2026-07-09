from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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
