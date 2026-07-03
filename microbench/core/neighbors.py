from __future__ import annotations

import math
import numpy as np

from microbench.types import NeighborObs


def _ttc_threat(ego_pos: np.ndarray, ego_vel: np.ndarray, nbr_pos: np.ndarray, nbr_vel: np.ndarray, horizon_s: float) -> float:
    rel_p = nbr_pos - ego_pos
    rel_v = nbr_vel - ego_vel
    v2 = float(np.dot(rel_v, rel_v))
    if v2 < 1e-9:
        return math.inf
    t = -float(np.dot(rel_p, rel_v)) / v2
    if t <= 0.0 or t > horizon_s:
        return math.inf
    return t


def select_neighbors(
    ego_idx: int,
    ego_pos: np.ndarray,
    ego_vel: np.ndarray,
    obs: list[NeighborObs],
    range_m: float,
    top_k: int,
    threat_metric: str,
    ttc_horizon_s: float,
) -> list[NeighborObs]:
    kept: list[tuple[float, int, NeighborObs]] = []
    for n in obs:
        if not n.valid:
            continue
        d = float(np.linalg.norm(n.pos - ego_pos))
        if d > range_m:
            continue
        if threat_metric == "ttc":
            score = _ttc_threat(ego_pos, ego_vel, n.pos, n.vel, ttc_horizon_s)
            if math.isinf(score):
                score = 1e9 + d
        else:
            score = d
        kept.append((score, int(n.idx), n))
    # Deterministic ordering: threat score, then neighbor id.
    kept.sort(key=lambda x: (x[0], x[1]))
    return [x[2] for x in kept[:top_k]]
