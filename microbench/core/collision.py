from __future__ import annotations

import numpy as np


def pairwise_stats(positions: np.ndarray, radii: np.ndarray, near_margin: float) -> tuple[int, int, float]:
    n = positions.shape[0]
    collisions = 0
    near_misses = 0
    min_sep = float("inf")
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(positions[i] - positions[j]))
            sep = d - (radii[i] + radii[j])
            if sep < min_sep:
                min_sep = sep
            if d < (radii[i] + radii[j]):
                collisions += 1
            elif d < (radii[i] + radii[j] + near_margin):
                near_misses += 1
    if n < 2:
        min_sep = 0.0
    return collisions, near_misses, min_sep
