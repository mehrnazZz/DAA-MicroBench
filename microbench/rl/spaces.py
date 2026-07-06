from __future__ import annotations

from typing import Any

import numpy as np


try:  # pragma: no cover - exercised when gymnasium is installed.
    from gymnasium import spaces as gym_spaces
except Exception:  # pragma: no cover - fallback is covered in the default test env.
    gym_spaces = None


class FallbackBox:
    """Small `gymnasium.spaces.Box` fallback for core installs.

    It intentionally implements only the methods needed by examples and tests:
    `sample`, `contains`, and basic attributes. Installing the `rl` extra uses
    Gymnasium's real Box class instead.
    """

    def __init__(self, *, low: float, high: float, shape: tuple[int, ...], dtype=np.float32):
        self.low = np.full(shape, low, dtype=dtype)
        self.high = np.full(shape, high, dtype=dtype)
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)

    def sample(self) -> np.ndarray:
        low = np.where(np.isfinite(self.low), self.low, -1.0)
        high = np.where(np.isfinite(self.high), self.high, 1.0)
        return np.random.default_rng().uniform(low, high).astype(self.dtype)

    def contains(self, x: Any) -> bool:
        arr = np.asarray(x, dtype=self.dtype)
        if arr.shape != self.shape:
            return False
        if not bool(np.all(np.isfinite(arr))):
            return False
        lower_ok = np.ones(self.shape, dtype=bool)
        upper_ok = np.ones(self.shape, dtype=bool)
        finite_low = np.isfinite(self.low)
        finite_high = np.isfinite(self.high)
        lower_ok[finite_low] = arr[finite_low] >= self.low[finite_low]
        upper_ok[finite_high] = arr[finite_high] <= self.high[finite_high]
        return bool(np.all(lower_ok & upper_ok))

    def __repr__(self) -> str:
        return f"FallbackBox(shape={self.shape}, dtype={self.dtype})"


def box(*, low: float, high: float, shape: tuple[int, ...], dtype=np.float32):
    if gym_spaces is not None:
        return gym_spaces.Box(low=low, high=high, shape=shape, dtype=dtype)
    return FallbackBox(low=low, high=high, shape=shape, dtype=dtype)
