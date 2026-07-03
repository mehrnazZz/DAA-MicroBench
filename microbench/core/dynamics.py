from __future__ import annotations

import numpy as np


def clamp_speed(v_cmd: np.ndarray, v_max: float) -> np.ndarray:
    spd = np.linalg.norm(v_cmd)
    if spd <= v_max or spd < 1e-9:
        return v_cmd
    return v_cmd / spd * v_max


def apply_dynamics(pos: np.ndarray, vel: np.ndarray, v_cmd: np.ndarray, v_max: float, a_max: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    v_limited = clamp_speed(v_cmd, v_max)
    dv = v_limited - vel
    max_dv = a_max * dt
    dv_norm = np.linalg.norm(dv)
    if dv_norm > max_dv and dv_norm > 1e-12:
        dv = dv / dv_norm * max_dv
    v_next = vel + dv
    p_next = pos + v_next * dt
    return p_next, v_next
