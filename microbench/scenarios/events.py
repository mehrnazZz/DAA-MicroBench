from __future__ import annotations

import math
import numpy as np


def _rotate_xz(v: np.ndarray, deg: float) -> np.ndarray:
    rad = math.radians(deg)
    c = math.cos(rad)
    s = math.sin(rad)
    x, y, z = v
    return np.array([c * x - s * z, y, s * x + c * z], dtype=float)


class EventEngine:
    def __init__(self, events_cfg: list[dict], rng: np.random.Generator):
        self.events_cfg = events_cfg
        self.rng = rng
        self.resolved: dict[int, list[int]] = {}

    def reset(self) -> None:
        self.resolved = {}

    def apply_overrides(self, t: float, states: list, v_cmds: list[np.ndarray]) -> list[np.ndarray]:
        out = [v.copy() for v in v_cmds]
        for ev_idx, ev in enumerate(self.events_cfg):
            if not ev.get("enabled", True):
                continue
            if ev.get("type") != "weather_maneuver":
                continue
            t0 = float(ev.get("t_start_s", 0.0))
            dur = float(ev.get("duration_s", 0.0))
            if not (t0 <= t <= t0 + dur):
                continue
            if ev_idx not in self.resolved:
                self.resolved[ev_idx] = self._select_agents(ev, states)
            forced = ev.get("forced_policy", {})
            for j, aid in enumerate(self.resolved[ev_idx]):
                sign = -1.0 if (j % 2 == 0) else 1.0
                cmd = out[aid]
                spd = np.linalg.norm(cmd)
                if spd < 1e-6:
                    cmd = states[aid].vel.copy()
                    spd = np.linalg.norm(cmd)
                ptype = forced.get("type")
                if ptype == "hard_turn_and_slow":
                    turn_deg = float(forced.get("turn_deg", 70.0))
                    speed_scale = float(forced.get("speed_scale", 0.35))
                    if spd > 1e-6:
                        turned = _rotate_xz(cmd, sign * turn_deg)
                        tnorm = np.linalg.norm(turned)
                        if tnorm > 1e-6:
                            turned = turned / tnorm * spd * speed_scale
                        out[aid] = turned
                elif ptype == "vertical_shift_and_slow":
                    speed_scale = float(forced.get("speed_scale", 0.5))
                    vertical_speed = float(forced.get("vertical_speed_mps", 1.5))
                    direction = str(forced.get("direction", "alternate")).lower()
                    horizontal = cmd.copy()
                    horizontal[1] = 0.0
                    hnorm = np.linalg.norm(horizontal)
                    if hnorm > 1e-6:
                        horizontal = horizontal / hnorm * max(0.0, spd * speed_scale)
                    else:
                        horizontal = np.zeros(3, dtype=float)
                    vertical_sign = sign
                    if direction == "up":
                        vertical_sign = 1.0
                    elif direction == "down":
                        vertical_sign = -1.0
                    elif direction == "random":
                        vertical_sign = 1.0 if self.rng.random() < 0.5 else -1.0
                    horizontal[1] = vertical_sign * vertical_speed
                    out[aid] = horizontal
        return out

    def _select_agents(self, ev: dict, states: list) -> list[int]:
        n_agents = int(ev.get("n_agents", 1))
        selection = ev.get("selection", "closest_to_gate")
        idxs = list(range(len(states)))
        if selection == "closest_to_gate":
            idxs.sort(key=lambda i: abs(states[i].pos[0]))
            return idxs[:n_agents]
        self.rng.shuffle(idxs)
        return idxs[:n_agents]
