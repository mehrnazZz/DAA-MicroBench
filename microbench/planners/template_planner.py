from __future__ import annotations

import numpy as np

from microbench.config import load_defaults
from microbench.planners.base import ILocalPlanner
from microbench.types import PlannerInput


class TemplatePlanner(ILocalPlanner):
    """Minimal planner skeleton for teammates.

    Notes:
    - Receives neighbors exactly as selected/provided by the harness.
    - Should return desired velocity command (sim enforces speed/accel clamps).
    """

    def __init__(self, cfg: dict | None = None):
        defaults = load_defaults()
        self.cfg = cfg or defaults.get("template_planner", {})
        self.goal_speed_scale = float(self.cfg.get("goal_speed_scale", 1.0))

    def reset(self, seed: int) -> None:
        _ = seed

    def compute_cmd(self, planner_input: PlannerInput) -> np.ndarray:
        ego = planner_input.ego
        goal_dir = planner_input.goal_dir
        neighbors = planner_input.neighbors

        # Use neighbors as passed; no hidden neighbor filtering here.
        _ = neighbors

        cmd = np.asarray(goal_dir, dtype=np.float32) * float(ego.v_max) * self.goal_speed_scale
        if cmd.shape != (3,):
            cmd = np.zeros(3, dtype=np.float32)
        return cmd
