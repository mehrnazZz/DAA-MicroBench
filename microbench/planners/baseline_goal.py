from __future__ import annotations

import numpy as np

from microbench.planners.base import ILocalPlanner
from microbench.types import PlannerInput


class BaselineGoalPlanner(ILocalPlanner):
    def reset(self, seed: int) -> None:
        _ = seed

    def compute_cmd(self, planner_input: PlannerInput) -> np.ndarray:
        return planner_input.goal_dir * planner_input.ego.v_max
