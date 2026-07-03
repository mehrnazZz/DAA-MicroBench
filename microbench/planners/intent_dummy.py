from __future__ import annotations

import numpy as np

from microbench.planners.base import ILocalPlanner
from microbench.types import IntentMsg, PlannerInput, PlannerOutput


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.zeros(3, dtype=np.float32)
    return (v / n).astype(np.float32)


class IntentDummyPlanner(ILocalPlanner):
    """Simple planner used for intent-channel plumbing tests."""

    def reset(self, seed: int) -> None:
        _ = seed

    def compute_cmd(self, planner_input: PlannerInput) -> np.ndarray | PlannerOutput:
        ego = planner_input.ego
        t = float(planner_input.t)
        goal_dir = _normalize(np.asarray(planner_input.goal_dir, dtype=np.float32))
        v_cmd = goal_dir * float(ego.v_max)

        step = 2.0
        points = np.stack(
            [
                np.asarray(ego.pos, dtype=np.float32),
                np.asarray(ego.pos, dtype=np.float32) + goal_dir * step,
                np.asarray(ego.pos, dtype=np.float32) + goal_dir * (2.0 * step),
            ],
            axis=0,
        )

        intent = IntentMsg(
            sender_id=int(ego.idx),
            timestamp_send_s=t,
            expiry_s=t + 1.0,
            kind="PROPOSED",
            tube_radius_m=float(ego.radius) + 0.2,
            points=points.astype(float),
            dt_plan_s=0.1,
            mode="normal",
        )
        return PlannerOutput(v_cmd=v_cmd.astype(float), intent_out=intent)
