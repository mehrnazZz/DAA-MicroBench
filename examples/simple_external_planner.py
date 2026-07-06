from __future__ import annotations

import numpy as np

from microbench.planners.base import ILocalPlanner
from microbench.types import PlannerInput, PlannerOutput


class SimpleExternalPlanner(ILocalPlanner):
    """Small public-contract planner example.

    The planner uses only PlannerInput fields:
    - goal direction
    - selected neighbor tracks
    - static AABB obstacles
    - agent_context memory

    It can be copied into microbench/planners/ and registered in the planner
    registry, or used directly in Python tests through EpisodeEngine's
    planner_factory callback.
    """

    def reset(self, agent_id: int | None = None, seed: int = 0, config: dict | None = None) -> None:
        self.agent_id = int(agent_id or 0)
        self.rng = np.random.default_rng(int(seed))
        self.config = dict(config or {})

    def compute_cmd(self, planner_input: PlannerInput) -> PlannerOutput:
        ego = planner_input.ego
        goal = np.asarray(planner_input.goal_dir, dtype=float)
        avoid = np.zeros(3, dtype=float)
        ego_pos = np.asarray(ego.pos, dtype=float)

        for neighbor in planner_input.neighbors:
            rel = ego_pos - np.asarray(neighbor.pos, dtype=float)
            dist = max(1e-6, float(np.linalg.norm(rel)))
            safety_radius = float(ego.radius + neighbor.radius + 1.0)
            if dist < safety_radius:
                avoid += rel / (dist * dist)

        for obstacle in planner_input.obstacles:
            center = np.asarray(obstacle.center, dtype=float)
            half = np.asarray(obstacle.half, dtype=float)
            closest = np.minimum(np.maximum(ego_pos, center - half), center + half)
            rel = ego_pos - closest
            dist = max(1e-6, float(np.linalg.norm(rel)))
            if dist < 1.5:
                avoid += rel / (dist * dist)

        if planner_input.planar:
            avoid[1] = 0.0

        direction = goal + 0.75 * avoid
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            v_cmd = np.zeros(3, dtype=float)
        else:
            v_cmd = direction / norm * float(ego.v_max)

        ctx = planner_input.agent_context
        ticks = 0
        if ctx is not None:
            ticks = int(ctx.memory.get("ticks", 0)) + 1
            ctx.memory["ticks"] = ticks

        return PlannerOutput(
            v_cmd=v_cmd,
            intent_out=None,
            messages_out=[],
            debug_info={
                "agent_id": self.agent_id,
                "ticks": ticks,
                "neighbors": len(planner_input.neighbors),
            },
        )
