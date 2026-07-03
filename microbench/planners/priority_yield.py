from __future__ import annotations

import numpy as np

from microbench.planners.base import ILocalPlanner
from microbench.types import AgentMessage, PlannerInput, PlannerOutput


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.zeros(3, dtype=np.float32)
    return (v / n).astype(np.float32)


class PriorityYieldPlanner(ILocalPlanner):
    """Tiny agentic baseline with memory and advisory yield messages.

    Lower agent id means higher priority. A lower-priority agent slows when it
    receives a YIELD message or detects an approaching higher-priority neighbor.
    """

    def reset(self, seed: int) -> None:
        self.seed = int(seed)

    def compute_cmd(self, planner_input: PlannerInput) -> PlannerOutput:
        ego = planner_input.ego
        ctx = planner_input.agent_context
        memory = ctx.memory if ctx is not None else {}
        now = float(planner_input.t)

        yield_until = float(memory.get("yield_until_s", -1.0))
        for msg in planner_input.messages:
            if not msg.valid or msg.kind != "YIELD":
                continue
            if msg.recipient_id is None or int(msg.recipient_id) == int(ego.idx):
                yield_until = max(yield_until, now + 0.5)
        messages_out: list[AgentMessage] = []

        for nbr in planner_input.neighbors:
            rel_pos = np.asarray(nbr.pos, dtype=np.float32) - np.asarray(ego.pos, dtype=np.float32)
            rel_vel = np.asarray(nbr.vel, dtype=np.float32) - np.asarray(ego.vel, dtype=np.float32)
            dist = float(np.linalg.norm(rel_pos))
            if dist > 8.0 or dist < 1e-6:
                continue
            closing = float(np.dot(rel_pos, rel_vel)) < 0.0
            if not closing:
                continue
            if int(nbr.idx) < int(ego.idx):
                yield_until = max(yield_until, now + 0.4)
            elif int(nbr.idx) > int(ego.idx):
                messages_out.append(
                    AgentMessage(
                        sender_id=int(ego.idx),
                        recipient_id=int(nbr.idx),
                        timestamp_send_s=now,
                        kind="YIELD",
                        payload={"reason": "priority_conflict", "priority": int(ego.idx)},
                        ttl_s=0.75,
                    )
                )

        memory["yield_until_s"] = yield_until

        speed_scale = 0.25 if now <= yield_until else 1.0
        v_cmd = _normalize(np.asarray(planner_input.goal_dir, dtype=np.float32)) * float(ego.v_max) * speed_scale
        return PlannerOutput(v_cmd=v_cmd.astype(float), messages_out=messages_out)
