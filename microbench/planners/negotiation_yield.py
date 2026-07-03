from __future__ import annotations

import numpy as np

from microbench.comm.messages import make_ack, make_negotiation_proposal
from microbench.planners.base import ILocalPlanner
from microbench.types import (
    MSG_ACK,
    MSG_NEGOTIATION_PROPOSAL,
    PlannerInput,
    PlannerOutput,
)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.zeros(3, dtype=np.float32)
    return (v / n).astype(np.float32)


def _sidestep(goal_dir: np.ndarray, agent_idx: int) -> np.ndarray:
    side = np.asarray([-goal_dir[2], 0.0, goal_dir[0]], dtype=np.float32)
    n = float(np.linalg.norm(side))
    if n < 1e-9:
        side = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        side = side / n
    sign = -1.0 if int(agent_idx) % 2 else 1.0
    return side * sign


class NegotiationYieldPlanner(ILocalPlanner):
    """Tiny proposal/ACK baseline for decentralized right-of-way negotiation.

    Lower numeric priority means higher right-of-way. A high-priority agent
    proposes a short yield commitment to lower-priority neighbors; recipients
    ACK and slow while the commitment is active.
    """

    def reset(self, seed: int) -> None:
        self.seed = int(seed)

    def compute_cmd(self, planner_input: PlannerInput) -> PlannerOutput:
        ego = planner_input.ego
        ctx = planner_input.agent_context
        memory = ctx.memory if ctx is not None else {}
        now = float(planner_input.t)
        ego_priority = int(ctx.priority) if ctx is not None else int(ego.idx)

        yield_until = float(memory.get("yield_until_s", -1.0))
        proposal_seq = int(memory.get("proposal_seq", 0))
        last_proposal_by_neighbor = dict(memory.get("last_proposal_by_neighbor", {}))
        acked_correlations = set(memory.get("acked_correlations", set()))

        messages_out = []
        acks_sent = 0
        proposals_sent = 0
        acks_received = 0

        for msg in planner_input.messages:
            if not msg.valid:
                continue
            if msg.kind == MSG_NEGOTIATION_PROPOSAL:
                duration_s = float(msg.payload.get("duration_s", 0.5))
                start_s = float(msg.payload.get("start_s", now))
                proposal_id = str(msg.payload.get("proposal_id", msg.message_id or ""))
                action = str(msg.payload.get("action", ""))
                if action in {"yield", "hold"} and proposal_id not in acked_correlations:
                    yield_until = max(yield_until, start_s + duration_s)
                    messages_out.append(
                        make_ack(
                            sender_id=int(ego.idx),
                            recipient_id=int(msg.sender_id),
                            now_s=now,
                            ack_message_id=str(msg.message_id or proposal_id),
                            status="accepted",
                            reason="yield_commitment",
                            ttl_s=0.75,
                        )
                    )
                    acked_correlations.add(str(msg.message_id or proposal_id))
                    acks_sent += 1
            elif msg.kind == MSG_ACK and str(msg.payload.get("status", "")) == "accepted":
                acks_received += 1

        for nbr in planner_input.neighbors:
            rel_pos = np.asarray(nbr.pos, dtype=np.float32) - np.asarray(ego.pos, dtype=np.float32)
            rel_vel = np.asarray(nbr.vel, dtype=np.float32) - np.asarray(ego.vel, dtype=np.float32)
            dist = float(np.linalg.norm(rel_pos))
            if dist > 8.0 or dist < 1e-6:
                continue
            closing = float(np.dot(rel_pos, rel_vel)) < 0.0
            if not closing:
                continue

            neighbor_priority = int(nbr.idx)
            if ego_priority <= neighbor_priority:
                last_t = float(last_proposal_by_neighbor.get(int(nbr.idx), -1e9))
                if now - last_t >= 0.5:
                    proposal_id = f"yield-{int(ego.idx)}-{int(nbr.idx)}-{proposal_seq}"
                    proposal_seq += 1
                    messages_out.append(
                        make_negotiation_proposal(
                            sender_id=int(ego.idx),
                            recipient_id=int(nbr.idx),
                            now_s=now,
                            proposal_id=proposal_id,
                            action="yield",
                            start_s=now,
                            duration_s=0.6,
                            priority=ego_priority,
                            reason="right_of_way_conflict",
                            params={
                                "requester_priority": ego_priority,
                                "speed_scale": 0.25,
                                "distance_m": dist,
                            },
                            ttl_s=0.75,
                        )
                    )
                    last_proposal_by_neighbor[int(nbr.idx)] = now
                    proposals_sent += 1
            else:
                yield_until = max(yield_until, now + 0.4)

        memory["yield_until_s"] = yield_until
        memory["proposal_seq"] = proposal_seq
        memory["last_proposal_by_neighbor"] = last_proposal_by_neighbor
        memory["acked_correlations"] = acked_correlations

        goal_dir = _normalize(np.asarray(planner_input.goal_dir, dtype=np.float32))
        yielding = bool(now <= yield_until)
        speed_scale = 0.25 if yielding else 1.0
        if yielding:
            sidestep = _sidestep(goal_dir, int(ego.idx))
            v_dir = _normalize(goal_dir * speed_scale + sidestep * 0.75)
            v_cmd = v_dir * float(ego.v_max)
        else:
            v_cmd = goal_dir * float(ego.v_max)
        return PlannerOutput(
            v_cmd=v_cmd.astype(float),
            messages_out=messages_out,
            debug_info={
                "yield_until_s": float(yield_until),
                "speed_scale": float(speed_scale),
                "sidestep_active": bool(yielding),
                "proposals_sent": int(proposals_sent),
                "acks_sent": int(acks_sent),
                "acks_received": int(acks_received),
            },
        )
