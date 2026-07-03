from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from microbench.types import (
    MSG_ABORT,
    MSG_ACK,
    MSG_EMERGENCY,
    MSG_INTENT_TRAJECTORY,
    MSG_NEGOTIATION_PROPOSAL,
    MSG_ODOMETRY,
    MSG_PRIORITY,
    MSG_STALE_BELIEF,
    MSG_YIELD,
    AgentMessage,
)


STANDARD_MESSAGE_KINDS = {
    MSG_ODOMETRY,
    MSG_INTENT_TRAJECTORY,
    MSG_YIELD,
    MSG_PRIORITY,
    MSG_NEGOTIATION_PROPOSAL,
    MSG_ACK,
    MSG_ABORT,
    MSG_EMERGENCY,
    MSG_STALE_BELIEF,
}

REQUIRED_PAYLOAD_FIELDS = {
    MSG_ODOMETRY: ("pos", "vel"),
    MSG_INTENT_TRAJECTORY: ("trajectory", "dt_plan_s", "expiry_s"),
    MSG_YIELD: ("reason",),
    MSG_PRIORITY: ("priority",),
    MSG_NEGOTIATION_PROPOSAL: ("proposal_id", "action", "start_s", "duration_s"),
    MSG_ACK: ("ack_message_id", "status"),
    MSG_ABORT: ("reason",),
    MSG_EMERGENCY: ("reason",),
    MSG_STALE_BELIEF: ("subject_id", "last_update_s"),
}


def validate_agent_message(msg: AgentMessage) -> tuple[bool, str | None]:
    kind = str(msg.kind)
    if kind not in STANDARD_MESSAGE_KINDS:
        return True, None

    payload = msg.payload or {}
    required = REQUIRED_PAYLOAD_FIELDS.get(kind, ())
    missing = [field for field in required if field not in payload]
    if missing:
        return False, "missing_payload_fields:" + ",".join(missing)

    if kind == MSG_INTENT_TRAJECTORY:
        trajectory = payload.get("trajectory")
        if not _is_vec3_sequence(trajectory):
            return False, "invalid_trajectory"
        dt_plan_s = _to_float(payload.get("dt_plan_s"))
        if dt_plan_s is None or dt_plan_s <= 0.0:
            return False, "invalid_dt_plan_s"
    elif kind == MSG_NEGOTIATION_PROPOSAL:
        if str(payload.get("action", "")) not in {"yield", "hold", "reroute", "climb", "descend"}:
            return False, "invalid_proposal_action"
        duration_s = _to_float(payload.get("duration_s"))
        start_s = _to_float(payload.get("start_s"))
        if start_s is None:
            return False, "invalid_start_s"
        if duration_s is None or duration_s <= 0.0:
            return False, "invalid_duration_s"
    elif kind == MSG_ACK:
        if str(payload.get("status", "")) not in {"accepted", "rejected", "received"}:
            return False, "invalid_ack_status"
    elif kind == MSG_ODOMETRY:
        if not _is_vec3(payload.get("pos")) or not _is_vec3(payload.get("vel")):
            return False, "invalid_odometry_vector"
    elif kind == MSG_PRIORITY:
        try:
            int(payload.get("priority"))
        except (TypeError, ValueError):
            return False, "invalid_priority"
    elif kind == MSG_STALE_BELIEF:
        try:
            int(payload.get("subject_id"))
            float(payload.get("last_update_s"))
        except (TypeError, ValueError):
            return False, "invalid_stale_belief"

    return True, None


def make_negotiation_proposal(
    *,
    sender_id: int,
    recipient_id: int,
    now_s: float,
    proposal_id: str,
    action: str,
    start_s: float,
    duration_s: float,
    priority: int = 0,
    reason: str = "",
    params: Mapping[str, Any] | None = None,
    ttl_s: float = 0.75,
) -> AgentMessage:
    payload = {
        "proposal_id": str(proposal_id),
        "action": str(action),
        "start_s": float(start_s),
        "duration_s": float(duration_s),
        "reason": str(reason),
        "params": dict(params or {}),
    }
    return AgentMessage(
        sender_id=int(sender_id),
        recipient_id=int(recipient_id),
        timestamp_send_s=float(now_s),
        kind=MSG_NEGOTIATION_PROPOSAL,
        payload=payload,
        ttl_s=float(ttl_s),
        message_id=str(proposal_id),
        correlation_id=str(proposal_id),
        channel="negotiation",
        priority=int(priority),
    )


def make_ack(
    *,
    sender_id: int,
    recipient_id: int,
    now_s: float,
    ack_message_id: str,
    status: str = "accepted",
    reason: str = "",
    ttl_s: float = 0.75,
) -> AgentMessage:
    payload = {
        "ack_message_id": str(ack_message_id),
        "status": str(status),
        "reason": str(reason),
    }
    return AgentMessage(
        sender_id=int(sender_id),
        recipient_id=int(recipient_id),
        timestamp_send_s=float(now_s),
        kind=MSG_ACK,
        payload=payload,
        ttl_s=float(ttl_s),
        correlation_id=str(ack_message_id),
        channel="negotiation",
    )


def make_stale_belief(
    *,
    sender_id: int,
    recipient_id: int | None,
    now_s: float,
    subject_id: int,
    last_update_s: float,
    reason: str,
    ttl_s: float = 0.5,
) -> AgentMessage:
    return AgentMessage(
        sender_id=int(sender_id),
        recipient_id=recipient_id,
        timestamp_send_s=float(now_s),
        kind=MSG_STALE_BELIEF,
        payload={
            "subject_id": int(subject_id),
            "last_update_s": float(last_update_s),
            "reason": str(reason),
        },
        ttl_s=float(ttl_s),
        channel="belief",
    )


def make_intent_trajectory(
    *,
    sender_id: int,
    recipient_id: int | None,
    now_s: float,
    trajectory: Sequence[Sequence[float]] | np.ndarray,
    dt_plan_s: float,
    expiry_s: float,
    tube_radius_m: float = 0.0,
    ttl_s: float = 0.75,
) -> AgentMessage:
    points = np.asarray(trajectory, dtype=float)
    return AgentMessage(
        sender_id=int(sender_id),
        recipient_id=recipient_id,
        timestamp_send_s=float(now_s),
        kind=MSG_INTENT_TRAJECTORY,
        payload={
            "trajectory": points.tolist(),
            "dt_plan_s": float(dt_plan_s),
            "expiry_s": float(expiry_s),
            "tube_radius_m": float(tube_radius_m),
        },
        ttl_s=float(ttl_s),
        channel="intent",
    )


def _is_vec3(value: object) -> bool:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    return bool(arr.shape == (3,))


def _is_vec3_sequence(value: object) -> bool:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return False
    return bool(arr.ndim == 2 and arr.shape[1] == 3 and arr.shape[0] >= 1)


def _to_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
