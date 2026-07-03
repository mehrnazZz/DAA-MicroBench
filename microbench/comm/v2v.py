from __future__ import annotations

from dataclasses import dataclass
import heapq
import json
import numpy as np

from microbench.comm.messages import validate_agent_message
from microbench.types import AgentMessage, AgentMessageObs, IntentMsg


@dataclass
class DeliveredMsg:
    sender: int
    timestamp: float
    pos: np.ndarray
    vel: np.ndarray
    radius: float


@dataclass(order=True)
class ScheduledMsg:
    deliver_time: float
    seq: int
    receiver: int
    msg: DeliveredMsg


@dataclass
class DeliveredIntentMsg:
    sender: int
    timestamp_send_s: float
    expiry_s: float
    kind: str
    tube_radius_m: float
    points: np.ndarray
    dt_plan_s: float | None = None
    mode: str | int | None = None


@dataclass(order=True)
class ScheduledIntentMsg:
    deliver_time: float
    seq: int
    receiver: int
    msg: DeliveredIntentMsg


@dataclass
class DeliveredAgentMessage:
    sender: int
    recipient: int | None
    timestamp_send_s: float
    kind: str
    payload: dict[str, object]
    ttl_s: float
    message_id: str | None
    correlation_id: str | None
    seq: int | None
    channel: str
    priority: int
    size_bytes: int


@dataclass(order=True)
class ScheduledAgentMessage:
    deliver_time: float
    seq: int
    receiver: int
    msg: DeliveredAgentMessage


class V2VEmulator:
    def __init__(
        self,
        profile: dict,
        age_cap_s: float,
        rng: np.random.Generator,
        intent_cfg: dict | None = None,
    ):
        self.profile = profile
        self.tx_rate_hz = float(profile["tx_rate_hz"])
        self.tx_period_s = 1.0 / self.tx_rate_hz
        self.intent_cfg = intent_cfg or {}
        intent_tx = float(self.intent_cfg.get("tx_rate_hz", self.tx_rate_hz))
        self.intent_tx_rate_hz = max(1e-9, intent_tx)
        self.intent_tx_period_s = 1.0 / self.intent_tx_rate_hz
        self.intent_enabled = bool(self.intent_cfg.get("enabled", False))
        self.intent_age_cap_s = float(self.intent_cfg.get("age_cap_s", age_cap_s))
        self.delay_cfg = profile.get("delay", {})
        self.loss_cfg = profile.get("loss", {})
        self.noise_cfg = profile.get("noise", {})
        self.agent_msg_cfg = profile.get("agent_messages", profile.get("message_bus", {})) or {}
        self.agent_msg_rate_limit_hz = float(self.agent_msg_cfg.get("rate_limit_hz", 0.0))
        self.agent_msg_bandwidth_limit_Bps = float(
            self.agent_msg_cfg.get("bandwidth_limit_Bps", self.agent_msg_cfg.get("bandwidth_limit_bytes_per_s", 0.0))
        )
        self.agent_msg_max_bytes = int(self.agent_msg_cfg.get("max_message_bytes", 0))
        self.agent_msg_overhead_bytes = int(self.agent_msg_cfg.get("overhead_bytes", 48))
        self.age_cap_s = float(age_cap_s)
        self.rng = rng
        self.next_tx_time: list[float] = []
        self.next_intent_tx_time: list[float] = []
        self.delivery_queue: list[ScheduledMsg] = []
        self.intent_delivery_queue: list[ScheduledIntentMsg] = []
        self.agent_message_delivery_queue: list[ScheduledAgentMessage] = []
        self.last_received: list[dict[int, DeliveredMsg]] = []
        self.last_intent_received: list[dict[int, DeliveredIntentMsg]] = []
        self.agent_messages_received: list[list[DeliveredAgentMessage]] = []
        self.pending_intent: list[IntentMsg | None] = []
        self.ge_state: dict[tuple[int, int], str] = {}
        self.agent_msg_rate_windows: list[list[float]] = []
        self.agent_msg_bandwidth_windows: list[list[tuple[float, int]]] = []
        self.agent_message_events: list[dict] = []
        self.agent_msg_stats: dict[str, int] = {}
        self.agent_msg_seq_by_sender: list[int] = []
        self.agent_msg_acked_correlations: set[str] = set()
        self._seq = 0

    def reset(self, n_agents: int) -> None:
        self.next_tx_time = [0.0 for _ in range(n_agents)]
        self.next_intent_tx_time = [0.0 for _ in range(n_agents)]
        self.delivery_queue = []
        self.intent_delivery_queue = []
        self.agent_message_delivery_queue = []
        self.last_received = [dict() for _ in range(n_agents)]
        self.last_intent_received = [dict() for _ in range(n_agents)]
        self.agent_messages_received = [[] for _ in range(n_agents)]
        self.pending_intent = [None for _ in range(n_agents)]
        self.ge_state = {}
        self.agent_msg_rate_windows = [[] for _ in range(n_agents)]
        self.agent_msg_bandwidth_windows = [[] for _ in range(n_agents)]
        self.agent_message_events = []
        self.agent_msg_stats = {
            "agent_msg_attempted": 0,
            "agent_msg_scheduled": 0,
            "agent_msg_delivered": 0,
            "agent_msg_dropped": 0,
            "agent_msg_expired": 0,
            "agent_msg_bytes_scheduled": 0,
            "agent_msg_bytes_delivered": 0,
            "agent_msg_negotiation_proposals": 0,
            "agent_msg_negotiation_acks": 0,
            "agent_msg_negotiation_correlations_acked": 0,
            "agent_msg_negotiation_rejections": 0,
        }
        self.agent_msg_seq_by_sender = [0 for _ in range(n_agents)]
        self.agent_msg_acked_correlations = set()
        self._seq = 0

    def step(self, t: float, states: list) -> None:
        n = len(states)
        for s_idx in range(n):
            while t + 1e-12 >= self.next_tx_time[s_idx]:
                send_time = self.next_tx_time[s_idx]
                self._broadcast(send_time, s_idx, states)
                self.next_tx_time[s_idx] += self.tx_period_s

        if self.intent_enabled:
            for s_idx in range(n):
                while t + 1e-12 >= self.next_intent_tx_time[s_idx]:
                    send_time = self.next_intent_tx_time[s_idx]
                    pending = self.pending_intent[s_idx]
                    if pending is not None:
                        self._broadcast_intent(send_time, pending, states)
                        self.pending_intent[s_idx] = None
                    self.next_intent_tx_time[s_idx] += self.intent_tx_period_s

        while self.delivery_queue and self.delivery_queue[0].deliver_time <= t + 1e-12:
            sched = heapq.heappop(self.delivery_queue)
            self.last_received[sched.receiver][sched.msg.sender] = sched.msg
        while self.intent_delivery_queue and self.intent_delivery_queue[0].deliver_time <= t + 1e-12:
            sched = heapq.heappop(self.intent_delivery_queue)
            self.last_intent_received[sched.receiver][sched.msg.sender] = sched.msg
        while self.agent_message_delivery_queue and self.agent_message_delivery_queue[0].deliver_time <= t + 1e-12:
            sched = heapq.heappop(self.agent_message_delivery_queue)
            self.agent_messages_received[sched.receiver].append(sched.msg)
            self.agent_msg_stats["agent_msg_delivered"] += 1
            self.agent_msg_stats["agent_msg_bytes_delivered"] += int(sched.msg.size_bytes)
            self._record_agent_event(
                {
                    "event": "delivered",
                    "t": float(t),
                    "sender_id": int(sched.msg.sender),
                    "receiver_id": int(sched.receiver),
                    "recipient_id": sched.msg.recipient,
                    "kind": str(sched.msg.kind),
                    "message_id": sched.msg.message_id,
                    "correlation_id": sched.msg.correlation_id,
                    "seq": sched.msg.seq,
                    "channel": sched.msg.channel,
                    "priority": int(sched.msg.priority),
                    "size_bytes": int(sched.msg.size_bytes),
                    "timestamp_send_s": float(sched.msg.timestamp_send_s),
                    "deliver_time_s": float(sched.deliver_time),
                }
            )

    def _broadcast(self, send_time: float, sender: int, states: list) -> None:
        s = states[sender]
        base_msg = DeliveredMsg(
            sender=sender,
            timestamp=send_time,
            pos=s.pos.copy(),
            vel=s.vel.copy(),
            radius=s.radius,
        )
        for receiver in range(len(states)):
            if receiver == sender:
                continue
            if self._drop(sender, receiver, channel="odom"):
                continue
            delay_s = self._sample_delay_sec()
            msg = self._apply_noise(base_msg)
            heapq.heappush(
                self.delivery_queue,
                ScheduledMsg(
                    deliver_time=send_time + delay_s,
                    seq=self._seq,
                    receiver=receiver,
                    msg=msg,
                ),
            )
            self._seq += 1

    def publish_intent(self, sender: int, intent: IntentMsg, now_s: float, max_points: int | None = None) -> None:
        _ = now_s
        if not self.intent_enabled:
            return
        points = np.asarray(intent.points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            return
        if max_points is not None and max_points > 0 and points.shape[0] > max_points:
            points = points[:max_points].copy()
        self.pending_intent[sender] = IntentMsg(
            sender_id=int(sender),
            timestamp_send_s=float(intent.timestamp_send_s),
            expiry_s=float(intent.expiry_s),
            kind=str(intent.kind),
            tube_radius_m=float(intent.tube_radius_m),
            points=points,
            dt_plan_s=float(intent.dt_plan_s) if intent.dt_plan_s is not None else None,
            mode=intent.mode,
        )

    def _broadcast_intent(self, send_time: float, intent: IntentMsg, states: list) -> None:
        sender = int(intent.sender_id)
        if sender < 0 or sender >= len(states):
            return
        msg = DeliveredIntentMsg(
            sender=sender,
            timestamp_send_s=float(intent.timestamp_send_s),
            expiry_s=float(intent.expiry_s),
            kind=str(intent.kind),
            tube_radius_m=float(intent.tube_radius_m),
            points=np.asarray(intent.points, dtype=float).copy(),
            dt_plan_s=float(intent.dt_plan_s) if intent.dt_plan_s is not None else None,
            mode=intent.mode,
        )
        for receiver in range(len(states)):
            if receiver == sender:
                continue
            if self._drop(sender, receiver, channel="intent"):
                continue
            delay_s = self._sample_delay_sec()
            heapq.heappush(
                self.intent_delivery_queue,
                ScheduledIntentMsg(
                    deliver_time=send_time + delay_s,
                    seq=self._seq,
                    receiver=receiver,
                    msg=msg,
                ),
            )
            self._seq += 1

    def publish_agent_message(self, sender: int, msg: AgentMessage, now_s: float, n_agents: int) -> None:
        sender = int(sender)
        if sender < 0 or sender >= n_agents:
            return
        recipient = msg.recipient_id
        if recipient is not None:
            recipient = int(recipient)
            if recipient < 0 or recipient >= n_agents or recipient == sender:
                return
            receivers = [recipient]
        else:
            receivers = [i for i in range(n_agents) if i != sender]

        timestamp_send_s = float(msg.timestamp_send_s)
        if timestamp_send_s < 0.0:
            timestamp_send_s = float(now_s)
        seq = int(msg.seq) if msg.seq is not None else int(self.agent_msg_seq_by_sender[sender])
        if msg.seq is None:
            self.agent_msg_seq_by_sender[sender] += 1
        message_id = msg.message_id or f"{sender}:{seq}:{timestamp_send_s:.6f}:{self._seq}"
        channel = str(msg.channel or "agent_msg")
        payload = dict(msg.payload or {})
        size_bytes = int(msg.size_bytes) if msg.size_bytes is not None else self._estimate_agent_message_bytes(
            sender=sender,
            recipient=recipient,
            kind=str(msg.kind),
            payload=payload,
            message_id=message_id,
            correlation_id=msg.correlation_id,
            seq=seq,
            channel=channel,
            priority=int(msg.priority),
            ttl_s=max(0.0, float(msg.ttl_s)),
        )
        delivered = DeliveredAgentMessage(
            sender=sender,
            recipient=recipient,
            timestamp_send_s=timestamp_send_s,
            kind=str(msg.kind),
            payload=payload,
            ttl_s=max(0.0, float(msg.ttl_s)),
            message_id=message_id,
            correlation_id=msg.correlation_id,
            seq=seq,
            channel=channel,
            priority=int(msg.priority),
            size_bytes=size_bytes,
        )
        valid_payload, invalid_reason = validate_agent_message(msg)
        for receiver in receivers:
            self.agent_msg_stats["agent_msg_attempted"] += 1
            if not valid_payload:
                self._drop_agent_message(now_s, sender, receiver, delivered, reason=invalid_reason or "invalid_payload")
                continue
            if self.agent_msg_max_bytes > 0 and size_bytes > self.agent_msg_max_bytes:
                self._drop_agent_message(now_s, sender, receiver, delivered, reason="max_message_bytes")
                continue
            if not self._reserve_agent_message_budget(sender, now_s, size_bytes):
                self._drop_agent_message(now_s, sender, receiver, delivered, reason="rate_or_bandwidth_limit")
                continue
            if self._drop(sender, receiver, channel="agent_msg"):
                self._drop_agent_message(now_s, sender, receiver, delivered, reason="loss_model")
                continue
            delay_s = self._sample_delay_sec()
            heapq.heappush(
                self.agent_message_delivery_queue,
                ScheduledAgentMessage(
                    deliver_time=float(now_s) + delay_s,
                    seq=self._seq,
                    receiver=receiver,
                    msg=delivered,
                ),
            )
            self.agent_msg_stats["agent_msg_scheduled"] += 1
            self.agent_msg_stats["agent_msg_bytes_scheduled"] += int(size_bytes)
            if str(delivered.kind) == "NEGOTIATION_PROPOSAL":
                self.agent_msg_stats["agent_msg_negotiation_proposals"] += 1
            self._record_agent_event(
                {
                    "event": "scheduled",
                    "t": float(now_s),
                    "sender_id": int(sender),
                    "receiver_id": int(receiver),
                    "recipient_id": recipient,
                    "kind": str(delivered.kind),
                    "message_id": delivered.message_id,
                    "correlation_id": delivered.correlation_id,
                    "seq": delivered.seq,
                    "channel": delivered.channel,
                    "priority": int(delivered.priority),
                    "size_bytes": int(delivered.size_bytes),
                    "timestamp_send_s": float(delivered.timestamp_send_s),
                    "deliver_time_s": float(now_s) + float(delay_s),
                }
            )
            self._seq += 1

    def _record_agent_event(self, event: dict) -> None:
        self.agent_message_events.append(event)

    def _drop_agent_message(
        self,
        now_s: float,
        sender: int,
        receiver: int,
        msg: DeliveredAgentMessage,
        *,
        reason: str,
    ) -> None:
        self.agent_msg_stats["agent_msg_dropped"] += 1
        self._record_agent_event(
            {
                "event": "dropped",
                "reason": str(reason),
                "t": float(now_s),
                "sender_id": int(sender),
                "receiver_id": int(receiver),
                "recipient_id": msg.recipient,
                "kind": str(msg.kind),
                "message_id": msg.message_id,
                "correlation_id": msg.correlation_id,
                "seq": msg.seq,
                "channel": msg.channel,
                "priority": int(msg.priority),
                "size_bytes": int(msg.size_bytes),
                "timestamp_send_s": float(msg.timestamp_send_s),
            }
        )

    def _estimate_agent_message_bytes(
        self,
        *,
        sender: int,
        recipient: int | None,
        kind: str,
        payload: dict[str, object],
        message_id: str | None,
        correlation_id: str | None,
        seq: int | None,
        channel: str,
        priority: int,
        ttl_s: float,
    ) -> int:
        body = {
            "sender_id": int(sender),
            "recipient_id": recipient,
            "kind": str(kind),
            "payload": payload,
            "message_id": message_id,
            "correlation_id": correlation_id,
            "seq": seq,
            "channel": channel,
            "priority": int(priority),
            "ttl_s": float(ttl_s),
        }
        raw = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        return int(len(raw) + max(0, self.agent_msg_overhead_bytes))

    def _reserve_agent_message_budget(self, sender: int, now_s: float, size_bytes: int) -> bool:
        cutoff = float(now_s) - 1.0
        rate_window = self.agent_msg_rate_windows[sender]
        while rate_window and rate_window[0] <= cutoff:
            rate_window.pop(0)
        if self.agent_msg_rate_limit_hz > 0.0:
            rate_limit = max(1, int(np.floor(self.agent_msg_rate_limit_hz)))
            if len(rate_window) >= rate_limit:
                return False

        bandwidth_window = self.agent_msg_bandwidth_windows[sender]
        while bandwidth_window and bandwidth_window[0][0] <= cutoff:
            bandwidth_window.pop(0)
        used_bytes = sum(int(x[1]) for x in bandwidth_window)
        if self.agent_msg_bandwidth_limit_Bps > 0.0 and used_bytes + int(size_bytes) > self.agent_msg_bandwidth_limit_Bps:
            return False

        rate_window.append(float(now_s))
        bandwidth_window.append((float(now_s), int(size_bytes)))
        return True

    def _sample_delay_sec(self) -> float:
        dtype = self.delay_cfg.get("type", "constant_ms")
        if dtype == "constant_ms":
            return max(0.0, float(self.delay_cfg.get("value_ms", 0.0)) / 1000.0)
        if dtype == "normal_trunc_ms":
            mean = float(self.delay_cfg.get("mean_ms", 0.0))
            std = max(1e-9, float(self.delay_cfg.get("std_ms", 1.0)))
            lo = float(self.delay_cfg.get("min_ms", 0.0))
            hi = float(self.delay_cfg.get("max_ms", max(lo, mean + 5 * std)))
            for _ in range(16):
                x = self.rng.normal(mean, std)
                if lo <= x <= hi:
                    return x / 1000.0
            return min(max(mean, lo), hi) / 1000.0
        raise ValueError(f"Unsupported delay type: {dtype}")

    def _drop(self, sender: int, receiver: int, channel: str) -> bool:
        ltype = self.loss_cfg.get("type", "iid")
        if ltype == "iid":
            p = float(self.loss_cfg.get("p", 0.0))
            return bool(self.rng.random() < p)
        if ltype == "gilbert_elliott":
            key = (sender, receiver, channel)
            state = self.ge_state.get(key, "good")
            p_gb = float(self.loss_cfg.get("p_good_to_bad", 0.0))
            p_bg = float(self.loss_cfg.get("p_bad_to_good", 1.0))
            p_loss_good = float(self.loss_cfg.get("p_loss_good", 0.0))
            p_loss_bad = float(self.loss_cfg.get("p_loss_bad", 1.0))
            if state == "good" and self.rng.random() < p_gb:
                state = "bad"
            elif state == "bad" and self.rng.random() < p_bg:
                state = "good"
            self.ge_state[key] = state
            p_loss = p_loss_bad if state == "bad" else p_loss_good
            return bool(self.rng.random() < p_loss)
        raise ValueError(f"Unsupported loss type: {ltype}")

    def _apply_noise(self, msg: DeliveredMsg) -> DeliveredMsg:
        sigma_pos = float(self.noise_cfg.get("sigma_pos_m", 0.0))
        sigma_vel = float(self.noise_cfg.get("sigma_vel_mps", 0.0))
        if sigma_pos <= 0.0 and sigma_vel <= 0.0:
            return msg
        pos = msg.pos.copy()
        vel = msg.vel.copy()
        if sigma_pos > 0.0:
            pos += self.rng.normal(0.0, sigma_pos, size=3)
        if sigma_vel > 0.0:
            vel += self.rng.normal(0.0, sigma_vel, size=3)
        return DeliveredMsg(
            sender=msg.sender,
            timestamp=msg.timestamp,
            pos=pos,
            vel=vel,
            radius=msg.radius,
        )

    def get_last(self, receiver: int, sender: int) -> DeliveredMsg | None:
        return self.last_received[receiver].get(sender)

    def message_age(self, now: float, msg: DeliveredMsg | None) -> tuple[bool, float]:
        if msg is None:
            return False, self.age_cap_s
        age = max(0.0, now - msg.timestamp)
        return True, min(age, self.age_cap_s)

    def get_last_intent(self, receiver: int, sender: int) -> DeliveredIntentMsg | None:
        if not self.intent_enabled:
            return None
        return self.last_intent_received[receiver].get(sender)

    def intent_status(self, now: float, msg: DeliveredIntentMsg | None) -> tuple[bool, float]:
        if msg is None:
            return False, self.intent_age_cap_s
        age = max(0.0, now - float(msg.timestamp_send_s))
        valid = now <= float(msg.expiry_s)
        return valid, min(age, self.intent_age_cap_s)

    def drain_agent_messages(self, receiver: int, now: float) -> list[AgentMessageObs]:
        if receiver < 0 or receiver >= len(self.agent_messages_received):
            return []
        pending = self.agent_messages_received[receiver]
        self.agent_messages_received[receiver] = []
        out: list[AgentMessageObs] = []
        for msg in pending:
            age = max(0.0, float(now) - float(msg.timestamp_send_s))
            valid = age <= float(msg.ttl_s)
            out.append(
                AgentMessageObs(
                    sender_id=int(msg.sender),
                    recipient_id=msg.recipient,
                    timestamp_send_s=float(msg.timestamp_send_s),
                    kind=str(msg.kind),
                    payload=dict(msg.payload),
                    msg_age_s=min(age, self.age_cap_s),
                    valid=bool(valid),
                    ttl_s=float(msg.ttl_s),
                    message_id=msg.message_id,
                    correlation_id=msg.correlation_id,
                    seq=msg.seq,
                    channel=msg.channel,
                    priority=int(msg.priority),
                    size_bytes=int(msg.size_bytes),
                )
            )
            if not valid:
                self.agent_msg_stats["agent_msg_expired"] += 1
                self._record_agent_event(
                    {
                        "event": "expired",
                        "t": float(now),
                        "sender_id": int(msg.sender),
                        "receiver_id": int(receiver),
                        "recipient_id": msg.recipient,
                        "kind": str(msg.kind),
                        "message_id": msg.message_id,
                        "correlation_id": msg.correlation_id,
                        "seq": msg.seq,
                        "channel": msg.channel,
                        "priority": int(msg.priority),
                        "size_bytes": int(msg.size_bytes),
                        "timestamp_send_s": float(msg.timestamp_send_s),
                        "msg_age_s": float(age),
                        "ttl_s": float(msg.ttl_s),
                    }
                )
            elif str(msg.kind) == "ACK":
                self.agent_msg_stats["agent_msg_negotiation_acks"] += 1
                status = str(msg.payload.get("status", ""))
                correlation = str(msg.correlation_id or msg.payload.get("ack_message_id", ""))
                if status == "accepted" and correlation:
                    if correlation not in self.agent_msg_acked_correlations:
                        self.agent_msg_acked_correlations.add(correlation)
                        self.agent_msg_stats["agent_msg_negotiation_correlations_acked"] += 1
                elif status == "rejected":
                    self.agent_msg_stats["agent_msg_negotiation_rejections"] += 1
        return out

    def drain_agent_message_events(self) -> list[dict]:
        events = list(self.agent_message_events)
        self.agent_message_events = []
        return events

    def agent_message_stats_snapshot(self) -> dict[str, int]:
        return dict(self.agent_msg_stats)
