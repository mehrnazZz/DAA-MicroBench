from __future__ import annotations

from dataclasses import dataclass
import heapq
import numpy as np

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
        delivered = DeliveredAgentMessage(
            sender=sender,
            recipient=recipient,
            timestamp_send_s=timestamp_send_s,
            kind=str(msg.kind),
            payload=dict(msg.payload or {}),
            ttl_s=max(0.0, float(msg.ttl_s)),
        )
        for receiver in receivers:
            if self._drop(sender, receiver, channel="agent_msg"):
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
            self._seq += 1

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
                )
            )
        return out
