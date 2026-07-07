from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


Vec3 = np.ndarray

MSG_ODOMETRY = "ODOMETRY"
MSG_INTENT_TRAJECTORY = "INTENT_TRAJECTORY"
MSG_YIELD = "YIELD"
MSG_PRIORITY = "PRIORITY"
MSG_NEGOTIATION_PROPOSAL = "NEGOTIATION_PROPOSAL"
MSG_ACK = "ACK"
MSG_ABORT = "ABORT"
MSG_EMERGENCY = "EMERGENCY"
MSG_STALE_BELIEF = "STALE_BELIEF"


@dataclass
class AgentState:
    idx: int
    pos: Vec3
    vel: Vec3
    goal: Vec3
    radius: float
    v_max: float
    a_max: float
    done: bool = False
    done_time_s: float | None = None
    path_length_m: float = 0.0


@dataclass
class NeighborObs:
    idx: int
    pos: Vec3
    vel: Vec3
    radius: float
    msg_age_sec: float
    valid: bool
    source: str = "v2v"
    track_age_sec: float = 0.0
    last_seen_s: float | None = None
    stale: bool = False
    occluded: bool = False


@dataclass
class AABBObs:
    center: Vec3
    half: Vec3


class AgentMemory(dict[str, object]):
    """Persistent per-agent scratchpad owned by one planner instance."""


@dataclass
class AgentProfile:
    agent_id: int
    method: str | None = None
    role: str | None = None
    priority: int = 0
    capabilities: dict[str, object] = field(default_factory=dict)
    mission: dict[str, object] = field(default_factory=dict)
    failure_modes: dict[str, object] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


@dataclass
class AgentContext:
    agent_id: int
    method: str
    seed: int
    memory: AgentMemory = field(default_factory=AgentMemory)
    role: str | None = None
    priority: int = 0
    capabilities: dict[str, object] = field(default_factory=dict)
    mission: dict[str, object] = field(default_factory=dict)
    failure_modes: dict[str, object] = field(default_factory=dict)
    profile: AgentProfile | None = None


@dataclass
class IntentMsg:
    sender_id: int
    timestamp_send_s: float
    expiry_s: float
    kind: str
    tube_radius_m: float
    points: Vec3
    dt_plan_s: float | None = None
    mode: str | int | None = None


@dataclass
class IntentObs:
    sender_id: int
    points: Vec3
    tube_radius_m: float
    kind: str
    expiry_s: float
    intent_age_s: float
    valid: bool
    dt_plan_s: float | None = None
    mode: str | int | None = None


@dataclass
class AgentMessage:
    sender_id: int
    timestamp_send_s: float
    kind: str
    payload: dict[str, object] = field(default_factory=dict)
    recipient_id: int | None = None
    ttl_s: float = 1.0
    message_id: str | None = None
    correlation_id: str | None = None
    seq: int | None = None
    channel: str = "agent_msg"
    priority: int = 0
    size_bytes: int | None = None


@dataclass
class AgentMessageObs:
    sender_id: int
    recipient_id: int | None
    timestamp_send_s: float
    kind: str
    payload: dict[str, object]
    msg_age_s: float
    valid: bool
    ttl_s: float
    message_id: str | None = None
    correlation_id: str | None = None
    seq: int | None = None
    channel: str = "agent_msg"
    priority: int = 0
    size_bytes: int = 0


@dataclass
class PlannerInput:
    ego: AgentState
    goal_dir: Vec3
    neighbors: list[NeighborObs]
    dt: float
    t: float
    obstacles: list[AABBObs] = field(default_factory=list)
    neighbor_intents: list[IntentObs] = field(default_factory=list)
    messages: list[AgentMessageObs] = field(default_factory=list)
    agent_context: AgentContext | None = None
    planar: bool = True


@dataclass
class PlannerOutput:
    v_cmd: Vec3
    intent_out: IntentMsg | None = None
    messages_out: list[AgentMessage] = field(default_factory=list)
    debug_info: dict[str, object] = field(default_factory=dict)


@dataclass
class RunSpec:
    scenario_path: str
    method: str
    n_agents: int
    seed: int
    comm_profile: str
    out_dir: str
    save_trace: bool
    agent_methods: list[str] | None = None
    policy_spec: str | None = None
