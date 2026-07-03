from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from microbench.comm.messages import make_negotiation_proposal, validate_agent_message
from microbench.comm.v2v import V2VEmulator
from microbench.runner import run_episode
from microbench.types import MSG_NEGOTIATION_PROPOSAL, AgentMessage, AgentState, PlannerOutput, RunSpec


def _state(idx: int) -> AgentState:
    return AgentState(
        idx=idx,
        pos=np.asarray([float(idx), 0.0, 0.0], dtype=np.float32),
        vel=np.zeros(3, dtype=np.float32),
        goal=np.asarray([10.0, 0.0, 0.0], dtype=np.float32),
        radius=0.5,
        v_max=1.0,
        a_max=1.0,
    )


class _MessageTestPlanner:
    received: list[tuple[int, str, int]] = []

    def reset(self, seed: int) -> None:
        _ = seed

    def compute_cmd(self, planner_input):
        ego = planner_input.ego
        for msg in planner_input.messages:
            if msg.valid:
                _MessageTestPlanner.received.append((int(ego.idx), str(msg.kind), int(msg.sender_id)))
        messages_out = []
        if int(ego.idx) == 0 and float(planner_input.t) < 1e-9:
            messages_out.append(
                AgentMessage(
                    sender_id=0,
                    recipient_id=1,
                    timestamp_send_s=float(planner_input.t),
                    kind="PING",
                    payload={"ok": True},
                    ttl_s=1.0,
                )
            )
        return PlannerOutput(v_cmd=np.zeros(3, dtype=np.float32), messages_out=messages_out)


class _BurstMessagePlanner:
    def reset(self, seed: int) -> None:
        _ = seed

    def compute_cmd(self, planner_input):
        messages_out = []
        if int(planner_input.ego.idx) == 0 and float(planner_input.t) < 1e-9:
            for seq in range(2):
                messages_out.append(
                    AgentMessage(
                        sender_id=0,
                        recipient_id=1,
                        timestamp_send_s=float(planner_input.t),
                        kind="NEGOTIATION_PROPOSAL",
                        payload={
                            "proposal_id": f"proposal-{seq}",
                            "action": "yield",
                            "start_s": float(planner_input.t),
                            "duration_s": 0.5,
                            "params": {"proposal": seq},
                        },
                        ttl_s=1.0,
                        message_id=f"proposal-{seq}",
                        correlation_id="encounter-0",
                        seq=seq,
                        channel="negotiation",
                        priority=5,
                    )
                )
        return PlannerOutput(v_cmd=np.zeros(3, dtype=np.float32), messages_out=messages_out)


class TestAgentMessages(unittest.TestCase):
    def test_standard_message_payload_validation(self):
        valid = make_negotiation_proposal(
            sender_id=0,
            recipient_id=1,
            now_s=0.0,
            proposal_id="p0",
            action="yield",
            start_s=0.0,
            duration_s=0.5,
        )
        self.assertEqual(validate_agent_message(valid), (True, None))

        invalid = AgentMessage(
            sender_id=0,
            recipient_id=1,
            timestamp_send_s=0.0,
            kind=MSG_NEGOTIATION_PROPOSAL,
            payload={"proposal_id": "bad", "action": "yield"},
        )
        ok, reason = validate_agent_message(invalid)
        self.assertFalse(ok)
        self.assertIn("missing_payload_fields", str(reason))

    def test_v2v_agent_message_delivery_and_expiry(self):
        profile = {
            "tx_rate_hz": 50,
            "delay": {"type": "constant_ms", "value_ms": 0.0},
            "loss": {"type": "iid", "p": 0.0},
            "noise": {"sigma_pos_m": 0.0, "sigma_vel_mps": 0.0},
        }
        v2v = V2VEmulator(profile=profile, age_cap_s=0.75, rng=np.random.default_rng(0))
        states = [_state(0), _state(1)]
        v2v.reset(2)
        v2v.publish_agent_message(
            sender=0,
            msg=AgentMessage(sender_id=0, recipient_id=1, timestamp_send_s=0.0, kind="HELLO", ttl_s=0.1),
            now_s=0.0,
            n_agents=2,
        )
        v2v.step(0.0, states)
        delivered = v2v.drain_agent_messages(1, 0.0)
        self.assertEqual(len(delivered), 1)
        self.assertEqual(delivered[0].kind, "HELLO")
        self.assertTrue(delivered[0].valid)

        v2v.publish_agent_message(
            sender=0,
            msg=AgentMessage(sender_id=0, recipient_id=1, timestamp_send_s=0.0, kind="STALE", ttl_s=0.1),
            now_s=0.0,
            n_agents=2,
        )
        v2v.step(0.0, states)
        expired = v2v.drain_agent_messages(1, 0.2)
        self.assertEqual(len(expired), 1)
        self.assertFalse(expired[0].valid)

    def test_v2v_agent_message_metadata_rate_limit_and_event_log(self):
        profile = {
            "tx_rate_hz": 50,
            "delay": {"type": "constant_ms", "value_ms": 0.0},
            "loss": {"type": "iid", "p": 0.0},
            "noise": {"sigma_pos_m": 0.0, "sigma_vel_mps": 0.0},
            "agent_messages": {"rate_limit_hz": 1, "overhead_bytes": 0},
        }
        v2v = V2VEmulator(profile=profile, age_cap_s=0.75, rng=np.random.default_rng(0))
        states = [_state(0), _state(1)]
        v2v.reset(2)
        for seq in range(2):
            v2v.publish_agent_message(
                sender=0,
                msg=AgentMessage(
                    sender_id=0,
                    recipient_id=1,
                    timestamp_send_s=0.0,
                    kind="NEGOTIATION_PROPOSAL",
                    payload={
                        "proposal_id": f"m-{seq}",
                        "action": "yield",
                        "start_s": 0.0,
                        "duration_s": 0.5,
                        "params": {"seq": seq},
                    },
                    message_id=f"m-{seq}",
                    correlation_id="c-0",
                    seq=seq,
                    channel="negotiation",
                    priority=4,
                ),
                now_s=0.0,
                n_agents=2,
            )
        v2v.step(0.0, states)
        delivered = v2v.drain_agent_messages(1, 0.0)
        events = v2v.drain_agent_message_events()
        stats = v2v.agent_message_stats_snapshot()

        self.assertEqual(len(delivered), 1)
        self.assertEqual(delivered[0].message_id, "m-0")
        self.assertEqual(delivered[0].correlation_id, "c-0")
        self.assertEqual(delivered[0].seq, 0)
        self.assertEqual(delivered[0].channel, "negotiation")
        self.assertEqual(delivered[0].priority, 4)
        self.assertGreater(delivered[0].size_bytes, 0)
        self.assertEqual(stats["agent_msg_attempted"], 2)
        self.assertEqual(stats["agent_msg_scheduled"], 1)
        self.assertEqual(stats["agent_msg_delivered"], 1)
        self.assertEqual(stats["agent_msg_dropped"], 1)
        self.assertEqual([e["event"] for e in events], ["scheduled", "dropped", "delivered"])
        self.assertEqual(events[1]["reason"], "rate_or_bandwidth_limit")

    def test_v2v_drops_invalid_standard_message_payload(self):
        profile = {
            "tx_rate_hz": 50,
            "delay": {"type": "constant_ms", "value_ms": 0.0},
            "loss": {"type": "iid", "p": 0.0},
            "noise": {"sigma_pos_m": 0.0, "sigma_vel_mps": 0.0},
        }
        v2v = V2VEmulator(profile=profile, age_cap_s=0.75, rng=np.random.default_rng(0))
        v2v.reset(2)
        v2v.publish_agent_message(
            sender=0,
            msg=AgentMessage(
                sender_id=0,
                recipient_id=1,
                timestamp_send_s=0.0,
                kind=MSG_NEGOTIATION_PROPOSAL,
                payload={"proposal_id": "bad", "action": "yield"},
            ),
            now_s=0.0,
            n_agents=2,
        )
        events = v2v.drain_agent_message_events()
        stats = v2v.agent_message_stats_snapshot()

        self.assertEqual(stats["agent_msg_attempted"], 1)
        self.assertEqual(stats["agent_msg_scheduled"], 0)
        self.assertEqual(stats["agent_msg_dropped"], 1)
        self.assertEqual(events[0]["event"], "dropped")
        self.assertIn("missing_payload_fields", events[0]["reason"])

    def test_runner_delivers_agent_messages_to_planner_input(self):
        _MessageTestPlanner.received = []
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            scenario = tmp / "scenario_messages.yaml"
            scenario.write_text(
                """
scenario:
  name: "message_smoke"
  duration_s: 0.1
world:
  planar: true
  fixed_y_m: 0.0
agent_params:
  radius_m: 0.2
  v_max_mps: 1.0
  a_max_mps2: 1.0
  goal_tolerance_m: 0.1
goals:
  min_goal_distance_m: 2.0
spawn:
  type: "rect_to_rect"
  start_region:
    center: [-2.0, 0.0, 0.0]
    half: [0.1, 0.0, 0.1]
  goal_region:
    center: [2.0, 0.0, 0.0]
    half: [0.1, 0.0, 0.1]
logging:
  save_events: false
  save_trace: false
""".strip(),
                encoding="utf-8",
            )
            with patch("microbench.runner.make_planner", side_effect=lambda _: _MessageTestPlanner()):
                run_episode(
                    RunSpec(
                        scenario_path=str(scenario),
                        method="message_test",
                        n_agents=2,
                        seed=0,
                        comm_profile="ideal_50hz",
                        out_dir=str(tmp / "runs"),
                        save_trace=False,
                    )
                )
        self.assertIn((1, "PING", 0), _MessageTestPlanner.received)

    def test_runner_records_message_events_and_comm_metrics(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            scenario = tmp / "scenario_message_metrics.yaml"
            scenario.write_text(
                """
scenario:
  name: "message_metrics"
  duration_s: 0.1
world:
  planar: true
  fixed_y_m: 0.0
agent_params:
  radius_m: 0.2
  v_max_mps: 1.0
  a_max_mps2: 1.0
  goal_tolerance_m: 0.1
goals:
  min_goal_distance_m: 2.0
spawn:
  type: "rect_to_rect"
  start_region:
    center: [-2.0, 0.0, 0.0]
    half: [0.1, 0.0, 0.1]
  goal_region:
    center: [2.0, 0.0, 0.0]
    half: [0.1, 0.0, 0.1]
comm:
  agent_messages:
    rate_limit_hz: 1
logging:
  save_events: false
  save_trace: true
  trace_save_failures_only: false
  trace_max_steps: 20
""".strip(),
                encoding="utf-8",
            )
            with patch("microbench.runner.make_planner", side_effect=lambda _: _BurstMessagePlanner()):
                row = run_episode(
                    RunSpec(
                        scenario_path=str(scenario),
                        method="burst_message_test",
                        n_agents=2,
                        seed=0,
                        comm_profile="ideal_50hz",
                        out_dir=str(tmp / "runs"),
                        save_trace=True,
                    )
                )

            self.assertEqual(row["comm_agent_msg_attempted"], 2)
            self.assertEqual(row["comm_agent_msg_scheduled"], 1)
            self.assertEqual(row["comm_agent_msg_dropped"], 1)
            self.assertEqual(row["comm_agent_msg_delivered"], 1)
            self.assertGreater(row["comm_agent_msg_bandwidth_Bps"], 0.0)

            trace_path = (
                tmp
                / "runs"
                / "episodes"
                / "scenario_message_metrics_burst_message_test_n2_seed0_comm_ideal_50hz"
                / "trace_episode.jsonl"
            )
            frames = [
                json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
                if json.loads(line).get("kind") == "frame"
            ]
            events = [e for frame in frames for e in frame.get("message_events", [])]
            self.assertIn("scheduled", [e["event"] for e in events])
            self.assertIn("dropped", [e["event"] for e in events])
            self.assertIn("delivered", [e["event"] for e in events])
            scheduled = next(e for e in events if e["event"] == "scheduled")
            self.assertEqual(scheduled["message_id"], "proposal-0")
            self.assertEqual(scheduled["correlation_id"], "encounter-0")

    def test_negotiation_yield_exchanges_proposal_and_ack(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            scenario = tmp / "scenario_negotiation.yaml"
            scenario.write_text(
                """
scenario:
  name: "negotiation_headon"
  duration_s: 0.3
world:
  planar: true
  fixed_y_m: 0.0
agent_params:
  radius_m: 0.2
  v_max_mps: 1.0
  a_max_mps2: 10.0
  goal_tolerance_m: 0.1
neighbors:
  range_m: 20.0
  top_k: 4
  threat_metric: distance
goals:
  min_goal_distance_m: 3.0
spawn:
  type: "circle_swap"
  center: [0.0, 0.0, 0.0]
  radius_m: 1.5
  jitter_m: 0.0
agents:
  by_id:
    0:
      priority: 0
    1:
      priority: 1
logging:
  save_events: false
  save_trace: true
  trace_save_failures_only: false
  trace_max_steps: 30
""".strip(),
                encoding="utf-8",
            )
            row = run_episode(
                RunSpec(
                    scenario_path=str(scenario),
                    method="negotiation_yield",
                    n_agents=2,
                    seed=0,
                    comm_profile="ideal_50hz",
                    out_dir=str(tmp / "runs"),
                    save_trace=True,
                )
            )

            self.assertGreaterEqual(row["comm_negotiation_proposals"], 1)
            self.assertGreaterEqual(row["comm_negotiation_acks"], 1)
            self.assertGreaterEqual(row["comm_negotiation_correlations_acked"], 1)

            trace_path = (
                tmp
                / "runs"
                / "episodes"
                / "scenario_negotiation_negotiation_yield_n2_seed0_comm_ideal_50hz"
                / "trace_episode.jsonl"
            )
            frames = [
                json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
                if json.loads(line).get("kind") == "frame"
            ]
            events = [e for frame in frames for e in frame.get("message_events", [])]
            kinds = [e.get("kind") for e in events if e.get("event") in {"scheduled", "delivered"}]
            self.assertIn("NEGOTIATION_PROPOSAL", kinds)
            self.assertIn("ACK", kinds)
            debug = [frame.get("planner_debug", []) for frame in frames]
            self.assertTrue(any(len(dbg) > 1 and float(dbg[1].get("speed_scale", 1.0)) < 1.0 for dbg in debug))


if __name__ == "__main__":
    unittest.main()
