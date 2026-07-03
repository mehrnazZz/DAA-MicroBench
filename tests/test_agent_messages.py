from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from microbench.comm.v2v import V2VEmulator
from microbench.runner import run_episode
from microbench.types import AgentMessage, AgentState, PlannerOutput, RunSpec


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


class TestAgentMessages(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
