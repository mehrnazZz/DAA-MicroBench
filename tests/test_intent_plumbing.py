from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from microbench.comm.v2v import V2VEmulator
from microbench.runner import run_episode
from microbench.types import AgentState, IntentMsg, RunSpec


def _state(idx: int, pos: tuple[float, float, float]) -> AgentState:
    return AgentState(
        idx=idx,
        pos=np.asarray(pos, dtype=np.float32),
        vel=np.zeros(3, dtype=np.float32),
        goal=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        radius=0.6,
        v_max=3.0,
        a_max=2.0,
    )


class TestIntentPlumbing(unittest.TestCase):
    def test_intent_delay_age_and_expiry(self):
        rng = np.random.default_rng(0)
        profile = {
            "tx_rate_hz": 50,
            "delay": {"type": "constant_ms", "value_ms": 100.0},
            "loss": {"type": "iid", "p": 0.0},
            "noise": {"sigma_pos_m": 0.0, "sigma_vel_mps": 0.0},
        }
        v2v = V2VEmulator(
            profile=profile,
            age_cap_s=0.75,
            rng=rng,
            intent_cfg={"enabled": True, "tx_rate_hz": 10.0, "age_cap_s": 0.75},
        )
        v2v.reset(2)
        states = [_state(0, (0.0, 0.0, 0.0)), _state(1, (1.0, 0.0, 0.0))]

        intent = IntentMsg(
            sender_id=0,
            timestamp_send_s=0.0,
            expiry_s=0.5,
            kind="PROPOSED",
            tube_radius_m=0.8,
            points=np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
            dt_plan_s=0.1,
            mode="normal",
        )
        v2v.publish_intent(sender=0, intent=intent, now_s=0.0, max_points=10)
        v2v.step(0.0, states)
        self.assertIsNone(v2v.get_last_intent(1, 0))

        v2v.step(0.09, states)
        self.assertIsNone(v2v.get_last_intent(1, 0))

        v2v.step(0.11, states)
        m = v2v.get_last_intent(1, 0)
        self.assertIsNotNone(m)
        valid, age = v2v.intent_status(0.11, m)
        self.assertTrue(valid)
        self.assertGreater(age, 0.09)
        self.assertLess(age, 0.2)

        valid2, age2 = v2v.intent_status(0.8, m)
        self.assertFalse(valid2)
        self.assertGreater(age2, 0.5)

    def test_runner_trace_includes_intent_state(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            scenario = tmp / "scenario_intent_headon.yaml"
            scenario.write_text(
            """
scenario:
  name: "intent_headon"
  duration_s: 8.0
world:
  planar: true
  fixed_y_m: 0.0
agent_params:
  radius_m: 1.0
  v_max_mps: 3.0
  a_max_mps2: 3.0
  goal_tolerance_m: 0.5
goals:
  min_goal_distance_m: 8.0
spawn:
  type: "four_way"
  extent_m: 6.0
  lane_half_width_m: 0.0
  y_m: 0.0
intent:
  enabled: true
  tx_rate_hz: 20.0
  max_points: 8
  age_cap_s: 0.75
logging:
  save_events: true
  save_trace_on_collision: true
  trace_window_s: 2.0
""".strip(),
                encoding="utf-8",
            )

            out_dir = tmp / "runs_intent"
            spec = RunSpec(
                scenario_path=str(scenario),
                method="intent_dummy",
                n_agents=2,
                seed=0,
                comm_profile="ideal_50hz",
                out_dir=str(out_dir),
                save_trace=False,
            )
            row = run_episode(spec)
            self.assertGreater(int(row["collisions"]), 0)

            ep_dir = out_dir / "episodes" / "scenario_intent_headon_intent_dummy_n2_seed0_comm_ideal_50hz"
            events_path = ep_dir / "events.jsonl"
            self.assertTrue(events_path.exists())
            lines = [json.loads(x) for x in events_path.read_text(encoding="utf-8").splitlines() if x.strip()]
            self.assertTrue(any("intent_i_of_j" in x and "intent_j_of_i" in x for x in lines))

            traces = sorted(ep_dir.glob("trace_collision_*.jsonl"))
            self.assertTrue(traces)
            first = traces[0].read_text(encoding="utf-8").splitlines()
            meta = json.loads(first[0])
            self.assertIn("intent_i_of_j", meta)
            self.assertIn("intent_j_of_i", meta)
            frame = json.loads(first[1])
            self.assertIn("selected_intents", frame)
            # Ensure receiver-side intent observations are present with age field.
            saw_valid_intent = False
            for line in first[1:]:
                rec = json.loads(line)
                if rec.get("kind") != "frame":
                    continue
                selected_intents = rec.get("selected_intents", {})
                for _, intents in selected_intents.items():
                    for it in intents:
                        self.assertIn("intent_age_s", it)
                        self.assertIn("valid", it)
                        if bool(it.get("valid", False)):
                            saw_valid_intent = True
            self.assertTrue(saw_valid_intent)


if __name__ == "__main__":
    unittest.main()
