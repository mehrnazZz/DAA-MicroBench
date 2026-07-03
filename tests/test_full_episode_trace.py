from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from microbench.runner import run_episode
from microbench.types import RunSpec


class TestFullEpisodeTrace(unittest.TestCase):
    def test_episode_trace_written_when_enabled(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            scenario = tmp / "scenario_trace.yaml"
            scenario.write_text(
                """
scenario:
  name: "trace_smoke"
  duration_s: 2.0
world:
  planar: true
  fixed_y_m: 0.0
agent_params:
  radius_m: 0.4
  v_max_mps: 2.0
  a_max_mps2: 2.0
  goal_tolerance_m: 0.5
goals:
  min_goal_distance_m: 6.0
spawn:
  type: "rect_to_rect"
  start_region:
    center: [-4.0, 0.0, 0.0]
    half: [0.1, 0.0, 0.1]
  goal_region:
    center: [4.0, 0.0, 0.0]
    half: [0.1, 0.0, 0.1]
logging:
  save_trace: true
  trace_save_failures_only: false
  trace_max_steps: 200
  save_events: false
  save_trace_on_collision: false
""".strip(),
                encoding="utf-8",
            )
            out_dir = tmp / "runs_trace"
            spec = RunSpec(
                scenario_path=str(scenario),
                method="template",
                n_agents=1,
                seed=0,
                comm_profile="ideal_50hz",
                out_dir=str(out_dir),
                save_trace=True,
            )
            run_episode(spec)

            ep_dir = out_dir / "episodes" / "scenario_trace_template_n1_seed0_comm_ideal_50hz"
            trace_path = ep_dir / "trace_episode.jsonl"
            self.assertTrue(trace_path.exists())
            lines = trace_path.read_text(encoding="utf-8").splitlines()
            self.assertGreater(len(lines), 2)
            meta = json.loads(lines[0])
            self.assertEqual(meta.get("trace_type"), "episode")
            frame = json.loads(lines[1])
            self.assertEqual(frame.get("kind"), "frame")
            self.assertIn("positions", frame)


if __name__ == "__main__":
    unittest.main()
