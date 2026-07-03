from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from microbench.core.perception import sense_neighbors
from microbench.runner import run_episode
from microbench.types import AgentState, RunSpec


def _agent(idx: int, pos: tuple[float, float, float], goal=(10.0, 0.0, 0.0)) -> AgentState:
    return AgentState(
        idx=idx,
        pos=np.asarray(pos, dtype=np.float32),
        vel=np.zeros(3, dtype=np.float32),
        goal=np.asarray(goal, dtype=np.float32),
        radius=0.3,
        v_max=1.0,
        a_max=1.0,
    )


class TestPerception(unittest.TestCase):
    def test_sensor_range_and_fov_filter_neighbors(self):
        ego = _agent(0, (0.0, 0.0, 0.0))
        states = [
            ego,
            _agent(1, (5.0, 0.0, 0.0)),
            _agent(2, (-5.0, 0.0, 0.0)),
            _agent(3, (12.0, 0.0, 0.0)),
        ]
        obs = sense_neighbors(
            ego=ego,
            states=states,
            goal_dir=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            obstacles=[],
            perception_cfg={"sensor": {"range_m": 10.0, "fov_deg": 90.0}},
            planar=True,
            rng=np.random.default_rng(0),
        )
        self.assertEqual([o.idx for o in obs], [1])
        self.assertEqual(obs[0].source, "sensor")

    def test_sensor_occlusion_blocks_line_of_sight(self):
        ego = _agent(0, (0.0, 0.0, 0.0))
        states = [ego, _agent(1, (5.0, 0.0, 0.0))]
        obstacles = [{"aabb": {"center": [2.5, 0.0, 0.0], "half": [0.5, 1.0, 1.0]}}]
        obs = sense_neighbors(
            ego=ego,
            states=states,
            goal_dir=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            obstacles=obstacles,
            perception_cfg={"sensor": {"range_m": 10.0, "fov_deg": 360.0, "occlusion": True}},
            planar=True,
            rng=np.random.default_rng(0),
        )
        self.assertEqual(obs, [])

    def test_runner_sensor_mode_trace_marks_observation_source(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            scenario = tmp / "sensor_mode.yaml"
            scenario.write_text(
                """
scenario:
  name: "sensor_mode"
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
  min_goal_distance_m: 4.0
spawn:
  type: "rect_to_rect"
  start_region:
    center: [-2.0, 0.0, 0.0]
    half: [0.0, 0.0, 0.0]
  goal_region:
    center: [2.0, 0.0, 0.0]
    half: [0.0, 0.0, 0.0]
perception:
  mode: "sensor"
  sensor:
    range_m: 10.0
    fov_deg: 360.0
logging:
  save_trace: true
  trace_save_failures_only: false
  trace_max_steps: 20
  save_events: false
""".strip(),
                encoding="utf-8",
            )
            out_dir = tmp / "runs"
            run_episode(
                RunSpec(
                    scenario_path=str(scenario),
                    method="template",
                    n_agents=2,
                    seed=0,
                    comm_profile="ideal_50hz",
                    out_dir=str(out_dir),
                    save_trace=True,
                )
            )
            trace_path = out_dir / "episodes" / "sensor_mode_template_n2_seed0_comm_ideal_50hz" / "trace_episode.jsonl"
            frames = [
                json.loads(line)
                for line in trace_path.read_text(encoding="utf-8").splitlines()
                if json.loads(line).get("kind") == "frame"
            ]
            self.assertTrue(
                any(
                    obs.get("source") == "sensor"
                    for frame in frames
                    for obs_list in frame.get("selected_obs", [])
                    for obs in obs_list
                )
            )

    def test_runner_observation_metrics_reflect_sensor_mode(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            scenario = tmp / "sensor_metrics.yaml"
            scenario.write_text(
                """
scenario:
  name: "sensor_metrics"
  duration_s: 0.2
world:
  planar: true
  fixed_y_m: 0.0
agent_params:
  radius_m: 0.2
  v_max_mps: 1.0
  a_max_mps2: 1.0
  goal_tolerance_m: 0.1
goals:
  min_goal_distance_m: 4.0
spawn:
  type: "rect_to_rect"
  start_region:
    center: [-2.0, 0.0, 0.0]
    half: [0.0, 0.0, 0.0]
  goal_region:
    center: [2.0, 0.0, 0.0]
    half: [0.0, 0.0, 0.0]
perception:
  mode: "sensor"
  sensor:
    range_m: 10.0
    fov_deg: 360.0
logging:
  save_events: false
  save_trace: false
""".strip(),
                encoding="utf-8",
            )
            row = run_episode(
                RunSpec(
                    scenario_path=str(scenario),
                    method="template",
                    n_agents=2,
                    seed=0,
                    comm_profile="ideal_50hz",
                    out_dir=str(tmp / "runs"),
                    save_trace=False,
                )
            )
        self.assertGreater(float(row["obs_neighbors_mean"]), 0.0)
        self.assertEqual(float(row["obs_sensor_fraction"]), 1.0)
        self.assertEqual(float(row["obs_v2v_fraction"]), 0.0)


if __name__ == "__main__":
    unittest.main()
