from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from microbench.config import load_yaml
from microbench.planners.orca_expert import OrcaExpertPlanner
from microbench.replay import render_interactive_trace
from microbench.replay.replay_matplotlib import render_trace
from microbench.scenarios import EventEngine, generate_spawns_goals
from microbench.runner import run_episode
from microbench.types import AgentState, NeighborObs, PlannerInput, RunSpec


class Test3DSupport(unittest.TestCase):
    def _smoke_copy(self, src: str, dst: Path, duration_s: float = 3.0) -> Path:
        cfg = load_yaml(src)
        cfg.setdefault("scenario", {})["duration_s"] = duration_s
        dst.write_text(json.dumps(cfg), encoding="utf-8")
        return dst

    def test_orca_nonplanar_preserves_vertical_component(self):
        planner = OrcaExpertPlanner(
            cfg={
                "time_horizon_s": 3.0,
                "safety_margin_m": 0.2,
                "stale_inflation_gain": 0.8,
                "stale_age_cap_s": 0.75,
                "max_neighbors": 8,
            },
            age_cap_s=0.75,
        )
        planner.reset(0)
        ego = AgentState(
            idx=0,
            pos=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            vel=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            goal=np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
            radius=0.5,
            v_max=3.0,
            a_max=2.0,
        )
        p_in = PlannerInput(
            ego=ego,
            goal_dir=np.asarray([0.70710677, 0.70710677, 0.0], dtype=np.float32),
            neighbors=[],
            dt=0.02,
            t=0.0,
            planar=False,
        )
        cmd = planner.compute_cmd(p_in)
        self.assertGreater(float(cmd[1]), 0.1)

    def test_orca_nonplanar_avoids_vertical_head_on(self):
        planner = OrcaExpertPlanner(
            cfg={
                "time_horizon_s": 3.0,
                "safety_margin_m": 0.2,
                "stale_inflation_gain": 0.8,
                "stale_age_cap_s": 0.75,
                "max_neighbors": 8,
            },
            age_cap_s=0.75,
        )
        planner.reset(0)
        ego = AgentState(
            idx=0,
            pos=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            vel=np.asarray([0.0, 2.0, 0.0], dtype=np.float32),
            goal=np.asarray([0.0, 5.0, 0.0], dtype=np.float32),
            radius=0.5,
            v_max=3.0,
            a_max=2.0,
        )
        nbr = NeighborObs(
            idx=1,
            pos=np.asarray([0.0, 0.8, 0.0], dtype=np.float32),
            vel=np.asarray([0.0, -2.0, 0.0], dtype=np.float32),
            radius=0.5,
            msg_age_sec=0.0,
            valid=True,
        )
        p_in = PlannerInput(
            ego=ego,
            goal_dir=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
            neighbors=[nbr],
            dt=0.02,
            t=0.0,
            planar=False,
        )
        cmd = planner.compute_cmd(p_in)
        self.assertLess(float(cmd[1]), 0.0)

    def test_nonplanar_episode_trace_and_replay(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            scenario = tmp / "scenario_3d.yaml"
            scenario.write_text(
                """
scenario:
  name: "support_3d"
  duration_s: 1.0
world:
  planar: false
agent_params:
  radius_m: 0.4
  v_max_mps: 2.0
  a_max_mps2: 2.0
  goal_tolerance_m: 0.1
goals:
  min_goal_distance_m: 1.0
spawn:
  type: "rect_to_rect"
  start_region:
    center: [0.0, 0.0, 0.0]
    half: [0.0, 0.0, 0.0]
  goal_region:
    center: [1.0, 1.0, 0.0]
    half: [0.0, 0.0, 0.0]
logging:
  save_trace: true
  trace_save_failures_only: false
  trace_max_steps: 100
  save_events: false
  save_trace_on_collision: false
""".strip(),
                encoding="utf-8",
            )
            out_dir = tmp / "runs_3d"
            run_episode(
                RunSpec(
                    scenario_path=str(scenario),
                    method="template",
                    n_agents=1,
                    seed=0,
                    comm_profile="ideal_50hz",
                    out_dir=str(out_dir),
                    save_trace=True,
                )
            )
            trace_path = out_dir / "episodes" / "scenario_3d_template_n1_seed0_comm_ideal_50hz" / "trace_episode.jsonl"
            self.assertTrue(trace_path.exists())
            lines = trace_path.read_text(encoding="utf-8").splitlines()
            frames = [json.loads(x) for x in lines if json.loads(x).get("kind") == "frame"]
            ys = [float(f["positions"][0][1]) for f in frames]
            self.assertGreater(max(ys) - min(ys), 1e-3)

            gif_path = tmp / "episode_3d.gif"
            try:
                render_trace(str(trace_path), str(gif_path), fps=10, tail=10, show_sensed=False)
            except RuntimeError as exc:
                self.skipTest(f"matplotlib unavailable in test env: {exc}")
            self.assertTrue(gif_path.exists())

            html_path = tmp / "episode_3d.html"
            render_interactive_trace(str(trace_path), str(html_path), tail=10, show_sensed=False)
            self.assertTrue(html_path.exists())
            html = html_path.read_text(encoding="utf-8")
            self.assertIn("Plotly.newPlot", html)
            self.assertIn("const replay =", html)
            self.assertIn("Neighbor Distances", html)
            self.assertIn("support_3d", html)

    def test_builtin_3d_scenario_runs_with_orca(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            scenario_path = self._smoke_copy("config/scenarios/stacked_swap_3d.yaml", tmp / "stacked_swap_3d_smoke.yaml")
            out_dir = Path(td) / "runs_3d_builtin"
            for method in ("orca_heuristic", "orca_with_staleness"):
                result = run_episode(
                    RunSpec(
                        scenario_path=str(scenario_path),
                        method=method,
                        n_agents=2,
                        seed=0,
                        comm_profile="ideal_50hz",
                        out_dir=str(out_dir),
                        save_trace=False,
                    )
                )
                self.assertEqual(result["scenario"], "stacked_swap_3d_smoke")
                self.assertEqual(result["method"], method)
                self.assertEqual(int(result["N"]), 2)
                self.assertIn("completion_rate", result)

    def test_layered_spawn_goal_shift(self):
        rng = np.random.default_rng(0)
        spawns, goals = generate_spawns_goals(
            {
                "spawn": {
                    "type": "rect_to_rect",
                    "start_region": {"center": [-10.0, 0.0, 0.0], "half": [1.0, 0.0, 1.0]},
                    "goal_region": {"center": [10.0, 0.0, 0.0], "half": [1.0, 0.0, 1.0]},
                    "start_layers_m": [-4.0, 0.0, 4.0],
                    "goal_layers_m": [-4.0, 0.0, 4.0],
                    "goal_layers_shift": 1,
                },
                "goals": {"min_goal_distance_m": 1.0},
            },
            6,
            rng,
        )
        self.assertListEqual(spawns[:, 1].tolist(), [-4.0, 0.0, 4.0, -4.0, 0.0, 4.0])
        self.assertListEqual(goals[:, 1].tolist(), [0.0, 4.0, -4.0, 0.0, 4.0, -4.0])

    def test_vertical_weather_event_overrides_y_command(self):
        rng = np.random.default_rng(0)
        events = EventEngine(
            [
                {
                    "type": "weather_maneuver",
                    "enabled": True,
                    "t_start_s": 1.0,
                    "duration_s": 2.0,
                    "n_agents": 1,
                    "selection": "closest_to_gate",
                    "forced_policy": {
                        "type": "vertical_shift_and_slow",
                        "vertical_speed_mps": 2.0,
                        "speed_scale": 0.5,
                        "direction": "up",
                    },
                }
            ],
            rng,
        )
        states = [
            AgentState(
                idx=0,
                pos=np.asarray([0.2, 0.0, 0.0], dtype=np.float32),
                vel=np.asarray([2.0, 0.0, 0.0], dtype=np.float32),
                goal=np.asarray([10.0, 0.0, 0.0], dtype=np.float32),
                radius=0.5,
                v_max=3.0,
                a_max=2.0,
            )
        ]
        v_cmds = [np.asarray([2.0, 0.0, 0.0], dtype=np.float32)]
        out = events.apply_overrides(1.5, states, v_cmds)
        self.assertGreater(float(out[0][1]), 1.5)
        self.assertLess(float(np.linalg.norm(out[0][[0, 2]])), 2.0)

    def test_new_builtin_3d_scenarios_run(self):
        scenarios = [
            "config/scenarios/layered_funnel_3d.yaml",
            "config/scenarios/layered_intersection_3d.yaml",
            "config/scenarios/weather_vertical_event_3d.yaml",
            "config/scenarios/vertical_crossing_obstacles_3d.yaml",
            "config/scenarios/urban_airspace_3d.yaml",
        ]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for i, scenario_path in enumerate(scenarios):
                smoke_path = self._smoke_copy(scenario_path, root / f"scenario_{i}.yaml")
                result = run_episode(
                    RunSpec(
                        scenario_path=str(smoke_path),
                        method="orca_heuristic",
                        n_agents=2,
                        seed=i,
                        comm_profile="ideal_50hz",
                        out_dir=str(root / f"run_{i}"),
                        save_trace=False,
                    )
                )
                self.assertGreaterEqual(float(result["completion_rate"]), 0.0)
                self.assertEqual(int(result["N"]), 2)


if __name__ == "__main__":
    unittest.main()
