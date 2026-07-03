from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from microbench.runner import run_episode
from microbench.types import RunSpec


class _StatefulTestPlanner:
    instances: list["_StatefulTestPlanner"] = []

    def __init__(self):
        self.reset_seed: int | None = None
        self.seen_agents: set[int] = set()
        _StatefulTestPlanner.instances.append(self)

    def reset(self, seed: int) -> None:
        self.reset_seed = int(seed)

    def compute_cmd(self, planner_input):
        self.seen_agents.add(int(planner_input.ego.idx))
        return np.asarray(planner_input.goal_dir, dtype=np.float32) * 0.1


class _NamedTestPlanner:
    calls: list[tuple[str, int]] = []

    def __init__(self, name: str):
        self.name = name

    def reset(self, seed: int) -> None:
        _ = seed

    def compute_cmd(self, planner_input):
        _NamedTestPlanner.calls.append((self.name, int(planner_input.ego.idx)))
        return np.asarray(planner_input.goal_dir, dtype=np.float32) * 0.1


def _write_agentic_scenario(path: Path, duration_s: float = 0.2) -> None:
    path.write_text(
        f"""
scenario:
  name: "agentic_instances"
  duration_s: {duration_s}
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
    center: [-4.0, 0.0, 0.0]
    half: [0.1, 0.0, 1.0]
  goal_region:
    center: [4.0, 0.0, 0.0]
    half: [0.1, 0.0, 1.0]
logging:
  save_events: false
  save_trace: false
""".strip(),
        encoding="utf-8",
    )


class TestAgenticPlannerInstances(unittest.TestCase):
    def test_runner_uses_independent_planner_instance_per_agent(self):
        _StatefulTestPlanner.instances = []
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            scenario = tmp / "scenario_agentic.yaml"
            _write_agentic_scenario(scenario)

            with patch("microbench.runner.make_planner", side_effect=lambda _: _StatefulTestPlanner()):
                run_episode(
                    RunSpec(
                        scenario_path=str(scenario),
                        method="stateful_test",
                        n_agents=3,
                        seed=7,
                        comm_profile="ideal_50hz",
                        out_dir=str(tmp / "runs"),
                        save_trace=False,
                    )
                )

        self.assertEqual(len(_StatefulTestPlanner.instances), 3)
        reset_seeds = [p.reset_seed for p in _StatefulTestPlanner.instances]
        self.assertEqual(len(set(reset_seeds)), 3)
        self.assertEqual([p.seen_agents for p in _StatefulTestPlanner.instances], [{0}, {1}, {2}])

    def test_runner_supports_explicit_heterogeneous_agent_methods(self):
        _NamedTestPlanner.calls = []
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            scenario = tmp / "scenario_agentic.yaml"
            _write_agentic_scenario(scenario)

            with patch("microbench.runner.make_planner", side_effect=lambda name: _NamedTestPlanner(name)):
                row = run_episode(
                    RunSpec(
                        scenario_path=str(scenario),
                        method="fallback",
                        n_agents=3,
                        seed=11,
                        comm_profile="ideal_50hz",
                        out_dir=str(tmp / "runs"),
                        save_trace=False,
                        agent_methods=["alpha", "beta", "alpha"],
                    )
                )

        self.assertEqual(row["method"], "mixed[alpha+beta+alpha]")
        first_seen = {}
        for method_name, agent_idx in _NamedTestPlanner.calls:
            first_seen.setdefault(agent_idx, method_name)
        self.assertEqual(first_seen, {0: "alpha", 1: "beta", 2: "alpha"})

    def test_runner_rejects_ambiguous_heterogeneous_agent_method_count(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            scenario = tmp / "scenario_agentic.yaml"
            _write_agentic_scenario(scenario)

            with self.assertRaisesRegex(ValueError, "agent_methods"):
                run_episode(
                    RunSpec(
                        scenario_path=str(scenario),
                        method="template",
                        n_agents=3,
                        seed=0,
                        comm_profile="ideal_50hz",
                        out_dir=str(tmp / "runs"),
                        save_trace=False,
                        agent_methods=["template", "baseline_goal"],
                    )
                )


if __name__ == "__main__":
    unittest.main()
