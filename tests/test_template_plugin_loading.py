from __future__ import annotations

import unittest
import numpy as np

from microbench.planners import make_planner
from microbench.types import AgentState, PlannerInput


class TestTemplatePluginLoading(unittest.TestCase):
    def test_template_alias_loads_and_runs(self):
        planner = make_planner("template")
        planner.reset(0)
        ego = AgentState(
            idx=0,
            pos=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            vel=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            goal=np.asarray([5.0, 0.0, 0.0], dtype=np.float32),
            radius=0.6,
            v_max=3.0,
            a_max=2.0,
        )
        p_in = PlannerInput(
            ego=ego,
            goal_dir=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            neighbors=[],
            dt=0.02,
            t=0.0,
        )
        cmd = planner.compute_cmd(p_in)
        self.assertEqual(tuple(cmd.shape), (3,))
        self.assertTrue(np.isfinite(cmd).all())


if __name__ == "__main__":
    unittest.main()
