from __future__ import annotations

import unittest
import numpy as np

from microbench.planners.template_planner import TemplatePlanner
from microbench.types import AgentState, NeighborObs, PlannerInput


class TestTemplatePlannerHeadOn(unittest.TestCase):
    def test_template_returns_valid_cmd_shape(self):
        planner = TemplatePlanner()
        planner.reset(0)

        ego = AgentState(
            idx=0,
            pos=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            vel=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            goal=np.asarray([10.0, 0.0, 0.0], dtype=np.float32),
            radius=0.6,
            v_max=3.0,
            a_max=2.0,
        )
        nbr = NeighborObs(
            idx=1,
            pos=np.asarray([5.0, 0.0, 0.0], dtype=np.float32),
            vel=np.asarray([-2.0, 0.0, 0.0], dtype=np.float32),
            radius=0.6,
            msg_age_sec=0.1,
            valid=True,
        )
        p_in = PlannerInput(
            ego=ego,
            goal_dir=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            neighbors=[nbr],
            dt=0.02,
            t=0.0,
        )

        cmd = planner.compute_cmd(p_in)
        self.assertEqual(cmd.shape, (3,))
        self.assertTrue(np.isfinite(cmd).all())


if __name__ == "__main__":
    unittest.main()
