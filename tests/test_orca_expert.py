from __future__ import annotations

import unittest
import numpy as np

from microbench.planners.orca_expert import OrcaExpertPlanner
from microbench.types import AABBObs, AgentState, NeighborObs, PlannerInput


def _agent(pos, vel=(0.0, 0.0, 0.0), goal=(10.0, 0.0, 0.0), radius=0.5, v_max=3.0, a_max=2.0):
    return AgentState(
        idx=0,
        pos=np.asarray(pos, dtype=np.float32),
        vel=np.asarray(vel, dtype=np.float32),
        goal=np.asarray(goal, dtype=np.float32),
        radius=radius,
        v_max=v_max,
        a_max=a_max,
    )


class TestOrcaExpert(unittest.TestCase):
    def setUp(self) -> None:
        self.planner = OrcaExpertPlanner(
            cfg={
                "time_horizon_s": 3.0,
                "safety_margin_m": 0.2,
                "stale_inflation_gain": 0.8,
                "stale_age_cap_s": 0.75,
                "max_neighbors": 8,
            },
            age_cap_s=0.75,
        )
        self.planner.reset(0)

    def test_no_neighbors_goes_to_goal(self):
        ego = _agent(pos=(0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0))
        pi = PlannerInput(
            ego=ego,
            goal_dir=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            neighbors=[],
            dt=0.02,
            t=0.0,
        )
        cmd = self.planner.compute_cmd(pi)
        self.assertGreater(cmd[0], 2.9)
        self.assertAlmostEqual(float(cmd[2]), 0.0, places=4)

    def test_head_on_changes_command(self):
        ego = _agent(pos=(0.0, 0.0, 0.0), vel=(2.0, 0.0, 0.0))
        n = NeighborObs(
            idx=1,
            pos=np.asarray([0.8, 0.0, 0.0], dtype=np.float32),
            vel=np.asarray([-2.0, 0.0, 0.0], dtype=np.float32),
            radius=0.5,
            msg_age_sec=0.0,
            valid=True,
        )
        pi = PlannerInput(
            ego=ego,
            goal_dir=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            neighbors=[n],
            dt=0.02,
            t=0.0,
        )
        cmd = self.planner.compute_cmd(pi)
        # In an imminent overlap, the expert should not keep driving forward.
        self.assertLess(float(cmd[0]), 0.0)

    def test_crossing_avoids_straight_pref(self):
        ego = _agent(pos=(0.0, 0.0, 0.0), vel=(2.0, 0.0, 0.0))
        n = NeighborObs(
            idx=1,
            pos=np.asarray([0.9, 0.0, -0.5], dtype=np.float32),
            vel=np.asarray([0.0, 0.0, 2.0], dtype=np.float32),
            radius=0.5,
            msg_age_sec=0.0,
            valid=True,
        )
        pi = PlannerInput(
            ego=ego,
            goal_dir=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            neighbors=[n],
            dt=0.02,
            t=0.0,
        )
        cmd = self.planner.compute_cmd(pi)
        self.assertTrue(abs(float(cmd[2])) > 1e-3 or float(cmd[0]) < 2.95)

    def test_stale_neighbor_more_conservative(self):
        ego = _agent(pos=(0.0, 0.0, 0.0), vel=(2.0, 0.0, 0.0))
        base_n = dict(
            idx=1,
            pos=np.asarray([3.0, 0.0, 0.0], dtype=np.float32),
            vel=np.asarray([-1.5, 0.0, 0.0], dtype=np.float32),
            radius=0.5,
            valid=True,
        )
        fresh = NeighborObs(msg_age_sec=0.0, **base_n)
        stale = NeighborObs(msg_age_sec=0.75, **base_n)

        pi_fresh = PlannerInput(
            ego=ego,
            goal_dir=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            neighbors=[fresh],
            dt=0.02,
            t=0.0,
        )
        pi_stale = PlannerInput(
            ego=ego,
            goal_dir=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            neighbors=[stale],
            dt=0.02,
            t=0.0,
        )

        cmd_fresh = self.planner.compute_cmd(pi_fresh)
        cmd_stale = self.planner.compute_cmd(pi_stale)

        pref = np.asarray([ego.v_max, 0.0, 0.0], dtype=np.float32)
        dev_fresh = np.linalg.norm(cmd_fresh - pref)
        dev_stale = np.linalg.norm(cmd_stale - pref)
        self.assertGreaterEqual(float(dev_stale), float(dev_fresh) - 1e-6)

    def test_obstacle_in_path_creates_avoidance(self):
        ego = _agent(pos=(0.0, 0.0, 0.0), vel=(2.0, 0.0, 0.0))
        pi = PlannerInput(
            ego=ego,
            goal_dir=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            neighbors=[],
            dt=0.02,
            t=0.0,
            obstacles=[
                AABBObs(
                    center=np.asarray([2.0, 0.0, 0.0], dtype=np.float32),
                    half=np.asarray([0.5, 0.5, 0.5], dtype=np.float32),
                )
            ],
        )
        cmd = self.planner.compute_cmd(pi)
        self.assertTrue(abs(float(cmd[2])) > 1e-3 or float(cmd[0]) < 2.9)


if __name__ == "__main__":
    unittest.main()
