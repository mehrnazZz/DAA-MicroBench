from __future__ import annotations

import unittest
import numpy as np

from microbench.core.neighbors import select_neighbors
from microbench.types import NeighborObs


class TestNeighborOrdering(unittest.TestCase):
    def test_tie_breaks_by_neighbor_id(self):
        ego_pos = np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
        ego_vel = np.zeros(3, dtype=np.float32)
        # Equal distance to ego, intentionally shuffled order.
        obs = [
            NeighborObs(idx=5, pos=np.asarray([1.0, 0.0, 0.0], dtype=np.float32), vel=np.zeros(3), radius=0.5, msg_age_sec=0.1, valid=True),
            NeighborObs(idx=2, pos=np.asarray([-1.0, 0.0, 0.0], dtype=np.float32), vel=np.zeros(3), radius=0.5, msg_age_sec=0.1, valid=True),
            NeighborObs(idx=3, pos=np.asarray([0.0, 0.0, 1.0], dtype=np.float32), vel=np.zeros(3), radius=0.5, msg_age_sec=0.1, valid=True),
        ]
        selected = select_neighbors(
            ego_idx=0,
            ego_pos=ego_pos,
            ego_vel=ego_vel,
            obs=obs,
            range_m=10.0,
            top_k=3,
            threat_metric="distance",
            ttc_horizon_s=6.0,
        )
        self.assertEqual([n.idx for n in selected], [2, 3, 5])


if __name__ == "__main__":
    unittest.main()
