from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np

from microbench.core import EpisodeEngine
from microbench.dataset.generate import generate_dataset


def _write_short_scenario(path: Path) -> None:
    path.write_text(
        """
scenario:
  name: "engine_smoke"
  duration_s: 0.12
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


def test_episode_engine_step_exposes_shared_trace_fields():
    with tempfile.TemporaryDirectory() as td:
        scenario = Path(td) / "scenario_engine.yaml"
        _write_short_scenario(scenario)

        engine = EpisodeEngine(
            scenario_path=str(scenario),
            method="baseline_goal",
            n_agents=2,
            seed=0,
            comm_profile="ideal_50hz",
        )
        step = engine.step()

        assert step is not None
        assert step.pos.shape == (2, 3)
        assert step.vel.shape == (2, 3)
        assert len(step.selected_neighbor_obs) == 2
        frame = step.trace_frame()
        assert frame["n_agents"] == 2
        assert len(frame["positions"]) == 2
        assert "selected_obs" in frame


def test_dataset_generation_uses_engine_path():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        scenario = root / "scenario_engine.yaml"
        _write_short_scenario(scenario)

        shards = generate_dataset(
            scenarios=[str(scenario)],
            method="baseline_goal",
            n_agents_list=[2],
            seeds=[0],
            T=2,
            dt_plan_s=0.04,
            out_dir=str(root / "dataset"),
            comm_profiles=["ideal_50hz"],
            shard_size=100,
        )

        assert len(shards) == 1
        payload = np.load(shards[0])
        assert payload["U0_raw"].shape[1:] == (2, 3)
        assert payload["cond_ego"].shape[1:] == (6,)
        assert payload["cond_nbh"].shape[2] == 9

