from __future__ import annotations

from pathlib import Path
import json
import tempfile
from unittest.mock import patch

import numpy as np

from microbench.core import EpisodeEngine
from microbench.dataset.generate import generate_dataset
from microbench.runner import run_episode
from microbench.types import PlannerOutput, RunSpec


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


class _LifecyclePlanner:
    reset_records: list[dict] = []
    finalize_records: list[dict] = []
    memory_ticks: dict[int, int] = {}

    def reset(self, agent_id: int, seed: int, config: dict) -> None:
        self.agent_id = int(agent_id)
        _LifecyclePlanner.reset_records.append(
            {
                "agent_id": int(agent_id),
                "seed": int(seed),
                "role": config.get("role"),
                "priority": int(config.get("priority", -1)),
                "capabilities": dict(config.get("capabilities", {})),
            }
        )

    def compute_cmd(self, planner_input):
        ctx = planner_input.agent_context
        assert ctx is not None
        ctx.memory["ticks"] = int(ctx.memory.get("ticks", 0)) + 1
        _LifecyclePlanner.memory_ticks[int(ctx.agent_id)] = int(ctx.memory["ticks"])
        return PlannerOutput(
            v_cmd=np.asarray(planner_input.goal_dir, dtype=np.float32) * 0.1,
            debug_info={
                "agent_id": int(ctx.agent_id),
                "ticks": int(ctx.memory["ticks"]),
                "priority": int(ctx.priority),
                "role": ctx.role,
            },
        )

    def finalize(self, agent_context, config: dict) -> None:
        _LifecyclePlanner.finalize_records.append(
            {
                "agent_id": int(agent_context.agent_id),
                "ticks": int(agent_context.memory.get("ticks", 0)),
                "role": config.get("role"),
            }
        )


class _CrashPlanner:
    def reset(self, seed: int) -> None:
        _ = seed

    def compute_cmd(self, planner_input):
        raise RuntimeError("planner exploded")


def test_engine_guardrails_fallback_on_planner_error():
    with tempfile.TemporaryDirectory() as td:
        scenario = Path(td) / "scenario_engine.yaml"
        _write_short_scenario(scenario)

        engine = EpisodeEngine(
            scenario_path=str(scenario),
            method="crash",
            n_agents=2,
            seed=0,
            comm_profile="ideal_50hz",
            planner_factory=lambda _: _CrashPlanner(),
        )
        step = engine.step()

        assert step is not None
        assert engine.planner_error_count == 2
        assert engine.planner_fallback_count == 2
        assert step.planner_debug[0]["engine_guardrail"] == "error"
        assert step.planner_debug[0]["error_type"] == "RuntimeError"


def test_engine_guardrails_soft_timeout_discards_slow_output():
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
        engine.planner_timeout_ms = 0.0
        step = engine.step()

        assert step is not None
        assert engine.planner_timeout_count == 2
        assert engine.planner_fallback_count == 2
        assert step.planner_debug[0]["engine_guardrail"] == "timeout"
        assert step.planner_debug[0]["planner_timeout_ms"] == 0.0


def test_runner_reports_planner_guardrail_counts():
    with tempfile.TemporaryDirectory() as td:
        scenario = Path(td) / "scenario_engine.yaml"
        _write_short_scenario(scenario)

        with patch("microbench.runner.make_planner", side_effect=lambda _: _CrashPlanner()):
            row = run_episode(
                RunSpec(
                    scenario_path=str(scenario),
                    method="crash",
                    n_agents=2,
                    seed=0,
                    comm_profile="ideal_50hz",
                    out_dir=str(Path(td) / "runs"),
                    save_trace=False,
                )
            )

        assert int(row["planner_error_count"]) > 0
        assert int(row["planner_fallback_count"]) == int(row["planner_error_count"])
        assert int(row["planner_timeout_count"]) == 0


def test_agent_profiles_lifecycle_and_memory_are_per_agent():
    _LifecyclePlanner.reset_records = []
    _LifecyclePlanner.finalize_records = []
    _LifecyclePlanner.memory_ticks = {}
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        scenario = root / "scenario_agent_profile.yaml"
        _write_short_scenario(scenario)
        text = scenario.read_text(encoding="utf-8")
        scenario.write_text(
            text
            + """
agents:
  defaults:
    capabilities:
      v_max_mps: 1.25
    mission:
      class: nominal
  by_id:
    0:
      role: lead
      priority: 7
      capabilities:
        v_max_mps: 0.5
    1:
      role: wing
      priority: 3
""",
            encoding="utf-8",
        )

        with patch("microbench.runner.make_planner", side_effect=lambda _: _LifecyclePlanner()):
            run_episode(
                RunSpec(
                    scenario_path=str(scenario),
                    method="lifecycle_test",
                    n_agents=2,
                    seed=5,
                    comm_profile="ideal_50hz",
                    out_dir=str(root / "runs"),
                    save_trace=False,
                )
            )

    assert len(_LifecyclePlanner.reset_records) == 2
    by_agent = {r["agent_id"]: r for r in _LifecyclePlanner.reset_records}
    assert by_agent[0]["role"] == "lead"
    assert by_agent[0]["priority"] == 7
    assert by_agent[0]["capabilities"]["v_max_mps"] == 0.5
    assert by_agent[1]["role"] == "wing"
    assert by_agent[1]["capabilities"]["v_max_mps"] == 1.25
    assert set(_LifecyclePlanner.memory_ticks) == {0, 1}
    assert {r["agent_id"] for r in _LifecyclePlanner.finalize_records} == {0, 1}
    assert all(r["ticks"] > 0 for r in _LifecyclePlanner.finalize_records)


def test_agent_failures_and_planner_debug_are_traced():
    _LifecyclePlanner.reset_records = []
    _LifecyclePlanner.finalize_records = []
    _LifecyclePlanner.memory_ticks = {}
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        scenario = root / "scenario_agent_failure.yaml"
        _write_short_scenario(scenario)
        text = scenario.read_text(encoding="utf-8")
        scenario.write_text(
            text
            + """
agents:
  by_id:
    0:
      failure_modes:
        frozen:
          enabled: true
          start_s: 0.0
          duration_s: 0.12
logging:
  save_trace: true
  trace_save_failures_only: false
  trace_max_steps: 20
""",
            encoding="utf-8",
        )

        with patch("microbench.runner.make_planner", side_effect=lambda _: _LifecyclePlanner()):
            run_episode(
                RunSpec(
                    scenario_path=str(scenario),
                    method="debug_test",
                    n_agents=2,
                    seed=0,
                    comm_profile="ideal_50hz",
                    out_dir=str(root / "runs"),
                    save_trace=True,
                )
            )

        trace_path = root / "runs" / "episodes" / "scenario_agent_failure_debug_test_n2_seed0_comm_ideal_50hz" / "trace_episode.jsonl"
        lines = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
        frame = next(row for row in lines if row.get("kind") == "frame")

    assert "frozen" in frame["agent_failures"][0]
    assert np.allclose(frame["v_cmd"][0], [0.0, 0.0, 0.0])
    assert frame["planner_debug"][1]["agent_id"] == 1
    assert frame["planner_debug"][1]["ticks"] >= 1


def test_sensor_dropout_filters_local_perception():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        scenario = root / "scenario_sensor_dropout.yaml"
        _write_short_scenario(scenario)
        text = scenario.read_text(encoding="utf-8")
        scenario.write_text(
            text
            + """
perception:
  mode: sensor
  sensor:
    range_m: 30.0
    fov_deg: 360.0
agents:
  by_id:
    0:
      failure_modes:
        sensor_dropout: true
""",
            encoding="utf-8",
        )

        engine = EpisodeEngine(
            scenario_path=str(scenario),
            method="baseline_goal",
            n_agents=2,
            seed=0,
            comm_profile="ideal_50hz",
        )
        step = engine.step()
        engine.close()

    assert step is not None
    assert "sensor_dropout" in step.agent_failures[0]
    assert step.selected_neighbor_obs[0] == []
    assert len(step.selected_neighbor_obs[1]) >= 1
