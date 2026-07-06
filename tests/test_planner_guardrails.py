from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from microbench.core import EpisodeEngine
from microbench.metrics import append_result, write_summary
from microbench.runner import run_episode
from microbench.types import PlannerOutput, RunSpec


def _write_guardrail_scenario(path: Path, *, planar: bool = True, timeout_ms: float | None = None) -> None:
    timeout_block = ""
    if timeout_ms is not None:
        timeout_block = f"""
planner_guardrails:
  timeout_ms: {timeout_ms}
  fallback_speed_scale: 0.5
"""
    world_block = (
        """
world:
  planar: true
  fixed_y_m: 0.0
"""
        if planar
        else """
world:
  planar: false
  bounds:
    x: [-5.0, 5.0]
    y: [-2.0, 2.0]
    z: [-5.0, 5.0]
"""
    )
    layers_block = (
        ""
        if planar
        else """
  start_layers_m: [-1.0, 1.0]
  goal_layers_m: [1.0, -1.0]
"""
    )
    path.write_text(
        f"""
scenario:
  name: "guardrail_smoke"
  duration_s: 0.08
sim:
  dt_s: 0.02
{world_block}
agent_params:
  radius_m: 0.25
  v_max_mps: 1.0
  a_max_mps2: 2.0
  goal_tolerance_m: 0.1
neighbors:
  range_m: 30.0
  top_k: 1
goals:
  min_goal_distance_m: 1.0
spawn:
  type: "circle_swap"
  center: [0.0, 0.0, 0.0]
  radius_m: 1.0
  jitter_m: 0.0
{layers_block}
logging:
  save_events: false
  save_trace: false
{timeout_block}
""".strip(),
        encoding="utf-8",
    )


class _InvalidShapePlanner:
    def reset(self, agent_id: int, seed: int, config: dict) -> None:
        self.agent_id = int(agent_id)

    def compute_cmd(self, planner_input):
        return PlannerOutput(
            v_cmd=np.asarray([1.0, 2.0], dtype=float),
            debug_info={"agent_id": int(self.agent_id)},
        )


class _InvalidObjectPlanner:
    def compute_cmd(self, planner_input):
        return object()


class _FiniteFastPlanner:
    def reset(self, seed: int) -> None:
        self.seed = int(seed)

    def compute_cmd(self, planner_input):
        return np.asarray(planner_input.goal_dir, dtype=float) * float(planner_input.ego.v_max)


class _AuditPlanner:
    seen: list[dict] = []

    def __init__(self, method_name: str):
        self.method_name = method_name

    def reset(self, agent_id: int, seed: int, config: dict) -> None:
        self.agent_id = int(agent_id)
        self.reset_config = dict(config)

    def compute_cmd(self, planner_input):
        forbidden = ("states", "all_states", "truth", "engine", "rng", "v2v", "cfg")
        for attr in forbidden:
            assert not hasattr(planner_input, attr), f"PlannerInput leaked privileged attr: {attr}"
        assert planner_input.agent_context is not None
        assert planner_input.agent_context.agent_id == self.agent_id
        assert planner_input.agent_context.method == self.method_name
        assert "capabilities" in self.reset_config
        self.seen.append(
            {
                "agent_id": int(self.agent_id),
                "method": self.method_name,
                "planar": bool(planner_input.planar),
                "neighbor_count": len(planner_input.neighbors),
                "memory_type": type(planner_input.agent_context.memory).__name__,
            }
        )
        return np.asarray(planner_input.goal_dir, dtype=float) * 0.1


@pytest.mark.parametrize("planar", [True, False])
def test_invalid_planner_output_uses_finite_fallback_in_2d_and_3d(tmp_path: Path, planar: bool) -> None:
    scenario = tmp_path / "guardrail.yaml"
    _write_guardrail_scenario(scenario, planar=planar)

    engine = EpisodeEngine(
        scenario_path=str(scenario),
        method="invalid_shape",
        n_agents=2,
        seed=0,
        comm_profile="ideal_50hz",
        planner_factory=lambda _: _InvalidShapePlanner(),
    )
    step = engine.step()
    engine.close()

    assert step is not None
    assert engine.planner_error_count == 2
    assert engine.planner_fallback_count == 2
    assert engine.planner_timeout_count == 0
    for cmd in step.v_cmds:
        assert cmd.shape == (3,)
        assert np.all(np.isfinite(cmd))
        assert float(np.linalg.norm(cmd)) <= 0.5 + 1e-9
        if planar:
            assert cmd[1] == pytest.approx(0.0)
    assert {dbg["engine_guardrail"] for dbg in step.planner_debug} == {"invalid_output"}
    assert {dbg["error_type"] for dbg in step.planner_debug} == {"ValueError"}


def test_uncoercible_planner_output_is_counted_as_error(tmp_path: Path) -> None:
    scenario = tmp_path / "guardrail.yaml"
    _write_guardrail_scenario(scenario, planar=True)

    engine = EpisodeEngine(
        scenario_path=str(scenario),
        method="invalid_object",
        n_agents=2,
        seed=0,
        comm_profile="ideal_50hz",
        planner_factory=lambda _: _InvalidObjectPlanner(),
    )
    step = engine.step()
    engine.close()

    assert step is not None
    assert engine.planner_error_count == 2
    assert engine.planner_fallback_count == 2
    assert {dbg["engine_guardrail"] for dbg in step.planner_debug} == {"invalid_output"}
    assert {dbg["error_type"] for dbg in step.planner_debug} == {"TypeError"}


def test_runner_writes_timeout_guardrail_metrics_to_summary(tmp_path: Path) -> None:
    scenario = tmp_path / "timeout_guardrail.yaml"
    _write_guardrail_scenario(scenario, planar=True, timeout_ms=0.0)
    out_dir = tmp_path / "runs"

    with patch("microbench.runner.make_planner", side_effect=lambda _: _FiniteFastPlanner()):
        row = run_episode(
            RunSpec(
                scenario_path=str(scenario),
                method="timeout_smoke",
                n_agents=2,
                seed=0,
                comm_profile="ideal_50hz",
                out_dir=str(out_dir),
                save_trace=False,
            )
        )

    append_result(out_dir, row)
    write_summary(out_dir)

    assert int(row["planner_timeout_count"]) > 0
    assert int(row["planner_error_count"]) == 0
    assert int(row["planner_fallback_count"]) == int(row["planner_timeout_count"])

    with (out_dir / "summary.csv").open("r", newline="", encoding="utf-8") as f:
        summary_rows = list(csv.DictReader(f))
    assert len(summary_rows) == 1
    summary = summary_rows[0]
    assert float(summary["planner_timeout_count_mean"]) == float(row["planner_timeout_count"])
    assert float(summary["planner_error_count_mean"]) == 0.0
    assert float(summary["planner_fallback_count_mean"]) == float(row["planner_fallback_count"])


def test_external_style_heterogeneous_planners_receive_public_input_only(tmp_path: Path) -> None:
    _AuditPlanner.seen = []
    scenario = tmp_path / "audit_3d.yaml"
    _write_guardrail_scenario(scenario, planar=False)

    def factory(name: str):
        return _AuditPlanner(name)

    engine = EpisodeEngine(
        scenario_path=str(scenario),
        method="audit",
        n_agents=2,
        seed=3,
        comm_profile="ideal_50hz",
        agent_methods=["audit_a", "audit_b"],
        planner_factory=factory,
    )
    step = engine.step()
    engine.close()

    assert step is not None
    assert engine.method_label == "mixed[audit_a+audit_b]"
    assert {entry["method"] for entry in _AuditPlanner.seen} == {"audit_a", "audit_b"}
    assert {entry["agent_id"] for entry in _AuditPlanner.seen} == {0, 1}
    assert all(entry["planar"] is False for entry in _AuditPlanner.seen)
    assert all(entry["memory_type"] == "AgentMemory" for entry in _AuditPlanner.seen)
