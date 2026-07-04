from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from microbench.planners import list_methods, make_planner
from microbench.types import AgentState, PlannerInput, PlannerOutput


PLANNER_DIR = Path(__file__).resolve().parents[1] / "microbench" / "planners"

FORBIDDEN_MICROBENCH_IMPORTS = (
    "microbench.acceptance",
    "microbench.cli",
    "microbench.core",
    "microbench.dataset",
    "microbench.logging",
    "microbench.metrics",
    "microbench.replay",
    "microbench.runner",
    "microbench.scenarios",
    "microbench.tools",
)


def _import_modules(tree: ast.AST) -> list[str]:
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def test_builtin_planners_do_not_import_privileged_simulator_internals() -> None:
    """Planner modules should depend on public input/types, not engine truth."""

    for path in sorted(PLANNER_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        modules = _import_modules(tree)
        for module in modules:
            assert not module.startswith(FORBIDDEN_MICROBENCH_IMPORTS), f"{path.name} imports {module}"


def test_builtin_planners_accept_public_planner_input_and_return_finite_vec3() -> None:
    ego = AgentState(
        idx=0,
        pos=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        vel=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        goal=np.asarray([10.0, 0.0, 0.0], dtype=np.float32),
        radius=0.5,
        v_max=3.0,
        a_max=2.0,
    )
    planner_input = PlannerInput(
        ego=ego,
        goal_dir=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        neighbors=[],
        dt=0.02,
        t=0.0,
        planar=True,
    )

    for method in list_methods():
        planner = make_planner(method)
        planner.reset(seed=123)
        out = planner.compute_cmd(planner_input)
        v_cmd = out.v_cmd if isinstance(out, PlannerOutput) else out
        v_cmd = np.asarray(v_cmd, dtype=float)

        assert v_cmd.shape == (3,), method
        assert np.all(np.isfinite(v_cmd)), method
        assert float(np.linalg.norm(v_cmd)) <= ego.v_max + 1e-6, method
