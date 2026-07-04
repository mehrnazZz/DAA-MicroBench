from __future__ import annotations

import ast
from pathlib import Path


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
