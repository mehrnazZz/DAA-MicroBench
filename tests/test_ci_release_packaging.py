from __future__ import annotations

from pathlib import Path

from microbench.config import builtin_scenario_paths, resolve_config_path


ROOT = Path(__file__).resolve().parents[1]


def test_github_actions_ci_workflow_contract() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "actions/setup-python@v5" in workflow
    assert 'python-version: ["3.10", "3.11", "3.12"]' in workflow
    assert "python -m pytest -q" in workflow
    assert "bash scripts/ci_sanity.sh" in workflow
    assert "bash scripts/package_smoke.sh" in workflow


def test_release_checklist_and_docs_index_cover_public_alpha_checks() -> None:
    checklist = (ROOT / "docs/RELEASE_CHECKLIST.md").read_text(encoding="utf-8")
    docs_index = (ROOT / "docs/README.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "python -m pytest -q" in checklist
    assert "bash scripts/ci_sanity.sh" in checklist
    assert "bash scripts/package_smoke.sh" in checklist
    assert "GitHub Actions CI is green for Python 3.10, 3.11, and 3.12" in checklist
    assert "installed-wheel smoke cannot run from outside the source checkout" in checklist
    assert "RELEASE_CHECKLIST.md" in docs_index
    assert "package_smoke.sh" in readme


def test_bundled_config_assets_match_source_tree() -> None:
    source_files = sorted((ROOT / "config").glob("*.yaml"))
    source_files.extend(sorted((ROOT / "config/scenarios").glob("*.yaml")))

    assert source_files
    for source in source_files:
        rel = source.relative_to(ROOT / "config")
        bundled = ROOT / "microbench/bundled_config" / rel
        assert bundled.exists(), rel
        assert bundled.read_bytes() == source.read_bytes(), rel


def test_builtin_scenario_resolver_uses_existing_assets() -> None:
    resolved = Path(resolve_config_path("config/scenarios/corridor.yaml"))
    assert resolved.exists()
    assert resolved.name == "corridor.yaml"

    builtins = [Path(p).name for p in builtin_scenario_paths()]
    assert "corridor.yaml" in builtins
    assert "stacked_swap_3d.yaml" in builtins
