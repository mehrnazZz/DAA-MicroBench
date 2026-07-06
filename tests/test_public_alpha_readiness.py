from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_has_public_status_badges_and_readiness_links() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "actions/workflows/ci.yml/badge.svg" in readme
    assert "license-Apache--2.0" in readme
    assert "scripts/release_readiness.sh" in readme
    assert "docs/PUBLIC_ALPHA_NOTES.md" in readme


def test_public_alpha_notes_cover_release_contract() -> None:
    notes = (ROOT / "docs/PUBLIC_ALPHA_NOTES.md").read_text(encoding="utf-8")

    assert "Status date: 2026-07-06" in notes
    assert "Python 3.10, 3.11, and 3.12" in notes
    assert "Result schema version: `0.4.0`" in notes
    assert "official_smoke_generated" in notes
    assert "official_3d_stress" in notes
    assert "official_agentic_stress" in notes
    assert "bash scripts/release_readiness.sh" in notes
    assert "not yet a stable v1 release" in notes


def test_release_readiness_script_runs_expected_checks() -> None:
    script_path = ROOT / "scripts/release_readiness.sh"
    script = script_path.read_text(encoding="utf-8")

    assert script_path.exists()
    assert "python -m pytest -q" in script
    assert "bash scripts/ci_sanity.sh" in script
    assert "bash scripts/package_smoke.sh" in script
    assert "golden-current-schema" in script
    assert "validate-scenarios --all-builtins --all-generated-suites --quiet" in script
    assert "DAA_REQUIRE_CLEAN" in script


def test_docs_index_and_checklist_reference_alpha_notes_and_dry_run() -> None:
    docs_index = (ROOT / "docs/README.md").read_text(encoding="utf-8")
    checklist = (ROOT / "docs/RELEASE_CHECKLIST.md").read_text(encoding="utf-8")

    assert "PUBLIC_ALPHA_NOTES.md" in docs_index
    assert "bash scripts/release_readiness.sh" in docs_index
    assert "bash scripts/release_readiness.sh" in checklist
    assert "DAA_REQUIRE_CLEAN=1 bash scripts/release_readiness.sh" in checklist
