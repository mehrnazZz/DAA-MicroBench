from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_public_alpha_metadata_and_entrypoints() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'name = "daa-microbench"' in text
    assert 'readme = "README.md"' in text
    assert "agentic multi-drone detect-and-avoid" in text
    assert 'daa-microbench = "microbench.cli:main"' in text
    assert 'microbench = "microbench.cli:main"' in text
    assert "Apache-2.0" in text


def test_module_cli_smoke_for_public_alpha() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "microbench.cli", "list-suites", "--json"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    suites = json.loads(proc.stdout)
    by_name = {entry["suite"]: entry for entry in suites}
    assert "official_smoke_generated" in by_name
    assert "official_3d_stress" in by_name
    assert "official_agentic_stress" in by_name
    assert "official_promotion_calibration" in by_name


def test_public_docs_index_and_templates_exist() -> None:
    required_docs = [
        "docs/README.md",
        "docs/DESIGN_V1.md",
        "docs/PLANNER_API.md",
        "docs/SCENARIO_SUITES.md",
        "docs/BASELINES.md",
        "docs/LEADERBOARD.md",
        "docs/RESULT_SUBMISSION.md",
        "docs/RL_STABLE_V1_FREEZE.md",
    ]
    for rel in required_docs:
        path = ROOT / rel
        assert path.exists(), rel
        assert path.read_text(encoding="utf-8").strip(), rel

    required_templates = [
        ".github/ISSUE_TEMPLATE/benchmark_result.md",
        ".github/ISSUE_TEMPLATE/bug_report.md",
        ".github/ISSUE_TEMPLATE/feature_request.md",
        ".github/ISSUE_TEMPLATE/learned_policy_submission.md",
        ".github/ISSUE_TEMPLATE/planner_submission.md",
        ".github/ISSUE_TEMPLATE/scenario_proposal.md",
        ".github/pull_request_template.md",
    ]
    for rel in required_templates:
        path = ROOT / rel
        assert path.exists(), rel
        assert path.read_text(encoding="utf-8").strip(), rel

    result_template = (ROOT / ".github/ISSUE_TEMPLATE/benchmark_result.md").read_text(encoding="utf-8")
    assert "result_schema.json" in result_template
    assert "planner_timeout_count_mean" in result_template
    assert "docs/RESULT_SUBMISSION.md" in result_template

    learned_template = (ROOT / ".github/ISSUE_TEMPLATE/learned_policy_submission.md").read_text(encoding="utf-8")
    assert "learned_submission_bundle.json" in learned_template
    assert "rl_contract.json" in learned_template
    assert "rl_freeze_check.json" in learned_template
    assert "rl_smoke.json" in learned_template
    assert "rl_calibration.json" in learned_template
    assert "Training Disclosure" in learned_template
