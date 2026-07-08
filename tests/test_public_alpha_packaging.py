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
        "docs/LEARNED_POLICY_ADOPTION.md",
        "docs/LEARNED_SUBMISSION_SCHEMAS.md",
        "docs/RL_STABLE_V1_FREEZE.md",
    ]
    for rel in required_docs:
        path = ROOT / rel
        assert path.exists(), rel
        assert path.read_text(encoding="utf-8").strip(), rel

    external_spec = ROOT / "examples/external_policy_spec.json"
    assert external_spec.exists()
    assert "tiny_linear_json" in external_spec.read_text(encoding="utf-8")
    model_spec = ROOT / "examples/external_policy_model_predict_spec.json"
    callable_spec = ROOT / "examples/external_policy_callable_spec.json"
    manifest_template = ROOT / "examples/learned_submission_manifest_template.json"
    schema_dir = ROOT / "microbench" / "bundled_config" / "schemas"
    assert model_spec.exists()
    assert callable_spec.exists()
    assert manifest_template.exists()
    assert (schema_dir / "learned_submission_manifest.schema.json").exists()
    assert (schema_dir / "learned_submission_bundle.schema.json").exists()
    assert (schema_dir / "learned_bundle_review.schema.json").exists()
    assert "model_predict" in model_spec.read_text(encoding="utf-8")
    assert "callable" in callable_spec.read_text(encoding="utf-8")
    assert "inference_packages" in manifest_template.read_text(encoding="utf-8")
    assert "bundled_config/schemas/*.json" in (ROOT / "pyproject.toml").read_text(encoding="utf-8")

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
    assert "learned_submission_manifest.json" in learned_template
    assert "learned_bundle_review.json" in learned_template
    assert "policy_spec.json" in learned_template
    assert "LEARNED_POLICY_ADOPTION.md" in learned_template
    assert "--submission-manifest" in learned_template
    assert "validate-learned-manifest" in learned_template
    assert "rl_contract.json" in learned_template
    assert "rl_freeze_check.json" in learned_template
    assert "rl_smoke.json" in learned_template
    assert "rl_calibration.json" in learned_template
    assert "validate-learned-bundle --bundle" in learned_template
    assert "review-learned-bundle --bundle" in learned_template
    assert "Training Disclosure" in learned_template
