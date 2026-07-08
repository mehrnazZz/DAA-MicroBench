from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from microbench.rl.submission_bundle import (
    LEARNED_SUBMISSION_BUNDLE_REVIEW_SCHEMA_VERSION,
    LEARNED_SUBMISSION_BUNDLE_SCHEMA_VERSION,
    LEARNED_SUBMISSION_MANIFEST_SCHEMA_VERSION,
    validate_learned_submission_manifest,
)
from microbench.rl.submission_schemas import (
    LEARNED_BUNDLE_REVIEW_SCHEMA_FILE,
    LEARNED_SUBMISSION_BUNDLE_SCHEMA_FILE,
    LEARNED_SUBMISSION_MANIFEST_SCHEMA_FILE,
    load_submission_schema,
    validate_with_schema_subset,
)


LEARNED_SUBMISSION_SCHEMA_CHECK_VERSION = "0.1"
_SCHEMA_FILES = (
    LEARNED_SUBMISSION_MANIFEST_SCHEMA_FILE,
    LEARNED_SUBMISSION_BUNDLE_SCHEMA_FILE,
    LEARNED_BUNDLE_REVIEW_SCHEMA_FILE,
)


def _check(name: str, ok: bool, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "details": details or {}}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def run_learned_submission_schema_check(*, root: str | Path = ".") -> dict[str, Any]:
    """Check that learned-submission schemas are ready for public/stable review."""

    repo = Path(root)
    schemas: dict[str, dict[str, Any]] = {}
    schema_load_errors: dict[str, str] = {}
    for filename in _SCHEMA_FILES:
        try:
            schemas[filename] = load_submission_schema(filename)
        except Exception as exc:
            schema_load_errors[filename] = f"{type(exc).__name__}: {exc}"

    manifest_schema = schemas.get(LEARNED_SUBMISSION_MANIFEST_SCHEMA_FILE, {})
    bundle_schema = schemas.get(LEARNED_SUBMISSION_BUNDLE_SCHEMA_FILE, {})
    review_schema = schemas.get(LEARNED_BUNDLE_REVIEW_SCHEMA_FILE, {})

    template_path = repo / "examples" / "learned_submission_manifest_template.json"
    overlay_path = repo / "examples" / "learned_submission_manifest_overlay_example.json"
    docs_path = repo / "docs" / "LEARNED_SUBMISSION_SCHEMAS.md"
    adoption_path = repo / "docs" / "LEARNED_POLICY_ADOPTION.md"
    result_submission_path = repo / "docs" / "RESULT_SUBMISSION.md"
    release_checklist_path = repo / "docs" / "RELEASE_CHECKLIST.md"
    release_readiness_path = repo / "scripts" / "release_readiness.sh"
    pyproject_path = repo / "pyproject.toml"

    try:
        manifest_template = _read_json(template_path)
        template_load_error = None
    except Exception as exc:
        manifest_template = {}
        template_load_error = f"{type(exc).__name__}: {exc}"
    template_schema_errors = validate_with_schema_subset(manifest_template, manifest_schema) if manifest_schema else []
    template_validator_report = (
        validate_learned_submission_manifest(manifest=template_path)
        if template_load_error is None
        else {"ok": False, "checks": []}
    )

    try:
        overlay_payload = _read_json(overlay_path)
        overlay_load_error = None
    except Exception as exc:
        overlay_payload = {}
        overlay_load_error = f"{type(exc).__name__}: {exc}"

    docs_text = _read_text(docs_path)
    adoption_text = _read_text(adoption_path)
    result_submission_text = _read_text(result_submission_path)
    release_checklist_text = _read_text(release_checklist_path)
    release_readiness_text = _read_text(release_readiness_path)
    pyproject_text = _read_text(pyproject_path)

    version_details = {
        "manifest_schema_const": (manifest_schema.get("properties") or {}).get("schema_version", {}).get("const"),
        "manifest_bundle_schema_const": (manifest_schema.get("properties") or {}).get("bundle_schema_version", {}).get("const"),
        "bundle_schema_const": (bundle_schema.get("properties") or {}).get("schema_version", {}).get("const"),
        "review_schema_const": (review_schema.get("properties") or {}).get("schema_version", {}).get("const"),
    }
    top_level_forward_compatible = all(
        schema.get("additionalProperties") is True
        for schema in (manifest_schema, bundle_schema, review_schema)
        if schema
    )

    checks = [
        _check(
            "learned_submission_schema_files_load",
            len(schemas) == len(_SCHEMA_FILES) and not schema_load_errors,
            {"schema_files": list(_SCHEMA_FILES), "errors": schema_load_errors},
        ),
        _check(
            "learned_submission_schema_versions_match_constants",
            version_details["manifest_schema_const"] == LEARNED_SUBMISSION_MANIFEST_SCHEMA_VERSION
            and version_details["manifest_bundle_schema_const"] == LEARNED_SUBMISSION_BUNDLE_SCHEMA_VERSION
            and version_details["bundle_schema_const"] == LEARNED_SUBMISSION_BUNDLE_SCHEMA_VERSION
            and version_details["review_schema_const"] == LEARNED_SUBMISSION_BUNDLE_REVIEW_SCHEMA_VERSION,
            version_details,
        ),
        _check(
            "learned_submission_schemas_forward_compatible",
            top_level_forward_compatible,
            {"top_level_additional_properties": top_level_forward_compatible},
        ),
        _check(
            "learned_submission_manifest_template_valid",
            template_load_error is None and not template_schema_errors and bool(template_validator_report.get("ok")),
            {
                "path": str(template_path),
                "load_error": template_load_error,
                "schema_errors": template_schema_errors[:20],
                "validator_failed": [
                    check["name"]
                    for check in template_validator_report.get("checks", [])
                    if not check.get("ok")
                ],
            },
        ),
        _check(
            "learned_submission_overlay_example_is_overlay",
            overlay_load_error is None
            and bool(overlay_payload)
            and "schema_version" not in overlay_payload
            and "benchmark" not in overlay_payload
            and "artifacts" not in overlay_payload
            and "training_disclosure" in overlay_payload,
            {"path": str(overlay_path), "load_error": overlay_load_error, "keys": sorted(overlay_payload)},
        ),
        _check(
            "learned_submission_schema_docs_present",
            docs_path.exists()
            and "Full Manifest Vs Overlay" in docs_text
            and "Compatibility Policy" in docs_text
            and LEARNED_SUBMISSION_MANIFEST_SCHEMA_FILE in docs_text
            and LEARNED_SUBMISSION_BUNDLE_SCHEMA_FILE in docs_text
            and LEARNED_BUNDLE_REVIEW_SCHEMA_FILE in docs_text,
            {"path": str(docs_path)},
        ),
        _check(
            "learned_submission_user_docs_link_schema_policy",
            "LEARNED_SUBMISSION_SCHEMAS.md" in adoption_text
            and "LEARNED_SUBMISSION_SCHEMAS.md" in result_submission_text
            and "learned_submission_manifest_overlay_example.json" in adoption_text,
            {
                "adoption_doc": str(adoption_path),
                "result_submission_doc": str(result_submission_path),
            },
        ),
        _check(
            "learned_submission_release_docs_include_schema_gate",
            "LEARNED_SUBMISSION_SCHEMAS.md" in release_checklist_text
            and "learned-submission-schema-check" in release_checklist_text
            and "learned-submission-schema-check" in release_readiness_text,
            {
                "release_checklist": str(release_checklist_path),
                "release_readiness": str(release_readiness_path),
            },
        ),
        _check(
            "learned_submission_schemas_packaged",
            "bundled_config/schemas/*.json" in pyproject_text,
            {"pyproject": str(pyproject_path)},
        ),
    ]

    return {
        "schema_version": LEARNED_SUBMISSION_SCHEMA_CHECK_VERSION,
        "ok": all(check["ok"] for check in checks),
        "root": str(repo),
        "schema_files": list(_SCHEMA_FILES),
        "schema_versions": version_details,
        "checks": checks,
    }
