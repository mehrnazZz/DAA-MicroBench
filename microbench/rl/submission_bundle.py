from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys
from typing import Any

from microbench.acceptance import check_acceptance
from microbench.metrics import append_result, write_summary
from microbench.planners import planner_metadata
from microbench.rl.calibration import run_rl_policy_calibration
from microbench.rl.evaluate import run_rl_policy_smoke
from microbench.rl.freeze import run_rl_freeze_check
from microbench.rl.policy_spec import portable_policy_spec_payload, resolve_policy_artifact_path
from microbench.rl.schema import interface_contract
from microbench.rl.submission_schemas import (
    LEARNED_BUNDLE_REVIEW_SCHEMA_FILE,
    LEARNED_SUBMISSION_BUNDLE_SCHEMA_FILE,
    LEARNED_SUBMISSION_MANIFEST_SCHEMA_FILE,
    load_submission_schema,
    validate_with_schema_subset,
)
from microbench.runner import run_episode
from microbench.scenarios import materialize_official_suite, suite_defaults
from microbench.types import RunSpec


LEARNED_SUBMISSION_BUNDLE_SCHEMA_VERSION = "0.1"
LEARNED_SUBMISSION_BUNDLE_VALIDATION_SCHEMA_VERSION = "0.1"
LEARNED_SUBMISSION_BUNDLE_REVIEW_SCHEMA_VERSION = "0.1"
LEARNED_SUBMISSION_MANIFEST_SCHEMA_VERSION = "0.1"
LEARNED_SUBMISSION_MANIFEST_VALIDATION_SCHEMA_VERSION = "0.1"
LEARNED_SUBMISSION_BUNDLE_FILENAME = "learned_submission_bundle.json"
LEARNED_SUBMISSION_MANIFEST_FILENAME = "learned_submission_manifest.json"
EXPECTED_BUNDLE_ARTIFACTS = {
    "learned_submission_manifest": LEARNED_SUBMISSION_MANIFEST_FILENAME,
    "rl_contract": "rl_contract.json",
    "rl_freeze_check": "rl_freeze_check.json",
    "rl_smoke": "rl_smoke.json",
    "rl_smoke_episodes": "rl_smoke/rl_smoke_episodes.csv",
    "rl_calibration": "rl_calibration.json",
    "rl_calibration_episodes": "rl_calibration/rl_calibration_episodes.csv",
    "planner_results": "planner_sweep/results.csv",
    "planner_summary": "planner_sweep/summary.csv",
    "planner_result_schema": "planner_sweep/result_schema.json",
    "planner_suite_manifest": "planner_sweep/_generated_scenarios/{suite}/suite_manifest.yaml",
    "planner_acceptance": "planner_sweep/acceptance.json",
}
OPTIONAL_BUNDLE_ARTIFACTS = {
    "policy_spec": "policy_spec.json",
    "policy_artifact": "policy_artifacts/{artifact_name}",
}
_DEPENDENCY_VERSION_OPERATORS = ("==", ">=", "<=", "~=", "!=", ">", "<")
_DEPENDENCY_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*(\[[A-Za-z0-9_,.-]+\])?$")


def _schema_errors(payload: Any, schema_file: str) -> list[str]:
    return validate_with_schema_subset(payload, load_submission_schema(schema_file))


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _check(name: str, ok: bool, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "details": details or {}}


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _rel(path: str | Path, root: Path) -> str:
    return str(Path(path).resolve().relative_to(root.resolve()))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _values(rows: list[dict[str, Any]], field: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        value = _to_float(row.get(field))
        if value is not None:
            out.append(value)
    return out


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _sum_values(values: list[float]) -> float | None:
    return sum(values) if values else None


def _round_or_none(value: float | None, ndigits: int = 6) -> float | None:
    return None if value is None else round(float(value), ndigits)


def _method_metadata(method: str) -> dict[str, Any] | None:
    by_method = {entry["method"]: entry for entry in planner_metadata(include_aliases=False)}
    return by_method.get(str(method))


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_manifest_overrides(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = _read_json(Path(path))
    return dict(payload)


def _artifact_record(*, bundle_root: Path, name: str, rel_path: str, required: bool) -> dict[str, Any]:
    path = bundle_root / rel_path
    record: dict[str, Any] = {
        "name": str(name),
        "path": str(rel_path),
        "required": bool(required),
        "kind": path.suffix.lstrip(".") or "file",
        "present": path.exists(),
    }
    if path.exists() and path.is_file():
        record["size_bytes"] = int(path.stat().st_size)
        record["sha256"] = _sha256(path)
    return record


def _default_training_disclosure() -> dict[str, Any]:
    return {
        "training_scenarios": [],
        "training_suites": [],
        "environment_steps": None,
        "random_seeds": [],
        "observation_normalization": "undisclosed",
        "action_post_processing": "normalized velocity action clipped by DAA Microbench action contract",
        "reward_configuration": "undisclosed",
        "external_data": "undisclosed",
        "pretrained_models": "undisclosed",
        "hardware": "undisclosed",
    }


def _build_learned_submission_manifest(
    *,
    bundle_root: Path,
    method: str,
    policy: str,
    suite: str,
    root: str | Path,
    n_agents: int,
    seeds: list[int],
    max_steps: int | None,
    max_runs: int | None,
    planner_sweep: dict[str, Any],
    artifact_paths: dict[str, str],
    policy_spec_summary: dict[str, Any] | None,
    method_metadata: dict[str, Any] | None,
    manifest_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact_records = [
        _artifact_record(
            bundle_root=bundle_root,
            name=name,
            rel_path=rel_path,
            required=name in EXPECTED_BUNDLE_ARTIFACTS,
        )
        for name, rel_path in sorted(artifact_paths.items())
        if name != "learned_submission_manifest"
    ]
    policy_spec_block = None
    if policy_spec_summary is not None:
        policy_spec_block = {
            "policy_name": policy_spec_summary.get("policy_name"),
            "adapter": policy_spec_summary.get("adapter"),
            "spec_path": policy_spec_summary.get("spec_path"),
            "deterministic": bool(policy_spec_summary.get("deterministic", True)),
            "clip": bool(policy_spec_summary.get("clip", True)),
            "artifact_path": policy_spec_summary.get("artifact_path"),
            "callable": policy_spec_summary.get("callable"),
            "factory": policy_spec_summary.get("factory"),
            "factory_kwargs": policy_spec_summary.get("factory_kwargs"),
        }

    manifest = {
        "schema_version": LEARNED_SUBMISSION_MANIFEST_SCHEMA_VERSION,
        "bundle_schema_version": LEARNED_SUBMISSION_BUNDLE_SCHEMA_VERSION,
        "policy": {
            "name": str(policy),
            "method": str(method),
            "method_metadata": method_metadata,
            "policy_spec": policy_spec_block,
        },
        "benchmark": {
            "suite": str(suite),
            "root": str(root),
            "n_agents": int(n_agents),
            "seeds": [int(seed) for seed in seeds],
            "max_steps": None if max_steps is None else int(max_steps),
            "max_runs": None if max_runs is None else int(max_runs),
            "planner_sweep_run_count": int(planner_sweep.get("run_count", 0) or 0),
            "planner_sweep_planned_run_count": int(planner_sweep.get("planned_run_count", 0) or 0),
        },
        "artifacts": artifact_records,
        "dependencies": {
            "python_version": sys.version.split()[0],
            "python_executable": sys.executable,
            "inference_packages": [],
            "notes": "Populate inference_packages through --submission-manifest for non-core dependencies.",
        },
        "training_disclosure": _default_training_disclosure(),
        "inference_disclosure": {
            "deterministic": bool(policy_spec_summary.get("deterministic", True)) if policy_spec_summary else True,
            "uses_external_services": "undisclosed",
            "external_services": [],
            "runtime_notes": "undisclosed",
        },
        "review_notes": {
            "privileged_information": "undisclosed",
            "intended_category": "external_submission" if policy_spec_summary else "built_in_fixture",
        },
    }
    if manifest_overrides:
        manifest = _deep_merge(manifest, manifest_overrides)
        manifest["schema_version"] = LEARNED_SUBMISSION_MANIFEST_SCHEMA_VERSION
        manifest["bundle_schema_version"] = LEARNED_SUBMISSION_BUNDLE_SCHEMA_VERSION
    return manifest


def _manifest_artifact_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    artifacts = manifest.get("artifacts", [])
    if not isinstance(artifacts, list):
        return {}
    return {str(item.get("name")): dict(item) for item in artifacts if isinstance(item, dict) and item.get("name")}


def _manifest_unknown_fields(manifest: dict[str, Any]) -> list[str]:
    unknown: list[str] = []
    for section_name in ("training_disclosure", "inference_disclosure", "review_notes"):
        section = manifest.get(section_name, {})
        if not isinstance(section, dict):
            unknown.append(section_name)
            continue
        for key, value in section.items():
            if value == "undisclosed" or value is None:
                unknown.append(f"{section_name}.{key}")
    return unknown


def _parse_dependency_string(spec: str) -> tuple[str, str | None]:
    cleaned = spec.strip()
    for op in _DEPENDENCY_VERSION_OPERATORS:
        if op in cleaned:
            name, version = cleaned.split(op, 1)
            return name.strip(), f"{op}{version.strip()}"
    return cleaned, None


def _normalize_dependency_entry(entry: Any, *, field: str, index: int) -> tuple[dict[str, Any] | None, list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if isinstance(entry, str):
        name, version = _parse_dependency_string(entry)
        normalized: dict[str, Any] = {
            "name": name,
            "version": version,
            "source": "unspecified",
            "optional": False,
        }
    elif isinstance(entry, dict):
        name = str(entry.get("name", "")).strip()
        version_value = entry.get("version", entry.get("specifier"))
        version = None if version_value is None or version_value == "" else str(version_value).strip()
        source = str(entry.get("source", "unspecified")).strip() or "unspecified"
        normalized = {
            "name": name,
            "version": version,
            "source": source,
            "optional": bool(entry.get("optional", False)),
        }
        if entry.get("purpose") is not None:
            normalized["purpose"] = str(entry["purpose"])
        unknown_keys = sorted(set(entry) - {"name", "version", "specifier", "source", "optional", "purpose"})
        if unknown_keys:
            warnings.append(f"{field}[{index}] has unrecognized keys: {','.join(unknown_keys)}")
    else:
        return None, [f"{field}[{index}] must be an object or requirement string"], warnings

    if not normalized["name"]:
        errors.append(f"{field}[{index}] is missing dependency name")
    elif not _DEPENDENCY_NAME_RE.match(str(normalized["name"])):
        errors.append(f"{field}[{index}] has invalid dependency name: {normalized['name']}")
    if not normalized["version"]:
        warnings.append(f"{field}[{index}] has no version/specifier")

    return normalized if not errors else None, errors, warnings


def _manifest_dependency_report(manifest: dict[str, Any]) -> dict[str, Any]:
    dependencies = manifest.get("dependencies", {})
    normalized: dict[str, list[dict[str, Any]]] = {}
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(dependencies, dict):
        return {
            "ok": False,
            "normalized": normalized,
            "errors": ["dependencies must be an object"],
            "warnings": warnings,
        }

    package_fields = sorted(key for key in dependencies if str(key).endswith("_packages"))
    if "inference_packages" not in package_fields:
        warnings.append("dependencies.inference_packages is not declared")

    for field in package_fields:
        entries = dependencies.get(field)
        if not isinstance(entries, list):
            errors.append(f"dependencies.{field} must be a list")
            continue
        normalized_entries: list[dict[str, Any]] = []
        for idx, entry in enumerate(entries):
            normalized_entry, entry_errors, entry_warnings = _normalize_dependency_entry(
                entry,
                field=f"dependencies.{field}",
                index=idx,
            )
            errors.extend(entry_errors)
            warnings.extend(entry_warnings)
            if normalized_entry is not None:
                normalized_entries.append(normalized_entry)
        normalized[field] = normalized_entries

    return {
        "ok": not errors,
        "normalized": normalized,
        "errors": errors,
        "warnings": warnings,
    }


def validate_learned_submission_manifest(
    *,
    manifest: str | Path,
    bundle_root: str | Path | None = None,
    allow_undisclosed: bool = False,
) -> dict[str, Any]:
    """Validate a standalone learned-policy submission manifest before bundling."""

    manifest_path = Path(manifest)
    artifact_root = Path(bundle_root) if bundle_root is not None else manifest_path.parent
    try:
        manifest_payload = _read_json(manifest_path)
        manifest_load_error = None
    except Exception as exc:
        manifest_payload = {}
        manifest_load_error = f"{type(exc).__name__}: {exc}"

    required_sections = ("policy", "benchmark", "dependencies", "training_disclosure", "inference_disclosure", "review_notes")
    sections_present = isinstance(manifest_payload, dict) and all(
        isinstance(manifest_payload.get(section), dict)
        for section in required_sections
    )
    policy = manifest_payload.get("policy", {}) if isinstance(manifest_payload.get("policy"), dict) else {}
    benchmark = manifest_payload.get("benchmark", {}) if isinstance(manifest_payload.get("benchmark"), dict) else {}

    artifacts = manifest_payload.get("artifacts", []) if isinstance(manifest_payload, dict) else []
    artifact_record_errors: list[str] = []
    artifact_missing: list[str] = []
    artifact_hash_mismatches: list[dict[str, Any]] = []
    if not isinstance(artifacts, list):
        artifact_record_errors.append("artifacts must be a list")
        artifacts = []
    for idx, item in enumerate(artifacts):
        if not isinstance(item, dict):
            artifact_record_errors.append(f"artifacts[{idx}] must be an object")
            continue
        name = str(item.get("name", "")).strip()
        rel_path = str(item.get("path", "")).strip()
        if not name:
            artifact_record_errors.append(f"artifacts[{idx}] is missing name")
        if not rel_path:
            artifact_record_errors.append(f"artifacts[{idx}] is missing path")
            continue
        artifact_path = artifact_root / rel_path
        if bundle_root is not None and not artifact_path.exists():
            artifact_missing.append(rel_path)
            continue
        declared_hash = item.get("sha256")
        if declared_hash and artifact_path.exists() and artifact_path.is_file():
            actual_hash = _sha256(artifact_path)
            if actual_hash != declared_hash:
                artifact_hash_mismatches.append(
                    {
                        "artifact": name,
                        "path": rel_path,
                        "declared": declared_hash,
                        "actual": actual_hash,
                    }
                )

    dependency_report = _manifest_dependency_report(manifest_payload)
    unknown_fields = _manifest_unknown_fields(manifest_payload) if isinstance(manifest_payload, dict) else []
    manifest_schema_errors = _schema_errors(manifest_payload, LEARNED_SUBMISSION_MANIFEST_SCHEMA_FILE)
    seeds = benchmark.get("seeds")
    checks = [
        _check("manifest_json_loads", manifest_load_error is None, {"error": manifest_load_error}),
        _check(
            "manifest_schema_supported",
            manifest_payload.get("schema_version") == LEARNED_SUBMISSION_MANIFEST_SCHEMA_VERSION,
            {"schema_version": manifest_payload.get("schema_version")},
        ),
        _check(
            "manifest_json_schema_valid",
            not manifest_schema_errors,
            {"schema_file": LEARNED_SUBMISSION_MANIFEST_SCHEMA_FILE, "errors": manifest_schema_errors[:20]},
        ),
        _check(
            "manifest_sections_present",
            sections_present,
            {"required_sections": list(required_sections), "present_sections": list(manifest_payload.keys())},
        ),
        _check(
            "manifest_policy_identity_present",
            bool(policy.get("name")) and bool(policy.get("method")),
            {"policy": {"name": policy.get("name"), "method": policy.get("method")}},
        ),
        _check(
            "manifest_benchmark_shape_present",
            bool(benchmark.get("suite"))
            and isinstance(benchmark.get("n_agents"), int)
            and int(benchmark.get("n_agents", 0) or 0) > 0
            and isinstance(seeds, list)
            and all(isinstance(seed, int) for seed in seeds),
            {
                "suite": benchmark.get("suite"),
                "n_agents": benchmark.get("n_agents"),
                "seeds": seeds,
            },
        ),
        _check(
            "manifest_dependencies_normalized",
            bool(dependency_report["ok"]),
            {
                "errors": dependency_report["errors"],
                "warnings": dependency_report["warnings"],
                "normalized": dependency_report["normalized"],
            },
        ),
        _check(
            "manifest_disclosures_complete",
            bool(allow_undisclosed or not unknown_fields),
            {"unknown_fields": unknown_fields, "allow_undisclosed": bool(allow_undisclosed)},
        ),
        _check(
            "manifest_artifact_records_valid",
            not artifact_record_errors and not artifact_missing,
            {"errors": artifact_record_errors, "missing": artifact_missing},
        ),
        _check(
            "manifest_artifact_hashes_match",
            not artifact_hash_mismatches,
            {"mismatches": artifact_hash_mismatches[:20]},
        ),
    ]

    return {
        "schema_version": LEARNED_SUBMISSION_MANIFEST_VALIDATION_SCHEMA_VERSION,
        "manifest_schema_version": manifest_payload.get("schema_version"),
        "ok": all(check["ok"] for check in checks),
        "manifest": str(manifest_path),
        "bundle_root": str(artifact_root),
        "policy": policy,
        "benchmark": benchmark,
        "artifact_count": len(artifacts),
        "dependencies": dependency_report,
        "unknown_fields": unknown_fields,
        "checks": checks,
    }


def _run_planner_sweep(
    *,
    out_dir: Path,
    suite: str,
    method: str,
    policy_spec: str | Path | None = None,
    max_runs: int | None,
    save_trace: bool,
) -> dict[str, Any]:
    if (out_dir / "results.csv").exists():
        raise RuntimeError(f"planner sweep output already exists: {out_dir / 'results.csv'}")

    generated = materialize_official_suite(
        suite,
        out_dir / "_generated_scenarios" / suite,
        overwrite=True,
    )
    defaults = suite_defaults(suite)
    scenarios = [Path(path) for path in generated["scenario_paths"]]
    n_agents = [int(value) for value in defaults["n_agents"]]
    seeds = [int(value) for value in defaults["seeds"]]
    comm_profiles = [str(value) for value in defaults["comm_profiles"]]

    specs: list[RunSpec] = []
    for scenario in scenarios:
        for comm_profile in comm_profiles:
            for n in n_agents:
                for seed in seeds:
                    specs.append(
                        RunSpec(
                            scenario_path=str(scenario),
                            method=str(method),
                            n_agents=int(n),
                            seed=int(seed),
                            comm_profile=str(comm_profile),
                            out_dir=str(out_dir),
                            save_trace=bool(save_trace),
                            policy_spec=None if policy_spec is None else str(policy_spec),
                        )
                    )

    planned_run_count = len(specs)
    if max_runs is not None:
        specs = specs[: max(0, int(max_runs))]

    rows: list[dict[str, Any]] = []
    for spec in specs:
        row = run_episode(spec)
        append_result(out_dir, row)
        rows.append(row)
    summary_csv = write_summary(out_dir)
    results_csv = out_dir / "results.csv"

    acceptance = check_acceptance(
        summary_csv=summary_csv,
        results_csv=results_csv,
        suite_manifest=generated["manifest_path"],
        methods=[str(method)],
    )

    guardrail_total = 0
    finite_metric_violations: list[dict[str, Any]] = []
    for row in rows:
        for field in ("planner_timeout_count", "planner_error_count", "planner_fallback_count"):
            guardrail_total += int(float(row.get(field, 0) or 0))
        for field in ("collision_episode", "completion_rate", "min_sep_min_m", "planner_ms_per_tick_per_agent_p95"):
            if not _finite(row.get(field)):
                finite_metric_violations.append(
                    {
                        "scenario": row.get("scenario"),
                        "comm_profile": row.get("comm_profile"),
                        "N": row.get("N"),
                        "seed": row.get("seed"),
                        "field": field,
                        "value": row.get(field),
                    }
                )

    return {
        "suite": str(suite),
        "method": str(method),
        "policy_spec": None if policy_spec is None else str(policy_spec),
        "planned_run_count": int(planned_run_count),
        "run_count": len(rows),
        "max_runs": None if max_runs is None else int(max_runs),
        "results_csv": str(results_csv),
        "summary_csv": str(summary_csv),
        "result_schema_json": str(out_dir / "result_schema.json"),
        "suite_manifest": str(generated["manifest_path"]),
        "acceptance_json": str(out_dir / "acceptance.json"),
        "acceptance": acceptance,
        "guardrail_total": int(guardrail_total),
        "finite_metric_violations": finite_metric_violations[:20],
        "rows": [
            {
                "method": row.get("method"),
                "scenario": row.get("scenario"),
                "comm_profile": row.get("comm_profile"),
                "N": row.get("N"),
                "seed": row.get("seed"),
                "collision_episode": row.get("collision_episode"),
                "completion_rate": row.get("completion_rate"),
                "min_sep_min_m": row.get("min_sep_min_m"),
                "planner_ms_per_tick_per_agent_p95": row.get("planner_ms_per_tick_per_agent_p95"),
                "planner_timeout_count": row.get("planner_timeout_count"),
                "planner_error_count": row.get("planner_error_count"),
                "planner_fallback_count": row.get("planner_fallback_count"),
            }
            for row in rows
        ],
    }


def run_learned_policy_submission_bundle(
    *,
    out_dir: str | Path,
    method: str = "learned_tiny",
    policy: str = "tiny_learned",
    policy_spec: str | Path | None = None,
    suite: str = "official_smoke_generated",
    root: str | Path = ".",
    n_agents: int = 4,
    seeds: tuple[int, ...] | list[int] | None = None,
    max_steps: int | None = None,
    max_runs: int | None = None,
    save_trace: bool = False,
    submission_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Create a reproducible learned-policy submission artifact bundle."""

    out = Path(out_dir)
    bundle_path = out / "learned_submission_bundle.json"
    if bundle_path.exists():
        raise RuntimeError(f"learned submission bundle already exists: {bundle_path}")
    out.mkdir(parents=True, exist_ok=True)

    seed_list = [int(seed) for seed in (seeds if seeds is not None else (0,))]
    contract = interface_contract(top_k=8)
    freeze = run_rl_freeze_check(root=root)
    smoke = run_rl_policy_smoke(
        out_dir=out / "rl_smoke",
        policy=str(policy),
        policy_spec=policy_spec,
        n_agents=int(n_agents),
        seeds=seed_list,
        max_steps=max_steps,
    )
    calibration = run_rl_policy_calibration(
        out_dir=out / "rl_calibration",
        policy=str(policy),
        policy_spec=policy_spec,
        n_agents=int(n_agents),
        seeds=seed_list,
        max_steps=max_steps,
    )
    planner_sweep = _run_planner_sweep(
        out_dir=out / "planner_sweep",
        suite=str(suite),
        method=str(method),
        policy_spec=policy_spec,
        max_runs=max_runs,
        save_trace=bool(save_trace),
    )

    _write_json(out / "rl_contract.json", contract)
    _write_json(out / "rl_freeze_check.json", freeze)
    _write_json(out / "rl_smoke.json", smoke)
    _write_json(out / "rl_calibration.json", calibration)
    _write_json(Path(planner_sweep["acceptance_json"]), planner_sweep["acceptance"])

    meta = _method_metadata(str(method))
    policy_spec_artifact: str | None = None
    policy_model_artifact: str | None = None
    if policy_spec is not None:
        spec_path = Path(policy_spec)
        model_artifact = resolve_policy_artifact_path(spec_path)
        spec_artifact_rel: str | None = None
        if model_artifact is not None and model_artifact.exists() and model_artifact.is_file():
            artifact_out = out / "policy_artifacts" / model_artifact.name
            artifact_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(model_artifact, artifact_out)
            policy_model_artifact = _rel(artifact_out, out)
            spec_artifact_rel = policy_model_artifact
        policy_spec_payload = portable_policy_spec_payload(spec_path, artifact_path=spec_artifact_rel)
        policy_spec_out = _write_json(out / "policy_spec.json", policy_spec_payload)
        policy_spec_artifact = _rel(policy_spec_out, out)

    artifact_paths = {
        "rl_contract": _rel(out / "rl_contract.json", out),
        "rl_freeze_check": _rel(out / "rl_freeze_check.json", out),
        "rl_smoke": _rel(out / "rl_smoke.json", out),
        "rl_smoke_episodes": _rel(out / "rl_smoke" / "rl_smoke_episodes.csv", out),
        "rl_calibration": _rel(out / "rl_calibration.json", out),
        "rl_calibration_episodes": _rel(out / "rl_calibration" / "rl_calibration_episodes.csv", out),
        "planner_results": _rel(planner_sweep["results_csv"], out),
        "planner_summary": _rel(planner_sweep["summary_csv"], out),
        "planner_result_schema": _rel(planner_sweep["result_schema_json"], out),
        "planner_suite_manifest": _rel(planner_sweep["suite_manifest"], out),
        "planner_acceptance": _rel(planner_sweep["acceptance_json"], out),
    }
    if policy_spec_artifact is not None:
        artifact_paths["policy_spec"] = policy_spec_artifact
    if policy_model_artifact is not None:
        artifact_paths["policy_artifact"] = policy_model_artifact
    artifact_paths["learned_submission_manifest"] = LEARNED_SUBMISSION_MANIFEST_FILENAME
    manifest = _build_learned_submission_manifest(
        bundle_root=out,
        method=str(method),
        policy=str(smoke.get("policy", policy)),
        suite=str(suite),
        root=root,
        n_agents=int(n_agents),
        seeds=seed_list,
        max_steps=max_steps,
        max_runs=max_runs,
        planner_sweep=planner_sweep,
        artifact_paths=artifact_paths,
        policy_spec_summary=smoke.get("policy_spec"),
        method_metadata=meta,
        manifest_overrides=_load_manifest_overrides(submission_manifest),
    )
    manifest_out = _write_json(out / LEARNED_SUBMISSION_MANIFEST_FILENAME, manifest)
    artifact_paths["learned_submission_manifest"] = _rel(manifest_out, out)
    missing_artifacts = [name for name, path in artifact_paths.items() if not (out / path).exists()]
    checks = [
        _check("method_metadata_present", meta is not None, {"method": str(method)}),
        _check("method_marked_learned", bool(meta and meta.get("learned")), {"learned": None if meta is None else meta.get("learned")}),
        _check("rl_freeze_check_ok", bool(freeze.get("ok")), {"failed": [c["name"] for c in freeze.get("checks", []) if not c.get("ok")]}),
        _check("rl_smoke_ok", bool(smoke.get("ok")), {"failed": [c["name"] for c in smoke.get("checks", []) if not c.get("ok")]}),
        _check(
            "rl_calibration_ok",
            bool(calibration.get("ok")),
            {"failed": [c["name"] for c in calibration.get("checks", []) if not c.get("ok")]},
        ),
        _check(
            "planner_sweep_ran",
            int(planner_sweep["run_count"]) > 0,
            {"run_count": planner_sweep["run_count"], "planned_run_count": planner_sweep["planned_run_count"]},
        ),
        _check(
            "planner_sweep_guardrails_clear",
            int(planner_sweep["guardrail_total"]) == 0,
            {"guardrail_total": planner_sweep["guardrail_total"]},
        ),
        _check(
            "planner_sweep_metrics_finite",
            not planner_sweep["finite_metric_violations"],
            {"violations": planner_sweep["finite_metric_violations"]},
        ),
        _check(
            "planner_acceptance_no_failures",
            bool(planner_sweep["acceptance"].get("ok")),
            {
                "status": planner_sweep["acceptance"].get("status"),
                "rules_failed": planner_sweep["acceptance"].get("rules_failed"),
            },
        ),
        _check("expected_artifacts_present", not missing_artifacts, {"missing": missing_artifacts}),
        _check(
            "learned_submission_manifest_written",
            (out / artifact_paths["learned_submission_manifest"]).exists(),
            {"path": artifact_paths["learned_submission_manifest"], "schema_version": manifest.get("schema_version")},
        ),
    ]

    report = {
        "schema_version": LEARNED_SUBMISSION_BUNDLE_SCHEMA_VERSION,
        "ok": all(check["ok"] for check in checks),
        "method": str(method),
        "policy": str(smoke.get("policy", policy)),
        "policy_spec": smoke.get("policy_spec"),
        "submission_manifest": {
            "schema_version": manifest.get("schema_version"),
            "path": artifact_paths["learned_submission_manifest"],
            "training_disclosure": manifest.get("training_disclosure"),
            "inference_disclosure": manifest.get("inference_disclosure"),
        },
        "suite": str(suite),
        "root": str(root),
        "n_agents": int(n_agents),
        "seeds": seed_list,
        "max_steps": None if max_steps is None else int(max_steps),
        "max_runs": None if max_runs is None else int(max_runs),
        "method_metadata": meta,
        "artifacts": artifact_paths,
        "checks": checks,
        "planner_sweep": {
            key: value
            for key, value in planner_sweep.items()
            if key not in {"acceptance"}
        },
        "acceptance": planner_sweep["acceptance"],
    }
    _write_json(bundle_path, report)
    return report


def _bundle_json_path(bundle: str | Path) -> Path:
    path = Path(bundle)
    if path.is_dir():
        return path / LEARNED_SUBMISSION_BUNDLE_FILENAME
    return path


def _expected_rel_for(name: str, suite: str | None) -> str | None:
    pattern = EXPECTED_BUNDLE_ARTIFACTS.get(name) or OPTIONAL_BUNDLE_ARTIFACTS.get(name)
    if pattern is None:
        return None
    return pattern.format(suite=suite or "*", artifact_name="*")


def _resolve_artifact(bundle_root: Path, report: dict[str, Any], name: str) -> Path:
    artifacts = report.get("artifacts", {})
    raw = artifacts.get(name) if isinstance(artifacts, dict) else None
    candidates: list[Path] = []
    if raw:
        raw_path = Path(str(raw))
        if raw_path.is_absolute():
            candidates.append(raw_path)
            candidates.append(bundle_root / raw_path.name)
        else:
            candidates.append(bundle_root / raw_path)
            candidates.append(raw_path)

    expected = _expected_rel_for(name, str(report.get("suite", "")))
    if expected is not None and "*" not in expected:
        candidates.append(bundle_root / expected)
    if name == "planner_suite_manifest":
        manifests = sorted((bundle_root / "planner_sweep" / "_generated_scenarios").glob("*/suite_manifest.yaml"))
        candidates.extend(manifests)
    if name == "policy_artifact":
        artifacts = sorted((bundle_root / "policy_artifacts").glob("*"))
        candidates.extend(path for path in artifacts if path.is_file())

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else bundle_root / str(name)


def validate_learned_policy_submission_bundle(*, bundle: str | Path) -> dict[str, Any]:
    """Validate an already-created learned-policy submission bundle."""

    bundle_json = _bundle_json_path(bundle)
    bundle_root = bundle_json.parent
    try:
        report = _read_json(bundle_json)
        bundle_load_error = None
    except Exception as exc:
        report = {}
        bundle_load_error = f"{type(exc).__name__}: {exc}"

    suite = str(report.get("suite", "")) if report else ""
    report_artifacts = report.get("artifacts") if isinstance(report.get("artifacts"), dict) else {}
    legacy_manifest_missing = bool(report) and "learned_submission_manifest" not in report_artifacts
    expected_artifact_names = [
        name
        for name in EXPECTED_BUNDLE_ARTIFACTS
        if name != "learned_submission_manifest" or not legacy_manifest_missing
    ]
    resolved = {
        name: _resolve_artifact(bundle_root, report, name)
        for name in expected_artifact_names
    }
    optional_declared = [
        name
        for name in OPTIONAL_BUNDLE_ARTIFACTS
        if name in report_artifacts
    ]
    optional_resolved = {
        name: _resolve_artifact(bundle_root, report, name)
        for name in optional_declared
    }
    all_resolved = {**resolved, **optional_resolved}
    missing = [name for name, path in resolved.items() if not path.exists()]
    optional_missing = [name for name, path in optional_resolved.items() if not path.exists()]

    json_payloads: dict[str, dict[str, Any]] = {}
    json_errors: list[dict[str, str]] = []
    json_artifact_names = [
        "rl_contract",
        "rl_freeze_check",
        "rl_smoke",
        "rl_calibration",
        "planner_acceptance",
    ]
    if "learned_submission_manifest" in all_resolved:
        json_artifact_names.insert(0, "learned_submission_manifest")
    if "policy_spec" in optional_resolved:
        json_artifact_names.append("policy_spec")
    for name in json_artifact_names:
        path = all_resolved[name]
        if not path.exists():
            continue
        try:
            json_payloads[name] = _read_json(path)
        except Exception as exc:
            json_errors.append({"artifact": name, "path": str(path), "error": f"{type(exc).__name__}: {exc}"})

    csv_errors: list[dict[str, str]] = []
    csv_counts: dict[str, int] = {}
    for name in ("rl_smoke_episodes", "rl_calibration_episodes", "planner_results", "planner_summary"):
        path = resolved[name]
        if not path.exists():
            continue
        try:
            csv_counts[name] = len(_read_csv_rows(path))
        except Exception as exc:
            csv_errors.append({"artifact": name, "path": str(path), "error": f"{type(exc).__name__}: {exc}"})

    schema_payload = None
    result_schema_error = None
    if resolved["planner_result_schema"].exists():
        try:
            schema_payload = _read_json(resolved["planner_result_schema"])
        except Exception as exc:
            result_schema_error = f"{type(exc).__name__}: {exc}"

    manifest_payload = json_payloads.get("learned_submission_manifest", {})
    manifest_artifacts = _manifest_artifact_map(manifest_payload) if isinstance(manifest_payload, dict) else {}
    report_artifact_names = set(report_artifacts.keys()) - {"learned_submission_manifest"}
    manifest_artifact_names = set(manifest_artifacts.keys())
    missing_manifest_artifacts = [] if legacy_manifest_missing else sorted(report_artifact_names - manifest_artifact_names)
    artifact_hash_mismatches: list[dict[str, Any]] = []
    if not legacy_manifest_missing:
        for name, item in manifest_artifacts.items():
            rel_path = str(item.get("path", ""))
            declared_hash = item.get("sha256")
            if not declared_hash:
                continue
            artifact_path = bundle_root / rel_path
            if artifact_path.exists() and artifact_path.is_file():
                actual_hash = _sha256(artifact_path)
                if actual_hash != declared_hash:
                    artifact_hash_mismatches.append(
                        {
                            "artifact": name,
                            "path": rel_path,
                            "declared": declared_hash,
                            "actual": actual_hash,
                        }
                    )
    manifest_policy_spec = (manifest_payload.get("policy") or {}).get("policy_spec") if isinstance(manifest_payload, dict) else None
    report_policy_spec = report.get("policy_spec")
    if legacy_manifest_missing:
        policy_spec_provenance_ok = True
    elif report_policy_spec is None:
        policy_spec_provenance_ok = manifest_policy_spec is None
    else:
        policy_spec_provenance_ok = isinstance(manifest_policy_spec, dict) and all(
            manifest_policy_spec.get(key) == report_policy_spec.get(key)
            for key in ("policy_name", "adapter")
        )
    manifest_required_sections_present = legacy_manifest_missing or (isinstance(manifest_payload, dict) and all(
        isinstance(manifest_payload.get(section), dict)
        for section in ("policy", "benchmark", "dependencies", "training_disclosure", "inference_disclosure")
    ))
    dependency_report = _manifest_dependency_report(manifest_payload) if isinstance(manifest_payload, dict) else {
        "ok": bool(legacy_manifest_missing),
        "normalized": {},
        "errors": [] if legacy_manifest_missing else ["learned_submission_manifest is missing or unreadable"],
        "warnings": [],
    }

    bundle_checks = report.get("checks", []) if isinstance(report.get("checks"), list) else []
    failed_bundle_checks = [check.get("name") for check in bundle_checks if not check.get("ok")]
    bundle_schema_errors = _schema_errors(report, LEARNED_SUBMISSION_BUNDLE_SCHEMA_FILE)
    checks = [
        _check("bundle_json_loads", bundle_load_error is None, {"error": bundle_load_error}),
        _check(
            "bundle_schema_supported",
            report.get("schema_version") == LEARNED_SUBMISSION_BUNDLE_SCHEMA_VERSION,
            {"schema_version": report.get("schema_version")},
        ),
        _check(
            "bundle_json_schema_valid",
            not bundle_schema_errors,
            {"schema_file": LEARNED_SUBMISSION_BUNDLE_SCHEMA_FILE, "errors": bundle_schema_errors[:20]},
        ),
        _check("bundle_report_ok", bool(report.get("ok")), {"failed_checks": failed_bundle_checks}),
        _check(
            "required_artifacts_declared",
            set(expected_artifact_names).issubset(set(report_artifacts.keys())),
            {"legacy_manifest_missing": legacy_manifest_missing},
        ),
        _check("required_artifacts_present", not missing, {"missing": missing}),
        _check("optional_artifacts_present", not optional_missing, {"missing": optional_missing, "declared": optional_declared}),
        _check("json_artifacts_parse", not json_errors and result_schema_error is None, {"errors": json_errors, "result_schema_error": result_schema_error}),
        _check(
            "learned_submission_manifest_schema_supported",
            legacy_manifest_missing
            or (isinstance(manifest_payload, dict)
            and manifest_payload.get("schema_version") == LEARNED_SUBMISSION_MANIFEST_SCHEMA_VERSION),
            {"schema_version": manifest_payload.get("schema_version") if isinstance(manifest_payload, dict) else None, "legacy_manifest_missing": legacy_manifest_missing},
        ),
        _check(
            "learned_submission_manifest_sections_present",
            manifest_required_sections_present,
            {"sections": list(manifest_payload.keys()) if isinstance(manifest_payload, dict) else []},
        ),
        _check(
            "learned_submission_manifest_artifacts_match",
            not missing_manifest_artifacts,
            {"missing_from_manifest": missing_manifest_artifacts},
        ),
        _check(
            "learned_submission_manifest_hashes_match",
            not artifact_hash_mismatches,
            {"mismatches": artifact_hash_mismatches[:20]},
        ),
        _check(
            "learned_submission_manifest_policy_spec_provenance",
            policy_spec_provenance_ok,
            {"manifest_policy_spec": manifest_policy_spec, "bundle_policy_spec": report_policy_spec},
        ),
        _check(
            "learned_submission_manifest_dependencies_normalized",
            bool(dependency_report["ok"]),
            {
                "errors": dependency_report["errors"],
                "warnings": dependency_report["warnings"],
                "normalized": dependency_report["normalized"],
                "legacy_manifest_missing": legacy_manifest_missing,
            },
        ),
        _check(
            "rl_reports_ok",
            bool(json_payloads.get("rl_freeze_check", {}).get("ok"))
            and bool(json_payloads.get("rl_smoke", {}).get("ok"))
            and bool(json_payloads.get("rl_calibration", {}).get("ok")),
        ),
        _check(
            "planner_acceptance_ok",
            bool(json_payloads.get("planner_acceptance", {}).get("ok")),
            {
                "status": json_payloads.get("planner_acceptance", {}).get("status"),
                "rules_failed": json_payloads.get("planner_acceptance", {}).get("rules_failed"),
            },
        ),
        _check(
            "csv_artifacts_nonempty",
            not csv_errors
            and csv_counts.get("rl_smoke_episodes", 0) > 0
            and csv_counts.get("rl_calibration_episodes", 0) > 0
            and csv_counts.get("planner_results", 0) > 0
            and csv_counts.get("planner_summary", 0) > 0,
            {"counts": csv_counts, "errors": csv_errors},
        ),
        _check(
            "planner_result_schema_present",
            isinstance(schema_payload, dict) and bool(schema_payload.get("schema_version")),
            {"schema_version": schema_payload.get("schema_version") if isinstance(schema_payload, dict) else None},
        ),
        _check(
            "method_marked_learned",
            bool((report.get("method_metadata") or {}).get("learned")),
            {"method": report.get("method")},
        ),
    ]

    return {
        "schema_version": LEARNED_SUBMISSION_BUNDLE_VALIDATION_SCHEMA_VERSION,
        "bundle_schema_version": report.get("schema_version"),
        "ok": all(check["ok"] for check in checks),
        "bundle_json": str(bundle_json),
        "bundle_root": str(bundle_root),
        "method": report.get("method"),
        "policy": report.get("policy"),
        "suite": suite,
        "artifacts": {name: str(path) for name, path in all_resolved.items()},
        "submission_manifest": {
            "schema_version": manifest_payload.get("schema_version") if isinstance(manifest_payload, dict) else None,
            "unknown_fields": _manifest_unknown_fields(manifest_payload) if isinstance(manifest_payload, dict) else [],
            "artifact_count": len(manifest_artifacts),
            "legacy_missing": legacy_manifest_missing,
            "dependencies": dependency_report,
        },
        "checks": checks,
    }


def _score_v0(row: dict[str, Any]) -> float | None:
    collision_episode_rate = _to_float(row.get("collision_episode_rate"))
    unique_collision_pairs_mean = _to_float(row.get("unique_collision_pairs_mean"))
    collision_pair_ticks_mean = _to_float(row.get("collision_pair_ticks_mean"))
    min_sep_p05_mean = _to_float(row.get("min_sep_p05_mean"))
    completion_rate_mean = _to_float(row.get("completion_rate_mean"))
    deadlock_time_pct_mean = _to_float(row.get("deadlock_time_pct_mean"))
    planner_ms_p95 = _to_float(row.get("planner_ms_p95"))

    required = (
        collision_episode_rate,
        unique_collision_pairs_mean,
        collision_pair_ticks_mean,
        min_sep_p05_mean,
        completion_rate_mean,
        deadlock_time_pct_mean,
        planner_ms_p95,
    )
    if any(value is None for value in required):
        return None

    mean_time_to_goal_mean = _to_float(row.get("mean_time_to_goal_mean")) or 0.0
    safety_penalty = (
        1000.0 * collision_episode_rate
        + 50.0 * unique_collision_pairs_mean
        + 0.1 * collision_pair_ticks_mean
        + 10.0 * max(0.0, -min_sep_p05_mean)
    )
    mission_penalty = (
        100.0 * (1.0 - completion_rate_mean)
        + 2.0 * deadlock_time_pct_mean
        + 0.01 * mean_time_to_goal_mean
    )
    compute_penalty = 0.1 * planner_ms_p95
    return safety_penalty + mission_penalty + compute_penalty


def _score_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for row in summary_rows:
        score = _score_v0(row)
        scored.append(
            {
                "method": row.get("method"),
                "scenario": row.get("scenario"),
                "comm_profile": row.get("comm_profile"),
                "N": row.get("N"),
                "episodes": row.get("episodes"),
                "score_v0": _round_or_none(score),
                "collision_episode_rate": _round_or_none(_to_float(row.get("collision_episode_rate"))),
                "completion_rate_mean": _round_or_none(_to_float(row.get("completion_rate_mean"))),
                "min_sep_p05_mean": _round_or_none(_to_float(row.get("min_sep_p05_mean"))),
                "planner_ms_p95": _round_or_none(_to_float(row.get("planner_ms_p95"))),
            }
        )
    return scored


def review_learned_policy_submission_bundle(*, bundle: str | Path) -> dict[str, Any]:
    """Build a concise reviewer summary for a learned-policy submission bundle."""

    validation = validate_learned_policy_submission_bundle(bundle=bundle)
    bundle_json = Path(validation["bundle_json"])
    try:
        bundle_report = _read_json(bundle_json)
        bundle_load_error = None
    except Exception as exc:
        bundle_report = {}
        bundle_load_error = f"{type(exc).__name__}: {exc}"

    artifacts = validation.get("artifacts", {})
    legacy_manifest_missing = bool(validation.get("submission_manifest", {}).get("legacy_missing"))
    manifest_payload: dict[str, Any] = {}
    manifest_path = Path(str(artifacts.get("learned_submission_manifest", "")))
    if manifest_path.exists():
        try:
            manifest_payload = _read_json(manifest_path)
        except Exception:
            manifest_payload = {}
    summary_rows: list[dict[str, str]] = []
    results_rows: list[dict[str, str]] = []
    csv_errors: list[dict[str, str]] = []
    for name, target in (("planner_summary", summary_rows), ("planner_results", results_rows)):
        path = Path(str(artifacts.get(name, "")))
        if not path.exists():
            continue
        try:
            target.extend(_read_csv_rows(path))
        except Exception as exc:
            csv_errors.append({"artifact": name, "path": str(path), "error": f"{type(exc).__name__}: {exc}"})

    scored_rows = _score_rows(summary_rows)
    scores = [float(row["score_v0"]) for row in scored_rows if row.get("score_v0") is not None]
    total_collision_episodes = int(_sum_values(_values(results_rows, "collision_episode")) or 0)
    total_timeouts = int(_sum_values(_values(results_rows, "planner_timeout_count")) or 0)
    total_errors = int(_sum_values(_values(results_rows, "planner_error_count")) or 0)
    total_fallbacks = int(_sum_values(_values(results_rows, "planner_fallback_count")) or 0)

    run_count = int(bundle_report.get("planner_sweep", {}).get("run_count", 0) or 0)
    planned_run_count = int(bundle_report.get("planner_sweep", {}).get("planned_run_count", 0) or 0)
    max_runs = bundle_report.get("max_runs")
    limited_sweep = max_runs is not None and planned_run_count > run_count

    limitations: list[str] = []
    if not validation["ok"]:
        limitations.append("bundle_validation_failed")
    if bundle_load_error is not None:
        limitations.append("bundle_report_unreadable")
    if limited_sweep:
        limitations.append("limited_planner_sweep")
    if total_collision_episodes > 0:
        limitations.append("collision_episodes_present")
    if total_timeouts or total_errors or total_fallbacks:
        limitations.append("planner_guardrails_present")
    if not scores:
        limitations.append("score_v0_unavailable")
    unknown_manifest_fields = _manifest_unknown_fields(manifest_payload) if manifest_payload else []
    if unknown_manifest_fields:
        limitations.append("submission_disclosure_incomplete")
    if legacy_manifest_missing:
        limitations.append("legacy_bundle_without_submission_manifest")

    checks = [
        _check("bundle_validation_ok", bool(validation["ok"])),
        _check("bundle_report_loads", bundle_load_error is None, {"error": bundle_load_error}),
        _check("planner_summary_rows_present", len(summary_rows) > 0, {"rows": len(summary_rows)}),
        _check("planner_results_rows_present", len(results_rows) > 0, {"rows": len(results_rows)}),
        _check("score_v0_computable", bool(scores), {"summary_rows": len(summary_rows), "scored_rows": len(scores)}),
        _check(
            "submission_manifest_present",
            bool(manifest_payload) or legacy_manifest_missing,
            {"path": str(manifest_path) if str(manifest_path) else None, "legacy_missing": legacy_manifest_missing},
        ),
    ]
    ok = all(check["ok"] for check in checks)
    if not ok:
        recommendation = "fix_artifacts"
    elif limited_sweep:
        recommendation = "manual_review_limited_sweep"
    elif total_collision_episodes or total_timeouts or total_errors or total_fallbacks:
        recommendation = "manual_review_required"
    elif unknown_manifest_fields or legacy_manifest_missing:
        recommendation = "manual_review_required"
    else:
        recommendation = "leaderboard_candidate"

    safety = {
        "collision_episode_count": total_collision_episodes,
        "collision_episode_rate_mean": _round_or_none(_mean(_values(summary_rows, "collision_episode_rate"))),
        "collision_episode_rate_max": _round_or_none(max(_values(summary_rows, "collision_episode_rate")) if _values(summary_rows, "collision_episode_rate") else None),
        "near_miss_episode_rate_mean": _round_or_none(_mean(_values(summary_rows, "near_miss_episode_rate"))),
        "min_sep_p05_min_m": _round_or_none(min(_values(summary_rows, "min_sep_p05_mean")) if _values(summary_rows, "min_sep_p05_mean") else None),
        "min_sep_min_m": _round_or_none(min(_values(summary_rows, "min_sep_min_mean")) if _values(summary_rows, "min_sep_min_mean") else None),
    }
    mission = {
        "completion_rate_mean": _round_or_none(_mean(_values(summary_rows, "completion_rate_mean"))),
        "completion_rate_min": _round_or_none(min(_values(summary_rows, "completion_rate_mean")) if _values(summary_rows, "completion_rate_mean") else None),
        "deadlock_time_pct_mean": _round_or_none(_mean(_values(summary_rows, "deadlock_time_pct_mean"))),
        "mean_time_to_goal_mean": _round_or_none(_mean(_values(summary_rows, "mean_time_to_goal_mean"))),
    }
    compute = {
        "planner_ms_p95_max": _round_or_none(max(_values(summary_rows, "planner_ms_p95")) if _values(summary_rows, "planner_ms_p95") else None),
        "planner_ms_mean_mean": _round_or_none(_mean(_values(summary_rows, "planner_ms_mean"))),
        "planner_timeout_count": total_timeouts,
        "planner_error_count": total_errors,
        "planner_fallback_count": total_fallbacks,
    }
    communication = {
        "bandwidth_Bps_mean": _round_or_none(_mean(_values(summary_rows, "comm_agent_msg_bandwidth_Bps_mean"))),
        "drop_fraction_max": _round_or_none(max(_values(summary_rows, "comm_agent_msg_drop_fraction_mean")) if _values(summary_rows, "comm_agent_msg_drop_fraction_mean") else None),
        "delivery_fraction_min": _round_or_none(min(_values(summary_rows, "comm_agent_msg_delivery_fraction_mean")) if _values(summary_rows, "comm_agent_msg_delivery_fraction_mean") else None),
        "negotiation_proposals_mean": _round_or_none(_mean(_values(summary_rows, "comm_negotiation_proposals_mean"))),
        "negotiation_acks_mean": _round_or_none(_mean(_values(summary_rows, "comm_negotiation_acks_mean"))),
    }
    observation = {
        "neighbors_mean": _round_or_none(_mean(_values(summary_rows, "obs_neighbors_mean"))),
        "v2v_fraction_mean": _round_or_none(_mean(_values(summary_rows, "obs_v2v_fraction_mean"))),
        "sensor_fraction_mean": _round_or_none(_mean(_values(summary_rows, "obs_sensor_fraction_mean"))),
        "stale_fraction_mean": _round_or_none(_mean(_values(summary_rows, "obs_stale_fraction_mean"))),
        "empty_fraction_mean": _round_or_none(_mean(_values(summary_rows, "obs_empty_fraction_mean"))),
    }

    review_report = {
        "schema_version": LEARNED_SUBMISSION_BUNDLE_REVIEW_SCHEMA_VERSION,
        "ok": ok,
        "recommendation": recommendation,
        "limitations": limitations,
        "method": validation.get("method"),
        "policy": validation.get("policy"),
        "suite": validation.get("suite"),
        "bundle_json": validation.get("bundle_json"),
        "bundle_root": validation.get("bundle_root"),
        "run_count": run_count,
        "planned_run_count": planned_run_count,
        "max_runs": max_runs,
        "summary_row_count": len(summary_rows),
        "result_row_count": len(results_rows),
        "score_v0": {
            "mean": _round_or_none(_mean(scores)),
            "worst": _round_or_none(max(scores) if scores else None),
            "best": _round_or_none(min(scores) if scores else None),
            "rows": scored_rows,
        },
        "submission_manifest": {
            "schema_version": manifest_payload.get("schema_version") if manifest_payload else None,
            "legacy_missing": legacy_manifest_missing,
            "policy": manifest_payload.get("policy") if manifest_payload else None,
            "benchmark": manifest_payload.get("benchmark") if manifest_payload else None,
            "dependencies": manifest_payload.get("dependencies") if manifest_payload else None,
            "training_disclosure": manifest_payload.get("training_disclosure") if manifest_payload else None,
            "inference_disclosure": manifest_payload.get("inference_disclosure") if manifest_payload else None,
            "review_notes": manifest_payload.get("review_notes") if manifest_payload else None,
            "unknown_fields": unknown_manifest_fields,
        },
        "dimensions": {
            "safety": safety,
            "mission": mission,
            "compute": compute,
            "communication": communication,
            "observation": observation,
        },
        "validation": validation,
        "checks": checks,
        "csv_errors": csv_errors,
    }
    review_schema_errors = _schema_errors(review_report, LEARNED_BUNDLE_REVIEW_SCHEMA_FILE)
    schema_check = _check(
        "learned_bundle_review_schema_valid",
        not review_schema_errors,
        {"schema_file": LEARNED_BUNDLE_REVIEW_SCHEMA_FILE, "errors": review_schema_errors[:20]},
    )
    checks.append(schema_check)
    if not schema_check["ok"] and "review_schema_invalid" not in limitations:
        limitations.append("review_schema_invalid")
    review_report["checks"] = checks
    review_report["limitations"] = limitations
    review_report["ok"] = all(check["ok"] for check in checks)
    if not review_report["ok"]:
        review_report["recommendation"] = "fix_artifacts"
    return review_report
