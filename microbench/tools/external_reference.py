from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml

from microbench.planners import BASELINE_FIDELITY_TIERS, canonical_method, list_methods


EXTERNAL_REFERENCE_SCHEMA_VERSION = "0.1"
EXTERNAL_REFERENCE_RUNNER_TYPES = ("external_process", "docker", "ros", "manual_import")
RMADER_METHOD_FAMILY = "rmader"
RMADER_REQUIRED_METHOD_CLAIMS = (
    "decentralized_asynchronous_planning",
    "communication_delay_robustness",
    "delay_check",
    "two_step_trajectory_publication",
    "trajectory_storing_and_checking",
    "minvo_interval_polyhedra",
    "hard_separating_hyperplanes",
    "dynamic_obstacles",
    "static_obstacles",
)
RMADER_REQUIRED_ADAPTER_FIELDS = (
    "scenario_mapping",
    "agent_authority_mapping",
    "observation_mapping",
    "communication_mapping",
    "obstacle_mapping",
    "output_mapping",
)
REQUIRED_TOP_LEVEL_FIELDS = (
    "schema_version",
    "reference_id",
    "method_family",
    "related_microbench_method",
    "fidelity",
    "implementation",
    "runner",
    "contract",
    "artifacts",
)
REQUIRED_RESULT_FIELDS = (
    "method",
    "scenario",
    "N",
    "seed",
    "collision_episode",
    "completion_rate",
)


def _read_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    raw = manifest_path.read_text(encoding="utf-8")
    if manifest_path.suffix.lower() == ".json":
        data = json.loads(raw)
    else:
        data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError("external reference manifest must be a mapping")
    return data


def _rel_or_abs(path: str | Path, *, base: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else base / p


def _bool_field(section: dict[str, Any], key: str) -> bool | None:
    value = section.get(key)
    if isinstance(value, bool):
        return value
    return None


def _nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _is_rmader_reference(data: dict[str, Any], related_method: str) -> bool:
    return str(data.get("method_family", "")).strip().lower() == RMADER_METHOD_FAMILY or related_method == RMADER_METHOD_FAMILY


def _csv_fields(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or [])


def validate_external_reference_manifest(
    *,
    manifest: str | Path,
    require_artifacts: bool = False,
) -> dict[str, Any]:
    manifest_path = Path(manifest)
    base = manifest_path.parent
    errors: list[str] = []
    warnings: list[str] = []
    data = _read_manifest(manifest_path)

    for field in REQUIRED_TOP_LEVEL_FIELDS:
        if field not in data:
            errors.append(f"missing_top_level_field:{field}")

    schema_version = str(data.get("schema_version", ""))
    if schema_version != EXTERNAL_REFERENCE_SCHEMA_VERSION:
        errors.append(f"unsupported_schema_version:{schema_version}")

    fidelity = str(data.get("fidelity", ""))
    if fidelity not in BASELINE_FIDELITY_TIERS:
        errors.append(f"unknown_fidelity:{fidelity}")
    if fidelity == "official_implementation":
        implementation = data.get("implementation", {})
        if not isinstance(implementation, dict):
            errors.append("implementation_not_mapping")
        else:
            if not str(implementation.get("source_url", "")).strip():
                errors.append("official_implementation_missing_source_url")
            if not str(implementation.get("license", "")).strip():
                errors.append("official_implementation_missing_license")
            if not str(implementation.get("commit", "")).strip():
                warnings.append("official_implementation_missing_commit")

    known_methods = set(list_methods(include_aliases=True))
    related_method = canonical_method(str(data.get("related_microbench_method", "")))
    if related_method not in known_methods:
        errors.append(f"unknown_related_microbench_method:{data.get('related_microbench_method')}")

    runner = data.get("runner", {})
    if not isinstance(runner, dict):
        errors.append("runner_not_mapping")
    else:
        runner_type = str(runner.get("type", ""))
        if runner_type not in EXTERNAL_REFERENCE_RUNNER_TYPES:
            errors.append(f"unknown_runner_type:{runner_type}")
        if not str(runner.get("command", "")).strip():
            warnings.append("runner_command_missing")

    contract = data.get("contract", {})
    if not isinstance(contract, dict):
        errors.append("contract_not_mapping")
    else:
        if _bool_field(contract, "privileged_information") is not False:
            errors.append("contract_must_declare_no_privileged_information")
        if _bool_field(contract, "uses_microbench_scenarios") is not True:
            errors.append("contract_must_use_microbench_scenarios")
        if _bool_field(contract, "decentralized_authority") is None:
            warnings.append("contract_decentralized_authority_not_declared")
        if not str(contract.get("output_format", "")).strip():
            warnings.append("contract_output_format_missing")

    rmader_claim_status: dict[str, Any] = {}
    if fidelity == "official_implementation" and _is_rmader_reference(data, related_method):
        method_claims = data.get("method_claims", {})
        if not isinstance(method_claims, dict):
            errors.append("rmader_method_claims_not_mapping")
            method_claims = {}
        for claim in RMADER_REQUIRED_METHOD_CLAIMS:
            value = _bool_field(method_claims, claim)
            rmader_claim_status[claim] = value
            if value is not True:
                errors.append(f"rmader_required_claim_missing_or_false:{claim}")
        solver_backend = str(method_claims.get("solver_backend", "")).strip().lower()
        rmader_claim_status["solver_backend"] = solver_backend or None
        if not solver_backend:
            warnings.append("rmader_solver_backend_not_declared")
        elif "gurobi" not in solver_backend:
            warnings.append(f"rmader_solver_backend_not_gurobi:{solver_backend}")

        adapter = data.get("adapter", {})
        if not isinstance(adapter, dict):
            errors.append("rmader_adapter_not_mapping")
            adapter = {}
        adapter_status = {}
        for field in RMADER_REQUIRED_ADAPTER_FIELDS:
            present = _nonempty(adapter.get(field))
            adapter_status[field] = present
            if not present:
                errors.append(f"rmader_adapter_field_missing:{field}")
        rmader_claim_status["adapter_fields"] = adapter_status

    artifacts = data.get("artifacts", {})
    artifact_status: dict[str, Any] = {}
    if not isinstance(artifacts, dict):
        errors.append("artifacts_not_mapping")
    else:
        results_csv = artifacts.get("results_csv")
        if require_artifacts and not results_csv:
            errors.append("artifacts_results_csv_required")
        for name, value in sorted(artifacts.items()):
            if value in (None, ""):
                continue
            path = _rel_or_abs(str(value), base=base)
            exists = path.exists()
            artifact_status[name] = {"path": str(path), "exists": exists}
            if require_artifacts and not exists:
                errors.append(f"artifact_missing:{name}")
            elif not exists:
                warnings.append(f"artifact_not_found:{name}")
        if results_csv:
            results_path = _rel_or_abs(str(results_csv), base=base)
            if results_path.exists():
                fields = _csv_fields(results_path)
                missing_fields = [field for field in REQUIRED_RESULT_FIELDS if field not in fields]
                artifact_status.setdefault("results_csv", {})["fields"] = fields
                if missing_fields:
                    errors.append("results_csv_missing_fields:" + ",".join(missing_fields))

    report = {
        "schema_version": EXTERNAL_REFERENCE_SCHEMA_VERSION,
        "ok": not errors,
        "manifest": str(manifest_path),
        "reference_id": data.get("reference_id"),
        "method_family": data.get("method_family"),
        "related_microbench_method": related_method,
        "fidelity": fidelity,
        "runner_type": runner.get("type") if isinstance(runner, dict) else None,
        "require_artifacts": bool(require_artifacts),
        "artifact_status": artifact_status,
        "method_claim_status": rmader_claim_status,
        "errors": errors,
        "warnings": warnings,
        "note": (
            "External reference manifests describe how an official or dependency-heavy implementation "
            "was run against DAA Microbench scenarios. The validator does not execute external code."
        ),
    }
    return report
