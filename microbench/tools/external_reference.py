from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import yaml

from microbench.config import resolve_config_path
from microbench.metrics.io import RESULT_FIELDS, SUMMARY_FIELDS, write_result_schema_manifest
from microbench.planners import BASELINE_FIDELITY_TIERS, canonical_method, list_methods
from microbench.scenarios import list_official_suites, materialize_official_suite


EXTERNAL_REFERENCE_SCHEMA_VERSION = "0.1"
EXTERNAL_REFERENCE_BUNDLE_SCHEMA_VERSION = "0.1"
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
DEFAULT_EXTERNAL_REFERENCE_SCENARIOS = (
    "urban_conflict_3d",
    "urban_throughput_3d",
    "stacked_swap_3d",
)
DEFAULT_EXTERNAL_REFERENCE_N_AGENTS = (4, 8)
DEFAULT_EXTERNAL_REFERENCE_SEEDS = (2,)
DEFAULT_EXTERNAL_REFERENCE_COMM_PROFILES = ("realistic_v2v_50hz",)


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


def _write_csv(path: Path, *, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _as_strings(values: tuple[str, ...] | list[str] | None, default: tuple[str, ...]) -> list[str]:
    return [str(v).strip() for v in (values if values is not None else default) if str(v).strip()]


def _as_ints(values: tuple[int, ...] | list[int] | None, default: tuple[int, ...]) -> list[int]:
    source = values if values is not None else default
    return [int(v) for v in source]


def _scenario_source(identifier: str) -> Path:
    raw = str(identifier).strip()
    candidate = resolve_config_path(raw)
    if Path(candidate).exists():
        return Path(candidate)
    if not raw.endswith((".yaml", ".yml")):
        scenario_candidate = resolve_config_path(f"config/scenarios/{raw}.yaml")
        if Path(scenario_candidate).exists():
            return Path(scenario_candidate)
    raise FileNotFoundError(f"Unknown scenario id/path for external-reference bundle: {identifier}")


def _method_family_label(method_family: str) -> str:
    clean = str(method_family).strip()
    if clean.lower() == RMADER_METHOD_FAMILY:
        return "RMADER"
    return clean


def _related_method(method_family: str, related_microbench_method: str | None) -> str:
    if related_microbench_method:
        return canonical_method(str(related_microbench_method))
    clean = str(method_family).strip().lower()
    try:
        return canonical_method(clean)
    except Exception:
        return clean


def _default_source_url(method_family: str) -> str:
    clean = str(method_family).strip().lower()
    if clean == RMADER_METHOD_FAMILY:
        return "https://github.com/mit-acl/rmader"
    return "<external-source-url>"


def _default_license(method_family: str) -> str:
    clean = str(method_family).strip().lower()
    if clean == RMADER_METHOD_FAMILY:
        return "BSD-3-Clause"
    return "<external-license>"


def _default_implementation_name(method_family: str) -> str:
    clean = str(method_family).strip().lower()
    if clean == RMADER_METHOD_FAMILY:
        return "MIT ACL RMADER"
    return f"{_method_family_label(method_family)} external reference"


def _default_reference_id(method_family: str) -> str:
    clean = str(method_family).strip().lower().replace("-", "_")
    if clean == RMADER_METHOD_FAMILY:
        return "rmader_official_ros_noetic"
    return f"{clean}_external_reference"


def _rmader_method_claims(method_family: str) -> dict[str, Any] | None:
    if str(method_family).strip().lower() != RMADER_METHOD_FAMILY:
        return None
    return {
        "decentralized_asynchronous_planning": True,
        "communication_delay_robustness": True,
        "delay_check": True,
        "two_step_trajectory_publication": True,
        "trajectory_storing_and_checking": True,
        "minvo_interval_polyhedra": True,
        "hard_separating_hyperplanes": True,
        "dynamic_obstacles": True,
        "static_obstacles": True,
        "solver_backend": "Gurobi",
    }


def _rmader_publication_urls(method_family: str) -> list[str]:
    if str(method_family).strip().lower() != RMADER_METHOD_FAMILY:
        return []
    return [
        "https://arxiv.org/abs/2303.06222",
        "https://doi.org/10.1109/ICRA48891.2023.10161244",
        "https://acl.mit.edu/projects/real-world-multi-agent-trajectory-planning",
    ]


def _adapter_template(method_family: str) -> dict[str, Any]:
    label = _method_family_label(method_family)
    out: dict[str, Any] = {
        "scenario_mapping": (
            "Map bundled DAA Microbench scenario YAMLs to external starts, goals, world bounds, static obstacles, "
            "dynamic obstacles/intruders, and timing parameters."
        ),
        "agent_authority_mapping": f"Run one {label} planner authority per drone; no centralized optimizer may choose multi-agent actions.",
        "observation_mapping": (
            "Each external planner receives only the ego state plus tracks/intents allowed by the declared Microbench "
            "sensing and V2V impairment filters."
        ),
        "communication_mapping": (
            "Map each Microbench communication profile row to the external transport's delay, jitter, loss, rate, "
            "and stale-belief behavior."
        ),
        "obstacle_mapping": "Map Microbench AABB/static/dynamic obstacle declarations to the external planner's obstacle representation.",
        "output_mapping": "Convert external trajectories/events into DAA Microbench results.csv rows and optional trace/MCAP artifacts.",
    }
    if str(method_family).strip().lower() == RMADER_METHOD_FAMILY:
        out["known_differences"] = [
            "External RMADER may use its original trajectory-command stack, while the built-in rmader baseline returns the first velocity command under the Microbench local-planner contract.",
            "Any perception, map, or communication assumption not present in the Microbench scenario must be disclosed in RUN_NOTES.md.",
        ]
    return out


def _manifest_template(
    *,
    method_family: str,
    reference_id: str,
    related_microbench_method: str,
    implementation_name: str,
    source_url: str,
    license_text: str,
    upstream_commit: str,
    runner_type: str,
    runner_command: str,
    runner_working_dir: str,
) -> dict[str, Any]:
    implementation: dict[str, Any] = {
        "name": implementation_name,
        "source_url": source_url,
        "license": license_text,
        "commit": upstream_commit,
        "dependency_notes": "External dependencies are kept outside the core Python package.",
    }
    publication_urls = _rmader_publication_urls(method_family)
    if publication_urls:
        implementation["publication_urls"] = publication_urls

    manifest: dict[str, Any] = {
        "schema_version": EXTERNAL_REFERENCE_SCHEMA_VERSION,
        "reference_id": reference_id,
        "method_family": _method_family_label(method_family),
        "related_microbench_method": related_microbench_method,
        "fidelity": "official_implementation",
        "implementation": implementation,
        "runner": {
            "type": runner_type,
            "command": runner_command,
            "working_dir": runner_working_dir,
        },
        "contract": {
            "input_format": "daa_microbench_external_reference_bundle",
            "output_format": "daa_microbench_results_csv",
            "uses_microbench_scenarios": True,
            "privileged_information": False,
            "decentralized_authority": True,
            "notes": "External process must only use each agent's allowed local observations/intents for each run-matrix row.",
        },
        "adapter": _adapter_template(method_family),
        "artifacts": {
            "results_csv": "results.csv",
            "summary_csv": "summary.csv",
            "mcap": "baseline_comparison.mcap",
            "notes": "RUN_NOTES.md",
        },
    }
    claims = _rmader_method_claims(method_family)
    if claims is not None:
        manifest["method_claims"] = claims
    return manifest


def _run_notes(
    *,
    method_family: str,
    reference_id: str,
    matrix_rows: list[dict[str, Any]],
    validation_command: str,
) -> str:
    return (
        f"# External Reference Run Notes: {reference_id}\n\n"
        f"Method family: `{_method_family_label(method_family)}`\n\n"
        "Fill this file before treating the external results as comparable evidence.\n\n"
        "## Required Disclosures\n\n"
        "- External repository URL and exact commit/tag:\n"
        "- External build environment, OS, ROS/Docker/runtime versions:\n"
        "- Solver backend and license status:\n"
        "- Hardware used for timing columns:\n"
        "- Adapter used to consume `run_matrix.csv` and `scenarios/*.yaml`:\n"
        "- Any deviation from Microbench sensing, communication, dynamics, obstacle, or authority contracts:\n"
        "- Any rows skipped, retried, timed out, or manually inspected:\n\n"
        "## Bundle Summary\n\n"
        f"- Planned rows: {len(matrix_rows)}\n"
        "- Scenario files: `scenarios/*.yaml`\n"
        "- Run matrix: `run_matrix.csv` and `run_matrix.json`\n"
        "- Expected output schema: `result_schema.json`, `results_template.csv`, `summary_template.csv`\n\n"
        "## Validation\n\n"
        "After the external stack writes `results.csv` and the declared artifacts, run:\n\n"
        f"```bash\n{validation_command} --require-artifacts --require-pass\n```\n"
    )


def _checksum_entries(out: Path, paths: list[Path]) -> list[dict[str, Any]]:
    entries = []
    for path in sorted(paths, key=lambda p: _rel(p, out)):
        entries.append(
            {
                "path": _rel(path, out),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    return entries


def build_external_reference_bundle(
    *,
    method_family: str,
    out_dir: str | Path,
    scenarios: tuple[str, ...] | list[str] | None = None,
    suites: tuple[str, ...] | list[str] | None = None,
    n_agents: tuple[int, ...] | list[int] | None = None,
    seeds: tuple[int, ...] | list[int] | None = None,
    comm_profiles: tuple[str, ...] | list[str] | None = None,
    reference_id: str | None = None,
    related_microbench_method: str | None = None,
    implementation_name: str | None = None,
    source_url: str | None = None,
    license_text: str | None = None,
    upstream_commit: str = "<external-repo-commit>",
    runner_type: str = "external_process",
    runner_command: str = "<external command that consumes run_matrix.csv and scenarios/*.yaml>",
    runner_working_dir: str = "<external-workspace>",
    overwrite: bool = False,
) -> dict[str, Any]:
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{out} already exists and is not empty; pass overwrite=True to replace bundle files")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    method_family_clean = str(method_family).strip()
    if not method_family_clean:
        raise ValueError("method_family must be nonempty")
    related = _related_method(method_family_clean, related_microbench_method)
    ref_id = str(reference_id or _default_reference_id(method_family_clean)).strip()
    runner_type_clean = str(runner_type).strip()
    if runner_type_clean not in EXTERNAL_REFERENCE_RUNNER_TYPES:
        raise ValueError(f"unknown runner type: {runner_type_clean}")

    scenario_out = out / "scenarios"
    if scenario_out.exists() and overwrite:
        shutil.rmtree(scenario_out)
    scenario_out.mkdir(parents=True, exist_ok=True)

    copied_scenarios: dict[str, Path] = {}
    scenario_values = _as_strings(scenarios, DEFAULT_EXTERNAL_REFERENCE_SCENARIOS)
    for value in scenario_values:
        src = _scenario_source(value)
        dest = scenario_out / src.name
        shutil.copyfile(src, dest)
        copied_scenarios[dest.stem] = dest

    suite_infos: list[dict[str, Any]] = []
    for suite_id in _as_strings(suites, ()):
        if suite_id not in list_official_suites():
            raise ValueError(f"unknown official generated suite for external-reference bundle: {suite_id}")
        suite_dir = scenario_out / f"suite_{suite_id}"
        if suite_dir.exists() and overwrite:
            shutil.rmtree(suite_dir)
        generated = materialize_official_suite(suite_id, suite_dir, overwrite=True)
        scenario_paths = [Path(p) for p in generated["scenario_paths"]]
        suite_infos.append(
            {
                "suite": suite_id,
                "suite_manifest": _rel(Path(generated["manifest_path"]), out),
                "scenario_count": len(scenario_paths),
            }
        )
        for path in scenario_paths:
            key = path.stem
            copied_scenarios[key] = path

    if not copied_scenarios:
        raise ValueError("external-reference bundle must include at least one scenario")

    n_values = _as_ints(n_agents, DEFAULT_EXTERNAL_REFERENCE_N_AGENTS)
    seed_values = _as_ints(seeds, DEFAULT_EXTERNAL_REFERENCE_SEEDS)
    comm_values = _as_strings(comm_profiles, DEFAULT_EXTERNAL_REFERENCE_COMM_PROFILES)

    matrix_rows: list[dict[str, Any]] = []
    for scenario_name, scenario_path in sorted(copied_scenarios.items()):
        for n_value in n_values:
            for seed in seed_values:
                for comm in comm_values:
                    run_id = f"{ref_id}_{scenario_name}_n{n_value}_seed{seed}_comm_{comm}"
                    matrix_rows.append(
                        {
                            "run_id": run_id,
                            "reference_id": ref_id,
                            "method_family": _method_family_label(method_family_clean),
                            "related_microbench_method": related,
                            "scenario": scenario_name,
                            "scenario_path": _rel(scenario_path, out),
                            "N": int(n_value),
                            "seed": int(seed),
                            "comm_profile": comm,
                            "expected_results_method": ref_id,
                        }
                    )

    matrix_fields = [
        "run_id",
        "reference_id",
        "method_family",
        "related_microbench_method",
        "scenario",
        "scenario_path",
        "N",
        "seed",
        "comm_profile",
        "expected_results_method",
    ]
    matrix_csv = out / "run_matrix.csv"
    _write_csv(matrix_csv, fieldnames=matrix_fields, rows=matrix_rows)
    matrix_json = out / "run_matrix.json"
    matrix_json.write_text(json.dumps(matrix_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = _manifest_template(
        method_family=method_family_clean,
        reference_id=ref_id,
        related_microbench_method=related,
        implementation_name=str(implementation_name or _default_implementation_name(method_family_clean)),
        source_url=str(source_url or _default_source_url(method_family_clean)),
        license_text=str(license_text or _default_license(method_family_clean)),
        upstream_commit=str(upstream_commit),
        runner_type=runner_type_clean,
        runner_command=str(runner_command),
        runner_working_dir=str(runner_working_dir),
    )
    manifest_path = out / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    schema_path = write_result_schema_manifest(out)
    result_template_rows = [
        {
            "run_id": row["run_id"],
            "method": row["expected_results_method"],
            "scenario": row["scenario"],
            "comm_profile": row["comm_profile"],
            "N": row["N"],
            "seed": row["seed"],
        }
        for row in matrix_rows
    ]
    results_template = out / "results_template.csv"
    _write_csv(results_template, fieldnames=RESULT_FIELDS, rows=result_template_rows)
    summary_template = out / "summary_template.csv"
    _write_csv(summary_template, fieldnames=SUMMARY_FIELDS, rows=[])

    validation_command = f"python -m microbench.cli validate-external-reference --manifest {manifest_path}"
    run_notes = out / "RUN_NOTES.md"
    run_notes.write_text(
        _run_notes(
            method_family=method_family_clean,
            reference_id=ref_id,
            matrix_rows=matrix_rows,
            validation_command=validation_command,
        ),
        encoding="utf-8",
    )

    checksum_targets = [
        *copied_scenarios.values(),
        *(scenario_out.glob("suite_*/suite_manifest.yaml")),
        matrix_csv,
        matrix_json,
        manifest_path,
        schema_path,
        results_template,
        summary_template,
        run_notes,
    ]
    checksums = {
        "schema_version": EXTERNAL_REFERENCE_BUNDLE_SCHEMA_VERSION,
        "algorithm": "sha256",
        "files": _checksum_entries(out, checksum_targets),
    }
    checksums_path = out / "checksums.json"
    checksums_path.write_text(json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    validation = validate_external_reference_manifest(manifest=manifest_path, require_artifacts=False)
    bundle_report = {
        "schema_version": EXTERNAL_REFERENCE_BUNDLE_SCHEMA_VERSION,
        "ok": bool(validation["ok"]),
        "reference_id": ref_id,
        "method_family": _method_family_label(method_family_clean),
        "related_microbench_method": related,
        "out_dir": str(out),
        "manifest": str(manifest_path),
        "validation_command": validation_command,
        "post_run_validation_command": f"{validation_command} --require-artifacts --require-pass",
        "run_matrix_csv": str(matrix_csv),
        "run_matrix_json": str(matrix_json),
        "run_count": len(matrix_rows),
        "scenario_count": len(copied_scenarios),
        "suites": suite_infos,
        "results_template": str(results_template),
        "summary_template": str(summary_template),
        "result_schema": str(schema_path),
        "checksums": str(checksums_path),
        "run_notes": str(run_notes),
        "validation": validation,
        "note": (
            "This bundle prepares official/dependency-heavy external implementation comparisons. "
            "It does not execute external code or claim that results.csv has been produced."
        ),
    }
    report_path = out / "external_reference_bundle.json"
    bundle_report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(bundle_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return bundle_report


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
