from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from microbench.planners import make_planner, planner_metadata
from microbench.scenarios import suite_registry_dicts


BASELINE_AUDIT_SCHEMA_VERSION = "0.1"

PUBLIC_ALPHA_REFERENCE_METHODS = (
    "orca_heuristic",
    "orca_with_staleness",
    "priority_yield",
    "negotiation_yield",
)

EXPERIMENTAL_METHODS = (
    "cbf_qp",
    "mpc_local",
    "mpc_nonlinear",
    "dmpc_best_response",
    "bvc_tube_dmpc",
    "dynamic_tube_dmpc",
    "rmader",
    "ego_swarm_opt",
    "velocity_obstacle",
    "reciprocal_velocity_obstacle",
    "learned_tiny",
)

REFERENCE_ROLES = {"reference_baseline", "agentic_reference_baseline"}
ILLUSTRATIVE_ROLES = {"illustrative_baseline", "agentic_example", "developer_template"}
BRIDGE_ROLES = {"submission_bridge"}


def _read_many(paths: list[Path]) -> str:
    chunks: list[str] = []
    for path in paths:
        if path.exists() and path.is_file():
            chunks.append(path.read_text(encoding="utf-8"))
    return "\n".join(chunks)


def _repo_docs_text(root: Path) -> str:
    docs = [
        root / "README.md",
        root / "docs" / "BASELINES.md",
        root / "docs" / "DESIGN_V1.md",
        root / "docs" / "LEADERBOARD.md",
        root / "docs" / "PUBLIC_ALPHA_NOTES.md",
        root / "docs" / "SCENARIO_SUITES.md",
    ]
    return _read_many(docs)


def _repo_tests_text(root: Path) -> str:
    tests_dir = root / "tests"
    return _read_many(sorted(tests_dir.glob("test_*.py"))) if tests_dir.exists() else ""


def _suite_coverage() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    defaults: dict[str, list[str]] = {}
    acceptance: dict[str, list[str]] = {}
    for suite in suite_registry_dicts():
        suite_id = str(suite["suite"])
        methods = [str(m) for m in suite.get("default_methods", [])]
        rules = list(suite.get("acceptance", {}).get("rules", []))
        for method in methods:
            defaults.setdefault(method, []).append(suite_id)
            for rule in rules:
                rule_method = str(rule.get("method", ""))
                if rule_method in {method, "*"}:
                    acceptance.setdefault(method, []).append(f"{suite_id}:{rule.get('name')}")
    return defaults, acceptance


def _factory_ok(method: str) -> tuple[bool, str | None]:
    try:
        make_planner(method)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


def _classify_method(entry: dict[str, Any]) -> str:
    role = str(entry.get("role", ""))
    status = str(entry.get("status", ""))
    if role in REFERENCE_ROLES and status in {"stable", "pre_v1"}:
        return "public_alpha_reference"
    if role in ILLUSTRATIVE_ROLES:
        return "illustrative"
    if role in BRIDGE_ROLES:
        return "submission_bridge"
    if status == "experimental" or str(entry.get("method")) in EXPERIMENTAL_METHODS:
        return "experimental"
    return "uncategorized"


def _method_entry(
    *,
    entry: dict[str, Any],
    docs_text: str,
    tests_text: str,
    suite_defaults: dict[str, list[str]],
    suite_acceptance: dict[str, list[str]],
) -> dict[str, Any]:
    method = str(entry["method"])
    dimensions = tuple(str(d) for d in entry.get("dimensions", ()))
    docs_mentioned = method in docs_text
    tests_mentioned = method in tests_text
    default_suites = sorted(suite_defaults.get(method, []))
    acceptance_rules = sorted(suite_acceptance.get(method, []))
    category = _classify_method(entry)
    if category == "submission_bridge":
        factory_ok, factory_error = True, "requires external --policy-spec"
    else:
        factory_ok, factory_error = _factory_ok(method)

    checks = {
        "factory_constructible": factory_ok,
        "docs_mentioned": docs_mentioned,
        "tests_mentioned": tests_mentioned,
        "supports_2d": "2d" in dimensions,
        "supports_3d": "3d" in dimensions,
        "in_official_suite_defaults": bool(default_suites),
        "has_acceptance_coverage": bool(acceptance_rules),
    }

    blockers: list[str] = []
    for key in ("factory_constructible", "docs_mentioned", "tests_mentioned", "supports_2d", "supports_3d"):
        if not checks[key]:
            blockers.append(key)

    if category == "public_alpha_reference":
        if not checks["in_official_suite_defaults"]:
            blockers.append("in_official_suite_defaults")
        if not checks["has_acceptance_coverage"]:
            blockers.append("has_acceptance_coverage")

    stable_v1_blockers: list[str] = []
    if category == "experimental":
        stable_v1_blockers.append("experimental_status_not_reference_ready")
    if category == "public_alpha_reference" and str(entry.get("status")) != "stable":
        stable_v1_blockers.append("pre_v1_status_not_stable_v1")

    if blockers:
        readiness = "blocked"
    elif category == "public_alpha_reference":
        readiness = "public_alpha_reference_ready"
    elif category == "experimental":
        readiness = "experimental_runnable"
    elif category == "submission_bridge":
        readiness = "externally_configured_bridge"
    elif category == "illustrative":
        readiness = "illustrative_or_template"
    else:
        readiness = "uncategorized"

    return {
        "method": method,
        "display_name": entry.get("display_name"),
        "role": entry.get("role"),
        "status": entry.get("status"),
        "planner_type": entry.get("planner_type"),
        "dimensions": list(dimensions),
        "category": category,
        "readiness": readiness,
        "checks": checks,
        "suite_defaults": default_suites,
        "acceptance_rules": acceptance_rules,
        "blockers": blockers,
        "stable_v1_blockers": stable_v1_blockers,
        "factory_error": factory_error,
    }


def build_baseline_audit(*, root: str | Path = ".") -> dict[str, Any]:
    repo_root = Path(root)
    docs_text = _repo_docs_text(repo_root)
    tests_text = _repo_tests_text(repo_root)
    suite_defaults, suite_acceptance = _suite_coverage()
    metadata = [entry for entry in planner_metadata(include_aliases=False)]
    methods = [
        _method_entry(
            entry=entry,
            docs_text=docs_text,
            tests_text=tests_text,
            suite_defaults=suite_defaults,
            suite_acceptance=suite_acceptance,
        )
        for entry in metadata
    ]

    by_method = {entry["method"]: entry for entry in methods}
    missing_required = [m for m in PUBLIC_ALPHA_REFERENCE_METHODS if m not in by_method]
    public_alpha_blockers: list[str] = [f"missing_required_reference:{m}" for m in missing_required]
    for method in PUBLIC_ALPHA_REFERENCE_METHODS:
        entry = by_method.get(method)
        if entry and entry["blockers"]:
            public_alpha_blockers.append(f"{method}:{','.join(entry['blockers'])}")

    stable_v1_blockers = [
        f"{entry['method']}:{','.join(entry['stable_v1_blockers'])}"
        for entry in methods
        if entry["stable_v1_blockers"]
    ]

    return {
        "schema_version": BASELINE_AUDIT_SCHEMA_VERSION,
        "required_public_alpha_reference_methods": list(PUBLIC_ALPHA_REFERENCE_METHODS),
        "experimental_methods": list(EXPERIMENTAL_METHODS),
        "public_alpha_ready": not public_alpha_blockers,
        "stable_v1_ready": not stable_v1_blockers,
        "summary": {
            "method_count": len(methods),
            "public_alpha_reference_ready_count": sum(
                1 for entry in methods if entry["readiness"] == "public_alpha_reference_ready"
            ),
            "experimental_runnable_count": sum(1 for entry in methods if entry["readiness"] == "experimental_runnable"),
            "illustrative_or_template_count": sum(
                1 for entry in methods if entry["readiness"] == "illustrative_or_template"
            ),
            "public_alpha_blockers": public_alpha_blockers,
            "stable_v1_blockers": stable_v1_blockers,
        },
        "methods": methods,
    }


def write_baseline_audit(*, out: str | Path, root: str | Path = ".") -> Path:
    report = build_baseline_audit(root=root)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path
