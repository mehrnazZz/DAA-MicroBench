from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from microbench.acceptance import check_acceptance
from microbench.metrics import append_result, write_summary
from microbench.runner import run_episode
from microbench.scenarios import materialize_official_suite, suite_defaults
from microbench.tools.baseline_audit import build_baseline_audit
from microbench.tools.baseline_behavior import run_baseline_behavior_smoke
from microbench.types import RunSpec


BASELINE_PROMOTION_SCHEMA_VERSION = "0.1"
PROMOTION_METHODS = ("cbf_qp", "mpc_local", "negotiation_yield")
EXPERIMENTAL_SUITE = "official_experimental_baselines"
EXPERIMENTAL_SUITE_METHODS = ("cbf_qp", "mpc_local")
METHOD_SIGNAL_CHECKS = {
    "cbf_qp": "cbf_qp_debug_contract",
    "mpc_local": "mpc_local_debug_contract",
    "negotiation_yield": "negotiation_yield_signal",
}
REFERENCE_ROLES = {"reference_baseline", "agentic_reference_baseline"}


def _as_list(values: tuple[str, ...] | list[str] | None, default: tuple[str, ...]) -> list[str]:
    return [str(v).strip() for v in (values if values is not None else default) if str(v).strip()]


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _check_by_name(report: dict[str, Any], name: str) -> dict[str, Any] | None:
    for check in report.get("checks", []):
        if check.get("name") == name:
            return check
    return None


def _check_ok(report: dict[str, Any], name: str) -> bool:
    check = _check_by_name(report, name)
    return bool(check and check.get("ok"))


def _method_rows(report: dict[str, Any], method: str) -> list[dict[str, Any]]:
    return [row for row in report.get("rows", []) if row.get("method") == method]


def _metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    collision_episodes = sum(1 for row in rows if (_to_float(row.get("collision_episode")) or 0.0) > 0.0)
    guardrail_total = 0.0
    planner_p95: list[float] = []
    min_sep: list[float] = []
    for row in rows:
        for field in ("planner_timeout_count", "planner_error_count", "planner_fallback_count"):
            guardrail_total += _to_float(row.get(field)) or 0.0
        value = _to_float(row.get("planner_ms_per_tick_per_agent_p95"))
        if value is not None:
            planner_p95.append(value)
        sep = _to_float(row.get("min_sep_min_m"))
        if sep is not None:
            min_sep.append(sep)
    return {
        "rows": len(rows),
        "collision_episode_count": int(collision_episodes),
        "guardrail_total": int(guardrail_total),
        "planner_p95_max_ms": max(planner_p95) if planner_p95 else None,
        "min_sep_min_m": min(min_sep) if min_sep else None,
    }


def _run_experimental_suite(*, out_dir: Path, methods: list[str]) -> dict[str, Any] | None:
    suite_methods = [method for method in methods if method in EXPERIMENTAL_SUITE_METHODS]
    if not suite_methods:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "results.csv").exists():
        raise RuntimeError(f"baseline promotion output already exists: {out_dir / 'results.csv'}")

    generated = materialize_official_suite(
        EXPERIMENTAL_SUITE,
        out_dir / "_generated_scenarios" / EXPERIMENTAL_SUITE,
        overwrite=True,
    )
    defaults = suite_defaults(EXPERIMENTAL_SUITE)
    n_agents = int(defaults["n_agents"][0])
    seed = int(defaults["seeds"][0])
    comm_profile = str(defaults["comm_profiles"][0])

    rows: list[dict[str, Any]] = []
    for scenario_path in generated["scenario_paths"]:
        for method in suite_methods:
            spec = RunSpec(
                scenario_path=str(scenario_path),
                method=method,
                n_agents=n_agents,
                seed=seed,
                comm_profile=comm_profile,
                out_dir=str(out_dir),
                save_trace=False,
            )
            row = run_episode(spec)
            append_result(out_dir, row)
            rows.append(row)
    summary_csv = write_summary(out_dir)

    acceptance = check_acceptance(
        summary_csv=summary_csv,
        results_csv=out_dir / "results.csv",
        suite_manifest=generated["manifest_path"],
        methods=suite_methods,
    )

    return {
        "suite": EXPERIMENTAL_SUITE,
        "methods": suite_methods,
        "run_count": len(rows),
        "results_csv": str(out_dir / "results.csv"),
        "summary_csv": str(summary_csv),
        "suite_manifest": str(generated["manifest_path"]),
        "acceptance": acceptance,
        "rows": [
            {
                "method": row.get("method"),
                "scenario": row.get("scenario"),
                "collision_episode": row.get("collision_episode"),
                "min_sep_min_m": row.get("min_sep_min_m"),
                "completion_rate": row.get("completion_rate"),
                "planner_ms_per_tick_per_agent_p95": row.get("planner_ms_per_tick_per_agent_p95"),
                "planner_timeout_count": row.get("planner_timeout_count"),
                "planner_error_count": row.get("planner_error_count"),
                "planner_fallback_count": row.get("planner_fallback_count"),
            }
            for row in rows
        ],
    }


def _method_acceptance(experimental_suite: dict[str, Any] | None, method: str) -> dict[str, Any] | None:
    if not experimental_suite or method not in EXPERIMENTAL_SUITE_METHODS:
        return None
    return check_acceptance(
        summary_csv=experimental_suite["summary_csv"],
        results_csv=experimental_suite["results_csv"],
        suite_manifest=experimental_suite["suite_manifest"],
        methods=[method],
    )


def _stable_v1_blockers(
    *,
    audit_entry: dict[str, Any],
    behavior_metrics: dict[str, Any],
    method_acceptance: dict[str, Any] | None,
) -> list[str]:
    blockers: list[str] = []
    if str(audit_entry.get("status")) != "stable":
        blockers.append("metadata_status_not_stable")
    if str(audit_entry.get("role")) not in REFERENCE_ROLES:
        blockers.append("role_not_reference_baseline")
    if behavior_metrics["collision_episode_count"] > 0:
        blockers.append("smoke_collision_episode_present")
    if method_acceptance is not None and method_acceptance.get("status") != "PASS":
        blockers.append("experimental_suite_acceptance_not_pass")

    blockers.append("stable_3d_stress_acceptance_bands_missing")
    blockers.append("degraded_comm_or_sensor_calibration_missing")
    return blockers


def run_baseline_promotion_calibration(
    *,
    out_dir: str | Path,
    root: str | Path = ".",
    methods: tuple[str, ...] | list[str] | None = None,
    behavior_report: str | Path | dict[str, Any] | None = None,
    include_experimental_suite: bool = True,
) -> dict[str, Any]:
    out = Path(out_dir)
    methods_list = _as_list(methods, PROMOTION_METHODS)
    unknown = sorted(set(methods_list) - set(PROMOTION_METHODS))
    if unknown:
        raise ValueError(f"Unknown promotion baseline(s): {','.join(unknown)}")

    if behavior_report is None:
        behavior = run_baseline_behavior_smoke(out_dir=out / "behavior_smoke", methods=methods_list)
        behavior_source = "generated"
    elif isinstance(behavior_report, dict):
        behavior = behavior_report
        behavior_source = "provided"
    else:
        behavior = json.loads(Path(behavior_report).read_text(encoding="utf-8"))
        behavior_source = str(behavior_report)

    experimental = (
        _run_experimental_suite(out_dir=out / "experimental_baselines", methods=methods_list)
        if include_experimental_suite
        else None
    )

    audit = build_baseline_audit(root=root)
    audit_by_method = {entry["method"]: entry for entry in audit["methods"]}

    method_entries: list[dict[str, Any]] = []
    for method in methods_list:
        audit_entry = audit_by_method.get(method, {})
        behavior_rows = _method_rows(behavior, method)
        behavior_metrics = _metric_summary(behavior_rows)
        method_signal = METHOD_SIGNAL_CHECKS[method]
        method_acceptance = _method_acceptance(experimental, method)

        audit_checks = audit_entry.get("checks", {})
        calibration_checks = {
            "factory_constructible": bool(audit_checks.get("factory_constructible")),
            "docs_mentioned": bool(audit_checks.get("docs_mentioned")),
            "tests_mentioned": bool(audit_checks.get("tests_mentioned")),
            "supports_2d": bool(audit_checks.get("supports_2d")),
            "supports_3d": bool(audit_checks.get("supports_3d")),
            "behavior_rows_present": bool(behavior_rows),
            "behavior_finite_metrics": _check_ok(behavior, "finite_key_metrics"),
            "behavior_zero_guardrails": _check_ok(behavior, "zero_planner_guardrails")
            and behavior_metrics["guardrail_total"] == 0,
            "behavior_2d_3d_coverage": _check_ok(behavior, "two_d_and_three_d_coverage"),
            "method_signal_contract": _check_ok(behavior, method_signal),
            "experimental_suite_acceptance_pass": (
                True if method not in EXPERIMENTAL_SUITE_METHODS else bool(method_acceptance and method_acceptance["status"] == "PASS")
            ),
        }
        calibration_blockers = [key for key, ok in calibration_checks.items() if not ok]
        stable_blockers = _stable_v1_blockers(
            audit_entry=audit_entry,
            behavior_metrics=behavior_metrics,
            method_acceptance=method_acceptance,
        )

        method_entries.append(
            {
                "method": method,
                "role": audit_entry.get("role"),
                "status": audit_entry.get("status"),
                "readiness": audit_entry.get("readiness"),
                "calibration_ready": not calibration_blockers,
                "stable_v1_ready": not calibration_blockers and not stable_blockers,
                "calibration_checks": calibration_checks,
                "calibration_blockers": calibration_blockers,
                "stable_v1_blockers": stable_blockers,
                "behavior_metrics": behavior_metrics,
                "behavior_signal_check": method_signal,
                "experimental_acceptance_status": method_acceptance.get("status") if method_acceptance else None,
                "experimental_acceptance_rules_passed": method_acceptance.get("rules_passed") if method_acceptance else None,
                "experimental_acceptance_rules_warned": method_acceptance.get("rules_warned") if method_acceptance else None,
                "experimental_acceptance_rules_failed": method_acceptance.get("rules_failed") if method_acceptance else None,
            }
        )

    report = {
        "schema_version": BASELINE_PROMOTION_SCHEMA_VERSION,
        "methods": methods_list,
        "public_alpha_calibrated": all(entry["calibration_ready"] for entry in method_entries),
        "stable_v1_ready": all(entry["stable_v1_ready"] for entry in method_entries),
        "summary": {
            "method_count": len(method_entries),
            "calibration_ready_count": sum(1 for entry in method_entries if entry["calibration_ready"]),
            "stable_v1_ready_count": sum(1 for entry in method_entries if entry["stable_v1_ready"]),
            "stable_v1_blockers": {
                entry["method"]: entry["stable_v1_blockers"]
                for entry in method_entries
                if entry["stable_v1_blockers"]
            },
        },
        "behavior_smoke": {
            "source": behavior_source,
            "ok": bool(behavior.get("ok")),
            "run_count": behavior.get("run_count"),
            "results_csv": behavior.get("results_csv"),
            "summary_csv": behavior.get("summary_csv"),
            "suite_manifest": behavior.get("suite_manifest"),
        },
        "experimental_suite": {
            "included": experimental is not None,
            "status": experimental.get("acceptance", {}).get("status") if experimental else None,
            "run_count": experimental.get("run_count") if experimental else 0,
            "results_csv": experimental.get("results_csv") if experimental else None,
            "summary_csv": experimental.get("summary_csv") if experimental else None,
            "suite_manifest": experimental.get("suite_manifest") if experimental else None,
        },
        "methods_detail": method_entries,
    }
    return report


def write_baseline_promotion_calibration(*, out_dir: str | Path, **kwargs: Any) -> Path:
    report = run_baseline_promotion_calibration(out_dir=out_dir, **kwargs)
    path = Path(out_dir) / "baseline_promotion.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
