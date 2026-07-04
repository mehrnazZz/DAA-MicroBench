from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from microbench.config import load_yaml


BLOCKING_SEVERITIES = {"required", "smoke"}
FILTER_FIELDS = ("method", "scenario", "comm_profile", "n_agents")


def _read_csv(path: str | Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")
    with p.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_filter_set(values: list[str] | tuple[str, ...] | None, *, numeric: bool = False) -> set[str]:
    if not values:
        return set()
    out: set[str] = set()
    for value in values:
        for part in str(value).split(","):
            p = part.strip()
            if not p:
                continue
            out.add(str(_to_int(p) if numeric and _to_int(p) is not None else p))
    return out


def _rule_value(rule: dict[str, Any], key: str) -> str:
    if key == "n_agents":
        return str(rule.get("n_agents", "*"))
    return str(rule.get(key, "*"))


def _row_value(row: dict[str, Any], key: str) -> str:
    if key == "n_agents":
        return str(row.get("N", row.get("n_agents", "")))
    return str(row.get(key, ""))


def _matches(value: str, pattern: str) -> bool:
    return pattern == "*" or value == pattern


def _rule_selected(rule: dict[str, Any], filters: dict[str, set[str]]) -> bool:
    for key, allowed in filters.items():
        if not allowed:
            continue
        target = _rule_value(rule, key)
        if target != "*" and target not in allowed:
            return False
    return True


def _row_selected(row: dict[str, Any], filters: dict[str, set[str]]) -> bool:
    for key, allowed in filters.items():
        if allowed and _row_value(row, key) not in allowed:
            return False
    return True


def _row_matches_rule(row: dict[str, Any], rule: dict[str, Any]) -> bool:
    return all(_matches(_row_value(row, key), _rule_value(rule, key)) for key in FILTER_FIELDS)


def _compare(observed: float, operator: str, threshold: float) -> bool:
    if operator == "<=":
        return observed <= threshold
    if operator == "<":
        return observed < threshold
    if operator == ">=":
        return observed >= threshold
    if operator == ">":
        return observed > threshold
    if operator == "==":
        return math.isclose(observed, threshold, rel_tol=1e-9, abs_tol=1e-12)
    if operator == "!=":
        return not math.isclose(observed, threshold, rel_tol=1e-9, abs_tol=1e-12)
    raise ValueError(f"Unsupported acceptance operator: {operator!r}")


def _blocking(rule: dict[str, Any]) -> bool:
    return str(rule.get("severity", "")).lower() in BLOCKING_SEVERITIES


def _failure_status(rule: dict[str, Any]) -> str:
    return "fail" if _blocking(rule) else "warn"


def _rule_summary(rule: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": rule.get("name"),
        "severity": rule.get("severity"),
        "scope": rule.get("scope", "summary"),
        "method": rule.get("method", "*"),
        "scenario": rule.get("scenario", "*"),
        "comm_profile": rule.get("comm_profile", "*"),
        "n_agents": rule.get("n_agents", "*"),
        "metric": rule.get("metric"),
        "operator": rule.get("operator"),
        "value": rule.get("value"),
        "description": rule.get("description", ""),
    }


def _evaluate_rule(
    rule: dict[str, Any],
    *,
    rows: list[dict[str, Any]],
    source_available: bool,
    filters: dict[str, set[str]],
) -> dict[str, Any]:
    check = _rule_summary(rule)

    if not _rule_selected(rule, filters):
        return {
            **check,
            "status": "skipped",
            "matched_rows": 0,
            "passed_rows": 0,
            "violations": [],
            "message": "rule skipped by CLI filter",
        }

    if not source_available:
        return {
            **check,
            "status": _failure_status(rule),
            "matched_rows": 0,
            "passed_rows": 0,
            "violations": [{"reason": f"{rule.get('scope', 'summary')} CSV unavailable"}],
            "message": "required CSV unavailable",
        }

    matching_rows = [row for row in rows if _row_selected(row, filters) and _row_matches_rule(row, rule)]
    if not matching_rows:
        return {
            **check,
            "status": _failure_status(rule),
            "matched_rows": 0,
            "passed_rows": 0,
            "violations": [{"reason": "no matching rows"}],
            "message": "no matching rows",
        }

    metric = str(rule.get("metric", ""))
    operator = str(rule.get("operator", ""))
    threshold = _to_float(rule.get("value"))
    violations: list[dict[str, Any]] = []
    passed_rows = 0

    for row in matching_rows:
        observed = _to_float(row.get(metric))
        key = {
            "method": row.get("method"),
            "scenario": row.get("scenario"),
            "comm_profile": row.get("comm_profile"),
            "N": row.get("N", row.get("n_agents")),
        }
        if observed is None or threshold is None:
            violations.append({**key, "observed": row.get(metric), "reason": "metric value is missing or non-finite"})
            continue
        try:
            passed = _compare(observed, operator, threshold)
        except ValueError as exc:
            violations.append({**key, "observed": observed, "reason": str(exc)})
            continue
        if passed:
            passed_rows += 1
        else:
            violations.append({**key, "observed": observed, "reason": "threshold violation"})

    if violations:
        status = _failure_status(rule)
        message = f"{len(violations)} row(s) violated rule"
    else:
        status = "pass"
        message = "all matching rows passed"

    return {
        **check,
        "status": status,
        "matched_rows": len(matching_rows),
        "passed_rows": passed_rows,
        "violations": violations,
        "message": message,
    }


def check_acceptance(
    *,
    summary_csv: str | Path,
    suite_manifest: str | Path,
    results_csv: str | Path | None = None,
    methods: list[str] | tuple[str, ...] | None = None,
    scenarios: list[str] | tuple[str, ...] | None = None,
    comm_profiles: list[str] | tuple[str, ...] | None = None,
    n_agents: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    manifest_path = Path(suite_manifest)
    manifest = load_yaml(manifest_path)
    acceptance = manifest.get("acceptance", {}) if isinstance(manifest, dict) else {}
    rules = acceptance.get("rules", []) if isinstance(acceptance, dict) else []
    if not isinstance(rules, list):
        raise ValueError("suite manifest acceptance.rules must be a list")

    summary_rows = _read_csv(summary_csv)
    result_rows = _read_csv(results_csv) if results_csv is not None else []
    filters = {
        "method": _as_filter_set(methods),
        "scenario": _as_filter_set(scenarios),
        "comm_profile": _as_filter_set(comm_profiles),
        "n_agents": _as_filter_set(n_agents, numeric=True),
    }

    checks: list[dict[str, Any]] = []
    for rule in rules:
        if not isinstance(rule, dict):
            checks.append(
                {
                    "name": None,
                    "status": "fail",
                    "severity": "required",
                    "matched_rows": 0,
                    "passed_rows": 0,
                    "violations": [{"reason": "rule is not a mapping"}],
                    "message": "invalid rule",
                }
            )
            continue
        scope = str(rule.get("scope", "summary")).lower()
        if scope == "results":
            rows = result_rows
            source_available = results_csv is not None
        elif scope == "summary":
            rows = summary_rows
            source_available = True
        else:
            checks.append(
                {
                    "name": rule.get("name"),
                    "status": "fail",
                    "severity": rule.get("severity", "required"),
                    "scope": scope,
                    "matched_rows": 0,
                    "passed_rows": 0,
                    "violations": [{"reason": f"unsupported scope {scope!r}"}],
                    "message": "invalid rule scope",
                }
            )
            continue
        checks.append(_evaluate_rule(rule, rows=rows, source_available=source_available, filters=filters))

    passed = sum(1 for check in checks if check["status"] == "pass")
    warnings = sum(1 for check in checks if check["status"] == "warn")
    failed = sum(1 for check in checks if check["status"] == "fail")
    skipped = sum(1 for check in checks if check["status"] == "skipped")
    status = "FAIL" if failed else "WARN" if warnings else "PASS"

    return {
        "status": status,
        "ok": failed == 0,
        "suite": manifest.get("suite") if isinstance(manifest, dict) else None,
        "acceptance_schema_version": acceptance.get("schema_version") if isinstance(acceptance, dict) else None,
        "suite_manifest": str(manifest_path),
        "summary_csv": str(Path(summary_csv)),
        "results_csv": str(Path(results_csv)) if results_csv is not None else None,
        "rules_total": len(checks),
        "rules_passed": passed,
        "rules_warned": warnings,
        "rules_failed": failed,
        "rules_skipped": skipped,
        "checks": checks,
    }
