from __future__ import annotations

import csv
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

from microbench.metrics import append_result, write_summary
from microbench.metrics.io import (
    RESULT_FIELDS,
    RESULT_SCHEMA_FILENAME,
    SUMMARY_FIELDS,
    result_schema_manifest,
)
from microbench.runner import run_episode
from microbench.types import RunSpec


CURRENT_SCHEMA_GOLDEN_RUN_ID = "daa_current_schema_m64"
CURRENT_SCHEMA_FILES = ("results.csv", "summary.csv", RESULT_SCHEMA_FILENAME)

RESULT_TIMING_FIELDS = frozenset(
    {
        "planner_ms_per_tick_per_agent_mean",
        "planner_ms_per_tick_per_agent_p95",
        "episode_runtime_s",
    }
)
SUMMARY_TIMING_FIELDS = frozenset({"planner_ms_mean", "planner_ms_p95"})
CURRENT_SCHEMA_FLOAT_ABS_TOL = 1e-5
CURRENT_SCHEMA_FLOAT_REL_TOL = 1e-9

RESULT_FLOAT_TOLERANCE_FIELDS = frozenset(
    {
        "jerk_mean",
        "min_sep_min_m",
        "min_sep_p05_m",
    }
)
SUMMARY_FLOAT_TOLERANCE_FIELDS = frozenset(
    {
        "min_sep_min_mean",
        "min_sep_p05_mean",
    }
)

CURRENT_SCHEMA_RUNS = (
    {
        "scenario_path": "config/scenarios/corridor.yaml",
        "method": "baseline_goal",
        "n_agents": 4,
        "seed": 0,
        "comm_profile": "ideal_50hz",
        "agent_methods": None,
    },
    {
        "scenario_path": "config/scenarios/corridor.yaml",
        "method": "mixed",
        "n_agents": 4,
        "seed": 1,
        "comm_profile": "ideal_50hz",
        "agent_methods": ["baseline_goal", "template", "baseline_goal", "template"],
    },
)


def _clean_artifacts(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in CURRENT_SCHEMA_FILES:
        path = out_dir / name
        if path.exists():
            path.unlink()


def _run_current_schema_fixture(run_dir: Path) -> None:
    _clean_artifacts(run_dir)
    for cfg in CURRENT_SCHEMA_RUNS:
        row = run_episode(
            RunSpec(
                scenario_path=str(cfg["scenario_path"]),
                method=str(cfg["method"]),
                n_agents=int(cfg["n_agents"]),
                seed=int(cfg["seed"]),
                comm_profile=str(cfg["comm_profile"]),
                out_dir=str(run_dir),
                save_trace=False,
                agent_methods=cfg["agent_methods"],
            )
        )
        append_result(run_dir, row)
    write_summary(run_dir)


def write_current_schema_golden(out_dir: str | Path) -> Path:
    """Regenerate the current-schema fixture while preserving its stable run id."""

    target = Path(out_dir)
    if target.name == CURRENT_SCHEMA_GOLDEN_RUN_ID:
        _run_current_schema_fixture(target)
        return target

    with tempfile.TemporaryDirectory(prefix="daa_current_schema_") as td:
        run_dir = Path(td) / CURRENT_SCHEMA_GOLDEN_RUN_ID
        _run_current_schema_fixture(run_dir)
        target.mkdir(parents=True, exist_ok=True)
        for name in CURRENT_SCHEMA_FILES:
            shutil.copy2(run_dir / name, target / name)
    return target


def build_current_schema_candidate(parent_dir: str | Path | None = None) -> Path:
    if parent_dir is None:
        parent = Path(tempfile.mkdtemp(prefix="daa_current_schema_candidate_"))
    else:
        parent = Path(parent_dir)
        parent.mkdir(parents=True, exist_ok=True)
    run_dir = parent / CURRENT_SCHEMA_GOLDEN_RUN_ID
    _run_current_schema_fixture(run_dir)
    return run_dir


def _read_header(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return next(csv.reader(f))


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _row_key(row: dict[str, str], *, summary: bool) -> tuple[str, str, str, str, str]:
    seed = "" if summary else str(row.get("seed", ""))
    return (
        str(row.get("method", "")),
        str(row.get("scenario", "")),
        str(row.get("comm_profile", "")),
        str(row.get("N", "")),
        seed,
    )


def _timing_value_ok(value: Any) -> bool:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(out) and out >= 0.0


def _numeric_values_close(expected: Any, actual: Any) -> bool:
    try:
        expected_f = float(expected)
        actual_f = float(actual)
    except (TypeError, ValueError):
        return False
    if math.isnan(expected_f) or math.isnan(actual_f):
        return math.isnan(expected_f) and math.isnan(actual_f)
    if not math.isfinite(expected_f) or not math.isfinite(actual_f):
        return False
    return math.isclose(
        expected_f,
        actual_f,
        rel_tol=CURRENT_SCHEMA_FLOAT_REL_TOL,
        abs_tol=CURRENT_SCHEMA_FLOAT_ABS_TOL,
    )


def _missing_file_mismatch(path: Path, file_name: str, role: str) -> dict[str, Any] | None:
    if path.exists():
        return None
    return {
        "file": file_name,
        "field": None,
        "row": None,
        "reason": f"missing_{role}_file",
        "expected": str(path),
        "actual": None,
    }


def _compare_manifest(golden_dir: Path, candidate_dir: Path) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    expected_path = golden_dir / RESULT_SCHEMA_FILENAME
    actual_path = candidate_dir / RESULT_SCHEMA_FILENAME
    for path, role in ((expected_path, "golden"), (actual_path, "candidate")):
        mismatch = _missing_file_mismatch(path, RESULT_SCHEMA_FILENAME, role)
        if mismatch is not None:
            mismatches.append(mismatch)
    if mismatches:
        return mismatches

    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    actual = json.loads(actual_path.read_text(encoding="utf-8"))
    declared = result_schema_manifest()
    if expected != declared:
        mismatches.append(
            {
                "file": RESULT_SCHEMA_FILENAME,
                "field": "manifest",
                "row": None,
                "reason": "golden_manifest_not_declared_schema",
                "expected": declared,
                "actual": expected,
            }
        )
    if actual != declared:
        mismatches.append(
            {
                "file": RESULT_SCHEMA_FILENAME,
                "field": "manifest",
                "row": None,
                "reason": "candidate_manifest_not_declared_schema",
                "expected": declared,
                "actual": actual,
            }
        )
    return mismatches


def _compare_csv(
    *,
    golden_dir: Path,
    candidate_dir: Path,
    file_name: str,
    fields: list[str],
    timing_fields: frozenset[str],
    tolerance_fields: frozenset[str],
    summary: bool,
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    expected_path = golden_dir / file_name
    actual_path = candidate_dir / file_name
    for path, role in ((expected_path, "golden"), (actual_path, "candidate")):
        mismatch = _missing_file_mismatch(path, file_name, role)
        if mismatch is not None:
            mismatches.append(mismatch)
    if mismatches:
        return mismatches

    expected_header = _read_header(expected_path)
    actual_header = _read_header(actual_path)
    if expected_header != fields:
        mismatches.append(
            {
                "file": file_name,
                "field": "header",
                "row": None,
                "reason": "golden_header_not_declared_schema",
                "expected": fields,
                "actual": expected_header,
            }
        )
    if actual_header != fields:
        mismatches.append(
            {
                "file": file_name,
                "field": "header",
                "row": None,
                "reason": "candidate_header_not_declared_schema",
                "expected": fields,
                "actual": actual_header,
            }
        )
    if mismatches:
        return mismatches

    expected_rows = sorted(_read_rows(expected_path), key=lambda row: _row_key(row, summary=summary))
    actual_rows = sorted(_read_rows(actual_path), key=lambda row: _row_key(row, summary=summary))
    if len(expected_rows) != len(actual_rows):
        mismatches.append(
            {
                "file": file_name,
                "field": None,
                "row": None,
                "reason": "row_count_mismatch",
                "expected": len(expected_rows),
                "actual": len(actual_rows),
            }
        )
        return mismatches

    for idx, (expected, actual) in enumerate(zip(expected_rows, actual_rows)):
        expected_key = _row_key(expected, summary=summary)
        actual_key = _row_key(actual, summary=summary)
        if expected_key != actual_key:
            mismatches.append(
                {
                    "file": file_name,
                    "field": "row_key",
                    "row": idx,
                    "reason": "row_key_mismatch",
                    "expected": expected_key,
                    "actual": actual_key,
                }
            )
            continue

        for field in fields:
            if field in timing_fields:
                if not _timing_value_ok(expected.get(field)) or not _timing_value_ok(actual.get(field)):
                    mismatches.append(
                        {
                            "file": file_name,
                            "field": field,
                            "row": idx,
                            "reason": "timing_value_not_finite_nonnegative",
                            "expected": expected.get(field),
                            "actual": actual.get(field),
                        }
                    )
                continue

            if field in tolerance_fields and _numeric_values_close(expected.get(field), actual.get(field)):
                continue

            if expected.get(field) != actual.get(field):
                mismatches.append(
                    {
                        "file": file_name,
                        "field": field,
                        "row": idx,
                        "reason": "semantic_value_mismatch",
                        "expected": expected.get(field),
                        "actual": actual.get(field),
                        "key": expected_key,
                    }
                )

    return mismatches


def compare_current_schema_golden(
    *,
    candidate_dir: str | Path,
    golden_dir: str | Path = "golden/current_schema",
) -> dict[str, Any]:
    golden = Path(golden_dir)
    candidate = Path(candidate_dir)
    mismatches: list[dict[str, Any]] = []
    mismatches.extend(_compare_manifest(golden, candidate))
    mismatches.extend(
        _compare_csv(
            golden_dir=golden,
            candidate_dir=candidate,
            file_name="results.csv",
            fields=RESULT_FIELDS,
            timing_fields=RESULT_TIMING_FIELDS,
            tolerance_fields=RESULT_FLOAT_TOLERANCE_FIELDS,
            summary=False,
        )
    )
    mismatches.extend(
        _compare_csv(
            golden_dir=golden,
            candidate_dir=candidate,
            file_name="summary.csv",
            fields=SUMMARY_FIELDS,
            timing_fields=SUMMARY_TIMING_FIELDS,
            tolerance_fields=SUMMARY_FLOAT_TOLERANCE_FIELDS,
            summary=True,
        )
    )

    return {
        "ok": not mismatches,
        "golden_dir": str(golden),
        "candidate_dir": str(candidate),
        "schema": result_schema_manifest()["schema"],
        "schema_version": result_schema_manifest()["schema_version"],
        "ignored_or_tolerated_timing_fields": {
            "results.csv": sorted(RESULT_TIMING_FIELDS),
            "summary.csv": sorted(SUMMARY_TIMING_FIELDS),
        },
        "tolerated_float_fields": {
            "abs_tol": CURRENT_SCHEMA_FLOAT_ABS_TOL,
            "rel_tol": CURRENT_SCHEMA_FLOAT_REL_TOL,
            "results.csv": sorted(RESULT_FLOAT_TOLERANCE_FIELDS),
            "summary.csv": sorted(SUMMARY_FLOAT_TOLERANCE_FIELDS),
        },
        "mismatches": mismatches,
    }
