from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

from microbench.metrics.io import RESULT_SCHEMA_FILENAME
from microbench.tools.current_schema_golden import (
    RESULT_TIMING_FIELDS,
    SUMMARY_TIMING_FIELDS,
    build_current_schema_candidate,
    compare_current_schema_golden,
)


ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = ROOT / "golden" / "current_schema"


def _copy_fixture(tmp_path: Path) -> Path:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    for name in ("results.csv", "summary.csv", RESULT_SCHEMA_FILENAME):
        shutil.copy2(GOLDEN_DIR / name, candidate / name)
    return candidate


def _rewrite_csv(path: Path, mutate) -> None:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    mutate(rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_current_schema_golden_matches_fresh_regeneration(tmp_path: Path) -> None:
    candidate = build_current_schema_candidate(tmp_path)

    report = compare_current_schema_golden(candidate_dir=candidate, golden_dir=GOLDEN_DIR)

    assert report["ok"], report["mismatches"]
    assert report["schema_version"] == "0.4.0"
    assert set(report["ignored_or_tolerated_timing_fields"]["results.csv"]) == set(RESULT_TIMING_FIELDS)
    assert set(report["ignored_or_tolerated_timing_fields"]["summary.csv"]) == set(SUMMARY_TIMING_FIELDS)


def test_current_schema_comparison_tolerates_timing_drift(tmp_path: Path) -> None:
    candidate = _copy_fixture(tmp_path)

    def mutate_results(rows):
        for row in rows:
            row["planner_ms_per_tick_per_agent_mean"] = "999.0"
            row["planner_ms_per_tick_per_agent_p95"] = "1000.0"
            row["episode_runtime_s"] = "1001.0"

    def mutate_summary(rows):
        for row in rows:
            row["planner_ms_mean"] = "999.0"
            row["planner_ms_p95"] = "1000.0"

    _rewrite_csv(candidate / "results.csv", mutate_results)
    _rewrite_csv(candidate / "summary.csv", mutate_summary)

    report = compare_current_schema_golden(candidate_dir=candidate, golden_dir=GOLDEN_DIR)

    assert report["ok"], report["mismatches"]


def test_current_schema_comparison_fails_semantic_drift(tmp_path: Path) -> None:
    candidate = _copy_fixture(tmp_path)

    def mutate(rows):
        rows[0]["collisions"] = "0"

    _rewrite_csv(candidate / "results.csv", mutate)

    report = compare_current_schema_golden(candidate_dir=candidate, golden_dir=GOLDEN_DIR)

    assert not report["ok"]
    assert any(
        mismatch["file"] == "results.csv"
        and mismatch["field"] == "collisions"
        and mismatch["reason"] == "semantic_value_mismatch"
        for mismatch in report["mismatches"]
    )


def test_current_schema_cli_candidate_json(tmp_path: Path) -> None:
    candidate = _copy_fixture(tmp_path)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "microbench.cli",
            "golden-current-schema",
            "--candidate",
            str(candidate),
            "--golden-dir",
            str(GOLDEN_DIR),
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    report = json.loads(proc.stdout)
    assert report["ok"] is True
    assert report["candidate_dir"] == str(candidate)
