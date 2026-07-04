from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

import yaml

from microbench.acceptance import check_acceptance
from microbench.scenarios import materialize_official_suite


SUMMARY_FIELDS = [
    "method",
    "scenario",
    "comm_profile",
    "N",
    "completion_rate_mean",
    "collision_episode_rate",
    "planner_ms_p95",
    "comm_agent_msg_attempted_mean",
    "comm_agent_msg_delivered_mean",
]


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _smoke_manifest(tmp_path: Path) -> Path:
    generated = materialize_official_suite("official_smoke_generated", tmp_path / "suite", overwrite=True)
    return generated["manifest_path"]


def _summary_rows(*, include_all_methods: bool = True, slow_orca: bool = False) -> list[dict]:
    scenarios = ["head_on_2d_easy", "sphere_swap_3d_medium", "heterogeneous_priority_crossing_3d_medium"]
    methods = ["baseline_goal", "orca_heuristic", "priority_yield"] if include_all_methods else ["baseline_goal"]
    rows = []
    for scenario in scenarios:
        for method in methods:
            rows.append(
                {
                    "method": method,
                    "scenario": scenario,
                    "comm_profile": "ideal_50hz",
                    "N": 4,
                    "completion_rate_mean": 0.5,
                    "collision_episode_rate": 0.0,
                    "planner_ms_p95": 50.0 if slow_orca and method == "orca_heuristic" else 0.5,
                    "comm_agent_msg_attempted_mean": 1.0 if method == "priority_yield" else 0.0,
                    "comm_agent_msg_delivered_mean": (
                        1.0 if method == "priority_yield" and scenario == "head_on_2d_easy" else 0.0
                    ),
                }
            )
    return rows


def _fixture_projection(report: dict) -> dict:
    return {
        "suite": report["suite"],
        "acceptance_schema_version": report["acceptance_schema_version"],
        "status": report["status"],
        "rules_total": report["rules_total"],
        "rules_passed": report["rules_passed"],
        "rules_warned": report["rules_warned"],
        "rules_failed": report["rules_failed"],
        "rules_skipped": report["rules_skipped"],
        "checks": [
            {
                "name": check["name"],
                "status": check["status"],
                "severity": check["severity"],
                "scope": check["scope"],
                "method": check["method"],
                "scenario": check["scenario"],
                "metric": check["metric"],
                "operator": check["operator"],
                "value": check["value"],
                "matched_rows": check["matched_rows"],
                "passed_rows": check["passed_rows"],
            }
            for check in report["checks"]
        ],
    }


def test_check_acceptance_passes_generated_smoke_summary(tmp_path: Path) -> None:
    manifest = _smoke_manifest(tmp_path)
    summary = tmp_path / "summary.csv"
    _write_csv(summary, _summary_rows())

    report = check_acceptance(summary_csv=summary, suite_manifest=manifest)

    assert report["status"] == "PASS"
    assert report["rules_passed"] == 7
    assert report["rules_failed"] == 0


def test_golden_acceptance_fixture_matches_generated_smoke_contract(tmp_path: Path) -> None:
    manifest = _smoke_manifest(tmp_path)
    summary = tmp_path / "summary.csv"
    _write_csv(summary, _summary_rows())

    report = check_acceptance(summary_csv=summary, suite_manifest=manifest)
    fixture = json.loads(
        (Path(__file__).resolve().parents[1] / "golden/acceptance/official_smoke_generated_acceptance.json").read_text(
            encoding="utf-8"
        )
    )

    assert _fixture_projection(report) == fixture


def test_check_acceptance_filters_to_run_method(tmp_path: Path) -> None:
    manifest = _smoke_manifest(tmp_path)
    summary = tmp_path / "summary.csv"
    _write_csv(summary, _summary_rows(include_all_methods=False))

    report = check_acceptance(summary_csv=summary, suite_manifest=manifest, methods=["baseline_goal"])

    assert report["status"] == "PASS"
    assert report["rules_passed"] == 3
    assert report["rules_skipped"] == 4


def test_check_acceptance_fails_missing_smoke_rows_without_filter(tmp_path: Path) -> None:
    manifest = _smoke_manifest(tmp_path)
    summary = tmp_path / "summary.csv"
    _write_csv(summary, _summary_rows(include_all_methods=False))

    report = check_acceptance(summary_csv=summary, suite_manifest=manifest)

    assert report["status"] == "FAIL"
    assert report["rules_failed"] == 4
    assert any(check["message"] == "no matching rows" for check in report["checks"])


def test_check_acceptance_fails_threshold_violation(tmp_path: Path) -> None:
    manifest = _smoke_manifest(tmp_path)
    summary = tmp_path / "summary.csv"
    _write_csv(summary, _summary_rows(slow_orca=True))

    report = check_acceptance(summary_csv=summary, suite_manifest=manifest)

    assert report["status"] == "FAIL"
    assert report["rules_failed"] == 1
    failed = [check for check in report["checks"] if check["status"] == "fail"]
    assert failed[0]["name"] == "orca_heuristic_smoke_runtime"
    assert len(failed[0]["violations"]) == 3


def test_check_acceptance_supports_results_scoped_rules(tmp_path: Path) -> None:
    manifest = tmp_path / "suite_manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "suite": "custom_acceptance",
                "acceptance": {
                    "schema_version": "0.1",
                    "rules": [
                        {
                            "name": "episode_runtime_bound",
                            "scope": "results",
                            "method": "baseline_goal",
                            "scenario": "*",
                            "comm_profile": "*",
                            "n_agents": "*",
                            "metric": "episode_runtime_s",
                            "operator": "<=",
                            "value": 10.0,
                            "severity": "required",
                            "description": "Episode runtime must be bounded.",
                        }
                    ],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    results = tmp_path / "results.csv"
    with results.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "scenario", "comm_profile", "N", "episode_runtime_s"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "method": "baseline_goal",
                "scenario": "head_on_2d_easy",
                "comm_profile": "ideal_50hz",
                "N": 4,
                "episode_runtime_s": 0.5,
            }
        )
    summary = tmp_path / "summary.csv"
    _write_csv(summary, _summary_rows(include_all_methods=False))

    report = check_acceptance(summary_csv=summary, results_csv=results, suite_manifest=manifest)

    assert report["status"] == "PASS"
    assert report["rules_passed"] == 1


def test_check_acceptance_cli_json_passes_with_method_filter(tmp_path: Path) -> None:
    manifest = _smoke_manifest(tmp_path)
    summary = tmp_path / "summary.csv"
    _write_csv(summary, _summary_rows(include_all_methods=False))

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "microbench.cli",
            "check-acceptance",
            "--summary",
            str(summary),
            "--suite-manifest",
            str(manifest),
            "--methods",
            "baseline_goal",
            "--json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )

    assert '"status": "PASS"' in proc.stdout
    assert '"rules_skipped": 4' in proc.stdout


def test_check_acceptance_cli_fails_on_missing_required_rows(tmp_path: Path) -> None:
    manifest = _smoke_manifest(tmp_path)
    summary = tmp_path / "summary.csv"
    _write_csv(summary, _summary_rows(include_all_methods=False))

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "microbench.cli",
            "check-acceptance",
            "--summary",
            str(summary),
            "--suite-manifest",
            str(manifest),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )

    assert proc.returncode != 0
    assert "acceptance: FAIL" in proc.stdout
