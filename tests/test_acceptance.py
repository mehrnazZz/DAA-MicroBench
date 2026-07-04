from __future__ import annotations

import csv
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
    methods = ["baseline_goal", "orca_expert", "priority_yield"] if include_all_methods else ["baseline_goal"]
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
                    "planner_ms_p95": 150.0 if slow_orca and method == "orca_expert" else 5.0,
                    "comm_agent_msg_attempted_mean": 1.0 if method == "priority_yield" else 0.0,
                }
            )
    return rows


def test_check_acceptance_passes_generated_smoke_summary(tmp_path: Path) -> None:
    manifest = _smoke_manifest(tmp_path)
    summary = tmp_path / "summary.csv"
    _write_csv(summary, _summary_rows())

    report = check_acceptance(summary_csv=summary, suite_manifest=manifest)

    assert report["status"] == "PASS"
    assert report["rules_passed"] == 5
    assert report["rules_failed"] == 0


def test_check_acceptance_filters_to_run_method(tmp_path: Path) -> None:
    manifest = _smoke_manifest(tmp_path)
    summary = tmp_path / "summary.csv"
    _write_csv(summary, _summary_rows(include_all_methods=False))

    report = check_acceptance(summary_csv=summary, suite_manifest=manifest, methods=["baseline_goal"])

    assert report["status"] == "PASS"
    assert report["rules_passed"] == 2
    assert report["rules_skipped"] == 3


def test_check_acceptance_fails_missing_smoke_rows_without_filter(tmp_path: Path) -> None:
    manifest = _smoke_manifest(tmp_path)
    summary = tmp_path / "summary.csv"
    _write_csv(summary, _summary_rows(include_all_methods=False))

    report = check_acceptance(summary_csv=summary, suite_manifest=manifest)

    assert report["status"] == "FAIL"
    assert report["rules_failed"] == 3
    assert any(check["message"] == "no matching rows" for check in report["checks"])


def test_check_acceptance_fails_threshold_violation(tmp_path: Path) -> None:
    manifest = _smoke_manifest(tmp_path)
    summary = tmp_path / "summary.csv"
    _write_csv(summary, _summary_rows(slow_orca=True))

    report = check_acceptance(summary_csv=summary, suite_manifest=manifest)

    assert report["status"] == "FAIL"
    assert report["rules_failed"] == 1
    failed = [check for check in report["checks"] if check["status"] == "fail"]
    assert failed[0]["name"] == "orca_expert_smoke_runtime"
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
    assert '"rules_skipped": 3' in proc.stdout


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
