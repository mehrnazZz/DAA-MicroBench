from __future__ import annotations

import json
from pathlib import Path
import sys
import types
from typing import Any

from microbench.logging.wandb_logger import build_leaderboard_wandb_payload, log_baseline_leaderboard


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sample_leaderboard_report(root: Path) -> dict[str, Any]:
    suite_dir = root / "official_smoke_generated"
    leaderboard_path = root / "baseline_leaderboard.json"
    suite_report_path = suite_dir / "baseline_report.json"
    acceptance_path = suite_dir / "acceptance.json"
    progress_path = suite_dir / "leaderboard_progress.json"
    results_path = suite_dir / "runs" / "results.csv"
    summary_path = suite_dir / "runs" / "summary.csv"
    schema_path = suite_dir / "runs" / "result_schema.json"
    manifest_path = suite_dir / "_generated_scenarios" / "official_smoke_generated" / "suite_manifest.yaml"

    for path, text in (
        (results_path, "run_id,method\nr0,baseline_goal\n"),
        (summary_path, "method,score_v0\nbaseline_goal,12.0\n"),
        (schema_path, '{"schema_version":"0.4.0"}\n'),
        (manifest_path, "suite_id: official_smoke_generated\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    _write_json(
        suite_report_path,
        {
            "method_summaries": [
                {
                    "method": "baseline_goal",
                    "score_v0_mean": 12.0,
                    "collision_episode_rate_mean": 0.0,
                    "completion_rate_mean": 1.0,
                }
            ],
            "rows": [
                {
                    "method": "baseline_goal",
                    "scenario": "head_on_3d",
                    "comm_profile": "ideal_50hz",
                    "N": 4,
                    "score_v0": 12.0,
                    "collision_episode_rate": 0.0,
                    "completion_rate_mean": 1.0,
                }
            ],
        },
    )
    _write_json(acceptance_path, {"ok": True})
    _write_json(progress_path, {"complete": True})

    report = {
        "schema_version": "0.2",
        "ok": True,
        "complete": True,
        "selected_complete": True,
        "timeout_run_count": 0,
        "out_dir": str(root),
        "leaderboard_path": str(leaderboard_path),
        "methods": ["baseline_goal"],
        "aggregate_ranking": [
            {
                "rank": 1,
                "method": "baseline_goal",
                "score_v0_mean": 12.0,
                "suite_count": 1,
                "episodes": 1,
            }
        ],
        "suites": [
            {
                "suite": "official_smoke_generated",
                "ok": True,
                "complete": True,
                "selected_complete": True,
                "planned_run_count": 1,
                "selected_run_count": 1,
                "selected_completed_count": 1,
                "run_count": 1,
                "timeout_run_count": 0,
                "scenario_count": 1,
                "truncated_by_max_runs": False,
                "stopped_by_wall_time": False,
                "report_path": str(suite_report_path.relative_to(root)),
                "acceptance_path": str(acceptance_path.relative_to(root)),
                "progress_path": str(progress_path.relative_to(root)),
                "results_csv": str(results_path.relative_to(root)),
                "summary_csv": str(summary_path.relative_to(root)),
                "suite_manifest": str(manifest_path.relative_to(root)),
            }
        ],
    }
    _write_json(leaderboard_path, report)
    return report


def test_build_leaderboard_wandb_payload_projects_suite_reports(tmp_path: Path) -> None:
    report = _sample_leaderboard_report(tmp_path)

    payload = build_leaderboard_wandb_payload(report)

    assert payload["aggregate_rows"][0]["method"] == "baseline_goal"
    assert payload["suite_rows"][0]["suite"] == "official_smoke_generated"
    assert payload["suite_method_rows"][0]["suite"] == "official_smoke_generated"
    assert payload["suite_method_rows"][0]["method"] == "baseline_goal"
    assert payload["component_rows"][0]["scenario"] == "head_on_3d"
    assert any(path.name == "result_schema.json" for path in payload["artifact_paths"])


def test_log_baseline_leaderboard_uses_tables_and_artifact(tmp_path: Path, monkeypatch) -> None:
    report = _sample_leaderboard_report(tmp_path)

    class FakeTable:
        def __init__(self, *, columns):
            self.columns = list(columns)
            self.rows = []

        def add_data(self, *values):
            self.rows.append(list(values))

    class FakeArtifact:
        def __init__(self, *, name, type):
            self.name = name
            self.type = type
            self.files = []

        def add_file(self, local_path, name=None):
            self.files.append((local_path, name))

    fake_wandb = types.ModuleType("wandb")
    fake_wandb.Table = FakeTable
    fake_wandb.Artifact = FakeArtifact
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    class FakeRun:
        id = "fake-run"
        config = {"run_id": "leaderboard_case"}

        def __init__(self):
            self.logged = []
            self.artifacts = []

        def log(self, data):
            self.logged.append(data)

        def log_artifact(self, artifact):
            self.artifacts.append(artifact)

    run = FakeRun()
    log_baseline_leaderboard(run, report)

    logged_keys = {key for entry in run.logged for key in entry.keys()}
    assert "leaderboard_aggregate" in logged_keys
    assert "leaderboard_suites" in logged_keys
    assert "leaderboard_suite_methods" in logged_keys
    assert "leaderboard_components" in logged_keys
    assert run.artifacts
    artifact = run.artifacts[0]
    assert artifact.type == "daa-microbench-leaderboard"
    assert any(name and name.endswith("baseline_leaderboard.json") for _, name in artifact.files)
    assert any(name and name.endswith("result_schema.json") for _, name in artifact.files)
