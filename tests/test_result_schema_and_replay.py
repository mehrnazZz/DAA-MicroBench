from __future__ import annotations

import csv
import json
from pathlib import Path

from microbench.metrics.io import (
    RESULT_FIELDS,
    RESULT_SCHEMA_FILENAME,
    RESULT_SCHEMA_VERSION,
    SUMMARY_FIELDS,
    append_result,
    result_schema_manifest,
    write_summary,
)
from microbench.replay import render_interactive_trace


ROOT = Path(__file__).resolve().parents[1]


def _header(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return next(csv.reader(f))


def test_result_schema_manifest_is_declared_and_deterministic() -> None:
    manifest = result_schema_manifest()

    assert RESULT_SCHEMA_VERSION == "0.4.0"
    assert manifest == result_schema_manifest()
    assert manifest["schema"] == "daa_microbench.results"
    assert manifest["schema_version"] == RESULT_SCHEMA_VERSION
    assert manifest["results"]["fields"] == RESULT_FIELDS
    assert manifest["summary"]["fields"] == SUMMARY_FIELDS


def test_append_result_and_write_summary_emit_schema_manifest(tmp_path: Path) -> None:
    append_result(
        tmp_path,
        {
            "run_id": "schema_smoke",
            "method": "baseline_goal",
            "scenario": "head_on_2d_easy",
            "comm_profile": "ideal_50hz",
            "N": 2,
            "collisions": 0,
            "collision_episode": 0,
            "unique_collision_pairs": 0,
            "near_miss_episode": 0,
            "completion_rate": 1.0,
        },
    )
    write_summary(tmp_path)

    manifest_path = tmp_path / RESULT_SCHEMA_FILENAME
    assert manifest_path.exists()
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == result_schema_manifest()
    assert _header(tmp_path / "results.csv") == RESULT_FIELDS
    assert _header(tmp_path / "summary.csv") == SUMMARY_FIELDS


def test_current_schema_golden_headers_and_manifest_match_declared_schema() -> None:
    golden_dir = ROOT / "golden" / "current_schema"

    assert _header(golden_dir / "results.csv") == RESULT_FIELDS
    assert _header(golden_dir / "summary.csv") == SUMMARY_FIELDS
    assert json.loads((golden_dir / RESULT_SCHEMA_FILENAME).read_text(encoding="utf-8")) == result_schema_manifest()


def test_golden_collision_trace_renders_interactive_html(tmp_path: Path) -> None:
    trace = ROOT / "golden" / "traces" / "trace_collision_0_9_t15.18.jsonl"
    out = tmp_path / "trace_collision_replay.html"

    render_interactive_trace(trace, out, tail=4, max_sensed_per_agent=2)

    html = out.read_text(encoding="utf-8")
    assert out.exists()
    assert "Plotly.newPlot" in html
    assert "const replay =" in html
    assert "trace_collision_0_9_t15.18" in html
    assert "collision_pair" in html
