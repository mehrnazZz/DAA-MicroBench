from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

from microbench.rl.learned_diagnostics import (
    LEARNED_POLICY_DIAGNOSTICS_SCHEMA_VERSION,
    build_learned_policy_diagnostics,
    write_learned_policy_diagnostics,
)
from microbench.rl.submission_bundle import run_learned_policy_submission_bundle


ROOT = Path(__file__).resolve().parents[1]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _fake_review(
    tmp_path: Path,
    *,
    name: str,
    policy: str,
    score: float,
    completion: float,
    min_sep: float,
    p05: float,
    collisions: int = 0,
    near_miss_ticks: int = 0,
    final_goal: float = 1.0,
) -> dict:
    planner_results = _write_csv(
        tmp_path / name / "planner_results.csv",
        [
            {
                "scenario": f"{name}_scenario",
                "seed": 7,
                "completion_rate": completion,
                "final_goal_dist_mean_m": final_goal,
                "min_sep_min_m": min_sep,
                "near_misses": near_miss_ticks,
                "collision_episode": collisions,
                "planner_timeout_count": 0,
                "planner_error_count": 0,
                "planner_fallback_count": 0,
            }
        ],
    )
    rl_rows = _write_csv(
        tmp_path / name / "rl_validation_matrix_episodes.csv",
        [
            {
                "lane_id": f"{name}_lane",
                "completion_rate": completion,
                "final_min_sep_m": min_sep,
                "collision_ticks": collisions,
                "near_miss_ticks": near_miss_ticks,
            }
        ],
    )
    return {
        "ok": True,
        "recommendation": "leaderboard_candidate",
        "limitations": [],
        "method": "learned_policy_spec",
        "policy": policy,
        "suite": "official_smoke_generated",
        "score_v0": {"mean": score, "best": score, "worst": score, "rows": []},
        "dimensions": {
            "safety": {
                "collision_episode_count": collisions,
                "near_miss_episode_rate_mean": 1.0 if near_miss_ticks else 0.0,
                "min_sep_min_m": min_sep,
                "min_sep_p05_min_m": p05,
            },
            "mission": {
                "completion_rate_mean": completion,
                "completion_rate_min": completion,
                "deadlock_time_pct_mean": 0.0,
            },
            "compute": {
                "planner_ms_p95_max": 0.5,
                "planner_timeout_count": 0,
                "planner_error_count": 0,
                "planner_fallback_count": 0,
            },
            "rl_validation_matrix": {
                "ok": True,
                "behavior_pass": collisions == 0,
                "collision_ticks": collisions,
                "near_miss_ticks": near_miss_ticks,
                "completion_rate_mean": completion,
                "final_min_sep_min_m": min_sep,
            },
        },
        "validation": {
            "artifacts": {
                "planner_results": str(planner_results),
                "rl_validation_matrix_episodes": str(rl_rows),
            }
        },
    }


def test_learned_policy_diagnostics_labels_tradeoffs(tmp_path: Path, monkeypatch) -> None:
    reviews = {
        "slow": _fake_review(
            tmp_path,
            name="slow",
            policy="safe_slow_policy",
            score=40.0,
            completion=0.65,
            min_sep=2.4,
            p05=3.0,
            final_goal=14.0,
        ),
        "close": _fake_review(
            tmp_path,
            name="close",
            policy="close_policy",
            score=5.0,
            completion=1.0,
            min_sep=0.4,
            p05=0.9,
            near_miss_ticks=2,
        ),
        "unsafe": _fake_review(
            tmp_path,
            name="unsafe",
            policy="unsafe_policy",
            score=20.0,
            completion=1.0,
            min_sep=0.1,
            p05=0.2,
            collisions=1,
        ),
    }
    monkeypatch.setattr(
        "microbench.rl.learned_diagnostics.review_learned_policy_submission_bundle",
        lambda bundle: reviews[str(bundle)],
    )

    report = build_learned_policy_diagnostics(bundles=["slow", "close", "unsafe"])

    assert report["schema_version"] == LEARNED_POLICY_DIAGNOSTICS_SCHEMA_VERSION
    by_policy = {row["policy"]: row for row in report["rows"]}
    assert by_policy["safe_slow_policy"]["diagnostic_label"] == "safe_but_slow"
    assert by_policy["close_policy"]["diagnostic_label"] == "fast_but_close"
    assert by_policy["unsafe_policy"]["diagnostic_label"] == "unsafe"
    assert report["summary"]["safety_leader"] == "safe_slow_policy"
    assert by_policy["safe_slow_policy"]["worst_scenario"] == "slow_scenario"
    assert "Increase horizon/progress weighting" in by_policy["safe_slow_policy"]["next_action"]

    out = tmp_path / "learned_policy_diagnostics.json"
    written = write_learned_policy_diagnostics(bundles=["slow", "close"], out=out)
    assert written["diagnostics_path"] == str(out)
    assert Path(written["diagnostics_csv"]).exists()
    assert Path(written["diagnostics_markdown"]).exists()
    assert "safe_but_slow" in Path(written["diagnostics_markdown"]).read_text(encoding="utf-8")


def test_learned_diagnostics_cli_writes_reports(tmp_path: Path) -> None:
    bundle = tmp_path / "tiny_bundle"
    run_learned_policy_submission_bundle(
        out_dir=bundle,
        method="learned_tiny",
        policy="tiny_learned",
        max_runs=1,
        max_steps=3,
    )

    out = tmp_path / "learned_policy_diagnostics.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "microbench.cli",
            "learned-diagnostics",
            "--bundle",
            str(bundle),
            "--out",
            str(out),
            "--require-pass",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    report = json.loads(proc.stdout)
    assert report["ok"] is True
    assert report["bundle_count"] == 1
    assert report["rows"][0]["policy"] == "tiny_learned"
    assert out.exists()
    assert out.with_suffix(".csv").exists()
    assert out.with_suffix(".md").exists()
