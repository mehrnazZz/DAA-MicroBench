from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from microbench.rl.learned_leaderboard import (
    LEARNED_POLICY_LEADERBOARD_SCHEMA_VERSION,
    build_learned_policy_leaderboard,
    write_learned_policy_leaderboard,
)
from microbench.rl.submission_bundle import run_learned_policy_submission_bundle


ROOT = Path(__file__).resolve().parents[1]


def _fake_review(*, method: str, policy: str, score: float, recommendation: str = "leaderboard_candidate") -> dict:
    return {
        "schema_version": "0.1",
        "ok": True,
        "recommendation": recommendation,
        "limitations": [] if recommendation == "leaderboard_candidate" else ["limited_planner_sweep"],
        "method": method,
        "policy": policy,
        "suite": "official_smoke_generated",
        "run_count": 3,
        "planned_run_count": 3,
        "score_v0": {"mean": score, "best": score - 1.0, "worst": score + 1.0, "rows": []},
        "dimensions": {
            "safety": {
                "collision_episode_count": 0,
                "collision_episode_rate_mean": 0.0,
                "min_sep_min_m": 0.5,
                "min_sep_p05_min_m": 0.6,
            },
            "mission": {
                "completion_rate_mean": 1.0,
                "completion_rate_min": 1.0,
                "deadlock_time_pct_mean": 0.0,
            },
            "compute": {
                "planner_ms_p95_max": 0.5,
                "planner_timeout_count": 0,
                "planner_error_count": 0,
                "planner_fallback_count": 0,
            },
            "communication": {},
            "observation": {},
            "rl_validation_matrix": {
                "ok": True,
                "behavior_pass": True,
                "run_count": 5,
                "lane_count": 5,
                "collision_ticks": 0,
                "near_miss_ticks": 0,
                "completion_rate_mean": 0.5,
                "final_min_sep_min_m": 0.4,
                "failed_gate_checks": [],
                "failed_behavior_checks": [],
            },
        },
        "checks": [],
        "validation": {},
    }


def test_learned_policy_leaderboard_ranks_reviewable_rows(tmp_path: Path, monkeypatch) -> None:
    reviews = {
        "tiny": _fake_review(method="learned_tiny", policy="tiny_learned", score=12.0),
        "mlp": _fake_review(method="learned_mlp", policy="mlp_learned", score=8.0),
        "limited": _fake_review(
            method="learned_policy_spec",
            policy="external_fixture",
            score=10.0,
            recommendation="manual_review_limited_sweep",
        ),
    }

    monkeypatch.setattr(
        "microbench.rl.learned_leaderboard.review_learned_policy_submission_bundle",
        lambda bundle: reviews[str(bundle)],
    )

    report = build_learned_policy_leaderboard(bundles=["tiny", "mlp", "limited"])

    assert report["schema_version"] == LEARNED_POLICY_LEADERBOARD_SCHEMA_VERSION
    assert report["ok"] is True
    assert report["bundle_count"] == 3
    assert report["leaderboard_candidate_count"] == 2
    assert [row["policy"] for row in report["rows"]] == ["mlp_learned", "external_fixture", "tiny_learned"]
    assert [row["development_rank"] for row in report["rows"]] == [1, 2, 3]
    assert report["rows"][1]["leaderboard_candidate"] is False
    assert report["rows"][1]["limitations"] == "limited_planner_sweep"

    out = tmp_path / "learned_policy_leaderboard.json"
    written = write_learned_policy_leaderboard(bundles=["tiny", "mlp"], out=out)
    assert written["leaderboard_path"] == str(out)
    assert Path(written["leaderboard_csv"]).exists()
    assert "score_v0_mean" in Path(written["leaderboard_csv"]).read_text(encoding="utf-8")


def test_learned_policy_leaderboard_cli_compares_bundles(tmp_path: Path) -> None:
    tiny_bundle = tmp_path / "tiny_bundle"
    mlp_bundle = tmp_path / "mlp_bundle"
    run_learned_policy_submission_bundle(
        out_dir=tiny_bundle,
        method="learned_tiny",
        policy="tiny_learned",
        max_runs=1,
        max_steps=3,
    )
    run_learned_policy_submission_bundle(
        out_dir=mlp_bundle,
        method="learned_mlp",
        policy="mlp_learned",
        max_runs=1,
        max_steps=3,
    )

    out = tmp_path / "learned_policy_leaderboard.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "microbench.cli",
            "learned-leaderboard",
            "--bundle",
            str(tiny_bundle),
            "--bundle",
            str(mlp_bundle),
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
    assert report["bundle_count"] == 2
    assert report["reviewable_count"] == 2
    assert {row["policy"] for row in report["rows"]} == {"tiny_learned", "mlp_learned"}
    assert out.exists()
    assert out.with_suffix(".csv").exists()
