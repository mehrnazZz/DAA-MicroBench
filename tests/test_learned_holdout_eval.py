from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from microbench.rl.learned_holdout import (
    DEFAULT_LEARNED_HOLDOUT_REFERENCE_METHODS,
    LEARNED_HOLDOUT_EVAL_SCHEMA_VERSION,
    LearnedHoldoutRunTimeout,
    parse_learned_policy_spec_entries,
    run_learned_holdout_eval,
)
from microbench.rl.policy_spec import RL_POLICY_SPEC_SCHEMA_VERSION


ROOT = Path(__file__).resolve().parents[1]


def _policy_spec(path: Path, *, adapter: str = "temporal_mlp_json", name: str = "temporal_fixture") -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": RL_POLICY_SPEC_SCHEMA_VERSION,
                "policy_name": name,
                "adapter": adapter,
                "artifact_path": "policy.json",
                "deterministic": True,
                "clip": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _fake_result_row(spec, *, collision_episode: int, min_sep: float, completion: float, p95_ms: float) -> dict[str, object]:
    return {
        "run_id": f"{Path(spec.scenario_path).stem}_{spec.method}_{spec.seed}",
        "method": spec.method,
        "scenario": Path(spec.scenario_path).stem,
        "comm_profile": spec.comm_profile,
        "N": spec.n_agents,
        "seed": spec.seed,
        "dt_s": 0.1,
        "duration_s": 0.2,
        "v_max_mps": 3.0,
        "a_max_mps2": 2.0,
        "range_m": 40.0,
        "top_k": 8,
        "spawn_goal_dist_min": 10.0,
        "spawn_goal_dist_mean": 10.0,
        "collisions": collision_episode,
        "near_misses": collision_episode,
        "collision_pair_ticks": 2 * collision_episode,
        "near_miss_pair_ticks": 3 * collision_episode,
        "unique_collision_pairs": collision_episode,
        "unique_near_miss_pairs": collision_episode,
        "collision_episode": collision_episode,
        "near_miss_episode": collision_episode,
        "time_to_first_collision_s": 0.1 if collision_episode else None,
        "min_sep_min_m": min_sep,
        "min_sep_p05_m": min_sep + 0.1,
        "completion_rate": completion,
        "final_goal_dist_mean_m": 3.0,
        "final_goal_dist_p95_m": 4.0,
        "goal_progress_mean_m": 7.0,
        "goal_progress_fraction_mean": completion,
        "goal_progress_fraction_p05": completion,
        "mean_time_to_goal_s": 1.0,
        "deadlock_time_pct": 0.0,
        "jerk_mean": 0.0,
        "planner_ms_per_tick_per_agent_mean": p95_ms,
        "planner_ms_per_tick_per_agent_p95": p95_ms,
        "obs_neighbors_mean": 2.0,
        "obs_v2v_fraction": 1.0,
        "obs_sensor_fraction": 0.0,
        "obs_stale_fraction": 0.0,
        "obs_sensor_track_stale_fraction": 0.0,
        "obs_sensor_track_age_mean_s": 0.0,
        "obs_sensor_track_age_p95_s": 0.0,
        "obs_occluded_fraction": 0.0,
        "obs_empty_fraction": 0.0,
        "comm_agent_msg_attempted": 0,
        "comm_agent_msg_scheduled": 0,
        "comm_agent_msg_delivered": 0,
        "comm_agent_msg_dropped": 0,
        "comm_agent_msg_expired": 0,
        "comm_agent_msg_bytes_scheduled": 0,
        "comm_agent_msg_bytes_delivered": 0,
        "comm_agent_msg_bandwidth_Bps": 0.0,
        "comm_agent_msg_drop_fraction": 0.0,
        "comm_agent_msg_delivery_fraction": 1.0,
        "comm_negotiation_proposals": 0,
        "comm_negotiation_acks": 0,
        "comm_negotiation_correlations_acked": 0,
        "comm_negotiation_rejections": 0,
        "planner_timeout_count": 0,
        "planner_error_count": 0,
        "planner_fallback_count": 0,
        "episode_runtime_s": 0.01,
    }


def test_parse_learned_policy_entries_accepts_labeled_temporal_spec(tmp_path: Path) -> None:
    spec_path = _policy_spec(tmp_path / "policy_spec.json")

    entries = parse_learned_policy_spec_entries([f"temporal={spec_path}"])

    assert len(entries) == 1
    assert entries[0].label == "temporal"
    assert entries[0].policy_name == "temporal_fixture"
    assert entries[0].adapter == "temporal_mlp_json"


def test_learned_holdout_eval_writes_scores_and_reference_deltas(tmp_path: Path, monkeypatch) -> None:
    spec_path = _policy_spec(tmp_path / "policy_spec.json")
    seen_agent_methods = []

    def fake_run_episode(spec):
        seen_agent_methods.append((spec.method, spec.agent_methods))
        if spec.method == "learned_policy_spec":
            return _fake_result_row(spec, collision_episode=1, min_sep=-0.2, completion=0.25, p95_ms=4.0)
        return _fake_result_row(spec, collision_episode=0, min_sep=1.2, completion=0.75, p95_ms=8.0)

    monkeypatch.setattr("microbench.rl.learned_holdout.run_episode", fake_run_episode)

    report = run_learned_holdout_eval(
        out_dir=tmp_path / "holdout",
        policy_specs=[f"temporal={spec_path}"],
        reference_methods=["ego_swarm_opt"],
        scenarios=["noncooperative_intruder_3d_hard"],
        seeds=[0],
        comm_profiles=["ideal_50hz"],
        n_agents=4,
    )

    assert report["schema_version"] == LEARNED_HOLDOUT_EVAL_SCHEMA_VERSION
    assert report["ok"] is True
    assert report["expected_runs_per_entry"] == 1
    assert report["learned_rows"][0]["adapter"] == "temporal_mlp_json"
    assert report["learned_rows"][0]["collision_episodes"] == 1
    assert report["reference_rows"][0]["collision_episodes"] == 0
    assert report["pairwise_deltas"][0]["collision_episodes_delta"] == 1.0
    assert report["pairwise_deltas"][0]["completion_rate_mean_delta"] == -0.5
    assert (tmp_path / "holdout" / "learned_holdout_table.csv").exists()
    assert (tmp_path / "holdout" / "learned_holdout_deltas.csv").exists()
    assert seen_agent_methods[0][1] == ["baseline_goal", "learned_policy_spec", "learned_policy_spec", "learned_policy_spec"]
    assert seen_agent_methods[1][1] == ["baseline_goal", "ego_swarm_opt", "ego_swarm_opt", "ego_swarm_opt"]


def test_learned_holdout_eval_collision_gate_can_fail(tmp_path: Path, monkeypatch) -> None:
    spec_path = _policy_spec(tmp_path / "policy_spec.json")

    def fake_run_episode(spec):
        return _fake_result_row(spec, collision_episode=1, min_sep=-0.1, completion=0.0, p95_ms=3.0)

    monkeypatch.setattr("microbench.rl.learned_holdout.run_episode", fake_run_episode)

    report = run_learned_holdout_eval(
        out_dir=tmp_path / "holdout_gate",
        policy_specs=[str(spec_path)],
        reference_methods=["baseline_goal"],
        scenarios=["sphere_swap_3d_medium"],
        seeds=[0],
        comm_profiles=["ideal_50hz"],
        n_agents=3,
        require_no_collision=True,
    )

    assert report["ok"] is False
    assert any(check["name"] == "learned_policies_collision_free" and not check["ok"] for check in report["checks"])


def test_learned_holdout_eval_records_timeout_rows(tmp_path: Path, monkeypatch) -> None:
    spec_path = _policy_spec(tmp_path / "policy_spec.json")

    def fake_run_episode(spec):
        if spec.method == "baseline_goal":
            raise LearnedHoldoutRunTimeout("synthetic timeout")
        return _fake_result_row(spec, collision_episode=0, min_sep=1.0, completion=1.0, p95_ms=2.0)

    monkeypatch.setattr("microbench.rl.learned_holdout.run_episode", fake_run_episode)

    report = run_learned_holdout_eval(
        out_dir=tmp_path / "holdout_timeout",
        policy_specs=[str(spec_path)],
        reference_methods=["baseline_goal"],
        scenarios=["sphere_swap_3d_medium"],
        seeds=[0],
        comm_profiles=["ideal_50hz"],
        n_agents=3,
        run_timeout_s=0.5,
    )

    assert report["ok"] is False
    assert report["run_timeout_s"] == 0.5
    assert report["reference_rows"][0]["planner_timeout_count"] == 1
    assert report["execution"]["baseline_goal"]["timeout_run_count"] == 1
    assert any(check["name"] == "no_planner_timeouts" and not check["ok"] for check in report["checks"])


def test_learned_holdout_eval_cli_smoke_max_runs_zero(tmp_path: Path) -> None:
    spec_path = _policy_spec(tmp_path / "policy_spec.json")
    out_dir = tmp_path / "cli_holdout"

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "microbench.cli",
            "learned-holdout-eval",
            "--out-dir",
            str(out_dir),
            "--policy-spec",
            f"temporal={spec_path}",
            "--reference-methods",
            "baseline_goal",
            "--scenarios",
            "sphere_swap_3d_medium",
            "--seeds",
            "0",
            "--comm",
            "ideal_50hz",
            "--n",
            "3",
            "--max-runs",
            "0",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["schema_version"] == LEARNED_HOLDOUT_EVAL_SCHEMA_VERSION
    assert payload["ok"] is False
    assert payload["reference_methods"] == ["baseline_goal"]
    assert payload["rows"][0]["adapter"] == "temporal_mlp_json"
    assert (out_dir / "learned_holdout_eval.json").exists()
    assert DEFAULT_LEARNED_HOLDOUT_REFERENCE_METHODS == ("dynamic_tube_dmpc", "ego_swarm_opt")
