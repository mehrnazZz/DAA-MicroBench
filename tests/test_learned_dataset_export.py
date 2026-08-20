from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from microbench.rl import (
    LEARNED_DENSE_SWARM_HARD_NEGATIVE_LANE_ID,
    LEARNED_DATASET_SCHEMA_VERSION,
    LEARNED_DATASET_TEACHER_POLICY,
    export_learned_policy_dataset,
    selected_learned_dataset_lanes,
)


ROOT = Path(__file__).resolve().parents[1]


def test_learned_dataset_export_writes_public_teacher_samples_and_replay(tmp_path: Path) -> None:
    out_dir = tmp_path / "learned_dataset"
    report = export_learned_policy_dataset(
        out_dir=out_dir,
        policy=LEARNED_DATASET_TEACHER_POLICY,
        lanes=["head_on"],
        max_steps=2,
        shard_size=5,
        save_replay=True,
    )

    assert report["schema_version"] == LEARNED_DATASET_SCHEMA_VERSION
    assert report["ok"] is True
    assert report["action_source"] == "bc_teacher"
    assert report["policy"] == "local_lateral_avoidance_teacher_v0"
    assert report["sample_count"] == 8
    assert report["episode_count"] == 1
    assert len(report["shards"]) == 2
    assert Path(report["manifest"]).exists()
    assert Path(report["episodes_csv"]).exists()
    assert report["public_observations_only"] is True
    assert report["privileged_global_state"] is False

    first_shard = np.load(report["shards"][0])
    assert first_shard["observations"].shape == (5, 89)
    assert first_shard["next_observations"].shape == (5, 89)
    assert first_shard["actions"].shape == (5, 3)
    assert np.all(np.isfinite(first_shard["observations"]))
    assert np.all(np.isfinite(first_shard["actions"]))
    assert set(first_shard.files) >= {
        "observations",
        "next_observations",
        "actions",
        "rewards",
        "done",
        "lane_id",
        "agent_id",
        "collision",
        "near_miss",
    }
    assert set(str(value) for value in first_shard["lane_id"]) == {"head_on"}

    replay_path = Path(report["episodes"][0]["replay_path"])
    lines = replay_path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["kind"] == "meta"
    assert json.loads(lines[1])["kind"] == "frame"
    assert json.loads(lines[1])["lane_id"] == "head_on"


def test_learned_dataset_export_plan_only(tmp_path: Path) -> None:
    report = export_learned_policy_dataset(
        out_dir=tmp_path / "learned_dataset_plan",
        lanes=["head_on", "crossing"],
        seeds=[0, 1],
        max_steps=1,
        plan_only=True,
    )

    assert report["plan_only"] is True
    assert report["ok"] is False
    assert report["planned_episode_count"] == 4
    assert report["sample_count"] == 0
    assert {entry["lane_id"] for entry in report["matrix"]} == {"head_on", "crossing"}
    assert Path(tmp_path / "learned_dataset_plan" / "learned_dataset_manifest.json").exists()


def test_learned_dataset_export_supports_dense_swarm_hard_negative_lane(tmp_path: Path) -> None:
    default_lane_ids = {lane.lane_id for lane in selected_learned_dataset_lanes(None)}
    assert LEARNED_DENSE_SWARM_HARD_NEGATIVE_LANE_ID not in default_lane_ids

    lanes = selected_learned_dataset_lanes([LEARNED_DENSE_SWARM_HARD_NEGATIVE_LANE_ID])
    assert [lane.lane_id for lane in lanes] == [LEARNED_DENSE_SWARM_HARD_NEGATIVE_LANE_ID]
    assert lanes[0].scenario == "dense_swarm_3d_hard"
    assert lanes[0].comm_profile == "degraded_20hz"

    report = export_learned_policy_dataset(
        out_dir=tmp_path / "dense_plan",
        lanes=[LEARNED_DENSE_SWARM_HARD_NEGATIVE_LANE_ID],
        plan_only=True,
    )

    assert report["plan_only"] is True
    assert report["planned_episode_count"] == 1
    assert report["matrix"][0]["scenario"] == "dense_swarm_3d_hard"
    assert LEARNED_DENSE_SWARM_HARD_NEGATIVE_LANE_ID in report["extra_lane_ids"]


def test_learned_dataset_export_cli_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "learned_dataset_cli"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "microbench.cli",
            "learned-dataset-export",
            "--out-dir",
            str(out_dir),
            "--policy",
            "bc_teacher",
            "--lanes",
            "head_on",
            "--max-steps",
            "2",
            "--shard-size",
            "4",
            "--save-replay",
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
    assert report["sample_count"] == 8
    assert len(report["shards"]) == 2
    assert (out_dir / "learned_dataset_manifest.json").exists()
    assert (out_dir / "learned_dataset_episodes.csv").exists()
    assert (out_dir / "replay").exists()
