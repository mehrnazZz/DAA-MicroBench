from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

from microbench.replay.foxglove_export import (
    WORLD_FRAME,
    _json_bytes,
    build_foxglove_frame_messages,
    build_foxglove_static_scene,
)
from microbench.replay.replay_interactive import _load_trace


ROOT = Path(__file__).resolve().parents[1]
GOLDEN_TRACE = ROOT / "golden" / "traces" / "trace_collision_0_9_t15.18.jsonl"


def test_static_scene_builds_world_bounds_and_obstacles() -> None:
    scene = build_foxglove_static_scene(
        {
            "world_bounds": {
                "xmin": -10.0,
                "xmax": 10.0,
                "ymin": -2.0,
                "ymax": 8.0,
                "zmin": -5.0,
                "zmax": 5.0,
            },
            "obstacles": [
                {
                    "aabb": {
                        "center": [1.0, 2.0, 3.0],
                        "half": [1.0, 2.0, 3.0],
                    }
                }
            ],
        }
    )

    assert scene["deletions"] == []
    assert [entity["id"] for entity in scene["entities"]] == ["world_bounds", "obstacle_0"]
    assert len(scene["entities"][0]["lines"][0]["points"]) == 24
    obstacle = scene["entities"][1]["cubes"][0]
    assert obstacle["pose"]["position"] == {"x": 1.0, "y": 3.0, "z": 2.0}
    assert obstacle["size"] == {"x": 2.0, "y": 6.0, "z": 4.0}


def test_frame_messages_map_native_altitude_to_foxglove_z_up() -> None:
    meta = {"agent_ids": [7], "agent_params": {"radius_m": 0.4}}
    frames = [
        {
            "kind": "frame",
            "t": 1.25,
            "agent_ids": [7],
            "positions": [[1.0, 2.0, 3.0]],
            "velocities": [[0.0, 1.0, 0.0]],
            "v_cmd": [[0.0, 0.0, 1.0]],
            "speed_saturated": [False],
            "accel_saturated": [True],
            "selected_obs": {
                "7": [
                    {
                        "idx": 8,
                        "valid": True,
                        "msg_age_sec": 0.12,
                        "pos": [4.0, 5.0, 6.0],
                        "vel": [0.0, 0.0, 0.0],
                        "radius": 0.4,
                    }
                ]
            },
        }
    ]

    messages = build_foxglove_frame_messages(meta=meta, frames=frames, frame_idx=0)
    transform = messages["transforms"]["transforms"][0]
    assert transform["parent_frame_id"] == WORLD_FRAME
    assert transform["child_frame_id"] == "drone_7"
    assert transform["translation"] == {"x": 1.0, "y": 3.0, "z": 2.0}

    agent_entity = messages["agents"]["entities"][0]
    assert agent_entity["frame_id"] == "drone_7"
    assert agent_entity["spheres"][0]["size"] == {"x": 0.8, "y": 0.8, "z": 0.8}
    assert agent_entity["texts"][0]["text"] == "agent 7"

    link_points = messages["sensing_links"]["entities"][0]["lines"][0]["points"]
    assert link_points == [{"x": 1.0, "y": 3.0, "z": 2.0}, {"x": 4.0, "y": 6.0, "z": 5.0}]

    diagnostics = messages["diagnostics"]
    assert diagnostics["frame"] == 0
    assert diagnostics["n_agents"] == 1
    assert diagnostics["selected_obs_count"] == 1
    assert diagnostics["max_msg_age_sec"] == 0.12
    assert diagnostics["accel_saturated_count"] == 1


def test_frame_messages_cover_golden_collision_trace() -> None:
    meta, frames = _load_trace(str(GOLDEN_TRACE))
    messages = build_foxglove_frame_messages(
        meta=meta,
        frames=frames,
        frame_idx=0,
        trail_frames=3,
        max_sensing_links=4,
    )

    assert set(messages) == {"transforms", "agents", "trails", "sensing_links", "diagnostics"}
    transforms = messages["transforms"]["transforms"]
    assert len(transforms) == 10
    native_pos = frames[0]["positions"][0]
    assert transforms[0]["translation"] == {
        "x": native_pos[0],
        "y": native_pos[2],
        "z": native_pos[1],
    }
    assert transforms[0]["parent_frame_id"] == WORLD_FRAME
    assert transforms[0]["child_frame_id"] == "drone_0"

    assert len(messages["agents"]["entities"]) == 10
    assert messages["agents"]["entities"][0]["spheres"]
    assert messages["agents"]["entities"][0]["arrows"]
    assert len(messages["trails"]["entities"]) == 10

    link_line = messages["sensing_links"]["entities"][0]["lines"][0]
    assert len(link_line["points"]) <= 8
    assert len(link_line["points"]) == len(link_line["colors"])
    assert messages["diagnostics"]["min_center_distance_m"] is not None
    assert messages["diagnostics"]["selected_obs_count"] > 0


def test_json_bytes_sanitizes_nonfinite_event_values() -> None:
    payload = json.loads(_json_bytes({"ttc_s": math.inf, "nested": {"bad": math.nan}}).decode("utf-8"))
    assert payload == {"ttc_s": "Infinity", "nested": {"bad": "NaN"}}


def test_foxglove_cli_reports_missing_optional_dependency_or_writes_mcap(tmp_path: Path) -> None:
    out_path = tmp_path / "episode.mcap"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "microbench.cli",
            "foxglove-export",
            "--trace",
            str(GOLDEN_TRACE),
            "--out",
            str(out_path),
            "--trail-frames",
            "4",
            "--max-sensing-links",
            "4",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    if importlib.util.find_spec("mcap") is None:
        assert proc.returncode != 0
        assert "optional MCAP dependency" in proc.stderr + proc.stdout
        assert not out_path.exists() or out_path.stat().st_size == 0
    else:
        assert proc.returncode == 0, proc.stderr
        assert out_path.exists()
        assert out_path.stat().st_size > 16
