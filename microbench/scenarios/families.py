from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy

import yaml


@dataclass(frozen=True)
class ScenarioFamily:
    scenario_id: str
    family: str
    dimension: str
    difficulty: str
    purpose: str
    expected_failure_modes: tuple[str, ...]
    recommended_n: tuple[int, ...]
    recommended_seeds: tuple[int, ...]
    recommended_comm_profiles: tuple[str, ...]
    config: dict


@dataclass(frozen=True)
class OfficialSuite:
    suite_id: str
    description: str
    scenario_ids: tuple[str, ...]
    default_methods: tuple[str, ...]
    n_agents: tuple[int, ...]
    stretch_n_agents: tuple[int, ...]
    seeds: tuple[int, ...]
    stretch_seeds: tuple[int, ...]
    comm_profiles: tuple[str, ...]


def _benchmark_meta(
    *,
    family: str,
    dimension: str,
    difficulty: str,
    purpose: str,
    expected_failure_modes: tuple[str, ...],
    recommended_n: tuple[int, ...],
    recommended_seeds: tuple[int, ...],
    recommended_comm_profiles: tuple[str, ...],
) -> dict:
    return {
        "family": family,
        "dimension": dimension,
        "difficulty": difficulty,
        "purpose": purpose,
        "expected_failure_modes": list(expected_failure_modes),
        "recommended": {
            "n_agents": list(recommended_n),
            "seeds": list(recommended_seeds),
            "comm_profiles": list(recommended_comm_profiles),
        },
    }


def _family(
    *,
    scenario_id: str,
    family: str,
    dimension: str,
    difficulty: str,
    purpose: str,
    expected_failure_modes: tuple[str, ...],
    recommended_n: tuple[int, ...],
    recommended_seeds: tuple[int, ...],
    recommended_comm_profiles: tuple[str, ...],
    config: dict,
) -> ScenarioFamily:
    cfg = copy.deepcopy(config)
    cfg["benchmark"] = _benchmark_meta(
        family=family,
        dimension=dimension,
        difficulty=difficulty,
        purpose=purpose,
        expected_failure_modes=expected_failure_modes,
        recommended_n=recommended_n,
        recommended_seeds=recommended_seeds,
        recommended_comm_profiles=recommended_comm_profiles,
    )
    cfg.setdefault("scenario", {})["name"] = scenario_id
    return ScenarioFamily(
        scenario_id=scenario_id,
        family=family,
        dimension=dimension,
        difficulty=difficulty,
        purpose=purpose,
        expected_failure_modes=expected_failure_modes,
        recommended_n=recommended_n,
        recommended_seeds=recommended_seeds,
        recommended_comm_profiles=recommended_comm_profiles,
        config=cfg,
    )


SCENARIO_FAMILIES: dict[str, ScenarioFamily] = {
    "head_on_2d_easy": _family(
        scenario_id="head_on_2d_easy",
        family="head_on",
        dimension="2d",
        difficulty="easy",
        purpose="Planar reciprocal encounter with simple crossing geometry.",
        expected_failure_modes=("late_yield", "symmetric_avoidance", "stale_v2v"),
        recommended_n=(2, 6),
        recommended_seeds=tuple(range(5)),
        recommended_comm_profiles=("ideal_50hz", "realistic_v2v_50hz"),
        config={
            "scenario": {
                "description": "Generated planar head-on corridor encounter.",
                "duration_s": 30.0,
            },
            "world": {
                "planar": True,
                "fixed_y_m": 0.0,
                "bounds": {"xmin": -42.0, "xmax": 42.0, "ymin": -1.0, "ymax": 1.0, "zmin": -12.0, "zmax": 12.0},
            },
            "agent_params": {"radius_m": 0.45, "v_max_mps": 3.0, "a_max_mps2": 2.0, "goal_tolerance_m": 1.0},
            "goals": {"min_goal_distance_m": 55.0},
            "spawn": {
                "type": "rect_to_rect",
                "start_region": {"center": [-32.0, 0.0, 0.0], "half": [1.0, 0.0, 4.0]},
                "goal_region": {"center": [32.0, 0.0, 0.0], "half": [1.0, 0.0, 4.0]},
            },
        },
    ),
    "crossing_2d_medium": _family(
        scenario_id="crossing_2d_medium",
        family="crossing",
        dimension="2d",
        difficulty="medium",
        purpose="Planar four-way crossing with converging flows.",
        expected_failure_modes=("priority_inversion", "deadlock", "dense_center_conflict"),
        recommended_n=(8, 12, 20),
        recommended_seeds=tuple(range(10)),
        recommended_comm_profiles=("ideal_50hz", "realistic_v2v_50hz", "degraded_20hz"),
        config={
            "scenario": {
                "description": "Generated planar four-way crossing.",
                "duration_s": 40.0,
            },
            "world": {
                "planar": True,
                "fixed_y_m": 0.0,
                "bounds": {"xmin": -45.0, "xmax": 45.0, "ymin": -1.0, "ymax": 1.0, "zmin": -45.0, "zmax": 45.0},
            },
            "agent_params": {"radius_m": 0.45, "v_max_mps": 3.0, "a_max_mps2": 2.0, "goal_tolerance_m": 1.0},
            "goals": {"min_goal_distance_m": 50.0},
            "spawn": {"type": "four_way", "extent_m": 38.0, "lane_half_width_m": 5.0, "y_m": 0.0},
        },
    ),
    "funnel_2d_hard": _family(
        scenario_id="funnel_2d_hard",
        family="funnel",
        dimension="2d",
        difficulty="hard",
        purpose="Planar bottleneck with static obstacle geometry.",
        expected_failure_modes=("bottleneck_deadlock", "wall_hugging", "late_merge"),
        recommended_n=(12, 20, 30),
        recommended_seeds=tuple(range(10)),
        recommended_comm_profiles=("ideal_50hz", "realistic_v2v_50hz", "degraded_20hz"),
        config={
            "scenario": {
                "description": "Generated planar obstacle funnel.",
                "duration_s": 45.0,
            },
            "world": {
                "planar": True,
                "fixed_y_m": 0.0,
                "bounds": {"xmin": -50.0, "xmax": 50.0, "ymin": -1.0, "ymax": 1.0, "zmin": -22.0, "zmax": 22.0},
            },
            "agent_params": {"radius_m": 0.45, "v_max_mps": 3.0, "a_max_mps2": 2.0, "goal_tolerance_m": 1.0},
            "goals": {"min_goal_distance_m": 70.0},
            "spawn": {
                "type": "rect_to_rect",
                "start_region": {"center": [-42.0, 0.0, 0.0], "half": [1.0, 0.0, 14.0]},
                "goal_region": {"center": [42.0, 0.0, 0.0], "half": [1.0, 0.0, 14.0]},
            },
            "obstacles": [
                {"aabb": {"center": [0.0, 0.0, -10.0], "half": [3.0, 1.0, 8.0]}},
                {"aabb": {"center": [0.0, 0.0, 10.0], "half": [3.0, 1.0, 8.0]}},
            ],
        },
    ),
    "sphere_swap_3d_medium": _family(
        scenario_id="sphere_swap_3d_medium",
        family="dense_volume",
        dimension="3d",
        difficulty="medium",
        purpose="True volumetric antipodal swap through shared airspace.",
        expected_failure_modes=("vertical_climb_conflict", "altitude_layer_crowding", "late_deconfliction"),
        recommended_n=(8, 12, 20),
        recommended_seeds=tuple(range(10)),
        recommended_comm_profiles=("ideal_50hz", "realistic_v2v_50hz"),
        config={
            "scenario": {
                "description": "Generated non-planar sphere swap through a shared 3D volume.",
                "duration_s": 40.0,
            },
            "world": {
                "planar": False,
                "bounds": {"xmin": -42.0, "xmax": 42.0, "ymin": -42.0, "ymax": 42.0, "zmin": -42.0, "zmax": 42.0},
            },
            "agent_params": {"radius_m": 0.5, "v_max_mps": 3.0, "a_max_mps2": 2.0, "goal_tolerance_m": 1.0},
            "goals": {"min_goal_distance_m": 48.0},
            "spawn": {
                "type": "sphere_swap",
                "center": [0.0, 0.0, 0.0],
                "radius_m": 30.0,
                "jitter_m": 1.5,
                "vertical_scale": 1.35,
                "min_abs_y_component": 0.18,
            },
        },
    ),
    "vertical_crossing_3d_hard": _family(
        scenario_id="vertical_crossing_3d_hard",
        family="vertical_crossing",
        dimension="3d",
        difficulty="hard",
        purpose="Layer-changing crossing around a central obstruction.",
        expected_failure_modes=("altitude_commitment_conflict", "obstacle_shadowing", "vertical_deadlock"),
        recommended_n=(8, 12, 20),
        recommended_seeds=tuple(range(10)),
        recommended_comm_profiles=("ideal_50hz", "realistic_v2v_50hz", "degraded_20hz"),
        config={
            "scenario": {
                "description": "Generated 3D vertical crossing with center obstacle.",
                "duration_s": 45.0,
            },
            "world": {
                "planar": False,
                "bounds": {"xmin": -48.0, "xmax": 48.0, "ymin": -6.0, "ymax": 16.0, "zmin": -48.0, "zmax": 48.0},
            },
            "agent_params": {"radius_m": 0.55, "v_max_mps": 3.0, "a_max_mps2": 2.0, "goal_tolerance_m": 1.0},
            "goals": {"min_goal_distance_m": 50.0},
            "obstacles": [{"aabb": {"center": [0.0, 4.0, 0.0], "half": [4.5, 3.0, 4.5]}}],
            "spawn": {
                "type": "four_way",
                "extent_m": 42.0,
                "lane_half_width_m": 7.0,
                "y_m": 0.0,
                "start_layers_m": [0.0, 0.0, 8.0, 8.0],
                "goal_layers_m": [8.0, 8.0, 0.0, 0.0],
            },
            "perception": {
                "mode": "fused",
                "sensor": {
                    "range_m": 28.0,
                    "fov_deg": 180.0,
                    "occlusion": True,
                    "occlusion_margin_m": 0.2,
                    "false_negative_p": 0.02,
                    "noise_sigma_pos_m": 0.04,
                    "noise_sigma_vel_mps": 0.04,
                    "track_ttl_s": 0.25,
                },
            },
        },
    ),
    "sensor_volume_3d_hard": _family(
        scenario_id="sensor_volume_3d_hard",
        family="sensor_degraded_volume",
        dimension="3d",
        difficulty="hard",
        purpose="Volumetric DAA under partial sensing, stale local tracks, and V2V degradation.",
        expected_failure_modes=("stale_track_collision", "fov_blind_spot", "message_sensor_disagreement"),
        recommended_n=(8, 12, 20),
        recommended_seeds=tuple(range(10)),
        recommended_comm_profiles=("realistic_v2v_50hz", "degraded_20hz", "bursty_stress_50hz"),
        config={
            "scenario": {
                "description": "Generated 3D sensor-degraded dense volume stress case.",
                "duration_s": 45.0,
            },
            "world": {
                "planar": False,
                "bounds": {"xmin": -44.0, "xmax": 44.0, "ymin": -44.0, "ymax": 44.0, "zmin": -44.0, "zmax": 44.0},
            },
            "agent_params": {"radius_m": 0.5, "v_max_mps": 3.0, "a_max_mps2": 2.0, "goal_tolerance_m": 1.0},
            "goals": {"min_goal_distance_m": 46.0},
            "spawn": {
                "type": "sphere_swap",
                "center": [0.0, 0.0, 0.0],
                "radius_m": 29.0,
                "jitter_m": 2.0,
                "vertical_scale": 1.5,
                "min_abs_y_component": 0.12,
            },
            "obstacles": [
                {"aabb": {"center": [0.0, 0.0, 0.0], "half": [3.0, 5.0, 3.0]}},
            ],
            "perception": {
                "mode": "fused",
                "sensor": {
                    "range_m": 24.0,
                    "fov_deg": 150.0,
                    "occlusion": True,
                    "occlusion_margin_m": 0.2,
                    "false_negative_p": 0.06,
                    "noise_sigma_pos_m": 0.08,
                    "noise_sigma_vel_mps": 0.08,
                    "track_ttl_s": 0.35,
                },
            },
        },
    ),
}


OFFICIAL_SUITES: dict[str, OfficialSuite] = {
    "official_alpha": OfficialSuite(
        suite_id="official_alpha",
        description="Pre-v1 official alpha suite mixing planar and 3D DAA families.",
        scenario_ids=(
            "head_on_2d_easy",
            "crossing_2d_medium",
            "funnel_2d_hard",
            "sphere_swap_3d_medium",
            "vertical_crossing_3d_hard",
            "sensor_volume_3d_hard",
        ),
        default_methods=("baseline_goal", "orca_expert"),
        n_agents=(6, 10),
        stretch_n_agents=(6, 10, 20),
        seeds=tuple(range(5)),
        stretch_seeds=tuple(range(20)),
        comm_profiles=("ideal_50hz", "realistic_v2v_50hz", "degraded_20hz"),
    ),
    "official_3d_stress": OfficialSuite(
        suite_id="official_3d_stress",
        description="Pre-v1 official 3D stress suite for volumetric and vertical DAA evaluation.",
        scenario_ids=("sphere_swap_3d_medium", "vertical_crossing_3d_hard", "sensor_volume_3d_hard"),
        default_methods=("orca_expert",),
        n_agents=(6, 10),
        stretch_n_agents=(6, 10, 20),
        seeds=tuple(range(10)),
        stretch_seeds=tuple(range(30)),
        comm_profiles=("ideal_50hz", "realistic_v2v_50hz", "degraded_20hz"),
    ),
}


def list_official_suites() -> list[str]:
    return sorted(OFFICIAL_SUITES)


def list_scenario_families() -> list[str]:
    return sorted(SCENARIO_FAMILIES)


def _has_3d_volume(cfg: dict) -> bool:
    world = cfg.get("world", {})
    bounds = world.get("bounds", {})
    ymin = float(bounds.get("ymin", 0.0))
    ymax = float(bounds.get("ymax", 0.0))
    if ymax - ymin > 1e-6:
        return True

    spawn = cfg.get("spawn", {})
    if spawn.get("type") == "sphere_swap":
        return True
    for key in ("start_region", "goal_region"):
        half = spawn.get(key, {}).get("half", [0.0, 0.0, 0.0])
        if len(half) >= 2 and abs(float(half[1])) > 1e-6:
            return True
    layers = list(spawn.get("start_layers_m", []) or []) + list(spawn.get("goal_layers_m", []) or [])
    return len({float(x) for x in layers}) > 1


def validate_scenario_family(family: ScenarioFamily) -> None:
    cfg = family.config
    for key in ("scenario", "world", "agent_params", "goals", "spawn", "benchmark"):
        if key not in cfg:
            raise ValueError(f"{family.scenario_id} missing required key: {key}")

    dimension = str(cfg.get("benchmark", {}).get("dimension", family.dimension)).lower()
    planar = bool(cfg.get("world", {}).get("planar", True))
    if dimension == "2d" and not planar:
        raise ValueError(f"{family.scenario_id} is marked 2d but world.planar is false")
    if dimension == "3d":
        if planar:
            raise ValueError(f"{family.scenario_id} is marked 3d but world.planar is true")
        if not _has_3d_volume(cfg):
            raise ValueError(f"{family.scenario_id} is marked 3d but has no vertical volume or layers")


def suite_defaults(suite_id: str, *, stretch: bool = False) -> dict:
    suite = OFFICIAL_SUITES[suite_id]
    return {
        "suite": suite.suite_id,
        "description": suite.description,
        "default_methods": list(suite.default_methods),
        "n_agents": list(suite.stretch_n_agents if stretch else suite.n_agents),
        "seeds": list(suite.stretch_seeds if stretch else suite.seeds),
        "comm_profiles": list(suite.comm_profiles),
    }


def build_suite_manifest(suite_id: str, scenario_paths: list[Path] | None = None, *, stretch: bool = False) -> dict:
    if suite_id not in OFFICIAL_SUITES:
        raise ValueError(f"Unknown official suite: {suite_id}")
    suite = OFFICIAL_SUITES[suite_id]
    paths = scenario_paths or [Path(f"{sid}.yaml") for sid in suite.scenario_ids]
    scenarios = []
    for sid, path in zip(suite.scenario_ids, paths):
        family = SCENARIO_FAMILIES[sid]
        scenarios.append(
            {
                "id": family.scenario_id,
                "path": str(path),
                "family": family.family,
                "dimension": family.dimension,
                "difficulty": family.difficulty,
                "purpose": family.purpose,
                "expected_failure_modes": list(family.expected_failure_modes),
                "recommended": {
                    "n_agents": list(family.recommended_n),
                    "seeds": list(family.recommended_seeds),
                    "comm_profiles": list(family.recommended_comm_profiles),
                },
            }
        )
    manifest = suite_defaults(suite_id, stretch=stretch)
    manifest["scenarios"] = scenarios
    return manifest


def materialize_official_suite(
    suite_id: str,
    out_dir: str | Path,
    *,
    overwrite: bool = False,
    stretch: bool = False,
) -> dict:
    if suite_id not in OFFICIAL_SUITES:
        raise ValueError(f"Unknown official suite: {suite_id}")

    suite = OFFICIAL_SUITES[suite_id]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    scenario_paths: list[Path] = []
    for scenario_id in suite.scenario_ids:
        family = SCENARIO_FAMILIES[scenario_id]
        validate_scenario_family(family)
        path = out / f"{scenario_id}.yaml"
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} already exists; pass overwrite=True to replace generated scenarios")
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(family.config, f, sort_keys=False)
        scenario_paths.append(path)

    manifest = build_suite_manifest(suite_id, [Path(p.name) for p in scenario_paths], stretch=stretch)
    manifest_path = out / "suite_manifest.yaml"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"{manifest_path} already exists; pass overwrite=True to replace generated manifest")
    with manifest_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    return {
        "suite": suite_id,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "scenario_paths": scenario_paths,
    }
