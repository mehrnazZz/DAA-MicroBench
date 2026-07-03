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
    status: str = "pre_v1_official"
    source: str = "generated"
    dimensions: tuple[str, ...] = ("2d", "3d")


@dataclass(frozen=True)
class SuiteRegistryEntry:
    suite_id: str
    status: str
    source: str
    dimensions: tuple[str, ...]
    description: str
    scenario_ids: tuple[str, ...]
    default_methods: tuple[str, ...]
    n_agents: tuple[int, ...]
    seeds: tuple[int, ...]
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
    "merge_3d_hard": _family(
        scenario_id="merge_3d_hard",
        family="merge",
        dimension="3d",
        difficulty="hard",
        purpose="Converging 3D streams merge through a constrained shared exit volume.",
        expected_failure_modes=("late_merge", "vertical_squeeze", "bottleneck_deadlock", "stale_intent"),
        recommended_n=(8, 12, 20),
        recommended_seeds=tuple(range(10)),
        recommended_comm_profiles=("ideal_50hz", "realistic_v2v_50hz", "degraded_20hz"),
        config={
            "scenario": {
                "description": "Generated 3D merge through a constrained exit corridor.",
                "duration_s": 45.0,
            },
            "world": {
                "planar": False,
                "bounds": {"xmin": -58.0, "xmax": 58.0, "ymin": -16.0, "ymax": 18.0, "zmin": -36.0, "zmax": 36.0},
            },
            "agent_params": {"radius_m": 0.5, "v_max_mps": 3.0, "a_max_mps2": 2.0, "goal_tolerance_m": 1.0},
            "goals": {"min_goal_distance_m": 55.0},
            "spawn": {
                "type": "rect_to_rect",
                "start_region": {"center": [-44.0, 0.0, -12.0], "half": [2.0, 10.0, 16.0]},
                "goal_region": {"center": [44.0, 4.0, 0.0], "half": [3.0, 8.0, 5.0]},
                "start_layers_m": [-8.0, -3.0, 3.0, 8.0],
                "goal_layers_m": [-2.0, 2.0, 6.0],
            },
            "obstacles": [
                {"aabb": {"center": [0.0, 4.0, -14.0], "half": [4.0, 8.0, 8.0]}},
                {"aabb": {"center": [0.0, 4.0, 14.0], "half": [4.0, 8.0, 8.0]}},
            ],
            "perception": {
                "mode": "fused",
                "sensor": {
                    "range_m": 28.0,
                    "fov_deg": 170.0,
                    "occlusion": True,
                    "occlusion_margin_m": 0.2,
                    "false_negative_p": 0.03,
                    "noise_sigma_pos_m": 0.05,
                    "noise_sigma_vel_mps": 0.05,
                    "track_ttl_s": 0.25,
                },
            },
        },
    ),
    "overtake_3d_medium": _family(
        scenario_id="overtake_3d_medium",
        family="overtake",
        dimension="3d",
        difficulty="medium",
        purpose="Same-direction 3D corridor traffic with heterogeneous speeds that induce overtakes.",
        expected_failure_modes=("rear_end_conflict", "unsafe_pass", "altitude_lane_change", "speed_heterogeneity"),
        recommended_n=(8, 12, 20),
        recommended_seeds=tuple(range(10)),
        recommended_comm_profiles=("ideal_50hz", "realistic_v2v_50hz"),
        config={
            "scenario": {
                "description": "Generated 3D overtake corridor with mixed-speed agents.",
                "duration_s": 40.0,
            },
            "world": {
                "planar": False,
                "bounds": {"xmin": -58.0, "xmax": 58.0, "ymin": -12.0, "ymax": 12.0, "zmin": -16.0, "zmax": 16.0},
            },
            "agent_params": {"radius_m": 0.45, "v_max_mps": 3.0, "a_max_mps2": 2.0, "goal_tolerance_m": 1.0},
            "goals": {"min_goal_distance_m": 60.0},
            "spawn": {
                "type": "rect_to_rect",
                "start_region": {"center": [-42.0, 0.0, 0.0], "half": [12.0, 6.0, 4.0]},
                "goal_region": {"center": [44.0, 0.0, 0.0], "half": [3.0, 6.0, 4.0]},
                "start_layers_m": [-6.0, -2.0, 2.0, 6.0],
                "goal_layers_m": [-4.0, 0.0, 4.0],
            },
            "agents": {
                "by_id": {
                    0: {"role": "slow_traffic", "capabilities": {"v_max_mps": 1.6}, "priority": 1},
                    1: {"role": "slow_traffic", "capabilities": {"v_max_mps": 1.8}, "priority": 1},
                    2: {"role": "fast_response", "capabilities": {"v_max_mps": 3.6, "a_max_mps2": 2.5}, "priority": 4},
                }
            },
            "perception": {
                "mode": "fused",
                "sensor": {
                    "range_m": 26.0,
                    "fov_deg": 150.0,
                    "occlusion": False,
                    "false_negative_p": 0.02,
                    "noise_sigma_pos_m": 0.04,
                    "noise_sigma_vel_mps": 0.04,
                    "track_ttl_s": 0.2,
                },
            },
        },
    ),
    "noncooperative_intruder_3d_hard": _family(
        scenario_id="noncooperative_intruder_3d_hard",
        family="noncooperative_intruder",
        dimension="3d",
        difficulty="hard",
        purpose="Sensor-driven 3D encounter with one faster intruder that does not share intent or messages.",
        expected_failure_modes=("unannounced_intruder", "sensor_only_late_detection", "occluded_climb_conflict"),
        recommended_n=(6, 10, 16),
        recommended_seeds=tuple(range(10)),
        recommended_comm_profiles=("ideal_50hz", "realistic_v2v_50hz", "degraded_20hz"),
        config={
            "scenario": {
                "description": "Generated 3D noncooperative intruder encounter.",
                "duration_s": 42.0,
            },
            "world": {
                "planar": False,
                "bounds": {"xmin": -46.0, "xmax": 46.0, "ymin": -36.0, "ymax": 36.0, "zmin": -46.0, "zmax": 46.0},
            },
            "agent_params": {"radius_m": 0.5, "v_max_mps": 3.0, "a_max_mps2": 2.0, "goal_tolerance_m": 1.0},
            "goals": {"min_goal_distance_m": 45.0},
            "spawn": {
                "type": "sphere_swap",
                "center": [0.0, 0.0, 0.0],
                "radius_m": 28.0,
                "jitter_m": 1.5,
                "vertical_scale": 1.45,
                "min_abs_y_component": 0.15,
            },
            "obstacles": [
                {"aabb": {"center": [0.0, 0.0, 0.0], "half": [3.5, 5.0, 3.5]}},
            ],
            "perception": {
                "mode": "sensor",
                "sensor": {
                    "range_m": 26.0,
                    "fov_deg": 160.0,
                    "occlusion": True,
                    "occlusion_margin_m": 0.2,
                    "false_negative_p": 0.04,
                    "noise_sigma_pos_m": 0.07,
                    "noise_sigma_vel_mps": 0.07,
                    "track_ttl_s": 0.3,
                },
            },
            "agents": {
                "by_id": {
                    0: {
                        "method": "baseline_goal",
                        "role": "noncooperative_intruder",
                        "priority": 100,
                        "capabilities": {"v_max_mps": 3.6, "a_max_mps2": 2.6, "radius_m": 0.55},
                        "failure_modes": {"noncooperative": True},
                    }
                }
            },
        },
    ),
    "heterogeneous_priority_crossing_3d_medium": _family(
        scenario_id="heterogeneous_priority_crossing_3d_medium",
        family="heterogeneous_priority",
        dimension="3d",
        difficulty="medium",
        purpose="3D crossing with mixed priorities, roles, capabilities, and altitude layer changes.",
        expected_failure_modes=("priority_inversion", "yield_deadlock", "high_priority_delay", "altitude_commitment_conflict"),
        recommended_n=(8, 12, 20),
        recommended_seeds=tuple(range(10)),
        recommended_comm_profiles=("ideal_50hz", "realistic_v2v_50hz", "degraded_20hz"),
        config={
            "scenario": {
                "description": "Generated 3D heterogeneous-priority crossing.",
                "duration_s": 42.0,
            },
            "world": {
                "planar": False,
                "bounds": {"xmin": -46.0, "xmax": 46.0, "ymin": -6.0, "ymax": 16.0, "zmin": -46.0, "zmax": 46.0},
            },
            "agent_params": {"radius_m": 0.48, "v_max_mps": 3.0, "a_max_mps2": 2.0, "goal_tolerance_m": 1.0},
            "goals": {"min_goal_distance_m": 50.0},
            "spawn": {
                "type": "four_way",
                "extent_m": 40.0,
                "lane_half_width_m": 6.0,
                "y_m": 0.0,
                "start_layers_m": [0.0, 2.0, 8.0, 12.0],
                "goal_layers_m": [12.0, 8.0, 2.0, 0.0],
            },
            "intent": {"enabled": True, "tx_rate_hz": 10.0, "max_points": 12},
            "comm": {
                "message_bus": {
                    "enabled": True,
                    "max_rate_hz": 20.0,
                    "bandwidth_Bps": 60000,
                    "max_message_size_bytes": 512,
                }
            },
            "agents": {
                "defaults": {"role": "standard", "priority": 5},
                "by_id": {
                    0: {
                        "role": "emergency_response",
                        "priority": 100,
                        "capabilities": {"v_max_mps": 3.6, "a_max_mps2": 2.5},
                        "mission": {"right_of_way": "high"},
                    },
                    1: {
                        "role": "survey_heavy",
                        "priority": 20,
                        "capabilities": {"v_max_mps": 2.2, "a_max_mps2": 1.5, "radius_m": 0.6},
                    },
                    2: {
                        "role": "low_priority_delivery",
                        "priority": 1,
                        "capabilities": {"v_max_mps": 2.6},
                    },
                },
            },
            "perception": {
                "mode": "fused",
                "sensor": {
                    "range_m": 28.0,
                    "fov_deg": 180.0,
                    "occlusion": False,
                    "false_negative_p": 0.02,
                    "noise_sigma_pos_m": 0.04,
                    "noise_sigma_vel_mps": 0.04,
                    "track_ttl_s": 0.25,
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
            "merge_3d_hard",
            "vertical_crossing_3d_hard",
            "heterogeneous_priority_crossing_3d_medium",
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
        scenario_ids=(
            "sphere_swap_3d_medium",
            "merge_3d_hard",
            "overtake_3d_medium",
            "vertical_crossing_3d_hard",
            "sensor_volume_3d_hard",
            "noncooperative_intruder_3d_hard",
            "heterogeneous_priority_crossing_3d_medium",
        ),
        default_methods=("orca_expert",),
        n_agents=(6, 10),
        stretch_n_agents=(6, 10, 20),
        seeds=tuple(range(10)),
        stretch_seeds=tuple(range(30)),
        comm_profiles=("ideal_50hz", "realistic_v2v_50hz", "degraded_20hz"),
        dimensions=("3d",),
    ),
    "official_agentic_stress": OfficialSuite(
        suite_id="official_agentic_stress",
        description="Pre-v1 official agentic stress suite for heterogeneous priorities and noncooperative traffic.",
        scenario_ids=(
            "heterogeneous_priority_crossing_3d_medium",
            "noncooperative_intruder_3d_hard",
            "sensor_volume_3d_hard",
            "vertical_crossing_3d_hard",
        ),
        default_methods=("priority_yield", "negotiation_yield", "orca_expert"),
        n_agents=(6, 10),
        stretch_n_agents=(6, 10, 16),
        seeds=tuple(range(10)),
        stretch_seeds=tuple(range(20)),
        comm_profiles=("ideal_50hz", "realistic_v2v_50hz", "degraded_20hz"),
        dimensions=("3d",),
    ),
}


HANDWRITTEN_SUITE_REGISTRY: tuple[SuiteRegistryEntry, ...] = (
    SuiteRegistryEntry(
        suite_id="primary",
        status="legacy_official",
        source="hand_written",
        dimensions=("2d",),
        description="Legacy hand-written planar canonical suite.",
        scenario_ids=("corridor", "intersection", "funnel", "ring", "crowd_swap", "weather_event"),
        default_methods=(),
        n_agents=(10, 20, 50),
        seeds=tuple(range(50)),
        comm_profiles=("ideal_50hz", "realistic_v2v_50hz", "degraded_20hz"),
    ),
    SuiteRegistryEntry(
        suite_id="baseline_sanity",
        status="smoke",
        source="hand_written",
        dimensions=("2d",),
        description="Small planar baseline sanity suite for quick method comparison.",
        scenario_ids=("corridor", "intersection", "funnel", "ring", "crowd_swap", "weather_event"),
        default_methods=("baseline_goal", "orca_expert"),
        n_agents=(10, 20),
        seeds=tuple(range(20)),
        comm_profiles=("ideal_50hz", "realistic_v2v_50hz"),
    ),
    SuiteRegistryEntry(
        suite_id="three_d",
        status="development",
        source="hand_written",
        dimensions=("3d",),
        description="Hand-written 3D development suite for non-planar planner debugging.",
        scenario_ids=(
            "stacked_swap_3d",
            "layered_funnel_3d",
            "layered_intersection_3d",
            "weather_vertical_event_3d",
            "vertical_crossing_obstacles_3d",
        ),
        default_methods=("orca_expert",),
        n_agents=(6, 10),
        seeds=tuple(range(10)),
        comm_profiles=("ideal_50hz",),
    ),
    SuiteRegistryEntry(
        suite_id="perception_stress",
        status="development",
        source="hand_written",
        dimensions=("2d",),
        description="Hand-written partial-observation suite for sensor/fused perception stress.",
        scenario_ids=("perception_sensor_occlusion", "perception_fused_degraded", "perception_stale_tracks"),
        default_methods=("priority_yield",),
        n_agents=(6, 10),
        seeds=tuple(range(10)),
        comm_profiles=("ideal_50hz", "degraded_20hz"),
    ),
)


def list_official_suites() -> list[str]:
    return sorted(OFFICIAL_SUITES)


def list_scenario_families() -> list[str]:
    return sorted(SCENARIO_FAMILIES)


def suite_registry_entries() -> list[SuiteRegistryEntry]:
    generated = [
        SuiteRegistryEntry(
            suite_id=suite.suite_id,
            status=suite.status,
            source=suite.source,
            dimensions=suite.dimensions,
            description=suite.description,
            scenario_ids=suite.scenario_ids,
            default_methods=suite.default_methods,
            n_agents=suite.n_agents,
            seeds=suite.seeds,
            comm_profiles=suite.comm_profiles,
        )
        for suite in OFFICIAL_SUITES.values()
    ]
    return sorted([*generated, *HANDWRITTEN_SUITE_REGISTRY], key=lambda x: x.suite_id)


def suite_registry_dicts() -> list[dict]:
    out = []
    for entry in suite_registry_entries():
        out.append(
            {
                "suite": entry.suite_id,
                "status": entry.status,
                "source": entry.source,
                "dimensions": list(entry.dimensions),
                "description": entry.description,
                "scenarios": list(entry.scenario_ids),
                "default_methods": list(entry.default_methods),
                "n_agents": list(entry.n_agents),
                "seed_count": len(entry.seeds),
                "seed_min": min(entry.seeds) if entry.seeds else None,
                "seed_max": max(entry.seeds) if entry.seeds else None,
                "comm_profiles": list(entry.comm_profiles),
            }
        )
    return out


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
