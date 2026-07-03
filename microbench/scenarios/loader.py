from __future__ import annotations

from pathlib import Path
import numpy as np

from microbench.config import load_yaml, deep_merge


def load_scenario(defaults: dict, scenario_path: str | Path) -> dict:
    scenario = load_yaml(scenario_path)
    return deep_merge(defaults, scenario)


def _sample_rect_point(rng: np.random.Generator, center: np.ndarray, half: np.ndarray) -> np.ndarray:
    return center + rng.uniform(-half, half)


def _apply_layer_y(cfg: dict, point: np.ndarray, idx: int, field: str) -> np.ndarray:
    layers = cfg.get(field)
    if not layers:
        return point
    shift = int(cfg.get(f"{field[:-2]}_shift", 0))
    layer_idx = (idx + shift) % len(layers)
    out = point.copy()
    out[1] = float(layers[layer_idx])
    return out


def _rect_to_rect(cfg: dict, n_agents: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    st = cfg.get("start_region", {})
    gt = cfg.get("goal_region", {})
    st_c = np.asarray(st.get("center", [0.0, 0.0, 0.0]), dtype=float)
    st_h = np.asarray(st.get("half", [1.0, 1.0, 1.0]), dtype=float)
    gt_c = np.asarray(gt.get("center", [10.0, 0.0, 0.0]), dtype=float)
    gt_h = np.asarray(gt.get("half", [1.0, 1.0, 1.0]), dtype=float)

    bias = cfg.get("spawn_bias", {})
    bias_type = bias.get("type", "none")
    sigma = float(bias.get("sigma_m", 1.0))

    spawns = np.zeros((n_agents, 3), dtype=float)
    goals = np.zeros((n_agents, 3), dtype=float)
    for i in range(n_agents):
        sp = _sample_rect_point(rng, st_c, st_h)
        if bias_type == "gaussian_z":
            sp[2] = st_c[2] + rng.normal(0.0, sigma)
            sp[2] = np.clip(sp[2], st_c[2] - st_h[2], st_c[2] + st_h[2])
        spawns[i] = _apply_layer_y(cfg, sp, i, "start_layers_m")
        goals[i] = _apply_layer_y(cfg, _sample_rect_point(rng, gt_c, gt_h), i, "goal_layers_m")
    return spawns, goals


def _circle_swap(cfg: dict, n_agents: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(cfg.get("center", [0.0, 0.0, 0.0]), dtype=float)
    radius = float(cfg.get("radius_m", 30.0))
    jitter = float(cfg.get("jitter_m", 1.0))
    spawns = np.zeros((n_agents, 3), dtype=float)
    goals = np.zeros((n_agents, 3), dtype=float)
    base_angles = np.linspace(0.0, 2.0 * np.pi, n_agents, endpoint=False)
    rng.shuffle(base_angles)
    for i, a in enumerate(base_angles):
        rs = radius + rng.normal(0.0, jitter)
        rg = radius + rng.normal(0.0, jitter)
        spawns[i] = _apply_layer_y(
            cfg,
            center + np.array([rs * np.cos(a), 0.0, rs * np.sin(a)], dtype=float),
            i,
            "start_layers_m",
        )
        goals[i] = _apply_layer_y(
            cfg,
            center + np.array([rg * np.cos(a + np.pi), 0.0, rg * np.sin(a + np.pi)], dtype=float),
            i,
            "goal_layers_m",
        )
    return spawns, goals


def _sphere_swap(cfg: dict, n_agents: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(cfg.get("center", [0.0, 0.0, 0.0]), dtype=float)
    radius = float(cfg.get("radius_m", 30.0))
    jitter = float(cfg.get("jitter_m", 1.0))
    vertical_scale = float(cfg.get("vertical_scale", 1.0))
    min_abs_y_component = float(cfg.get("min_abs_y_component", 0.0))

    spawns = np.zeros((n_agents, 3), dtype=float)
    goals = np.zeros((n_agents, 3), dtype=float)
    for i in range(n_agents):
        direction = np.zeros(3, dtype=float)
        for _ in range(128):
            candidate = rng.normal(0.0, 1.0, size=3)
            candidate[1] *= vertical_scale
            n = float(np.linalg.norm(candidate))
            if n < 1e-9:
                continue
            candidate = candidate / n
            if abs(float(candidate[1])) >= min_abs_y_component:
                direction = candidate
                break
        if float(np.linalg.norm(direction)) < 1e-9:
            direction = np.array([0.0, 1.0, 0.0], dtype=float)

        rs = radius + rng.normal(0.0, jitter)
        rg = radius + rng.normal(0.0, jitter)
        spawns[i] = center + direction * rs
        goals[i] = center - direction * rg
    return spawns, goals


def _four_way(cfg: dict, n_agents: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    extent = float(cfg.get("extent_m", 40.0))
    lane_hw = float(cfg.get("lane_half_width_m", 5.0))
    dirs = ["west", "east", "south", "north"]

    def sample_side(side: str, y_m: float) -> np.ndarray:
        if side == "west":
            return np.array([-extent, y_m, rng.uniform(-lane_hw, lane_hw)], dtype=float)
        if side == "east":
            return np.array([extent, y_m, rng.uniform(-lane_hw, lane_hw)], dtype=float)
        if side == "south":
            return np.array([rng.uniform(-lane_hw, lane_hw), y_m, -extent], dtype=float)
        return np.array([rng.uniform(-lane_hw, lane_hw), y_m, extent], dtype=float)

    opposite = {"west": "east", "east": "west", "south": "north", "north": "south"}

    spawns = np.zeros((n_agents, 3), dtype=float)
    goals = np.zeros((n_agents, 3), dtype=float)
    for i in range(n_agents):
        side = dirs[i % 4]
        sp = sample_side(side, float(cfg.get("y_m", 0.0)))
        gt = sample_side(opposite[side], float(cfg.get("y_m", 0.0)))
        spawns[i] = _apply_layer_y(cfg, sp, i, "start_layers_m")
        goals[i] = _apply_layer_y(cfg, gt, i, "goal_layers_m")
    return spawns, goals


def generate_spawns_goals(cfg: dict, n_agents: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    spawn_cfg = cfg.get("spawn", {})
    goals_cfg = cfg.get("goals", {})
    min_goal_dist = float(goals_cfg.get("min_goal_distance_m", 0.0))
    max_attempts = int(goals_cfg.get("max_attempts", 5000))

    stype = spawn_cfg.get("type", "rect_to_rect")
    if stype == "rect_to_rect":
        raw_spawns, raw_goals = _rect_to_rect(spawn_cfg, n_agents, rng)
    elif stype == "circle_swap":
        raw_spawns, raw_goals = _circle_swap(spawn_cfg, n_agents, rng)
    elif stype == "sphere_swap":
        raw_spawns, raw_goals = _sphere_swap(spawn_cfg, n_agents, rng)
    elif stype == "four_way":
        raw_spawns, raw_goals = _four_way(spawn_cfg, n_agents, rng)
    else:
        raise ValueError(f"Unsupported spawn type: {stype}")

    spawns = raw_spawns.copy()
    goals = raw_goals.copy()
    for i in range(n_agents):
        if np.linalg.norm(goals[i] - spawns[i]) >= min_goal_dist:
            continue
        ok = False
        for _ in range(max_attempts):
            if stype == "rect_to_rect":
                _, retry_goals = _rect_to_rect(spawn_cfg, 1, rng)
                candidate = retry_goals[0]
            elif stype == "circle_swap":
                _, retry_goals = _circle_swap(spawn_cfg, 1, rng)
                candidate = retry_goals[0]
            elif stype == "sphere_swap":
                _, retry_goals = _sphere_swap(spawn_cfg, 1, rng)
                candidate = retry_goals[0]
            else:
                _, retry_goals = _four_way(spawn_cfg, 1, rng)
                candidate = retry_goals[0]
            if np.linalg.norm(candidate - spawns[i]) >= min_goal_dist:
                goals[i] = candidate
                ok = True
                break
        if not ok:
            raise RuntimeError(
                f"Failed goal assignment: min_goal_distance_m={min_goal_dist} too strict for scenario"
            )

    return spawns, goals
