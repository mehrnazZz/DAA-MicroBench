from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from microbench.config import load_yaml
from microbench.metrics.io import RESULT_FIELDS, SUMMARY_FIELDS


AXES = (("x", 0), ("y", 1), ("z", 2))
BOUND_KEYS = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
PERCEPTION_MODES = {"v2v", "sensor", "fused"}
SPAWN_TYPES = {"rect_to_rect", "circle_swap", "sphere_swap", "four_way"}
ACCEPTANCE_OPERATORS = {"<=", "<", ">=", ">", "==", "!="}
ACCEPTANCE_SCOPES = {"summary", "results"}
ACCEPTANCE_SEVERITIES = {"smoke", "required", "warning", "informational"}


@dataclass
class ValidationReport:
    path: str
    kind: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def add_error(self, loc: str, message: str) -> None:
        self.errors.append(f"{loc}: {message}")

    def add_warning(self, loc: str, message: str) -> None:
        self.warnings.append(f"{loc}: {message}")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value))


def _num(report: ValidationReport, value: Any, loc: str) -> float | None:
    if not _is_number(value):
        report.add_error(loc, "must be a finite number")
        return None
    return float(value)


def _vec3(report: ValidationReport, value: Any, loc: str) -> np.ndarray | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        report.add_error(loc, "must be a 3-element list")
        return None
    out: list[float] = []
    for i, item in enumerate(value):
        v = _num(report, item, f"{loc}[{i}]")
        if v is None:
            return None
        out.append(v)
    return np.asarray(out, dtype=float)


def _positive(report: ValidationReport, cfg: dict, key: str, loc: str) -> float | None:
    if key not in cfg:
        return None
    v = _num(report, cfg[key], f"{loc}.{key}")
    if v is not None and v <= 0.0:
        report.add_error(f"{loc}.{key}", "must be > 0")
    return v


def _nonnegative(report: ValidationReport, cfg: dict, key: str, loc: str) -> float | None:
    if key not in cfg:
        return None
    v = _num(report, cfg[key], f"{loc}.{key}")
    if v is not None and v < 0.0:
        report.add_error(f"{loc}.{key}", "must be >= 0")
    return v


def _unit_probability(report: ValidationReport, cfg: dict, key: str, loc: str) -> None:
    if key not in cfg:
        return
    v = _num(report, cfg[key], f"{loc}.{key}")
    if v is not None and not (0.0 <= v <= 1.0):
        report.add_error(f"{loc}.{key}", "must be in [0, 1]")


def _bounds(report: ValidationReport, world: dict, *, required: bool) -> dict[str, float] | None:
    bounds = world.get("bounds")
    if bounds is None:
        if required:
            report.add_error("world.bounds", "is required for 3D validation")
        return None
    if not isinstance(bounds, dict):
        report.add_error("world.bounds", "must be a mapping")
        return None
    out: dict[str, float] = {}
    for key in BOUND_KEYS:
        if key not in bounds:
            report.add_error(f"world.bounds.{key}", "is required")
            continue
        v = _num(report, bounds[key], f"world.bounds.{key}")
        if v is not None:
            out[key] = v
    for axis in ("x", "y", "z"):
        lo_key = f"{axis}min"
        hi_key = f"{axis}max"
        if lo_key in out and hi_key in out and out[hi_key] <= out[lo_key]:
            report.add_error(f"world.bounds.{axis}", "max must be greater than min")
    return out if len(out) == len(BOUND_KEYS) else None


def _box_inside_bounds(
    report: ValidationReport,
    bounds: dict[str, float] | None,
    center: np.ndarray,
    half: np.ndarray,
    loc: str,
    *,
    warn_only: bool = False,
) -> None:
    if bounds is None:
        return
    lo = center - half
    hi = center + half
    for axis, idx in AXES:
        if lo[idx] < bounds[f"{axis}min"] - 1e-9 or hi[idx] > bounds[f"{axis}max"] + 1e-9:
            message = (
                f"{axis}-extent [{lo[idx]:.3g}, {hi[idx]:.3g}] lies outside "
                f"world bounds [{bounds[f'{axis}min']:.3g}, {bounds[f'{axis}max']:.3g}]"
            )
            if warn_only:
                report.add_warning(loc, message)
            else:
                report.add_error(loc, message)


def _combine_extents(extents: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray] | None:
    if not extents:
        return None
    lo = np.min(np.asarray([e[0] for e in extents], dtype=float), axis=0)
    hi = np.max(np.asarray([e[1] for e in extents], dtype=float), axis=0)
    return lo, hi


def _region_extent(report: ValidationReport, region: Any, loc: str) -> tuple[np.ndarray, np.ndarray] | None:
    if not isinstance(region, dict):
        report.add_error(loc, "must be a mapping")
        return None
    center = _vec3(report, region.get("center"), f"{loc}.center")
    half = _vec3(report, region.get("half"), f"{loc}.half")
    if center is None or half is None:
        return None
    if np.any(half < 0.0):
        report.add_error(f"{loc}.half", "entries must be >= 0")
    return center - half, center + half


def _layers_or_y(spawn: dict, field: str, fallback_y: float) -> list[float]:
    values = spawn.get(field)
    if isinstance(values, list) and values:
        return [float(x) for x in values if _is_number(x)]
    return [fallback_y]


def _spawn_extent(report: ValidationReport, spawn: dict) -> tuple[np.ndarray, np.ndarray] | None:
    stype = str(spawn.get("type", "rect_to_rect"))
    if stype not in SPAWN_TYPES:
        report.add_error("spawn.type", f"unsupported spawn type {stype!r}")
        return None

    if stype == "rect_to_rect":
        extents = []
        for key in ("start_region", "goal_region"):
            extent = _region_extent(report, spawn.get(key), f"spawn.{key}")
            if extent is not None:
                extents.append(extent)
        return _combine_extents(extents)

    if stype == "circle_swap":
        center = _vec3(report, spawn.get("center", [0.0, 0.0, 0.0]), "spawn.center")
        radius = _positive(report, spawn, "radius_m", "spawn")
        jitter = _nonnegative(report, spawn, "jitter_m", "spawn") or 0.0
        if center is None or radius is None:
            return None
        planar_radius = radius + 3.0 * jitter
        ys = _layers_or_y(spawn, "start_layers_m", float(center[1])) + _layers_or_y(
            spawn, "goal_layers_m", float(center[1])
        )
        lo = center - np.asarray([planar_radius, 0.0, planar_radius], dtype=float)
        hi = center + np.asarray([planar_radius, 0.0, planar_radius], dtype=float)
        lo[1] = min(ys)
        hi[1] = max(ys)
        return lo, hi

    if stype == "sphere_swap":
        center = _vec3(report, spawn.get("center", [0.0, 0.0, 0.0]), "spawn.center")
        radius = _positive(report, spawn, "radius_m", "spawn")
        jitter = _nonnegative(report, spawn, "jitter_m", "spawn") or 0.0
        vertical_scale = _positive(report, spawn, "vertical_scale", "spawn")
        min_abs = _nonnegative(report, spawn, "min_abs_y_component", "spawn")
        if min_abs is not None and min_abs >= 1.0:
            report.add_error("spawn.min_abs_y_component", "must be < 1")
        if center is None or radius is None:
            return None
        if "vertical_scale" in spawn and vertical_scale is None:
            return None
        extent = radius + 3.0 * jitter
        half = np.asarray([extent, extent, extent], dtype=float)
        return center - half, center + half

    extent = _positive(report, spawn, "extent_m", "spawn")
    lane = _positive(report, spawn, "lane_half_width_m", "spawn")
    y_m = _num(report, spawn.get("y_m", 0.0), "spawn.y_m")
    if extent is None or lane is None or y_m is None:
        return None
    ys = _layers_or_y(spawn, "start_layers_m", y_m) + _layers_or_y(spawn, "goal_layers_m", y_m)
    lo = np.asarray([-extent, min(ys), -extent], dtype=float)
    hi = np.asarray([extent, max(ys), extent], dtype=float)
    return lo, hi


def _validate_obstacles(report: ValidationReport, obstacles: Any, bounds: dict[str, float] | None) -> None:
    if obstacles is None:
        return
    if not isinstance(obstacles, list):
        report.add_error("obstacles", "must be a list")
        return
    for idx, obstacle in enumerate(obstacles):
        loc = f"obstacles[{idx}]"
        if not isinstance(obstacle, dict) or "aabb" not in obstacle:
            report.add_error(loc, "must contain an aabb mapping")
            continue
        aabb = obstacle["aabb"]
        if not isinstance(aabb, dict):
            report.add_error(f"{loc}.aabb", "must be a mapping")
            continue
        center = _vec3(report, aabb.get("center"), f"{loc}.aabb.center")
        half = _vec3(report, aabb.get("half"), f"{loc}.aabb.half")
        if center is None or half is None:
            continue
        if np.any(half <= 0.0):
            report.add_error(f"{loc}.aabb.half", "entries must be > 0")
        _box_inside_bounds(report, bounds, center, half, loc, warn_only=True)


def _validate_perception(report: ValidationReport, cfg: dict) -> None:
    perception = cfg.get("perception", {})
    if not isinstance(perception, dict):
        report.add_error("perception", "must be a mapping")
        return
    mode = str(perception.get("mode", "v2v")).lower()
    if mode not in PERCEPTION_MODES:
        report.add_error("perception.mode", f"must be one of {sorted(PERCEPTION_MODES)}")
    sensor = perception.get("sensor", {})
    if not isinstance(sensor, dict):
        report.add_error("perception.sensor", "must be a mapping")
        return
    _positive(report, sensor, "range_m", "perception.sensor")
    fov = _positive(report, sensor, "fov_deg", "perception.sensor")
    if fov is not None and fov > 360.0:
        report.add_error("perception.sensor.fov_deg", "must be <= 360")
    _unit_probability(report, sensor, "false_negative_p", "perception.sensor")
    _nonnegative(report, sensor, "noise_sigma_pos_m", "perception.sensor")
    _nonnegative(report, sensor, "noise_sigma_vel_mps", "perception.sensor")
    _nonnegative(report, sensor, "track_ttl_s", "perception.sensor")
    _nonnegative(report, sensor, "occlusion_margin_m", "perception.sensor")


def validate_scenario_config(cfg: dict, *, path: str = "<scenario>") -> ValidationReport:
    report = ValidationReport(path=str(path), kind="scenario")
    if not isinstance(cfg, dict):
        report.add_error("root", "must be a mapping")
        return report

    for key in ("scenario", "world", "agent_params", "goals", "spawn"):
        if key not in cfg:
            report.add_error(key, "is required")

    scenario = cfg.get("scenario", {})
    if isinstance(scenario, dict):
        name = scenario.get("name")
        if name is not None and not str(name).strip():
            report.add_error("scenario.name", "must not be empty")
        _positive(report, scenario, "duration_s", "scenario")
    else:
        report.add_error("scenario", "must be a mapping")

    world = cfg.get("world", {})
    if not isinstance(world, dict):
        report.add_error("world", "must be a mapping")
        world = {}
    planar = bool(world.get("planar", True))
    benchmark = cfg.get("benchmark", {}) if isinstance(cfg.get("benchmark", {}), dict) else {}
    dimension = str(benchmark.get("dimension", "2d" if planar else "3d")).lower()
    if dimension not in {"2d", "3d"}:
        report.add_error("benchmark.dimension", "must be '2d' or '3d'")
    if dimension == "2d" and not planar:
        report.add_error("benchmark.dimension", "2d scenarios must set world.planar: true")
    if dimension == "3d" and planar:
        report.add_error("benchmark.dimension", "3d scenarios must set world.planar: false")

    bounds = _bounds(report, world, required=(dimension == "3d" or not planar))

    agent_params = cfg.get("agent_params", {})
    if isinstance(agent_params, dict):
        _positive(report, agent_params, "radius_m", "agent_params")
        _positive(report, agent_params, "v_max_mps", "agent_params")
        _positive(report, agent_params, "a_max_mps2", "agent_params")
        _positive(report, agent_params, "goal_tolerance_m", "agent_params")
    else:
        report.add_error("agent_params", "must be a mapping")

    goals = cfg.get("goals", {})
    if isinstance(goals, dict):
        _nonnegative(report, goals, "min_goal_distance_m", "goals")
        _positive(report, goals, "max_attempts", "goals")
    else:
        report.add_error("goals", "must be a mapping")

    spawn = cfg.get("spawn", {})
    if not isinstance(spawn, dict):
        report.add_error("spawn", "must be a mapping")
        spawn_extent = None
    else:
        spawn_extent = _spawn_extent(report, spawn)
        if spawn_extent is not None and bounds is not None:
            center = (spawn_extent[0] + spawn_extent[1]) * 0.5
            half = (spawn_extent[1] - spawn_extent[0]) * 0.5
            _box_inside_bounds(report, bounds, center, half, "spawn")

    _validate_obstacles(report, cfg.get("obstacles"), bounds)
    _validate_perception(report, cfg)

    if dimension == "3d" or not planar:
        if bounds is not None and bounds["ymax"] - bounds["ymin"] <= 1e-6:
            report.add_error("world.bounds.y", "3D scenarios must have nonzero vertical span")
        if spawn_extent is not None and spawn_extent[1][1] - spawn_extent[0][1] <= 1e-6:
            report.add_error("spawn", "3D scenarios must have vertical spawn or goal variation")

    return report


def validate_scenario_file(path: str | Path) -> ValidationReport:
    p = Path(path)
    try:
        cfg = load_yaml(p)
    except Exception as exc:
        report = ValidationReport(path=str(p), kind="scenario")
        report.add_error("yaml", f"failed to load: {exc}")
        return report
    return validate_scenario_config(cfg, path=str(p))


def _nonempty_list(report: ValidationReport, value: Any, loc: str) -> list:
    if not isinstance(value, list) or not value:
        report.add_error(loc, "must be a non-empty list")
        return []
    return value


def _validate_acceptance(report: ValidationReport, acceptance: Any, loc: str) -> None:
    if acceptance is None:
        return
    if not isinstance(acceptance, dict):
        report.add_error(loc, "must be a mapping")
        return

    schema_version = acceptance.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version.strip():
        report.add_error(f"{loc}.schema_version", "must be a non-empty string")

    rules = acceptance.get("rules")
    if not isinstance(rules, list):
        report.add_error(f"{loc}.rules", "must be a list")
        return

    summary_fields = set(SUMMARY_FIELDS)
    result_fields = set(RESULT_FIELDS)
    for idx, rule in enumerate(rules):
        rloc = f"{loc}.rules[{idx}]"
        if not isinstance(rule, dict):
            report.add_error(rloc, "must be a mapping")
            continue

        for key in ("name", "scope", "method", "metric", "operator", "value", "severity", "description"):
            if key not in rule:
                report.add_error(f"{rloc}.{key}", "is required")

        for key in ("name", "method", "metric", "operator", "severity", "description"):
            if key in rule and (not isinstance(rule[key], str) or not rule[key].strip()):
                report.add_error(f"{rloc}.{key}", "must be a non-empty string")

        scope = str(rule.get("scope", "")).lower()
        if scope not in ACCEPTANCE_SCOPES:
            report.add_error(f"{rloc}.scope", f"must be one of {sorted(ACCEPTANCE_SCOPES)}")

        operator = str(rule.get("operator", ""))
        if operator not in ACCEPTANCE_OPERATORS:
            report.add_error(f"{rloc}.operator", f"must be one of {sorted(ACCEPTANCE_OPERATORS)}")

        severity = str(rule.get("severity", "")).lower()
        if severity not in ACCEPTANCE_SEVERITIES:
            report.add_error(f"{rloc}.severity", f"must be one of {sorted(ACCEPTANCE_SEVERITIES)}")

        metric = str(rule.get("metric", ""))
        if scope == "summary" and metric not in summary_fields:
            report.add_error(f"{rloc}.metric", f"unknown summary metric {metric!r}")
        if scope == "results" and metric not in result_fields:
            report.add_error(f"{rloc}.metric", f"unknown results metric {metric!r}")

        _num(report, rule.get("value"), f"{rloc}.value")

        n_agents = rule.get("n_agents", "*")
        if not (n_agents == "*" or (isinstance(n_agents, int) and not isinstance(n_agents, bool) and n_agents > 0)):
            report.add_error(f"{rloc}.n_agents", "must be '*' or a positive integer")


def _resolve_manifest_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    base_candidate = base_dir / path
    if base_candidate.exists():
        return base_candidate
    return path


def validate_suite_manifest_config(
    manifest: dict,
    *,
    path: str = "<suite_manifest>",
    base_dir: str | Path | None = None,
    validate_scenarios: bool = True,
) -> ValidationReport:
    report = ValidationReport(path=str(path), kind="suite_manifest")
    if not isinstance(manifest, dict):
        report.add_error("root", "must be a mapping")
        return report

    for key in ("suite", "description", "default_methods", "n_agents", "seeds", "comm_profiles", "scenarios"):
        if key not in manifest:
            report.add_error(key, "is required")

    _nonempty_list(report, manifest.get("default_methods"), "default_methods")
    n_agents = _nonempty_list(report, manifest.get("n_agents"), "n_agents")
    seeds = _nonempty_list(report, manifest.get("seeds"), "seeds")
    comm_profiles = _nonempty_list(report, manifest.get("comm_profiles"), "comm_profiles")
    for i, n in enumerate(n_agents):
        if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
            report.add_error(f"n_agents[{i}]", "must be a positive integer")
    for i, seed in enumerate(seeds):
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            report.add_error(f"seeds[{i}]", "must be a non-negative integer")
    for i, profile in enumerate(comm_profiles):
        if not str(profile).strip():
            report.add_error(f"comm_profiles[{i}]", "must not be empty")
    _positive(report, manifest, "duration_override_s", "root")

    _validate_acceptance(report, manifest.get("acceptance"), "acceptance")

    scenarios = _nonempty_list(report, manifest.get("scenarios"), "scenarios")
    base = Path(base_dir) if base_dir is not None else Path(path).parent
    for idx, item in enumerate(scenarios):
        loc = f"scenarios[{idx}]"
        if not isinstance(item, dict):
            report.add_error(loc, "must be a mapping")
            continue
        for key in ("id", "path", "family", "dimension", "difficulty", "purpose", "expected_failure_modes", "recommended"):
            if key not in item:
                report.add_error(f"{loc}.{key}", "is required")
        dimension = str(item.get("dimension", "")).lower()
        if dimension not in {"2d", "3d"}:
            report.add_error(f"{loc}.dimension", "must be '2d' or '3d'")
        if not isinstance(item.get("expected_failure_modes"), list) or not item.get("expected_failure_modes"):
            report.add_error(f"{loc}.expected_failure_modes", "must be a non-empty list")
        recommended = item.get("recommended")
        if not isinstance(recommended, dict):
            report.add_error(f"{loc}.recommended", "must be a mapping")
        else:
            _nonempty_list(report, recommended.get("n_agents"), f"{loc}.recommended.n_agents")
            _nonempty_list(report, recommended.get("seeds"), f"{loc}.recommended.seeds")
            _nonempty_list(report, recommended.get("comm_profiles"), f"{loc}.recommended.comm_profiles")

        raw_path = item.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue
        scenario_path = _resolve_manifest_path(base, raw_path)
        if not scenario_path.exists():
            report.add_error(f"{loc}.path", f"does not exist: {raw_path}")
            continue
        if validate_scenarios:
            scenario_report = validate_scenario_file(scenario_path)
            report.errors.extend(f"{loc}.{err}" for err in scenario_report.errors)
            cfg = load_yaml(scenario_path)
            benchmark = cfg.get("benchmark", {}) if isinstance(cfg.get("benchmark", {}), dict) else {}
            for key in ("family", "dimension", "difficulty"):
                if key in item and key in benchmark and str(item[key]) != str(benchmark[key]):
                    report.add_error(
                        f"{loc}.{key}",
                        f"manifest value {item[key]!r} does not match scenario benchmark value {benchmark[key]!r}",
                    )
    return report


def validate_suite_manifest_file(path: str | Path, *, validate_scenarios: bool = True) -> ValidationReport:
    p = Path(path)
    try:
        manifest = load_yaml(p)
    except Exception as exc:
        report = ValidationReport(path=str(p), kind="suite_manifest")
        report.add_error("yaml", f"failed to load: {exc}")
        return report
    return validate_suite_manifest_config(
        manifest,
        path=str(p),
        base_dir=p.parent,
        validate_scenarios=validate_scenarios,
    )
