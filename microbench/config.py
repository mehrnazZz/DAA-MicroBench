from __future__ import annotations

from importlib import resources
from pathlib import Path
import copy
import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG_ROOT = ROOT / "config"


def _resource_config_root():
    return resources.files("microbench").joinpath("bundled_config")


def _join(base, *parts: str):
    out = base
    for part in parts:
        out = out / part if isinstance(out, Path) else out.joinpath(part)
    return out


def default_config_root():
    """Return source-tree configs when available, otherwise packaged configs."""
    if SOURCE_CONFIG_ROOT.exists():
        return SOURCE_CONFIG_ROOT
    return _resource_config_root()


def resolve_config_path(path: str | Path):
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    if candidate.parts and candidate.parts[0] == "config":
        return _join(default_config_root(), *candidate.parts[1:])
    return candidate


def builtin_scenario_paths() -> list[str]:
    scenarios_dir = _join(default_config_root(), "scenarios")
    if isinstance(scenarios_dir, Path):
        paths = sorted(scenarios_dir.glob("*.yaml"))
    else:
        paths = sorted((p for p in scenarios_dir.iterdir() if p.name.endswith(".yaml")), key=lambda p: p.name)
    return [str(p) for p in paths]


def load_yaml(path: str | Path) -> dict:
    resolved = resolve_config_path(path) if isinstance(path, (str, Path)) else path
    with resolved.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_defaults(config_dir: str | Path | None = None) -> dict:
    base = Path(config_dir) if config_dir else default_config_root()
    return load_yaml(_join(base, "defaults.yaml"))


def load_comm_profiles(config_dir: str | Path | None = None) -> dict:
    base = Path(config_dir) if config_dir else default_config_root()
    return load_yaml(_join(base, "comm_profiles.yaml")).get("profiles", {})
