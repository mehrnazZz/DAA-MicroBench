from __future__ import annotations

from pathlib import Path
import copy
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
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
    base = Path(config_dir) if config_dir else ROOT / "config"
    return load_yaml(base / "defaults.yaml")


def load_comm_profiles(config_dir: str | Path | None = None) -> dict:
    base = Path(config_dir) if config_dir else ROOT / "config"
    return load_yaml(base / "comm_profiles.yaml").get("profiles", {})
