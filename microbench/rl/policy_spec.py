from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

import yaml

from microbench.learned import TinyLinearPolicyModel
from microbench.rl.adapters import CallablePolicyAdapter, ModelPredictPolicyAdapter
from microbench.rl.policies import RlPolicy, make_policy


RL_POLICY_SPEC_SCHEMA_VERSION = "0.1"
SUPPORTED_POLICY_SPEC_ADAPTERS = ("builtin", "callable", "model_predict", "tiny_linear_json")


@dataclass
class NamedRlPolicy:
    policy: RlPolicy
    policy_name: str
    spec: dict[str, Any]
    spec_path: str

    def reset(self, seed: int) -> None:
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset(int(seed))

    def action(self, agent: str, observation, action_space: Any, info: dict[str, Any]):
        return self.policy.action(agent, observation, action_space, info)


@dataclass
class LoadedPolicySpec:
    policy: NamedRlPolicy
    policy_name: str
    spec: dict[str, Any]
    spec_path: str
    summary: dict[str, Any]


def _read_policy_spec(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"RL policy spec must contain an object: {path}")
    return payload


def load_policy_spec(path: str | Path) -> dict[str, Any]:
    spec_path = Path(path)
    spec = _read_policy_spec(spec_path)
    if spec.get("schema_version") != RL_POLICY_SPEC_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported RL policy spec schema {spec.get('schema_version')!r}; "
            f"expected {RL_POLICY_SPEC_SCHEMA_VERSION!r}"
        )
    adapter = str(spec.get("adapter", "")).strip()
    if adapter not in SUPPORTED_POLICY_SPEC_ADAPTERS:
        raise ValueError(f"Unsupported RL policy spec adapter {adapter!r}; expected one of {','.join(SUPPORTED_POLICY_SPEC_ADAPTERS)}")
    if not str(spec.get("policy_name", "")).strip():
        raise ValueError("RL policy spec requires a nonempty policy_name")
    return spec


def _resolve_relative(path: str | Path, *, spec_path: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    spec_relative = spec_path.parent / candidate
    if spec_relative.exists():
        return spec_relative
    return candidate


def resolve_policy_artifact_path(path: str | Path, spec: dict[str, Any] | None = None) -> Path | None:
    spec_path = Path(path)
    payload = load_policy_spec(spec_path) if spec is None else spec
    artifact = payload.get("artifact_path")
    if not artifact:
        return None
    return _resolve_relative(str(artifact), spec_path=spec_path)


def portable_policy_spec_payload(path: str | Path, *, artifact_path: str | None = None) -> dict[str, Any]:
    spec_path = Path(path)
    payload = dict(load_policy_spec(spec_path))
    payload["source_spec_path"] = str(spec_path)
    if artifact_path is not None:
        payload["artifact_path"] = str(artifact_path)
    return payload


def _pythonpath_entries(spec: dict[str, Any], *, spec_path: Path) -> list[str]:
    entries: list[str] = []
    for raw in spec.get("pythonpath", []) or []:
        entries.append(str(_resolve_relative(str(raw), spec_path=spec_path)))
    return entries


def _resolve_factory_kwargs(value: Any, *, spec_path: Path) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(item, str) and str(key).endswith(("_path", "_dir")):
                out[str(key)] = str(_resolve_relative(item, spec_path=spec_path))
            else:
                out[str(key)] = _resolve_factory_kwargs(item, spec_path=spec_path)
        return out
    if isinstance(value, list):
        return [_resolve_factory_kwargs(item, spec_path=spec_path) for item in value]
    return value


def _import_object(path: str, *, pythonpath: list[str]) -> Any:
    for entry in reversed(pythonpath):
        if entry and entry not in sys.path:
            sys.path.insert(0, entry)
    if ":" in path:
        module_name, attr = path.split(":", 1)
    else:
        module_name, attr = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    obj: Any = module
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def _summary(spec: dict[str, Any], *, spec_path: Path) -> dict[str, Any]:
    return {
        "schema_version": spec.get("schema_version"),
        "policy_name": spec.get("policy_name"),
        "adapter": spec.get("adapter"),
        "spec_path": str(spec_path),
        "deterministic": bool(spec.get("deterministic", True)),
        "clip": bool(spec.get("clip", True)),
        "description": spec.get("description"),
        "artifact_path": spec.get("artifact_path"),
        "callable": spec.get("callable"),
        "factory": spec.get("factory"),
        "factory_kwargs": spec.get("factory_kwargs"),
    }


def load_policy_from_spec(path: str | Path, *, seed: int = 0) -> LoadedPolicySpec:
    spec_path = Path(path)
    spec = load_policy_spec(spec_path)
    adapter = str(spec["adapter"])
    policy_name = str(spec["policy_name"])
    clip = bool(spec.get("clip", True))
    deterministic = bool(spec.get("deterministic", True))
    pythonpath = _pythonpath_entries(spec, spec_path=spec_path)

    if adapter == "builtin":
        policy = make_policy(str(spec.get("policy", policy_name)), seed=int(seed))
    elif adapter == "tiny_linear_json":
        artifact = spec.get("artifact_path")
        model_path = _resolve_relative(str(artifact), spec_path=spec_path) if artifact else None
        policy = ModelPredictPolicyAdapter(
            TinyLinearPolicyModel.from_path(model_path),
            deterministic=deterministic,
            clip=clip,
        )
    elif adapter == "callable":
        target = str(spec.get("callable", ""))
        if not target:
            raise ValueError("callable policy spec requires callable")
        fn = _import_object(target, pythonpath=pythonpath)
        policy = CallablePolicyAdapter(
            fn,
            signature=str(spec.get("signature", "full")),  # type: ignore[arg-type]
            clip=clip,
        )
    elif adapter == "model_predict":
        target = str(spec.get("factory", ""))
        if not target:
            raise ValueError("model_predict policy spec requires factory")
        factory = _import_object(target, pythonpath=pythonpath)
        kwargs = _resolve_factory_kwargs(dict(spec.get("factory_kwargs", {}) or {}), spec_path=spec_path)
        model = factory(**kwargs) if callable(factory) else factory
        policy = ModelPredictPolicyAdapter(model, deterministic=deterministic, clip=clip)
    else:  # pragma: no cover - validated before this branch.
        raise ValueError(f"Unsupported RL policy spec adapter {adapter!r}")

    wrapped = NamedRlPolicy(
        policy=policy,
        policy_name=policy_name,
        spec=spec,
        spec_path=str(spec_path),
    )
    wrapped.reset(int(seed))
    return LoadedPolicySpec(
        policy=wrapped,
        policy_name=policy_name,
        spec=spec,
        spec_path=str(spec_path),
        summary=_summary(spec, spec_path=spec_path),
    )


def policy_factory_from_spec(path: str | Path) -> tuple[Callable[[int], RlPolicy], dict[str, Any]]:
    spec_path = Path(path)
    spec = load_policy_spec(spec_path)
    summary = _summary(spec, spec_path=spec_path)

    def factory(seed: int) -> RlPolicy:
        return load_policy_from_spec(spec_path, seed=int(seed)).policy

    return factory, summary
