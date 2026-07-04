from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

from microbench.config import deep_merge, load_defaults
from microbench.planners.base import ILocalPlanner
from microbench.planners.baseline_goal import BaselineGoalPlanner
from microbench.planners.cbf_qp import CbfQpPlanner
from microbench.planners.intent_dummy import IntentDummyPlanner
from microbench.planners.negotiation_yield import NegotiationYieldPlanner
from microbench.planners.orca_expert import OrcaExpertPlanner
from microbench.planners.priority_yield import PriorityYieldPlanner
from microbench.planners.template_planner import TemplatePlanner


@dataclass(frozen=True)
class PlannerMetadata:
    method: str
    display_name: str
    planner_type: str
    role: str
    status: str
    dimensions: tuple[str, ...]
    observation_sources: tuple[str, ...]
    aliases: tuple[str, ...] = ()
    uses_v2v: bool = False
    uses_local_sensing: bool = False
    uses_intent: bool = False
    uses_agent_messages: bool = False
    uses_obstacles: bool = False
    learned: bool = False
    deterministic: bool = True
    description: str = ""
    limitations: tuple[str, ...] = ()
    canonical_method: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _orca_cfg(defaults: dict, key: str) -> dict:
    base = defaults.get("orca_heuristic", defaults.get("orca_expert", {}))
    if key == "orca_heuristic":
        return dict(base)
    return deep_merge(base, defaults.get(key, {}))


def _make_orca_heuristic() -> ILocalPlanner:
    defaults = load_defaults()
    age_cap = float(defaults.get("comm", {}).get("age_cap_s", 0.75))
    return OrcaExpertPlanner(cfg=_orca_cfg(defaults, "orca_heuristic"), age_cap_s=age_cap)


def _make_orca_with_staleness() -> ILocalPlanner:
    defaults = load_defaults()
    age_cap = float(defaults.get("comm", {}).get("age_cap_s", 0.75))
    return OrcaExpertPlanner(cfg=_orca_cfg(defaults, "orca_with_staleness"), age_cap_s=age_cap)


def _make_cbf_qp() -> ILocalPlanner:
    defaults = load_defaults()
    return CbfQpPlanner(cfg=defaults.get("cbf_qp", {}))


_FACTORIES: dict[str, Callable[[], ILocalPlanner]] = {
    "baseline_goal": BaselineGoalPlanner,
    "orca_heuristic": _make_orca_heuristic,
    "orca_with_staleness": _make_orca_with_staleness,
    "cbf_qp": _make_cbf_qp,
    "template": TemplatePlanner,
    "intent_dummy": IntentDummyPlanner,
    "priority_yield": PriorityYieldPlanner,
    "negotiation_yield": NegotiationYieldPlanner,
}


_METADATA: dict[str, PlannerMetadata] = {
    "baseline_goal": PlannerMetadata(
        method="baseline_goal",
        display_name="Goal-only baseline",
        planner_type="reactive",
        role="illustrative_baseline",
        status="stable",
        dimensions=("2d", "3d"),
        observation_sources=("ego_state", "goal"),
        description="Accelerates each drone toward its goal and ignores all traffic.",
        limitations=("Does not use neighbor, obstacle, intent, or message information.",),
    ),
    "orca_heuristic": PlannerMetadata(
        method="orca_heuristic",
        display_name="ORCA-like geometric heuristic",
        planner_type="geometric_heuristic",
        role="reference_baseline",
        status="pre_v1",
        dimensions=("2d", "3d"),
        observation_sources=("local_neighbors", "v2v", "sensor", "fused"),
        aliases=("orca_expert",),
        uses_v2v=True,
        uses_local_sensing=True,
        uses_obstacles=True,
        description=(
            "Obstacle-aware ORCA-like candidate-velocity planner with stale-track inflation "
            "and 2D/3D support."
        ),
        limitations=(
            "Not a formally validated ORCA implementation.",
            "Not an expert oracle and should not be treated as ground-truth DAA behavior.",
            "Uses the benchmark's provided local neighbor tracks rather than raw sensor processing.",
        ),
    ),
    "orca_with_staleness": PlannerMetadata(
        method="orca_with_staleness",
        display_name="ORCA-like stale-aware heuristic",
        planner_type="geometric_heuristic",
        role="reference_baseline",
        status="pre_v1",
        dimensions=("2d", "3d"),
        observation_sources=("local_neighbors", "v2v", "sensor", "fused"),
        uses_v2v=True,
        uses_local_sensing=True,
        uses_obstacles=True,
        description=(
            "ORCA-like candidate-velocity planner preset with stronger stale-track inflation "
            "and age-based responsibility for degraded communication or sensor-track runs."
        ),
        limitations=(
            "Not a formally validated ORCA implementation.",
            "May trade mission progress for conservative behavior under stale observations.",
            "Uses the benchmark's provided local neighbor tracks rather than raw sensor processing.",
        ),
    ),
    "cbf_qp": PlannerMetadata(
        method="cbf_qp",
        display_name="CBF-QP projection skeleton",
        planner_type="optimization_skeleton",
        role="experimental_baseline",
        status="experimental",
        dimensions=("2d", "3d"),
        observation_sources=("local_neighbors", "v2v", "sensor", "fused"),
        uses_v2v=True,
        uses_local_sensing=True,
        uses_obstacles=True,
        description=(
            "Solver-free control-barrier-function baseline skeleton using deterministic "
            "halfspace projection with a bounded fallback."
        ),
        limitations=(
            "Not yet a solver-backed quadratic program.",
            "Experimental; not recommended as a leaderboard anchor until calibrated.",
            "Uses local tracks supplied by PlannerInput rather than raw sensor processing.",
        ),
    ),
    "template": PlannerMetadata(
        method="template",
        display_name="Template planner",
        planner_type="example",
        role="developer_template",
        status="stable",
        dimensions=("2d", "3d"),
        observation_sources=("ego_state", "goal"),
        aliases=("template_planner",),
        description="Minimal plugin example for implementing a custom local planner.",
        limitations=("Intended for API examples, not for benchmark scoring.",),
    ),
    "intent_dummy": PlannerMetadata(
        method="intent_dummy",
        display_name="Intent-sharing dummy",
        planner_type="agentic_example",
        role="illustrative_baseline",
        status="experimental",
        dimensions=("2d", "3d"),
        observation_sources=("ego_state", "goal", "agent_messages"),
        uses_intent=True,
        uses_agent_messages=True,
        description="Simple planner that exercises intent message emission and receipt.",
        limitations=("Useful for plumbing tests; not designed as a competitive DAA baseline.",),
    ),
    "priority_yield": PlannerMetadata(
        method="priority_yield",
        display_name="Priority-yield baseline",
        planner_type="agentic_heuristic",
        role="agentic_reference_baseline",
        status="pre_v1",
        dimensions=("2d", "3d"),
        observation_sources=("local_neighbors", "v2v", "sensor", "fused", "agent_messages"),
        uses_v2v=True,
        uses_local_sensing=True,
        uses_agent_messages=True,
        description="Decentralized yielding heuristic that uses per-agent priority and agent messages.",
        limitations=("Negotiation is one-shot and heuristic rather than formally optimized.",),
    ),
    "negotiation_yield": PlannerMetadata(
        method="negotiation_yield",
        display_name="Negotiation-yield baseline",
        planner_type="agentic_heuristic",
        role="agentic_reference_baseline",
        status="experimental",
        dimensions=("2d", "3d"),
        observation_sources=("local_neighbors", "v2v", "sensor", "fused", "agent_messages"),
        uses_v2v=True,
        uses_local_sensing=True,
        uses_agent_messages=True,
        description="Agentic yielding baseline that exchanges proposals and acknowledgments.",
        limitations=("Early reference behavior; not yet a leaderboard anchor.",),
    ),
}


_ALIASES = {
    alias: method
    for method, metadata in _METADATA.items()
    for alias in metadata.aliases
}


def canonical_method(name: str) -> str:
    key = name.strip().lower()
    return _ALIASES.get(key, key)


def list_methods(*, include_aliases: bool = False) -> list[str]:
    methods = list(_METADATA)
    if include_aliases:
        methods.extend(sorted(_ALIASES))
    return methods


def planner_metadata(*, include_aliases: bool = False) -> list[dict]:
    entries = [metadata.to_dict() for metadata in _METADATA.values()]
    if include_aliases:
        for alias in sorted(_ALIASES):
            canonical = _ALIASES[alias]
            metadata = _METADATA[canonical]
            entry = metadata.to_dict()
            entry["method"] = alias
            entry["display_name"] = f"{metadata.display_name} compatibility alias"
            entry["status"] = "alias"
            entry["aliases"] = ()
            entry["canonical_method"] = canonical
            entries.append(entry)
    return entries


def make_planner(name: str) -> ILocalPlanner:
    canonical = canonical_method(name)
    try:
        return _FACTORIES[canonical]()
    except KeyError as exc:
        raise ValueError(f"Unknown planner method: {name}") from exc
