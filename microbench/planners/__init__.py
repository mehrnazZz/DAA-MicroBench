from __future__ import annotations

from microbench.config import load_defaults
from microbench.planners.base import ILocalPlanner
from microbench.planners.baseline_goal import BaselineGoalPlanner
from microbench.planners.intent_dummy import IntentDummyPlanner
from microbench.planners.negotiation_yield import NegotiationYieldPlanner
from microbench.planners.orca_expert import OrcaExpertPlanner
from microbench.planners.priority_yield import PriorityYieldPlanner
from microbench.planners.template_planner import TemplatePlanner


def list_methods() -> list[str]:
    return [
        "baseline_goal",
        "orca_expert",
        "template",
        "template_planner",
        "intent_dummy",
        "priority_yield",
        "negotiation_yield",
    ]


def make_planner(name: str) -> ILocalPlanner:
    key = name.strip().lower()
    if key == "baseline_goal":
        return BaselineGoalPlanner()
    if key == "orca_expert":
        defaults = load_defaults()
        orca_cfg = defaults.get("orca_expert", {})
        age_cap = float(defaults.get("comm", {}).get("age_cap_s", 0.75))
        return OrcaExpertPlanner(cfg=orca_cfg, age_cap_s=age_cap)
    if key == "intent_dummy":
        return IntentDummyPlanner()
    if key == "priority_yield":
        return PriorityYieldPlanner()
    if key == "negotiation_yield":
        return NegotiationYieldPlanner()
    if key == "template":
        return TemplatePlanner()
    if key == "template_planner":
        return TemplatePlanner()
    raise ValueError(f"Unknown planner method: {name}")
