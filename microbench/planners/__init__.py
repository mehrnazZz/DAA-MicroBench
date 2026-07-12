from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from dataclasses import asdict, dataclass

from microbench.config import deep_merge, load_defaults
from microbench.planners.base import ILocalPlanner
from microbench.planners.baseline_goal import BaselineGoalPlanner
from microbench.planners.bvc_tube_dmpc import BvcTubeDmpcPlanner
from microbench.planners.cbf_qp import CbfQpPlanner
from microbench.planners.dmpc_best_response import DistributedMpcBestResponsePlanner
from microbench.planners.ego_swarm import EgoSwarmPlanner
from microbench.planners.ego_swarm_opt import EgoSwarmOptimizingPlanner
from microbench.planners.intent_dummy import IntentDummyPlanner
from microbench.planners.learned_tiny import LearnedTinyPlanner
from microbench.planners.mpc_local import MpcLocalPlanner
from microbench.planners.mpc_nonlinear import NonlinearMpcPlanner
from microbench.planners.negotiation_yield import NegotiationYieldPlanner
from microbench.planners.orca_expert import OrcaExpertPlanner
from microbench.planners.priority_yield import PriorityYieldPlanner
from microbench.planners.rmader import RmaderPlanner
from microbench.planners.template_planner import TemplatePlanner
from microbench.planners.velocity_obstacle import ReciprocalVelocityObstaclePlanner, VelocityObstaclePlanner


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


def _make_mpc_local() -> ILocalPlanner:
    defaults = load_defaults()
    return MpcLocalPlanner(cfg=defaults.get("mpc_local", {}))


def _make_mpc_nonlinear() -> ILocalPlanner:
    defaults = load_defaults()
    return NonlinearMpcPlanner(cfg=defaults.get("mpc_nonlinear", {}))


def _make_dmpc_best_response() -> ILocalPlanner:
    defaults = load_defaults()
    return DistributedMpcBestResponsePlanner(cfg=defaults.get("dmpc_best_response", {}))


def _make_bvc_tube_dmpc() -> ILocalPlanner:
    defaults = load_defaults()
    return BvcTubeDmpcPlanner(cfg=defaults.get("bvc_tube_dmpc", {}))


def _make_rmader() -> ILocalPlanner:
    defaults = load_defaults()
    return RmaderPlanner(cfg=defaults.get("rmader", {}))


def _make_ego_swarm() -> ILocalPlanner:
    defaults = load_defaults()
    return EgoSwarmPlanner(cfg=defaults.get("ego_swarm", {}))


def _make_ego_swarm_opt() -> ILocalPlanner:
    defaults = load_defaults()
    return EgoSwarmOptimizingPlanner(cfg=defaults.get("ego_swarm_opt", {}))


def _make_velocity_obstacle() -> ILocalPlanner:
    defaults = load_defaults()
    return VelocityObstaclePlanner(cfg=defaults.get("velocity_obstacle", {}))


def _make_reciprocal_velocity_obstacle() -> ILocalPlanner:
    defaults = load_defaults()
    return ReciprocalVelocityObstaclePlanner(cfg=defaults.get("reciprocal_velocity_obstacle", {}))


_FACTORIES: dict[str, Callable[[], ILocalPlanner]] = {
    "baseline_goal": BaselineGoalPlanner,
    "orca_heuristic": _make_orca_heuristic,
    "orca_with_staleness": _make_orca_with_staleness,
    "cbf_qp": _make_cbf_qp,
    "mpc_local": _make_mpc_local,
    "mpc_nonlinear": _make_mpc_nonlinear,
    "dmpc_best_response": _make_dmpc_best_response,
    "bvc_tube_dmpc": _make_bvc_tube_dmpc,
    "rmader": _make_rmader,
    "ego_swarm": _make_ego_swarm,
    "ego_swarm_opt": _make_ego_swarm_opt,
    "velocity_obstacle": _make_velocity_obstacle,
    "reciprocal_velocity_obstacle": _make_reciprocal_velocity_obstacle,
    "template": TemplatePlanner,
    "intent_dummy": IntentDummyPlanner,
    "learned_tiny": LearnedTinyPlanner,
    "priority_yield": PriorityYieldPlanner,
    "negotiation_yield": NegotiationYieldPlanner,
}


def _make_learned_policy_spec(policy_spec: str | Path | None) -> ILocalPlanner:
    if policy_spec is None:
        raise ValueError("learned_policy_spec requires --policy-spec")
    from microbench.planners.learned_policy_spec import LearnedPolicySpecPlanner

    return LearnedPolicySpecPlanner(policy_spec=policy_spec)


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
        display_name="CBF-QP safety-filter baseline",
        planner_type="cbf_qp_safety_filter",
        role="experimental_baseline",
        status="experimental",
        dimensions=("2d", "3d"),
        observation_sources=("local_neighbors", "v2v", "sensor", "fused"),
        uses_v2v=True,
        uses_local_sensing=True,
        uses_obstacles=True,
        description=(
            "Control-barrier-function safety-filter baseline with deterministic projection, "
            "optional SciPy SLSQP solve, obstacle barriers, and stale-track inflation."
        ),
        limitations=(
            "Experimental; not recommended as a leaderboard anchor until calibrated.",
            "Projection fallback is not a substitute for a fully validated CBF-QP controller.",
            "Uses local tracks supplied by PlannerInput rather than raw sensor processing.",
        ),
    ),
    "mpc_local": PlannerMetadata(
        method="mpc_local",
        display_name="Local MPC sampling baseline",
        planner_type="predictive_sampling",
        role="experimental_baseline",
        status="experimental",
        dimensions=("2d", "3d"),
        observation_sources=("local_neighbors", "v2v", "sensor", "fused"),
        uses_v2v=True,
        uses_local_sensing=True,
        uses_obstacles=True,
        description=(
            "Deterministic local predictive baseline that samples one-step-reachable velocity "
            "commands and scores short-horizon rollouts against neighbors, stale tracks, and obstacles."
        ),
        limitations=(
            "Not a full nonlinear MPC solver.",
            "Experimental; not recommended as a leaderboard anchor until calibrated.",
            "Uses constant-velocity neighbor predictions from PlannerInput.",
        ),
    ),
    "mpc_nonlinear": PlannerMetadata(
        method="mpc_nonlinear",
        display_name="Nonlinear MPC trajectory-optimization baseline",
        planner_type="nonlinear_mpc_trajectory_optimization",
        role="experimental_baseline",
        status="experimental",
        dimensions=("2d", "3d"),
        observation_sources=("local_neighbors", "v2v", "sensor", "fused", "intent"),
        uses_v2v=True,
        uses_local_sensing=True,
        uses_intent=True,
        uses_obstacles=True,
        description=(
            "Clean-room nonlinear MPC baseline that optimizes finite-horizon acceleration "
            "controls with double-integrator dynamics, warm starts, multistart avoidance seeds, "
            "neighbor/intent/obstacle penalties, and trajectory intent output."
        ),
        limitations=(
            "Simplified translational NMPC for the benchmark contract, not a full quadrotor attitude/rotor model.",
            "Uses supplied local tracks, intent trajectories, and AABB obstacles rather than raw onboard sensing or mapping.",
            "Experimental; needs broader dense-3D and solver-mode calibration before becoming a reference baseline.",
        ),
    ),
    "dmpc_best_response": PlannerMetadata(
        method="dmpc_best_response",
        display_name="Distributed MPC best-response baseline",
        planner_type="distributed_mpc_best_response",
        role="experimental_baseline",
        status="experimental",
        dimensions=("2d", "3d"),
        observation_sources=("local_neighbors", "v2v", "sensor", "fused", "intent", "agent_messages"),
        uses_v2v=True,
        uses_local_sensing=True,
        uses_intent=True,
        uses_agent_messages=True,
        uses_obstacles=True,
        description=(
            "Clean-room distributed-MPC-style best-response baseline. Each agent optimizes its own "
            "finite-horizon trajectory, treats received neighbor intents as coupled trajectory "
            "constraints, falls back to inflated constant-velocity tracks for stale or missing plans, "
            "and republishes its optimized plan for the next coordination round."
        ),
        limitations=(
            "Implements asynchronous one-best-response-round-per-simulator-tick coordination rather than a centralized joint solve.",
            "Uses DAA Microbench local tracks, intent trajectories, and AABB obstacles rather than raw onboard mapping.",
            "Experimental; needs dense-3D communication-limited calibration before becoming a reference baseline.",
        ),
    ),
    "bvc_tube_dmpc": PlannerMetadata(
        method="bvc_tube_dmpc",
        display_name="BVC tube-DMPC baseline",
        planner_type="tube_dmpc_buffered_voronoi_cells",
        role="experimental_baseline",
        status="experimental",
        dimensions=("2d", "3d"),
        observation_sources=("local_neighbors", "v2v", "sensor", "fused", "intent", "agent_messages"),
        uses_v2v=True,
        uses_local_sensing=True,
        uses_intent=True,
        uses_agent_messages=True,
        uses_obstacles=True,
        description=(
            "Clean-room tube-based distributed MPC baseline that builds time-indexed buffered "
            "Voronoi-cell halfspace tubes, projects planned waypoints into hard non-overlapping "
            "cells, and publishes the selected tube trajectory as intent."
        ),
        limitations=(
            "Not an official port of a BVC/B-UAVC or Schoellig-lab DMPC implementation.",
            "Uses DAA Microbench local tracks, intent trajectories, and AABB obstacles rather than raw onboard perception.",
            "Experimental; needs dense-3D stress and heterogeneous-policy calibration before becoming a reference baseline.",
        ),
    ),
    "rmader": PlannerMetadata(
        method="rmader",
        display_name="RMADER MINVO robust trajectory baseline",
        planner_type="rmader_minvo_hyperplane_trajectory_optimization",
        role="experimental_baseline",
        status="experimental",
        dimensions=("2d", "3d"),
        observation_sources=("local_neighbors", "v2v", "sensor", "fused", "intent", "agent_messages"),
        uses_v2v=True,
        uses_local_sensing=True,
        uses_intent=True,
        uses_agent_messages=True,
        uses_obstacles=True,
        description=(
            "Clean-room RMADER/MADER-style baseline with cubic B-spline plans, continuous MINVO "
            "interval polyhedra, hard separating hyperplanes against dynamic and static hulls, "
            "kinematic feasibility checks, and robust publish/check/commit trajectory sharing."
        ),
        limitations=(
            "Not a ROS/Gurobi port of the MIT ACL implementation; it is adapted to the DAA Microbench velocity-command contract.",
            "Uses supplied local tracks, intent trajectories, and AABB obstacles rather than onboard mapping or raw perception.",
            "Experimental; needs dense-3D delay/loss stress calibration before becoming a reference baseline.",
        ),
    ),
    "velocity_obstacle": PlannerMetadata(
        method="velocity_obstacle",
        display_name="Velocity-obstacle cone baseline",
        planner_type="velocity_obstacle_sampling",
        role="experimental_baseline",
        status="experimental",
        dimensions=("2d", "3d"),
        observation_sources=("local_neighbors", "v2v", "sensor", "fused"),
        uses_v2v=True,
        uses_local_sensing=True,
        uses_obstacles=True,
        description=(
            "Deterministic 2D/3D velocity-obstacle cone sampler that scores finite-horizon "
            "candidate commands against local tracks and static obstacles."
        ),
        limitations=(
            "Experimental; not a formally validated VO/RVO/HRVO implementation.",
            "Uses constant-velocity local track predictions from PlannerInput.",
            "Not a stable-v1 leaderboard anchor until calibrated on official stress suites.",
        ),
    ),
    "ego_swarm": PlannerMetadata(
        method="ego_swarm",
        display_name="EGO-Swarm-inspired trajectory-sharing baseline",
        planner_type="decentralized_trajectory_optimization",
        role="experimental_baseline",
        status="experimental",
        dimensions=("2d", "3d"),
        observation_sources=("local_neighbors", "v2v", "sensor", "fused", "intent"),
        uses_v2v=True,
        uses_local_sensing=True,
        uses_intent=True,
        uses_obstacles=True,
        description=(
            "Clean-room EGO-Swarm-inspired baseline that samples smooth local trajectory "
            "topologies, scores swarm/obstacle clearance and smoothness, and advertises "
            "the selected local trajectory as an intent message."
        ),
        limitations=(
            "Not a port of the GPL ROS/C++ EGO-Swarm implementation.",
            "Uses DAA Microbench local tracks and AABB obstacles rather than onboard mapping or B-spline optimization.",
            "Experimental; not a stable-v1 leaderboard anchor until calibrated on official stress suites.",
        ),
    ),
    "ego_swarm_opt": PlannerMetadata(
        method="ego_swarm_opt",
        display_name="EGO-Swarm-style optimized trajectory-sharing baseline",
        planner_type="decentralized_control_point_trajectory_optimization",
        role="experimental_baseline",
        status="experimental",
        dimensions=("2d", "3d"),
        observation_sources=("local_neighbors", "v2v", "sensor", "fused", "intent"),
        uses_v2v=True,
        uses_local_sensing=True,
        uses_intent=True,
        uses_obstacles=True,
        description=(
            "Clean-room EGO-Swarm-style baseline that seeds topological local trajectories, "
            "optimizes smooth control points against dynamic feasibility, swarm clearance, "
            "obstacle clearance, and warm-start costs, then advertises the optimized trajectory."
        ),
        limitations=(
            "Not a port of the GPL ROS/C++ EGO-Swarm implementation.",
            "Optimizes DAA Microbench control points against supplied local tracks and AABB obstacles; it is not an onboard ESDF mapper.",
            "Experimental; needs official dense-3D and degraded-intent calibration before becoming a reference baseline.",
        ),
    ),
    "reciprocal_velocity_obstacle": PlannerMetadata(
        method="reciprocal_velocity_obstacle",
        display_name="Hybrid reciprocal velocity-obstacle baseline",
        planner_type="reciprocal_velocity_obstacle_sampling",
        role="experimental_baseline",
        status="experimental",
        dimensions=("2d", "3d"),
        observation_sources=("local_neighbors", "v2v", "sensor", "fused"),
        uses_v2v=True,
        uses_local_sensing=True,
        uses_obstacles=True,
        description=(
            "Hybrid reciprocal velocity-obstacle sampler with deterministic responsibility "
            "sharing, stale-track responsibility inflation, and tangent-boundary candidate commands."
        ),
        limitations=(
            "Experimental; not a certified HRVO implementation.",
            "Assumes neighbors share avoidance responsibility according to deterministic priority/id heuristics.",
            "Uses constant-velocity local track predictions from PlannerInput.",
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
    "learned_tiny": PlannerMetadata(
        method="learned_tiny",
        display_name="Tiny frozen learned-policy baseline",
        planner_type="learned_policy",
        role="experimental_baseline",
        status="experimental",
        dimensions=("2d", "3d"),
        observation_sources=("ego_state", "goal", "local_neighbors", "v2v", "sensor", "fused"),
        uses_v2v=True,
        uses_local_sensing=True,
        learned=True,
        description=(
            "Frozen linear-tanh learned-policy fixture loaded from a versioned JSON weight artifact. "
            "It maps public local planner features to normalized velocity commands."
        ),
        limitations=(
            "Tiny synthetic behavior-cloning fixture, not a competitive or certified DAA controller.",
            "Included to exercise learned-model packaging, disclosure, and benchmark-result plumbing.",
            "Uses local tracks supplied by PlannerInput rather than raw sensor processing.",
        ),
    ),
    "learned_policy_spec": PlannerMetadata(
        method="learned_policy_spec",
        display_name="External learned-policy spec bridge",
        planner_type="learned_policy",
        role="submission_bridge",
        status="experimental",
        dimensions=("2d", "3d"),
        observation_sources=("ego_state", "goal", "local_neighbors", "v2v", "sensor", "fused"),
        uses_v2v=True,
        uses_local_sensing=True,
        learned=True,
        description=(
            "Loads a trusted JSON/YAML RL policy spec and evaluates it as a normal local planner "
            "using the stable DAA RL observation/action contract."
        ),
        limitations=(
            "External specs can execute local Python code for callable/model adapters; only run trusted specs.",
            "Experimental submission bridge, not a built-in reference behavior.",
            "Uses local tracks supplied by PlannerInput rather than raw sensor processing.",
        ),
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
        status="pre_v1",
        dimensions=("2d", "3d"),
        observation_sources=("local_neighbors", "v2v", "sensor", "fused", "agent_messages"),
        uses_v2v=True,
        uses_local_sensing=True,
        uses_agent_messages=True,
        description="Agentic yielding baseline that exchanges proposals, acknowledgments, and local separation actions.",
        limitations=("Pre-v1 reference behavior; not yet a stable-v1 leaderboard anchor.",),
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


def make_planner(name: str, *, policy_spec: str | Path | None = None) -> ILocalPlanner:
    canonical = canonical_method(name)
    if canonical == "learned_policy_spec":
        return _make_learned_policy_spec(policy_spec)
    try:
        return _FACTORIES[canonical]()
    except KeyError as exc:
        raise ValueError(f"Unknown planner method: {name}") from exc
