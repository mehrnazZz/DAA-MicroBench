from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from microbench.learned.rl_bridge import (
    agent_name,
    clamp_normalized_velocity_action,
    clamp_velocity,
    normalized_velocity_action_space,
    planner_input_to_rl_info,
    planner_input_to_rl_observation,
)
from microbench.planners.base import ILocalPlanner
from microbench.types import PlannerInput, PlannerOutput


class LearnedPolicySpecPlanner(ILocalPlanner):
    """Planner bridge for trusted external RL policy specs.

    The loaded policy receives the same local observation/action contract used by
    the PettingZoo/Gymnasium wrappers, then its normalized action is scaled into
    the public planner velocity command contract.
    """

    def __init__(self, policy_spec: str | Path):
        if policy_spec is None or not str(policy_spec).strip():
            raise ValueError("learned_policy_spec requires --policy-spec")
        self.policy_spec = str(policy_spec)
        self.loaded_policy: Any | None = None
        self.seed = 0
        self.agent_id = -1
        self.top_k = 8
        self.n_agents: int | None = None
        self.action_space = normalized_velocity_action_space()

    def reset(self, seed: int, agent_id: int | None = None, config: dict | None = None) -> None:
        from microbench.rl.policy_spec import load_policy_from_spec

        cfg = dict(config or {})
        self.seed = int(seed)
        if agent_id is not None:
            self.agent_id = int(agent_id)
        elif cfg.get("agent_id") is not None:
            self.agent_id = int(cfg["agent_id"])
        if cfg.get("neighbor_top_k") is not None:
            self.top_k = max(0, int(cfg["neighbor_top_k"]))
        elif cfg.get("top_k") is not None:
            self.top_k = max(0, int(cfg["top_k"]))
        self.n_agents = int(cfg["n_agents"]) if cfg.get("n_agents") is not None else None
        self.loaded_policy = load_policy_from_spec(self.policy_spec, seed=self.seed)

    @property
    def policy_name(self) -> str:
        if self.loaded_policy is None:
            return Path(self.policy_spec).stem
        return str(self.loaded_policy.policy_name)

    @property
    def policy_adapter(self) -> str | None:
        if self.loaded_policy is None:
            return None
        return str(self.loaded_policy.summary.get("adapter"))

    def compute_cmd(self, planner_input: PlannerInput) -> PlannerOutput:
        if self.loaded_policy is None:
            self.reset(self.seed, agent_id=int(planner_input.ego.idx))

        assert self.loaded_policy is not None
        agent_id = self.agent_id if self.agent_id >= 0 else int(planner_input.ego.idx)
        observation = planner_input_to_rl_observation(
            planner_input,
            top_k=self.top_k,
            n_agents=self.n_agents,
        )
        info = planner_input_to_rl_info(
            planner_input,
            top_k=self.top_k,
            n_agents=self.n_agents,
            policy_name=self.loaded_policy.policy_name,
        )
        raw_action = self.loaded_policy.policy.action(
            agent_name(agent_id),
            observation,
            self.action_space,
            info,
        )
        action = clamp_normalized_velocity_action(raw_action, self.action_space)
        if planner_input.planar:
            action[1] = 0.0
        v_cmd = clamp_velocity(action * float(planner_input.ego.v_max), float(planner_input.ego.v_max))

        return PlannerOutput(
            v_cmd=v_cmd.astype(float),
            debug_info={
                "learned_model": True,
                "learned_policy_spec": True,
                "learned_policy_name": self.loaded_policy.policy_name,
                "learned_policy_adapter": self.loaded_policy.summary.get("adapter"),
                "learned_policy_spec_path": self.loaded_policy.spec_path,
                "learned_policy_action_norm": float(np.linalg.norm(action)),
                "learned_policy_observation_dim": int(observation.shape[0]),
                "learned_policy_observation_top_k": int(self.top_k),
            },
        )
