from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from microbench.core import EpisodeEngine, EpisodeStep
from microbench.planners import make_planner
from microbench.planners.base import ILocalPlanner
from microbench.rl.schema import (
    AGENT_NAME_PREFIX,
    DEFAULT_REWARD_WEIGHTS,
    OBS_BASE_DIM,
    OBS_NEIGHBOR_DIM,
    OBSERVATION_LAYOUT,
    RL_POLICY_METHOD,
    action_schema,
    interface_contract,
    observation_schema,
    reward_schema,
)
from microbench.rl.spaces import box
from microbench.types import PlannerInput, PlannerOutput


try:  # pragma: no cover - exercised when pettingzoo is installed.
    from pettingzoo.utils.env import ParallelEnv as _PettingZooParallelEnv
except Exception:  # pragma: no cover - default test env uses the lightweight fallback.
    _PettingZooParallelEnv = object

try:  # pragma: no cover - exercised when gymnasium is installed.
    import gymnasium as _gymnasium
except Exception:  # pragma: no cover - default test env uses the lightweight fallback.
    _gymnasium = None


def agent_name(agent_id: int) -> str:
    return f"{AGENT_NAME_PREFIX}{int(agent_id)}"


def agent_id_from_name(name: str) -> int:
    if not str(name).startswith(AGENT_NAME_PREFIX):
        raise ValueError(f"Invalid agent name {name!r}; expected {AGENT_NAME_PREFIX}<id>")
    return int(str(name)[len(AGENT_NAME_PREFIX) :])


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.zeros(3, dtype=np.float32)
    return (np.asarray(v, dtype=np.float32) / n).astype(np.float32)


def _clamp_speed(v: np.ndarray, v_max: float) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(arr))
    if n <= float(v_max) or n < 1e-9:
        return arr
    return (arr / n * float(v_max)).astype(np.float32)


@dataclass
class _ActionProvider:
    strict_actions: bool = True
    actions: dict[int, np.ndarray] | None = None

    def set_actions(self, actions: dict[int, np.ndarray]) -> None:
        self.actions = {int(k): np.asarray(v, dtype=np.float32) for k, v in actions.items()}

    def command_for(self, agent_id: int, planner_input: PlannerInput) -> np.ndarray:
        if self.actions is None or int(agent_id) not in self.actions:
            if self.strict_actions:
                raise ValueError(f"Missing RL action for {agent_name(agent_id)}")
            return np.zeros(3, dtype=np.float32)
        action = np.asarray(self.actions[int(agent_id)], dtype=np.float32)
        if action.shape != (3,):
            raise ValueError(f"RL action for {agent_name(agent_id)} must have shape (3,), got {action.shape}")
        if not np.all(np.isfinite(action)):
            raise ValueError(f"RL action for {agent_name(agent_id)} must be finite")
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        if planner_input.planar:
            action[1] = 0.0
        return _clamp_speed(action * float(planner_input.ego.v_max), float(planner_input.ego.v_max))


class _ExternalActionPlanner(ILocalPlanner):
    def __init__(self, provider: _ActionProvider):
        self.provider = provider
        self.agent_id = -1

    def reset(self, seed: int, agent_id: int | None = None, config: dict | None = None) -> None:
        _ = seed, config
        if agent_id is not None:
            self.agent_id = int(agent_id)

    def compute_cmd(self, planner_input: PlannerInput) -> PlannerOutput:
        agent_id = self.agent_id if self.agent_id >= 0 else int(planner_input.ego.idx)
        v_cmd = self.provider.command_for(agent_id, planner_input)
        return PlannerOutput(
            v_cmd=v_cmd.astype(float),
            debug_info={
                "rl_policy": True,
                "rl_action_norm": float(np.linalg.norm(v_cmd) / max(1e-6, float(planner_input.ego.v_max))),
            },
        )


class DaaParallelEnv(_PettingZooParallelEnv):
    """PettingZoo-style parallel environment over the DAA Microbench engine.

    The environment controls agents whose resolved planner method is
    `controlled_method` (`rl_policy` by default). Scenario-configured agents
    with another method, such as noncooperative `baseline_goal` intruders, remain
    part of the simulation as background traffic.
    """

    metadata = {
        "name": "daa_microbench_parallel_v0",
        "render_modes": ["trace_frame"],
        "is_parallelizable": True,
    }

    def __init__(
        self,
        *,
        scenario_path: str,
        n_agents: int,
        seed: int = 0,
        comm_profile: str = "ideal_50hz",
        method: str = RL_POLICY_METHOD,
        controlled_method: str = RL_POLICY_METHOD,
        controlled_agents: list[int] | tuple[int, ...] | None = None,
        agent_methods: list[str] | None = None,
        terminate_on_collision: bool = False,
        strict_actions: bool = True,
        reward_config: dict[str, float] | None = None,
    ):
        self.scenario_path = str(scenario_path)
        self.n_agents = int(n_agents)
        self.seed_value = int(seed)
        self.comm_profile = str(comm_profile)
        self.method = str(method)
        self.controlled_method = str(controlled_method)
        self.controlled_agents_override = None if controlled_agents is None else [int(x) for x in controlled_agents]
        self.agent_methods = list(agent_methods) if agent_methods is not None else None
        self.terminate_on_collision = bool(terminate_on_collision)
        self.strict_actions = bool(strict_actions)
        self.reward_config = {**DEFAULT_REWARD_WEIGHTS, **(reward_config or {})}

        self._engine: EpisodeEngine | None = None
        self._last_step: EpisodeStep | None = None
        self._provider = _ActionProvider(strict_actions=self.strict_actions)
        self._last_goal_dist = np.zeros(self.n_agents, dtype=float)
        self._last_done = np.zeros(self.n_agents, dtype=bool)
        self.possible_agents = [agent_name(i) for i in (self.controlled_agents_override or range(self.n_agents))]
        self.agents: list[str] = []
        self._obs_space = None
        self._action_space = None

    def _planner_factory(self, method: str):
        if str(method) == self.controlled_method:
            return _ExternalActionPlanner(self._provider)
        return make_planner(method)

    def _new_engine(self, seed: int) -> EpisodeEngine:
        return EpisodeEngine(
            scenario_path=self.scenario_path,
            method=self.method,
            n_agents=self.n_agents,
            seed=int(seed),
            comm_profile=self.comm_profile,
            agent_methods=self.agent_methods,
            planner_factory=self._planner_factory,
        )

    def _controlled_agent_ids(self) -> list[int]:
        if self.controlled_agents_override is not None:
            return list(self.controlled_agents_override)
        if self._engine is None:
            return list(range(self.n_agents))
        return [i for i, method in enumerate(self._engine.agent_methods) if method == self.controlled_method]

    def reset(self, seed: int | None = None, options: dict | None = None):
        _ = options
        if self._engine is not None:
            self._engine.close()
        if seed is not None:
            self.seed_value = int(seed)
        self._provider = _ActionProvider(strict_actions=self.strict_actions)
        self._engine = self._new_engine(self.seed_value)
        controlled = self._controlled_agent_ids()
        self.possible_agents = [agent_name(i) for i in controlled]
        self.agents = list(self.possible_agents)
        self._last_step = None
        self._last_goal_dist = self._goal_distances()
        self._last_done = np.array([s.done for s in self._engine.states], dtype=bool)
        observations = {name: self._observation(agent_id_from_name(name)) for name in self.agents}
        infos = {name: self._info(agent_id_from_name(name), reset=True) for name in self.agents}
        return observations, infos

    def step(self, actions: dict[str, np.ndarray]):
        if self._engine is None:
            raise RuntimeError("Call reset() before step().")

        current_agents = list(self.agents)
        if not current_agents:
            return {}, {}, {}, {}, {}

        action_ids = {agent_id_from_name(name): np.asarray(action, dtype=np.float32) for name, action in actions.items()}
        expected_ids = {agent_id_from_name(name) for name in current_agents}
        if self.strict_actions:
            missing = sorted(expected_ids - set(action_ids))
            extra = sorted(set(action_ids) - expected_ids)
            if missing or extra:
                raise ValueError(
                    "RL action keys must match current agents; "
                    f"missing={[agent_name(i) for i in missing]} extra={[agent_name(i) for i in extra]}"
                )
        else:
            action_ids = {i: action_ids.get(i, np.zeros(3, dtype=np.float32)) for i in expected_ids}

        previous_goal_dist = self._goal_distances()
        previous_done = np.array([s.done for s in self._engine.states], dtype=bool)
        self._provider.set_actions(action_ids)
        step = self._engine.step()
        if step is None:
            observations = {name: self._observation(agent_id_from_name(name)) for name in current_agents}
            rewards = {name: 0.0 for name in current_agents}
            terminations = {name: True for name in current_agents}
            truncations = {name: False for name in current_agents}
            infos = {name: self._info(agent_id_from_name(name)) for name in current_agents}
            self.agents = []
            return observations, rewards, terminations, truncations, infos

        self._last_step = step
        new_goal_dist = self._goal_distances()
        collision_agents = {idx for pair in step.collision_pairs for idx in pair}
        near_agents = {idx for pair in step.near_miss_pairs for idx in pair}
        horizon_truncated = bool(self._engine.k >= self._engine.steps)

        observations: dict[str, np.ndarray] = {}
        rewards: dict[str, float] = {}
        terminations: dict[str, bool] = {}
        truncations: dict[str, bool] = {}
        infos: dict[str, dict] = {}
        for name in current_agents:
            idx = agent_id_from_name(name)
            progress = float(previous_goal_dist[idx] - new_goal_dist[idx])
            newly_done = bool(step.done[idx] and not previous_done[idx])
            collision = idx in collision_agents
            near_miss = idx in near_agents
            reward = (
                float(self.reward_config["progress"]) * progress
                + float(self.reward_config["time"])
                + (float(self.reward_config["goal"]) if newly_done else 0.0)
                + (float(self.reward_config["collision"]) if collision else 0.0)
                + (float(self.reward_config["near_miss"]) if near_miss and not collision else 0.0)
            )
            terminated = bool(step.done[idx] or (self.terminate_on_collision and collision))
            truncated = bool(horizon_truncated and not terminated)
            observations[name] = self._observation(idx)
            rewards[name] = float(reward)
            terminations[name] = terminated
            truncations[name] = truncated
            infos[name] = self._info(
                idx,
                collision=collision,
                near_miss=near_miss,
                progress_m=progress,
                newly_done=newly_done,
            )

        self._last_goal_dist = new_goal_dist
        self._last_done = np.array([s.done for s in self._engine.states], dtype=bool)
        self.agents = [name for name in current_agents if not (terminations[name] or truncations[name])]
        return observations, rewards, terminations, truncations, infos

    def _goal_distances(self) -> np.ndarray:
        if self._engine is None:
            return np.zeros(self.n_agents, dtype=float)
        return np.asarray(
            [float(np.linalg.norm(state.goal - state.pos)) for state in self._engine.states],
            dtype=float,
        )

    def _observation_dim(self) -> int:
        top_k = 8
        if self._engine is not None:
            top_k = int(self._engine.neighbor_cfg.get("top_k", 8))
        return OBS_BASE_DIM + max(0, top_k) * OBS_NEIGHBOR_DIM

    def _observation(self, agent_id: int) -> np.ndarray:
        if self._engine is None:
            return np.zeros(self._observation_dim(), dtype=np.float32)
        state = self._engine.states[int(agent_id)]
        goal_delta = np.asarray(state.goal, dtype=float) - np.asarray(state.pos, dtype=float)
        goal_dist = float(np.linalg.norm(goal_delta))
        goal_dir = _normalize(goal_delta)
        t = float(self._engine.k * self._engine.dt)
        context = self._engine.agent_contexts[int(agent_id)]
        base = [
            *np.asarray(state.pos, dtype=float).tolist(),
            *np.asarray(state.vel, dtype=float).tolist(),
            *goal_dir.tolist(),
            goal_dist,
            1.0 if state.done else 0.0,
            t,
            float(agent_id) / max(1.0, float(self.n_agents - 1)),
            float(context.priority),
            float(state.radius),
            float(state.v_max),
            float(state.a_max),
        ]

        neighbors: list[dict[str, Any]] = []
        if self._last_step is not None and int(agent_id) < len(self._last_step.selected_obs):
            neighbors = list(self._last_step.selected_obs[int(agent_id)])
        top_k = int((self._engine.neighbor_cfg if self._engine is not None else {}).get("top_k", 8))
        features: list[float] = []
        for obs in neighbors[:top_k]:
            rel_pos = np.asarray(obs.get("pos", [0.0, 0.0, 0.0]), dtype=float) - np.asarray(state.pos, dtype=float)
            rel_vel = np.asarray(obs.get("vel", [0.0, 0.0, 0.0]), dtype=float) - np.asarray(state.vel, dtype=float)
            features.extend(
                [
                    1.0,
                    *rel_pos.tolist(),
                    *rel_vel.tolist(),
                    float(obs.get("radius", 0.0)),
                    float(obs.get("msg_age_sec", 0.0)),
                ]
            )
        while len(features) < top_k * OBS_NEIGHBOR_DIM:
            features.extend([0.0] * OBS_NEIGHBOR_DIM)
        return np.asarray([*base, *features], dtype=np.float32)

    def _info(self, agent_id: int, *, reset: bool = False, **extra) -> dict:
        if self._engine is None:
            return dict(extra)
        context = self._engine.agent_contexts[int(agent_id)]
        info = {
            "agent_id": int(agent_id),
            "method": str(context.method),
            "role": context.role,
            "priority": int(context.priority),
            "controlled": str(context.method) == self.controlled_method,
            "reset": bool(reset),
            "t": float(self._engine.k * self._engine.dt),
        }
        if self._last_step is not None:
            info.update(
                {
                    "done": bool(self._last_step.done[int(agent_id)]),
                    "min_sep_m": float(self._last_step.min_sep),
                    "collisions": int(self._last_step.collisions),
                    "near_misses": int(self._last_step.near_misses),
                    "planner_debug": self._last_step.planner_debug[int(agent_id)],
                    "perception_debug": self._last_step.perception_debug[int(agent_id)],
                }
            )
        info.update(extra)
        return info

    def observation_space(self, agent: str):
        if self._obs_space is None or getattr(self._obs_space, "shape", None) != (self._observation_dim(),):
            self._obs_space = box(low=-np.inf, high=np.inf, shape=(self._observation_dim(),), dtype=np.float32)
        _ = agent
        return self._obs_space

    def action_space(self, agent: str):
        if self._action_space is None:
            self._action_space = box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        _ = agent
        return self._action_space

    def observation_schema(self) -> dict[str, Any]:
        return observation_schema(top_k=(self._observation_dim() - OBS_BASE_DIM) // OBS_NEIGHBOR_DIM)

    def action_schema(self) -> dict[str, Any]:
        return action_schema()

    def reward_schema(self) -> dict[str, Any]:
        return reward_schema(self.reward_config)

    def interface_contract(self) -> dict[str, Any]:
        return interface_contract(
            top_k=(self._observation_dim() - OBS_BASE_DIM) // OBS_NEIGHBOR_DIM,
            reward_config=self.reward_config,
        )

    def render(self):
        if self._last_step is None:
            return None
        return self._last_step.trace_frame()

    @property
    def episode_step_limit(self) -> int | None:
        return None if self._engine is None else int(self._engine.steps)

    @property
    def planar(self) -> bool | None:
        return None if self._engine is None else bool(self._engine.planar)

    def close(self) -> None:
        if self._engine is not None:
            self._engine.close()
        self._engine = None


class DaaSingleAgentEnv(_gymnasium.Env if _gymnasium is not None else object):
    """Gymnasium-style single-ego wrapper with benchmark traffic as background."""

    metadata = {"render_modes": ["trace_frame"], "render_fps": 50}

    def __init__(
        self,
        *,
        scenario_path: str,
        n_agents: int,
        ego_agent_id: int = 0,
        seed: int = 0,
        comm_profile: str = "ideal_50hz",
        background_method: str = "orca_heuristic",
        terminate_on_collision: bool = False,
    ):
        self.ego_agent_id = int(ego_agent_id)
        if self.ego_agent_id < 0 or self.ego_agent_id >= int(n_agents):
            raise ValueError(f"ego_agent_id must be in [0, {int(n_agents)})")
        agent_methods = [str(background_method) for _ in range(int(n_agents))]
        agent_methods[self.ego_agent_id] = RL_POLICY_METHOD
        self.parallel_env = DaaParallelEnv(
            scenario_path=str(scenario_path),
            n_agents=int(n_agents),
            seed=int(seed),
            comm_profile=str(comm_profile),
            method=str(background_method),
            agent_methods=agent_methods,
            controlled_agents=[self.ego_agent_id],
            terminate_on_collision=terminate_on_collision,
        )
        self.agent_name = agent_name(self.ego_agent_id)
        self.observation_space = self.parallel_env.observation_space(self.agent_name)
        self.action_space = self.parallel_env.action_space(self.agent_name)

    def reset(self, seed: int | None = None, options: dict | None = None):
        observations, infos = self.parallel_env.reset(seed=seed, options=options)
        return observations[self.agent_name], infos[self.agent_name]

    def step(self, action: np.ndarray):
        if self.agent_name not in self.parallel_env.agents:
            raise RuntimeError("Cannot step inactive ego agent; call reset() before continuing.")
        observations, rewards, terminations, truncations, infos = self.parallel_env.step({self.agent_name: action})
        return (
            observations[self.agent_name],
            rewards[self.agent_name],
            terminations[self.agent_name],
            truncations[self.agent_name],
            infos[self.agent_name],
        )

    def render(self):
        return self.parallel_env.render()

    def close(self) -> None:
        self.parallel_env.close()


def parallel_env(**kwargs) -> DaaParallelEnv:
    return DaaParallelEnv(**kwargs)


def single_agent_env(**kwargs) -> DaaSingleAgentEnv:
    return DaaSingleAgentEnv(**kwargs)
