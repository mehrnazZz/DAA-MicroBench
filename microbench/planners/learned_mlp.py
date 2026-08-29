from __future__ import annotations

import numpy as np

from microbench.learned import (
    FrozenMlpPolicyModel,
    mlp_learned_model_path,
    mlp_model_id_for_feature_set,
    planner_input_to_mlp_features,
)
from microbench.planners.base import ILocalPlanner
from microbench.types import PlannerInput, PlannerOutput


class LearnedMlpPlanner(ILocalPlanner):
    """Frozen MLP learned-model baseline over public local planner features."""

    def __init__(self, model: FrozenMlpPolicyModel | None = None):
        self.model = model if model is not None else FrozenMlpPolicyModel.from_path()
        self.seed = 0

    def reset(self, seed: int) -> None:
        self.seed = int(seed)

    def compute_cmd(self, planner_input: PlannerInput) -> PlannerOutput:
        action = self.model.action_from_planner_input(planner_input)
        debug_features = planner_input_to_mlp_features(planner_input)
        if planner_input.planar:
            action[1] = 0.0
        v_cmd = np.asarray(action, dtype=np.float32) * float(planner_input.ego.v_max)
        speed = float(np.linalg.norm(v_cmd))
        if speed > float(planner_input.ego.v_max) + 1e-9:
            v_cmd = v_cmd / speed * float(planner_input.ego.v_max)

        return PlannerOutput(
            v_cmd=v_cmd.astype(float),
            debug_info={
                "learned_model": True,
                "learned_model_id": self.model.model_id,
                "learned_model_expected_id": mlp_model_id_for_feature_set(self.model.feature_set),
                "learned_weight_artifact": mlp_learned_model_path(),
                "learned_policy_architecture": "mlp_tanh",
                "learned_policy_feature_set": self.model.feature_set,
                "learned_policy_feature_dim": int(len(self.model.feature_names)),
                "learned_policy_hidden_dim": int(self.model.hidden_dim),
                "learned_policy_action_norm": float(np.linalg.norm(action)),
                "learned_policy_threat_scalar": float(debug_features[-2]),
                "learned_policy_neighbor_count_frac": float(debug_features[-1]),
            },
        )
