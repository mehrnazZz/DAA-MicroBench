from __future__ import annotations

import numpy as np

from microbench.planners.base import ILocalPlanner
from microbench.learned import (
    TINY_LEARNED_MODEL_ID,
    TinyLinearPolicyModel,
    planner_input_to_tiny_features,
    tiny_learned_model_path,
)
from microbench.types import PlannerInput, PlannerOutput


class LearnedTinyPlanner(ILocalPlanner):
    """Frozen tiny learned-model baseline.

    This planner is intentionally transparent and modest: it loads a small
    linear-tanh policy from a versioned JSON weight artifact and maps public
    `PlannerInput` features to a normalized velocity action. It is a learned
    model plumbing baseline, not a safety-certified controller.
    """

    def __init__(self, model: TinyLinearPolicyModel | None = None):
        self.model = model if model is not None else TinyLinearPolicyModel.from_path()
        self.seed = 0

    def reset(self, seed: int) -> None:
        self.seed = int(seed)

    def compute_cmd(self, planner_input: PlannerInput) -> PlannerOutput:
        features = planner_input_to_tiny_features(planner_input)
        action = self.model.action_from_features(features)
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
                "learned_model_expected_id": TINY_LEARNED_MODEL_ID,
                "learned_weight_artifact": tiny_learned_model_path(),
                "learned_policy_action_norm": float(np.linalg.norm(action)),
                "learned_policy_threat_scalar": float(features[-2]),
                "learned_policy_neighbor_count_frac": float(features[-1]),
            },
        )
