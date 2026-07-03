from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

from microbench.types import PlannerInput, PlannerOutput


class ILocalPlanner(ABC):
    @abstractmethod
    def reset(self, seed: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def compute_cmd(self, planner_input: PlannerInput) -> np.ndarray | PlannerOutput:
        raise NotImplementedError
