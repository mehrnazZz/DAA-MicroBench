from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from microbench.types import IntentObs, NeighborObs, PlannerInput


@dataclass(frozen=True)
class LocalTrafficContext:
    input_id: int
    neighbors: tuple[NeighborObs, ...]
    intents_by_sender: dict[int, IntentObs]
    intent_only: tuple[IntentObs, ...]
    input_neighbor_count: int
    input_valid_intent_count: int
    selected_intent_count: int
    pruned_intent_count: int


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _finite_points(intent: IntentObs) -> np.ndarray:
    points = np.asarray(intent.points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    mask = np.all(np.isfinite(points), axis=1)
    return points[mask].astype(np.float32)


def _dedupe_valid_intents(intents: Iterable[IntentObs]) -> dict[int, IntentObs]:
    by_sender: dict[int, IntentObs] = {}
    for intent in intents:
        if not bool(intent.valid):
            continue
        if _finite_points(intent).size == 0:
            continue
        sender = int(intent.sender_id)
        previous = by_sender.get(sender)
        if previous is None or float(intent.intent_age_s) < float(previous.intent_age_s):
            by_sender[sender] = intent
    return by_sender


def _intent_priority(planner_input: PlannerInput, intent: IntentObs) -> tuple[float, float, int]:
    ego_pos = np.asarray(planner_input.ego.pos, dtype=np.float32).copy()
    points = _finite_points(intent)
    if planner_input.planar:
        ego_pos[1] = float(planner_input.ego.pos[1])
        points[:, 1] = float(planner_input.ego.pos[1])
    lookahead = points[: min(points.shape[0], 4)]
    dist = min((_norm(point - ego_pos) for point in lookahead), default=float("inf"))
    age = float(intent.intent_age_s)
    if not math.isfinite(age):
        age = float("inf")
    return (float(dist), age, int(intent.sender_id))


def select_local_traffic(
    planner_input: PlannerInput,
    *,
    max_neighbors: int,
    max_intents: int | None = None,
) -> LocalTrafficContext:
    neighbor_budget = max(0, int(max_neighbors))
    neighbors = tuple(planner_input.neighbors[:neighbor_budget])
    neighbor_ids = {int(nobs.idx) for nobs in neighbors}
    valid_intents = _dedupe_valid_intents(planner_input.neighbor_intents)

    associated = {sender: intent for sender, intent in valid_intents.items() if sender in neighbor_ids}
    intent_budget = len(associated) if max_intents is None else max(len(associated), int(max_intents))
    extras = [
        intent
        for sender, intent in valid_intents.items()
        if sender not in associated
    ]
    extras.sort(key=lambda intent: _intent_priority(planner_input, intent))
    selected_extras = extras[: max(0, intent_budget - len(associated))]

    intents_by_sender = dict(associated)
    for intent in selected_extras:
        intents_by_sender[int(intent.sender_id)] = intent

    return LocalTrafficContext(
        input_id=id(planner_input),
        neighbors=neighbors,
        intents_by_sender=intents_by_sender,
        intent_only=tuple(selected_extras),
        input_neighbor_count=len(planner_input.neighbors),
        input_valid_intent_count=len(valid_intents),
        selected_intent_count=len(intents_by_sender),
        pruned_intent_count=max(0, len(valid_intents) - len(intents_by_sender)),
    )
