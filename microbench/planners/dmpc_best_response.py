from __future__ import annotations

from typing import Any

import numpy as np

from microbench.comm.messages import make_intent_trajectory
from microbench.planners.mpc_nonlinear import NonlinearMpcPlanner, _clamp_speed, _limit_delta, _norm
from microbench.types import IntentObs, NeighborObs, PlannerInput, PlannerOutput


class DistributedMpcBestResponsePlanner(NonlinearMpcPlanner):
    """Distributed best-response MPC baseline.

    This planner keeps the nonlinear MPC backend from ``mpc_nonlinear`` but
    changes the coupling model to a distributed-MPC style best response:
    neighbor intent trajectories are treated as the primary coupled predictions,
    stale or missing plans fall back to constant-velocity tracks with extra
    inflation, and every tick publishes the optimized plan for the next
    best-response round.
    """

    def __init__(self, cfg: dict | None = None):
        cfg = dict(cfg or {})
        super().__init__(cfg=cfg)
        self.coordination_mode = str(cfg.get("coordination_mode", "sequential_best_response"))
        self.coordination_rounds = int(cfg.get("coordination_rounds", 1))
        self.intent_trust_horizon_s = float(cfg.get("intent_trust_horizon_s", 0.85))
        self.missing_intent_inflation_m = float(cfg.get("missing_intent_inflation_m", 0.35))
        self.stale_intent_inflation_m = float(cfg.get("stale_intent_inflation_m", 0.55))
        self.priority_responsibility_gain = float(cfg.get("priority_responsibility_gain", 0.25))
        self.priority_inflation_gain = float(cfg.get("priority_inflation_gain", 0.15))
        self.intent_coupling_weight = float(cfg.get("intent_coupling_weight", 0.75))
        self.hard_constraint_slack_weight = float(cfg.get("hard_constraint_slack_weight", 4500.0))
        self.emit_agent_messages = bool(cfg.get("emit_agent_messages", True))
        self.coordination_message_ttl_s = float(cfg.get("coordination_message_ttl_s", 0.75))
        self.velocity_guard_enabled = bool(cfg.get("velocity_guard_enabled", True))
        self.velocity_guard_margin_m = float(cfg.get("velocity_guard_margin_m", 0.35))
        self.velocity_guard_activation_margin_m = float(cfg.get("velocity_guard_activation_margin_m", 1.25))
        self.velocity_guard_ttc_s = float(cfg.get("velocity_guard_ttc_s", 3.0))
        self.velocity_guard_alpha = float(cfg.get("velocity_guard_alpha", 1.4))
        self.velocity_guard_iterations = int(cfg.get("velocity_guard_iterations", 3))
        self.velocity_guard_brake_enabled = bool(cfg.get("velocity_guard_brake_enabled", True))
        self.velocity_guard_brake_clearance_m = float(cfg.get("velocity_guard_brake_clearance_m", 0.4))
        self.velocity_guard_brake_scale = float(cfg.get("velocity_guard_brake_scale", 0.6))
        self._local_memory: dict[str, object] = {}
        self._dmpc_seed_stats: dict[str, dict[str, object]] = {}
        self._dmpc_last_eval_stats: dict[str, object] = {}
        self._dmpc_current_seed: str | None = None

    def reset(self, seed: int) -> None:
        super().reset(seed)
        self._local_memory.clear()
        self._dmpc_seed_stats.clear()
        self._dmpc_last_eval_stats.clear()
        self._dmpc_current_seed = None

    def compute_cmd(self, planner_input: PlannerInput) -> PlannerOutput:
        self._dmpc_seed_stats = {}
        self._dmpc_last_eval_stats = {}
        out = super().compute_cmd(planner_input)

        intent = out.intent_out
        base_debug = dict(out.debug_info)
        current = np.asarray(planner_input.ego.vel, dtype=np.float32).copy()
        if planner_input.planar:
            current[1] = 0.0
        v_cmd, guard_info = self._apply_velocity_guard(
            planner_input,
            np.asarray(out.v_cmd, dtype=np.float32),
            current,
        )
        if intent is not None and (
            guard_info["dmpc_best_response_velocity_guard_adjusted"]
            or guard_info["dmpc_best_response_velocity_guard_brake_applied"]
        ):
            self._override_intent_first_step(planner_input, intent, v_cmd)
        best_seed = str(base_debug.get("mpc_nonlinear_best_seed", ""))
        stats = dict(self._dmpc_seed_stats.get(best_seed, self._dmpc_last_eval_stats))

        memory = self._memory(planner_input)
        plan_version = int(memory.get("dmpc_best_response_plan_version", 0)) + 1
        prior_mode = memory.get("dmpc_best_response_last_mode")
        memory["dmpc_best_response_plan_version"] = plan_version
        memory["dmpc_best_response_last_mode"] = best_seed

        messages_out = list(out.messages_out or [])
        if intent is not None:
            intent.kind = "DMPC_BEST_RESPONSE_TRAJECTORY"
            intent.mode = f"{self.coordination_mode}:{best_seed}"
            if self.emit_agent_messages:
                messages_out.append(
                    make_intent_trajectory(
                        sender_id=int(planner_input.ego.idx),
                        recipient_id=None,
                        now_s=float(planner_input.t),
                        trajectory=intent.points,
                        dt_plan_s=float(intent.dt_plan_s or self._dt()),
                        expiry_s=float(intent.expiry_s),
                        tube_radius_m=float(intent.tube_radius_m),
                        ttl_s=self.coordination_message_ttl_s,
                    )
                )

        debug = dict(base_debug)
        for key, value in base_debug.items():
            if key.startswith("mpc_nonlinear_"):
                debug[f"dmpc_best_response_{key.removeprefix('mpc_nonlinear_')}"] = value
        debug.update(stats)
        debug.update(
            {
                "dmpc_best_response_algorithm": "distributed_best_response_nonlinear_mpc",
                "dmpc_best_response_mpc_backend": "clean_room_multiple_shooting_nonlinear_mpc",
                "dmpc_best_response_coordination_mode": self.coordination_mode,
                "dmpc_best_response_coordination_rounds": int(self.coordination_rounds),
                "dmpc_best_response_plan_version": int(plan_version),
                "dmpc_best_response_prior_mode": None if prior_mode is None else str(prior_mode),
                "dmpc_best_response_intent_kind": getattr(intent, "kind", None),
                "dmpc_best_response_agent_messages": int(len(messages_out)),
                "dmpc_best_response_agent_priority": int(self._agent_priority(planner_input)),
                "dmpc_best_response_velocity_guard_enabled": bool(self.velocity_guard_enabled),
                **guard_info,
            }
        )
        try:
            return PlannerOutput(
                v_cmd=np.asarray(v_cmd, dtype=float),
                intent_out=intent,
                messages_out=messages_out,
                debug_info=debug,
            )
        finally:
            self._traffic_context = None

    def _apply_velocity_guard(
        self,
        planner_input: PlannerInput,
        v_cmd: np.ndarray,
        current: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if not self.velocity_guard_enabled:
            return np.asarray(v_cmd, dtype=np.float32), {
                "dmpc_best_response_velocity_guard_adjusted": False,
                "dmpc_best_response_velocity_guard_constraint_count": 0,
                "dmpc_best_response_velocity_guard_min_clearance_m": None,
                "dmpc_best_response_velocity_guard_min_ttc_s": None,
                "dmpc_best_response_velocity_guard_delta_norm": 0.0,
                "dmpc_best_response_velocity_guard_brake_applied": False,
            }

        ego = planner_input.ego
        guarded = np.asarray(v_cmd, dtype=np.float32).copy()
        start = guarded.copy()
        max_delta = float(ego.a_max) * float(planner_input.dt)
        active: list[tuple[np.ndarray, float]] = []
        min_clearance: float | None = None
        min_ttc: float | None = None
        brake_applied = False

        for nobs in self._traffic(planner_input).neighbors:
            if not bool(nobs.valid):
                continue
            rel = np.asarray(ego.pos, dtype=np.float32) - np.asarray(nobs.pos, dtype=np.float32)
            other_v = np.asarray(nobs.vel, dtype=np.float32)
            if planner_input.planar:
                rel[1] = 0.0
                other_v[1] = 0.0
            dist = _norm(rel)
            if dist < 1.0e-6:
                rel = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
                dist = 1.0
            unit = rel / dist
            safe = (
                float(ego.radius)
                + float(nobs.radius)
                + self.safety_margin_m
                + self._neighbor_inflation(nobs)
                + max(0.0, self._priority_inflation(planner_input, nobs))
            )
            safe += max(0.0, float(self.velocity_guard_margin_m))
            clearance = dist - safe
            min_clearance = clearance if min_clearance is None else min(min_clearance, clearance)
            rel_v_current = np.asarray(current, dtype=np.float32) - other_v
            vv = float(np.dot(rel_v_current, rel_v_current))
            ttc: float | None = None
            if vv > 1.0e-9:
                tau = -float(np.dot(rel, rel_v_current)) / vv
                if tau >= 0.0:
                    ttc = tau
                    min_ttc = tau if min_ttc is None else min(min_ttc, tau)
            within_range = dist <= safe + max(0.0, float(self.velocity_guard_activation_margin_m))
            closing_soon = ttc is not None and ttc <= max(0.0, float(self.velocity_guard_ttc_s))
            if not within_range and not closing_soon:
                continue
            rhs = float(np.dot(unit, other_v) - float(self.velocity_guard_alpha) * clearance)
            active.append((unit.astype(np.float32), rhs))

        for _ in range(max(1, self.velocity_guard_iterations)):
            for unit, rhs in active:
                lhs = float(np.dot(unit, guarded))
                if lhs < rhs:
                    guarded = guarded + unit * (rhs - lhs)
            guarded = _limit_delta(guarded, current, max_delta)
            guarded = _clamp_speed(guarded, float(ego.v_max))
            if planner_input.planar:
                guarded[1] = 0.0

        if (
            self.velocity_guard_brake_enabled
            and active
            and min_clearance is not None
            and float(min_clearance) <= max(0.0, float(self.velocity_guard_brake_clearance_m))
        ):
            brake_scale = max(0.0, min(1.0, float(self.velocity_guard_brake_scale)))
            guarded = _limit_delta(guarded * brake_scale, current, max_delta)
            guarded = _clamp_speed(guarded, float(ego.v_max))
            if planner_input.planar:
                guarded[1] = 0.0
            brake_applied = True

        delta = _norm(guarded - start)
        return guarded.astype(np.float32), {
            "dmpc_best_response_velocity_guard_adjusted": bool(delta > 1.0e-7),
            "dmpc_best_response_velocity_guard_constraint_count": int(len(active)),
            "dmpc_best_response_velocity_guard_min_clearance_m": min_clearance,
            "dmpc_best_response_velocity_guard_min_ttc_s": min_ttc,
            "dmpc_best_response_velocity_guard_delta_norm": float(delta),
            "dmpc_best_response_velocity_guard_brake_applied": bool(brake_applied),
        }

    def _override_intent_first_step(self, planner_input: PlannerInput, intent: Any, v_cmd: np.ndarray) -> None:
        points = np.asarray(intent.points, dtype=np.float32).copy()
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 3:
            return
        dt = float(intent.dt_plan_s) if intent.dt_plan_s is not None and intent.dt_plan_s > 1.0e-9 else self._dt()
        points[1] = np.asarray(planner_input.ego.pos, dtype=np.float32) + np.asarray(v_cmd, dtype=np.float32) * dt
        if planner_input.planar:
            points[1, 1] = float(planner_input.ego.pos[1])
        intent.points = points.astype(float)

    def _optimize_seed(self, planner_input: PlannerInput, seed):
        previous = self._dmpc_current_seed
        self._dmpc_current_seed = str(seed.label)
        try:
            result = super()._optimize_seed(planner_input, seed)
            self._dmpc_seed_stats[str(result.label)] = dict(self._dmpc_last_eval_stats)
            return result
        finally:
            self._dmpc_current_seed = previous

    def _collision_penalty_and_grad(
        self,
        planner_input: PlannerInput,
        positions: np.ndarray,
    ) -> tuple[float, np.ndarray, float | None, bool]:
        grad = np.zeros_like(positions, dtype=np.float32)
        penalty = 0.0
        min_clearance: float | None = None
        conflict = False
        slack_penalty = 0.0
        dt = self._dt()

        traffic = self._traffic(planner_input)
        intent_by_sender = traffic.intents_by_sender
        usable_intent_ids = {
            sender_id for sender_id, intent in intent_by_sender.items() if self._intent_usable_for_coupling(intent)
        }
        stale_intent_ids = {
            sender_id
            for sender_id, intent in intent_by_sender.items()
            if bool(intent.valid) and sender_id not in usable_intent_ids
        }
        missing_intent_ids: set[int] = set()
        neighbor_ids_seen: set[int] = set()
        primary_predictions = 0
        fallback_predictions = 0
        coupled_constraints = 0

        for step_idx, pos in enumerate(positions, start=1):
            t = step_idx * dt
            for nobs in traffic.neighbors:
                neighbor_id = int(nobs.idx)
                neighbor_ids_seen.add(neighbor_id)
                intent = intent_by_sender.get(neighbor_id)
                using_intent = intent is not None and neighbor_id in usable_intent_ids
                if using_intent:
                    other = self._intent_prediction(intent, step_idx)
                    primary_predictions += 1
                else:
                    other = np.asarray(nobs.pos, dtype=np.float32) + np.asarray(nobs.vel, dtype=np.float32) * t
                    fallback_predictions += 1
                    if intent is None:
                        missing_intent_ids.add(neighbor_id)

                safe_radius = self._safe_radius_for_neighbor(
                    planner_input=planner_input,
                    nobs=nobs,
                    intent=intent if using_intent else None,
                    used_stale_fallback=bool(intent is not None and not using_intent),
                )
                p, g, clearance = self._sphere_clearance_penalty_and_grad(
                    pos,
                    other,
                    safe_radius=safe_radius,
                    collision_weight=self.collision_weight,
                    clearance_weight=self.clearance_weight,
                )
                responsibility = self._responsibility_scale(planner_input, nobs)
                p *= responsibility
                g = g * responsibility
                if clearance < 0.0:
                    extra, extra_grad = self._slack_penalty_and_grad(pos, other, clearance)
                    p += extra
                    g = g + extra_grad
                    slack_penalty += extra
                penalty += p
                grad[step_idx - 1] += g
                min_clearance = clearance if min_clearance is None else min(min_clearance, clearance)
                conflict = conflict or clearance < 0.0
                coupled_constraints += 1

            for sender_id, intent in intent_by_sender.items():
                if sender_id in neighbor_ids_seen or sender_id not in usable_intent_ids:
                    continue
                other = self._intent_prediction(intent, step_idx)
                safe_radius = float(planner_input.ego.radius) + float(intent.tube_radius_m) + self.safety_margin_m
                safe_radius += self._intent_inflation(intent)
                p, g, clearance = self._sphere_clearance_penalty_and_grad(
                    pos,
                    other,
                    safe_radius=safe_radius,
                    collision_weight=self.intent_collision_weight,
                    clearance_weight=self.intent_clearance_weight,
                )
                if clearance < 0.0:
                    extra, extra_grad = self._slack_penalty_and_grad(pos, other, clearance)
                    p += extra
                    g = g + extra_grad
                    slack_penalty += extra
                penalty += p
                grad[step_idx - 1] += g
                min_clearance = clearance if min_clearance is None else min(min_clearance, clearance)
                conflict = conflict or clearance < 0.0
                primary_predictions += 1
                coupled_constraints += 1

        self._dmpc_last_eval_stats = {
            "dmpc_best_response_seed": self._dmpc_current_seed,
            "dmpc_best_response_neighbor_intent_count_considered": int(len(usable_intent_ids)),
            "dmpc_best_response_stale_intent_count": int(len(stale_intent_ids)),
            "dmpc_best_response_missing_intent_count": int(len(missing_intent_ids)),
            "dmpc_best_response_intent_primary_predictions": int(primary_predictions),
            "dmpc_best_response_fallback_cv_predictions": int(fallback_predictions),
            "dmpc_best_response_coupled_constraints": int(coupled_constraints),
            "dmpc_best_response_pairwise_slack_penalty": float(slack_penalty),
            "dmpc_best_response_min_coupled_clearance_m": min_clearance,
            "dmpc_best_response_predicted_coupled_conflict": bool(conflict),
        }
        return float(penalty), grad, min_clearance, bool(conflict)

    def _intent_penalty_and_grad(self, planner_input: PlannerInput, positions: np.ndarray) -> tuple[float, np.ndarray]:
        penalty, grad = super()._intent_penalty_and_grad(planner_input, positions)
        return float(self.intent_coupling_weight * penalty), (self.intent_coupling_weight * grad).astype(np.float32)

    def _safe_radius_for_neighbor(
        self,
        *,
        planner_input: PlannerInput,
        nobs: NeighborObs,
        intent: IntentObs | None,
        used_stale_fallback: bool,
    ) -> float:
        safe_radius = (
            float(planner_input.ego.radius)
            + float(nobs.radius)
            + self.safety_margin_m
            + self._neighbor_inflation(nobs)
            + self._priority_inflation(planner_input, nobs)
        )
        if intent is not None:
            safe_radius += self._intent_inflation(intent)
        elif used_stale_fallback:
            safe_radius += self.stale_intent_inflation_m
        else:
            safe_radius += self.missing_intent_inflation_m
        return float(safe_radius)

    def _intent_usable_for_coupling(self, intent: IntentObs) -> bool:
        if not bool(intent.valid):
            return False
        if float(intent.intent_age_s) > self.intent_trust_horizon_s:
            return False
        points = np.asarray(intent.points, dtype=np.float32)
        return bool(points.ndim == 2 and points.shape[1] == 3 and points.shape[0] >= 1)

    def _slack_penalty_and_grad(self, point: np.ndarray, other: np.ndarray, clearance: float) -> tuple[float, np.ndarray]:
        rel = np.asarray(point, dtype=np.float32) - np.asarray(other, dtype=np.float32)
        dist = _norm(rel)
        direction = np.asarray([1.0, 0.0, 0.0], dtype=np.float32) if dist <= 1e-9 else rel / dist
        penalty = self.hard_constraint_slack_weight * float(clearance) * float(clearance)
        grad = 2.0 * self.hard_constraint_slack_weight * float(clearance) * direction
        return float(penalty), np.asarray(grad, dtype=np.float32)

    def _priority_inflation(self, planner_input: PlannerInput, nobs: NeighborObs) -> float:
        ego_priority = self._agent_priority(planner_input)
        neighbor_priority = self._neighbor_priority(nobs)
        if neighbor_priority > ego_priority:
            return self.priority_inflation_gain
        if neighbor_priority < ego_priority:
            return -0.5 * self.priority_inflation_gain
        return 0.0

    def _responsibility_scale(self, planner_input: PlannerInput, nobs: NeighborObs) -> float:
        ego_priority = self._agent_priority(planner_input)
        neighbor_priority = self._neighbor_priority(nobs)
        if neighbor_priority > ego_priority:
            return float(1.0 + self.priority_responsibility_gain)
        if neighbor_priority < ego_priority:
            return float(max(0.55, 1.0 - 0.5 * self.priority_responsibility_gain))
        return 1.0

    @staticmethod
    def _agent_priority(planner_input: PlannerInput) -> int:
        ctx = planner_input.agent_context
        return int(ctx.priority) if ctx is not None else int(planner_input.ego.idx)

    @staticmethod
    def _neighbor_priority(nobs: NeighborObs) -> int:
        return int(nobs.priority) if nobs.priority is not None else int(nobs.idx)

    def _memory(self, planner_input: PlannerInput) -> dict[str, Any]:
        ctx = planner_input.agent_context
        if ctx is not None:
            return ctx.memory
        return self._local_memory
