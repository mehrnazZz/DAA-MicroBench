from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from microbench.comm.messages import make_intent_trajectory
from microbench.planners.base import ILocalPlanner
from microbench.types import AABBObs, IntentMsg, IntentObs, NeighborObs, PlannerInput, PlannerOutput


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _normalize(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32)
    n = _norm(arr)
    if n < 1e-9:
        return np.zeros(3, dtype=np.float32)
    return (arr / n).astype(np.float32)


def _clamp_speed(v: np.ndarray, v_max: float) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32)
    n = _norm(arr)
    if n <= float(v_max) or n < 1e-9:
        return arr
    return (arr / n * float(v_max)).astype(np.float32)


def _limit_delta(v: np.ndarray, current: np.ndarray, max_delta: float) -> np.ndarray:
    target = np.asarray(v, dtype=np.float32)
    base = np.asarray(current, dtype=np.float32)
    delta = target - base
    n = _norm(delta)
    if n <= float(max_delta) or n < 1e-9:
        return target
    return (base + delta / n * float(max_delta)).astype(np.float32)


def _closest_point_on_aabb(point: np.ndarray, obs: AABBObs) -> np.ndarray:
    center = np.asarray(obs.center, dtype=np.float32)
    half = np.asarray(obs.half, dtype=np.float32)
    return np.minimum(np.maximum(np.asarray(point, dtype=np.float32), center - half), center + half)


def _perp_xz(v: np.ndarray) -> np.ndarray:
    return np.asarray([float(v[2]), 0.0, -float(v[0])], dtype=np.float32)


@dataclass(frozen=True)
class _LinearConstraint:
    step_idx: int
    source_kind: str
    source_id: int
    normal: np.ndarray
    b: float


@dataclass(frozen=True)
class _TubeState:
    nominal: np.ndarray
    deformed: np.ndarray
    active_obstacle_count: int
    max_shift_m: float
    connected: bool
    update_trigger: str


@dataclass(frozen=True)
class _QpProblem:
    hessian: np.ndarray
    linear: np.ndarray
    constraints_a: np.ndarray
    constraints_b: np.ndarray
    base_positions: np.ndarray
    base_velocities: np.ndarray
    bpos: np.ndarray
    bvel: np.ndarray
    targets: np.ndarray
    tube_state: _TubeState
    tube_constraints: list[_LinearConstraint]
    collision_constraints: list[_LinearConstraint]
    risk_agent_count: int
    first_risk_step: int | None
    accel_bound_mps2: float
    velocity_bound_mps: float


@dataclass(frozen=True)
class _ConstraintReport:
    max_violation_m: float
    sum_violation_m: float
    min_slack_m: float | None
    hard_ok: bool


@dataclass(frozen=True)
class _KinematicReport:
    max_speed_violation_mps: float
    max_accel_violation_mps2: float
    ok: bool


@dataclass(frozen=True)
class _PlanResult:
    controls: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    qp: _QpProblem
    initial_cost: float
    final_cost: float
    iterations: int
    solver_status: str
    fallback: str
    tube_report: _ConstraintReport
    collision_report: _ConstraintReport
    kinematic_report: _KinematicReport
    min_swarm_clearance_m: float | None
    min_obstacle_clearance_m: float | None


class DynamicTubeDmpcPlanner(ILocalPlanner):
    """Dynamic tube-based distributed MPC from Dai/Liao/Chen 2026.

    This planner implements the paper's benchmark-relevant algorithmic contract:
    a condensed double-integrator QP over acceleration controls, elastic virtual
    tube reconstruction, risk-triggered linearized collision constraints, local
    tube halfspace constraints, and assumed predicted trajectory broadcasts.
    It is adapted to DAA Microbench's local velocity-command interface.
    """

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.step_dt_s = float(cfg.get("step_dt_s", 0.2))
        self.horizon_steps = int(cfg.get("horizon_steps", 15))
        self.replan_period_s = float(cfg.get("replan_period_s", 0.2))
        self.max_neighbors = int(cfg.get("max_neighbors", 12))
        self.max_intent_points = int(cfg.get("max_intent_points", 16))
        self.intent_ttl_s = float(cfg.get("intent_ttl_s", 1.0))
        self.intent_tube_margin_m = float(cfg.get("intent_tube_margin_m", 0.35))
        self.coordination_message_ttl_s = float(cfg.get("coordination_message_ttl_s", 0.75))

        self.r_min_prime_m = float(cfg.get("r_min_prime_m", 0.4))
        self.localization_sigma_m = float(cfg.get("localization_sigma_m", 0.03))
        self.collision_probability_delta = float(cfg.get("collision_probability_delta", 0.0027))
        self.theta_vertical = float(cfg.get("theta_vertical", 2.0))
        self.safety_margin_m = float(cfg.get("safety_margin_m", 0.0))
        self.risk_activation_margin_m = float(cfg.get("risk_activation_margin_m", 0.0))
        self.obstacle_margin_m = float(cfg.get("obstacle_margin_m", 0.5))
        self.hard_tolerance_m = float(cfg.get("hard_tolerance_m", 0.04))

        self.tube_radius_m = float(cfg.get("tube_radius_m", 0.8))
        self.environment_margin_m = float(cfg.get("environment_margin_m", 1.0))
        self.tube_waypoints = int(cfg.get("tube_waypoints", 25))
        self.sensing_horizon_m = float(cfg.get("sensing_horizon_m", 6.0))
        self.local_subgoal_advance_m = float(cfg.get("local_subgoal_advance_m", 4.0))
        self.tube_update_period_s = float(cfg.get("tube_update_period_s", 0.4))
        self.obstacle_motion_trigger_m = float(cfg.get("obstacle_motion_trigger_m", 0.25))
        self.task_trigger_distance_m = float(cfg.get("task_trigger_distance_m", 1.5))

        self.elastic_delta_safe_m = float(cfg.get("elastic_delta_safe_m", 1.5))
        self.elastic_gamma = float(cfg.get("elastic_gamma", 0.16))
        self.elastic_zeta = float(cfg.get("elastic_zeta", 0.05))
        self.elastic_step_m = float(cfg.get("elastic_step_m", 0.22))
        self.elastic_max_shift_m = float(cfg.get("elastic_max_shift_m", 2.0))
        self.gaussian_kernel_radius = int(cfg.get("gaussian_kernel_radius", 2))
        self.gaussian_sigma = float(cfg.get("gaussian_sigma", 1.0))
        self.temporal_smoothing_alpha = float(cfg.get("temporal_smoothing_alpha", 0.65))

        self.position_weight = float(cfg.get("position_weight", 0.8))
        self.terminal_weight = float(cfg.get("terminal_weight", 5.5))
        self.control_weight = float(cfg.get("control_weight", 0.06))
        self.input_delta_weight = float(cfg.get("input_delta_weight", 0.28))
        self.qp_regularization = float(cfg.get("qp_regularization", 1e-5))
        self.qp_iterations = int(cfg.get("qp_iterations", 65))
        self.projection_iterations = int(cfg.get("projection_iterations", 5))
        self.line_search_shrink = float(cfg.get("line_search_shrink", 0.55))
        self.step_scale = float(cfg.get("step_scale", 0.92))

        self.seed = 0
        self._nominal_centerline: np.ndarray | None = None
        self._deformed_centerline: np.ndarray | None = None
        self._last_goal: np.ndarray | None = None
        self._last_controls: np.ndarray | None = None
        self._last_positions: np.ndarray | None = None
        self._last_velocities: np.ndarray | None = None
        self._last_plan_t: float | None = None
        self._last_replan_t: float | None = None
        self._last_tube_update_t: float | None = None
        self._last_obstacle_signature: np.ndarray | None = None

    def reset(self, seed: int) -> None:
        self.seed = int(seed)
        self._nominal_centerline = None
        self._deformed_centerline = None
        self._last_goal = None
        self._last_controls = None
        self._last_positions = None
        self._last_velocities = None
        self._last_plan_t = None
        self._last_replan_t = None
        self._last_tube_update_t = None
        self._last_obstacle_signature = None

    def compute_cmd(self, planner_input: PlannerInput) -> PlannerOutput:
        ego = planner_input.ego
        current = np.asarray(ego.vel, dtype=np.float32).copy()
        if planner_input.planar:
            current[1] = 0.0

        cached = self._maybe_reuse_plan(planner_input)
        replanned = cached is None
        if cached is None:
            plan = self._solve_plan(planner_input)
            self._last_replan_t = float(planner_input.t)
        else:
            plan = cached

        first_accel = plan.controls[0] if plan.controls.size else np.zeros(3, dtype=np.float32)
        v_cmd = current + np.asarray(first_accel, dtype=np.float32) * float(planner_input.dt)
        v_cmd = _limit_delta(v_cmd, current, float(ego.a_max) * float(planner_input.dt))
        v_cmd = _clamp_speed(v_cmd, float(ego.v_max))
        if planner_input.planar:
            v_cmd[1] = 0.0

        intent_points = self._intent_points(planner_input, plan.positions)
        intent = IntentMsg(
            sender_id=int(ego.idx),
            timestamp_send_s=float(planner_input.t),
            expiry_s=float(planner_input.t) + self.intent_ttl_s,
            kind="DYNAMIC_TUBE_DMPC_TRAJECTORY",
            tube_radius_m=float(ego.radius) + self.intent_tube_margin_m,
            points=intent_points.astype(float),
            dt_plan_s=float(self.step_dt_s),
            mode=f"{plan.solver_status}:{plan.fallback}",
        )
        msg = make_intent_trajectory(
            sender_id=int(ego.idx),
            recipient_id=None,
            now_s=float(planner_input.t),
            trajectory=plan.positions,
            dt_plan_s=float(self.step_dt_s),
            expiry_s=float(planner_input.t) + self.intent_ttl_s,
            tube_radius_m=float(ego.radius) + self.intent_tube_margin_m,
            ttl_s=self.coordination_message_ttl_s,
        )
        msg.payload.update(
            {
                "algorithm": "dynamic_tube_dmpc",
                "paper": "Dai_Liao_Chen_2026_dynamic_tube_dmpc",
                "risk_agent_count": int(plan.qp.risk_agent_count),
                "tube_reconstruction_active": bool(plan.qp.tube_state.active_obstacle_count > 0),
                "fallback": str(plan.fallback),
            }
        )

        self._last_controls = plan.controls.copy()
        self._last_positions = plan.positions.copy()
        self._last_velocities = plan.velocities.copy()
        self._last_plan_t = float(planner_input.t)

        debug = {
            "dynamic_tube_dmpc_algorithm": "paper_dynamic_tube_based_distributed_mpc",
            "dynamic_tube_dmpc_reference": "Dai_Liao_Chen_2026_Drones_10_177",
            "dynamic_tube_dmpc_solver": "condensed_qp_projected_gradient",
            "dynamic_tube_dmpc_solver_status": str(plan.solver_status),
            "dynamic_tube_dmpc_replanned": bool(replanned),
            "dynamic_tube_dmpc_cached_reuse": bool(not replanned),
            "dynamic_tube_dmpc_fallback": str(plan.fallback),
            "dynamic_tube_dmpc_horizon_steps": int(self.horizon_steps),
            "dynamic_tube_dmpc_step_dt_s": float(self.step_dt_s),
            "dynamic_tube_dmpc_replan_period_s": float(self.replan_period_s),
            "dynamic_tube_dmpc_iterations": int(plan.iterations),
            "dynamic_tube_dmpc_initial_cost": float(plan.initial_cost),
            "dynamic_tube_dmpc_final_cost": float(plan.final_cost),
            "dynamic_tube_dmpc_cost_reduction": float(plan.initial_cost - plan.final_cost),
            "dynamic_tube_dmpc_qp_variables": int(plan.controls.size),
            "dynamic_tube_dmpc_qp_constraint_count": int(plan.qp.constraints_a.shape[0]),
            "dynamic_tube_dmpc_velocity_constraint_count": int(2 * plan.qp.bvel.shape[0]),
            "dynamic_tube_dmpc_tube_constraint_count": int(len(plan.qp.tube_constraints)),
            "dynamic_tube_dmpc_collision_constraint_count": int(len(plan.qp.collision_constraints)),
            "dynamic_tube_dmpc_risk_agent_count": int(plan.qp.risk_agent_count),
            "dynamic_tube_dmpc_first_risk_step": plan.qp.first_risk_step,
            "dynamic_tube_dmpc_risk_triggered_activation": bool(plan.qp.risk_agent_count > 0),
            "dynamic_tube_dmpc_tube_reconstruction_active": bool(plan.qp.tube_state.active_obstacle_count > 0),
            "dynamic_tube_dmpc_tube_update_trigger": str(plan.qp.tube_state.update_trigger),
            "dynamic_tube_dmpc_tube_connected": bool(plan.qp.tube_state.connected),
            "dynamic_tube_dmpc_tube_max_shift_m": float(plan.qp.tube_state.max_shift_m),
            "dynamic_tube_dmpc_active_obstacle_count": int(plan.qp.tube_state.active_obstacle_count),
            "dynamic_tube_dmpc_tube_max_violation_m": float(plan.tube_report.max_violation_m),
            "dynamic_tube_dmpc_tube_min_slack_m": plan.tube_report.min_slack_m,
            "dynamic_tube_dmpc_tube_hard_ok": bool(plan.tube_report.hard_ok),
            "dynamic_tube_dmpc_collision_max_violation_m": float(plan.collision_report.max_violation_m),
            "dynamic_tube_dmpc_collision_min_slack_m": plan.collision_report.min_slack_m,
            "dynamic_tube_dmpc_collision_hard_ok": bool(plan.collision_report.hard_ok),
            "dynamic_tube_dmpc_kinematic_ok": bool(plan.kinematic_report.ok),
            "dynamic_tube_dmpc_max_speed_violation_mps": float(plan.kinematic_report.max_speed_violation_mps),
            "dynamic_tube_dmpc_max_accel_violation_mps2": float(plan.kinematic_report.max_accel_violation_mps2),
            "dynamic_tube_dmpc_min_swarm_clearance_m": plan.min_swarm_clearance_m,
            "dynamic_tube_dmpc_min_obstacle_clearance_m": plan.min_obstacle_clearance_m,
            "dynamic_tube_dmpc_r_min_prime_m": float(self._paper_safe_radius(planner_input)),
            "dynamic_tube_dmpc_theta_vertical": float(self.theta_vertical),
            "dynamic_tube_dmpc_planar": bool(planner_input.planar),
            "dynamic_tube_dmpc_agent_messages": 1,
            "dynamic_tube_dmpc_intent_points": int(intent_points.shape[0]),
            "dynamic_tube_dmpc_equations": "1-3,21-23,24-27,28-32,33-41",
        }
        return PlannerOutput(v_cmd=v_cmd.astype(float), intent_out=intent, messages_out=[msg], debug_info=debug)

    def _solve_plan(self, planner_input: PlannerInput) -> _PlanResult:
        qp = self._build_qp(planner_input)
        initial = self._initial_controls(planner_input)
        initial = self._project(qp, initial)
        initial_cost = self._qp_cost(qp, initial)

        controls = initial.copy()
        cost = initial_cost
        status = "projected_qp_converged"
        iterations = 0
        lipschitz = max(1e-6, float(np.linalg.norm(qp.hessian, ord=2)))
        base_step = self.step_scale / lipschitz
        for _ in range(max(0, self.qp_iterations)):
            iterations += 1
            grad = qp.hessian @ controls + qp.linear
            if _norm(grad) < 1e-8:
                status = "projected_qp_stationary"
                break
            step = base_step
            accepted = False
            for _ls in range(8):
                candidate = self._project(qp, controls - step * grad)
                candidate_cost = self._qp_cost(qp, candidate)
                if candidate_cost <= cost + 1e-7:
                    controls = candidate
                    cost = candidate_cost
                    accepted = True
                    break
                step *= max(0.1, min(0.95, self.line_search_shrink))
            if not accepted:
                status = "projected_qp_line_search_stalled"
                break

        positions, velocities = self._rollout(planner_input, controls)
        tube_report = self._constraint_report(qp.tube_constraints, positions)
        collision_report = self._constraint_report(qp.collision_constraints, positions)
        kin = self._kinematic_report(planner_input, controls, velocities)
        fallback = "none"
        if not tube_report.hard_ok or not collision_report.hard_ok or not kin.ok:
            brake = self._braking_controls(planner_input)
            brake = self._project(qp, brake)
            bpos, bvel = self._rollout(planner_input, brake)
            b_tube = self._constraint_report(qp.tube_constraints, bpos)
            b_coll = self._constraint_report(qp.collision_constraints, bpos)
            b_kin = self._kinematic_report(planner_input, brake, bvel)
            if (
                max(b_tube.max_violation_m, b_coll.max_violation_m, b_kin.max_speed_violation_mps)
                <= max(tube_report.max_violation_m, collision_report.max_violation_m, kin.max_speed_violation_mps)
                + 1e-9
            ):
                controls = brake
                positions = bpos
                velocities = bvel
                tube_report = b_tube
                collision_report = b_coll
                kin = b_kin
                fallback = "braking_trajectory"

        return _PlanResult(
            controls=np.asarray(controls, dtype=np.float32).reshape(self._steps(), 3),
            positions=np.asarray(positions, dtype=np.float32),
            velocities=np.asarray(velocities, dtype=np.float32),
            qp=qp,
            initial_cost=float(initial_cost),
            final_cost=float(self._qp_cost(qp, controls)),
            iterations=int(iterations),
            solver_status=status,
            fallback=fallback,
            tube_report=tube_report,
            collision_report=collision_report,
            kinematic_report=kin,
            min_swarm_clearance_m=self._min_swarm_clearance(planner_input, positions),
            min_obstacle_clearance_m=self._min_obstacle_clearance(planner_input, positions),
        )

    def _maybe_reuse_plan(self, planner_input: PlannerInput) -> _PlanResult | None:
        if self.replan_period_s <= 0.0:
            return None
        if (
            self._last_controls is None
            or self._last_positions is None
            or self._last_velocities is None
            or self._last_plan_t is None
            or self._last_replan_t is None
        ):
            return None
        if float(planner_input.t) - float(self._last_replan_t) >= self.replan_period_s:
            return None

        controls = self._shift_controls()
        qp = self._build_qp(planner_input, force_tube_reuse=True)
        controls = self._project(qp, controls)
        positions, velocities = self._rollout(planner_input, controls)
        tube_report = self._constraint_report(qp.tube_constraints, positions)
        collision_report = self._constraint_report(qp.collision_constraints, positions)
        kin = self._kinematic_report(planner_input, controls, velocities)
        if not tube_report.hard_ok or not collision_report.hard_ok:
            return None
        return _PlanResult(
            controls=controls.astype(np.float32).reshape(self._steps(), 3),
            positions=positions.astype(np.float32),
            velocities=velocities.astype(np.float32),
            qp=qp,
            initial_cost=float(self._qp_cost(qp, controls)),
            final_cost=float(self._qp_cost(qp, controls)),
            iterations=0,
            solver_status="cached_receding_qp_solution",
            fallback="none",
            tube_report=tube_report,
            collision_report=collision_report,
            kinematic_report=kin,
            min_swarm_clearance_m=self._min_swarm_clearance(planner_input, positions),
            min_obstacle_clearance_m=self._min_obstacle_clearance(planner_input, positions),
        )

    def _build_qp(self, planner_input: PlannerInput, *, force_tube_reuse: bool = False) -> _QpProblem:
        k_steps = self._steps()
        dim = 3 * k_steps
        base_pos, bpos, base_vel, bvel = self._prediction_matrices(planner_input)
        tube_state = self._tube_state(planner_input, force_reuse=force_tube_reuse)
        targets = self._subtargets(planner_input, tube_state.deformed)

        w = np.full(k_steps, self.position_weight, dtype=np.float64)
        w[-1] = self.terminal_weight
        wpos = np.repeat(w, 3)
        target_vec = targets.reshape(dim).astype(np.float64)
        base_vec = base_pos.reshape(dim).astype(np.float64)
        hessian = 2.0 * (bpos.T @ (wpos[:, None] * bpos))
        linear = 2.0 * (bpos.T @ (wpos * (base_vec - target_vec)))

        hessian += 2.0 * self.control_weight * np.eye(dim, dtype=np.float64)
        rmat, uref = self._input_delta_matrix(k_steps)
        hessian += 2.0 * self.input_delta_weight * (rmat.T @ rmat)
        linear += -2.0 * self.input_delta_weight * (rmat.T @ uref)
        hessian += self.qp_regularization * np.eye(dim, dtype=np.float64)

        constraint_rows: list[np.ndarray] = []
        constraint_bounds: list[float] = []
        velocity_rows, velocity_bounds = self._velocity_constraints(planner_input, base_vel, bvel)
        constraint_rows.extend(velocity_rows)
        constraint_bounds.extend(velocity_bounds)

        tube_constraints = self._tube_constraints(planner_input, tube_state.deformed, targets)
        for c in tube_constraints:
            row_slice = slice(3 * (c.step_idx - 1), 3 * c.step_idx)
            constraint_rows.append(np.asarray(c.normal, dtype=np.float64) @ bpos[row_slice, :])
            constraint_bounds.append(float(c.b - np.dot(c.normal, base_pos[c.step_idx - 1])))

        collision_constraints, risk_count, first_risk = self._collision_constraints(planner_input, bpos, base_pos, targets)
        for c in collision_constraints:
            row_slice = slice(3 * (c.step_idx - 1), 3 * c.step_idx)
            constraint_rows.append(np.asarray(c.normal, dtype=np.float64) @ bpos[row_slice, :])
            constraint_bounds.append(float(c.b - np.dot(c.normal, base_pos[c.step_idx - 1])))

        if constraint_rows:
            amat = np.vstack(constraint_rows).astype(np.float64)
            bvec = np.asarray(constraint_bounds, dtype=np.float64)
        else:
            amat = np.zeros((0, dim), dtype=np.float64)
            bvec = np.zeros(0, dtype=np.float64)

        return _QpProblem(
            hessian=hessian.astype(np.float64),
            linear=linear.astype(np.float64),
            constraints_a=amat,
            constraints_b=bvec,
            base_positions=base_pos.astype(np.float32),
            base_velocities=base_vel.astype(np.float32),
            bpos=bpos.astype(np.float64),
            bvel=bvel.astype(np.float64),
            targets=targets.astype(np.float32),
            tube_state=tube_state,
            tube_constraints=tube_constraints,
            collision_constraints=collision_constraints,
            risk_agent_count=int(risk_count),
            first_risk_step=first_risk,
            accel_bound_mps2=float(planner_input.ego.a_max),
            velocity_bound_mps=float(planner_input.ego.v_max),
        )

    def _prediction_matrices(self, planner_input: PlannerInput) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        k_steps = self._steps()
        h = float(self.step_dt_s)
        p0 = np.asarray(planner_input.ego.pos, dtype=np.float64)
        v0 = np.asarray(planner_input.ego.vel, dtype=np.float64)
        if planner_input.planar:
            v0[1] = 0.0
        base_pos = np.zeros((k_steps, 3), dtype=np.float64)
        base_vel = np.zeros((k_steps, 3), dtype=np.float64)
        bpos = np.zeros((3 * k_steps, 3 * k_steps), dtype=np.float64)
        bvel = np.zeros((3 * k_steps, 3 * k_steps), dtype=np.float64)
        for k in range(1, k_steps + 1):
            base_pos[k - 1] = p0 + float(k) * h * v0
            base_vel[k - 1] = v0
            row = slice(3 * (k - 1), 3 * k)
            for m in range(k):
                col = slice(3 * m, 3 * (m + 1))
                bpos[row, col] = np.eye(3) * (h * h * (float(k - m) - 0.5))
                bvel[row, col] = np.eye(3) * h
        return base_pos, bpos, base_vel, bvel

    def _input_delta_matrix(self, k_steps: int) -> tuple[np.ndarray, np.ndarray]:
        dim = 3 * k_steps
        rmat = np.eye(dim, dtype=np.float64)
        for k in range(1, k_steps):
            rmat[3 * k : 3 * (k + 1), 3 * (k - 1) : 3 * k] = -np.eye(3)
        uref = np.zeros(dim, dtype=np.float64)
        if self._last_controls is not None and self._last_controls.size >= 3:
            uref[:3] = np.asarray(self._last_controls[0], dtype=np.float64)
        return rmat, uref

    def _velocity_constraints(
        self,
        planner_input: PlannerInput,
        base_vel: np.ndarray,
        bvel: np.ndarray,
    ) -> tuple[list[np.ndarray], list[float]]:
        rows: list[np.ndarray] = []
        bounds: list[float] = []
        vmax = float(planner_input.ego.v_max)
        for k in range(self._steps()):
            for axis in range(3):
                if planner_input.planar and axis == 1:
                    continue
                idx = 3 * k + axis
                row = bvel[idx, :].copy()
                rows.append(row)
                bounds.append(float(vmax - base_vel[k, axis]))
                rows.append(-row)
                bounds.append(float(vmax + base_vel[k, axis]))
        return rows, bounds

    def _tube_state(self, planner_input: PlannerInput, *, force_reuse: bool = False) -> _TubeState:
        nominal = self._nominal_tube_centerline(planner_input)
        trigger = self._tube_update_trigger(planner_input)
        if force_reuse and self._deformed_centerline is not None:
            deformed = self._deformed_centerline.copy()
            return _TubeState(nominal=nominal, deformed=deformed, active_obstacle_count=0, max_shift_m=float(np.max(np.linalg.norm(deformed - nominal, axis=1))), connected=True, update_trigger="cached")
        if trigger == "reuse" and self._deformed_centerline is not None:
            deformed = self._deformed_centerline.copy()
            return _TubeState(nominal=nominal, deformed=deformed, active_obstacle_count=0, max_shift_m=float(np.max(np.linalg.norm(deformed - nominal, axis=1))), connected=True, update_trigger="reuse")

        deformed, active = self._elastic_reconstruct(planner_input, nominal)
        if self._deformed_centerline is not None and self._deformed_centerline.shape == deformed.shape:
            alpha = min(1.0, max(0.0, self.temporal_smoothing_alpha))
            deformed = alpha * deformed + (1.0 - alpha) * self._deformed_centerline
        deformed[0] = nominal[0]
        deformed[-1] = nominal[-1]
        if planner_input.planar:
            deformed[:, 1] = float(planner_input.ego.pos[1])
        self._deformed_centerline = deformed.astype(np.float32)
        self._last_tube_update_t = float(planner_input.t)
        self._last_obstacle_signature = self._obstacle_signature(planner_input)
        shift = np.linalg.norm(self._deformed_centerline - nominal, axis=1)
        return _TubeState(
            nominal=nominal,
            deformed=self._deformed_centerline.copy(),
            active_obstacle_count=int(active),
            max_shift_m=float(np.max(shift)) if shift.size else 0.0,
            connected=bool(self._tube_connected(nominal, self._deformed_centerline)),
            update_trigger=trigger,
        )

    def _nominal_tube_centerline(self, planner_input: PlannerInput) -> np.ndarray:
        ego = planner_input.ego
        goal = np.asarray(ego.goal, dtype=np.float32).copy()
        if planner_input.planar:
            goal[1] = float(ego.pos[1])
        needs_reset = self._nominal_centerline is None or self._last_goal is None or _norm(goal - self._last_goal) > 1e-3
        if needs_reset:
            start = np.asarray(ego.pos, dtype=np.float32).copy()
            if planner_input.planar:
                start[1] = float(ego.pos[1])
            n = max(3, self.tube_waypoints)
            tau = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
            self._nominal_centerline = (1.0 - tau) * start[None, :] + tau * goal[None, :]
            self._last_goal = goal.copy()
        return np.asarray(self._nominal_centerline, dtype=np.float32)

    def _tube_update_trigger(self, planner_input: PlannerInput) -> str:
        if self._deformed_centerline is None or self._last_tube_update_t is None:
            return "initial"
        if float(planner_input.t) - float(self._last_tube_update_t) >= self.tube_update_period_s:
            return "time"
        sig = self._obstacle_signature(planner_input)
        if sig.size and self._last_obstacle_signature is not None and self._last_obstacle_signature.shape == sig.shape:
            if float(np.max(np.linalg.norm(sig - self._last_obstacle_signature, axis=1))) >= self.obstacle_motion_trigger_m:
                return "obstacle_motion"
        closest_idx = self._closest_centerline_index(np.asarray(planner_input.ego.pos, dtype=np.float32), self._deformed_centerline)
        if closest_idx >= self._deformed_centerline.shape[0] - 3:
            return "task"
        goal_dist = _norm(np.asarray(planner_input.ego.goal, dtype=np.float32) - np.asarray(planner_input.ego.pos, dtype=np.float32))
        if goal_dist < self.task_trigger_distance_m:
            return "task"
        return "reuse"

    def _elastic_reconstruct(self, planner_input: PlannerInput, nominal: np.ndarray) -> tuple[np.ndarray, int]:
        deformed = np.asarray(nominal, dtype=np.float32).copy()
        active = 0
        for i in range(1, deformed.shape[0] - 1):
            shift = np.zeros(3, dtype=np.float32)
            point = deformed[i]
            for obs in planner_input.obstacles:
                closest = _closest_point_on_aabb(point, obs)
                diff = point - closest
                dist = _norm(diff)
                if dist <= self.elastic_delta_safe_m:
                    direction = _normalize(diff if dist > 1e-6 else point - np.asarray(obs.center, dtype=np.float32))
                    norm_d = max(0.0, dist / max(1e-6, self.elastic_delta_safe_m))
                    mag = self.elastic_gamma / ((norm_d + self.elastic_zeta) ** 2)
                    shift += direction * min(self.elastic_step_m, mag)
                    active += 1
            if _norm(shift) > self.elastic_max_shift_m:
                shift = _normalize(shift) * self.elastic_max_shift_m
            deformed[i] = point + shift
        deformed = self._gaussian_smooth(deformed)
        return deformed.astype(np.float32), active

    def _gaussian_smooth(self, points: np.ndarray) -> np.ndarray:
        radius = max(0, int(self.gaussian_kernel_radius))
        if radius <= 0 or points.shape[0] <= 2:
            return points
        offsets = np.arange(-radius, radius + 1, dtype=np.float32)
        weights = np.exp(-0.5 * (offsets / max(1e-6, self.gaussian_sigma)) ** 2)
        out = points.copy()
        for i in range(1, points.shape[0] - 1):
            acc = np.zeros(3, dtype=np.float32)
            total = 0.0
            for off, w in zip(offsets.astype(int), weights):
                j = min(points.shape[0] - 1, max(0, i + int(off)))
                acc += points[j] * float(w)
                total += float(w)
            out[i] = acc / max(1e-6, total)
        out[0] = points[0]
        out[-1] = points[-1]
        return out.astype(np.float32)

    def _tube_connected(self, nominal: np.ndarray, deformed: np.ndarray) -> bool:
        if nominal.shape[0] < 2:
            return True
        nominal_step = max(1e-6, float(np.max(np.linalg.norm(np.diff(nominal, axis=0), axis=1))))
        deformed_step = float(np.max(np.linalg.norm(np.diff(deformed, axis=0), axis=1)))
        return bool(deformed_step <= max(3.0 * nominal_step, nominal_step + 2.0 * self.tube_radius_m))

    def _obstacle_signature(self, planner_input: PlannerInput) -> np.ndarray:
        if not planner_input.obstacles:
            return np.zeros((0, 3), dtype=np.float32)
        return np.vstack([np.asarray(obs.center, dtype=np.float32) for obs in planner_input.obstacles]).astype(np.float32)

    def _subtargets(self, planner_input: PlannerInput, centerline: np.ndarray) -> np.ndarray:
        k_steps = self._steps()
        pos = np.asarray(planner_input.ego.pos, dtype=np.float32)
        cumulative = self._centerline_cumulative(centerline)
        closest_idx = self._closest_centerline_index(pos, centerline)
        start_s = float(cumulative[closest_idx])
        targets = np.zeros((k_steps, 3), dtype=np.float32)
        for k in range(1, k_steps + 1):
            advance = min(self.local_subgoal_advance_m, float(planner_input.ego.v_max) * self.step_dt_s * k)
            targets[k - 1] = self._sample_centerline(centerline, cumulative, start_s + advance)
        if planner_input.planar:
            targets[:, 1] = float(planner_input.ego.pos[1])
        return targets.astype(np.float32)

    def _tube_constraints(
        self,
        planner_input: PlannerInput,
        centerline: np.ndarray,
        targets: np.ndarray,
    ) -> list[_LinearConstraint]:
        constraints: list[_LinearConstraint] = []
        cumulative = self._centerline_cumulative(centerline)
        pos = np.asarray(planner_input.ego.pos, dtype=np.float32)
        closest_idx = self._closest_centerline_index(pos, centerline)
        s0 = float(cumulative[closest_idx])
        smax = min(float(cumulative[-1]), s0 + self.sensing_horizon_m)
        p_start = self._sample_centerline(centerline, cumulative, s0)
        p_end = self._sample_centerline(centerline, cumulative, smax)
        cap_tangent = _normalize(p_end - p_start)
        if _norm(cap_tangent) < 1e-6:
            cap_tangent = _normalize(np.asarray(planner_input.goal_dir, dtype=np.float32))
        radius = self._effective_tube_radius(planner_input)
        for k, center in enumerate(targets, start=1):
            tangent = self._local_tangent(centerline, center)
            lateral = _normalize(_perp_xz(tangent))
            if _norm(lateral) < 1e-6:
                lateral = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
            axes = [lateral]
            if not planner_input.planar:
                axes.append(np.asarray([0.0, 1.0, 0.0], dtype=np.float32))
            for axis_id, axis in enumerate(axes):
                constraints.append(
                    _LinearConstraint(
                        step_idx=k,
                        source_kind="tube_boundary",
                        source_id=axis_id,
                        normal=axis.astype(np.float32),
                        b=float(np.dot(axis, center) + radius),
                    )
                )
                constraints.append(
                    _LinearConstraint(
                        step_idx=k,
                        source_kind="tube_boundary",
                        source_id=axis_id,
                        normal=(-axis).astype(np.float32),
                        b=float(np.dot(-axis, center) + radius),
                    )
                )
            if _norm(cap_tangent) > 1e-6:
                constraints.append(
                    _LinearConstraint(
                        step_idx=k,
                        source_kind="tube_cap",
                        source_id=1,
                        normal=cap_tangent.astype(np.float32),
                        b=float(np.dot(cap_tangent, p_end) + radius),
                    )
                )
                constraints.append(
                    _LinearConstraint(
                        step_idx=k,
                        source_kind="tube_cap",
                        source_id=0,
                        normal=(-cap_tangent).astype(np.float32),
                        b=float(np.dot(-cap_tangent, p_start) + radius),
                    )
                )
        return constraints

    def _collision_constraints(
        self,
        planner_input: PlannerInput,
        bpos: np.ndarray,
        base_pos: np.ndarray,
        assumed_ego: np.ndarray,
    ) -> tuple[list[_LinearConstraint], int, int | None]:
        del bpos, base_pos
        constraints: list[_LinearConstraint] = []
        risk_agents = 0
        first_risk: int | None = None
        intent_by_sender = {
            int(intent.sender_id): intent
            for intent in planner_input.neighbor_intents
            if bool(intent.valid) and np.asarray(intent.points).size > 0
        }
        seen: set[int] = set()
        for nobs in planner_input.neighbors[: self.max_neighbors]:
            seen.add(int(nobs.idx))
            pred = self._neighbor_prediction_sequence(nobs, intent_by_sender.get(int(nobs.idx)))
            added, risk_step = self._constraints_against_prediction(
                planner_input,
                assumed_ego,
                pred,
                source_kind="risk_neighbor",
                source_id=int(nobs.idx),
                neighbor_radius=float(nobs.radius),
            )
            if added:
                risk_agents += 1
                first_risk = risk_step if first_risk is None else min(first_risk, risk_step)
                constraints.extend(added)
        for sender_id, intent in intent_by_sender.items():
            if sender_id in seen:
                continue
            pred = self._intent_prediction_sequence(intent)
            added, risk_step = self._constraints_against_prediction(
                planner_input,
                assumed_ego,
                pred,
                source_kind="risk_intent",
                source_id=int(sender_id),
                neighbor_radius=float(intent.tube_radius_m),
            )
            if added:
                risk_agents += 1
                first_risk = risk_step if first_risk is None else min(first_risk, risk_step)
                constraints.extend(added)
        return constraints, risk_agents, first_risk

    def _constraints_against_prediction(
        self,
        planner_input: PlannerInput,
        assumed_ego: np.ndarray,
        other_pred: np.ndarray,
        *,
        source_kind: str,
        source_id: int,
        neighbor_radius: float,
    ) -> tuple[list[_LinearConstraint], int | None]:
        safe = self._safe_radius(planner_input, neighbor_radius)
        scaled = [self._scaled_distance(assumed_ego[k] - other_pred[k]) for k in range(self._steps())]
        threshold = safe + self.risk_activation_margin_m
        risk_steps = [idx + 1 for idx, value in enumerate(scaled) if value <= threshold]
        if not risk_steps:
            return [], None
        risk_step = min(risk_steps)
        out: list[_LinearConstraint] = []
        theta_inv2 = self._theta_inv2(planner_input)
        theta_inv = self._theta_inv(planner_input)
        for k in range(risk_step, self._steps() + 1):
            ego_anchor = np.asarray(assumed_ego[k - 1], dtype=np.float32)
            other = np.asarray(other_pred[k - 1], dtype=np.float32)
            rel = other - ego_anchor
            if planner_input.planar:
                rel[1] = 0.0
            if _norm(rel) < 1e-6:
                continue
            normal = theta_inv2 @ rel
            if _norm(normal) < 1e-9:
                continue
            midpoint = 0.5 * (ego_anchor + other)
            scaled_sep = _norm(theta_inv @ rel)
            b = float(np.dot(normal, midpoint) - 0.5 * safe * scaled_sep)
            out.append(
                _LinearConstraint(
                    step_idx=k,
                    source_kind=source_kind,
                    source_id=int(source_id),
                    normal=normal.astype(np.float32),
                    b=b,
                )
            )
        return out, risk_step

    def _neighbor_prediction_sequence(self, nobs: NeighborObs, intent: IntentObs | None) -> np.ndarray:
        if intent is not None and bool(intent.valid) and np.asarray(intent.points).size > 0:
            return self._intent_prediction_sequence(intent)
        out = np.zeros((self._steps(), 3), dtype=np.float32)
        p = np.asarray(nobs.pos, dtype=np.float32)
        v = np.asarray(nobs.vel, dtype=np.float32)
        for k in range(1, self._steps() + 1):
            out[k - 1] = p + v * (float(k) * self.step_dt_s)
        return out

    def _intent_prediction_sequence(self, intent: IntentObs) -> np.ndarray:
        out = np.zeros((self._steps(), 3), dtype=np.float32)
        for k in range(1, self._steps() + 1):
            out[k - 1] = self._intent_prediction(intent, float(k) * self.step_dt_s)
        return out

    def _intent_prediction(self, intent: IntentObs, t: float) -> np.ndarray:
        points = np.asarray(intent.points, dtype=np.float32)
        if points.ndim != 2 or points.shape[0] == 0:
            return np.zeros(3, dtype=np.float32)
        if points.shape[0] == 1:
            return points[0].astype(np.float32)
        dt = float(intent.dt_plan_s) if intent.dt_plan_s is not None and intent.dt_plan_s > 1e-9 else self.step_dt_s
        tau = max(0.0, t) / max(1e-6, dt)
        lo = min(points.shape[0] - 1, max(0, int(math.floor(tau))))
        hi = min(points.shape[0] - 1, lo + 1)
        alpha = min(1.0, max(0.0, tau - lo))
        return ((1.0 - alpha) * points[lo] + alpha * points[hi]).astype(np.float32)

    def _project(self, qp: _QpProblem, controls: np.ndarray) -> np.ndarray:
        u = np.asarray(controls, dtype=np.float64).reshape(-1).copy()
        amax = float(qp.accel_bound_mps2)
        for _ in range(max(1, self.projection_iterations)):
            u = self._project_accel_balls(u, amax)
            u = self._project_velocity_balls(qp, u)
            for row, bound in zip(qp.constraints_a, qp.constraints_b):
                violation = float(np.dot(row, u) - bound)
                if violation > 0.0:
                    denom = float(np.dot(row, row)) + 1e-9
                    u -= row * (violation / denom)
            u = self._project_accel_balls(u, amax)
            u = self._project_velocity_balls(qp, u)
        return u.astype(np.float64)

    def _project_accel_balls(self, controls: np.ndarray, amax: float) -> np.ndarray:
        u = np.asarray(controls, dtype=np.float64).reshape(self._steps(), 3).copy()
        for k in range(u.shape[0]):
            n = float(np.linalg.norm(u[k]))
            if n > float(amax) and n > 1e-12:
                u[k] = u[k] / n * float(amax)
        return u.reshape(-1)

    def _project_velocity_balls(self, qp: _QpProblem, controls: np.ndarray) -> np.ndarray:
        u = np.asarray(controls, dtype=np.float64).reshape(-1).copy()
        vmax = float(qp.velocity_bound_mps)
        for k in range(self._steps()):
            row_slice = slice(3 * k, 3 * (k + 1))
            vel = np.asarray(qp.base_velocities[k], dtype=np.float64) + qp.bvel[row_slice, :] @ u
            n = float(np.linalg.norm(vel))
            if n > vmax and n > 1e-12:
                direction = vel / n
                row = direction @ qp.bvel[row_slice, :]
                bound = vmax - float(np.dot(direction, qp.base_velocities[k]))
                violation = float(np.dot(row, u) - bound)
                if violation > 0.0:
                    denom = float(np.dot(row, row)) + 1e-9
                    u -= row * (violation / denom)
        return u

    def _qp_cost(self, qp: _QpProblem, controls: np.ndarray) -> float:
        u = np.asarray(controls, dtype=np.float64).reshape(-1)
        return float(0.5 * np.dot(u, qp.hessian @ u) + np.dot(qp.linear, u))

    def _initial_controls(self, planner_input: PlannerInput) -> np.ndarray:
        if self._last_controls is not None and self._last_controls.shape[0] == self._steps():
            return self._shift_controls().reshape(-1).astype(np.float64)
        k_steps = self._steps()
        controls = np.zeros((k_steps, 3), dtype=np.float32)
        current = np.asarray(planner_input.ego.vel, dtype=np.float32)
        desired = _normalize(np.asarray(planner_input.goal_dir, dtype=np.float32)) * float(planner_input.ego.v_max)
        if planner_input.planar:
            desired[1] = 0.0
            current[1] = 0.0
        accel = (desired - current) / max(1e-6, self.step_dt_s)
        n = _norm(accel)
        if n > float(planner_input.ego.a_max):
            accel = accel / n * float(planner_input.ego.a_max)
        controls[0] = accel.astype(np.float32)
        return controls.reshape(-1).astype(np.float64)

    def _shift_controls(self) -> np.ndarray:
        if self._last_controls is None:
            return np.zeros((self._steps(), 3), dtype=np.float64)
        old = np.asarray(self._last_controls, dtype=np.float64)
        out = np.zeros((self._steps(), 3), dtype=np.float64)
        keep = min(max(0, old.shape[0] - 1), self._steps() - 1)
        if keep > 0:
            out[:keep] = old[1 : 1 + keep]
        return out.reshape(-1)

    def _braking_controls(self, planner_input: PlannerInput) -> np.ndarray:
        controls = np.zeros((self._steps(), 3), dtype=np.float32)
        vel = np.asarray(planner_input.ego.vel, dtype=np.float32).copy()
        if planner_input.planar:
            vel[1] = 0.0
        for k in range(self._steps()):
            speed = _norm(vel)
            if speed > 1e-9:
                accel = -vel / speed * float(planner_input.ego.a_max)
            else:
                accel = np.zeros(3, dtype=np.float32)
            controls[k] = accel
            vel = vel + accel * self.step_dt_s
        return controls.reshape(-1).astype(np.float64)

    def _rollout(self, planner_input: PlannerInput, controls: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u = np.asarray(controls, dtype=np.float32).reshape(self._steps(), 3)
        p = np.asarray(planner_input.ego.pos, dtype=np.float32).copy()
        v = np.asarray(planner_input.ego.vel, dtype=np.float32).copy()
        if planner_input.planar:
            v[1] = 0.0
        positions = np.zeros((self._steps(), 3), dtype=np.float32)
        velocities = np.zeros((self._steps(), 3), dtype=np.float32)
        h = float(self.step_dt_s)
        for k in range(self._steps()):
            a = u[k]
            p = p + h * v + 0.5 * h * h * a
            v = v + h * a
            if planner_input.planar:
                p[1] = float(planner_input.ego.pos[1])
                v[1] = 0.0
            positions[k] = p
            velocities[k] = v
        return positions, velocities

    def _constraint_report(self, constraints: list[_LinearConstraint], positions: np.ndarray) -> _ConstraintReport:
        max_v = 0.0
        sum_v = 0.0
        min_slack: float | None = None
        for c in constraints:
            value = float(np.dot(c.normal, positions[c.step_idx - 1]) - c.b)
            violation = max(0.0, value)
            slack = -value
            max_v = max(max_v, violation)
            sum_v += violation
            min_slack = slack if min_slack is None else min(min_slack, slack)
        return _ConstraintReport(
            max_violation_m=float(max_v),
            sum_violation_m=float(sum_v),
            min_slack_m=None if min_slack is None else float(min_slack),
            hard_ok=bool(max_v <= self.hard_tolerance_m),
        )

    def _kinematic_report(self, planner_input: PlannerInput, controls: np.ndarray, velocities: np.ndarray) -> _KinematicReport:
        u = np.asarray(controls, dtype=np.float32).reshape(self._steps(), 3)
        max_acc = max(0.0, max((_norm(a) - float(planner_input.ego.a_max) for a in u), default=0.0))
        max_speed = max(0.0, max((_norm(v) - float(planner_input.ego.v_max) for v in velocities), default=0.0))
        return _KinematicReport(
            max_speed_violation_mps=float(max_speed),
            max_accel_violation_mps2=float(max_acc),
            ok=bool(max(max_speed, max_acc) <= 1e-5),
        )

    def _min_swarm_clearance(self, planner_input: PlannerInput, positions: np.ndarray) -> float | None:
        values: list[float] = []
        for nobs in planner_input.neighbors[: self.max_neighbors]:
            pred = self._neighbor_prediction_sequence(nobs, None)
            safe = self._safe_radius(planner_input, float(nobs.radius))
            for p, q in zip(positions, pred):
                values.append(self._scaled_distance(p - q) - safe)
        return min(values) if values else None

    def _min_obstacle_clearance(self, planner_input: PlannerInput, positions: np.ndarray) -> float | None:
        values: list[float] = []
        for obs in planner_input.obstacles:
            for p in positions:
                closest = _closest_point_on_aabb(p, obs)
                values.append(_norm(p - closest) - self.obstacle_margin_m)
        return min(values) if values else None

    def _safe_radius(self, planner_input: PlannerInput, other_radius: float) -> float:
        ego = planner_input.ego
        return float(max(self._paper_safe_radius(planner_input), float(ego.radius) + float(other_radius) + self.safety_margin_m))

    def _paper_safe_radius(self, planner_input: PlannerInput) -> float:
        del planner_input
        # The paper uses r'_min = rmin + 2 epsilon. The default is the
        # simulation/hardware value reported in Table 1: 0.4 m.
        return float(self.r_min_prime_m)

    def _effective_tube_radius(self, planner_input: PlannerInput) -> float:
        r_prime = self._paper_safe_radius(planner_input)
        return float(max(self.environment_margin_m - 0.5 * r_prime, 0.5 * r_prime, self.tube_radius_m))

    def _theta_inv(self, planner_input: PlannerInput) -> np.ndarray:
        diag = np.asarray([1.0, self.theta_vertical, 1.0], dtype=np.float32)
        if planner_input.planar:
            diag[1] = 1.0
        return np.diag(1.0 / diag).astype(np.float32)

    def _theta_inv2(self, planner_input: PlannerInput) -> np.ndarray:
        theta_inv = self._theta_inv(planner_input)
        return (theta_inv @ theta_inv).astype(np.float32)

    def _scaled_distance(self, delta: np.ndarray) -> float:
        arr = np.asarray(delta, dtype=np.float32).copy()
        arr[1] = arr[1] / max(1e-6, self.theta_vertical)
        return _norm(arr)

    def _steps(self) -> int:
        return max(2, int(self.horizon_steps))

    def _centerline_cumulative(self, centerline: np.ndarray) -> np.ndarray:
        if centerline.shape[0] == 0:
            return np.zeros(0, dtype=np.float32)
        diffs = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
        return np.concatenate([[0.0], np.cumsum(diffs)]).astype(np.float32)

    def _closest_centerline_index(self, point: np.ndarray, centerline: np.ndarray) -> int:
        if centerline.shape[0] == 0:
            return 0
        dist = np.linalg.norm(centerline - np.asarray(point, dtype=np.float32)[None, :], axis=1)
        return int(np.argmin(dist))

    def _sample_centerline(self, centerline: np.ndarray, cumulative: np.ndarray, s: float) -> np.ndarray:
        if centerline.shape[0] == 0:
            return np.zeros(3, dtype=np.float32)
        if centerline.shape[0] == 1 or s <= float(cumulative[0]):
            return centerline[0].astype(np.float32)
        if s >= float(cumulative[-1]):
            return centerline[-1].astype(np.float32)
        hi = int(np.searchsorted(cumulative, s, side="right"))
        lo = max(0, hi - 1)
        hi = min(centerline.shape[0] - 1, hi)
        alpha = (float(s) - float(cumulative[lo])) / max(1e-6, float(cumulative[hi] - cumulative[lo]))
        return ((1.0 - alpha) * centerline[lo] + alpha * centerline[hi]).astype(np.float32)

    def _local_tangent(self, centerline: np.ndarray, point: np.ndarray) -> np.ndarray:
        idx = self._closest_centerline_index(point, centerline)
        lo = max(0, idx - 1)
        hi = min(centerline.shape[0] - 1, idx + 1)
        tangent = centerline[hi] - centerline[lo]
        return _normalize(tangent)

    def _intent_points(self, planner_input: PlannerInput, points: np.ndarray) -> np.ndarray:
        ego_pos = np.asarray(planner_input.ego.pos, dtype=np.float32)
        out = np.asarray(points, dtype=np.float32).copy() if points.size else ego_pos.reshape(1, 3)
        if out.shape[0] == 0 or _norm(out[0] - ego_pos) > 1e-5:
            out = np.vstack([ego_pos, out])
        if self.max_intent_points > 0 and out.shape[0] > self.max_intent_points:
            idx = np.linspace(0, out.shape[0] - 1, self.max_intent_points).round().astype(int)
            out = out[idx]
        return out.astype(np.float32)
