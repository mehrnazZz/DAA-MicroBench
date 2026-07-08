from __future__ import annotations

import numpy as np

from microbench.planners.velocity_obstacle import VelocityObstaclePlanner
from microbench.types import AABBObs, AgentState, NeighborObs, PlannerInput


def _ego(*, vel: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> AgentState:
    return AgentState(
        idx=0,
        pos=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        vel=np.asarray(vel, dtype=np.float32),
        goal=np.asarray([12.0, 0.0, 0.0], dtype=np.float32),
        radius=0.5,
        v_max=3.0,
        a_max=2.0,
    )


def _neighbor(
    *,
    pos: tuple[float, float, float] = (5.0, 0.0, 0.0),
    vel: tuple[float, float, float] = (-2.0, 0.0, 0.0),
    age: float = 0.0,
) -> NeighborObs:
    return NeighborObs(
        idx=1,
        pos=np.asarray(pos, dtype=np.float32),
        vel=np.asarray(vel, dtype=np.float32),
        radius=0.5,
        msg_age_sec=age,
        valid=True,
    )


def _input(
    *,
    ego: AgentState | None = None,
    neighbors: list[NeighborObs] | None = None,
    obstacles: list[AABBObs] | None = None,
    planar: bool = True,
) -> PlannerInput:
    return PlannerInput(
        ego=ego or _ego(),
        goal_dir=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        neighbors=list(neighbors or []),
        obstacles=list(obstacles or []),
        dt=0.02,
        t=0.0,
        planar=planar,
    )


def test_velocity_obstacle_open_space_tracks_goal() -> None:
    planner = VelocityObstaclePlanner()
    planner.reset(0)

    out = planner.compute_cmd(_input())

    assert out.v_cmd.shape == (3,)
    assert out.v_cmd[0] > 2.9
    assert abs(out.v_cmd[1]) < 1e-9
    assert abs(out.v_cmd[2]) < 1e-6
    assert out.debug_info["vo_candidates"] > 0
    assert out.debug_info["vo_conflict_count"] == 0
    assert out.debug_info["vo_planar"] is True


def test_velocity_obstacle_head_on_avoids_full_speed_cone() -> None:
    planner = VelocityObstaclePlanner()
    planner.reset(0)
    ego = _ego(vel=(2.0, 0.0, 0.0))

    out = planner.compute_cmd(_input(ego=ego, neighbors=[_neighbor()]))

    speed = float(np.linalg.norm(out.v_cmd))
    assert speed <= ego.v_max + 1e-6
    assert out.v_cmd[0] < 2.9 or abs(out.v_cmd[2]) > 0.1
    assert out.debug_info["vo_candidates"] > 0
    assert out.debug_info["vo_min_pred_clearance_m"] is not None
    assert out.debug_info["vo_min_ttc_s"] is not None


def test_velocity_obstacle_stale_track_is_more_conservative() -> None:
    fresh = VelocityObstaclePlanner()
    stale = VelocityObstaclePlanner()
    ego = _ego(vel=(2.0, 0.0, 0.0))

    fresh_out = fresh.compute_cmd(_input(ego=ego, neighbors=[_neighbor(age=0.0)]))
    stale_out = stale.compute_cmd(_input(ego=ego, neighbors=[_neighbor(age=1.0)]))

    assert stale_out.debug_info["vo_min_pred_clearance_m"] <= fresh_out.debug_info["vo_min_pred_clearance_m"]


def test_velocity_obstacle_obstacle_in_path_redirects_or_slows() -> None:
    planner = VelocityObstaclePlanner()
    planner.reset(0)
    obstacle = AABBObs(
        center=np.asarray([4.0, 0.0, 0.0], dtype=np.float32),
        half=np.asarray([0.6, 1.0, 0.6], dtype=np.float32),
    )

    out = planner.compute_cmd(_input(obstacles=[obstacle]))

    assert out.v_cmd[0] < 2.9 or abs(out.v_cmd[2]) > 0.1
    assert out.debug_info["vo_min_pred_clearance_m"] is not None
    assert out.debug_info["vo_obstacle_penalty"] >= 0.0


def test_velocity_obstacle_preserves_3d_command_shape() -> None:
    planner = VelocityObstaclePlanner()
    planner.reset(0)
    ego = _ego(vel=(1.5, 0.0, 0.0))
    neighbor = _neighbor(pos=(4.0, 0.2, 0.0), vel=(-1.5, 0.0, 0.0))

    out = planner.compute_cmd(_input(ego=ego, neighbors=[neighbor], planar=False))

    assert out.v_cmd.shape == (3,)
    assert np.all(np.isfinite(out.v_cmd))
    assert float(np.linalg.norm(out.v_cmd)) <= ego.v_max + 1e-6
    assert out.debug_info["vo_planar"] is False
    assert out.debug_info["vo_candidates"] > 0
