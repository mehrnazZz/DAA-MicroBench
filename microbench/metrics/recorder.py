from __future__ import annotations

from pathlib import Path
import json
from typing import Any

from microbench.metrics.ring_buffer import EpisodeRingBuffer


class FailureRecorder:
    def __init__(
        self,
        out_dir: str,
        scenario: str,
        method: str,
        n_agents: int,
        seed: int,
        comm_profile: str | None,
        save_trace: bool,
        trace_max_steps: int,
        trace_save_failures_only: bool,
        save_events: bool,
        save_trace_on_collision: bool,
        trace_agents_mode: str,
    ):
        self.save_events = bool(save_events)
        self.save_trace = bool(save_trace)
        self.trace_save_failures_only = bool(trace_save_failures_only)
        self.save_trace_on_collision = bool(save_trace_on_collision)
        self.trace_agents_mode = trace_agents_mode
        self.episode_dir = Path(out_dir) / "episodes" / episode_dir_name(
            scenario=scenario,
            method=method,
            n_agents=n_agents,
            seed=seed,
            comm_profile=comm_profile,
        )
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.episode_dir / "events.jsonl"
        self.episode_trace_path = self.episode_dir / "trace_episode.jsonl"
        self._event_fh = None
        self._dumped_pairs: set[tuple[int, int]] = set()
        self._had_failure = False
        self._full_trace = EpisodeRingBuffer(max_frames=max(1, int(trace_max_steps)))
        self._episode_meta: dict[str, Any] = {}

    def set_episode_meta(self, meta: dict[str, Any]) -> None:
        self._episode_meta = dict(meta)

    def push_episode_frame(self, frame: dict[str, Any]) -> None:
        if not self.save_trace:
            return
        self._full_trace.push(frame)

    def mark_failure(self) -> None:
        self._had_failure = True

    def _write_event(self, payload: dict[str, Any]) -> None:
        if not self.save_events:
            return
        if self._event_fh is None:
            self._event_fh = self.events_path.open("w", encoding="utf-8")
        self._event_fh.write(json.dumps(payload) + "\n")

    def record_proximity_event(self, event: dict[str, Any]) -> None:
        self.mark_failure()
        self._write_event(event)

    def maybe_dump_collision_trace(
        self,
        pair: tuple[int, int],
        t: float,
        ring_snapshot: list[dict[str, Any]],
        collision_meta: dict[str, Any],
    ) -> Path | None:
        if not self.save_trace_on_collision:
            return None
        key = tuple(sorted(pair))
        if key in self._dumped_pairs:
            return None
        self._dumped_pairs.add(key)

        i, j = key
        trace_path = self.episode_dir / f"trace_collision_{i}_{j}_t{t:.2f}.jsonl"
        ids = self._select_agent_ids(key, ring_snapshot)

        with trace_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"kind": "meta", **self._episode_meta, **collision_meta, "agent_ids": ids}) + "\n")
            for frame in ring_snapshot:
                trimmed = self._trim_frame(frame, ids)
                f.write(json.dumps({"kind": "frame", **trimmed}) + "\n")

        return trace_path

    def _select_agent_ids(self, pair: tuple[int, int], ring_snapshot: list[dict[str, Any]]) -> list[int]:
        i, j = pair
        if self.trace_agents_mode == "all" or not ring_snapshot:
            n_agents = int(ring_snapshot[-1]["n_agents"]) if ring_snapshot else 0
            return list(range(n_agents))

        ids = {i, j}
        latest = ring_snapshot[-1]
        selected = latest.get("selected_neighbors", [])
        if i < len(selected):
            ids.update(selected[i])
        if j < len(selected):
            ids.update(selected[j])
        return sorted(ids)

    def _trim_frame(self, frame: dict[str, Any], agent_ids: list[int]) -> dict[str, Any]:
        idx = agent_ids
        keep = set(idx)
        return {
            "t": frame["t"],
            "n_agents": len(idx),
            "agent_ids": idx,
            "positions": [frame["positions"][a] for a in idx],
            "velocities": [frame["velocities"][a] for a in idx],
            "v_cmd": [frame["v_cmd"][a] for a in idx],
            "goal_dirs": [frame["goal_dirs"][a] for a in idx],
            "speed_saturated": [frame["speed_saturated"][a] for a in idx],
            "accel_saturated": [frame["accel_saturated"][a] for a in idx],
            "selected_neighbors": {
                str(a): [n for n in frame["selected_neighbors"][a] if n in keep] for a in idx
            },
            "selected_obs": {
                str(a): [o for o in frame["selected_obs"][a] if int(o["idx"]) in keep] for a in idx
            },
            "selected_intents": {
                str(a): [o for o in frame.get("selected_intents", [])[a] if int(o["idx"]) in keep] for a in idx
            }
            if "selected_intents" in frame
            else {},
            "selected_messages": {
                str(a): [o for o in frame.get("selected_messages", [])[a] if int(o.get("sender_id", -1)) in keep] for a in idx
            }
            if "selected_messages" in frame
            else {},
        }

    def close(self) -> None:
        if self._event_fh is not None:
            self._event_fh.close()
            self._event_fh = None
        self._maybe_dump_episode_trace()

    def _maybe_dump_episode_trace(self) -> None:
        if not self.save_trace:
            return
        if self.trace_save_failures_only and not self._had_failure:
            return
        frames = self._full_trace.snapshot()
        if not frames:
            return
        with self.episode_trace_path.open("w", encoding="utf-8") as f:
            meta = {
                "kind": "meta",
                "trace_type": "episode",
                "agent_ids": list(range(int(frames[-1].get("n_agents", 0)))),
            }
            meta.update(self._episode_meta)
            f.write(json.dumps(meta) + "\n")
            for frame in frames:
                f.write(json.dumps({"kind": "frame", **frame}) + "\n")


def episode_dir_name(
    scenario: str,
    method: str,
    n_agents: int,
    seed: int,
    comm_profile: str | None = None,
) -> str:
    base = f"{scenario}_{method}_n{n_agents}_seed{seed}"
    if comm_profile:
        return f"{base}_comm_{comm_profile}"
    return base
