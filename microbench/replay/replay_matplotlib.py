from __future__ import annotations

from pathlib import Path
import json


def _age_color(age: float) -> str:
    if age < 0.05:
        return "#2CA02C"
    if age < 0.20:
        return "#F2A104"
    return "#D62728"


def _frame_obs_list(frm: dict, ego_id: int, ego_local_idx: int) -> list[dict]:
    selected_obs = frm.get("selected_obs", {})
    if isinstance(selected_obs, dict):
        return selected_obs.get(str(ego_id), [])
    if isinstance(selected_obs, list) and ego_local_idx < len(selected_obs):
        return selected_obs[ego_local_idx]
    return []


def render_trace(
    trace_path: str,
    out_path: str,
    fps: int = 25,
    tail: int = 25,
    show_sensed: bool = True,
    max_sensed_per_agent: int = 8,
) -> Path:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from matplotlib.patches import Circle
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for replay rendering") from exc

    tpath = Path(trace_path)
    opath = Path(out_path)

    meta = None
    frames = []
    with tpath.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("kind") == "meta":
                meta = rec
            elif rec.get("kind") == "frame":
                frames.append(rec)

    if not frames:
        raise ValueError(f"Trace has no frames: {trace_path}")

    agent_ids = frames[0].get("agent_ids")
    if agent_ids is None:
        agent_ids = meta.get("agent_ids") if meta else None
    if agent_ids is None:
        agent_ids = list(range(len(frames[0]["positions"])))
    id_to_local = {aid: i for i, aid in enumerate(agent_ids)}
    collision_pair = None
    if meta and "i" in meta and "j" in meta and meta["i"] in id_to_local and meta["j"] in id_to_local:
        collision_pair = (id_to_local[meta["i"]], id_to_local[meta["j"]])

    all_pos = [frm["positions"] for frm in frames]
    xs = [p[0] for frm in all_pos for p in frm]
    ys = [p[1] for frm in all_pos for p in frm]
    zs = [p[2] for frm in all_pos for p in frm]
    is_3d = (max(ys) - min(ys)) > 1e-6
    pad = 2.0
    xmin, xmax = min(xs) - pad, max(xs) + pad
    ymin, ymax = min(ys) - pad, max(ys) + pad
    zmin, zmax = min(zs) - pad, max(zs) + pad

    if is_3d:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    else:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect("equal")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(zmin, zmax)
        ax.set_xlabel("x")
        ax.set_ylabel("z")
    ax.set_title(f"Replay: {tpath.name}")

    circles = []
    if is_3d:
        for i in range(len(agent_ids)):
            color = "#E45756" if collision_pair and i in collision_pair else "#4C78A8"
            marker = ax.plot([0.0], [0.0], [0.0], marker="o", markersize=8, color=color, alpha=0.85)[0]
            circles.append(marker)
        trails = [ax.plot([], [], [], linewidth=1.0, alpha=0.6)[0] for _ in agent_ids]
    else:
        for i in range(len(agent_ids)):
            c = Circle((0.0, 0.0), radius=0.6, edgecolor="black", facecolor="#4C78A8", alpha=0.85)
            if collision_pair and i in collision_pair:
                c.set_facecolor("#E45756")
            ax.add_patch(c)
            circles.append(c)
        trails = [ax.plot([], [], linewidth=1.0, alpha=0.6)[0] for _ in agent_ids]
    if is_3d:
        t_text = ax.text2D(0.02, 0.98, "", transform=ax.transAxes, va="top")
    else:
        t_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")
    sensed_lines = []
    sensed_age_text = []

    if show_sensed:
        if is_3d:
            ax.text2D(
                0.02,
                0.93,
                "Sensed neighbors: dashed links, label=msgAge(s)",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
                color="#555555",
            )
        else:
            ax.text(
                0.02,
                0.93,
                "Sensed neighbors: dashed links, label=msgAge(s)",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
                color="#555555",
            )

    def update(frame_idx: int):
        for artist in sensed_lines:
            artist.remove()
        sensed_lines.clear()
        for artist in sensed_age_text:
            artist.remove()
        sensed_age_text.clear()

        frm = frames[frame_idx]
        pos = frm["positions"]
        for i, c in enumerate(circles):
            if is_3d:
                c.set_data([pos[i][0]], [pos[i][1]])
                c.set_3d_properties([pos[i][2]])
            else:
                c.center = (pos[i][0], pos[i][2])

            t0 = max(0, frame_idx - tail)
            trail_x = [frames[k]["positions"][i][0] for k in range(t0, frame_idx + 1)]
            trail_y = [frames[k]["positions"][i][1] for k in range(t0, frame_idx + 1)]
            trail_z = [frames[k]["positions"][i][2] for k in range(t0, frame_idx + 1)]
            if is_3d:
                trails[i].set_data(trail_x, trail_y)
                trails[i].set_3d_properties(trail_z)
            else:
                trails[i].set_data(trail_x, trail_z)
            if collision_pair and i in collision_pair:
                trails[i].set_color("#E45756")
            else:
                trails[i].set_color("#4C78A8")

        if show_sensed:
            if collision_pair:
                focus = list(collision_pair)
            else:
                focus = list(range(len(agent_ids)))
            for ego_local_idx in focus:
                ego_id = agent_ids[ego_local_idx]
                obs_list = _frame_obs_list(frm, ego_id, ego_local_idx)
                ex, ey, ez = pos[ego_local_idx][0], pos[ego_local_idx][1], pos[ego_local_idx][2]
                for obs in obs_list[:max_sensed_per_agent]:
                    nbr_id = int(obs["idx"])
                    if nbr_id not in id_to_local:
                        continue
                    nbr_local_idx = id_to_local[nbr_id]
                    nx, ny, nz = pos[nbr_local_idx][0], pos[nbr_local_idx][1], pos[nbr_local_idx][2]
                    age = float(obs.get("msg_age_sec", 0.0))
                    color = _age_color(age)
                    if is_3d:
                        line = ax.plot([ex, nx], [ey, ny], [ez, nz], linestyle="--", linewidth=0.8, alpha=0.75, color=color)[0]
                        txt = ax.text((ex + nx) * 0.5, (ey + ny) * 0.5, (ez + nz) * 0.5, f"{age:.2f}", fontsize=6, color=color)
                    else:
                        line = ax.plot([ex, nx], [ez, nz], linestyle="--", linewidth=0.8, alpha=0.75, color=color)[0]
                        txt = ax.text((ex + nx) * 0.5, (ez + nz) * 0.5, f"{age:.2f}", fontsize=6, color=color)
                    sensed_lines.append(line)
                    sensed_age_text.append(txt)

        t_text.set_text(f"t={frm['t']:.2f}s")
        return circles + trails + [t_text] + sensed_lines + sensed_age_text

    ani = FuncAnimation(fig, update, frames=len(frames), interval=1000 / max(1, fps), blit=False)

    suffix = opath.suffix.lower()
    if suffix == ".gif":
        ani.save(opath, writer="pillow", fps=fps)
    else:
        ani.save(opath, fps=fps)

    plt.close(fig)
    return opath
