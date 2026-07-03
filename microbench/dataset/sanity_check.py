from __future__ import annotations

from pathlib import Path
import numpy as np


def sanity_check_shard(shard_path: str, out_plot: str | None = None) -> dict:
    path = Path(shard_path)
    d = np.load(path, allow_pickle=True)

    cond_nbh = d["cond_nbh"]
    valid = cond_nbh[..., 8]
    padded_mask = valid < 0.5
    padded_vals = cond_nbh[..., :8][padded_mask]
    padded_abs_max = float(np.max(np.abs(padded_vals))) if padded_vals.size else 0.0

    cond_ego = d["cond_ego"]
    cond_goal = d["cond_goal"]
    U0_raw = d["U0_raw"]

    stats = {
        "num_samples": int(cond_ego.shape[0]),
        "k": int(cond_nbh.shape[1]),
        "T": int(d["T"]),
        "dt_plan_s": float(d["dt_plan_s"]),
        "cond_ego_abs_max": float(np.max(np.abs(cond_ego))) if cond_ego.size else 0.0,
        "cond_goal_abs_max": float(np.max(np.abs(cond_goal))) if cond_goal.size else 0.0,
        "cond_nbh_abs_max": float(np.max(np.abs(cond_nbh[..., :8]))) if cond_nbh.size else 0.0,
        "padding_abs_max": padded_abs_max,
        "u0_raw_mean": float(np.mean(U0_raw)) if U0_raw.size else 0.0,
        "u0_raw_std": float(np.std(U0_raw)) if U0_raw.size else 0.0,
        "collision_label_rate": float(np.mean(d["collision_in_next_H"])) if d["collision_in_next_H"].size else 0.0,
    }

    if out_plot:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 4))
            vals = U0_raw.reshape(-1)
            ax.hist(vals, bins=80, color="#4C78A8", alpha=0.9)
            ax.set_title("U0_raw Distribution")
            ax.set_xlabel("velocity (m/s)")
            ax.set_ylabel("count")
            outp = Path(out_plot)
            outp.parent.mkdir(parents=True, exist_ok=True)
            fig.tight_layout()
            fig.savefig(outp)
            plt.close(fig)
            stats["plot_path"] = str(outp)
        except Exception as exc:  # pragma: no cover
            stats["plot_error"] = str(exc)

    return stats
