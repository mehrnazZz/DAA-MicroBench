from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from microbench.rl import LEARNED_BASELINE_SCHEMA_VERSION, TINY_LEARNED_FEATURE_NAMES


def _unit_vectors(rng: np.random.Generator, n: int) -> np.ndarray:
    raw = rng.normal(size=(n, 3)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / np.maximum(norms, 1e-6)


def _make_synthetic_dataset(seed: int, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    goal = _unit_vectors(rng, n_samples)
    ego_vel = rng.uniform(-0.7, 0.7, size=(n_samples, 3)).astype(np.float32)
    avoid_dir = _unit_vectors(rng, n_samples)
    threat = rng.uniform(0.0, 1.0, size=(n_samples, 1)).astype(np.float32)
    avoid = avoid_dir * threat
    rel_vel = rng.uniform(-1.0, 1.0, size=(n_samples, 3)).astype(np.float32)
    neighbor_count = rng.integers(0, 9, size=(n_samples, 1)).astype(np.float32) / 8.0
    features = np.concatenate([goal, ego_vel, avoid, rel_vel, threat, neighbor_count], axis=1).astype(np.float32)

    # Label policy: transparent local goal-seeking plus inverse-clearance repulsion.
    labels = np.tanh(goal - 0.2 * ego_vel + 1.6 * avoid - 0.12 * rel_vel - 0.05 * threat).astype(np.float32)
    return features, labels


def _fit_linear_tanh_policy(features: np.ndarray, labels: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float64)
    y = np.arctanh(np.clip(np.asarray(labels, dtype=np.float64), -0.999, 0.999))
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    reg = float(ridge) * np.eye(x_aug.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0
    coef = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y)
    weights = coef[:-1, :].T.astype(float)
    bias = coef[-1, :].astype(float)
    return weights, bias


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the tiny linear learned-policy fixture on synthetic labels.")
    parser.add_argument("--out", default="tiny_linear_policy.generated.json")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--ridge", type=float, default=1e-6)
    args = parser.parse_args()

    features, labels = _make_synthetic_dataset(seed=int(args.seed), n_samples=int(args.samples))
    weights, bias = _fit_linear_tanh_policy(features, labels, ridge=float(args.ridge))
    spec = {
        "schema_version": LEARNED_BASELINE_SCHEMA_VERSION,
        "model_id": "tiny_linear_goal_avoidance_v0",
        "display_name": "Tiny linear learned-policy baseline",
        "model_type": "linear_tanh_policy",
        "action_shape": [3],
        "input_features": list(TINY_LEARNED_FEATURE_NAMES),
        "activation": "tanh",
        "weights": weights.round(8).tolist(),
        "bias": bias.round(8).tolist(),
        "training": {
            "recipe": "examples/rl_train_tiny_linear_policy.py",
            "source": "deterministic synthetic behavior-cloning fixture",
            "seed": int(args.seed),
            "samples": int(args.samples),
            "ridge": float(args.ridge),
            "label_policy": "goal direction plus local inverse-clearance repulsion",
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "samples": int(args.samples), "feature_dim": int(features.shape[1])}, indent=2))


if __name__ == "__main__":
    main()
