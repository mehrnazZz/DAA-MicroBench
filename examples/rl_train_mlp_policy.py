from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from microbench.rl import LEARNED_BASELINE_SCHEMA_VERSION, MLP_LEARNED_FEATURE_NAMES, MLP_LEARNED_MODEL_ID


def _unit_vectors(rng: np.random.Generator, n: int) -> np.ndarray:
    raw = rng.normal(size=(n, 3)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / np.maximum(norms, 1e-6)


def _make_synthetic_dataset(seed: int, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    goal = _unit_vectors(rng, n_samples)
    ego_vel = rng.uniform(-0.8, 0.8, size=(n_samples, 3)).astype(np.float32)
    avoid_dir = _unit_vectors(rng, n_samples)
    threat = rng.uniform(0.0, 1.0, size=(n_samples, 1)).astype(np.float32)
    avoid = avoid_dir * threat
    rel_vel = rng.uniform(-1.0, 1.0, size=(n_samples, 3)).astype(np.float32)
    neighbor_count = rng.integers(0, 9, size=(n_samples, 1)).astype(np.float32) / 8.0
    features = np.concatenate([goal, ego_vel, avoid, rel_vel, threat, neighbor_count], axis=1).astype(np.float32)

    lateral = np.cross(goal, avoid_dir).astype(np.float32)
    lateral_norm = np.linalg.norm(lateral, axis=1, keepdims=True)
    lateral = lateral / np.maximum(lateral_norm, 1e-6)
    nonlinear_avoid_gain = 0.8 + 1.8 * threat + 0.4 * neighbor_count
    goal_gate = 1.0 - 0.55 * threat
    labels = np.tanh(
        goal_gate * goal
        - 0.16 * ego_vel
        + nonlinear_avoid_gain * avoid
        - 0.10 * rel_vel
        + 0.18 * threat * lateral
    ).astype(np.float32)
    return features, labels


def _fit_random_feature_mlp(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    hidden_dim: int,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(int(seed) + 1009)
    x = np.asarray(features, dtype=np.float64)
    y = np.arctanh(np.clip(np.asarray(labels, dtype=np.float64), -0.999, 0.999))
    w1 = rng.normal(0.0, 0.75 / np.sqrt(x.shape[1]), size=(int(hidden_dim), x.shape[1]))
    b1 = rng.normal(0.0, 0.08, size=(int(hidden_dim),))
    hidden = np.tanh(x @ w1.T + b1)
    hidden_aug = np.concatenate([hidden, np.ones((hidden.shape[0], 1), dtype=np.float64)], axis=1)
    reg = float(ridge) * np.eye(hidden_aug.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0
    coef = np.linalg.solve(hidden_aug.T @ hidden_aug + reg, hidden_aug.T @ y)
    w2 = coef[:-1, :].T
    b2 = coef[-1, :]
    pred = np.tanh(hidden @ w2.T + b2)
    rmse = float(np.sqrt(np.mean((pred - labels) ** 2)))
    return w1.astype(float), b1.astype(float), w2.astype(float), b2.astype(float), rmse


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the frozen MLP learned-policy fixture on synthetic labels.")
    parser.add_argument("--out", default="mlp_policy.generated.json")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--ridge", type=float, default=1e-4)
    args = parser.parse_args()

    features, labels = _make_synthetic_dataset(seed=int(args.seed), n_samples=int(args.samples))
    w1, b1, w2, b2, rmse = _fit_random_feature_mlp(
        features,
        labels,
        seed=int(args.seed),
        hidden_dim=int(args.hidden_dim),
        ridge=float(args.ridge),
    )
    spec = {
        "schema_version": LEARNED_BASELINE_SCHEMA_VERSION,
        "model_id": MLP_LEARNED_MODEL_ID,
        "display_name": "Frozen MLP learned-policy baseline",
        "model_type": "mlp_tanh_policy",
        "hidden_dim": int(args.hidden_dim),
        "action_shape": [3],
        "input_features": list(MLP_LEARNED_FEATURE_NAMES),
        "hidden_activation": "tanh",
        "output_activation": "tanh",
        "layer1_weights": w1.round(8).tolist(),
        "layer1_bias": b1.round(8).tolist(),
        "layer2_weights": w2.round(8).tolist(),
        "layer2_bias": b2.round(8).tolist(),
        "training": {
            "recipe": "examples/rl_train_mlp_policy.py",
            "source": "deterministic synthetic behavior-cloning fixture",
            "seed": int(args.seed),
            "samples": int(args.samples),
            "hidden_dim": int(args.hidden_dim),
            "ridge": float(args.ridge),
            "fit_rmse": round(rmse, 8),
            "label_policy": "nonlinear goal gating plus local inverse-clearance repulsion and lateral deconfliction",
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(out),
                "samples": int(args.samples),
                "feature_dim": int(features.shape[1]),
                "hidden_dim": int(args.hidden_dim),
                "fit_rmse": round(rmse, 8),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
