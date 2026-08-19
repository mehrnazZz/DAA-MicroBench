# RL Interface

DAA Microbench exposes a lightweight PettingZoo/Gymnasium-style interface for learning researchers. The wrappers use the same `EpisodeEngine` as benchmark runs, so scenarios, sensing, V2V impairment, agent messages, heterogeneous agents, collisions, and dynamics stay aligned with official evaluation.

Install the optional RL integrations when you want real Gymnasium/PettingZoo space classes:

```bash
pip install -e ".[rl]"
```

The core package still imports without those extras. In that case DAA Microbench uses a tiny fallback `Box` space with `sample()` and `contains()` for smoke tests and simple scripts.

Current public-alpha RL interface version: `0.1.0`.

## Parallel Multi-Agent

Use `DaaParallelEnv` for decentralized multi-drone control. It follows the PettingZoo `ParallelEnv` shape:

```python
import numpy as np

from microbench.rl import DaaParallelEnv

env = DaaParallelEnv(
    scenario_path="config/scenarios/stacked_swap_3d.yaml",
    n_agents=4,
    seed=0,
    comm_profile="ideal_50hz",
)

observations, infos = env.reset()
while env.agents:
    actions = {
        agent: env.action_space(agent).sample()
        for agent in env.agents
    }
    observations, rewards, terminations, truncations, infos = env.step(actions)

env.close()
```

Actions are normalized desired velocity commands with shape `(3,)` and bounds `[-1, 1]`. The wrapper scales each action by the controlled drone's `v_max_mps`; the simulator still applies speed and acceleration limits.

Observations are fixed-size local vectors:

- ego position, velocity, goal direction, goal distance, done flag, time, normalized agent id, priority, radius, `v_max`, and `a_max`
- padded top-k local neighbor tracks from the benchmark observation pipeline, using relative position, relative velocity, radius, and message age

The vector intentionally uses the same local information surface exposed to planners, not privileged global state for all drones.

Base observation layout:

| Field | Indices | Meaning |
| --- | ---: | --- |
| `ego_pos` | `0:3` | ego position `(x, y, z)` in meters |
| `ego_vel` | `3:6` | ego velocity `(vx, vy, vz)` in m/s |
| `goal_dir` | `6:9` | unit direction from ego to goal |
| `goal_dist` | `9` | distance to goal in meters |
| `done` | `10` | simulator goal-completion flag |
| `time_s` | `11` | episode time in seconds |
| `agent_id_norm` | `12` | agent id normalized to `[0, 1]` |
| `priority` | `13` | scenario priority value |
| `radius_m` | `14` | collision radius |
| `v_max_mps` | `15` | speed limit |
| `a_max_mps2` | `16` | acceleration limit |
| `neighbors` | `17:` | padded top-k neighbor blocks |

Each neighbor block has 9 values: present flag, relative position `(3)`, relative velocity `(3)`, neighbor radius, and message age.

You can inspect the machine-readable contract:

```bash
python -m microbench.cli rl-contract --json
```

The JSON includes schema versions for actions, observations, and rewards:

- action schema: normalized `(3,)` `float32` desired-velocity actions in `[-1, 1]`
- observation schema: fixed `float32` vector with base ego fields plus padded top-k neighbor blocks
- reward schema: default public-alpha training reward weights and term descriptions

## Background Traffic

The default controlled method is `rl_policy`. Scenario-configured agents with another method remain background traffic. This matters for official agentic stress cases such as `multi_intruder_3d_hard`, where noncooperative intruders can be configured as `baseline_goal` while cooperative agents are controlled by the learner.

You can also provide explicit methods:

```python
env = DaaParallelEnv(
    scenario_path="config/scenarios/stacked_swap_3d.yaml",
    n_agents=4,
    agent_methods=["rl_policy", "orca_heuristic", "orca_heuristic", "orca_heuristic"],
    controlled_agents=[0],
)
```

## Single-Agent Gymnasium Style

Use `DaaSingleAgentEnv` for single-ego experiments with benchmark baselines as background traffic:

```python
from microbench.rl import DaaSingleAgentEnv

env = DaaSingleAgentEnv(
    scenario_path="config/scenarios/stacked_swap_3d.yaml",
    n_agents=4,
    ego_agent_id=0,
    background_method="orca_heuristic",
)

obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

## Reward

The default reward is intentionally simple:

- positive progress toward goal
- small time penalty
- collision penalty
- near-miss penalty
- goal-completion bonus

Override weights with `reward_config` on `DaaParallelEnv`. Stable leaderboard comparisons should still use benchmark metrics and official suite reports, not training reward alone.

## Smoke Evaluation

Run a compact 2D/3D wrapper check before handing the environment to a trainer:

```bash
python -m microbench.cli rl-smoke \
  --out-dir runs_rl_smoke \
  --policy goal_direction \
  --require-pass
```

The command materializes `official_smoke_generated`, runs one 2D and one 3D scenario through the RL wrapper, writes `rl_smoke.json`, and writes per-episode rows to `rl_smoke_episodes.csv`. Built-in smoke policies are `zero`, `random`, `goal_direction`, `tiny_learned`, and `mlp_learned`.

Run compact 3D/degraded calibration before submitting learned-policy results:

```bash
python -m microbench.cli rl-calibration \
  --out-dir runs_rl_calibration \
  --policy goal_direction \
  --require-pass
```

The calibration command materializes `official_promotion_calibration`, runs a compact 3D volumetric lane and a degraded V2V/fused-sensing lane, writes `rl_calibration.json`, and writes per-episode rows to `rl_calibration_episodes.csv`. Passing it means the wrapper, policy interface, and finite rollout metrics survived stronger 3D/degraded exposure; it is not a leaderboard score.

Run the canonical validation matrix when a learned policy should be judged on the same encounter families as classical baselines:

```bash
python -m microbench.cli rl-validation-matrix \
  --out-dir runs_rl_validation_matrix \
  --policy goal_direction \
  --require-pass
```

This runs head-on, crossing, urban-obstacle, communication-delay, and high-N dense-merge lanes through `DaaParallelEnv` and writes `rl_validation_matrix.json` plus `rl_validation_matrix_episodes.csv`. `ok` is the hard interface/rollout gate; `behavior_pass` is separate collision/clearance/completion evidence and is not required for wrapper health.

The same runner is available from Python:

```python
from microbench.rl import run_rl_policy_smoke

report = run_rl_policy_smoke(
    out_dir="runs_rl_smoke",
    policy="random",
    max_steps=100,
)
assert report["ok"]
```

See `examples/rl_random_policy.py` for a minimal runnable script.

For custom training scripts, use the rollout helpers directly:

```python
from microbench.rl import DaaParallelEnv, rollout_parallel_env

env = DaaParallelEnv(
    scenario_path="config/scenarios/stacked_swap_3d.yaml",
    n_agents=4,
)
try:
    row = rollout_parallel_env(env, "goal_direction", seed=0, max_steps=100)
finally:
    env.close()
```

For small scenario/seed matrices, `run_parallel_policy_rollouts(...)` creates and closes one environment per row and returns the same per-episode fields used by `rl-smoke`.

## Learned-Policy Adapters

For external policy objects, wrap model inference in one of the dependency-free adapters:

```python
from microbench.rl import ModelPredictPolicyAdapter

policy = ModelPredictPolicyAdapter(my_model)
```

`ModelPredictPolicyAdapter` supports objects with `compute_single_action(observation)`, `predict(observation, deterministic=...)`, `predict(observation)`, or direct callable behavior. Tuple returns such as `(action, state)` are accepted. `CallablePolicyAdapter` supports plain functions shaped as `f(observation)`, `f(observation, info)`, or `f(agent, observation, action_space, info)`.

Both adapters validate finite `(3,)` actions and clip to the action space bounds. See `examples/rl_external_policy_adapter.py` for a runnable learned-policy adapter example that does not require any external RL framework.

## External Policy Specs

Use a JSON or YAML policy spec when you want the CLI to load an exported learned policy without editing DAA Microbench source. The dependency-free tiny-linear example is in `examples/external_policy_spec.json`:

```json
{
  "schema_version": "0.1",
  "policy_name": "external_tiny_linear_fixture",
  "adapter": "tiny_linear_json",
  "artifact_path": "../microbench/bundled_config/learned_baselines/tiny_linear_policy.json",
  "deterministic": true,
  "clip": true
}
```

For production-style import paths, see `examples/external_policy_model_predict_spec.json` and `examples/external_policy_callable_spec.json`. They load `examples/exported_linear_policy.json` through `examples/exported_policy.py` without requiring any external RL framework.

Supported public-alpha adapters are:

- `tiny_linear_json`: loads a DAA tiny-linear JSON weight artifact.
- `mlp_json`: loads a DAA frozen MLP JSON weight artifact, including artifacts from `train-learned-bc`.
- `callable`: imports a Python callable with `callable: "module:function"` and optional `signature`.
- `model_predict`: imports a Python factory/class with `factory: "module:Factory"` and wraps objects exposing `predict(...)`, `compute_single_action(...)`, or callable inference.
- `builtin`: aliases an existing built-in policy name for reproducible command files.

Relative `artifact_path` and `pythonpath` entries resolve from the spec file. Path-like `factory_kwargs` such as `artifact_path` and `checkpoint_dir` also resolve from the spec file. Import-based specs execute local Python code, so only use specs from trusted sources.

Run the spec through the learned-policy health gates:

```bash
python -m microbench.cli rl-smoke \
  --out-dir runs_external_rl_smoke \
  --policy-spec examples/external_policy_spec.json \
  --require-pass

python -m microbench.cli rl-calibration \
  --out-dir runs_external_rl_calibration \
  --policy-spec examples/external_policy_spec.json \
  --require-pass

python -m microbench.cli rl-validation-matrix \
  --out-dir runs_external_rl_validation_matrix \
  --policy-spec examples/external_policy_spec.json \
  --require-pass

python -m microbench.cli rl-smoke \
  --out-dir runs_external_model_predict_smoke \
  --policy-spec examples/external_policy_model_predict_spec.json \
  --max-steps 3 \
  --require-pass
```

Run the same external spec as a normal benchmark planner to produce standard `results.csv` and `summary.csv` rows:

```bash
python -m microbench.cli run \
  --scenario config/scenarios/stacked_swap_3d.yaml \
  --method learned_policy_spec \
  --policy-spec examples/external_policy_spec.json \
  --n 4 \
  --seed 0 \
  --comm ideal_50hz \
  --out-dir runs_external_policy_planner
```

The `learned_policy_spec` planner bridge loads the spec once per drone, converts each public `PlannerInput` into the stable RL local observation vector, calls `policy.action(agent, observation, action_space, info)`, clamps the normalized `(3,)` action, and scales it by the drone's `v_max`. It is a submission bridge, not a built-in reference baseline.

Build, validate, and summarize a learned submission bundle with the same spec:

```bash
python -m microbench.cli learned-submission-bundle \
  --out-dir runs_external_learned_bundle \
  --method learned_policy_spec \
  --policy-spec examples/external_policy_spec.json \
  --require-pass

python -m microbench.cli validate-learned-bundle \
  --bundle runs_external_learned_bundle \
  --require-pass

python -m microbench.cli review-learned-bundle \
  --bundle runs_external_learned_bundle \
  --require-pass
```

The bundle command writes a portable `policy_spec.json`; when the spec has a file `artifact_path`, it also copies the artifact under `policy_artifacts/` and rewrites the bundled spec to point at that copy. It also writes `rl_validation_matrix.json` on the canonical learned-policy validation lanes and `learned_submission_manifest.json`, which records policy-spec provenance, artifact hashes, dependency declarations, and training/inference disclosure fields. Pass `--submission-manifest path/to/overrides.json` to merge reviewer-ready disclosures into the generated manifest.

Validate disclosure drafts before creating a bundle:

```bash
python -m microbench.cli validate-learned-manifest \
  --manifest examples/learned_submission_manifest_template.json \
  --require-pass
```

See [LEARNED_POLICY_ADOPTION.md](LEARNED_POLICY_ADOPTION.md) for the full exported-policy-to-bundle workflow and submission manifest checklist.

DAA Microbench also ships frozen learned-policy fixtures:

```bash
python -m microbench.cli rl-smoke \
  --out-dir runs_rl_tiny_learned \
  --policy tiny_learned \
  --require-pass

python -m microbench.cli rl-smoke \
  --out-dir runs_rl_mlp_learned \
  --policy mlp_learned \
  --require-pass
```

The matching official planner methods are `learned_tiny` and `learned_mlp`, which produce normal benchmark `results.csv` and `summary.csv` rows. Their deterministic synthetic training recipes are in `examples/rl_train_tiny_linear_policy.py` and `examples/rl_train_mlp_policy.py`; checked-in weight artifacts live under `microbench/bundled_config/learned_baselines/`.

## Behavior-Cloned Training Workflow

Train a portable learned policy from real DAA Microbench RL validation-lane observations:

```bash
python -m microbench.cli train-learned-bc \
  --out-dir runs_bc_mlp_policy \
  --lanes head_on,crossing,urban_obstacle,communication_delay,high_n_dense_merge \
  --eval-lanes head_on,crossing,urban_obstacle \
  --require-pass
```

The trainer rolls out a transparent local DAA teacher over the public RL observation contract, fits a dependency-free two-layer tanh MLP, and writes:

- `bc_mlp_policy.json`: frozen MLP JSON weights using the same model contract as `mlp_learned`
- `policy_spec.json`: portable `mlp_json` policy spec for `rl-smoke`, `rl-validation-matrix`, `learned_policy_spec`, and learned-submission bundles
- `bc_training_report.json`: sample counts, lane/seed provenance, fit error, and optional validation-matrix evidence

Generated BC artifacts also declare a small inference guardrail: a goal-direction forward-progress floor plus a unit-norm action clamp before normal action-space clipping. This is disclosed in the model JSON and learned-submission manifest. The workflow is behavior cloning from a local teacher, not an upper-bound oracle or a certified DAA controller.

Create a reviewer-facing bundle and learned-policy leaderboard comparison against the frozen tiny/MLP fixtures:

```bash
python -m microbench.cli learned-bc-evidence \
  --out-dir runs_bc_mlp_evidence \
  --lanes head_on,crossing,urban_obstacle \
  --max-runs 1 \
  --require-pass
```

This command trains the BC policy, writes a disclosure overlay for the trained policy, creates a `learned_policy_spec` bundle for the trained artifact, creates comparable `learned_tiny` and `learned_mlp` bundles, and writes `learned_policy_leaderboard.json` plus CSV. Capped evidence is useful for development review; uncapped suite runs are still needed for final leaderboard claims.

## Learned Submission Bundle

Use the bundle command when preparing learned-policy artifacts for review:

```bash
python -m microbench.cli learned-submission-bundle \
  --out-dir runs_learned_bundle \
  --method learned_tiny \
  --policy tiny_learned \
  --require-pass
```

The bundle writes:

- `learned_submission_bundle.json`
- `rl_contract.json`
- `rl_freeze_check.json`
- `rl_smoke.json`
- `rl_calibration.json`
- `rl_validation_matrix.json`
- `rl_validation_matrix/rl_validation_matrix_episodes.csv`
- `planner_sweep/results.csv`
- `planner_sweep/summary.csv`
- `planner_sweep/result_schema.json`
- `planner_sweep/_generated_scenarios/<suite>/suite_manifest.yaml`
- `planner_sweep/acceptance.json`

By default the planner sweep uses `official_smoke_generated`. Use `--suite`, `--max-runs`, and `--save-trace` to control the official planner CSV workload. Larger leaderboard claims should still run the full intended official suite and submit the generated suite manifest, `results.csv`, `summary.csv`, and `result_schema.json`.

Review an existing bundle without rerunning simulations:

```bash
python -m microbench.cli validate-learned-bundle \
  --bundle runs_learned_bundle \
  --require-pass
```

The validator accepts either the bundle directory or `learned_submission_bundle.json`. It checks required artifacts, parses JSON/CSV files, confirms RL smoke/calibration/validation-matrix/freeze reports are passing, confirms planner acceptance has no failures, and verifies the planner CSVs are present and nonempty.

Summarize the same bundle for manual leaderboard review:

```bash
python -m microbench.cli review-learned-bundle \
  --bundle runs_learned_bundle \
  --require-pass
```

The reviewer does not rerun simulations. It validates the bundle, computes the documented v0 score from `summary.csv`, reports safety/mission/compute/communication/observation dimensions, and flags limitations such as limited planner sweeps, collision episodes, or planner guardrails.

Export public observation/action samples for learned-policy training or replay debugging:

```bash
python -m microbench.cli learned-dataset-export \
  --out-dir runs_learned_dataset \
  --lanes head_on,crossing,urban_obstacle \
  --max-steps 64 \
  --save-replay \
  --require-pass
```

This writes `learned_dataset_manifest.json`, `learned_dataset_episodes.csv`, compressed `shards/shard_*.npz`, and optional replay JSONL. Shards include `observations`, `actions`, `next_observations`, rewards, termination flags, lane metadata, and collision/near-miss diagnostics. The default action source is `bc_teacher`; pass `--policy-spec` to export samples from a submitted learned policy.

Run a diagnostics-driven hard-lane retraining loop:

```bash
python -m microbench.cli learned-hard-lane-loop \
  --out-dir runs_hard_lane_loop \
  --diagnostics runs_bc_mlp_evidence/learned_policy_diagnostics.json \
  --max-lanes 3 \
  --max-runs 1 \
  --require-pass
```

This command selects canonical weak validation lanes from learned diagnostics, exports matching public dataset shards, trains the same portable BC MLP from those shards, packages the trained spec, and writes a fresh learned leaderboard plus diagnostics report.

Compare multiple learned-policy bundles without rerunning simulations:

```bash
python -m microbench.cli learned-leaderboard \
  --bundle runs_learned_bundle \
  --bundle runs_external_learned_bundle \
  --out runs_learned_leaderboard/learned_policy_leaderboard.json \
  --require-pass

python -m microbench.cli learned-diagnostics \
  --bundle runs_learned_bundle \
  --bundle runs_external_learned_bundle \
  --out runs_learned_diagnostics/learned_policy_diagnostics.json \
  --require-pass
```

The leaderboard writes a JSON report and sibling CSV with one row per bundle, combining planner `summary.csv` score fields with RL validation-matrix lane evidence. Diagnostics writes JSON/CSV/Markdown labels that explain the main behavior pattern and weakest scenario/lane. These are development review tables; only rows marked `leaderboard_candidate` should be treated as candidate leaderboard evidence.

## Compatibility Check

For custom adapters, use the lightweight compatibility checker without installing PettingZoo's optional test helpers:

```python
from microbench.rl import DaaParallelEnv, check_parallel_env_api

env = DaaParallelEnv(
    scenario_path="config/scenarios/stacked_swap_3d.yaml",
    n_agents=4,
)
try:
    report = check_parallel_env_api(env, seed=0, steps=2)
    assert report["ok"]
finally:
    env.close()
```

The checker validates reset/step dictionary keys, observation/action-space shapes, finite rewards, boolean termination/truncation flags, and agent-list consistency.

When optional extras are installed, run the integration tests:

```bash
pip install -e ".[rl]"
python -m pytest tests/test_rl_optional_integrations.py -q
```

These tests are skipped in core installs and verify the wrappers inherit from Gymnasium/PettingZoo base classes when those packages are present.

## Stable-v1 Freeze Criteria

The public-alpha interface is versioned but not frozen. To inspect stable-v1 freeze readiness:

```bash
python -m microbench.cli rl-freeze-check --require-pass --json
```

The check covers the versioned contract, action shape/bounds, observation layout, lack of privileged global observation state, reward documentation, wrapper health gates, and dependency-free adapter examples. See `docs/RL_STABLE_V1_FREEZE.md` for the compatibility policy and learned-policy artifact expectations.

## Public Alpha Caveats

The RL interface is pre-v1. Observation vector layout, reward defaults, and helper wrappers may still change before stable v1. Official benchmark results should continue to report `results.csv`, `summary.csv`, suite manifests, and result schema sidecars.
