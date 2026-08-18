# Baseline Fidelity And External References

DAA Microbench separates **benchmark performance** from **implementation fidelity**. A method can be useful and well tested without being an official implementation of a paper, and an official implementation can be valuable while still requiring external dependencies that do not belong in the core Python package.

Check the current machine-readable metadata with:

```bash
python -m microbench.cli list-methods --json --include-aliases
python -m microbench.cli baseline-audit --json
```

## Fidelity Tiers

| Tier | Meaning |
|---|---|
| `official_implementation` | Code comes from, or is wrapped around, an upstream official implementation. These should usually live outside the core package and enter via an external-reference manifest. |
| `faithful_reimplementation` | Clean-room implementation intended to reproduce the main algorithmic structure of a paper or method family, adapted to the DAA Microbench planner contract. |
| `inspired_clean_room` | Clean-room method inspired by a paper/family, but not a claim of exact reproduction. |
| `benchmark_utility` | Built-in baseline, fixture, lower bound, tutorial method, or plumbing check. Useful for benchmark interpretation but not a paper-fidelity claim. |
| `submission_bridge` | Adapter path for externally supplied learned policies or reference implementations. |

## Current Fidelity Matrix

| Method | Fidelity | External reference candidate | Provenance |
|---|---|---:|---|
| `baseline_goal` | `benchmark_utility` | no | Built-in lower-bound behavior for scenario sanity checks. |
| `orca_heuristic` | `inspired_clean_room` | no | Clean-room ORCA/RVO-family geometric heuristic adapted to the local planner contract. References: [ORCA](https://gamma-web.iacs.umd.edu/ORCA/), [RVO2](https://gamma-web.iacs.umd.edu/RVO2/downloads/). |
| `orca_with_staleness` | `inspired_clean_room` | no | Clean-room ORCA/RVO-family stale-track variant adapted to the benchmark observation model. |
| `cbf_qp` | `inspired_clean_room` | no | Clean-room CBF-style safety filter with dependency-free projection and optional SciPy solve. |
| `mpc_local` | `benchmark_utility` | no | Deterministic predictive sampling baseline for API and metric comparisons. |
| `mpc_nonlinear` | `inspired_clean_room` | no | Clean-room nonlinear MPC-style trajectory optimizer adapted to the velocity-command contract. |
| `dmpc_best_response` | `inspired_clean_room` | no | Clean-room distributed-MPC-style best-response coordinator for intent-sharing experiments. |
| `bvc_tube_dmpc` | `inspired_clean_room` | no | Clean-room buffered-Voronoi/tube-DMPC-style spatial partitioning baseline. |
| `dynamic_tube_dmpc` | `faithful_reimplementation` | yes | Clean-room paper-specific dynamic tube-DMPC reimplementation adapted to Microbench commands and AABB obstacles. Reference: [Drones 10(3), 177](https://www.mdpi.com/2504-446X/10/3/177). |
| `rmader` | `faithful_reimplementation` | yes | Clean-room RMADER/MADER-family reimplementation with MINVO hulls and separating hyperplanes; not a ROS/Gurobi port. Reference: [MIT ACL RMADER](https://github.com/mit-acl/rmader). |
| `ego_swarm` | `inspired_clean_room` | yes | Clean-room EGO-Swarm-inspired trajectory-sharing baseline; not a port or vendored copy of the upstream GPL ROS/C++ implementation. Reference: [EGO-Planner-Swarm](https://github.com/ZJU-FAST-Lab/ego-planner-swarm). |
| `ego_swarm_opt` | `inspired_clean_room` | yes | Clean-room EGO-Swarm-style control-point optimizer; not a port or vendored copy of the upstream GPL ROS/C++ implementation. References: [EGO-Planner-Swarm](https://github.com/ZJU-FAST-Lab/ego-planner-swarm), [EGO-Planner-v2](https://github.com/ZJU-FAST-Lab/EGO-Planner-v2). |
| `velocity_obstacle` | `inspired_clean_room` | no | Clean-room velocity-obstacle-family sampler adapted to DAA Microbench tracks and obstacles. |
| `reciprocal_velocity_obstacle` | `inspired_clean_room` | no | Clean-room reciprocal/HRVO-style sampler adapted to degraded observation metadata. |
| `learned_tiny` | `benchmark_utility` | no | Frozen synthetic learned-policy fixture for packaging and submission plumbing. |
| `learned_policy_spec` | `submission_bridge` | no | Bridge for trusted external learned-policy specs using the DAA RL observation/action contract. |
| `priority_yield` | `benchmark_utility` | no | Deterministic agentic right-of-way baseline for priority/message scenarios. |
| `negotiation_yield` | `benchmark_utility` | no | Structured proposal/ACK agentic baseline for decentralized communication tests. |
| `intent_dummy` | `benchmark_utility` | no | Intent and trace message-plumbing fixture. |
| `template` | `benchmark_utility` | no | Developer example for planner API tutorials. |

## External Reference Manifests

External official implementations should be compared through a manifest rather than vendored into the package. This is especially important for stacks with ROS, Gurobi, CUDA, GPL licensing, or hardware-specific build assumptions.

Validate a manifest without executing external code:

```bash
python -m microbench.cli validate-external-reference \
  --manifest examples/external_reference_rmader_manifest.yaml \
  --json
python -m microbench.cli validate-external-reference \
  --manifest examples/external_reference_ego_swarm_manifest.yaml \
  --json
```

Prepare a portable bundle for running an official external implementation against Microbench scenarios:

```bash
python -m microbench.cli external-reference-bundle \
  --method-family rmader \
  --out-dir runs_external_references/rmader_official_bundle \
  --scenarios urban_conflict_3d,urban_throughput_3d,stacked_swap_3d \
  --n 4,8 \
  --seeds 2 \
  --comm realistic_v2v_50hz \
  --runner-type ros
python -m microbench.cli external-reference-bundle \
  --method-family ego_swarm \
  --out-dir runs_external_references/ego_swarm_official_bundle \
  --scenarios urban_conflict_3d,urban_throughput_3d,stacked_swap_3d \
  --n 4,8 \
  --seeds 2 \
  --comm realistic_v2v_50hz \
  --runner-type ros
```

The bundle writes copied `scenarios/*.yaml`, `run_matrix.csv`, `run_matrix.json`, `manifest.yaml`, `result_schema.json`, `results_template.csv`, `summary_template.csv`, `RUN_NOTES.md`, `checksums.json`, and `external_reference_bundle.json`. The external stack should consume the scenario files and run matrix, then write the declared `results.csv` / `summary.csv` artifacts before final validation.

After an external run writes artifacts, require declared artifacts to exist and expose the core result fields:

```bash
python -m microbench.cli validate-external-reference \
  --manifest runs_external_references/rmader_official/manifest.yaml \
  --require-artifacts \
  --require-pass
```

The manifest must declare:

- upstream implementation name, source URL, license, and preferably commit
- related Microbench method family
- runner type and command notes
- no privileged information
- use of Microbench scenarios
- output artifacts, especially `results.csv`

The validator does **not** run ROS, Docker, Gurobi, or arbitrary scripts. It only checks disclosure, contract declarations, and optional artifact presence/schema.

### RMADER External Reference Contract

`rmader` has a stricter external-reference manifest because the upstream method makes specific safety and communication-delay claims. The example manifest at `examples/external_reference_rmader_manifest.yaml` is a capture template for an official MIT ACL RMADER run, not a claim that the official run has already been executed inside this repository.

For an official RMADER comparison, the manifest must explicitly declare these upstream method claims:

- decentralized asynchronous planning
- communication-delay robustness
- Delay Check
- two-step trajectory publication
- trajectory storing/checking
- MINVO interval polyhedra
- hard separating hyperplanes
- dynamic obstacle handling
- static obstacle handling
- solver backend, normally Gurobi for the upstream ROS stack

It must also explain the Microbench adapter boundary:

- scenario YAML to ROS starts, goals, obstacles, dynamic tracks, and timing
- one RMADER authority per drone
- local observation and intent filtering with no privileged global/future state
- mapping of Microbench V2V delay, jitter, loss, and staleness to the external process
- obstacle conversion
- conversion back to `results.csv`, optional traces, and optional MCAP

The built-in `rmader` planner remains a clean-room Python reimplementation adapted to the Microbench velocity-command contract. The external-reference manifest is how we compare against the official ROS/Gurobi implementation when that dependency-heavy stack is run outside the package.

### EGO-Swarm External Reference Contract

`ego_swarm` and `ego_swarm_opt` are clean-room Python comparators for trajectory-sharing experiments. They are not ports of ZJU FAST-Lab's EGO-Planner-Swarm ROS/C++ stack. The example manifest at `examples/external_reference_ego_swarm_manifest.yaml` is a capture template for an official upstream run, with `ego_swarm_opt` as the closest built-in comparator.

For an official EGO-Swarm comparison, the manifest must explicitly declare these upstream method claims:

- decentralized swarm planning
- asynchronous planning
- onboard sensing and compute
- unknown cluttered environment navigation
- trajectory sharing
- B-spline trajectory representation
- gradient-based optimization
- ESDF-free local planning
- static obstacle avoidance
- inter-agent collision avoidance
- simulator mode disclosure
- local sensing mode disclosure
- GPL-3.0 license disclosure

It must also explain the Microbench adapter boundary:

- scenario YAML to upstream starts, goals, world bounds, obstacles, dynamic tracks, and timing
- one EGO-Swarm authority per drone
- local observation and intent filtering with no privileged global/future state
- mapping of Microbench V2V delay, jitter, loss, rate, and stale trajectories to upstream trajectory broadcasts
- obstacle and map/local_sensing conversion
- simulator mode, such as fake_drone, quadrotor_simulator_so3, hardware logs, or another disclosed simulator
- GPL boundary for any upstream code, adapters, and redistributed artifacts
- conversion back to `results.csv`, optional traces, and optional MCAP

The built-in EGO-Swarm-style planners remain clean-room Python baselines adapted to the Microbench velocity-command contract. The external-reference manifest is how we compare against the official GPL ROS/C++ implementation when that dependency-heavy stack is run outside the package.

## Promotion Rule

Do not promote a baseline to a stable reference role only because it wins a leaderboard row. Stable reference status requires:

- explicit fidelity/provenance metadata
- docs and tests
- behavior evidence on 2D and 3D scenarios
- stress-suite or high-volume evidence
- clear known limitations
- external-reference comparison when an official implementation exists and is practical to run
