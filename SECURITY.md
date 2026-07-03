# Security Policy

DAA Microbench is a local benchmark harness, not flight software. It should not be used as the sole validation path for real aircraft, autonomous systems, or safety-critical deployments.

## Supported Versions

Security fixes are accepted for the current main branch and the latest tagged release once releases begin.

## Reporting a Vulnerability

Please report vulnerabilities privately once the public repository security contact is configured. Until then, do not disclose exploitable issues publicly in issues or pull requests.

Useful report details:

- affected version or commit
- operating system and Python version
- reproduction steps
- expected and actual behavior
- whether arbitrary code execution, data exposure, or supply-chain risk is involved

## Scope

In scope:

- unsafe deserialization or arbitrary code execution in benchmark tooling
- dependency or packaging issues that affect users
- malicious artifact handling in replays, traces, datasets, or configs

Out of scope:

- planner algorithms colliding in simulated scenarios
- benchmark scores being poor or unstable
- use of this repository as flight-certification evidence

## Safety Disclaimer

DAA Microbench is for research benchmarking. Passing its scenarios does not imply airworthiness, certification readiness, or safe real-world operation.
