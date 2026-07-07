#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${1:-$(mktemp -d "${TMPDIR:-/tmp}/daa_release_readiness.XXXXXX")}"

mkdir -p "${WORK_DIR}"

cd "${ROOT}"

if [[ "${DAA_REQUIRE_CLEAN:-0}" == "1" ]]; then
  git diff --quiet
  git diff --cached --quiet
fi

python -m pytest -q
bash scripts/ci_sanity.sh "${WORK_DIR}/ci_sanity"
bash scripts/package_smoke.sh "${WORK_DIR}/package_smoke"
python -m microbench.cli golden-current-schema --golden-dir golden/current_schema
python -m microbench.cli validate-scenarios --all-builtins --all-generated-suites --quiet
python -m microbench.cli baseline-audit --require-public-alpha-ready --json >"${WORK_DIR}/baseline_audit.json"
python -m microbench.cli baseline-smoke --out-dir "${WORK_DIR}/baseline_smoke" --require-pass --json >"${WORK_DIR}/baseline_smoke.json"
python -m microbench.cli baseline-promotion --out-dir "${WORK_DIR}/baseline_promotion" --behavior-report "${WORK_DIR}/baseline_smoke.json" --require-calibrated --json >"${WORK_DIR}/baseline_promotion.json"
python -m microbench.cli baseline-evidence --out-dir "${WORK_DIR}/baseline_evidence" --require-pass --json >"${WORK_DIR}/baseline_evidence.json"
python -m microbench.cli rl-smoke --out-dir "${WORK_DIR}/rl_smoke" --require-pass --json >"${WORK_DIR}/rl_smoke.json"
python -m microbench.cli rl-smoke --out-dir "${WORK_DIR}/rl_tiny_learned" --policy tiny_learned --require-pass --json >"${WORK_DIR}/rl_tiny_learned.json"
python -m microbench.cli rl-calibration --out-dir "${WORK_DIR}/rl_calibration" --require-pass --json >"${WORK_DIR}/rl_calibration.json"
python -m microbench.cli rl-contract --json >"${WORK_DIR}/rl_contract.json"
python -m microbench.cli rl-freeze-check --require-pass --json >"${WORK_DIR}/rl_freeze_check.json"
python -m microbench.cli list-suites --json >"${WORK_DIR}/suites.json"
python -m microbench.cli list-methods --json --include-aliases >"${WORK_DIR}/methods.json"

echo "release_readiness: PASS"
echo "work_dir: ${WORK_DIR}"
