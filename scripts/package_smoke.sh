#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${1:-$(mktemp -d "${TMPDIR:-/tmp}/daa_package_smoke.XXXXXX")}"
WHEEL_DIR="${WORK_DIR}/wheel"
INSTALL_DIR="${WORK_DIR}/install"
RUN_DIR="${WORK_DIR}/run"

mkdir -p "${WHEEL_DIR}" "${INSTALL_DIR}"

python -m pip wheel "${ROOT}" --no-deps --no-build-isolation -w "${WHEEL_DIR}"
python -m pip install --target "${INSTALL_DIR}" --no-deps --force-reinstall "${WHEEL_DIR}"/daa_microbench-*.whl

(
  cd "${TMPDIR:-/tmp}"
  export PYTHONPATH="${INSTALL_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
  python -m microbench.cli list-suites --json >/dev/null
  python -m microbench.cli validate-scenarios --all-builtins --all-generated-suites --quiet
  python -m microbench.cli canonical-sweep \
    --suite official_smoke_generated \
    --methods baseline_goal \
    --max-runs 1 \
    --out-dir "${RUN_DIR}"
)

test -s "${RUN_DIR}/results.csv"
test -s "${RUN_DIR}/summary.csv"

echo "package_smoke: PASS"
echo "work_dir: ${WORK_DIR}"
