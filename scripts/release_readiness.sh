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
python -m microbench.cli rl-smoke --out-dir "${WORK_DIR}/rl_external_policy_spec" --policy-spec examples/external_policy_spec.json --max-steps 3 --require-pass --json >"${WORK_DIR}/rl_external_policy_spec.json"
python -m microbench.cli rl-smoke --out-dir "${WORK_DIR}/rl_external_model_predict_spec" --policy-spec examples/external_policy_model_predict_spec.json --max-steps 3 --require-pass --json >"${WORK_DIR}/rl_external_model_predict_spec.json"
python -m microbench.cli validate-learned-manifest --manifest examples/learned_submission_manifest_template.json --require-pass --json >"${WORK_DIR}/learned_manifest_template_validation.json"
python -m microbench.cli learned-submission-schema-check --require-pass --json >"${WORK_DIR}/learned_submission_schema_check.json"
python -m microbench.cli learned-submission-bundle --out-dir "${WORK_DIR}/learned_submission_policy_spec_bundle" --method learned_policy_spec --policy-spec examples/external_policy_spec.json --max-runs 1 --max-steps 3 --require-pass --json >"${WORK_DIR}/learned_submission_policy_spec_bundle.json"
python -m microbench.cli rl-smoke --out-dir "${WORK_DIR}/rl_tiny_learned" --policy tiny_learned --require-pass --json >"${WORK_DIR}/rl_tiny_learned.json"
python -m microbench.cli rl-smoke --out-dir "${WORK_DIR}/rl_mlp_learned" --policy mlp_learned --require-pass --json >"${WORK_DIR}/rl_mlp_learned.json"
python -m microbench.cli rl-calibration --out-dir "${WORK_DIR}/rl_calibration" --require-pass --json >"${WORK_DIR}/rl_calibration.json"
python -m microbench.cli learned-dataset-export --out-dir "${WORK_DIR}/learned_dataset_export" --lanes head_on --max-steps 2 --shard-size 16 --require-pass --json >"${WORK_DIR}/learned_dataset_export.json"
python -m microbench.cli train-learned-bc --out-dir "${WORK_DIR}/bc_mlp_policy" --lanes head_on --max-steps 2 --eval-lanes head_on --eval-max-steps 2 --hidden-dim 8 --require-pass --json >"${WORK_DIR}/bc_mlp_policy.json"
python -m microbench.cli learned-bc-evidence --out-dir "${WORK_DIR}/bc_mlp_evidence" --lanes head_on --max-steps 2 --eval-lanes head_on --eval-max-steps 2 --hidden-dim 8 --bundle-max-steps 2 --max-runs 1 --skip-fixtures --require-pass --json >"${WORK_DIR}/bc_mlp_evidence.json"
python -m microbench.cli learned-hard-lane-loop --out-dir "${WORK_DIR}/hard_lane_loop" --diagnostics "${WORK_DIR}/bc_mlp_evidence/learned_policy_diagnostics.json" --max-lanes 1 --mix-lanes crossing --sample-weighting safety --dataset-max-steps 2 --dataset-shard-size 16 --hidden-dim 8 --eval-lanes head_on --eval-max-steps 2 --bundle-max-steps 2 --max-runs 1 --skip-fixtures --require-pass --json >"${WORK_DIR}/hard_lane_loop.json"
python -m microbench.cli learned-closed-loop-finetune --out-dir "${WORK_DIR}/closed_loop_finetune" --base-policy-spec "${WORK_DIR}/bc_mlp_policy/policy_spec.json" --lanes head_on --train-max-steps 2 --generations 0 --population-size 1 --trainable-parameters all_layers --sigma 0.01 --eval-lanes head_on --eval-max-steps 2 --holdout-profile broad_3d_stress --holdout-scenarios sphere_swap_3d_medium --holdout-seeds 0 --holdout-comm ideal_50hz --holdout-n 3 --holdout-max-runs 1 --require-pass --require-promotion --json >"${WORK_DIR}/closed_loop_finetune.json"
python -m microbench.cli learned-closed-loop-study --out-dir "${WORK_DIR}/closed_loop_study" --base-policy-spec "${WORK_DIR}/bc_mlp_policy/policy_spec.json" --lanes head_on --train-max-steps 2 --generations 0 --population-size 1 --trainable-parameters all_layers --sigma 0.01 --eval-lanes head_on --eval-max-steps 2 --holdout-scenarios sphere_swap_3d_medium --holdout-seeds 0 --holdout-comm ideal_50hz --holdout-n 3 --holdout-max-runs 1 --bundle-max-steps 2 --bundle-max-runs 1 --require-pass --require-promotion --json >"${WORK_DIR}/closed_loop_study.json"
python -m microbench.cli rl-contract --json >"${WORK_DIR}/rl_contract.json"
python -m microbench.cli rl-freeze-check --require-pass --json >"${WORK_DIR}/rl_freeze_check.json"
python -m microbench.cli learned-submission-bundle --out-dir "${WORK_DIR}/learned_submission_bundle" --method learned_tiny --policy tiny_learned --max-runs 1 --max-steps 3 --require-pass --json >"${WORK_DIR}/learned_submission_bundle.json"
python -m microbench.cli validate-learned-bundle --bundle "${WORK_DIR}/learned_submission_bundle" --require-pass --json >"${WORK_DIR}/learned_bundle_validation.json"
python -m microbench.cli review-learned-bundle --bundle "${WORK_DIR}/learned_submission_bundle" --require-pass --json >"${WORK_DIR}/learned_bundle_review.json"
python -m microbench.cli learned-leaderboard --bundle "${WORK_DIR}/learned_submission_bundle" --bundle "${WORK_DIR}/learned_submission_policy_spec_bundle" --out "${WORK_DIR}/learned_policy_leaderboard.json" --require-pass --json >"${WORK_DIR}/learned_policy_leaderboard.stdout.json"
python -m microbench.cli learned-diagnostics --bundle "${WORK_DIR}/learned_submission_bundle" --bundle "${WORK_DIR}/learned_submission_policy_spec_bundle" --out "${WORK_DIR}/learned_policy_diagnostics.json" --require-pass --json >"${WORK_DIR}/learned_policy_diagnostics.stdout.json"
python -m microbench.cli list-suites --json >"${WORK_DIR}/suites.json"
python -m microbench.cli list-methods --json --include-aliases >"${WORK_DIR}/methods.json"

echo "release_readiness: PASS"
echo "work_dir: ${WORK_DIR}"
