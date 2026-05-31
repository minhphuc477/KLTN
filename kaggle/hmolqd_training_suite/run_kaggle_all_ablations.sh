#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python}"
KAGGLE_OUTPUTS_ROOT="${KAGGLE_OUTPUTS_ROOT:-/kaggle/working/kaggle_outputs}"
OUT_ROOT="${OUT_ROOT:-${KAGGLE_OUTPUTS_ROOT}/hmolqd_training_suite}"
TOKENIZERS="${TOKENIZERS:-vqvae vqvae2}"
BRANCHES="${BRANCHES:-stage_full stage_tokens_only stage_trace_only stage_loss010 stage_loss050}"
FORCE_FULL_SUITE="${FORCE_FULL_SUITE:-1}"
STRICT_CHECKPOINTS="${STRICT_CHECKPOINTS:-1}"
RUN_TRAINING_ARTIFACT_COLLECTION="${RUN_TRAINING_ARTIFACT_COLLECTION:-0}"
RUN_FINAL_ARTIFACT_COLLECTION="${RUN_FINAL_ARTIFACT_COLLECTION:-1}"

export KAGGLE_OUTPUTS_ROOT OUT_ROOT TOKENIZERS BRANCHES

if [[ "${FORCE_FULL_SUITE}" == "1" ]]; then
  QUICK="0"
  RUN_VQVAE="1"
  RUN_DIFFUSION="1"
  RUN_FAST_SAMPLER="1"
  RUN_MASKED_ROOM="1"
  RUN_PREFLIGHT="1"
  RUN_CONDITIONING_LOGICNET_REPAIR="1"
  RUN_FIXED_GRAPH="1"
  RUN_GENERATED_GRAPH="1"
  RUN_ABLATION_STUDY="1"
  RUN_RANDOM_BASELINE="1"
  RUN_MATCHED_BUDGET="1"
  RUN_PCG_BENCHMARK="1"
  RUN_OOD_BLINDED="1"
  RUN_DESIGNER_CONTROLLABILITY="1"
  RUN_PCBS_SWEEP="1"
  RUN_PCBS_COMPONENT_ABLATION="1"
  RUN_PROTOCOL_COMPARE="1"
  RUN_COMPUTE_CONSOLIDATION="1"
  if [[ -n "${PCBS_TELEMETRY_PATHS:-}" ]]; then
    RUN_PCBS_TELEMETRY_CALIBRATION="1"
  else
    RUN_PCBS_TELEMETRY_CALIBRATION="0"
  fi
  export QUICK RUN_VQVAE RUN_DIFFUSION RUN_FAST_SAMPLER RUN_MASKED_ROOM RUN_PREFLIGHT
  export RUN_CONDITIONING_LOGICNET_REPAIR RUN_FIXED_GRAPH RUN_GENERATED_GRAPH RUN_ABLATION_STUDY
  export RUN_RANDOM_BASELINE RUN_MATCHED_BUDGET RUN_PCG_BENCHMARK RUN_OOD_BLINDED
  export RUN_DESIGNER_CONTROLLABILITY RUN_PCBS_SWEEP RUN_PCBS_COMPONENT_ABLATION
  export RUN_PCBS_TELEMETRY_CALIBRATION RUN_PROTOCOL_COMPARE RUN_COMPUTE_CONSOLIDATION
fi

mkdir -p "${KAGGLE_OUTPUTS_ROOT}" "${OUT_ROOT}/artifacts" "${OUT_ROOT}/research"

echo "[all-ablations] output root: ${OUT_ROOT}"
echo "[all-ablations] tokenizers: ${TOKENIZERS}"
echo "[all-ablations] branches: ${BRANCHES}"
echo "[all-ablations] force full suite: ${FORCE_FULL_SUITE}"
echo "[all-ablations] strict checkpoint audit: ${STRICT_CHECKPOINTS}"

echo "[all-ablations] training tokenizer and branch matrix"
RUN_ARTIFACT_COLLECTION="${RUN_TRAINING_ARTIFACT_COLLECTION}" \
  bash "${SCRIPT_DIR}/run_kaggle_training_suite.sh"

if [[ "${STRICT_CHECKPOINTS}" == "1" ]]; then
  "${PYTHON}" "${SCRIPT_DIR}/verify_kaggle_checkpoints.py" \
    --run-root "${OUT_ROOT}" \
    --tokenizers ${TOKENIZERS} \
    --branches ${BRANCHES} \
    --output-json "${OUT_ROOT}/artifacts/checkpoint_completeness.json" \
    --output-tsv "${OUT_ROOT}/artifacts/checkpoint_completeness.tsv" \
    --strict
fi

for tokenizer in ${TOKENIZERS}; do
  for branch in ${BRANCHES}; do
    result_root="${OUT_ROOT}/research/${tokenizer}_${branch}"
    echo
    echo "[all-ablations] evidence tokenizer=${tokenizer} branch=${branch}"
    EVAL_TOKENIZER="${tokenizer}" \
    EVAL_BRANCH="${branch}" \
    RESULT_ROOT="${result_root}" \
    RUN_ARTIFACT_COLLECTION=0 \
      bash "${SCRIPT_DIR}/run_kaggle_research_suite.sh"
  done
done

if [[ "${RUN_FINAL_ARTIFACT_COLLECTION}" == "1" ]]; then
  "${PYTHON}" "${SCRIPT_DIR}/collect_training_artifacts.py" \
    --run-root "${OUT_ROOT}" \
    --out-dir "${OUT_ROOT}/artifacts" \
    --zip-name "hmolqd_kaggle_all_ablations_artifacts.zip" \
    --include-checkpoints
fi

if [[ "${STRICT_CHECKPOINTS}" == "1" ]]; then
  "${PYTHON}" "${SCRIPT_DIR}/verify_kaggle_checkpoints.py" \
    --run-root "${OUT_ROOT}" \
    --tokenizers ${TOKENIZERS} \
    --branches ${BRANCHES} \
    --output-json "${OUT_ROOT}/artifacts/checkpoint_completeness.json" \
    --output-tsv "${OUT_ROOT}/artifacts/checkpoint_completeness.tsv" \
    --strict
fi

echo
echo "[done] Kaggle all-ablation suite outputs: ${OUT_ROOT}"
