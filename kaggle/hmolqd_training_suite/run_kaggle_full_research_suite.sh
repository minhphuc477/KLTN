#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

echo "[full-suite] training phase"
RUN_ARTIFACT_COLLECTION="${RUN_TRAINING_ARTIFACT_COLLECTION:-0}" \
  bash "${SCRIPT_DIR}/run_kaggle_training_suite.sh"

echo "[full-suite] research/evidence phase"
RUN_ARTIFACT_COLLECTION="${RUN_FINAL_ARTIFACT_COLLECTION:-1}" \
  bash "${SCRIPT_DIR}/run_kaggle_research_suite.sh"

echo "[full-suite] complete"
