#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-configs/zelda_hmolqd.yaml}"
STAGE="${STAGE:-diffusion}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
MASTER_PORT="${MASTER_PORT:-29500}"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export PYTHONUNBUFFERED=1

GPU_COUNT="$(python -c "import torch; print(torch.cuda.device_count())")"
if [[ "${GPU_COUNT}" -lt 2 ]]; then
  echo "Expected 2 visible CUDA devices for Kaggle dual-T4 training, found ${GPU_COUNT}." >&2
  exit 1
fi

python main.py train \
  --config "${CONFIG_PATH}" \
  --stage "${STAGE}" \
  --distributed-enabled \
  --nproc-per-node 2 \
  --distributed-backend nccl \
  --master-port "${MASTER_PORT}" \
  --device cuda \
  "$@"
