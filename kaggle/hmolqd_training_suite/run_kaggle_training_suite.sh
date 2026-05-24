#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python}"
BASE_CONFIG="${BASE_CONFIG:-configs/zelda_hmolqd.yaml}"
DATA_DIR="${DATA_DIR:-Data/The Legend of Zelda}"
OUT_ROOT="${OUT_ROOT:-/kaggle/working/hmolqd_training_suite}"
PROFILE="${PROFILE:-auto}"
TOKENIZERS="${TOKENIZERS:-vqvae2}"
BRANCHES="${BRANCHES:-stage_full}"
SEED="${SEED:-42}"
RUN_VQVAE="${RUN_VQVAE:-1}"
RUN_DIFFUSION="${RUN_DIFFUSION:-1}"
RUN_FAST_SAMPLER="${RUN_FAST_SAMPLER:-1}"
RUN_MASKED_ROOM="${RUN_MASKED_ROOM:-1}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
RUN_ARTIFACT_COLLECTION="${RUN_ARTIFACT_COLLECTION:-1}"
SINGLE_GPU_DEVICE="${SINGLE_GPU_DEVICE:-0}"
MASTER_PORT="${MASTER_PORT:-29500}"
QUICK="${QUICK:-0}"
VQVAE_CHECKPOINT_ROOT="${VQVAE_CHECKPOINT_ROOT:-}"

VQVAE_EPOCHS="${VQVAE_EPOCHS:-}"
DIFFUSION_EPOCHS="${DIFFUSION_EPOCHS:-}"
FAST_SAMPLER_EPOCHS="${FAST_SAMPLER_EPOCHS:-}"
MASKED_ROOM_EPOCHS="${MASKED_ROOM_EPOCHS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

mkdir -p "${OUT_ROOT}/configs" "${OUT_ROOT}/logs" "${OUT_ROOT}/tokenizers" "${OUT_ROOT}/downstream" "${OUT_ROOT}/artifacts"

GPU_INFO="$(${PYTHON} - <<'PY'
import json
try:
    import torch
    payload = {
        "cuda": bool(torch.cuda.is_available()),
        "count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
    }
except Exception as exc:
    payload = {"cuda": False, "count": 0, "names": [], "error": repr(exc)}
print(json.dumps(payload))
PY
)"
echo "[env] GPU info: ${GPU_INFO}"
GPU_COUNT="$(${PYTHON} - <<'PY'
try:
    import torch
    print(int(torch.cuda.device_count()) if torch.cuda.is_available() else 0)
except Exception:
    print(0)
PY
)"

if [[ "${PROFILE}" == "auto" ]]; then
  if [[ "${GPU_COUNT}" -ge 2 ]]; then
    PROFILE="t4x2"
  elif [[ "${GPU_COUNT}" -eq 1 ]]; then
    PROFILE="p100"
  else
    PROFILE="cpu"
  fi
fi
echo "[env] selected profile=${PROFILE}"

if [[ "${PROFILE}" == "t4x2" && "${GPU_COUNT}" -lt 2 ]]; then
  echo "[warn] PROFILE=t4x2 requested but fewer than two CUDA devices are visible; falling back to p100/single GPU."
  PROFILE="p100"
fi

"${PYTHON}" - "${OUT_ROOT}/artifacts/run_environment.json" "${PROFILE}" "${GPU_INFO}" "${TOKENIZERS}" "${BRANCHES}" "${BASE_CONFIG}" "${DATA_DIR}" <<'PY'
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

out_path = Path(sys.argv[1])
profile = sys.argv[2]
gpu_info_raw = sys.argv[3]
try:
    gpu_info = json.loads(gpu_info_raw)
except Exception:
    gpu_info = {"raw": gpu_info_raw}
payload = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "profile": profile,
    "gpu_info": gpu_info,
    "tokenizers": sys.argv[4].split(),
    "branches": sys.argv[5].split(),
    "base_config": sys.argv[6],
    "data_dir": sys.argv[7],
    "python": sys.version,
    "platform": platform.platform(),
}
try:
    import torch
    payload["torch"] = {
        "version": getattr(torch, "__version__", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": getattr(torch.version, "cuda", None),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
except Exception as exc:
    payload["torch"] = {"error": repr(exc)}
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(f"[env] wrote {out_path}")
PY

make_config() {
  local output="$1"
  local output_dir="$2"
  local experiment_name="$3"
  local tokenizer="$4"
  local branch="$5"
  local vqvae_checkpoint="${6:-}"
  local args=(
    "${SCRIPT_DIR}/make_kaggle_config.py"
    --base-config "${BASE_CONFIG}"
    --output "${output}"
    --output-dir "${output_dir}"
    --data-dir "${DATA_DIR}"
    --experiment-name "${experiment_name}"
    --profile "${PROFILE}"
    --tokenizer "${tokenizer}"
    --branch "${branch}"
    --seed "${SEED}"
    --summary-json "${output%.yaml}.summary.json"
  )
  if [[ -n "${vqvae_checkpoint}" ]]; then
    args+=(--vqvae-checkpoint "${vqvae_checkpoint}")
  fi
  if [[ "${QUICK}" == "1" ]]; then
    args+=(--quick)
  fi
  if [[ -n "${BATCH_SIZE}" ]]; then
    args+=(--batch-size "${BATCH_SIZE}")
  fi
  if [[ -n "${VQVAE_EPOCHS}" ]]; then
    args+=(--vqvae-epochs "${VQVAE_EPOCHS}")
  fi
  if [[ -n "${DIFFUSION_EPOCHS}" ]]; then
    args+=(--diffusion-epochs "${DIFFUSION_EPOCHS}")
  fi
  if [[ -n "${FAST_SAMPLER_EPOCHS}" ]]; then
    args+=(--fast-sampler-epochs "${FAST_SAMPLER_EPOCHS}")
  fi
  if [[ -n "${MASKED_ROOM_EPOCHS}" ]]; then
    args+=(--masked-room-epochs "${MASKED_ROOM_EPOCHS}")
  fi
  "${PYTHON}" "${args[@]}"
}

run_preflight() {
  local config="$1"
  local output_dir="$2"
  if [[ "${RUN_PREFLIGHT}" != "1" ]]; then
    return
  fi
  echo "[preflight] ${config}"
  "${PYTHON}" scripts/check_training_hyperparameters.py \
    --config "${config}" \
    --output "${output_dir}/preflight" \
    --probe-data
}

train_stage_single_gpu() {
  local config="$1"
  local stage="$2"
  if [[ "${PROFILE}" == "cpu" ]]; then
    echo "[train] CPU stage=${stage} config=${config}"
    "${PYTHON}" main.py train \
      --config "${config}" \
      --stage "${stage}" \
      --device cpu \
      --no-distributed-enabled \
      --nproc-per-node 1 \
      --cuda-visible-devices ""
  else
    echo "[train] single GPU stage=${stage} config=${config}"
    CUDA_VISIBLE_DEVICES="${SINGLE_GPU_DEVICE}" "${PYTHON}" main.py train \
      --config "${config}" \
      --stage "${stage}" \
      --device cuda \
      --no-distributed-enabled \
      --nproc-per-node 1 \
      --cuda-visible-devices "${SINGLE_GPU_DEVICE}"
  fi
}

train_diffusion() {
  local config="$1"
  if [[ "${PROFILE}" == "t4x2" ]]; then
    echo "[train] T4 x2 DDP diffusion config=${config}"
    CUDA_VISIBLE_DEVICES="0,1" MASTER_PORT="${MASTER_PORT}" "${PYTHON}" main.py train \
      --config "${config}" \
      --stage diffusion \
      --device cuda \
      --distributed-enabled \
      --distributed-backend nccl \
      --nproc-per-node 2 \
      --master-port "${MASTER_PORT}" \
      --cuda-visible-devices "0,1"
  else
    train_stage_single_gpu "${config}" diffusion
  fi
}

for tokenizer in ${TOKENIZERS}; do
  tokenizer_dir="${OUT_ROOT}/tokenizers/${tokenizer}"
  tokenizer_config="${OUT_ROOT}/configs/${tokenizer}_tokenizer.yaml"
  make_config "${tokenizer_config}" "${tokenizer_dir}" "kaggle_${tokenizer}_tokenizer" "${tokenizer}" base
  run_preflight "${tokenizer_config}" "${tokenizer_dir}"

  if [[ "${RUN_VQVAE}" == "1" ]]; then
    train_stage_single_gpu "${tokenizer_config}" vqvae | tee "${OUT_ROOT}/logs/${tokenizer}_vqvae.log"
  fi

  if [[ -n "${VQVAE_CHECKPOINT_ROOT}" ]]; then
    vqvae_checkpoint="${VQVAE_CHECKPOINT_ROOT}/${tokenizer}/checkpoints/vqvae/vqvae_pretrained.pth"
  else
    vqvae_checkpoint="${tokenizer_dir}/checkpoints/vqvae/vqvae_pretrained.pth"
  fi
  if [[ ! -f "${vqvae_checkpoint}" ]]; then
    echo "[error] missing tokenizer checkpoint: ${vqvae_checkpoint}" >&2
    exit 1
  fi

  for branch in ${BRANCHES}; do
    run_name="${tokenizer}_${branch}"
    run_dir="${OUT_ROOT}/downstream/${run_name}"
    run_config="${OUT_ROOT}/configs/${run_name}.yaml"
    make_config "${run_config}" "${run_dir}" "kaggle_${run_name}" "${tokenizer}" "${branch}" "${vqvae_checkpoint}"
    run_preflight "${run_config}" "${run_dir}"

    if [[ "${RUN_DIFFUSION}" == "1" ]]; then
      train_diffusion "${run_config}" | tee "${OUT_ROOT}/logs/${run_name}_diffusion.log"
    fi
    if [[ "${RUN_FAST_SAMPLER}" == "1" ]]; then
      train_stage_single_gpu "${run_config}" fast_sampler | tee "${OUT_ROOT}/logs/${run_name}_fast_sampler.log"
    fi
    if [[ "${RUN_MASKED_ROOM}" == "1" ]]; then
      train_stage_single_gpu "${run_config}" masked_room | tee "${OUT_ROOT}/logs/${run_name}_masked_room.log"
    fi
  done
done

if [[ "${RUN_ARTIFACT_COLLECTION}" == "1" ]]; then
  "${PYTHON}" "${SCRIPT_DIR}/collect_training_artifacts.py" \
    --run-root "${OUT_ROOT}" \
    --out-dir "${OUT_ROOT}/artifacts" \
    --zip-name "hmolqd_kaggle_${PROFILE}_artifacts.zip" \
    --include-checkpoints
fi

echo "[done] Kaggle training suite outputs: ${OUT_ROOT}"
