#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python}"
BASE_CONFIG="${BASE_CONFIG:-configs/zelda_hmolqd.yaml}"
DATA_DIR="${DATA_DIR:-Data/The Legend of Zelda}"
KAGGLE_OUTPUTS_ROOT="${KAGGLE_OUTPUTS_ROOT:-/kaggle/working/kaggle_outputs}"
OUT_ROOT="${OUT_ROOT:-${KAGGLE_OUTPUTS_ROOT}/hmolqd_training_suite}"
TOKENIZERS="${TOKENIZERS:-vqvae2}"
BRANCHES="${BRANCHES:-stage_full}"
SEED="${SEED:-42}"
QUICK="${QUICK:-0}"

PRIMARY_TOKENIZER="${EVAL_TOKENIZER:-${TOKENIZERS%% *}}"
PRIMARY_BRANCH="${EVAL_BRANCH:-${BRANCHES%% *}}"
PRIMARY_RUN_NAME="${PRIMARY_TOKENIZER}_${PRIMARY_BRANCH}"
RUN_DIR="${RUN_DIR:-${OUT_ROOT}/downstream/${PRIMARY_RUN_NAME}}"
RUN_CONFIG="${RUN_CONFIG:-${OUT_ROOT}/configs/${PRIMARY_RUN_NAME}.yaml}"
RESULT_ROOT="${RESULT_ROOT:-${OUT_ROOT}/research}"
LOG_DIR="${RESULT_ROOT}/logs"
ARTIFACT_DIR="${OUT_ROOT}/artifacts"

first_existing_checkpoint() {
  local fallback="$1"
  shift
  local candidate
  for candidate in "$fallback" "$@"; do
    if [[ -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  printf '%s\n' "${fallback}"
}

VQVAE_CHECKPOINT="${VQVAE_CHECKPOINT:-$(first_existing_checkpoint \
  "${OUT_ROOT}/tokenizers/${PRIMARY_TOKENIZER}/checkpoints/vqvae/vqvae_pretrained.pth" \
  "${OUT_ROOT}/tokenizers/${PRIMARY_TOKENIZER}/checkpoints/vqvae/best_model.pth" \
  "${OUT_ROOT}/tokenizers/${PRIMARY_TOKENIZER}/checkpoints/vqvae/final_model.pth" \
  "${OUT_ROOT}/tokenizers/${PRIMARY_TOKENIZER}/vqvae_pretrained.pth")}"
DIFFUSION_CHECKPOINT="${DIFFUSION_CHECKPOINT:-$(first_existing_checkpoint \
  "${RUN_DIR}/checkpoints/diffusion/best_model.pth" \
  "${RUN_DIR}/checkpoints/diffusion/final_model.pth" \
  "${RUN_DIR}/checkpoints/diffusion/diffusion_pretrained.pth")}"
if [[ -n "${LOGIC_NET_CHECKPOINT:-}" ]]; then
  LOGIC_CHECKPOINT="${LOGIC_NET_CHECKPOINT}"
elif [[ -f "${RUN_DIR}/checkpoints/diffusion/best_logic_model.pth" ]]; then
  LOGIC_CHECKPOINT="${RUN_DIR}/checkpoints/diffusion/best_logic_model.pth"
else
  LOGIC_CHECKPOINT="${DIFFUSION_CHECKPOINT}"
fi
MASKED_ROOM_CHECKPOINT="${MASKED_ROOM_CHECKPOINT:-$(first_existing_checkpoint \
  "${RUN_DIR}/checkpoints/masked_room/masked_room_best.pth" \
  "${RUN_DIR}/checkpoints/masked_room/best_model.pth" \
  "${RUN_DIR}/checkpoints/masked_room/masked_room_final.pth" \
  "${RUN_DIR}/checkpoints/masked_room/final_model.pth")}"
FAST_SAMPLER_CHECKPOINT="${FAST_SAMPLER_CHECKPOINT:-$(first_existing_checkpoint \
  "${RUN_DIR}/checkpoints/fast_sampler/fast_sampler_best.pth" \
  "${RUN_DIR}/checkpoints/fast_sampler/best_model.pth" \
  "${RUN_DIR}/checkpoints/fast_sampler/fast_sampler_best_reselected.pth" \
  "${RUN_DIR}/checkpoints/fast_sampler/fast_sampler_final.pth" \
  "${RUN_DIR}/checkpoints/fast_sampler/final_model.pth")}"

RUN_CONDITIONING_LOGICNET_REPAIR="${RUN_CONDITIONING_LOGICNET_REPAIR:-1}"
RUN_FIXED_GRAPH="${RUN_FIXED_GRAPH:-1}"
RUN_GENERATED_GRAPH="${RUN_GENERATED_GRAPH:-1}"
RUN_ABLATION_STUDY="${RUN_ABLATION_STUDY:-1}"
RUN_RANDOM_BASELINE="${RUN_RANDOM_BASELINE:-1}"
RUN_MATCHED_BUDGET="${RUN_MATCHED_BUDGET:-1}"
RUN_PCG_BENCHMARK="${RUN_PCG_BENCHMARK:-1}"
RUN_OOD_BLINDED="${RUN_OOD_BLINDED:-1}"
RUN_DESIGNER_CONTROLLABILITY="${RUN_DESIGNER_CONTROLLABILITY:-1}"
RUN_PCBS_SWEEP="${RUN_PCBS_SWEEP:-1}"
RUN_PCBS_COMPONENT_ABLATION="${RUN_PCBS_COMPONENT_ABLATION:-1}"
RUN_PCBS_TELEMETRY_CALIBRATION="${RUN_PCBS_TELEMETRY_CALIBRATION:-0}"
PCBS_TELEMETRY_PATHS="${PCBS_TELEMETRY_PATHS:-}"
RUN_PROTOCOL_COMPARE="${RUN_PROTOCOL_COMPARE:-1}"
RUN_COMPUTE_CONSOLIDATION="${RUN_COMPUTE_CONSOLIDATION:-1}"
RUN_ARTIFACT_COLLECTION="${RUN_ARTIFACT_COLLECTION:-1}"
CONTINUE_ON_EVIDENCE_FAILURE="${CONTINUE_ON_EVIDENCE_FAILURE:-0}"

if [[ "${QUICK}" == "1" ]]; then
  CONDITIONING_SEEDS="${CONDITIONING_SEEDS:-42}"
  FIXED_GRAPH_SEEDS="${FIXED_GRAPH_SEEDS:-20260404}"
  GENERATED_GRAPH_SEEDS="${GENERATED_GRAPH_SEEDS:-20260514:20260514}"
  EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-2}"
  ABLATION_NUM_SAMPLES="${ABLATION_NUM_SAMPLES:-2}"
  POPULATION_SIZE="${POPULATION_SIZE:-8}"
  GENERATIONS="${GENERATIONS:-3}"
  DIFFUSION_STEPS="${DIFFUSION_STEPS:-8}"
  TIMEOUT_ASTAR="${TIMEOUT_ASTAR:-5000}"
  TIMEOUT_PCBS="${TIMEOUT_PCBS:-1000}"
  ABLATION_CONFIGS="${ABLATION_CONFIGS:-FULL,NO_LOGIC}"
  PCBS_LEVELS="${PCBS_LEVELS:-1}"
  PCBS_VARIANTS="${PCBS_VARIANTS:-1}"
  PCBS_PERSONAS="${PCBS_PERSONAS:-novice,balanced}"
else
  CONDITIONING_SEEDS="${CONDITIONING_SEEDS:-42,43,44}"
  FIXED_GRAPH_SEEDS="${FIXED_GRAPH_SEEDS:-20260404 20260405 20260406}"
  GENERATED_GRAPH_SEEDS="${GENERATED_GRAPH_SEEDS:-20260514:20260518}"
  EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-8}"
  ABLATION_NUM_SAMPLES="${ABLATION_NUM_SAMPLES:-8}"
  POPULATION_SIZE="${POPULATION_SIZE:-24}"
  GENERATIONS="${GENERATIONS:-24}"
  DIFFUSION_STEPS="${DIFFUSION_STEPS:-25}"
  TIMEOUT_ASTAR="${TIMEOUT_ASTAR:-200000}"
  TIMEOUT_PCBS="${TIMEOUT_PCBS:-50000}"
  ABLATION_CONFIGS="${ABLATION_CONFIGS:-}"
  PCBS_LEVELS="${PCBS_LEVELS:-1,2,3,4,5,6,7,8,9}"
  PCBS_VARIANTS="${PCBS_VARIANTS:-1,2}"
  PCBS_PERSONAS="${PCBS_PERSONAS:-novice,balanced,speedrunner}"
fi

GENERATED_GRAPH_VARIANTS="${GENERATED_GRAPH_VARIANTS:-}"
if [[ -z "${GENERATED_GRAPH_VARIANTS}" ]]; then
  GENERATED_GRAPH_VARIANTS="diffusion_no_logic,diffusion_logic1"
  if [[ -f "${FAST_SAMPLER_CHECKPOINT}" ]]; then
    GENERATED_GRAPH_VARIANTS="${GENERATED_GRAPH_VARIANTS},fast"
  fi
  if [[ -f "${MASKED_ROOM_CHECKPOINT}" ]]; then
    GENERATED_GRAPH_VARIANTS="${GENERATED_GRAPH_VARIANTS},masked"
  fi
fi

mkdir -p "${KAGGLE_OUTPUTS_ROOT}" "${RESULT_ROOT}" "${LOG_DIR}" "${ARTIFACT_DIR}"

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "${path}" ]]; then
    echo "[missing] ${label}: ${path}" >&2
    return 1
  fi
}

require_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "${path}" ]]; then
    echo "[missing] ${label}: ${path}" >&2
    return 1
  fi
}

run_step() {
  local name="$1"
  shift
  local log_path="${LOG_DIR}/${name}.log"
  echo
  echo "[step] ${name}"
  echo "[cmd] $*"
  set +e
  "$@" 2>&1 | tee "${log_path}"
  local status="${PIPESTATUS[0]}"
  set -e
  printf "%s\t%s\t%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${name}" "${status}" >> "${RESULT_ROOT}/steps.tsv"
  if [[ "${status}" -ne 0 ]]; then
    echo "[fail] ${name} exited with ${status}; log=${log_path}" >&2
    if [[ "${CONTINUE_ON_EVIDENCE_FAILURE}" == "1" ]]; then
      return 0
    fi
    exit "${status}"
  fi
}

write_manifest() {
  "${PYTHON}" - "${RESULT_ROOT}/research_suite_manifest.json" <<PY
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

payload = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "repo_root": r"${REPO_ROOT}",
    "out_root": r"${OUT_ROOT}",
    "kaggle_outputs_root": r"${KAGGLE_OUTPUTS_ROOT}",
    "result_root": r"${RESULT_ROOT}",
    "run_dir": r"${RUN_DIR}",
    "run_config": r"${RUN_CONFIG}",
    "primary_tokenizer": r"${PRIMARY_TOKENIZER}",
    "primary_branch": r"${PRIMARY_BRANCH}",
    "quick": r"${QUICK}" == "1",
    "data_dir": r"${DATA_DIR}",
    "base_config": r"${BASE_CONFIG}",
    "checkpoints": {
        "vqvae": r"${VQVAE_CHECKPOINT}",
        "diffusion": r"${DIFFUSION_CHECKPOINT}",
        "logic_net": r"${LOGIC_CHECKPOINT}",
        "fast_sampler": r"${FAST_SAMPLER_CHECKPOINT}",
        "masked_room": r"${MASKED_ROOM_CHECKPOINT}",
    },
    "budgets": {
        "conditioning_seeds": r"${CONDITIONING_SEEDS}",
        "fixed_graph_seeds": r"${FIXED_GRAPH_SEEDS}",
        "generated_graph_seeds": r"${GENERATED_GRAPH_SEEDS}",
        "eval_num_samples": int("${EVAL_NUM_SAMPLES}"),
        "ablation_num_samples": int("${ABLATION_NUM_SAMPLES}"),
        "population_size": int("${POPULATION_SIZE}"),
        "generations": int("${GENERATIONS}"),
        "diffusion_steps": int("${DIFFUSION_STEPS}"),
        "timeout_astar": int("${TIMEOUT_ASTAR}"),
        "timeout_pcbs": int("${TIMEOUT_PCBS}"),
        "generated_graph_variants": r"${GENERATED_GRAPH_VARIANTS}",
        "pcbs_telemetry_paths": r"${PCBS_TELEMETRY_PATHS}",
    },
    "python": sys.version,
    "platform": platform.platform(),
    "env_flags": {
        key: os.environ.get(key)
        for key in sorted(os.environ)
        if key.startswith("RUN_") or key in {"PROFILE", "TOKENIZERS", "BRANCHES", "QUICK", "PCBS_TELEMETRY_PATHS"}
    },
}
try:
    import torch
    payload["torch"] = {
        "version": getattr(torch, "__version__", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
    }
except Exception as exc:
    payload["torch"] = {"error": repr(exc)}
path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(f"[manifest] wrote {path}")
PY
}

write_manifest

if [[ "${RUN_CONDITIONING_LOGICNET_REPAIR}" == "1" ]]; then
  require_file "${RUN_CONFIG}" "run config"
  require_file "${VQVAE_CHECKPOINT}" "VQ-VAE checkpoint"
  require_file "${DIFFUSION_CHECKPOINT}" "diffusion checkpoint"
  require_file "${LOGIC_CHECKPOINT}" "LogicNet checkpoint"
  conditioning_args=(
    "${PYTHON}" scripts/run_conditioning_logicnet_repair_ablation.py
    --execute
    --config "${RUN_CONFIG}"
    --output "${RESULT_ROOT}/conditioning_logicnet_repair"
    --seeds "${CONDITIONING_SEEDS}"
    --vqvae-checkpoint "${VQVAE_CHECKPOINT}"
    --diffusion-checkpoint "${DIFFUSION_CHECKPOINT}"
    --logic-net-checkpoint "${LOGIC_CHECKPOINT}"
    --num-diffusion-steps "${DIFFUSION_STEPS}"
    --timeout-astar "${TIMEOUT_ASTAR}"
    --timeout-pcbs "${TIMEOUT_PCBS}"
    --population-size "${POPULATION_SIZE}"
    --generations "${GENERATIONS}"
  )
  if [[ "${QUICK}" == "1" ]]; then
    conditioning_args+=(--quick)
  fi
  run_step conditioning_logicnet_repair "${conditioning_args[@]}"
fi

if [[ "${RUN_FIXED_GRAPH}" == "1" ]]; then
  require_dir "${RUN_DIR}" "run directory"
  fixed_args=(
    "${PYTHON}" scripts/run_fixed_graph_multi_seed_audit.py
    --run-dir "${RUN_DIR}"
    --output-dir "${RESULT_ROOT}/fixed_graph"
    --seeds
  )
  # shellcheck disable=SC2206
  fixed_seed_array=(${FIXED_GRAPH_SEEDS})
  fixed_args+=("${fixed_seed_array[@]}")
  if [[ "${FIXED_GRAPH_INCLUDE_NO_FALLBACK:-0}" == "1" ]]; then
    fixed_args+=(--include-no-fallback-ablations)
  fi
  if [[ "${FIXED_GRAPH_INCLUDE_PUZZLE_ABLATIONS:-0}" == "1" ]]; then
    fixed_args+=(--include-puzzle-ablations)
  fi
  run_step fixed_graph "${fixed_args[@]}"
fi

if [[ "${RUN_GENERATED_GRAPH}" == "1" ]]; then
  require_dir "${RUN_DIR}" "run directory"
  generated_args=(
    "${PYTHON}" scripts/run_generated_graph_full_pipeline_eval.py
    --run-dir "${RUN_DIR}"
    --output-dir "${RESULT_ROOT}/generated_graph_full_pipeline"
    --variants "${GENERATED_GRAPH_VARIANTS}"
    --seeds "${GENERATED_GRAPH_SEEDS}"
    --data-root "${DATA_DIR}"
    --population-size "${POPULATION_SIZE}"
    --generations "${GENERATIONS}"
    --reuse-existing
  )
  if [[ "${GENERATED_GRAPH_INCLUDE_OOD:-0}" == "1" ]]; then
    generated_args+=(--include-ood)
  fi
  run_step generated_graph_full_pipeline "${generated_args[@]}"
fi

if [[ "${RUN_ABLATION_STUDY}" == "1" ]]; then
  require_file "${RUN_CONFIG}" "run config"
  require_file "${VQVAE_CHECKPOINT}" "VQ-VAE checkpoint"
  require_file "${DIFFUSION_CHECKPOINT}" "diffusion checkpoint"
  ablation_args=(
    "${PYTHON}" scripts/run_ablation_study.py
    --config "${RUN_CONFIG}"
    --output "${RESULT_ROOT}/ablation"
    --data-root "${DATA_DIR}"
    --num-samples "${ABLATION_NUM_SAMPLES}"
    --seed "${SEED}"
    --diffusion-steps "${DIFFUSION_STEPS}"
    --evolution-population "${POPULATION_SIZE}"
    --evolution-generations "${GENERATIONS}"
    --vqvae-checkpoint "${VQVAE_CHECKPOINT}"
    --diffusion-checkpoint "${DIFFUSION_CHECKPOINT}"
    --logic-net-checkpoint "${LOGIC_CHECKPOINT}"
  )
  if [[ -f "${MASKED_ROOM_CHECKPOINT}" ]]; then
    ablation_args+=(--masked-room-checkpoint "${MASKED_ROOM_CHECKPOINT}")
  fi
  if [[ -n "${ABLATION_CONFIGS}" ]]; then
    ablation_args+=(--configs "${ABLATION_CONFIGS}")
  fi
  if [[ "${QUICK}" == "1" ]]; then
    ablation_args+=(--quick --core-only)
  else
    ablation_args+=(--kaggle-t4x2)
  fi
  run_step ablation "${ablation_args[@]}"
fi

if [[ "${RUN_RANDOM_BASELINE}" == "1" ]]; then
  run_step random_baseline \
    "${PYTHON}" scripts/random_baseline.py \
    --num-samples "${RANDOM_BASELINE_SAMPLES:-${EVAL_NUM_SAMPLES}}" \
    --archive-cells "${RANDOM_BASELINE_ARCHIVE_CELLS:-256}" \
    --seeds 42 43 44 \
    --output-dir "${RESULT_ROOT}/random_baseline"
fi

if [[ "${RUN_MATCHED_BUDGET}" == "1" ]]; then
  matched_args=(
    "${PYTHON}" scripts/run_matched_budget_topology_benchmark.py
    --methods "${MATCHED_BUDGET_METHODS:-RANDOM,ES,GA,MAP_ELITES,FULL}"
    --num-samples "${EVAL_NUM_SAMPLES}"
    --seed "${SEED}"
    --eval-budget "${MATCHED_BUDGET_EVAL_BUDGET:-256}"
    --population-hint "${POPULATION_SIZE}"
    --data-root "${DATA_DIR}"
    --output "${RESULT_ROOT}/matched_budget"
  )
  if [[ "${QUICK}" == "1" ]]; then
    matched_args+=(--quick)
  else
    matched_args+=(--kaggle-t4x2)
  fi
  run_step matched_budget "${matched_args[@]}"
fi

if [[ "${RUN_PCG_BENCHMARK}" == "1" ]]; then
  run_step pcg_benchmark_alignment \
    "${PYTHON}" scripts/run_pcg_benchmark_alignment.py \
    --output "${RESULT_ROOT}/pcg_benchmark_alignment" \
    --data-root "${DATA_DIR}" \
    --methods "${PCG_BENCHMARK_METHODS:-FULL_GA,FULL_CVT}" \
    --num-samples "${EVAL_NUM_SAMPLES}" \
    --seed "${SEED}" \
    --population-size "${POPULATION_SIZE}" \
    --generations "${GENERATIONS}"
fi

if [[ "${RUN_OOD_BLINDED}" == "1" ]]; then
  run_step ood_blinded \
    "${PYTHON}" scripts/run_ood_scaling_and_blinded_eval.py \
    --output "${RESULT_ROOT}/ood_blinded_eval" \
    --data-root "${DATA_DIR}" \
    --methods "${OOD_METHODS:-FULL_GA,FULL_CVT}" \
    --num-samples "${EVAL_NUM_SAMPLES}" \
    --seed "${SEED}" \
    --population-size "${POPULATION_SIZE}" \
    --generations "${GENERATIONS}" \
    --blinded-per-condition "${BLINDED_PER_CONDITION:-6}"
fi

if [[ "${RUN_DESIGNER_CONTROLLABILITY}" == "1" ]]; then
  designer_args=(
    "${PYTHON}" scripts/run_designer_controllability_proof.py
    --execute
    --output "${RESULT_ROOT}/designer_controllability"
    --data-root "${DATA_DIR}"
    --methods "${DESIGNER_METHODS:-FULL_GA,FULL_CVT}"
    --samples-per-target "${DESIGNER_SAMPLES_PER_TARGET:-${EVAL_NUM_SAMPLES}}"
    --seed "${SEED}"
    --population-size "${POPULATION_SIZE}"
    --generations "${GENERATIONS}"
  )
  if [[ "${DESIGNER_WRITE_GRAPHS:-0}" == "1" ]]; then
    designer_args+=(--write-graphs)
  fi
  run_step designer_controllability "${designer_args[@]}"
fi

if [[ "${RUN_PCBS_SWEEP}" == "1" ]]; then
  run_step pcbs_persona_map_sweep \
    "${PYTHON}" scripts/run_pcbs_persona_map_sweep.py \
    --levels "${PCBS_LEVELS}" \
    --variants "${PCBS_VARIANTS}" \
    --personas "${PCBS_PERSONAS}" \
    --timeout-astar "${TIMEOUT_ASTAR}" \
    --timeout-pcbs "${TIMEOUT_PCBS}" \
    --seed "${SEED}" \
    --output-dir "${RESULT_ROOT}/pcbs_persona_map_sweep" \
    --data-root "${DATA_DIR}" \
    --quiet
fi

if [[ "${RUN_PCBS_COMPONENT_ABLATION}" == "1" ]]; then
  pcbs_component_args=(
    "${PYTHON}" scripts/run_pcbs_component_ablation.py
    --levels "${PCBS_COMPONENT_LEVELS:-1,2,3}"
    --variants "${PCBS_COMPONENT_VARIANTS:-1,2}"
    --persona "${PCBS_COMPONENT_PERSONA:-novice}"
    --timeout-astar "${TIMEOUT_ASTAR}"
    --timeout-pcbs "${TIMEOUT_PCBS}"
    --seed "${SEED}"
    --output-dir "${RESULT_ROOT}/pcbs_component_ablation"
    --data-root "${DATA_DIR}"
    --quiet
  )
  if [[ "${QUICK}" == "1" ]]; then
    pcbs_component_args+=(--quick)
  fi
  run_step pcbs_component_ablation "${pcbs_component_args[@]}"
fi

if [[ "${RUN_PCBS_TELEMETRY_CALIBRATION}" == "1" ]]; then
  if [[ -z "${PCBS_TELEMETRY_PATHS}" ]]; then
    echo "[missing] PCBS_TELEMETRY_PATHS is required when RUN_PCBS_TELEMETRY_CALIBRATION=1" >&2
    exit 1
  fi
  pcbs_calibration_args=(
    "${PYTHON}" scripts/calibrate_pcbs_personas_from_telemetry.py
    --output-dir "${RESULT_ROOT}/pcbs_telemetry_calibration"
    --personas "${PCBS_PERSONAS}"
    --telemetry
  )
  # shellcheck disable=SC2206
  pcbs_telemetry_array=(${PCBS_TELEMETRY_PATHS})
  pcbs_calibration_args+=("${pcbs_telemetry_array[@]}")
  sweep_csv="${RESULT_ROOT}/pcbs_persona_map_sweep/pcbs_persona_map_sweep.csv"
  if [[ -f "${sweep_csv}" ]]; then
    pcbs_calibration_args+=(--pcbs-sweep-csv "${sweep_csv}")
  fi
  run_step pcbs_telemetry_calibration "${pcbs_calibration_args[@]}"
fi

if [[ "${RUN_PROTOCOL_COMPARE}" == "1" ]]; then
  fixed_summary="${RESULT_ROOT}/fixed_graph/summary.json"
  matched_report="${RESULT_ROOT}/matched_budget/matched_budget_report.json"
  pcg_report="${RESULT_ROOT}/pcg_benchmark_alignment/pcg_benchmark_alignment_report.json"
  if [[ -f "${fixed_summary}" && -f "${matched_report}" && -f "${pcg_report}" ]]; then
    run_step protocol_to_baselines \
      "${PYTHON}" scripts/compare_protocol_to_baselines.py \
      --fixed-graph-summary "${fixed_summary}" \
      --matched-budget-report "${matched_report}" \
      --pcg-benchmark-report "${pcg_report}" \
      --output-dir "${RESULT_ROOT}/protocol_to_baselines"
  else
    echo "[skip] protocol_to_baselines missing one or more inputs:"
    echo "       ${fixed_summary}"
    echo "       ${matched_report}"
    echo "       ${pcg_report}"
  fi
fi

if [[ "${RUN_COMPUTE_CONSOLIDATION}" == "1" ]]; then
  run_step compute_sample_efficiency \
    "${PYTHON}" scripts/consolidate_compute_sample_efficiency.py \
    --roots "${OUT_ROOT}" "${RESULT_ROOT}" \
    --output "${RESULT_ROOT}/compute_sample_efficiency"
fi

if [[ "${RUN_ARTIFACT_COLLECTION}" == "1" ]]; then
  run_step collect_research_artifacts \
    "${PYTHON}" "${SCRIPT_DIR}/collect_training_artifacts.py" \
    --run-root "${OUT_ROOT}" \
    --out-dir "${ARTIFACT_DIR}" \
    --zip-name "hmolqd_kaggle_research_artifacts.zip" \
    --include-checkpoints
fi

echo
echo "[done] Kaggle research suite outputs: ${RESULT_ROOT}"
