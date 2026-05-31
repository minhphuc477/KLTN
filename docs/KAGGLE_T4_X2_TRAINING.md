# Kaggle Dual-T4 Training

This repo supports Kaggle-oriented full-stack training under
[`../kaggle/hmolqd_training_suite`](../kaggle/hmolqd_training_suite).

## Recommended Hardware

Use T4 x2 when Kaggle exposes two CUDA devices. The repo can use two GPUs for
the diffusion stage through PyTorch DDP, while VQ-VAE/VQ-VAE-2, fast-sampler,
and masked-room stages remain single-process. Use the P100/single-GPU profile
as the fallback when only one GPU is visible.

## Notebook Cell

Kaggle notebooks execute Python cells by default. Use `%%bash` as the first
line of the cell. Do not paste a bare `bash ...` command into a Python cell.

```bash
%%bash
cd /kaggle/working/KLTN
bash kaggle/hmolqd_training_suite/run_kaggle_training_suite.sh
```

Default behavior:

- auto-detects `t4x2`, single-GPU, or CPU profile
- trains `vqvae2`
- runs the `stage_full` branch
- writes outputs to `/kaggle/working/kaggle_outputs/hmolqd_training_suite`
- records `artifacts/run_environment.json`

To train and then run the full research evidence suite:

```bash
%%bash
cd /kaggle/working/KLTN
bash kaggle/hmolqd_training_suite/run_kaggle_full_research_suite.sh
```

To run only the post-training evidence suite:

```bash
%%bash
cd /kaggle/working/KLTN
bash kaggle/hmolqd_training_suite/run_kaggle_research_suite.sh
```

To run every packaged tokenizer and stage-conditioning ablation locally inside
the Kaggle environment:

```bash
%%bash
cd /kaggle/working/KLTN
bash kaggle/hmolqd_training_suite/run_kaggle_all_ablations.sh
```

## Kaggle API Kernel

```bash
cp kaggle/hmolqd_training_suite/kernel-metadata.template.json \
   kaggle/hmolqd_training_suite/kernel-metadata.json

kaggle kernels push \
  -p kaggle/hmolqd_training_suite \
  --accelerator NvidiaTeslaT4 \
  --timeout 43200
```

Edit `kernel-metadata.json` before pushing. Kaggle metadata controls GPU and
internet enablement, while the `--accelerator` flag selects the accelerator ID
for the run.

## Common Runs

VQ-VAE-2 full stack:

```bash
%%bash
cd /kaggle/working/KLTN
TOKENIZERS="vqvae2" BRANCHES="stage_full" \
  bash kaggle/hmolqd_training_suite/run_kaggle_training_suite.sh
```

Tokenizer ablation only:

```bash
%%bash
cd /kaggle/working/KLTN
TOKENIZERS="vqvae vqvae2" \
RUN_DIFFUSION=0 RUN_FAST_SAMPLER=0 RUN_MASKED_ROOM=0 \
  bash kaggle/hmolqd_training_suite/run_kaggle_training_suite.sh
```

Stage-conditioning ablation:

```bash
%%bash
cd /kaggle/working/KLTN
TOKENIZERS="vqvae2" \
BRANCHES="stage_full stage_tokens_only stage_trace_only stage_loss010 stage_loss050" \
  bash kaggle/hmolqd_training_suite/run_kaggle_training_suite.sh
```

Smoke check:

```bash
%%bash
cd /kaggle/working/KLTN
QUICK=1 TOKENIZERS="vqvae2" BRANCHES="stage_full" \
  bash kaggle/hmolqd_training_suite/run_kaggle_training_suite.sh
```

Full smoke check, including evidence scripts:

```bash
%%bash
cd /kaggle/working/KLTN
QUICK=1 TOKENIZERS="vqvae2" BRANCHES="stage_full" \
  bash kaggle/hmolqd_training_suite/run_kaggle_full_research_suite.sh
```

Full ablation matrix:

```bash
%%bash
cd /kaggle/working/KLTN
TOKENIZERS="vqvae vqvae2" \
BRANCHES="stage_full stage_tokens_only stage_trace_only stage_loss010 stage_loss050" \
  bash kaggle/hmolqd_training_suite/run_kaggle_all_ablations.sh
```

The all-ablation runner defaults to `FORCE_FULL_SUITE=1` and
`STRICT_CHECKPOINTS=1`, so inherited skip flags are overridden, quick mode is
disabled, and the run fails if any required checkpoint is missing.

## Research Evidence Suite

The evidence suite fills the currently missing execution layer. It runs:

- conditioning / LogicNet / repair ablation
- fixed-graph multi-seed protocol
- generated-graph full-pipeline evaluation
- core component ablation
- random, matched-budget, and PCG Benchmark-alignment baselines
- OOD scaling and blinded-eval packet generation
- designer controllability proof
- P-CBS persona and component ablations
- compute/sample-efficiency consolidation

Outputs are written under:

```text
/kaggle/working/kaggle_outputs/hmolqd_training_suite/research/
```

The packaged zip is:

```text
/kaggle/working/kaggle_outputs/hmolqd_training_suite/artifacts/hmolqd_kaggle_research_artifacts.zip
```

The strict all-ablation path also writes
`artifacts/checkpoint_completeness.json` and
`artifacts/checkpoint_completeness.tsv`.

Detailed controls and required thesis artifacts are documented in
[`KAGGLE_RESEARCH_EVIDENCE_RUNBOOK.md`](KAGGLE_RESEARCH_EVIDENCE_RUNBOOK.md).

## Notes

- Use `distributed.backend=nccl` for CUDA diffusion training.
- Validation and checkpoint writing run on rank 0.
- Training data is sharded with `DistributedSampler` for distributed diffusion.
- Non-diffusion stages intentionally run single-process because `main.py`
  validates multi-process distributed training only for `training.stage=diffusion`.
- Diffusion training caches frozen VQ-VAE room latents in memory by default
  (`diffusion.latent_cache_enabled=true`). This removes repeated Block-II
  encoding for real rooms and teacher-forced neighbor maps across epochs. Set
  `--no-latent-cache-enabled` or `diffusion.latent_cache_max_items: 0` for
  memory-constrained debugging runs.
