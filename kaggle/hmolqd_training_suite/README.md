# H-MOLQD Kaggle Training Suite

This folder contains the Kaggle-oriented training path for the current
repository code. It is separate from the older notebook so the full training
stack can be run from shell cells and produce clean per-run folders.

## Hardware Choice

Use **GPU T4 x2** when Kaggle offers it. This repo can use two GPUs for the
diffusion stage through `torchrun`/DDP, while VQ-VAE, VQ-VAE-2, fast-sampler,
and masked-room stages run single-process. T4 also has Tensor Cores and strong
FP16 throughput, which is a better fit for diffusion-style training than a
single P100. P100 is still a valid fallback for single-GPU runs.

## Quick Start In A Kaggle Notebook Cell

Kaggle notebook cells are Python by default. Put `%%bash` on the first line of
the cell before shell commands.

```bash
%%bash
cd /kaggle/working/KLTN
bash kaggle/hmolqd_training_suite/run_kaggle_training_suite.sh
```

Default behavior:

- profile: auto-detect, preferring `t4x2` when two CUDA devices are visible
- tokenizer: `vqvae2`
- branch: `stage_full`
- stages: VQ-VAE/VQ-VAE-2, diffusion, fast-sampler, masked-room
- output root: `/kaggle/working/hmolqd_training_suite`

To train and then run the full thesis evidence suite:

```bash
%%bash
cd /kaggle/working/KLTN
bash kaggle/hmolqd_training_suite/run_kaggle_full_research_suite.sh
```

To run only the evidence suite after checkpoints already exist:

```bash
%%bash
cd /kaggle/working/KLTN
bash kaggle/hmolqd_training_suite/run_kaggle_research_suite.sh
```

## Kaggle API Script Kernel

This folder can also be pushed as a Kaggle script kernel. Copy
`kernel-metadata.template.json` to `kernel-metadata.json`, edit the `id`, attach
the dataset if needed, and push with the accelerator flag:

```bash
kaggle kernels push \
  -p kaggle/hmolqd_training_suite \
  --accelerator NvidiaTeslaT4 \
  --timeout 43200
```

`kaggle_kernel.py` is the script entrypoint. If the full repo is not already
present in the Kaggle working directory, it clones the repo before launching the
selected suite. Use `--mode training`, `--mode evidence`, or `--mode full`.

## Recommended Paper Runs

Train the VQ-VAE-2 full stack:

```bash
%%bash
cd /kaggle/working/KLTN
TOKENIZERS="vqvae2" \
BRANCHES="stage_full" \
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

Stage-conditioning ablation using one VQ-VAE-2 tokenizer:

```bash
%%bash
cd /kaggle/working/KLTN
TOKENIZERS="vqvae2" \
BRANCHES="stage_full stage_tokens_only stage_trace_only stage_loss010 stage_loss050" \
bash kaggle/hmolqd_training_suite/run_kaggle_training_suite.sh
```

Smoke test:

```bash
%%bash
cd /kaggle/working/KLTN
QUICK=1 \
TOKENIZERS="vqvae2" \
BRANCHES="stage_full" \
bash kaggle/hmolqd_training_suite/run_kaggle_training_suite.sh
```

Full smoke test, including evidence scripts:

```bash
%%bash
cd /kaggle/working/KLTN
QUICK=1 \
TOKENIZERS="vqvae2" \
BRANCHES="stage_full" \
bash kaggle/hmolqd_training_suite/run_kaggle_full_research_suite.sh
```

## Important Environment Variables

- `DATA_DIR`: dataset root, default `Data/The Legend of Zelda`
- `OUT_ROOT`: output root, default `/kaggle/working/hmolqd_training_suite`
- `PROFILE`: `auto`, `t4x2`, `p100`, or `cpu`
- `TOKENIZERS`: space-separated `vqvae` and/or `vqvae2`
- `BRANCHES`: space-separated `base`, `stage_full`, `stage_tokens_only`,
  `stage_trace_only`, `stage_loss010`, `stage_loss050`
- `VQVAE_EPOCHS`, `DIFFUSION_EPOCHS`, `FAST_SAMPLER_EPOCHS`,
  `MASKED_ROOM_EPOCHS`: optional epoch overrides
- `BATCH_SIZE`: optional global batch override
- `VQVAE_CHECKPOINT_ROOT`: optional existing tokenizer root with
  `<root>/<tokenizer>/checkpoints/vqvae/vqvae_pretrained.pth`
- `RUN_VQVAE`, `RUN_DIFFUSION`, `RUN_FAST_SAMPLER`, `RUN_MASKED_ROOM`: set to
  `0` to skip a stage
- `RUN_CONDITIONING_LOGICNET_REPAIR`, `RUN_FIXED_GRAPH`,
  `RUN_GENERATED_GRAPH`, `RUN_ABLATION_STUDY`, `RUN_MATCHED_BUDGET`,
  `RUN_PCG_BENCHMARK`, `RUN_OOD_BLINDED`, `RUN_DESIGNER_CONTROLLABILITY`,
  `RUN_PCBS_SWEEP`, `RUN_PCBS_COMPONENT_ABLATION`: evidence-suite section
  switches; set to `0` to skip a section

## Outputs

The suite writes separate folders:

```text
/kaggle/working/hmolqd_training_suite/
  configs/
  tokenizers/
  downstream/
  research/
  logs/
  artifacts/
```

`artifacts/kaggle_training_manifest.json` indexes summaries, metrics, and
checkpoint sizes. `artifacts/hmolqd_kaggle_<profile>_artifacts.zip` contains
the manifest, configs, logs, metadata, and best/final checkpoints.
`artifacts/run_environment.json` records the selected profile, GPU inventory,
Python/PyTorch versions, tokenizers, branches, config, and data root.

The evidence suite writes `research/research_suite_manifest.json`,
`research/steps.tsv`, per-experiment result folders, and
`artifacts/hmolqd_kaggle_research_artifacts.zip`.

See [`../../docs/KAGGLE_RESEARCH_EVIDENCE_RUNBOOK.md`](../../docs/KAGGLE_RESEARCH_EVIDENCE_RUNBOOK.md)
for the full experiment matrix and artifact checklist.

## Notes

- Diffusion uses DDP only when `PROFILE=t4x2` and two CUDA devices are visible.
- Non-diffusion stages intentionally run on one GPU because `main.py` currently
  validates distributed multi-process training only for `training.stage=diffusion`.
- VQ-VAE runs already export runtime, epoch-to-best, validation loss, codebook
  utilization, and checkpoint size in `vqvae_run_summary.json`.
