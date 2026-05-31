# H-MOLQD Kaggle Workflows

Use [`hmolqd_training_suite/`](hmolqd_training_suite/) for current training.
The older notebooks are kept under [`legacy/`](legacy/) for provenance only.

## Current Full Training Suite

In a Kaggle notebook cell after the repo is available at `/kaggle/working/KLTN`:

```bash
%%bash
cd /kaggle/working/KLTN
bash kaggle/hmolqd_training_suite/run_kaggle_training_suite.sh
```

The suite auto-detects visible GPUs, prefers the dual-T4 profile when two CUDA
devices are available, and falls back to a single-GPU profile otherwise.

To run training plus the thesis evidence matrix:

```bash
%%bash
cd /kaggle/working/KLTN
bash kaggle/hmolqd_training_suite/run_kaggle_full_research_suite.sh
```

To run only the post-training evidence matrix after checkpoints exist:

```bash
%%bash
cd /kaggle/working/KLTN
bash kaggle/hmolqd_training_suite/run_kaggle_research_suite.sh
```

Useful smoke run:

```bash
%%bash
cd /kaggle/working/KLTN
QUICK=1 TOKENIZERS="vqvae2" BRANCHES="stage_full" \
  bash kaggle/hmolqd_training_suite/run_kaggle_training_suite.sh
```

Useful full smoke run:

```bash
%%bash
cd /kaggle/working/KLTN
QUICK=1 TOKENIZERS="vqvae2" BRANCHES="stage_full" \
  bash kaggle/hmolqd_training_suite/run_kaggle_full_research_suite.sh
```

## Kaggle API Script Kernel

Copy the template metadata, edit the `id`, and push the suite folder:

```bash
cp kaggle/hmolqd_training_suite/kernel-metadata.template.json \
   kaggle/hmolqd_training_suite/kernel-metadata.json

kaggle kernels push \
  -p kaggle/hmolqd_training_suite \
  --accelerator NvidiaTeslaT4 \
  --timeout 43200
```

The script entrypoint is
[`hmolqd_training_suite/kaggle_kernel.py`](hmolqd_training_suite/kaggle_kernel.py).
If it is launched without the full repo present, it clones the repo into
`/kaggle/working/KLTN` before running the shell suite. Attach the Zelda dataset
or set `DATA_DIR`/`--data-dir` to the dataset path exposed by Kaggle.
Use `--mode training`, `--mode evidence`, or `--mode full` to select which
suite the kernel entrypoint runs.

## Legacy Notebooks

- [`legacy/train_h_molqd_kaggle.ipynb`](legacy/train_h_molqd_kaggle.ipynb)
- [`legacy/test_checkpoint_resume.ipynb`](legacy/test_checkpoint_resume.ipynb)

These notebooks predate the current VQ-VAE-2 / downstream suite and should not
be used for new paper runs.
