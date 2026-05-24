# H-MOLQD Kaggle Workflows

Use [`hmolqd_training_suite/`](hmolqd_training_suite/) for current training.
The older notebooks are kept under [`legacy/`](legacy/) for provenance only.

## Current Full Training Suite

In a Kaggle notebook cell after the repo is available at `/kaggle/working/KLTN`:

```bash
cd /kaggle/working/KLTN
bash kaggle/hmolqd_training_suite/run_kaggle_training_suite.sh
```

The suite auto-detects visible GPUs, prefers the dual-T4 profile when two CUDA
devices are available, and falls back to a single-GPU profile otherwise.

Useful smoke run:

```bash
cd /kaggle/working/KLTN
QUICK=1 TOKENIZERS="vqvae2" BRANCHES="stage_full" \
  bash kaggle/hmolqd_training_suite/run_kaggle_training_suite.sh
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

## Legacy Notebooks

- [`legacy/train_h_molqd_kaggle.ipynb`](legacy/train_h_molqd_kaggle.ipynb)
- [`legacy/test_checkpoint_resume.ipynb`](legacy/test_checkpoint_resume.ipynb)

These notebooks predate the current VQ-VAE-2 / downstream suite and should not
be used for new paper runs.
