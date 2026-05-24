# Kaggle Dual-T4 Training

This repo now supports single-node multi-process training for the diffusion
stage with PyTorch distributed collectives.

Recommended full-suite Kaggle usage:

```bash
cd /kaggle/working/KLTN
bash kaggle/hmolqd_training_suite/run_kaggle_training_suite.sh
```

The suite trains tokenizers and downstream stages into separate folders under
`/kaggle/working/hmolqd_training_suite`. It defaults to the VQ-VAE-2
stage-conditioned stack and uses dual-T4 DDP for the diffusion stage when two
CUDA devices are visible.

Use T4 x2 when available. Kaggle's kernel accelerator list includes both
`NvidiaTeslaT4` and `NvidiaTeslaP100`; the repo benefits from two visible T4s
because only diffusion currently supports multi-process DDP. T4 also has
Tensor Cores for mixed-precision training, while P100 remains a good fallback
single-GPU profile with 16 GB HBM2.

Current suite folder:

```text
kaggle/hmolqd_training_suite/
```

Legacy single-stage usage:

```bash
bash scripts/run_kaggle_t4x2_train.sh \
  --config configs/zelda_hmolqd.yaml \
  --stage diffusion
```

Equivalent manual command:

```bash
python main.py train \
  --config configs/zelda_hmolqd.yaml \
  --stage diffusion \
  --distributed-enabled \
  --distributed-backend nccl \
  --nproc-per-node 2 \
  --master-port 29500 \
  --device cuda
```

Notes:

- Use `distributed.backend: nccl` for CUDA training.
- Validation and checkpoint writing run on rank 0.
- Training data is sharded with `DistributedSampler`; each GPU sees a different
  slice of the epoch.
- The current distributed wiring targets the diffusion stage. Other training
  stages remain single-process unless extended in the same pattern.


cd /kaggle/working/KLTN
bash kaggle/hmolqd_training_suite/run_kaggle_training_suite.sh