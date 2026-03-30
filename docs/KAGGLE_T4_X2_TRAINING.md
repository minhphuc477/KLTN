# Kaggle Dual-T4 Training

This repo now supports single-node multi-process training for the diffusion
stage with PyTorch distributed collectives.

Recommended Kaggle usage:

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
