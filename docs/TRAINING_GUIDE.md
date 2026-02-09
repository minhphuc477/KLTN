# H-MOLQD Training Guide

**Two-Stage Training Pipeline for Dungeon Generation**

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Data Pipeline](#data-pipeline)
3. [Stage 1: VQ-VAE Pre-training](#stage-1-vq-vae-pre-training)
4. [Stage 2: Latent Diffusion Training](#stage-2-latent-diffusion-training)
5. [Checkpoint Management](#checkpoint-management)
6. [Kaggle GPU Training](#kaggle-gpu-training)
7. [Known Limitations](#known-limitations)
8. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

H-MOLQD generates Zelda-style dungeon layouts using a 5-block architecture:

```
Block I:   MissionGrammar    → Mission graph (rooms + lock/key ordering)
Block II:  SemanticVQVAE     → Discrete latent representation of dungeon grids
Block III: ConditionEncoder  → Graph-aware conditioning vector (dual-stream)
Block IV:  LatentDiffusion   → Denoising in VQ-VAE latent space
Block V:   LogicNet          → Differentiable solvability guidance
```

**Training is two-stage:**

```
Stage 1:  Train VQ-VAE (Block II) to reconstruct dungeon grids
              ↓ produces checkpoints/vqvae_pretrained.pth
Stage 2:  Train Diffusion (Block IV) + LogicNet (Block V) in VQ-VAE latent space
              ↓ produces checkpoints/final_model.pth
```

---

## Data Pipeline

### Source Data
- **18 VGLC stitched dungeons** from `data/The Legend of Zelda/`
- Dungeons 1–9, each with 2 quest variants
- Loaded via `ZeldaDungeonAdapter` from `src/data/zelda_core.py`

### Normalization (CRITICAL)
All training code uses a **fixed constant** for normalization:

```
Normalize:   tile_ids / 43   → float [0, 1]
Denormalize: float * 43      → round → clamp(0, 43) → integer tile IDs
One-hot:     F.one_hot(tile_ids, num_classes=44) → [B, 44, H, W]
```

- `43 = TileID.PUZZLE`, the highest valid tile ID
- This ensures an **exact round-trip** for all dungeons regardless of which tiles appear
- Defined in `src/core/definitions.py` as `TileID` IntEnum (VOID=0 through PUZZLE=43)

### Grid Format
- VQ-VAE input: `[B, 44, H, W]` one-hot encoded tile classes
- Diffusion operates on VQ-VAE latent space: `[B, 64, H', W']`
- VQ-VAE has **4× spatial downsampling** (2 stride-2 convolutions in encoder)
- Dungeon grids are padded to `128×88` (max observed size)

---

## Stage 1: VQ-VAE Pre-training

### Purpose
Train the Semantic VQ-VAE to faithfully reconstruct dungeon grids using a discrete codebook. The diffusion model will later operate in this learned latent space.

### Command
```bash
python -m src.train_vqvae \
    --data-dir "data/The Legend of Zelda" \
    --epochs 300 \
    --batch-size 4 \
    --lr 3e-4 \
    --codebook-size 512 \
    --latent-dim 64 \
    --save-dir checkpoints \
    --verbose
```

### Key Hyperparameters
| Parameter | Default | Notes |
|-----------|---------|-------|
| `--epochs` | 300 | Full convergence; 50–100 sufficient for >95% accuracy |
| `--batch-size` | 4 | Small dataset (18 samples), upsampled to 64/epoch |
| `--lr` | 3e-4 | Adam learning rate |
| `--codebook-size` | 512 | Number of discrete latent codes |
| `--latent-dim` | 64 | Dimension of each codebook vector |
| `--min-samples-per-epoch` | 64 | Upsamples 18 dungeons to 64 per epoch |

### Architecture Details
- **Encoder**: Conv2d → 3-level (×1, ×2, ×4 channel multipliers) with ResBlocks → Stride-2 downsampling at levels 0 and 1
- **Vector Quantizer**: 512 entries, 64-dim, EMA codebook updates, dead code reset every 100 batches
- **Decoder**: Mirror of encoder with ConvTranspose2d upsampling + `F.interpolate` for exact output size
- **Total parameters**: ~31M
- **Reconstruction loss**: Weighted cross-entropy (rare tiles weighted 5×)

### Expected Training Metrics
| Epoch | Loss | Accuracy | Perplexity | Notes |
|-------|------|----------|------------|-------|
| 1 | ~2.3 | ~57% | ~4 | Initial learning |
| 5 | ~0.6 | ~90% | ~8 | Rapid improvement |
| 7 | ~0.5 | ~93% | ~62 | After codebook reset (484/512 dead codes) |
| 15 | ~0.2 | ~94% | ~88 | Good codebook utilization |
| 20 | ~0.14 | ~96% | ~76 | Near convergence |

### Output
- `checkpoints/vqvae_pretrained.pth` — best model (lowest loss)
- `checkpoints/vqvae_epochNNNN.pth` — periodic checkpoints every 50 epochs
- `checkpoints/vqvae_training_history.json` — full training log

---

## Stage 2: Latent Diffusion Training

### Purpose
Train the latent diffusion model to generate dungeon layouts in VQ-VAE latent space, with LogicNet providing differentiable solvability guidance.

### Command
```bash
python -m src.train_diffusion \
    --data-dir "data/The Legend of Zelda" \
    --vqvae-checkpoint checkpoints/vqvae_pretrained.pth \
    --epochs 100 \
    --batch-size 4 \
    --lr 1e-4 \
    --alpha-logic 0.1 \
    --guidance-scale 1.0 \
    --checkpoint-dir checkpoints \
    --verbose
```

### Key Hyperparameters
| Parameter | Default | Notes |
|-----------|---------|-------|
| `--vqvae-checkpoint` | None | **REQUIRED** — path to pretrained VQ-VAE |
| `--epochs` | 100 | Full training |
| `--lr` | 1e-4 | AdamW with cosine annealing |
| `--alpha-logic` | 0.1 | Weight for LogicNet solvability loss |
| `--guidance-scale` | 1.0 | Gradient guidance strength at inference |

### Training Strategy
1. **VQ-VAE is frozen** — only provides latent encoding/decoding
2. **Diffusion loss**: Standard ε-prediction with cosine noise schedule (1000 timesteps)
3. **LogicNet loss**: Computed on real `z_0` latents (detached from VQ-VAE graph)
4. **EMA model**: Exponential moving average of diffusion weights (decay=0.9999) used for validation/sampling
5. **Warmup**: First 5 epochs use random conditioning (unconditional baseline)
6. **Curriculum**: Epochs 5–10 use simple 3-node chain graphs; after that, full 5–12 node graphs

### Conditioning Pipeline
Training uses **synthetic** 5-dim graph features (not real VGLC dungeon graphs):
```
[is_start, feature_1, feature_2, feature_3, has_triforce]
```
The condition encoder's `GlobalStreamEncoder` processes these through a GNN to produce `[B, 256]` conditioning vectors.

### Output
- `checkpoints/final_model.pth` — final checkpoint (contains ALL model weights)
- `checkpoints/best_model.pth` — best by validation solvability
- `checkpoints/checkpoint_epoch_NNNN.pth` — periodic saves

### Checkpoint Contents
```python
{
    'epoch': int,
    'global_step': int,
    'vqvae_state_dict': ...,      # VQ-VAE weights (frozen copy)
    'diffusion_state_dict': ...,   # U-Net + noise schedule
    'ema_diffusion_state_dict': ...,  # EMA copy for stable sampling
    'condition_encoder_state_dict': ...,
    'logic_net_state_dict': ...,   # Also inside diffusion_state_dict
    'optimizer_state_dict': ...,
    'scheduler_state_dict': ...,
    'config': dict,
    'metrics': dict,
    'schedule_type': str,          # 'cosine' or 'linear'
}
```

---

## Checkpoint Management

### Resume Training
```bash
# VQ-VAE resume
python -m src.train_vqvae --resume checkpoints/vqvae_pretrained.pth ...

# Diffusion — manual resume: load checkpoint in DiffusionTrainer.load_checkpoint()
```

### Using Checkpoints in GUI
The GUI (`gui_runner.py`) loads from `checkpoints/final_model.pth`:
1. Loads diffusion (prefers EMA weights)
2. Loads VQ-VAE from main checkpoint, or falls back to `checkpoints/vqvae_pretrained.pth`
3. Loads condition encoder
4. LogicNet is loaded as a submodule of `diffusion.guidance`

---

## Kaggle GPU Training

For GPU-accelerated training on Kaggle, use the provided notebook:
`notebook/hmolqd_kaggle_training.ipynb`

**Key benefits:**
- T4/P100 GPU: ~10× faster than CPU training
- Full 300-epoch VQ-VAE + 100-epoch Diffusion in ~1–2 hours total
- Automatic checkpoint saving and download

See the notebook for complete instructions.

---

## Known Limitations

### 1. Synthetic Conditioning (HIGH)
Training uses randomly-generated 5-dim graph features, not real VGLC dungeon topologies. This means:
- The condition encoder learns to process graph structures but hasn't seen real Zelda-specific patterns
- At inference, MissionGrammar produces 12-dim features that are manually projected to 5-dim
- Conditioning influences generation but doesn't capture true Zelda dungeon semantics

### 2. LogicNet Graph Losses ≈ 0 (MEDIUM)
During `train_step()`, LogicNet receives only the latent `z_0` tensor without graph topology data. This means:
- `LogicNet.forward(z)` computes tile-level losses (path existence, key-lock ordering)
- Graph-level losses that require `graph_data` are effectively zero
- The solvability score is approximate, based on learned spatial patterns in latents

### 3. Node Feature Dimension Mismatch (MEDIUM)
- Training: `node_feature_dim=5` (synthetic features)
- Grammar output: 12-dim (8 NodeType one-hot + 2 position + 2 extra)
- GUI manually projects 12→5, dropping PUZZLE/ITEM/EMPTY information
- This is acceptable given training never saw 12-dim features

### 4. Small Dataset
- Only 18 training samples (9 dungeons × 2 quests)
- Upsampled to 64 per epoch with replacement
- Data augmentation could improve generalization (not currently implemented)

---

## Troubleshooting

### "No dungeon samples found"
- Check that `data/The Legend of Zelda/Processed/` contains the VGLC text files
- Ensure `use_vglc=True` is set (default for both training scripts)

### VQ-VAE producing noise
- Check that normalization uses fixed `÷43` (not per-sample max)
- Verify `grids_to_onehot` multiplies by `43` to recover tile IDs
- Ensure codebook is not collapsed (check perplexity > 10)

### Low diffusion quality
- Ensure VQ-VAE is well-trained first (>95% accuracy)
- Pass `--vqvae-checkpoint` to diffusion training
- Check that VQ-VAE weights are frozen during diffusion training

### GPU out of memory
- Reduce `--batch-size` to 2
- Model is ~31M params (VQ-VAE) + diffusion — should fit in 8GB VRAM easily

### Checkpoint not loading
- Ensure checkpoint was saved with `schedule_type` field
- Check that LogicNet is wired into `diffusion.guidance` before loading
