# H-MOLQD Kaggle Training

Production-ready Kaggle notebook for training the H-MOLQD dungeon generator with robust checkpoint management.

## 📋 Table of Contents
- [Quick Start](#quick-start)
- [Checkpoint Management](#checkpoint-management)
- [Configuration Options](#configuration-options)
- [Output Files](#output-files)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Upload to Kaggle
1. Copy `train_h_molqd_kaggle.ipynb` to a new Kaggle notebook
2. Attach your VGLC dataset to the notebook
3. Enable GPU accelerator (recommended: T4 or P100)

### 2. Configure Training
Edit the configuration in Cell 2:

```python
CONFIG = {
    'batch_size': 32,           # Adjust based on GPU memory
    'num_epochs': 100,          # Total epochs
    'learning_rate': 1e-4,      # Learning rate
    'checkpoint_every': 5,      # Save every N epochs
    ...
}
```

### 3. Run Training
- **First time**: Execute all cells → Training starts from epoch 0
- **Resume training**: Re-run notebook → Auto-resumes from latest checkpoint

---

## Checkpoint Management

### 🔄 Auto-Resume
The notebook **automatically resumes** from the latest checkpoint if available:

```python
CONFIG['auto_resume'] = True  # Default behavior
```

When you re-run the notebook:
- ✅ Automatically finds latest checkpoint
- ✅ Restores model weights
- ✅ Restores optimizer state
- ✅ Restores learning rate scheduler
- ✅ Continues from next epoch

### 📌 Manual Resume
To resume from a specific checkpoint:

```python
CONFIG['resume_from'] = '/kaggle/working/checkpoints/checkpoint_epoch_50.pt'
CONFIG['auto_resume'] = False  # Disable auto-resume
```

### 💾 What Gets Saved

Each checkpoint contains:
- **Model weights**: Full model state dict
- **Optimizer state**: Momentum, learning rates, etc.
- **Scheduler state**: LR schedule position
- **Training metrics**: Loss history, best metrics
- **Epoch number**: For resume tracking
- **Timestamp**: When checkpoint was saved
- **Config**: Training configuration snapshot

### 📂 Checkpoint Files

Three types of checkpoints are saved:

1. **Regular checkpoints**: `checkpoint_epoch_N.pt`
   - Saved every N epochs (configurable)
   - Useful for resuming or analyzing training progression

2. **Latest checkpoint**: `checkpoint_latest.pt`
   - Always points to most recent checkpoint
   - Used for auto-resume

3. **Best checkpoint**: `checkpoint_best.pt`
   - Saved when validation loss improves
   - Your best performing model

### 🔍 Checkpoint Location

```
/kaggle/working/checkpoints/
├── checkpoint_epoch_5.pt      (37 MB)
├── checkpoint_epoch_10.pt     (37 MB)
├── checkpoint_epoch_15.pt     (37 MB)
├── checkpoint_latest.pt       (37 MB) ← Auto-resume uses this
└── checkpoint_best.pt         (37 MB)
```

---

## Configuration Options

### Training Parameters

```python
'batch_size': 32              # Training batch size
'num_epochs': 100             # Total training epochs
'learning_rate': 1e-4         # Initial learning rate
'weight_decay': 1e-5          # L2 regularization
'checkpoint_every': 5         # Save every N epochs
```

### Model Parameters

```python
'latent_dim': 64              # Latent space dimension
'num_embeddings': 512         # VQ-VAE codebook size
'num_timesteps': 1000         # Diffusion timesteps
'embedding_dim': 256          # Embedding dimension
```

### Paths

```python
'data_dir': '/kaggle/input/zelda-vglc-data'
'checkpoint_dir': '/kaggle/working/checkpoints'
'output_dir': '/kaggle/working/outputs'
```

### Resume Options

```python
'resume_from': None           # Specific checkpoint path or None
'auto_resume': True           # Auto-resume from latest
```

### Performance Options

```python
'use_amp': True               # Mixed precision training (GPU only)
'log_every': 10               # Log every N batches
'val_every': 1                # Validate every N epochs
```

---

## Output Files

After training completes, these files are available for download:

### 📦 Archive (`h_molqd_outputs.zip`)
Complete package containing everything below:

### 📊 Results
```
/kaggle/working/outputs/
├── config.json                   # Training configuration
├── training_history.json         # Full training metrics
├── training_curves.png           # Loss/LR plots
├── model_final.pt                # Final model weights
└── sample_*.png                  # Generated samples
```

### 💾 Checkpoints
```
/kaggle/working/checkpoints/
├── checkpoint_epoch_*.pt         # Regular checkpoints
├── checkpoint_latest.pt          # Latest checkpoint
└── checkpoint_best.pt            # Best model
```

### Download Instructions

**Option 1: Download Archive**
```python
# Cell 11 creates comprehensive archive
h_molqd_outputs.zip  # Download from Kaggle output panel
```

**Option 2: Individual Files**
- Navigate to "Output" tab in Kaggle
- Right-click files → Download

---

## Troubleshooting

### 🔥 Out of Memory

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
```python
# Reduce batch size
CONFIG['batch_size'] = 16  # Try 16, 8, or even 4

# Disable mixed precision (uses more memory but more stable)
CONFIG['use_amp'] = False

# Enable gradient checkpointing (if model supports it)
# model.enable_gradient_checkpointing()
```

### 🔄 Training Interrupted

**What happens**: Checkpoints are saved automatically every N epochs

**To resume**:
1. Re-run the notebook
2. Auto-resume will load latest checkpoint
3. Training continues from last saved epoch

**Manual control**:
```python
# Force resume from specific checkpoint
CONFIG['resume_from'] = '/kaggle/working/checkpoints/checkpoint_epoch_45.pt'
```

### 🆕 Start Fresh (Ignore Existing Checkpoints)

**Option 1**: Delete checkpoint directory
```python
import shutil
shutil.rmtree('/kaggle/working/checkpoints', ignore_errors=True)
```

**Option 2**: Disable auto-resume
```python
CONFIG['auto_resume'] = False
CONFIG['resume_from'] = None
```

### ⚠️ Checkpoint Not Found

**Symptoms**: `No checkpoint found - starting from scratch`

**Causes**:
- First time running (expected behavior)
- Checkpoint directory was cleared
- You specified a checkpoint path that doesn't exist

**Solution**: This is normal for first run. Training will start from epoch 0.

### 📉 Loss Not Improving

**Diagnostics**:
1. Check training curves (Cell 9)
2. Review learning rate schedule
3. Check for data loading issues

**Solutions**:
```python
# Adjust learning rate
CONFIG['learning_rate'] = 1e-3  # Try different values

# Adjust scheduler
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode='min', patience=5, factor=0.5
# )

# Check data augmentation
# May need more/less augmentation
```

### 💾 Checkpoint File Too Large

**Symptoms**: Download fails or takes too long

**Solutions**:
```python
# Save only specific epochs
CONFIG['checkpoint_every'] = 10  # Save less frequently

# Delete old checkpoints
import os
from pathlib import Path

checkpoint_dir = Path('/kaggle/working/checkpoints')
checkpoints = sorted(checkpoint_dir.glob('checkpoint_epoch_*.pt'))

# Keep only latest N checkpoints
keep_last_n = 3
for ckpt in checkpoints[:-keep_last_n]:
    os.remove(ckpt)
```

### 🔍 Verify Checkpoint Integrity

Test checkpoint save/load with `test_checkpoint_resume.ipynb`:

```bash
# Run test notebook to verify checkpoint functionality
# All tests should pass ✅
```

---

## Advanced Usage

### Multi-Stage Training

Train different components sequentially:

```python
# Stage 1: Train VQ-VAE
for epoch in range(0, 50):
    train_vqvae()
    if epoch == 49:
        save_checkpoint(...)

# Stage 2: Train Diffusion
for epoch in range(50, 100):
    train_diffusion()
    if checkpoint_manager.should_save(epoch):
        save_checkpoint(...)

# Stage 3: Joint Fine-tuning
for epoch in range(100, 150):
    train_joint()
    save_checkpoint(...)
```

### Custom Checkpoint Logic

```python
# Save checkpoint based on custom criteria
if val_loss < best_loss and val_accuracy > 0.95:
    checkpoint_manager.save_checkpoint(
        epoch=epoch,
        model=model,
        optimizer=optimizer,
        is_best=True
    )
```

### Export for Inference

```python
# Export model for inference (without optimizer state)
inference_checkpoint = {
    'model_state_dict': model.state_dict(),
    'config': CONFIG,
    'metrics': training_history[-1]
}

torch.save(
    inference_checkpoint,
    '/kaggle/working/model_inference_only.pt'
)
```

---

## Best Practices

### ✅ DO:
- Save checkpoints regularly (every 5-10 epochs)
- Keep "best" checkpoint based on validation loss
- Test checkpoint loading before long runs
- Monitor GPU memory usage
- Save config with checkpoints
- Document training runs

### ❌ DON'T:
- Save every single epoch (wastes space/time)
- Delete checkpoints during training
- Change model architecture mid-training
- Ignore GPU memory warnings
- Forget to validate periodically

---

## Questions?

### How often should I save checkpoints?
**Recommendation**: Every 5-10 epochs for training runs > 50 epochs

### Can I change the config when resuming?
**Yes**, but be careful:
- ✅ Safe: `batch_size`, `log_every`, `val_every`
- ⚠️ Careful: `learning_rate` (will override scheduler state)
- ❌ Don't: Model architecture parameters

### What if my training is going well?
**Let it run!** Checkpoints are saved automatically. You can download them anytime from the Kaggle output panel, even while training is running.

### How do I use the best model for generation?
```python
# Load best checkpoint
checkpoint = torch.load('/kaggle/working/checkpoints/checkpoint_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Generate samples
with torch.no_grad():
    samples = model.generate(num_samples=10)
```

---

## Feedback & Support

- **Issues**: Check Troubleshooting section
- **Questions**: Refer to Best Practices
- **Testing**: Run `test_checkpoint_resume.ipynb` to verify functionality

**Happy Training! 🚀**
