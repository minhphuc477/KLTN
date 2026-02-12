# Kaggle Multi-Path Checkpoint Discovery System

## Overview

This system enables **robust checkpoint resumption** across Kaggle's complex directory structure, allowing you to seamlessly continue training from previous notebook runs without manual checkpoint management.

## Problem Statement

Kaggle notebooks face unique challenges for long-running training:
- **Time limits**: Notebooks auto-stop after 9-12 hours
- **Restart from scratch**: Without checkpointing, progress is lost
- **Complex filesystem**: Checkpoints can be in multiple locations
- **Manual dataset versioning**: Users upload outputs as new datasets

## Kaggle Directory Structure

```
/kaggle/
├── working/                    # ✅ WRITABLE - Current run outputs
│   └── checkpoints/           # Checkpoints saved during THIS run
├── input/                     # ❌ READ-ONLY - Input datasets
│   ├── <dataset-name-1>/     # User-uploaded dataset
│   │   └── checkpoints/      # Previous run checkpoints (uploaded manually)
│   ├── <dataset-name-2>/     # Another dataset
│   └── notebooks/            # Auto-generated from notebook outputs
│       └── <user>/
│           └── <notebook-name>/
│               └── checkpoints/  # Checkpoints from previous notebook versions
└── tmp/                       # Temporary files (cleared on restart)
```

## Multi-Path Search Strategy

### Search Priority

The checkpoint discovery system searches in this order:

1. **`/kaggle/working/checkpoints/`** (Current run)
   - Highest priority for **saving** new checkpoints
   - Most recent training state
   - **Writable**: Can be updated during training

2. **`/kaggle/input/*/checkpoints/`** (User-uploaded datasets)
   - Previous runs uploaded as datasets
   - User controls versioning
   - **Read-only**: Must copy to working directory

3. **`/kaggle/input/notebooks/*/*/checkpoints/`** (Auto-versioned outputs)
   - Kaggle automatically versions notebook outputs
   - Historical checkpoints from older runs
   - **Read-only**: Must copy to working directory

4. **`/kaggle/input/*/`** (Direct checkpoint files)
   - Fallback for flat dataset uploads
   - User may have uploaded `.pth` files directly
   - **Read-only**: Must copy to working directory

## Key Functions

### 1. `find_checkpoint_locations()`

**Purpose**: Discover all directories that might contain checkpoints.

**Returns**: Dictionary mapping location types to paths.

```python
locations = find_checkpoint_locations()
# {
#   'working': [Path('/kaggle/working/checkpoints')],
#   'input_datasets': [Path('/kaggle/input/my-training-outputs/checkpoints')],
#   'notebook_outputs': [Path('/kaggle/input/notebooks/user/notebook/checkpoints')],
#   'direct_files': []
# }
```

### 2. `get_checkpoint_info()`

**Purpose**: Extract and validate metadata from a checkpoint file.

**Returns**: `CheckpointInfo` dataclass with:
- `path`: Absolute path to checkpoint
- `source_type`: Where it came from (working/input_datasets/etc.)
- `source_location`: Human-readable location
- `epoch`: Training epoch (if available)
- `accuracy`: Validation accuracy (if available)
- `solvability`: Solvability metric (if available)
- `is_valid`: Whether checkpoint passed validation
- `validation_msg`: Validation result or error message

```python
info = get_checkpoint_info(
    Path('/kaggle/input/hmolqd/checkpoints/vqvae_pretrained.pth'),
    source_type='input_datasets',
    source_location='input/hmolqd/checkpoints',
    required_keys=['epoch', 'model_state_dict', 'accuracy']
)
# CheckpointInfo(
#   path=Path('/kaggle/input/.../vqvae_pretrained.pth'),
#   epoch=45,
#   accuracy=0.875,
#   is_valid=True,
#   validation_msg='Valid'
# )
```

### 3. `find_best_checkpoint_across_sources()`

**Purpose**: Find the best checkpoint across ALL available sources.

**Selection Logic**:
- If `prefer_metric` specified → choose checkpoint with best metric value
- Otherwise → priority by source type (working > input > notebook)
- Within same priority → most recent by modification time

```python
best_path, best_info, all_valid = find_best_checkpoint_across_sources(
    checkpoint_filename='vqvae_pretrained.pth',
    required_keys=['epoch', 'model_state_dict', 'accuracy'],
    prefer_metric='accuracy'  # Choose checkpoint with highest accuracy
)

if best_path:
    print(f"Loading checkpoint from: {best_info.source_location}")
    print(f"Epoch {best_info.epoch}, accuracy={best_info.accuracy:.3f}")
    checkpoint = torch.load(best_path, map_location='cpu')
```

**Example Output**:
```
🔍 Searching for checkpoints across all sources...
   ✅ [working        ] working/checkpoints/vqvae_pretrained.pth
      23.4MB, epoch=45, acc=0.875 - Valid
   ✅ [input_datasets ] input/hmolqd/checkpoints/vqvae_pretrained.pth
      23.1MB, epoch=40, acc=0.860 - Valid
   ✅ [notebook_outputs] input/notebooks/user/hmolqd-training/vqvae_pretrained.pth
      22.8MB, epoch=30, acc=0.840 - Valid

🎯 Selected checkpoint by best accuracy:
   📂 working/checkpoints/vqvae_pretrained.pth
   📊 Epoch 45, accuracy=0.875
```

### 4. `copy_checkpoint_to_working()`

**Purpose**: Copy checkpoint from read-only input to writable working directory.

**Why needed**: Input datasets are read-only; training must save to working directory.

```python
# Checkpoint found in read-only input dataset
source = Path('/kaggle/input/hmolqd/checkpoints/vqvae_pretrained.pth')

# Copy to working directory so we can overwrite with better checkpoints
working_path = copy_checkpoint_to_working(source, 'vqvae_pretrained.pth')
# Returns: Path('/kaggle/working/checkpoints/vqvae_pretrained.pth')

# Now training can save improved checkpoints to working_path
```

### 5. `discover_and_validate_all_checkpoints()`

**Purpose**: Comprehensive scan for debugging and exploration.

**Use case**: Understand what checkpoints are available and why they might be invalid.

```python
all_ckpts = discover_and_validate_all_checkpoints(show_invalid=True)
```

**Example Output**:
```
🔎 Comprehensive checkpoint scan:
======================================================================

📁 WORKING:
   /kaggle/working/checkpoints
      ✅ vqvae_pretrained.pth (23.4MB)
         Epoch 45, acc=0.875
      ✅ checkpoint_0050.pth (48.2MB)
         Epoch 50, solv=0.720

📁 INPUT DATASETS:
   /kaggle/input/hmolqd/checkpoints
      ✅ vqvae_pretrained.pth (23.1MB)
         Epoch 40, acc=0.860
      ❌ corrupted_checkpoint.pth (15.2MB)
         ⚠️  Load error: invalid load key

📁 NOTEBOOK OUTPUTS:
   /kaggle/input/notebooks/user123/hmolqd-training/checkpoints
      ✅ final_model.pth (48.0MB)
         Epoch 100, solv=0.750
```

## Integration Examples

### VQ-VAE Training (Stage 1)

```python
# Multi-path checkpoint discovery
best_ckpt_path, best_ckpt_info, all_ckpts = find_best_checkpoint_across_sources(
    checkpoint_filename='vqvae_pretrained.pth',
    required_keys=['epoch', 'model_state_dict', 'optimizer_state_dict', 'accuracy'],
    prefer_metric='accuracy'  # Choose checkpoint with highest accuracy
)

if best_ckpt_path is not None:
    # Load the best checkpoint found
    ckpt = torch.load(best_ckpt_path, map_location='cpu', weights_only=False)
    resume_epoch = ckpt['epoch'] + 1
    vqvae_history = ckpt.get('history', [])
    
    print(f"⚡ Loaded VQ-VAE checkpoint from: {best_ckpt_info.source_location}")
    print(f"   Epoch {resume_epoch}, accuracy={ckpt['accuracy']:.3f}")
    
    # Copy to working directory if from read-only input
    if IS_KAGGLE and best_ckpt_info.source_type != 'working':
        VQVAE_SAVE_PATH = copy_checkpoint_to_working(best_ckpt_path, 'vqvae_pretrained.pth')
        print(f"   📋 Copied to working directory for incremental saves")
    else:
        VQVAE_SAVE_PATH = best_ckpt_path
    
    # Resume model state
    vqvae_model.load_state_dict(ckpt['model_state_dict'])
    trainer.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
else:
    print("ℹ️  No existing VQ-VAE checkpoint found - starting fresh training")
```

### Diffusion Training (Stage 2)

```python
# Search for numbered checkpoints with glob pattern
latest_ckpt_path, latest_ckpt_info = find_latest_checkpoint_multi_source(
    checkpoint_patterns=['final_model.pth', 'best_model.pth', 'checkpoint_*.pth'],
    required_keys=['epoch', 'diffusion_state_dict', 'ema_diffusion_state_dict']
)

if latest_ckpt_path is not None:
    # Copy to working directory if from input dataset
    if IS_KAGGLE and latest_ckpt_info.source_type != 'working':
        working_ckpt_path = copy_checkpoint_to_working(latest_ckpt_path, latest_ckpt_path.name)
        print(f"📋 Copied to working directory for incremental saves")
    
    # Load checkpoint
    diff_trainer.load_checkpoint(str(latest_ckpt_path))
    start_epoch = diff_trainer.epoch + 1
    print(f"🔄 Resuming from epoch {start_epoch}")
else:
    print("ℹ️  No existing diffusion checkpoint found - starting fresh training")
```

## Workflow: Resume Training Across Runs

### Scenario: Multi-day training with 9-hour Kaggle limit

**Day 1 (0-9 hours)**:
1. Start fresh training
2. Checkpoints saved to `/kaggle/working/checkpoints/`
3. Download outputs at end of session

**Day 2 (9-18 hours)**:
1. Upload previous outputs as Kaggle dataset (name: `hmolqd-run1`)
2. Add dataset to notebook inputs
3. Notebook automatically finds checkpoints in `/kaggle/input/hmolqd-run1/checkpoints/`
4. Copies to `/kaggle/working/checkpoints/` (writable)
5. Resumes training from epoch 50
6. New checkpoints saved to working directory
7. Download outputs at end

**Day 3 (18-27 hours)**:
1. Upload Day 2 outputs as new dataset version
2. System finds latest checkpoint (epoch 100)
3. Resumes from there
4. Training continues...

### Manual Dataset Upload Steps

1. **Download Kaggle outputs**:
   - Go to your notebook
   - Click "Output" tab
   - Download all files (includes `checkpoints/`)

2. **Create/Update dataset**:
   - Go to [kaggle.com/datasets](https://www.kaggle.com/datasets)
   - Click "+ New Dataset" or "New Version" (for existing)
   - Upload the downloaded files
   - Preserve directory structure (`checkpoints/` folder)

3. **Add to notebook**:
   - Open your notebook
   - Right sidebar → "Add Data"
   - Search for your dataset
   - Click "Add"

4. **Run notebook**:
   - The system automatically finds and loads checkpoints
   - No code changes needed!

## Best Practices

### ✅ DO:

1. **Preserve directory structure** when uploading datasets
   ```
   my-dataset/
   ├── checkpoints/
   │   ├── vqvae_pretrained.pth
   │   ├── checkpoint_0050.pth
   │   └── final_model.pth
   └── output/
       └── generated_dungeons/
   ```

2. **Use descriptive dataset names**
   - Good: `hmolqd-training-run1`, `hmolqd-epoch100-v2`
   - Bad: `untitled-dataset`, `my-data`

3. **Version your datasets**
   - Use Kaggle's dataset versioning feature
   - Keeps history of training progress

4. **Download checkpoints regularly**
   - Don't rely solely on Kaggle storage
   - Keep local backups of important checkpoints

5. **Use atomic saves** (already implemented in notebook)
   ```python
   # Save to temp file first, then rename (prevents corruption)
   temp_path = checkpoint_dir / f".{checkpoint_name}.tmp"
   torch.save(state, temp_path)
   temp_path.rename(checkpoint_path)
   ```

### ❌ DON'T:

1. **Don't hardcode checkpoint paths**
   ```python
   # Bad
   ckpt = torch.load('/kaggle/input/my-dataset/checkpoint.pth')
   
   # Good
   best_path, _, _ = find_best_checkpoint_across_sources('checkpoint.pth')
   ckpt = torch.load(best_path)
   ```

2. **Don't save directly to input directories**
   ```python
   # Bad - will fail (read-only)
   torch.save(state, '/kaggle/input/my-dataset/checkpoint.pth')
   
   # Good - save to working directory
   torch.save(state, '/kaggle/working/checkpoints/checkpoint.pth')
   ```

3. **Don't assume checkpoint locations**
   - Always use the discovery functions
   - They handle different Kaggle dataset naming patterns

4. **Don't skip checkpoint validation**
   - The system validates before loading
   - Prevents crashes from corrupted files

5. **Don't upload gigantic checkpoints**
   - Use compact checkpoints for Kaggle (already implemented)
   - Strip unnecessary state (old optimizer states, etc.)

## Troubleshooting

### Checkpoint not found

**Symptom**: `ℹ️  No existing checkpoint found - starting fresh training`

**Causes**:
1. Dataset not added to notebook inputs
2. Wrong directory structure in dataset
3. Checkpoint filename mismatch

**Solution**:
```python
# Run diagnostic scan
discover_and_validate_all_checkpoints(show_invalid=True)
# Shows all available checkpoints and why they might be invalid
```

### Invalid checkpoint

**Symptom**: `⚠️  Skipping invalid checkpoint: Missing keys: ['model_state_dict']`

**Causes**:
1. Incomplete checkpoint save (interrupted)
2. Version mismatch (old checkpoint format)
3. Corrupted file (partial upload/download)

**Solution**:
- Check checkpoint keys: `torch.load(path, map_location='cpu').keys()`
- Re-download from source if corrupted
- Use previous version of dataset if available

### Training not resuming correctly

**Symptom**: Training starts from epoch 0 despite checkpoint found

**Causes**:
1. Checkpoint loaded but state not applied
2. Model architecture mismatch
3. Exception during load_state_dict

**Debug**:
```python
try:
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    print(f"✅ Successfully loaded state")
except Exception as e:
    print(f"❌ Failed to load state: {e}")
    # Model architecture likely changed
```

### Out of disk space on Kaggle

**Symptom**: `OSError: [Errno 28] No space left on device`

**Solutions**:
1. Use checkpoint pruning (keep only N most recent):
   ```python
   _prune_checkpoints(checkpoint_dir, keep=3, pattern='checkpoint_*.pth')
   ```

2. Use compact checkpoints (exclude optimizer/scheduler for interim saves):
   ```python
   COMPACT_CHECKPOINTS = True  # Already set for IS_KAGGLE
   ```

3. Clean up output files:
   ```python
   # Remove old generated samples, training curves, etc.
   import shutil
   shutil.rmtree('/kaggle/working/output', ignore_errors=True)
   ```

## Advanced: Merging Training Histories

If you want to combine training histories from multiple runs:

```python
def merge_training_histories(checkpoint_dir: Path) -> List[Dict]:
    """Combine histories from previous runs with current run."""
    merged_history = []
    
    # Find all history files across sources
    locations = find_checkpoint_locations()
    history_files = []
    
    for source_type, dirs in locations.items():
        for d in dirs:
            history_path = d / 'vqvae_history.json'
            if history_path.exists():
                history_files.append((history_path, source_type))
    
    # Sort by source priority
    priority = {'working': 0, 'input_datasets': 1, 'notebook_outputs': 2}
    history_files.sort(key=lambda x: priority.get(x[1], 99))
    
    # Merge histories (deduplicate by epoch)
    seen_epochs = set()
    for history_path, source in history_files:
        with open(history_path, 'r') as f:
            history = json.load(f)
            for record in history:
                epoch = record.get('epoch', -1)
                if epoch not in seen_epochs:
                    merged_history.append(record)
                    seen_epochs.add(epoch)
    
    # Sort by epoch
    merged_history.sort(key=lambda x: x.get('epoch', 0))
    return merged_history

# Usage
merged = merge_training_histories(CHECKPOINT_DIR)
print(f"Total training history: {len(merged)} epochs across all runs")
```

## Summary

The multi-path checkpoint discovery system provides:

- ✅ **Automatic resumption** from previous runs
- ✅ **No manual path configuration** needed
- ✅ **Intelligent checkpoint selection** (best accuracy, most recent epoch, etc.)
- ✅ **Validation before loading** (prevents crashes)
- ✅ **Support for multiple checkpoint sources** (working dir, input datasets, notebook outputs)
- ✅ **Seamless copy from read-only to writable** locations
- ✅ **Comprehensive diagnostic tools** for troubleshooting

**Key takeaway**: Upload your notebook outputs as Kaggle datasets, add them to your notebook inputs, and the system automatically finds and resumes from the best checkpoint. No code changes required!
