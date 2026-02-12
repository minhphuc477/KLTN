# Kaggle Multi-Path Checkpoint Discovery System - Implementation Summary

## Overview

This document summarizes the comprehensive multi-path checkpoint discovery system designed for Kaggle notebooks that need to resume training across multiple runs.

## Problem Solved

**Challenge**: Kaggle notebooks have 9-12 hour time limits, but training often requires days. Resuming training across multiple runs is complex because:

1. Checkpoints from previous runs are in **read-only** `/kaggle/input/` directories
2. Multiple datasets may contain different versions of checkpoints
3. Kaggle auto-versions notebook outputs with unpredictable naming
4. Manual path configuration is fragile and error-prone

**Solution**: Automated checkpoint discovery that searches ALL possible locations, validates checkpoints, selects the best one, and handles writable/read-only directory complexities.

## Implementation Location

All code is in **Cell 1.5** of `notebook/hmolqd_kaggle_training.ipynb`

## Core Components

### 1. Data Structures

**`CheckpointInfo` dataclass**:
```python
@dataclass
class CheckpointInfo:
    path: Path                    # Absolute path to checkpoint
    source_type: str              # 'working', 'input_dataset', 'notebook_output'
    source_location: str          # Human-readable description
    epoch: Optional[int]          # Training epoch
    accuracy: Optional[float]     # Validation accuracy
    solvability: Optional[float]  # Solvability metric
    file_size_mb: float          # File size
    modified_time: Optional[float] # Last modified timestamp
    is_valid: bool               # Validation result
    validation_msg: str          # Validation message or error
```

### 2. Core Functions

| Function | Purpose | Returns |
|----------|---------|---------|
| `find_checkpoint_locations()` | Discover all checkpoint directories | `Dict[str, List[Path]]` |
| `get_checkpoint_info()` | Extract & validate checkpoint metadata | `CheckpointInfo` |
| `find_best_checkpoint_across_sources()` | Find best checkpoint across all sources | `(Path, CheckpointInfo, List[CheckpointInfo])` |
| `copy_checkpoint_to_working()` | Copy from read-only to writable location | `Path` |
| `discover_and_validate_all_checkpoints()` | Comprehensive diagnostic scan | `Dict[str, List[CheckpointInfo]]` |
| `find_latest_checkpoint_multi_source()` | Find latest diffusion checkpoint | `(Path, CheckpointInfo)` |

### 3. Search Strategy

**Priority Order**:
1. `/kaggle/working/checkpoints/` - Current run (WRITABLE)
2. `/kaggle/input/*/checkpoints/` - User-uploaded datasets (READ-ONLY)
3. `/kaggle/input/notebooks/*/*/checkpoints/` - Auto-versioned outputs (READ-ONLY)
4. `/kaggle/input/*/` - Direct checkpoint files (READ-ONLY)

**Selection Logic**:
- If `prefer_metric` specified → choose checkpoint with best metric value
- Otherwise → prioritize by source type (working > input > notebook)
- Within same priority → most recent by modification time

### 4. Validation

Each checkpoint is validated before use:
- ✅ File can be loaded successfully
- ✅ Required keys are present
- ✅ Epoch value is non-negative
- ✅ Accuracy/solvability in valid range [0, 1]

Invalid checkpoints are logged but skipped.

## Integration Points

### Stage 1: VQ-VAE Training (Cell 3)

**Before**:
```python
if VQVAE_SAVE_PATH.exists():
    ckpt = validate_vqvae_checkpoint(VQVAE_SAVE_PATH)
    if ckpt:
        # Resume from single hardcoded path
```

**After**:
```python
best_ckpt_path, best_ckpt_info, all_ckpts = find_best_checkpoint_across_sources(
    checkpoint_filename='vqvae_pretrained.pth',
    required_keys=['epoch', 'model_state_dict', 'optimizer_state_dict', 'accuracy'],
    prefer_metric='accuracy'
)

if best_ckpt_path is not None:
    # Automatically finds best checkpoint from ANY source
    # Copies to working directory if needed
```

**Benefits**:
- Searches all input datasets automatically
- Selects checkpoint with highest accuracy
- Handles read-only → writable copy
- Detailed logging shows source and metrics

### Stage 2: Diffusion Training (Cell 4)

**Before**:
```python
def find_latest_checkpoint(checkpoint_dir):
    # Searches only ONE directory
    candidates = [checkpoint_dir / 'final_model.pth', ...]
```

**After**:
```python
def find_latest_checkpoint_multi_source(
    checkpoint_patterns=['final_model.pth', 'best_model.pth', 'checkpoint_*.pth'],
    required_keys=None
):
    # Searches ALL directories across all sources
    # Supports glob patterns for numbered checkpoints
```

**Benefits**:
- Finds latest checkpoint across all runs
- Supports glob patterns (`checkpoint_*.pth`)
- Validates before returning
- Auto-copies to working directory

## User Workflow

### First Run (0-9 hours)
```bash
1. Start Kaggle notebook
2. Training saves to /kaggle/working/checkpoints/
3. Download outputs at end of session
4. Upload to Kaggle Datasets as "hmolqd-run1"
```

### Resume Run (9-18 hours)
```bash
1. Add "hmolqd-run1" dataset to notebook inputs
2. Run notebook
3. System automatically:
   - Finds checkpoints in /kaggle/input/hmolqd-run1/checkpoints/
   - Validates and selects best checkpoint
   - Copies to /kaggle/working/checkpoints/
   - Resumes training from epoch N+1
```

### Subsequent Runs (18+ hours)
```bash
1. Add new dataset versions to inputs
2. System finds most recent/best checkpoint across ALL versions
3. Training continues seamlessly
```

**Zero manual configuration required!**

## Example Output

```
🔍 Searching for checkpoints across all sources...
   ✅ [working        ] working/checkpoints/vqvae_pretrained.pth
      23.4MB, epoch=50, acc=0.875 - Valid
   ✅ [input_datasets ] input/hmolqd-run1/checkpoints/vqvae_pretrained.pth
      23.1MB, epoch=45, acc=0.860 - Valid
   ✅ [input_datasets ] input/hmolqd-run2/checkpoints/vqvae_pretrained.pth
      23.5MB, epoch=55, acc=0.880 - Valid
   ❌ [notebook_outputs] input/notebooks/user/nb/vqvae_pretrained.pth
      12.1MB, epoch=?, acc=? - Load error: invalid load key

🎯 Selected checkpoint by best accuracy:
   📂 input/hmolqd-run2/checkpoints/vqvae_pretrained.pth
   📊 Epoch 55, accuracy=0.880

📋 Copying checkpoint to working directory...
   Source: /kaggle/input/hmolqd-run2/checkpoints/vqvae_pretrained.pth
   Target: /kaggle/working/checkpoints/vqvae_pretrained.pth
   ✅ Copied 23.5MB

⚡ Loaded VQ-VAE checkpoint from: input/hmolqd-run2/checkpoints
   Epoch 56, accuracy=0.880
   📋 Copied to working directory for incremental saves
   🔄 Resuming from epoch 56
   📊 Loaded 55 epochs of history
```

## Testing & Diagnostics

### Automatic Test (on Kaggle only)

Cell 1.5 automatically runs diagnostic scan when `IS_KAGGLE=True`:
```python
if IS_KAGGLE:
    discover_and_validate_all_checkpoints(show_invalid=True)
```

Shows:
- All checkpoint directories found
- All checkpoint files in each directory
- Validation status for each checkpoint
- Epoch, accuracy, solvability (if available)
- Why invalid checkpoints failed

### Manual Diagnostic

```python
# See all available checkpoint locations
locations = find_checkpoint_locations()
print(locations)

# Comprehensive scan with invalid checkpoints shown
all_ckpts = discover_and_validate_all_checkpoints(show_invalid=True)

# Test specific checkpoint
info = get_checkpoint_info(
    Path('/kaggle/input/my-dataset/checkpoints/model.pth'),
    'input_datasets',
    'input/my-dataset/checkpoints',
    required_keys=['epoch', 'model_state_dict']
)
print(f"Valid: {info.is_valid}, Message: {info.validation_msg}")
```

## Error Handling

All functions include comprehensive error handling:

1. **File loading errors**: Caught and logged, checkpoint marked invalid
2. **Missing keys**: Validated and logged, checkpoint skipped
3. **Invalid values**: Sanity-checked (epoch >= 0, accuracy in [0,1])
4. **No checkpoints found**: Returns `None`, training starts from epoch 0
5. **Copy failures**: Logged with clear error message

**No crashes** - system always degrades gracefully to fresh training if needed.

## Documentation

### Comprehensive Guide
- **`docs/KAGGLE_CHECKPOINT_RESUME_GUIDE.md`**
  - Full technical documentation
  - All functions explained with examples
  - Integration patterns
  - Troubleshooting section
  - Best practices

### Quick Reference
- **`docs/KAGGLE_CHECKPOINT_QUICK_REF.md`**
  - One-page cheat sheet
  - Quick start (3 steps)
  - Common commands
  - Troubleshooting tips

### Visual Guide
- **`docs/KAGGLE_CHECKPOINT_VISUAL_GUIDE.md`**
  - System architecture diagram
  - Search priority flow
  - Multi-run timeline
  - Example scenarios with diagrams

## Key Features

✅ **Zero Configuration**: No hardcoded paths, no manual setup
✅ **Multi-Source Search**: Finds checkpoints in ALL possible locations
✅ **Intelligent Selection**: Choose by accuracy, epoch, or priority
✅ **Validation Before Load**: Prevents crashes from corrupted checkpoints
✅ **Automatic Copy**: Handles read-only → writable transparently
✅ **Detailed Logging**: Always shows what was found and why
✅ **Error Recovery**: Graceful fallback to fresh training
✅ **Glob Pattern Support**: Find numbered checkpoints (`checkpoint_*.pth`)
✅ **Comprehensive Diagnostics**: Built-in troubleshooting tools
✅ **Well Documented**: 3 detailed guides with examples

## Testing Checklist

Before deployment, verify:

- [ ] Works with no checkpoints (fresh training)
- [ ] Works with checkpoint in working directory only
- [ ] Works with checkpoint in input dataset only
- [ ] Works with checkpoints in multiple locations (selects correct one)
- [ ] Handles invalid/corrupted checkpoints gracefully
- [ ] Copies checkpoints from input to working correctly
- [ ] `prefer_metric` selection works correctly
- [ ] Glob patterns find numbered checkpoints
- [ ] Diagnostic functions show complete information
- [ ] VQ-VAE training resumes correctly
- [ ] Diffusion training resumes correctly
- [ ] Training history loaded and merged correctly

## Future Enhancements (Optional)

Potential improvements:

1. **History Merging**: Automatically merge training histories from multiple runs
2. **Checkpoint Pruning**: Auto-delete old checkpoints to save space
3. **Cloud Sync**: Automatically upload/download from cloud storage (S3, GCS)
4. **Checksum Validation**: Verify file integrity with checksums
5. **Metadata Database**: SQLite database tracking all checkpoints
6. **Web Dashboard**: Visualize checkpoint history and metrics
7. **Email Notifications**: Alert when training completes/fails

## Performance Impact

- **Minimal overhead**: Checkpoint discovery runs once at startup (~1-2 seconds)
- **No impact on training**: Discovery happens before training loop
- **Efficient validation**: Only loads checkpoint metadata, not full model weights
- **Caching possible**: Could cache checkpoint locations between runs (not implemented)

## Maintenance

To update the system:

1. **Add new checkpoint types**: Update required_keys in function calls
2. **Change selection logic**: Modify `find_best_checkpoint_across_sources()`
3. **Add new metrics**: Extend `CheckpointInfo` dataclass
4. **Custom validation**: Extend `get_checkpoint_info()` with new checks

Code is modular and well-documented for easy maintenance.

## Conclusion

This multi-path checkpoint discovery system provides a **robust, automatic, and user-friendly** solution for resuming training across multiple Kaggle sessions. It eliminates manual path configuration, handles complex directory structures, validates checkpoints before use, and provides detailed logging for debugging.

**Result**: Users can focus on training and experiments instead of managing checkpoints manually.

## Quick Links

- Implementation: `notebook/hmolqd_kaggle_training.ipynb` (Cell 1.5, Cell 3, Cell 4)
- Full Guide: [`docs/KAGGLE_CHECKPOINT_RESUME_GUIDE.md`](KAGGLE_CHECKPOINT_RESUME_GUIDE.md)
- Quick Ref: [`docs/KAGGLE_CHECKPOINT_QUICK_REF.md`](KAGGLE_CHECKPOINT_QUICK_REF.md)
- Visual Guide: [`docs/KAGGLE_CHECKPOINT_VISUAL_GUIDE.md`](KAGGLE_CHECKPOINT_VISUAL_GUIDE.md)

---

**Implementation Date**: February 10, 2026  
**Status**: ✅ Complete and Ready for Testing  
**Testing Required**: Yes (follow checklist above)
