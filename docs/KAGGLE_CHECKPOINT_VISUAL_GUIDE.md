# Kaggle Checkpoint Discovery Flow - Visual Guide

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    KAGGLE NOTEBOOK ENVIRONMENT                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │         Multi-Path Checkpoint Discovery System          │  │
│  └─────────────────────────────────────────────────────────┘  │
│                            │                                    │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │     find_checkpoint_locations()                         │  │
│  │     • Scans all /kaggle/ directories                    │  │
│  │     • Returns: {working, input_datasets, notebook_outputs}│ │
│  └─────────────────────────────────────────────────────────┘  │
│                            │                                    │
│           ┌────────────────┼────────────────┐                  │
│           ▼                ▼                ▼                  │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐          │
│  │   Working    │ │    Input     │ │   Notebook   │          │
│  │  Directory   │ │   Datasets   │ │   Outputs    │          │
│  │              │ │              │ │              │          │
│  │ /kaggle/     │ │ /kaggle/     │ │ /kaggle/     │          │
│  │ working/     │ │ input/       │ │ input/       │          │
│  │ checkpoints/ │ │ my-dataset/  │ │ notebooks/   │          │
│  │              │ │ checkpoints/ │ │ user/nb/     │          │
│  │ ✅ WRITABLE  │ │ ❌ READ-ONLY │ │ ❌ READ-ONLY │          │
│  └──────────────┘ └──────────────┘ └──────────────┘          │
│           │                │                │                  │
│           └────────────────┼────────────────┘                  │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │     get_checkpoint_info()                               │  │
│  │     • Loads checkpoint metadata                         │  │
│  │     • Validates required keys                           │  │
│  │     • Returns: CheckpointInfo (epoch, accuracy, etc.)   │  │
│  └─────────────────────────────────────────────────────────┘  │
│                            │                                    │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │     find_best_checkpoint_across_sources()               │  │
│  │     • Compares all valid checkpoints                    │  │
│  │     • Selects by metric or priority                     │  │
│  │     • Returns: best_path, best_info                     │  │
│  └─────────────────────────────────────────────────────────┘  │
│                            │                                    │
│                    ┌───────┴───────┐                           │
│                    ▼               ▼                           │
│        ┌──────────────────┐ ┌──────────────────┐             │
│        │  From Working    │ │  From Input      │             │
│        │  (Already        │ │  (Need to Copy)  │             │
│        │   writable)      │ │                  │             │
│        └──────────────────┘ └──────────────────┘             │
│                    │               │                           │
│                    │               ▼                           │
│                    │    ┌─────────────────────┐               │
│                    │    │ copy_checkpoint_    │               │
│                    │    │ to_working()        │               │
│                    │    └─────────────────────┘               │
│                    │               │                           │
│                    └───────┬───────┘                           │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │     TRAINING RESUMES FROM CHECKPOINT                    │  │
│  │     • Model state loaded                                │  │
│  │     • Optimizer state loaded                            │  │
│  │     • Training history merged                           │  │
│  │     • Continue from epoch N+1                           │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Search Priority Flow

```
START: find_best_checkpoint_across_sources()
│
├─> Search Priority 1: /kaggle/working/checkpoints/
│   │
│   ├─> Found vqvae_pretrained.pth? ✅
│   │   ├─> Valid? ✅ → epoch=50, acc=0.89
│   │   └─> Add to candidates
│   │
│   └─> No checkpoint → continue
│
├─> Search Priority 2: /kaggle/input/*/checkpoints/
│   │
│   ├─> Dataset: hmolqd-run1
│   │   ├─> Found vqvae_pretrained.pth? ✅
│   │   ├─> Valid? ✅ → epoch=40, acc=0.85
│   │   └─> Add to candidates
│   │
│   ├─> Dataset: hmolqd-run2
│   │   ├─> Found vqvae_pretrained.pth? ✅
│   │   ├─> Valid? ❌ → corrupted
│   │   └─> Skip
│   │
│   └─> Continue searching...
│
├─> Search Priority 3: /kaggle/input/notebooks/*/*/
│   │
│   └─> Found vqvae_pretrained.pth? ✅
│       ├─> Valid? ✅ → epoch=30, acc=0.82
│       └─> Add to candidates
│
├─> Selection Logic
│   │
│   ├─> prefer_metric='accuracy'?
│   │   │
│   │   ├─> YES → Select checkpoint with highest accuracy
│   │   │          working/checkpoints/ (acc=0.89) ← WINNER
│   │   │
│   │   └─> NO → Select by priority
│   │              working > input > notebook
│   │              working/checkpoints/ ← WINNER (priority 0)
│   │
│   └─> Return: best_path, best_info
│
├─> Check if writable
│   │
│   ├─> From /kaggle/working/? ✅
│   │   └─> Use directly (already writable)
│   │
│   └─> From /kaggle/input/? ❌
│       └─> copy_checkpoint_to_working()
│           • Copy to /kaggle/working/checkpoints/
│           • Now writable for future saves
│
└─> RESUME TRAINING ✅
    • Load model state
    • Load optimizer state
    • Set start_epoch = loaded_epoch + 1
    • Continue training loop
```

## Multi-Run Timeline

```
                    TRAINING ACROSS MULTIPLE KAGGLE SESSIONS

┌─────────────────────────────────────────────────────────────────────────┐
│ RUN 1: Initial Training (0-9 hours)                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Start: epoch 0                                                          │
│    │                                                                     │
│    ├─> Training loop (epochs 0-50)                                      │
│    │   • Save checkpoints to /kaggle/working/checkpoints/               │
│    │   • checkpoint_0010.pth                                            │
│    │   • checkpoint_0020.pth                                            │
│    │   • ...                                                            │
│    │   • vqvae_pretrained.pth (epoch 50, acc=0.85)                      │
│    │                                                                     │
│    └─> Session timeout (9 hours) ⏰                                      │
│                                                                          │
│  User Action:                                                            │
│    1. Download outputs (/kaggle/working/checkpoints/)                   │
│    2. Upload to Kaggle Datasets as "hmolqd-run1"                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ RUN 2: Resume Training (9-18 hours)                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Setup:                                                                  │
│    • Add "hmolqd-run1" dataset to notebook inputs                       │
│    • Start notebook                                                     │
│                                                                          │
│  Checkpoint Discovery:                                                   │
│    🔍 Searching for checkpoints...                                       │
│       ✅ [input_datasets] input/hmolqd-run1/checkpoints/                │
│          vqvae_pretrained.pth (epoch=50, acc=0.85)                      │
│                                                                          │
│    📋 Copying to working directory...                                    │
│       /kaggle/working/checkpoints/vqvae_pretrained.pth                  │
│                                                                          │
│  Resume: epoch 51                                                        │
│    │                                                                     │
│    ├─> Training loop (epochs 51-100)                                    │
│    │   • Overwrite /kaggle/working/checkpoints/vqvae_pretrained.pth    │
│    │   • Now: epoch 100, acc=0.90                                       │
│    │                                                                     │
│    └─> Session timeout (9 hours) ⏰                                      │
│                                                                          │
│  User Action:                                                            │
│    1. Download outputs                                                   │
│    2. Upload as "hmolqd-run2" (or new version of "hmolqd-run1")        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ RUN 3: Continue Training (18-27 hours)                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Setup:                                                                  │
│    • "hmolqd-run1" still in inputs (epoch 50)                           │
│    • Add "hmolqd-run2" to inputs (epoch 100)                            │
│                                                                          │
│  Checkpoint Discovery:                                                   │
│    🔍 Searching for checkpoints...                                       │
│       ✅ [input_datasets] input/hmolqd-run1/checkpoints/                │
│          vqvae_pretrained.pth (epoch=50, acc=0.85)                      │
│       ✅ [input_datasets] input/hmolqd-run2/checkpoints/                │
│          vqvae_pretrained.pth (epoch=100, acc=0.90)                     │
│                                                                          │
│    🎯 Selected: epoch=100, acc=0.90 (highest accuracy)                  │
│                                                                          │
│  Resume: epoch 101                                                       │
│    └─> Training continues from most recent checkpoint!                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Checkpoint Selection Examples

### Example 1: Multiple Checkpoints, Select by Accuracy

```
Available Checkpoints:
┌──────────────────────┬────────┬──────────┬────────┐
│ Location             │ Epoch  │ Accuracy │ Valid  │
├──────────────────────┼────────┼──────────┼────────┤
│ working/checkpoints/ │   50   │  0.875   │   ✅   │
│ input/run1/          │   45   │  0.860   │   ✅   │
│ input/run2/          │   40   │  0.900   │   ✅   │  ← WINNER (highest acc)
│ notebooks/user/nb/   │   30   │  0.840   │   ✅   │
└──────────────────────┴────────┴──────────┴────────┘

Selection:
  prefer_metric='accuracy'
  → Choose input/run2/ (acc=0.900)
```

### Example 2: Multiple Checkpoints, Select by Priority

```
Available Checkpoints:
┌──────────────────────┬────────┬──────────┬──────────┐
│ Location             │ Epoch  │ Accuracy │ Priority │
├──────────────────────┼────────┼──────────┼──────────┤
│ working/checkpoints/ │   50   │  0.875   │    0     │  ← WINNER (priority)
│ input/run1/          │   55   │  0.880   │    1     │
│ notebooks/user/nb/   │   60   │  0.885   │    2     │
└──────────────────────┴────────┴──────────┴──────────┘

Selection:
  prefer_metric=None (use priority)
  → Choose working/checkpoints/ (priority 0, even though older)
  
Reasoning: Working directory has most recent "active" checkpoint
           that we can continue saving to.
```

### Example 3: Invalid Checkpoints Filtered

```
Scan Results:
┌──────────────────────┬────────┬──────────┬────────┬─────────────────┐
│ Location             │ Epoch  │ Accuracy │ Valid  │ Reason          │
├──────────────────────┼────────┼──────────┼────────┼─────────────────┤
│ working/checkpoints/ │   50   │  0.875   │   ✅   │                 │  ← WINNER
│ input/run1/          │   45   │  None    │   ❌   │ Missing keys    │
│ input/run2/          │   40   │  0.860   │   ❌   │ Load error      │
│ notebooks/user/nb/   │   30   │  0.840   │   ✅   │                 │
└──────────────────────┴────────┴──────────┴────────┴─────────────────┘

Valid Candidates After Filtering:
  • working/checkpoints/ (epoch=50, acc=0.875)  ← Selected
  • notebooks/user/nb/ (epoch=30, acc=0.840)
```

## Error Handling Flow

```
find_best_checkpoint_across_sources()
│
├─> For each checkpoint found:
│   │
│   ├─> Load checkpoint
│   │   ├─> Success ✅
│   │   │   └─> Continue validation
│   │   │
│   │   └─> Exception ❌
│   │       ├─> Log error: "Load error: <exception>"
│   │       └─> Mark is_valid=False, skip
│   │
│   ├─> Validate required keys
│   │   ├─> All present ✅
│   │   │   └─> Continue validation
│   │   │
│   │   └─> Missing keys ❌
│   │       ├─> Log error: "Missing keys: [...]"
│   │       └─> Mark is_valid=False, skip
│   │
│   ├─> Sanity checks
│   │   ├─> epoch >= 0 ✅
│   │   ├─> 0 <= accuracy <= 1 ✅
│   │   │   └─> Mark is_valid=True
│   │   │
│   │   └─> Invalid values ❌
│   │       ├─> Log error: "Invalid epoch/accuracy"
│   │       └─> Mark is_valid=False, skip
│   │
│   └─> Add to candidates list
│
├─> Filter candidates (keep only is_valid=True)
│   │
│   ├─> No valid candidates found?
│   │   └─> Return (None, None, all_candidates)
│   │       • Training starts from epoch 0
│   │
│   └─> Valid candidates exist
│       └─> Continue to selection
│
└─> Select best checkpoint
    • Apply selection criteria (metric or priority)
    • Return best checkpoint
```

## Directory Structure Examples

### Ideal Dataset Structure (Recommended)

```
hmolqd-training-outputs/
├── checkpoints/
│   ├── vqvae_pretrained.pth        # Stage 1 final checkpoint
│   ├── checkpoint_0050.pth         # Stage 2 checkpoint
│   ├── checkpoint_0100.pth
│   ├── final_model.pth
│   └── best_model.pth
├── output/
│   ├── generated_dungeons/
│   ├── vqvae_curves.png
│   └── diffusion_curves.png
└── history/
    ├── vqvae_history.json
    └── diffusion_history.json
```

### Flat Structure (Supported but not recommended)

```
hmolqd-checkpoints/
├── vqvae_pretrained.pth
├── checkpoint_0050.pth
├── final_model.pth
└── (system can find these via 'direct_files' search)
```

### Nested Structure (Supported)

```
my-training-run/
├── run1/
│   └── checkpoints/
│       └── vqvae_pretrained.pth
└── run2/
    └── checkpoints/
        └── vqvae_pretrained.pth

(system searches recursively within /kaggle/input/*/)
```

## Benefits Summary

```
┌────────────────────────────────────────────────────────────┐
│  WITHOUT Multi-Path Discovery       │  WITH Multi-Path     │
├────────────────────────────────────────────────────────────┤
│  ❌ Manual path configuration       │  ✅ Automatic search │
│  ❌ Hardcoded checkpoint locations  │  ✅ Dynamic discovery│
│  ❌ Fails if dataset renamed        │  ✅ Finds any name   │
│  ❌ No validation before load       │  ✅ Pre-validated    │
│  ❌ Single checkpoint source        │  ✅ Multiple sources │
│  ❌ Manual copy from input          │  ✅ Auto-copy        │
│  ❌ No metric-based selection       │  ✅ Best by metric   │
│  ❌ Silent failures                 │  ✅ Detailed logging │
└────────────────────────────────────────────────────────────┘
```

## Key Takeaways

1. **Zero Configuration**: Upload datasets, add to inputs, run → automatic resume
2. **Intelligent Selection**: Finds best checkpoint by accuracy/epoch/priority
3. **Robust Validation**: Catches corrupted/incomplete checkpoints before loading
4. **Multi-Source Support**: Searches working dir, input datasets, notebook outputs
5. **Automatic Copy**: Handles read-only input → writable working directory
6. **Detailed Logging**: Shows exactly what was found and why it was selected
7. **Error Recovery**: Gracefully falls back to fresh training if no valid checkpoint

**Result**: Seamless multi-day training on Kaggle without manual intervention! 🎉
