# Kaggle Checkpoint Resume - Quick Reference Card

## 🚀 Quick Start (3 Steps)

### First Run
1. Run your notebook → checkpoints save to `/kaggle/working/checkpoints/`
2. Download outputs from notebook (Output tab)
3. Upload to [kaggle.com/datasets](https://www.kaggle.com/datasets) as new dataset

### Resume Run
1. Add your dataset to notebook (right sidebar → "Add Data")
2. Run notebook → automatically finds and loads checkpoints
3. Training continues from where it left off!

## 📁 Where Checkpoints Live

```
/kaggle/
├── working/checkpoints/     ← Current run (WRITABLE)
└── input/
    ├── my-dataset/
    │   └── checkpoints/     ← Previous runs (READ-ONLY)
    └── notebooks/user/notebook/
        └── checkpoints/     ← Auto-versioned (READ-ONLY)
```

## 🔍 Core Functions

### Find Best Checkpoint (by metric)
```python
best_path, info, all = find_best_checkpoint_across_sources(
    checkpoint_filename='vqvae_pretrained.pth',
    required_keys=['epoch', 'model_state_dict', 'accuracy'],
    prefer_metric='accuracy'  # or 'epoch', 'solvability'
)
```

### Search All Locations
```python
locations = find_checkpoint_locations()
# {'working': [...], 'input_datasets': [...], 'notebook_outputs': [...]}
```

### Copy to Writable Location
```python
working_path = copy_checkpoint_to_working(
    source_path=Path('/kaggle/input/my-dataset/checkpoints/model.pth'),
    target_filename='model.pth'
)
# Now you can overwrite with better checkpoints
```

### Diagnostic Scan
```python
discover_and_validate_all_checkpoints(show_invalid=True)
# Shows ALL checkpoints and validation status
```

## 🎯 Selection Priority

When multiple checkpoints found:

1. **With `prefer_metric`**: Highest metric value wins
2. **Without `prefer_metric`**: 
   - Working dir > Input datasets > Notebook outputs
   - Within same level: Most recent by modified time

## 📊 Example Output

```
🔍 Searching for checkpoints across all sources...
   ✅ [working        ] working/checkpoints/vqvae_pretrained.pth
      23.4MB, epoch=45, acc=0.875 - Valid
   ✅ [input_datasets ] input/hmolqd/checkpoints/vqvae_pretrained.pth
      23.1MB, epoch=40, acc=0.860 - Valid

🎯 Selected checkpoint by best accuracy:
   📂 working/checkpoints/vqvae_pretrained.pth
   📊 Epoch 45, accuracy=0.875
```

## ✅ Best Practices

### DO
- ✅ Upload outputs with `checkpoints/` folder structure
- ✅ Use descriptive dataset names (`hmolqd-run1`, not `untitled`)
- ✅ Version datasets (keeps training history)
- ✅ Download checkpoints regularly (local backup)
- ✅ Use the discovery functions (handles complexity)

### DON'T
- ❌ Hardcode checkpoint paths
- ❌ Save to `/kaggle/input/` (read-only, will fail)
- ❌ Assume specific checkpoint locations
- ❌ Skip checkpoint validation
- ❌ Upload gigantic checkpoints (use compact mode)

## 🐛 Troubleshooting

### Checkpoint not found?
```python
# Run diagnostic
discover_and_validate_all_checkpoints(show_invalid=True)
```

**Common causes**:
- Dataset not added to notebook inputs
- Wrong folder structure in dataset
- Filename doesn't match

### Training not resuming?
```python
# Check if state loading succeeded
try:
    model.load_state_dict(ckpt['model_state_dict'])
    print("✅ State loaded")
except Exception as e:
    print(f"❌ Failed: {e}")
    # Likely model architecture changed
```

### Out of disk space?
```python
# Prune old checkpoints (keep only 3 most recent)
_prune_checkpoints(checkpoint_dir, keep=3, pattern='checkpoint_*.pth')

# Use compact checkpoints (already enabled for Kaggle)
COMPACT_CHECKPOINTS = True
```

## 🔄 Multi-Day Training Workflow

**Day 1 (0-9h)**:
```
Start notebook → Save to working → Download outputs
```

**Day 2 (9-18h)**:
```
Upload outputs as dataset → Add to inputs → Resume automatically
```

**Day 3 (18-27h)**:
```
Upload Day 2 outputs → Add to inputs → Resume from epoch 100
```

No manual path configuration needed!

## 📚 Integration Examples

### VQ-VAE Training
```python
best_path, info, _ = find_best_checkpoint_across_sources(
    'vqvae_pretrained.pth',
    prefer_metric='accuracy'
)

if best_path:
    ckpt = torch.load(best_path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    
    # Copy to working if from input
    if IS_KAGGLE and info.source_type != 'working':
        SAVE_PATH = copy_checkpoint_to_working(best_path, 'vqvae_pretrained.pth')
```

### Diffusion Training
```python
latest_path, info = find_latest_checkpoint_multi_source(
    checkpoint_patterns=['final_model.pth', 'checkpoint_*.pth'],
    required_keys=['epoch', 'diffusion_state_dict']
)

if latest_path:
    trainer.load_checkpoint(str(latest_path))
    start_epoch = trainer.epoch + 1
```

## 💡 Key Insight

**The system handles all the complexity automatically:**
- Searches multiple locations
- Validates checkpoints before loading
- Selects best checkpoint
- Copies to writable location if needed
- Provides detailed logging

**You just**: Upload outputs as dataset → Add to inputs → Run notebook

## 🆘 Need Help?

1. **Read full guide**: `docs/KAGGLE_CHECKPOINT_RESUME_GUIDE.md`
2. **Run diagnostic**: `discover_and_validate_all_checkpoints(show_invalid=True)`
3. **Check logs**: Look for `🔍 Searching for checkpoints...` output
4. **Verify dataset**: Make sure it's added to notebook inputs (right sidebar)

---

**TIP**: The first cell in your notebook runs `discover_and_validate_all_checkpoints()` when `IS_KAGGLE=True`, showing you exactly what's available!
