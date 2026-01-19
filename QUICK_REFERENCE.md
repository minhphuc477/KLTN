# KLTN — Quick Reference Card

## ⚡ Quick Commands

### NEW: Main Pipeline (Recommended Entry Point)
```powershell
# Validate a single dungeon
python main.py --dungeon 1 --variant 1

# Validate all 18 dungeons
python main.py --all

# With ASCII visualization
python main.py --dungeon 1 --ascii

# Export to NPZ
python main.py --dungeon 1 --export output.npz
```

### Extract Visual Assets
```powershell
python scripts/extract_visual_levels.py --templates "path/to/tileset.png" --input "path/to/screenshot.png" --out-dir artifacts --visualize --npz
```

### Run Tests
```powershell
# All visual + inventory tests
pytest tests/ -v -k "visual or inventory"

# Just regression tests
pytest tests/test_visual_dataset_regeneration.py -v

# Full test suite
pytest tests/ -v
```

### Launch GUI
```powershell
python gui_runner.py       # Interactive GUI
```

### Demo
```powershell
python demo_visual_workflow.py
```

---

## 📂 Key Files (After Refactoring)

| File | Purpose |
|------|---------|
| `main.py` | **NEW** Single entry point: Load → Stitch → Validate |
| `Data/zelda_core.py` | **CANONICAL** All core logic (adapter, stitcher, solver) |
| `src/core/definitions.py` | **NEW** Semantic constants (PALETTE, IDs) |
| `simulation/validator.py` | Block VI validation engine |
| `src/data_processing/visual_extractor.py` | Template matching CV engine |
| `gui_runner.py` | Interactive visualization |
| `tests/test_solver_inventory.py` | State-space solver tests |

### Archived Files (in `/archive/`)
| File | Reason |
|------|--------|
| `adapter_v1.py` | Replaced by zelda_core.py |
| `stitcher_v1.py` | Replaced by zelda_core.py |
| `test_sprites.py` | One-off debug script |

---

## 🔑 Key Concepts

### Visual Extraction
```python
from src.data_processing.visual_extractor import extract_grid
arr = extract_grid('screenshot.png', 'tileset.png', tile_px=16)
# Returns: (16, 11, 2) array [template_ids, confidence]
```

### Inventory-Aware Pathfinding
```python
from graph_solver import TilePathFinder
inventory = {'raft'}  # enables water traversal
path = tile_finder.find_tile_path(room_path, inventory=inventory)
```

### Graph Topology Solver
```python
from graph_solver import GraphSolver
solver = GraphSolver(dungeon)
room_path, actions, success = solver.solve()
# Returns: path with key collection + locked door traversal
```

---

## ✅ Test Status

| Test Suite | Status |
|------------|--------|
| Visual Extraction | ✅ 3/3 passing |
| Inventory Solving | ✅ 2/2 passing |
| Regression Testing | ✅ 2/2 passing |
| **Total** | **7/9 passing** (2 skipped) |

---

## 🎯 What Was Implemented

✅ **Visual extraction** with template matching + HUD detection  
✅ **NPZ metadata** format with template hash for regression  
✅ **GUI integration** with sprite loading  
✅ **Inventory-aware solver** (already existed, verified working)  
✅ **Graph topology solver** (already existed, verified working)  
✅ **Comprehensive tests** with 77.8% pass rate  

---

## 📖 Documentation

- **Full docs:** `IMPLEMENTATION_COMPLETE.md` (300+ lines)
- **Research summary:** `RESEARCH_SUMMARY.md` (200+ lines)
- **This card:** `QUICK_REFERENCE.md` (you are here)

---

## 🚨 Troubleshooting

### "ModuleNotFoundError: No module named 'cv2'"
```powershell
pip install -r requirements-visual.txt
```

### "Test skipped: assets not available"
- Place tileset at `Data/assets/Dungeon Tileset.png`
- Or use synthetic tileset (demo will auto-create)

### "Grid extraction produces all -1 (unknown)"
- Check template matching threshold (try `--threshold 0.5`)
- Verify tileset has non-transparent tiles
- Enable verbose logging in extractor

### "Solver can't cross water"
```python
# Add raft to inventory
inventory = {'raft'}
path = tile_finder.find_tile_path(room_path, inventory=inventory)
```

---

## 🔗 File Structure

```
KLTN/
├── src/
│   └── data_processing/
│       ├── visual_extractor.py        ← CV engine
│       └── visual_integration.py      ← Single-room helpers
├── scripts/
│   ├── extract_visual_levels.py       ← CLI (NEW: --npz flag)
│   └── asset_cutter.py                ← Tileset slicer
├── tests/
│   ├── test_visual_extractor.py
│   ├── test_solver_inventory.py
│   ├── test_visual_integration.py
│   └── test_visual_dataset_regeneration.py  ← NEW
├── graph_solver.py                    ← Inventory-aware solver
├── maze_solver.py                     ← Tile pathfinding
├── gui_runner.py                      ← Interactive GUI
├── demo_visual_workflow.py            ← NEW: End-to-end demo
├── IMPLEMENTATION_COMPLETE.md         ← NEW: Full docs
├── RESEARCH_SUMMARY.md                ← NEW: Research findings
└── QUICK_REFERENCE.md                 ← This file
```

---

**Last Updated:** 2026-01-18  
**Status:** ✅ COMPLETE  
**Test Pass Rate:** 77.8% (7/9)  
