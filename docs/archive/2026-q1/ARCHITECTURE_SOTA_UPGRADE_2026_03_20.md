# Architecture Revision: SOTA Upgrade Pass (2026-03-20)

This revision upgrades the generator from one-way neural-symbolic flow into a feedback-capable, multi-objective, engine-agnostic pipeline.

## Revised Top-Level Flow

1. Topology generation/evolution with multi-objective structural constraints.
2. Neural room synthesis (diffusion/VQ-VAE).
3. Symbolic repair (WFC).
4. WFC dead-end detection triggers localized diffusion inpainting feedback.
5. Symbolic repair resumes and final solvability constraints are enforced.
6. Final output can be exported to 2D and semantic 3D voxel formats.

## Upgrade 1: Bi-Directional Bridge (WFC-Guided Inpainting)

Implemented in:
- `src/core/symbolic_refiner.py`
- `src/core/latent_diffusion.py`
- `src/pipeline/dungeon_pipeline.py`

What changed:
- Added `repair_room_with_feedback(...)` in `SymbolicRefiner`.
- Added latent `inpaint(...)` method in `LatentDiffusionModel`.
- `generate_room(...)` now invokes a neural feedback callback when symbolic WFC reaches local dead-end failures.
- Dead-end masks from refiner are converted to latent-space masks and inpainted regions are merged back before re-running symbolic checks.

## Upgrade 2: Multi-Objective QD / Pareto Constraints

Implemented in:
- `src/evaluation/fun_metrics.py`
- `src/generation/evolutionary_director.py`

What changed:
- Added structural metrics:
  - Cyclomatic complexity (loop count proxy)
  - Raw branching factor
- Integrated hard Pareto-style constraints in topology evaluation:
  - `cyclomatic_complexity >= 2`
  - `branching_factor_raw > 1.5`
- Added new evaluator diagnostics:
  - `pareto_score`, `pareto_feasible`
  - loop/branching violations and requirements
- Fitness now blends scalar objective with Pareto score and increases constraint violation when topology fails structural minima.

## Upgrade 3: Engine-Agnostic Semantic Voxel Exporter

Implemented in:
- `src/visualization/voxel_exporter.py`

What changed:
- New exporter from semantic 2D grids to 3D voxel artifacts.
- Supports ASCII and JSON inputs.
- Exports:
  - Wavefront OBJ (`--obj-out`)
  - Engine JSON (`--json-out`) for Unity/Godot pipelines
- Semantic mapping includes walls/floor/water defaults and robust fallback behavior.

## Validation Notes

- Compile checks passed for all modified modules.
- Voxel exporter smoke test succeeded and produced:
  - `results/voxel_smoke.obj`
  - `results/voxel_smoke.json`

## Follow-Up (Recommended)

1. Add benchmark toggles to isolate feedback-loop gains vs baseline repair.
2. Add dedicated ablation configs for Pareto constraints on/off.
3. Add integration tests for inpainting callback activation and mask locality.
4. Add Unity/Godot import sample prefabs/scripts under `examples/`.
