# GUI Runner Demo Tutorial

This guide explains how to run `gui_runner.py` as a thesis/demo interface, how it is connected to the best available trained output checkpoint, and how to demonstrate a real generated level without reducing the story to a short straight-line solve.

## What Is Connected

`gui_runner.py` now auto-selects a trained checkpoint when no `--checkpoint` or `KLTN_CHECKPOINT_PATH` override is provided.

Default demo checkpoint:

```powershell
outputs\zelda_hmolqd_downstream_stageconditioned_semantics_v3_puzzlefix\checkpoints\diffusion\best_model.pth
```

Reason for this choice:

- It has checkpoint metadata and a matched VQ-VAE tokenizer path.
- It is the preferred trained checkpoint under `outputs`, and its metadata resolves the matched VQ-VAE tokenizer.
- The demo artifact below is from the Chapter 4 generated-topology run that uses the same model family and full neural-symbolic pipeline: Block I topology, diffusion room generation, symbolic repair/marker overlay, post-generation validation, and P-CBS.
- The demo is configured for live solving with `P-CBS (Balanced)` so the path is computed during the presentation.

The GUI still supports explicit overrides:

```powershell
python gui_runner.py --checkpoint outputs\zelda_hmolqd_downstream_stageconditioned_semantics_v3_puzzlefix\checkpoints\diffusion\best_model.pth
```

or:

```powershell
$env:KLTN_CHECKPOINT_PATH="outputs\zelda_hmolqd_downstream_stageconditioned_semantics_v3_puzzlefix\checkpoints\diffusion\best_model.pth"
python gui_runner.py
```

## Demo Level

Prepared hard demo file:

```powershell
examples\gui_demo_hard_real_model_level.txt
```

The level is a numeric GUI-importable version of:

```powershell
results\ch4_generated_topology_real_pdrop035_seed20260418_fixedvalidator\diffusion_cfg3_logic0_steps50\dungeon_grid_ids.json
```

Demo metrics:

- Generator branch: `diffusion_cfg3_logic0_steps50`
- Rooms: `12`
- Grid size: `32 x 77`
- Generation time in the recorded run: `21.46s`
- Repair rate: `0.9167`
- Total repaired tiles: `147`
- Puzzle plan count / stage count: `7 / 9`
- Final graph-marker match after overlay: `1.0`
- Live P-CBS balanced solve: usually `~218-232` steps in under one second after solver startup
- Live route coverage: `7` unique rooms with repeated room transitions/revisits
- Validation: `tests\test_gui_demo_validated_level_artifact.py` asserts this level solves live through the GUI solver worker
- Provenance/regression coverage: `tests\test_gui_demo_validated_level_artifact.py`

The older `examples\gui_demo_validated_level.txt` is still useful as a quick hard-A* sanity check, but it is shorter and should not be the main thesis demo.

## Fast Demo Script

From the repository root:

```powershell
cd <repo root>
python gui_runner.py --advanced --load-txt examples\gui_demo_hard_real_model_level.txt --solver-algorithm 7 --solver-timeout 900
```

In the GUI:

1. Wait for the imported map to appear.
2. Press `T` or enable `Show Topology Overlay` / `Show Room Graph`.
3. Use `F` to auto-fit if the full dungeon is not visible.
4. Explain the room graph first: this is a stitched dungeon from the generated topology pipeline, not a single corridor map.
5. Point out the selected algorithm: `P-CBS (Balanced)`.
6. Press `SPACE` or click `Solve Level`. The GUI computes a fresh route during the demo and then animates it through `7` rooms.
7. Press `H` only if you want to show search heatmap/debug behavior, then `ESC` to quit.

This launch avoids waiting for a fresh neural generation run while still showing a real generated artifact and a real-time solver pass. The main story is topology-conditioned generation, repair, and validation; the live solve is the visual proof that the generated dungeon can be traversed.

## Demo Story

Use this order when presenting so the demo shows the pipeline rather than a boring line traversal:

1. `Checkpoint`: show the selected trained checkpoint path and explain that the GUI auto-selects it from `outputs`.
2. `Topology`: press `T` and show that the full grid is interpreted as stitched rooms, not a single flat maze.
3. `Semantics`: point out room-scale semantic tiles: start, triforce, boss, boss key, key item, puzzle gates, stairs, doors, enemies, blocks, floors, walls, and void.
4. `Repair`: explain that the artifact is post neural-symbolic repair, so invalid neural samples are converted into a playable semantic grid.
5. `Behavioral validation`: press `SPACE` to run P-CBS balanced live. The expected route is roughly `218-232` steps and visits `7` rooms.
6. `Live generation`: click `Generate Level` / `AI Generate` only after the prepared artifact, because fresh generation can take longer and individual seeds may still require validation or rejection.

## Live AI Generation Demo

Use this when you want to show the model producing a new level during the demo:

```powershell
cd <repo root>
$env:KLTN_AI_SEED="20260405"
python gui_runner.py --advanced
```

Then click `Generate Level` or `AI Generate`.

Notes:

- `Generate Level` now uses the auto-selected checkpoint instead of falling back to procedural generation.
- Fresh CPU generation can take a while because the model loads the neural-symbolic stack and samples room content.
- For a deterministic demo, set `KLTN_AI_SEED` before launching.
- For strict checkpoint validation, add:

```powershell
$env:KLTN_STRICT_CHECKPOINTS="1"
```

## Buttons And Controls

- `Generate Level`: uses the selected AI checkpoint; falls back to procedural only if no checkpoint is available.
- `Load Model`: manually choose another `.pth`, `.pt`, or `.ckpt` checkpoint.
- `AI Generate`: starts the AI dungeon generation worker directly.
- `Solve Level` / `SPACE`: runs the selected solver on the current level.
- `Ctrl+S`: export the current semantic grid.
- `I`: import a numeric `.txt` level.
- `Ctrl+A`: toggle advanced controls.
- `H`: toggle heatmap.
- `R`: reset current map.
- `ESC`: quit.

## Talking Points For The Demo

Use these points when explaining the prepared generated level:

- The level is not hand-authored; it comes from a recorded Chapter 4 full-pipeline result under `results`.
- It is a generated 12-room stitched dungeon, not a single room or a linear corridor.
- It contains a non-trivial live behavioral route: P-CBS balanced typically travels more than `200` adjacent steps before reaching the triforce.
- The topology overlay can now infer room nodes for imported numeric grids, so the prepared artifact still shows room-level structure even though `.txt` import does not preserve Python graph objects.
- The generation stack combines learned room generation with symbolic repair and validation, which is why the final map is both visually structured and live-solvable.

## Troubleshooting

If the GUI starts but `Generate Level` says no checkpoint exists, run with the explicit checkpoint:

```powershell
python gui_runner.py --advanced --checkpoint outputs\zelda_hmolqd_downstream_stageconditioned_semantics_v3_puzzlefix\checkpoints\diffusion\best_model.pth
```

If importing a file fails, use `examples\gui_demo_hard_real_model_level.txt`, not the VGLC-style character export. The GUI import path expects numeric tile IDs.

If solving takes too long with another algorithm, return to `--solver-algorithm 7`. A* is an oracle-style solver and can be slower or produce shortcut-style paths on this imported topology; P-CBS balanced is the intended live demo solver for this artifact.
