# Topology Commands

Canonical command guide for:

1. generating Block I topology graph PNGs,
2. comparing room-generation branches on one fixed manual topology,
3. rerunning one fixed topology across multiple seeds,
4. exporting one generated topology through the trained room generators,
5. materializing every selected topology-QD elite into validated final maps.

Run all commands from the repository root.

## 1. Block I topology graph galleries

Use the top-level CLI when you want generated-vs-reference topology PNGs.

```powershell
python main.py topology-visualize `
  --seed 20260406 `
  --num-generated 12 `
  --num-show 12 `
  --population-size 24 `
  --generations 24 `
  --min-rooms 8 `
  --max-rooms 16 `
  --rule-space full `
  --search-strategy ga `
  --output-dir results\topology_visuals_manual_check
```

Key outputs:

- `generated_gallery.png`
- `reference_gallery.png`
- `descriptor_scatter.png`
- `summary.json`

Legacy direct module entry still works when needed:

```powershell
python scripts\visualize_block_i_graphs.py --help
```

## 2. Manual fixed-topology comparison across branches

This is the best command when you want one exact mission graph and a side-by-side
comparison of:

- diffusion
- fast sampler
- masked room

Built-in rich topology:

```powershell
python main.py topology-compare-manual `
  --run-dir outputs\zelda_hmolqd_semantic_anchor_retrain_v1 `
  --output-dir outputs\zelda_hmolqd_semantic_anchor_retrain_v1\full_architecture_verification\manual_compare_v1 `
  --seed 20260406
```

User-authored `mission_graph.json`:

```powershell
python main.py topology-compare-manual `
  --run-dir outputs\zelda_hmolqd_semantic_anchor_retrain_v1 `
  --mission-graph my_manual_graph.json `
  --output-dir outputs\zelda_hmolqd_semantic_anchor_retrain_v1\full_architecture_verification\manual_custom_compare_v1 `
  --seed 20260406
```

Runtime ablation example with the promoted semantic-anchor / fallback knobs:

```powershell
python main.py topology-compare-manual `
  --run-dir outputs\zelda_hmolqd_semantic_anchor_retrain_v1 `
  --mission-graph my_manual_graph.json `
  --output-dir outputs\zelda_hmolqd_semantic_anchor_retrain_v1\full_architecture_verification\manual_custom_compare_ablation_v1 `
  --seed 20260406 `
  --semantic-role-prior-strength 0.25 `
  --semantic-puzzle-offset 3 `
  --no-fast-sampler-teacher-fallback-enabled
```

Key outputs:

- `mission_graph.json`
- `mission_graph_layout.png`
- `graph_summary.json`
- `search_algorithm_comparison.json`
- `dungeon_alignment_comparison.png`
- `rooms_sheet_comparison.png`
- `comparisons\...\summary.json`
- `report.md`

## 3. Fixed-graph multi-seed audit

Use this after you already have a fixed `mission_graph.json` and want stability
results without Block I topology variance.

```powershell
python main.py topology-audit-fixed-graph `
  --run-dir outputs\zelda_hmolqd_semantic_anchor_retrain_v1 `
  --mission-graph outputs\zelda_hmolqd_semantic_anchor_retrain_v1\full_architecture_verification\manual_compare_v1\mission_graph.json `
  --output-dir outputs\zelda_hmolqd_semantic_anchor_retrain_v1\full_architecture_verification\fixed_graph_multi_seed_v1 `
  --seeds 20260404 20260405 20260406
```

Runtime ablation example:

```powershell
python main.py topology-audit-fixed-graph `
  --run-dir outputs\zelda_hmolqd_semantic_anchor_retrain_v1 `
  --mission-graph my_manual_graph.json `
  --output-dir outputs\zelda_hmolqd_semantic_anchor_retrain_v1\full_architecture_verification\fixed_graph_multi_seed_ablation_v1 `
  --seeds 20260404 20260405 20260406 `
  --semantic-role-prior-strength 0.25 `
  --semantic-puzzle-offset 3 `
  --no-fast-sampler-teacher-fallback-enabled
```

Strict no-fallback / pure-neural ablation bundle:

```powershell
python main.py topology-audit-fixed-graph `
  --run-dir outputs\zelda_hmolqd_semantic_anchor_retrain_v1 `
  --mission-graph my_manual_graph.json `
  --output-dir outputs\zelda_hmolqd_semantic_anchor_retrain_v1\full_architecture_verification\fixed_graph_multi_seed_no_fallback_v1 `
  --seeds 20260404 20260405 20260406 `
  --include-no-fallback-ablations
```

Key outputs:

- per-seed variant folders
- aggregate `summary.json`
- aggregate `search_algorithm_comparison.json`

Validation interpretation:

- `summary.json` carries the end-to-end aggregate metrics
- `search_algorithm_comparison.json` carries the canonical validation solver stack
- the report-facing hard oracle is `graph_guided_oracle + graph progression + softlock`
- monolithic stitched tile-state `A*` is retained as a stricter stress probe
- CBS is a bounded-rational probe, not the correctness oracle

## 4. Automatic topology + end-to-end export

Use this when you want one generated topology and all room-generation branches
exported from trained checkpoints.

```powershell
python scripts\export_semantic_anchor_end_to_end.py `
  --run-dir outputs\zelda_hmolqd_semantic_anchor_retrain_v1 `
  --output-dir outputs\zelda_hmolqd_semantic_anchor_retrain_v1\full_architecture_verification\auto_topology_export_v1 `
  --seed 20260406 `
  --num-rooms 8 `
  --topology-population 50 `
  --topology-generations 100
```

The same runtime ablation flags also work here:

- `--semantic-role-prior-strength`
- `--semantic-puzzle-offset`
- `--fast-sampler-teacher-fallback-enabled` / `--no-fast-sampler-teacher-fallback-enabled`

This path is still script-level because it is an export workflow rather than a
core root command.

## 4a. Training-time semantic-anchor threshold

`generation.semantic_anchor_threshold` is already promoted into the YAML / train
CLI, but it is currently a training-time lever rather than a meaningful
room-export runtime ablation. Use it on `train`, for example:

```powershell
python main.py train `
  --config configs\zelda_hmolqd.yaml `
  --stage masked_room `
  --output-dir outputs\zelda_hmolqd_anchor_threshold_ablation_v1 `
  --semantic-anchor-threshold 0.65 `
  --no-auto-resume `
  --verbose
```

## 5. Materialize a topology-QD archive

Topology archive coverage is graph-level evidence. It is not final-map
coverage until every selected elite survives graph export, room generation,
stitching, and the exact tile-state validator. New archives store the evaluated
phenotype snapshot and grammar/fitness provenance needed for this operation.
Genome-only legacy pickle archives must be regenerated because grammar replay
depends on population-global RNG state.

```powershell
python main.py topology-materialize-archive `
  --archive results\topology_qd.pkl `
  --trust-pickle `
  --config configs\zelda_hmolqd.yaml `
  --output-dir results\topology_qd_final_maps `
  --seeds 42,43,44 `
  --vqvae-checkpoint outputs\vqvae_pretrained.pth `
  --diffusion-checkpoint outputs\diffusion_best.pth `
  --condition-encoder-checkpoint outputs\diffusion_best.pth `
  --logic-net-checkpoint outputs\diffusion_best.pth `
  --require-logic-net `
  --verify-solver-consistency
```

`--trust-pickle` is mandatory because native Python pickle is executable. Use
it only for an archive produced by this repository under your control.
Warm-started topology search has the equivalent fail-closed configuration
contract: both `topology.qd_load_archive: true` and
`topology.qd_trust_archive_pickle: true` are required.

Key outputs:

- `materialization_summary.json`: graph archive coverage, graph-cell drift,
  per-seed acceptance, source-elite survival, and final-descriptor archive
  coverage. A full scan is distinguished from complete successful
  materialization.
- `materialization_progress.json`: atomically updated failure ledger suitable
  for resuming diagnosis after an interrupted long run.
- `cell_*/mission_graph.json`: finalized graph sent to room generation.
- `cell_*/seed_*/dungeon_grid.npy` and `metrics.json`: accepted exact artifacts.

Using `--max-elites` is diagnostic only. The summary marks such a run as an
incomplete archive materialization, so it cannot be reported as full coverage.

## 6. Manual topology JSON format

The manual graph format is NetworkX node-link JSON using:

- `nodes`
- `links`

Minimal example:

```json
{
  "directed": false,
  "multigraph": false,
  "graph": {
    "name": "my_manual_graph"
  },
  "nodes": [
    { "id": 0, "label": "START", "type": "START", "pos": [0, 0], "is_start": true },
    { "id": 1, "label": "KEY", "type": "KEY", "pos": [0, 1], "has_key": true, "key_id": 1 },
    { "id": 2, "label": "GOAL", "type": "GOAL", "pos": [1, 1], "is_goal": true, "has_triforce": true }
  ],
  "links": [
    { "source": 0, "target": 1, "label": "path", "edge_type": "PATH" },
    { "source": 1, "target": 2, "label": "key_locked", "edge_type": "KEY_LOCKED", "key_required": 1 }
  ]
}
```

Recommended node fields:

- `id`
- `label`
- `type`
- `pos`

Useful optional node flags:

- `is_start`
- `is_goal`
- `has_triforce`
- `has_enemy`
- `enemy_count`
- `has_key`
- `key_id`
- `has_item`
- `item_type`
- `has_puzzle`
- `has_boss`

Useful edge fields:

- `source`
- `target`
- `label`
- `edge_type`
- `key_required`
- `item_required`

## 7. Practical workflow

1. Inspect Block I:
   `python main.py topology-visualize ...`
2. Freeze one topology:
   `python main.py topology-compare-manual ...`
3. Stress-test stability:
   `python main.py topology-audit-fixed-graph ...`
4. Only then do longer training or ablation runs.
5. Materialize the complete topology archive before reporting end-to-end QD
   coverage.
