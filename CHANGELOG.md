# Changelog

All notable architectural, training, validation, and data-pipeline changes to the KLTN Neuro-Symbolic Dungeon Generation project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] - 2026-11-24

### Added
- **Multi-Criterion Checkpoint Saving, Non-Neural Reference Baselines & Automated Falsification Audit:**
  - `train_masked_room` ([`src/train_masked_room.py`](src/train_masked_room.py)) now tracks `best_val_rare_tile_recall` alongside `best_metric_name` (`val_loss`), persisting `masked_room_best.pth` (`min val_loss`), `masked_room_best_rare_recall.pth` (`max val_rare_tile_recall`), and `masked_room_final.pth` (final epoch) to mitigate loss–solvability decoupling (`Baghal2026_SokobanDiffusion`).
  - Added deterministic non-neural reference baselines (`--baseline empty_floor`, `--baseline vglc_template`), multi-checkpoint expansion (`--checkpoint-dir-spec`), paired unmask-order sweeps (`--eval-both-unmask-orders`), and automated falsification checks (`run_falsification_checks`: `H20_floor_shortcut_collapse`, `H40_beats_empty_floor_baseline`, `H10_memorization_alarm`, `H30_unmask_order_delta`, `H40_repair_dominance`) in [`scripts/evaluate_glmdm_benchmark.py`](scripts/evaluate_glmdm_benchmark.py), with unit coverage in [`tests/test_evaluate_glmdm_benchmark.py`](tests/test_evaluate_glmdm_benchmark.py).
- **Literature-Grounded Minimalist Architectural Redesign (Block IV + LogicNet + V4 Symmetry):**
  - **Factorized 2D Row+Col Embeddings (`DiscreteMaskedRoomModel`):** Replaced the unfactorized `[1, hidden_dim, 16, 11]` spatial embedding with factorized `row_embedding` (`[1, C, 16, 1]`) and `col_embedding` (`[1, C, 1, 11]`) in [`src/core/discrete_masked_model.py`](src/core/discrete_masked_model.py) (`Baghal2026_SokobanDiffusion`), sharing row/column inductive bias across the `45` unique VGLC training interiors.
  - **Balanced 4-Layer Denoiser & Corrector Guard (`configs/glmdm_*.yaml`):** Reconfigured `hidden_dim: 48 -> 32` with `unet_num_res_blocks: 2` and `unet_channel_mult: [1, 2]` (`4` Transformer encoder layers, `4` heads, `head_dim=8`, `denoiser=59,756` params `<= 100K`, `cond_enc=764,209`, `total=823,965` for `G-LG`/`G-L`/`G`; `cond_enc=382,961`, `total=442,717` for `N`) following `Shao2026_DiscretePlayground`. Explicitly disabled low-confidence corrector remasking (`masked_room_corrector_steps: 0`, `masked_room_corrector_mask_ratio: 0.0`) so inference does not strip unpinned rare tiles (`BLOCK`, `ENEMY`, `ELEMENT`) after `unmask_order="random"`.
  - **Per-Sample Mask-Ratio Gating in `LogicNet`:** Added optional `sample_weights` `[B]` to `LogicNet.forward` ([`src/core/logic_net.py`](src/core/logic_net.py)) and updated `MaskedRoomTrainer._step` ([`src/train_masked_room.py`](src/train_masked_room.py)) to weight each room's grid reachability, trace, and anchor terms by its individual `(1 - mask_ratio_i)^2` rather than averaging the gate across the entire dungeon batch.
  - **Full `V4` Position Equivariance:** Updated `apply_room_symmetry_transform` in [`src/zelda_data/symmetry.py`](src/zelda_data/symmetry.py) to reflect `node_positions` and `room_position` around the graph bounding-box midpoint under `flip_h` and `flip_v`.
  - **New Regression Tests:** Added `test_discrete_masked_model_factorized_2d_embeddings_and_4layer_budget` and `test_logic_net_per_sample_weights_zeroes_masked_sample_gradient` in [`tests/test_discrete_masked_room_model.py`](tests/test_discrete_masked_room_model.py), plus `test_apply_room_symmetry_transform_reflects_node_and_room_positions` in [`tests/test_zelda_dataset_fixed_split.py`](tests/test_zelda_dataset_fixed_split.py).
- **Bipartite & Degree-Cap Enforcement in Block I (`EvolutionaryTopologyGenerator`):**
  - Extracted `@staticmethod _edge_prune_priority` and added `_edge_removal_preserves_connectivity` and `_enforce_bipartite_cycles` in [`src/generation/evolutionary_director/executor.py`](src/generation/evolutionary_director/executor.py).
  - Wired `_enforce_max_degree(..., max_degree=4)` and `_enforce_bipartite_cycles(...)` into both candidate execution (`GraphGrammarExecutor.execute`) and final export (`EvolutionaryTopologyGenerator._finalize_graph_output` in [`src/generation/evolutionary_director/generator.py`](src/generation/evolutionary_director/generator.py)).
  - Non-bridge odd-cycle `PATH`/`SHORTCUT` edges are pruned deterministically before export so graphs remain 2-colorable and `max_degree <= 4` for strict 2D orthogonal grid stitching.
- **Mask-Ratio Gating for LogicNet Training Supervision:**
  - Exposed `mask_ratio_per_sample` in `DiscreteMaskedRoomModel.training_loss(..., return_aux=True)` ([`src/core/discrete_masked_model.py`](src/core/discrete_masked_model.py)).
  - Multiplied `logic_loss` by `low_mask_gate = (1.0 - mask_ratio_per_sample.clamp(0.0, 1.0)).pow(2).mean()` in `MaskedRoomTrainer._run_step` ([`src/train_masked_room.py`](src/train_masked_room.py)) so noisy high-mask predictions (`mask_ratio -> 1.0`) do not overwhelm tile-level MLE reconstruction while preserving dungeon-batch alignment.
- **Regression Tests:**
  - `test_logicnet_vglc_two_tile_border_interior_anchors` in [`tests/test_logicnet_fixes.py`](tests/test_logicnet_fixes.py).
  - `test_block_in_rare_tiles_and_mask_ratio_aux_and_stochastic_sample` and `test_masked_room_dataloader_uses_dungeon_batch_mode_and_kfold_with_augmentation` in [`tests/test_discrete_masked_room_model.py`](tests/test_discrete_masked_room_model.py).
  - `test_executor_enforce_bipartite_cycles_and_degree_cap` in [`tests/test_room_stitching.py`](tests/test_room_stitching.py).
  - `test_build_dungeon_kfold_splits_partitions_cleanly` in [`tests/test_zelda_dataset_fixed_split.py`](tests/test_zelda_dataset_fixed_split.py).
- **Dungeon-Grouped K-Fold / LODO Cross-Validation & Fair Batching:**
  - Added `build_dungeon_kfold_splits` in [`src/zelda_data/splits.py`](src/zelda_data/splits.py) and wired `--val-dungeons`, `--num-folds`, `--fold-index`, and `--dungeon-batch-mode` into [`src/train_masked_room.py`](src/train_masked_room.py) so all 4 ablation arms respect `dataset.dungeon_batch_mode` identically (removing the previous `~5.6x` optimizer-step imbalance between LogicNet arms and `G`/`N`).
- **Uniform Random Unmasking (`unmask_order="random"`):**
  - Added `unmask_order` (`"confidence"` or `"random"`) to `DiscreteMaskedRoomModel.sample` / `_iterative_fill` ([`src/core/discrete_masked_model.py`](src/core/discrete_masked_model.py)) and wired `generation.masked_room_unmask_order` through [`src/config_system.py`](src/config_system.py), [`src/pipeline/config.py`](src/pipeline/config.py), [`src/pipeline/config_bridge.py`](src/pipeline/config_bridge.py), [`src/pipeline/runtime.py`](src/pipeline/runtime.py), and [`src/pipeline/generation/sampler.py`](src/pipeline/generation/sampler.py).

### Fixed
- **Preserve Authentic `TileID.BLOCK` Pillars in Structural Post-Processing:**
  - Updated `_strip_structural_room_artifacts` in [`src/pipeline/generation/room_processing.py`](src/pipeline/generation/room_processing.py#L428-L430) to target isolated `<= 4`-tile `TileID.WALL` islands only rather than erasing `TileID.BLOCK` clusters (which corrupted authentic VGLC `1x1`–`2x2` block pillars on `vglc_template` from `JS_raw=0.0085` to `JS_rep=0.0708`).
- **LogicNet False Outer-Wall Penalty on 2-Tile VGLC Room Borders:**
  - Diagnostic autopsy on 48 real VGLC training rooms (`349` rooms total; `171` rooms / `49.0%` contain interior `BLOCK` tiles; `8.61%` interior `BLOCK` density; `0.0%` interior `WALL`) showed `LogicNet` previously assigned `total=1.7929` (`topology_anchor_loss=1.0000`, `grid_reach_loss=0.5276`, `topology_trace_loss=0.4162`) to *ground-truth* VGLC rooms because anchors and directional `logic_source_mask`/`logic_target_mask` landed on the outer `WALL` border (`rows 0, 15`, `cols 0, 10`) rather than interior doorway aprons.
  - Updated `_graph_room_start_goal` and `_direction_mask` in [`src/zelda_data/graph_extraction.py`](src/zelda_data/graph_extraction.py) and `_direction_mask` in [`src/pipeline/generation/graph_context.py`](src/pipeline/generation/graph_context.py) to target interior doorway aprons strictly inside `[2, ROOM_HEIGHT - 3] x [2, ROOM_WIDTH - 3]`.
  - Updated `LogicNet._resolve_room_logic_targets` and `LogicNet.forward` in [`src/core/logic_net.py`](src/core/logic_net.py) to prefer interior explicit/start/goal anchors over outer-ring `door_target` and mask the 2-tile outer wall perimeter out of `topology_trace_loss`.
  - Post-fix verification on real VGLC ground-truth rooms reduced `LogicNet` `topology_anchor_loss` from `1.0000 -> 0.0625` (`total=1.7929 -> 0.5741`, `grid_reach_loss=0.5276 -> 0.1325`).
- **Self-Reinforcing Empty-Floor Cascade & Missing `BLOCK` Rare-Tile Weighting:**
  - Added `int(TileID.BLOCK)` to `DiscreteMaskedRoomModel.RARE_TILES` in [`src/core/discrete_masked_model.py`](src/core/discrete_masked_model.py) (previously `BLOCK` was excluded despite representing `8.61%` of interior VGLC tiles and `49.0%` of rooms).
  - Fixed `DiscreteMaskedRoomModel._iterative_fill` so stochastic unmasking keeps `token_temperature = max(1e-6, float(temperature))` without `anneal` temperature crushing, and ranks unmask order using cell-level `probs.max(dim=1).values` + annealed Gumbel noise instead of sampled-class probability `P(sampled_token)`.
  - Fixed deterministic test setup in `test_discrete_masked_model_rare_tile_weighting_loss_shift` ([`tests/test_discrete_masked_room_model.py`](tests/test_discrete_masked_room_model.py)) so weighted vs. unweighted normalized loss comparison uses a mixed `FLOOR` + `rare_id` grid without dropout RNG variance.
- **Batched `pooled` Conditioning Forward Alignment (Arm `N`):**
  - Forwarded `graph_conditioning_mode` inside `graph_ctx_for_guidance` in [`src/pipeline/generation/sampler.py`](src/pipeline/generation/sampler.py), synced `use_graph_node_cross_attention` in `pipeline_kwargs_from_resolved_config` ([`src/pipeline/config_bridge.py`](src/pipeline/config_bridge.py)) and `load_masked_room_model` ([`src/pipeline/models/model_manager.py`](src/pipeline/models/model_manager.py)), and aligned the fallback in [`src/pipeline/generation/graph_context.py`](src/pipeline/generation/graph_context.py).

### Changed (Breaking Checkpoint Schema — Retraining Required)
- Rebalanced conditioning capacity and cleaned inactive sampling knobs across all 4 canonical G-LMDM ablation YAMLs ([`configs/glmdm_canonical.yaml`](configs/glmdm_canonical.yaml), [`configs/glmdm_g_l_local_logic.yaml`](configs/glmdm_g_l_local_logic.yaml), [`configs/glmdm_g_graph.yaml`](configs/glmdm_g_graph.yaml), [`configs/glmdm_n_local.yaml`](configs/glmdm_n_local.yaml)):
  - `dataset.grid_augmentation`: enabled (`true`, equivariant V4 symmetry via `apply_room_symmetry_transform`)
  - `generation.guidance_scale`: `3.0 -> 1.0`, `generation.logic_guidance_scale`: `1.0 -> 0.0` (removed unused continuous-diffusion no-op overrides), `generation.masked_room_unmask_order`: `random`
  - `masked_room.context_dim`: `256 -> 96`
  - `masked_room.condition_hidden_dim`: `192 -> 96`
  - `masked_room.condition_num_attention_heads`: `8 -> 4`
  - `masked_room.rare_tile_weight`: `1.0 -> 2.0`
  - `masked_room.logic_topology_trace_weight` & `logic_topology_anchor_weight` (`G-LG`, `G-L`): `0.25 -> 0.1`
- Parameter budget reduction (verified via `MaskedRoomTrainer`):
  - `G-LG`: `denoiser=77,084`, `cond_enc=764,209`, `total=841,293` (down from `3,358,989`).
  - `N`: `denoiser=77,084`, `cond_enc=382,961`, `total=460,045` (down from `2,022,477`).

### Known Notes & Operational Caveats
- **Existing checkpoints under `outputs/glmdm_*/` are incompatible** with the updated `96`-dim conditioning configs and must be retrained from scratch for all 4 arms (`G-LG`, `G-L`, `G`, `N`).
- Because `masked_room.context_dim` is `96` while legacy `diffusion.context_dim` defaults to `256`, `check_training_hyperparameters.py --probe-data` logs an informational notice (not an error); masked-room checkpoints are not cross-loadable into the continuous diffusion stage.
- Publication evaluation requires `50–100` seeds per arm with Schema 2.0 attribution, Diversity/Novelty vs. the 45 unique VGLC interior layouts, Expressive Range Analysis (ERA), and external baselines (e.g., Graph Grammar + GAN) after retraining.
