# Changelog

All notable architectural, training, validation, and data-pipeline changes to the KLTN Neuro-Symbolic Dungeon Generation project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] - 2026-11-24

### Added
- **Bipartite & Degree-Cap Enforcement in Block I (`EvolutionaryTopologyGenerator`):**
  - Extracted `@staticmethod _edge_prune_priority` and added `_edge_removal_preserves_connectivity` and `_enforce_bipartite_cycles` in [`src/generation/evolutionary_director/executor.py`](src/generation/evolutionary_director/executor.py).
  - Wired `_enforce_max_degree(..., max_degree=4)` and `_enforce_bipartite_cycles(...)` into both candidate execution (`GraphGrammarExecutor.execute`) and final export (`EvolutionaryTopologyGenerator._finalize_graph_output` in [`src/generation/evolutionary_director/generator.py`](src/generation/evolutionary_director/generator.py)).
  - Non-bridge odd-cycle `PATH`/`SHORTCUT` edges are pruned deterministically before export so graphs remain 2-colorable and `max_degree <= 4` for strict 2D orthogonal grid stitching.
- **Mask-Ratio Gating for LogicNet Training Supervision:**
  - Exposed `mask_ratio_per_sample` in `DiscreteMaskedRoomModel.training_loss(..., return_aux=True)` ([`src/core/discrete_masked_model.py`](src/core/discrete_masked_model.py)).
  - Multiplied `logic_loss` by `low_mask_gate = (1.0 - mask_ratio_per_sample.clamp(0.0, 1.0)).pow(2).mean()` in `MaskedRoomTrainer._run_step` ([`src/train_masked_room.py`](src/train_masked_room.py)) so noisy high-mask predictions (`mask_ratio -> 1.0`) do not overwhelm tile-level MLE reconstruction while preserving dungeon-batch alignment.
- **Regression Tests:**
  - `test_logicnet_vglc_two_tile_border_interior_anchors` in [`tests/test_logicnet_fixes.py`](tests/test_logicnet_fixes.py).
  - `test_block_in_rare_tiles_and_mask_ratio_aux_and_stochastic_sample` in [`tests/test_discrete_masked_room_model.py`](tests/test_discrete_masked_room_model.py).
  - `test_executor_enforce_bipartite_cycles_and_degree_cap` in [`tests/test_room_stitching.py`](tests/test_room_stitching.py).

### Fixed
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
- Rebalanced conditioning capacity across all 4 canonical G-LMDM ablation YAMLs ([`configs/glmdm_canonical.yaml`](configs/glmdm_canonical.yaml), [`configs/glmdm_g_l_local_logic.yaml`](configs/glmdm_g_l_local_logic.yaml), [`configs/glmdm_g_graph.yaml`](configs/glmdm_g_graph.yaml), [`configs/glmdm_n_local.yaml`](configs/glmdm_n_local.yaml)):
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
