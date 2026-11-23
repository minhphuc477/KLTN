# Current Architecture

Last updated: 2026-11-12

This is the concise, code-aligned description of the current stack. Use this
file as the canonical architecture reference for what exists in code now.

## System Summary

The project centers on **`G-LMDM` (`Graph-Conditioned Logic-Regularized Masked Discrete Model`)**, a repair-aware neuro-symbolic Zelda dungeon generator evaluated against continuous latent diffusion comparators and external symbolic baselines under the locked 1-to-1 protocol (`RG1(FG1)–RG4(FG4) -> PS1–PS4 -> RO1–RO4 -> RQ1–RQ4 -> H1–H4` in [`publication_guidance_card.json`](publication_guidance_card.json)):

1. build or accept a mission graph (`Block I`)
2. condition via causal `Block III GraphGPS + RRWP + neighbor/boundary` encoder
3. generate $16\times 11$ rooms directly in discrete token space with **`G-LMDM`** (`masked_room`, `3.58M` train / `3.36M` infer; `84.8K` denoiser) + direct $16\times 11$ Soft Bellman-Ford `LogicNet`, or compare against continuous `VQ-VAE` (`17.37M`) + `UNet[1,2]` (`47.13M` total) / `DiT+Flow` (`22.44M` total) (`Block II / IV / V`, `RQ2 / H2`)
4. enforce mission-critical semantics and attribute every step across `raw -> constrained -> repaired -> compiled` (`Block VI`)
5. stitch the rooms into a full dungeon
6. run independent formal verification (`StateSpaceAStar` solution witnesses vs. finite-frontier proofs vs. indeterminate cutoffs)
7. run `P-CBS` strictly as a bounded-rational proxy probe (`proxy_only`) and export Schema 2.0 artifacts

VGLC empirical corpus ground truth (`Data/The Legend of Zelda`): `450` room instances across `9` dungeons (`Quests 1–2`); held-out split allocates Dungeons `1–8` to training (`349` room instances, `217` unique $16\times 11$ grids, **`43` unique $12\times 7$ structural interiors**, `8` active tile IDs of `44` with `88.60%` `WALL+FLOOR` vs `0.68%` `ENEMY` and `0.05%` `STAIR`) and unseen Dungeon `9` (`101` room instances) strictly to testing.

Fast sampler remains a distilled efficiency extension, whereas continuous `VQ-VAE + UNet[1,2]` and `DiT+Flow` serve as matched-data comparators against primary discrete `G-LMDM` under `RO2`/`RQ2`/`H2`.

## Canonical Entry Points

- Training: `python main.py train --config configs/zelda_hmolqd.yaml --stage ...`
- Generation and evaluation: `python -m src.generate ...`
- GUI path: `python gui_runner.py`
- Canonical runtime orchestrator:
  [`src/pipeline/dungeon_pipeline.py`](../src/pipeline/dungeon_pipeline.py)

## Block Map

| Block | Role | Architectural Status | Main code |
|---|---|---|---|
| Block 0 | data parsing, room/graph alignment, stitching utilities | Active (Common) | [`src/data_processing/data_adapter.py`](../src/data_processing/data_adapter.py), [`src/zelda_data/zelda_core.py`](../src/zelda_data/zelda_core.py) |
| Block I | mission/topology graph generation and validation | Active (Stage 1) | `src/generation/*`, `src/evaluation/benchmark_suite.py` |
| Block II | continuous room tokenizer (`SemanticVQVAE` / `SemanticVQVAE2`) | **Eliminated in G-LMDM** (Retained strictly as continuous baseline comparator for $H_2$) | [`src/core/vqvae.py`](../src/core/vqvae.py) |
| Block III | local + global graph conditioning (GraphGPS + RRWP + Boundary Clamping) | Active (Stage 2) | [`src/core/condition_encoder.py`](../src/core/condition_encoder.py), [`src/pipeline/room_topology_conditioning.py`](../src/pipeline/room_topology_conditioning.py) |
| Block IV | discrete masked room generator (**G-LMDM**, primary) vs. latent diffusion (`UNet [1, 2]` / `DiT`, comparator) | Active (Stage 3) | [`src/core/discrete_masked_model.py`](../src/core/discrete_masked_model.py), [`src/train_masked_room.py`](../src/train_masked_room.py) |
| Block V | direct $16\times 11$ Soft Bellman-Ford `LogicNet` regularization | Active (Stage 4) | [`src/core/logic_net.py`](../src/core/logic_net.py) |
| Block VI | symbolic repair (`WaveFunctionCollapse` / `PathAnalyzer`), constructive stateful puzzle compiler (`PuzzleGenerator`), graph marker overlay, physical topology enforcement, stitching | Active (Stage 5) | [`src/pipeline/dungeon_pipeline.py`](../src/pipeline/dungeon_pipeline.py), [`src/generation/puzzle_generator.py`](../src/generation/puzzle_generator.py), [`src/core/symbolic_refiner.py`](../src/core/symbolic_refiner.py), [`src/generation/graph_constraint_enforcer.py`](../src/generation/graph_constraint_enforcer.py) |
| Block VII | dual-tier evaluation: exact `StateSpaceAStar` oracle vs. bounded `P-CBS` proxy diagnostics | Active (Stage 6) | `src/evaluation/*`, `src/simulation/*` |

## Core Module Classification & Pruning Policy (`src/core/`)

To prevent cargo-cult feature bloat on VGLC's small-data regime (`349` training rooms / `43` unique $12\times 7$ interiors), `src/core/` explicitly partitions modules into three tiers (enforced via `CANONICAL_GLMDM_MODULES`, `CONTINUOUS_COMPARATOR_MODULES`, and `DEPRECATED_ABLATION_ONLY_MODULES` in [`src/core/__init__.py`](../src/core/__init__.py)):

| Tier | Status in Canonical `G-LMDM` | Modules |
|---|---|---|
| **Tier 1 — Canonical `G-LMDM` Active Stack** | Active (5-Block Core) | `definitions.py`, `adapter.py`, `stitcher.py`, `condition_encoder.py` (GraphGPS+RRWP), `discrete_masked_model.py` (84.8K Transformer denoiser), `logic_net.py` (Direct Soft Bellman-Ford, `alpha_logic=0.1`), `symbolic_refiner.py`, `neural_guided_repair.py` (paired with [`src/generation/puzzle_generator.py`](../src/generation/puzzle_generator.py)) |
| **Tier 2 — Continuous Latent Comparators** | Comparator-Only (`H1`/`H2`) | `vqvae.py`, `latent_diffusion.py` (`UNet[1,2]`, `DiT+Flow`) |
| **Tier 3 — Deprecated / Ablation-Only** | Inactive (`loss_weight=0.0` / Excluded) | `gaussian_vae.py`, `perturb_and_map.py`, `puzzle_stage_semantics.py`, `attention_kernels.py` |

## Runtime Flow

### 1. Primary Proposed Pipeline: G-LMDM (Zero VQ-VAE)
```text
Mission graph G = (V, E, X_V, X_E) [Block I]
  -> Causal GraphGPS + RRWP topological encodings + t=0 hard boundary clamping [Block III]
  -> Direct 16x11 absorbing-state discrete masked diffusion Q_t [G-LMDM / Block IV]
  -> Direct 16x11 Soft Bellman-Ford LogicNet regularization [Block V]
  -> Direct discrete token output (Zero VQ-VAE decode)
  -> Structural cleanup + semantic constrained decode [Schema 2.0 Stage 2]
  -> Stateful puzzle grammar & deterministic graph-marker overlay [Schema 2.0 Stage 3]
  -> Symbolic WFC / boundary passage repair [Block VI]
  -> Stitched complete dungeon grid [Schema 2.0 Stage 4]
  -> Dual-tier verification: Formal StateSpaceAStar proofs vs. Bounded P-CBS diagnostics [Block VII]
```

### 2. Comparative Baseline Pipeline: H-MOLQD (Continuous Comparator)
```text
Mission graph G [Block I]
  -> Condition encoder [Block III]
  -> Continuous latent diffusion / flow matching [Block IV: UNet/DiT]
  -> Continuous latent decode z_q -> 16x11 [Block II: SemanticVQVAE]
  -> Symbolic repair & evaluation [Blocks VI - VII]
```

## Current Canonical Config

The validated config surface is
[`configs/zelda_hmolqd.yaml`](../configs/zelda_hmolqd.yaml).

Current important defaults:

| Area | Current value |
|---|---|
| Dataset schema | `zelda_v1`, `44` classes, `16x11` rooms |
| Topology anchor policy | `2026-04-11.semantic_anchor_v8_puzzle_subtype_channels` |
| Room-topology channels | `54` |
| Canonical VQ-VAE YAML default | `architecture=vqvae` (`vqvae2` optional), `hidden_dim=96`, `codebook_size=256`, `latent_dim=64`, VQ or FSQ |
| Canonical diffusion comparator (`UNet` / `DiT`) | `denoiser_backbone=unet` (`dit` optional), `unet_channel_mult=[1, 2]`, `unet_attention_resolutions=[0, 1]`, `model_channels=96`, `condition_hidden_dim=192`, `condition_gnn_type=gps`, `training_objective=diffusion` (`flow_matching` optional), `latent_clip_bound=5.0`, `scheduler_t0=100`, `use_amp=true` (`fp16`) |
| Canonical **`G-LMDM`** discrete config (`masked_room`) | `hidden_dim=48`, `unet_num_heads=4`, `condition_hidden_dim=192`, `condition_gnn_type=gps`, `logic_net_enabled=true`, `alpha_logic=0.1`, `room_topology_channels=54` |
| Graph node features | `14` baseline; optional `15`th normalized floor/z feature for labeled multi-floor ablations |
| Generation defaults | constrained decode `on`, deterministic marker overlay `on`, repair `on`, puzzle scaffold `on`, puzzle novelty search `on` |
| Diffusion/fast-sampler training efficiency | frozen VQ-VAE latent cache `on`, `4096` max entries |

Note: [`configs/zelda_hmolqd.yaml`](../configs/zelda_hmolqd.yaml) configures primary discrete **`G-LMDM`** (`masked_room`: `84,764` transformer denoiser + `3,272,497` GPS/RRWP+ref conditioner + `227,644` LogicNet = **`3,584,905` (`3.58M` train) / `3,357,261` (`3.36M` inference)**, paired with [`configs/zelda_hmolqd_masked_small.yaml`](../configs/zelda_hmolqd_masked_small.yaml) for the `G`-only `logic_net_enabled=false` ablation) alongside continuous comparators (`VQ-VAE` `17,365,900` + `UNet[1,2]` `26,332,390` + `ConditionEncoder` `3,200,113` + `LogicNet` `227,644` = **`47,126,047` (`47.13M`)**, or `DiT+Flow` `1,642,694` denoiser = **`22,436,351` (`22.44M`)**).

## Conditioned Semantics

The current room-generation contract is stronger than the older generic
"puzzle room" setup.

The topology conditioning path now carries:

- graph node role channels
- puzzle subtype channels
  - `tutorial_puzzle`
  - `combat_puzzle`
  - `complex_puzzle`
  - `switch_puzzle`
- edge-semantic gate families
  - `key_locked`
  - `bombable`
  - `item_gate` / `item_locked`
  - `switch_locked`
  - `on_off_gate` / `state_block`
- optional explicit floor/z identity when `dataset.node_feature_dim=15`; this
  mode requires at least two floor labels and therefore cannot be claimed from the
  original single-floor corpus

Runtime puzzle scaffolds can then specialize to those semantics instead of
using one generic obstacle template.

## Production vs Research Branches

### Production branch

- `latent diffusion`
- graph-conditioned decode
- deterministic overlay
- symbolic repair

### Active Primary Architecture & Comparative Branches

- **Primary Discrete Architecture (`G-LMDM`)**: [`DiscreteMaskedRoomModel`](../src/core/discrete_masked_model.py) (`condition_gnn_type=gps`, `logic_net_enabled=true`, `alpha_logic=0.1`; `3.58M` train / `3.36M` infer), paired with [`configs/zelda_hmolqd_masked_small.yaml`](../configs/zelda_hmolqd_masked_small.yaml) (`G` without LogicNet)
- **Continuous Comparators (`RQ2 / H2`)**: `SemanticVQVAE` + `UNet[1,2]` (`47.13M` total) and `DiT + Flow Matching` (`22.44M` total)
- `fast sampler`: distilled research extension ([`src/train_lcm.py`](../src/train_lcm.py))
- attributed 4-stage ablation runs (`raw -> constrained -> repaired -> compiled`)

Until full multi-seed comparative runs on Dungeon 9 complete (`refine-logs/EXPERIMENT_PLAN.md`), both the discrete `G-LMDM` stack and the continuous `VQ-VAE + UNet[1,2]` stack remain supported in the canonical pipeline without premature performance claims.

## Model-Block Efficiency Boundary

The current training bottleneck is not only U-Net attention. The production
path already uses PyTorch SDPA where available and has a linear graph-attention
option for larger graph contexts. On this Zelda corpus, the more concrete
repeat cost is frozen Block-II encoding inside downstream training:

- Block IV diffusion revisits the same room maps for many epochs.
- Block IV-B fast-sampler distillation reuses `DiffusionTrainer` as its frozen
  teacher bundle, so it inherits the same tokenizer path.
- teacher-forced neighbor maps also pass through the frozen VQ-VAE.

The code now caches frozen VQ-VAE latents during downstream training. This does
not cache trainable Block-III graph conditioning, LogicNet predicted-latent
losses, or Block-VI repair, because those either need gradients, depend on
runtime choices, or are correctness checks rather than repeated frozen
preprocessing.

## Learned Stage Semantics

The repo now has an explicit `trainable ordered puzzle-stage semantics path`.

It is implemented in two layers, not only in runtime grammar:

- loader/runtime build ordered `puzzle_stage_condition` metadata from the same
  validator-aware room semantics
- diffusion and masked-room conditioning can append deterministic stage tokens
- room-topology priors can optionally rasterize ordered stage traces
- diffusion, masked-room, and fast-sampler training now also have an auxiliary
  learned `puzzle-stage semantics head`
  - it predicts gate family
  - sequence-required flag
  - stage count
  - ordered stage slots
  - from generated room logits during training

Important boundary:

- this stronger path is implemented in code
- it is `off by default`
- old checkpoints did **not** learn it
- the older `stageconditioned_v1` branch used token/trace conditioning only and
  is now outdated for the stronger claim
- any claim about `learned multi-step puzzle semantics` now requires retraining
  with:
  - `diffusion.puzzle_stage_conditioning_enabled=true`
  - `diffusion.puzzle_stage_semantics_loss_weight>0`
  - `masked_room.puzzle_stage_conditioning_enabled=true`
  - `masked_room.puzzle_stage_semantics_loss_weight>0`
  - optionally `fast_sampler.puzzle_stage_conditioning_enabled=true`
  - optionally `fast_sampler.puzzle_stage_semantics_loss_weight>0`

## Validation Contract

Protocol exports now carry explicit validation/search artifacts instead of only
tile-cleanup metrics:

- per-variant `validation_search_stats.json`
- root-level `search_algorithm_comparison.json` for manual compare and fixed-graph audits
- graph progression / goal-gauntlet checks alongside grid A* and CBS

Validation roles are now explicit.

- report-facing hard oracle: hybrid mechanical contract
  - graph-guided room oracle
  - graph progression validator
  - deterministic softlock check
- stricter stress probe: monolithic stitched tile-state `A*`
- comparison solvers: BFS, Dijkstra, Greedy, D* Lite, DFS/IDDFS, Bidirectional A*
- behavioral probe: `P-CBS` / `CognitiveBoundedSearch`
- optional learned-policy probe: per-level tabular RL playtesters in
  `run_pcbs_persona_map_sweep.py --include-rl-ablation`; these are ablations,
  never hard solvability oracles
- VGLC benchmark handoff preserves the stitched mission graph and room/node
  mappings. Grid-only evaluation is reserved for artifacts with no graph
  transitions; the cardinal-only tabular RL arm explicitly skips certified
  paths that require stairs or warps.
- A* and its Dijkstra safeguard consume one shared state-expansion budget and
  report aggregate work. MAP-Elites archive sampling uses checkpointed,
  instance-local RNG state for reproducible resumed runs.
- excluded from canonical export comparison: `parallel_astar`, `multi_goal`,
  `key_economy_validator`, and `solver_comparison`

This makes the repo's "playable" claim depend on observable search evidence and
the current patched stateful-puzzle contract, not only on visual quality or
repair counts.

Current protocol exports also expose an end-to-end structural evaluation layer
at the stitched room-grid level:

- `room_unique_ratio`
- `room_pairwise_ncd`
- `room_nearest_reference_ncd`
- room / dungeon symbol entropy

These do not replace the topology benchmark descriptors. They complement them so
the final exported dungeons have a report-facing diversity/novelty signal, not
only graph-level expressivity metrics.

Important clarification:

- monolithic stitched tile-state `A*` is still useful and is still reported
- on the current stateful multi-step puzzle slice it times out often enough
  that it should be treated as a harsher stress probe, not the only
  report-facing pass/fail gate

## Checkpoint and Reproducibility Contract

The current stack expects:

- checkpoints to carry enough metadata to reconstruct component shapes
- generation to reuse the nearest `resolved_config.yaml/json` snapshot
- CLI/YAML overrides to flow through `src/config_system.py`

This means "current behavior" is defined by:

1. resolved config
2. checkpoint metadata
3. runtime generation overrides

not by private hardcoded script defaults.

## Where To Read Next

- High-level rationale, live parameter counts, and ablations:
  [`CANONICAL_MODEL_RATIONALE_ABLATION_AND_COMPLEXITY_GUIDE.md`](CANONICAL_MODEL_RATIONALE_ABLATION_AND_COMPLEXITY_GUIDE.md)
- Block I/O tensor and contract specifications:
  [`BLOCK_IO_REFERENCE.md`](BLOCK_IO_REFERENCE.md)
- Grammar rules, progression constraints, and constraint enforcement:
  [`GRAMMAR_REFERENCE.md`](GRAMMAR_REFERENCE.md)
- Search algorithms, strict admissibility, and GUI reference:
  [`SOLVERS_AND_GUI_REFERENCE.md`](SOLVERS_AND_GUI_REFERENCE.md)
- Topology commands and manual graph workflows:
  [`TOPOLOGY_COMMANDS.md`](TOPOLOGY_COMMANDS.md)
- Current non-GUI audit ledger and claim boundary:
  [`architecture_audit_research_notes.md`](architecture_audit_research_notes.md)
