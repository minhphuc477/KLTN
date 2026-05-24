# Current Architecture

Last updated: 2026-05-24

This is the concise, code-aligned description of the current stack. Use this
file as the canonical architecture reference. Older deep-dive notes remain in
`docs/` and `docs/archive/`, but this file owns the "what exists now" answer.

## System Summary

The project is a `graph-first hybrid neural-symbolic Zelda dungeon generator`.
The current thesis-safe path is:

1. build or accept a mission graph
2. generate room layouts with the diffusion branch under graph conditioning
3. enforce mission-critical semantics with constrained decode, overlay, repair,
   and hybrid stateful puzzle grammar
4. stitch the rooms into a full dungeon
5. run the hybrid mechanical contract
6. run `P-CBS` as the bounded-rational behavioral probe
7. export metrics/artifacts

Fast sampler and masked-room still exist, but they remain research branches.

## Canonical Entry Points

- Training: `python main.py train --config configs/zelda_hmolqd.yaml --stage ...`
- Generation and evaluation: `python -m src.generate ...`
- GUI path: `python gui_runner.py`
- Canonical runtime orchestrator:
  [`src/pipeline/dungeon_pipeline.py`](../src/pipeline/dungeon_pipeline.py)

## Block Map

| Block | Role | Main code |
|---|---|---|
| Block 0 | data parsing, room/graph alignment, stitching utilities | [`src/data_processing/data_adapter.py`](../src/data_processing/data_adapter.py), [`src/zelda_data/zelda_core.py`](../src/zelda_data/zelda_core.py) |
| Block I | mission/topology graph generation and validation | `src/generation/*`, `src/evaluation/benchmark_suite.py` |
| Block II | semantic room tokenizer (VQ-VAE) | [`src/core/vqvae.py`](../src/core/vqvae.py) |
| Block III | local + global graph conditioning | [`src/core/condition_encoder.py`](../src/core/condition_encoder.py), [`src/pipeline/room_topology_conditioning.py`](../src/pipeline/room_topology_conditioning.py) |
| Block IV | latent diffusion teacher | [`src/core/latent_diffusion.py`](../src/core/latent_diffusion.py) |
| Block V | LogicNet guidance | [`src/core/logic_net.py`](../src/core/logic_net.py), [`src/core/latent_diffusion.py`](../src/core/latent_diffusion.py) |
| Block VI | symbolic repair, graph marker overlay, puzzle scaffolds, stitching | [`src/pipeline/dungeon_pipeline.py`](../src/pipeline/dungeon_pipeline.py), [`src/core/symbolic_refiner.py`](../src/core/symbolic_refiner.py) |
| Block VII | metrics, validation, QD / MAP-Elites hooks | `src/evaluation/*`, `src/simulation/*` |

Auxiliary branches:

- fast sampler distillation: [`src/train_lcm.py`](../src/train_lcm.py)
- masked-room branch: [`src/core/discrete_masked_model.py`](../src/core/discrete_masked_model.py), [`src/train_masked_room.py`](../src/train_masked_room.py)

Auxiliary-branch training now also includes `topology-focused supervision`
that upweights sparse anchors, doors, typed gates, and traversability traces.
This is the current path toward stronger neural semantics without dropping the
hybrid runtime safeguards prematurely.

## Runtime Flow

```text
Mission graph
  -> room order + graph features + topology maps
  -> condition encoder
  -> diffusion / fast sampler / masked-room branch
  -> VQ-VAE decode to semantic room grid
  -> structural cleanup + semantic constrained decode
  -> hybrid stateful puzzle grammar / interaction-sequence enforcement
  -> deterministic graph-marker overlay
  -> symbolic repair / fallback
  -> stitched dungeon
  -> validation handoff contract
  -> hybrid mechanical contract
  -> P-CBS
  -> reports
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
| Canonical VQ-VAE YAML default | `hidden_dim=96`, `codebook_size=256`, `latent_dim=64` |
| Best-tested tokenizer by held-out VQ-VAE validation loss | `codebook256` |
| Canonical diffusion config | `model_channels=96`, `condition_hidden_dim=192`, `condition_gnn_type=gps` |
| Canonical masked-room config | `hidden_dim=48`, `condition_gnn_type=gcn`, `room_topology_channels=54` |
| Generation defaults | constrained decode `on`, deterministic marker overlay `on`, repair `on`, puzzle scaffold `on`, puzzle novelty search `on` |
| Diffusion/fast-sampler training efficiency | frozen VQ-VAE latent cache `on`, `4096` max entries |

Important distinction:

- the `canonical YAML` still uses a `256`-entry VQ-VAE
- the latest verified downstream experimental branch used an explicit external
  `codebook512` VQ-VAE checkpoint during downstream training/inference

That distinction is intentional. The codebase supports both:

- stable canonical defaults in `configs/zelda_hmolqd.yaml`
- high-capacity downstream comparison branches via explicit checkpoint handoff

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

Runtime puzzle scaffolds can then specialize to those semantics instead of
using one generic obstacle template.

## Production vs Research Branches

### Production branch

- `latent diffusion`
- graph-conditioned decode
- deterministic overlay
- symbolic repair

### Research branches

- `fast sampler`
- `masked room`
- pure-neural / reduced-fallback ablations

Current evidence still says diffusion is the only branch that should be treated
as the production baseline.

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

- High-level rationale and report-writing detail:
  [`CANONICAL_MODEL_RATIONALE_ABLATION_AND_COMPLEXITY_GUIDE.md`](CANONICAL_MODEL_RATIONALE_ABLATION_AND_COMPLEXITY_GUIDE.md)
- Topology commands and manual graph workflows:
  [`TOPOLOGY_COMMANDS.md`](TOPOLOGY_COMMANDS.md)
- Current VQ-VAE-2, LogicNet, repair, and ablation protocol:
  [`VQVAE2_LOGICNET_REPAIR_ABLATION_PROTOCOL_2026_05_23.md`](VQVAE2_LOGICNET_REPAIR_ABLATION_PROTOCOL_2026_05_23.md)
- Current artifact / checkpoint status and retraining alerts:
  [`ARTIFACT_AND_CHECKPOINT_STATUS_2026_04_18.md`](ARTIFACT_AND_CHECKPOINT_STATUS_2026_04_18.md)
- Final production/finalization review and remaining required runs:
  [`PRODUCTION_FINALIZATION_REVIEW_2026_04_18.md`](PRODUCTION_FINALIZATION_REVIEW_2026_04_18.md)
- Current chat handoff context:
  [`NEXT_CHAT_CONTEXT_2026_04_18.md`](NEXT_CHAT_CONTEXT_2026_04_18.md)
- Archived auxiliary/neural-semantics and playability provenance notes:
  [`archive/2026-q2/AUXILIARY_BRANCH_AND_NEURAL_SEMANTICS_AUDIT_2026_04_15.md`](archive/2026-q2/AUXILIARY_BRANCH_AND_NEURAL_SEMANTICS_AUDIT_2026_04_15.md),
  [`archive/2026-q2/PLAYABILITY_EVALUATION_AND_CBS_RESEARCH_2026_04_16.md`](archive/2026-q2/PLAYABILITY_EVALUATION_AND_CBS_RESEARCH_2026_04_16.md)
- Latest VQ-VAE audit:
  [`VQVAE_RESEARCH_AUDIT_2026_04_10.md`](VQVAE_RESEARCH_AUDIT_2026_04_10.md)
