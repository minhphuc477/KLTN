# Architecture Positioning and Attention Notes

Last updated: 2026-04-02

This note answers five recurring questions about the current Zelda generator:

1. How to compare it fairly against "single-level" or "single-example" diffusion papers.
2. Why the architecture is large and decomposed instead of monolithic.
3. Why generation is topology-first and room-by-room instead of whole-dungeon-at-once.
4. Which attention layers and non-standard twists actually exist in code.
5. Which hyperparameters are still hard-coded or runtime-defaulted outside the main YAML training surface.

Primary research anchors:

- Dai et al., *Procedural Level Generation with Diffusion Models from a Single Example*, AAAI 2024. https://ojs.aaai.org/index.php/AAAI/article/view/28865
- *Text-to-Level Diffusion Models With Various Text Encoders for Super Mario Bros*, arXiv 2507.00184. https://arxiv.org/abs/2507.00184
- Xu et al., *Prompt-Free Diffusion*, CVPR 2024. https://openaccess.thecvf.com/content/CVPR2024/papers/Xu_Prompt-Free_Diffusion_Taking_Text_out_of_Text-to-Image_Diffusion_Models_CVPR_2024_paper.pdf
- Inoue et al., *LayoutDM*, CVPR 2023. https://openaccess.thecvf.com/content/CVPR2023/papers/Inoue_LayoutDM_Discrete_Diffusion_Model_for_Controllable_Layout_Generation_CVPR_2023_paper.pdf
- Shabani et al., *HouseDiffusion*, CVPR 2023. https://openaccess.thecvf.com/content/CVPR2023/papers/Shabani_HouseDiffusion_Vector_Floorplan_Generation_via_a_Diffusion_Model_With_Discrete_CVPR_2023_paper.pdf
- Dormans and Bakkes, *Generating Missions and Spaces for Adaptable Play Experiences*, IEEE TCIAIG 2011. https://pure.hva.nl/ws/files/149264/453867_Dormans_Bakkes_-_Generating_Missions_and_Spaces_for_Adaptable_Play_Experiences.pdf

Status of this note:

- `Implemented and verified`: the repo already exposes nearly all training-time methodology knobs through `src/config_system.py`.
- `Documented here`: the remaining hidden knobs are mainly inference/runtime defaults and fallback constructor defaults, not the canonical training surface.
- `Still open`: benchmark evidence, not code, is the main remaining gap when comparing against room-only single-example diffusion papers.

## 1. Comparison Boundaries

The right way to protect this project from unfair comparison is not to avoid comparison. It is to define the comparison target correctly.

### What the single-example diffusion papers actually solve

The AAAI 2024 single-example paper is a room-scale or local-level synthesis result under extreme data scarcity. Its strongest claim is that dense semantic representations and locality-biased denoisers can learn useful level structure from a very small amount of content. That is highly relevant to our room generator.

It is not a fair full-system baseline for:

- mission-graph generation,
- progression logic,
- lock-key ordering,
- multi-room controllability,
- symbolic repair,
- mixed neural-symbolic validity enforcement.

The July 2025 Mario paper points in the same direction: compact in-domain conditioning works well, but longer-horizon level construction remains hard, and mixed-initiative composition is still useful.

### Fair benchmark framing for this repo

This project should be compared in three separate tiers:

1. `Topology generation` against mission-graph / graph-generation baselines such as ASP-style Zelda graph generation, GraphRNN, DiGress, and search-based dungeon graph methods.
2. `Room generation` against room/layout baselines such as single-example diffusion PCG, LayoutDM-style controllable structured generation, or other room-local generative baselines.
3. `Full-stack dungeon generation` against hybrid graph-plus-room systems, not against room-only generators.

If a reviewer compares the whole stack directly against a single-example room generator, the right response is:

- that is a valid `room-branch` comparison;
- it is not a valid `full-dungeon progression-system` comparison;
- both should be reported, but they should not be collapsed into one headline claim.

### Thesis-safe claim

The thesis-safe claim is:

- this repo is a `hybrid controllable dungeon generator` with explicit progression structure and room-local neural synthesis;
- it should not be marketed as a direct drop-in substitute for a `single-example room generator`;
- it solves a broader and stricter problem than those room-only papers, but at higher system complexity.

## 2. Why the Architecture Is This Large

The architecture is large because it is solving several different subproblems at once, and they do not share the same inductive bias.

| Subproblem | Current block | Why one small model is not enough |
|---|---|---|
| Progression structure | Block I topology generator | Keys, locks, branching, pacing, and reachability are sparse symbolic dependencies, not just local texture generation. |
| Room appearance and local semantics | Blocks II-IV / masked-room branch | Tile synthesis is spatial and local; it benefits from convolution, attention, and local conditioning. |
| Global validity pressure during sampling | Block V LogicNet guidance | Playability is not guaranteed by a plain generator. |
| Hard constraint cleanup | Block VI symbolic refiner | Even a strong neural model can still violate local tile rules or door consistency. |
| Diversity/search | Block VII MAP-Elites | Quality-diversity is an evaluation/search layer, not a denoising layer. |

So the repo is not "big because we wanted a big model." It is big because it is really a stack of:

- a graph generator,
- a room generator,
- a guidance model,
- a symbolic constraint repairer,
- and an archive/evaluation layer.

If we removed the symbolic and topology layers, the system would become smaller, but it would also stop solving the current research question.

## 3. Why Topology First and Room by Room

This design is intentional and still the correct one for this codebase.

### Why not generate the whole dungeon in one pass

Generating a whole Zelda dungeon as one grid would force one model to learn all of the following jointly:

- room content,
- room boundaries,
- inter-room adjacency,
- progression semantics,
- global pacing,
- long-horizon lock/key consistency.

That is exactly the kind of long-horizon structured dependency that the recent PCG diffusion papers still struggle with.

### Why topology first

Topology first is supported by Dormans and Bakkes and also by the empirical limits visible in the newer diffusion-PCG work.

The topology graph carries:

- start-goal structure,
- room ordering,
- branch shape,
- gating constraints,
- target room count,
- and designer control signals.

Once that is fixed, the room model only needs to solve `room realization under context`, which is a much better fit for convolutional and attention-based generators.

### Why room by room

Room-by-room generation helps in four concrete ways:

1. `Sample efficiency`: the dataset becomes a room dataset, not a tiny set of full dungeons.
2. `Factorization`: progression is solved in graph space; tile realization is solved in room space.
3. `Local conditioning`: neighboring room maps and boundary constraints become meaningful conditioning signals.
4. `Repairability`: if one room is bad, we can re-generate or repair one room instead of the entire dungeon.

## 4. How Such a Small Dataset Can Feed the Model

The short answer is: only because the repo is heavily biased in its favor.

### What makes the data regime survivable

1. `Schema lock`
   The repo is locked to `zelda_v1`: fixed room size `16x11`, fixed tile vocabulary `44`, fixed graph feature schema. That dramatically reduces entropy.

2. `Room-level decomposition`
   Training is mostly on rooms, not on a tiny number of stitched full dungeons. The VQ-VAE stage explicitly trains with `room_level=True`, and diffusion/masked-room stages also work room-wise.

3. `Replacement sampling`
   VQ-VAE training uses replacement sampling with `min_samples_per_epoch`, so tiny datasets still produce a usable number of gradient steps per epoch.

4. `Strong inductive bias`
   The model is not learning from raw pixels. It gets:
   - semantic tiles,
   - graph features,
   - topology maps,
   - neighboring room maps,
   - and symbolic repair.

5. `Latent compression`
   The VQ-VAE shrinks room representation before diffusion, so the denoiser does not have to model full-resolution tile grids directly.

6. `Mixed neural-symbolic recovery`
   The symbolic refiner can repair some failures instead of asking the neural model to do everything perfectly.

### What this does not mean

It does not mean the dataset is "sufficient" in a generic ML sense. It means the repository has been engineered so that a small Zelda corpus is still usable for this very specific problem.

This is also why the single-example paper is not automatically an existential threat to the architecture. That paper shows what strong locality bias can do for one local synthesis task. Our repo survives small data by combining:

- locality bias,
- schema lock,
- room-level factorization,
- graph-conditioned control,
- and symbolic recovery.

That is a different operating point from "one denoiser learns everything."

## 5. Attention Layer Inventory

This section lists the real attention-bearing modules in the code and the non-standard twists added on top of standard attention.

### Block III: condition encoder attention

| Module | File | Type | Standard part | Repo-specific twist |
|---|---|---|---|---|
| `GPSLayer.global_attn` | `src/core/condition_encoder.py` | `nn.MultiheadAttention` | Standard transformer-style global self-attention over graph tokens | Paired with local GATv2 / fallback message passing in a GraphGPS-style block |
| `CrossAttentionFusion` | `src/core/condition_encoder.py` | Cross-attention from local room anchor to global graph tokens | Standard Q from local, K/V from global | Used only as a single-query fusion layer between local neighbor context and graph tokens |

### Block IV: diffusion attention

| Module | File | Type | Standard part | Repo-specific twist |
|---|---|---|---|---|
| `SelfAttention` | `src/core/latent_diffusion.py` | U-Net self-attention | Standard MHSA with SDPA fallback | None beyond the usual U-Net integration |
| `CrossAttention` | `src/core/latent_diffusion.py` | Conditioning cross-attention | Standard Q from latent tokens, K/V from conditioner | Optional `linear_hedgehog` kernel; optional topology refinement of context tokens before K/V |
| `AttentionBlock` | `src/core/latent_diffusion.py` | Combined self + cross attention block | Standard U-Net attention pattern | Wrapped with spatial graph conditioning and room-topology conditioning |

### Graph-to-grid attention path

| Module | File | Type | Standard part | Repo-specific twist |
|---|---|---|---|---|
| `GraphToGridCrossAttention` | `src/core/graph_grid_attention.py` | Per-position grid-to-graph cross-attention | Q from grid positions, K/V from graph nodes | 2D grid positional encoding, graph node position encoding, degree/current-distance bias, optional lightweight GCN prepass, optional `linear_hedgehog`, auto-switch to linear above a node-count threshold |
| `RoomTopologyConditioner` | `src/core/graph_grid_attention.py` | Spatial conditioning over topology maps | Additive bias or SPADE-style modulation | Explicit room-topology map injection |
| `SpatialGraphConditioner` | `src/core/graph_grid_attention.py` | Combined topology + graph conditioning | Standard residual conditioning wrapper | Learnable sigmoid gates on graph and topology contributions, initialized small but non-zero |
| `EnhancedAttentionBlock` | `src/core/graph_grid_attention.py` | Alternate richer U-Net attention block | Self-attn + graph cross-attn + fallback context cross-attn | Exists as an upgrade path but is not the main canonical path documented in the YAML |

### Masked-room branch

The masked-room model in `src/core/discrete_masked_model.py` reuses the same `UNetDenoiser` attention path as the latent diffusion branch. So the masked-room branch inherits:

- standard self-attention,
- standard cross-attention,
- optional `linear_hedgehog`,
- spatial graph conditioning,
- additive or SPADE topology conditioning,
- graph/topology gates.

## 6. What the Attention "Twists" Actually Are

These are the main non-standard modifications beyond a plain diffusion U-Net:

1. `linear_hedgehog` attention
   Implemented in `src/core/attention_kernels.py`.
   This replaces softmax attention with a trainable feature-map approximation when selected, and graph-to-grid attention can auto-switch to it above a node threshold.

2. `topology_refinement_mode`
   Implemented inside `src/core/latent_diffusion.py::CrossAttention`.
   Options:
   - `none`
   - `lightweight`
   - `gat2`

   This refines graph/token context before K/V projection, which is not part of standard text-to-image cross-attention.

3. `current_node_distance` bias
   Implemented in graph conditioning and graph-grid attention.
   This explicitly biases conditioning toward the current room's relative position in the mission graph.

4. `RoomTopologyConditioner`
   Injects room-topology maps either additively or via SPADE-style affine modulation.

5. `graph/topology gates`
   `SpatialGraphConditioner` uses learnable sigmoid gates so graph and topology signals start weak and are learned into the denoiser, instead of being slammed in at full strength from step 0.

6. `CFG scheduling`
   The diffusion model supports `constant`, `linear_decay`, and `cosine_decay` schedules for classifier-free guidance, instead of a single fixed CFG scale across all denoising steps.

7. `LogicNet gradient guidance`
   This is not attention, but it is a major extra steering path layered onto diffusion sampling and should be described alongside the attention twists.

## 7. Remaining Hard-Coded Hyperparameters

Most training-time hyperparameters are already in `src/config_system.py`. The main remaining hard-coded items are runtime and fallback defaults.

### A. Inference-time defaults that are still hard-coded in the pipeline

| Parameter | Current value | Location | Why it matters |
|---|---:|---|---|
| `guidance_scale` | `7.5` | `src/pipeline/dungeon_pipeline.py` | Default CFG at inference is stronger than training default `3.0`; this can distort comparisons if not reported. |
| `logic_guidance_scale` | `1.0` | `src/pipeline/dungeon_pipeline.py` | Default LogicNet guidance at inference is still hard-coded. |
| `start_goal_coords` | `((1,5),(14,5))` | `src/pipeline/advanced_pipeline.py` | Advanced pipeline fallback injects a specific start/goal pattern. |

These should be promoted if the runtime path is meant to be fully reproducible from YAML alone.

Important distinction:

- these are `real reproducibility concerns` for generation;
- they are `not` hidden training hyperparameters for the canonical experiment path.

### B. Random-init fallback model defaults in the pipeline

These are mostly compatibility defaults used only when a checkpoint is missing or incomplete.

| Component | Current fallback defaults | Location |
|---|---|---|
| Condition encoder | `condition_hidden_dim=256`, `context_dim=256`, `condition_num_gnn_layers=3`, `condition_num_attention_heads=8`, `condition_dropout=0.1` | `src/pipeline/dungeon_pipeline.py` |
| Diffusion | `num_timesteps=1000`, `prediction_type=epsilon`, `cfg_scale=3.0`, `min_snr_gamma=5.0`, `model_channels=128`, `unet_channel_mult=(1,2,4)`, `unet_num_res_blocks=2`, `unet_attention_resolutions=(1,2)`, `unet_num_heads=8`, `unet_dropout=0.1`, `graph_auto_linear_attention_nodes=128`, gate init `-2.0` | `src/pipeline/dungeon_pipeline.py` |
| LogicNet | `num_iterations=20`, topology weights `0.25 / 0.25` | `src/pipeline/dungeon_pipeline.py` |
| Masked-room | `hidden_dim=64`, `model_channels=128`, `context_dim=condition_encoder.output_dim or 256`, `attention_mode=diffusion_attention_mode`, `topology_conditioning_mode=additive`, `unet_channel_mult=(1,2,4)`, `unet_num_res_blocks=2`, `unet_num_heads=8` | `src/pipeline/dungeon_pipeline.py` |

These are not the canonical experiment settings, but they should still be called out because they can affect ad hoc interactive runs and fallback behavior.

### C. Constructor defaults duplicated outside the YAML path

`NeuralSymbolicDungeonPipeline.__init__` still repeats many defaults that also exist in the validated config:

- topology defaults,
- condition encoder defaults,
- fast-sampling defaults,
- masked sampling defaults,
- symbolic repair defaults.

This is acceptable for compatibility, but it means the repo still has two layers of defaults:

1. the canonical `config_system.py` defaults,
2. the pipeline constructor defaults.

For strict reproducibility, the pipeline should eventually be instantiated only from resolved config-derived values.

### D. Defaults that look scary but are not the main reproducibility problem

Some defaults still appear in constructors in `src/train_diffusion.py`, `src/train_masked_room.py`, and `src/core/latent_diffusion.py`, but those classes are already built from resolved config in the canonical path. They matter mainly for:

- direct ad hoc imports,
- partial manual instantiation,
- test/demo code.

So they should be documented, but they are less serious than the pipeline inference defaults above.

## 8. Direct Answers

### "How do we protect our model from comparison with the one-level paper?"

By framing the problem correctly and benchmarking in layers. The one-level paper is a fair comparison for the room generator, not for the whole dungeon stack.

### "Why do we need this big architecture?"

Because the project is solving progression, room synthesis, controllability, validity guidance, symbolic repair, and diversity search at once. A smaller monolithic model would be simpler, but it would solve a different and weaker problem.

### "Why topology graph first and room by room?"

Because long-horizon progression is symbolic and sparse, while room realization is local and spatial. The decomposition matches the problem structure and is much more data-efficient.

### "How can so little data feed the model?"

Only because the repo is heavily constrained:

- fixed room schema,
- room-level decomposition,
- replacement sampling,
- latent compression,
- graph/topology conditioning,
- neighboring room exemplars,
- symbolic repair.

Without those constraints, the current data regime would not support the full stack well.

## 9. Recommended Next Cleanup

If we want the next high-value cleanup pass, the best remaining move is:

1. promote inference-time `guidance_scale` and `logic_guidance_scale` into the validated config/runtime path,
2. remove or clearly isolate random-init fallback defaults from canonical experiment runs,
3. benchmark room-only vs topology-only vs full-stack separately in all reports.

## 10. Done vs Open

### Done

- The comparison boundary is now explicitly documented: room-only papers are room-branch baselines, not whole-stack baselines.
- The attention inventory and non-standard conditioning twists are documented in one place.
- The small-data explanation is now tied to actual repo mechanisms instead of hand-waving.
- Remaining hard-coded values are separated into `canonical training`, `runtime inference`, and `fallback compatibility` buckets.

### Still open

- External matched-budget room-baseline runs.
- Full-stack benchmark tables that report room, topology, and full-dungeon results separately.
- Optional promotion of runtime inference defaults into the resolved config path if we want fully config-driven generation, not only config-driven training.
