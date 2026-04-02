# Architecture Research Audit

Last updated: 2026-03-31

Scope:

- `src/core/vqvae.py`
- `src/core/condition_encoder.py`
- `src/core/latent_diffusion.py`
- `src/core/logic_net.py`
- `src/core/discrete_masked_model.py`
- `src/pipeline/dungeon_pipeline.py`
- `src/train_vqvae.py`
- `src/train_diffusion.py`
- `src/train_masked_room.py`
- `main.py`

This pass is code-first and literature-backed. Claims are tagged as:

- `Code evidence`: verified directly in this repository.
- `Literature-backed`: supported by cited publications.
- `Inference-based`: reasoned from code plus literature where no direct paper states the exact repo-specific claim.

## Executive Summary

The architecture is a serious neural-symbolic Zelda generator, but it is not yet benchmark-proven as state of the art. Its strongest ideas are the mission-graph-first decomposition, room-level latent diffusion, explicit topological conditioning, differentiable logical guidance, and symbolic repair. Its weakest points are small-data exposure, a very strong `zelda_v1` schema lock, and incomplete benchmark evidence against modern diffusion-PCG baselines.

The most important implementation gap found in this pass was a theory-vs-implementation mismatch in Block II:

1. `Critical - Fixed in code`: canonical VQ-VAE training was still defaulting to dungeon-level samples even though the documented architecture, the diffusion stage, and the inference pipeline all consume room latents.
2. `Medium - Fixed in code`: the VQ-VAE stage still omitted several validated dataset/runtime controls from the canonical stage mapping.
3. `Low - Fixed in code`: `main.py` duplicated Block II argument assembly instead of reusing the shared validated config translation.
4. `Low - Fixed in code`: the fast-sampler stage still hardcoded `use_vglc=True` instead of honoring the resolved dataset config.

## Step 1 - Deep Research and Literature Review

### Most relevant publications

| Topic | Publication | Venue | Why it matters here | Key result used in this audit |
|---|---|---|---|---|
| Diffusion foundations | Ho et al., *Denoising Diffusion Probabilistic Models* [1] | NeurIPS 2020 | Block IV noise-prediction foundation | CIFAR-10 IS `9.46`, FID `3.17` |
| Latent diffusion | Rombach et al., *High-Resolution Image Synthesis With Latent Diffusion Models* [2] | CVPR 2022 | Justifies latent-space denoising over pixel-space denoising | At least `2.7x` speedup and `1.6x` better FID than pixel diffusion in their inpainting efficiency comparison |
| Guidance | Nichol et al., *GLIDE* [3] | ICML 2022 | Supports classifier-free guidance and inpainting paths already present in Block IV | Human evaluators preferred classifier-free guidance over CLIP guidance |
| Diffusion reweighting | Hang et al., *Efficient Diffusion Training via Min-SNR Weighting Strategy* [4] | ICCV 2023 | Supports `min_snr_gamma` in this repo | ImageNet-256 FID `2.06`; reported `3.4x` faster convergence than prior training |
| Structured discrete diffusion | Austin et al., *Structured Denoising Diffusion Models in Discrete State-Spaces* [5] | NeurIPS 2021 | Strongest theory match for the masked-room stage | Transition-matrix design materially affects quality; auxiliary cross-entropy helps |
| Layout-conditioned diffusion | Inoue et al., *LayoutDM* [6] | CVPR 2023 | Strong evidence that structured conditioning quality matters | Beats strong layout baselines on multiple conditional layout tasks |
| Small-data PCG diffusion | Dai et al., *Procedural Level Generation with Diffusion Models from a Single Example* [7] | AAAI 2024 | Most relevant modern PCG diffusion baseline for scarce-data regimes | Explicitly uses constrained receptive fields and compact representation to survive extreme data scarcity |
| Mission-graph-first generation | Dormans and Bakkes, *Generating Missions and Spaces for Adaptable Play Experiences* [8] | IEEE TCIAIG 2011 | Direct support for Block I | Recommends mission-first then space-generation decomposition for adventure games |
| Zelda-specific graph+room hybrid | Gutierrez and Schrum, *GAN Rooms in Graph Grammar Dungeons for The Legend of Zelda* [9] | IEEE CEC 2020 | Closest Zelda-domain baseline | 30-subject user study found Graph+GAN dungeons roughly comparable to original dungeons on most survey metrics |
| RL PCG baseline | Khalifa et al., *PCGRL* [10] | IEEE Transactions on Games 2020 | Strong non-diffusion controllable PCG baseline | Demonstrated strong controllability across multiple game domains |
| Graph attention expressivity | Brody et al., *How Attentive are Graph Attention Networks?* [11] | ICLR 2022 | Supports dynamic attention over static GAT | GATv2 outperformed GAT on 12 OGB and related benchmarks at matched parametric cost |
| Structural graph bias | Ying et al., *Do Transformers Really Perform Bad for Graph Representation?* [12] | NeurIPS 2021 | Supports shortest-path and structural bias in graph conditioning | Graphormer reported strong results on large OGB graph benchmarks |
| Spatial modulation | Park et al., *Semantic Image Synthesis With Spatially-Adaptive Normalization* [13] | CVPR 2019 | Best literature match for `topology_conditioning_mode=spade` | Superior visual fidelity and alignment to input layouts |
| Coordinate injection | Liu et al., *CoordConv* [14] | NeurIPS 2018 | Supports `use_coordconv` in Block II | Solved coordinate-transform failures much faster and with fewer parameters; also improved MNIST detection IoU |
| Generative-model bias risk | Chen et al., *Would Deep Generative Models Amplify Bias in Future Models?* [15] | CVPR 2024 | Supports architectural bias-risk discussion | Bias does not increase uniformly, but generated data can change downstream bias behavior in complex ways |

### Literature-backed takeaways

1. `Literature-backed`: latent diffusion is the correct efficiency bias for Block IV, especially when room structure is low-resolution and highly discrete [2].
2. `Literature-backed`: CFG and Min-SNR are not optional extras anymore; they are standard stability and controllability tools for diffusion training/inference [3][4].
3. `Literature-backed`: structured generation quality depends heavily on explicit conditioning pathways, not only on the denoiser backbone [6][13].
4. `Literature-backed`: dynamic graph attention and structural bias matter when the task depends on ranking graph neighbors or shortest-path-aware structure [11][12].
5. `Literature-backed`: low-data diffusion only works well when the architecture is deliberately biased toward locality and compact structure [7].
6. `Literature-backed + Inference-based`: the repo's `zelda_v1` schema lock is not just a dataset choice; it is a major part of why the current stack is trainable at all on this corpus [7][8][9].

## Step 2 - Assumptions Validation

### Architecture-level assumptions

| Assumption | Evidence | Judgment |
|---|---|---|
| Inputs are Zelda rooms with canonical shape `(16, 11)` | `src/core/definitions.py`, loader/schema validators | `Fragile but intentional` |
| Tile vocabulary is fixed at `44` classes | config schema, VQ-VAE, diffusion decode, symbolic repair | `Fragile but intentional` |
| Mission graphs are small enough for per-room graph conditioning and guidance | graph batching, guidance caps, shortest-path features | `Scale-sensitive` |
| Strong graph priors and symbolic repair can offset small-data limitations | overall design | `Plausible, but high-risk without matched-budget evidence` |
| Relative target-room graph distance is useful conditioning | current-node distance path in Block III/IV | `Supported by graph-structure literature` [11][12] |
| Spatial topology maps improve alignment | SPADE/additive topology conditioning | `Reasonable and literature-backed` [13] |
| Differentiable logical guidance remains numerically stable | gradient caps and active-fraction scheduling | `Likely valid at Zelda scale, not guaranteed outside it` |
| Block II learns room latents | documented architecture and downstream usage | `Valid by design, but implementation drift existed before this pass` |

### Hardcoded assumptions that should remain explicit in config/schema

| Assumption | Status |
|---|---|
| `dataset.schema_profile=zelda_v1` | Keep explicit; this is a hard repository contract, not a hidden default |
| `dataset.num_classes=44` | Keep explicit |
| `dataset.room_height=16`, `dataset.room_width=11` | Keep explicit |
| `dataset.node_feature_dim=14`, `dataset.edge_feature_dim=16`, `dataset.tpe_dim=8` | Keep explicit |
| `diffusion.room_topology_channels=50` | Keep explicit |
| `diffusion.guidance_max_*` caps | Keep explicit |
| `symbolic_max_repair_attempts`, `symbolic_repair_margin`, `symbolic_adjacency_threshold` | Already promoted; should stay configurable |

## Step 3 - Logical Audit of the Architecture

### Key findings

1. `Critical - Code evidence - Fixed`
   Block II training had drifted away from the documented room-level latent contract. The architecture claims a room VQ-VAE, the diffusion stage consumes room latents, and the pipeline decodes per-room outputs, but `src/train_vqvae.py` still built a dungeon-level dataset by default. This pass fixes that by propagating `dataset.room_level`, `dataset.normalize`, `dataset.use_vglc`, and `runtime.quick` through the canonical Block II training path.

2. `High - Code evidence`
   The repo is still highly domain-locked. The config system correctly documents this, but any claim beyond Zelda-style dungeons would currently be overstated. This is acceptable for a thesis artifact; it is not acceptable as a generic dungeon generator claim.

3. `Medium - Literature-backed`
   The masked-room branch is almost as large as the latent diffusion branch (`65.8M` vs `66.2M` parameters in the current default profile) while operating on a far smaller discrete grid. On this corpus, that is expensive relative to the amount of evidence showing it adds unique value.

4. `Medium - Literature-backed`
   The architecture is sensible as a hybrid stack, but there is still no matched-budget evidence that the full Block I-VII system beats simpler baselines such as Graph+GAN, PCGRL, or the newer single-example diffusion-PCG setup [7][9][10].

## Step 4 - Theory vs Implementation Consistency Check

### Now aligned

1. Block II canonical training now follows the room-level contract used by Block IV and the pipeline.
2. Canonical stage-1 argument construction now reuses the same validated config translation as the other training stages.
3. Block II now honors `runtime.quick` in the same way as the later training stages.

### Remaining theory/implementation caveats

1. `Inference-based`
   The masked-room branch is present and trainable, but the repo still treats latent diffusion as the primary generation path. The masked-room model is therefore better described as an auxiliary/ablation branch than as a co-equal production generator.
2. `Code evidence`
   Several repository documents discuss SOTA competitiveness more strongly than the available matched-budget experiments justify. That is a benchmarking/documentation issue, not a core implementation bug.

## Step 5 - Gap and Bug Analysis

| Gap | Why it matters | Source | Action |
|---|---|---|---|
| No matched-budget benchmark proving full-stack superiority | Prevents any robust SOTA claim | [7][9][10] | `Benchmark harnesses implemented; external/full-stack evidence still requires experiment runs` |
| No formal fairness/bias evaluation on generated Zelda outputs | Ethical deployment risk | [15] | `Lightweight fairness/bias audit harness implemented; full study still requires experiment runs` |
| Masked-room branch may be over-capacity for dataset size | Efficiency and overfitting risk | [7] + local counts | `Fixed in defaults/config; dedicated small-profile ablation path added` |
| Schema lock is broad and global | Limits transfer and reproducibility across datasets | code + [8] | `Intentional, but must stay explicit` |
| Legacy no-config entrypoints still exist | Maintenance overhead | code | `Low risk; wrappers are acceptable for now` |

### New config recommendations surfaced in this pass

This pass required both propagation fixes and new schema exposure for previously hidden methodology knobs. The following fields are now explicit in the validated YAML/CLI schema:

- `dataset.room_level: bool`
- `dataset.normalize: bool`
- `dataset.use_vglc: bool`
- `runtime.quick: bool`
- `topology.default_target_curve: list[float]`
- `topology.num_rooms: int`
- `topology.population_size: int`
- `topology.generations: int`
- `topology.mutation_rate: float`
- `topology.crossover_rate: float`
- `topology.genome_length: int`
- `topology.rule_space: enum`
- `topology.transition_mix: float`
- `topology.search_strategy: enum`
- `topology.qd_archive_cells: int`
- `topology.qd_init_random_fraction: float`
- `topology.qd_emitter_mutation_rate: float`
- `topology.max_lock_key_rules: int`
- `topology.enable_rule_credit_assignment: bool`
- `topology.enforce_generation_constraints: bool`
- `topology.allow_candidate_repairs: bool`
- `masked_room.attention_mode: enum`
- `masked_room.topology_conditioning_mode: enum`
- `masked_room.hedgehog_feature_dim: int`
- `masked_room.graph_auto_linear_attention_nodes: int`
- `masked_room.spatial_graph_gate_init: float`
- `masked_room.spatial_topology_gate_init: float`
- `masked_room.unet_channel_mult: list[int]`
- `masked_room.unet_num_res_blocks: int`
- `masked_room.unet_attention_resolutions: list[int]`
- `masked_room.unet_num_heads: int`
- `masked_room.unet_dropout: float`
- `masked_room.min_mask_ratio: float`
- `masked_room.max_mask_ratio: float`

## Step 6 - Redundancy and Unnecessary Work Analysis

1. `Code evidence - Fixed`
   `main.py` duplicated Block II stage argument assembly. This was redundant and risked future drift; the stage now reuses `vqvae_training_kwargs_from_resolved_config(...)`.

2. `Inference-based`
   The masked-room branch is expensive enough that it should justify itself empirically. Until an ablation shows meaningful gains in controllability, robustness, or fidelity, it remains the most likely over-engineered component.

3. `Inference-based`
   MAP-Elites is valuable for evaluation and archive-building, but it is not free. If the immediate research question is room-generation fidelity rather than quality-diversity exploration, running MAP-Elites on every experiment is unnecessary overhead.

## Step 7 - Computational Complexity Analysis

### Big-O summary

| Block | Main cost |
|---|---|
| Block II VQ-VAE | `O(HW * C^2 * k^2)` convolutional cost |
| Block III local encoder | `O(HW * d)` after latent pooling; small |
| Block III graph encoder | `O(L_g * E * d)` for message passing; `O(N^2 * d)` if transformer-style global mixing is used |
| Block IV denoiser per step | `O(sum_l H_l W_l C_l^2 k^2 + HWN d + N^2 d)` |
| Block V LogicNet | roughly `O(I * (HW + N^2))`, with `I=num_logic_iterations` |
| Block VI WFC repair | worst-case exponential; bounded in practice by local masks and repair budgets |
| Block VII MAP-Elites | `O(num_evals * archive_update_cost + solver/evaluator cost)` |

### Measured local parameter counts

Measured on the current default YAML profile:

| Module | Parameters |
|---|---|
| VQ-VAE | `31.10M` |
| Condition encoder | `3.13M` |
| Latent diffusion | `66.21M` |
| LogicNet | `0.43M` |
| Masked-room model | `65.78M` |

### Approximate forward FLOPs

Measured locally on batch size `1` with canonical room size `(16,11)` and an 8-node graph. These are implementation-specific approximations, not paper-reported numbers:

| Module | Approx forward FLOPs |
|---|---|
| VQ-VAE forward | `1.52G` |
| Condition encoder forward | `24.6M` |
| Diffusion denoiser forward per step | `385M` |
| LogicNet forward | `75.0M` |
| Masked-room forward | `2.05G` |

Interpretation:

1. `Inference-based`: the denoiser is cheap per step because the latent grid is only `4x3`, but total diffusion cost still scales linearly with the number of reverse steps.
2. `Inference-based`: a 50-step DDIM-style sample is roughly `19.3G` denoiser FLOPs before logic-guidance backward cost.
3. `Inference-based`: a 1000-step DDPM sample is roughly `385G` denoiser FLOPs before guidance/backward cost and is therefore not practical for frequent interactive use.

### Complexity-sensitive parameters that must remain configurable

| Parameter | Default | Safe range | Notes |
|---|---:|---:|---|
| `diffusion.model_channels` | `96` | `64..160` | Strong effect on params and memory |
| `diffusion.unet_channel_mult` | `[1,2,4]` | small positive tuples | Expands width by level |
| `diffusion.unet_num_res_blocks` | `2` | `1..3` | Linear-ish compute multiplier |
| `diffusion.unet_num_heads` | `8` | divisors of active channel widths | Must divide every attention width |
| `diffusion.condition_hidden_dim` | `192` | `128..320` | Affects Block III cost and overfitting risk |
| `diffusion.condition_num_gnn_layers` | `2` | `1..4` | Deeper is costlier and can over-smooth |
| `diffusion.num_timesteps` | `1000` | `100..1000` for training | Sampling cost scales linearly |
| `masked_room.model_channels` | `96` | `64..128` | Large sensitivity for an already heavy branch |
| `masked_room.hidden_dim` | `64` | `32..96` | Token backbone width |
| `distributed.nproc_per_node` | `1` | hardware-bound | Only supported for diffusion training |

## Step 8 - Hyperparameter Sensitivity Analysis

### Highest-sensitivity hyperparameters

| Parameter | Type | Default | Safe range | Why sensitive | Dependency |
|---|---|---:|---|---|---|
| `diffusion.cfg_scale` | float | `3.0` | `1.0..5.0` | High values can oversharpen and destabilize semantics [3] | interacts with `cfg_schedule_*` |
| `diffusion.min_snr_gamma` | float | `5.0` | `1.0..7.0` | Training weighting changes convergence behavior [4] | depends on prediction type and schedule |
| `diffusion.alpha_logic` | float | `0.1` | `0.0..0.3` | Too high can overpower visual fidelity | depends on LogicNet calibration |
| `diffusion.logic_topology_trace_weight` | float | `0.25` | `0.0..0.5` | Too high can force topology priors over tile realism | pairs with `logic_topology_anchor_weight` |
| `diffusion.logic_topology_anchor_weight` | float | `0.25` | `0.0..0.5` | Same as above | pairs with `logic_topology_trace_weight` |
| `diffusion.condition_gnn_type` | enum | `gps` | `{gcn,gat,sage,gps}` | Changes expressivity and compute | graph scale matters |
| `diffusion.graph_conditioning_mode` | enum | `node_sequence` | `{node_sequence,pooled}` | Strong effect on controllability | node-sequence benefits more from distance features |
| `diffusion.use_current_node_distance_features` | bool | `true` | `{true,false}` | Can materially affect target-room specificity | most useful with node-sequence |
| `diffusion.topology_conditioning_mode` | enum | `spade` | `{additive,spade}` | Directly changes spatial prior injection [13] | depends on quality of topology maps |
| `diffusion.guidance_active_fraction` | float | `0.30` | `0.1..0.5` | Higher fractions cost more and can overshape later steps | interacts with `guidance_scale` |
| `vqvae.hidden_dim` | int | `128` | `96..192` | Strong cost/reconstruction tradeoff | affects codebook utilization |
| `vqvae.codebook_size` | int | `256` | `128..1024` | Too small collapses diversity; too large wastes capacity | canonical default reduced for the current tiny Zelda corpus; interacts with dataset size |
| `masked_room.model_channels` | int | `96` | `64..128` | Largest cost lever in auxiliary branch | interacts with `hidden_dim` |

### Dangerous combinations

1. `cfg_scale > 5` with `guidance_scale > 1`: can overconstrain sampling and degrade diversity.
2. `alpha_logic > 0.3` with high topology weights: likely to damage local tile realism.
3. `condition_num_gnn_layers >= 4` on this small graph regime: increased compute with limited evidence of benefit.
4. `masked_room.model_channels >= 128` with current dataset size: high overfitting risk.

## Step 9 - Failure Mode and Edge Case Analysis

1. OOD mission graphs with many nodes or rich cross-links can make graph conditioning and guidance brittle.
2. Missing or noisy room-topology maps can mislead SPADE/additive modulation.
3. Training-data scarcity can yield memorization or deceptive benchmark wins on the tiny Zelda corpus [7][9].
4. Generated layouts may satisfy local room constraints while still failing global pacing or novelty expectations.
5. Symbolic repair can preserve validity while reducing diversity, especially when invoked frequently.

## Step 10 - Scalability and Generalization Boundary Analysis

### Practical boundary summary

| Dimension | Current boundary |
|---|---|
| Data scale | Good fit for tiny Zelda corpus, but evidence outside this regime is absent |
| Graph scale | Safe for small mission graphs; large graphs will push `N^2` graph-conditioning costs |
| Model scale | Current diffusion profile is viable; larger widths are difficult to justify on this dataset |
| Task transfer | Limited by `zelda_v1` schema lock |
| Interactive inference | Reasonable with fast sampler or low-step DDIM; poor with high-step DDPM plus guidance |

### Scale-relevant config fields

- `distributed.nproc_per_node`
- `diffusion.model_channels`
- `diffusion.condition_hidden_dim`
- `diffusion.condition_num_gnn_layers`
- `diffusion.num_timesteps`
- `vqvae.hidden_dim`
- `masked_room.model_channels`
- `vqvae.epochs`
- `diffusion.epochs`

## Step 11 - Comparison Against State-of-the-Art Baselines

### Judgment

The architecture is `meaningfully novel as a hybrid engineering design`, but `not benchmark-validated as SOTA`.

Why:

1. Compared with Graph+GAN Zelda baselines [9], this repo is architecturally richer and should have stronger controllability, but no controlled head-to-head evidence is included.
2. Compared with single-example diffusion-PCG [7], this repo has stronger symbolic structure and mission-graph control, but also far more moving parts and no matched-budget comparison.
3. Compared with PCGRL [10], this repo likely has better tile-level generative diversity but weaker evidence on controllability and robustness.
4. Compared with structured diffusion/layout work [2][5][6], the repo uses many of the right ingredients, but its benchmark suite is domain-specific and not externally comparable enough to support SOTA claims.

## Step 12 - Bias and Ethical Risk Analysis

1. `Literature-backed`
   Generative models can alter downstream bias behavior in non-obvious ways [15]. Even if bias is not monotonically amplified, generated Zelda data could distort room semantics, item placement frequency, or difficulty pacing.
2. `Inference-based`
   Mission-graph-first generation may encode a normative view of progression and lock/key pacing that excludes alternative play styles.
3. `Inference-based`
   Symbolic repair can preferentially collapse outputs toward hand-coded priors, which may reduce stylistic diversity and hide model failures.

## Step 13 - Evidence-Based Decision Table

### Consolidated config recommendations

| Parameter | Type | Default | Valid range | Source | Notes |
|---|---|---:|---|---|---|
| `dataset.room_level` | bool | `true` | `{true,false}` | Code evidence | Block II must honor this; fixed in this pass |
| `dataset.normalize` | bool | `true` | `{true,false}` | Code evidence | Block II must honor this; fixed in this pass |
| `dataset.use_vglc` | bool | `true` | `{true,false}` | Code evidence | Block II now propagates it |
| `runtime.quick` | bool | `false` | `{true,false}` | Code evidence | Block II now honors it |
| `diffusion.cfg_scale` | float | `3.0` | `1.0..5.0` | [3] + inference | Main controllability knob |
| `diffusion.min_snr_gamma` | float | `5.0` | `1.0..7.0` | [4] | Training stability/speed |
| `diffusion.condition_gnn_type` | enum | `gps` | `{gcn,gat,sage,gps}` | [11][12] + inference | Expressivity/compute tradeoff |
| `diffusion.graph_conditioning_mode` | enum | `node_sequence` | `{node_sequence,pooled}` | [6][12] + inference | Major controllability toggle |
| `diffusion.use_current_node_distance_features` | bool | `true` | `{true,false}` | [11][12] + inference | Target-room specificity |
| `diffusion.topology_conditioning_mode` | enum | `spade` | `{additive,spade}` | [13] | Spatial prior injection |
| `diffusion.alpha_logic` | float | `0.1` | `0.0..0.3` | inference | Balance with visual fidelity |
| `diffusion.logic_topology_trace_weight` | float | `0.25` | `0.0..0.5` | inference | Only meaningful if topology maps are good |
| `diffusion.logic_topology_anchor_weight` | float | `0.25` | `0.0..0.5` | inference | Same dependency as above |
| `vqvae.hidden_dim` | int | `128` | `96..192` | [7][14] + inference | Capacity vs small-data risk |
| `vqvae.codebook_size` | int | `256` | `128..1024` | inference | Canonical default reduced for small-data stability; monitor codebook usage/perplexity |
| `masked_room.model_channels` | int | `96` | `64..128` | [5][7] + inference | Largest masked-room cost knob |
| `masked_room.unet_channel_mult` | list[int] | `[1,2,4]` | short positive tuples | inference | Controls masked-room compute per scale level |
| `masked_room.unet_num_heads` | int | `8` | divisors of active channel widths | inference | Must divide every masked-room attention width |
| `masked_room.min_mask_ratio` | float | `0.15` | `0.10..0.40` | [5] + inference | Too low weakens denoising signal |
| `masked_room.max_mask_ratio` | float | `0.90` | `0.60..0.95` | [5] + inference | Too high can destabilize hard conditional anchors |
| `topology.population_size` | int | `50` | `16..128` | [8][9] + inference | Direct Block I compute knob |
| `topology.generations` | int | `100` | `12..200` | [8][9] + inference | Linear runtime multiplier for Block I |
| `topology.mutation_rate` | float | `0.15` | `0.05..0.40` | [8] + inference | Governs exploration vs. convergence |
| `topology.max_lock_key_rules` | int | `3` | `0..6` | [8][9] + inference | Prevents degenerate over-gating in small Zelda graphs |

## Step 14 - Ablation Study Recommendation

### Recommended ablations

1. Remove current-node distance features.
   What varies: `diffusion.use_current_node_distance_features=false`, `masked_room.use_current_node_distance_features=false`
   Measure: topology alignment, room solvability, validation loss
   Expected outcome: weaker target-room specificity
   Proves/disproves: whether explicit target-relative graph distance matters

```bash
python main.py train --config configs/zelda_hmolqd.yaml --stage diffusion --diffusion-use-current-node-distance-features false
python main.py train --config configs/zelda_hmolqd.yaml --stage masked_room --masked-room-use-current-node-distance-features false
```

2. Compare `node_sequence` against `pooled` graph conditioning.
   What varies: `diffusion.graph_conditioning_mode`
   Measure: controllability and topology-conditioned fidelity
   Expected outcome: `node_sequence` should outperform pooled conditioning

```bash
python main.py train --config configs/zelda_hmolqd.yaml --stage diffusion --diffusion-graph-conditioning-mode pooled
```

3. Remove logical pressure.
   What varies: `diffusion.alpha_logic=0`, `diffusion.guidance_scale=0`, topology logic weights `0`
   Measure: solvability, repair rate, tile realism
   Expected outcome: more raw visual freedom, worse solvability and higher repair dependence

```bash
python main.py train --config configs/zelda_hmolqd.yaml --stage diffusion --diffusion-alpha-logic 0 --diffusion-guidance-scale 0 --diffusion-logic-topology-trace-weight 0 --diffusion-logic-topology-anchor-weight 0
```

4. Compare topology conditioning modes.
   What varies: `diffusion.topology_conditioning_mode=additive` vs `spade`
   Measure: room-plan adherence and reconstruction fidelity
   Expected outcome: `spade` should improve spatial adherence if topology maps are reliable

```bash
python main.py train --config configs/zelda_hmolqd.yaml --stage diffusion --diffusion-topology-conditioning-mode additive
```

5. Downsize the masked-room branch.
   What varies: `masked_room.model_channels=64`, `masked_room.hidden_dim=48`
   Measure: validation loss, generation quality, compute
   Expected outcome: smaller model may retain most quality at much lower cost

```yaml
masked_room:
  model_channels: 64
  hidden_dim: 48
```

## Step 15 - Priority Ranking

### Prioritized findings

| Priority | Finding | Reproducibility risk |
|---|---|---|
| Critical | Block II room-level mismatch between documented architecture and training path | Yes |
| High | No completed matched-budget evidence for SOTA competitiveness | Yes |
| High | Strong `zelda_v1` schema lock limits generalization claims | Yes |
| Medium | Masked-room branch may be over-capacity for the dataset | Yes |
| Medium | Fairness/bias behavior still needs experiment execution beyond the new audit harness | Yes |
| Low | Legacy wrappers still exist, but they now mostly delegate to canonical paths | No |

## Step 16 - Immediate Implementation

### Changes applied in this pass

1. `src/train_vqvae.py`
   Block II now propagates `dataset.room_level`, `dataset.normalize`, `dataset.use_vglc`, and `runtime.quick` instead of silently training on stitched dungeons.

2. `src/train_vqvae.py`
   Block II now uses shared seeding and logs small-data capacity guardrails.

3. `main.py`
   The canonical stage-1 runner now reuses `vqvae_training_kwargs_from_resolved_config(...)` instead of maintaining a second manual mapping.

4. `tests/test_config_system.py`
   Added assertions that Block II canonical config propagation includes `room_level`, `normalize`, and `quick`.

5. `src/train_lcm.py`
   The fast-sampler trainer now honors `dataset.use_vglc` instead of forcing `use_vglc=True` in its dataloaders.

6. `main.py`
   The canonical fast-sampler stage runner now overwrites `base_diffusion_checkpoint` in a local kwargs dict instead of double-passing it to `FastSamplerTrainingConfig(...)`.

7. `src/zelda_data/zelda_loader.py`, `src/train_diffusion.py`, `src/train_masked_room.py`, `src/core/condition_encoder.py`, `src/pipeline/dungeon_pipeline.py`, `src/config_system.py`
   The topology graph contract is now unified across config, loader, training, and runtime with a richer `14/16/8` graph schema. Training now consumes explicit `edge_features` and richer `node_features` instead of relying on the legacy `6/8` path.

8. `src/pipeline/graph_features.py`
   Edge features now preserve one-way directionality and battery/switch-cardinality semantics; node features now expose richer secret/hub-style structure hints.

9. `src/pipeline/block_contracts.py`
   Runtime feature validation now defers exact width alignment to the condition encoder's pad/truncate compatibility logic instead of dropping to zero conditioning on schema-width mismatches.

10. `tests/test_zelda_loader_graph_conditioning.py`, `tests/test_train_diffusion_conditioning_shapes.py`, `tests/test_block_integration.py`
   Added regression coverage for richer graph schemas and explicit topology edge-feature propagation.

11. `src/config_system.py`, `configs/zelda_hmolqd.yaml`
    The validated config system now exposes Block I topology defaults and hidden masked-room U-Net/mask-schedule assumptions as explicit YAML/CLI fields, with validation for attention divisibility and mask-ratio ordering.

12. `src/pipeline/dungeon_pipeline.py`, `src/generation/evolutionary_director.py`
    Block I topology generation now consumes configurable defaults instead of local hardcoded constants, and the generator now threads `max_lock_key_rules` into the grammar executor. A helper `topology_generation_kwargs_from_resolved_config(...)` was added for reproducible generation calls.

13. `src/train_masked_room.py`, `src/core/discrete_masked_model.py`
    The masked-room branch now exposes and checkpoints its previously hidden attention kernel, topology-conditioning mode, hedgehog width, graph/topology gate initializations, U-Net shape, and stochastic masking schedule. Training now forwards `min_mask_ratio` and `max_mask_ratio` explicitly into loss computation.

14. `src/core/condition_encoder.py`
    Corrected the top-level Block III formulation so the documented local stream matches the actual four-neighbor implementation instead of the stale north-west-only description.

15. `tests/test_config_system.py`, `tests/test_discrete_masked_room_model.py`, `tests/test_neural_pipeline.py`
   Added regression coverage for the new topology config helper, masked-room hidden-knob propagation, configurable mask schedule forwarding, and pipeline topology-default consumption.

16. `src/config_system.py`, `configs/zelda_hmolqd.yaml`, `configs/zelda_hmolqd_masked_small.yaml`
   The default validated config profile is now aligned with the recommended small-data operating regime from this audit: reduced `model_channels`, shallower condition encoders, `gps` diffusion graph conditioning, and reference-room conditioning enabled by default. The dedicated downsized masked-room ablation profile is now checked into config form.

17. `src/utils/style_tokens.py`, `src/zelda_data/zelda_loader.py`, `src/pipeline/dungeon_pipeline.py`, `src/core/condition_encoder.py`
   The style-token path now resolves the repo's canonical symbolic sector themes from both numeric IDs and free-form compound labels such as `fire-temple` or `shadow_dungeon`, so the style pathway is no longer limited to explicit numeric metadata only.

18. `src/evaluation/pcg_benchmark_alignment.py`, `scripts/run_pcg_benchmark_alignment.py`, `tests/test_pcg_benchmark_alignment.py`
   The external PCG benchmark bridge now fails closed on missing progression semantics instead of inventing start/key/goal anchors. Invalid graphs are flagged explicitly in the outputs via `semantic_valid` / `semantic_error`, preventing inflated benchmark scores from semantically incomplete inputs.

19. `src/evaluation/fairness_assessment.py`, `scripts/run_fairness_bias_audit.py`, `tests/test_fairness_assessment.py`
   Added a lightweight structural fairness/bias audit harness that reports distribution drift, entropy, active-class coverage, and invalid-tile counts in JSON/Markdown form. This closes the reproducibility gap for routine bias smoke tests, while still not claiming a full human-centered fairness study.

20. `scripts/run_ablation_study.py`, `scripts/run_room_branch_benchmark.py`, `scripts/run_priority_research_suite.py`
   Added a dedicated matched-budget room-branch benchmark harness for latent-vs-masked generation and reference-room conditioning on/off comparisons, and registered it in the consolidated research suite. This closes the benchmark-harness gap for Block III/IV/Masked-room internal comparisons, while external layout-baseline runs still remain an experiment layer.

### Verification

- `python -m pytest tests/test_config_system.py -q`
- `python -m pytest tests/test_hmolqd/test_vqvae.py -q`
- `python -m pytest tests/test_config_system.py tests/test_zelda_loader_graph_conditioning.py tests/test_train_diffusion_conditioning_shapes.py tests/test_block_integration.py tests/test_architecture_audit_fixes.py -q`
- `python -m pytest tests/test_config_system.py -q`
- `python -m pytest tests/test_discrete_masked_room_model.py -q`
- `python -m pytest tests/test_neural_pipeline.py -q -k "prepare_dungeon_generation or topology_defaults"`
- `python -m pytest tests/test_architecture_audit_fixes.py -q -k "masked_room or bound_component_dimensions"`
- `python -m pytest tests/test_architecture_audit_fixes.py tests/test_zelda_loader_graph_conditioning.py tests/test_config_system.py tests/test_pcg_benchmark_alignment.py tests/test_fairness_assessment.py -q`
- `python -m py_compile src/utils/style_tokens.py src/pipeline/dungeon_pipeline.py src/core/condition_encoder.py src/config_system.py src/evaluation/pcg_benchmark_alignment.py scripts/run_pcg_benchmark_alignment.py scripts/run_ablation_study.py scripts/run_room_branch_benchmark.py scripts/run_fairness_bias_audit.py`

## References

1. Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS 2020. https://arxiv.org/abs/2006.11239
2. Rombach et al., *High-Resolution Image Synthesis With Latent Diffusion Models*, CVPR 2022. https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf
3. Nichol et al., *GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models*, ICML 2022. https://proceedings.mlr.press/v162/nichol22a.html
4. Hang et al., *Efficient Diffusion Training via Min-SNR Weighting Strategy*, ICCV 2023. https://openaccess.thecvf.com/content/ICCV2023/papers/Hang_Efficient_Diffusion_Training_via_Min-SNR_Weighting_Strategy_ICCV_2023_paper.pdf
5. Austin et al., *Structured Denoising Diffusion Models in Discrete State-Spaces*, NeurIPS 2021. https://papers.nips.cc/paper/2021/hash/958c530554f78bcd8e97125b70e6973d-Abstract.html
6. Inoue et al., *LayoutDM: Discrete Diffusion Model for Controllable Layout Generation*, CVPR 2023. https://openaccess.thecvf.com/content/CVPR2023/papers/Inoue_LayoutDM_Discrete_Diffusion_Model_for_Controllable_Layout_Generation_CVPR_2023_paper.pdf
7. Dai et al., *Procedural Level Generation with Diffusion Models from a Single Example*, AAAI 2024. https://ojs.aaai.org/index.php/AAAI/article/view/28865
8. Dormans and Bakkes, *Generating Missions and Spaces for Adaptable Play Experiences*, IEEE TCIAIG 2011. https://pure.hva.nl/ws/files/149264/453867_Dormans_Bakkes_-_Generating_Missions_and_Spaces_for_Adaptable_Play_Experiences.pdf
9. Gutierrez and Schrum, *GAN Rooms in Graph Grammar Dungeons for The Legend of Zelda*, IEEE CEC 2020. https://people.southwestern.edu/~schrum2/SCOPE/gutierrez.cec2020.pdf
10. Khalifa et al., *Procedural Content Generation via Reinforcement Learning*, IEEE Transactions on Games 2020. https://arxiv.org/abs/1910.01603
11. Brody et al., *How Attentive are Graph Attention Networks?*, ICLR 2022. https://iclr.cc/virtual/2022/poster/6366
12. Ying et al., *Do Transformers Really Perform Bad for Graph Representation?*, NeurIPS 2021. https://arxiv.org/abs/2106.05234
13. Park et al., *Semantic Image Synthesis With Spatially-Adaptive Normalization*, CVPR 2019. https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Semantic_Image_Synthesis_With_Spatially-Adaptive_Normalization_CVPR_2019_paper.html
14. Liu et al., *An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution*, NeurIPS 2018. https://papers.nips.cc/paper/8169-an-intriguing-failing-of-convolutional-neural-networks-and-the-coordconv-solution
15. Chen et al., *Would Deep Generative Models Amplify Bias in Future Models?*, CVPR 2024. https://openaccess.thecvf.com/content/CVPR2024/html/Chen_Would_Deep_Generative_Models_Amplify_Bias_in_Future_Models_CVPR_2024_paper.html
