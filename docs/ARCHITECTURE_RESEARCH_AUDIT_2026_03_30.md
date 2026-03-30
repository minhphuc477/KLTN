# Architecture Research Audit

Last updated: 2026-03-30

This document is a code-first and literature-backed audit of the current Zelda dungeon generation stack implemented in:

- `src/pipeline/dungeon_pipeline.py`
- `src/core/vqvae.py`
- `src/core/condition_encoder.py`
- `src/core/latent_diffusion.py`
- `src/core/logic_net.py`
- `src/core/symbolic_refiner.py`
- `src/simulation/map_elites.py`
- `src/train_diffusion.py`
- `src/train_masked_room.py`
- `main.py`

It is organized to match the requested 16-step workflow. Claims are tagged as either:

- `Code evidence`: directly verified in the repository.
- `Literature-backed`: supported by cited publications.
- `Inference-based`: reasoned from code + literature where no direct paper states the exact claim.

## Executive Summary

The current architecture is a credible neural-symbolic hybrid for Zelda-style dungeon generation:

1. Block I builds a mission graph first.
2. Block II compresses rooms with a semantic VQ-VAE.
3. Block III encodes local neighbor context plus global graph context.
4. Block IV denoises in latent space with graph-aware and room-topology-aware conditioning.
5. Block V injects differentiable logical solvability pressure.
6. Block VI repairs local failures symbolically.
7. Block VII evaluates/archive-scores outputs with MAP-Elites.

The strongest parts are architectural modularity, graph-conditioning depth, explicit repair, and a validated YAML configuration system. The weakest parts are small-data dependence, strong Zelda-specific schema assumptions, and the ongoing need to keep docs, objectives, and checkpoint-loading behavior aligned with a fast-moving implementation.

Immediate code fixes implemented in this pass:

1. `src/train_diffusion.py` standalone training now accepts `--config` and can inherit the full validated YAML schema instead of silently exposing only a thin subset of diffusion hyperparameters.
2. `main.py` now reuses shared diffusion-stage config construction instead of duplicating the mapping logic.
3. `src/train_masked_room.py` no longer hardcodes `num_classes=44` and `latent_dim=64`; those values are now passed through config-derived arguments.
4. `main.py` now reuses shared masked-room-stage config construction.

Immediate architecture improvement implemented in the current follow-up pass:

5. Block III and Block IV now receive explicit current-room distance features derived from the mission graph, instead of relying on `current_node_idx` only as a late token-selection hint.
6. The new feature is exposed in YAML/CLI as `use_current_node_distance_features` and `current_node_distance_max` for both diffusion and masked-room training.

Immediate reproducibility and small-data fixes implemented in the current one-pass closeout:

7. `src/train.py` is now only a compatibility wrapper over `main.py train`, removing the last divergent legacy training surface.
8. The Zelda data contract is now explicit via `dataset.schema_profile=zelda_v1`, and reproducibility snapshots record the schema lock in metadata.
9. `configs/zelda_hmolqd.yaml` now encodes the audit-recommended reduced-capacity profile for the current small dataset.
10. `src/train_diffusion.py` and `src/train_masked_room.py` now log parameter-count guardrails and warn when model capacity is oversized relative to dataset size.
11. `src/core/logic_net.py` now directly optimizes room-topology trace and anchor targets, instead of only describing topology-aware logical pressure while training mostly on room-grid reachability.
12. `src/train_diffusion.py` now passes the full batched diffusion graph context, including `room_topology_map` and `boundary_constraints`, into LogicNet training and validation.
13. `src/core/latent_diffusion.py` now allows guidance to run in room-topology-only mode, even when mission-level adjacency is absent.
14. `src/pipeline/dungeon_pipeline.py` now reconstructs condition encoder, diffusion, and LogicNet modules from composite diffusion checkpoints using embedded config values instead of silently hardcoded defaults.

## Step 1 - Deep Research and Literature Review

### Most Relevant Publications

| Topic | Publication | Venue | Why it matters here | Key benchmark or result |
|---|---|---|---|---|
| Diffusion foundations | Ho et al., *Denoising Diffusion Probabilistic Models* | NeurIPS 2020 | Baseline reverse-diffusion training objective used by Block IV | CIFAR-10 FID `3.17` reported on the paper page |
| Latent diffusion | Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models* | CVPR 2022 | Justifies doing diffusion in compressed latent space rather than pixel space | LDM-4 reports CelebA-HQ FID `5.11` and LSUN Bedrooms FID `2.95` |
| Classifier-free guidance | Nichol et al., *GLIDE* | ICML 2022 | Supports CFG-style conditional/unconditional interpolation already used in Block IV | Human evaluators preferred CFG to CLIP guidance in GLIDE |
| Min-SNR reweighting | Hang et al., *Efficient Diffusion Training via Min-SNR Weighting Strategy* | ICCV 2023 | Supports the repo's `min_snr_gamma` loss reweighting | Literature-backed training efficiency/stability improvement |
| Layout diffusion | Chai et al., *LayoutDM* | CVPR 2023 | Strongest nearby evidence that conditional diffusion improves structured layout generation over GAN/VAE baselines | Paper reports outperforming prior SOTA layout generators on quality and diversity |
| Small-data diffusion for levels | Dai et al., *Procedural Level Generation with Diffusion Models from a Single Example* | AAAI 2024 | Shows diffusion can work in low-data PCG only with strong inductive bias and constrained receptive fields | Generates arbitrary-size levels with fewer artifacts than GAN baselines |
| Mission graphs | Dormans and Bakkes, *Generating Missions and Spaces for Adaptable Play Experiences* | IEEE TCIAIG 2011 | Canonical support for mission-graph-first dungeon generation | Establishes mission/space duality used by Block I |
| Zelda graph+room hybrid baseline | Gutierrez and Schrum, *Generative Adversarial Network Rooms in Generative Graph Grammar Dungeons for The Legend of Zelda* | IEEE CEC 2020 | Closest prior Zelda-specific hybrid baseline | User study with `70` players reported on the project page |
| RL-based PCG baseline | Khalifa et al., *PCGRL: Procedural Content Generation via Reinforcement Learning* | IEEE Transactions on Games 2020 | Strong alternative baseline under low-example conditions | Reported stronger controllability across multiple domains than prior handcrafted methods |
| Graph backbone scaling | Rampasek et al., *Recipe for a General, Powerful, Scalable Graph Transformer (GraphGPS)* | 2022 preprint / NeurIPS-era widely adopted recipe | Most relevant justification for the repo's `condition_gnn_type=gps` option | Strong graph-transformer recipe for accuracy/scalability tradeoff |
| Relative structural bias | Ying et al., *Do Transformers Really Perform Bad for Graph Representation?* (Graphormer) | NeurIPS 2021 | Most relevant support for shortest-path-aware attention bias in graph transformers | Shows structural encodings and attention bias materially improve graph reasoning |
| Attention expressivity | Brody et al., *How Attentive are Graph Attention Networks?* | ICLR 2022 | Supports using GATv2-style dynamic attention instead of static GAT when topology ranking matters | GATv2 outperformed GAT on `11` OGB and related benchmarks at similar cost |
| Spatial modulation | Park et al., *Semantic Image Synthesis with Spatially-Adaptive Normalization* | CVPR 2019 | Best literature match for the repo's `topology_conditioning_mode=spade` | Strong conditional fidelity improvements for spatial control |
| Coordinate injection | Liu et al., *An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution* | NeurIPS 2018 | Supports coordinate-aware encoders in Zelda's rigid room geometry | Clear gains on coordinate transform tasks with little downside |
| Ethical risk | Chen et al., *Would Deep Generative Models Amplify Bias in Future Models?* | CVPR 2024 | Relevant to architectural bias amplification in generative models | Shows generative-model self-training can amplify bias over iterations |

### Literature-Based Takeaways

1. `Literature-backed`: latent-space diffusion is the correct efficiency choice for this project's room generation scale.
2. `Literature-backed`: CFG and Min-SNR are not optional "nice to have" tricks anymore; they are now standard diffusion-stability controls.
3. `Literature-backed`: for structured generation, explicit conditioning quality matters at least as much as the denoiser architecture.
4. `Literature-backed`: graph attention choice matters; GAT-style static attention can be a real expressive bottleneck.
5. `Literature-backed`: low-data PCG requires stronger inductive bias than general-purpose image diffusion.
6. `Literature-backed + Inference-based`: graph-conditioned generation should encode target-node-relative structure explicitly. Graphormer's shortest-path attention bias and distance-encoding work both support making "distance to the room being generated" a first-class signal instead of an implicit side effect of the graph encoder.

## Step 2 - Assumptions Validation

### Architecture-Level Assumptions

| Assumption | Evidence in code | Real-world validity | Judgment |
|---|---|---|---|
| Rooms always use Zelda's canonical `16x11` grid | `src/core/definitions.py`, loader/validator/pipeline constants | Valid only for this dataset family | `Fragile but intentional` |
| Tile vocabulary is fixed at `44` semantic classes | dataset schema + VQ-VAE + masked-room path | Valid only for the current semantic palette | `Fragile but intentional` |
| Graph schema is fixed at `6` node features, `8` edge features, `8` TPE channels | `src/config_system.py`, graph feature builders | Not generally valid outside this codebase | `Fragile` |
| Training data is room-level and normalized to `[0,1]` tile IDs | `create_dataloader(...)`, `encode_to_latent(...)`, `_to_token_ids(...)` | Valid for the current loader only | `Undocumented coupling` |
| Mission graph size stays modest | guidance caps and graph attention thresholding | True for VGLC Zelda, not true for general dungeon corpora | `Scale-sensitive` |
| Torch Geometric may be absent | fallback GNN paths exist | Good defensive assumption | `Robust` |
| Diffusion can be trained on a very small corpus if symbolic repair and conditioning are strong enough | whole stack design | Only partially valid; literature suggests extra inductive bias is required in low-data PCG | `High-risk` |
| Node-sequence graph conditioning is superior to pooled conditioning | implemented default | Reasonable for structured tasks, but requires ablation to justify in this repo | `Plausible, not proven locally` |
| Room-topology maps are useful conditioning priors | `room_topology_map` path in loader and diffusion | Consistent with spatially conditioned generation literature | `Reasonable` |
| LogicNet guidance remains numerically stable under bounded graph sizes | gradient caps and thresholds | Likely valid under current Zelda scales, not guaranteed beyond them | `Scale-sensitive` |

### Hardcoded Assumptions Identified in Implementation

Candidates for configuration promotion or explicit derivation:

1. `Code evidence`: masked-room training had a hardcoded `latent_dim=64`.
   Status: fixed in this pass by wiring config-derived `latent_dim`.
2. `Code evidence`: masked-room training had a hardcoded `num_classes=44`.
   Status: fixed in this pass by wiring config-derived `num_classes`.
3. `Code evidence`: the standalone diffusion trainer previously exposed only a subset of its real methodology knobs.
   Status: fixed in this pass by adding `--config` support to `src/train_diffusion.py`.
4. `Code evidence`: `src/train.py` historically duplicated a thinner, legacy training surface than `main.py train`.
   Status: fixed in this pass by turning it into a compatibility wrapper over the canonical entrypoint.
5. `Code evidence`: the codebase still hard-fails non-`16x11`, non-`44`-class data in the shared schema validator.
   Status: still intentional, but now exposed explicitly via `dataset.schema_profile=zelda_v1` and recorded in metadata.
6. `Code evidence`: pipeline loaders previously instantiated major modules with hardcoded latent widths, hidden widths, timestep counts, and state-dict key assumptions.
   Status: fixed in this follow-up by reconstructing modules from composite checkpoint `config` payloads and accepted metadata types.

## Step 3 - Logical Audit of the Architecture

### Findings

1. `High - Code evidence + Literature-backed`
   Standalone diffusion training and canonical YAML training previously described different effective methodologies. The model implementation already supported CFG schedules, Min-SNR, guidance caps, topology-conditioning modes, and distributed settings, but the direct diffusion CLI did not expose them. This is logically inconsistent because literature-backed diffusion behavior is sensitive to these settings.

2. `High - Code evidence`
   The masked-room stage silently assumed the same latent width and class count forever. That contradicted the repository's own config-system goal of validated, reproducible experiment control.

3. `High - Code evidence + Inference-based`
   The architecture is nominally modular, but data/schema rigidity remains global. The config system validates room size, tile count, node feature count, edge feature count, and TPE width to fixed constants. This is logically acceptable for a Zelda-only thesis artifact, but not for a claimed general dungeon generator.

4. `Medium - Code evidence + Literature-backed`
   The denoiser is large for the amount of training data available. The default latent diffusion model alone is about `107.2M` parameters, while the current VGLC-derived Zelda corpus is tiny. AAAI 2024 single-example level diffusion specifically needed constrained receptive fields and strong representation design to make low-data diffusion work.

5. `Medium - Code evidence`
   The repository historically contained two training front doors: `main.py train` and `src/train.py`. That drift risk is now mitigated because `src/train.py` delegates to `main.py train` instead of maintaining separate argument/config logic.

6. `High - Code evidence + Literature-backed`
   Current-room awareness was previously too implicit. `current_node_idx` existed, but in Block III it was only used to slice one token after global encoding, and in Block IV it was dropped before graph-to-grid attention. That is weaker than the graph-transformer literature recommends, because relative shortest-path structure to the target node is known to be a useful attention bias. This pass fixes that gap by adding explicit current-room distance features to both stages.

7. `Critical - Code evidence + Inference-based`
   Block V's stated role was stronger than its actual optimized objective. Before this follow-up, `LogicNet.forward()` returned `reach_weight * grid_reach_loss`, while `graph_reach_loss` and `lock_loss` were diagnostic only, and room-topology priors were not directly enforced in the scalar loss. That made the architecture description "graph-aware logical solvability guidance" partially overstated. This follow-up fixes the most actionable part of that gap by adding explicit `topology_trace_loss` and `topology_anchor_loss` terms tied to `room_topology_map` and `boundary_constraints`.

8. `High - Code evidence`
   Pipeline checkpoint loading drifted away from the training surface. Composite diffusion checkpoints already stored `diffusion_state_dict`, `condition_encoder_state_dict`, `logic_net_state_dict`, and flat config metadata, but the pipeline loaders still instantiated several modules from hardcoded defaults and only partially recognized composite checkpoint layouts. That made non-default experiments fragile at inference time. This follow-up fixes the loader path and adds tests for composite checkpoint reconstruction.

## Step 4 - Theory vs Implementation Consistency Check

### What the code actually does

The current implementation is more advanced than several older repo documents suggest:

1. Graph conditioning is not just a pooled graph vector. It supports `node_sequence` token conditioning and room-anchor prepending.
2. Diffusion conditioning is not only token cross-attention. It also includes `SpatialGraphConditioner` and `room_topology_map` injection.
3. Diffusion is not plain epsilon-prediction only. It supports `prediction_type in {epsilon, v}` plus `min_snr_gamma`.
4. Sampling is not plain unconditional DDPM. It supports CFG, DDIM, inpainting, fast-sampler adapters, topology caching, and scheduled LogicNet guidance.

### Silent gaps found

1. `Previously fixed, now fixed`
   Hyperparameters already present in theory/method code but absent from the standalone diffusion script: `cfg_*`, `prediction_type`, `min_snr_gamma`, guidance schedule/cap parameters, optimizer/scheduler details, validation sampling count, and distributed flags beyond a tiny subset.

2. `Fixed in this pass`
   `src/train.py` no longer presents a narrower staged interface; it forwards directly into the canonical `main.py train` configuration path.

3. `Fixed in this pass`
   The live docs in `docs/SOTA_COMPARISON_AND_BENCHMARKS.md` and `docs/CURRENT_ARCHITECTURE.md` now describe the actual conditioning, schema, and training surfaces implemented in code.

4. `Fixed in this follow-up`
   Block V previously described a stronger topology-aware objective than the optimized scalar actually used. The implementation now exposes and optimizes `logic_topology_trace_weight` and `logic_topology_anchor_weight`, making the room-topology logical objective explicit instead of implicit.

5. `Fixed in this follow-up`
   Pipeline inference previously risked silent architecture drift because composite diffusion checkpoints were loaded into modules instantiated from hardcoded defaults. The loaders now derive latent widths, hidden widths, timestep counts, attention modes, topology modes, and LogicNet iteration counts from the checkpoint's embedded config, and they accept `model_type="diffusion"` sidecars when loading bundled submodules.

## Step 5 - Gap and Bug Analysis

### Gaps that matter most

| Gap | Why it matters | Evidence | Fix status |
|---|---|---|---|
| Diffusion standalone CLI was method-incomplete | Reproducibility and experiment parity failure | `src/train_diffusion.py` vs `main.py` | `Fixed` |
| Masked-room latent/class assumptions were hardcoded | Hidden dependency on VQ-VAE/data schema | `src/train_masked_room.py` | `Fixed` |
| Current room was not explicitly encoded as a relative graph anchor in Block III/IV | Weakens graph-conditioning precision on small structured graphs | `src/core/condition_encoder.py`, `src/core/graph_grid_attention.py` + Graphormer-style literature | `Fixed in this pass` |
| LogicNet topology priors were not directly optimized in the returned loss | Architecture narrative overstated topology-aware solvability pressure | `src/core/logic_net.py`, `src/train_diffusion.py` | `Fixed in this follow-up` |
| Pipeline loaders hardcoded non-default widths and expected narrow checkpoint layouts | Non-default checkpoints could be reconstructed incorrectly at inference time | `src/pipeline/dungeon_pipeline.py`, `src/pipeline/block_contracts.py` | `Fixed in this follow-up` |
| Legacy `src/train.py` remained thinner than canonical training path | Duplicate surface and documentation drift | `src/train.py` | `Fixed in this pass` |
| Dataset schema is Zelda-locked in validator | Prevents clean transfer to other corpora | `src/config_system.py` | `Open by design, now explicit via schema_profile` |
| No current repo-local evidence that the full hybrid stack beats recent diffusion-PCG baselines | Competitiveness claim remains weak without benchmark parity | literature + repo docs | `Research gap` |

### Parameters that should stay documented

| Parameter | Type | Default | Valid range | Source | Notes |
|---|---|---:|---|---|---|
| `diffusion.model_channels` | int | `128` | `64..192` practical | inference-based + code | Main capacity knob; large effect on memory |
| `diffusion.unet_channel_mult` | list[int] | `[1,2,4]` | non-empty positive ints | code | Must remain divisible by `unet_num_heads` after scaling |
| `diffusion.unet_num_heads` | int | `8` | `4..8` practical | code | Divisibility constraint already enforced |
| `diffusion.condition_gnn_type` | str | `gcn` schema / `gps` recommended config | `{gcn,gat,sage,gps}` | literature-backed | `gps` is the most future-proof option |
| `diffusion.topology_conditioning_mode` | str | `additive` schema / `spade` recommended config | `{additive,spade}` | literature-backed | `spade` is the higher-capacity spatial control path |
| `diffusion.prediction_type` | str | `epsilon` | `{epsilon,v}` | literature-backed | `v` is worth testing for stability, but not forced |
| `diffusion.min_snr_gamma` | float | `5.0` | `0..5` | literature-backed | `0` disables; `5` is the common recommendation |
| `diffusion.cfg_scale` | float | `3.0` | `1..4` safe | literature-backed + inference-based | Very high values risk oversharpening and mode collapse |
| `diffusion.cfg_dropout_prob` | float | `0.1` | `0.05..0.2` | literature-backed | Needed for classifier-free guidance training |
| `diffusion.guidance_scale` | float | `1.0` | `0..1.5` safe | inference-based | Logic guidance can become brittle above this on small graphs |
| `diffusion.guidance_active_fraction` | float | `0.30` | `0.1..0.5` | inference-based | Limits expensive and unstable late-step guidance |
| `diffusion.logic_topology_trace_weight` | float | `0.25` | `>=0`, practical `0..1` | inference-based | Extra pressure for traversability traces implied by room-topology priors |
| `diffusion.logic_topology_anchor_weight` | float | `0.25` | `>=0`, practical `0..1` | inference-based | Extra pressure on start/goal/door anchors; too large can over-constrain walkability |
| `diffusion.graph_auto_linear_attention_nodes` | int | `128` | `64..256` | inference-based | Softmax is fine for Zelda graphs; threshold matters when scaling out |
| `vqvae.latent_dim` | int | `64` | `32..96` practical | inference-based + code | Also now feeds masked-room conditioning assumptions |
| `dataset.num_classes` | int | `44` | fixed today | code | Still dataset-locked in validator |

## Step 6 - Redundancy and Unnecessary Work Analysis

1. `Medium`
   `src/train.py` is now intentionally thin. The redundant logic has been removed, but the compatibility module should stay documented as a wrapper only.

2. `Low`
   Some repo docs still describe older "single context vector" or thinner diffusion behavior even though the code now contains node-sequence tokens, topology refinement, room-topology modulation, and fast-sampling adapters.

3. `Low`
   The codebase contains both legacy and canonical phrasings for LCM/fast-sampler paths. That is documentation overhead more than runtime overhead.

## Step 7 - Computational Complexity Analysis

### Parameter Counts at Current Defaults

Measured from the current code:

| Module | Parameters |
|---|---:|
| `SemanticVQVAE` | `31,095,276` |
| `DualStreamConditionEncoder` (`gps`, hidden `256`) | `5,846,320` |
| `LatentDiffusionModel` | `107,167,309` |
| `LogicNet` | `425,613` |
| `DiscreteMaskedRoomModel` | `107,184,313` |

### Complexity

Let:

- `L_lat = H_lat * W_lat` be latent spatial tokens.
- `N` be graph nodes.
- `E` be graph edges.
- `d` be hidden width.
- `T` be sampling steps.

Then:

1. U-Net self-attention: `O(B * L_lat^2 * d)` per attention block.
2. Graph-to-grid softmax cross-attention: `O(B * L_lat * N * d)`.
3. Linear hedgehog cross-attention: approximately `O(B * (L_lat + N) * d * k)` where `k = hedgehog_feature_dim`.
4. Topology refinement over graph tokens: approximately `O(Layers * (E * d + N * d^2))` for message passing projections.
5. Full sampling cost: multiply denoiser cost by `T`.

For the current Zelda rooms, the latent grid is tiny and graph sizes are modest, so the dominant runtime cost is not attention quadraticity inside one pass. It is the repeated U-Net evaluation across many diffusion timesteps.

### Practical Scale Judgment

1. `Literature-backed`: latent diffusion is the right efficiency choice relative to pixel-space diffusion.
2. `Inference-based`: the current default denoiser is viable for Zelda room scale, but expensive relative to corpus size.
3. `Inference-based`: graph linear-attention fallback is more a future-proofing mechanism than a necessity for the current dataset.

## Step 8 - Hyperparameter Sensitivity Analysis

### Highest-Sensitivity Parameters

| Parameter | Sensitivity | Why |
|---|---|---|
| `diffusion.min_snr_gamma` | High | Directly changes timestep weighting during training |
| `diffusion.cfg_scale` | High | Alters fidelity/diversity tradeoff at inference |
| `diffusion.cfg_dropout_prob` | High | Misconfigured dropout breaks CFG training usefulness |
| `diffusion.guidance_scale` | High | Too large makes logic guidance unstable |
| `diffusion.condition_gnn_type` | Medium-High | Changes graph expressivity and compute profile |
| `diffusion.topology_conditioning_mode` | Medium-High | Changes whether room-topology maps are weak bias or full affine modulation |
| `diffusion.use_teacher_forced_neighbor_latents` | High | Strong effect on room-to-room coherence during room-level training |
| `vqvae.latent_dim` | Medium-High | Changes compression bottleneck and all downstream modules |
| `diffusion.model_channels` | High | Main capacity/memory lever |
| `diffusion.unet_num_heads` | Medium | Affects attention expressivity and divisibility constraints |

### Safe Operating Guidance

1. Keep `diffusion.cfg_scale` in the `1..4` range unless you explicitly trade diversity for stronger conditioning.
2. Keep `diffusion.min_snr_gamma` in the `3..5` range for stable reweighting; set to `0` only for ablation.
3. Keep `diffusion.guidance_scale` at or below `1.5` until larger-graph stress tests show otherwise.
4. Keep `diffusion.condition_num_gnn_layers` in the `2..4` range; deeper stacks are likely to oversmooth on small mission graphs.
5. Prefer `diffusion.condition_gnn_type=gps` for best long-term scalability, but keep `gcn` or `sage` as cheaper baselines.

## Step 9 - Failure Mode and Edge Case Analysis

1. `High`
   Domain shift away from Zelda semantics: the model depends on Zelda-specific tile IDs, room dimensions, graph feature schemas, and symbolic rules.
2. `High`
   Data scarcity overfitting: the training corpus is small enough that reported sample quality may reflect memorization pressure without careful holdout analysis.
3. `High`
   Large graph or dense lock/key structures: logic guidance may become expensive or capped out by `guidance_max_*` thresholds.
4. `Medium`
   Missing graph side-information: the pipeline has fallbacks, but generation quality and coherence degrade sharply if graph features or topology maps are absent.
5. `Medium`
   Excessive CFG or logic guidance: stronger conditioning can force visually plausible but distribution-shifted rooms.

## Step 10 - Scalability and Generalization Boundary Analysis

### Practical Boundaries

1. Data scale boundary:
   The architecture is much more plausible as a Zelda-specialized thesis stack than as a general game-level foundation model.
2. Graph scale boundary:
   Current defaults are comfortable for tens of nodes, not hundreds-to-thousands without broader retraining and profiling.
3. Compute boundary:
   The latent diffusion model is over `107M` parameters before VQ-VAE and condition encoder are counted. Full DDPM training is therefore disproportionately expensive relative to the current dataset size.
4. Transfer boundary:
   Generalization beyond Zelda-like lock-and-key mission graphs is limited by schema-locked feature engineering and symbolic validators.

### Scale-Relevant Parameters

| Parameter | Lower bound | Upper bound | Notes |
|---|---:|---:|---|
| `distributed.nproc_per_node` | `1` | visible GPUs | Diffusion stage only |
| `diffusion.model_channels` | `64` | `192` practical | Main memory lever |
| `diffusion.unet_num_heads` | `4` | `8` practical | Must divide channel widths |
| `diffusion.num_timesteps` | `250` | `1000` practical | Higher improves training fidelity but increases cost |
| `diffusion.graph_auto_linear_attention_nodes` | `0` | `256` | `0` disables threshold |
| `diffusion.guidance_max_graph_nodes` | `64` | `512` default ceiling | Higher increases logic-guidance cost |
| `diffusion.epochs` | problem-dependent | compute-limited | Small-data regime increases overfitting risk quickly |

## Step 11 - Comparison Against State-of-the-Art Baselines

### Overall Judgment

The current stack is `architecturally strong but not empirically SOTA`.

Why:

1. `Literature-backed`
   Recent diffusion/layout papers show strong results when trained with rich datasets or very task-specific inductive biases.
2. `Code evidence + Inference-based`
   This repo has good inductive bias and repair structure, but its empirical claims are still tied to a very small Zelda corpus and mostly repo-local benchmarks.
3. `Literature-backed`
   Strong competing paradigms exist:
   - graph-grammar + learned room generator hybrids,
   - RL-based PCG such as PCGRL,
   - recent diffusion-PCG methods specialized for low-data or single-example settings.

The repo is meaningfully novel in integration:

1. mission graph first,
2. graph-conditioned latent diffusion,
3. differentiable logic guidance,
4. symbolic repair,
5. QD-oriented evaluation.

But that is not the same as being benchmark-dominant.

## Step 12 - Bias and Ethical Risk Analysis

### Architectural Risks

1. `Literature-backed`
   Strong conditioning and self-reinforcing generative loops can amplify representational bias over iterations.
2. `Inference-based`
   This codebase encodes design priors from a narrow historical corpus. Even when demographic bias is not the main concern, gameplay/style bias can be amplified toward one narrow notion of "correct Zelda design".
3. `Inference-based`
   Hard validators can encode unfair exclusions if the target design space later expands to alternative dungeon styles or accessibility-oriented layouts.

### Mitigations

1. Keep symbolic constraints separate from stylistic preferences.
2. Benchmark on multiple style regimes before claiming broad playability.
3. Do not recursively bootstrap future datasets from model outputs without bias monitoring.

## Step 13 - Evidence-Based Decision Table

### Consolidated Configuration Schema Recommendations

| Parameter Name | Type | Default | Valid Range / Options | Source | Notes |
|---|---|---:|---|---|---|
| `diffusion.cfg_schedule_mode` | str | `constant` | `constant, linear_decay, cosine_decay` | literature-backed | Already implemented; now reachable from standalone YAML flow |
| `diffusion.cfg_schedule_min_scale` | float | `1.0` | `>=0` | inference-based | Keep close to `1.0` to avoid disabling conditional signal late |
| `diffusion.cfg_schedule_power` | float | `1.0` | `>0` | inference-based | Shapes late-step conditioning strength |
| `diffusion.prediction_type` | str | `epsilon` | `epsilon, v` | literature-backed | Important methodology toggle |
| `diffusion.min_snr_gamma` | float | `5.0` | `0..5` | literature-backed | Standard diffusion training control |
| `diffusion.guidance_clamp_magnitude` | float | `1.0` | `>=0` | inference-based | Caps unstable guidance gradients |
| `diffusion.guidance_relative_norm_cap` | float | `0.25` | `>=0` | inference-based | Prevents oversized logic steps |
| `diffusion.guidance_active_fraction` | float | `0.30` | `0.05..1.0` | inference-based | Guidance schedule sparsity control |
| `diffusion.guidance_decay_power` | float | `1.0` | `>=0.25` | inference-based | Shapes timestep guidance decay |
| `diffusion.logic_topology_trace_weight` | float | `0.25` | `>=0`, practical `0..1` | inference-based | Makes room-topology traversability supervision explicit in LogicNet |
| `diffusion.logic_topology_anchor_weight` | float | `0.25` | `>=0`, practical `0..1` | inference-based | Makes door/start/goal anchor supervision explicit in LogicNet |
| `diffusion.use_teacher_forced_neighbor_latents` | bool | `true` | `true,false` | inference-based | Important room-coherence lever |
| `diffusion.use_current_node_distance_features` | bool | `true` | `true,false` | literature-backed + inference-based | Enables current-room-relative graph conditioning in Block III/IV |
| `diffusion.current_node_distance_max` | int | `8` | `>=1`, practical `4..16` | inference-based | Distance clip/normalization radius; larger values smooth locality bias |
| `diffusion.topology_conditioning_mode` | str | `additive` | `additive, spade` | literature-backed | Spatial control capacity toggle |
| `diffusion.attention_mode` | str | `softmax` | `softmax, linear_hedgehog` | inference-based | Complexity/quality tradeoff |
| `diffusion.graph_auto_linear_attention_nodes` | int | `128` | `>=0` | inference-based | Auto-switch threshold |
| `masked_room.num_classes` | int | `44` | fixed today | code | Fixed hardcode removed in this pass |
| `masked_room.latent_dim` | int | derived from `vqvae.latent_dim` | positive int | code + inference-based | Fixed hardcode removed in this pass |
| `masked_room.use_current_node_distance_features` | bool | `true` | `true,false` | literature-backed + inference-based | Keeps masked-room graph conditioning aligned with diffusion conditioning |
| `masked_room.current_node_distance_max` | int | `8` | `>=1`, practical `4..16` | inference-based | Should match diffusion unless intentionally ablated |

## Step 14 - Recommended Ablation Plan

Each command assumes the canonical entrypoint:

```bash
python main.py train --config configs/zelda_hmolqd.yaml
```

### Ablations

1. Node-sequence vs pooled graph conditioning

```bash
python main.py train --config configs/zelda_hmolqd.yaml --diffusion-graph-conditioning-mode pooled
```

Expected outcome:
Node-sequence should improve room-graph alignment and solvability consistency. If not, the extra token complexity is not justified.

2. GNN backbone swap

```bash
python main.py train --config configs/zelda_hmolqd.yaml --diffusion-condition-gnn-type gcn
python main.py train --config configs/zelda_hmolqd.yaml --diffusion-condition-gnn-type gps
```

Expected outcome:
`gps` should help when mission graphs become more structurally varied; `gcn` should be cheaper but less expressive.

3. Disable Min-SNR weighting

```bash
python main.py train --config configs/zelda_hmolqd.yaml --diffusion-min-snr-gamma 0
```

Expected outcome:
Training should become less stable or less sample-efficient.

4. Topology-conditioning path

```bash
python main.py train --config configs/zelda_hmolqd.yaml --diffusion-topology-conditioning-mode additive
python main.py train --config configs/zelda_hmolqd.yaml --diffusion-topology-conditioning-mode spade
```

Expected outcome:
`spade` should improve boundary and topology-map fidelity if the topology maps are informative.

5. Teacher-forced neighbor latent ablation

```bash
python main.py train --config configs/zelda_hmolqd.yaml --no-diffusion-use-teacher-forced-neighbor-latents
```

Expected outcome:
Inter-room coherence should drop if teacher forcing is carrying local consistency.

6. Logic-loss target mode

```bash
python main.py train --config configs/zelda_hmolqd.yaml --diffusion-logic-loss-mode detached_real
python main.py train --config configs/zelda_hmolqd.yaml --diffusion-logic-loss-mode predicted_latent
```

Expected outcome:
`predicted_latent` should better train the generative path because logic gradients reach the denoiser.

7. Current-room distance encoding ablation

```bash
python main.py train --config configs/zelda_hmolqd.yaml --no-diffusion-use-current-node-distance-features
python main.py train --config configs/zelda_hmolqd.yaml --diffusion-current-node-distance-max 4
python main.py train --config configs/zelda_hmolqd.yaml --diffusion-current-node-distance-max 8
python main.py train --config configs/zelda_hmolqd.yaml --diffusion-current-node-distance-max 16
```

Expected outcome:
Turning the feature off should hurt room-to-graph alignment and solvability consistency most on progression-heavy dungeons. Increasing the max radius too far should weaken the locality bias and reduce the gain.

8. LogicNet topology-loss ablation

```bash
python main.py train --config configs/zelda_hmolqd.yaml --diffusion-logic-topology-trace-weight 0.0 --diffusion-logic-topology-anchor-weight 0.0
python main.py train --config configs/zelda_hmolqd.yaml --diffusion-logic-topology-trace-weight 0.25 --diffusion-logic-topology-anchor-weight 0.25
python main.py train --config configs/zelda_hmolqd.yaml --diffusion-logic-topology-trace-weight 0.6 --diffusion-logic-topology-anchor-weight 0.4
```

Expected outcome:
Zeroing the topology-aware LogicNet weights should reduce room-topology fidelity and weaken logical guidance. Increasing them moderately should help only if topology priors are accurate; pushing them too high risks over-constraining walkability and hurting diversity.

## Step 15 - Priority Ranking

| Priority | Finding | Reproducibility Risk |
|---|---|---|
| Critical | Standalone diffusion CLI previously hid core methodology controls from the canonical config system | Yes |
| Critical | Block V previously optimized a weaker logical objective than the architecture description implied | Yes |
| High | Current-room graph awareness was previously implicit and under-specified in Block III/IV | Yes |
| High | Pipeline loaders previously reconstructed non-default checkpoints from hardcoded defaults | Yes |
| High | Masked-room stage hardcoded class count and latent width | Yes |
| High | Dataset and graph schema are globally Zelda-locked | Yes |
| High | Full stack is large relative to corpus size | No |
| Medium | `src/train.py` duplicated a thinner legacy training surface | Yes, mitigated |
| Medium | Docs under-describe actual node-sequence/topology-conditioned diffusion behavior | Yes |
| Low | Naming/docs drift around fast sampler / LCM terminology | Yes |

## Step 16 - Immediate Implementation Summary

Changes landed in this pass:

1. `src/train_diffusion.py`
   Added shared-config-aware stage construction so `--config` now unlocks the full validated YAML methodology surface for diffusion-only runs.

2. `main.py`
   Replaced duplicated diffusion-stage kwarg assembly with a shared helper from `src/train_diffusion.py`.

3. `src/train_masked_room.py`
   Added explicit `num_classes` and `latent_dim` config plumbing.
   Removed masked-room hardcodes for class count and latent width.
   Added `--config` support for standalone masked-room runs.

4. `main.py`
   Replaced duplicated masked-room-stage kwarg assembly with a shared helper from `src/train_masked_room.py`.

5. `tests/test_config_system.py`
   Added focused coverage for the shared diffusion and masked-room config builders.

6. `src/pipeline/graph_features.py`, `src/core/condition_encoder.py`, `src/core/graph_grid_attention.py`, `src/core/latent_diffusion.py`
   Added current-room distance encoding for graph conditioning.
   The feature is computed from mission-graph shortest paths and injected both as node-aligned features and as an attention bias.

7. `src/pipeline/dungeon_pipeline.py`, `src/train_diffusion.py`, `src/train_masked_room.py`, `src/core/discrete_masked_model.py`
   Threaded current-room distance features through training, inference, batching, and masked-room conditioning.
   Added YAML/CLI controls: `use_current_node_distance_features` and `current_node_distance_max`.

8. `tests/test_train_diffusion_conditioning_shapes.py`, `tests/test_ml_components.py`, `tests/test_architecture_audit_fixes.py`
   Added coverage proving the new feature is built, batched, and consumed by the attention path.

9. `src/config_system.py`, `configs/zelda_hmolqd.yaml`
   Made the Zelda schema lock explicit as `dataset.schema_profile=zelda_v1`.
   Recorded the schema lock in reproducibility metadata and updated the canonical YAML to the reduced-capacity small-data profile recommended by the audit.

10. `src/train.py`
   Replaced the old parallel training surface with a compatibility wrapper over `main.py train`.

11. `src/utils/model_capacity.py`, `src/train_diffusion.py`, `src/train_masked_room.py`
   Added runtime capacity guardrails that log trainable parameter counts and warn when the configured model is oversized relative to the dataset.

12. `src/core/logic_net.py`, `src/train_diffusion.py`, `src/core/latent_diffusion.py`
   Added explicit room-topology-aware LogicNet supervision via `topology_trace_loss` and `topology_anchor_loss`.
   Batched `room_topology_map` and `boundary_constraints` now flow into LogicNet training, validation, and gradient guidance paths.
   Added YAML/CLI controls: `logic_topology_trace_weight` and `logic_topology_anchor_weight`.

13. `src/pipeline/dungeon_pipeline.py`, `src/pipeline/block_contracts.py`
   Composite diffusion checkpoints now load bundled condition encoder and LogicNet weights from `model_type="diffusion"` sidecars.
   Loader reconstruction now uses embedded checkpoint config for latent width, hidden width, timestep count, topology mode, attention mode, and LogicNet iterations instead of silently hardcoded defaults.
   Invalid diffusion checkpoints without a loadable state dict now fail clearly instead of falling through to confusing downstream errors.

14. `tests/test_hmolqd/test_logic_net.py`, `tests/test_block_integration.py`, `tests/test_train_diffusion_conditioning_shapes.py`, `tests/test_architecture_audit_fixes.py`, `tests/test_config_system.py`
   Added focused coverage for topology-aware LogicNet loss composition, room-topology-only guidance, batched topology context flow into LogicNet, composite diffusion checkpoint reconstruction, missing-state-dict rejection, and the new config knobs.

15. `docs/CURRENT_ARCHITECTURE.md`, `docs/SOTA_COMPARISON_AND_BENCHMARKS.md`
   Refreshed the live docs so they match the canonical training path, schema lock, and current-room distance conditioning.

## Source Links

1. Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS 2020
   https://papers.nips.cc/paper_files/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html
2. Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models*, CVPR 2022
   https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf
3. Nichol et al., *GLIDE*, ICML 2022
   https://proceedings.mlr.press/v162/nichol22a.html
4. Hang et al., *Efficient Diffusion Training via Min-SNR Weighting Strategy*, ICCV 2023
   https://openaccess.thecvf.com/content/ICCV2023/supplemental/Hang_Efficient_Diffusion_Training_ICCV_2023_supplemental.pdf
5. Chai et al., *LayoutDM*, CVPR 2023
   https://openaccess.thecvf.com/content/CVPR2023/html/Chai_LayoutDM_Transformer-Based_Diffusion_Model_for_Layout_Generation_CVPR_2023_paper.html
6. Dai et al., *Procedural Level Generation with Diffusion Models from a Single Example*, AAAI 2024
   https://ojs.aaai.org/index.php/AAAI/article/view/28865
7. Dormans and Bakkes, *Generating Missions and Spaces for Adaptable Play Experiences*, IEEE TCIAIG 2011
   https://dblp.org/rec/journals/tciaig/DormansB11
8. Gutierrez and Schrum, *Generative Adversarial Network Rooms in Generative Graph Grammar Dungeons for The Legend of Zelda*, CEC 2020 project page
   https://people.southwestern.edu/~schrum2/SCOPE/zelda-graphgan.php
9. Khalifa et al., *PCGRL: Procedural Content Generation via Reinforcement Learning*, IEEE Transactions on Games 2020
   https://arxiv.org/abs/2001.09212
10. Rampasek et al., *Recipe for a General, Powerful, Scalable Graph Transformer*
   https://dblp.org/rec/journals/corr/abs-2205-12454
11. Ying et al., *Do Transformers Really Perform Bad for Graph Representation?*, NeurIPS 2021
   https://arxiv.org/pdf/2106.05234.pdf
12. Brody et al., *How Attentive are Graph Attention Networks?*, ICLR 2022
   https://openreview.net/forum?id=F72ximsx7C1
13. Li et al., *Distance Encoding: Design Provably More Powerful Neural Networks for Graph Representation Learning*, 2020
   https://papers.nips.cc/paper/2020/hash/2f73168bf3656f697507752ec592c437-Abstract.html
14. Park et al., *Semantic Image Synthesis with Spatially-Adaptive Normalization*, CVPR 2019
   https://dblp.org/rec/conf/cvpr/Park0WZ19
15. Liu et al., *An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution*, NeurIPS 2018
   https://arxiv.org/abs/1807.03247
16. Chen et al., *Would Deep Generative Models Amplify Bias in Future Models?*, CVPR 2024
   https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_Would_Deep_Generative_Models_Amplify_Bias_in_Future_Models_CVPR_2024_paper.pdf
