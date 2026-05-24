# Architecture Research Audit: Topology Signal and Generation Quality

Last updated: 2026-04-04

This audit is the continuation of:

- [ARCHITECTURE_RESEARCH_AUDIT_2026_03_31.md](ARCHITECTURE_RESEARCH_AUDIT_2026_03_31.md)
- [ARCHITECTURE_RESEARCH_AUDIT_DIFFUSION_CONDITIONING_ADDENDUM_2026_03_31.md](ARCHITECTURE_RESEARCH_AUDIT_DIFFUSION_CONDITIONING_ADDENDUM_2026_03_31.md)
- [TOPOLOGY_GRAPH_RESEARCH_AUDIT_2026_03_31.md](TOPOLOGY_GRAPH_RESEARCH_AUDIT_2026_03_31.md)

This pass focuses on one concrete question:

`How do we make the current Zelda architecture use topology information more effectively so room generation is stronger and more semantically aligned?`

It also records the implementation changes applied in this pass and the cleanup performed on stale result folders.

## Scope

This is a targeted audit of the current hybrid pipeline:

1. `Block I`: topology graph generation
2. `Block II`: VQ-VAE room compression
3. `Block III`: graph and local conditioning
4. `Block IV`: latent diffusion teacher + fast sampler student
5. `Masked-room branch`: discrete parallel room generator
6. `Runtime`: symbolic repair, graph-owned marker placement, and rendering

The audit is intentionally narrower than a whole-repo survey. It focuses on the generation-critical path and the topology-to-room information contract.

## Phase 1 - Research

## Step 1 - Deep Research and Literature Review

### Primary literature reviewed

| Area | Paper | Venue | Why it matters here |
|---|---|---|---|
| Discrete latent compression | van den Oord et al., *Neural Discrete Representation Learning* | NeurIPS 2017 | Justifies the VQ-VAE room-latent stage under small-data regimes. |
| Latent diffusion | Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models* | CVPR 2022 | Justifies latent-space diffusion instead of pixel-space diffusion for efficiency and conditioning. |
| Guidance | Ho and Salimans, *Classifier-Free Diffusion Guidance* | NeurIPS workshop / arXiv 2022 | Grounds the CFG path used by the teacher and fast-sampler regime selection. |
| Diffusion optimization | Hang et al., *Efficient Diffusion Training via Min-SNR Weighting Strategy* | ICCV 2023 | Supports the exposed `min_snr_gamma` training knob for convergence and quality. |
| Few-step distillation | Song et al., *Consistency Models* | ICML 2023 | Provides the conceptual baseline for few-step sampling and distillation. |
| Parallel masked generation | Chang et al., *MaskGIT* | CVPR 2022 | Strongest direct analogue for the masked-room branch. |
| Graph transformers | Rampasek et al., *Recipe for a General, Powerful, Scalable Graph Transformer* | NeurIPS 2022 workshop-era graph literature / arXiv | Supports the `gps` conditioner option and hybrid local-global graph processing. |
| Graph-conditioned structured generation | Shabani et al., *HouseDiffusion* | CVPR 2023 | Supports richer structured conditioning and graph-aware generation for geometry/layout tasks. |
| Graph-conditioned layout generation | Hu et al., *Graph2Plan* | CVPR 2020 | Supports graph-first generation of structured spaces and explicit graph-to-layout conditioning. |
| Game-PCG under tight constraints | Rodriguez Torrado et al., *Bootstrapping Conditional GANs for Video Game Level Generation* | IEEE CoG / arXiv 2019 | Direct evidence that playability and constraint satisfaction are hard in low-data level generation. |

### Key findings from the literature

1. `Latent diffusion remains the right efficiency bias`.
   Rombach et al. show diffusion in latent space preserves conditioning flexibility while reducing training and inference cost relative to pixel-space diffusion. That still matches this repo’s room-level setup well. Source: https://arxiv.org/abs/2112.10752

2. `CFG is useful, but it is a fidelity-diversity tradeoff, not a free win`.
   Ho and Salimans explicitly frame classifier-free guidance as a tradeoff between fidelity and mode coverage. That matches the repo’s earlier fast-sampler failures under mismatched runtime CFG. Source: https://arxiv.org/abs/2207.12598

3. `Min-SNR weighting is one of the few diffusion-training tweaks with strong empirical backing`.
   Hang et al. report `3.4x` faster convergence and ImageNet-256 FID `2.06`, which supports keeping `diffusion.min_snr_gamma` exposed and tuned rather than hardcoded. Source: https://arxiv.org/abs/2303.09556

4. `Few-step students inherit teacher quality limits`.
   Consistency Models and related distillation work show strong few-step generation is possible, but the student still depends on a clean teacher objective and aligned runtime regime. Source: https://arxiv.org/abs/2303.01469

5. `MaskGIT-style parallel token generation is a good fit for room grids, but only when conditioning is explicit and local`.
   Chang et al. show masked-token iterative synthesis can outperform raster autoregression and accelerate decoding by up to `64x`. That supports the masked-room branch, but not a topology-free variant. Source: https://arxiv.org/abs/2202.04200

6. `Graph-conditioned structure generation works best when the graph contract is explicit`.
   Graph2Plan and HouseDiffusion both support the repo’s graph-first factorization rather than end-to-end monolithic dungeon generation. Sources:
   - https://arxiv.org/abs/2004.13204
   - https://arxiv.org/abs/2211.13287

7. `Low-data level generation needs stronger constraint scaffolding than generic image generation`.
   Rodriguez Torrado et al. explicitly note that aesthetic appeal and playability are both hard under limited training data. That supports the repo’s symbolic repair and graph-owned semantics. Source: https://arxiv.org/abs/1910.01603

### State-of-the-art benchmark anchors

These are not direct apples-to-apples Zelda benchmarks, but they establish the method frontier the current architecture should be judged against:

| Paper | Benchmark claim from source |
|---|---|
| *Efficient Diffusion Training via Min-SNR Weighting Strategy* | `3.4x` faster convergence; ImageNet-256 FID `2.06` |
| *Consistency Models* | one-step FID `3.55` on CIFAR-10; `6.20` on ImageNet 64x64 |
| *MaskGIT* | up to `64x` faster decoding than prior autoregressive transformer baselines |

For Zelda generation specifically, the more relevant takeaway is not the exact benchmark number, but which design patterns consistently win:

- efficient latent-space generation
- strong local semantics
- explicit structured conditioning
- fewer hidden runtime mismatches

## Phase 2 - Deep Analysis

## Step 2 - Assumptions Validation

### Assumptions the current architecture makes

| Assumption | Status | Evidence / comment |
|---|---|---|
| Rooms can be factorized from global dungeon generation | `Valid` | Supported by Graph2Plan, HouseDiffusion, and prior repo audits. |
| Room size is fixed to `16x11` and tile vocabulary to `44` classes | `Valid for this corpus, fragile in general` | Good for Zelda schema-lock; not portable without retraining and schema changes. |
| Mission progression is better handled symbolically than by a monolithic denoiser | `Valid` | Supported by the game-PCG literature and the repo’s Block I results. |
| Diffusion teacher can rely on strong topology priors to stay on-manifold | `Mostly valid` | Works if topology priors are consistent between loader, training, and runtime. |
| Masked-room branch can learn from the same topology tensor as runtime | `Previously false in part` | Runtime knew about semantic role anchors; training hard-fixed only start/goal/doors. Fixed in this pass. |
| Current checkpoints are only expected to work on the canonical Zelda schema | `Valid but under-documented` | This is an intended repository constraint, but it should remain explicit. |
| Existing graph labels are semantically consistent (`S` means stair, not start) | `Fragile historically` | This was already a real failure mode and had to be fixed. |

### Hardcoded implementation assumptions that should be considered config candidates

These are not all bugs. Some are reasonable heuristics. But they are currently silent policy choices:

| Location | Hardcoded assumption | Candidate config? |
|---|---|---|
| `src/pipeline/room_topology_conditioning.py` | room-role prior fill strength `0.15` | `Yes` |
| `src/pipeline/room_topology_conditioning.py` | semantic anchor interpolation alphas `0.55 / 0.72 / 0.38 / 0.62` | `Yes` |
| `src/pipeline/room_topology_conditioning.py` | puzzle perpendicular offset magnitude `2` | `Yes` |
| `src/core/discrete_masked_model.py` | fixed-token threshold `> 0.5` on topology channels | `Maybe` |
| `src/pipeline/dungeon_pipeline.py` | teacher fallback policy for fast-sampler suspicious rooms | `Yes` |
| `src/pipeline/dungeon_pipeline.py` | preferred room-marker search order and radius | `Yes` |

## Step 3 - Logical Audit of the Architecture

### Finding 1 - The topology contract was richer at runtime than during masked-room training

This was the highest-value logical gap.

What happened:

- the topology tensor already contained localized semantic anchors for roles like key, item, boss, and puzzle
- runtime placement used those anchors
- masked-room training only hard-fixed `start`, `goal`, and `door_*`

Why it is problematic:

- the model was asked to learn semantic placement from a weaker target contract than the one enforced at generation time
- this creates train/runtime skew
- structured-generation literature consistently rewards alignment between conditioning contract and training objective

Decision:

- fixed in this pass by making `build_fixed_mask_from_topology_map(...)` also preserve localized semantic role anchors

### Finding 2 - Some semantics remain intentionally deterministic rather than fully learned

This is not a bug, but it is a design tradeoff the repo must describe honestly.

The model does not fully own:

- start placement
- triforce placement
- final item semantics
- some puzzle/stair markers

Instead, the graph and symbolic pipeline own those markers.

Why this is acceptable:

- it improves controllability and progression correctness
- it matches the low-data PCG literature better than trusting the model to invent mission semantics reliably

Why it is also limiting:

- it caps how “fully neural” the result really is
- it reduces the novelty of semantic placement

### Finding 3 - The fast sampler is not the main quality ceiling anymore

After the earlier runtime fixes, the fast-sampler branch is mostly a teacher-fidelity problem, not a hidden-sampler-bug problem.

Why this matters:

- future quality work should primarily target teacher training, topology signal quality, and semantic alignment
- not another round of sampler-only patching

## Step 4 - Theory vs. Implementation Consistency Check

### Consistent now

- topology-first generation matches the documented hybrid architecture
- diffusion and masked-room both consume room-topology maps
- runtime semantic placement uses shared semantic anchors

### Previously inconsistent, fixed in this pass

- `Theory`: topology maps should communicate mission-critical room semantics
- `Implementation`: masked-room training only froze start/goal/doors
- `Now`: masked-room training freezes semantic role anchors too

### Remaining theory-vs-implementation gap

The repo is still best described as:

`neural room structure + symbolic semantic placement`

not:

`fully neural end-to-end semantic room synthesis`

That difference should remain explicit in documentation and any claims.

## Step 5 - Gap and Bug Analysis

### Fixed now

1. `Critical`: masked-room training underused semantic topology anchors.
2. `Low`: stale verification folders made it harder to compare current outputs against current code.

### Still missing / recommended next

1. `High`: expose semantic-anchor policy knobs to YAML/CLI if systematic ablations are planned.
2. `High`: retrain masked-room and diffusion branches after this topology-signal change to realize the full benefit.
3. `Medium`: add an explicit metric for semantic-anchor adherence at room level.
4. `Medium`: add checkpoint metadata for topology-anchor policy versioning.

### Config additions recommended from this gap analysis

| Parameter name | Type | Default | Valid range | Why |
|---|---|---:|---|---|
| `generation.semantic_role_prior_strength` | float | `0.15` | `0.0..1.0` | Controls broadcast prior strength in topology maps. |
| `generation.semantic_anchor_threshold` | float | `0.5` | `0.0..1.0` | Controls topology-to-fixed-mask thresholding. |
| `generation.semantic_puzzle_offset` | int | `2` | `0..4` | Controls puzzle anchor displacement. |
| `generation.fast_sampler_teacher_fallback_enabled` | bool | `true` | `{true,false}` | Makes the fallback policy reproducible and experimentable. |

These are recommendations from the audit. They were not all promoted in this pass because the training-path fix was higher-value and lower-risk.

## Step 6 - Redundancy and Unnecessary Work Analysis

### Redundancy found

1. `Old result folders`
   They were consuming space and confusing comparisons after the architecture moved on. Removed in this pass.

2. `Semantic duplication between runtime placement and training contract`
   This was partially redundant and partially inconsistent. The runtime had richer anchor semantics than the trainer. The fix removes the inconsistency without adding new overhead.

### Over-engineering watchlist

1. The masked-room branch remains an auxiliary branch unless it can show unique value beyond the teacher and repair pipeline.
2. The fast-sampler branch should not be asked to solve semantic-placement quality that the teacher and symbolic layers still own.

## Step 7 - Computational Complexity Analysis

### Big-O view

| Block | Approximate complexity |
|---|---|
| VQ-VAE | `O(HW * C^2 * k^2)` |
| Condition encoder local path | `O(HW * d)` plus small CNN cost on up to four neighbor rooms |
| Condition encoder graph path | `O(L_g * E * d)` for message passing plus `O(N^2 * d)` for global attention when enabled |
| Diffusion U-Net per denoising step | `O(sum_l H_l W_l C_l^2 k^2 + HWN d + N^2 d)` |
| Masked-room U-Net per iteration | similar U-Net cost, but iterative masked-token refinement instead of diffusion time steps |
| LogicNet | roughly `O(I * (HW + N^2))` |

### Current measured parameter counts on the canonical YAML

Measured locally in this pass:

| Module | Parameters |
|---|---:|
| VQ-VAE | `17,623,948` |
| Diffusion denoiser | `66,599,302` |
| Diffusion condition encoder | `3,206,961` |
| LogicNet | `274,957` |
| Teacher branch total | `70,081,220` |
| Masked-room model | `10,502,534` |
| Masked-room condition encoder | `1,946,289` |
| Masked-room total | `12,448,823` |

### FLOPs / memory note

Exact current FLOPs were not re-profiled in this pass because the environment does not include a FLOP profiler such as `fvcore`, `ptflops`, or `thop`.

What we can say confidently:

- diffusion cost still scales linearly with denoising steps
- the denoiser remains the dominant cost center
- the masked-room branch is now much cheaper than its earlier oversized profile, but still non-trivial
- the semantic-anchor training fix adds negligible compute because it only expands a boolean mask over existing topology channels

### Complexity-sensitive parameters that must remain configurable

| Parameter | Safe range | Notes |
|---|---|---|
| `diffusion.model_channels` | `64..128` | strongest single cost knob for the teacher |
| `diffusion.unet_channel_mult` | short positive tuples | scale-depth/cost tradeoff |
| `diffusion.unet_num_res_blocks` | `1..3` | near-linear multiplier |
| `diffusion.unet_num_heads` | divisors of active widths | must divide attention widths |
| `diffusion.condition_hidden_dim` | `128..256` | graph conditioner cost/overfitting balance |
| `diffusion.condition_num_gnn_layers` | `1..4` | deeper can oversmooth |
| `diffusion.num_timesteps` | `100..1000` | training and high-step inference cost |
| `masked_room.model_channels` | `48..96` | main masked-room cost knob |
| `masked_room.hidden_dim` | `32..96` | token backbone width |
| `topology.population_size` | `16..128` | direct Block I runtime multiplier |
| `topology.generations` | `12..200` | direct Block I runtime multiplier |

## Step 8 - Hyperparameter Sensitivity Analysis

### Most sensitive generation-quality knobs

| Parameter | Type | Default | Valid range | Interdependency |
|---|---|---:|---|---|
| `diffusion.cfg_scale` | float | `3.0` | `1.0..5.0` | interacts with fast-sampler distillation regime |
| `diffusion.min_snr_gamma` | float | `5.0` | `1.0..7.0` | depends on schedule and prediction type |
| `diffusion.alpha_logic` | float | `0.1` | `0.0..0.3` | too high can degrade visual fidelity |
| `diffusion.logic_topology_trace_weight` | float | `0.25` | `0.0..0.5` | pairs with anchor weight |
| `diffusion.logic_topology_anchor_weight` | float | `0.25` | `0.0..0.5` | pairs with trace weight |
| `diffusion.topology_conditioning_mode` | enum | `additive` | `{additive,spade}` | depends on topology-map quality |
| `masked_room.min_mask_ratio` | float | `0.12` | `0.05..0.4` | must be `<= max_mask_ratio` |
| `masked_room.max_mask_ratio` | float | `0.85` | `0.5..0.95` | must be `>= min_mask_ratio` |
| `masked_room.model_channels` | int | `64` | `48..96` | interacts with hidden_dim and total params |
| `masked_room.hidden_dim` | int | `48` | `32..96` | interacts with context_dim |

### Newly important after this pass

| Parameter | Type | Default | Valid range | Source |
|---|---|---:|---|---|
| `generation.semantic_role_prior_strength` | float | `0.15` | `0.0..1.0` | inference-based candidate from this audit |
| `generation.semantic_anchor_threshold` | float | `0.5` | `0.0..1.0` | inference-based candidate from this audit |

## Step 9 - Failure Mode and Edge Case Analysis

1. `OOD topology semantics`
   If a graph contains role combinations or labels not covered by the anchor builder, semantic alignment will degrade or fall back to heuristics.

2. `Schema drift`
   Any change to room shape, palette size, or graph feature schema can silently invalidate checkpoints.

3. `Teacher quality ceiling`
   Fast-sampler quality still degrades if the teacher itself is weak, even when runtime is now aligned.

4. `Over-strong semantic priors`
   If future experiments push anchor priors too strongly, the generator can overfit to marker positions and lose room diversity.

## Step 10 - Scalability and Generalization Boundary Analysis

### Where this architecture degrades

| Axis | Boundary |
|---|---|
| Data scale | Below the current Zelda-room regime, the teacher becomes highly dependent on repair and deterministic semantics. |
| Model scale | The teacher branch is already `~70M` parameters; much larger models are hard to justify on this corpus. |
| Domain shift | The current schema-lock makes transfer outside canonical Zelda-style rooms brittle. |
| Long-horizon generation | Still delegated to Block I; room generators alone are not enough. |

### Graceful degradation vs hard failure

- `Graceful`: room-local style weakens, more repair is needed
- `Hard failure`: schema mismatch, label mismatch, or checkpoint/conditioning contract drift

## Step 11 - Comparison Against SOTA Baselines

### Verdict

The architecture is:

- `Competitive as a controllable hybrid PCG system`
- `Not SOTA as an end-to-end neural generator`
- `Meaningfully novel in its hybrid factorization, but not novel enough to skip matched-budget ablations`

### Why

Compared with the literature:

- it is stronger on explicit controllability and progression than room-only diffusion baselines
- it is weaker on pure visual simplicity and elegance than specialized layout or image generators
- it relies more on symbolic scaffolding than current SOTA end-to-end generative work

That is acceptable if the claim is:

`controllable hybrid Zelda dungeon generation`

and not:

`purely learned end-to-end SOTA generation`

## Step 12 - Bias and Ethical Risk Analysis

This architecture has limited demographic fairness relevance because the output domain is Zelda-style level generation rather than human-centered prediction.

The architecture-specific risks are instead:

- hidden determinism disguised as learned behavior
- over-claiming neural semantic placement when markers are graph-owned
- poor reproducibility if runtime heuristics and fallback policies are undocumented

These are primarily scientific-validity and reproducibility risks, not demographic fairness risks.

## Phase 3 - Synthesis

## Step 13 - Evidence-Based Decision Making

### Core decisions and evidence

| Decision | Backing |
|---|---|
| Keep latent diffusion as the teacher path | LDM efficiency and conditioning flexibility |
| Keep CFG moderate and teacher-aligned | CFG paper + prior repo failures |
| Keep Min-SNR exposed | Hang et al. |
| Keep graph-first topology factorization | Graph2Plan, HouseDiffusion, game-PCG literature |
| Strengthen training use of semantic anchors | inference from structured-generation literature + direct repo train/runtime mismatch |

### Consolidated configuration schema table for recommended knobs

| Parameter Name | Type | Default | Valid Range | Source | Notes |
|---|---|---:|---|---|---|
| `diffusion.cfg_scale` | float | `3.0` | `1.0..5.0` | CFG paper + repo evidence | teacher and student should stay aligned |
| `diffusion.min_snr_gamma` | float | `5.0` | `1.0..7.0` | Min-SNR paper | keep exposed |
| `diffusion.topology_conditioning_mode` | enum | `additive` | `{additive,spade}` | literature + repo evidence | ablate rather than freeze forever |
| `masked_room.min_mask_ratio` | float | `0.12` | `0.05..0.4` | inference + repo stability | |
| `masked_room.max_mask_ratio` | float | `0.85` | `0.5..0.95` | inference + repo stability | |
| `masked_room.model_channels` | int | `64` | `48..96` | inference + current capacity audit | |
| `masked_room.hidden_dim` | int | `48` | `32..96` | inference + current capacity audit | |
| `generation.semantic_role_prior_strength` | float | `0.15` | `0.0..1.0` | inference-based | config candidate |
| `generation.semantic_anchor_threshold` | float | `0.5` | `0.0..1.0` | inference-based | config candidate |
| `generation.fast_sampler_teacher_fallback_enabled` | bool | `true` | `{true,false}` | inference-based | config candidate |

## Step 14 - Ablation Study Recommendation

### A1 - Semantic-anchor training mask on vs off

What changes:

- preserve only start/goal/doors vs preserve start/goal/doors plus semantic role anchors

What it tests:

- whether richer topology supervision improves semantic placement stability and anchor adherence

Metric:

- room-level semantic-anchor accuracy
- repair rate
- final graph-marker correction count

Expected outcome:

- anchor-aware training should reduce semantic jitter and reduce correction pressure

Reproducible command:

```powershell
python main.py train --config configs/zelda_hmolqd.yaml --stage masked_room --output-dir outputs/ablation_masked_room_semantic_anchor_on --no-auto-resume --verbose
```

Candidate counterfactual after config promotion:

```powershell
python main.py train --config configs/zelda_hmolqd.yaml --stage masked_room --generation-semantic-anchor-threshold 1.1 --output-dir outputs/ablation_masked_room_semantic_anchor_off --no-auto-resume --verbose
```

### A2 - Additive vs SPADE topology conditioning

What changes:

- `diffusion.topology_conditioning_mode` or `masked_room.topology_conditioning_mode`

Metric:

- visual fidelity
- topology adherence
- repair rate

Expected outcome:

- one mode will likely dominate on this tiny corpus; do not assume SPADE is always better

```powershell
python main.py train --config configs/zelda_hmolqd.yaml --stage diffusion --diffusion-topology-conditioning-mode additive --output-dir outputs/ablation_diffusion_topo_additive --no-auto-resume --verbose
```

```powershell
python main.py train --config configs/zelda_hmolqd.yaml --stage diffusion --diffusion-topology-conditioning-mode spade --output-dir outputs/ablation_diffusion_topo_spade --no-auto-resume --verbose
```

### A3 - Fast sampler vs teacher under fixed topology

Metric:

- room-diff audit tile disagreement
- repair count
- generation time

Expected outcome:

- fast sampler should stay close in geometry but still lose some structure in difficult rooms

```powershell
python scripts/run_fixed_graph_multi_seed_audit.py --output-dir outputs/ablation_fixed_graph_multi_seed
```

## Step 15 - Priority Ranking of Findings

| Priority | Finding | Reproducibility Risk |
|---|---|---|
| `Critical` | masked-room training previously ignored semantic topology anchors used at runtime | `Yes` |
| `High` | topology-anchor policy still depends on hidden heuristics (`0.15`, interpolation alphas, threshold `0.5`) | `Yes` |
| `High` | teacher quality remains the main ceiling for fast-sampler quality | `No` |
| `Medium` | semantic placement remains partially deterministic rather than learned | `Yes` |
| `Medium` | no direct metric yet for semantic-anchor adherence | `Yes` |
| `Low` | stale result folders were cluttering comparisons | `No` |

## Phase 4 - Implementation

## Step 16 - Immediate Implementation Applied in This Pass

### Code changes made now

1. `Strengthened masked-room training with topology semantic anchors`

Changed:

- [discrete_masked_model.py](/F:/KLTN/src/core/discrete_masked_model.py)

What changed:

- `build_fixed_mask_from_topology_map(...)` now preserves localized semantic role anchors in addition to start/goal/door positions.

Why:

- removes the train/runtime mismatch
- makes the masked-room branch learn from the same topology signal that runtime uses

2. `Added regression coverage`

Changed:

- [test_discrete_masked_room_model.py](/F:/KLTN/tests/test_discrete_masked_room_model.py)

What changed:

- added a test that verifies role-key and role-boss anchors are actually frozen by the topology-derived fixed mask

3. `Cleaned stale result folders but preserved checkpoints`

Removed:

- `outputs/zelda_hmolqd_fulltrain_rerun/semantic_anchor_verification`
- `outputs/zelda_hmolqd_teacher_retrain_v2/full_architecture_verification`

Preserved:

- every `checkpoints/` directory and checkpoint artifact

### Verification

Executed:

```powershell
python -m pytest tests\test_discrete_masked_room_model.py tests\test_zelda_loader_graph_conditioning.py -q
```

Result:

- `24 passed`

Executed:

```powershell
python -m py_compile src\core\discrete_masked_model.py tests\test_discrete_masked_room_model.py
```

Result:

- passed

## Bottom line

The highest-value generation-quality fix in this pass was not another sampler trick. It was removing a topology-information mismatch inside masked-room training.

The architecture is now more internally consistent:

- topology semantics are richer
- runtime and training are better aligned
- stale results are cleaned away
- checkpoints are preserved

The next best move is empirical, not architectural:

1. retrain masked-room on the updated topology contract
2. rerun fixed-graph teacher-vs-student audits
3. promote the remaining hidden anchor heuristics into config only if those ablations show sensitivity

## Sources

- van den Oord et al., *Neural Discrete Representation Learning*: https://arxiv.org/abs/1711.00937
- Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models*: https://arxiv.org/abs/2112.10752
- Ho and Salimans, *Classifier-Free Diffusion Guidance*: https://arxiv.org/abs/2207.12598
- Hang et al., *Efficient Diffusion Training via Min-SNR Weighting Strategy*: https://arxiv.org/abs/2303.09556
- Song et al., *Consistency Models*: https://arxiv.org/abs/2303.01469
- Chang et al., *MaskGIT*: https://arxiv.org/abs/2202.04200
- Rampasek et al., *Recipe for a General, Powerful, Scalable Graph Transformer*: https://arxiv.org/abs/2205.12454
- Hu et al., *Graph2Plan*: https://arxiv.org/abs/2004.13204
- Shabani et al., *HouseDiffusion*: https://arxiv.org/abs/2211.13287
- Rodriguez Torrado et al., *Bootstrapping Conditional GANs for Video Game Level Generation*: https://arxiv.org/abs/1910.01603
