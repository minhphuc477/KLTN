# Stateful Puzzle Architecture Audit And Immediate Fixes

Last updated: 2026-04-09

Scope:

- canonical config: [`configs/zelda_hmolqd.yaml`](../configs/zelda_hmolqd.yaml)
- puzzle/runtime implementation: [`src/pipeline/dungeon_pipeline.py`](../src/pipeline/dungeon_pipeline.py)
- topology conditioning: [`src/pipeline/room_topology_conditioning.py`](../src/pipeline/room_topology_conditioning.py)
- current topology-anchor policy version: `2026-04-09.semantic_anchor_v7_stateful_puzzle_edge_semantics`

This document is the research-backed audit for the current Zelda generation
stack, focused on the remaining weakness around puzzle rooms and semantic
room logic. It also records the immediate code changes applied in this pass.

The current model remains a hybrid neural-symbolic stack. That is still the
correct choice for this repository. The small dataset, fixed grid size, and
mission-graph constraints make fully neural end-to-end room semantics too
fragile today. The correct direction is to make the hybrid contract better:

- topology should carry richer, more directional puzzle intent
- topology conditioning should encode that intent more faithfully
- runtime room scaffolds should reflect the specific edge semantics
- symbolic repair should be configurable and measurable, not hidden

## Step 1. Deep Research And Literature Review

### Most relevant sources

| Topic | Source | Venue | Relevance to this repo |
|---|---|---|---|
| Discrete latent bottlenecks | van den Oord et al., *Neural Discrete Representation Learning* | NeurIPS 2017 | Supports VQ-VAE as the right latent bottleneck for discrete tile layouts. |
| Diffusion foundations | Ho et al., *Denoising Diffusion Probabilistic Models* | NeurIPS 2020 | Base teacher formulation. |
| Fast deterministic sampling | Song et al., *Denoising Diffusion Implicit Models* | ICLR 2021 | Supports the DDIM-style teacher sampler already used. |
| Latent diffusion | Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models* | CVPR 2022 | Supports latent-space generation instead of raw tile diffusion. |
| Parallel masked generation | Chang et al., *MaskGIT* | CVPR 2022 | Closest high-impact analogue for the masked-room branch. |
| Graph-to-layout generation | Hu et al., *Graph2Plan* | CVPR 2020 | Strong support for graph-first structured generation. |
| Graph-conditioned diffusion | Shabani et al., *HouseDiffusion* | CVPR 2023 | Strong support for constrained graph-conditioned geometry generation. |
| Layout-conditioned diffusion | Inoue et al., *LayoutDM* | arXiv 2023 | Supports pushing layout constraints earlier into generation instead of only post-hoc repair. |
| Graph transformers | Rampasek et al., *Recipe for a General, Powerful, Scalable Graph Transformer* | NeurIPS 2022 workshop-era graph literature | Supports richer graph encoders than shallow message passing alone. |
| Spatial modulation | Park et al., *Semantic Image Synthesis with Spatially-Adaptive Normalization* | CVPR 2019 | Supports SPADE-style topology conditioning. |
| RL for constructive level generation | Khalifa et al., *PCGRL: Procedural Content Generation via Reinforcement Learning* | AAAI 2020 | Supports using explicit constraints/rewards for puzzle structure instead of unconstrained emergence. |
| Hybrid Zelda dungeon generation | Gutierrez and Schrum, *Generative Adversarial Network Rooms in Generative Graph Grammar Dungeons for The Legend of Zelda* | IEEE CoG 2020 | Supports the graph-first / room-second factorization. |
| Mission and space grammars | Dormans and Bakkes, *Generating Missions and Spaces for Adaptable Play Experiences* | IEEE TCIAIG 2011 | Strong support for explicit mission progression graphs. |
| Locked-door mission generation | Viana and dos Santos, *Procedural generation of dungeons’ maps and locked-door missions through an evolutionary algorithm validated with players* | Expert Systems With Applications 2021 | Strong support for explicit lock/key density, progression constraints, and player-validated mission structure. |

### Key findings from the literature

1. `Graph-first control is the right bias`.
   Graph2Plan, HouseDiffusion, and Zelda graph+room work all support the same
   design pattern: decide high-level structure first, then realize local
   geometry. This directly supports keeping Block I explicit instead of moving
   everything into a single room model.

2. `Small-data procedural generation benefits from explicit constraints`.
   PCGRL, Dormans-style mission graphs, and the Expert Systems With
   Applications locked-door paper all support explicit lock/key and progression
   structure. This is especially important when the domain is small and the
   design goal is playability rather than raw visual diversity.

3. `Semantic room logic should be encoded before, not after, generation where possible`.
   HouseDiffusion and LayoutDM are not Zelda papers, but they strongly support
   constraint-aware generation over pure post-hoc repair. That is why the
   improvements in this pass target topology conditioning and puzzle templates,
   not only repair.

4. `Few-step students only help if the teacher and structure are already strong`.
   DDIM, LDM, and consistency-style distillation literature all imply the same
   thing: fast samplers cannot compensate for weak teacher semantics. This is
   why fast-sampler quality remains a secondary concern behind Block I quality
   and topology-aware room semantics.

### State-of-the-art comparison context

There is no direct, apples-to-apples SOTA benchmark for `graph-conditioned
Zelda dungeon generation with room-level semantics and symbolic repair`.
Therefore:

- exact cross-paper metric comparison is limited
- architecture decisions must be justified by adjacent structured-generation
  literature plus repo-specific evidence

This document therefore distinguishes:

- `citation-backed claims` for general architectural direction
- `inference-based repo judgments` for the exact implementation

## Step 2. Assumptions Validation

### Core architectural assumptions

| Assumption | Where it appears | Assessment |
|---|---|---|
| The graph should own progression semantics | Block I, marker overlay, puzzle scaffold | Valid and well supported by graph/layout literature. |
| The dataset is too small for fully neural room semantics | training and runtime design | Valid. Repo evidence and PCG literature both support extra structure under limited data. |
| Room geometry and mission semantics can be separated safely | VQ-VAE + diffusion + overlay | Mostly valid, but only if semantic overlays remain measurable and limited. |
| Fixed room size `16x11` is acceptable | almost every room model and renderer path | Valid for this Zelda corpus, fragile for transfer to other domains. |
| Symbolic repair can safely rescue small structural errors | runtime generation | Partly valid. Useful as a safety net, but high repair load hides model weakness. |
| Masked-room can act as a production branch | earlier experiments | Not validated. Current repo evidence says no. It still needs teacher fallback in bad cases. |
| Puzzle semantics can be approximated by generic “puzzle blocks” | old scaffold path | Invalid. This was one of the main weaknesses fixed in this pass. |

### Hardcoded assumptions identified in code

These are real implementation assumptions that should be configurable if they
affect reproducibility or behavior:

| Assumption | Current status |
|---|---|
| symbolic repair max attempts | Promoted in this pass |
| symbolic repair margin | Promoted in this pass |
| symbolic repair adjacency threshold | Promoted in this pass |
| switch pocket depth | Promoted in this pass |
| bombable bypass offset | Promoted in this pass |
| item-slot depth | Promoted in this pass |
| toggle corridor offset | Promoted in this pass |
| key pocket depth | Promoted in this pass |
| room size `16x11` | Still fixed in code, not suitable for generalization |
| default start/goal coordinates | Already configurable |
| semantic marker decode biases | Already configurable |
| puzzle scaffold density/budget | Already configurable |

## Step 3. Logical Audit Of The Architecture

### Good logic that should stay

1. `Block I -> Block II/III/IV -> repair/overlay` is still the right overall
   factorization.
2. The system should keep `mission correctness` and `room geometry` partially
   separate.
3. Symbolic repair should remain a last-resort validity layer.

### Logical problems found

1. `Puzzle semantics were under-specified at runtime`.
   Before this pass, a `switch_locked`, `bombable`, `item_gate`, and `key_locked` room
   could all end up looking like generic transverse bars. That is logically
   inconsistent with the graph semantics.

2. `Room type -> puzzle role inference was incomplete`.
   `switch` and `complex_puzzle` node types did not always imply
   `has_puzzle=True`. That made topology conditioning and anchor placement less
   faithful than intended.

3. `Symbolic repair had hidden hyperparameters`.
   Repair was important to correctness, but its key thresholds were not exposed
   to config. That made ablation and reproducibility weaker than the architecture
   description implied.

4. `Masked-room remains conceptually over-claimed`.
   The codebase still supports it, but the current behavior shows it should be
   treated as guarded/experimental, not a peer production branch.

## Step 4. Theory vs Implementation Consistency Check

### Consistent parts

- The code does implement graph-first generation.
- The topology map really does influence the room generator.
- Semantic constrained decoding is implemented.
- Deterministic graph marker overlay is implemented.

### Gaps between theory and implementation

1. `“Topology-aware puzzle rooms” was weaker in practice than on paper`.
   Before this pass, puzzle scaffolds were only archetype-aware, not directly
   tied to edge semantics.

2. `“Hybrid but tunable” was incomplete`.
   Repair and stateful puzzle offsets were still partly hardcoded.

3. `“Semantic anchors” were better described than enforced for stateful puzzles`.
   The previous scaffold used puzzle families, but not the most appropriate
   local anchor for each gate family.

### Hyperparameters that were present in the effective method but missing from config

Promoted in this pass:

- `generation.symbolic_max_repair_attempts`
- `generation.symbolic_repair_margin`
- `generation.symbolic_adjacency_threshold`
- `generation.puzzle_room_switch_pocket_depth`
- `generation.puzzle_room_resource_bypass_offset`
- `generation.puzzle_room_key_pocket_depth`
- `generation.puzzle_room_item_slot_depth`
- `generation.puzzle_room_toggle_corridor_offset`

## Step 5. Gap And Bug Analysis

### Gaps fixed now

1. `Stateful gate-family puzzle templates`
   Implemented in [`src/pipeline/dungeon_pipeline.py`](../src/pipeline/dungeon_pipeline.py):
   - `switch_locked -> switch pocket + blocked gate line`
   - `bombable -> side-pocket + destructible bypass`
   - `item_gate / item_locked -> item-slot alcove + centered unlock lane`
   - `on_off_gate / state_block -> toggle-state corridor`
   - `key_locked -> local key-before-gate template when a key anchor exists`

2. `Puzzle role inference bug`
   `switch`, `tutorial_puzzle`, `combat_puzzle`, and `complex_puzzle` now map
   to `has_puzzle=True` more faithfully.

3. `Hidden repair parameters`
   Promoted into YAML/runtime config.

### Remaining gaps

1. `Room-level puzzles are still constructive, not fully stateful simulations`.
   They read much better now, but they still do not simulate richer Zelda
   mechanics like time-varying switch states or multi-step causal chains.

2. `Topology still under-specifies puzzle subtypes in many generated graphs`.
   The room logic can only be as specific as the graph semantics allow.

3. `Teacher quality remains a ceiling for fast sampling`.
   Not fixed in this pass.

### New config fields added now

| Parameter | Type | Default | Valid range | Why |
|---|---|---:|---|---|
| `generation.puzzle_room_switch_pocket_depth` | int | 3 | `1..6` | Controls how far the switch pocket sits from the gate line. |
| `generation.puzzle_room_resource_bypass_offset` | int | 2 | `1..5` | Controls how far bombable bypasses deviate from the center line. |
| `generation.puzzle_room_key_pocket_depth` | int | 3 | `1..6` | Controls the size/depth of the key-before-gate alcove when a local key exists. |
| `generation.puzzle_room_item_slot_depth` | int | 3 | `1..6` | Controls the depth of the local item-slot alcove for item-gate unlock rooms. |
| `generation.puzzle_room_toggle_corridor_offset` | int | 2 | `1..5` | Controls the half-width of toggle-state corridors for on/off and state-block rooms. |
| `generation.symbolic_max_repair_attempts` | int | 5 | `>=1` | Makes repair aggressiveness reproducible. |
| `generation.symbolic_repair_margin` | int | 2 | `>=0` | Makes repair locality tunable. |
| `generation.symbolic_adjacency_threshold` | float | 0.01 | `>=0` | Makes repair connectivity sensitivity tunable. |

## Step 6. Redundancy And Unnecessary Work Analysis

### Still useful

- deterministic overlay
- symbolic repair
- graph-first topology generation

### Potentially redundant or overused

1. `Masked-room as a peer runtime branch`.
   Current evidence suggests it adds maintenance complexity without equal value.
   It remains useful as an ablation/research branch.

2. `High repair load`.
   If repairs remain consistently large, then repair is compensating for weak
   generation instead of just enforcing final safety.

3. `Teacher fallback on poor students`.
   Necessary now, but computationally redundant once the student becomes
   stronger.

## Step 7. Computational Complexity Analysis

### Parameter counts already measured in this repo

From the canonical repo measurements:

- VQ-VAE: about `17.62M` parameters
- diffusion teacher stack: about `70.08M` parameters
- masked-room branch: about `12.45M` parameters

### Complexity by block

| Block | Complexity sketch | Notes |
|---|---|---|
| VQ-VAE | `O(HW * C * k^2 * L)` | Cheap relative to diffusion; fixed small room size keeps it manageable. |
| Graph encoder | usually `O(|E|d^2)` to `O(|V|^2 d)` depending on attention mode | Graph size is small here, so not the main bottleneck. |
| Diffusion teacher | `O(T * U-Net(H,W,C))` | Dominant runtime cost. |
| Fast sampler | `O(T_fast * U-Net(H,W,C))` | Cheaper but bounded by teacher quality. |
| Masked-room | iterative masked-token decoding | Lighter than diffusion, but quality is still weaker. |
| Repair/scaffold | roughly linear in grid size | Negligible compared with neural generation, but can hide semantic/model weakness. |

### Complexity-relevant fixed parameters

Now configurable or already exposed:

- `diffusion.model_channels`
- `diffusion.unet_channel_mult`
- `diffusion.unet_num_heads`
- `masked_room.model_channels`
- `masked_room.hidden_dim`
- `fast_sampler.num_inference_steps`
- `generation.puzzle_room_block_budget`
- `generation.symbolic_max_repair_attempts`

### Practical tradeoff

The new stateful puzzle templates increase runtime logic slightly but do not
materially change neural FLOPs. This is a favorable tradeoff because the added
cost is in tiny room-grid constructive logic, not in the teacher network.

## Step 8. Hyperparameter Sensitivity Analysis

### Most fragile hyperparameters

| Parameter | Sensitivity | Notes |
|---|---|---|
| `diffusion.cfg_scale` | High | Too high harms room quality and teacher-student alignment. |
| `diffusion.min_snr_gamma` | High | Affects training stability. |
| `generation.semantic_marker_logit_bias` | High | Too low weakens semantic control; too high acts like a hard constraint. |
| `generation.semantic_marker_suppression_bias` | High | Too low leaves stray semantics; too high can over-prune. |
| `generation.puzzle_room_branch_density` | Medium-high | Too low yields empty puzzle rooms; too high yields clutter. |
| `generation.puzzle_room_block_budget` | Medium-high | Too low weakens readability; too high overfills tiny rooms. |
| `generation.puzzle_room_switch_pocket_depth` | Medium | Too small collapses the switch alcove into the gate line; too large wastes space. |
| `generation.puzzle_room_resource_bypass_offset` | Medium | Too small reads like a normal gate; too large makes the bypass disconnected. |
| `generation.puzzle_room_key_pocket_depth` | Medium | Too small makes the key alcove unreadable; too large overcommits scarce room area. |
| `generation.symbolic_max_repair_attempts` | Medium | Too low leaves broken rooms; too high hides neural weakness. |

### Safe operating guidance

| Parameter | Default | Safe range | Source |
|---|---:|---|---|
| `generation.puzzle_room_switch_pocket_depth` | 3 | `2..4` | inference-based repo tuning |
| `generation.puzzle_room_resource_bypass_offset` | 2 | `1..3` | inference-based repo tuning |
| `generation.puzzle_room_key_pocket_depth` | 3 | `2..4` | inference-based repo tuning |
| `generation.puzzle_room_branch_density` | 0.75 | `0.45..0.9` | inference-based repo tuning |
| `generation.puzzle_room_block_budget` | 28 | `18..34` | inference-based repo tuning |
| `generation.symbolic_max_repair_attempts` | 5 | `3..7` | inference-based repo tuning |

## Step 9. Failure Modes And Edge Cases

1. `Topology says puzzle, room lacks space`.
   Very small or over-constrained rooms can still produce cramped puzzle logic.

2. `Key-gate room without a local key`.
   The new code handles this by falling back to a generic gate template instead
   of forcing a fake key alcove. This is intentional.

3. `Bombable/item-gate without an item anchor`.
   The new code falls back to the puzzle anchor. Better than before, but still a
   weaker semantic story than a true item-owned room.

4. `Masked-room drift`.
   Still a live failure mode.

5. `Block I distribution mismatch`.
   If Block I emits graphs that are too lenient or too sparse, room-level
   improvements cannot fully compensate.

## Step 10. Scalability And Generalization Boundaries

### Lower bound

The current hybrid design is viable because:

- room grids are tiny
- graphs are small
- semantics are explicit

### Upper boundary

The current design will become fragile if:

- room size increases substantially without architectural changes
- graph size grows beyond the current small-dungeon regime
- the dataset shifts to another game with materially different room semantics

### Generalization ceiling

This system is not yet a general “layout diffusion for arbitrary dungeon
domains.” It is best understood as a Zelda-specific structured generator.

## Step 11. Comparison Against State Of The Art Baselines

### Where it is strong

- more controllable than an unconstrained room generator
- more mission-faithful than a room-only generative baseline
- more practical than a purely neural end-to-end approach on the current data scale

### Where it is behind

- not fully end-to-end
- teacher is still computationally heavy
- masked-room quality is not yet competitive enough to replace diffusion
- richer stateful puzzle logic is still constructive, not learned

### Overall judgment

The architecture is `meaningfully novel as a repo-specific hybrid system`, but
it is `not SOTA in the broader generative-layout sense`. It is closer to a
careful, domain-specific engineering synthesis of graph grammars, topology-aware
conditioning, diffusion, and symbolic repair.

## Step 12. Bias And Ethical Risk Analysis

There is limited classical demographic bias exposure here because the domain is
game maps, not human identity data. The relevant risks are instead:

- gameplay unfairness across player skill levels
- hidden difficulty spikes from bad topology or over-repair
- overfitting to one Zelda corpus style and calling that “general dungeon design”

The graph-first design helps here because difficulty, gating, and progression
can be audited explicitly.

## Step 13. Evidence-Based Decision Summary

### Citation-backed decisions

- Keep the graph-first factorization.
- Keep topology-aware generation and semantic anchors explicit.
- Keep repair configurable and measured.
- Prefer stateful puzzle templates over generic random obstacle bars.

### Inference-based but strongly supported by repo evidence

- Treat masked-room as experimental.
- Keep the hybrid semantic path in production.
- Use stateful gate-family scaffolds as the next improvement step instead of
  moving to purely neural puzzle semantics immediately.

## Step 14. Recommended Ablation Study

No source code changes required for the following.

### A. Disable stateful templates, keep old generic gate logic

Expected outcome:

- lower puzzle readability
- more rooms that look like random block bars

Command:

```powershell
python main.py topology-compare-manual `
  --run-dir outputs\zelda_hmolqd_semantic_anchor_retrain_v1 `
  --output-dir outputs\ablation_generic_gate_templates_v1 `
  --seed 20260409 `
  --puzzle-room-switch-pocket-depth 1 `
  --puzzle-room-resource-bypass-offset 1 `
  --puzzle-room-key-pocket-depth 1
```

### B. Stronger switch pocket

Expected outcome:

- clearer switch-side logic
- risk of overusing room area

```powershell
python main.py topology-compare-manual `
  --run-dir outputs\zelda_hmolqd_semantic_anchor_retrain_v1 `
  --output-dir outputs\ablation_switch_pocket_depth_v1 `
  --seed 20260409 `
  --puzzle-room-switch-pocket-depth 4
```

### C. Wider item/bomb bypass

Expected outcome:

- stronger side-route readability
- risk of disjoint pathing on tiny rooms

```powershell
python main.py topology-compare-manual `
  --run-dir outputs\zelda_hmolqd_semantic_anchor_retrain_v1 `
  --output-dir outputs\ablation_resource_bypass_v1 `
  --seed 20260409 `
  --puzzle-room-resource-bypass-offset 3
```

### D. Disable symbolic repair to expose upstream weakness

Expected outcome:

- lower validity
- clearer signal of what the generator itself is really doing

```powershell
python main.py topology-compare-manual `
  --run-dir outputs\zelda_hmolqd_semantic_anchor_retrain_v1 `
  --output-dir outputs\ablation_no_repair_v1 `
  --seed 20260409 `
  --no-apply-repair
```

## Step 15. Priority Ranking

### Critical

1. Keep improving Block I distribution quality.
2. Keep improving topology-to-room semantic faithfulness.

### High

3. Retrain diffusion / fast-sampler / masked-room on the corrected stateful
   topology conditioning path.
4. Keep masked-room behind guardrails until it improves materially.

### Medium

5. Expose and track repair/scaffold hyperparameters.
6. Expand puzzle subtypes in Block I so room logic can specialize further.

### Low

7. Purely neural semantic placement as a future research branch.

### Reproducibility risks

- hidden repair hyperparameters: fixed in this pass
- hard-coded puzzle pocket/bypass offsets: fixed in this pass
- fixed room size and tile vocabulary: still structural assumptions

## Step 16. Immediate Implementation Applied In This Pass

### Code changes

Implemented in:

- [`src/pipeline/dungeon_pipeline.py`](../src/pipeline/dungeon_pipeline.py)
- [`src/config_system.py`](../src/config_system.py)
- [`configs/zelda_hmolqd.yaml`](../configs/zelda_hmolqd.yaml)
- [`scripts/run_fast_sampler_visual_audit.py`](../scripts/run_fast_sampler_visual_audit.py)
- [`tests/test_architecture_audit_fixes.py`](../tests/test_architecture_audit_fixes.py)
- [`tests/test_config_system.py`](../tests/test_config_system.py)

### What was added

1. `Stateful puzzle templates tied directly to edge semantics`
   - `switch_locked -> switch pocket + blocked gate line`
   - `bombable -> side-pocket + destructible bypass`
   - `item_gate / item_locked -> item-slot alcove + centered unlock lane`
   - `on_off_gate / state_block -> toggle-state corridor`
   - `key_locked -> local key-before-gate room template when a key anchor exists`

2. `Gate-family-aware route templates`
   The route skeleton now depends on both archetype and semantic gate family.

3. `Anchor-aware local pockets`
   - switch rooms prefer the puzzle anchor
   - bombable rooms prefer the puzzle anchor
   - item-unlock rooms prefer the item anchor, else the puzzle anchor
   - toggle rooms prefer the puzzle anchor as the control point
   - key-gate rooms only use a local key pocket when a key anchor exists

4. `Puzzle role inference bugfix`
   Room types like `switch` and `complex_puzzle` now correctly imply
   `has_puzzle=True`.

5. `Config promotion`
   New puzzle and repair knobs are now in YAML and runtime override paths.

### Repo evidence after this pass

Lightweight manual-node puzzle export:

- [`outputs/puzzle_scaffold_manual_nodes_v5/summary.json`](../outputs/puzzle_scaffold_manual_nodes_v5/summary.json)

This export now clearly distinguishes:

- `switch_gate`
- `toggle_gate`
- `bombable_gate`
- `item_gate`
- `key_gate_no_local_key`
- `key_gate_local_key`
- richer manual puzzle nodes like `COMPLEX_PUZZLE` and `COMBAT_PUZZLE`

### Validation status

Focused regression suites passing after the final patch set:

- `tests/test_architecture_audit_fixes.py`
- `tests/test_config_system.py`
- `tests/test_zelda_loader_graph_conditioning.py`

And the broader room-generation regressions also passed during this pass:

- `tests/test_neural_pipeline.py`
- `tests/test_fast_sampler_integration.py`

## References

1. van den Oord et al., *Neural Discrete Representation Learning*, NeurIPS 2017. https://arxiv.org/abs/1711.00937
2. Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS 2020. https://arxiv.org/abs/2006.11239
3. Song et al., *Denoising Diffusion Implicit Models*, ICLR 2021. https://arxiv.org/abs/2010.02502
4. Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models*, CVPR 2022. https://arxiv.org/abs/2112.10752
5. Chang et al., *MaskGIT*, CVPR 2022. https://arxiv.org/abs/2202.04200
6. Hu et al., *Graph2Plan*, CVPR 2020. https://arxiv.org/abs/2004.13204
7. Shabani et al., *HouseDiffusion*, CVPR 2023. https://arxiv.org/abs/2211.13287
8. Inoue et al., *LayoutDM*, 2023. https://arxiv.org/abs/2303.08137
9. Rampasek et al., *Recipe for a General, Powerful, Scalable Graph Transformer*, 2022. https://arxiv.org/abs/2205.12454
10. Park et al., *Semantic Image Synthesis with Spatially-Adaptive Normalization*, CVPR 2019. https://arxiv.org/abs/1903.07291
11. Khalifa et al., *PCGRL: Procedural Content Generation via Reinforcement Learning*, AAAI 2020. https://ojs.aaai.org/index.php/AIIDE/article/view/7416
12. Gutierrez and Schrum, *Generative Adversarial Network Rooms in Generative Graph Grammar Dungeons for The Legend of Zelda*, IEEE CoG 2020. https://arxiv.org/abs/2001.05065
13. Dormans and Bakkes, *Generating Missions and Spaces for Adaptable Play Experiences*, IEEE TCIAIG 2011. https://repository.tudelft.nl/file/File_ba6582ba-7f55-4b55-93f7-9f278f3e2d94
14. Viana and dos Santos, *Procedural generation of dungeons’ maps and locked-door missions through an evolutionary algorithm validated with players*, Expert Systems with Applications 2021. https://www.sciencedirect.com/science/article/pii/S0957417421000909
