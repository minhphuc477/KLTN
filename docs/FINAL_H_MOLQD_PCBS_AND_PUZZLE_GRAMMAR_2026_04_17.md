# Final H-MOLQD, P-CBS, And Puzzle Grammar Plan

Last updated: 2026-04-17

## Scope

This note is the current report-ready summary for:

- final end-to-end H-MOLQD pipeline handoff
- ablation and baseline methodology
- `P-CBS` (`Persona-Driven Cognitive Bounded Search`) positioning
- rigid puzzle-grammar anchoring for room generation

## 1. Final End-To-End Pipeline

The production pipeline should be treated as:

1. `MAP-Elites` mission-DAG generation
2. graph-conditioned latent room generation (`VQ-VAE + diffusion`)
3. symbolic repair / discretization
4. grammar-anchor cleanup for puzzle rooms
5. deterministic graph-marker overlay
6. room stitching into the global dungeon
7. global validation:
   - `A*` tile-state oracle
   - graph progression validator
   - soft-lock checker
   - `P-CBS` persona probe

### Required Python handoff contract

The stitched dungeon should only be exported after every room has passed:

- semantic sanitization
- boundary shell enforcement
- structural artifact stripping
- puzzle grammar anchoring
- graph-marker overlay

The current implementation point is the `generate_room -> generate_rooms_for_graph -> stitch/export -> validation` chain in:

- [dungeon_pipeline.py](f:/KLTN/src/pipeline/dungeon_pipeline.py)
- [run_fast_sampler_visual_audit.py](f:/KLTN/scripts/run_fast_sampler_visual_audit.py)
- [run_fixed_graph_multi_seed_audit.py](f:/KLTN/scripts/run_fixed_graph_multi_seed_audit.py)

## 2. Report Metrics

### Mechanical quality

- `Solvability% = solved_dungeons / generated_dungeons`
- `RepairRate = repaired_rooms / total_rooms`
- `OverwriteRate = overwritten_graph_markers / expected_graph_markers`
- `AnchorError = (1 / N) Σ_i d_1(pred_i, target_i)`
  where `d_1` is Manhattan distance
- `GenerationTime = t_export_end - t_export_start`

### Expressive-range metrics

- `Leniency = 1 - (combat_hazard_tiles + mandatory_gate_tiles) / walkable_tiles`
- `Linearity = shortest_path_length / traversable_tiles`
- `Density = occupied_structure_tiles / room_area`
- `PuzzleDensity = puzzle_structure_tiles / room_area`
- `TopologyBranching = (1 / |V|) Σ_v outdeg(v)`

### Human-like validation metrics (`P-CBS`)

- `ConfusionIndex = revisits / unique_tiles_visited`
- `NavigationEntropy = - Σ_a p(a) log2 p(a)`
- `RoomEntropy = - Σ_x p(x) log2 p(x)`
  where `x` is tile/room visit mass
- `CognitiveLoad = memory_usage_ratio × (1 + belief_confidence_variance)`
- `AhaLatency = t_reach_goal - t_first_goal_seen`
- `Replans = number of direction changes`
- `AffordanceReactivations =` number of remembered gates/puzzle anchors reactivated after inventory changes
- `CognitiveGapRate = P(P-CBS fails | hard oracle solves)`

## 3. Ablation Matrix

| ID | Variant | What changes | Main outputs |
|---|---|---|---|
| A1 | `full_hybrid` | full H-MOLQD | reference quality/performance |
| A2 | `no_wfc` | disable symbolic repair/refiner | measure pure neural validity drop |
| A3 | `pure_wfc` | disable latent diffusion, use symbolic-only constructive baseline | measure appearance/control loss |
| A4 | `no_router` | remove hybrid router / topology-conditioned handoff | measure semantic drift |
| A5 | `no_puzzle_grammar` | disable grammar-anchor pocket/frame logic | isolate puzzle readability gain |
| A6 | `no_marker_overlay` | disable deterministic graph overlay | isolate learned semantic placement |
| A7 | `pcbs_off` | replace persona validation with only hard oracle | isolate behavioral validation contribution |

### Core report fields per ablation

- `Solvability%`
- `GoalGauntletValid%`
- `SoftlockSafe%`
- `GenerationTimeSec`
- `OverwriteRate`
- `PreOverlayAnchorError`
- `P-CBS confusion / entropy / load`

## 4. Baseline Panel

### Internal baselines already supported

- `RANDOM`
- `ES`
- `GA`
- `MAP_ELITES`
- `FULL`

### Room/layout baselines

- `pure_wfc`
- `pure_diffusion` (`no_wfc`)
- `symbolic_only_repair`

### Paper-facing comparison dimensions

| Family | Solvability | Expressive Range | Controllability | Compute |
|---|---|---|---|---|
| BSP / constructive | high | low-medium | medium | low |
| pure WFC | medium-high | medium | medium | medium |
| pure evolutionary | medium | high | medium | high |
| H-MOLQD | target high | high | high | high |

Matched-budget external comparison should continue using:

- `results/matched_budget_topology_v1`
- `results/pcg_benchmark_alignment_v2`

## 5. P-CBS Definition

`P-CBS` is a bounded-rational dungeon validator with persona-conditioned utility:

`U(a) = αΔg(a) + βI(a) - γR(a) - ρV(a) + λL(a) - κC(a) - ψQ(a) - ωW(a) + ηF(a) + ξA(a) - ζM(a)`

where:

- `Δg(a)`: goal progress
- `I(a)`: information gain / curiosity
- `R(a)`: immediate hazard risk
- `V(a)`: revisit penalty
- `L(a)`: loot value
- `C(a)`: combat aversion
- `Q(a)`: local puzzle/branching complexity penalty
- `W(a)`: conditional uncertainty penalty
- `F(a)`: frontier bonus
- `A(a)`: affordance-resumption bonus toward remembered, now-solvable progression gates
- `M(a)`: affordance-forgetting penalty when conditional structure is no longer well supported by memory

Bounded memory is modeled through:

- finite working-memory capacity
- exponential salience decay
- partial observability / confidence-weighted belief map
- progression-affordance memory that is reactivated by inventory changes

### Personas

| Persona | Goal | Loot | Risk | Complexity | Behavior |
|---|---|---|---|---|---|
| `speedrunner` | very high | very low | low | low | shortest safe finish |
| `explorer` | low | medium | low | low | frontier seeking / full coverage |
| `completionist` | low | very high | low-medium | low | exhaustive item collection |
| `cautious` | medium | low | high | medium-high | avoids threats and uncertainty |
| `forgetful` | medium-low | low | medium | medium | backtracks under decay |
| `novice` | low-medium | low | very high | very high | avoids combat and complex rooms |
| `balanced` | medium | medium | medium-low | medium-low | mixed policy |

## 6. Novelty Claim

### Defensible claim

`P-CBS` is a repo-novel persona-driven bounded-rational validator for dungeon PCG. It is not the hard mechanical oracle; it complements the hard oracle by measuring how different player archetypes experience the same dungeon under bounded memory, partial observation, asymmetric utility over loot/risk/complexity, and remembered progression affordances that can be reactivated after inventory changes.

### Non-defensible claim

Do not claim `P-CBS` is a new universally recognized general search family replacing `A*`, `D* Lite`, or MAPF `CBS`. The novelty is the bounded persona-validation formulation and its integration into PCG evaluation and generation guidance.

## 7. Thesis Abstract Snippet

We introduce `Persona-Driven Cognitive Bounded Search (P-CBS)`, a bounded-rational automated playtesting algorithm for procedural dungeon validation. Unlike classical search procedures that optimize only mechanical reachability, P-CBS augments state-space navigation with persona-conditioned utility, finite working memory, belief uncertainty, and explicit penalties for revisitation, combat exposure, and local puzzle complexity. This produces synthetic validators that approximate distinct player styles such as speedrunners, explorers, cautious players, completionists, and novices while remaining computationally analyzable.

Within H-MOLQD, P-CBS is not used as the sole correctness oracle. Instead, it complements an exact mechanical validation stack based on `A*`, graph progression, and soft-lock checks. This separation allows the system to distinguish mechanical solvability from experiential readability. The resulting evaluation protocol captures both whether a dungeon can be solved and how it is likely to be experienced by different bounded-rational players, which is a more informative criterion for neuro-symbolic PCG than perfect-play search alone.

## 8. Puzzle Grammar Fix

The current production fix is a local grammar-anchor layer applied inside room generation before final marker overlay:

- detect puzzle room from DAG role / edge semantics
- classify gate family:
  - `switch`
  - `toggle`
  - `bombable`
  - `item_unlock`
  - `key`
  - `combat`
- build a route skeleton
- reserve traversability path
- force a local anchor pocket:
  - pocket floor around the interaction anchor
  - block frame with one intentional opening
- place additional scaffold segments only if quality beats baseline

This logic now lives in:

- [dungeon_pipeline.py](f:/KLTN/src/pipeline/dungeon_pipeline.py)

### Intended structural meanings

| Gate family | Forced meaning |
|---|---|
| `switch` / `toggle` | readable switch pocket and blocked gate line |
| `bombable` | visible side bypass / destructible wall pressure |
| `item_unlock` | alcove / slot around item anchor |
| `key` | local key pocket before gate |
| `combat` | tighter encounter-focused obstruction with enemy anchor |

## 9. Current Production Recommendation

As of `2026-04-17`, the safest production branch is still:

- `outputs/zelda_hmolqd_downstream_codebook512_puzzle_subtype_v1`
- room branch: `diffusion`

Recent manual export evidence:

- `protocol_manual_compare_currentcode_v7`
- zero exported room `VOID` leakage
- lower manual-seed `P-CBS` confusion than earlier current-code runs

## 10. Remaining Evidence Gaps

The repo is stronger, but the thesis should still avoid claiming it surpasses prior publications until all of these are complete:

- fresh full multi-seed fixed-graph audit on the latest code path
- matched-budget external baseline tables aligned with the final branch
- rerun full `1-9 x variants x personas` benchmark on the patched `P-CBS`

## References

- [PCGRL, AIIDE 2020](https://arxiv.org/abs/2001.09212)
- [Automated Playtesting with Procedural Personas, 2018](https://arxiv.org/abs/1802.06881)
- [Procedural Personas as Critics for Dungeon Generation](https://antoniosliapis.com/papers/procedural_personas_as_critics_for_dungeon_generation.pdf)
- [Towards Objective Metrics for Procedurally Generated Video Game Levels, 2022](https://arxiv.org/abs/2201.10334)
- [Video Game Level Repair via Mixed Integer Linear Programming, 2020](https://arxiv.org/abs/2010.06627)
- [Guided Game Level Repair via Explainable AI, 2024](https://arxiv.org/abs/2410.23101)
