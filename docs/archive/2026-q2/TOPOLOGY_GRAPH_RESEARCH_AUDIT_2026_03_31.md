# Topology Graph Research Audit

Last updated: 2026-03-31

Scope:

- `src/generation/grammar.py`
- `src/generation/grammar_validators.py`
- `src/pipeline/graph_features.py`
- `src/pipeline/room_topology_conditioning.py`
- `src/pipeline/dungeon_pipeline.py`
- `src/zelda_data/zelda_loader.py`
- `src/config_system.py`
- `configs/zelda_hmolqd.yaml`

## Summary

The topology stack is stronger than a basic mission-graph pipeline. It already models lock-key progression, boss gates, item gates, switch/state blocks, hazards, shortcuts, secrets, vertical transitions, and room-level topology priors. The main weakness is not absence of graph logic; it is compression. Several progression semantics exist in the symbolic mission graph but are flattened away before the learned graph encoder and room-topology maps consume them.

The two most important findings are:

1. The repo contains richer symbolic topology than the learned topology representation actually exposes.
2. The old graph schema drift was real, but it has now been fixed in code by unifying training/config/runtime on a richer `node_feature_dim=14`, `edge_feature_dim=16`, `tpe_dim=8` contract.

## What The Current Topology Graph Already Covers

Code evidence:

- Mission graph nodes support start, goal, key, lock, enemy, puzzle, item, switch, big key, boss door, boss, stairs, secret, token, arena, treasure, protection item, tutorial, mini-boss, scenic, resource-farm, and more in `src/generation/grammar.py`.
- Mission edges support path, locked, one-way, hidden, shortcut, on/off gate, boss locked, item gate, state block, warp, stairs, visual link, shutter, hazard, and multi-lock in `src/generation/grammar.py`.
- Battery patterns and switch reachability are explicitly validated in `src/generation/grammar_validators.py`.
- Edge feature extraction already encodes open, key-locked, bombable, soft-locked, boss-locked, item-locked, stair, switch, hazard, shutter, multi-lock, state-block, and hidden/secret semantics in `src/pipeline/graph_features.py`.
- Room topology conditioning now provides a 50-channel tensor with traversability, start/goal, directional doors, generic directional gated doors, typed directional gate families, and room-role broadcasts in `src/pipeline/room_topology_conditioning.py`.
- Current-room relative graph distance is encoded explicitly in `compute_current_node_distance_features(...)` and used by diffusion/masked-room training.

This is a meaningful topology stack. The missing pieces are mostly about fidelity and completeness of representation.

## Additional Literature Worth Bringing In

### Most useful papers for the next topology pass

1. Dormans and Bakkes, *Generating Missions and Spaces for Adaptable Play Experiences*.
   Link: https://pure.hva.nl/ws/files/149264/453867_Dormans_Bakkes_-_Generating_Missions_and_Spaces_for_Adaptable_Play_Experiences.pdf
   Why it matters: still the clearest mission-first formalization for adventure-game progression graphs.

2. Ying et al., *Do Transformers Really Perform Bad for Graph Representation?* (Graphormer, NeurIPS 2021).
   Link: https://arxiv.org/abs/2106.05234
   Why it matters: shows that shortest-path distance, centrality, and edge-aware structural bias materially improve graph representations.

3. Rampasek et al., *Recipe for a General, Powerful, Scalable Graph Transformer* (GraphGPS, NeurIPS 2022).
   Link: https://arxiv.org/abs/2205.12454
   Why it matters: directly motivates richer positional and structural encodings such as RWSE, LapPE, and relative encodings for scalable graph conditioning.

4. Vignac et al., *DiGress: Discrete Denoising Diffusion for Graph Generation* (ICLR 2023).
   Link: https://openreview.net/forum?id=UaAD-Nu86WX
   Why it matters: relevant if you want a learned topology-graph generator baseline instead of relying only on grammar search.

5. Pereira et al., *Procedural generation of dungeons' maps and locked-door missions through an evolutionary algorithm validated with players* (Expert Systems with Applications, 2021).
   Link: https://www.sciencedirect.com/science/article/pii/S0957417421004504
   Why it matters: domain-specific evidence that locked-door mission structure, linearity, and path redundancy are measurable and player-relevant.

### What those papers imply here

- Dormans and Bakkes supports the mission-then-space decomposition already used here.
- Graphormer suggests your topology encoder should not stop at node labels plus a small TPE vector; relative path structure and centrality matter.
- GraphGPS suggests the repo should treat positional/structural encodings as a first-class design surface, not a fixed `tpe_dim=8` artifact.
- DiGress is relevant if you want to test whether grammar-only topology generation is leaving performance on the table.
- Pereira et al. supports adding explicit topology evaluation metrics such as mission linearity and path redundancy, instead of relying only on solvability and ad hoc structure checks.

## Missing Elements In The Learned Topology Representation

### 1. One-way directionality is modeled symbolically but not encoded explicitly

Status: `Fixed in code for edge features`

Code evidence:

- `MissionEdge` stores `preferred_direction` in `src/generation/grammar.py`.
- The evolutionary export/import path preserves `preferred_direction` in `src/generation/evolutionary_director.py`.
- `encode_edge_feature_vector(...)` in `src/pipeline/graph_features.py` now preserves one-way directionality explicitly in the edge feature vector while still keeping soft-lock compatibility.

Why this matters:

- A one-way shortcut and a shutter room are not equivalent progression constraints.
- Graphormer-style structural bias suggests edge semantics should preserve relation identity when that relation changes traversal behavior.

Remaining research direction:

- Test whether explicit directional channels improve downstream controllability and room-generation fidelity.
- Decide whether one-way direction should also be promoted into the room-topology tensor, not only the graph edge vector.

### 2. Battery and multi-switch dependencies are present in the grammar but mostly invisible to the neural topology encoder

Status: `Partially fixed in code`

Code evidence:

- `battery_id` and `switches_required` exist in `src/generation/grammar.py`.
- `validate_battery_reachability(...)` checks them in `src/generation/grammar_validators.py`.
- `encode_edge_feature_vector(...)` in `src/pipeline/graph_features.py` now preserves switch-cardinality and battery membership in the learned edge vector.
- `build_room_topology_condition_map(...)` reduces all gated semantics to per-direction binary gated channels in `src/pipeline/room_topology_conditioning.py`.

Why this matters:

- A gate controlled by one switch is not the same topology as a gate requiring a battery of switches across multiple rooms.
- The symbolic layer can reason about it; the learned layer largely cannot.

Remaining research direction:

- Add room-level channels for switch/state gates, not just generic `gated_*`.
- Consider hyperedge-style or factor-node modeling if multi-switch dependencies become important.

### 3. Advanced node semantics exist in the grammar but are mostly dropped before graph conditioning

Status: `Partially fixed in code`

Code evidence:

- `MissionNode` contains `required_item`, `is_hub`, `is_secret`, `sector_id`, `virtual_layer`, `is_arena`, `is_big_room`, tutorial flags, resource flags, and more in `src/generation/grammar.py`.
- `extract_node_feature_vector(...)` in `src/pipeline/graph_features.py` now exposes a richer 14-D node schema and includes secret/hub-style structure hints, but it still does not encode every advanced grammar attribute.
- The config is now schema-locked to `dataset.node_feature_dim=14` and `dataset.edge_feature_dim=16` in `src/config_system.py` and `configs/zelda_hmolqd.yaml`.

Why this matters:

- Hubs, secret rooms, sectors, and vertical layers are topology-relevant, not cosmetic.
- The symbolic generator may create these distinctions while the neural stack only sees a reduced proxy.

Research direction:

- Promote hub/secret/sector/layer semantics into the learned graph schema.
- If keeping a compact schema, validate by ablation which symbolic attributes truly help downstream room generation.

### 4. Room topology maps used to collapse too many gate types into a generic gated-door channel

Status: `Fixed in code`

Code evidence:

- `src/core/definitions.py` now centralizes the room-topology contract, including typed directional gate families.
- `build_room_topology_condition_map(...)` still exposes backward-compatible `gated_*` strips, but now also paints typed channels such as `gate_key_*`, `gate_switch_*`, `gate_bomb_*`, `gate_secret_*`, and `gate_hazard_*`.

Why this matters:

- A bombable wall, a boss lock, a soft lock, and a hazard gate imply different room semantics and likely different tile layouts.
- SPADE/additive conditioning can only use what the topology tensor gives it.

Research direction:

- The minimum useful semantic split has now been implemented.
- The next question is whether the full 50-channel contract is better than a compressed typed variant under equal training budget.

### 5. TPE is useful, but still narrow for this topology regime

Code evidence:

- `compute_tpe_features(...)` in `src/pipeline/graph_features.py` uses only distance from start, distance to goal, degree, on-main-path flag, key flag, lock flag, difficulty, and key-id presence.

Why this matters:

- This misses articulation points, dominator structure, cycle membership, branch depth, backtracking burden, and redundancy.
- Pereira et al. explicitly treats linearity and path redundancy as meaningful dungeon properties.
- Graphormer and GraphGPS both support richer structural bias than a fixed 8D summary.

Research direction:

- Add articulation-point, betweenness, branch-depth, and redundancy features.
- Consider pairwise shortest-path bias in the graph encoder instead of relying only on node-wise TPE.

### 6. The topology contract is inconsistent across config, training, and runtime

Status: `Fixed in code`

Code evidence:

- `dataset.node_feature_dim=14`, `dataset.edge_feature_dim=16`, `dataset.tpe_dim=8` are now schema-locked in `src/config_system.py` and `configs/zelda_hmolqd.yaml`.
- Training loaders now emit explicit `edge_features` and richer `node_features`.
- Diffusion and masked-room training now pass schema widths into the condition encoder instead of silently defaulting to legacy `6/8`.
- `src/pipeline/dungeon_pipeline.py` now defaults the runtime condition encoder to the same shared graph schema contract.

Why this matters:

- Topology experiments become path-dependent.
- A result observed in training may not match the feature schema used at inference or evaluation time.

Result:

- The schema is unified end-to-end, so topology experiments are no longer path-dependent on the old `6/8` vs `12/14` split.

## Implementation Closure In This Pass

The topology audit also exposed a reproducibility gap in Block I itself: topology generation defaults were still buried inside `prepare_dungeon_generation(...)` as local constants. This is now fixed in code.

Implemented:

- `src/config_system.py` and `configs/zelda_hmolqd.yaml` now expose explicit `topology.*` fields for target curve, room budget, evolutionary budget, mutation/crossover, rule space, search strategy, QD knobs, and constraint/repair behavior.
- `src/pipeline/dungeon_pipeline.py` now stores those defaults on the pipeline and consumes them when `generate_topology=True`.
- `src/pipeline/dungeon_pipeline.py` now exports `topology_generation_kwargs_from_resolved_config(...)` so generation experiments can be reproduced from resolved YAML without hand-copying kwargs.
- `src/generation/evolutionary_director.py` now threads `max_lock_key_rules` through to `GraphGrammarExecutor`, closing a previously hidden generation-time assumption.

## Missing Evaluation Dimensions

The topology graph side needs more than solvability.

Recommended metrics to add:

- mission linearity
- path redundancy
- articulation count
- key-before-lock satisfaction rate by pair
- switch-before-gate satisfaction rate by battery
- average backtracking burden from start to goal
- branch utility rate
- secret-content discoverability rate

This is directly supported by the dungeon-topology literature and would make topology changes measurable instead of aesthetic.

## Highest-Priority Next Research Questions

1. Does explicit directionality for one-way edges improve conditioned room generation and topology fidelity?
2. Does separating gate semantics in the room topology map improve diffusion alignment more than the current generic `gated_*` channels?
3. Do hub/sector/layer features improve graph conditioning, or are they symbolic-only detail with no downstream gain?
4. Is the grammar-based topology generator still competitive against a learned graph generator baseline such as DiGress for this domain size?
5. Which structural features best predict downstream room-generation correctness: current-node distance, richer TPE, or pairwise structural bias?
6. Does the current `14/16` contract capture enough topology, or do room-topology tensors and graph structure still under-represent stateful progression?

## Recommended Ablations

1. Compare the current `edge_feature_dim=16` encoding against an even richer relation-expanded encoding that also lifts typed gate semantics into the room-topology tensor.
2. Compare current 50-channel room topology maps against a compressed typed-topology tensor.
3. Compare current 8D TPE against an expanded structural encoding with articulation and redundancy features.
4. Compare grammar-only topology generation against a learned graph generator baseline.
5. Compare the current unified `14/16` schema against a larger ablation that also lifts sector/layer and typed gate channels.

## Bottom Line

Nothing is fundamentally missing at the level of "can this repo represent topology at all". It can. The real missing element is representational faithfulness: several mission-graph semantics are generated, validated, and preserved symbolically, but then compressed away before the neural blocks learn from them.

If you want the next topology-focused upgrade to matter, the best direction is not adding more exotic grammar rules first. It is making the learned topology representation faithful to the symbolic topology you already have.
