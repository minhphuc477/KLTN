# Mission Grammar Reference

This document is the code-accurate reference for the mission-graph grammar implemented in `src/generation/grammar.py`.

Last updated: 2026-02-23.

## 1. Purpose

The mission grammar builds a directed mission graph for Zelda-like dungeon progression.

It combines:
- Topology expansion rules (branches, hubs, shortcuts, floors)
- Progression rules (keys/locks, item gates, boss hierarchy)
- Quality/pedagogical rules (tutorial chains, pacing, battery gates)
- Validation and repair passes to keep generated graphs playable

## 2. Core Data Model

### 2.1 MissionGraph

`MissionGraph` contains:
- `nodes: Dict[int, MissionNode]`
- `edges: List[MissionEdge]`
- `generation_stats: Dict[str, Any]` (repair accounting metadata)
- `_adjacency: Dict[int, List[int]]` (traversal index)
- `_key_to_lock: Dict[int, int]` (key-lock mapping)

Key integrity methods:
- `rebuild_adjacency()` rebuilds adjacency from edge list and drops dangling edges.
- `sanitize()` normalizes graph internals after any rewrite.
- `record_repair()` increments repair counters used by benchmarks.

### 2.2 MissionNode fields (important)

- Identity/type: `id`, `node_type`
- Spatial: `position=(row,col,floor)`
- Progression metadata: `key_id`, `required_item`, `item_type`, `switch_id`
- Structural metadata: `is_hub`, `is_secret`, `room_size`, `sector_id`, `virtual_layer`, `is_arena`, `is_big_room`
- Wave-3 quality metadata: `difficulty_rating`, `is_sanctuary`, `drops_resource`, `is_tutorial`, `is_mini_boss`, `tension_value`

### 2.3 MissionEdge fields (important)

- Connectivity: `source`, `target`, `edge_type`
- Gate metadata: `key_required`, `item_required`, `switch_id`
- Extended progression: `requires_key_count`, `token_count`, `token_id`
- Wave-3 quality metadata: `battery_id`, `switches_required`, `path_savings`

## 3. Grammar Symbols

### 3.1 NodeType

`START`, `GOAL`, `KEY`, `LOCK`, `ENEMY`, `PUZZLE`, `ITEM`, `EMPTY`, `SWITCH`, `BIG_KEY`, `BOSS_DOOR`, `BOSS`, `STAIRS_UP`, `STAIRS_DOWN`, `SECRET`, `TOKEN`, `ARENA`, `TREASURE`, `PROTECTION_ITEM`, `MINI_BOSS`, `SCENIC`, `RESOURCE_FARM`, `TUTORIAL_PUZZLE`, `COMBAT_PUZZLE`, `COMPLEX_PUZZLE`.

### 3.2 EdgeType

`PATH`, `LOCKED`, `ONE_WAY`, `HIDDEN`, `SHORTCUT`, `ON_OFF_GATE`, `BOSS_LOCKED`, `ITEM_GATE`, `STATE_BLOCK`, `WARP`, `STAIRS`, `VISUAL_LINK`, `SHUTTER`, `HAZARD`, `MULTI_LOCK`.

## 4. Production Rules (All Rules)

Rule order is the active order in `MissionGrammar.rules`.

1. `StartRule`
- Initializes `START(0) -> GOAL(1)`.

2. `InsertChallengeRule(NodeType.ENEMY)`
3. `InsertChallengeRule(NodeType.PUZZLE)`
- Splits a `PATH` edge and inserts a challenge room.
- Safety rule: only splits `PATH` edges.

4. `InsertLockKeyRule`
- Inserts `KEY` then inserts `LOCK` later on a START->GOAL path with `LOCKED` edge.
- Uses path-aware placement and keeps key-before-lock ordering intent.

5. `BranchRule`
- Adds side branches from low-degree nodes.

6. `MergeRule`
- Adds shortcut/cycle structure.

7. `InsertSwitchRule`
- Adds switch-based progression with state-gated edges.

8. `AddBossGauntlet`
- Inserts `BOSS_DOOR` before goal and places matching `BIG_KEY` in distant branch.

9. `AddItemGateRule`
- Places an `ITEM` node and creates later `ITEM_GATE` requiring that item.

10. `CreateHubRule`
- Builds multi-branch hub structure.

11. `AddStairsRule`
- Adds vertical progression (`STAIRS_UP`/`STAIRS_DOWN`, `STAIRS` edges).

12. `AddSecretRule`
- Adds hidden room/link structure.

13. `AddTeleportRule`
- Adds warp connection.

14. `PruneGraphRule`
- Prunes long linear empty chains.

15. `AddFungibleLockRule`
- Adds key-inventory lock economy via `requires_key_count` on `LOCKED` edges.
- New safety behavior:
  - lock edge is only selected if at least one key is reachable pre-gate,
  - if no valid lock candidate exists, temporary key insertion is rolled back.

16. `FormBigRoomRule`
- Merges room semantics into larger macro rooms.

17. `AddValveRule`
- Adds one-way valves in cycles.

18. `AddForeshadowingRule`
- Adds `VISUAL_LINK` between spatially close but topologically far rooms.

19. `AddCollectionChallengeRule`
- Adds distributed `TOKEN` collection and `MULTI_LOCK` gate.
- New safety behavior:
  - `MULTI_LOCK` candidate requires all tokens reachable pre-gate,
  - if no viable gate exists, inserted token nodes are rolled back.

20. `AddArenaRule`
- Converts a room to arena and applies `SHUTTER` behavior.

21. `AddSectorRule`
- Creates thematic sectors (zone tagging).

22. `AddEntangledBranchesRule`
- Creates cross-branch switch dependency (`STATE_BLOCK`).

23. `AddHazardGateRule`
- Adds hazardous path with optional protection item.

24. `SplitRoomRule`
- Creates virtual layered room node with `ONE_WAY` or `STAIRS` linkage.

25. `AddSkillChainRule`
- Converts post-item successors into tutorial progression:
  - `TUTORIAL_PUZZLE`
  - `COMBAT_PUZZLE`
  - `COMPLEX_PUZZLE`

26. `AddPacingBreakerRule`
- Inserts `SCENIC` sanctuary after high-tension chains.
- Safety rule: rewires only `PATH` outgoing edges.

27. `AddResourceLoopRule`
- Creates/assigns `RESOURCE_FARM` near item gates.
- Safety rule: never retags protected anchor/boss-door nodes.

28. `AddGatekeeperRule`
- Converts item predecessor to `MINI_BOSS` and seals reward path.
- Safety rule: never retags `START`/`GOAL`/`BOSS_DOOR`.

29. `AddMultiLockRule`
- Builds battery pattern (`battery_id`, `switches_required`) on `STATE_BLOCK` edges.

30. `AddItemShortcutRule`
- Creates item-gated return shortcut from distant item location.

31. `PruneDeadEndRule`
- Removes non-valuable dead-end rooms.
- Safety rule: prune only if remaining graph stays connected.

## 5. Validation Pipeline

`validate_all_constraints()` runs:
- `validate_anchor_nodes()`
- `validate_lock_key_ordering()`
- `validate_progression_constraints()`
- `validate_skill_chains()`
- `validate_battery_reachability()`
- `validate_resource_loops()`

### 5.3 Hard vs Soft constraints

Hard constraints (must hold for valid progression):
- anchor nodes (`START`, `GOAL`) exist and are reachable,
- lock/key and gate prerequisites are satisfiable pre-gate,
- battery/resource reachability for required dependencies.

Soft constraints (quality/tuning objectives):
- pacing smoothness and tension-wave shape,
- pedagogical ordering quality in skill chains,
- branch/loop richness and expressive topology variety.

### 5.1 Lock/Progression checks

- Lock node checks: each `LOCK`/`BOSS_DOOR` key must exist and be reachable pre-lock.
- Edge gate checks:
  - `LOCKED`/`BOSS_LOCKED` keys
  - Fungible key counts (`requires_key_count`)
  - `ITEM_GATE` provider reachability
  - `MULTI_LOCK` token counts
  - `STATE_BLOCK` switch reachability

### 5.2 Wave-3 checks

- Tutorial chain ordering checks nearest pedagogical successors only.
- Battery validation ensures required switches are pre-gate reachable.
- Resource-loop validation ensures farms are pre-gate reachable for matching gates.

## 6. Repair Pipeline

Generation includes a convergence loop with bounded repair rounds.

Repair passes:
- `_ensure_anchor_nodes()`
  - Restores a single `START` and single `GOAL` if any rule retagged them.
- `_fix_lock_key_ordering()`
  - Demotes invalid lock nodes/edges to preserve solvability.
- `_repair_progression_constraints()`
  - Relaxes unsatisfied gate requirements (key/item/token/switch).
  - Repairs malformed lock edges with missing key metadata.
- `_repair_wave3_constraints()`
  - Normalizes pedagogical difficulty ordering.
  - Trims invalid battery requirements.
  - Demotes unreachable resource farms.

Repair counters recorded in `generation_stats`:
- `lock_key_repairs`
- `progression_repairs`
- `wave3_repairs`
- `repair_rounds`
- `total_repairs`
- `repair_applied`

## 7. Generation Algorithm Summary

`MissionGrammar.generate()`:

1. Start graph with `StartRule`.
2. Iteratively sample weighted applicable rules until room budget / iteration cap.
3. Run convergence repairs and validations.
4. Apply final layout positioning (`_layout_graph`).
5. Return mission graph.

## 8. Invariants (Current Best Practice)

The implementation now enforces these invariants:
- Exactly one `START` and one `GOAL` at output.
- Challenge insertion only splits `PATH` edges (no gate metadata loss).
- Dead-end pruning never mutates graph when connectivity would break.
- Protected nodes (`START`/`GOAL`/`BOSS_DOOR`) are not retagged by gatekeeper/resource rules.
- Validation and repairs are iterative, not one-shot.

## 9. Related Files

- Main implementation: `src/generation/grammar.py`
- Evolutionary executor integration: `src/generation/evolutionary_director.py`
- Tests:
  - `tests/test_wave3_pedagogical_rules.py`
  - `tests/test_advanced_rules_integration.py`
  - `tests/test_topology_generation_fixes.py`

## 10. Research Context

The grammar and constraints follow lock-and-key mission-graph patterns common in procedural dungeon research:
- Dormans & Bakkes mission/space generation framework.
- VGDL/VGLC Zelda level structure conventions.
- Zelda-style gating and progression pedagogy.

References:
- Dormans, J. et al. *Generating Missions and Spaces for Adaptable Play Experiences* (2011). DOI: `10.1109/TCIAIG.2010.2067210`
- Summerville et al. *The Video Game Level Corpus* (2016): https://arxiv.org/abs/1606.07487
- VGLC repository: https://github.com/TheVGLC/TheVGLC
