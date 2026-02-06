# Zelda Dungeon Solver — Feature Documentation

> **File**: `src/simulation/validator.py`  
> **Purpose**: Automated playtesting and validation for Zelda-style dungeon levels  
> **Version**: TIER 2 Enhanced (with multi-floor support, bitset optimization, and graph-guided validation)

---

## Table of Contents

1. [Overview](#overview)
2. [Core Classes](#core-classes)
3. [Public Methods Reference](#public-methods-reference)
4. [Configuration Options](#configuration-options)
5. [Tile Types](#tile-types)
6. [Edge Types](#edge-types)
7. [Inventory System](#inventory-system)
8. [Heuristic Modes](#heuristic-modes)
9. [Caching Mechanisms](#caching-mechanisms)
10. [Advanced Features](#advanced-features)
11. [Usage Examples](#usage-examples)

---

## Overview

The Zelda Dungeon Solver is a comprehensive validation system that provides:

| Module | Purpose |
|--------|---------|
| `ZeldaLogicEnv` | State machine simulator for dungeon logic |
| `StateSpaceAStar` | A* pathfinder with inventory state tracking |
| `SanityChecker` | Pre-validation structural checks |
| `MetricsEngine` | Solvability, reachability, and diversity metrics |
| `DiversityEvaluator` | Mode collapse detection for batch generation |
| `ZeldaValidator` | Main orchestrator for single/batch validation |
| `GraphGuidedValidator` | Graph-topology-based validation for incomplete dungeons |

---

## Core Classes

### 1. `GameState`

**Purpose**: Represents the complete game state at any point during play.

```python
@dataclass
class GameState:
    position: Tuple[int, int]           # Current (row, col) position
    keys: int = 0                       # Number of small keys held
    has_bomb: bool = False              # Whether player has bombs
    has_boss_key: bool = False          # Whether player has boss key
    has_item: bool = False              # Whether player has key item (ladder/raft)
    opened_doors: Set[Tuple[int, int]]  # Positions of opened doors
    collected_items: Set[Tuple[int, int]]  # Positions of collected items
    pushed_blocks: Set[Tuple[...]]      # Block push history (from_pos, to_pos)
    current_floor: int = 0              # Multi-floor dungeon support
```

**Key Design Note**: `pushed_blocks` is **NOT included in hash** to prevent state explosion (117 blocks = 2^117 states). Blocks are handled as transient modifications.

---

### 2. `GameStateBitset` (Optimized)

**Purpose**: Memory-optimized GameState using bitsets for 5-10× faster hashing.

**Bit Allocation (64-bit integer)**:
| Bits | Purpose | Max Count |
|------|---------|-----------|
| 0-29 | Doors | 30 doors |
| 30-49 | Items | 20 items |
| 50-63 | Blocks | 14 blocks |

**Performance Improvement**:
- Hash time: 5-10× faster
- Memory: 50% reduction
- A* search: 10-20% faster

---

### 3. `ZeldaLogicEnv`

**Purpose**: Discrete state simulator for Zelda dungeon logic (headless environment).

**Constructor Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `semantic_grid` | `np.ndarray` | 2D array of semantic tile IDs |
| `render_mode` | `bool` | Enable Pygame rendering (default: False) |
| `graph` | `NetworkX` | Graph for stair/warp connections |
| `room_to_node` | `Dict` | Mapping of room positions to graph nodes |
| `room_positions` | `Dict` | Mapping of room positions to grid offsets |
| `node_to_room` | `Dict` | Reverse mapping (includes virtual nodes) |
| `solver_options` | `SolverOptions` | Configurable starting inventory |

---

### 4. `StateSpaceAStar`

**Purpose**: A* pathfinder operating on game state space (not just positions).

**Why State-Space**: Enables solving puzzles requiring:
- Picking up keys before opening doors
- Getting bombs before bombing walls
- Proper sequencing of item collection

---

## Public Methods Reference

### ZeldaLogicEnv Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `reset()` | `() → GameState` | Reset environment to initial state |
| `step()` | `(action: int) → Tuple[GameState, float, bool, Dict]` | Execute one action (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT) |
| `get_valid_actions()` | `() → List[int]` | Get list of valid actions from current state |
| `render()` | `() → None` | Render current state (requires Pygame) |
| `close()` | `() → None` | Clean up resources |

### StateSpaceAStar Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `solve()` | `() → Tuple[bool, List[Tuple[int,int]], int]` | Find solution path using A* |
| `solve_with_diagnostics()` | `() → Tuple[bool, List[...], SolverDiagnostics]` | Solve with detailed statistics |

**`solve()` Returns**:
- `success: bool` — Whether solution was found
- `path: List[Tuple[int, int]]` — Positions visited
- `states_explored: int` — Number of states explored

### ZeldaValidator Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `validate_single()` | `(grid, render, persona_mode) → ValidationResult` | Validate one map |
| `validate_batch()` | `(grids, verbose, persona_mode) → BatchValidationResult` | Validate multiple maps |
| `validate_batch_multi_persona()` | `(grids, personas) → Dict[str, BatchValidationResult]` | Validate with multiple personas |
| `check_soft_locks()` | `(grid, sample_count) → Tuple[bool, List[str]]` | Detect soft-lock traps |

### GraphGuidedValidator Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `validate_dungeon_with_graph()` | `(dungeon_data) → GraphValidationResult` | Validate using graph topology |
| `validate_with_edge_types()` | `(dungeon_data, inventory) → GraphValidationResult` | Validate with inventory constraints |

---

## Configuration Options

### SolverOptions

**Purpose**: Configurable settings for the solver and starting inventory.

```python
@dataclass
class SolverOptions:
    start_keys: int = 0          # Starting key count
    start_bombs: int = 1         # Starting bomb count (Zelda-style default)
    start_boss_key: bool = False # Start with boss key
    start_item: bool = False     # Start with ladder/raft
    timeout: int = 200000        # Max states to explore
    allow_diagonals: bool = False # Enable 8-directional movement
    heuristic_mode: str = "balanced"  # Solver persona
```

**Factory Methods**:

```python
# For bomb-heavy dungeons
SolverOptions.for_level("bomb_heavy")  # start_bombs=3

# For key-heavy dungeons
SolverOptions.for_level("key_heavy")   # start_keys=1, start_bombs=1

# For speedrun testing
SolverOptions.for_level("speedrun")    # allow_diagonals=True, mode="speedrunner"
```

### Priority Options (StateSpaceAStar)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `tie_break` | `bool` | `False` | Use locked-door count as tie-breaker |
| `key_boost` | `bool` | `False` | Slight priority boost for key pickups |
| `enable_ara` | `bool` | `False` | Enable ARA* (Anytime Repairing A*) |
| `ara_weight` | `float` | `1.0` | Heuristic weight for ARA* |
| `allow_diagonals` | `bool` | `False` | Enable 8-directional movement |

**Example**:
```python
solver = StateSpaceAStar(
    env, 
    priority_options={
        'tie_break': True,
        'key_boost': True,
        'allow_diagonals': True
    }
)
```

---

## Tile Types

### Semantic Palette

All tile types are defined in `src/core/definitions.py`:

| ID | Name | Description | Walkable? |
|----|------|-------------|-----------|
| 0 | `VOID` | Empty space outside map | ❌ Blocking |
| 1 | `FLOOR` | Walkable floor tile | ✅ Walkable |
| 2 | `WALL` | Solid wall | ❌ Blocking |
| 3 | `BLOCK` | Pushable block | 🔸 Conditional (push) |
| 10 | `DOOR_OPEN` | Open passage | ✅ Walkable |
| 11 | `DOOR_LOCKED` | Key-locked door | 🔑 Requires key |
| 12 | `DOOR_BOMB` | Bombable wall | 💣 Requires bomb |
| 13 | `DOOR_PUZZLE` | Puzzle/switch door | ✅ Auto-passable |
| 14 | `DOOR_BOSS` | Boss key door | 🔑 Requires boss key |
| 15 | `DOOR_SOFT` | One-way door | ✅ Walkable |
| 20 | `ENEMY` | Monster | ✅ Walkable (with cost) |
| 21 | `START` | Starting position | ✅ Walkable |
| 22 | `TRIFORCE` | Goal/win condition | ✅ Walkable |
| 23 | `BOSS` | Boss enemy | ✅ Walkable |
| 30 | `KEY_SMALL` | Consumable key | ✅ Pickup |
| 31 | `KEY_BOSS` | Permanent boss key | ✅ Pickup |
| 32 | `KEY_ITEM` | Key item (ladder/raft/bomb) | ✅ Pickup |
| 33 | `ITEM_MINOR` | Minor item (grants bombs) | ✅ Pickup |
| 40 | `ELEMENT` | Hazard (water/lava) | 🌊 Requires ladder |
| 41 | `ELEMENT_FLOOR` | Element with floor | ✅ Walkable |
| 42 | `STAIR` | Stair/warp point | ✅ Transition |

### Tile Categories

```python
WALKABLE_IDS = {FLOOR, DOOR_OPEN, DOOR_SOFT, START, TRIFORCE, 
                KEY_SMALL, KEY_BOSS, KEY_ITEM, ITEM_MINOR, 
                ELEMENT_FLOOR, STAIR, ENEMY}

BLOCKING_IDS = {VOID, WALL}

CONDITIONAL_IDS = {DOOR_LOCKED, DOOR_BOMB, DOOR_BOSS, DOOR_PUZZLE}

PUSHABLE_IDS = {BLOCK}

WATER_IDS = {ELEMENT}  # Requires KEY_ITEM (ladder) to cross

PICKUP_IDS = {KEY_SMALL, KEY_BOSS, KEY_ITEM, ITEM_MINOR}

TRANSITION_IDS = {STAIR, DOOR_OPEN, DOOR_SOFT}  # Allow teleportation
```

### Movement Costs

| Tile Type | Cost | Reason |
|-----------|------|--------|
| `FLOOR` | 1.0 | Baseline |
| `ENEMY` | 10.0 | Combat time/health loss |
| `DOOR` (unlocked) | 2.0 | Opening time |
| `DOOR_BOMB` | 3.0 | Bombing time |
| `DOOR_PUZZLE` | 2.5 | Puzzle solving |
| `PICKUP` | 1.5 | Collection delay |
| `BLOCKING` | ∞ | Impassable |

---

## Edge Types

Graph edges define room-to-room connections:

| Edge Code | Type | Requirement | Consumed? |
|-----------|------|-------------|-----------|
| `''` (empty) | `open` | None | No |
| `k` | `key_locked` | Small key | Yes (1 key) |
| `K` | `boss_locked` | Boss key | No |
| `b` | `bombable` | Bomb | No (unlimited bombs) |
| `l` | `soft_locked` | None | No (one-way) |
| `s` | `stair` | None | No (warp) |
| `S` | `switch` | Puzzle completion | No |
| `I` | `item_locked` | Key item | No |

### Edge Restriction Priority

When combining edges (multi-hop traversal), the most restrictive is kept:

```
boss > bomb > locked > puzzle > open
```

---

## Inventory System

### Inventory Items

| Item | Field | Effect | Pickup Tile |
|------|-------|--------|-------------|
| Small Key | `keys: int` | Opens `DOOR_LOCKED` (consumed) | `KEY_SMALL` |
| Boss Key | `has_boss_key: bool` | Opens `DOOR_BOSS` (permanent) | `KEY_BOSS` |
| Bomb | `has_bomb: bool` | Opens `DOOR_BOMB` (unlimited) | `KEY_ITEM`, `ITEM_MINOR` |
| Key Item | `has_item: bool` | Cross `ELEMENT` tiles (ladder/raft) | `KEY_ITEM` |

### Pickup Rewards

```python
KEY_SMALL:  keys += 1,      reward = 5.0
KEY_BOSS:   has_boss_key = True, reward = 15.0
KEY_ITEM:   has_item = True, has_bomb = True, reward = 10.0
ITEM_MINOR: has_bomb = True, reward = 1.0
```

### State Persistence

- **Opened doors**: Tracked in `opened_doors` set, persist across visits
- **Collected items**: Tracked in `collected_items` set, not re-collectable
- **Pushed blocks**: Tracked in `pushed_blocks` set, blocks stay pushed

---

## Heuristic Modes

Three persona-based heuristic profiles:

### 1. Balanced (default)

**Purpose**: Standard pathfinding with balanced priorities.

```python
solver = StateSpaceAStar(env, heuristic_mode="balanced")
```

**Behavior**:
- Manhattan distance to goal
- Standard door penalties
- No item collection bias

### 2. Speedrunner

**Purpose**: Optimize for fastest completion.

```python
solver = StateSpaceAStar(env, heuristic_mode="speedrunner")
```

**Behavior**:
- 10% reduced heuristic weight (`h *= 0.9`)
- 30% reduced door penalties (`door_scale = 0.7`)
- Ignores optional pickups
- Best for: Route optimization, time trials

### 3. Completionist

**Purpose**: Explore all areas, collect all items.

```python
solver = StateSpaceAStar(env, heuristic_mode="completionist")
```

**Behavior**:
- Penalty for uncollected items (`+2 per remaining pickup`)
- Encourages full exploration
- Best for: 100% completion, coverage testing

### Multi-Persona Validation

```python
validator = ZeldaValidator()
results = validator.validate_batch_multi_persona(
    grids,
    personas=["speedrunner", "balanced", "completionist"]
)
# Returns: Dict[str, BatchValidationResult]
```

---

## Caching Mechanisms

### 1. Stair Destination Cache

**Location**: `StateSpaceAStar._stair_dest_cache`

**Purpose**: Cache stair tile → destination room mappings.

```python
# Structure: Dict[Tuple[int,int], List[Tuple[int,int]]]
# Key: Current position
# Value: List of valid stair destinations
```

**Why**: Graph traversal for stair connections is expensive. Caching provides O(1) lookup after first query.

### 2. Virtual Transition Cache

**Location**: `StateSpaceAStar._virtual_transition_cache`

**Purpose**: Cache virtual node traversal results.

```python
# Structure: Dict[Tuple[int,int], List[Tuple[dest, cost, edge_type]]]
```

**Why**: Virtual nodes (hidden passages) require graph BFS. Cache reuses computed paths.

### 3. Best State at Position (Dominance)

**Location**: `StateSpaceAStar._best_at_pos`, `_best_g_at_pos`

**Purpose**: Track Pareto-optimal states for dominance pruning.

```python
# _best_at_pos: Dict[position, GameState] - Best inventory at each position
# _best_g_at_pos: Dict[position, float] - Best g-score at each position
```

**Performance**: Reduces search space by 20-40% on multi-key dungeons.

### 4. Locked Door Distance Cache

**Location**: `StateSpaceAStar.min_locked_needed_node`

**Purpose**: Precomputed minimum locked doors from each graph node to goal.

**How**: Dijkstra on graph with edge costs = 1 for locked edges, 0 otherwise.

**Use**: Improves heuristic accuracy, especially for lock-heavy dungeons.

---

## Advanced Features

### 1. State Domination Pruning

**What**: Skip states that are strictly worse than previously visited states.

**Domination Criteria** (all must hold):
- Same position
- Fewer or equal keys
- Fewer or equal items
- Subset of opened doors
- Subset of collected items
- Worse or equal g-score

**Performance**: 20-40% state reduction.

### 2. Diagonal Movement

**Enable**: `priority_options={'allow_diagonals': True}`

**Costs**:
- Cardinal (N/S/E/W): 1.0
- Diagonal: √2 ≈ 1.414

**Corner-Cutting Prevention**: Blocked if either adjacent tile is wall/door.

**Performance**: 30× speedup on large maps (but changes animation behavior).

### 3. Block Pushing (Zelda Mechanic)

**Behavior**:
1. Check if target tile is `BLOCK`
2. Calculate push direction (player → block)
3. Check if destination is walkable and not blocked
4. Track push in `pushed_blocks` set
5. Player moves into block's original position

**Chained Pushes**: Supported via state tracking.

### 4. Multi-Floor Support

**Field**: `GameState.current_floor`

**Purpose**: Track which dungeon floor the player is on.

**Use Case**: Dungeons with multiple levels connected by stairs.

### 5. Soft-Lock Detection

```python
is_safe, traps = validator.check_soft_locks(grid, sample_count=10)
```

**Algorithm**:
1. Solve START → GOAL
2. Sample random walkable positions
3. Test if GOAL is reachable from each position
4. Report positions that cannot reach GOAL

### 6. ARA* (Anytime Repairing A*)

**Enable**: `priority_options={'enable_ara': True, 'ara_weight': 2.0}`

**What**: Weighted A* for faster suboptimal solutions.

**Formula**: `f = g + w × h` where `w > 1` inflates heuristic.

**Use Case**: Quick validation when optimality is not required.

---

## Usage Examples

### Basic Validation

```python
from src.simulation.validator import ZeldaValidator

validator = ZeldaValidator()
result = validator.validate_single(my_grid)

print(f"Solvable: {result.is_solvable}")
print(f"Path length: {result.path_length}")
print(f"Reachability: {result.reachability:.1%}")
```

### Custom Starting Inventory

```python
from src.simulation.validator import ZeldaLogicEnv, StateSpaceAStar, SolverOptions

options = SolverOptions(
    start_keys=2,
    start_bombs=1,
    start_boss_key=True
)

env = ZeldaLogicEnv(grid, solver_options=options)
solver = StateSpaceAStar(env)
success, path, states = solver.solve()
```

### Using Priority Options

```python
solver = StateSpaceAStar(
    env,
    timeout=100000,
    heuristic_mode="speedrunner",
    priority_options={
        'tie_break': True,
        'key_boost': True,
        'allow_diagonals': True,
        'enable_ara': True,
        'ara_weight': 1.5
    }
)
```

### Detailed Diagnostics

```python
success, path, diagnostics = solver.solve_with_diagnostics()

print(diagnostics.summary())
# Output:
# === Solver Diagnostics ===
# Status: SUCCESS
# States Explored: 1,234
# States Pruned (dominated): 567
# Pruning Efficiency: 31.5%
# Max Queue Size: 890
# Time Taken: 45.2ms
# Path Length: 78
# ==========================
```

### Graph-Guided Validation

```python
from src.simulation.validator import GraphGuidedValidator

validator = GraphGuidedValidator()
result = validator.validate_dungeon_with_graph(dungeon_data)

if result.is_solvable:
    print(f"Path through rooms: {result.graph_path}")
else:
    print(f"Missing rooms: {result.missing_rooms}")
    print(f"Error: {result.error_message}")
```

### With Stitched Dungeon

```python
from src.data.zelda_core import ZeldaDungeonAdapter

adapter = ZeldaDungeonAdapter("Data/The Legend of Zelda")
dungeon = adapter.load_dungeon(1, variant=1)
stitched = adapter.stitch_dungeon(dungeon)

env = ZeldaLogicEnv(
    semantic_grid=stitched.global_grid,
    graph=stitched.graph,
    room_positions=stitched.room_positions,
    room_to_node=stitched.room_to_node,
    node_to_room=stitched.node_to_room
)

solver = StateSpaceAStar(env, timeout=200000)
success, path, states = solver.solve()
```

### Batch Validation with Diversity

```python
grids = [generate_dungeon() for _ in range(100)]
results = validator.validate_batch(grids, verbose=True)

print(results.summary())
# === Batch Validation Summary ===
# Total Maps: 100
# Valid Syntax: 98 (98.0%)
# Solvable: 87 (87.0%)
# Avg Reachability: 62.3%
# Avg Path Length: 145.2
# Avg Backtracking: 0.18
# Diversity Score: 0.734
# ================================
```

---

## Data Classes Reference

### ValidationResult

```python
@dataclass
class ValidationResult:
    is_solvable: bool
    is_valid_syntax: bool
    reachability: float
    path_length: int
    backtracking_score: float
    logical_errors: List[str]
    path: List[Tuple[int, int]]
    error_message: str
```

### SolverDiagnostics

```python
@dataclass
class SolverDiagnostics:
    success: bool
    states_explored: int
    states_pruned_dominated: int
    max_queue_size: int
    time_taken_ms: float
    failure_reason: str
    path_length: int
    final_inventory: Dict[str, Any]
```

### GraphValidationResult

```python
@dataclass
class GraphValidationResult:
    is_solvable: bool
    graph_path: List[int]
    subgraph_path: List[int]
    missing_rooms: List[int]
    room_validations: Dict[int, Dict]
    connectivity_score: float
    start_node: int
    triforce_node: int
    error_message: str
```

---

## Performance Guidelines

| Dungeon Size | Recommended Timeout | Expected States | Notes |
|--------------|---------------------|-----------------|-------|
| Small (1-4 rooms) | 10,000 | < 1,000 | Fast |
| Medium (5-9 rooms) | 50,000 | 2,000-10,000 | Standard |
| Large (10+ rooms) | 200,000 | 5,000-50,000 | Enable diagonals |
| Complex (many keys) | 500,000 | 20,000-100,000 | Use ARA* |

**Optimization Tips**:
1. Enable `allow_diagonals` for large maps (30× speedup)
2. Use `heuristic_mode="speedrunner"` when optimality isn't needed
3. Enable `enable_ara` with `ara_weight=2.0` for fast suboptimal solutions
4. Pre-validate with `SanityChecker` to catch obvious failures

---

*Generated from `src/simulation/validator.py` analysis*  
*Last updated: February 2026*
