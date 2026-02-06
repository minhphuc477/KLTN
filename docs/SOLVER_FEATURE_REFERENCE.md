# Zelda Dungeon Solver - Feature Reference

## Overview

The `StateSpaceAStar` solver in `src/simulation/validator.py` is a sophisticated A* pathfinder that operates on game state space, not just positions. This allows finding solutions that require proper sequencing of item collection, key usage, and door traversal.

---

## Core Classes

### 1. GameState

Represents the complete state of the game at a point in time.

| Field | Type | Description |
|-------|------|-------------|
| `position` | `Tuple[int, int]` | Current (row, col) position |
| `keys` | `int` | Number of small keys available |
| `has_bomb` | `bool` | Whether player has bombs |
| `has_boss_key` | `bool` | Whether player has the boss key |
| `has_item` | `bool` | Whether player has KEY_ITEM (ladder/raft) |
| `opened_doors` | `Set[Tuple[int, int]]` | Doors that have been opened |
| `collected_items` | `Set[Tuple[int, int]]` | Items that have been collected |
| `pushed_blocks` | `Set[Tuple[Tuple, Tuple]]` | Blocks that have been pushed (from→to) |
| `current_floor` | `int` | Multi-floor dungeon support |

### 2. ZeldaLogicEnv

Environment wrapper that holds the dungeon grid and graph.

| Property | Description |
|----------|-------------|
| `semantic_grid` | 2D numpy array of tile IDs |
| `graph` | NetworkX graph of room connections |
| `room_positions` | Dict mapping room coords to grid offsets |
| `room_to_node` | Dict mapping room coords to graph node IDs |
| `node_to_room` | Dict mapping graph node IDs to room coords |
| `start_pos` | Starting position (row, col) |
| `goal_pos` | Goal/Triforce position (row, col) |

### 3. StateSpaceAStar

The main A* solver class.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `env` | `ZeldaLogicEnv` | required | Environment to solve |
| `timeout` | `int` | 200000 | Max states to explore |
| `heuristic_mode` | `str` | "balanced" | Heuristic calculation mode |
| `priority_options` | `dict` | None | Priority queue options |

---

## Tile Types (18 Total)

### Walkable Tiles (Always Passable)

| Tile | ID | Description |
|------|----|--------------| 
| `FLOOR` | 1 | Basic walkable floor |
| `DOOR_OPEN` | 10 | Open passage (no lock) |
| `DOOR_SOFT` | 15 | One-way door (passable in one direction) |
| `START` | 21 | Starting position |
| `TRIFORCE` | 22 | Goal/Triforce piece |
| `ENEMY` | 20 | Monster (fought or avoided) |
| `BOSS` | 23 | Boss enemy (must be fought) |
| `PUZZLE` | 43 | Puzzle element (interact to solve) |
| `STAIR` | 42 | Stairs/ladder/warp point |
| `ELEMENT_FLOOR` | 41 | Element with floor underneath |
| `KEY_SMALL` | 30 | Small key pickup (+1 key) |
| `KEY_BOSS` | 31 | Boss key pickup (permanent) |
| `KEY_ITEM` | 32 | Key item pickup (enables water crossing) |
| `ITEM_MINOR` | 33 | Minor item (bombs in VGLC) |

### Blocking Tiles (Impassable)

| Tile | ID | Description |
|------|----|--------------| 
| `VOID` | 0 | Empty space (outside map) |
| `WALL` | 2 | Solid wall |

### Conditional Tiles (Require Items)

| Tile | ID | Requirement | Description |
|------|----|--------------|--------------| 
| `DOOR_LOCKED` | 11 | 1 small key (consumed) | Key-locked door |
| `DOOR_BOMB` | 12 | Bomb (not consumed) | Bombable wall |
| `DOOR_BOSS` | 14 | Boss key (permanent) | Boss door |
| `DOOR_PUZZLE` | 13 | None (simplified) | Puzzle/switch door |

### Special Tiles

| Tile | ID | Requirement | Description |
|------|----|--------------|--------------| 
| `BLOCK` | 3 | Empty space behind | Pushable block |
| `ELEMENT` | 40 | KEY_ITEM (ladder) | Water/lava hazard |

---

## Graph Edge Types (8 Total)

The dungeon graph encodes room-to-room connections with specific traversal requirements:

| Edge Type | VGLC Code | Requirement | Description |
|-----------|-----------|-------------|-------------|
| `open` | ` ` (empty) | None | Normal open passage |
| `soft_locked` | `l` | One direction only | One-way door |
| `key_locked` | `k` | Small key (consumed) | Key-locked passage |
| `boss_locked` | `K` | Boss key (permanent) | Boss key required |
| `bombable` | `b` | Bomb | Bombable wall passage |
| `stair` | `s` | None | Stair/warp connection |
| `item_locked` | `I` | KEY_ITEM | Key item required |
| `switch` | `S` | None (simplified) | Puzzle-activated door |

---

## Heuristic Modes

### 1. Balanced (Default)

Standard pathfinding with inventory-aware penalties:

- **Manhattan distance** to goal as baseline
- **Graph-based distance** when available (tighter bound)
- **Key penalty**: +10 per locked door when keys insufficient
- **Boss key penalty**: +20 when boss doors exist without boss key
- **Bomb penalty**: +15 when bomb doors exist without bombs
- **Ladder penalty**: +15 when water tiles exist without KEY_ITEM

```python
solver = StateSpaceAStar(env, heuristic_mode="balanced")
```

### 2. Speedrunner

Optimized for fastest path, ignoring optional items:

- 30% reduced door penalties (×0.7)
- 10% reduced overall heuristic (×0.9)
- No penalty for uncollected items

```python
solver = StateSpaceAStar(env, heuristic_mode="speedrunner")
```

### 3. Completionist

Penalizes leaving items uncollected:

- Standard door penalties
- +2 penalty per remaining pickup item

```python
solver = StateSpaceAStar(env, heuristic_mode="completionist")
```

---

## Priority Queue Options

Configure via `priority_options` dict:

```python
options = {
    'tie_break': True,       # Use state hash for tie-breaking
    'key_boost': True,       # Slight priority boost for key pickups
    'enable_ara': True,      # Enable weighted heuristic (WA*)
    'ara_weight': 1.5,       # Weight multiplier for heuristic
    'allow_diagonals': True  # Enable 8-directional movement
}
solver = StateSpaceAStar(env, priority_options=options)
```

### Options Explained

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `tie_break` | bool | False | Use state hash to break f-score ties |
| `key_boost` | bool | False | Small priority boost (-0.01) for key pickups |
| `enable_ara` | bool | False | Enable weighted A* (faster, suboptimal) |
| `ara_weight` | float | 1.0 | Heuristic weight (>1 = faster, less optimal) |
| `allow_diagonals` | bool | False | Enable 8-directional movement |

---

## Movement System

### Cardinal Movement (4-directional)

Default movement: UP, DOWN, LEFT, RIGHT

- Cost: 1.0 per move
- Always available

### Diagonal Movement (8-directional)

When `allow_diagonals=True`:

- Additional moves: UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT
- Cost: √2 ≈ 1.414 per diagonal move
- **30× speedup** on large maps

### Corner-Cutting Prevention

Diagonal moves are blocked if:
- Adjacent orthogonal tiles are in `BLOCKING_IDS` (VOID, WALL)
- Adjacent orthogonal tiles are in `CONDITIONAL_IDS` (locked doors)

This prevents "cutting corners" through walls.

### Teleportation/Warping

Players can teleport via graph connections when:

1. Standing on a **STAIR** tile (immediate warp)
2. Standing on a **DOOR** tile with graph edge to non-adjacent room

Teleportation respects edge type requirements (keys, bombs, etc.)

---

## Block Pushing (Zelda Mechanic)

Blocks (`BLOCK` tiles) can be pushed if:

1. Space behind block (in push direction) is walkable
2. Block hasn't been pushed there already

Push mechanics:
- Player moves INTO block position
- Block moves to position behind it
- Push is tracked in `GameState.pushed_blocks`

---

## Inventory System

### Small Keys
- Collected from `KEY_SMALL` tiles
- Consumed when opening `DOOR_LOCKED`
- Tracked as integer count

### Boss Key
- Collected from `KEY_BOSS` tiles
- Permanent (not consumed)
- Required for `DOOR_BOSS`

### Bombs
- Collected from `KEY_ITEM` or `ITEM_MINOR` tiles
- Not consumed (unlimited use in solver)
- Required for `DOOR_BOMB` and `bombable` edges

### KEY_ITEM (Ladder/Raft)
- Collected from `KEY_ITEM` tiles
- Permanent (not consumed)
- Required for crossing `ELEMENT` (water) tiles
- Required for `item_locked` edges

---

## Caching Systems

### 1. Stair Destination Cache
- Key: Current position
- Value: List of stair destinations
- Populated lazily on first access

### 2. Virtual Transition Cache
- Key: Position
- Value: BFS results through virtual nodes
- Enables traversal of hidden passages

### 3. Door/Element Position Caches
- `_locked_doors_cache`: All DOOR_LOCKED positions
- `_boss_doors_cache`: All DOOR_BOSS positions
- `_bomb_doors_cache`: All DOOR_BOMB positions
- `_element_tiles_cache`: All ELEMENT positions
- **Performance**: Avoids O(width×height) grid scans in heuristic

### 4. Graph Distance Cache
- `min_locked_needed_node`: Min locked doors from each node to goal
- Precomputed via Dijkstra at initialization

---

## State Dominance Pruning

A state A dominates state B if:
1. Same position
2. A has ≥ keys as B
3. A has all items B has
4. A has opened all doors B has opened

When A dominates B, B is pruned (will never lead to better solution).

---

## API Reference

### StateSpaceAStar.solve()

```python
def solve(self) -> Tuple[bool, List[Tuple[int, int]], int]:
    """
    Find a solution path using A* on state space.
    
    Returns:
        success: Whether a solution was found
        path: List of (row, col) positions visited
        states_explored: Number of states explored
    """
```

### Example Usage

```python
from src.simulation.validator import StateSpaceAStar, ZeldaLogicEnv

# Create environment
env = ZeldaLogicEnv(
    semantic_grid=grid,
    graph=graph,
    room_positions=room_positions,
    room_to_node=room_to_node,
    node_to_room=node_to_room
)

# Create solver with options
solver = StateSpaceAStar(
    env,
    timeout=200000,
    heuristic_mode="balanced",
    priority_options={'allow_diagonals': False}
)

# Solve
success, path, states = solver.solve()

if success:
    print(f"Found path with {len(path)} steps, explored {states} states")
else:
    print(f"No solution found after {states} states")
```

---

## Known Limitations

1. **Bombs are unlimited**: Once acquired, bombs are never consumed (simplified model)
2. **Soft-lock direction not enforced**: One-way doors treated as bidirectional in tile logic
3. **Pushed blocks excluded from hash**: Prevents state explosion but may miss some block puzzle solutions
4. **Puzzle tiles simplified**: Always passable (no actual puzzle solving)

---

## Performance Tips

1. **Enable diagonals** for 30× speedup on large maps:
   ```python
   priority_options={'allow_diagonals': True}
   ```

2. **Use speedrunner mode** for faster heuristic:
   ```python
   heuristic_mode="speedrunner"
   ```

3. **Enable ARA*** for faster (suboptimal) solutions:
   ```python
   priority_options={'enable_ara': True, 'ara_weight': 2.0}
   ```

4. **Reduce timeout** for quick validation:
   ```python
   timeout=10000
   ```

---

## Version History

- **v1.0**: Initial A* implementation
- **v2.0**: Added graph-based room traversal
- **v3.0**: Added state dominance pruning
- **v4.0**: Added virtual node traversal
- **v5.0**: Added teleportation from door tiles (Feb 2026)
- **v5.1**: Added BOSS/PUZZLE tile handling, item_locked/switch edges, cached heuristic positions (Feb 2026)
