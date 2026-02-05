# Solver Consolidation Plan

## Overview

The KLTN codebase has **4 redundant solver implementations** that have evolved organically during development. This document outlines a plan to consolidate them into a single authoritative solver while preserving the unique capabilities of each.

---

## Current Solvers

### 1. `DungeonSolver` (Data/zelda_core.py:3208)
**Purpose**: Built-in solver for `ZeldaDungeonAdapter` - validates dungeon solvability using state-space pathfinding.

**Capabilities**:
- Multiple validation modes: STRICT, REALISTIC, FULL
- Inventory tracking (keys, bombs)
- Graph-aware with room-level pathfinding
- Fallback to grid-based BFS

**Limitations**:
- Tightly coupled to `zelda_core.py`
- Limited heuristic options
- No detailed traversal metadata

---

### 2. `GraphSolver` (graph_solver.py:91)
**Purpose**: Graph-based solver using DOT topology for room-level pathfinding.

**Capabilities**:
- State-space search with room positions + keys
- Locked door identification from edge labels
- Key room identification from graph attributes
- Room-level path output

**Limitations**:
- Requires graph to be well-formed
- No tile-level pathfinding within rooms
- Visualization-focused (includes COLORS dict)

---

### 3. `MazeSolver` (maze_solver.py:68)
**Purpose**: Grid-based A* pathfinding on walkable floor tiles.

**Capabilities**:
- Pure tile-level A* navigation
- Respects walkable/non-walkable tile semantics
- Simple and fast for grid traversal

**Limitations**:
- No inventory tracking
- No key/lock mechanics
- No graph awareness
- Cannot solve dungeons with locked doors

---

### 4. `StateSpaceAStar` (simulation/validator.py:963) ⭐ **RECOMMENDED**
**Purpose**: Full-featured state-space A* solver with comprehensive game logic.

**Capabilities**:
- **Full inventory tracking** (keys, bombs, boss keys, items)
- **Heuristic modes**: balanced, aggressive, relaxed
- **Priority options**: tie-breaking, key boost, ARA* support
- **Diagonal movement** option for performance
- **Detailed traversal metrics** (states explored, path, items collected)
- **Integration with ZeldaLogicEnv** for accurate game state simulation
- **Stair/warp handling** via graph edge types
- **Timeout protection** for large dungeons

**Limitations**:
- Requires `ZeldaLogicEnv` to be set up
- Heavier initialization overhead

---

## Recommendation

### Keep: `StateSpaceAStar` as the **Authoritative Ground-Truth Validator**

**Rationale**:
1. **Most comprehensive**: Handles all game mechanics (keys, bombs, soft-locks, stairs)
2. **Proven reliable**: Used in GUI runner for animated solving
3. **Configurable**: Multiple heuristic modes and priority options
4. **Well-tested**: Has extensive integration with the simulation environment
5. **Documented**: Clear docstrings and parameter descriptions

### Deprecation Plan

| Solver | Action | Timeline |
|--------|--------|----------|
| `StateSpaceAStar` | **KEEP** - Authoritative solver | N/A |
| `DungeonSolver` | **DEPRECATE** - Replace calls with `StateSpaceAStar` | Phase 1 |
| `GraphSolver` | **DEPRECATE** - Extract useful graph-matching logic | Phase 2 |
| `MazeSolver` | **DEPRECATE** - Merge tile-level A* into utility | Phase 3 |

---

## Implementation Steps

### Phase 1: Deprecate `DungeonSolver` (zelda_core.py)

1. **Add deprecation warning** to `DungeonSolver.solve()`:
   ```python
   import warnings
   warnings.warn(
       "DungeonSolver is deprecated. Use simulation.validator.StateSpaceAStar instead.",
       DeprecationWarning,
       stacklevel=2
   )
   ```

2. **Update callers** in:
   - `gui_runner.py` - Already uses `StateSpaceAStar`
   - `tests/test_dungeon_solvability.py` - Switch to `StateSpaceAStar`
   - `src/simulation/map_elites.py` - Switch to `StateSpaceAStar`

3. **Keep as facade** (optional): Wrap `StateSpaceAStar` internally for backward compatibility.

### Phase 2: Deprecate `GraphSolver` (graph_solver.py)

1. **Extract useful logic**: Move room-node mapping to `zelda_core.py` or `StateSpaceAStar`
2. **Add deprecation warning**
3. **Archive file**: Move to `archive/` or delete after verification

### Phase 3: Deprecate `MazeSolver` (maze_solver.py)

1. **Extract utility**: Create `src/utils/grid_astar.py` for pure grid pathfinding
2. **Update imports** if any remain
3. **Archive file**: Move to `archive/` or delete

### Phase 4: Cleanup

1. **Remove deprecated files**: `graph_solver.py`, `maze_solver.py`
2. **Remove `DungeonSolver` class** from `zelda_core.py`
3. **Update documentation**
4. **Run full test suite** to verify no regressions

---

## Migration Guide

### Before (using DungeonSolver):
```python
from Data.zelda_core import DungeonSolver, ValidationMode
solver = DungeonSolver()
result = solver.solve(stitched, mode=ValidationMode.FULL)
```

### After (using StateSpaceAStar):
```python
from simulation.validator import ZeldaLogicEnv, StateSpaceAStar

env = ZeldaLogicEnv(
    global_grid=stitched.global_grid,
    graph=stitched.graph,
    room_positions=stitched.room_positions,
    room_to_node=stitched.room_to_node,
)
solver = StateSpaceAStar(env, timeout=100000)
result = solver.solve()
```

---

## Verification Checklist

- [ ] All tests pass after Phase 1
- [ ] GUI runner works correctly with `StateSpaceAStar`
- [ ] No import errors after removing deprecated files
- [ ] Performance benchmarks show no regression
- [ ] Documentation updated

---

## Notes

- The `StateSpaceAStar` solver is already used by `gui_runner.py` for animated solving
- Consider exposing a simpler wrapper function for common use cases
- The `ValidationMode` enum should remain in `src/core/definitions.py`
