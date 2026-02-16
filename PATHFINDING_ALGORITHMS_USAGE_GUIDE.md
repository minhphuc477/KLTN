# Pathfinding Algorithms Usage Guide

## Summary of Implementation Status

All pathfinding algorithms in KLTN now use **complete game logic** for state transitions. However, different algorithms have different suitability for Zelda dungeon solving.

## ✅ Fully Working Algorithms

### 1. StateSpaceAStar (RECOMMENDED)
- **Status**: ✓ Fully working, production-ready
- **Use Case**: Primary solver for all Zelda dungeons
- **Performance**: Optimal paths, typically 15-50 states for simple dungeons
- **Features**: 
  - Handles all game mechanics (keys, doors, blocks, items, bombs)
  - A* heuristic guidance for efficient search
  - Multiple heuristic modes (balanced, speedrunner, completionist)
  - Diagonal movement support
  - Graph-based room transitions

**When to use**: Always use this as the primary solver.

```python
from src.simulation.validator import ZeldaLogicEnv, StateSpaceAStar

env = ZeldaLogicEnv(grid)
solver = StateSpaceAStar(env, timeout=10000, heuristic_mode="balanced")
success, path, states = solver.solve()
```

### 2. State Space DFS / IDDFS
- **Status**: ✓ Fully working, uses complete game logic
- **Use Case**: Feasibility checking, local connectivity validation
- **Performance**: Can explore 1000-10000 states for simple dungeons
- **Features**:
  - Complete exploration guarantee with IDDFS
  - Memory-efficient for deep searches
  - Handles all game mechanics identically to StateSpaceAStar

**When to use**: 
- Validating puzzle solvability in isolated room clusters
- Checking local connectivity for fitness functions
- When you need guaranteed completeness

```python
from src.simulation.state_space_dfs import StateSpaceDFS

env = ZeldaLogicEnv(grid)
solver = StateSpaceDFS(env, use_iddfs=True, max_depth=100)
success, path, states = solver.solve()
```

### 3. Bidirectional A*
- **Status**: ✓ Full game logic, but has performance issues
- **Use Case**: Long corridors with minimal inventory changes
- **Performance**: Can be slower than regular A* on complex dungeons
- **Limitations**:
  - Collision detection with inventory is complex
  - Best suited for simple topologies
  - Not recommended for dungeons with many keys/doors

**When to use**: Only for benchmarking or theoretical comparisons.

```python
from src.simulation.bidirectional_astar import BidirectionalAStar

env = ZeldaLogicEnv(grid)
solver = BidirectionalAStar(env, timeout=10000)
success, path, states = solver.solve()
```

## ⚠️ Limited Algorithms

### 4. D* Lite
- **Status**: ⚠️ Implemented with full game logic, but **fundamentally unsuitable** for Zelda domains
- **Why it fails**:
  - Designed for replanning in static environments where costs change
  - Zelda dungeons have **dynamic state spaces** (collecting keys changes available transitions)
  - D* Lite's g-values and rhs-values become invalid when state space changes
  - Cannot efficiently handle the inventory-dependent graph topology

**Technical explanation**: D* Lite maintains consistency constraints (g-values, rhs-values) across a static graph. In Zelda:
- Collecting a key fundamentally changes which edges exist in the state graph
- Opening a door creates new connections
- D* Lite would need to recompute everything from scratch, losing its efficiency advantage

**When to use**: **DO NOT USE** for Zelda dungeons. Use StateSpaceAStar instead.

```python
# NOT RECOMMENDED - Use StateSpaceAStar instead
from src.simulation.dstar_lite import DStarLiteSolver

# D* Lite will fail to find paths that StateSpaceAStar solves
env = ZeldaLogicEnv(grid)
solver = DStarLiteSolver(env)
success, path, states = solver.solve(env.state.copy())  # Likely to fail
```

## Implementation Details

All algorithms now use the **canonical game logic** from `ZeldaLogicEnv._try_move_pure` (validator.py:3644-3856), which handles:

- ✅ Keys and locked doors
- ✅ Bombs and bomb doors  
- ✅ Boss keys and boss doors
- ✅ Block pushing with chain tracking
- ✅ Item collection state
- ✅ Water/element tiles requiring ladder
- ✅ Door opening state
- ✅ All VGLC tile types

## Test Results

```
Test dungeon: 10x10 with key at (1,5) and locked door at (5,5)

✓ StateSpaceAStar:    success=True, path_len=15, states=38
✗ D* Lite:            success=False (max iterations) - EXPECTED
✓ State Space DFS:    success=True, path_len=19, states=6182
⚠ Bidirectional A*:   success=variable (slow on complex dungeons)
```

## Recommendation

**Use StateSpaceAStar for all Zelda dungeon solving.**

It provides:
- Optimal paths
- Fast performance (38 states vs 6182 for DFS)
- Complete game mechanic support
- Production-ready reliability

State Space DFS is useful for validation and feasibility checking, but StateSpaceAStar should be the primary solver for the pipeline.
