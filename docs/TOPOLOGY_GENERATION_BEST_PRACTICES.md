# Topology Generation Best Practices & Bug Fix Documentation

## Overview

This document describes the bug fixes and best practices implemented for the KLTN topology generation and pathfinding systems.

## Critical Bug Fixes

### 1. **dungeon_pipeline.py: Invalid Parameter Passing**

**Issue:** Line 503 passed `max_rooms=num_rooms` to `EvolutionaryTopologyGenerator()`, but this parameter didn't exist in the constructor.

**Fix:**
```python
# BEFORE (BROKEN):
topology_generator = EvolutionaryTopologyGenerator(
    target_curve=target_curve,
    population_size=population_size,
    generations=generations,
    max_rooms=num_rooms,  # ❌ Parameter doesn't exist!
    seed=seed,
)

# AFTER (FIXED):
target_genome_length = max(10, int(num_rooms * 0.7))
topology_generator = EvolutionaryTopologyGenerator(
    target_curve=target_curve,
    population_size=population_size,
    generations=generations,
    genome_length=target_genome_length,  # ✅ Controls rule count
    max_nodes=num_rooms,  # ✅ Direct room count constraint
    seed=seed,
)
```

**Impact:**
- Prevents `TypeError` at runtime
- Provides direct control over maximum room count
- Maintains empirical relationship: `genome_length ≈ 0.7 × target_rooms`

---

### 2. **EvolutionaryTopologyGenerator: Missing max_nodes Parameter**

**Issue:** No way to constrain maximum dungeon size during evolution. The `GraphGrammarExecutor.execute()` had `max_nodes=20` hardcoded, but it wasn't exposed at the generator level.

**Fix:**
```python
# Constructor now accepts max_nodes
def __init__(
    self,
    target_curve: List[float],
    zelda_transition_matrix: Optional[Dict[str, Dict[str, float]]] = None,
    population_size: int = 50,
    generations: int = 100,
    mutation_rate: float = 0.15,
    crossover_rate: float = 0.7,
    genome_length: int = 18,
    max_nodes: int = 20,  # ✅ NEW PARAMETER
    seed: Optional[int] = None,
):
    ...
    self.max_nodes = max_nodes
    
    # Validate
    if max_nodes < 5:
        logger.warning(f"max_nodes={max_nodes} is very low, setting to minimum of 5")
        self.max_nodes = 5

# Passes max_nodes to executor
individual.phenotype = self.executor.execute(
    individual.genome,
    difficulty=0.5,
    max_nodes=self.max_nodes,  # ✅ Uses instance parameter
)
```

**Impact:**
- Direct control over dungeon size bounds
- Prevents graph explosion during evolution
- Configurable per-experiment

---

### 3. **D* Lite: Incorrect Predecessor State Computation**

**Issue:** `update_vertex()` computed `rhs(s)` by creating fake predecessor states—just copying the current state and changing position without validating if the transition is actually possible.

**Broken Code:**
```python
# BEFORE (BROKEN):
# Create predecessor state (simplified - assume same inventory)
pred_state = state.copy()
pred_state.position = (pred_r, pred_c)
pred_hash = hash(pred_state)

pred_g = self.g_scores.get(pred_hash, float('inf'))
if pred_g < float('inf'):
    cost = self._get_edge_cost(pred_state, state)
    min_rhs = min(min_rhs, pred_g + cost)
```

**Why This Failed:**
- Assumed all positions are reachable without checking doors, keys, or walls
- Created invalid predecessor states that don't respect game mechanics
- Led to incorrect `rhs` values and failed pathfinding

**Fixed Code:**
```python
# AFTER (FIXED):
# Create a hypothetical predecessor state at pred_pos
pred_state = state.copy()
pred_state.position = (pred_r, pred_c)

# Check if this predecessor can actually reach current state
# by attempting the forward move
target_tile = self.env.grid[state.position[0], state.position[1]]
can_reach, _ = self.env._try_move_pure(pred_state, state.position, target_tile)

if not can_reach:
    continue  # Skip invalid predecessors

pred_hash = hash(pred_state)
pred_g = self.g_scores.get(pred_hash, float('inf'))
if pred_g < float('inf'):
    cost = self._get_edge_cost(pred_state, state)
    min_rhs = min(min_rhs, pred_g + cost)
```

**Impact:**
- D* Lite now correctly handles keys, doors, bombs, and blocks
- `rhs(s)` values are computed only from valid predecessors
- Pathfinding succeeds in dungeons with state-based mechanics

---

### 4. **Bidirectional A*: Incomplete Collision Detection**

**Issue:** `_check_approximate_collision()` only checked inventory compatibility (keys, bombs, boss_key, item) but ignored `opened_doors` and `collected_items` sets. This caused false collisions where forward and backward searches met at the same position but with incompatible states.

**Broken Code:**
```python
# BEFORE (BROKEN):
if is_forward:
    # Only checks inventory
    if (node.state.keys <= other_node.state.keys and
        node.state.bomb_count <= other_node.state.bomb_count and
        (other_node.state.has_boss_key or not node.state.has_boss_key) and
        (other_node.state.has_item or not node.state.has_item)):
        return other_node  # ❌ Missing state set checks!
```

**Why This Failed:**
- Forward state might have `opened_doors={(2,3)}` 
- Backward state might have `opened_doors={(2,3), (5,6)}`
- They'd collide at same position, but backward state assumes more doors are open
- Path reconstruction would be invalid

**Fixed Code:**
```python
# AFTER (FIXED):
if is_forward:
    # Check inventory compatibility
    inventory_compatible = (
        node.state.keys <= other_node.state.keys and
        node.state.bomb_count <= other_node.state.bomb_count and
        (other_node.state.has_boss_key or not node.state.has_boss_key) and
        (other_node.state.has_item or not node.state.has_item)
    )
    
    # ✅ CRITICAL: Check state sets compatibility
    # Forward opened_doors must be a subset of backward opened_doors
    # Forward collected_items must be a subset of backward collected_items
    state_sets_compatible = (
        node.state.opened_doors.issubset(other_node.state.opened_doors) and
        node.state.collected_items.issubset(other_node.state.collected_items)
    )
    
    if inventory_compatible and state_sets_compatible:
        return other_node
```

**Impact:**
- Collision detection now correctly validates state compatibility
- Prevents invalid meeting points
- Bidirectional search returns valid paths

---

## Best Practices

### Room Count Control

**Problem:** Indirect relationship between `genome_length` and final room count.

**Solution:**
```python
# Empirical formula: genome_length ≈ 0.7 × target_rooms
# (Not all rules apply, some are skipped)

target_rooms = 12
genome_length = max(10, int(target_rooms * 0.7))

generator = EvolutionaryTopologyGenerator(
    target_curve=target_curve,
    genome_length=genome_length,  # Controls how many rules to try
    max_nodes=target_rooms,       # Hard upper bound
    seed=seed
)
```

**Rationale:**
- `genome_length`: Suggests how many rules to execute
- `max_nodes`: Prevents graph explosion (safety net)
- Empirical 0.7x factor accounts for skipped/inapplicable rules

---

### State-Based Pathfinding

**Problem:** Algorithms must respect game mechanics (keys, doors, items, blocks).

**Solution:**
```python
# ✅ DO: Use environment's state transition logic
target_tile = self.env.grid[new_r, new_c]
success, new_state = self.env._try_move_pure(state, (new_r, new_c), target_tile)

if success:
    successors.append(new_state)

# ❌ DON'T: Assume all positions are reachable
new_state = state.copy()
new_state.position = (new_r, new_c)  # WRONG! Ignores doors, keys, etc.
successors.append(new_state)
```

**Rationale:**
- `_try_move_pure()` handles all game mechanics:
  - Key collection and consumption
  - Door opening
  - Block pushing
  - Item interaction
- Returns both success flag and properly updated state
- Maintains state consistency

---

### Bidirectional Search State Compatibility

**Problem:** Forward and backward searches must meet at compatible states.

**Solution:**
```python
# Check ALL state components for compatibility
def states_compatible(forward_state, backward_state):
    # Position must match
    if forward_state.position != backward_state.position:
        return False
    
    # Forward inventory ≤ backward inventory
    inventory_ok = (
        forward_state.keys <= backward_state.keys and
        forward_state.bomb_count <= backward_state.bomb_count
    )
    
    # Forward state sets ⊆ backward state sets
    state_sets_ok = (
        forward_state.opened_doors.issubset(backward_state.opened_doors) and
        forward_state.collected_items.issubset(backward_state.collected_items)
    )
    
    return inventory_ok and state_sets_ok
```

**Rationale:**
- Forward search starts with empty inventory, collects items along the way
- Backward search assumes maximal inventory at goal
- Meeting point must satisfy: forward ⊆ backward (subset relationship)

---

## Validation Guidelines

### 1. Topology Validation

Always validate generated graphs:

```python
from src.data.vglc_utils import validate_topology, filter_virtual_nodes

# Generate topology
graph = generator.evolve()

# VGLC compliance: filter virtual nodes
physical_graph = filter_virtual_nodes(graph)

# Validate
report = validate_topology(physical_graph)
if not report.is_valid:
    logger.warning(f"Topology validation failed: {report.summary()}")
else:
    logger.info("Topology validation: PASSED")
```

---

### 2. Solvability Testing

Test pathfinding after generation:

```python
from src.simulation.validator import ZeldaLogicEnv
from src.simulation.state_space_astar import StateSpaceAStar

# Create environment
env = ZeldaLogicEnv(grid)

# Test solvability
solver = StateSpaceAStar(env, timeout=50000)
success, path, nodes = solver.solve()

assert success, "Generated dungeon must be solvable"
assert len(path) > 0, "Path must exist"
```

---

### 3. Deterministic Testing

Always use seeds for reproducibility:

```python
# ✅ DO: Use explicit seeds
generator = EvolutionaryTopologyGenerator(
    target_curve=curve,
    seed=42  # Reproducible results
)

# ✅ DO: Test multiple seeds
for seed in [42, 123, 456]:
    generator = EvolutionaryTopologyGenerator(
        target_curve=curve,
        seed=seed
    )
    graph = generator.evolve()
    # Validate...
```

---

## Testing Strategy

### Unit Tests

Test individual components:

1. **Parameter Validation:**
   - `max_nodes` parameter accepted by EvolutionaryTopologyGenerator
   - Parameter bounds enforced (min 5 nodes)

2. **Algorithm Correctness:**
   - D* Lite finds paths with keys and doors
   - Bidirectional A* collision detection with state sets
   - GraphGrammarExecutor respects `max_nodes`

3. **State Transitions:**
   - `_try_move_pure()` handles all mechanics
   - State hashing includes all components
   - State equality checks all fields

### Integration Tests

Test complete pipelines:

1. **Topology Generation:**
   - Generate graph with target room count
   - Validate VGLC compliance
   - Check solvability

2. **Pathfinding:**
   - D* Lite on complex dungeons
   - Bidirectional A* with multiple keys/doors
   - State-space A* as ground truth

3. **Pipeline End-to-End:**
   - `dungeon_pipeline.py` with `generate_topology=True`
   - Room count control via `num_rooms`
   - Validation passes

---

## Performance Considerations

### Graph Size Bounds

```python
# Recommended ranges
MAX_NODES_SMALL = 8     # Quick testing
MAX_NODES_MEDIUM = 15   # Standard dungeons
MAX_NODES_LARGE = 25    # Complex dungeons
MAX_NODES_EXTREME = 40  # Research/stress testing

# Avoid very small values
if max_nodes < 5:
    logger.warning("max_nodes < 5 may produce trivial dungeons")
```

### Search Timeouts

```python
# State-space search scales with: O(N × I)
# N = grid size, I = inventory states

# Recommended timeouts
TIMEOUT_SMALL = 10000    # Simple dungeons
TIMEOUT_MEDIUM = 50000   # Standard dungeons
TIMEOUT_LARGE = 100000   # Complex dungeons

solver = StateSpaceAStar(env, timeout=TIMEOUT_MEDIUM)
```

---

## Common Pitfalls

### ❌ Pitfall 1: Ignoring State Transitions

```python
# WRONG: Direct position manipulation
new_state = state.copy()
new_state.position = new_pos
# BUG: Doesn't check if move is valid!

# RIGHT: Use environment logic
success, new_state = env._try_move_pure(state, new_pos, target_tile)
if success:
    # new_state is properly updated
```

### ❌ Pitfall 2: Incomplete State Hashing

```python
# NOTE: pushed_blocks is intentionally excluded from hash
# for performance (see GameState.__hash__ comment)

# States with same position/inventory but different
# pushed_blocks are considered equal for search purposes

# Block positions are checked during movement in _try_move_pure()
```

### ❌ Pitfall 3: Hardcoded Constants

```python
# WRONG: Hardcoded max_nodes in executor call
graph = executor.execute(genome, difficulty=0.5, max_nodes=20)

# RIGHT: Use configurable parameter
graph = executor.execute(genome, difficulty=0.5, max_nodes=self.max_nodes)
```

---

## Migration Guide

### Updating Existing Code

If you have code using the old API:

```python
# OLD API (BROKEN):
generator = EvolutionaryTopologyGenerator(
    target_curve=curve,
    max_rooms=15,  # ❌ Doesn't exist
)

# NEW API (FIXED):
generator = EvolutionaryTopologyGenerator(
    target_curve=curve,
    genome_length=max(10, int(15 * 0.7)),  # Empirical formula
    max_nodes=15,  # ✅ Now available
)
```

### Testing After Migration

Run validation suite:

```bash
# Run bug fix tests
pytest tests/test_topology_generation_fixes.py -v

# Run full test suite
pytest tests/ -v

# Validate a specific config
python scripts/validate_configs.py --config configs/nn_tuning.json
```

---

## References

### Academic Sources

1. **D* Lite Algorithm:**
   - Koenig & Likhachev (2002). "D* Lite." AAAI Conference.

2. **Bidirectional Search:**
   - Pohl (1971). "Bi-directional Search." Machine Intelligence, 6.
   - Kaindl & Kainz (1997). "Bidirectional Heuristic Search Reconsidered." JAIR, 7.

3. **Evolutionary PCG:**
   - Togelius et al. (2011). "Search-Based Procedural Content Generation."
   - Dormans & Bakkes (2011). "Generating Missions and Spaces."

### Internal Documentation

- `QUICK_START.md` - Setup and quick commands
- `.github/copilot-instructions.md` - AI agent guidance
- `docs/` - Architecture and design docs
- Source code docstrings - Implementation details

---

## Changelog

### 2026-02-16: Critical Bug Fixes

1. **dungeon_pipeline.py (Line 503)**: Fixed `max_rooms` parameter error
   - Removed non-existent parameter
   - Added `genome_length` calculation
   - Added `max_nodes` parameter

2. **EvolutionaryTopologyGenerator**: Added `max_nodes` parameter
   - Constructor accepts `max_nodes` with default 20
   - Validates minimum 5 nodes
   - Passes to executor during evolution

3. **D* Lite `update_vertex()`**: Fixed predecessor computation
   - Now validates transitions using `_try_move_pure()`
   - Skips invalid predecessors
   - Produces correct `rhs` values

4. **Bidirectional A* `_check_approximate_collision()`**: Fixed state compatibility
   - Added `opened_doors` subset check
   - Added `collected_items` subset check
   - Prevents invalid meeting points

---

## Contact & Contributions

For questions or bug reports:
1. Check existing documentation in `docs/`
2. Review test cases in `tests/test_topology_generation_fixes.py`
3. Consult `.github/copilot-instructions.md` for coding standards

**Always test changes with deterministic seeds and validate VGLC compliance!**
