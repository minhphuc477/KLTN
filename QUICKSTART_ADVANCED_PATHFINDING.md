# Quick Reference: Advanced Pathfinding Algorithms

## Using in GUI

1. **Launch GUI**:
   ```bash
   python gui_runner.py
   ```

2. **Select Algorithm** from dropdown (top-right sidebar):
   - A* (default)
   - BFS
   - Dijkstra
   - Greedy
   - **D* Lite** ← New!
   - **DFS/IDDFS** ← New!
   - **Bidirectional A*** ← New!
   - CBS (Balanced/Explorer/Cautious/Forgetful/Speedrunner/Greedy)

3. **Solve Dungeon**: Press `SPACE` or click "Solve" button

4. **View Results**: Path animated on screen, metrics logged to console

## Using Programmatically

### D* Lite (Incremental Replanning)
```python
from src.simulation.dstar_lite import DStarLiteSolver
from src.simulation.validator import ZeldaLogicEnv

# Create environment
env = ZeldaLogicEnv(dungeon_grid)

# Initialize solver
solver = DStarLiteSolver(env, heuristic_mode="balanced")

# Initial solve
start_state = env.state.copy()
success, path, nodes = solver.solve(start_state)

# Environment changes (door unlocked)
current_state.opened_doors.add((5, 5))

# Replan efficiently
success, new_path, updated = solver.replan(current_state)
print(f"Replanned with {updated} states updated (vs {nodes} initial)")
```

### DFS/IDDFS (Complete Exploration)
```python
from src.simulation.state_space_dfs import StateSpaceDFS
from src.simulation.validator import ZeldaLogicEnv

# Create environment
env = ZeldaLogicEnv(dungeon_grid)

# Initialize solver (use IDDFS for better completeness)
solver = StateSpaceDFS(
    env,
    timeout=100000,
    max_depth=500,
    allow_diagonals=False,
    use_iddfs=True  # Iterative deepening
)

# Solve
success, path, nodes = solver.solve()

# Check metrics
print(f"Max depth: {solver.metrics.max_depth_reached}")
print(f"Backtracks: {solver.metrics.backtrack_count}")
print(f"Cycles detected: {solver.metrics.cycle_detections}")
```

### Bidirectional A* (Meet-in-the-Middle)
```python
from src.simulation.bidirectional_astar import BidirectionalAStar
from src.simulation.validator import ZeldaLogicEnv

# Create environment
env = ZeldaLogicEnv(dungeon_grid)

# Initialize solver
solver = BidirectionalAStar(
    env,
    timeout=100000,
    allow_diagonals=False,
    heuristic_mode="balanced"
)

# Solve
success, path, nodes = solver.solve()

# Check where frontiers met
print(f"Meeting point: {solver.meeting_point}")
print(f"Collision checks: {solver.collision_checks}")
```

## Algorithm Selection Guide

**Use D* Lite when:**
- Environment changes during search (doors unlock, enemies defeated)
- Need efficient replanning without full restart
- Testing evolutionary mutations on dungeon graphs

**Use DFS/IDDFS when:**
- Checking feasibility (does ANY path exist?)
- Small dungeons (< 15x15)
- Memory constrained environments
- Need complete exploration guarantee

**Use Bidirectional A* when:**
- Long paths (>20 steps)
- Direct corridors between start and goal
- Want to reduce nodes expanded
- Comparing against A* baseline

**Use A* (baseline) when:**
- Need optimal path
- Standard dungeons without special requirements
- Proven, reliable algorithm

## Performance Tips

### D* Lite
- Initial search ≈ A* performance
- Replanning 10-100× faster than A* restart
- Best when <10% of map changes between replans

### DFS/IDDFS
- Set appropriate `max_depth` (default: 500)
- Use `use_iddfs=True` for better completeness
- Expect 2-5× more nodes than A* but lower memory

### Bidirectional A*
- Ideal for long, narrow dungeons
- Can reduce nodes by 30-50% vs A*
- Slightly slower per node due to collision checks

## Common Issues

### "D* Lite found no solution"
- Check that initial state has required items
- Verify environment changes are tracked in state
- Try increasing timeout

### "DFS timed out"
- Increase `timeout` parameter
- Reduce `max_depth` for faster shallow search
- Switch to IDDFS mode (`use_iddfs=True`)
- Consider using A* instead for large dungeons

### "Bidirectional A* path invalid"
- Inventory reversal may have failed
- Check for one-way doors (edge directionality)
- Verify goal state inventory heuristic is correct

## Testing

Run test suite:
```bash
pytest tests/test_advanced_pathfinding.py -v
```

Quick manual test:
```python
python tests/test_advanced_pathfinding.py
```

## Debugging

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check solver metrics:
```python
# D* Lite
print(f"Replans: {solver.replans_count}")
print(f"States updated: {solver.states_updated}")

# DFS
print(f"Depth: {solver.metrics.max_depth_reached}")
print(f"Backtracks: {solver.metrics.backtrack_count}")

# Bidirectional A*
print(f"Forward explored: {len(solver.forward_closed)}")
print(f"Backward explored: {len(solver.backward_closed)}")
```

## Examples

See full examples in:
- `tests/test_advanced_pathfinding.py`
- `ADVANCED_PATHFINDING_REPORT.md`
- Individual algorithm docstrings

## References

- **D* Lite**: Koenig & Likhachev (2002). "D* Lite." AAAI Conference.
- **IDDFS**: Korf (1985). "Depth-First Iterative-Deepening." AI Journal.
- **Bidirectional A***: Pohl (1971). "Bi-directional Search." Machine Intelligence.
