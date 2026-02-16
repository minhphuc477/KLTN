"""Debug A* pathfinding with block"""
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.definitions import SEMANTIC_PALETTE
from src.simulation.validator import ZeldaLogicEnv, StateSpaceAStar, GameState

# Simple block test
grid = np.full((5, 8), SEMANTIC_PALETTE['WALL'], dtype=np.int64)
grid[2, 1:7] = SEMANTIC_PALETTE['FLOOR']
grid[2, 1] = SEMANTIC_PALETTE['START']
grid[2, 6] = SEMANTIC_PALETTE['TRIFORCE']
grid[2, 3] = SEMANTIC_PALETTE['BLOCK']

print("Grid:")
for r in range(5):
    for c in range(8):
        tile = grid[r, c]
        if tile == SEMANTIC_PALETTE['WALL']:
            print('#', end='')
        elif tile == SEMANTIC_PALETTE['START']:
            print('S', end='')
        elif tile == SEMANTIC_PALETTE['TRIFORCE']:
            print('T', end='')
        elif tile == SEMANTIC_PALETTE['BLOCK']:
            print('B', end='')
        elif tile == SEMANTIC_PALETTE['FLOOR']:
            print('.', end='')
    print()

env = ZeldaLogicEnv(grid)  # Use default (no graph for simple dungeon)
env.reset()

print(f"\nStart: {env.start_pos}")
print(f"Goal: {env.goal_pos}")
print(f"Initial state: {env.state}")
print(f"Graph: {env.graph}")
print()

# Manually test state transitions
print("Manual state transitions:")
state = env.state.copy()
print(f"1. Start at {state.position}")

# Move right from (2,1) to (2,2)
success, state = env._try_move_pure(state, (2, 2), grid[2, 2])
print(f"2. Move to (2,2): success={success}, pos={state.position}")

# Try to push block from (2,2) to (2,3)
success, state = env._try_move_pure(state, (2, 3), grid[2, 3])
print(f"3. Push block to (2,3): success={success}, pos={state.position}, pushed={state.pushed_blocks}")

# Continue to goal
success, state = env._try_move_pure(state, (2, 4), grid[2, 4])
print(f"4. Move to (2,4): success={success}, pos={state.position}")

success, state = env._try_move_pure(state, (2, 5), grid[2, 5])
print(f"5. Move to (2,5): success={success}, pos={state.position}")

success, state = env._try_move_pure(state, (2, 6), grid[2, 6])
print(f"6. Move to goal (2,6): success={success}, pos={state.position}")

print("\nNow testing A*...")
env2 = ZeldaLogicEnv(grid)
astar = StateSpaceAStar(env2)
success, path, diag = astar.solve_with_diagnostics()

print(f"Success: {success}")
print(f"Path: {path}")
print(f"Failure: {diag.failure_reason}")
