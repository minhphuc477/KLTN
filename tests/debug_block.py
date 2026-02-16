"""Debug block pushing issue"""
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.definitions import SEMANTIC_PALETTE
from src.simulation.validator import ZeldaLogicEnv, StateSpaceAStar

# Try a simpler block test
grid = np.full((5, 8), SEMANTIC_PALETTE['WALL'], dtype=np.int64)
grid[2, 1:7] = SEMANTIC_PALETTE['FLOOR']
grid[2, 1] = SEMANTIC_PALETTE['START']
grid[2, 6] = SEMANTIC_PALETTE['TRIFORCE']
grid[2, 3] = SEMANTIC_PALETTE['BLOCK']

print("Grid layout:")
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

print("\nTesting A*...")
env = ZeldaLogicEnv(grid)
astar = StateSpaceAStar(env)
success, path, diag = astar.solve_with_diagnostics()

print(f"Success: {success}")
print(f"Path: {path}")
print(f"Failure reason: {diag.failure_reason}")
print(f"States explored: {diag.states_explored}")
