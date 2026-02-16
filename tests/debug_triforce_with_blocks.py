"""Debug TRIFORCE with pushed blocks in state"""
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.definitions import SEMANTIC_PALETTE
from src.simulation.validator import ZeldaLogicEnv, GameState

grid = np.full((5, 8), SEMANTIC_PALETTE['WALL'], dtype=np.int64)
grid[2, 1:7] = SEMANTIC_PALETTE['FLOOR']
grid[2, 1] = SEMANTIC_PALETTE['START']
grid[2, 3] = SEMANTIC_PALETTE['BLOCK']
grid[2, 6] = SEMANTIC_PALETTE['TRIFORCE']

env = ZeldaLogicEnv(grid)

# Simulate the full path with block push
state = GameState(position=(2, 1))

# Move to (2,2)
success, state = env._try_move_pure(state, (2, 2), grid[2, 2])
print(f"1. Move to (2,2): success={success}, pushed_blocks={state.pushed_blocks}")

# Push block from (2,2) to (2,3) - agent moves to (2,3), block goes to (2,4)
success, state = env._try_move_pure(state, (2, 3), grid[2, 3])
print(f"2. Push block: success={success}, pos={state.position}, pushed_blocks={state.pushed_blocks}")

# Move to (2,4) - where the block now is
success, state = env._try_move_pure(state, (2, 4), grid[2, 4])
print(f"3. Move to (2,4): success={success}, pos={state.position}, pushed_blocks={state.pushed_blocks}")

# Move to (2,5)
success, state = env._try_move_pure(state, (2, 5), grid[2, 5])
print(f"4. Move to (2,5): success={success}, pos={state.position}, pushed_blocks={state.pushed_blocks}")

# Move to (2,6) - TRIFORCE
print(f"\nMoving to TRIFORCE at (2,6)...")
print(f"Current state: pos={state.position}, pushed_blocks={state.pushed_blocks}")
print(f"Target tile at (2,6): {grid[2,6]} (TRIFORCE={SEMANTIC_PALETTE['TRIFORCE']})")
print()

# Check if (2,6) might be in pushed_blocks
for from_pos, to_pos in state.pushed_blocks:
    print(f"  Pushed block: {from_pos} -> {to_pos}")
    if to_pos == (2, 6):
        print(f"  WARNING: Block was pushed TO (2,6)!")
    if from_pos == (2, 6):
        print(f"  INFO: Block was pushed FROM (2,6)")

success, state = env._try_move_pure(state, (2, 6), grid[2, 6])
print(f"\n5. Move to TRIFORCE (2,6): success={success}, pos={state.position}")
