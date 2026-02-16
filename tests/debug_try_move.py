"""Debug _try_move_pure for block pushing"""
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.definitions import SEMANTIC_PALETTE
from src.simulation.validator import ZeldaLogicEnv, GameState

# Simple block test
grid = np.full((5, 8), SEMANTIC_PALETTE['WALL'], dtype=np.int64)
grid[2, 1:7] = SEMANTIC_PALETTE['FLOOR']
grid[2, 1] = SEMANTIC_PALETTE['START']
grid[2, 6] = SEMANTIC_PALETTE['TRIFORCE']
grid[2, 3] = SEMANTIC_PALETTE['BLOCK']

env = ZeldaLogicEnv(grid)

# Try to move from (2,2) to (2,3) where the block is
state = GameState(position=(2, 2), keys=0, bomb_count=0)
target_pos = (2, 3)
target_tile = grid[2, 3]

print(f"Current position: {state.position}")
print(f"Target position: {target_pos}")
print(f"Target tile: {target_tile} (BLOCK={SEMANTIC_PALETTE['BLOCK']})")
print(f"Tile behind block: (2,4) = {grid[2, 4]} (FLOOR={SEMANTIC_PALETTE['FLOOR']})")
print()

success, new_state = env._try_move_pure(state, target_pos, target_tile)

print(f"Can move: {success}")
print(f"New position: {new_state.position}")
print(f"Pushed blocks: {new_state.pushed_blocks}")
