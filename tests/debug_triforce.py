"""Debug TRIFORCE tile"""
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.definitions import SEMANTIC_PALETTE
from src.simulation.validator import ZeldaLogicEnv, GameState, WALKABLE_IDS

grid = np.full((5, 8), SEMANTIC_PALETTE['WALL'], dtype=np.int64)
grid[2, 1:7] = SEMANTIC_PALETTE['FLOOR']
grid[2, 1] = SEMANTIC_PALETTE['START']
grid[2, 6] = SEMANTIC_PALETTE['TRIFORCE']

env = ZeldaLogicEnv(grid)

# Check what tile (2,6) is
print(f"Tile at (2,6): {grid[2,6]}")
print(f"TRIFORCE ID: {SEMANTIC_PALETTE['TRIFORCE']}")
print(f"TRIFORCE in WALKABLE_IDS: {SEMANTIC_PALETTE['TRIFORCE'] in WALKABLE_IDS}")
print()

# Try to move to TRIFORCE
state = GameState(position=(2, 5))
target_pos = (2, 6)
target_tile = grid[2, 6]

print(f"Moving from {state.position} to {target_pos}")
print(f"Target tile: {target_tile}")

success, new_state = env._try_move_pure(state, target_pos, target_tile)

print(f"Success: {success}")
print(f"New position: {new_state.position}")
