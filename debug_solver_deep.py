"""Deep debug of StateSpaceAStar solver"""
from Data.zelda_core import ZeldaDungeonAdapter, SEMANTIC_PALETTE
from simulation.validator import ZeldaLogicEnv, StateSpaceAStar, WALKABLE_IDS, BLOCKING_IDS, PUSHABLE_IDS, WATER_IDS, GameState
from pathlib import Path
import numpy as np

data_root = Path('Data/The Legend of Zelda')
adapter = ZeldaDungeonAdapter(str(data_root))

dungeon = adapter.load_dungeon(1, variant=1)
stitched = adapter.stitch_dungeon(dungeon)

# Create environment 
env = ZeldaLogicEnv(
    stitched.global_grid,
    graph=stitched.graph,
    room_to_node=stitched.room_to_node,
    room_positions=stitched.room_positions
)

print('=== SOLVER DEBUG ===')
print('Start:', env.start_pos)
print('Goal:', env.goal_pos)
print()

# Manually check moves from START
grid = env.grid
start_r, start_c = env.start_pos

print('Tiles around START:')
for dr in range(-2, 3):
    row_tiles = []
    for dc in range(-2, 3):
        nr, nc = start_r + dr, start_c + dc
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
            tile = grid[nr, nc]
            if tile in WALKABLE_IDS:
                row_tiles.append(f'{tile:2}W')  # Walkable
            elif tile in BLOCKING_IDS:
                row_tiles.append(f'{tile:2}B')  # Blocking
            elif tile in PUSHABLE_IDS:
                row_tiles.append(f'{tile:2}P')  # Pushable
            elif tile in WATER_IDS:
                row_tiles.append(f'{tile:2}~')  # Water
            else:
                row_tiles.append(f'{tile:2}?')  # Unknown
        else:
            row_tiles.append('OOB')
    print('  ', ' '.join(row_tiles))

print()

# Check which tiles are unknown (not in any category)
all_categorized = WALKABLE_IDS | BLOCKING_IDS | PUSHABLE_IDS | WATER_IDS
unique_tiles = set(np.unique(grid))
uncategorized = unique_tiles - all_categorized
print('Uncategorized tiles:', uncategorized)
print()

# Check CONDITIONAL_IDS
from simulation.validator import CONDITIONAL_IDS
print('CONDITIONAL_IDS:', CONDITIONAL_IDS)
print('Tiles in CONDITIONAL_IDS in grid:', 
      {t: np.sum(grid == t) for t in CONDITIONAL_IDS if np.sum(grid == t) > 0})

print()

# Check if ENEMY tiles are walkable or blocking
print('ENEMY (20) in WALKABLE_IDS:', SEMANTIC_PALETTE['ENEMY'] in WALKABLE_IDS)
print('ENEMY (20) in BLOCKING_IDS:', SEMANTIC_PALETTE['ENEMY'] in BLOCKING_IDS)
print('ENEMY (20) count in grid:', np.sum(grid == SEMANTIC_PALETTE['ENEMY']))

# Enemies might be blocking progress!
# Let's check the path from start to goal with ENEMY tiles considered
print()
print('=== Checking if ENEMY tiles block path ===')
enemy_positions = list(zip(*np.where(grid == SEMANTIC_PALETTE['ENEMY'])))
print('ENEMY positions:', enemy_positions[:10], '...' if len(enemy_positions) > 10 else '')
