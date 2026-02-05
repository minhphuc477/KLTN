"""Check item reachability"""
from Data.zelda_core import ZeldaDungeonAdapter, SEMANTIC_PALETTE
from simulation.validator import WALKABLE_IDS
from pathlib import Path
import numpy as np
from collections import deque

data_root = Path('Data/The Legend of Zelda')
adapter = ZeldaDungeonAdapter(str(data_root))

dungeon = adapter.load_dungeon(1, variant=1)
stitched = adapter.stitch_dungeon(dungeon)
grid = stitched.global_grid

print('=== CHECKING ITEM REACHABILITY ===')
start = stitched.start_global
print('START:', start)

# Find KEY_ITEM positions
key_item_positions = list(zip(*np.where(grid == SEMANTIC_PALETTE['KEY_ITEM'])))
key_small_positions = list(zip(*np.where(grid == SEMANTIC_PALETTE['KEY_SMALL'])))
print('KEY_ITEM positions:', key_item_positions)
print('KEY_SMALL positions:', key_small_positions)
print()

# BFS from START - only considers tiles already in WALKABLE_IDS
visited = set()
queue = deque([start])
visited.add(start)

while queue:
    r, c = queue.popleft()
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
            if (nr, nc) not in visited:
                tile = grid[nr, nc]
                if tile in WALKABLE_IDS:
                    visited.add((nr, nc))
                    queue.append((nr, nc))

print('Reachability from START (only WALKABLE_IDS):')
for pos in key_item_positions:
    status = 'REACHABLE' if pos in visited else 'UNREACHABLE'
    print(f'  KEY_ITEM at {pos}: {status}')
for pos in key_small_positions:
    status = 'REACHABLE' if pos in visited else 'UNREACHABLE'
    print(f'  KEY_SMALL at {pos}: {status}')

print()
print('GOAL reachable:', stitched.triforce_global in visited)
print()

# Now check: is KEY_ITEM (32) in WALKABLE_IDS?
print('Is KEY_ITEM (32) in WALKABLE_IDS?', SEMANTIC_PALETTE['KEY_ITEM'] in WALKABLE_IDS)
print('WALKABLE_IDS:', WALKABLE_IDS)
