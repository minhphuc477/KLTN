"""Find if there's ANY path avoiding unpushable blocks"""
from Data.zelda_core import ZeldaDungeonAdapter, SEMANTIC_PALETTE
from simulation.validator import WALKABLE_IDS, BLOCKING_IDS, PUSHABLE_IDS, WATER_IDS
from pathlib import Path
import numpy as np
from collections import deque

data_root = Path('Data/The Legend of Zelda')
adapter = ZeldaDungeonAdapter(str(data_root))
dungeon = adapter.load_dungeon(1, variant=1)
stitched = adapter.stitch_dungeon(dungeon)
grid = stitched.global_grid

start = stitched.start_global
goal = stitched.triforce_global

print('=== REALISTIC PATH FINDING ===')
print('Start:', start)
print('Goal:', goal)
print()

# Approach 1: BFS that treats blocks and water as BLOCKING
# This simulates "no inventory" state
print('Approach 1: Blocks and Water are BLOCKING')
impassable = BLOCKING_IDS | PUSHABLE_IDS | WATER_IDS

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
                if tile not in impassable:
                    visited.add((nr, nc))
                    queue.append((nr, nc))

print(f'  Goal reachable: {goal in visited}')
print(f'  Tiles visited: {len(visited)}')
print()

# Approach 2: Blocks BLOCKING, Water WALKABLE (if we had KEY_ITEM)
print('Approach 2: Blocks BLOCKING, Water WALKABLE (has item)')
impassable = BLOCKING_IDS | PUSHABLE_IDS  # Water becomes walkable

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
                if tile not in impassable:
                    visited.add((nr, nc))
                    queue.append((nr, nc))

print(f'  Goal reachable: {goal in visited}')
print(f'  Tiles visited: {len(visited)}')
print()

# Approach 3: Everything except BLOCKING walkable
print('Approach 3: Only WALL/VOID blocking (baseline)')
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
                if tile not in BLOCKING_IDS:
                    visited.add((nr, nc))
                    queue.append((nr, nc))

print(f'  Goal reachable: {goal in visited}')
print(f'  Tiles visited: {len(visited)}')
print()

# Check: Are KEY_ITEM tiles reachable without crossing water/blocks?
key_item_positions = list(zip(*np.where(grid == SEMANTIC_PALETTE['KEY_ITEM'])))
print('=== KEY_ITEM REACHABILITY (no blocks/water) ===')
print(f'KEY_ITEM positions: {key_item_positions}')

# BFS avoiding blocks and water
impassable = BLOCKING_IDS | PUSHABLE_IDS | WATER_IDS
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
                if tile not in impassable:
                    visited.add((nr, nc))
                    queue.append((nr, nc))

for pos in key_item_positions:
    status = 'REACHABLE' if pos in visited else 'UNREACHABLE'
    print(f'  KEY_ITEM at {pos}: {status}')
