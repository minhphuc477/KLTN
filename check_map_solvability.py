from Data.zelda_core import ZeldaDungeonAdapter, SEMANTIC_PALETTE, ID_TO_NAME
import numpy as np

adapter = ZeldaDungeonAdapter('Data/The Legend of Zelda')
d = adapter.load_dungeon(1, 1)
s = adapter.stitch_dungeon(d)
grid = s.global_grid

print('Tile distribution (obstacles/keys):')
for tid in [SEMANTIC_PALETTE['KEY_SMALL'], SEMANTIC_PALETTE['KEY_BOSS'], 
            SEMANTIC_PALETTE['KEY_ITEM'], SEMANTIC_PALETTE['DOOR_LOCKED'], 
            SEMANTIC_PALETTE['DOOR_BOSS'], SEMANTIC_PALETTE['DOOR_BOMB'], 
            SEMANTIC_PALETTE['ELEMENT']]:
    count = np.sum(grid == tid)
    if count > 0:
        print(f'  {ID_TO_NAME[tid]}: {count}')

# Check if path from start to goal crosses any obstacles
start = (19, 2)
goal = (88, 16)

# Simple flood fill to check reachability ignoring keys
from collections import deque

def is_walkable_simple(tile_id):
    """Check if tile is walkable without any keys."""
    return tile_id in [SEMANTIC_PALETTE['FLOOR'], SEMANTIC_PALETTE['START'], 
                       SEMANTIC_PALETTE['TRIFORCE'], SEMANTIC_PALETTE['DOOR_OPEN'],
                       SEMANTIC_PALETTE['KEY_SMALL'], SEMANTIC_PALETTE['KEY_BOSS'],
                       SEMANTIC_PALETTE['KEY_ITEM'], SEMANTIC_PALETTE['ITEM_MINOR'],
                       SEMANTIC_PALETTE['ELEMENT_FLOOR'], SEMANTIC_PALETTE['STAIR']]

visited = set()
queue = deque([start])
visited.add(start)
found = False

while queue:
    r, c = queue.popleft()
    
    if (r, c) == goal:
        found = True
        break
    
    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
            if (nr, nc) not in visited and is_walkable_simple(grid[nr, nc]):
                visited.add((nr, nc))
                queue.append((nr, nc))

print(f'\nSimple reachability (no keys required):')
print(f'  Can reach goal: {found}')
print(f'  Positions reachable: {len(visited)}')

if not found:
    print('\n❌ GOAL IS NOT REACHABLE WITHOUT KEYS/ITEMS')
    print('   Need to collect keys or cross obstacles')
