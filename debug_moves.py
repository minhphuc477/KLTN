"""Debug StateSpaceAStar internals"""
from Data.zelda_core import ZeldaDungeonAdapter, SEMANTIC_PALETTE
from simulation.validator import (
    ZeldaLogicEnv, StateSpaceAStar, WALKABLE_IDS, BLOCKING_IDS, 
    GameState, PUSHABLE_IDS, WATER_IDS, PICKUP_IDS
)
from pathlib import Path
import numpy as np
import heapq

data_root = Path('Data/The Legend of Zelda')
adapter = ZeldaDungeonAdapter(str(data_root))
dungeon = adapter.load_dungeon(1, variant=1)
stitched = adapter.stitch_dungeon(dungeon)

env = ZeldaLogicEnv(
    stitched.global_grid,
    graph=stitched.graph,
    room_to_node=stitched.room_to_node,
    room_positions=stitched.room_positions
)

print('=== DEBUG StateSpaceAStar._try_move_pure ===')
print()

grid = env.grid
start_pos = env.start_pos
goal_pos = env.goal_pos

# Create initial state
initial_state = GameState(position=start_pos)

# Test moving from start position
print(f'Testing moves from START {start_pos}:')
for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
    nr, nc = start_pos[0] + dr, start_pos[1] + dc
    if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
        tile = grid[nr, nc]
        
        # Simulate _try_move_pure logic
        can_move = True
        reason = ""
        
        if tile in BLOCKING_IDS:
            can_move = False
            reason = "BLOCKING"
        elif (nr, nc) in initial_state.opened_doors:
            reason = "already opened door"
        elif (nr, nc) in initial_state.collected_items:
            reason = "already collected item"
        elif tile in WALKABLE_IDS:
            reason = "WALKABLE"
        elif tile == SEMANTIC_PALETTE['DOOR_LOCKED']:
            if initial_state.keys > 0:
                reason = "locked door (has key)"
            else:
                can_move = False
                reason = "locked door (NO KEY)"
        elif tile in PUSHABLE_IDS:
            # Check push destination
            push_r, push_c = nr + dr, nc + dc
            if 0 <= push_r < grid.shape[0] and 0 <= push_c < grid.shape[1]:
                push_tile = grid[push_r, push_c]
                if push_tile in WALKABLE_IDS:
                    reason = f"pushable block (dest {push_tile} is walkable)"
                else:
                    can_move = False
                    reason = f"pushable block (dest {push_tile} NOT walkable)"
            else:
                can_move = False
                reason = "pushable block (dest out of bounds)"
        elif tile in WATER_IDS:
            if initial_state.has_item:
                reason = "water (has ladder/item)"
            else:
                can_move = False
                reason = "water (NO LADDER)"
        else:
            reason = f"default (unknown tile {tile})"
        
        status = "CAN" if can_move else "CANNOT"
        print(f'  ({nr:3}, {nc:3}): tile={tile:2} -> {status} move ({reason})')

print()

# Now check tiles along the path from simple A*
print('=== Checking tiles that might block inventory-based search ===')

# Find all PUSHABLE (BLOCK) tiles
block_positions = list(zip(*np.where(grid == SEMANTIC_PALETTE['BLOCK'])))
print(f'BLOCK (3) tiles in grid: {len(block_positions)}')

# Find all WATER tiles
water_positions = list(zip(*np.where(grid == SEMANTIC_PALETTE['ELEMENT'])))
print(f'ELEMENT/WATER (40) tiles in grid: {len(water_positions)}')

print()

# The issue: BLOCK tiles require pushing, and WATER tiles require ladder
# But if blocks can't be pushed (destination blocked), or water can't be crossed...

# Check if any BLOCK tiles are on the simple path
from collections import deque
visited = set()
queue = deque([start_pos])
visited.add(start_pos)
came_from = {start_pos: None}

while queue:
    r, c = queue.popleft()
    if (r, c) == goal_pos:
        break
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
            if (nr, nc) not in visited:
                tile = grid[nr, nc]
                if tile not in BLOCKING_IDS:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
                    came_from[(nr, nc)] = (r, c)

# Reconstruct path
path = []
pos = goal_pos
while pos is not None:
    path.append(pos)
    pos = came_from.get(pos)
path.reverse()

print(f'Simple BFS path length: {len(path)}')

# Check tiles along this path
block_on_path = []
water_on_path = []
for pos in path:
    tile = grid[pos[0], pos[1]]
    if tile == SEMANTIC_PALETTE['BLOCK']:
        block_on_path.append(pos)
    if tile == SEMANTIC_PALETTE['ELEMENT']:
        water_on_path.append(pos)

print(f'BLOCK tiles on simple path: {len(block_on_path)} - {block_on_path}')
print(f'WATER tiles on simple path: {len(water_on_path)} - {water_on_path}')

print()
# If there are BLOCK tiles, check if they can be pushed
if block_on_path:
    print('=== Checking if blocks can be pushed ===')
    for block_pos in block_on_path:
        # Find how we approach this block
        idx = path.index(block_pos)
        if idx > 0:
            prev_pos = path[idx - 1]
            dr = block_pos[0] - prev_pos[0]
            dc = block_pos[1] - prev_pos[1]
            push_r = block_pos[0] + dr
            push_c = block_pos[1] + dc
            if 0 <= push_r < grid.shape[0] and 0 <= push_c < grid.shape[1]:
                push_tile = grid[push_r, push_c]
                can_push = push_tile in WALKABLE_IDS
                print(f'Block at {block_pos}: push to ({push_r}, {push_c}), tile={push_tile}, can_push={can_push}')
            else:
                print(f'Block at {block_pos}: push destination out of bounds')
