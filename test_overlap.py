"""Check for room overlap in stitcher."""
import sys
sys.path.insert(0, '.')
import importlib.util
import numpy as np

spec = importlib.util.spec_from_file_location('adapter', 'data/adapter.py')
adapter_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter_mod)

IntelligentDataAdapter = adapter_mod.IntelligentDataAdapter
SEMANTIC_PALETTE = adapter_mod.SEMANTIC_PALETTE
ID_TO_NAME = adapter_mod.ID_TO_NAME

adapter = IntelligentDataAdapter('Data/The Legend of Zelda')
dungeon = adapter.process_dungeon(
    'Data/The Legend of Zelda/Processed/tloz1_1.txt',
    'Data/The Legend of Zelda/Graph Processed/LoZ_1.dot',
    'zelda_1_quest1'
)

# Manual stitch with detailed logging
import networkx as nx

ROOM_HEIGHT = 16
ROOM_WIDTH = 11
PADDING = 1

graph = dungeon.graph
start_node = 7

# BFS layout (same as before)
positions = {start_node: (0, 0)}
visited = {start_node}
queue = [start_node]
direction_cycle = [(0, 1), (1, 0), (0, -1), (-1, 0)]

while queue:
    current = queue.pop(0)
    curr_pos = positions[current]
    neighbors = list(graph.neighbors(current))
    dir_idx = 0
    
    for neighbor in neighbors:
        if neighbor in visited:
            continue
        placed = False
        for attempt in range(8):
            dr, dc = direction_cycle[(dir_idx + attempt) % 4]
            new_pos = (curr_pos[0] + dr, curr_pos[1] + dc)
            if new_pos not in positions.values():
                positions[neighbor] = new_pos
                visited.add(neighbor)
                queue.append(neighbor)
                dir_idx += attempt + 1
                placed = True
                break
        if not placed:
            offset = 1
            for outer_attempt in range(20):
                for dr, dc in direction_cycle:
                    new_pos = (curr_pos[0] + dr * offset, curr_pos[1] + dc * offset)
                    if new_pos not in positions.values():
                        positions[neighbor] = new_pos
                        visited.add(neighbor)
                        queue.append(neighbor)
                        placed = True
                        break
                if placed:
                    break
                offset += 1

if positions:
    min_r = min(p[0] for p in positions.values())
    min_c = min(p[1] for p in positions.values())
    positions = {k: (v[0] - min_r, v[1] - min_c) for k, v in positions.items()}

max_row = max(p[0] for p in positions.values())
max_col = max(p[1] for p in positions.values())
grid_height = (max_row + 1) * (ROOM_HEIGHT + PADDING)
grid_width = (max_col + 1) * (ROOM_WIDTH + PADDING)

global_grid = np.full((grid_height, grid_width), SEMANTIC_PALETTE['VOID'], dtype=np.int64)

# Track what's written to position (27, 24)
target_r, target_c = 27, 24

print("Placing rooms in order...")
for room_id, (layout_row, layout_col) in positions.items():
    room_key = str(room_id)
    
    if room_key not in dungeon.rooms:
        continue
    
    room = dungeon.rooms[room_key]
    room_grid = room.grid
    
    r_offset = layout_row * (ROOM_HEIGHT + PADDING)
    c_offset = layout_col * (ROOM_WIDTH + PADDING)
    
    rh, rw = room_grid.shape
    r_end = min(r_offset + rh, grid_height)
    c_end = min(c_offset + rw, grid_width)
    
    actual_rh = r_end - r_offset
    actual_rw = c_end - c_offset
    
    # Check if this room overlaps target position
    if r_offset <= target_r < r_end and c_offset <= target_c < c_end:
        local_r = target_r - r_offset
        local_c = target_c - c_offset
        if local_r < actual_rh and local_c < actual_rw:
            tile = room_grid[local_r, local_c]
            print(f"Room {room_id} WRITES to ({target_r}, {target_c}): {ID_TO_NAME.get(tile, 'UNKNOWN')} (ID={tile})")
    
    global_grid[r_offset:r_end, c_offset:c_end] = room_grid[:actual_rh, :actual_rw]

print(f"\nFinal tile at ({target_r}, {target_c}): {global_grid[target_r, target_c]} ({ID_TO_NAME.get(global_grid[target_r, target_c], 'UNKNOWN')})")
print(f"Total START count: {np.sum(global_grid == SEMANTIC_PALETTE['START'])}")
