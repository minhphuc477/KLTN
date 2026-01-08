"""Debug stitcher directly."""
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

# Manual stitch to debug
import networkx as nx

ROOM_HEIGHT = 16
ROOM_WIDTH = 11
PADDING = 1

# Get layout
graph = dungeon.graph

# Find start node
start_node = None
for node, attrs in graph.nodes(data=True):
    if attrs.get('is_start', False):
        start_node = node
        break
if start_node is None:
    start_node = min(graph.nodes())

print(f"Start node: {start_node}")

# BFS for layout
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

# Normalize
if positions:
    min_r = min(p[0] for p in positions.values())
    min_c = min(p[1] for p in positions.values())
    positions = {k: (v[0] - min_r, v[1] - min_c) for k, v in positions.items()}

print(f"Layout positions: {positions}")
print(f"Room 7 layout: {positions.get(7, 'NOT IN LAYOUT!')}")

# Check if room 7 is in dungeon.rooms
print(f"\nRooms in dungeon: {list(dungeon.rooms.keys())}")

# Build global grid
max_row = max(p[0] for p in positions.values())
max_col = max(p[1] for p in positions.values())
grid_height = (max_row + 1) * (ROOM_HEIGHT + PADDING)
grid_width = (max_col + 1) * (ROOM_WIDTH + PADDING)

print(f"\nGrid dimensions: {grid_height}x{grid_width}")

global_grid = np.full((grid_height, grid_width), SEMANTIC_PALETTE['VOID'], dtype=np.int64)

# Place room 7 specifically
if 7 in positions:
    layout_row, layout_col = positions[7]
    room_key = str(7)
    
    if room_key in dungeon.rooms:
        room = dungeon.rooms[room_key]
        room_grid = room.grid
        
        r_offset = layout_row * (ROOM_HEIGHT + PADDING)
        c_offset = layout_col * (ROOM_WIDTH + PADDING)
        
        rh, rw = room_grid.shape
        r_end = min(r_offset + rh, grid_height)
        c_end = min(c_offset + rw, grid_width)
        
        print(f"\nRoom 7 placement:")
        print(f"  layout: ({layout_row}, {layout_col})")
        print(f"  offset: ({r_offset}, {c_offset})")
        print(f"  room shape: {rh}x{rw}")
        print(f"  bounds: r_end={r_end}, c_end={c_end}")
        
        actual_rh = r_end - r_offset
        actual_rw = c_end - c_offset
        
        print(f"  actual dims: {actual_rh}x{actual_rw}")
        
        # Check START in room before placement
        start_in_room = room_grid == SEMANTIC_PALETTE['START']
        print(f"  START count in room_grid before placement: {np.sum(start_in_room)}")
        
        # Do the placement
        global_grid[r_offset:r_end, c_offset:c_end] = room_grid[:actual_rh, :actual_rw]
        
        # Check START after placement
        start_in_global = global_grid == SEMANTIC_PALETTE['START']
        print(f"  START count in global_grid after placement: {np.sum(start_in_global)}")
        
        # Check specific position
        expected_start_r = r_offset + 10
        expected_start_c = c_offset + 24
        print(f"  Expected START at: ({expected_start_r}, {expected_start_c})")
        
        if expected_start_c < c_end:
            tile = global_grid[expected_start_r, expected_start_c]
            print(f"  Tile at expected START: {tile} ({ID_TO_NAME.get(tile, 'UNKNOWN')})")
        else:
            print(f"  Expected START column {expected_start_c} is >= c_end {c_end} - TRUNCATED!")
    else:
        print(f"  Room key '7' not in dungeon.rooms!")
