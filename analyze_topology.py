"""
Match VGLC rooms to graph nodes using topology.

Key insight: The graph defines connectivity, VGLC defines spatial layout.
We need to find a consistent mapping where adjacent VGLC rooms match graph edges.
"""
import networkx as nx
import numpy as np
from pathlib import Path

# Parse the graph
G = nx.drawing.nx_pydot.read_dot('Data/The Legend of Zelda/Graph Processed/LoZ_1.dot')

# Load VGLC
vglc_file = 'Data/The Legend of Zelda/Processed/tloz1_1.txt'
with open(vglc_file) as f:
    lines = [line.rstrip('\n') for line in f]

ROOM_WIDTH = 11
ROOM_HEIGHT = 16

# Extract rooms with their content
print("=== VGLC ROOM ANALYSIS ===")
vglc_rooms = {}
for slot_r in range(6):
    for slot_c in range(6):
        start_row = slot_r * ROOM_HEIGHT
        start_col = slot_c * ROOM_WIDTH
        
        # Extract room content
        room_chars = []
        for r in range(start_row, min(start_row + ROOM_HEIGHT, len(lines))):
            row_chars = []
            for c in range(start_col, start_col + ROOM_WIDTH):
                char = lines[r][c] if r < len(lines) and c < len(lines[r]) else '-'
                row_chars.append(char)
            room_chars.append(row_chars)
        
        room_grid = np.array(room_chars)
        
        # Check if room (not gap)
        if np.sum(room_grid == '-') < room_grid.size * 0.7:
            vglc_rooms[(slot_r, slot_c)] = room_grid
            
            # Analyze content
            has_stair = np.any(room_grid == 'S')
            has_monster = np.any(room_grid == 'M')
            has_element = np.any(np.isin(room_grid, ['P', 'O']))
            block_count = np.sum(room_grid == 'B')
            
            # Check for doors on each side
            top_row = ''.join(room_grid[0, :])
            bottom_row = ''.join(room_grid[-1, :])
            left_col = ''.join(room_grid[:, 0])
            right_col = ''.join(room_grid[:, -1])
            
            has_top_door = 'D' in top_row or 'D' in ''.join(room_grid[1, :])
            has_bottom_door = 'D' in bottom_row or 'D' in ''.join(room_grid[-2, :])
            has_left_door = 'D' in left_col or 'D' in ''.join(room_grid[:, 1])
            has_right_door = 'D' in right_col or 'D' in ''.join(room_grid[:, -2])
            
            doors = []
            if has_top_door: doors.append('N')
            if has_bottom_door: doors.append('S')
            if has_left_door: doors.append('W')
            if has_right_door: doors.append('E')
            
            print(f"  ({slot_r},{slot_c}): {'STAIR' if has_stair else '     '} "
                  f"{'MON' if has_monster else '   '} {'ELEM' if has_element else '    '} "
                  f"B={block_count:2d} doors={','.join(doors) or 'none'}")

print(f"\nTotal VGLC rooms: {len(vglc_rooms)}")

# Create visual map
print("\n=== VGLC SPATIAL LAYOUT ===")
print("Room positions in the dungeon grid:")
for r in range(6):
    row_str = ""
    for c in range(6):
        if (r, c) in vglc_rooms:
            row_str += f"[{r},{c}] "
        else:
            row_str += "  .   "
    print(row_str)

# Find adjacencies
print("\n=== VGLC ADJACENCIES (rooms that share edges) ===")
for pos in sorted(vglc_rooms.keys()):
    r, c = pos
    neighbors = []
    if (r-1, c) in vglc_rooms: neighbors.append(f"({r-1},{c})N")
    if (r+1, c) in vglc_rooms: neighbors.append(f"({r+1},{c})S")
    if (r, c-1) in vglc_rooms: neighbors.append(f"({r},{c-1})W")
    if (r, c+1) in vglc_rooms: neighbors.append(f"({r},{c+1})E")
    print(f"  ({r},{c}) -> {neighbors}")

# Now show graph structure
print("\n=== GRAPH STRUCTURE ===")
print("Nodes and their labels:")
for node in sorted(G.nodes(), key=lambda x: int(x)):
    label = G.nodes[node].get('label', '').strip('"')
    neighbors = list(G.successors(node)) + list(G.predecessors(node))
    neighbors = sorted(set(neighbors), key=lambda x: int(x))
    print(f"  {node}: '{label}' -> neighbors: {neighbors}")

# The key insight: START room in VGLC is at (1,0)
# Graph node 7 has label 's' (start)
# So graph node 7 should map to VGLC (1,0)
print("\n=== PROPOSED MAPPING ===")
print("Based on START position and adjacency propagation:")
print("  Graph node 7 (s=start) -> VGLC (1,0) [has STAIR]")
