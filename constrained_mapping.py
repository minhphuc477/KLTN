"""
CONSTRAINED MAPPING: Use door directions to match graph edges to VGLC adjacencies.

Key insight: If graph node A connects to node B, and A is at VGLC pos (r,c),
then B must be at an adjacent VGLC position where:
- If A has SOUTH door and B is adjacent south -> B at (r+1, c)
- If A has NORTH door and B is adjacent north -> B at (r-1, c)
etc.

This creates constraints that must be satisfied for valid mapping.
"""
import networkx as nx
import numpy as np
from collections import deque
from pathlib import Path
from itertools import permutations

# Parse graph
G = nx.drawing.nx_pydot.read_dot('Data/The Legend of Zelda/Graph Processed/LoZ_1.dot')
UG = G.to_undirected()

# Load VGLC
vglc_file = 'Data/The Legend of Zelda/Processed/tloz1_1.txt'
with open(vglc_file) as f:
    lines = [line.rstrip('\n') for line in f]

ROOM_WIDTH = 11
ROOM_HEIGHT = 16

# Extract VGLC rooms with door analysis
vglc_rooms = {}
for slot_r in range(6):
    for slot_c in range(6):
        start_row = slot_r * ROOM_HEIGHT
        start_col = slot_c * ROOM_WIDTH
        
        room_chars = []
        for r in range(start_row, min(start_row + ROOM_HEIGHT, len(lines))):
            row_chars = []
            for c in range(start_col, start_col + ROOM_WIDTH):
                char = lines[r][c] if r < len(lines) and c < len(lines[r]) else '-'
                row_chars.append(char)
            room_chars.append(row_chars)
        
        room_grid = np.array(room_chars)
        if np.sum(room_grid == '-') < room_grid.size * 0.7:
            has_stair = np.any(room_grid == 'S')
            has_monster = np.any(room_grid == 'M')
            
            has_north = 'D' in ''.join(room_grid[0:2, :].flatten())
            has_south = 'D' in ''.join(room_grid[-2:, :].flatten())
            has_west = 'D' in ''.join(room_grid[:, 0:2].flatten())
            has_east = 'D' in ''.join(room_grid[:, -2:].flatten())
            
            vglc_rooms[(slot_r, slot_c)] = {
                'grid': room_grid,
                'stair': has_stair,
                'monster': has_monster,
                'doors': {'N': has_north, 'S': has_south, 'W': has_west, 'E': has_east}
            }

def vglc_neighbors_with_direction(pos):
    """Return dict of direction -> neighbor position"""
    r, c = pos
    neighbors = {}
    if (r-1, c) in vglc_rooms:
        neighbors['N'] = (r-1, c)
    if (r+1, c) in vglc_rooms:
        neighbors['S'] = (r+1, c)
    if (r, c-1) in vglc_rooms:
        neighbors['W'] = (r, c-1)
    if (r, c+1) in vglc_rooms:
        neighbors['E'] = (r, c+1)
    return neighbors

# Get graph info
print("=== CONSTRAINT ANALYSIS ===")

# START: Node 7 must be at VGLC (1,0) - only room with STAIR
print("\nFixed mapping: Node 7 -> (1,0) [START with STAIR]")

# Check constraints from START
start_vglc = (1, 0)
start_doors = vglc_rooms[start_vglc]['doors']
print(f"  VGLC (1,0) has doors: {[d for d,v in start_doors.items() if v]}")

start_vglc_neighbors = vglc_neighbors_with_direction(start_vglc)
print(f"  VGLC (1,0) neighbors: {start_vglc_neighbors}")

start_graph_neighbors = list(UG.neighbors('7'))
print(f"  Graph node 7 neighbors: {start_graph_neighbors}")

# Node 7 has ONLY node 8 as neighbor, and VGLC (1,0) has ONLY south door to (2,0)
print("\n=> Node 8 MUST be at (2,0)")

# Continue constraint propagation
print("\n=== CONSTRAINED BFS MAPPING ===")
mapping = {'7': (1, 0)}
reverse = {(1, 0): '7'}
queue = deque(['7'])
visited = {'7'}

while queue:
    current = queue.popleft()
    current_pos = mapping[current]
    current_info = vglc_rooms[current_pos]
    
    # Get graph neighbors
    graph_neighbors = list(UG.neighbors(current))
    
    # Get VGLC neighbors with their directions
    vglc_neighs = vglc_neighbors_with_direction(current_pos)
    
    # For each door direction in current VGLC room, try to match graph neighbor
    unmapped_graph = [n for n in graph_neighbors if n not in visited]
    available_vglc = {d: p for d, p in vglc_neighs.items() if p not in reverse}
    
    # Match based on doors
    for direction, vglc_pos in available_vglc.items():
        if not current_info['doors'][direction]:
            continue  # No door in this direction
        
        if unmapped_graph:
            # Pick first unmapped neighbor
            node = unmapped_graph.pop(0)
            mapping[node] = vglc_pos
            reverse[vglc_pos] = node
            visited.add(node)
            queue.append(node)
            
            label = G.nodes[node].get('label', '').strip('"')
            print(f"  Map node {node} '{label}' -> VGLC {vglc_pos} (via {direction} door from {current})")

# Show unmapped
print(f"\n=== RESULTS ===")
print(f"Mapped: {len(mapping)}/19 nodes")
unmapped = sorted(set(G.nodes()) - set(mapping.keys()), key=lambda x: int(x))
print(f"Unmapped nodes: {unmapped}")
for node in unmapped:
    label = G.nodes[node].get('label', '').strip('"')
    neighbors = list(UG.neighbors(node))
    mapped_neighbors = [n for n in neighbors if n in mapping]
    print(f"  {node} '{label}' - neighbors in map: {mapped_neighbors}")

unused_vglc = sorted(set(vglc_rooms.keys()) - set(reverse.keys()))
print(f"\nUnused VGLC rooms: {unused_vglc}")
