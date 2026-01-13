"""
IMPROVED MAPPING: Use degree matching + door direction analysis.

Key insight: Graph edges correspond to doors in VGLC rooms.
A room with 3 doors should map to a graph node with degree ~3.
"""
import networkx as nx
import numpy as np
from collections import deque
from pathlib import Path

# Parse graph
G = nx.drawing.nx_pydot.read_dot('Data/The Legend of Zelda/Graph Processed/LoZ_1.dot')

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
            # Analyze doors
            has_stair = np.any(room_grid == 'S')
            has_monster = np.any(room_grid == 'M')
            
            # Check doors by looking at specific rows/columns
            has_north = 'D' in ''.join(room_grid[0:2, :].flatten())
            has_south = 'D' in ''.join(room_grid[-2:, :].flatten())
            has_west = 'D' in ''.join(room_grid[:, 0:2].flatten())
            has_east = 'D' in ''.join(room_grid[:, -2:].flatten())
            
            door_count = sum([has_north, has_south, has_west, has_east])
            
            vglc_rooms[(slot_r, slot_c)] = {
                'grid': room_grid,
                'stair': has_stair,
                'monster': has_monster,
                'doors': {'N': has_north, 'S': has_south, 'W': has_west, 'E': has_east},
                'door_count': door_count
            }

# Analyze graph node degrees (undirected)
print("=== GRAPH DEGREES ===")
UG = G.to_undirected()
node_degrees = {}
for node in sorted(G.nodes(), key=lambda x: int(x)):
    degree = UG.degree(node)
    label = G.nodes[node].get('label', '').strip('"')
    node_degrees[node] = {'degree': degree, 'label': label}
    print(f"  Node {node:2s}: degree={degree}, label='{label}'")

print("\n=== VGLC DOOR COUNTS ===")
for pos in sorted(vglc_rooms.keys()):
    info = vglc_rooms[pos]
    doors = [d for d, has in info['doors'].items() if has]
    print(f"  VGLC {pos}: doors={info['door_count']} ({','.join(doors)}), "
          f"{'STAIR' if info['stair'] else ''} {'MON' if info['monster'] else ''}")

# Build degree-matched candidates
print("\n=== DEGREE-BASED MATCHING ===")
print("VGLC rooms grouped by door count:")
by_doors = {}
for pos, info in vglc_rooms.items():
    dc = info['door_count']
    if dc not in by_doors:
        by_doors[dc] = []
    by_doors[dc].append(pos)
    
for dc in sorted(by_doors.keys()):
    print(f"  {dc} doors: {by_doors[dc]}")

print("\nGraph nodes grouped by degree:")
by_degree = {}
for node, info in node_degrees.items():
    d = info['degree']
    if d not in by_degree:
        by_degree[d] = []
    by_degree[d].append(node)
    
for d in sorted(by_degree.keys()):
    labels = [f"{n}:{node_degrees[n]['label']}" for n in by_degree[d]]
    print(f"  degree {d}: {labels}")

# Compare counts
print("\n=== MISMATCH ANALYSIS ===")
for count in range(1, 5):
    vglc_count = len(by_doors.get(count, []))
    graph_count = len(by_degree.get(count, []))
    print(f"  {count} connections: VGLC has {vglc_count} rooms, Graph has {graph_count} nodes")
