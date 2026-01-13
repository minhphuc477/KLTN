"""
COMPLETE MANUAL MAPPING based on careful VGLC and Graph analysis.

Key observations from VGLC analysis:
1. Room (1,0) has STAIR = START (graph node 7 's')
2. Room (5,1) has MONSTERS in boss pattern = BOSS (graph node 15 'b')
3. Room (3,5) has door only at NORTH, blocks = TRIFORCE (graph node 11 't')
4. Room (2,5) has MON+ELEMENT = high enemy room (connected to boss path)

Graph path analysis: START -> TRIFORCE
7 -> 8 -> 4 -> 3 -> 13 -> 1 -> 17 -> 15 -> 11

VGLC connected path from START:
(1,0) -> (2,0) -> (2,1) -> (2,2) -> (1,2) OR (3,2)...

The key insight: We need to trace BOTH graph connectivity AND VGLC doors
"""
import networkx as nx
import numpy as np
from collections import deque
from pathlib import Path

# Parse graph
G = nx.drawing.nx_pydot.read_dot('Data/The Legend of Zelda/Graph Processed/LoZ_1.dot')
UG = G.to_undirected()

# Load VGLC with complete analysis
vglc_file = 'Data/The Legend of Zelda/Processed/tloz1_1.txt'
with open(vglc_file) as f:
    lines = [line.rstrip('\n') for line in f]

ROOM_WIDTH = 11
ROOM_HEIGHT = 16

# Extract all VGLC room data
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
            has_element = np.any(np.isin(room_grid, ['P', 'O']))
            block_count = np.sum(room_grid == 'B')
            
            has_north = 'D' in ''.join(room_grid[0:2, :].flatten())
            has_south = 'D' in ''.join(room_grid[-2:, :].flatten())
            has_west = 'D' in ''.join(room_grid[:, 0:2].flatten())
            has_east = 'D' in ''.join(room_grid[:, -2:].flatten())
            
            vglc_rooms[(slot_r, slot_c)] = {
                'grid': room_grid,
                'stair': has_stair,
                'monster': has_monster,
                'element': has_element,
                'blocks': block_count,
                'doors': {'N': has_north, 'S': has_south, 'W': has_west, 'E': has_east}
            }

# Manual mapping based on analysis
# Key anchors:
# - Graph 7 (s) -> VGLC (1,0) [STAIR]
# - Graph 15 (b) -> VGLC (5,1) [BOSS - many monsters, blocks]
# - Graph 11 (t) -> VGLC ??? [TRIFORCE - need to find]

# Graph edge analysis for critical nodes:
print("=== GRAPH PATH ANALYSIS ===")
print("Path from START (7) to TRIFORCE (11):")
path = nx.shortest_path(UG, '7', '11')
print(f"  {' -> '.join(path)}")

print("\nDetailed path with labels:")
for i, node in enumerate(path):
    label = G.nodes[node].get('label', '').strip('"')
    if i < len(path) - 1:
        # Check edge to next node
        next_node = path[i+1]
        for u, v, data in G.edges(data=True):
            if (u == node and v == next_node) or (u == next_node and v == node):
                edge_label = data.get('label', '').strip('"')
                print(f"  {node} '{label}' --[{edge_label or 'open'}]--> ", end="")
                break
    else:
        print(f"  {node} '{label}'")

# Try to build mapping by matching path
print("\n=== MAPPING BY PATH MATCHING ===")
# We know the start and need to trace through

# Build VGLC adjacency graph
vglc_graph = nx.Graph()
for pos in vglc_rooms:
    vglc_graph.add_node(pos)
    r, c = pos
    info = vglc_rooms[pos]
    # Add edges based on doors
    if info['doors']['N'] and (r-1, c) in vglc_rooms:
        vglc_graph.add_edge(pos, (r-1, c), direction='N')
    if info['doors']['S'] and (r+1, c) in vglc_rooms:
        vglc_graph.add_edge(pos, (r+1, c), direction='S')
    if info['doors']['W'] and (r, c-1) in vglc_rooms:
        vglc_graph.add_edge(pos, (r, c-1), direction='W')
    if info['doors']['E'] and (r, c+1) in vglc_rooms:
        vglc_graph.add_edge(pos, (r, c+1), direction='E')

print(f"VGLC graph: {vglc_graph.number_of_nodes()} nodes, {vglc_graph.number_of_edges()} edges")

# Find connected components in VGLC
components = list(nx.connected_components(vglc_graph))
print(f"\nVGLC connected components: {len(components)}")
for i, comp in enumerate(components):
    print(f"  Component {i}: {sorted(comp)}")

# The main component should contain START
main_component = None
for comp in components:
    if (1, 0) in comp:
        main_component = comp
        break

print(f"\nMain component (with START): {sorted(main_component)}")

# Find all paths in VGLC graph from START
print("\n=== VGLC PATHS FROM START ===")
start_vglc = (1, 0)

# BFS to find all reachable rooms and their distances
vglc_distances = {}
queue = deque([(start_vglc, 0)])
visited_vglc = {start_vglc}
while queue:
    pos, dist = queue.popleft()
    vglc_distances[pos] = dist
    for neighbor in vglc_graph.neighbors(pos):
        if neighbor not in visited_vglc:
            visited_vglc.add(neighbor)
            queue.append((neighbor, dist + 1))

print("VGLC rooms by distance from START:")
by_dist = {}
for pos, dist in sorted(vglc_distances.items(), key=lambda x: x[1]):
    if dist not in by_dist:
        by_dist[dist] = []
    by_dist[dist].append(pos)
for dist in sorted(by_dist.keys()):
    rooms = by_dist[dist]
    room_info = []
    for pos in rooms:
        info = vglc_rooms[pos]
        attrs = []
        if info['stair']: attrs.append('STAIR')
        if info['monster']: attrs.append('MON')
        if info['element']: attrs.append('ELEM')
        if info['blocks'] > 10: attrs.append(f'B={info["blocks"]}')
        room_info.append(f"{pos}{attrs if attrs else ''}")
    print(f"  Distance {dist}: {room_info}")

# Now the key question: which VGLC rooms are NOT reachable via doors?
unreachable = set(vglc_rooms.keys()) - visited_vglc
print(f"\nVGLC rooms NOT reachable via doors from START: {sorted(unreachable)}")
for pos in sorted(unreachable):
    info = vglc_rooms[pos]
    doors = [d for d,v in info['doors'].items() if v]
    print(f"  {pos}: doors={doors}, MON={info['monster']}, B={info['blocks']}")
