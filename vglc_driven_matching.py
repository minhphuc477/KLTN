"""
ADJACENCY-CONSTRAINED MAPPING with graph validation.

The key insight: VGLC doors define PHYSICAL adjacency.
Graph edges define LOGICAL connectivity.

Strategy:
1. BFS from START using VGLC doors (physical adjacency)
2. At each step, pick the graph node that best matches the VGLC room content
3. Validate that graph connectivity is consistent with physical placement
4. Ghost rooms ONLY for graph nodes that truly have no physical room
"""
import networkx as nx
import numpy as np
from collections import deque
from pathlib import Path

# Parse graph
G = nx.drawing.nx_pydot.read_dot('Data/The Legend of Zelda/Graph Processed/LoZ_1.dot')
UG = G.to_undirected()

# Load VGLC
vglc_file = 'Data/The Legend of Zelda/Processed/tloz1_1.txt'
with open(vglc_file) as f:
    lines = [line.rstrip('\n') for line in f]

ROOM_WIDTH = 11
ROOM_HEIGHT = 16

# Build VGLC room data with adjacency
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

def get_vglc_neighbors(pos):
    """Get adjacent VGLC rooms via doors."""
    r, c = pos
    info = vglc_rooms[pos]
    neighbors = []
    if info['doors']['N'] and (r-1, c) in vglc_rooms:
        neighbors.append((r-1, c))
    if info['doors']['S'] and (r+1, c) in vglc_rooms:
        neighbors.append((r+1, c))
    if info['doors']['W'] and (r, c-1) in vglc_rooms:
        neighbors.append((r, c-1))
    if info['doors']['E'] and (r, c+1) in vglc_rooms:
        neighbors.append((r, c+1))
    return neighbors

def content_score(graph_node, vglc_pos):
    """Score content match."""
    node_label = G.nodes[graph_node].get('label', '').strip('"')
    info = vglc_rooms[vglc_pos]
    
    score = 0
    if 's' in node_label and info['stair']:
        score += 1000
    if node_label == 'b' and info['monster'] and info['blocks'] > 15:
        score += 500
    if 't' in node_label and info['blocks'] > 5:
        score += 400
    if 'e' in node_label and info['monster']:
        score += 10
    if 'p' in node_label and info['element']:
        score += 20
    return score

# BFS through VGLC, assign graph nodes
print("=== VGLC-DRIVEN MATCHING ===")
print("BFS through VGLC rooms, assigning best graph node at each step")

mapping = {}  # graph_node -> vglc_pos
reverse = {}  # vglc_pos -> graph_node
used_nodes = set()

# Start with fixed mapping
mapping['7'] = (1, 0)
reverse[(1, 0)] = '7'
used_nodes.add('7')
print(f"  START: Node 7 -> VGLC (1, 0)")

# BFS through VGLC
vglc_queue = deque([(1, 0)])
vglc_visited = {(1, 0)}

while vglc_queue:
    current_vglc = vglc_queue.popleft()
    current_node = reverse.get(current_vglc)
    
    if current_node is None:
        continue
    
    # Get VGLC neighbors (via doors)
    vglc_neighbors = get_vglc_neighbors(current_vglc)
    
    # Get graph neighbors of current node
    graph_neighbors = list(UG.neighbors(current_node))
    available_graph = [n for n in graph_neighbors if n not in used_nodes]
    
    for vglc_neighbor in vglc_neighbors:
        if vglc_neighbor in vglc_visited:
            continue
        
        vglc_visited.add(vglc_neighbor)
        vglc_queue.append(vglc_neighbor)
        
        if not available_graph:
            # No graph node available for this VGLC room - skip
            info = vglc_rooms[vglc_neighbor]
            attrs = []
            if info['stair']: attrs.append('STAIR')
            if info['monster']: attrs.append('MON')
            if info['blocks'] > 10: attrs.append(f'B={info["blocks"]}')
            print(f"  VGLC {vglc_neighbor} {attrs} -> NO GRAPH NEIGHBOR AVAILABLE")
            continue
        
        # Score all available graph neighbors against this VGLC room
        scores = [(content_score(n, vglc_neighbor), n) for n in available_graph]
        scores.sort(reverse=True)
        
        # Pick best match
        best_score, best_node = scores[0]
        
        mapping[best_node] = vglc_neighbor
        reverse[vglc_neighbor] = best_node
        used_nodes.add(best_node)
        available_graph.remove(best_node)
        
        label = G.nodes[best_node].get('label', '').strip('"')
        info = vglc_rooms[vglc_neighbor]
        attrs = []
        if info['stair']: attrs.append('STAIR')
        if info['monster']: attrs.append('MON')
        if info['blocks'] > 10: attrs.append(f'B={info["blocks"]}')
        print(f"  VGLC {vglc_neighbor} {attrs} -> Node {best_node} '{label}' (score={best_score})")

# Summary
print(f"\n=== RESULTS ===")
print(f"Mapped: {len(mapping)}/{len(G.nodes())} graph nodes")
print(f"VGLC rooms matched: {len(reverse)}/{len(vglc_rooms)}")

ghost = sorted(set(G.nodes()) - used_nodes, key=lambda x: int(x))
print(f"Ghost nodes: {ghost}")
for n in ghost:
    label = G.nodes[n].get('label', '').strip('"')
    print(f"  {n} '{label}'")

unused_vglc = sorted(set(vglc_rooms.keys()) - set(reverse.keys()))
print(f"\nUnused VGLC rooms: {unused_vglc}")
for pos in unused_vglc:
    info = vglc_rooms[pos]
    attrs = []
    if info['stair']: attrs.append('STAIR')
    if info['monster']: attrs.append('MON')
    if info['element']: attrs.append('ELEM')
    print(f"  {pos} {attrs}")

# Validation: check if graph edges match VGLC adjacency
print("\n=== VALIDATION ===")
violations = []
for node in mapping:
    vglc_pos = mapping[node]
    graph_neighbors = set(UG.neighbors(node))
    vglc_neighbors = set(get_vglc_neighbors(vglc_pos))
    
    for g_neighbor in graph_neighbors:
        if g_neighbor in mapping:
            g_neighbor_vglc = mapping[g_neighbor]
            if g_neighbor_vglc not in vglc_neighbors:
                violations.append((node, g_neighbor, vglc_pos, g_neighbor_vglc))

if violations:
    print(f"Found {len(violations)} edge violations:")
    for v in violations[:10]:
        print(f"  Graph {v[0]}-{v[1]} but VGLC {v[2]} not adjacent to {v[3]}")
else:
    print("All mapped graph edges match VGLC adjacency!")
