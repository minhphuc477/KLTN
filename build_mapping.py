"""
TOPOLOGY-BASED MAPPING: Match graph nodes to VGLC rooms by adjacency.

Algorithm:
1. START node 7 -> VGLC (1,0) [only room with STAIR]
2. BFS from START: for each graph neighbor, find matching adjacent VGLC room
3. Key constraint: graph edges must match VGLC adjacencies
4. Create ghost rooms for nodes that don't have VGLC representation
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

# Extract VGLC rooms
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
            vglc_rooms[(slot_r, slot_c)] = room_grid

# Get VGLC neighbors
def get_vglc_neighbors(pos):
    r, c = pos
    neighbors = []
    for nr, nc in [(r-1,c), (r+1,c), (r,c-1), (r,c+1)]:
        if (nr, nc) in vglc_rooms:
            neighbors.append((nr, nc))
    return neighbors

# Get graph neighbors
def get_graph_neighbors(node):
    successors = list(G.successors(str(node)))
    predecessors = list(G.predecessors(str(node)))
    return sorted(set(successors + predecessors), key=lambda x: int(x))

# BFS mapping from START
print("=== TOPOLOGY-BASED MAPPING ===")
mapping = {}  # graph_node -> vglc_pos
reverse_mapping = {}  # vglc_pos -> graph_node

# Step 1: Map START
start_node = '7'  # Graph node with 's' label
start_vglc = (1, 0)  # Only VGLC room with STAIR
mapping[start_node] = start_vglc
reverse_mapping[start_vglc] = start_node
print(f"Step 1: Map START node {start_node} -> VGLC {start_vglc}")

# Step 2: BFS propagation
queue = deque([start_node])
visited = {start_node}

while queue:
    current = queue.popleft()
    current_vglc = mapping.get(current)
    
    if current_vglc is None:
        continue
    
    # Get graph neighbors
    graph_neighbors = get_graph_neighbors(current)
    
    # Get VGLC neighbors of current position
    vglc_neighbors = get_vglc_neighbors(current_vglc)
    
    # Available VGLC positions (not yet mapped)
    available_vglc = [pos for pos in vglc_neighbors if pos not in reverse_mapping]
    
    # Unmapped graph neighbors
    unmapped_graph = [n for n in graph_neighbors if n not in visited]
    
    # Try to match unmapped graph neighbors to available VGLC positions
    for graph_neighbor in unmapped_graph:
        neighbor_label = G.nodes[graph_neighbor].get('label', '').strip('"')
        
        if available_vglc:
            # Assign the first available VGLC position
            # In a more sophisticated approach, we'd score by content
            vglc_pos = available_vglc.pop(0)
            mapping[graph_neighbor] = vglc_pos
            reverse_mapping[vglc_pos] = graph_neighbor
            visited.add(graph_neighbor)
            queue.append(graph_neighbor)
            print(f"  Map node {graph_neighbor} '{neighbor_label}' -> VGLC {vglc_pos}")
        else:
            # No available adjacent VGLC - this neighbor needs ghost room
            visited.add(graph_neighbor)
            queue.append(graph_neighbor)
            print(f"  Node {graph_neighbor} '{neighbor_label}' - NO ADJACENT VGLC AVAILABLE (ghost)")

# Show results
print(f"\n=== MAPPING RESULTS ===")
print(f"Mapped: {len(mapping)}/{len(G.nodes())} graph nodes")
print(f"VGLC rooms used: {len(reverse_mapping)}/{len(vglc_rooms)}")

# Check unmapped
unmapped_nodes = set(G.nodes()) - set(mapping.keys())
print(f"\nUnmapped graph nodes: {sorted(unmapped_nodes, key=lambda x: int(x))}")
for node in sorted(unmapped_nodes, key=lambda x: int(x)):
    label = G.nodes[node].get('label', '').strip('"')
    neighbors = get_graph_neighbors(node)
    print(f"  {node} '{label}' - neighbors: {neighbors}")

unused_vglc = set(vglc_rooms.keys()) - set(reverse_mapping.keys())
print(f"\nUnused VGLC rooms: {sorted(unused_vglc)}")

# Final mapping table
print("\n=== FINAL MAPPING TABLE ===")
print("Graph Node -> VGLC Position -> Label")
for node in sorted(G.nodes(), key=lambda x: int(x)):
    label = G.nodes[node].get('label', '').strip('"')
    vglc_pos = mapping.get(node, "GHOST")
    print(f"  {node:2s} -> {str(vglc_pos):8s} -> '{label}'")
