"""
GREEDY BFS MATCHING: Match graph nodes to VGLC rooms by distance from START + content.

Strategy:
1. Both START: Graph node 7 -> VGLC (1,0)
2. BFS both graphs simultaneously
3. At each distance level, match nodes to rooms by content similarity
4. Graph nodes without VGLC match become "ghost" rooms
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

# Build VGLC room data and graph
vglc_rooms = {}
vglc_graph = nx.Graph()

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
            
            vglc_graph.add_node((slot_r, slot_c))
            
            # Add edges for doors
            if has_north and (slot_r-1, slot_c) in vglc_rooms:
                vglc_graph.add_edge((slot_r, slot_c), (slot_r-1, slot_c))
            if has_south and (slot_r+1, slot_c) in vglc_rooms:
                vglc_graph.add_edge((slot_r, slot_c), (slot_r+1, slot_c))
            if has_west and (slot_r, slot_c-1) in vglc_rooms:
                vglc_graph.add_edge((slot_r, slot_c), (slot_r, slot_c-1))
            if has_east and (slot_r, slot_c+1) in vglc_rooms:
                vglc_graph.add_edge((slot_r, slot_c), (slot_r, slot_c+1))

# Add remaining edges (need to do second pass)
for pos in vglc_rooms:
    r, c = pos
    info = vglc_rooms[pos]
    if info['doors']['N'] and (r-1, c) in vglc_rooms:
        vglc_graph.add_edge(pos, (r-1, c))
    if info['doors']['S'] and (r+1, c) in vglc_rooms:
        vglc_graph.add_edge(pos, (r+1, c))
    if info['doors']['W'] and (r, c-1) in vglc_rooms:
        vglc_graph.add_edge(pos, (r, c-1))
    if info['doors']['E'] and (r, c+1) in vglc_rooms:
        vglc_graph.add_edge(pos, (r, c+1))

# BFS distances from START
def bfs_distances(graph, start):
    distances = {}
    queue = deque([(start, 0)])
    visited = {start}
    while queue:
        node, dist = queue.popleft()
        distances[node] = dist
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return distances

# Get distances
graph_distances = bfs_distances(UG, '7')  # From START node
vglc_distances = bfs_distances(vglc_graph, (1, 0))  # From VGLC START

# Group by distance
def group_by_distance(distances):
    groups = {}
    for node, dist in distances.items():
        if dist not in groups:
            groups[dist] = []
        groups[dist].append(node)
    return groups

graph_by_dist = group_by_distance(graph_distances)
vglc_by_dist = group_by_distance(vglc_distances)

print("=== DISTANCE COMPARISON ===")
max_dist = max(max(graph_by_dist.keys()), max(vglc_by_dist.keys()))
for d in range(max_dist + 1):
    g_nodes = graph_by_dist.get(d, [])
    v_rooms = vglc_by_dist.get(d, [])
    print(f"Distance {d}: {len(g_nodes)} graph nodes, {len(v_rooms)} VGLC rooms")
    
# Content scoring function
def content_score(graph_node, vglc_pos):
    """Score how well a graph node matches a VGLC room."""
    node_label = G.nodes[graph_node].get('label', '').strip('"')
    info = vglc_rooms[vglc_pos]
    
    score = 0
    
    # START match
    if 's' in node_label and info['stair']:
        score += 1000
    
    # BOSS match (many monsters in complex pattern)
    if 'b' in node_label and node_label != 'b':  # 'b' alone is boss
        pass
    if node_label == 'b':  # Boss room
        if info['monster'] and info['blocks'] > 15:
            score += 500
    
    # Triforce - typically at end of dungeon, after boss
    if 't' in node_label:
        # Triforce room is usually sparse, no monsters
        if not info['monster'] and info['blocks'] > 5:
            score += 500
    
    # Enemy match
    if 'e' in node_label and info['monster']:
        score += 10
    
    # Element rooms (puzzle rooms often have hazards)
    if 'p' in node_label and info['element']:
        score += 20
    
    # Key item rooms often have blocks
    if 'I' in node_label and info['blocks'] > 0:
        score += 5
    
    return score

# Greedy matching by distance level
print("\n=== GREEDY BFS MATCHING ===")
mapping = {}  # graph_node -> vglc_pos
reverse = {}  # vglc_pos -> graph_node
ghost_nodes = []

for d in range(max_dist + 1):
    g_nodes = sorted(graph_by_dist.get(d, []), key=lambda x: int(x))
    v_rooms = sorted(vglc_by_dist.get(d, []))
    
    if not g_nodes:
        continue
    
    # Available VGLC rooms at this distance
    available = [r for r in v_rooms if r not in reverse]
    
    # Score all pairs
    scores = []
    for node in g_nodes:
        if node in mapping:
            continue
        for room in available:
            score = content_score(node, room)
            scores.append((score, node, room))
    
    # Sort by score (highest first) and assign greedily
    scores.sort(reverse=True)
    assigned_nodes = set()
    assigned_rooms = set()
    
    for score, node, room in scores:
        if node in assigned_nodes or room in assigned_rooms:
            continue
        if node in mapping:
            continue
            
        mapping[node] = room
        reverse[room] = node
        assigned_nodes.add(node)
        assigned_rooms.add(room)
        
        label = G.nodes[node].get('label', '').strip('"')
        info = vglc_rooms[room]
        attrs = []
        if info['stair']: attrs.append('STAIR')
        if info['monster']: attrs.append('MON')
        if info['blocks'] > 10: attrs.append(f'B={info["blocks"]}')
        print(f"  d={d}: Node {node} '{label}' -> VGLC {room} {attrs} (score={score})")
    
    # Mark unmatched nodes as ghost
    for node in g_nodes:
        if node not in mapping:
            ghost_nodes.append(node)
            label = G.nodes[node].get('label', '').strip('"')
            print(f"  d={d}: Node {node} '{label}' -> GHOST (no VGLC match)")

# Summary
print(f"\n=== FINAL MAPPING SUMMARY ===")
print(f"Mapped: {len(mapping)}/19 graph nodes")
print(f"Ghost nodes: {ghost_nodes}")

print("\nComplete mapping:")
for node in sorted(G.nodes(), key=lambda x: int(x)):
    label = G.nodes[node].get('label', '').strip('"')
    if node in mapping:
        pos = mapping[node]
        print(f"  {node:2s} '{label:6s}' -> VGLC {pos}")
    else:
        print(f"  {node:2s} '{label:6s}' -> GHOST")
