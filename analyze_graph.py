"""
Analyze the graph and VGLC file to understand the correct room mapping.
"""
import networkx as nx
from pathlib import Path

# Load the graph
graph_file = Path('Data/The Legend of Zelda/Graph Processed/LoZ_1.dot')
G = nx.drawing.nx_pydot.read_dot(str(graph_file))

print('=== GRAPH ANALYSIS ===')
print(f'Total nodes: {len(G.nodes())}')
print(f'Total edges: {len(G.edges())}')

print('\nGraph nodes and their labels:')
for node in sorted(G.nodes(), key=lambda x: int(x)):
    attrs = G.nodes[node]
    label = attrs.get('label', '""').strip('"')
    print(f'  Node {node}: label="{label}"')

print('\n=== VGLC ANALYSIS ===')
# Load VGLC file
vglc_file = Path('Data/The Legend of Zelda/Processed/tloz1_1.txt')
with open(vglc_file) as f:
    lines = [line.rstrip('\n') for line in f]

ROOM_WIDTH = 11
ROOM_HEIGHT = 16

total_rows = len(lines)
total_cols = max(len(l) for l in lines)

print(f'VGLC dimensions: {total_rows} rows x {total_cols} cols')

# Find rooms and extract their content
rooms_found = []
for slot_r in range(6):
    for slot_c in range(6):
        start_row = slot_r * ROOM_HEIGHT
        start_col = slot_c * ROOM_WIDTH
        
        # Extract room content
        room_chars = set()
        has_content = False
        for r in range(start_row, min(start_row + ROOM_HEIGHT, total_rows)):
            line = lines[r] if r < len(lines) else ''
            for c in range(start_col, start_col + ROOM_WIDTH):
                char = line[c] if c < len(line) else '-'
                if char != '-':
                    has_content = True
                    room_chars.add(char)
        
        if has_content:
            # Determine room features
            features = []
            if 'S' in room_chars:
                features.append('START')
            if 'M' in room_chars:
                features.append('MONSTER')
            if 'B' in room_chars:
                features.append('BLOCK')
            if 'P' in room_chars:
                features.append('ELEMENT')
            if 'O' in room_chars:
                features.append('ELEM+FLOOR')
            if 'I' in room_chars:
                features.append('ELEM+BLOCK')
            
            rooms_found.append({
                'slot': (slot_r, slot_c),
                'chars': room_chars,
                'features': features
            })
            
            print(f'  Slot ({slot_r}, {slot_c}): chars={sorted(room_chars)}, features={features}')

print(f'\nTotal VGLC rooms: {len(rooms_found)}')

# Look for START room
print('\n=== FINDING KEY ROOMS ===')
for room in rooms_found:
    if 'START' in room['features']:
        print(f'  START room at slot {room["slot"]}')
