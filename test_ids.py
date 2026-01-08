"""Check IDs and trace the START injection."""
import sys
sys.path.insert(0, '.')

import importlib.util
spec = importlib.util.spec_from_file_location('adapter', 'data/adapter.py')
adapter_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter_mod)

SEMANTIC_PALETTE = adapter_mod.SEMANTIC_PALETTE
IntelligentDataAdapter = adapter_mod.IntelligentDataAdapter

print("Semantic IDs:")
print(f"  START: {SEMANTIC_PALETTE['START']}")
print(f"  STAIR: {SEMANTIC_PALETTE['STAIR']}")
print(f"  TRIFORCE: {SEMANTIC_PALETTE['TRIFORCE']}")

# Load adapter and process a room
adapter = IntelligentDataAdapter('Data/The Legend of Zelda')
rooms = adapter.extract_rooms_simple('Data/The Legend of Zelda/Processed/tloz1_1.txt')
graph = adapter.parse_dot_graph('Data/The Legend of Zelda/Graph Processed/LoZ_1.dot')

# Find room 7 (the start room)
room_7_char = None
for room_id, char_grid in rooms:
    if room_id == 7:
        room_7_char = char_grid
        break

if room_7_char is not None:
    print(f"\nRoom 7 char_grid shape: {room_7_char.shape}")
    print("Sample of room 7 characters:")
    for row in room_7_char[:5]:
        print(''.join(row))
    
    # Check for 'S' characters
    s_count = sum(1 for r in room_7_char for c in r if c == 'S')
    print(f"\n'S' characters in room 7: {s_count}")
    
    # Get node attributes for room 7
    node_attrs = {}
    if 7 in graph.nodes:
        node_data = graph.nodes[7]
        node_attrs = {
            'is_start': node_data.get('is_start', False),
            'has_triforce': node_data.get('has_triforce', False),
            'has_key': node_data.get('has_key', False),
        }
    
    print(f"\nNode 7 attrs from graph: {node_attrs}")
    
    # Now call defensive_mapper
    semantic_grid = adapter.defensive_mapper(room_7_char, 7, {}, node_attrs)
    
    # Check for START in semantic grid
    import numpy as np
    start_positions = np.where(semantic_grid == SEMANTIC_PALETTE['START'])
    stair_positions = np.where(semantic_grid == SEMANTIC_PALETTE['STAIR'])
    
    print(f"\nAfter defensive_mapper:")
    print(f"  START positions: {list(zip(start_positions[0], start_positions[1]))}")
    print(f"  STAIR positions: {list(zip(stair_positions[0], stair_positions[1]))}")
    
    # Check unique values in semantic grid
    unique = np.unique(semantic_grid)
    print(f"  Unique semantic IDs: {unique}")
