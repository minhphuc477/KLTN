"""Test room-to-node mapping."""
import sys
sys.path.insert(0, '.')

import importlib.util
spec = importlib.util.spec_from_file_location('adapter', 'data/adapter.py')
adapter_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter_mod)

IntelligentDataAdapter = adapter_mod.IntelligentDataAdapter
adapter = IntelligentDataAdapter('Data/The Legend of Zelda')

# Check dungeon 1 specifically
graph = adapter.parse_dot_graph('Data/The Legend of Zelda/Graph Processed/LoZ_1.dot')
rooms = adapter.extract_rooms_simple('Data/The Legend of Zelda/Processed/tloz1_1.txt')

print(f'Graph has {graph.number_of_nodes()} nodes')
print(f'Extracted {len(rooms)} rooms')

print('\nGraph nodes with start/triforce:')
for node, attrs in graph.nodes(data=True):
    is_s = attrs.get('is_start')
    has_t = attrs.get('has_triforce')
    if is_s or has_t:
        print(f'  Node {node}: is_start={is_s}, has_triforce={has_t}')

print('\nExtracted room IDs:')
for room_id, grid in rooms:
    print(f'  Room {room_id}: shape {grid.shape}')

# The problem: graph has nodes 0-18 (19 nodes)
# But we extracted 12 rooms with IDs 0-11
# The START node (7) and TRIFORCE node (11) won't match extracted rooms!
