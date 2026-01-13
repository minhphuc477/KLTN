"""Debug D4."""
from vglc_first_adapter import VGLCFirstAdapter, SEMANTIC_PALETTE
import numpy as np

adapter = VGLCFirstAdapter('Data')
vglc_rooms = adapter._extract_vglc_rooms('Data/The Legend of Zelda/Processed/tloz4_1.txt')

# Check for STAIR
stair_rooms = [(pos, info) for pos, info in vglc_rooms.items() if info['has_stair']]
print('STAIR rooms:', [(pos, sum(info['doors'].values())) for pos, info in stair_rooms])

# Find dead-end rooms (potential TRIFORCE candidates)
print('\nDead-end rooms:')
for pos, info in vglc_rooms.items():
    door_count = sum(info['doors'].values())
    if door_count == 1:
        print(f'  {pos}: doors={info["doors"]}, has_stair={info["has_stair"]}')

# Check _match_semantics output
import networkx as nx
dot_graph = nx.drawing.nx_pydot.read_dot('Data/The Legend of Zelda/Graph Processed/LoZ_4.dot')
semantic_mapping = adapter._match_semantics(vglc_rooms, dot_graph)
print('\nSemantic mapping:')
for pos, sem in semantic_mapping.items():
    print(f'  {pos}: {sem}')
