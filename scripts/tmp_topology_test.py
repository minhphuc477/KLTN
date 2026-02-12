import sys
sys.path.insert(0, '.')
from src.data.zelda_core import ZeldaDungeonAdapter
adapter = ZeldaDungeonAdapter('Data/The Legend of Zelda')
D = adapter.load_dungeon(1,1)
print('graph_nodes=', len(D.graph.nodes()))
print('graph_edges=', len(D.graph.edges()))
print('sample_nodes=', list(D.graph.nodes(data=True))[:6])
S = adapter.stitch_dungeon(D)
print('global_grid_shape=', S.global_grid.shape)
print('start_global=', S.start_global, 'triforce_global=', S.triforce_global)
import numpy as np
if S.start_global:
    r,c = S.start_global
    print('start_tile=', S.global_grid[r,c])
    print('start_room_slice:')
    print(S.global_grid[r-3:r+4,c-3:c+4])
print('some_edges=', list(D.graph.edges(data=True))[:6])
print('\nroom_to_node sample (first 6):')
print(dict(list(S.room_to_node.items())[:6]))
print('\nnode_to_room sample (first 6):')
print(dict(list(S.node_to_room.items())[:6]))
