"""Debug room 7 connectivity"""
import sys
sys.path.insert(0, 'Data')
import numpy as np
from adapter import IntelligentDataAdapter, SEMANTIC_PALETTE, ID_TO_NAME
from stitcher import DungeonStitcher

adapter = IntelligentDataAdapter('Data/The Legend of Zelda')
dungeon = adapter.process_dungeon(
    'Data/The Legend of Zelda/Processed/tloz1_1.txt',
    'Data/The Legend of Zelda/Graph Processed/LoZ_1.dot',
    'zelda_1_quest1'
)

stitcher = DungeonStitcher()
result = stitcher.stitch(dungeon)

# Find room 7 (START) in the stitched grid
room7_bounds = result.room_bounds.get('7')
print(f'Room 7 bounds: {room7_bounds}')

# Get all edges from node 7
print('\nGraph edges from node 7:')
for u, v, data in dungeon.graph.edges(data=True):
    if u == 7 or v == 7:
        label = data.get('label', '')
        print(f'  {u} <-> {v}, label="{label}"')

# Show room 7 grid
if '7' in dungeon.rooms:
    room7 = dungeon.rooms['7']
    print(f'\nRoom 7 semantic grid ({room7.grid.shape}):')
    for row in room7.grid:
        line = ''
        for tile in row:
            name = ID_TO_NAME.get(tile, '?')
            line += name[:1]
        print(f'  {line}')
    
    # Count door tiles in room
    door_count = sum(
        np.sum(room7.grid == did) 
        for did in [10, 11, 12, 13, 14, 15]
    )
    print(f'Door tiles in room 7: {door_count}')

# Check layout positions
print('\nLayout positions:')
for rid, room in dungeon.rooms.items():
    print(f'  Room {rid}: position {room.position}')

# Check stitcher layout
print('\nStitcher layout mapping:')
for node_id, pos in sorted(stitcher.layout.items()):
    print(f'  Node {node_id} -> layout {pos}')

# Check room bounds
print('\nRoom bounds in global grid:')
for rid, bounds in sorted(result.room_bounds.items(), key=lambda x: int(x[0])):
    r1, c1, r2, c2 = bounds
    print(f'  Room {rid}: rows {r1}-{r2}, cols {c1}-{c2}')
