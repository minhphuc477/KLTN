"""Debug room doors."""
import numpy as np
from vglc_first_adapter import VGLCFirstAdapter, SEMANTIC_PALETTE
import sys

dungeon_num = int(sys.argv[1]) if len(sys.argv) > 1 else 7

adapter = VGLCFirstAdapter('Data')

# Check dungeon
vglc_rooms = adapter._extract_vglc_rooms(f'Data/The Legend of Zelda/Processed/tloz{dungeon_num}_1.txt')

print(f'Dungeon {dungeon_num} VGLC rooms:')
for pos, info in sorted(vglc_rooms.items()):
    print(f'  {pos}: doors={info["doors"]}, stair={info["has_stair"]}')

# Find START room and show its neighbors
print('\n=== START ROOM ANALYSIS ===')
for pos, info in sorted(vglc_rooms.items()):
    if info['has_stair'] and sum(info['doors'].values()) > 0:
        print(f'\nSTART room {pos}:')
        print(f'  Doors: {info["doors"]}')
        
        grid = info['grid']
        
        # Find all D tiles
        door_locs = [(r, c) for r in range(grid.shape[0]) for c in range(grid.shape[1]) if grid[r,c] == 'D']
        print(f'  Door tile positions: {door_locs}')
        
        # Check expected door positions
        print('  Expected positions:')
        print(f'    North door row 1: {"".join(grid[1, :])}')
        print(f'    South door row 14: {"".join(grid[14, :])}')
        print(f'    West door col 1: {"".join(grid[:, 1])}')
        print(f'    East door col 9: {"".join(grid[:, 9])}')
        
        # Show grid
        print('  Grid:')
        for r, row in enumerate(grid):
            print(f'    {r:2d}: {"".join(row)}')
        
        # Show neighbors
        r, c = pos
        neighbors = [(r-1, c, 'N'), (r+1, c, 'S'), (r, c-1, 'W'), (r, c+1, 'E')]
        print('\n  Neighbors:')
        for nr, nc, direction in neighbors:
            if (nr, nc) in vglc_rooms:
                neighbor_info = vglc_rooms[(nr, nc)]
                print(f'    {direction}: ({nr},{nc}) doors={neighbor_info["doors"]}')
            else:
                print(f'    {direction}: ({nr},{nc}) - NO ROOM')
        
        break
