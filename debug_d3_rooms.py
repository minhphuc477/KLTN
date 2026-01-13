"""Debug room connectivity."""
from vglc_first_adapter import VGLCFirstAdapter, SEMANTIC_PALETTE, ID_TO_NAME
from test_vglc_solver import stitch_dungeon
import numpy as np

adapter = VGLCFirstAdapter('Data')

# Process dungeon 3
dungeon = adapter.process_dungeon(
    'Data/The Legend of Zelda/Processed/tloz3_1.txt',
    'Data/The Legend of Zelda/Graph Processed/LoZ_3.dot',
    'D3'
)

# Stitch
full_grid, min_row, min_col = stitch_dungeon(dungeon)

# Find positions of rooms (1,0) and (2,0) in stitched grid
room_height, room_width = 16, 11

room_1_0 = dungeon.rooms.get('1_0')
room_2_0 = dungeon.rooms.get('2_0')

if room_1_0:
    r, c = room_1_0.position
    norm_r = r - min_row
    norm_c = c - min_col
    start_row = norm_r * room_height
    start_col = norm_c * room_width
    
    print(f"Room 1_0 in stitched grid (rows {start_row}-{start_row+15}, cols {start_col}-{start_col+10}):")
    for row_idx in range(start_row, start_row + room_height):
        row_vals = full_grid[row_idx, start_col:start_col+room_width]
        names = [ID_TO_NAME.get(v, 'UNK')[:4] for v in row_vals]
        print(f"  {row_idx-start_row:2d}: {' '.join(names)}")

if room_2_0:
    r, c = room_2_0.position
    norm_r = r - min_row
    norm_c = c - min_col
    start_row = norm_r * room_height
    start_col = norm_c * room_width
    
    print(f"\nRoom 2_0 in stitched grid (rows {start_row}-{start_row+15}, cols {start_col}-{start_col+10}):")
    for row_idx in range(start_row, start_row + room_height):
        row_vals = full_grid[row_idx, start_col:start_col+room_width]
        names = [ID_TO_NAME.get(v, 'UNK')[:4] for v in row_vals]
        print(f"  {row_idx-start_row:2d}: {' '.join(names)}")

# Check boundary between rooms (1,0) and (2,0)
if room_1_0 and room_2_0:
    print("\nBoundary rows (1,0 row 14-15, 2,0 row 0-1):")
    r1, c1 = room_1_0.position
    norm_r1 = r1 - min_row
    start_row_1 = norm_r1 * room_height
    
    for row_offset in [14, 15, 16, 17]:  # Last 2 rows of 1_0 and first 2 rows of 2_0
        row_idx = start_row_1 + row_offset
        if row_idx < full_grid.shape[0]:
            row_vals = full_grid[row_idx, start_col:start_col+room_width]
            names = [ID_TO_NAME.get(v, 'UNK')[:4] for v in row_vals]
            print(f"  Row {row_idx} ({row_offset}): {' '.join(names)}")
