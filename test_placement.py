"""Debug stitcher room placement."""
import sys
sys.path.insert(0, '.')

import importlib.util
import numpy as np

spec = importlib.util.spec_from_file_location('adapter', 'data/adapter.py')
adapter_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter_mod)

spec2 = importlib.util.spec_from_file_location('stitcher', 'Data/stitcher.py')
stitcher_mod = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(stitcher_mod)

IntelligentDataAdapter = adapter_mod.IntelligentDataAdapter
SEMANTIC_PALETTE = adapter_mod.SEMANTIC_PALETTE
DungeonStitcher = stitcher_mod.DungeonStitcher

# Process dungeon 1
adapter = IntelligentDataAdapter('Data/The Legend of Zelda')
dungeon = adapter.process_dungeon(
    'Data/The Legend of Zelda/Processed/tloz1_1.txt',
    'Data/The Legend of Zelda/Graph Processed/LoZ_1.dot',
    'zelda_1_quest1'
)

# Check Room 7 specifically
room_7 = dungeon.rooms.get('7')
if room_7:
    print(f"Room 7 grid shape: {room_7.grid.shape}")
    start_pos = np.where(room_7.grid == SEMANTIC_PALETTE['START'])
    print(f"START in room 7 at: {list(zip(start_pos[0], start_pos[1]))}")

# Check graph layout
stitcher = DungeonStitcher(dungeon)
layout = stitcher.compute_layout(dungeon.graph)
print(f"\nLayout for graph nodes:")
for node, pos in sorted(layout.items()):
    print(f"  Node {node}: layout position {pos}")

# Check what Room 7's layout position is
if 7 in layout:
    layout_row, layout_col = layout[7]
    r_offset = layout_row * (stitcher.ROOM_HEIGHT + stitcher.PADDING)
    c_offset = layout_col * (stitcher.ROOM_WIDTH + stitcher.PADDING)
    print(f"\nRoom 7 placement:")
    print(f"  layout pos: ({layout_row}, {layout_col})")
    print(f"  ROOM_HEIGHT: {stitcher.ROOM_HEIGHT}, ROOM_WIDTH: {stitcher.ROOM_WIDTH}")
    print(f"  r_offset: {r_offset}, c_offset: {c_offset}")
    
    # Calculate grid dimensions
    max_row = max(p[0] for p in layout.values())
    max_col = max(p[1] for p in layout.values())
    grid_height = (max_row + 1) * (stitcher.ROOM_HEIGHT + stitcher.PADDING)
    grid_width = (max_col + 1) * (stitcher.ROOM_WIDTH + stitcher.PADDING)
    print(f"  grid_height: {grid_height}, grid_width: {grid_width}")
    
    # Check if START position will be within bounds
    start_local = (10, 24)  # from earlier test
    global_r = r_offset + start_local[0]
    global_c = c_offset + start_local[1]
    print(f"  Expected START global position: ({global_r}, {global_c})")
    
    # Check room bounds
    rh, rw = room_7.grid.shape
    r_end = min(r_offset + rh, grid_height)
    c_end = min(c_offset + rw, grid_width)
    actual_rh = r_end - r_offset
    actual_rw = c_end - c_offset
    print(f"  Room 7 actual dims: {rh}x{rw}")
    print(f"  After placement bounds: r_end={r_end}, c_end={c_end}")
    print(f"  Actual placed dims: {actual_rh}x{actual_rw}")
    
    # Check if START is within placed region
    if start_local[0] < actual_rh and start_local[1] < actual_rw:
        print(f"  START IS within placed region ✓")
    else:
        print(f"  START is TRUNCATED! ✗")
