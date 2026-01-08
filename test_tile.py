"""Check exact tile at Room 7 position (10, 24)."""
import sys
sys.path.insert(0, '.')
import importlib.util
import numpy as np

spec = importlib.util.spec_from_file_location('adapter', 'data/adapter.py')
adapter_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter_mod)

IntelligentDataAdapter = adapter_mod.IntelligentDataAdapter
SEMANTIC_PALETTE = adapter_mod.SEMANTIC_PALETTE
ID_TO_NAME = adapter_mod.ID_TO_NAME

adapter = IntelligentDataAdapter('Data/The Legend of Zelda')
dungeon = adapter.process_dungeon(
    'Data/The Legend of Zelda/Processed/tloz1_1.txt',
    'Data/The Legend of Zelda/Graph Processed/LoZ_1.dot',
    'zelda_1_quest1'
)

room_7 = dungeon.rooms.get('7')
if room_7:
    tile = room_7.grid[10, 24]
    name = ID_TO_NAME.get(tile, "UNKNOWN")
    print(f"Room 7 grid[10, 24] = {tile} ({name})")
    
    # Find all START tiles
    start_pos = np.where(room_7.grid == SEMANTIC_PALETTE['START'])
    print(f"All START positions in room 7: {list(zip(start_pos[0], start_pos[1]))}")
    
    # Show the grid area around (10, 24)
    print("\nGrid around (10, 24):")
    for r in range(8, 13):
        row_str = ""
        for c in range(22, 27):
            if c < room_7.grid.shape[1]:
                val = room_7.grid[r, c]
                row_str += f"{val:3d}"
            else:
                row_str += "  -"
        print(f"  Row {r}: {row_str}")
