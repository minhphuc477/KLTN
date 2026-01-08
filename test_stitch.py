"""Test stitched dungeon to see why it's not solvable."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import importlib.util
import numpy as np

spec = importlib.util.spec_from_file_location('adapter', 'data/adapter.py')
adapter_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter_mod)

spec2 = importlib.util.spec_from_file_location('stitcher', 'Data/stitcher.py')
stitcher_mod = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(stitcher_mod)

from simulation.validator import ZeldaValidator, SanityChecker

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

print(f"Dungeon has {len(dungeon.rooms)} rooms")

# Check which rooms have START and TRIFORCE
for room_id, room in dungeon.rooms.items():
    has_start = SEMANTIC_PALETTE['START'] in room.grid
    has_triforce = SEMANTIC_PALETTE['TRIFORCE'] in room.grid
    if has_start or has_triforce:
        print(f"  Room {room_id}: START={has_start}, TRIFORCE={has_triforce}")
        # Find position
        if has_start:
            pos = np.where(room.grid == SEMANTIC_PALETTE['START'])
            print(f"    START at local pos: {list(zip(pos[0], pos[1]))}")
        if has_triforce:
            pos = np.where(room.grid == SEMANTIC_PALETTE['TRIFORCE'])
            print(f"    TRIFORCE at local pos: {list(zip(pos[0], pos[1]))}")

# Stitch
stitcher = DungeonStitcher(dungeon)
stitched = stitcher.stitch()

print(f"\nStitched dungeon: {stitched.global_grid.shape}")
print(f"START position: {stitched.start_pos}")
print(f"GOAL position: {stitched.goal_pos}")

# Check what's actually at the START position
if stitched.start_pos:
    sr, sc = stitched.start_pos
    tile_at_start = stitched.global_grid[sr, sc]
    tile_name = adapter_mod.ID_TO_NAME.get(tile_at_start, f"UNKNOWN({tile_at_start})")
    print(f"Tile at START position: {tile_name} (ID={tile_at_start})")

# Check what's actually at the GOAL position
if stitched.goal_pos:
    gr, gc = stitched.goal_pos
    tile_at_goal = stitched.global_grid[gr, gc]
    tile_name = adapter_mod.ID_TO_NAME.get(tile_at_goal, f"UNKNOWN({tile_at_goal})")
    print(f"Tile at GOAL position: {tile_name} (ID={tile_at_goal})")

# Run sanity check
checker = SanityChecker(stitched.global_grid)
is_valid, errors = checker.check_all()
print(f"\nSanity check: valid={is_valid}")
if errors:
    for e in errors:
        print(f"  - {e}")

# Count START and TRIFORCE explicitly
start_count = np.sum(stitched.global_grid == SEMANTIC_PALETTE['START'])
triforce_count = np.sum(stitched.global_grid == SEMANTIC_PALETTE['TRIFORCE'])
stair_count = np.sum(stitched.global_grid == SEMANTIC_PALETTE['STAIR'])
print(f"\nDirect count:")
print(f"  START (ID=21): {start_count}")
print(f"  TRIFORCE (ID=22): {triforce_count}")
print(f"  STAIR (ID=42): {stair_count}")

