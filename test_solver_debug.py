"""Debug script to investigate solver issues."""
from src.simulation.validator import WALKABLE_IDS, CONDITIONAL_IDS, SEMANTIC_PALETTE, BLOCKING_IDS, WATER_IDS
from src.data.zelda_core import ZeldaDungeonAdapter
from src.core.definitions import ID_TO_NAME
from pathlib import Path
from collections import deque
import numpy as np

a = ZeldaDungeonAdapter(str(Path('Data/The Legend of Zelda')))
d = a.load_dungeon(1, variant=2)
s = a.stitch_dungeon(d)
grid = s.global_grid

print("Grid shape:", grid.shape)
start = s.start_global
goal = s.triforce_global
print("Start:", start, "Goal:", goal)

# BFS
visited = set()
queue = deque([start])
visited.add(start)

while queue:
    r, c = queue.popleft()
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
            if (nr, nc) not in visited:
                tid = grid[nr, nc]
                if tid in WALKABLE_IDS:
                    visited.add((nr, nc))
                    queue.append((nr, nc))

min_c = min(c for r, c in visited)
max_c = max(c for r, c in visited)
print(f"Visited column range: {min_c} to {max_c}")
print(f"Goal is at column: {goal[1]}")
print(f"Visited {len(visited)} tiles")

# Check what's at column 22 (the boundary of room 1)
print()
print("=== Column 21-23 boundary check ===")
for c in [21, 22, 23]:
    print(f"Column {c}:")
    for r in range(grid.shape[0]):
        tid = grid[r, c]
        if tid not in [0, 2]:  # Skip void and wall
            name = ID_TO_NAME.get(tid, str(tid))
            walkable = "W" if tid in WALKABLE_IDS else "X"
            in_visited = "V" if (r, c) in visited else " "
            print(f"  ({r}, {c}): {tid} ({name}) {walkable} {in_visited}")


# BFS from first key position
start = (8, 16)
print(f"=== BFS from first key at {start} ===")

visited = set()
queue = deque([start])
visited.add(start)
reachable_cond = []

while queue:
    r, c = queue.popleft()
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
            if (nr, nc) not in visited:
                tid = grid[nr, nc]
                if tid in WALKABLE_IDS:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
                elif tid in CONDITIONAL_IDS:
                    reachable_cond.append((nr, nc, tid))

print(f"Visited {len(visited)} tiles from first key")
print(f"Goal (24, 71) reachable without keys: {(24, 71) in visited}")
print(f"Second key (24, 38) reachable without keys: {(24, 38) in visited}")
print(f"Conditional doors reachable: {reachable_cond}")
print()

# Check the second key position
key2_pos = (24, 38)
print(f"=== Tile at second key {key2_pos} ===")
print(f"Tile value: {grid[key2_pos]}")
print(f"Expected KEY_SMALL: {SEMANTIC_PALETTE['KEY_SMALL']}")
print(f"Is in WALKABLE_IDS: {grid[key2_pos] in WALKABLE_IDS}")
print()

# Check tiles around second key
print("Tiles around second key (24, 38):")
for dr in [-1, 0, 1]:
    for dc in [-1, 0, 1]:
        nr, nc = key2_pos[0] + dr, key2_pos[1] + dc
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
            tid = grid[nr, nc]
            status = "WALK" if tid in WALKABLE_IDS else ("BLOCK" if tid in BLOCKING_IDS else ("COND" if tid in CONDITIONAL_IDS else f"OTHER({tid})"))
            print(f"  ({nr}, {nc}): tile={tid} -> {status}")
print()

# Check the locked doors
print("=== Locked door analysis ===")
door_locked = SEMANTIC_PALETTE['DOOR_LOCKED']
for r in range(grid.shape[0]):
    for c in range(grid.shape[1]):
        if grid[r, c] == door_locked:
            print(f"DOOR_LOCKED at ({r}, {c})")
            # Check neighbors
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    if (nr, nc) in visited:
                        print(f"  -> Reachable from first key side: ({nr}, {nc})")
