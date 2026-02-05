"""Debug solver state exploration"""
from Data.zelda_core import ZeldaDungeonAdapter, SEMANTIC_PALETTE
from simulation.validator import ZeldaLogicEnv, StateSpaceAStar, WALKABLE_IDS, BLOCKING_IDS, GameState
from pathlib import Path
import numpy as np
import heapq

data_root = Path('Data/The Legend of Zelda')
adapter = ZeldaDungeonAdapter(str(data_root))
dungeon = adapter.load_dungeon(1, variant=1)
stitched = adapter.stitch_dungeon(dungeon)

env = ZeldaLogicEnv(
    stitched.global_grid,
    graph=stitched.graph,
    room_to_node=stitched.room_to_node,
    room_positions=stitched.room_positions
)

print('=== MANUAL STATE-SPACE BFS ===')
print('Start:', env.start_pos)
print('Goal:', env.goal_pos)
print()

grid = env.grid
height, width = grid.shape
goal = env.goal_pos

# Manual A* implementation to debug
from dataclasses import dataclass, field
from typing import Set, Tuple

# Simplified state - just track position, keys, and collected items
visited_positions = set()  # Just positions, ignore inventory for now
queue = [(0, env.start_pos)]  # (cost, position)
came_from = {env.start_pos: None}

iterations = 0
max_iter = 100000

while queue and iterations < max_iter:
    iterations += 1
    _, current = heapq.heappop(queue)
    
    if current == goal:
        print(f'FOUND GOAL at iteration {iterations}!')
        # Reconstruct path
        path = []
        pos = current
        while pos is not None:
            path.append(pos)
            pos = came_from.get(pos)
        path.reverse()
        print(f'Path length: {len(path)}')
        break
    
    if current in visited_positions:
        continue
    visited_positions.add(current)
    
    # Explore neighbors
    curr_r, curr_c = current
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = curr_r + dr, curr_c + dc
        if 0 <= nr < height and 0 <= nc < width:
            tile = grid[nr, nc]
            # Simple check - not blocking
            if tile not in BLOCKING_IDS:
                neighbor = (nr, nc)
                if neighbor not in visited_positions:
                    cost = iterations + abs(nr - goal[0]) + abs(nc - goal[1])
                    heapq.heappush(queue, (cost, neighbor))
                    if neighbor not in came_from:
                        came_from[neighbor] = current
else:
    if iterations >= max_iter:
        print(f'Stopped at max iterations: {max_iter}')
    else:
        print('Queue exhausted without finding goal')
    print(f'Visited {len(visited_positions)} positions')
    
    # Check if goal is even reachable
    print(f'Goal in visited: {goal in visited_positions}')

print()
print(f'Total iterations: {iterations}')
