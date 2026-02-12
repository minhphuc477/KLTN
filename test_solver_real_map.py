"""Test solver dispatch with actual Zelda dungeon data to verify different algorithms produce different results."""
import sys
import os
import time
sys.path.insert(0, os.path.dirname(__file__))

# Load a real dungeon using the same adapter the GUI uses
from src.data.zelda_core import ZeldaDungeonAdapter, DungeonSolver
from gui_runner import _solve_in_subprocess

data_root = os.path.join(os.path.dirname(__file__), "Data", "The Legend of Zelda")
adapter = ZeldaDungeonAdapter(str(data_root))

# Load dungeon 1, quest 1
dungeon = adapter.load_dungeon(1, variant=1)
stitched = adapter.stitch_dungeon(dungeon)
print(f"Loaded dungeon 1 quest 1")
grid = stitched.global_grid

# Find start and goal
import numpy as np
from src.simulation.validator import SEMANTIC_PALETTE

start_pos = None
goal_pos = None

# Find START tile
for r in range(grid.shape[0]):
    for c in range(grid.shape[1]):
        if grid[r, c] == SEMANTIC_PALETTE.get('START', -1):
            start_pos = (r, c)
        elif grid[r, c] == SEMANTIC_PALETTE.get('TRIFORCE', -1):
            goal_pos = (r, c)

if start_pos is None or goal_pos is None:
    print(f"start_pos={start_pos}, goal_pos={goal_pos}")
    # Try ZeldaLogicEnv
    from src.simulation.validator import ZeldaLogicEnv
    graph = getattr(dungeon, 'graph', None)
    room_to_node = getattr(dungeon, 'room_to_node', None)
    room_positions = getattr(dungeon, 'room_positions', None)
    node_to_room = getattr(dungeon, 'node_to_room', None)
    env = ZeldaLogicEnv(grid, render_mode=False, graph=graph, 
                        room_to_node=room_to_node, room_positions=room_positions,
                        node_to_room=node_to_room)
    start_pos = env.start_pos
    goal_pos = env.goal_pos
    print(f"From env: start={start_pos}, goal={goal_pos}")

if start_pos is None or goal_pos is None:
    print("ERROR: Could not find start/goal!")
    sys.exit(1)

print(f"\nGrid shape: {grid.shape}")
print(f"Start: {start_pos}, Goal: {goal_pos}")

# Extract graph data from STITCHED dungeon (has room mappings after stitching)
graph = getattr(stitched, 'graph', None) or getattr(dungeon, 'graph', None)
room_to_node = getattr(stitched, 'room_to_node', None) or getattr(dungeon, 'room_to_node', None)
room_positions = getattr(stitched, 'room_positions', None) or getattr(dungeon, 'room_positions', None)
node_to_room = getattr(stitched, 'node_to_room', None) or getattr(dungeon, 'node_to_room', None)

algorithms = {
    0: "A*",
    1: "BFS",
    2: "Dijkstra",
    3: "Greedy",
    5: "CBS (Balanced)",
    6: "CBS (Explorer)",
}

results = {}
for idx, name in algorithms.items():
    print(f"\n{'='*60}")
    print(f"Testing: {name} (algorithm_idx={idx})")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        r = _solve_in_subprocess(grid, tuple(start_pos), tuple(goal_pos), idx, {}, 
                                 {'allow_diagonals': True},
                                 graph=graph, room_to_node=room_to_node, 
                                 room_positions=room_positions, node_to_room=node_to_room)
        elapsed = time.time() - t0
        path = r.get('path')
        success = r.get('success', False)
        path_len = len(path) if path else 0
        solver_result = r.get('solver_result', {}) or {}
        nodes = solver_result.get('nodes', 'N/A')
        
        results[idx] = {
            'name': name,
            'success': success,
            'path_len': path_len,
            'nodes': nodes,
            'time': elapsed,
            'path_hash': hash(tuple(path)) if path else None,
        }
        
        print(f"  Success: {success}")
        print(f"  Path length: {path_len}")
        print(f"  Nodes explored: {nodes}")
        print(f"  Time: {elapsed*1000:.1f}ms")
        if not success:
            print(f"  Message: {r.get('message', 'N/A')}")
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        results[idx] = {'name': name, 'success': False, 'time': elapsed}

# Summary
print(f"\n\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"{'Algorithm':<20} {'Status':>6} {'Path Len':>10} {'Nodes':>10} {'Time (ms)':>10}")
print("-" * 60)
for idx, data in results.items():
    status = "OK" if data.get('success') else "FAIL"
    path_len = data.get('path_len', 0)
    nodes = data.get('nodes', 'N/A')
    time_ms = data.get('time', 0) * 1000
    print(f"{data['name']:<20} {status:>6} {path_len:>10} {str(nodes):>10} {time_ms:>10.1f}")

# Check if paths differ
print(f"\n{'='*60}")
print("PATH DIFFERENCES")
print(f"{'='*60}")
hashes = {idx: data.get('path_hash') for idx, data in results.items() if data.get('success')}
unique_hashes = set(hashes.values())
print(f"  Successful algorithms: {len(hashes)}")
print(f"  Unique paths: {len(unique_hashes)}")
if len(unique_hashes) == 1:
    print("  WARNING: All algorithms produce identical paths!")
    print("  (This can happen when there's only one valid solution)")
elif len(unique_hashes) > 1:
    print("  Different algorithms produce DIFFERENT paths!")
    for idx, h in hashes.items():
        print(f"    {results[idx]['name']}: path_hash={h}")
