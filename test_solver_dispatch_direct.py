"""Direct test of solver dispatch to verify different algorithms produce different results."""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.simulation.validator import GameState, SEMANTIC_PALETTE, ACTION_DELTAS
from gui_runner import _solve_in_subprocess

# Create a simple test grid with path variations
grid = np.zeros((11, 11), dtype=np.int64)
wall_id = SEMANTIC_PALETTE['WALL']
floor_id = SEMANTIC_PALETTE['FLOOR']

# Fill with floor
grid[:] = floor_id
# Border walls
grid[0,:] = wall_id
grid[-1,:] = wall_id
grid[:,0] = wall_id
grid[:,-1] = wall_id
# Add some internal walls to create path variations
grid[2, 1:8] = wall_id
grid[4, 3:10] = wall_id
grid[6, 1:7] = wall_id
grid[8, 4:10] = wall_id

start = (1, 1)
goal = (9, 9)

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
    print(f"Testing algorithm_idx={idx} -> {name}")
    print(f"{'='*60}")
    try:
        r = _solve_in_subprocess(grid, start, goal, idx, {}, {})
        path = r.get('path')
        success = r.get('success', False)
        path_len = len(path) if path else 0
        solver_result = r.get('solver_result', {})
        nodes = solver_result.get('nodes', 'N/A') if solver_result else 'N/A'
        algo_reported = solver_result.get('algorithm', 'N/A') if solver_result else 'N/A'
        cbs_metrics = solver_result.get('cbs_metrics') if solver_result else None
        
        results[idx] = {
            'name': name,
            'success': success,
            'path_len': path_len,
            'nodes': nodes,
            'algo_reported': algo_reported,
            'path': path,
            'cbs_metrics': cbs_metrics
        }
        
        print(f"  Success: {success}")
        print(f"  Path length: {path_len}")
        print(f"  Nodes explored: {nodes}")
        print(f"  Algorithm reported: {algo_reported}")
        if path:
            print(f"  First 3 steps: {path[:3]}")
            print(f"  Last 3 steps: {path[-3:]}")
        if cbs_metrics:
            print(f"  CBS Confusion: {cbs_metrics['confusion_index']}")
            print(f"  CBS Cognitive Load: {cbs_metrics['cognitive_load']}")
        if not success:
            print(f"  Message: {r.get('message', 'N/A')}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        results[idx] = {'name': name, 'success': False, 'error': str(e)}

# Summary
print(f"\n\n{'='*60}")
print("SUMMARY - PATH COMPARISON")
print(f"{'='*60}")
for idx, data in results.items():
    status = "OK" if data.get('success') else "FAIL"
    path_len = data.get('path_len', 0)
    nodes = data.get('nodes', 'N/A')
    print(f"  [{status}] {data['name']:20s} path_len={path_len:4d}  nodes={nodes}")

# Check identical paths
print(f"\n{'='*60}")
print("PATH IDENTITY CHECKS")
print(f"{'='*60}")
if results.get(0, {}).get('path') and results.get(1, {}).get('path'):
    same = results[0]['path'] == results[1]['path']
    print(f"  A* == BFS?       {same}")
if results.get(0, {}).get('path') and results.get(2, {}).get('path'):
    same = results[0]['path'] == results[2]['path']
    print(f"  A* == Dijkstra?  {same}")
if results.get(0, {}).get('path') and results.get(3, {}).get('path'):
    same = results[0]['path'] == results[3]['path']
    print(f"  A* == Greedy?    {same}")
if results.get(0, {}).get('path') and results.get(5, {}).get('path'):
    same = results[0]['path'] == results[5]['path']
    print(f"  A* == CBS Bal?   {same}")
if results.get(0, {}).get('path') and results.get(6, {}).get('path'):
    same = results[0]['path'] == results[6]['path']
    print(f"  A* == CBS Expl?  {same}")
