import time
import sys
import os
# Ensure project root is on sys.path so we can import gui_runner when running from scripts/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gui_runner import ZeldaGUI

# Create GUI object but avoid calling run()
g = ZeldaGUI()
print('Loaded map, start', g.env.start_pos, 'goal', g.env.goal_pos)

alg_names = ['A*', 'BFS', 'Dijkstra', 'Greedy', 'D* Lite']
for alg in range(len(alg_names)):
    g.algorithm_idx = alg
    name = alg_names[alg]
    start = time.time()
    success, path, teleports = g._smart_grid_path()
    elapsed = (time.time() - start) * 1000
    nodes = getattr(g, 'last_search_iterations', None)
    print(f"Alg={alg} ({name}): success={success}, path_len={len(path)}, teleports={teleports}, nodes={nodes}, time_ms={elapsed:.0f}")
