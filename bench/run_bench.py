"""Simple benchmark runner for grid solvers. Produces CSV-like stdout for later import into plots."""
import time
from bench.grid_solvers import astar, jps


def synthetic_open_grid(size=50, obstacle_ratio=0.1, seed=42):
    import random
    rnd = random.Random(seed)
    grid = [[0]*size for _ in range(size)]
    # place obstacles
    for r in range(size):
        for c in range(size):
            if rnd.random() < obstacle_ratio:
                grid[r][c] = 1
    return grid


def run_simple_bench():
    grid = synthetic_open_grid(30, obstacle_ratio=0.12)
    start = (0, 0)
    goal = (29, 29)

    t0 = time.time()
    path, nodes = astar(grid, start, goal)
    dt = time.time() - t0
    print(f"solver,grid_size,ob_ratio,time_sec,nodes_expanded,path_len")
    print(f"astar,{len(grid)},{0.12},{dt:.4f},{nodes},{len(path) if path else -1}")

    # JPS placeholder try/except
    try:
        t0 = time.time()
        path, nodes = jps(grid, start, goal)
        dt = time.time() - t0
        print(f"jps,{len(grid)},{0.12},{dt:.4f},{nodes},{len(path) if path else -1}")
    except NotImplementedError:
        print("jps,not_implemented")


if __name__ == '__main__':
    run_simple_bench()
