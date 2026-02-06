"""Automated benchmark suite for comparing A* and JPS across map families.

Produces CSV with fields: solver, map_type, size, obstacle_ratio, allow_diagonal, time_sec, nodes_expanded, path_len
"""
import time
import csv
import os
from bench.grid_solvers import astar, jps


def synthetic_open_grid(size=50, obstacle_ratio=0.1, seed=42):
    import random
    rnd = random.Random(seed)
    grid = [[0]*size for _ in range(size)]
    for r in range(size):
        for c in range(size):
            if rnd.random() < obstacle_ratio:
                grid[r][c] = 1
    return grid


def corridor_grid(size=50, spacing=3):
    grid = [[0]*size for _ in range(size)]
    for c in range(spacing-1, size, spacing):
        for r in range(size):
            grid[r][c] = 1
    # openholes every so often
    for i, c in enumerate(range(spacing-1, size, spacing)):
        hole = (i * 7) % size
        grid[hole][c] = 0
    return grid


def maze_grid(size=50, seed=42):
    # simple recursive division / random maze generator (coarse)
    import random
    grid = [[1]*size for _ in range(size)]
    def carve(r0,c0,r1,c1):
        if r1 - r0 < 2 or c1 - c0 < 2:
            return
        # carve a room
        for r in range(r0+1, r1):
            for c in range(c0+1, c1):
                grid[r][c] = 0
        # place walls
        # stop recursion to keep it fast
    carve(0,0,size-1,size-1)
    # make some openings
    import random
    for _ in range(size*3):
        r = random.randrange(1,size-1)
        c = random.randrange(1,size-1)
        grid[r][c] = 1 if grid[r][c]==0 else 0
    return grid


def run_suite(out_csv='bench/results.csv'):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    maps = [
        ('open', synthetic_open_grid(50, 0.05, seed=1)),
        ('open', synthetic_open_grid(50, 0.15, seed=2)),
        ('corridor', corridor_grid(50, spacing=3)),
        ('corridor', corridor_grid(50, spacing=4)),
        ('maze', maze_grid(50, seed=3)),
    ]

    rows = []
    for map_type, grid in maps:
        for allow_diag in (False, True):
            for solver_name in ('astar', 'jps'):
                t0 = time.time()
                if solver_name == 'astar':
                    path, nodes = astar(grid, (0,0), (len(grid)-1,len(grid)-1), allow_diagonal=allow_diag)
                else:
                    res = jps(grid, (0,0), (len(grid)-1,len(grid)-1), allow_diagonal=allow_diag)
                    if len(res) == 3:
                        path, nodes, trace = res
                    else:
                        path, nodes = res
                dt = time.time() - t0
                rows.append({'solver': solver_name, 'map_type': map_type, 'size': len(grid), 'allow_diagonal': allow_diag, 'time_sec': dt, 'nodes_expanded': nodes, 'path_len': len(path) if path else -1})
                print(f"{solver_name} {map_type} diag={allow_diag} time={dt:.4f}s nodes={nodes} pathlen={len(path) if path else -1}")
    # write CSV
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print('Wrote results to', out_csv)

if __name__ == '__main__':
    run_suite()
