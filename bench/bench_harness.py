"""Benchmark harness for grid solvers (A*, JPS) with deterministic seeding.

Usage: python -m bench.bench_harness --solver astar --map open --size 100 --seed 42 --repeats 50

Outputs CSV to stdout with columns: solver, map_family, size, seed, repeat, runtime_ms, nodes_expanded, path_len
"""
import argparse
import time
import csv
import random
from bench.grid_solvers import astar, jps


def generate_map(family: str, size: int, seed: int):
    rnd = random.Random(seed)
    if family == 'open':
        return [[0]*size for _ in range(size)]
    if family == 'maze':
        # simple random obstacles
        grid = [[0]*size for _ in range(size)]
        for r in range(size):
            for c in range(size):
                grid[r][c] = 1 if rnd.random() < 0.3 else 0
        return grid
    if family == 'corridor':
        grid = [[1]*size for _ in range(size)]
        mid = size // 2
        for r in range(size):
            grid[r][mid] = 0
        return grid
    # default random sparse
    grid = [[0]*size for _ in range(size)]
    for r in range(size):
        for c in range(size):
            grid[r][c] = 1 if rnd.random() < 0.15 else 0
    return grid


def run_once(solver_name, grid, start, goal, allow_diagonal=True, allow_corner_cutting=False):
    start_time = time.time()
    if solver_name == 'astar':
        path, nodes = astar(grid, start, goal, allow_diagonal=allow_diagonal, allow_corner_cutting=allow_corner_cutting)
        t = (time.time() - start_time) * 1000.0
        return t, nodes, len(path) - 1 if path else -1
    elif solver_name == 'jps':
        res = jps(grid, start, goal, allow_diagonal=allow_diagonal, trace=False, allow_corner_cutting=allow_corner_cutting)
        if isinstance(res, tuple) and len(res) >= 2:
            path = res[0]
            nodes = res[1]
        else:
            path, nodes = None, 0
        t = (time.time() - start_time) * 1000.0
        return t, nodes, len(path) - 1 if path else -1
    else:
        raise ValueError('Unknown solver')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--solver', choices=['astar', 'jps'], default='astar')
    p.add_argument('--map', choices=['open', 'maze', 'corridor', 'random'], default='open')
    p.add_argument('--size', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--repeats', type=int, default=10)
    p.add_argument('--allow_diagonal', action='store_true')
    p.add_argument('--allow_corner_cutting', action='store_true')
    args = p.parse_args()

    writer = csv.writer(open('bench_results.csv', 'w'))
    writer.writerow(['solver','map_family','size','seed','repeat','runtime_ms','nodes_expanded','path_len'])

    for r in range(args.repeats):
        grid = generate_map(args.map, args.size, args.seed + r)
        start = (0,0)
        goal = (args.size-1, args.size-1)
        t, nodes, plen = run_once(args.solver, grid, start, goal, allow_diagonal=args.allow_diagonal, allow_corner_cutting=args.allow_corner_cutting)
        writer.writerow([args.solver, args.map, args.size, args.seed, r, t, nodes, plen])


if __name__ == '__main__':
    main()
