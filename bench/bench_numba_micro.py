"""Micro-benchmark for neighbors generation: compare Numba helper vs Python fallback.

Run: python -m bench.bench_numba_micro --size 200 --repeats 1000
"""
import argparse
import time
import random
import numpy as np
from bench import numba_solvers


def run_once(func, idx, h, w, flat):
    t0 = time.perf_counter()
    out = func(idx, h, w, flat)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--size', type=int, default=100)
    p.add_argument('--repeats', type=int, default=1000)
    args = p.parse_args()

    size = args.size
    rnd = random.Random(0)
    grid = [[0]*size for _ in range(size)]
    # add some obstacles
    for r in range(size):
        for c in range(size):
            if rnd.random() < 0.15:
                grid[r][c] = 1
    flat = np.array([cell for row in grid for cell in row], dtype=np.int8)
    h = size; w = size

    indices = [rnd.randrange(0, size*size) for _ in range(args.repeats)]

    # Warm up numba
    try:
        _ = numba_solvers.neighbors_flat_numba(indices[0], h, w, flat)
    except Exception:
        pass

    # Benchmark python fallback
    t_py = 0.0
    for idx in indices:
        t, _ = run_once(numba_solvers.neighbors_flat, idx, h, w, flat)
        t_py += t

    # Benchmark numba version
    t_nb = 0.0
    for idx in indices:
        t, _ = run_once(numba_solvers.neighbors_flat_numba, idx, h, w, flat)
        t_nb += t

    print('Python neighbors total ms: {:.3f}'.format(t_py))
    print('Numba neighbors total ms: {:.3f}'.format(t_nb))
    print('Speedup: {:.2f}x'.format(max(1e-9, t_py / max(1e-9, t_nb))))

if __name__ == '__main__':
    main()
