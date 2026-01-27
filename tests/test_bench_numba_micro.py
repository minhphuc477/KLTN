import pytest
import random
import numpy as np

try:
    from bench import numba_solvers
except Exception:
    pytest.skip('bench.numba_solvers import failed', allow_module_level=True)


def test_neighbors_numba_matches_python():
    size = 20
    rnd = random.Random(3)
    grid = [[0]*size for _ in range(size)]
    for r in range(size):
        for c in range(size):
            if rnd.random() < 0.2:
                grid[r][c] = 1
    flat = np.array([cell for row in grid for cell in row], dtype=np.int8)
    h = size; w = size

    for idx in [0, 1, size//2 * size + size//2, size*size-1]:
        py = numba_solvers.neighbors_flat(idx, h, w, flat)
        nb = numba_solvers.neighbors_flat_numba(idx, h, w, flat)
        assert sorted(py) == sorted(nb)
