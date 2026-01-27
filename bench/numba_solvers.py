"""Numba sketched helpers for accelerating grid solvers (proof-of-concept).

This module contains small numba-accelerated helpers such as neighbor generation
on flattened grid arrays. Full numba-accelerated A*/JPS implementations are
left as next steps, but these helpers demonstrate the pattern to follow.
"""
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

import numpy as np


if NUMBA_AVAILABLE:
    @njit
    def neighbors_flat(idx, h, w, grid_flat):
        """Generator-style helper (returns array of neighbors indices).

        idx: flattened index
        grid_flat: 1D array with 0=free, 1=blocked
        """
        r = idx // w
        c = idx % w
        res = []
        # up
        ni = idx - w
        if r > 0 and grid_flat[ni] == 0:
            res.append(ni)
        # down
        ni = idx + w
        if r < h-1 and grid_flat[ni] == 0:
            res.append(ni)
        # left
        ni = idx - 1
        if c > 0 and grid_flat[ni] == 0:
            res.append(ni)
        # right
        ni = idx + 1
        if c < w-1 and grid_flat[ni] == 0:
            res.append(ni)
        return res

    @njit
    def neighbors_flat_array(idx, h, w, grid_flat, out_buf):
        """Numba-friendly: write neighbor indices into preallocated out_buf and return count."""
        r = idx // w
        c = idx % w
        cnt = 0
        # up
        ni = idx - w
        if r > 0 and grid_flat[ni] == 0:
            out_buf[cnt] = ni; cnt += 1
        # down
        ni = idx + w
        if r < h-1 and grid_flat[ni] == 0:
            out_buf[cnt] = ni; cnt += 1
        # left
        ni = idx - 1
        if c > 0 and grid_flat[ni] == 0:
            out_buf[cnt] = ni; cnt += 1
        # right
        ni = idx + 1
        if c < w-1 and grid_flat[ni] == 0:
            out_buf[cnt] = ni; cnt += 1
        return cnt

    def neighbors_flat_numba(idx, h, w, grid_flat):
        import numpy as _np
        out = _np.empty(4, dtype=_np.int64)
        cnt = neighbors_flat_array(idx, h, w, grid_flat, out)
        return [int(out[i]) for i in range(cnt)]

else:
    def neighbors_flat(idx, h, w, grid_flat):
        # Python fallback
        r = idx // w
        c = idx % w
        res = []
        if r > 0 and grid_flat[idx - w] == 0:
            res.append(idx - w)
        if r < h-1 and grid_flat[idx + w] == 0:
            res.append(idx + w)
        if c > 0 and grid_flat[idx - 1] == 0:
            res.append(idx - 1)
        if c < w-1 and grid_flat[idx + 1] == 0:
            res.append(idx + 1)
        return res

    def neighbors_flat_numba(idx, h, w, grid_flat):
        # Fallback to Python implementation
        return neighbors_flat(idx, h, w, grid_flat)


def smoke_test_neighbors():
    g = np.zeros((5,5), dtype=np.int8)
    flat = g.ravel()
    return neighbors_flat(6, 5, 5, flat)
