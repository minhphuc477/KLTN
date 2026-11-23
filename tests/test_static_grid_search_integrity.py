"""Compare static search with an independent exhaustive Dijkstra oracle."""

import heapq
import math

import pytest

from bench.grid_solvers import astar, jps


def _oracle(grid, diagonal, corners):
    heap = [(0.0, (0, 0))]
    settled = set()
    while heap:
        cost, pos = heapq.heappop(heap)
        if pos in settled:
            continue
        settled.add(pos)
        if pos == (2, 2):
            return cost
        r, c = pos
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (dr == dc == 0) or (dr and dc and not diagonal):
                    continue
                nr, nc = r + dr, c + dc
                if not (0 <= nr < 3 and 0 <= nc < 3) or grid[nr][nc]:
                    continue
                if dr and dc and not corners and (grid[nr][c] or grid[r][nc]):
                    continue
                heapq.heappush(heap, (cost + math.hypot(dr, dc), (nr, nc)))
    return None


@pytest.mark.parametrize("solver", [astar, jps])
@pytest.mark.parametrize("diagonal,corners", [(False, False), (True, False), (True, True)])
def test_all_three_by_three_obstacle_maps(solver, diagonal, corners):
    cells = [(r, c) for r in range(3) for c in range(3) if (r, c) not in {(0, 0), (2, 2)}]
    for mask in range(128):
        grid = [[0] * 3 for _ in range(3)]
        for bit, (r, c) in enumerate(cells):
            grid[r][c] = (mask >> bit) & 1
        expected = _oracle(grid, diagonal, corners)
        path, _count = solver(grid, (0, 0), (2, 2), allow_diagonal=diagonal,
                              allow_corner_cutting=corners)
        assert (path is None) == (expected is None), (mask, path, expected)
        if path is None:
            continue
        assert path[0] == (0, 0) and path[-1] == (2, 2)
        cost = 0.0
        for (r, c), (nr, nc) in zip(path, path[1:]):
            dr, dc = abs(nr-r), abs(nc-c)
            assert 0 < max(dr, dc) <= 1 and not grid[nr][nc]
            assert diagonal or dr + dc == 1
            if dr and dc and not corners:
                assert not grid[nr][c] and not grid[r][nc]
            cost += math.hypot(dr, dc)
        assert cost == pytest.approx(expected), (mask, path)


@pytest.mark.parametrize("solver", [astar, jps])
@pytest.mark.parametrize("grid,start,goal", [([], (0, 0), (0, 0)),
    ([[1]], (0, 0), (0, 0)), ([[0]], (-1, 0), (0, 0)), ([[0]], (0, 0), (1, 0))])
def test_invalid_endpoints_cannot_be_reported_as_solved(solver, grid, start, goal):
    assert solver(grid, start, goal) == (None, 0)


def test_jump_trace_contract_for_trivial_and_invalid_requests():
    for grid, expected in [([[0]], [(0, 0)]), ([[1]], None)]:
        path, count, trace = jps(grid, (0, 0), (0, 0), trace=True)
        assert path == expected
        assert count == 0
        assert trace["algorithm"] == "jump_augmented_astar"
