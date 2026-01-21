import json
from scripts.parallel_worker import run_grid_algorithm


def test_parallel_worker_teleports():
    # Grid with two stair tiles connecting distant locations
    grid = [[1]*7 for _ in range(7)]
    # Surround with floor
    grid[1][1] = 21  # start
    grid[5][5] = 22  # goal
    grid[2][2] = 42  # stair A
    grid[4][4] = 42  # stair B

    res_bfs = run_grid_algorithm(grid, (1,1), (5,5), alg=1, allow_teleports=True)
    # With teleports, BFS should find path possibly using stairs; ensure function runs
    assert isinstance(res_bfs, dict)
    assert 'success' in res_bfs

    # Now test priority-based (A*)
    res_a = run_grid_algorithm(grid, (1,1), (5,5), alg=0, allow_teleports=True)
    assert isinstance(res_a, dict)
    assert 'time_ms' in res_a
