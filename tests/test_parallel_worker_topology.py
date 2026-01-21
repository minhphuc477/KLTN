from scripts.parallel_worker import run_grid_algorithm


def test_parallel_worker_topology_restricts_teleport():
    # 9x9 grid with interior floors
    grid = [[1]*9 for _ in range(9)]
    # Start stair in room A
    start = (2, 2)
    grid[start[0]][start[1]] = 42  # stair A
    # Middle stair in room B
    mid = (2, 6)
    grid[mid[0]][mid[1]] = 42  # stair B
    # Goal stair in room C
    goal = (6, 6)
    grid[goal[0]][goal[1]] = 42  # stair C

    # Topology: A connected only to B; B connected to C; A not connected directly to C
    topology = {
        'room_to_stairs': {
            'A': [start],
            'B': [mid],
            'C': [goal],
        },
        'adj': {
            'A': ['B'],
            'B': ['C'],
            'C': []
        }
    }

    res = run_grid_algorithm(grid, start, goal, alg=1, allow_teleports=True, topology=topology)
    assert isinstance(res, dict)
    assert res.get('success') is True
    path = res.get('path', [])
    # Find indices of stair nodes if present
    idx_A = path.index(start) if start in path else None
    idx_B = path.index(mid) if mid in path else None
    idx_C = path.index(goal) if goal in path else None
    # Teleport chain should include B before C
    assert idx_B is not None and idx_C is not None
    assert idx_B < idx_C


def test_parallel_worker_no_direct_teleport_when_topology_missing():
    # Ensure that without topology, teleports may go between any stairs
    grid = [[1]*5 for _ in range(5)]
    a = (1,1); b = (1,3); c = (3,3)
    grid[a[0]][a[1]] = 42
    grid[b[0]][b[1]] = 42
    grid[c[0]][c[1]] = 42
    res = run_grid_algorithm(grid, a, c, alg=1, allow_teleports=True)
    assert isinstance(res, dict)
    assert res.get('success') is True
