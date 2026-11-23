"""Independent small-map and stateful regressions; no saved experiment inputs."""

from collections import deque

import networkx as nx
import numpy as np
import pytest

from src.core.definitions import ROOM_HEIGHT, ROOM_WIDTH, SEMANTIC_PALETTE as P
from src.simulation.dstar_lite import DStarLiteSolver
from src.simulation.search_base import GameStateSearchConfig, SearchRepresentation
from src.simulation.bidirectional_astar import BidirectionalAStar
from src.simulation.search_factory import run_game_state_solver
from src.simulation.state_space_dfs import StateSpaceDFS
from src.simulation.validator import StateSpaceAStar, ZeldaLogicEnv, game_state_key


def _stair_env(*, directed=True, locked=False, key=False):
    grid = np.full((ROOM_HEIGHT, ROOM_WIDTH * 3), P["WALL"], dtype=np.int64)
    grid[1, 1] = P["START"]
    grid[1, 2] = P["STAIR"]
    grid[1, 1 + ROOM_WIDTH * 2] = P["STAIR"]
    grid[1, 2 + ROOM_WIDTH * 2] = P["TRIFORCE"]
    if key:
        grid[2, 1] = P["KEY_SMALL"]
    graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_edge(0, 1, edge_type="stair", label="locked" if locked else "stair")
    return ZeldaLogicEnv(
        grid, graph=graph, room_to_node={(0, 0): 0, (0, 2): 1},
        room_positions={(0, 0): (0, 0), (0, 2): (0, ROOM_WIDTH * 2)},
    )


@pytest.mark.parametrize("mode", list(SearchRepresentation))
def test_search_representation_roundtrips_enum_and_string(mode):
    assert SearchRepresentation.parse(mode) is mode
    assert GameStateSearchConfig(representation=mode.value).to_priority_options()["representation"] == mode.value


def test_reusing_bidirectional_solver_resets_fallback_budget_and_statistics():
    env = _stair_env(locked=True, key=True)
    solver = BidirectionalAStar(env, timeout=100)
    first = solver.solve()
    assert first[0]
    assert solver.used_fallback
    second = solver.solve()
    assert second == first
    assert solver.states_explored == second[2]


@pytest.mark.parametrize("iddfs", [False, True])
def test_dfs_supports_graph_stairs_and_replayed_witness(iddfs):
    env = _stair_env()
    result = run_game_state_solver(env, 5, GameStateSearchConfig(max_depth=6, use_iddfs=iddfs))
    assert result.success
    valid, reason, _state, _cost = StateSpaceAStar(env).verify_position_path(result.path)
    assert valid, reason


def test_stair_lookup_supports_undirected_mission_graphs():
    env = _stair_env(directed=False)
    success, path, _ = StateSpaceAStar(env).solve_with_verified_diagnostics()
    assert success
    assert path[-1] == env.goal_pos


@pytest.mark.parametrize("key", [False, True])
def test_stairs_cannot_bypass_attached_key_gate(key):
    env = _stair_env(locked=True, key=key)
    success, _path, diagnostics = StateSpaceAStar(env).solve_with_verified_diagnostics()
    assert success == key
    if success:
        assert diagnostics.final_inventory["keys"] == 0


@pytest.mark.parametrize("has_item", [False, True])
def test_explicit_composite_gate_metadata_is_not_lost_by_primary_edge_type(has_item):
    env = _stair_env()
    env.graph.edges[0, 1].update(edge_type="stair", label="", edge_constraints=["stair", "key_locked", "item_locked"])
    state = env.state.copy()
    state.keys = 1
    state.has_item = has_item
    solver = StateSpaceAStar(env)
    success, path, diagnostics = solver.solve_with_verified_diagnostics(start_state=state)
    assert success == has_item
    if success:
        assert diagnostics.final_inventory["keys"] == 0
        assert path[-1] == env.goal_pos


@pytest.mark.parametrize("iddfs", [False, True])
def test_dfs_rejects_diagonal_shortcut_between_locked_doors(iddfs):
    grid = np.full((4, 4), P["WALL"], dtype=np.int64)
    grid[1, 1], grid[2, 2] = P["START"], P["TRIFORCE"]
    grid[1, 2] = grid[2, 1] = P["DOOR_LOCKED"]
    result = StateSpaceDFS(ZeldaLogicEnv(grid), allow_diagonals=True, use_iddfs=iddfs).solve()
    assert not result[0]


def test_depth_limited_dfs_reopens_state_reached_with_more_remaining_depth():
    grid = np.full((4, 5), P["WALL"], dtype=np.int64)
    grid[1:3, 1:3] = P["FLOOR"]
    grid[1, 1], grid[1, 3] = P["START"], P["TRIFORCE"]
    success, path, _ = StateSpaceDFS(ZeldaLogicEnv(grid), max_depth=3, use_iddfs=False).solve()
    assert success
    assert len(path) - 1 <= 3


def test_dstar_stateful_fallback_preserves_replan_origin_and_inventory():
    grid = np.full((3, 7), P["WALL"], dtype=np.int64)
    grid[1, 1:6] = P["FLOOR"]
    grid[1, 1], grid[1, 4], grid[1, 5] = P["START"], P["DOOR_LOCKED"], P["TRIFORCE"]
    env = ZeldaLogicEnv(grid)
    current = env.state.copy()
    current.position, current.keys = (1, 3), 1
    original = game_state_key(env.state)
    solver = DStarLiteSolver(env, timeout=100)
    success, path, _ = solver.solve(current)
    assert success
    assert solver.used_fallback
    assert path[0] == current.position
    assert path[-1] == env.goal_pos
    assert game_state_key(env.state) == original
    assert current.keys == 1


def test_dstar_failed_replan_clears_old_route_and_preserves_exhaustion_certificate():
    grid = np.full((3, 7), P["WALL"], dtype=np.int64)
    grid[1, 1:6] = P["FLOOR"]
    grid[1, 1], grid[1, 5] = P["START"], P["TRIFORCE"]
    env = ZeldaLogicEnv(grid)
    solver = DStarLiteSolver(env, timeout=100)
    success, path, states = solver.solve(env.state.copy())
    assert success and path
    assert states == solver._last_compute_iterations
    env.grid[1, 3] = P["WALL"]
    success, path, states = solver.replan(env.state.copy())
    assert not success and not path
    assert solver.current_path == []
    assert solver.last_termination_status == "exhausted"
    assert states == solver._last_compute_iterations


def _independent_distance(grid):
    start, goal = (1, 1), (3, 3)
    queue, seen = deque([(start, 0)]), {start}
    while queue:
        (row, col), distance = queue.popleft()
        if (row, col) == goal:
            return distance
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor = row + dr, col + dc
            if neighbor not in seen and grid[neighbor] != P["WALL"]:
                seen.add(neighbor)
                queue.append((neighbor, distance + 1))
    return None


@pytest.mark.parametrize("algorithm", range(8))
def test_registered_searches_match_independent_small_grid_oracle(algorithm, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr("src.simulation.learned_astar_search.MLHeuristicAStar", lambda *args, **kwargs:
        SimpleNamespace(heuristic=lambda state: float(10000 * state.position[0] + state.position[1]), trained_map_shape=None))
    # Exhaust all 128 obstacle patterns in a 3x3 interior with fixed endpoints.
    interior = [(r, c) for r in range(1, 4) for c in range(1, 4) if (r, c) not in {(1, 1), (3, 3)}]
    for mask in range(1 << len(interior)):
        grid = np.full((5, 5), P["WALL"], dtype=np.int64)
        grid[1, 1], grid[3, 3] = P["START"], P["TRIFORCE"]
        for bit, position in enumerate(interior):
            if mask & (1 << bit):
                grid[position] = P["FLOOR"]
        expected = _independent_distance(grid)
        result = run_game_state_solver(
            ZeldaLogicEnv(grid), algorithm,
            GameStateSearchConfig(timeout=10_000, max_depth=9),
        )
        assert result.success == (expected is not None), (algorithm, mask, result)
        if result.success:
            assert result.path[0] == (1, 1) and result.path[-1] == (3, 3)
            assert all(grid[position] != P["WALL"] for position in result.path)
            assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(result.path, result.path[1:]))
            if algorithm != 3:  # Greedy best-first search has no optimality claim.
                assert len(result.path) - 1 == expected, (algorithm, mask)


@pytest.mark.parametrize("algorithm", range(8))
def test_all_registered_searches_use_same_rule_profile_when_environment_is_reused(algorithm, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr("src.simulation.learned_astar_search.MLHeuristicAStar", lambda *args, **kwargs:
        SimpleNamespace(heuristic=lambda state: 0.0, trained_map_shape=None))
    grid = np.full((4, 4), P["WALL"], dtype=np.int64)
    grid[1:3, 1:3] = P["FLOOR"]
    grid[1, 1], grid[2, 2] = P["START"], P["TRIFORCE"]
    env = ZeldaLogicEnv(grid)
    for profile in ("strict_original", "extended", "vglc_strict"):
        result = run_game_state_solver(env, algorithm, GameStateSearchConfig(rules_profile=profile, allow_diagonals=True))
        assert result.success
        assert env.rules_profile == profile
        assert env.strict_original_mode == (profile == "strict_original")
        assert env.vglc_strict_mode == (profile == "vglc_strict")
        if profile == "strict_original":
            assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(result.path, result.path[1:]))


@pytest.mark.parametrize("weight", [float("nan"), float("inf"), -1.0, 0.0, "bad"])
def test_invalid_heuristic_weight_is_rejected_before_search(weight):
    env = _stair_env()
    with pytest.raises(ValueError, match="finite positive"):
        StateSpaceAStar(env, priority_options={"heuristic_weight": weight})


def test_learned_tiebreak_cannot_silently_become_weighted_astar():
    with pytest.raises(ValueError, match="canonical unweighted"):
        run_game_state_solver(_stair_env(), 7, GameStateSearchConfig(enable_weighted_astar=True, heuristic_weight=2.0))
