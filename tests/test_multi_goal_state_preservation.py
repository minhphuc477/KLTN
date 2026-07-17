import networkx as nx
import numpy as np
import pytest

from src.core.definitions import SEMANTIC_PALETTE
from src.simulation import MultiGoalResult as ExportedMultiGoalResult
from src.simulation.multi_goal import MultiGoalPathfinder, MultiGoalResult
from src.simulation.state import GameState
from src.simulation.validation_types import SolverOptions
from src.simulation.validator import StateSpaceAStar, ZeldaLogicEnv


P = SEMANTIC_PALETTE


def _corridor(width: int) -> np.ndarray:
    grid = np.full((3, width), P["WALL"], dtype=np.int64)
    grid[1, 1 : width - 1] = P["FLOOR"]
    return grid


def test_route_preserves_inventory_and_opened_door_between_objectives():
    grid = _corridor(7)
    grid[1, 1] = P["START"]
    grid[1, 2] = P["KEY_SMALL"]
    grid[1, 3] = P["DOOR_LOCKED"]
    grid[1, 5] = P["TRIFORCE"]
    env = ZeldaLogicEnv(grid)
    start_state = env.state.copy()
    original_env_state = env.state.copy()

    result = MultiGoalPathfinder(env, max_states=1_000).find_optimal_collection_order(
        start_state,
        [P["KEY_SMALL"]],
    )

    assert isinstance(result, MultiGoalResult)
    assert MultiGoalResult is ExportedMultiGoalResult
    assert result.success
    assert result.optimal
    assert result.failure_reason is None
    assert result.full_path == [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)]
    assert result.waypoints == [(1, 2), (1, 5)]
    assert result.collected_targets == [(1, 2)]
    assert result.remaining_targets == []
    assert result.total_cost == pytest.approx(5.5)

    key_state, goal_state = result.segment_end_states
    assert key_state.keys == 1
    assert key_state.collected_items == {(1, 2)}
    assert goal_state.keys == 0
    assert goal_state.opened_doors == {(1, 3)}
    assert result.terminal_state == goal_state

    # Planning is pure with respect to both caller-owned state objects.
    assert start_state == GameState(
        position=(1, 1),
        bomb_count=original_env_state.bomb_count,
    )
    assert env.state == original_env_state

    replay_solver = StateSpaceAStar(
        env,
        priority_options={"representation": "tile"},
    )
    verified, error, replay_state, replay_cost = replay_solver.verify_position_path(
        result.full_path
    )
    assert verified, error
    assert replay_state == result.terminal_state
    assert replay_cost == pytest.approx(result.total_cost)


def test_search_chooses_the_globally_cheapest_collection_order():
    grid = np.full((5, 9), P["WALL"], dtype=np.int64)
    grid[1:4, 1:8] = P["FLOOR"]
    grid[2, 4] = P["START"]
    grid[1, 2] = P["KEY_SMALL"]
    grid[3, 6] = P["KEY_SMALL"]
    grid[2, 7] = P["TRIFORCE"]
    env = ZeldaLogicEnv(grid)

    result = MultiGoalPathfinder(env, max_states=20_000).find_optimal_collection_order(
        env.state.copy(),
        [P["KEY_SMALL"]],
    )

    assert result.success, result.failure_reason
    assert result.waypoints == [(1, 2), (3, 6), (2, 7)]
    assert result.collected_targets == [(1, 2), (3, 6)]
    assert result.optimal


def test_route_preserves_staged_puzzle_progress_between_waypoints():
    grid = _corridor(8)
    grid[1, 1] = P["START"]
    grid[1, 2] = P["KEY_SMALL"]
    grid[1, 3] = P["KEY_ITEM"]
    grid[1, 4] = P["PUZZLE"]
    grid[1, 5] = P["DOOR_PUZZLE"]
    grid[1, 6] = P["TRIFORCE"]
    puzzle_metadata = {
        "version": "stateful_v1",
        "plans": {
            "corridor": {
                "plan_id": "corridor",
                "controlled_doors_global": [[1, 5]],
                "stage_sequence": [
                    {
                        "stage_index": 0,
                        "kind": "collect_key",
                        "global_anchor": [1, 2],
                        "trigger_tile_id": int(P["KEY_SMALL"]),
                    },
                    {
                        "stage_index": 1,
                        "kind": "collect_item",
                        "global_anchor": [1, 3],
                        "trigger_tile_id": int(P["KEY_ITEM"]),
                    },
                    {
                        "stage_index": 2,
                        "kind": "step_on_puzzle",
                        "global_anchor": [1, 4],
                        "trigger_tile_id": int(P["PUZZLE"]),
                    },
                ],
            }
        },
    }
    env = ZeldaLogicEnv(grid, room_puzzle_metadata=puzzle_metadata)

    result = MultiGoalPathfinder(env, max_states=2_000).find_optimal_collection_order(
        env.state.copy(),
        [P["KEY_SMALL"], P["KEY_ITEM"]],
    )

    assert result.success, result.failure_reason
    assert result.waypoints == [(1, 2), (1, 3), (1, 6)]
    assert result.segment_end_states[0].completed_puzzle_stages == {("corridor", 0)}
    assert result.segment_end_states[1].completed_puzzle_stages == {
        ("corridor", 0),
        ("corridor", 1),
    }
    assert result.terminal_state is not None
    assert result.terminal_state.completed_puzzle_stages == {
        ("corridor", 0),
        ("corridor", 1),
        ("corridor", 2),
    }
    assert (1, 5) in result.terminal_state.opened_doors
    assert result.terminal_state.has_item
    assert {(1, 2), (1, 3)}.issubset(result.terminal_state.collected_items)


def test_route_preserves_consumed_inventory_and_open_graph_edge():
    grid = np.full((16, 38), P["WALL"], dtype=np.int64)
    grid[1:10, 1:15] = P["FLOOR"]
    grid[1:10, 23:37] = P["FLOOR"]
    grid[1, 1] = P["START"]
    grid[1, 2] = P["KEY_SMALL"]
    grid[1, 24] = P["TRIFORCE"]
    graph = nx.DiGraph()
    graph.add_edge(0, 1, edge_type="key_locked")
    env = ZeldaLogicEnv(
        grid,
        graph=graph,
        room_positions={(0, 0): (0, 0), (0, 2): (0, 22)},
        room_to_node={(0, 0): 0, (0, 2): 1},
        node_to_room={0: (0, 0), 1: (0, 2)},
        solver_options=SolverOptions(rules_profile="extended", start_bombs=0),
    )

    result = MultiGoalPathfinder(env, max_states=5_000).find_optimal_collection_order(
        env.state.copy(),
        [P["KEY_SMALL"]],
    )

    assert result.success, result.failure_reason
    assert result.terminal_state is not None
    assert result.terminal_state.keys == 0
    assert result.terminal_state.opened_graph_edges == {(0, 1)}
    assert (1, 2) in result.terminal_state.collected_items

    replay_solver = StateSpaceAStar(
        env,
        priority_options={"representation": "tile", "rules_profile": "extended"},
    )
    verified, error, replay_state, replay_cost = replay_solver.verify_position_path(
        result.full_path
    )
    assert verified, error
    assert replay_state == result.terminal_state
    assert replay_cost == pytest.approx(result.total_cost)


def test_target_bound_returns_explicit_failure_without_searching():
    grid = _corridor(8)
    grid[1, 1] = P["START"]
    grid[1, 2] = P["KEY_SMALL"]
    grid[1, 4] = P["KEY_SMALL"]
    grid[1, 6] = P["TRIFORCE"]
    env = ZeldaLogicEnv(grid)

    result = MultiGoalPathfinder(
        env,
        max_states=100,
        max_targets=1,
    ).find_optimal_collection_order(env.state.copy(), [P["KEY_SMALL"]])

    assert not result.success
    assert not result.optimal
    assert result.exploration_count == 0
    assert result.remaining_targets == [(1, 2), (1, 4)]
    assert result.failure_reason == (
        "target limit exceeded: 2 objectives is greater than max_targets=1"
    )


def test_state_bound_is_reported_as_inconclusive():
    grid = _corridor(6)
    grid[1, 1] = P["START"]
    grid[1, 4] = P["TRIFORCE"]
    env = ZeldaLogicEnv(grid)

    result = MultiGoalPathfinder(env, max_states=1).find_optimal_collection_order(
        env.state.copy(),
        [],
    )

    assert not result.success
    assert result.state_limit_reached
    assert result.exploration_count == 1
    assert result.failure_reason == "state limit reached: max_states=1"
    assert result.terminal_state is None
