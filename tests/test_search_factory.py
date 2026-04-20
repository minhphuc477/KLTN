from __future__ import annotations

import numpy as np

from src.core.definitions import SEMANTIC_PALETTE
from src.simulation.search_base import GameStateSearchConfig
from src.simulation.search_factory import (
    SUPPORTED_GAME_STATE_ALGORITHMS,
    iter_game_state_algorithm_specs,
    run_game_state_solver,
)
from src.simulation.validator import ZeldaLogicEnv


def _simple_grid() -> np.ndarray:
    grid = np.full((6, 8), int(SEMANTIC_PALETTE["WALL"]), dtype=np.int64)
    grid[1:-1, 1:-1] = int(SEMANTIC_PALETTE["FLOOR"])
    grid[1, 1] = int(SEMANTIC_PALETTE["START"])
    grid[4, 6] = int(SEMANTIC_PALETTE["TRIFORCE"])
    return grid


def test_search_factory_exposes_advanced_state_space_algorithms():
    assert SUPPORTED_GAME_STATE_ALGORITHMS[4] == "D* Lite"
    assert SUPPORTED_GAME_STATE_ALGORITHMS[5] == "DFS/IDDFS"
    assert SUPPORTED_GAME_STATE_ALGORITHMS[6] == "Bidirectional A*"


def test_search_factory_exposes_canonical_validation_registry():
    specs = list(iter_game_state_algorithm_specs())
    assert [spec.key for spec in specs] == [
        "astar",
        "bfs",
        "dijkstra",
        "greedy",
        "dstar_lite",
        "dfs_iddfs",
        "bidirectional_astar",
    ]
    assert specs[0].validation_role == "oracle"
    assert specs[0].canonical_use == "hard_oracle"
    assert specs[4].validation_role == "replanning"
    assert specs[4].canonical_use == "incremental_replanning"


def test_search_factory_runs_advanced_algorithms_on_simple_grid():
    env = ZeldaLogicEnv(_simple_grid(), render_mode=False)
    config = GameStateSearchConfig(timeout=20000, allow_diagonals=False, max_depth=128, use_iddfs=True)

    for algorithm_idx in (4, 5, 6):
        result = run_game_state_solver(env, algorithm_idx, config)
        assert result.success is True
        assert len(result.path) > 0
        assert result.states_explored > 0


def test_search_factory_astar_exposes_oracle_metadata():
    env = ZeldaLogicEnv(_simple_grid(), render_mode=False)
    config = GameStateSearchConfig(timeout=20000, allow_diagonals=False)
    result = run_game_state_solver(env, 0, config)
    assert result.success is True
    assert result.metadata["oracle_status"] == "solved"
    assert "failure_reason" in result.metadata
