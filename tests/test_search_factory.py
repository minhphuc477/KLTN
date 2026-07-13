from __future__ import annotations

import numpy as np
import pytest

from src.core.definitions import SEMANTIC_PALETTE
from src.ml.heuristic_learning import HeuristicTrainer
from src.simulation.search_base import GameStateSearchConfig
from src.simulation.search_factory import (
    SUPPORTED_GAME_STATE_ALGORITHMS,
    iter_game_state_algorithm_specs,
    run_game_state_solver,
)
from src.simulation.validator import StateSpaceAStar, ZeldaLogicEnv


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
    assert SUPPORTED_GAME_STATE_ALGORITHMS[7] == "A* + Learned Tie-Break"


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
        "learned_tiebreak_astar",
    ]
    assert specs[0].validation_role == "oracle"
    assert specs[0].canonical_use == "hard_oracle"
    assert specs[4].validation_role == "replanning_diagnostic"
    assert specs[4].canonical_use == "incremental_replanning"
    assert specs[7].validation_role == "learned_guidance_ablation"


def test_weighted_astar_uses_canonical_config_keys_with_legacy_alias_support():
    canonical = GameStateSearchConfig(
        enable_weighted_astar=True,
        heuristic_weight=1.75,
    ).to_priority_options()
    legacy = GameStateSearchConfig(
        enable_ara=True,
        ara_weight=1.5,
    ).to_priority_options()

    assert canonical["enable_weighted_astar"] is True
    assert canonical["heuristic_weight"] == 1.75
    assert legacy["enable_weighted_astar"] is True
    assert legacy["heuristic_weight"] == 1.5


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


def test_learned_tiebreak_solver_requires_an_explicit_checkpoint():
    env = ZeldaLogicEnv(_simple_grid(), render_mode=False)
    config = GameStateSearchConfig(timeout=20000)

    with pytest.raises(ValueError, match="checkpoint path is required"):
        run_game_state_solver(env, 7, config)


def test_learned_tiebreak_solver_preserves_astar_path_cost(tmp_path):
    grid = _simple_grid()
    env = ZeldaLogicEnv(grid, render_mode=False)
    trainer = HeuristicTrainer(map_height=grid.shape[0], map_width=grid.shape[1])
    checkpoint = tmp_path / "heuristic.pth"
    trainer.save_model(str(checkpoint))
    config = GameStateSearchConfig(
        timeout=20000,
        learned_heuristic_model_path=str(checkpoint),
    )

    canonical = run_game_state_solver(env, 0, config)
    learned = run_game_state_solver(env, 7, config)

    assert canonical.success is True
    assert learned.success is True
    assert len(learned.path) == len(canonical.path)
    assert learned.metadata["optimality_contract"] == "canonical_f_primary_neural_tiebreak_only"
    assert learned.metadata["secondary_heuristic"] == "learned_cost"


def test_diagnostic_astar_executes_secondary_tiebreaker_without_changing_primary_f():
    env = ZeldaLogicEnv(_simple_grid(), render_mode=False)
    calls = []

    def secondary(state):
        calls.append(tuple(state.position))
        return float(state.position[0] + state.position[1])

    solver = StateSpaceAStar(
        env,
        timeout=20000,
        search_mode="astar",
        priority_options={
            "secondary_heuristic": secondary,
            "secondary_heuristic_name": "probe",
            "representation": "tile",
        },
    )
    success, path, _diagnostics = solver.solve_with_diagnostics()

    assert success is True
    assert len(path) > 0
    assert len(calls) > 1
