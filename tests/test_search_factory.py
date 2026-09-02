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
from src.simulation.dstar_lite import DStarLiteSolver
from src.simulation.validator import StateSpaceAStar, ZeldaLogicEnv, ZeldaValidator


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
        assert result.metadata["termination_status"] == "solved"
        assert result.metadata["solver_status"] == "solved"


def test_search_factory_astar_exposes_oracle_metadata():
    env = ZeldaLogicEnv(_simple_grid(), render_mode=False)
    config = GameStateSearchConfig(timeout=20000, allow_diagonals=False)
    result = run_game_state_solver(env, 0, config)
    assert result.success is True
    assert result.metadata["oracle_status"] == "solved"
    assert result.metadata["termination_status"] == "solved"
    assert "failure_reason" in result.metadata


def test_comparison_solvers_preserve_explicit_budget_status():
    for algorithm_idx in (0, 1, 2, 3, 4, 5, 6):
        env = ZeldaLogicEnv(_simple_grid(), render_mode=False)
        result = run_game_state_solver(
            env,
            algorithm_idx,
            GameStateSearchConfig(timeout=1, allow_diagonals=False),
        )
        status_key = "oracle_status" if algorithm_idx == 0 else "solver_status"
        assert result.success is False
        assert result.metadata["termination_status"] == "budget_exhausted"
        assert result.metadata[status_key] == "timeout"


def test_dstar_fallback_shares_the_configured_search_budget():
    env = ZeldaLogicEnv(_simple_grid(), render_mode=False)
    solver = DStarLiteSolver(env, timeout=1, allow_diagonals=False)

    success, path, states = solver.solve(env.state.copy())

    assert success is False
    assert path == []
    assert solver.used_fallback is True
    assert states <= 1


def test_iddfs_distinguishes_depth_cutoff_from_complete_no_path():
    env = ZeldaLogicEnv(_simple_grid(), render_mode=False)
    cutoff = run_game_state_solver(
        env,
        5,
        GameStateSearchConfig(timeout=10_000, max_depth=1, use_iddfs=True),
    )

    blocked = _simple_grid()
    blocked[2, 1:-1] = int(SEMANTIC_PALETTE["WALL"])
    no_path = run_game_state_solver(
        ZeldaLogicEnv(blocked, render_mode=False),
        5,
        GameStateSearchConfig(timeout=10_000, max_depth=32, use_iddfs=True),
    )

    assert cutoff.success is False
    assert cutoff.metadata["termination_status"] == "depth_limit"
    assert cutoff.metadata["solver_status"] == "inconclusive"
    assert no_path.success is False
    assert no_path.metadata["termination_status"] == "exhausted"
    assert no_path.metadata["solver_status"] == "no_path"


def test_iterative_dfs_checks_goal_at_the_exact_depth_limit():
    grid = np.full((3, 4), int(SEMANTIC_PALETTE["WALL"]), dtype=np.int64)
    grid[1, 1] = int(SEMANTIC_PALETTE["START"])
    grid[1, 2] = int(SEMANTIC_PALETTE["TRIFORCE"])
    result = run_game_state_solver(
        ZeldaLogicEnv(grid, render_mode=False),
        5,
        GameStateSearchConfig(timeout=100, max_depth=1, use_iddfs=False),
    )

    assert result.success is True
    assert result.path == [(1, 1), (1, 2)]
    assert result.metadata["termination_status"] == "solved"


def test_exact_validator_does_not_rerun_after_complete_frontier_exhaustion():
    blocked = _simple_grid()
    blocked[2, 1:-1] = int(SEMANTIC_PALETTE["WALL"])

    result = ZeldaValidator().validate_single(
        blocked,
        solver_timeout=10_000,
        run_dijkstra_fallback=True,
    )

    assert result.is_solvable is False
    assert result.termination_status == "exhausted"
    assert result.proven_unsolvable is True
    assert result.solver_used == "astar"


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
