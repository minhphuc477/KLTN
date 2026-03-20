import numpy as np

from src.core.definitions import SEMANTIC_PALETTE
from src.simulation.validator import GameState, SolverOptions, StateSpaceAStar, ZeldaLogicEnv


def _build_two_room_grid():
    grid = np.full((16, 22), SEMANTIC_PALETTE['FLOOR'], dtype=np.int64)
    grid[8, 9] = SEMANTIC_PALETTE['START']
    grid[8, 15] = SEMANTIC_PALETTE['TRIFORCE']
    grid[6, 4] = SEMANTIC_PALETTE['ENEMY']
    # Soft door on the room boundary crossing tile.
    grid[8, 11] = SEMANTIC_PALETTE['DOOR_SOFT']
    return grid


def test_strict_original_soft_door_requires_room_clear():
    grid = _build_two_room_grid()
    env = ZeldaLogicEnv(
        grid,
        render_mode=False,
        room_positions={(0, 0): (0, 0), (0, 1): (0, 11)},
        solver_options=SolverOptions(rules_profile="strict_original"),
    )

    state = env.state.copy()
    state.position = (8, 10)  # Left room, directly before soft door.

    can_move, _ = env._try_move_pure(state, (8, 11), int(grid[8, 11]))
    assert not can_move

    cleared = state.copy()
    cleared.defeated_enemies.add((6, 4))
    can_move_cleared, _ = env._try_move_pure(cleared, (8, 11), int(grid[8, 11]))
    assert can_move_cleared


def test_extended_mode_keeps_soft_door_passable():
    grid = _build_two_room_grid()
    env = ZeldaLogicEnv(
        grid,
        render_mode=False,
        room_positions={(0, 0): (0, 0), (0, 1): (0, 11)},
        solver_options=SolverOptions(rules_profile="extended"),
    )

    state = env.state.copy()
    state.position = (8, 10)
    can_move, _ = env._try_move_pure(state, (8, 11), int(grid[8, 11]))
    assert can_move


def test_strict_original_solver_disables_diagonal_and_hierarchical():
    grid = _build_two_room_grid()
    env = ZeldaLogicEnv(
        grid,
        render_mode=False,
        room_positions={(0, 0): (0, 0), (0, 1): (0, 11)},
        solver_options=SolverOptions(rules_profile="strict_original"),
    )
    solver = StateSpaceAStar(
        env,
        priority_options={"allow_diagonals": True, "rules_profile": "strict_original", "representation": "hybrid"},
    )

    assert solver.strict_original_mode is True
    assert solver.allow_diagonals is False
    assert solver.enable_hierarchical is False
