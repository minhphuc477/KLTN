import numpy as np

from src.core.definitions import SEMANTIC_PALETTE
from src.simulation.validator import ZeldaLogicEnv


def _two_block_corridor():
    grid = np.full((5, 8), SEMANTIC_PALETTE["WALL"], dtype=np.int64)
    grid[2, 1:7] = SEMANTIC_PALETTE["FLOOR"]
    grid[2, 1] = SEMANTIC_PALETTE["START"]
    grid[2, 3] = SEMANTIC_PALETTE["BLOCK"]
    grid[2, 4] = SEMANTIC_PALETTE["BLOCK"]
    grid[2, 6] = SEMANTIC_PALETTE["TRIFORCE"]
    return grid


def _assert_vacated_block_origin_is_reusable(move):
    grid = _two_block_corridor()
    env = ZeldaLogicEnv(grid)
    state = env.state.copy()

    state.position = (2, 3)
    can_move, state = move(env, state, (2, 4), int(grid[2, 4]))
    assert can_move
    assert ((2, 4), (2, 5)) in state.pushed_blocks

    state.position = (2, 2)
    can_move, state = move(env, state, (2, 3), int(grid[2, 3]))
    assert can_move
    assert ((2, 3), (2, 4)) in state.pushed_blocks

    state.position = (2, 3)
    can_move, unchanged_state = move(env, state, (2, 4), int(grid[2, 4]))
    assert not can_move
    assert unchanged_state == state


def test_canonical_validator_pushes_into_vacated_block_origin_but_not_occupied_destination():
    """Static block origins become floor, while current dynamic positions remain occupied."""

    def move(env, state, target_pos, target_tile):
        return env.try_move_pure(state, target_pos, target_tile)

    _assert_vacated_block_origin_is_reusable(move)


def test_cognitive_search_uses_canonical_block_transition_rules():
    """P-CBS validity checks and moves must not walk through an immovable block."""
    from src.simulation.cognitive_bounded_search import CognitiveBoundedSearch

    grid = np.full((5, 6), SEMANTIC_PALETTE["WALL"], dtype=np.int64)
    grid[2, 1:5] = SEMANTIC_PALETTE["FLOOR"]
    grid[2, 1] = SEMANTIC_PALETTE["START"]
    grid[2, 2] = SEMANTIC_PALETTE["BLOCK"]
    grid[2, 3] = SEMANTIC_PALETTE["WALL"]
    grid[2, 4] = SEMANTIC_PALETTE["TRIFORCE"]

    env = ZeldaLogicEnv(grid)
    search = CognitiveBoundedSearch.__new__(CognitiveBoundedSearch)
    search.env = env

    state = env.state.copy()
    assert not search._can_move_to(state, (2, 2), int(grid[2, 2]))
    can_move, unchanged_state = search._try_move(state, (2, 2), int(grid[2, 2]))
    assert not can_move
    assert unchanged_state == state
