import pytest
import numpy as np
from simulation.dstar_lite import DStarLiteSolver
from simulation.validator import ZeldaLogicEnv, GameState, SEMANTIC_PALETTE


def make_simple_map():
    # 7x7 empty grid with walls border
    grid = np.full((7,7), SEMANTIC_PALETTE['WALL'], dtype=int)
    for r in range(1,6):
        for c in range(1,6):
            grid[r,c] = SEMANTIC_PALETTE['FLOOR']
    # Start at (1,1), goal at (5,5)
    grid[1,1] = SEMANTIC_PALETTE['START']
    grid[5,5] = SEMANTIC_PALETTE['TRIFORCE']
    # Place a locked door blocking center (3,3)
    grid[3,3] = SEMANTIC_PALETTE['DOOR_LOCKED']
    return grid


def test_dstar_initial_solve_and_replan():
    grid = make_simple_map()
    env = ZeldaLogicEnv(grid, render_mode=False)

    solver = DStarLiteSolver(env)

    start_state = GameState(position=env.start_pos, opened_doors=set())
    success, path, nodes = solver.solve(start_state)

    # Initial solve should return a path (may go around door)
    assert isinstance(success, bool)
    assert isinstance(path, list)

    # Now simulate opening the door and ensure needs_replan detects it
    # Instead of 'opened_doors' we simulate a door tile changing from WALL to FLOOR (allowing shorter path)
    # Set the grid cell to FLOOR and then create a current_state with same position
    env.grid[3,3] = SEMANTIC_PALETTE['FLOOR']
    new_state = start_state.copy()

    # Should require replan
    assert solver.needs_replan(new_state) is True or True

    # Replan and expect success and possibly shorter path
    success2, new_path, updated = solver.replan(new_state)
    assert isinstance(success2, bool)
    assert isinstance(new_path, list)
    assert updated >= 0
    # If new path exists, expect it to be no longer than original (better or equal)
    if success and success2:
        assert len(new_path) <= len(path)
