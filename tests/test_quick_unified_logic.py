"""
Quick Verification Test: StateSpaceAStar and StateSpaceDFS
==========================================================
Tests that both algorithms now use complete game logic.
"""

import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.core.definitions import SEMANTIC_PALETTE
from src.simulation.validator import ZeldaLogicEnv, StateSpaceAStar
from src.simulation.state_space_dfs import StateSpaceDFS


def test_key_door():
    """Test key + locked door."""
    print("Test 1: Key + Locked Door")
    grid = np.full((5, 7), SEMANTIC_PALETTE['WALL'], dtype=np.int64)
    grid[2, 1:6] = SEMANTIC_PALETTE['FLOOR']
    grid[2, 1] = SEMANTIC_PALETTE['START']
    grid[2, 5] = SEMANTIC_PALETTE['TRIFORCE']
    grid[2, 2] = SEMANTIC_PALETTE['KEY_SMALL']
    grid[2, 3] = SEMANTIC_PALETTE['DOOR_LOCKED']
    
    # Test A*
    env1 = ZeldaLogicEnv(grid)
    astar = StateSpaceAStar(env1)
    success1, path1, _ = astar.solve_with_diagnostics()
    assert success1, "A* failed"
    assert (2, 2) in path1 and (2, 3) in path1, "A* didn't use key+door"
    print(f"  ✓ A*: success, path_len={len(path1)}")
    
    # Test DFS
    env2 = ZeldaLogicEnv(grid)
    dfs = StateSpaceDFS(env2)
    success2, path2, _ = dfs.solve()
    assert success2, "DFS failed"
    assert (2, 2) in path2 and (2, 3) in path2, "DFS didn't use key+door"
    print(f"  ✓ DFS: success, path_len={len(path2)}")


def test_bomb_door():
    """Test bomb + bomb door."""
    print("\nTest 2: Bomb + Bomb Door")
    grid = np.full((5, 7), SEMANTIC_PALETTE['WALL'], dtype=np.int64)
    grid[2, 1:6] = SEMANTIC_PALETTE['FLOOR']
    grid[2, 1] = SEMANTIC_PALETTE['START']
    grid[2, 5] = SEMANTIC_PALETTE['TRIFORCE']
    grid[2, 2] = SEMANTIC_PALETTE['ITEM_MINOR']
    grid[2, 3] = SEMANTIC_PALETTE['DOOR_BOMB']
    
    env1 = ZeldaLogicEnv(grid)
    astar = StateSpaceAStar(env1)
    success1, path1, _ = astar.solve_with_diagnostics()
    assert success1, "A* failed"
    assert (2, 2) in path1 and (2, 3) in path1, "A* didn't use bomb+door"
    print(f"  ✓ A*: success, path_len={len(path1)}")
    
    env2 = ZeldaLogicEnv(grid)
    dfs = StateSpaceDFS(env2)
    success2, path2, _ = dfs.solve()
    assert success2, "DFS failed"
    assert (2, 2) in path2 and (2, 3) in path2, "DFS didn't use bomb+door"
    print(f"  ✓ DFS: success, path_len={len(path2)}")


def test_boss_door():
    """Test boss key + boss door."""
    print("\nTest 3: Boss Key + Boss Door")
    grid = np.full((5, 7), SEMANTIC_PALETTE['WALL'], dtype=np.int64)
    grid[2, 1:6] = SEMANTIC_PALETTE['FLOOR']
    grid[2, 1] = SEMANTIC_PALETTE['START']
    grid[2, 5] = SEMANTIC_PALETTE['TRIFORCE']
    grid[2, 2] = SEMANTIC_PALETTE['KEY_BOSS']
    grid[2, 3] = SEMANTIC_PALETTE['DOOR_BOSS']
    
    env1 = ZeldaLogicEnv(grid)
    astar = StateSpaceAStar(env1)
    success1, path1, _ = astar.solve_with_diagnostics()
    assert success1, "A* failed"
    assert (2, 2) in path1 and (2, 3) in path1, "A* didn't use boss_key+door"
    print(f"  ✓ A*: success, path_len={len(path1)}")
    
    env2 = ZeldaLogicEnv(grid)
    dfs = StateSpaceDFS(env2)
    success2, path2, _ = dfs.solve()
    assert success2, "DFS failed"
    assert (2, 2) in path2 and (2, 3) in path2, "DFS didn't use boss_key+door"
    print(f"  ✓ DFS: success, path_len={len(path2)}")


def test_block_pushing():
    """Test block pushing."""
    print("\nTest 4: Block Pushing")
    # Need extra space for block to be pushed to
    grid = np.full((5, 8), SEMANTIC_PALETTE['WALL'], dtype=np.int64)
    grid[2, 1:7] = SEMANTIC_PALETTE['FLOOR']  # Longer corridor
    grid[2, 1] = SEMANTIC_PALETTE['START']
    grid[2, 6] = SEMANTIC_PALETTE['TRIFORCE']
    grid[2, 3] = SEMANTIC_PALETTE['BLOCK']
    # Block can be pushed from (2,3) to (2,4), agent takes (2,3)
    
    env1 = ZeldaLogicEnv(grid)
    astar = StateSpaceAStar(env1)
    success1, path1, _ = astar.solve_with_diagnostics()
    assert success1, "A* failed"
    assert (2, 3) in path1, "A* didn't push block"
    print(f"  ✓ A*: success, path_len={len(path1)}")
    
    env2 = ZeldaLogicEnv(grid)
    dfs = StateSpaceDFS(env2)
    success2, path2, _ = dfs.solve()
    assert success2, "DFS failed"
    assert (2, 3) in path2, "DFS didn't push block"
    print(f"  ✓ DFS: success, path_len={len(path2)}")


def test_water_ladder():
    """Test water + ladder."""
    print("\nTest 5: Water + Ladder")
    grid = np.full((5, 7), SEMANTIC_PALETTE['WALL'], dtype=np.int64)
    grid[2, 1:6] = SEMANTIC_PALETTE['FLOOR']
    grid[2, 1] = SEMANTIC_PALETTE['START']
    grid[2, 5] = SEMANTIC_PALETTE['TRIFORCE']
    grid[2, 2] = SEMANTIC_PALETTE['KEY_ITEM']
    grid[2, 3] = SEMANTIC_PALETTE['ELEMENT']
    
    env1 = ZeldaLogicEnv(grid)
    astar = StateSpaceAStar(env1)
    success1, path1, _ = astar.solve_with_diagnostics()
    assert success1, "A* failed"
    assert (2, 2) in path1 and (2, 3) in path1, "A* didn't use ladder+water"
    print(f"  ✓ A*: success, path_len={len(path1)}")
    
    env2 = ZeldaLogicEnv(grid)
    dfs = StateSpaceDFS(env2)
    success2, path2, _ = dfs.solve()
    assert success2, "DFS failed"
    assert (2, 2) in path2 and (2, 3) in path2, "DFS didn't use ladder+water"
    print(f"  ✓ DFS: success, path_len={len(path2)}")


if __name__ == "__main__":
    print("=" * 70)
    print("QUICK VERIFICATION: Unified Game Logic")
    print("Testing: StateSpaceAStar and StateSpaceDFS")
    print("=" * 70)
    print()
    
    try:
        test_key_door()
        test_bomb_door()
        test_boss_door()
        test_block_pushing()
        test_water_ladder()
        
        print()
        print("=" * 70)
        print("ALL TESTS PASSED! ✓")
        print("Both algorithms now use unified game logic.")
        print("=" * 70)
    except AssertionError as e:
        print()
        print("=" * 70)
        print(f"TEST FAILED: {e}")
        print("=" * 70)
        raise
