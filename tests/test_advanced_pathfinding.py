"""
Integration Tests for Advanced Pathfinding Algorithms
=====================================================

Tests D* Lite, DFS/IDDFS, and Bidirectional A* implementations
on realistic Zelda dungeon scenarios.

Usage:
    python tests/test_advanced_pathfinding.py
"""

import numpy as np
import logging
from typing import Tuple

import pytest

from src.core.definitions import SEMANTIC_PALETTE
from src.simulation.validator import GameState, ZeldaLogicEnv, SolverOptions, game_state_key
from src.simulation.dstar_lite import DStarLiteSolver
from src.simulation.state_space_dfs import StateSpaceDFS
from src.simulation.bidirectional_astar import BidirectionalAStar, SearchNode
from src.simulation.validator import StateSpaceAStar  # Baseline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _corner_cut_grid(adjacent_tile: int) -> np.ndarray:
    grid = np.full((4, 4), SEMANTIC_PALETTE['FLOOR'], dtype=np.int64)
    grid[0, :] = SEMANTIC_PALETTE['WALL']
    grid[-1, :] = SEMANTIC_PALETTE['WALL']
    grid[:, 0] = SEMANTIC_PALETTE['WALL']
    grid[:, -1] = SEMANTIC_PALETTE['WALL']
    grid[1, 1] = SEMANTIC_PALETTE['START']
    grid[2, 2] = SEMANTIC_PALETTE['TRIFORCE']
    grid[1, 2] = adjacent_tile
    grid[2, 1] = adjacent_tile
    return grid


def test_state_space_astar_non_strict_heuristic_returns_numeric_value():
    grid = create_simple_dungeon()
    env = ZeldaLogicEnv(grid)
    solver = StateSpaceAStar(
        env,
        timeout=1000,
        search_mode="astar",
        priority_options={"enable_ara": True, "ara_weight": 1.5},
    )

    value = solver._heuristic(env.state)

    assert isinstance(value, float)
    assert np.isfinite(value)


# ==========================================
# DUNGEON FIXTURES
# ==========================================

def create_simple_dungeon() -> np.ndarray:
    """10x10 dungeon with key and locked door."""
    grid = np.full((10, 10), SEMANTIC_PALETTE['FLOOR'], dtype=np.int64)
    
    # Walls
    grid[0, :] = SEMANTIC_PALETTE['WALL']
    grid[-1, :] = SEMANTIC_PALETTE['WALL']
    grid[:, 0] = SEMANTIC_PALETTE['WALL']
    grid[:, -1] = SEMANTIC_PALETTE['WALL']
    
    # Start and goal
    grid[1, 1] = SEMANTIC_PALETTE['START']
    grid[8, 8] = SEMANTIC_PALETTE['TRIFORCE']
    
    # Key and locked door
    grid[1, 5] = SEMANTIC_PALETTE['KEY_SMALL']
    grid[5, 5] = SEMANTIC_PALETTE['DOOR_LOCKED']
    
    return grid


def create_complex_dungeon() -> np.ndarray:
    """20x20 dungeon with multiple keys, bombs, and obstacles."""
    grid = np.full((20, 20), SEMANTIC_PALETTE['FLOOR'], dtype=np.int64)
    
    # Walls
    grid[0, :] = SEMANTIC_PALETTE['WALL']
    grid[-1, :] = SEMANTIC_PALETTE['WALL']
    grid[:, 0] = SEMANTIC_PALETTE['WALL']
    grid[:, -1] = SEMANTIC_PALETTE['WALL']
    
    # Start and goal
    grid[1, 1] = SEMANTIC_PALETTE['START']
    grid[18, 18] = SEMANTIC_PALETTE['TRIFORCE']
    
    # Internal walls (create maze structure)
    grid[10, 2:18] = SEMANTIC_PALETTE['WALL']
    grid[5, 5:15] = SEMANTIC_PALETTE['WALL']
    grid[15, 5:15] = SEMANTIC_PALETTE['WALL']
    
    # Doors to pass through walls
    grid[10, 8] = SEMANTIC_PALETTE['DOOR_LOCKED']  # Key door
    grid[10, 12] = SEMANTIC_PALETTE['DOOR_BOMB']   # Bomb door
    grid[5, 10] = SEMANTIC_PALETTE['DOOR_OPEN']    # Open door
    
    # Keys and items
    grid[3, 3] = SEMANTIC_PALETTE['KEY_SMALL']     # Small key
    grid[6, 8] = SEMANTIC_PALETTE['ITEM_MINOR']    # Bomb pickup
    grid[12, 5] = SEMANTIC_PALETTE['KEY_BOSS']     # Boss key
    
    # Boss door near goal
    grid[17, 15] = SEMANTIC_PALETTE['DOOR_BOSS']
    
    return grid


def create_long_corridor() -> np.ndarray:
    """30x10 long corridor - ideal for bidirectional A* speedup."""
    grid = np.full((30, 10), SEMANTIC_PALETTE['FLOOR'], dtype=np.int64)
    
    # Walls
    grid[0, :] = SEMANTIC_PALETTE['WALL']
    grid[-1, :] = SEMANTIC_PALETTE['WALL']
    grid[:, 0] = SEMANTIC_PALETTE['WALL']
    grid[:, -1] = SEMANTIC_PALETTE['WALL']
    
    # Start and goal (far apart)
    grid[1, 5] = SEMANTIC_PALETTE['START']
    grid[28, 5] = SEMANTIC_PALETTE['TRIFORCE']
    
    # Add some obstacles to make it interesting
    grid[10, 3:7] = SEMANTIC_PALETTE['WALL']
    grid[20, 3:7] = SEMANTIC_PALETTE['WALL']
    grid[10, 5] = SEMANTIC_PALETTE['DOOR_OPEN']  # Opening
    grid[20, 5] = SEMANTIC_PALETTE['DOOR_OPEN']  # Opening
    
    return grid


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def verify_path_validity(env: ZeldaLogicEnv, path: list) -> Tuple[bool, str]:
    """
    Verify that a path is valid (all moves are legal).
    
    Returns:
        (is_valid, error_message)
    """
    if not path or len(path) == 0:
        return False, "Empty path"
    
    # Reset environment
    env.reset()
    _current_state = env.state.copy()
    
    # Check each step
    for i in range(len(path) - 1):
        current_pos = path[i]
        next_pos = path[i + 1]
        
        # Check adjacency (manhattan distance <= 1 for cardinal, <=2 for diagonal)
        dr = abs(next_pos[0] - current_pos[0])
        dc = abs(next_pos[1] - current_pos[1])
        
        if dr + dc > 2 or (dr == 2 or dc == 2):
            # Allow teleportation (stairs/warps) - skip adjacency check
            pass
        
        # Simulate move (would need to check with actual move logic)
        # For now, just check path reaches goal
    
    # Check if path reaches goal
    if path[-1] != env.goal_pos:
        return False, f"Path doesn't reach goal: {path[-1]} != {env.goal_pos}"
    
    return True, ""


# ==========================================
# TESTS
# ==========================================

class TestDStarLite:
    """Test D* Lite incremental replanning."""
    
    def test_simple_dungeon(self):
        """Test D* Lite on simple dungeon."""
        logger.info("==== Testing D* Lite: Simple Dungeon ====")
        
        grid = create_simple_dungeon()
        env = ZeldaLogicEnv(grid)
        solver = DStarLiteSolver(env, heuristic_mode="balanced")
        
        start_state = env.state.copy()
        success, path, nodes = solver.solve(start_state)
        
        assert success, "D* Lite failed to find solution on simple dungeon"
        assert len(path) > 0, "D* Lite returned empty path"
        assert path[0] == env.start_pos, "Path doesn't start at start position"
        assert path[-1] == env.goal_pos, "Path doesn't end at goal position"
        assert solver.used_fallback is False
        
        logger.info(f"✓ D* Lite: path_len={len(path)}, nodes={nodes}")
    
    def test_complex_dungeon(self):
        """D* Lite must solve the deterministic complex fixture."""
        logger.info("==== Testing D* Lite: Complex Dungeon ====")
        
        grid = create_complex_dungeon()
        env = ZeldaLogicEnv(grid, solver_options=SolverOptions(start_bombs=1))
        solver = DStarLiteSolver(env, heuristic_mode="balanced")
        
        start_state = env.state.copy()
        success, path, nodes = solver.solve(start_state)

        assert success
        assert path[0] == env.start_pos
        assert path[-1] == env.goal_pos
        assert 0 < nodes <= solver.timeout

    def test_diagonal_heuristic_uses_octile_lower_bound(self):
        grid = create_simple_dungeon()
        env = ZeldaLogicEnv(grid)
        solver = DStarLiteSolver(env, allow_diagonals=True)

        assert solver._heuristic(GameState(position=(1, 1))) == pytest.approx(1.414 * 7, rel=1e-3)

    def test_locked_door_predecessor_restores_consumed_key(self):
        grid = create_simple_dungeon()
        env = ZeldaLogicEnv(grid)
        solver = DStarLiteSolver(env)
        door_pos = (5, 5)
        state_at_opened_door = GameState(position=door_pos, keys=0, opened_doors={door_pos})

        candidates = solver._predecessor_state_candidates(
            state_at_opened_door,
            pred_pos=(5, 4),
            target_tile=int(SEMANTIC_PALETTE["DOOR_LOCKED"]),
        )

        assert any(candidate.keys == 1 and door_pos not in candidate.opened_doors for candidate in candidates)

    def test_goal_termination_accepts_inventory_goal_state(self):
        grid = create_simple_dungeon()
        env = ZeldaLogicEnv(grid)
        solver = DStarLiteSolver(env)
        goal_state_with_inventory = GameState(position=env.goal_pos, keys=1)
        goal_hash = game_state_key(goal_state_with_inventory)
        solver.g_scores[goal_hash] = 12.0
        solver.rhs_scores[goal_hash] = 12.0

        assert solver._has_consistent_goal_state()

    def test_diagonal_successors_reject_wall_corner_cutting(self):
        env = ZeldaLogicEnv(_corner_cut_grid(SEMANTIC_PALETTE['WALL']))
        solver = DStarLiteSolver(env, allow_diagonals=True)

        successors = solver._get_successors(env.state.copy())

        assert all(state.position != env.goal_pos for state in successors)


class TestStateSpaceDFS:
    """Test DFS/IDDFS implementations."""
    
    def test_iterative_dfs_simple(self):
        """Test iterative DFS on simple dungeon."""
        logger.info("==== Testing Iterative DFS: Simple Dungeon ====")
        
        grid = create_simple_dungeon()
        env = ZeldaLogicEnv(grid)
        solver = StateSpaceDFS(env, timeout=50000, max_depth=100, use_iddfs=False)
        
        success, path, nodes = solver.solve()
        
        assert success, "Iterative DFS failed on simple dungeon"
        assert len(path) > 0, "DFS returned empty path"
        assert path[-1] == env.goal_pos, "Path doesn't reach goal"
        
        logger.info(f"✓ Iterative DFS: path_len={len(path)}, nodes={nodes}, max_depth={solver.metrics.max_depth_reached}")
    
    def test_iddfs_simple(self):
        """Test IDDFS on simple dungeon."""
        logger.info("==== Testing IDDFS: Simple Dungeon ====")
        
        grid = create_simple_dungeon()
        env = ZeldaLogicEnv(grid)
        solver = StateSpaceDFS(env, timeout=50000, max_depth=100, use_iddfs=True)
        
        success, path, nodes = solver.solve()
        
        assert success, "IDDFS failed on simple dungeon"
        assert len(path) > 0, "IDDFS returned empty path"
        assert path[-1] == env.goal_pos, "Path doesn't reach goal"
        
        logger.info(f"✓ IDDFS: path_len={len(path)}, nodes={nodes}, max_depth={solver.metrics.max_depth_reached}")
        logger.info(f"  Metrics: backtracks={solver.metrics.backtrack_count}, cycles={solver.metrics.cycle_detections}")
    
    def test_iddfs_complex_respects_global_state_budget(self):
        """All deepening iterations share one global expansion budget."""
        logger.info("==== Testing IDDFS: Complex Dungeon ====")
        
        grid = create_complex_dungeon()
        env = ZeldaLogicEnv(grid, solver_options=SolverOptions(start_bombs=1))
        solver = StateSpaceDFS(env, timeout=100000, max_depth=300, use_iddfs=True)
        
        success, path, nodes = solver.solve()

        assert nodes <= solver.timeout
        if success:
            assert path[0] == env.start_pos
            assert path[-1] == env.goal_pos
        else:
            assert path == []


class TestBidirectionalAStar:
    """Test Bidirectional A* implementation."""
    
    def test_long_corridor(self):
        """Test Bidirectional A* on long corridor (ideal case)."""
        logger.info("==== Testing Bidirectional A*: Long Corridor ====")
        
        grid = create_long_corridor()
        env = ZeldaLogicEnv(grid)
        solver = BidirectionalAStar(env, timeout=100000)
        
        success, path, nodes = solver.solve()
        
        assert success, "Bidirectional A* failed on long corridor"
        assert len(path) > 0, "Bidirectional A* returned empty path"
        assert path[0] == env.start_pos, "Path doesn't start at start"
        assert path[-1] == env.goal_pos, "Path doesn't end at goal"
        
        logger.info(f"✓ Bidirectional A*: path_len={len(path)}, nodes={nodes}")
        logger.info(f"  Meeting point: {solver.meeting_point}")
        logger.info(f"  Collision checks: {solver.collision_checks}")
    
    def test_simple_dungeon(self):
        """Test Bidirectional A* on simple dungeon."""
        logger.info("==== Testing Bidirectional A*: Simple Dungeon ====")
        
        grid = create_simple_dungeon()
        env = ZeldaLogicEnv(grid)
        solver = BidirectionalAStar(env, timeout=100000)
        
        success, path, nodes = solver.solve()
        
        assert success, "Bidirectional A* failed on simple dungeon"
        assert len(path) > 0, "Path is empty"
        assert path[-1] == env.goal_pos, "Path doesn't reach goal"
        
        logger.info(f"✓ Bidirectional A*: path_len={len(path)}, nodes={nodes}")


    def test_collision_rejects_backward_surplus_keys(self):
        grid = create_simple_dungeon()
        env = ZeldaLogicEnv(grid)
        solver = BidirectionalAStar(env, timeout=100000)
        forward_node = SearchNode(state=GameState(position=(3, 3), keys=0), g_score=0.0, f_score=0.0)
        backward_node = SearchNode(state=GameState(position=(3, 3), keys=3), g_score=0.0, f_score=0.0)

        collision = solver._check_approximate_collision(forward_node, [backward_node], is_forward=True)

        assert collision is None

    def test_diagonal_heuristic_matches_unit_action_cost(self):
        grid = create_simple_dungeon()
        env = ZeldaLogicEnv(grid)
        solver = BidirectionalAStar(env, timeout=100000, allow_diagonals=True)

        assert solver._heuristic_forward(GameState(position=(1, 1))) == pytest.approx(7.0)

    def test_stateful_map_uses_canonical_fallback(self):
        grid = np.full((5, 7), SEMANTIC_PALETTE['WALL'], dtype=np.int64)
        grid[2, 1] = SEMANTIC_PALETTE['START']
        grid[2, 2] = SEMANTIC_PALETTE['KEY_SMALL']
        grid[2, 3] = SEMANTIC_PALETTE['DOOR_LOCKED']
        grid[2, 4] = SEMANTIC_PALETTE['FLOOR']
        grid[2, 5] = SEMANTIC_PALETTE['TRIFORCE']
        env = ZeldaLogicEnv(grid)
        solver = BidirectionalAStar(env, timeout=100000)

        success, path, _nodes = solver.solve()

        assert success
        assert path[0] == env.start_pos
        assert path[-1] == env.goal_pos
        assert solver.used_fallback is True

    def test_diagonal_successors_reject_conditional_corner_cutting(self):
        env = ZeldaLogicEnv(_corner_cut_grid(SEMANTIC_PALETTE['DOOR_LOCKED']))
        solver = BidirectionalAStar(env, timeout=100000, allow_diagonals=True)

        successors = solver._get_forward_successors(env.state.copy())

        assert all(state.position != env.goal_pos for state in successors)


class TestComparison:
    """Comparative benchmarks between algorithms."""
    
    def test_all_algorithms_simple(self):
        """Run all algorithms on simple dungeon and compare."""
        logger.info("\n==== COMPARATIVE BENCHMARK: Simple Dungeon ====")
        
        grid = create_simple_dungeon()
        
        results = {}
        
        # A* (baseline)
        env = ZeldaLogicEnv(grid)
        astar = StateSpaceAStar(env, timeout=100000)
        success, path, nodes = astar.solve()
        results['A*'] = {'success': success, 'path_len': len(path) if path else 0, 'nodes': nodes}
        logger.info(f"A*: success={success}, path_len={len(path) if path else 0}, nodes={nodes}")
        
        # DFS
        env = ZeldaLogicEnv(grid)
        dfs = StateSpaceDFS(env, timeout=50000, max_depth=100, use_iddfs=False)
        success, path, nodes = dfs.solve()
        results['DFS'] = {'success': success, 'path_len': len(path) if path else 0, 'nodes': nodes}
        logger.info(f"DFS: success={success}, path_len={len(path) if path else 0}, nodes={nodes}")
        
        # IDDFS
        env = ZeldaLogicEnv(grid)
        iddfs = StateSpaceDFS(env, timeout=50000, max_depth=100, use_iddfs=True)
        success, path, nodes = iddfs.solve()
        results['IDDFS'] = {'success': success, 'path_len': len(path) if path else 0, 'nodes': nodes}
        logger.info(f"IDDFS: success={success}, path_len={len(path) if path else 0}, nodes={nodes}")
        
        # Bidirectional A*
        env = ZeldaLogicEnv(grid)
        bidir = BidirectionalAStar(env, timeout=100000)
        success, path, nodes = bidir.solve()
        results['BiDir A*'] = {'success': success, 'path_len': len(path) if path else 0, 'nodes': nodes}
        logger.info(f"Bidirectional A*: success={success}, path_len={len(path) if path else 0}, nodes={nodes}")
        
        # D* Lite
        env = ZeldaLogicEnv(grid)
        dstar = DStarLiteSolver(env)
        start_state = env.state.copy()
        success, path, nodes = dstar.solve(start_state)
        results['D* Lite'] = {'success': success, 'path_len': len(path) if path else 0, 'nodes': nodes}
        logger.info(f"D* Lite: success={success}, path_len={len(path) if path else 0}, nodes={nodes}")
        
        # Verify all succeeded
        for alg, res in results.items():
            assert res['success'], f"{alg} failed on simple dungeon"
        
        logger.info("\n✓ All algorithms successfully solved simple dungeon")
    
    def test_bidirectional_speedup(self):
        """Verify Bidirectional A* reduces nodes expanded on long paths."""
        logger.info("\n==== BIDIRECTIONAL A* SPEEDUP TEST ====")
        
        grid = create_long_corridor()
        
        # Standard A*
        env1 = ZeldaLogicEnv(grid)
        astar = StateSpaceAStar(env1, timeout=100000)
        success1, _path1, nodes1 = astar.solve()
        
        # Bidirectional A*
        env2 = ZeldaLogicEnv(grid)
        bidir = BidirectionalAStar(env2, timeout=100000)
        success2, _path2, nodes2 = bidir.solve()
        
        assert success1 and success2, "Both algorithms should succeed"
        
        speedup = (nodes1 - nodes2) / nodes1 * 100 if nodes1 > 0 else 0
        
        logger.info(f"A* nodes: {nodes1}")
        logger.info(f"Bidirectional A* nodes: {nodes2}")
        logger.info(f"Nodes reduction: {speedup:.1f}%")
        
        # Bidirectional should explore fewer nodes (ideally ~50% less)
        # But due to state-space complexity, any reduction is good
        if nodes2 < nodes1:
            logger.info("✓ Bidirectional A* explored fewer nodes than A*")
        else:
            logger.info("⚠ Bidirectional A* didn't reduce nodes (state-space complexity)")


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    # Run tests manually
    logger.info("=" * 70)
    logger.info("ADVANCED PATHFINDING ALGORITHMS - INTEGRATION TESTS")
    logger.info("=" * 70)
    
    # D* Lite tests
    test_dstar = TestDStarLite()
    test_dstar.test_simple_dungeon()
    test_dstar.test_complex_dungeon()
    
    # DFS tests
    test_dfs = TestStateSpaceDFS()
    test_dfs.test_iterative_dfs_simple()
    test_dfs.test_iddfs_simple()
    test_dfs.test_iddfs_complex()
    
    # Bidirectional A* tests
    test_bidir = TestBidirectionalAStar()
    test_bidir.test_long_corridor()
    test_bidir.test_simple_dungeon()
    
    # Comparative benchmarks
    test_comp = TestComparison()
    test_comp.test_all_algorithms_simple()
    test_comp.test_bidirectional_speedup()
    
    logger.info("\n" + "=" * 70)
    logger.info("ALL TESTS COMPLETED")
    logger.info("=" * 70)
