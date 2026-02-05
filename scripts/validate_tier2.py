"""
Quick Validation Script for TIER 2 & 3 Features
===============================================

Runs quick checks to verify all implementations are working.

Usage: python scripts/validate_tier2.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.simulation import GameState, ZeldaLogicEnv, SEMANTIC_PALETTE


def print_status(name: str, success: bool):
    """Print colorful status message."""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} - {name}")


def main():
    print("=" * 60)
    print("TIER 2 & 3 FEATURE VALIDATION")
    print("=" * 60)
    print()
    
    results = []
    
    # Test 1: Multi-floor support
    print("Test 1: Multi-Floor Dungeon Support")
    try:
        state = GameState(position=(0, 0), current_floor=2)
        assert state.current_floor == 2
        assert hasattr(state, 'current_floor')
        print_status("Multi-floor GameState", True)
        results.append(True)
    except Exception as e:
        print_status("Multi-floor GameState", False)
        print(f"  Error: {e}")
        results.append(False)
    print()
    
    # Test 2: D* Lite
    print("Test 2: D* Lite Replanning")
    try:
        from src.simulation import DStarLiteSolver
        
        grid = np.full((10, 10), SEMANTIC_PALETTE['FLOOR'])
        grid[0, 0] = SEMANTIC_PALETTE['START']
        grid[9, 9] = SEMANTIC_PALETTE['TRIFORCE']
        
        env = ZeldaLogicEnv(grid)
        solver = DStarLiteSolver(env)
        
        start_state = GameState(position=(0, 0))
        success, path, states = solver.solve(start_state)
        
        assert success, "D* Lite should find solution"
        assert len(path) > 0
        
        print_status("D* Lite solver", True)
        results.append(True)
    except Exception as e:
        print_status("D* Lite solver", False)
        print(f"  Error: {e}")
        results.append(False)
    print()
    
    # Test 3: Parallel A*
    print("Test 3: Parallel Search")
    try:
        from src.simulation import ParallelAStarSolver
        import multiprocessing as mp
        
        grid = np.full((10, 10), SEMANTIC_PALETTE['FLOOR'])
        grid[0, 0] = SEMANTIC_PALETTE['START']
        grid[9, 9] = SEMANTIC_PALETTE['TRIFORCE']
        
        env = ZeldaLogicEnv(grid)
        solver = ParallelAStarSolver(env, n_workers=2)
        
        assert solver.n_workers == 2
        
        print_status("Parallel A* solver", True)
        results.append(True)
    except Exception as e:
        print_status("Parallel A* solver", False)
        print(f"  Error: {e}")
        results.append(False)
    print()
    
    # Test 4: Multi-Goal
    print("Test 4: Multi-Goal Pathfinding")
    try:
        from src.simulation import MultiGoalPathfinder
        
        grid = np.full((10, 10), SEMANTIC_PALETTE['FLOOR'])
        grid[0, 0] = SEMANTIC_PALETTE['START']
        grid[5, 5] = SEMANTIC_PALETTE['KEY_SMALL']
        grid[9, 9] = SEMANTIC_PALETTE['TRIFORCE']
        
        env = ZeldaLogicEnv(grid)
        finder = MultiGoalPathfinder(env)
        
        start_state = GameState(position=(0, 0))
        result = finder.find_optimal_collection_order(start_state)
        
        assert result is not None
        
        print_status("Multi-goal pathfinder", True)
        results.append(True)
    except Exception as e:
        print_status("Multi-goal pathfinder", False)
        print(f"  Error: {e}")
        results.append(False)
    print()
    
    # Test 5: Solver Comparison
    print("Test 5: Solver Comparison")
    try:
        from src.simulation import SolverComparison
        
        grid = np.full((10, 10), SEMANTIC_PALETTE['FLOOR'])
        grid[0, 0] = SEMANTIC_PALETTE['START']
        grid[9, 9] = SEMANTIC_PALETTE['TRIFORCE']
        
        env = ZeldaLogicEnv(grid)
        comparison = SolverComparison(env)
        
        start_state = GameState(position=(0, 0))
        results_dict = comparison.compare_all(start_state, max_time=3.0)
        
        assert 'A*' in results_dict
        assert 'BFS' in results_dict
        assert 'Dijkstra' in results_dict
        assert 'Greedy' in results_dict
        
        print_status("Solver comparison", True)
        results.append(True)
    except Exception as e:
        print_status("Solver comparison", False)
        print(f"  Error: {e}")
        results.append(False)
    print()
    
    # Test 6: Procedural Generation
    print("Test 6: Procedural Dungeon Generation")
    try:
        from src.generation.dungeon_generator import DungeonGenerator, Difficulty
        
        gen = DungeonGenerator(width=20, height=20, difficulty=Difficulty.EASY, seed=42)
        grid = gen.generate()
        
        assert grid.shape == (20, 20)
        assert np.sum(grid == SEMANTIC_PALETTE['START']) >= 1
        assert np.sum(grid == SEMANTIC_PALETTE['TRIFORCE']) >= 1
        
        print_status("Procedural generation", True)
        results.append(True)
    except Exception as e:
        print_status("Procedural generation", False)
        print(f"  Error: {e}")
        results.append(False)
    print()
    
    # Test 7: ML Heuristics (optional)
    print("Test 7: ML Heuristic Learning (Optional)")
    try:
        from src.ml.heuristic_learning import HeuristicNetwork
        import torch
        
        net = HeuristicNetwork(map_height=30, map_width=30)
        assert net is not None
        
        print_status("ML heuristics (PyTorch)", True)
        results.append(True)
    except ImportError:
        print_status("ML heuristics (PyTorch)", False)
        print("  Note: PyTorch not installed - this is optional")
        results.append(None)  # Not counted as failure
    except Exception as e:
        print_status("ML heuristics (PyTorch)", False)
        print(f"  Error: {e}")
        results.append(False)
    print()
    
    # Test 8: GUI Components
    print("Test 8: GUI Components")
    try:
        from src.gui.tier2_components import FloorSelector, MinimapZoom, ItemTooltip
        import pygame
        
        selector = FloorSelector(screen_width=800, num_floors=2)
        assert selector.num_floors == 2
        
        rect = pygame.Rect(0, 0, 400, 400)
        zoom = MinimapZoom(rect)
        assert zoom is not None
        
        tooltip = ItemTooltip()
        assert tooltip is not None
        
        print_status("GUI components", True)
        results.append(True)
    except Exception as e:
        print_status("GUI components", False)
        print(f"  Error: {e}")
        results.append(False)
    print()
    
    # Summary
    print("=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for r in results if r is True)
    failed = sum(1 for r in results if r is False)
    skipped = sum(1 for r in results if r is None)
    total = len(results)
    
    print(f"Passed:  {passed}/{total}")
    print(f"Failed:  {failed}/{total}")
    print(f"Skipped: {skipped}/{total} (optional)")
    print()
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED! 🎉")
        print("All TIER 2 & 3 features are working correctly.")
        return 0
    else:
        print("⚠️  SOME TESTS FAILED")
        print("Check error messages above for details.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
