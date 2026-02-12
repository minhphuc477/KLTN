"""
Test Algorithm Dispatch - Verify Different Algorithms Run Correctly
===================================================================

This script tests that each algorithm (A*, BFS, Dijkstra, Greedy, D* Lite, CBS)
is actually being invoked and produces different results.

Run: python test_algorithm_dispatch.py
"""

import sys
import os
import logging
import numpy as np

# Configure logging to show algorithm dispatch
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from simulation.validator import ZeldaLogicEnv, SEMANTIC_PALETTE

# Import the solver function
from gui_runner import _solve_in_subprocess

def create_simple_test_grid():
    """Create a simple 10x20 test grid with a clear path."""
    W = SEMANTIC_PALETTE['WALL']
    F = SEMANTIC_PALETTE['FLOOR']
    S = SEMANTIC_PALETTE['START']
    T = SEMANTIC_PALETTE['TRIFORCE']
    
    grid = np.full((10, 20), W, dtype=np.int64)
    
    # Create a simple corridor with some branches
    # This should give different algorithms different exploration patterns
    for r in range(1, 9):
        for c in range(1, 19):
            grid[r, c] = F
    
    # Add start and goal
    grid[5, 2] = S
    grid[5, 17] = T
    
    # Add some walls to create branching paths
    for c in range(7, 14):
        grid[3, c] = W
        grid[7, c] = W
    
    return grid

def test_algorithm(algorithm_idx, algorithm_name):
    """Test a single algorithm."""
    print(f"\n{'='*70}")
    print(f"Testing: {algorithm_name} (idx={algorithm_idx})")
    print('='*70)
    
    grid = create_simple_test_grid()
    start = (5, 2)
    goal = (5, 17)
    
    result = _solve_in_subprocess(
        grid=grid,
        start_pos=start,
        goal_pos=goal,
        algorithm_idx=algorithm_idx,
        feature_flags={},
        priority_options={},
        graph=None,
        room_to_node=None,
        room_positions=None,
        node_to_room=None
    )
    
    if result['success']:
        path_len = len(result['path']) if result['path'] else 0
        nodes = result.get('solver_result', {}).get('nodes', 'N/A')
        algo = result.get('solver_result', {}).get('algorithm', 'N/A')
        print(f"✓ SUCCESS: Path length={path_len}, Nodes explored={nodes}, Algorithm={algo}")
        
        # Show first few and last few moves of the path
        if result['path'] and len(result['path']) > 4:
            print(f"  Path start: {result['path'][:3]}")
            print(f"  Path end: {result['path'][-3:]}")
        
        return True, path_len, nodes
    else:
        msg = result.get('message', 'Unknown error')
        print(f"✗ FAILED: {msg}")
        return False, 0, 0

def main():
    """Run all algorithm tests."""
    print("\n" + "="*70)
    print("ALGORITHM DISPATCH TEST")
    print("="*70)
    print("This test verifies that different algorithms are being invoked")
    print("and producing different exploration patterns.")
    print("="*70)
    
    algorithms = [
        (0, "A*"),
        (1, "BFS"),
        (2, "Dijkstra"),
        (3, "Greedy"),
        (4, "D* Lite"),
        (6, "CBS (Explorer)"),
    ]
    
    results = []
    
    for idx, name in algorithms:
        success, path_len, nodes = test_algorithm(idx, name)
        results.append((name, success, path_len, nodes))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Algorithm':<20} {'Status':<10} {'Path Length':<15} {'Nodes Explored'}")
    print("-"*70)
    
    for name, success, path_len, nodes in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{name:<20} {status:<10} {path_len:<15} {nodes}")
    
    # Check for differences
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)
    
    path_lengths = [pl for _, success, pl, _ in results if success]
    nodes_explored = [n for _, success, _, n in results if success]
    
    if len(set(path_lengths)) > 1:
        print("✓ GOOD: Different algorithms produced different path lengths")
        print(f"  Path lengths: {sorted(set(path_lengths))}")
    else:
        print("⚠ WARNING: All algorithms produced the same path length")
        print("  This might be okay for simple grids, but suggests limited variation")
    
    if len(set(nodes_explored)) > 1:
        print("✓ GOOD: Different algorithms explored different numbers of nodes")
        print(f"  Node counts: {sorted(set(nodes_explored))}")
    else:
        print("⚠ WARNING: All algorithms explored the same number of nodes")
    
    # Check if all succeeded
    all_success = all(success for _, success, _, _ in results)
    if all_success:
        print("\n✓ ALL TESTS PASSED: All algorithms found paths")
    else:
        failed = [name for name, success, _, _ in results if not success]
        print(f"\n✗ SOME TESTS FAILED: {', '.join(failed)}")
    
    print("="*70)
    print("\nTo see detailed algorithm dispatch logs, check the terminal output above.")
    print("Look for lines like: '🔍 SOLVER DISPATCH: algorithm_idx=X → AlgorithmName'")
    print("="*70)

if __name__ == '__main__':
    main()
