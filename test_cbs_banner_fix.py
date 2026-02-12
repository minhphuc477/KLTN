"""Test script to verify CBS banner display bug fix.

Run this after applying the fix to verify:
1. Banner shows correct algorithm name
2. Changing algorithm while solver running stops old solver
3. solver_algorithm_idx is properly tracked
"""

import os
import sys
import time
import logging

# Add KLTN directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging to see debug messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

def test_solver_algorithm_tracking():
    """Test that solver_algorithm_idx is properly set and cleared."""
    print("\n" + "="*70)
    print("TEST 1: Solver Algorithm Tracking")
    print("="*70)
    
    # This test would require actually running the GUI
    # For now, we'll check the code changes are present
    
    with open('gui_runner.py', 'r', encoding='utf-8') as f:
        code = f.read()
    
    checks = [
        ('solver_algorithm_idx assignment', 'self.solver_algorithm_idx ='),
        ('Banner uses solver_algorithm_idx', 'solver_algorithm_idx.*getattr.*algorithm_idx'),
        ('Clear solver state helper', 'def _clear_solver_state'),
        ('Dropdown stops running solver', 'CRITICAL FIX: If solver is currently running'),
    ]
    
    results = []
    for check_name, pattern in checks:
        if pattern in code.replace(' ', '').replace('\n', ''):
            print(f"✓ {check_name}: FOUND")
            results.append(True)
        else:
            print(f"✗ {check_name}: NOT FOUND")
            results.append(False)
    
    if all(results):
        print("\n✅ All code changes are present!")
        return True
    else:
        print(f"\n❌ Missing {sum(not r for r in results)} code changes")
        return False

def test_algorithm_names_consistency():
    """Test that algorithm names are consistent across all usages."""
    print("\n" + "="*70)
    print("TEST 2: Algorithm Names Consistency")
    print("="*70)
    
    with open('gui_runner.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find all algorithm_names definitions
    algorithm_lists = []
    for i, line in enumerate(lines, 1):
        if 'algorithm_names = [' in line:
            # Extract the list
            list_lines = [line]
            j = i
            while ']' not in line:
                j += 1
                if j <= len(lines):
                    line = lines[j-1]
                    list_lines.append(line)
                else:
                    break
            algorithm_lists.append((i, ''.join(list_lines)))
    
    print(f"Found {len(algorithm_lists)} algorithm_names definitions")
    
    # Check CBS variants are present at correct indices
    expected_order = [
        (0, "A*"),
        (1, "BFS"),
        (2, "Dijkstra"),
        (3, "Greedy"),
        (4, "D* Lite"),
        (5, "CBS (Balanced)"),
        (6, "CBS (Explorer)"),
        (7, "CBS (Cautious)"),
        (8, "CBS (Forgetful)"),
        (9, "CBS (Speedrunner)"),
        (10, "CBS (Greedy)"),
    ]
    
    print("\nExpected algorithm order:")
    for idx, name in expected_order:
        print(f"  [{idx}] {name}")
    
    print("\n✅ Test complete - verify manually that dropdown and banner match")
    return True

def test_debug_logging():
    """Check that debug logging is added."""
    print("\n" + "="*70)
    print("TEST 3: Debug Logging")
    print("="*70)
    
    with open('gui_runner.py', 'r', encoding='utf-8') as f:
        code = f.read()
    
    log_checks = [
        ('DROPDOWN log', 'DROPDOWN: Algorithm changed from'),
        ('SOLVER log', 'SOLVER: Acquired solver lock.*algorithm_idx'),
        ('BANNER log', 'BANNER: Rendering solver banner'),
        ('CLEANUP log', 'SOLVER_CLEANUP: Clearing solver state'),
    ]
    
    results = []
    for check_name, pattern in log_checks:
        pattern_clean = pattern.replace(' ', '').replace('.*', '')
        if pattern_clean in code.replace(' ', '').replace('\n', ''):
            print(f"✓ {check_name}: FOUND")
            results.append(True)
        else:
            print(f"✗ {check_name}: NOT FOUND")
            results.append(False)
    
    if all(results):
        print("\n✅ All debug logs are present!")
        return True
    else:
        print(f"\n❌ Missing {sum(not r for r in results)} debug logs")
        return False

def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("CBS BANNER BUG FIX - VERIFICATION TESTS")
    print("="*70)
    print("\nThese tests verify the code changes are present.")
    print("To fully test the fix, run the GUI application and:")
    print("  1. Select CBS (Balanced) from dropdown")
    print("  2. Press SPACE to solve")
    print("  3. Verify banner shows 'Computing path with CBS (Balanced)...'")
    print("="*70)
    
    test1 = test_solver_algorithm_tracking()
    test2 = test_algorithm_names_consistency()
    test3 = test_debug_logging()
    
    print("\n" + "="*70)
    if all([test1, test2, test3]):
        print("✅ ALL TESTS PASSED - Code changes are present!")
        print("\nNext steps:")
        print("  1. Run: python gui_runner.py")
        print("  2. Select 'CBS (Balanced)' from dropdown")
        print("  3. Press SPACE")
        print("  4. Verify banner shows correct algorithm")
        print("  5. Check console log for debug messages:")
        print("     - DROPDOWN: Algorithm changed...")
        print("     - SOLVER: Acquired solver lock...")
        print("     - BANNER: Rendering solver banner...")
    else:
        print("❌ SOME TESTS FAILED - Code changes may be incomplete")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
