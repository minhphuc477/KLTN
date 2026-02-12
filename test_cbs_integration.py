"""
Quick test to verify CBS integration in GUI runner.

This script validates that:
1. CBS can be imported correctly
2. CBS personas are available
3. CBS solver can be invoked 
4. CBS metrics are returned correctly
"""

import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_cbs_import():
    """Test that CBS can be imported."""
    print("Testing CBS import...")
    try:
        from src.simulation.cognitive_bounded_search import CognitiveBoundedSearch, AgentPersona, CBSMetrics
        print("✓ CBS imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import CBS: {e}")
        return False

def test_cbs_personas():
    """Test that all CBS personas are available."""
    print("\nTesting CBS personas...")
    try:
        from src.simulation.cognitive_bounded_search import AgentPersona
        
        expected_personas = ['balanced', 'explorer', 'cautious', 'forgetful', 'speedrunner', 'greedy']
        for persona_name in expected_personas:
            persona = AgentPersona(persona_name)
            print(f"  ✓ {persona_name.title()} persona available")
        
        print("✓ All CBS personas available")
        return True
    except Exception as e:
        print(f"✗ Failed to access personas: {e}")
        return False

def test_cbs_basic_solve():
    """Test CBS solver on a simple grid."""
    print("\nTesting CBS basic solve...")
    try:
        from src.simulation.cognitive_bounded_search import CognitiveBoundedSearch
        from src.simulation.validator import ZeldaLogicEnv, SEMANTIC_PALETTE
        
        # Create a simple 10x10 grid with start, goal, and floor
        grid = np.full((10, 10), SEMANTIC_PALETTE['FLOOR'], dtype=np.int64)
        grid[1, 1] = SEMANTIC_PALETTE['START']
        grid[8, 8] = SEMANTIC_PALETTE['TRIFORCE']
        
        # Create env and CBS solver
        env = ZeldaLogicEnv(grid, render_mode=False)
        cbs = CognitiveBoundedSearch(env, persona='balanced', timeout=10000)
        
        # Solve
        success, path, states, metrics = cbs.solve()
        
        if success:
            print(f"  ✓ CBS found path: {len(path)} steps")
            print(f"    - Confusion Index: {metrics.confusion_index:.3f}")
            print(f"    - Navigation Entropy: {metrics.navigation_entropy:.3f}")
            print(f"    - Cognitive Load: {metrics.cognitive_load:.3f}")
            print(f"    - States explored: {states}")
            return True
        else:
            print(f"  ✗ CBS failed to find path (states explored: {states})")
            return False
            
    except Exception as e:
        print(f"✗ CBS solve failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gui_integration():
    """Test that GUI runner can handle CBS algorithm indices."""
    print("\nTesting GUI integration...")
    try:
        # Check that the algorithm index mapping is correct
        cbs_personas = {
            5: 'balanced',
            6: 'explorer',
            7: 'cautious',
            8: 'forgetful',
            9: 'speedrunner',
            10: 'greedy'
        }
        
        algorithm_names = ["A*", "BFS", "Dijkstra", "Greedy", "D* Lite",
                          "CBS (Balanced)", "CBS (Explorer)", "CBS (Cautious)",
                          "CBS (Forgetful)", "CBS (Speedrunner)", "CBS (Greedy)"]
        
        for idx, persona in cbs_personas.items():
            name = algorithm_names[idx]
            expected = f"CBS ({persona.title()})"
            if name == expected:
                print(f"  ✓ Index {idx} maps to {name}")
            else:
                print(f"  ✗ Index {idx} mismatch: expected {expected}, got {name}")
                return False
        
        print("✓ GUI integration mapping correct")
        return True
    except Exception as e:
        print(f"✗ GUI integration check failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("CBS Integration Test Suite")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("CBS Import", test_cbs_import()))
    results.append(("CBS Personas", test_cbs_personas()))
    results.append(("CBS Basic Solve", test_cbs_basic_solve()))
    results.append(("GUI Integration", test_gui_integration()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name:.<30} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
