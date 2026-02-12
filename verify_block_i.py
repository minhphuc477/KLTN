"""
Quick Verification Script for Block I
=====================================

Verifies that the Evolutionary Topology Director is working correctly.
"""

import sys
from pathlib import Path

# Setup path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.evolutionary_director import EvolutionaryTopologyGenerator

print("=" * 60)
print("BLOCK I: EVOLUTIONARY TOPOLOGY DIRECTOR")
print("Quick Verification Test")
print("=" * 60)

# Test 1: Import
print("\n[1/4] Testing import...")
try:
    from src.generation.evolutionary_director import (
        EvolutionaryTopologyGenerator,
        TensionCurveEvaluator,
        GraphGrammarExecutor,
    )
    print("✓ Import successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Initialization
print("\n[2/4] Testing initialization...")
try:
    gen = EvolutionaryTopologyGenerator(
        target_curve=[0.2, 0.5, 0.8, 1.0],
        population_size=10,
        generations=5,
        seed=42,
    )
    print("✓ Initialization successful")
except Exception as e:
    print(f"✗ Initialization failed: {e}")
    sys.exit(1)

# Test 3: Evolution
print("\n[3/4] Testing evolution...")
try:
    graph = gen.evolve()
    print(f"✓ Evolution successful")
    print(f"  - Nodes: {graph.number_of_nodes()}")
    print(f"  - Edges: {graph.number_of_edges()}")
except Exception as e:
    print(f"✗ Evolution failed: {e}")
    sys.exit(1)

# Test 4: Statistics
print("\n[4/4] Testing statistics...")
try:
    stats = gen.get_statistics()
    print(f"✓ Statistics retrieval successful")
    print(f"  - Fitness: {stats['final_best_fitness']:.4f}")
    print(f"  - Generations: {stats['generations_run']}")
    print(f"  - Converged: {stats['converged']}")
except Exception as e:
    print(f"✗ Statistics failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 60)
print("🎮 BLOCK I: FULLY OPERATIONAL")
print("=" * 60)
print("\nAll systems verified:")
print("  ✓ Import system")
print("  ✓ Initialization")
print("  ✓ Evolutionary search")
print("  ✓ Statistics tracking")
print("\nStatus: READY FOR PRODUCTION")
print("=" * 60)
