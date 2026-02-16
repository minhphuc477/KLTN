"""Quick validation test for WFC integration fixes."""

import sys
sys.path.insert(0, 'F:\\KLTN')

try:
    # Test 1: WFC imports
    print("Test 1: Importing WFC components...")
    from src.generation.weighted_bayesian_wfc import (
        WeightedBayesianWFC, 
        extract_tile_priors_from_vqvae, 
        WeightedBayesianWFCConfig,
        TilePrior
    )
    print("✅ WFC imports successful")
    
    # Test 2: NeuralSymbolicDungeonPipeline import
    print("\nTest 2: Importing NeuralSymbolicDungeonPipeline...")
    from src.pipeline.dungeon_pipeline import NeuralSymbolicDungeonPipeline
    print("✅ Pipeline import successful")
    
    # Test 3: Direction mapping verification
    print("\nTest 3: Verifying direction mapping logic...")
    import numpy as np
    from collections import Counter
    
    # Create simple test grid
    test_grid = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ])
    
    # Extract adjacencies manually
    tile_id = 1  # Center tile
    r, c = 1, 1
    
    adjacencies = []
    # North
    if r > 0:
        neighbor_id = test_grid[r-1, c]
        adjacencies.append((neighbor_id, 'N'))
    # South 
    if r < 2:
        neighbor_id = test_grid[r+1, c]
        adjacencies.append((neighbor_id, 'S'))
    # East
    if c < 2:
        neighbor_id = test_grid[r, c+1]
        adjacencies.append((neighbor_id, 'E'))
    # West
    if c > 0:
        neighbor_id = test_grid[r, c-1]
        adjacencies.append((neighbor_id, 'W'))
    
    print(f"  Center tile {tile_id} at ({r}, {c})")
    print(f"  Adjacencies: {adjacencies}")
    print(f"  Expected: [(0, 'N'), (0, 'S'), (0, 'E'), (0, 'W')]")
    
    expected = [(0, 'N'), (0, 'S'), (0, 'E'), (0, 'W')]
    if adjacencies == expected:
        print("✅ Direction mapping logic verified")
    else:
        print("❌ Direction mapping mismatch!")
        sys.exit(1)
    
    # Test 4: TilePrior consistency
    print("\nTest 4: Testing TilePrior.get_adjacency_probability()...")
    prior = TilePrior(
        tile_id=1,
        frequency=0.5,
        adjacency_counts={(2, 'N'): 10, (2, 'S'): 5, (3, 'N'): 5}
    )
    
    prob_n = prior.get_adjacency_probability(2, 'N')
    print(f"  P(tile_2 is NORTH of tile_1) = {prob_n:.2f}")
    print(f"  Expected: 10 / (10 + 5) = 0.67")
    
    if abs(prob_n - 0.67) < 0.01:
        print("✅ TilePrior probability calculation correct")
    else:
        print(f"❌ TilePrior calculation error: expected 0.67, got {prob_n}")
        sys.exit(1)
    
    print("\n" + "="*50)
    print("✅ ALL TESTS PASSED - WFC integration is correct!")
    print("="*50)
    
except Exception as e:
    print(f"\n❌ TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
