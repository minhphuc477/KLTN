"""
KLTN Pipeline Module
====================

Master pipeline orchestration for neural-symbolic dungeon generation.

This module provides the complete end-to-end pipeline that integrates
all 7 H-MOLQD blocks for Legend of Zelda dungeon generation.

Usage:
    from src.pipeline import NeuralSymbolicDungeonPipeline, create_pipeline
    
    # Create pipeline with checkpoints
    pipeline = create_pipeline(checkpoint_dir="./checkpoints")
    
    # Generate dungeon
    result = pipeline.generate_dungeon(
        mission_graph=my_graph,
        seed=42
    )
    
    # Access results
    print(f"Generated {result.metrics['num_rooms']} rooms")
    print(f"Repair rate: {result.metrics['repair_rate']:.1%}")
    
    # Save dungeon
    np.save("dungeon.npy", result.dungeon_grid)
"""

from src.pipeline.dungeon_pipeline import (
    NeuralSymbolicDungeonPipeline,
    RoomGenerationResult,
    DungeonGenerationResult,
    create_pipeline,
)

__all__ = [
    'NeuralSymbolicDungeonPipeline',
    'RoomGenerationResult',
    'DungeonGenerationResult',
    'create_pipeline',
]
