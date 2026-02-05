"""
KLTN Core Module - H-MOLQD Neural Components
============================================

Core neural network components and definitions for Zelda dungeon generation.

Original Components:
- definitions: Semantic constants and type definitions
- adapter: Data loading and transformation
- stitcher: Room stitching algorithms

H-MOLQD Architecture Blocks (NEW):
- Block II:   vqvae.py - Semantic VQ-VAE for discrete representation learning
- Block III:  condition_encoder.py - Dual-stream contextual encoding
- Block IV:   latent_diffusion.py - Latent diffusion with gradient guidance
- Block V:    logic_net.py - Differentiable pathfinding for solvability
- Block VII:  symbolic_refiner.py - Path-guided WFC repair

Usage:
    # Original components
    from src.core import SEMANTIC_PALETTE, ID_TO_NAME
    from src.core.adapter import ZeldaAdapter
    from src.core.stitcher import DungeonStitcher
    
    # H-MOLQD neural components
    from src.core.vqvae import SemanticVQVAE
    from src.core.condition_encoder import DualStreamConditionEncoder
    from src.core.latent_diffusion import LatentDiffusionModel
    from src.core.logic_net import LogicNet
    from src.core.symbolic_refiner import SymbolicRefiner
"""

from src.core.definitions import (
    TileID,
    ValidationMode,
    SEMANTIC_PALETTE,
    ID_TO_NAME,
    CHAR_TO_SEMANTIC,
    WALKABLE_CHARS,
    WALL_CHARS,
    ROOM_HEIGHT,
    ROOM_WIDTH,
    EDGE_TYPE_MAP,
    NODE_CONTENT_MAP,
)

__all__ = [
    # Definitions
    'TileID',
    'ValidationMode', 
    'SEMANTIC_PALETTE',
    'ID_TO_NAME',
    'CHAR_TO_SEMANTIC',
    'WALKABLE_CHARS',
    'WALL_CHARS',
    'ROOM_HEIGHT',
    'ROOM_WIDTH',
    'EDGE_TYPE_MAP',
    'NODE_CONTENT_MAP',
]
