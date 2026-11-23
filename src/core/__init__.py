"""
KLTN Core Module - Canonical G-LMDM Neuro-Symbolic Stack & H-MOLQD Comparators
================================================================================

Core neural network components and definitions for Zelda dungeon generation.

Tier 1 - Canonical G-LMDM Active Stack (Primary Model):
- definitions.py:            Semantic constants, TileID palette, room dimensions
- adapter.py:                VGLC data loading and graph/grid transformations
- stitcher.py:               Dungeon-level room stitching algorithms
- condition_encoder.py:      Block III - Dual-stream GraphGPS + RRWP contextual encoding
- discrete_masked_model.py:  Block IV  - Absorbing-state masked discrete room Transformer
- logic_net.py:              Block V   - Direct 16x11 Soft Bellman-Ford differentiable pathfinding (alpha=0.1)
- symbolic_refiner.py:       Block VI  - Path-guided WFC / minimum-edit symbolic repair
- neural_guided_repair.py:   Block VI  - Neural-guided repair utilities (paired with src.generation.puzzle_generator)

Tier 2 - Continuous Latent Comparators (H-MOLQD Baselines for H1/H2):
- vqvae.py:                  Block II (Comparator) - Semantic VQ-VAE tokenizer
- latent_diffusion.py:       Block IV (Comparator) - UNet / DiT continuous latent diffusion

Tier 3 - Deprecated / Ablation-Only Modules (Not exported in __all__; inactive in canonical G-LMDM):
- gaussian_vae.py:           Continuous Gaussian VAE ablation baseline
- perturb_and_map.py:        Perturb-and-MAP LogicNet ablation surrogate
- puzzle_stage_semantics.py: Disabled auxiliary puzzle-stage CNN head (loss_weight=0.0)
- attention_kernels.py:      Linear Hedgehog attention kernel (unneeded on L=176 token grids)

Usage:
    # Definitions
    from src.core import SEMANTIC_PALETTE, ID_TO_NAME

    # Canonical G-LMDM components
    from src.core.condition_encoder import DualStreamConditionEncoder
    from src.core.discrete_masked_model import DiscreteMaskedRoomModel
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

# H-MOLQD Continuous Comparators & Canonical G-LMDM Neural Blocks
from src.core.vqvae import SemanticVQVAE, create_vqvae
from src.core.condition_encoder import DualStreamConditionEncoder, create_condition_encoder
from src.core.latent_diffusion import LatentDiffusionModel, create_latent_diffusion
from src.core.discrete_masked_model import DiscreteMaskedRoomModel, create_discrete_masked_model
from src.core.logic_net import LogicNet
from src.core.symbolic_refiner import (
    SymbolicRefiner, create_symbolic_refiner, LearnedTileStatistics
)
from src.core.neural_guided_repair import NeuralGuidedRepair, NeuralRepairGuidance

CANONICAL_GLMDM_MODULES = (
    'definitions',
    'adapter',
    'stitcher',
    'condition_encoder',
    'discrete_masked_model',
    'logic_net',
    'symbolic_refiner',
    'neural_guided_repair',
)

CONTINUOUS_COMPARATOR_MODULES = (
    'vqvae',
    'latent_diffusion',
)

DEPRECATED_ABLATION_ONLY_MODULES = (
    'gaussian_vae',
    'perturb_and_map',
    'puzzle_stage_semantics',
    'attention_kernels',
)

__all__ = [
    # Module Audit Registry
    'CANONICAL_GLMDM_MODULES',
    'CONTINUOUS_COMPARATOR_MODULES',
    'DEPRECATED_ABLATION_ONLY_MODULES',
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
    # Canonical G-LMDM & Comparator Blocks
    'SemanticVQVAE',
    'create_vqvae',
    'DualStreamConditionEncoder',
    'create_condition_encoder',
    'LatentDiffusionModel',
    'create_latent_diffusion',
    'DiscreteMaskedRoomModel',
    'create_discrete_masked_model',
    'LogicNet',
    'SymbolicRefiner',
    'create_symbolic_refiner',
    'LearnedTileStatistics',
    'NeuralGuidedRepair',
    'NeuralRepairGuidance',
]
