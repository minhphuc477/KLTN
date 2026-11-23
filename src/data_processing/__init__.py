"""
H-MOLQD Data Processing Module
==============================

Block I: Intelligent Data Adapter
- VGLC text file parsing
- Graphviz DOT topology parsing
- Auto-phase alignment
- Graph fingerprinting
"""

from .data_adapter import (
    IntelligentDataAdapter,
    VGLCParser,
    GraphvizParser,
    PhaseAligner,
    GraphFingerprinter,
    DungeonTensor,
)
from src.zelda_data.zelda_loader import (
    ZeldaRoomDataset,
    apply_room_symmetry_transform,
    compute_room_tile_frequencies,
)

__all__ = [
    'IntelligentDataAdapter',
    'VGLCParser',
    'GraphvizParser', 
    'PhaseAligner',
    'GraphFingerprinter',
    'DungeonTensor',
    'ZeldaRoomDataset',
    'apply_room_symmetry_transform',
    'compute_room_tile_frequencies',
]
