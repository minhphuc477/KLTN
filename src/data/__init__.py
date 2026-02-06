"""
Data Loading Module for KLTN PCG Training
=========================================

Provides dataset classes and data loaders for Zelda dungeon training.

Classes:
    ZeldaDungeonDataset: PyTorch Dataset for loading dungeon grids
    ZeldaDungeonAdapter: Adapter for loading VGLC dungeon data
    DungeonSolver: Solver for dungeon solvability checking
    
Constants:
    TILE_MAPPING: ASCII character to integer ID mapping
    SEMANTIC_PALETTE: Semantic tile ID mapping
"""

from .zelda_loader import ZeldaDungeonDataset, TILE_MAPPING, create_dataloader
from .zelda_core import (
    ZeldaDungeonAdapter,
    DungeonSolver,
    ValidationMode,
    StitchedDungeon,
    RoomGraphMatcher,
    GridBasedRoomExtractor,
    Dungeon,
    Room,
)
from src.core.definitions import SEMANTIC_PALETTE, ID_TO_NAME, ROOM_HEIGHT, ROOM_WIDTH

__all__ = [
    # Dataset
    'ZeldaDungeonDataset',
    'TILE_MAPPING',
    'create_dataloader',
    # Zelda core
    'ZeldaDungeonAdapter',
    'DungeonSolver',
    'ValidationMode',
    'StitchedDungeon',
    'RoomGraphMatcher',
    'GridBasedRoomExtractor',
    'Dungeon',
    'Room',
    # Constants
    'SEMANTIC_PALETTE',
    'ID_TO_NAME',
    'ROOM_HEIGHT',
    'ROOM_WIDTH',
]
