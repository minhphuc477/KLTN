"""
Data Loading Module for KLTN PCG Training
=========================================

Provides dataset classes and data loaders for Zelda dungeon training.

Classes:
    ZeldaDungeonDataset: PyTorch Dataset for loading dungeon grids
    
Constants:
    TILE_MAPPING: ASCII character to integer ID mapping
"""

from .zelda_loader import ZeldaDungeonDataset, TILE_MAPPING, create_dataloader

__all__ = ['ZeldaDungeonDataset', 'TILE_MAPPING', 'create_dataloader']
