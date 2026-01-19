"""
KLTN Core Module
================
Core data processing components for Zelda dungeon analysis.

This module contains:
- definitions: Semantic constants and type definitions
- adapter: Data loading and transformation
- stitcher: Room stitching algorithms

Usage:
    from src.core import SEMANTIC_PALETTE, ID_TO_NAME
    from src.core.adapter import ZeldaAdapter
    from src.core.stitcher import DungeonStitcher
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
