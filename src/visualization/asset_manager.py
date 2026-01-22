"""
Asset Manager - Robust Asset Loading System
============================================

Handles loading of visual assets (sprites, tiles) with automatic fallback
to procedurally generated colored rectangles when assets are missing.

Key Features:
- Never crashes on missing assets
- Generates colored fallback rectangles based on semantic tile ID
- Caches loaded assets for performance
- Supports hot-reloading for development

Scientific Rationale:
--------------------
Visual debugging is critical for validating pathfinding algorithms. By ensuring
the visualization never crashes due to missing assets, researchers can focus on
algorithm correctness rather than asset pipeline issues.

Reference: Zelda VGLC Semantic Palette (zelda_core.py)

Author: KLTN Visualization Module
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)

try:
    import pygame
    from pygame import Surface
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    logger.warning("Pygame not available - asset loading disabled")


# ==========================================
# SEMANTIC COLOR PALETTE
# ==========================================

# Fallback colors for procedural tile generation
# Maps semantic tile IDs to RGB colors
SEMANTIC_COLORS: Dict[int, Tuple[int, int, int]] = {
    # Void and structural
    0:  (20, 20, 25),        # VOID - near black
    1:  (180, 165, 130),     # FLOOR - sand/stone
    2:  (50, 55, 90),        # WALL - dark blue-gray
    3:  (100, 75, 50),       # BLOCK - brown (pushable)
    
    # Doors (10-15)
    10: (90, 75, 55),        # DOOR_OPEN - wood brown
    11: (180, 140, 50),      # DOOR_LOCKED - golden (key required)
    12: (180, 80, 80),       # DOOR_BOMB - red-brown (bomb required)
    13: (140, 80, 180),      # DOOR_PUZZLE - purple
    14: (180, 40, 40),       # DOOR_BOSS - blood red
    15: (100, 100, 60),      # DOOR_SOFT - yellow-gray (one-way)
    
    # Entities (20-23)
    20: (200, 60, 60),       # ENEMY - red
    21: (80, 180, 80),       # START - green
    22: (255, 215, 0),       # TRIFORCE - gold (goal)
    23: (150, 25, 25),       # BOSS - dark red
    
    # Items (30-33)
    30: (255, 200, 50),      # KEY_SMALL - yellow
    31: (200, 100, 50),      # KEY_BOSS - orange
    32: (100, 200, 255),     # KEY_ITEM - cyan (ladder, etc.)
    33: (200, 200, 200),     # ITEM_MINOR - gray
    
    # Environment (40-43)
    40: (50, 80, 180),       # ELEMENT - water/lava (blue)
    41: (80, 100, 160),      # ELEMENT_FLOOR - walkable water
    42: (120, 100, 80),      # STAIR - stairs (teleport)
    43: (180, 100, 180),     # PUZZLE - puzzle tile
}

# Human-readable names for debugging
SEMANTIC_NAMES: Dict[int, str] = {
    0: "VOID", 1: "FLOOR", 2: "WALL", 3: "BLOCK",
    10: "DOOR_OPEN", 11: "DOOR_LOCKED", 12: "DOOR_BOMB",
    13: "DOOR_PUZZLE", 14: "DOOR_BOSS", 15: "DOOR_SOFT",
    20: "ENEMY", 21: "START", 22: "TRIFORCE", 23: "BOSS",
    30: "KEY_SMALL", 31: "KEY_BOSS", 32: "KEY_ITEM", 33: "ITEM_MINOR",
    40: "ELEMENT", 41: "ELEMENT_FLOOR", 42: "STAIR", 43: "PUZZLE",
}


class AssetManager:
    """
    Manages loading and caching of visual assets with graceful fallbacks.
    
    Design Philosophy:
    -----------------
    "Never Crash" - If an asset file is missing, generate a colored rectangle
    based on the semantic meaning of the tile. This allows debugging of logic
    without worrying about graphics pipeline.
    
    Usage:
    ------
        assets = AssetManager(tile_size=32)
        surface = assets.get_tile(FLOOR)  # Returns pygame.Surface
        link_sprite = assets.get_link_sprite()
    """
    
    def __init__(self, tile_size: int = 32, assets_path: Optional[str] = None):
        """
        Initialize asset manager.
        
        Args:
            tile_size: Size of tiles in pixels (square)
            assets_path: Optional path to assets folder. If None, uses procedural only.
        """
        if not PYGAME_AVAILABLE:
            raise RuntimeError("Pygame is required for AssetManager")
        
        self.tile_size = tile_size
        self.assets_path = Path(assets_path) if assets_path else None
        
        # Caches
        self._tile_cache: Dict[Tuple[int, int], Surface] = {}
        self._sprite_cache: Dict[str, Surface] = {}
        
        # Track missing assets (for logging, not crashing)
        self._missing_assets: set = set()
        
        logger.info(f"AssetManager initialized: tile_size={tile_size}")
    
    def set_tile_size(self, tile_size: int) -> None:
        """
        Change tile size and clear caches.
        
        Args:
            tile_size: New tile size in pixels
        """
        if tile_size != self.tile_size:
            self.tile_size = tile_size
            self._tile_cache.clear()
            self._sprite_cache.clear()
            logger.debug(f"Tile size changed to {tile_size}, caches cleared")
    
    def get_tile(self, tile_id: int) -> Surface:
        """
        Get a tile surface for the given semantic ID.
        
        This method NEVER raises an exception. If the sprite file is missing,
        it returns a procedurally generated colored rectangle.
        
        Args:
            tile_id: Semantic tile ID (from SEMANTIC_PALETTE)
            
        Returns:
            pygame.Surface of the tile
        """
        cache_key = (tile_id, self.tile_size)
        
        if cache_key in self._tile_cache:
            return self._tile_cache[cache_key]
        
        # Try loading from file first
        surface = self._try_load_tile_sprite(tile_id)
        
        if surface is None:
            # Generate procedural fallback
            surface = self._generate_procedural_tile(tile_id)
        
        self._tile_cache[cache_key] = surface
        return surface
    
    def get_link_sprite(self) -> Surface:
        """
        Get the Link (player) sprite.
        
        Returns:
            pygame.Surface of Link, scaled to tile_size - 4 pixels
        """
        cache_key = f"link_{self.tile_size}"
        
        if cache_key in self._sprite_cache:
            return self._sprite_cache[cache_key]
        
        # Try loading from file
        surface = self._try_load_sprite("link.png")
        
        if surface is None:
            # Generate procedural Link
            surface = self._generate_procedural_link()
        
        self._sprite_cache[cache_key] = surface
        return surface
    
    def _try_load_tile_sprite(self, tile_id: int) -> Optional[Surface]:
        """Attempt to load tile sprite from file."""
        if self.assets_path is None:
            return None
        
        # Try common naming conventions
        name = SEMANTIC_NAMES.get(tile_id, f"tile_{tile_id}")
        candidates = [
            self.assets_path / f"{name.lower()}.png",
            self.assets_path / f"tile_{tile_id}.png",
            self.assets_path / "tiles" / f"{name.lower()}.png",
        ]
        
        for path in candidates:
            if path.exists():
                try:
                    surface = pygame.image.load(str(path)).convert_alpha()
                    return pygame.transform.scale(surface, (self.tile_size, self.tile_size))
                except pygame.error as e:
                    logger.warning(f"Failed to load tile {path}: {e}")
        
        # Log missing asset once
        if tile_id not in self._missing_assets:
            self._missing_assets.add(tile_id)
            logger.debug(f"Tile sprite not found for ID {tile_id} ({name}), using procedural")
        
        return None
    
    def _try_load_sprite(self, filename: str) -> Optional[Surface]:
        """Attempt to load a named sprite from file."""
        if self.assets_path is None:
            return None
        
        candidates = [
            self.assets_path / filename,
            self.assets_path / "sprites" / filename,
        ]
        
        for path in candidates:
            if path.exists():
                try:
                    surface = pygame.image.load(str(path)).convert_alpha()
                    target_size = self.tile_size - 4
                    return pygame.transform.scale(surface, (target_size, target_size))
                except pygame.error as e:
                    logger.warning(f"Failed to load sprite {path}: {e}")
        
        return None
    
    def _generate_procedural_tile(self, tile_id: int) -> Surface:
        """
        Generate a procedural tile with visual indicators.
        
        Creates a gradient-filled tile with semantic decorations
        (keyholes, cracks, arrows, etc.)
        """
        surface = Surface((self.tile_size, self.tile_size), pygame.SRCALPHA)
        
        # Get base color (default to hot pink for unknown IDs - easy to spot bugs)
        base_color = SEMANTIC_COLORS.get(tile_id, (255, 0, 255))
        
        # Create gradient effect (lighter at top)
        highlight = self._lighten(base_color, 1.3)
        shadow = self._darken(base_color, 0.7)
        
        # Draw gradient background
        for y in range(self.tile_size):
            t = y / max(1, self.tile_size - 1)
            r = int(highlight[0] * (1 - t) + base_color[0] * t)
            g = int(highlight[1] * (1 - t) + base_color[1] * t)
            b = int(highlight[2] * (1 - t) + base_color[2] * t)
            pygame.draw.line(surface, (r, g, b), (0, y), (self.tile_size - 1, y))
        
        # Draw border for depth
        pygame.draw.rect(surface, shadow, (0, 0, self.tile_size, self.tile_size), 1)
        
        # Add semantic decorations
        self._draw_tile_decoration(surface, tile_id)
        
        return surface
    
    def _draw_tile_decoration(self, surface: Surface, tile_id: int) -> None:
        """Add visual indicators based on tile type."""
        size = self.tile_size
        center = size // 2
        
        if tile_id == 11:  # DOOR_LOCKED - keyhole
            pygame.draw.circle(surface, (255, 200, 50), (center, center - 4), max(3, size // 8))
            pygame.draw.rect(surface, (255, 200, 50), (center - 2, center, 4, size // 4))
        
        elif tile_id == 12:  # DOOR_BOMB - crack pattern
            crack = (40, 40, 40)
            pygame.draw.line(surface, crack, (size // 4, size // 4), (size * 3 // 4, size * 3 // 4), 2)
            pygame.draw.line(surface, crack, (size * 3 // 4, size // 4), (size // 4, size * 3 // 4), 2)
        
        elif tile_id == 14:  # DOOR_BOSS - skull indicator
            pygame.draw.circle(surface, (255, 255, 255), (center, center - 2), max(4, size // 6))
            pygame.draw.circle(surface, (0, 0, 0), (center - 2, center - 3), max(1, size // 16))
            pygame.draw.circle(surface, (0, 0, 0), (center + 2, center - 3), max(1, size // 16))
        
        elif tile_id == 21:  # START - spawn marker
            points = [
                (center, size // 4), (size * 3 // 4, center),
                (center, size * 3 // 4), (size // 4, center)
            ]
            pygame.draw.polygon(surface, (255, 255, 255), points, 2)
        
        elif tile_id == 22:  # TRIFORCE - golden triangle
            points = [(center, size // 6), (size // 6, size * 5 // 6), (size * 5 // 6, size * 5 // 6)]
            pygame.draw.polygon(surface, (255, 255, 200), points)
            pygame.draw.polygon(surface, (200, 180, 0), points, 2)
        
        elif tile_id == 30:  # KEY_SMALL - key shape
            pygame.draw.circle(surface, (200, 150, 0), (center, size // 3), max(3, size // 6))
            pygame.draw.rect(surface, (200, 150, 0), (center - 2, size // 3, 4, size // 2))
            pygame.draw.rect(surface, (200, 150, 0), (center, size * 2 // 3, size // 6, 3))
        
        elif tile_id == 42:  # STAIR - step pattern
            step_color = (80, 60, 40)
            for i in range(4):
                y = size // 5 + i * size // 6
                w = size - (i * size // 8)
                x = (size - w) // 2
                pygame.draw.rect(surface, step_color, (x, y, w, size // 8))
        
        elif tile_id in (20, 23):  # ENEMY or BOSS - eyes
            eye_color = (255, 255, 255) if tile_id == 20 else (255, 200, 0)
            pygame.draw.circle(surface, eye_color, (center - size // 6, center - size // 8), max(2, size // 12))
            pygame.draw.circle(surface, eye_color, (center + size // 6, center - size // 8), max(2, size // 12))
            pygame.draw.circle(surface, (0, 0, 0), (center - size // 6, center - size // 8), max(1, size // 20))
            pygame.draw.circle(surface, (0, 0, 0), (center + size // 6, center - size // 8), max(1, size // 20))
    
    def _generate_procedural_link(self) -> Surface:
        """
        Generate a procedural Link sprite.
        
        Creates a recognizable green-tunic character without external assets.
        """
        size = self.tile_size - 4
        surface = Surface((size, size), pygame.SRCALPHA)
        
        # Colors
        tunic = (0, 168, 0)
        tunic_dark = (0, 120, 0)
        skin = (252, 216, 168)
        hair = (136, 112, 0)
        shield = (200, 150, 50)
        sword = (180, 180, 180)
        
        # Scale factor for different tile sizes
        s = size / 28  # Normalize to 28px reference
        
        # Body (green tunic)
        pygame.draw.rect(surface, tunic, (int(8*s), int(12*s), int(12*s), int(12*s)))
        pygame.draw.rect(surface, tunic_dark, (int(6*s), int(18*s), int(4*s), int(8*s)))
        pygame.draw.rect(surface, tunic_dark, (int(18*s), int(18*s), int(4*s), int(8*s)))
        
        # Head
        pygame.draw.rect(surface, skin, (int(8*s), int(2*s), int(12*s), int(10*s)))
        pygame.draw.circle(surface, (0, 0, 0), (int(11*s), int(6*s)), max(1, int(2*s)))
        pygame.draw.circle(surface, (0, 0, 0), (int(17*s), int(6*s)), max(1, int(2*s)))
        
        # Hair/cap
        pygame.draw.rect(surface, hair, (int(6*s), int(0*s), int(16*s), int(4*s)))
        pygame.draw.rect(surface, hair, (int(4*s), int(2*s), int(4*s), int(6*s)))
        pygame.draw.rect(surface, hair, (int(20*s), int(2*s), int(4*s), int(6*s)))
        
        # Shield
        pygame.draw.rect(surface, (136, 112, 0), (int(2*s), int(14*s), int(6*s), int(10*s)))
        pygame.draw.rect(surface, shield, (int(3*s), int(15*s), int(4*s), int(8*s)))
        
        # Sword
        pygame.draw.rect(surface, sword, (int(22*s), int(12*s), int(4*s), int(14*s)))
        pygame.draw.rect(surface, hair, (int(22*s), int(10*s), int(4*s), int(4*s)))
        
        return surface
    
    @staticmethod
    def _lighten(color: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
        """Lighten a color by a factor."""
        return tuple(min(255, int(c * factor)) for c in color)
    
    @staticmethod
    def _darken(color: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
        """Darken a color by a factor."""
        return tuple(max(0, int(c * factor)) for c in color)
    
    def clear_cache(self) -> None:
        """Clear all cached surfaces."""
        self._tile_cache.clear()
        self._sprite_cache.clear()
        logger.debug("Asset caches cleared")


# Factory function for convenience
def create_asset_manager(tile_size: int = 32, assets_path: Optional[str] = None) -> AssetManager:
    """
    Create an AssetManager instance.
    
    Args:
        tile_size: Tile size in pixels
        assets_path: Optional path to assets folder
        
    Returns:
        Configured AssetManager instance
    """
    return AssetManager(tile_size=tile_size, assets_path=assets_path)
