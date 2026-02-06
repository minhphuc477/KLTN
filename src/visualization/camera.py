"""
Camera System - Viewport Management for Large Maps
===================================================

Implements a camera/viewport system to handle maps larger than the screen.
Keeps the player (Link) centered while clamping to map boundaries.

Key Features:
- Smooth camera following with lerp interpolation
- Edge clamping to prevent showing void outside map
- Support for multiple zoom levels
- Coordinate transformation utilities

Scientific Rationale:
--------------------
Zelda dungeons are stitched from multiple rooms (16x11 tiles each), resulting
in maps up to 96x66 tiles. At 32px/tile, this is 3072x2112 pixels - far larger
than typical screen sizes. The camera system enables visualization of the
complete solution path while keeping the active agent visible.

Reference:
- VGLC room dimensions: 16 rows × 11 columns
- Typical dungeon: 6×6 room grid = 96×66 tiles

Author: KLTN Visualization Module
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class Viewport:
    """
    Represents the visible area of the world.
    
    Attributes:
        x: Left edge of viewport in world coordinates
        y: Top edge of viewport in world coordinates
        width: Width of viewport in pixels
        height: Height of viewport in pixels
    """
    x: float = 0.0
    y: float = 0.0
    width: int = 800
    height: int = 600


class Camera:
    """
    Camera system for tracking a target position on large maps.
    
    The camera keeps the target (typically the player) centered while
    ensuring the viewport never shows areas outside the world bounds.
    
    Design:
    -------
    - Uses smooth interpolation (lerp) for pleasant camera movement
    - Clamps to world boundaries to avoid showing void
    - Provides coordinate transformation methods for rendering
    
    Usage:
    ------
        camera = Camera(
            world_width=96 * 32,  # Map width in pixels
            world_height=66 * 32,  # Map height in pixels
            viewport_width=800,
            viewport_height=600
        )
        
        # In update loop:
        camera.set_target(player_x, player_y)
        camera.update(delta_time)
        
        # In render loop:
        screen_pos = camera.world_to_screen(entity_x, entity_y)
    """
    
    def __init__(
        self,
        world_width: int,
        world_height: int,
        viewport_width: int = 800,
        viewport_height: int = 600,
        smoothing: float = 8.0,
        tile_size: int = 32
    ):
        """
        Initialize the camera.
        
        Args:
            world_width: Total world width in pixels
            world_height: Total world height in pixels
            viewport_width: Viewport (screen) width in pixels
            viewport_height: Viewport (screen) height in pixels
            smoothing: Lerp smoothing factor (higher = faster following)
            tile_size: Size of one tile in pixels (for grid calculations)
        """
        self.world_width = world_width
        self.world_height = world_height
        self.tile_size = tile_size
        self.smoothing = smoothing
        
        # Viewport
        self.viewport = Viewport(0, 0, viewport_width, viewport_height)
        
        # Current and target camera positions (center of viewport)
        self._position_x: float = viewport_width / 2
        self._position_y: float = viewport_height / 2
        self._target_x: float = self._position_x
        self._target_y: float = self._position_y
        
        # Immediate mode (skip interpolation)
        self._immediate_next = False
    
    @property
    def x(self) -> float:
        """Camera X offset (left edge of viewport in world coordinates)."""
        return self._position_x - self.viewport.width / 2
    
    @property
    def y(self) -> float:
        """Camera Y offset (top edge of viewport in world coordinates)."""
        return self._position_y - self.viewport.height / 2
    
    @property
    def offset(self) -> Tuple[int, int]:
        """Get camera offset as integer tuple for rendering."""
        return (int(self.x), int(self.y))
    
    def set_target(
        self,
        world_x: float,
        world_y: float,
        immediate: bool = False
    ) -> None:
        """
        Set the target position for the camera to follow.
        
        The camera will smoothly interpolate toward this position,
        keeping it centered (subject to boundary clamping).
        
        Args:
            world_x: Target X position in world coordinates (pixels)
            world_y: Target Y position in world coordinates (pixels)
            immediate: If True, snap to target without interpolation
        """
        self._target_x = world_x
        self._target_y = world_y
        
        if immediate:
            self._immediate_next = True
    
    def set_target_grid(
        self,
        row: int,
        col: int,
        immediate: bool = False
    ) -> None:
        """
        Set target position using grid coordinates.
        
        Converts (row, col) to world pixels and sets as target.
        
        Args:
            row: Grid row (Y in tiles)
            col: Grid column (X in tiles)
            immediate: If True, snap without interpolation
        """
        world_x = col * self.tile_size + self.tile_size / 2
        world_y = row * self.tile_size + self.tile_size / 2
        self.set_target(world_x, world_y, immediate)
    
    def update(self, dt: float) -> None:
        """
        Update camera position with smooth interpolation.
        
        Call this once per frame before rendering.
        
        Args:
            dt: Delta time in seconds since last update
        """
        if self._immediate_next:
            # Snap directly to target
            self._position_x = self._target_x
            self._position_y = self._target_y
            self._immediate_next = False
        else:
            # Smooth interpolation (lerp)
            t = min(1.0, dt * self.smoothing)
            self._position_x += (self._target_x - self._position_x) * t
            self._position_y += (self._target_y - self._position_y) * t
        
        # Clamp to world boundaries
        self._clamp_to_bounds()
    
    def _clamp_to_bounds(self) -> None:
        """
        Clamp camera position to prevent showing outside world bounds.
        
        Ensures the viewport never shows the void beyond map edges.
        """
        half_w = self.viewport.width / 2
        half_h = self.viewport.height / 2
        
        # Minimum camera center position (keeps left/top edges in bounds)
        min_x = half_w
        min_y = half_h
        
        # Maximum camera center position (keeps right/bottom edges in bounds)
        max_x = max(half_w, self.world_width - half_w)
        max_y = max(half_h, self.world_height - half_h)
        
        self._position_x = max(min_x, min(self._position_x, max_x))
        self._position_y = max(min_y, min(self._position_y, max_y))
    
    def set_viewport_size(self, width: int, height: int) -> None:
        """
        Update viewport dimensions (e.g., on window resize).
        
        Args:
            width: New viewport width in pixels
            height: New viewport height in pixels
        """
        self.viewport.width = width
        self.viewport.height = height
        self._clamp_to_bounds()
    
    def set_world_size(self, width: int, height: int) -> None:
        """
        Update world dimensions (e.g., when loading a new map).
        
        Args:
            width: New world width in pixels
            height: New world height in pixels
        """
        self.world_width = width
        self.world_height = height
        self._clamp_to_bounds()
    
    def world_to_screen(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """
        Convert world coordinates to screen (viewport) coordinates.
        
        Use this when rendering entities to determine their screen position.
        
        Args:
            world_x: X position in world coordinates
            world_y: Y position in world coordinates
            
        Returns:
            (screen_x, screen_y) tuple
        """
        screen_x = int(world_x - self.x)
        screen_y = int(world_y - self.y)
        return (screen_x, screen_y)
    
    def screen_to_world(self, screen_x: int, screen_y: int) -> Tuple[float, float]:
        """
        Convert screen coordinates to world coordinates.
        
        Use this for mouse input to determine clicked world position.
        
        Args:
            screen_x: X position on screen
            screen_y: Y position on screen
            
        Returns:
            (world_x, world_y) tuple
        """
        world_x = screen_x + self.x
        world_y = screen_y + self.y
        return (world_x, world_y)
    
    def grid_to_screen(self, row: int, col: int) -> Tuple[int, int]:
        """
        Convert grid coordinates to screen position.
        
        Args:
            row: Grid row
            col: Grid column
            
        Returns:
            (screen_x, screen_y) of tile's top-left corner
        """
        world_x = col * self.tile_size
        world_y = row * self.tile_size
        return self.world_to_screen(world_x, world_y)
    
    def is_visible(self, world_x: float, world_y: float, margin: int = 0) -> bool:
        """
        Check if a world position is within the visible viewport.
        
        Args:
            world_x: X position in world coordinates
            world_y: Y position in world coordinates
            margin: Extra margin around viewport (for culling large sprites)
            
        Returns:
            True if position is visible
        """
        left = self.x - margin
        top = self.y - margin
        right = self.x + self.viewport.width + margin
        bottom = self.y + self.viewport.height + margin
        
        return left <= world_x <= right and top <= world_y <= bottom
    
    def is_tile_visible(self, row: int, col: int) -> bool:
        """
        Check if a tile is at least partially visible.
        
        Args:
            row: Tile row
            col: Tile column
            
        Returns:
            True if tile is at least partially visible
        """
        tile_x = col * self.tile_size
        tile_y = row * self.tile_size
        
        return (
            tile_x + self.tile_size > self.x and
            tile_x < self.x + self.viewport.width and
            tile_y + self.tile_size > self.y and
            tile_y < self.y + self.viewport.height
        )
    
    def get_visible_tile_range(self) -> Tuple[int, int, int, int]:
        """
        Get the range of visible tiles for efficient rendering.
        
        Returns:
            (start_row, end_row, start_col, end_col) - exclusive end indices
        """
        start_col = max(0, int(self.x // self.tile_size))
        start_row = max(0, int(self.y // self.tile_size))
        
        end_col = min(
            int((self.x + self.viewport.width) // self.tile_size) + 2,
            self.world_width // self.tile_size
        )
        end_row = min(
            int((self.y + self.viewport.height) // self.tile_size) + 2,
            self.world_height // self.tile_size
        )
        
        return (start_row, end_row, start_col, end_col)
    
    def center_on_world(self) -> None:
        """Center the camera on the world (useful for initial view)."""
        self._target_x = self.world_width / 2
        self._target_y = self.world_height / 2
        self._immediate_next = True


def create_camera_for_map(
    map_rows: int,
    map_cols: int,
    tile_size: int = 32,
    viewport_width: int = 800,
    viewport_height: int = 600
) -> Camera:
    """
    Factory function to create a camera for a given map size.
    
    Args:
        map_rows: Number of tile rows in the map
        map_cols: Number of tile columns in the map
        tile_size: Size of tiles in pixels
        viewport_width: Viewport width in pixels
        viewport_height: Viewport height in pixels
        
    Returns:
        Configured Camera instance
    """
    world_width = map_cols * tile_size
    world_height = map_rows * tile_size
    
    return Camera(
        world_width=world_width,
        world_height=world_height,
        viewport_width=viewport_width,
        viewport_height=viewport_height,
        tile_size=tile_size
    )
