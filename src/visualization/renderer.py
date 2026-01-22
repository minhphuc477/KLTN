"""
KLTN Visualization - Core Renderer
===================================

High-quality rendering system for Zelda dungeon visualization.
Provides smooth animations, procedural tile generation, and sprite management.

Features:
- Delta-time based animations
- Smooth interpolated movement
- Procedural gradient tiles (no sprites required)
- Heatmap visualization for A* search
- Modern semi-dark theme

"""

from __future__ import annotations

import os
import math
from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum

try:
    import pygame
    from pygame import Surface
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

if TYPE_CHECKING:
    import numpy as np


# ==========================================
# THEME CONFIGURATION
# ==========================================

@dataclass
class ThemeConfig:
    """
    Configurable visual theme for the renderer.
    
    All colors use RGB or RGBA tuples. The theme uses a modern
    semi-dark palette that's easy on the eyes and looks professional.
    """
    # Tile rendering
    TILE_SIZE: int = 32
    ANIMATION_SPEED: float = 8.0  # Lerp speed multiplier
    
    # Color palette - Modern semi-dark theme
    COLORS: Dict[str, Tuple[int, ...]] = field(default_factory=lambda: {
        # Background and void
        'background': (25, 25, 35),
        'void': (15, 15, 22),
        
        # Floor and structural
        'floor': (180, 165, 130),
        'floor_highlight': (200, 185, 150),
        'wall': (50, 60, 100),
        'wall_highlight': (70, 80, 130),
        'block': (100, 75, 50),
        'block_highlight': (120, 95, 70),
        
        # Doors
        'door_open': (100, 80, 60),
        'door_open_highlight': (120, 100, 80),
        'door_locked': (200, 160, 60),
        'door_locked_highlight': (220, 180, 80),
        'door_bomb': (200, 80, 80),
        'door_bomb_highlight': (220, 100, 100),
        'door_boss': (180, 40, 40),
        'door_boss_highlight': (200, 60, 60),
        'door_puzzle': (140, 80, 180),
        'door_puzzle_highlight': (160, 100, 200),
        'door_soft': (100, 100, 60),
        'door_soft_highlight': (120, 120, 80),
        
        # Interactive elements
        'start': (80, 180, 80),
        'start_highlight': (100, 200, 100),
        'triforce': (255, 215, 0),
        'triforce_highlight': (255, 235, 100),
        'triforce_glow': (255, 240, 150, 80),
        
        # Enemies
        'enemy': (180, 60, 60),
        'enemy_highlight': (200, 80, 80),
        'boss': (150, 20, 20),
        'boss_highlight': (180, 40, 40),
        
        # Items
        'key': (255, 200, 50),
        'key_highlight': (255, 220, 100),
        'key_boss': (200, 100, 50),
        'key_boss_highlight': (220, 120, 70),
        'item': (200, 200, 200),
        'item_highlight': (220, 220, 220),
        
        # Environment
        'element': (50, 80, 180),
        'element_highlight': (70, 100, 200),
        'element_floor': (80, 100, 160),
        'stair': (120, 100, 80),
        'stair_highlight': (140, 120, 100),
        'puzzle': (180, 100, 180),
        
        # Path visualization
        'path_trail': (100, 150, 255, 100),
        'path_glow': (120, 180, 255, 150),
        'path_visited': (80, 120, 200, 60),
        
        # Heatmap
        'heatmap_cold': (50, 50, 200),
        'heatmap_mid': (200, 50, 200),
        'heatmap_hot': (200, 50, 50),
        
        # HUD
        'hud_panel': (40, 40, 60, 200),
        'hud_panel_border': (70, 70, 100),
        'hud_text': (220, 220, 220),
        'hud_text_highlight': (255, 255, 255),
        'hud_text_dim': (140, 140, 160),
        'hud_success': (100, 255, 100),
        'hud_warning': (255, 200, 50),
        'hud_error': (255, 80, 80),
        
        # Link (player)
        'link_tunic': (0, 168, 0),
        'link_tunic_dark': (0, 120, 0),
        'link_skin': (252, 216, 168),
        'link_hair': (136, 112, 0),
        'link_shield': (200, 150, 50),
        'link_sword': (180, 180, 180),
    })
    
    def get_color(self, name: str, alpha: int = None) -> Tuple[int, ...]:
        """Get a color by name, optionally with custom alpha."""
        color = self.COLORS.get(name, (128, 128, 128))
        if alpha is not None:
            if len(color) == 4:
                return (*color[:3], alpha)
            return (*color, alpha)
        return color


# ==========================================
# VECTOR MATH
# ==========================================

class Vector2:
    """
    2D vector with interpolation and math support.
    
    Used for smooth position interpolation and distance calculations.
    """
    
    __slots__ = ('x', 'y')
    
    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = float(x)
        self.y = float(y)
    
    def lerp(self, target: 'Vector2', t: float) -> 'Vector2':
        """
        Linear interpolation toward target.
        
        Args:
            target: Target position to interpolate toward
            t: Interpolation factor (0.0 to 1.0)
            
        Returns:
            New interpolated Vector2
        """
        t = max(0.0, min(1.0, t))  # Clamp t
        return Vector2(
            self.x + (target.x - self.x) * t,
            self.y + (target.y - self.y) * t
        )
    
    def distance_to(self, other: 'Vector2') -> float:
        """Calculate Euclidean distance to another vector."""
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx * dx + dy * dy)
    
    def manhattan_distance(self, other: 'Vector2') -> float:
        """Calculate Manhattan distance to another vector."""
        return abs(self.x - other.x) + abs(self.y - other.y)
    
    def to_tuple(self) -> Tuple[float, float]:
        """Convert to tuple."""
        return (self.x, self.y)
    
    def to_int_tuple(self) -> Tuple[int, int]:
        """Convert to integer tuple (for pixel coordinates)."""
        return (int(self.x), int(self.y))
    
    def __add__(self, other: 'Vector2') -> 'Vector2':
        return Vector2(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: 'Vector2') -> 'Vector2':
        return Vector2(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar: float) -> 'Vector2':
        return Vector2(self.x * scalar, self.y * scalar)
    
    def __repr__(self) -> str:
        return f"Vector2({self.x:.2f}, {self.y:.2f})"
    
    @classmethod
    def from_tuple(cls, t: Tuple[float, float]) -> 'Vector2':
        """Create from a tuple."""
        return cls(t[0], t[1])
    
    @classmethod
    def from_grid(cls, row: int, col: int) -> 'Vector2':
        """Create from grid coordinates (row, col)."""
        return cls(col, row)  # Note: col=x, row=y


# ==========================================
# PROCEDURAL TILE RENDERER
# ==========================================

class ProceduralTileRenderer:
    """
    Generates beautiful procedural tiles without requiring sprite assets.
    
    Each tile is rendered with:
    - Gradient fills (lighter at top for 3D effect)
    - Borders for depth
    - Visual indicators for special tiles (keyholes, cracks, etc.)
    """
    
    def __init__(self, theme: ThemeConfig):
        self.theme = theme
        self._tile_cache: Dict[Tuple[int, int], Surface] = {}
    
    def get_tile(self, tile_id: int, tile_size: int) -> Surface:
        """
        Get a procedurally generated tile surface.
        
        Args:
            tile_id: Semantic tile ID
            tile_size: Size of the tile in pixels
            
        Returns:
            Pygame Surface with the rendered tile
        """
        cache_key = (tile_id, tile_size)
        if cache_key in self._tile_cache:
            return self._tile_cache[cache_key]
        
        surface = self._render_tile(tile_id, tile_size)
        self._tile_cache[cache_key] = surface
        return surface
    
    def clear_cache(self):
        """Clear the tile cache (call when changing tile size)."""
        self._tile_cache.clear()
    
    def _render_tile(self, tile_id: int, tile_size: int) -> Surface:
        """Render a single tile with gradients and decorations."""
        if not PYGAME_AVAILABLE:
            raise RuntimeError("Pygame required for rendering")
        
        surface = pygame.Surface((tile_size, tile_size), pygame.SRCALPHA)
        
        # Map tile ID to color name
        color_name = self._get_color_name(tile_id)
        base_color = self.theme.get_color(color_name)
        highlight_name = f"{color_name}_highlight"
        highlight_color = self.theme.get_color(highlight_name, None)
        if highlight_color == (128, 128, 128):  # Default fallback
            highlight_color = self._lighten_color(base_color, 1.2)
        
        # Draw gradient fill
        self._draw_gradient_rect(surface, base_color, highlight_color, tile_size)
        
        # Draw border for depth
        darker = self._darken_color(base_color, 0.7)
        pygame.draw.rect(surface, darker, (0, 0, tile_size, tile_size), 1)
        
        # Draw special indicators
        self._draw_tile_decoration(surface, tile_id, tile_size)
        
        return surface
    
    def _get_color_name(self, tile_id: int) -> str:
        """Map semantic tile ID to theme color name."""
        # Standard tile ID mappings
        id_map = {
            0: 'void',
            1: 'floor',
            2: 'wall',
            3: 'block',
            10: 'door_open',
            11: 'door_locked',
            12: 'door_bomb',
            13: 'door_puzzle',
            14: 'door_boss',
            15: 'door_soft',
            20: 'enemy',
            21: 'start',
            22: 'triforce',
            23: 'boss',
            30: 'key',
            31: 'key_boss',
            32: 'item',
            33: 'item',
            40: 'element',
            41: 'element_floor',
            42: 'stair',
            43: 'puzzle',
        }
        return id_map.get(tile_id, 'floor')
    
    def _draw_gradient_rect(self, surface: Surface, base_color: Tuple[int, ...], 
                           highlight_color: Tuple[int, ...], size: int):
        """Draw a gradient-filled rectangle (lighter at top)."""
        for i in range(size):
            # Interpolate from highlight (top) to base (bottom)
            t = i / max(1, size - 1)
            r = int(highlight_color[0] * (1 - t) + base_color[0] * t)
            g = int(highlight_color[1] * (1 - t) + base_color[1] * t)
            b = int(highlight_color[2] * (1 - t) + base_color[2] * t)
            color = (min(255, max(0, r)), min(255, max(0, g)), min(255, max(0, b)))
            pygame.draw.line(surface, color, (0, i), (size - 1, i))
    
    def _draw_tile_decoration(self, surface: Surface, tile_id: int, size: int):
        """Draw visual indicators for special tiles."""
        center = size // 2
        
        if tile_id == 11:  # DOOR_LOCKED - keyhole
            pygame.draw.circle(surface, (255, 200, 50), (center, center - 4), max(3, size // 8))
            pygame.draw.rect(surface, (255, 200, 50), 
                           (center - 2, center, 4, max(4, size // 4)))
        
        elif tile_id == 12:  # DOOR_BOMB - crack pattern
            crack_color = (40, 40, 40)
            pygame.draw.line(surface, crack_color, (size // 4, size // 4), 
                           (size * 3 // 4, size * 3 // 4), 2)
            pygame.draw.line(surface, crack_color, (size * 3 // 4, size // 4), 
                           (size // 4, size * 3 // 4), 2)
        
        elif tile_id == 14:  # DOOR_BOSS - skull/danger indicator
            pygame.draw.circle(surface, (255, 255, 255), (center, center - 2), max(4, size // 6))
            pygame.draw.circle(surface, (0, 0, 0), (center - 2, center - 3), max(1, size // 16))
            pygame.draw.circle(surface, (0, 0, 0), (center + 2, center - 3), max(1, size // 16))
            pygame.draw.arc(surface, (0, 0, 0), (center - 4, center, 8, 4), 0, 3.14, 1)
        
        elif tile_id == 21:  # START - arrow/marker
            arrow_color = (255, 255, 255)
            points = [
                (center, size // 4),
                (size * 3 // 4, center),
                (center, size * 3 // 4),
                (size // 4, center)
            ]
            pygame.draw.polygon(surface, arrow_color, points, 2)
        
        elif tile_id == 22:  # TRIFORCE - golden triangle
            triforce_color = (255, 255, 200)
            border_color = (200, 180, 0)
            points = [(center, size // 6), (size // 6, size * 5 // 6), (size * 5 // 6, size * 5 // 6)]
            pygame.draw.polygon(surface, triforce_color, points)
            pygame.draw.polygon(surface, border_color, points, 2)
        
        elif tile_id == 30:  # KEY - key shape
            key_color = (200, 150, 0)
            pygame.draw.circle(surface, key_color, (center, size // 3), max(3, size // 6))
            pygame.draw.rect(surface, key_color, (center - 2, size // 3, 4, size // 2))
            pygame.draw.rect(surface, key_color, (center, size * 2 // 3, size // 6, 3))
        
        elif tile_id == 31:  # KEY_BOSS - ornate key
            key_color = (200, 100, 50)
            pygame.draw.circle(surface, key_color, (center, size // 4), max(4, size // 5))
            pygame.draw.circle(surface, (150, 70, 30), (center, size // 4), max(2, size // 8))
            pygame.draw.rect(surface, key_color, (center - 2, size // 4, 4, size // 2))
        
        elif tile_id == 20 or tile_id == 23:  # ENEMY or BOSS
            eye_color = (255, 255, 255) if tile_id == 20 else (255, 200, 0)
            pygame.draw.circle(surface, eye_color, (center - size // 6, center - size // 8), max(2, size // 12))
            pygame.draw.circle(surface, eye_color, (center + size // 6, center - size // 8), max(2, size // 12))
            pygame.draw.circle(surface, (0, 0, 0), (center - size // 6, center - size // 8), max(1, size // 20))
            pygame.draw.circle(surface, (0, 0, 0), (center + size // 6, center - size // 8), max(1, size // 20))
        
        elif tile_id == 42:  # STAIR - step pattern
            step_color = (80, 60, 40)
            for i in range(4):
                y = size // 5 + i * size // 6
                w = size - (i * size // 8)
                x = (size - w) // 2
                pygame.draw.rect(surface, step_color, (x, y, w, size // 8))
        
        elif tile_id == 40:  # ELEMENT (water/lava) - wave pattern
            wave_color = (80, 120, 200)
            for i in range(3):
                y = size // 4 + i * size // 4
                for x in range(0, size, size // 4):
                    pygame.draw.arc(surface, wave_color, 
                                   (x - size // 8, y - size // 8, size // 4, size // 4),
                                   0, 3.14, 2)
        
        elif tile_id == 2:  # WALL - brick pattern
            brick_color = self._lighten_color(self.theme.get_color('wall'), 1.1)
            pygame.draw.line(surface, brick_color, (0, size // 2), (size, size // 2), 1)
            pygame.draw.line(surface, brick_color, (size // 2, 0), (size // 2, size // 2), 1)
            pygame.draw.line(surface, brick_color, (size // 4, size // 2), (size // 4, size), 1)
            pygame.draw.line(surface, brick_color, (size * 3 // 4, size // 2), (size * 3 // 4, size), 1)
    
    def _lighten_color(self, color: Tuple[int, ...], factor: float) -> Tuple[int, ...]:
        """Lighten a color by a factor."""
        r = min(255, int(color[0] * factor))
        g = min(255, int(color[1] * factor))
        b = min(255, int(color[2] * factor))
        if len(color) == 4:
            return (r, g, b, color[3])
        return (r, g, b)
    
    def _darken_color(self, color: Tuple[int, ...], factor: float) -> Tuple[int, ...]:
        """Darken a color by a factor."""
        r = max(0, int(color[0] * factor))
        g = max(0, int(color[1] * factor))
        b = max(0, int(color[2] * factor))
        if len(color) == 4:
            return (r, g, b, color[3])
        return (r, g, b)


# ==========================================
# SPRITE MANAGER
# ==========================================

class SpriteManager:
    """
    Load and manage sprite assets with auto-scaling and fallback.
    
    Automatically scales sprites to match TILE_SIZE and falls back
    to procedural rendering if sprites are unavailable.
    """
    
    def __init__(self, theme: ThemeConfig, assets_dir: str = None):
        self.theme = theme
        self.assets_dir = assets_dir
        self.procedural = ProceduralTileRenderer(theme)
        self._sprites: Dict[str, Surface] = {}
        self._link_sprites: Dict[str, Surface] = {}
        self._loaded = False
        
        if assets_dir:
            self._load_sprites()
    
    def _load_sprites(self):
        """Load sprite sheets from assets directory."""
        if not PYGAME_AVAILABLE or not self.assets_dir:
            return
        
        if not os.path.exists(self.assets_dir):
            return
        
        # Try to load tileset
        tileset_path = os.path.join(self.assets_dir, 
                                    "NES - The Legend of Zelda - Tilesets - Dungeon Tileset.png")
        if os.path.exists(tileset_path):
            try:
                self._sprites['tileset'] = pygame.image.load(tileset_path).convert_alpha()
                self._loaded = True
            except Exception:
                pass
        
        # Try to load Link sprite
        link_path = os.path.join(self.assets_dir,
                                 "NES - The Legend of Zelda - Playable Characters - Link.png")
        if os.path.exists(link_path):
            try:
                self._link_sprites['sheet'] = pygame.image.load(link_path).convert_alpha()
            except Exception:
                pass
    
    def get_tile(self, tile_id: int, tile_size: int) -> Surface:
        """
        Get a tile surface, using sprites if available, otherwise procedural.
        """
        # For now, always use procedural (can extend to use sprite sheet later)
        return self.procedural.get_tile(tile_id, tile_size)
    
    def get_link_sprite(self, tile_size: int, direction: str = 'down') -> Surface:
        """
        Get Link sprite, either from sprite sheet or procedurally generated.
        
        Args:
            tile_size: Size to render at
            direction: 'up', 'down', 'left', 'right'
        """
        cache_key = f"link_{tile_size}_{direction}"
        if cache_key in self._sprites:
            return self._sprites[cache_key]
        
        # Generate procedural Link sprite
        sprite = self._create_procedural_link(tile_size, direction)
        self._sprites[cache_key] = sprite
        return sprite
    
    def _create_procedural_link(self, tile_size: int, direction: str) -> Surface:
        """Create a procedural Link sprite."""
        if not PYGAME_AVAILABLE:
            raise RuntimeError("Pygame required")
        
        size = tile_size - 4  # Slight padding
        surface = pygame.Surface((size, size), pygame.SRCALPHA)
        
        # Colors
        green = self.theme.get_color('link_tunic')
        dark_green = self.theme.get_color('link_tunic_dark')
        skin = self.theme.get_color('link_skin')
        hair = self.theme.get_color('link_hair')
        shield = self.theme.get_color('link_shield')
        sword = self.theme.get_color('link_sword')
        
        center = size // 2
        
        # Body (green tunic)
        body_rect = pygame.Rect(size // 4, size // 3, size // 2, size // 2)
        pygame.draw.rect(surface, green, body_rect)
        pygame.draw.rect(surface, dark_green, body_rect, 1)
        
        # Head
        head_rect = pygame.Rect(size // 4, size // 8, size // 2, size // 3)
        pygame.draw.rect(surface, skin, head_rect)
        
        # Hair
        hair_rect = pygame.Rect(size // 5, 0, size * 3 // 5, size // 6)
        pygame.draw.rect(surface, hair, hair_rect)
        
        # Eyes
        eye_y = size // 4
        pygame.draw.circle(surface, (0, 0, 0), (center - size // 8, eye_y), max(1, size // 16))
        pygame.draw.circle(surface, (0, 0, 0), (center + size // 8, eye_y), max(1, size // 16))
        
        # Shield (left side)
        shield_rect = pygame.Rect(2, size // 3, size // 5, size // 3)
        pygame.draw.rect(surface, shield, shield_rect)
        pygame.draw.rect(surface, hair, shield_rect, 1)
        
        # Sword (right side)
        sword_rect = pygame.Rect(size - size // 6, size // 4, size // 8, size // 2)
        pygame.draw.rect(surface, sword, sword_rect)
        pygame.draw.rect(surface, (100, 100, 100), sword_rect, 1)
        
        return surface
    
    def clear_cache(self):
        """Clear sprite cache (call when changing tile size)."""
        self.procedural.clear_cache()
        # Keep loaded sprites, just clear generated ones
        keys_to_remove = [k for k in self._sprites if k not in ('tileset',)]
        for k in keys_to_remove:
            del self._sprites[k]


# ==========================================
# ANIMATION CONTROLLER
# ==========================================

class AnimationController:
    """
    Manages all animations with delta-time updates.
    
    Handles:
    - Smooth agent movement interpolation
    - Path trail fading
    - Effect animations (pop, flash, ripple)
    """
    
    def __init__(self, theme: ThemeConfig):
        self.theme = theme
        self.agent_pos: Optional[Vector2] = None
        self.agent_target: Optional[Vector2] = None
        self.agent_direction: str = 'down'
        self.path_trail: List[Tuple[Vector2, float]] = []  # (pos, alpha)
        self.active_effects: List[Any] = []
        self._accumulated_time: float = 0.0
    
    def set_agent_position(self, row: int, col: int, immediate: bool = False):
        """
        Set the agent's target position.
        
        Args:
            row: Grid row
            col: Grid column
            immediate: If True, snap to position without animation
        """
        target = Vector2.from_grid(row, col)
        
        if self.agent_pos is None or immediate:
            self.agent_pos = target
            self.agent_target = target
        else:
            # Determine direction
            if self.agent_target:
                dx = target.x - self.agent_target.x
                dy = target.y - self.agent_target.y
                if abs(dx) > abs(dy):
                    self.agent_direction = 'right' if dx > 0 else 'left'
                elif dy != 0:
                    self.agent_direction = 'down' if dy > 0 else 'up'
            
            self.agent_target = target
            
            # Add to trail
            if self.agent_pos:
                self.path_trail.append((Vector2(self.agent_pos.x, self.agent_pos.y), 1.0))
    
    def update(self, dt: float):
        """
        Update all animations.
        
        Args:
            dt: Delta time in seconds since last update
        """
        self._accumulated_time += dt
        
        # Update agent position (lerp toward target)
        if self.agent_pos and self.agent_target:
            lerp_speed = self.theme.ANIMATION_SPEED * dt
            self.agent_pos = self.agent_pos.lerp(self.agent_target, min(1.0, lerp_speed))
            
            # Snap if very close
            if self.agent_pos.distance_to(self.agent_target) < 0.01:
                self.agent_pos = Vector2(self.agent_target.x, self.agent_target.y)
        
        # Fade path trail
        fade_rate = 0.5 * dt  # Fade over ~2 seconds
        new_trail = []
        for pos, alpha in self.path_trail:
            new_alpha = alpha - fade_rate
            if new_alpha > 0.05:
                new_trail.append((pos, new_alpha))
        self.path_trail = new_trail[-50:]  # Keep max 50 trail points
        
        # Update effects
        remaining_effects = []
        for effect in self.active_effects:
            if hasattr(effect, 'update'):
                effect.update(dt)
            if hasattr(effect, 'is_active') and effect.is_active():
                remaining_effects.append(effect)
        self.active_effects = remaining_effects
    
    def add_effect(self, effect: Any):
        """Add a visual effect to be updated and rendered."""
        self.active_effects.append(effect)
    
    def get_agent_render_position(self, tile_size: int) -> Tuple[int, int]:
        """Get the agent's current render position in pixels."""
        if self.agent_pos is None:
            return (0, 0)
        return (int(self.agent_pos.x * tile_size), int(self.agent_pos.y * tile_size))
    
    def clear(self):
        """Clear all animation state."""
        self.agent_pos = None
        self.agent_target = None
        self.path_trail.clear()
        self.active_effects.clear()


# ==========================================
# MAIN RENDERER
# ==========================================

class ZeldaRenderer:
    """
    Main rendering orchestrator for Zelda dungeon visualization.
    
    Coordinates:
    - Tile rendering (procedural or sprite-based)
    - Agent rendering with smooth movement
    - Path and heatmap overlays
    - HUD rendering
    
    Usage:
        renderer = ZeldaRenderer(tile_size=32)
        renderer.set_agent_position(start_row, start_col, immediate=True)
        
        # In game loop:
        renderer.update(delta_time)
        renderer.render_map(screen, grid, camera_offset)
        renderer.render_agent(screen, camera_offset)
    """
    
    def __init__(self, tile_size: int = 32, assets_dir: str = None):
        """
        Initialize the renderer.
        
        Args:
            tile_size: Size of each tile in pixels
            assets_dir: Optional path to sprite assets
        """
        if not PYGAME_AVAILABLE:
            raise RuntimeError("Pygame is required for ZeldaRenderer")
        
        self.tile_size = tile_size
        self.theme = ThemeConfig()
        self.theme.TILE_SIZE = tile_size
        
        self.sprite_manager = SpriteManager(self.theme, assets_dir)
        self.tile_renderer = self.sprite_manager  # Alias for backward compatibility
        self.animations = AnimationController(self.theme)
        
        # Search visualization
        self.search_heatmap: Dict[Tuple[int, int], int] = {}
        self.show_heatmap: bool = False
        self.solution_path: List[Tuple[int, int]] = []
        self.path_progress: int = 0
    
    @property
    def agent_visual_pos(self) -> Optional[Vector2]:
        """Get the agent's current visual position (for smooth animation)."""
        return self.animations.agent_pos
    
    def set_tile_size(self, tile_size: int):
        """Change the tile size (clears caches)."""
        self.tile_size = tile_size
        self.theme.TILE_SIZE = tile_size
        self.sprite_manager.clear_cache()
    
    def set_agent_position(self, row: int, col: int, immediate: bool = False):
        """Set the agent's grid position."""
        self.animations.set_agent_position(row, col, immediate)
    
    def set_solution_path(self, path: List[Tuple[int, int]]):
        """Set the solution path for visualization."""
        self.solution_path = path
        self.path_progress = 0
    
    def advance_path(self):
        """Advance the path progress by one step."""
        if self.path_progress < len(self.solution_path):
            self.path_progress += 1
    
    def set_search_heatmap(self, heatmap: Dict[Tuple[int, int], int]):
        """Set the search heatmap (position -> visit count)."""
        self.search_heatmap = heatmap
    
    def toggle_heatmap(self):
        """Toggle heatmap visibility."""
        self.show_heatmap = not self.show_heatmap
    
    def update(self, dt: float):
        """
        Update all animations.
        
        Args:
            dt: Delta time in seconds since last frame
        """
        self.animations.update(dt)
    
    def render_map(self, surface: Surface, grid: 'np.ndarray', 
                   camera_offset: Tuple[int, int] = (0, 0),
                   viewport: Tuple[int, int] = None):
        """
        Render the dungeon map.
        
        Args:
            surface: Pygame surface to render to
            grid: 2D numpy array of tile IDs
            camera_offset: (x, y) offset for camera panning
            viewport: (width, height) of visible area, or None for full surface
        """
        if viewport is None:
            viewport = surface.get_size()
        
        view_w, view_h = viewport
        cam_x, cam_y = camera_offset
        
        # Calculate visible tile range
        start_col = max(0, cam_x // self.tile_size)
        start_row = max(0, cam_y // self.tile_size)
        end_col = min(grid.shape[1], start_col + (view_w // self.tile_size) + 2)
        end_row = min(grid.shape[0], start_row + (view_h // self.tile_size) + 2)
        
        # Render visible tiles
        for row in range(start_row, end_row):
            for col in range(start_col, end_col):
                tile_id = int(grid[row, col])
                tile_surface = self.sprite_manager.get_tile(tile_id, self.tile_size)
                
                screen_x = col * self.tile_size - cam_x
                screen_y = row * self.tile_size - cam_y
                
                surface.blit(tile_surface, (screen_x, screen_y))
    
    def render_heatmap(self, surface: Surface, camera_offset: Tuple[int, int] = (0, 0)):
        """
        Render A* search density as a color gradient overlay.
        
        Blue (cold) = few visits, Red (hot) = many visits
        """
        if not self.show_heatmap or not self.search_heatmap:
            return
        
        max_visits = max(self.search_heatmap.values()) if self.search_heatmap else 1
        cam_x, cam_y = camera_offset
        
        for (row, col), visits in self.search_heatmap.items():
            t = visits / max_visits  # 0 to 1
            color = self._heatmap_color(t)
            
            # Create semi-transparent overlay
            overlay = pygame.Surface((self.tile_size, self.tile_size), pygame.SRCALPHA)
            overlay.fill((*color, 80))
            
            screen_x = col * self.tile_size - cam_x
            screen_y = row * self.tile_size - cam_y
            surface.blit(overlay, (screen_x, screen_y))
    
    def _heatmap_color(self, t: float) -> Tuple[int, int, int]:
        """Interpolate heatmap color from cold (blue) to hot (red)."""
        cold = self.theme.get_color('heatmap_cold')
        mid = self.theme.get_color('heatmap_mid')
        hot = self.theme.get_color('heatmap_hot')
        
        if t < 0.5:
            # Blue to Purple
            t2 = t * 2
            return (
                int(cold[0] + (mid[0] - cold[0]) * t2),
                int(cold[1] + (mid[1] - cold[1]) * t2),
                int(cold[2] + (mid[2] - cold[2]) * t2),
            )
        else:
            # Purple to Red
            t2 = (t - 0.5) * 2
            return (
                int(mid[0] + (hot[0] - mid[0]) * t2),
                int(mid[1] + (hot[1] - mid[1]) * t2),
                int(mid[2] + (hot[2] - mid[2]) * t2),
            )
    
    def render_path(self, surface: Surface, camera_offset: Tuple[int, int] = (0, 0)):
        """
        Render the solution path with a glowing trail effect.
        """
        if not self.solution_path:
            return
        
        cam_x, cam_y = camera_offset
        
        # Draw the full path as a faint preview
        if len(self.solution_path) > 1:
            points = []
            for row, col in self.solution_path:
                x = col * self.tile_size + self.tile_size // 2 - cam_x
                y = row * self.tile_size + self.tile_size // 2 - cam_y
                points.append((x, y))
            
            preview_color = self.theme.get_color('path_visited')[:3]
            pygame.draw.lines(surface, preview_color, False, points, 1)
        
        # Draw visited portion with glow
        for i in range(min(self.path_progress, len(self.solution_path))):
            row, col = self.solution_path[i]
            
            # Calculate alpha based on recency
            recency = self.path_progress - i
            alpha = max(60, 180 - recency * 3)
            
            # Create glowing overlay
            glow = pygame.Surface((self.tile_size, self.tile_size), pygame.SRCALPHA)
            glow_color = (*self.theme.get_color('path_glow')[:3], alpha)
            pygame.draw.rect(glow, glow_color, (2, 2, self.tile_size - 4, self.tile_size - 4),
                           border_radius=4)
            
            screen_x = col * self.tile_size - cam_x
            screen_y = row * self.tile_size - cam_y
            surface.blit(glow, (screen_x, screen_y))
    
    def render_trail(self, surface: Surface, camera_offset: Tuple[int, int] = (0, 0)):
        """Render the smooth movement trail."""
        cam_x, cam_y = camera_offset
        
        for pos, alpha in self.animations.path_trail:
            overlay = pygame.Surface((self.tile_size - 4, self.tile_size - 4), pygame.SRCALPHA)
            trail_color = self.theme.get_color('path_trail')
            overlay.fill((*trail_color[:3], int(alpha * trail_color[3] if len(trail_color) > 3 else alpha * 100)))
            
            screen_x = int(pos.x * self.tile_size) - cam_x + 2
            screen_y = int(pos.y * self.tile_size) - cam_y + 2
            surface.blit(overlay, (screen_x, screen_y))
    
    def render_agent(self, surface: Surface, camera_offset: Tuple[int, int] = (0, 0)):
        """Render Link at the interpolated position."""
        if self.animations.agent_pos is None:
            return
        
        cam_x, cam_y = camera_offset
        
        # Get Link sprite
        link_sprite = self.sprite_manager.get_link_sprite(
            self.tile_size, 
            self.animations.agent_direction
        )
        
        # Calculate screen position (with smooth interpolation)
        screen_x = int(self.animations.agent_pos.x * self.tile_size) - cam_x + 2
        screen_y = int(self.animations.agent_pos.y * self.tile_size) - cam_y + 2
        
        surface.blit(link_sprite, (screen_x, screen_y))
    
    def render_effects(self, surface: Surface, camera_offset: Tuple[int, int] = (0, 0)):
        """Render all active visual effects."""
        cam_x, cam_y = camera_offset
        
        for effect in self.animations.active_effects:
            if hasattr(effect, 'render'):
                effect.render(surface, self.tile_size, (cam_x, cam_y))


# ==========================================
# CONVENIENCE FUNCTIONS
# ==========================================

def create_renderer(tile_size: int = 32, assets_dir: str = None) -> ZeldaRenderer:
    """
    Factory function to create a configured renderer.
    
    Args:
        tile_size: Tile size in pixels
        assets_dir: Optional path to sprite assets
        
    Returns:
        Configured ZeldaRenderer instance
    """
    return ZeldaRenderer(tile_size, assets_dir)


__all__ = [
    'ThemeConfig',
    'Vector2', 
    'ProceduralTileRenderer',
    'SpriteManager',
    'AnimationController',
    'ZeldaRenderer',
    'create_renderer',
]
