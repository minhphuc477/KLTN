"""
KLTN Visualization - Modern HUD System
======================================

Professional HUD components with semi-transparency and rounded corners.

Components:
- HUDPanel: Base semi-transparent panel with rounded corners
- IconDisplay: Display icons with counts (Keys x3, Bombs x2)
- StatusBar: Bottom status bar with position, steps, state
- InventoryPanel: Shows collected items
- ModernHUD: Complete HUD layout manager

All components use delta-time for animations and smooth transitions.

"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

try:
    import pygame
    from pygame import Surface
    from pygame.font import Font
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


# ==========================================
# HUD THEME
# ==========================================

@dataclass
class HUDTheme:
    """Theme configuration for HUD elements."""
    
    # Panel styling
    panel_color: Tuple[int, int, int, int] = (40, 40, 60, 200)
    panel_border_color: Tuple[int, int, int] = (70, 70, 100)
    panel_border_width: int = 2
    panel_corner_radius: int = 8
    
    # Text colors
    text_primary: Tuple[int, int, int] = (220, 220, 220)
    text_secondary: Tuple[int, int, int] = (160, 160, 180)
    text_highlight: Tuple[int, int, int] = (255, 220, 100)
    text_success: Tuple[int, int, int] = (100, 255, 100)
    text_warning: Tuple[int, int, int] = (255, 200, 50)
    text_error: Tuple[int, int, int] = (255, 80, 80)
    
    # Font sizes
    font_size_large: int = 20
    font_size_medium: int = 14
    font_size_small: int = 12
    
    # Spacing
    padding: int = 10
    line_height: int = 18
    icon_size: int = 16


# ==========================================
# HUD PANEL
# ==========================================

class HUDPanel:
    """
    Semi-transparent rounded panel for HUD elements.
    
    Provides a consistent visual base for all HUD components.
    """
    
    def __init__(self, x: int, y: int, width: int, height: int, 
                 theme: HUDTheme = None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.theme = theme or HUDTheme()
        self._surface: Optional[Surface] = None
        self._needs_rebuild = True
    
    def set_position(self, x: int, y: int):
        """Update panel position."""
        self.x = x
        self.y = y
    
    def set_size(self, width: int, height: int):
        """Update panel size."""
        if width != self.width or height != self.height:
            self.width = width
            self.height = height
            self._needs_rebuild = True
    
    def get_rect(self) -> Tuple[int, int, int, int]:
        """Get panel rectangle (x, y, width, height)."""
        return (self.x, self.y, self.width, self.height)
    
    def _rebuild_surface(self):
        """Rebuild the panel surface."""
        if not PYGAME_AVAILABLE:
            return
        
        self._surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        # Draw rounded rectangle background
        rect = pygame.Rect(0, 0, self.width, self.height)
        pygame.draw.rect(self._surface, self.theme.panel_color, rect,
                        border_radius=self.theme.panel_corner_radius)
        
        # Draw border
        if self.theme.panel_border_width > 0:
            pygame.draw.rect(self._surface, self.theme.panel_border_color, rect,
                           self.theme.panel_border_width,
                           border_radius=self.theme.panel_corner_radius)
        
        self._needs_rebuild = False
    
    def render(self, surface: Surface):
        """Render the panel to a surface."""
        if not PYGAME_AVAILABLE:
            return
        
        if self._needs_rebuild or self._surface is None:
            self._rebuild_surface()
        
        surface.blit(self._surface, (self.x, self.y))
    
    def render_with_content(self, surface: Surface, content_surface: Surface):
        """Render panel with content overlaid."""
        self.render(surface)
        surface.blit(content_surface, (self.x, self.y))


# ==========================================
# ICON DISPLAY
# ==========================================

class IconDisplay:
    """
    Display icons with counts (Keys x3, Bombs x2, etc).
    
    Renders a small icon followed by a count number.
    """
    
    def __init__(self, theme: HUDTheme = None):
        self.theme = theme or HUDTheme()
        self._icon_cache: Dict[str, Surface] = {}
        self._font: Optional[Font] = None
    
    def _get_font(self) -> Font:
        """Get or create the display font."""
        if not PYGAME_AVAILABLE:
            raise RuntimeError("Pygame required")
        
        if self._font is None:
            self._font = pygame.font.SysFont('Arial', self.theme.font_size_small, bold=True)
        return self._font
    
    def _get_icon(self, icon_type: str) -> Surface:
        """Get or create an icon surface."""
        if icon_type in self._icon_cache:
            return self._icon_cache[icon_type]
        
        size = self.theme.icon_size
        surface = pygame.Surface((size, size), pygame.SRCALPHA)
        
        if icon_type == 'key':
            # Yellow key icon
            color = (255, 220, 100)
            pygame.draw.circle(surface, color, (size // 2, size // 3), size // 4)
            pygame.draw.rect(surface, color, (size // 2 - 2, size // 3, 4, size // 2))
        
        elif icon_type == 'bomb':
            # Black bomb icon
            color = (60, 60, 60)
            pygame.draw.circle(surface, color, (size // 2, size // 2 + 2), size // 3)
            pygame.draw.rect(surface, (100, 100, 100), (size // 2 - 1, 2, 2, size // 4))
            pygame.draw.circle(surface, (255, 150, 50), (size // 2, 2), 2)
        
        elif icon_type == 'boss_key':
            # Ornate boss key
            color = (200, 100, 50)
            pygame.draw.circle(surface, color, (size // 2, size // 4), size // 4)
            pygame.draw.circle(surface, (150, 70, 30), (size // 2, size // 4), size // 6)
            pygame.draw.rect(surface, color, (size // 2 - 2, size // 4, 4, size // 2))
        
        elif icon_type == 'heart':
            # Red heart
            color = (255, 80, 80)
            # Simple heart shape using circles and triangle
            pygame.draw.circle(surface, color, (size // 3, size // 3), size // 4)
            pygame.draw.circle(surface, color, (size * 2 // 3, size // 3), size // 4)
            points = [(size // 6, size // 3), (size * 5 // 6, size // 3), 
                     (size // 2, size - 2)]
            pygame.draw.polygon(surface, color, points)
        
        elif icon_type == 'triforce':
            # Golden triforce
            color = (255, 215, 0)
            points = [(size // 2, 2), (2, size - 2), (size - 2, size - 2)]
            pygame.draw.polygon(surface, color, points)
            pygame.draw.polygon(surface, (200, 170, 0), points, 1)
        
        else:
            # Generic item (gray square)
            pygame.draw.rect(surface, (180, 180, 180), (2, 2, size - 4, size - 4))
        
        self._icon_cache[icon_type] = surface
        return surface
    
    def render(self, surface: Surface, x: int, y: int, 
               icon_type: str, count: int, 
               highlight: bool = False) -> int:
        """
        Render an icon with count.
        
        Args:
            surface: Surface to render to
            x, y: Position
            icon_type: Type of icon ('key', 'bomb', 'boss_key', 'heart')
            count: Number to display
            highlight: Whether to highlight the text
            
        Returns:
            Width of rendered content
        """
        if not PYGAME_AVAILABLE:
            return 0
        
        # Draw icon
        icon = self._get_icon(icon_type)
        surface.blit(icon, (x, y))
        
        # Draw count
        font = self._get_font()
        color = self.theme.text_highlight if highlight else self.theme.text_primary
        text = f"×{count}"
        text_surface = font.render(text, True, color)
        surface.blit(text_surface, (x + self.theme.icon_size + 4, y + 2))
        
        return self.theme.icon_size + 4 + text_surface.get_width()


# ==========================================
# STATUS BAR
# ==========================================

class StatusBar:
    """
    Bottom status bar with position, steps, and state information.
    """
    
    def __init__(self, theme: HUDTheme = None):
        self.theme = theme or HUDTheme()
        self._font: Optional[Font] = None
        self._message: str = ""
        self._message_color: Tuple[int, int, int] = (220, 220, 220)
        self._position: Tuple[int, int] = (0, 0)
        self._steps: int = 0
    
    def _get_font(self) -> Font:
        """Get or create the status font."""
        if not PYGAME_AVAILABLE:
            raise RuntimeError("Pygame required")
        
        if self._font is None:
            self._font = pygame.font.SysFont('Arial', self.theme.font_size_medium)
        return self._font
    
    def set_message(self, message: str, 
                    color: Tuple[int, int, int] = None):
        """Set the status message."""
        self._message = message
        self._message_color = color or self.theme.text_primary
    
    def set_position(self, row: int, col: int):
        """Set the displayed position."""
        self._position = (row, col)
    
    def set_steps(self, steps: int):
        """Set the step count."""
        self._steps = steps
    
    def render(self, surface: Surface, x: int, y: int, width: int):
        """Render the status bar."""
        if not PYGAME_AVAILABLE:
            return
        
        font = self._get_font()
        
        # Message (left)
        if self._message:
            msg_surface = font.render(self._message, True, self._message_color)
            surface.blit(msg_surface, (x, y))
        
        # Position (center)
        pos_text = f"Pos: ({self._position[0]}, {self._position[1]})"
        pos_surface = font.render(pos_text, True, self.theme.text_secondary)
        center_x = x + width // 2 - pos_surface.get_width() // 2
        surface.blit(pos_surface, (center_x, y))
        
        # Steps (right)
        steps_text = f"Steps: {self._steps}"
        steps_surface = font.render(steps_text, True, self.theme.text_secondary)
        right_x = x + width - steps_surface.get_width()
        surface.blit(steps_surface, (right_x, y))


# ==========================================
# INVENTORY PANEL
# ==========================================

class InventoryPanel:
    """
    Panel showing collected items and their counts.
    """
    
    def __init__(self, theme: HUDTheme = None):
        self.theme = theme or HUDTheme()
        self.icon_display = IconDisplay(theme)
        self._font: Optional[Font] = None
        
        # Inventory state
        self.keys: int = 0
        self.bombs: int = 0
        self.has_boss_key: bool = False
        self.has_map: bool = False
        self.has_compass: bool = False
    
    def _get_font(self) -> Font:
        """Get or create the title font."""
        if not PYGAME_AVAILABLE:
            raise RuntimeError("Pygame required")
        
        if self._font is None:
            self._font = pygame.font.SysFont('Arial', self.theme.font_size_medium, bold=True)
        return self._font
    
    def update_inventory(self, keys: int = 0, bombs: int = 0,
                        has_boss_key: bool = False,
                        has_map: bool = False,
                        has_compass: bool = False):
        """Update inventory state."""
        self.keys = keys
        self.bombs = bombs
        self.has_boss_key = has_boss_key
        self.has_map = has_map
        self.has_compass = has_compass
    
    def render(self, surface: Surface, x: int, y: int) -> int:
        """
        Render the inventory panel.
        
        Returns:
            Height of rendered content
        """
        if not PYGAME_AVAILABLE:
            return 0
        
        font = self._get_font()
        current_y = y
        
        # Title
        title = font.render("Inventory", True, self.theme.text_highlight)
        surface.blit(title, (x, current_y))
        current_y += self.theme.line_height + 4
        
        # Keys
        self.icon_display.render(surface, x + 4, current_y, 'key', self.keys,
                                highlight=self.keys > 0)
        current_y += self.theme.line_height
        
        # Bombs
        if self.bombs > 0:
            self.icon_display.render(surface, x + 4, current_y, 'bomb', self.bombs)
            current_y += self.theme.line_height
        
        # Boss key
        if self.has_boss_key:
            small_font = pygame.font.SysFont('Arial', self.theme.font_size_small)
            bk_text = small_font.render("Boss Key ✓", True, self.theme.text_success)
            surface.blit(bk_text, (x + 4, current_y))
            current_y += self.theme.line_height
        
        return current_y - y


# ==========================================
# PATH ANALYSIS PANEL
# ==========================================

class PathAnalysisPanel:
    """
    Panel showing solver/path analysis results.
    """
    
    def __init__(self, theme: HUDTheme = None):
        self.theme = theme or HUDTheme()
        self._font: Optional[Font] = None
        self._small_font: Optional[Font] = None
        
        # Analysis state
        self.path_length: int = 0
        self.keys_found: int = 0
        self.keys_used: int = 0
        self.edge_types: Dict[str, int] = {}
        self.teleports: int = 0
    
    def _get_fonts(self) -> Tuple[Font, Font]:
        """Get fonts."""
        if not PYGAME_AVAILABLE:
            raise RuntimeError("Pygame required")
        
        if self._font is None:
            self._font = pygame.font.SysFont('Arial', self.theme.font_size_medium, bold=True)
            self._small_font = pygame.font.SysFont('Arial', self.theme.font_size_small)
        return self._font, self._small_font
    
    def update_analysis(self, path_length: int = 0,
                       keys_found: int = 0, keys_used: int = 0,
                       edge_types: Dict[str, int] = None,
                       teleports: int = 0):
        """Update path analysis data."""
        self.path_length = path_length
        self.keys_found = keys_found
        self.keys_used = keys_used
        self.edge_types = edge_types or {}
        self.teleports = teleports
    
    def render(self, surface: Surface, x: int, y: int) -> int:
        """
        Render the path analysis panel.
        
        Returns:
            Height of rendered content
        """
        if not PYGAME_AVAILABLE:
            return 0
        
        font, small_font = self._get_fonts()
        current_y = y
        
        # Title
        title = font.render("Path Analysis", True, (100, 200, 255))
        surface.blit(title, (x, current_y))
        current_y += self.theme.line_height + 4
        
        # Path length
        path_text = small_font.render(f"Length: {self.path_length} steps", True, 
                                      self.theme.text_primary)
        surface.blit(path_text, (x + 4, current_y))
        current_y += self.theme.line_height - 2
        
        # Keys info
        if self.keys_found > 0:
            key_color = self.theme.text_highlight if self.keys_used > 0 else self.theme.text_secondary
            key_text = small_font.render(f"Keys: {self.keys_found} found, {self.keys_used} used",
                                        True, key_color)
            surface.blit(key_text, (x + 4, current_y))
            current_y += self.theme.line_height - 2
        
        # Teleports
        if self.teleports > 0:
            tp_text = small_font.render(f"Teleports: {self.teleports}", True,
                                       (100, 200, 255))
            surface.blit(tp_text, (x + 4, current_y))
            current_y += self.theme.line_height - 2
        
        # Edge type breakdown
        if self.edge_types:
            edge_colors = {
                'open': self.theme.text_success,
                'key_locked': self.theme.text_highlight,
                'bombable': (255, 150, 50),
                'soft_locked': (180, 100, 255),
                'stair': (100, 200, 255),
            }
            
            for edge_type, count in self.edge_types.items():
                color = edge_colors.get(edge_type, self.theme.text_secondary)
                name = edge_type.replace('_', ' ').title()
                et_text = small_font.render(f"  {name}: {count}", True, color)
                surface.blit(et_text, (x + 4, current_y))
                current_y += self.theme.line_height - 4
        
        return current_y - y


# ==========================================
# CONTROLS PANEL
# ==========================================

class ControlsPanel:
    """
    Panel showing keyboard controls.
    """
    
    CONTROLS = [
        ("↑↓←→", "Move"),
        ("SPACE", "Auto-solve"),
        ("R", "Reset map"),
        ("N/P", "Next/Prev"),
        ("+/-", "Zoom"),
        ("H", "Heatmap"),
        ("F11", "Fullscreen"),
        ("ESC", "Quit"),
    ]
    
    def __init__(self, theme: HUDTheme = None):
        self.theme = theme or HUDTheme()
        self._fonts: Optional[Tuple[Font, Font]] = None
    
    def _get_fonts(self) -> Tuple[Font, Font]:
        """Get fonts."""
        if not PYGAME_AVAILABLE:
            raise RuntimeError("Pygame required")
        
        if self._fonts is None:
            title_font = pygame.font.SysFont('Arial', self.theme.font_size_medium, bold=True)
            text_font = pygame.font.SysFont('Arial', self.theme.font_size_small)
            self._fonts = (title_font, text_font)
        return self._fonts
    
    def render(self, surface: Surface, x: int, y: int) -> int:
        """
        Render the controls panel.
        
        Returns:
            Height of rendered content
        """
        if not PYGAME_AVAILABLE:
            return 0
        
        title_font, text_font = self._get_fonts()
        current_y = y
        
        # Title
        title = title_font.render("Controls", True, self.theme.text_success)
        surface.blit(title, (x, current_y))
        current_y += self.theme.line_height + 2
        
        # Control list
        for key, action in self.CONTROLS:
            # Key
            key_surface = text_font.render(key, True, self.theme.text_highlight)
            surface.blit(key_surface, (x + 4, current_y))
            
            # Action
            action_surface = text_font.render(action, True, self.theme.text_secondary)
            surface.blit(action_surface, (x + 60, current_y))
            
            current_y += self.theme.line_height - 4
        
        return current_y - y


# ==========================================
# MODERN HUD
# ==========================================

class ModernHUD:
    """
    Complete HUD layout manager combining all components.
    
    Manages:
    - Title panel
    - Inventory panel
    - Status bar
    - Path analysis panel
    - Controls panel
    
    Usage:
        hud = ModernHUD()
        hud.update_game_state(keys=3, position=(5, 10), steps=42)
        hud.render(screen, screen_width, screen_height, sidebar_width=220)
    """
    
    def __init__(self, theme: HUDTheme = None):
        self.theme = theme or HUDTheme()
        
        # Components
        self.inventory = InventoryPanel(theme)
        self.status_bar = StatusBar(theme)
        self.path_analysis = PathAnalysisPanel(theme)
        self.controls = ControlsPanel(theme)
        
        # State
        self.title: str = "ZAVE"
        self.subtitle: str = ""
        self.map_info: str = ""
        self.zoom_level: int = 32
        self.victory: bool = False
        
        # Fonts
        self._title_font: Optional[Font] = None
        self._subtitle_font: Optional[Font] = None
        self._small_font: Optional[Font] = None
    
    def _get_fonts(self) -> Tuple[Font, Font, Font]:
        """Get fonts."""
        if not PYGAME_AVAILABLE:
            raise RuntimeError("Pygame required")
        
        if self._title_font is None:
            self._title_font = pygame.font.SysFont('Arial', 20, bold=True)
            self._subtitle_font = pygame.font.SysFont('Arial', 14, bold=True)
            self._small_font = pygame.font.SysFont('Arial', 12)
        return self._title_font, self._subtitle_font, self._small_font
    
    def set_title(self, title: str, subtitle: str = ""):
        """Set the HUD title and subtitle."""
        self.title = title
        self.subtitle = subtitle
    
    def set_map_info(self, info: str):
        """Set map information string."""
        self.map_info = info
    
    def set_zoom_level(self, zoom: int):
        """Set displayed zoom level."""
        self.zoom_level = zoom
    
    def set_victory(self, victory: bool):
        """Set victory state."""
        self.victory = victory
    
    def update_game_state(self, keys: int = 0, bombs: int = 0,
                         has_boss_key: bool = False,
                         position: Tuple[int, int] = (0, 0),
                         steps: int = 0,
                         message: str = "",
                         message_color: Tuple[int, int, int] = None):
        """Update game state for all components."""
        self.inventory.update_inventory(keys, bombs, has_boss_key)
        self.status_bar.set_position(*position)
        self.status_bar.set_steps(steps)
        self.status_bar.set_message(message, message_color)
    
    def update_solver_result(self, result: Dict[str, Any] = None):
        """Update with solver result data."""
        if result:
            edge_types = {}
            for et in result.get('edge_types', []):
                edge_types[et] = edge_types.get(et, 0) + 1
            
            self.path_analysis.update_analysis(
                path_length=result.get('path_length', 0),
                keys_found=result.get('keys_available', 0),
                keys_used=result.get('keys_used', 0),
                edge_types=edge_types,
                teleports=result.get('teleports', 0)
            )
    
    def render_sidebar(self, surface: Surface, x: int, width: int, height: int):
        """
        Render the sidebar HUD.
        
        Args:
            surface: Surface to render to
            x: X position of sidebar
            width: Width of sidebar
            height: Height of sidebar
        """
        if not PYGAME_AVAILABLE:
            return
        
        title_font, subtitle_font, small_font = self._get_fonts()
        
        # Background
        sidebar_panel = HUDPanel(x, 0, width, height, self.theme)
        sidebar_panel.theme.panel_corner_radius = 0  # Square sidebar
        sidebar_panel.render(surface)
        
        current_y = self.theme.padding
        
        # Title
        title_surface = title_font.render(self.title, True, (100, 200, 255))
        surface.blit(title_surface, (x + self.theme.padding, current_y))
        current_y += 24
        
        # Subtitle (dungeon name)
        if self.subtitle:
            sub_surface = subtitle_font.render(self.subtitle, True, 
                                               self.theme.text_highlight)
            surface.blit(sub_surface, (x + self.theme.padding, current_y))
            current_y += 18
        
        # Map info
        if self.map_info:
            info_surface = small_font.render(self.map_info, True,
                                            self.theme.text_secondary)
            surface.blit(info_surface, (x + self.theme.padding, current_y))
            current_y += 16
        
        # Divider
        current_y += 4
        pygame.draw.line(surface, self.theme.panel_border_color,
                        (x + self.theme.padding, current_y),
                        (x + width - self.theme.padding, current_y))
        current_y += 8
        
        # Inventory
        inv_height = self.inventory.render(surface, x + self.theme.padding, current_y)
        current_y += inv_height + 8
        
        # Path analysis (if available)
        if self.path_analysis.path_length > 0:
            pygame.draw.line(surface, self.theme.panel_border_color,
                            (x + self.theme.padding, current_y),
                            (x + width - self.theme.padding, current_y))
            current_y += 8
            
            analysis_height = self.path_analysis.render(surface, 
                                                        x + self.theme.padding, 
                                                        current_y)
            current_y += analysis_height + 8
        
        # Zoom info
        zoom_surface = small_font.render(f"Zoom: {self.zoom_level}px", True,
                                        self.theme.text_secondary)
        surface.blit(zoom_surface, (x + self.theme.padding, current_y))
        current_y += 20
        
        # Divider
        pygame.draw.line(surface, self.theme.panel_border_color,
                        (x + self.theme.padding, current_y),
                        (x + width - self.theme.padding, current_y))
        current_y += 8
        
        # Controls
        self.controls.render(surface, x + self.theme.padding, current_y)
    
    def render_bottom_bar(self, surface: Surface, width: int, height: int,
                          bar_height: int = 100):
        """
        Render the bottom status bar.
        
        Args:
            surface: Surface to render to
            width: Width of the bar
            height: Total screen height (bar_y = height - bar_height)
            bar_height: Height of the status bar
        """
        if not PYGAME_AVAILABLE:
            return
        
        y = height - bar_height
        
        # Background panel
        panel = HUDPanel(0, y, width, bar_height, self.theme)
        panel.theme.panel_corner_radius = 0
        panel.render(surface)
        
        # Border line at top
        pygame.draw.line(surface, self.theme.panel_border_color,
                        (0, y), (width, y), 2)
        
        # Status bar
        self.status_bar.render(surface, 
                               self.theme.padding, 
                               y + self.theme.padding,
                               width - self.theme.padding * 2)
        
        # Victory message
        if self.victory:
            title_font = self._get_fonts()[0]
            victory_text = "★ VICTORY! ★"
            victory_surface = title_font.render(victory_text, True, (255, 215, 0))
            surface.blit(victory_surface, 
                        (self.theme.padding, y + 40))


# ==========================================
# EXPORTS
# ==========================================

__all__ = [
    'HUDTheme',
    'HUDPanel',
    'IconDisplay',
    'StatusBar',
    'InventoryPanel',
    'PathAnalysisPanel',
    'ControlsPanel',
    'ModernHUD',
]
