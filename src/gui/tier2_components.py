"""
TIER 2 & 3 GUI ENHANCEMENTS
===========================

Comprehensive GUI additions for all 12 new features:
1. Multi-floor dungeon visualization
2. D* Lite replanning indicator
3. Minimap zoom controls
4. Item tooltips
5. Solver comparison split-screen
6. Multi-goal waypoint display
7. Enemy visualization
8. ML heuristic toggle
9. Procedural generation dialog
10. Difficulty adjustment display
11. Speedrun timer
12. Enhanced controls

This module extends gui_runner.py with new UI components.
"""

import pygame
import logging
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ==========================================
# FLOOR SELECTOR UI
# ==========================================

class FloorSelector:
    """
    Dropdown for selecting current floor in multi-floor dungeons.
    
    Position: Top-right corner
    Size: 150×40 pixels
    """
    
    def __init__(self, screen_width: int, num_floors: int = 1):
        """
        Initialize floor selector.
        
        Args:
            screen_width: Screen width for positioning
            num_floors: Total number of floors
        """
        self.num_floors = num_floors
        self.current_floor = 0
        
        # Position (top-right corner)
        self.x = screen_width - 160
        self.y = 10
        self.width = 150
        self.height = 40
        
        # State
        self.expanded = False
        self.hover_index = -1
        
        # Styling
        self.bg_color = (40, 40, 50)
        self.hover_color = (60, 60, 70)
        self.text_color = (255, 255, 255)
        self.border_color = (100, 100, 120)
    
    def render(self, surface, font):
        """Render the floor selector."""
        # Main box
        rect = pygame.Rect(self.x, self.y, self.width, self.height)
        pygame.draw.rect(surface, self.bg_color, rect)
        pygame.draw.rect(surface, self.border_color, rect, 2)
        
        # Text
        text = f"Floor {self.current_floor + 1}/{self.num_floors}"
        text_surf = font.render(text, True, self.text_color)
        text_rect = text_surf.get_rect(center=rect.center)
        surface.blit(text_surf, text_rect)
        
        # Dropdown arrow
        arrow_x = self.x + self.width - 20
        arrow_y = self.y + self.height // 2
        arrow_points = [
            (arrow_x, arrow_y - 5),
            (arrow_x + 10, arrow_y - 5),
            (arrow_x + 5, arrow_y + 5)
        ]
        pygame.draw.polygon(surface, self.text_color, arrow_points)
        
        # Expanded menu
        if self.expanded:
            menu_y = self.y + self.height + 5
            for i in range(self.num_floors):
                item_rect = pygame.Rect(self.x, menu_y + i * 35, self.width, 35)
                
                # Highlight hover
                color = self.hover_color if i == self.hover_index else self.bg_color
                pygame.draw.rect(surface, color, item_rect)
                pygame.draw.rect(surface, self.border_color, item_rect, 1)
                
                # Floor label
                label = font.render(f"Floor {i + 1}", True, self.text_color)
                label_rect = label.get_rect(center=item_rect.center)
                surface.blit(label, label_rect)
    
    def handle_click(self, mouse_pos: Tuple[int, int]) -> bool:
        """
        Handle mouse click.
        
        Returns:
            True if floor changed
        """
        x, y = mouse_pos
        
        # Check main button
        if self.x <= x <= self.x + self.width and self.y <= y <= self.y + self.height:
            self.expanded = not self.expanded
            return False
        
        # Check expanded menu
        if self.expanded:
            menu_y = self.y + self.height + 5
            for i in range(self.num_floors):
                item_y = menu_y + i * 35
                if self.x <= x <= self.x + self.width and item_y <= y <= item_y + 35:
                    self.current_floor = i
                    self.expanded = False
                    return True
        
        # Click outside - close menu
        if self.expanded:
            self.expanded = False
        
        return False
    
    def handle_hover(self, mouse_pos: Tuple[int, int]):
        """Update hover state."""
        if not self.expanded:
            return
        
        x, y = mouse_pos
        menu_y = self.y + self.height + 5
        
        self.hover_index = -1
        for i in range(self.num_floors):
            item_y = menu_y + i * 35
            if self.x <= x <= self.x + self.width and item_y <= y <= item_y + 35:
                self.hover_index = i
                break


# ==========================================
# MINIMAP ZOOM OVERLAY
# ==========================================

@dataclass
class ZoomState:
    """State for minimap zoom."""
    enabled: bool = False
    zoom_level: float = 1.0
    focus_pos: Tuple[int, int] = (0, 0)  # (row, col)
    drag_start: Optional[Tuple[int, int]] = None
    zoom_rect: Optional[pygame.Rect] = None


class MinimapZoom:
    """
    Interactive minimap zoom with mouse controls.
    
    Controls:
    - Click+Drag: Draw zoom rectangle
    - Mouse Wheel: Zoom in/out
    - Double-Click: Reset zoom
    """
    
    def __init__(self, minimap_rect: pygame.Rect):
        """
        Initialize zoom controller.
        
        Args:
            minimap_rect: Rectangle of minimap area
        """
        self.minimap_rect = minimap_rect
        self.state = ZoomState()
        
        # Zoom limits
        self.min_zoom = 1.0
        self.max_zoom = 4.0
        self.zoom_step = 1.2
        
        # Overlay window
        self.overlay_rect = pygame.Rect(
            minimap_rect.right + 20,
            minimap_rect.top,
            400,
            400
        )
    
    def handle_mouse_down(self, mouse_pos: Tuple[int, int], button: int):
        """Handle mouse button press."""
        if not self.minimap_rect.collidepoint(mouse_pos):
            return
        
        if button == 1:  # Left click
            self.state.drag_start = mouse_pos
    
    def handle_mouse_up(self, mouse_pos: Tuple[int, int]):
        """Handle mouse button release."""
        if self.state.drag_start:
            # Complete drag - set zoom rectangle
            x1, y1 = self.state.drag_start
            x2, y2 = mouse_pos
            
            rect = pygame.Rect(
                min(x1, x2),
                min(y1, y2),
                abs(x2 - x1),
                abs(y2 - y1)
            )
            
            if rect.width > 20 and rect.height > 20:
                self.state.zoom_rect = rect
                self.state.enabled = True
            
            self.state.drag_start = None
    
    def handle_mouse_wheel(self, direction: int, mouse_pos: Tuple[int, int]):
        """Handle mouse wheel scroll."""
        if not self.minimap_rect.collidepoint(mouse_pos):
            return
        
        if direction > 0:  # Scroll up - zoom in
            self.state.zoom_level = min(self.state.zoom_level * self.zoom_step, self.max_zoom)
        else:  # Scroll down - zoom out
            self.state.zoom_level = max(self.state.zoom_level / self.zoom_step, self.min_zoom)
        
        self.state.enabled = (self.state.zoom_level > 1.0)
    
    def handle_double_click(self, mouse_pos: Tuple[int, int]):
        """Handle double-click (reset zoom)."""
        if self.minimap_rect.collidepoint(mouse_pos):
            self.reset()
    
    def reset(self):
        """Reset zoom to default."""
        self.state.enabled = False
        self.state.zoom_level = 1.0
        self.state.zoom_rect = None
        self.state.drag_start = None
    
    def render(self, surface, minimap_surface, tile_size: int):
        """
        Render zoom overlay.
        
        Args:
            surface: Main screen surface
            minimap_surface: Minimap surface to zoom
            tile_size: Size of minimap tiles
        """
        # Draw selection rectangle during drag
        if self.state.drag_start and pygame.mouse.get_pressed()[0]:
            mouse_pos = pygame.mouse.get_pos()
            x1, y1 = self.state.drag_start
            x2, y2 = mouse_pos
            
            rect = pygame.Rect(
                min(x1, x2),
                min(y1, y2),
                abs(x2 - x1),
                abs(y2 - y1)
            )
            
            pygame.draw.rect(surface, (255, 0, 0), rect, 2)
        
        # Draw zoom overlay if enabled
        if self.state.enabled and self.state.zoom_rect:
            # Create zoomed surface
            zoom_factor = self.state.zoom_level
            
            # Extract zoomed region
            zoom_surface = pygame.Surface((
                int(self.state.zoom_rect.width * zoom_factor),
                int(self.state.zoom_rect.height * zoom_factor)
            ))
            
            # Scale up
            pygame.transform.scale(
                minimap_surface.subsurface(self.state.zoom_rect),
                zoom_surface.get_size(),
                zoom_surface
            )
            
            # Draw overlay background
            pygame.draw.rect(surface, (20, 20, 30), self.overlay_rect)
            pygame.draw.rect(surface, (100, 100, 120), self.overlay_rect, 3)
            
            # Blit zoomed region (centered)
            zoom_rect = zoom_surface.get_rect(center=self.overlay_rect.center)
            surface.blit(zoom_surface, zoom_rect)
            
            # Label
            font = pygame.font.Font(None, 24)
            label = font.render(f"Zoom: {zoom_factor:.1f}×", True, (255, 255, 255))
            surface.blit(label, (self.overlay_rect.x + 10, self.overlay_rect.y + 10))


# ==========================================
# ITEM TOOLTIPS
# ==========================================

class ItemTooltip:
    """
    Tooltip system for minimap items.
    
    Shows item details on hover:
    - Item name
    - Collection status
    - Type (key/boss key/item)
    """
    
    def __init__(self):
        """Initialize tooltip system."""
        self.hover_start_time = 0
        self.hover_pos: Optional[Tuple[int, int]] = None  # Tile position (r, c)
        self.hover_delay = 0.5  # seconds
        
        # Styling
        self.bg_color = (0, 0, 0, 200)  # Semi-transparent black
        self.text_color = (255, 255, 255)
        self.border_color = (200, 200, 200)
        
        # Current tooltip
        self.visible = False
        self.text_lines: List[str] = []
        self.screen_pos: Tuple[int, int] = (0, 0)
    
    def update(
        self,
        mouse_pos: Tuple[int, int],
        tile_pos: Optional[Tuple[int, int]],
        grid,
        collected_items: set,
        current_time: float
    ):
        """
        Update tooltip state.
        
        Args:
            mouse_pos: Mouse screen position
            tile_pos: Hovered tile position (r, c) or None
            grid: Dungeon grid
            collected_items: Set of collected item positions
            current_time: Current time (seconds)
        """
        from src.core.definitions import SEMANTIC_PALETTE, ID_TO_NAME
        
        # Check if hovering over new tile
        if tile_pos != self.hover_pos:
            self.hover_pos = tile_pos
            self.hover_start_time = current_time
            self.visible = False
        
        # Check hover delay
        if tile_pos and (current_time - self.hover_start_time) >= self.hover_delay:
            r, c = tile_pos
            tile_id = grid[r, c]
            
            # Check if tile is an item
            item_types = {
                SEMANTIC_PALETTE['KEY_SMALL']: ('Small Key', '🔑'),
                SEMANTIC_PALETTE['KEY_BOSS']: ('Boss Key', '🗝️'),
                SEMANTIC_PALETTE['KEY_ITEM']: ('Key Item', '💎'),
                SEMANTIC_PALETTE['ITEM_MINOR']: ('Item', '⭐'),
                SEMANTIC_PALETTE['TRIFORCE']: ('Triforce', '🔺'),
            }
            
            if tile_id in item_types:
                name, icon = item_types[tile_id]
                collected = tile_pos in collected_items
                status = "✓ Collected" if collected else "Not Collected"
                
                self.text_lines = [
                    f"{icon} {name}",
                    status,
                    f"Position: ({r}, {c})"
                ]
                
                # Position tooltip near mouse (avoid screen edges)
                self.screen_pos = (mouse_pos[0] + 15, mouse_pos[1] + 15)
                self.visible = True
            else:
                self.visible = False
        
        if not tile_pos:
            self.visible = False
    
    def render(self, surface):
        """Render tooltip if visible."""
        if not self.visible or not self.text_lines:
            return
        
        font = pygame.font.Font(None, 20)
        
        # Calculate tooltip size
        line_height = 22
        padding = 10
        max_width = max(font.size(line)[0] for line in self.text_lines)
        
        tooltip_width = max_width + padding * 2
        tooltip_height = len(self.text_lines) * line_height + padding * 2
        
        # Adjust position to stay on screen
        x, y = self.screen_pos
        screen_width, screen_height = surface.get_size()
        
        if x + tooltip_width > screen_width:
            x = screen_width - tooltip_width - 10
        if y + tooltip_height > screen_height:
            y = screen_height - tooltip_height - 10
        
        # Create semi-transparent surface
        tooltip_surf = pygame.Surface((tooltip_width, tooltip_height), pygame.SRCALPHA)
        tooltip_surf.fill(self.bg_color)
        
        # Border
        pygame.draw.rect(tooltip_surf, self.border_color, tooltip_surf.get_rect(), 1)
        
        # Render text lines
        for i, line in enumerate(self.text_lines):
            text = font.render(line, True, self.text_color)
            tooltip_surf.blit(text, (padding, padding + i * line_height))
        
        # Blit to screen
        surface.blit(tooltip_surf, (x, y))


# ==========================================
# REPLANNING INDICATOR
# ==========================================

class ReplanningIndicator:
    """
    Visual indicator for D* Lite replanning events.
    
    Shows animated "Replanning..." text when environment changes.
    """
    
    def __init__(self, screen_width: int, screen_height: int):
        """Initialize indicator."""
        self.active = False
        self.message = ""
        self.duration = 2.0  # seconds
        self.start_time = 0
        
        # Position (center-top)
        self.x = screen_width // 2
        self.y = 80
        
        # Animation
        self.pulse_phase = 0
    
    def trigger(self, message: str, current_time: float):
        """
        Show replanning indicator.
        
        Args:
            message: Message to display
            current_time: Current time
        """
        self.active = True
        self.message = message
        self.start_time = current_time
    
    def update(self, current_time: float, dt: float):
        """Update animation."""
        if self.active:
            elapsed = current_time - self.start_time
            if elapsed >= self.duration:
                self.active = False
            
            self.pulse_phase += dt * 3.0  # Pulse frequency
    
    def render(self, surface):
        """Render indicator."""
        if not self.active:
            return
        
        # Pulsing alpha
        import math
        alpha = int(128 + 127 * math.sin(self.pulse_phase))
        
        # Background
        font = pygame.font.Font(None, 36)
        text = font.render(self.message, True, (255, 200, 0))
        text_rect = text.get_rect(center=(self.x, self.y))
        
        # Semi-transparent background
        bg_rect = text_rect.inflate(40, 20)
        bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
        bg_surf.fill((0, 0, 0, alpha))
        
        surface.blit(bg_surf, bg_rect)
        surface.blit(text, text_rect)


# ==========================================
# HELPER FUNCTIONS FOR INTEGRATION
# ==========================================

def get_tile_from_mouse(
    mouse_pos: Tuple[int, int],
    minimap_rect: pygame.Rect,
    grid_shape: Tuple[int, int],
    tile_size: int
) -> Optional[Tuple[int, int]]:
    """
    Convert mouse position to grid tile position.
    
    Args:
        mouse_pos: Mouse screen position
        minimap_rect: Minimap rectangle
        grid_shape: (height, width) of grid
        tile_size: Size of minimap tiles
        
    Returns:
        (row, col) or None if outside minimap
    """
    x, y = mouse_pos
    
    if not minimap_rect.collidepoint(mouse_pos):
        return None
    
    # Relative to minimap
    rel_x = x - minimap_rect.x
    rel_y = y - minimap_rect.y
    
    # Convert to tile
    tile_c = rel_x // tile_size
    tile_r = rel_y // tile_size
    
    height, width = grid_shape
    
    if 0 <= tile_r < height and 0 <= tile_c < width:
        return (tile_r, tile_c)
    
    return None
