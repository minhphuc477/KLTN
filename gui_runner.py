"""
GUI Runner for ZAVE (Zelda AI Validation Environment)
====================================================

Interactive visual interface for validating Zelda dungeon maps.

Features:
- Real-time visualization of map and agent
- Manual play mode (arrow keys)
- Auto-solve mode (A* pathfinding)
- Map loading from processed data
- Smooth delta-time animations
- Heatmap overlay for A* search visualization
- Modern semi-transparent HUD

Controls:
- Arrow Keys: Move Link
- SPACE: Run A* solver (auto-solve)
- R: Reset map
- N: Next map (if multiple loaded)
- P: Previous map
- H: Toggle heatmap overlay
- ESC: Quit


"""

import sys
import os
import time
import math
import logging
import numpy as np
from typing import Tuple, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import simulation components

from simulation.validator import (
    ZeldaLogicEnv, 
    ZeldaValidator, 
    StateSpaceAStar,
    SanityChecker,
    create_test_map,
    SEMANTIC_PALETTE,
    Action,
    GameState
)

# Try to import Pygame
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    logger.warning("Pygame not installed. Run 'pip install pygame' for GUI support.")

# Try to import new visualization system
try:
    from src.visualization.renderer import ZeldaRenderer, ThemeConfig, Vector2
    from src.visualization.effects import (
        EffectManager, PopEffect, FlashEffect, RippleEffect,
        ItemCollectionEffect, ItemUsageEffect, ItemMarkerEffect
    )
    from src.visualization.hud import ModernHUD, HUDTheme
    from src.visualization.path_preview import PathPreviewDialog
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logger.warning("New visualization system not available, using fallback rendering.")

# Try to import GUI widgets
try:
    from src.gui.widgets import (
        CheckboxWidget, DropdownWidget, ButtonWidget,
        WidgetManager, WidgetTheme
    )
    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False
    logger.warning("GUI widgets not available, using keyboard controls only.")


class ToastNotification:
    """Floating toast message with auto-dismiss and fade animations."""
    def __init__(self, message: str, duration: float = 3.0, toast_type: str = 'info'):
        self.message = message
        self.duration = duration
        self.toast_type = toast_type  # 'info', 'success', 'error', 'warning'
        self.created_at = time.time()
        
        # Colors by type
        self.colors = {
            'info': (100, 200, 255),
            'success': (100, 255, 150),
            'error': (255, 100, 100),
            'warning': (255, 200, 100)
        }
    
    def is_expired(self) -> bool:
        """Check if toast should be removed."""
        return time.time() - self.created_at > self.duration
    
    def get_alpha(self) -> int:
        """Calculate current alpha for fade in/out animation."""
        elapsed = time.time() - self.created_at
        
        # Fade in (0.3s)
        if elapsed < 0.3:
            return int((elapsed / 0.3) * 240)
        
        # Fade out (last 0.5s)
        if elapsed > self.duration - 0.5:
            remaining = self.duration - elapsed
            return int((remaining / 0.5) * 240)
        
        # Hold (middle)
        return 240
    
    def render(self, surface: pygame.Surface, center_x: int, y: int):
        """Render toast notification at specified position."""
        alpha = self.get_alpha()
        font = pygame.font.Font(None, 20)
        text_surf = font.render(self.message, True, (255, 255, 255))
        
        padding = 15
        toast_w = text_surf.get_width() + padding * 2
        toast_h = text_surf.get_height() + padding * 2
        
        toast_surf = pygame.Surface((toast_w, toast_h), pygame.SRCALPHA)
        
        # Background with border
        bg_rect = pygame.Rect(0, 0, toast_w, toast_h)
        pygame.draw.rect(toast_surf, (50, 60, 80, alpha), bg_rect, border_radius=8)
        
        # Colored border by type
        border_color = (*self.colors[self.toast_type][:3], alpha)
        pygame.draw.rect(toast_surf, border_color, bg_rect, 2, border_radius=8)
        
        # Text with alpha
        text_with_alpha = text_surf.copy()
        text_with_alpha.set_alpha(alpha)
        toast_surf.blit(text_with_alpha, (padding, padding))
        
        # Render centered
        surface.blit(toast_surf, (center_x - toast_w // 2, y))


class ZeldaGUI:
    """
    Interactive GUI for Zelda dungeon validation.
    
    Features:
    - Resizable window (drag corners/edges)
    - Zoom in/out with +/- keys or mouse wheel
    - Pan with middle mouse or WASD when zoomed
    - Fullscreen toggle with F11
    - Smooth delta-time based animations
    - Heatmap overlay for A* search (toggle with H)
    """
    
    # Zoom levels available
    ZOOM_LEVELS = [16, 24, 32, 48, 64]
    DEFAULT_ZOOM_IDX = 2  # 32px default
    
    # Minimum window size
    MIN_WIDTH = 400
    MIN_HEIGHT = 300
    
    def __init__(self, maps: list = None, map_names: list = None):
        """
        Initialize GUI.
        
        Args:
            maps: List of semantic grids to visualize
            map_names: List of names for each map
        """
        if not PYGAME_AVAILABLE:
            raise ImportError("Pygame is required for GUI. Install with: pip install pygame")
        
        self.maps = maps if maps else [create_test_map()]
        self.map_names = map_names if map_names else [f"Map {i+1}" for i in range(len(self.maps))]
        self.current_map_idx = 0
        
        # Initialize Pygame
        try:
            pygame.init()
        except Exception as e:
            logger.exception("Failed to initialize Pygame")
            raise
        
        # Display settings
        self.zoom_idx = self.DEFAULT_ZOOM_IDX
        self.TILE_SIZE = self.ZOOM_LEVELS[self.zoom_idx]
        self.HUD_HEIGHT = 10  # Minimal bottom margin (status/message moved to sidebar)
        self.SIDEBAR_WIDTH = 220  # Wider for dungeon names
        
        # Get screen info for smart sizing
        display_info = pygame.display.Info()
        max_screen_w = display_info.current_w - 100
        max_screen_h = display_info.current_h - 100
        
        # Calculate initial window size (fit largest map)
        # Handle both raw grids and StitchedDungeon objects
        max_map_h = max(m.global_grid.shape[0] if hasattr(m, 'global_grid') else m.shape[0] for m in self.maps)
        max_map_w = max(m.global_grid.shape[1] if hasattr(m, 'global_grid') else m.shape[1] for m in self.maps)
        
        # Smart sizing: fit map with some padding, but don't exceed screen
        ideal_w = max_map_w * self.TILE_SIZE + self.SIDEBAR_WIDTH
        ideal_h = max_map_h * self.TILE_SIZE + self.HUD_HEIGHT
        
        self.screen_w = min(ideal_w, max_screen_w)
        self.screen_h = min(ideal_h, max_screen_h)
        
        # Ensure minimum size
        self.screen_w = max(self.screen_w, self.MIN_WIDTH)
        self.screen_h = max(self.screen_h, self.MIN_HEIGHT)
        
        # Create resizable window
        self.screen = pygame.display.set_mode(
            (self.screen_w, self.screen_h), 
            pygame.RESIZABLE
        )
        pygame.display.set_caption("ZAVE: Zelda AI Validation Environment")
        
        # View offset for panning
        self.view_offset_x = 0
        self.view_offset_y = 0
        self.dragging = False
        self.drag_start = (0, 0)
        
        # Fullscreen state
        self.fullscreen = False
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 14, bold=True)
        self.big_font = pygame.font.SysFont('Arial', 20, bold=True)
        self.small_font = pygame.font.SysFont('Arial', 12)
        
        # Delta-time tracking for smooth animations
        self.last_frame_time = time.time()
        self.delta_time = 0.0
        
        # New visualization system
        if VISUALIZATION_AVAILABLE:
            self.renderer = ZeldaRenderer(self.TILE_SIZE)
            self.effects = EffectManager()
            self.modern_hud = ModernHUD()
        else:
            self.renderer = None
            self.effects = None
            self.modern_hud = None
        
        # Heatmap state for A* visualization
        self.show_heatmap = False
        self.search_heatmap = {}  # position -> visit count
        
        # Load assets (fallback for when new system unavailable)
        self._load_assets()
        
        # Initialize environment
        self.env = None
        self.solver = None
        self.auto_path = []
        self.auto_step_idx = 0
        self.auto_mode = False
        self.message = "Press SPACE to auto-solve, Arrow keys to move"
        self.message_time = time.time()  # Track when message was set
        self.message_duration = 3.0  # How long to show messages (seconds)
        self.error_message = None
        self.error_time = 0
        self.status_message = "Ready"
        self.show_help = False  # Toggle help overlay
        
        # State-space solver tracking (inventory/edge info)
        self.solver_result = None  # Stores keys_available, keys_used, edge_types etc.
        self.current_keys_held = 0  # Keys currently held during auto-solve
        self.current_keys_used = 0  # Keys used so far during auto-solve
        self.current_edge_types = []  # Edge types traversed so far
        self.door_unlock_times = {}  # Track when doors are unlocked for visual feedback
        
        # Path preview dialog (Feature 5)
        self.path_preview_dialog = None  # PathPreviewDialog instance when showing preview
        self.path_preview_mode = False  # True when showing path preview
        # If True, show a blocking modal dialog. If False, show non-modal overlay + sidebar summary.
        # Default: False to avoid blocking the map view (user prefers sidebar preview).
        self.preview_modal_enabled = False
        # When True the map will show the path overlay and a small sidebar preview box (non-modal)
        self.preview_overlay_visible = False

        # Topology overlay and DOT export
        self.show_topology = False
        self.topology_export_path = None

        # Solver metrics and comparison results
        self.last_solver_metrics = None  # dict: {name,nodes,time_ms,path_len}
        self.solver_comparison_results = None  # list of dicts
        self.show_solver_comparison_overlay = False

        # Presets
        self.presets = ['Debugging', 'Fast Approx', 'Optimal', 'Speedrun']
        self.current_preset_idx = 0

        # D* Lite integration
        self.dstar_solver = None
        self.dstar_active = False

        # Parallel search state
        self.parallel_search_thread = None
        self.parallel_search_done = False
        self.parallel_search_result = None
        
        # Smooth agent animation state
        self.agent_visual_pos = None  # Vector2 for smooth movement
        self.agent_target_pos = None  # Grid position target
        
        # Speed control system
        self.speed_levels = [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
        self.speed_index = 2  # Start at 1.0x
        self.speed_multiplier = self.speed_levels[self.speed_index]
        
        # Game metrics
        self.step_count = 0  # Total steps taken
        self.item_pickup_times = {}  # Track when items were picked up for animation
        
        # Item totals for "X/Y collected" display
        self.total_keys = 0  # Total keys in dungeon
        self.total_bombs = 0  # Total bomb items
        self.total_boss_keys = 0  # Total boss keys
        self.keys_collected = 0  # Keys collected so far
        self.bombs_collected = 0  # Bombs collected
        self.boss_keys_collected = 0  # Boss keys collected
        
        # Toast notification system
        self.toast_notifications = []  # List of ToastNotification objects
        # Debug overlay & logging
        self.debug_overlay_enabled = False
        self.debug_click_log = []  # List of (pos, time, handled_widget_name)
        
        # Continuous movement (hold key to move)
        self.keys_held = {pygame.K_UP: False, pygame.K_DOWN: False, pygame.K_LEFT: False, pygame.K_RIGHT: False}
        self.move_timer = 0.0  # Timer for continuous movement
        self.move_delay = 0.15  # Delay between moves (seconds)
        
        # Minimap settings
        self.show_minimap = True  # Toggle minimap display
        self.minimap_size = 150  # Pixel size of minimap
        self.minimap_clickable = True  # Allow clicking minimap to navigate
        
        # === NEW: Item tracking for enhanced visualization ===
        self.collected_items = []  # List of (pos, item_type, timestamp)
        self.used_items = []       # List of (pos, item_type, target_pos, timestamp)
        self.item_markers = {}     # Dict: position -> ItemMarkerEffect
        self.collection_effects = []  # Active collection effects
        self.usage_effects = []    # Active usage effects
        
        # === Toast Notification System ===
        self.toast_notifications = []  # List of ToastNotification objects
        
        # === NEW: GUI Control Panel ===
        self.control_panel_enabled = WIDGETS_AVAILABLE
        self.widget_manager = None
        self.control_panel_width = 300  # Logical expanded width
        self.control_panel_width_current = float(self.control_panel_width)  # Animated visual width
        self.control_panel_collapsed = False  # Track collapsed state
        self.control_panel_rect = None
        self.collapse_button_rect = None  # Rectangle for collapse button

        # Animation state for smooth collapse/expand
        self.control_panel_animating = False
        self.control_panel_anim_start = 0.0
        self.control_panel_anim_from = float(self.control_panel_width)
        self.control_panel_anim_to = float(self.control_panel_width)
        self.control_panel_anim_duration = 0.22
        self.control_panel_target_collapsed = False
        self.control_panel_x = None  # Custom X position (None = default right side)
        self.control_panel_y = None  # Custom Y position (None = default below minimap)
        self.dragging_panel = False
        self.drag_panel_offset = (0, 0)
        self.resizing_panel = False
        self.resize_edge = None  # 'left', 'right', 'top', 'bottom'

        # Control panel scroll state (for small screens)
        self.control_panel_scroll = 0
        self.control_panel_scroll_step = 20
        self.control_panel_can_scroll = False
        self.control_panel_scroll_max = 0
        self.control_panel_scroll_track_rect = None
        self.control_panel_scroll_thumb_rect = None
        self.control_panel_scroll_dragging = False
        self.control_panel_scroll_drag_offset = 0
        self.control_panel_content_height = 0

        # Scroll inertia/momentum
        self.control_panel_scroll_velocity = 0.0  # pixels per second
        self.control_panel_scroll_damping = 6.0   # damping factor (higher = faster stop)
        # Ignore clicks during active scroll or shortly after to avoid accidental toggles
        self.control_panel_ignore_click_until = 0.0

        self.min_panel_width = 250
        self.max_panel_width = 500
        self.min_panel_height = 300
        
        # Feature toggles (controlled by checkboxes)
        self.feature_flags = {
            'solver_comparison': False,
            'parallel_search': False,
            'multi_goal': False,
            'ml_heuristic': False,
            'dstar_lite': False,
            'show_heatmap': False,
            'show_minimap': True,
            'diagonal_movement': False,
            'speedrun_mode': False,
            'dynamic_difficulty': False,
            'force_grid': False,
        }
        # Toggle to force using selected grid algorithm even when graph info exists
        self.force_grid_algorithm = False
        
        # Dropdown selections
        self.current_floor = 1
        self.zoom_level_idx = 3  # 100%
        self.difficulty_idx = 1  # Medium
        self.algorithm_idx = 0   # A*
        
        self._load_current_map()
        self._center_view()  # Center the map in view
        
        # Initialize control panel after map loaded
        if self.control_panel_enabled:
            self._init_control_panel()

    
    def _load_assets(self):
        """Load tile images - using colored squares for reliability."""
        self.images = {}
        
        # Color definitions for tile rendering
        color_map = {
            SEMANTIC_PALETTE['VOID']: (20, 20, 20),
            SEMANTIC_PALETTE['FLOOR']: (200, 180, 140),
            SEMANTIC_PALETTE['WALL']: (60, 60, 140),
            SEMANTIC_PALETTE['BLOCK']: (139, 90, 43),
            SEMANTIC_PALETTE['DOOR_OPEN']: (100, 80, 60),
            SEMANTIC_PALETTE['DOOR_LOCKED']: (139, 69, 19),
            SEMANTIC_PALETTE['DOOR_BOMB']: (80, 80, 80),
            SEMANTIC_PALETTE['DOOR_BOSS']: (180, 40, 40),
            SEMANTIC_PALETTE['DOOR_PUZZLE']: (140, 80, 180),
            SEMANTIC_PALETTE['DOOR_SOFT']: (100, 100, 60),
            SEMANTIC_PALETTE['ENEMY']: (200, 50, 50),
            SEMANTIC_PALETTE['START']: (80, 180, 80),
            SEMANTIC_PALETTE['TRIFORCE']: (255, 215, 0),
            SEMANTIC_PALETTE['BOSS']: (150, 20, 20),
            SEMANTIC_PALETTE['KEY_SMALL']: (255, 200, 50),
            SEMANTIC_PALETTE['KEY_BOSS']: (200, 100, 50),
            SEMANTIC_PALETTE['KEY_ITEM']: (100, 200, 255),
            SEMANTIC_PALETTE['ITEM_MINOR']: (200, 200, 200),
            SEMANTIC_PALETTE['ELEMENT']: (50, 80, 180),
            SEMANTIC_PALETTE['ELEMENT_FLOOR']: (80, 100, 160),
            SEMANTIC_PALETTE['STAIR']: (120, 100, 80),
            SEMANTIC_PALETTE['PUZZLE']: (180, 100, 180),
        }
        
        # Create colored square tiles for each semantic ID
        for tile_id, color in color_map.items():
            surf = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE))
            surf.fill(color)
            
            # Add visual indicators for special tiles
            if tile_id == SEMANTIC_PALETTE['DOOR_LOCKED']:
                # Draw keyhole indicator
                pygame.draw.circle(surf, (255, 200, 50), 
                                 (self.TILE_SIZE//2, self.TILE_SIZE//2 - 4), 4)
                pygame.draw.rect(surf, (255, 200, 50),
                               (self.TILE_SIZE//2 - 2, self.TILE_SIZE//2, 4, 8))
            elif tile_id == SEMANTIC_PALETTE['DOOR_BOMB']:
                # Draw crack pattern
                pygame.draw.line(surf, (40, 40, 40), (8, 8), (24, 24), 2)
                pygame.draw.line(surf, (40, 40, 40), (24, 8), (8, 24), 2)
            elif tile_id == SEMANTIC_PALETTE['KEY_SMALL']:
                # Draw key with glow effect for better visibility
                # Outer glow (yellow)
                pygame.draw.circle(surf, (255, 255, 100), (16, 10), 9)
                # Key head (circle)
                pygame.draw.circle(surf, (255, 215, 0), (16, 10), 6)
                # Key shaft
                pygame.draw.rect(surf, (255, 215, 0), (14, 10, 4, 16))
                # Key teeth
                pygame.draw.rect(surf, (255, 215, 0), (14, 22, 2, 3))
                pygame.draw.rect(surf, (255, 215, 0), (16, 24, 2, 2))
                # Inner shine
                pygame.draw.circle(surf, (255, 255, 200), (17, 9), 2)
                pygame.draw.circle(surf, (255, 255, 100, 150), (16, 10), 9)
                # Key head (circle)
                pygame.draw.circle(surf, (255, 215, 0), (16, 10), 6)
                # Key shaft
                pygame.draw.rect(surf, (255, 215, 0), (14, 10, 4, 16))
                # Key teeth
                pygame.draw.rect(surf, (255, 215, 0), (14, 22, 2, 3))
                pygame.draw.rect(surf, (255, 215, 0), (16, 24, 2, 2))
                # Inner shine
                pygame.draw.circle(surf, (255, 255, 200), (17, 9), 2)
            elif tile_id == SEMANTIC_PALETTE['TRIFORCE']:
                # Draw golden triangle
                points = [(16, 4), (4, 28), (28, 28)]
                pygame.draw.polygon(surf, (255, 255, 200), points)
                pygame.draw.polygon(surf, (200, 180, 0), points, 2)
            elif tile_id == SEMANTIC_PALETTE['ENEMY']:
                # Draw enemy indicator (red circle with eyes)
                pygame.draw.circle(surf, (255, 100, 100), (16, 16), 10)
                pygame.draw.circle(surf, (0, 0, 0), (12, 12), 3)
                pygame.draw.circle(surf, (0, 0, 0), (20, 12), 3)
            elif tile_id == SEMANTIC_PALETTE['START']:
                # Draw stair pattern
                pygame.draw.rect(surf, (60, 140, 60), (4, 4, 24, 24))
                for i in range(4):
                    pygame.draw.line(surf, (40, 100, 40), (8, 8+i*6), (24, 8+i*6), 2)
            elif tile_id == SEMANTIC_PALETTE['STAIR']:
                # Draw stair steps
                for i in range(4):
                    pygame.draw.rect(surf, (100, 80, 60), (4+i*4, 20-i*4, 20-i*4, 4))
            elif tile_id == SEMANTIC_PALETTE['WALL']:
                # Add brick pattern to walls
                pygame.draw.rect(surf, (50, 50, 120), (2, 2, 28, 28), 2)
                pygame.draw.line(surf, (70, 70, 150), (0, 16), (32, 16), 1)
                pygame.draw.line(surf, (70, 70, 150), (16, 0), (16, 32), 1)
            elif tile_id == SEMANTIC_PALETTE['BLOCK']:
                # Add block texture
                pygame.draw.rect(surf, (100, 60, 30), (2, 2, 28, 28), 2)
            elif tile_id == SEMANTIC_PALETTE['DOOR_OPEN']:
                # Draw open doorway
                pygame.draw.rect(surf, (40, 30, 20), (8, 0, 16, 32))
            elif tile_id == SEMANTIC_PALETTE['ELEMENT']:
                # Water/lava pattern
                for i in range(4):
                    pygame.draw.arc(surf, (80, 120, 200), (i*8, 8, 16, 16), 0, 3.14, 2)
                    pygame.draw.arc(surf, (80, 120, 200), (i*8, 16, 16, 16), 3.14, 6.28, 2)
            
            self.images[tile_id] = surf
        
        # Create Link sprite
        self.link_img = self._create_link_sprite()

        # Create a small stair sprite (glowing marker) for visual emphasis
        try:
            # Force stair sprite to full tile size and use a bright, high-contrast overlay
            sprite_size = self.TILE_SIZE
            self.stair_sprite = pygame.Surface((sprite_size, sprite_size), pygame.SRCALPHA)
            self.stair_sprite.fill((0, 0, 0, 0))

            # Full-tile translucent fill (warm gold)
            pygame.draw.rect(self.stair_sprite, (255, 220, 100, 180), (0, 0, sprite_size, sprite_size))
            # Strong border for clear visibility
            pygame.draw.rect(self.stair_sprite, (255, 200, 50), (1, 1, sprite_size-2, sprite_size-2), 4)

            # Center triangle to indicate stair direction
            pts = [(sprite_size//2, sprite_size//6), (sprite_size//6, sprite_size*5//6), (sprite_size*5//6, sprite_size*5//6)]
            pygame.draw.polygon(self.stair_sprite, (255, 245, 180), pts)
            pygame.draw.polygon(self.stair_sprite, (255, 200, 50), pts, 2)

            # Slight inner highlight circle
            pygame.draw.circle(self.stair_sprite, (255, 255, 220, 64), (sprite_size//2, sprite_size//2), max(6, sprite_size//6))

            self.stair_anim_phase = 0.0
        except Exception:
            self.stair_sprite = None
            self.stair_anim_phase = 0.0
    
    def _create_link_sprite(self) -> pygame.Surface:
        """Create a detailed Link sprite using pygame drawing."""
        link_img = pygame.Surface((self.TILE_SIZE - 4, self.TILE_SIZE - 4), pygame.SRCALPHA)
        
        # Transparent background
        link_img.fill((0, 0, 0, 0))
        
        # Body colors
        green = (0, 168, 0)
        skin = (252, 216, 168)
        brown = (136, 112, 0)
        dark_green = (0, 120, 0)
        
        # Draw Link's body (green tunic)
        pygame.draw.rect(link_img, green, (8, 12, 12, 12))  # Torso
        pygame.draw.rect(link_img, dark_green, (6, 18, 4, 8))  # Left arm
        pygame.draw.rect(link_img, dark_green, (18, 18, 4, 8))  # Right arm
        
        # Draw head
        pygame.draw.rect(link_img, skin, (8, 2, 12, 10))  # Face
        pygame.draw.circle(link_img, (0, 0, 0), (11, 6), 2)  # Left eye
        pygame.draw.circle(link_img, (0, 0, 0), (17, 6), 2)  # Right eye
        
        # Draw hair/cap (brown)
        pygame.draw.rect(link_img, brown, (6, 0, 16, 4))  # Hair top
        pygame.draw.rect(link_img, brown, (4, 2, 4, 6))  # Hair left
        pygame.draw.rect(link_img, brown, (20, 2, 4, 6))  # Hair right
        
        # Draw shield (brown rectangle on left side)
        pygame.draw.rect(link_img, brown, (2, 14, 6, 10))
        pygame.draw.rect(link_img, (200, 150, 50), (3, 15, 4, 8))  # Shield front
        
        # Draw sword (on right side)
        pygame.draw.rect(link_img, (180, 180, 180), (22, 12, 4, 14))  # Blade
        pygame.draw.rect(link_img, brown, (22, 10, 4, 4))  # Hilt
        
        return link_img
    
    def _init_control_panel(self):
        """Initialize the GUI control panel with widgets."""
        if not WIDGETS_AVAILABLE:
            return
        
        self.widget_manager = WidgetManager()
        self._update_control_panel_positions()
    
    def _update_control_panel_positions(self):
        """Update control panel and widget positions (called on resize)."""
        if not WIDGETS_AVAILABLE or not self.widget_manager:
            return
        
        # Control panel position
        # Use custom position if set, otherwise default to right side
        collapsed_width = 40
        # Use animated visual width for current layout
        panel_width = int(max(collapsed_width, min(self.control_panel_width_current, self.max_panel_width)))

        # Compute default dock position and allow custom drag; then clamp to sidebar area
        sidebar_x = self.screen_w - self.SIDEBAR_WIDTH
        if self.control_panel_x is not None and self.control_panel_y is not None:
            # Use custom dragged position
            panel_x = self.control_panel_x
            panel_y = self.control_panel_y
        else:
            # Default position (docked to right side, near sidebar)
            panel_x = sidebar_x - panel_width - 10
            panel_y = self.minimap_size + 20 if self.show_minimap else 10

        # Clamp panel inside usable area
        min_x = 10
        max_x = max(min_x, sidebar_x - panel_width - 10)
        panel_x = max(min_x, min(panel_x, max_x))

        # Calculate panel height (align with _render_control_panel logic)
        max_available_height = self.screen_h - panel_y - self.HUD_HEIGHT - 20
        min_panel_height = 120
        if max_available_height < min_panel_height:
            panel_height = min_panel_height
        else:
            panel_height = min(max_available_height, 700)

        self.control_panel_rect = pygame.Rect(
            panel_x, panel_y,
            panel_width,
            panel_height
        )

        # Collapse/Expand button (always visible) - top-right
        button_size = 32
        self.collapse_button_rect = pygame.Rect(
            panel_x + panel_width - 34, panel_y + 4,
            button_size, button_size
        )
        
        # Don't update widgets if effectively collapsed (small width)
        if panel_width <= collapsed_width + 8:
            return
        
        # Only rebuild widgets if none exist yet
        widgets_exist = hasattr(self, 'widget_manager') and self.widget_manager and len(self.widget_manager.widgets) > 0
        if widgets_exist:
            # Just update widget positions instead of rebuilding
            self._reposition_widgets(panel_x, panel_y)
            return
        
        # Clear existing widgets if reinitializing
        if hasattr(self, 'widget_manager') and self.widget_manager:
            self.widget_manager.widgets.clear()
        
        # === CONSISTENT LAYOUT CONSTANTS ===
        margin_left = 12
        margin_top = 48  # Space for collapse button + "FEATURES" title
        checkbox_spacing = 26
        dropdown_spacing = 36
        section_gap = 18
        
        # Start position (below "FEATURES" title)
        y_offset = panel_y + margin_top
        x_offset = panel_x + margin_left
        
        # === CHECKBOXES SECTION ===
        checkbox_labels = [
            ('solver_comparison', 'Solver Comparison'),
            ('parallel_search', 'Parallel Search'),
            ('multi_goal', 'Multi-Goal Pathfinding'),
            ('ml_heuristic', 'ML Heuristic'),
            ('dstar_lite', 'D* Lite Replanning'),
            ('show_heatmap', 'Show Heatmap Overlay'),
            ('show_topology', 'Show Topology Overlay'),
            ('show_minimap', 'Show Minimap'),
            ('diagonal_movement', 'Diagonal Movement'),
            ('speedrun_mode', 'Speedrun Mode'),
            ('dynamic_difficulty', 'Dynamic Difficulty'),
            ('force_grid', 'Force Grid Solver'),
        ]
        
        for flag_name, label in checkbox_labels:
            checkbox = CheckboxWidget(
                (x_offset, y_offset),
                label,
                checked=self.feature_flags.get(flag_name, False)
            )
            checkbox.flag_name = flag_name
            self.widget_manager.add_widget(checkbox)
            y_offset += checkbox_spacing
        
        # Section gap before dropdowns
        y_offset += section_gap
        
        # === DROPDOWNS SECTION ===
        # Floor selector
        floor_dropdown = DropdownWidget(
            (x_offset, y_offset),
            "Floor",
            ["Floor 1", "Floor 2", "Floor 3"],
            selected=self.current_floor - 1
        )
        floor_dropdown.control_name = 'floor'
        self.widget_manager.add_widget(floor_dropdown)
        y_offset += dropdown_spacing
        
        # Zoom level
        zoom_dropdown = DropdownWidget(
            (x_offset, y_offset),
            "Zoom",
            ["25%", "50%", "75%", "100%", "150%", "200%"],
            selected=self.zoom_level_idx
        )
        zoom_dropdown.control_name = 'zoom'
        self.widget_manager.add_widget(zoom_dropdown)
        y_offset += dropdown_spacing
        
        # Difficulty
        difficulty_dropdown = DropdownWidget(
            (x_offset, y_offset),
            "Difficulty",
            ["Easy", "Medium", "Hard", "Expert"],
            selected=self.difficulty_idx
        )
        difficulty_dropdown.control_name = 'difficulty'
        self.widget_manager.add_widget(difficulty_dropdown)
        y_offset += dropdown_spacing
        
        # Presets
        presets_dropdown = DropdownWidget(
            (x_offset, y_offset),
            "Presets",
            self.presets,
            selected=self.current_preset_idx
        )
        presets_dropdown.control_name = 'presets'
        self.widget_manager.add_widget(presets_dropdown)
        y_offset += dropdown_spacing

        # Algorithm
        algorithm_dropdown = DropdownWidget(
            (x_offset, y_offset),
            "Solver",
            ["A*", "BFS", "Dijkstra", "Greedy", "D* Lite"],
            selected=self.algorithm_idx
        )
        algorithm_dropdown.control_name = 'algorithm'
        self.widget_manager.add_widget(algorithm_dropdown)
        y_offset += dropdown_spacing
        
        # Section gap before buttons
        y_offset += section_gap
        
        # === BUTTONS SECTION ===
        button_width = 125
        button_height = 30
        buttons_per_row = 2
        button_h_spacing = 8
        button_v_spacing = 8
        
        # Primary action buttons
        primary_buttons = [
            ("Start Auto-Solve", self._start_auto_solve),
            ("Stop", self._stop_auto_solve),
            ("Generate Dungeon", self._generate_dungeon),
            ("Reset", self._reset_map),
        ]
        
        # Secondary action buttons
        secondary_buttons = [
            ("Path Preview", self._show_path_preview),
            ("Clear Path", self._clear_path),
            ("Export Route", self._export_route),
            ("Load Route", self._load_route),
            ("Export Topology", self._export_topology),
            ("Compare Solvers", self._run_solver_comparison),
        ]
        
        # Render primary buttons in 2x2 grid
        button_start_y = y_offset
        for i, (label, callback) in enumerate(primary_buttons):
            row = i // buttons_per_row
            col = i % buttons_per_row
            button_x = x_offset + col * (button_width + button_h_spacing)
            button_y = button_start_y + row * (button_height + button_v_spacing)
            
            button = ButtonWidget(
                (button_x, button_y),
                label,
                callback,
                width=button_width,
                height=button_height
            )
            self.widget_manager.add_widget(button)
        
        # Update y_offset for secondary buttons
        y_offset = button_start_y + (len(primary_buttons) // buttons_per_row) * (button_height + button_v_spacing) + 12
        
        # Render secondary buttons in 2x2 grid
        for i, (label, callback) in enumerate(secondary_buttons):
            row = i // buttons_per_row
            col = i % buttons_per_row
            button_x = x_offset + col * (button_width + button_h_spacing)
            button_y = y_offset + row * (button_height + button_v_spacing)
            
            button = ButtonWidget(
                (button_x, button_y),
                label,
                callback,
                width=button_width,
                height=button_height
            )
            self.widget_manager.add_widget(button)
        
        # === POST-BUILD: compute content height and decide if we should scroll ===
        # Compute bottom-most widget to determine content height
        max_widget_bottom = 0
        for w in self.widget_manager.widgets:
            bottom = None
            if hasattr(w, 'full_rect'):
                bottom = w.full_rect.bottom
            elif hasattr(w, 'dropdown_rect'):
                bottom = w.rect.bottom
            elif hasattr(w, 'rect'):
                bottom = w.rect.bottom
            if bottom and bottom > max_widget_bottom:
                max_widget_bottom = bottom
        # Desired content height (space required to show everything)
        content_height = max_widget_bottom - panel_y + 12 if max_widget_bottom > 0 else min_panel_height
        self.control_panel_content_height = content_height

        # If content exceeds available height, enable scrolling instead of enlarging
        if content_height > max_available_height:
            # Keep panel_height limited and enable scrolling
            panel_height = min(max_available_height, 700)
            self.control_panel_can_scroll = True
            self.control_panel_scroll = getattr(self, 'control_panel_scroll', 0)
            self.control_panel_scroll = max(0, min(self.control_panel_scroll, content_height - panel_height))
        else:
            # Grow to fit content when possible
            panel_height = min(max_available_height, max(content_height, min_panel_height))
            self.control_panel_can_scroll = False
            self.control_panel_scroll = 0

        # Update control panel rect height and collapse button Y
        self.control_panel_rect.height = panel_height
        self.collapse_button_rect.y = panel_y + 4
        # Reposition widgets in case layout changed
        self._reposition_widgets(panel_x, panel_y)
    
    def _reposition_widgets(self, panel_x: int, panel_y: int):
        """Reposition existing widgets when panel is dragged (without rebuilding)."""
        if not self.widget_manager or not self.widget_manager.widgets:
            return
        
        # Define consistent spacing
        margin_left = 12
        # Compute dynamic top margin from collapse button size to align items under header
        button_size = 28
        button_margin = 6
        margin_top = button_margin + max(button_size, self.font.get_height()) + 8
        item_spacing = 30  # Vertical spacing between checkboxes
        section_gap = 20  # Gap between sections
        
        # Starting Y position (below "FEATURES" title)
        current_y = panel_y + margin_top
        
        # Counters for different widget types
        checkbox_idx = 0
        dropdown_idx = 0
        button_idx = 0
        
        # Button layout configuration
        button_width = 125
        button_height = 30
        buttons_per_row = 2
        button_h_spacing = 8
        button_v_spacing = 8
        
        for widget in self.widget_manager.widgets:
            if isinstance(widget, CheckboxWidget):
                # Position checkboxes in a vertical list
                widget.pos = (panel_x + margin_left, current_y + checkbox_idx * item_spacing)
                checkbox_idx += 1
                
            elif isinstance(widget, DropdownWidget):
                # Position dropdowns after checkboxes with section gap
                if checkbox_idx > 0 and dropdown_idx == 0:
                    current_y += checkbox_idx * item_spacing + section_gap
                
                widget.pos = (panel_x + margin_left, current_y + dropdown_idx * 38)
                dropdown_idx += 1
                
            elif isinstance(widget, ButtonWidget):
                # Position buttons in a grid after dropdowns
                if dropdown_idx > 0 and button_idx == 0:
                    current_y += dropdown_idx * 38 + section_gap
                
                row = button_idx // buttons_per_row
                col = button_idx % buttons_per_row
                widget.pos = (
                    panel_x + margin_left + col * (button_width + button_h_spacing),
                    current_y + row * (button_height + button_v_spacing)
                )
                button_idx += 1
    
    def _track_item_collection(self, old_state, new_state):
        """Detect when items are collected by comparing states."""
        # Check for key collection
        if new_state.keys > old_state.keys:
            keys_collected = new_state.keys - old_state.keys
            pos = new_state.position
            timestamp = time.time()
            
            self.collected_items.append((pos, 'key', timestamp))
            self.keys_collected += keys_collected
            
            # Remove marker if exists
            if pos in self.item_markers and self.item_markers[pos].item_type == 'key':
                del self.item_markers[pos]
            
            # Add collection effect
            if self.effects:
                effect = ItemCollectionEffect(pos, 'key', 'KEY', 
                                             f'Key collected at ({pos[0]}, {pos[1]})!')
                self.effects.add_effect(effect)
                self.collection_effects.append(effect)
            
            # Show toast notification
            self._show_toast(f"Key collected! Now have {self.keys_collected}/{self.total_keys}", 
                           duration=2.5, toast_type='success')
        
        # Check for bomb collection
        if new_state.has_bomb and not old_state.has_bomb:
            pos = new_state.position
            timestamp = time.time()
            
            self.collected_items.append((pos, 'bomb', timestamp))
            self.bombs_collected += 1
            
            # Remove marker
            if pos in self.item_markers and self.item_markers[pos].item_type == 'bomb':
                del self.item_markers[pos]
            
            # Add collection effect
            if self.effects:
                effect = ItemCollectionEffect(pos, 'bomb', 'BOMB',
                                             f'Bomb collected at ({pos[0]}, {pos[1]})!')
                self.effects.add_effect(effect)
                self.collection_effects.append(effect)
            
            # Show toast notification
            self._show_toast("Bomb acquired! Can now blow up weak walls", 
                           duration=3.0, toast_type='success')
        
        # Check for boss key collection
        if new_state.has_boss_key and not old_state.has_boss_key:
            pos = new_state.position
            timestamp = time.time()
            
            self.collected_items.append((pos, 'boss_key', timestamp))
            self.boss_keys_collected += 1
            
            # Remove marker
            if pos in self.item_markers and self.item_markers[pos].item_type == 'boss_key':
                del self.item_markers[pos]
            
            # Add collection effect
            if self.effects:
                effect = ItemCollectionEffect(pos, 'boss_key', 'BOSS KEY',
                                             f'Boss Key collected at ({pos[0]}, {pos[1]})!')
                self.effects.add_effect(effect)
                self.collection_effects.append(effect)
            
            # Show toast notification
            self._show_toast("Boss Key acquired! Can now face the boss", 
                           duration=3.0, toast_type='success')
    
    def _track_item_usage(self, old_state, new_state):
        """Detect when items are used (doors opened, walls bombed)."""
        # Check if key was used
        if new_state.keys < old_state.keys:
            keys_used = old_state.keys - new_state.keys
            pos = new_state.position
            timestamp = time.time()
            
            self.used_items.append((pos, 'key', pos, timestamp))
            
            # Add usage effect
            if self.effects:
                effect = ItemUsageEffect(old_state.position, pos, 'key')
                self.effects.add_effect(effect)
                self.usage_effects.append(effect)
        
        # Check if bomb was used
        if old_state.has_bomb and not new_state.has_bomb:
            pos = new_state.position
            timestamp = time.time()
            
            self.used_items.append((pos, 'bomb', pos, timestamp))
            
            # Add explosion effect
            if self.effects:
                effect = ItemUsageEffect(old_state.position, pos, 'bomb')
                self.effects.add_effect(effect)
                self.usage_effects.append(effect)
        
        # Check if boss key was used
        if old_state.has_boss_key and not new_state.has_boss_key:
            pos = new_state.position
            timestamp = time.time()
            
            self.used_items.append((pos, 'boss_key', pos, timestamp))
            
            # Add usage effect
            if self.effects:
                effect = ItemUsageEffect(old_state.position, pos, 'boss_key')
                self.effects.add_effect(effect)
                self.usage_effects.append(effect)
    
    def _scan_and_mark_items(self):
        """Scan the map for all items and create markers."""
        self.item_markers.clear()
        self.total_keys = 0
        self.total_bombs = 0
        self.total_boss_keys = 0
        
        if not self.env:
            return
        
        h, w = self.env.height, self.env.width
        
        for r in range(h):
            for c in range(w):
                tile_id = self.env.grid[r, c]
                pos = (r, c)
                
                # Check if already collected
                if pos in self.env.state.collected_items:
                    continue
                
                # Create marker for items
                if tile_id == SEMANTIC_PALETTE['KEY_SMALL']:
                    self.total_keys += 1
                    marker = ItemMarkerEffect(pos, 'key', 'K')
                    self.item_markers[pos] = marker
                    if self.effects:
                        self.effects.add_effect(marker)
                
                elif tile_id == SEMANTIC_PALETTE.get('ITEM_BOMB', -1):
                    self.total_bombs += 1
                    marker = ItemMarkerEffect(pos, 'bomb', 'B')
                    self.item_markers[pos] = marker
                    if self.effects:
                        self.effects.add_effect(marker)
                
                elif tile_id == SEMANTIC_PALETTE['KEY_BOSS']:
                    self.total_boss_keys += 1
                    marker = ItemMarkerEffect(pos, 'boss_key', 'BK')
                    self.item_markers[pos] = marker
                    if self.effects:
                        self.effects.add_effect(marker)
                
                elif tile_id == SEMANTIC_PALETTE['TRIFORCE']:
                    marker = ItemMarkerEffect(pos, 'triforce', 'TRI')
                    self.item_markers[pos] = marker
                    if self.effects:
                        self.effects.add_effect(marker)
    
    def _render_item_legend(self, surface):
        """Render legend showing item counts."""
        if not self.env:
            return
        
        legend_x = 10
        legend_y = self.screen_h - 60
        
        # Background
        legend_bg = pygame.Surface((250, 50), pygame.SRCALPHA)
        legend_bg.fill((30, 30, 40, 200))
        surface.blit(legend_bg, (legend_x, legend_y))
        
        # Text
        keys_remaining = self.total_keys - self.keys_collected
        bombs_remaining = self.total_bombs - self.bombs_collected
        
        legend_text = [
            f"Keys: {keys_remaining} remaining",
            f"Bombs: {bombs_remaining} remaining",
        ]
        
        y_offset = legend_y + 8
        for text in legend_text:
            text_surf = self.small_font.render(text, True, (255, 255, 200))
            surface.blit(text_surf, (legend_x + 10, y_offset))
            y_offset += 20
    
    def _render_error_banner(self, surface):
        """Render error message banner at top of screen with fade effect."""
        if hasattr(self, 'error_message') and self.error_message:
            elapsed = time.time() - self.error_time
            if elapsed < 5.0:  # Show for 5 seconds
                # Calculate fade (0.0 to 1.0)
                alpha = 1.0 if elapsed < 4.0 else (5.0 - elapsed)
                
                # Draw red banner at top
                banner_height = 45
                banner_rect = pygame.Rect(0, 0, self.screen_w, banner_height)
                banner_surface = pygame.Surface((self.screen_w, banner_height), pygame.SRCALPHA)
                banner_surface.fill((200, 0, 0, int(220 * alpha)))
                surface.blit(banner_surface, (0, 0))
                
                # Draw error icon and text
                font = pygame.font.Font(None, 28)
                text = f"[!] {self.error_message}"
                text_surf = font.render(text, True, (255, 255, 255))
                text_surf.set_alpha(int(255 * alpha))
                text_rect = text_surf.get_rect(center=(self.screen_w // 2, banner_height // 2))
                surface.blit(text_surf, text_rect)
                
                # Draw border
                pygame.draw.rect(surface, (150, 0, 0), banner_rect, 2)
            else:
                self.error_message = None
    
    def _render_status_bar(self, surface):
        """Render status bar at bottom of screen."""
        bar_height = 30
        bar_y = self.screen_h - bar_height
        bar_rect = pygame.Rect(0, bar_y, self.screen_w, bar_height)
        
        # Background
        bar_surface = pygame.Surface((self.screen_w, bar_height), pygame.SRCALPHA)
        bar_surface.fill((40, 40, 50, 200))
        surface.blit(bar_surface, (0, bar_y))
        
        # Status text
        font = pygame.font.Font(None, 20)
        
        # Left: Status
        status_text = f"Status: {self.status_message}"
        status_surf = font.render(status_text, True, (180, 220, 255))
        surface.blit(status_surf, (10, bar_y + 7))
        
        # Center: Current action/message
        if self.auto_mode:
            progress = f"Step {self.auto_step_idx + 1}/{len(self.auto_path)}"
            progress_surf = font.render(progress, True, (100, 255, 100))
            progress_rect = progress_surf.get_rect(center=(self.screen_w // 2, bar_y + bar_height // 2))
            surface.blit(progress_surf, progress_rect)
        
        # Right: FPS
        fps_text = f"FPS: {int(self.clock.get_fps())}"
        fps_surf = font.render(fps_text, True, (255, 255, 180))
        fps_rect = fps_surf.get_rect(right=self.screen_w - 10, centery=bar_y + bar_height // 2)
        surface.blit(fps_surf, fps_rect)
        
        # Border
        pygame.draw.rect(surface, (60, 60, 80), bar_rect, 1)
    
    def _render_control_panel(self, surface):
        """Render the control panel with all GUI widgets and metrics."""
        if not self.control_panel_enabled or not self.widget_manager:
            return
        logger.debug(f"_render_control_panel: width_current={self.control_panel_width_current}, collapsed={self.control_panel_collapsed}, animating={getattr(self, 'control_panel_animating', False)}")

        # Collapsed threshold (always defined so later checks won't fail)
        collapsed_width = 40

        # Ensure control panel rects/positions are up-to-date (keeps animation offsets consistent)
        try:
            self._update_control_panel_positions()
        except Exception:
            pass

        # If update set a rect, use its coordinates for rendering; otherwise fallback to local computation
        if getattr(self, 'control_panel_rect', None):
            panel_rect = self.control_panel_rect
            panel_x, panel_y, panel_width, panel_height = panel_rect.x, panel_rect.y, panel_rect.width, panel_rect.height
        else:
            # Use animated width for current visual state
            panel_width = int(max(collapsed_width, min(self.control_panel_width_current, self.max_panel_width)))
            panel_width = max(collapsed_width, min(panel_width, self.max_panel_width))

            # Compute default dock position and allow custom drag
            sidebar_x = self.screen_w - self.SIDEBAR_WIDTH
            if self.control_panel_x is not None and self.control_panel_y is not None:
                panel_x = self.control_panel_x
                panel_y = self.control_panel_y
            else:
                panel_x = sidebar_x - panel_width - 10
                panel_y = 10

            min_x = 10
            max_x = max(min_x, sidebar_x - panel_width - 10)
            panel_x = max(min_x, min(panel_x, max_x))
            panel_y = max(10, min(panel_y, self.screen_h - 150))

            max_available_height = self.screen_h - panel_y - self.HUD_HEIGHT - 20
            min_panel_height = 120
            if max_available_height < min_panel_height:
                panel_height = min_panel_height
            else:
                panel_height = min(max_available_height, 700)

            if panel_width <= 0 or panel_height <= 0:
                return
            panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
            self.control_panel_rect = panel_rect


        # Create panel surface (use SRCALPHA so we can fade during animation)
        panel_surf = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        # Draw background onto panel surface
        bg_rect = pygame.Rect(0, 0, panel_width, panel_height)
        pygame.draw.rect(panel_surf, (40, 45, 60, 255), bg_rect, border_radius=8)
        pygame.draw.rect(panel_surf, (60, 60, 80, 255), bg_rect, 2, border_radius=8)

        # If animating, compute alpha for fade-in/out based on visual progress
        alpha = 255
        if getattr(self, 'control_panel_animating', False):
            a_from = self.control_panel_anim_from
            a_to = self.control_panel_anim_to
            denom = (a_to - a_from) if abs(a_to - a_from) > 1e-6 else 1.0
            progress = max(0.0, min(1.0, (self.control_panel_width_current - a_from) / denom))
            ease = progress * progress * (3 - 2 * progress)
            # If we're expanding, fade from 0->255; if collapsing, fade out
            alpha = int(255 * ease)

        # Blit background (we will blit widgets separately with alpha applied to whole panel)
        # Panel surface will be blitted at the end after widget rendering to ensure alpha applies to entire block
        

        # Update rect for mouse interaction (ensure using shifted coordinates from _update_control_panel_positions)
        # Note: _update_control_panel_positions has already computed self.control_panel_rect
        # and self.collapse_button_rect with any slide offset applied.
        # If they haven't been computed yet, fallback to unshifted rects.
        if not getattr(self, 'control_panel_rect', None):
            self.control_panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        if not getattr(self, 'collapse_button_rect', None):
            self.collapse_button_rect = pygame.Rect(
                panel_x + panel_width - 28 - 6,
                panel_y + 6,
                28,
                28
            )
        
        # Button will be drawn after panel surface is blitted so it remains on top
        # Track mouse hover for collapse button appearance
        mouse_pos = pygame.mouse.get_pos()
        is_hovering = self.collapse_button_rect.collidepoint(mouse_pos)
        
        # If visual width is small (collapsed visual), only show the collapse button (it will be drawn later)
        if panel_width <= collapsed_width + 8:
            # Draw only collapse button on main surface
            if is_hovering:
                button_color = (80, 120, 180)
                border_color = (150, 200, 255)
            else:
                button_color = (60, 80, 120)
                border_color = (100, 150, 200)
            pygame.draw.rect(surface, button_color, self.collapse_button_rect, border_radius=4)
            pygame.draw.rect(surface, border_color, self.collapse_button_rect, 2, border_radius=4)
            if getattr(self, 'control_panel_animating', False):
                button_text = ">" if self.control_panel_target_collapsed else "<"
            else:
                button_text = "<" if not self.control_panel_collapsed else ">"
            button_surf = self.font.render(button_text, True, (200, 220, 255))
            button_rect = button_surf.get_rect(center=self.collapse_button_rect.center)
            surface.blit(button_surf, button_rect)
            return
        
        # Backup widget positions to restore after local rendering
        backups = [(w, getattr(w, 'pos', None)) for w in self.widget_manager.widgets]

        # Update widget positions to match panel location (including any slide offset computed previously)
        self._reposition_widgets(panel_x, panel_y)
        
        # Render widgets individually into the panel surface with per-widget alpha
        # We'll draw the header on top later so scrolling content never overlaps it

        content_alpha = 255
        if getattr(self, 'control_panel_animating', False):
            # Use the same progress/ease as alpha computed earlier
            a_from = self.control_panel_anim_from
            a_to = self.control_panel_anim_to
            denom = (a_to - a_from) if abs(a_to - a_from) > 1e-6 else 1.0
            progress = max(0.0, min(1.0, (self.control_panel_width_current - a_from) / denom))
            ease = progress * progress * (3 - 2 * progress)
            # Snap alpha near the end to avoid near-invisible final frames
            if ease >= 0.98:
                content_alpha = 255
            elif ease <= 0.02:
                content_alpha = 0
            else:
                content_alpha = int(255 * ease)
        
        for widget, orig_pos in backups:
            try:
                # Compute widget local position relative to panel and apply scroll offset
                if orig_pos is None:
                    continue
                # widget.rect is global; calculate local coords
                local_x = widget.rect.x - panel_x
                local_y = widget.rect.y - panel_y - (getattr(self, 'control_panel_scroll', 0) if getattr(self, 'control_panel_can_scroll', False) else 0)

                # Create a temp surface sized to the widget's interactive area
                full_rect = getattr(widget, 'full_rect', widget.rect)
                dropdown_rect = getattr(widget, 'dropdown_rect', None)
                # If this is a dropdown and it is open, avoid rendering the expanded menu into the
                # panel surface (it may extend beyond panel bounds). Render only the main button
                # here and draw the expanded menu later on the main surface.
                if getattr(widget, '__class__', None).__name__ == 'DropdownWidget' and getattr(widget, 'is_open', False):
                    target_w = min(panel_width - 24, max(full_rect.width, widget.rect.width))
                    target_h = widget.rect.height
                else:
                    target_w = min(panel_width - 24, max(full_rect.width, dropdown_rect.width if dropdown_rect is not None else 0))
                    target_h = max(widget.rect.height, dropdown_rect.height if dropdown_rect is not None else widget.rect.height)

                # Skip drawing if widget is outside the visible panel area (vertical clipping)
                # Ensure widgets do not draw into the header area at the top
                header_height = self.font.get_height() + 12
                if local_y + target_h < header_height or local_y > panel_height:
                    continue

                temp_surf = pygame.Surface((max(1, target_w), max(1, target_h)), pygame.SRCALPHA)

                # Temporarily set widget.pos to (0,0) so it draws at origin of temp_surf
                widget.pos = (0, 0)
                widget.render(temp_surf)

                # Apply per-widget alpha
                if content_alpha < 255:
                    temp_surf.set_alpha(content_alpha)

                # Blit into panel surface
                blit_x = local_x
                blit_y = local_y
                panel_surf.blit(temp_surf, (blit_x, blit_y))

            except Exception as e:
                logger.warning(f"Per-widget render failed: {e}")

        # Draw scrollbar if content exceeds visible area
        if getattr(self, 'control_panel_can_scroll', False):
            track_w = 10
            track_margin = 8
            track_local_x = panel_width - track_w - track_margin
            track_local_y = 16
            track_h = panel_height - 32
            track_rect = pygame.Rect(track_local_x, track_local_y, track_w, track_h)
            # Visuals: subtle track + thumb
            pygame.draw.rect(panel_surf, (60, 65, 80, 200), track_rect, border_radius=6)
            # Compute visible content height excluding header
            header_height = self.font.get_height() + 12
            visible_h = max(10, panel_height - header_height - 16)
            content_h = getattr(self, 'control_panel_content_height', visible_h)
            max_scroll = max(content_h - visible_h, 0)
            thumb_h = max(int((visible_h / content_h) * track_h) if content_h > 0 else track_h, 20)
            if max_scroll > 0:
                thumb_y_local = track_local_y + int((getattr(self, 'control_panel_scroll', 0) / max_scroll) * (track_h - thumb_h))
            else:
                thumb_y_local = track_local_y
            thumb_rect = pygame.Rect(track_local_x + 1, thumb_y_local, track_w - 2, thumb_h)
            pygame.draw.rect(panel_surf, (100, 130, 180, 220), thumb_rect, border_radius=6)
            # Store global rects for mouse interaction handling
            self.control_panel_scroll_track_rect = pygame.Rect(panel_x + track_rect.x, panel_y + track_rect.y, track_rect.width, track_rect.height)
            self.control_panel_scroll_thumb_rect = pygame.Rect(panel_x + thumb_rect.x, panel_y + thumb_rect.y, thumb_rect.width, thumb_rect.height)
            self.control_panel_scroll_max = max_scroll
        else:
            # Clear any previous scroll rects
            self.control_panel_scroll_track_rect = None
            self.control_panel_scroll_thumb_rect = None
            self.control_panel_scroll_max = 0

        # Draw header (fixed) on top of scrolled content so it never overlaps
        header_height = self.font.get_height() + 12
        header_rect = pygame.Rect(0, 0, panel_width, header_height)
        header_surf = pygame.Surface((panel_width, header_height), pygame.SRCALPHA)
        # Slightly darker strip to separate header from scroll area
        pygame.draw.rect(header_surf, (35, 40, 55, 230), pygame.Rect(0, 0, panel_width, header_height), border_radius=0)
        # Draw FEATURES title centered vertically in the header
        features_title = self.font.render("FEATURES", True, (100, 200, 100))
        header_surf.blit(features_title, (12, (header_height - self.font.get_height()) // 2))
        panel_surf.blit(header_surf, (0, 0))

        # Apply overall panel alpha (for expand/collapse animation)
        if alpha < 255:
            panel_surf.set_alpha(alpha)
        # Blit panel surface onto main surface (panel bg + widgets)
        surface.blit(panel_surf, (panel_x, panel_y))
        # Restore panel surface alpha to full (avoid side-effects)
        if alpha < 255:
            panel_surf.set_alpha(255)

        # Restore widget global positions
        for widget, orig_pos in backups:
            try:
                if orig_pos is not None:
                    widget.pos = orig_pos
            except Exception:
                pass

        # Draw collapse button on top of the panel so it's always visible
        if is_hovering:
            button_color = (80, 120, 180)
            border_color = (150, 200, 255)
        else:
            button_color = (60, 80, 120)
            border_color = (100, 150, 200)
        pygame.draw.rect(surface, button_color, self.collapse_button_rect, border_radius=4)
        pygame.draw.rect(surface, border_color, self.collapse_button_rect, 2, border_radius=4)
        if getattr(self, 'control_panel_animating', False):
            button_text = ">" if self.control_panel_target_collapsed else "<"
        else:
            button_text = "<" if not self.control_panel_collapsed else ">"
        button_surf = self.font.render(button_text, True, (200, 220, 255))
        button_rect = button_surf.get_rect(center=self.collapse_button_rect.center)
        surface.blit(button_surf, button_rect)

        # Render dropdown menus on top of everything so they do not get clipped by the panel
        try:
            for widget in self.widget_manager.widgets:
                if getattr(widget, '__class__', None).__name__ == 'DropdownWidget' and getattr(widget, 'is_open', False):
                    try:
                        # Use the same content alpha used for widget fading during animation
                        scroll_offset = self.control_panel_scroll if getattr(self, 'control_panel_can_scroll', False) else 0
                        widget.render_menu(surface, alpha=content_alpha, scroll_offset=scroll_offset, panel_rect=self.control_panel_rect)
                    except Exception as e:
                        logger.warning(f"Dropdown menu render failed: {e}")
        except Exception:
            pass

        # Render tooltips if mouse over a widget
        self._render_tooltips(surface, mouse_pos)    
    def _render_tooltips(self, surface, mouse_pos):
        """Render tooltips for widgets under mouse cursor."""
        if not self.widget_manager:
            return
        
        # Tooltip definitions for each control
        tooltips = {
            'show_heatmap': 'Toggle A* search heatmap visualization',
            'show_minimap': 'Toggle minimap display in top-right corner',
            'show_path': 'Show/hide the solution path preview',
            'smooth_camera': 'Enable smooth camera transitions',
            'show_grid': 'Toggle grid overlay on map',
            'zoom': 'Adjust map zoom level (also use +/- keys)',
            'difficulty': 'Select map difficulty level',
            'algorithm': 'Choose pathfinding algorithm for auto-solve',
            'ml_heuristic': 'Use experimental ML-style heuristic (may be non-admissible)',
            'parallel_search': 'Run multiple strategies in parallel and pick fastest result',
            'solver_comparison': 'Run a comparison of available solvers and report metrics',
            'dstar_lite': 'Enable D* Lite incremental replanning (if implemented)',
            'show_topology': 'Draw room nodes & edges from topology graph on the map',
        }
        
        # Check which widget is under mouse
        for widget in self.widget_manager.widgets:
            # When panel is scrolled, translate mouse into scrolled coords for hit tests
            test_pos = mouse_pos
            header_height = self.font.get_height() + 12
            if getattr(self, 'control_panel_can_scroll', False) and getattr(self, 'control_panel_rect', None) and self.control_panel_rect.collidepoint(mouse_pos):
                # Only translate if mouse is below the fixed header area
                panel_top = self.control_panel_rect.y
                if mouse_pos[1] > panel_top + header_height:
                    test_pos = (mouse_pos[0], mouse_pos[1] + getattr(self, 'control_panel_scroll', 0))
            if hasattr(widget, 'rect') and widget.rect.collidepoint(test_pos) and widget.rect.y >= self.control_panel_rect.y + header_height:
                # Get tooltip text
                tooltip_text = None
                if hasattr(widget, 'flag_name') and widget.flag_name in tooltips:
                    tooltip_text = tooltips[widget.flag_name]
                elif hasattr(widget, 'control_name') and widget.control_name in tooltips:
                    tooltip_text = tooltips[widget.control_name]
                elif isinstance(widget, ButtonWidget) and hasattr(widget, 'label'):
                    # Button tooltips
                    button_tooltips = {
                        'Start Auto-Solve': 'Begin automatic pathfinding solution (SPACE)',
                        'Stop': 'Stop the current auto-solve operation',
                        'Generate Dungeon': 'Create a new random dungeon map',
                        'Reset': 'Reset current map to initial state (R key)',
                        'Path Preview': 'Preview the complete solution path',
                        'Clear Path': 'Clear the displayed path overlay',
                        'Export Route': 'Save current path to file',
                        'Load Route': 'Load path from file',
                    }
                    tooltip_text = button_tooltips.get(widget.label)
                
                if tooltip_text:
                    self._draw_tooltip(surface, mouse_pos, tooltip_text)
                break
    
    def _draw_tooltip(self, surface, pos, text):
        """Draw a tooltip box at the specified position."""
        font = pygame.font.Font(None, 18)
        padding = 8
        
        # Render text
        text_surf = font.render(text, True, (255, 255, 255))
        text_rect = text_surf.get_rect()
        
        # Calculate tooltip position (offset from mouse)
        tooltip_x = pos[0] + 15
        tooltip_y = pos[1] + 15
        
        # Keep tooltip on screen
        if tooltip_x + text_rect.width + padding * 2 > self.screen_w:
            tooltip_x = pos[0] - text_rect.width - padding * 2 - 15
        if tooltip_y + text_rect.height + padding * 2 > self.screen_h:
            tooltip_y = pos[1] - text_rect.height - padding * 2 - 15
        
        # Draw background
        bg_rect = pygame.Rect(
            tooltip_x - padding,
            tooltip_y - padding,
            text_rect.width + padding * 2,
            text_rect.height + padding * 2
        )
        bg_surface = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
        bg_surface.fill((50, 50, 60, 240))
        surface.blit(bg_surface, bg_rect.topleft)
        
        # Draw border
        pygame.draw.rect(surface, (100, 150, 200), bg_rect, 2)
        
        # Draw text
        surface.blit(text_surf, (tooltip_x, tooltip_y))
    
    def _handle_control_panel_click(self, pos, button, event_type='down'):
        """Handle mouse clicks on control panel widgets."""
        if not self.control_panel_enabled or not self.widget_manager:
            return False
        
        if event_type == 'down':
            # If the panel is currently being scrolled (dragging or inertia), don't forward clicks to widgets
            if (getattr(self, 'control_panel_scroll_dragging', False) or time.time() < getattr(self, 'control_panel_ignore_click_until', 0.0)):
                return True
            # Translate mouse pos into scrolled widget coordinates when panel is scrolled
            if getattr(self, 'control_panel_can_scroll', False) and getattr(self, 'control_panel_rect', None) and self.control_panel_rect.collidepoint(pos):
                header_height = 45
                panel_top = self.control_panel_rect.y
                # Only translate clicks that occur below the fixed header area
                if pos[1] > panel_top + header_height:
                    sc_pos = (pos[0], pos[1] + getattr(self, 'control_panel_scroll', 0))
                else:
                    sc_pos = pos
            else:
                sc_pos = pos
            handled = self.widget_manager.handle_mouse_down(sc_pos, button)
            if handled:
                # Update feature flags from checkboxes
                for widget in self.widget_manager.widgets:
                    if isinstance(widget, CheckboxWidget) and hasattr(widget, 'flag_name'):
                        old_value = self.feature_flags.get(widget.flag_name, False)
                        self.feature_flags[widget.flag_name] = widget.checked
                        
                        # Apply flags immediately with visual feedback
                        if widget.flag_name == 'show_heatmap' and old_value != widget.checked:
                            self.show_heatmap = widget.checked
                            if self.renderer:
                                self.renderer.show_heatmap = widget.checked
                            self._set_message(f"Heatmap: {'ON' if widget.checked else 'OFF'}")
                        elif widget.flag_name == 'show_minimap':
                            self.show_minimap = widget.checked
                            self._set_message(f"Minimap: {'ON' if widget.checked else 'OFF'}")
                        elif widget.flag_name == 'force_grid' and old_value != widget.checked:
                            # Special handling for force-grid toggle
                            self.force_grid_algorithm = widget.checked
                            self._set_message(f"Force Grid Solver: {'ON' if widget.checked else 'OFF'}")
                        elif old_value != widget.checked:
                            # Generic feedback for other toggles
                            status = 'enabled' if widget.checked else 'disabled'
                            self._set_message(f"{widget.label}: {status}")
                    
                    # Handle dropdown selections
                    elif isinstance(widget, DropdownWidget) and hasattr(widget, 'control_name'):
                        if widget.control_name == 'floor':
                            self.current_floor = widget.selected + 1
                            self.message = f"Floor {self.current_floor} selected"
                        elif widget.control_name == 'zoom':
                            old_zoom_idx = self.zoom_level_idx
                            self.zoom_level_idx = widget.selected
                            if old_zoom_idx != self.zoom_level_idx:
                                # Map zoom_level_idx to actual zoom: [25%, 50%, 75%, 100%, 150%, 200%]
                                zoom_map = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3, 5: 4}  # Map to ZOOM_LEVELS indices
                                if self.zoom_level_idx < len(zoom_map):
                                    new_zoom_idx = zoom_map.get(self.zoom_level_idx, 2)
                                    if new_zoom_idx != self.zoom_idx:
                                        self.zoom_idx = new_zoom_idx
                                        self.TILE_SIZE = self.ZOOM_LEVELS[self.zoom_idx]
                                        self._load_assets()
                                        self._center_view()
                                        zoom_labels = ["25%", "50%", "75%", "100%", "150%", "200%"]
                                        self.message = f"Zoom: {zoom_labels[self.zoom_level_idx]}"
                        elif widget.control_name == 'difficulty':
                            self.difficulty_idx = widget.selected
                            difficulty_names = ["Easy", "Medium", "Hard", "Expert"]
                            self.message = f"Difficulty: {difficulty_names[self.difficulty_idx]}"
                        elif widget.control_name == 'algorithm':
                            old_algorithm_idx = self.algorithm_idx
                            self.algorithm_idx = widget.selected
                            if old_algorithm_idx != self.algorithm_idx:
                                algorithm_names = ["A*", "BFS", "Dijkstra", "Greedy", "D* Lite"]
                                self.message = f"Solver: {algorithm_names[self.algorithm_idx]}"
                                # Clear any existing path when algorithm changes
                                if self.auto_path:
                                    self.auto_path = []
                                    self.auto_mode = False
                        elif widget.control_name == 'presets':
                            old = self.current_preset_idx
                            self.current_preset_idx = widget.selected
                            if old != self.current_preset_idx:
                                p = self.presets[self.current_preset_idx]
                                # Apply simple presets
                                if p == 'Debugging':
                                    self.feature_flags['show_heatmap'] = True
                                    self.feature_flags['solver_comparison'] = False
                                    self.feature_flags['ml_heuristic'] = False
                                elif p == 'Fast Approx':
                                    self.feature_flags['ml_heuristic'] = True
                                    self.feature_flags['parallel_search'] = True
                                elif p == 'Optimal':
                                    self.feature_flags['ml_heuristic'] = False
                                    self.feature_flags['parallel_search'] = False
                                elif p == 'Speedrun':
                                    self.feature_flags['speedrun_mode'] = True
                                    self.feature_flags['ml_heuristic'] = True
                                self._set_message(f"Preset applied: {p}")

            return handled
        elif event_type == 'up':
            # Translate pos like we do for mouse-down when scrolled
            if getattr(self, 'control_panel_can_scroll', False) and getattr(self, 'control_panel_rect', None) and self.control_panel_rect.collidepoint(pos):
                sc_pos = (pos[0], pos[1] + getattr(self, 'control_panel_scroll', 0))
            else:
                sc_pos = pos
            return self.widget_manager.handle_mouse_up(sc_pos, button)
        return False
    
    # Button callbacks
    def _stop_auto_solve(self):
        """Stop auto-solve."""
        self.auto_mode = False
        self.message = "Auto-solve stopped"
    
    def _generate_dungeon(self):
        """Generate a new random dungeon (placeholder)."""
        self.message = "Dungeon generation not implemented yet"
    
    def _reset_map(self):
        """Reset the current map."""
        self._load_current_map()
        self._center_view()
        if self.effects:
            self.effects.clear()
        self.step_count = 0
        self.message = "Map Reset"
    
    def _show_path_preview(self):
        """Show path preview dialog (placeholder)."""
        self.message = "Path preview not available"
    
    def _clear_path(self):
        """Clear the current path."""
        self.auto_path = []
        self.auto_mode = False
        self.message = "Path cleared"
    
    def _export_route(self):
        """Export the current route (placeholder)."""
        self.message = "Route export not implemented yet"
    
    def _load_route(self):
        """Load a saved route (placeholder)."""
        self.message = "Route loading not implemented yet"

    def load_visual_assets(self, templates_dir: str = None, link_sprite_path: str = None):
        """Optional: override GUI assets with extracted visual tiles/sprites.

        Usage (copy-paste into startup code):
            gui = ZeldaGUI(maps)
            gui.load_visual_assets('data/tileset.png', 'data/link_sprite.png')

        Behaviour:
        - If `templates_dir` is a folder of tile images, create pygame surfaces from them
          and assign to `self.images` keyed by semantic id (best-effort).
        - If `link_sprite_path` is provided, attempt to cut a Link sprite and replace `self.link_img`.
        """
        try:
            from src.data_processing.visual_extractor import extract_grid
            from PIL import Image
        except Exception:
            return

        if templates_dir and os.path.isdir(templates_dir):
            # load any PNGs found and map into fallback image slots
            for fn in sorted(os.listdir(templates_dir)):
                if not fn.lower().endswith('.png'):
                    continue
                name = os.path.splitext(fn)[0]
                try:
                    im = Image.open(os.path.join(templates_dir, fn)).convert('RGBA')
                    surf = pygame.image.fromstring(im.tobytes(), im.size, im.mode)
                    # best-effort: if filename contains 'floor','wall','door','key' map to semantic ids
                    ln = name.lower()
                    if 'floor' in ln or 'f' == ln:
                        self.images[SEMANTIC_PALETTE['FLOOR']] = pygame.transform.scale(surf, (self.TILE_SIZE, self.TILE_SIZE))
                    if 'wall' in ln:
                        self.images[SEMANTIC_PALETTE['WALL']] = pygame.transform.scale(surf, (self.TILE_SIZE, self.TILE_SIZE))
                    if 'door' in ln:
                        self.images[SEMANTIC_PALETTE['DOOR_OPEN']] = pygame.transform.scale(surf, (self.TILE_SIZE, self.TILE_SIZE))
                    if 'key' in ln:
                        self.images[SEMANTIC_PALETTE['KEY']] = pygame.transform.scale(surf, (self.TILE_SIZE, self.TILE_SIZE))
                except Exception:
                    continue

        if link_sprite_path and os.path.exists(link_sprite_path):
            try:
                im = Image.open(link_sprite_path).convert('RGBA')
                im = im.resize((self.TILE_SIZE - 4, self.TILE_SIZE - 4), Image.NEAREST)
                self.link_img = pygame.image.fromstring(im.tobytes(), im.size, im.mode)
            except Exception as e:
                print(f"Warning: failed to load link sprite from {link_sprite_path}: {e}")
        
        print(f"Loaded {len([k for k in self.images if k in (1,2,10,12)])} visual assets")
        return True

    def load_visual_map(self, image_path: str, templates_dir: str | None = None):
        """Public API: create a GUI map from a screenshot and switch to it.

        - `image_path` can be a full screenshot (HUD allowed).
        - `templates_dir` is passed to the visual extractor (tileset or folder).

        This method is intentionally permissive and returns a bool for success
        so automated tests can call it without a file dialog.
        """
        try:
            from src.data_processing.visual_integration import visual_extract_to_room, make_stitched_for_single_room, infer_inventory_from_room
        except Exception:
            return False

        try:
            ids, conf = visual_extract_to_room(image_path, templates_dir or '')
            # convert single-room semantic -> environment grid expected by GUI
            stitched = make_stitched_for_single_room(ids)
            self.maps = [stitched.global_grid]
            self.current_map_idx = 0
            self._load_current_map()
            self.message = f"Loaded visual map: {image_path}"
            return True
        except Exception as e:
            self.message = f"Visual load failed: {e}"
            return False

    def _load_current_map(self):
        """Load and initialize the current map."""
        current_dungeon = self.maps[self.current_map_idx]
        # Extract grid and graph info from StitchedDungeon
        if hasattr(current_dungeon, 'global_grid'):
            # It's a StitchedDungeon object
            grid = current_dungeon.global_grid
            graph = current_dungeon.graph
            room_to_node = current_dungeon.room_to_node
            room_positions = current_dungeon.room_positions
        else:
            # Legacy: just a grid array
            grid = current_dungeon
            graph = None
            room_to_node = None
            room_positions = None
        
        self.env = ZeldaLogicEnv(grid, render_mode=False, graph=graph, 
                                  room_to_node=room_to_node, room_positions=room_positions)
        self.solver = StateSpaceAStar(self.env)
        self.auto_path = []
        self.auto_step_idx = 0
        self.auto_mode = False
        
        # Clear solver result when loading new map
        self.solver_result = None
        self.current_keys_held = 0
        self.current_keys_used = 0
        self.current_edge_types = []
        
        # Count total items in dungeon for "X/Y collected" display
        self.total_keys = len(self.env._find_all_positions(SEMANTIC_PALETTE['KEY_SMALL']))
        # Bombs are boolean in this game (has_bomb state), not collectible items
        # Check if dungeon requires bombs by looking for bomb doors
        bomb_doors = self.env._find_all_positions(SEMANTIC_PALETTE['DOOR_BOMB'])
        self.total_bombs = 1 if len(bomb_doors) > 0 else 0
        self.total_boss_keys = len(self.env._find_all_positions(SEMANTIC_PALETTE['KEY_BOSS']))
        self.keys_collected = 0
        self.bombs_collected = 0
        self.boss_keys_collected = 0
        
        # Clear search heatmap
        self.search_heatmap = {}
        
        # Initialize renderer agent position
        if self.renderer and self.env.start_pos:
            self.renderer.set_agent_position(
                self.env.start_pos[0],
                self.env.start_pos[1],
                immediate=True
            )
        
        # Run sanity check
        checker = SanityChecker(grid)
        is_valid, errors = checker.check_all()
        
        if not is_valid:
            self.message = f"Map Error: {errors[0]}"
        else:
            self.message = f"Map {self.current_map_idx + 1}/{len(self.maps)} - Press SPACE to solve"
        
        # Auto-fit zoom to show entire map
        self._auto_fit_zoom()
    
    def _center_view(self):
        """Center the current map in the view."""
        if self.env is None:
            return
        map_w = self.env.width * self.TILE_SIZE
        map_h = self.env.height * self.TILE_SIZE
        view_w = self.screen_w - self.SIDEBAR_WIDTH
        view_h = self.screen_h - self.HUD_HEIGHT
        self.view_offset_x = max(0, (map_w - view_w) // 2)
        self.view_offset_y = max(0, (map_h - view_h) // 2)
    
    def _auto_fit_zoom(self):
        """Automatically set zoom level to fit the entire map in view."""
        if self.env is None:
            return
        
        view_w = self.screen_w - self.SIDEBAR_WIDTH - 20  # padding
        view_h = self.screen_h - self.HUD_HEIGHT - 20
        
        map_h = self.env.height
        map_w = self.env.width
        
        # Find the largest zoom level that fits
        best_zoom_idx = 0
        for idx, tile_size in enumerate(self.ZOOM_LEVELS):
            if map_w * tile_size <= view_w and map_h * tile_size <= view_h:
                best_zoom_idx = idx
            else:
                break
        
        # Apply the zoom
        if best_zoom_idx != self.zoom_idx:
            self.zoom_idx = best_zoom_idx
            self.TILE_SIZE = self.ZOOM_LEVELS[self.zoom_idx]
            self._load_assets()
            # Update renderer tile size
            if self.renderer:
                self.renderer.set_tile_size(self.TILE_SIZE)
        
        self._center_view()
        
        # Initialize agent position in renderer
        if self.renderer and self.env and self.env.start_pos:
            self.renderer.set_agent_position(
                self.env.start_pos[0], 
                self.env.start_pos[1], 
                immediate=True
            )
    
    def _change_zoom(self, delta: int):
        """Change zoom level by delta steps."""
        old_idx = self.zoom_idx
        self.zoom_idx = max(0, min(len(self.ZOOM_LEVELS) - 1, self.zoom_idx + delta))
        if self.zoom_idx != old_idx:
            self.TILE_SIZE = self.ZOOM_LEVELS[self.zoom_idx]
            self._load_assets()  # Reload assets at new size
            # Update renderer tile size
            if self.renderer:
                self.renderer.set_tile_size(self.TILE_SIZE)
            self._center_view()
            self.message = f"Zoom: {self.TILE_SIZE}px"
    
    def _toggle_fullscreen(self):
        """Toggle fullscreen mode."""
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            self.screen_w, self.screen_h = self.screen.get_size()
        else:
            self.screen = pygame.display.set_mode(
                (800, 600), pygame.RESIZABLE
            )
            self.screen_w, self.screen_h = 800, 600
        self._center_view()

    # ------------------ Control Panel Animation ------------------
    def _start_toggle_panel_animation(self, target_collapsed: bool):
        """Begin animated transition to collapsed or expanded state."""
        collapsed_width = 40
        # Compute target width
        target_width = float(collapsed_width) if target_collapsed else float(self.control_panel_width)
        # Initialize animation state
        self.control_panel_anim_from = float(self.control_panel_width_current)
        self.control_panel_anim_to = float(target_width)
        self.control_panel_anim_start = time.time()
        self.control_panel_anim_duration = 0.22
        self.control_panel_target_collapsed = target_collapsed
        self.control_panel_animating = True

    def _update_control_panel_animation(self):
        """Update animation state; should be called each frame."""
        if not getattr(self, 'control_panel_animating', False):
            return
        # Keep widget positions up to date during animation to maintain correct hitboxes
        try:
            self._update_control_panel_positions()
        except Exception:
            pass
        elapsed = time.time() - self.control_panel_anim_start
        t = min(1.0, elapsed / max(1e-6, self.control_panel_anim_duration))
        # Smoothstep easing
        ease = t * t * (3 - 2 * t)
        self.control_panel_width_current = self.control_panel_anim_from + (self.control_panel_anim_to - self.control_panel_anim_from) * ease
        # If finished
        if t >= 1.0:
            self.control_panel_animating = False
            self.control_panel_width_current = self.control_panel_anim_to
            # Apply final collapsed flag
            self.control_panel_collapsed = bool(self.control_panel_target_collapsed)
            # Ensure widgets are (re)initialized / repositioned after animation completes
            try:
                if not self.control_panel_collapsed:
                    self._update_control_panel_positions()
                self._set_message(f"Panel: {'collapsed' if self.control_panel_collapsed else 'expanded'}")
            except Exception:
                # Avoid raising during animation cleanup
                pass

    def _update_control_panel_scroll(self):
        """Per-frame update that applies inertia (momentum) and clamps scroll."""
        if not getattr(self, 'control_panel_can_scroll', False):
            return
        # If user is actively dragging the thumb, don't apply inertia
        if getattr(self, 'control_panel_scroll_dragging', False):
            return
        vel = getattr(self, 'control_panel_scroll_velocity', 0.0)
        # Nothing to do if no velocity
        if abs(vel) < 1.0:
            self.control_panel_scroll_velocity = 0.0
            return
        # Advance scroll by velocity (pixels per second) scaled by delta_time
        prev = self.control_panel_scroll
        self.control_panel_scroll = max(0, min(getattr(self, 'control_panel_scroll_max', 0), self.control_panel_scroll + vel * max(1e-6, self.delta_time)))
        # If hit bounds, zero velocity
        if self.control_panel_scroll <= 0 or self.control_panel_scroll >= getattr(self, 'control_panel_scroll_max', 0):
            self.control_panel_scroll_velocity = 0.0
        else:
            # Apply simple linear damping per second
            damping = getattr(self, 'control_panel_scroll_damping', 6.0)
            self.control_panel_scroll_velocity *= max(0.0, 1.0 - damping * self.delta_time)
            if abs(self.control_panel_scroll_velocity) < 1.0:
                self.control_panel_scroll_velocity = 0.0
        # Set a short ignore window to avoid accidental toggles while momentum is active
        if abs(self.control_panel_scroll - prev) > 0.5:
            self.control_panel_ignore_click_until = time.time() + 0.12

    def run(self):
        """Main game loop with delta-time support."""
        running = True
        
        while running:
            # Calculate delta time for smooth animations
            current_time = time.time()
            self.delta_time = current_time - self.last_frame_time
            self.last_frame_time = current_time
            
            # Handle events
            for event in pygame.event.get():
                # Developer debug overlay toggle
                if event.type == pygame.KEYDOWN and event.key == pygame.K_F12:
                    self.debug_overlay_enabled = not getattr(self, 'debug_overlay_enabled', False)
                    if self.debug_overlay_enabled:
                        self._set_message('Debug overlay ON (F12 to toggle)')
                    else:
                        self._set_message('Debug overlay OFF')
                    continue
                # Page Up / Page Down to scroll control panel when visible and hovered
                if event.type == pygame.KEYDOWN and event.key in (pygame.K_PAGEUP, pygame.K_PAGEDOWN):
                    if self.control_panel_enabled and getattr(self, 'control_panel_can_scroll', False) and getattr(self, 'control_panel_rect', None) and self.control_panel_rect.collidepoint(pygame.mouse.get_pos()) and not self.control_panel_collapsed:
                        # Page amount: visible content height (excluding header area)
                        page_amount = max(1, self.control_panel_rect.height - 32)
                        if event.key == pygame.K_PAGEUP:
                            self.control_panel_scroll = max(0, int(self.control_panel_scroll - page_amount))
                        else:
                            self.control_panel_scroll = min(getattr(self, 'control_panel_scroll_max', 0), int(self.control_panel_scroll + page_amount))
                        # Stop any momentum when keyboard-driven
                        self.control_panel_scroll_velocity = 0.0
                        self.control_panel_ignore_click_until = time.time() + 0.12
                        continue
                if event.type == pygame.KEYDOWN and event.key == pygame.K_F11 and (pygame.key.get_mods() & pygame.KMOD_SHIFT):
                    # Shift-F11 clears debug click log
                    self.debug_click_log = []
                    self._set_message('Debug log cleared')
                    continue
                # Handle path preview dialog input first (if active)
                # If a non-modal overlay is active, handle its quick interactions (start/dismiss)
                if getattr(self, 'preview_overlay_visible', False) and (self.path_preview_dialog or getattr(self, 'auto_path', None)):
                    # Keyboard shortcuts for non-modal overlay
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            # Dismiss overlay (keep planned path stored but hide overlay)
                            self.preview_overlay_visible = False
                            self.path_preview_dialog = None
                            self.message = "Path preview dismissed"
                            continue
                        if event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                            # Start auto-solve from overlay
                            self._execute_auto_solve_from_preview()
                            continue
                    # Mouse click on sidebar buttons handled further down when button rects exist

                if self.path_preview_mode and self.path_preview_dialog:
                    result = self.path_preview_dialog.handle_input(event)
                    if result == 'start':
                        # User confirmed - start auto-solve
                        self._execute_auto_solve_from_preview()
                        continue
                    elif result == 'cancel':
                        # User cancelled - switch to non-modal overlay (keep planned path visible)
                        self.path_preview_mode = False
                        # Keep the dialog instance for overlay rendering if available
                        self.preview_overlay_visible = True
                        self.message = "Path preview closed; overlay visible in sidebar/map (Enter to start or Esc to dismiss)"
                        continue

                # Dismiss solver comparison overlay with Esc
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE and getattr(self, 'show_solver_comparison_overlay', False):
                    self.show_solver_comparison_overlay = False
                    self._set_message('Solver comparison closed', 1.2)
                    continue
                
                # If non-modal overlay visible and clicked in sidebar buttons, handle them
                if getattr(self, 'preview_overlay_visible', False) and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mouse_pos = event.pos
                    if getattr(self, 'sidebar_start_button_rect', None) and self.sidebar_start_button_rect.collidepoint(mouse_pos):
                        self._execute_auto_solve_from_preview()
                        continue
                    if getattr(self, 'sidebar_dismiss_button_rect', None) and self.sidebar_dismiss_button_rect.collidepoint(mouse_pos):
                        self.preview_overlay_visible = False
                        self.path_preview_dialog = None
                        self.message = "Path preview dismissed"
                        continue

                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.VIDEORESIZE:
                    # Handle window resize
                    self.screen_w = max(event.w, self.MIN_WIDTH)
                    self.screen_h = max(event.h, self.MIN_HEIGHT)
                    if not self.fullscreen:
                        self.screen = pygame.display.set_mode(
                            (self.screen_w, self.screen_h), pygame.RESIZABLE
                        )
                    # Update control panel widget positions
                    if self.control_panel_enabled:
                        self._update_control_panel_positions()
                
                elif event.type == pygame.MOUSEWHEEL:
                    mouse_pos = pygame.mouse.get_pos()
                    # If mouse is over control panel and scrolling is enabled, apply momentum to panel
                    if self.control_panel_enabled and getattr(self, 'control_panel_can_scroll', False) and getattr(self, 'control_panel_rect', None) and self.control_panel_rect.collidepoint(mouse_pos) and not self.control_panel_collapsed:
                        # Use wheel to add velocity (pixels per second)
                        wheel_power = getattr(self, 'control_panel_scroll_step', 20) * 12
                        # Negative event.y means scroll down? We want positive y to scroll up (decrease coord)
                        self.control_panel_scroll_velocity += -event.y * wheel_power
                        # Clamp velocity to reasonable bounds
                        max_v = 2000
                        self.control_panel_scroll_velocity = max(-max_v, min(max_v, self.control_panel_scroll_velocity))
                        # Ignore immediate clicks while momentum is active
                        self.control_panel_ignore_click_until = time.time() + 0.12
                    else:
                        # Zoom with mouse wheel when not over panel
                        self._change_zoom(event.y)
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    # Record click to debug log
                    if getattr(self, 'debug_click_log', None) is not None:
                        self.debug_click_log.insert(0, (mouse_pos, time.time()))
                        # Keep bounded history
                        if len(self.debug_click_log) > 50:
                            self.debug_click_log.pop()
                    
                    # Handle collapse button click first (animated)
                    if self.control_panel_enabled and self.collapse_button_rect and self.collapse_button_rect.collidepoint(mouse_pos):
                        # Ignore input if animation already running
                        if not getattr(self, 'control_panel_animating', False):
                            target_collapsed = not self.control_panel_collapsed
                            self._start_toggle_panel_animation(target_collapsed)
                        continue
                    
                    # Check if starting to drag panel (click on title bar area)
                    if self.control_panel_enabled and self.control_panel_rect and not self.control_panel_collapsed:
                        # Check if clicking on scrollbar thumb to start drag
                        if event.button == 1 and getattr(self, 'control_panel_scroll_thumb_rect', None) and self.control_panel_scroll_thumb_rect.collidepoint(mouse_pos):
                            self.control_panel_scroll_dragging = True
                            self.control_panel_scroll_drag_offset = mouse_pos[1] - self.control_panel_scroll_thumb_rect.y
                            continue
                        # Clicking on track -> page to that location
                        if event.button == 1 and getattr(self, 'control_panel_scroll_track_rect', None) and self.control_panel_scroll_track_rect.collidepoint(mouse_pos):
                            tr = self.control_panel_scroll_track_rect
                            rel = mouse_pos[1] - tr.y
                            max_move = tr.height - getattr(self, 'control_panel_scroll_thumb_rect', pygame.Rect(0,0,0,20)).height
                            ratio = max(0.0, min(1.0, rel / tr.height))
                            self.control_panel_scroll = int(ratio * getattr(self, 'control_panel_scroll_max', 0))
                            continue

                        title_bar_height = 45
                        title_bar_rect = pygame.Rect(
                            self.control_panel_rect.x,
                            self.control_panel_rect.y,
                            self.control_panel_rect.width,
                            title_bar_height
                        )
                        if title_bar_rect.collidepoint(mouse_pos) and not self.collapse_button_rect.collidepoint(mouse_pos):
                            self.dragging_panel = True
                            self.drag_panel_offset = (mouse_pos[0] - self.control_panel_rect.x, mouse_pos[1] - self.control_panel_rect.y)
                            continue
                        
                        # Check if starting to resize panel (near edges)
                        edge_threshold = 8
                        mx, my = mouse_pos
                        rect = self.control_panel_rect
                        
                        if abs(mx - rect.left) < edge_threshold and rect.top <= my <= rect.bottom:
                            self.resizing_panel = True
                            self.resize_edge = 'left'
                            continue
                        elif abs(mx - rect.right) < edge_threshold and rect.top <= my <= rect.bottom:
                            self.resizing_panel = True
                            self.resize_edge = 'right'
                            continue
                        elif abs(my - rect.bottom) < edge_threshold and rect.left <= mx <= rect.right:
                            self.resizing_panel = True
                            self.resize_edge = 'bottom'
                            continue
                    
                    # Handle control panel clicks
                    if self.control_panel_enabled and self._handle_control_panel_click(mouse_pos, event.button, 'down'):
                        continue  # Control panel handled the click
                    
                    if event.button == 1:  # Left click - check minimap and start map drag if on map
                        if self._handle_minimap_click(mouse_pos):
                            pass  # Minimap click handled
                        else:
                            # Start map drag with left button when clicking on the main map area (not on sidebar or panel)
                            sidebar_x = self.screen_w - self.SIDEBAR_WIDTH
                            if mouse_pos[0] < sidebar_x and not (self.control_panel_enabled and getattr(self, 'control_panel_rect', None) and self.control_panel_rect.collidepoint(mouse_pos)):
                                self.dragging = True
                                self.dragging_button = 1
                                self.drag_start = event.pos
                    elif event.button == 2:  # Middle mouse
                        self.dragging = True
                        self.dragging_button = 2
                        self.drag_start = event.pos
                
                elif event.type == pygame.MOUSEBUTTONUP:
                    mouse_pos = pygame.mouse.get_pos()
                    # Stop dragging/resizing panel
                    if self.dragging_panel:
                        self.dragging_panel = False
                    if self.resizing_panel:
                        self.resizing_panel = False
                        self.resize_edge = None
                    # Stop scrollbar drag if active
                    if getattr(self, 'control_panel_scroll_dragging', False):
                        self.control_panel_scroll_dragging = False
                    # Handle control panel clicks
                    if self.control_panel_enabled and self._handle_control_panel_click(mouse_pos, event.button, 'up'):
                        continue
                    
                    if event.button == 2 or (hasattr(self, 'dragging_button') and event.button == getattr(self, 'dragging_button')):
                        self.dragging = False
                        self.dragging_button = None
                
                elif event.type == pygame.MOUSEMOTION:
                    mouse_pos = event.pos
                    
                    # Handle panel dragging
                    if self.dragging_panel:
                        self.control_panel_x = mouse_pos[0] - self.drag_panel_offset[0]
                        self.control_panel_y = mouse_pos[1] - self.drag_panel_offset[1]
                        # Clamp to screen bounds
                        self.control_panel_x = max(0, min(self.control_panel_x, self.screen_w - self.control_panel_width))
                        self.control_panel_y = max(0, min(self.control_panel_y, self.screen_h - 100))
                        # Update widget positions to follow panel
                        self._reposition_widgets(self.control_panel_x, self.control_panel_y)
                    
                    # Handle panel resizing
                    elif self.resizing_panel and self.control_panel_rect:
                        if self.resize_edge == 'left':
                            old_right = self.control_panel_rect.right
                            new_x = mouse_pos[0]
                            new_width = old_right - new_x
                            if self.min_panel_width <= new_width <= self.max_panel_width:
                                self.control_panel_width = new_width
                                self.control_panel_x = new_x
                        elif self.resize_edge == 'right':
                            new_width = mouse_pos[0] - self.control_panel_rect.x
                            if self.min_panel_width <= new_width <= self.max_panel_width:
                                self.control_panel_width = new_width
                        elif self.resize_edge == 'bottom':
                            new_height = mouse_pos[1] - self.control_panel_rect.y
                            if self.min_panel_height <= new_height <= self.screen_h - self.control_panel_rect.y - 20:
                                pass  # Height is auto-calculated, just update visual feedback
                    
                    # Handle scrollbar thumb dragging
                    elif getattr(self, 'control_panel_scroll_dragging', False) and getattr(self, 'control_panel_can_scroll', False) and getattr(self, 'control_panel_scroll_track_rect', None):
                        track_rect = self.control_panel_scroll_track_rect
                        thumb_rect = getattr(self, 'control_panel_scroll_thumb_rect', None)
                        if thumb_rect is None:
                            continue
                        # Compute local mouse position inside track
                        rel_y = mouse_pos[1] - track_rect.y
                        max_move = track_rect.height - thumb_rect.height
                        new_thumb_top = max(0, min(rel_y - getattr(self, 'control_panel_scroll_drag_offset', 0), max_move))
                        if max_move > 0:
                            ratio = new_thumb_top / max_move
                            self.control_panel_scroll = int(ratio * getattr(self, 'control_panel_scroll_max', 0))
                            # Clamp
                            self.control_panel_scroll = max(0, min(self.control_panel_scroll, getattr(self, 'control_panel_scroll_max', 0)))
                    
                    # Update cursor for resize edges (when not dragging)
                    elif self.control_panel_enabled and self.control_panel_rect and not self.control_panel_collapsed:
                        edge_threshold = 8
                        mx, my = mouse_pos
                        rect = self.control_panel_rect
                        
                        if abs(mx - rect.left) < edge_threshold and rect.top <= my <= rect.bottom:
                            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_SIZEWE)
                        elif abs(mx - rect.right) < edge_threshold and rect.top <= my <= rect.bottom:
                            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_SIZEWE)
                        elif abs(my - rect.bottom) < edge_threshold and rect.left <= mx <= rect.right:
                            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_SIZENS)
                        else:
                            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)
                    elif self.dragging:
                        dx = self.drag_start[0] - event.pos[0]
                        dy = self.drag_start[1] - event.pos[1]
                        self.view_offset_x += dx
                        self.view_offset_y += dy
                        self.drag_start = event.pos
                        self._clamp_view_offset()
                    elif self.dragging:
                        dx = self.drag_start[0] - event.pos[0]
                        dy = self.drag_start[1] - event.pos[1]
                        self.view_offset_x += dx
                        self.view_offset_y += dy
                        self.drag_start = event.pos
                        self._clamp_view_offset()

                    # Update cursor for resize edges (when not dragging)
                    elif self.control_panel_enabled and self.control_panel_rect and not self.control_panel_collapsed:
                        edge_threshold = 8
                        mx, my = mouse_pos
                        rect = self.control_panel_rect
                        
                        if abs(mx - rect.left) < edge_threshold and rect.top <= my <= rect.bottom:
                            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_SIZEWE)
                        elif abs(mx - rect.right) < edge_threshold and rect.top <= my <= rect.bottom:
                            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_SIZEWE)
                        elif abs(my - rect.bottom) < edge_threshold and rect.left <= mx <= rect.right:
                            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_SIZENS)
                        else:
                            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)
                    
                    # Update cursor for resize edges (when not dragging)
                    elif self.control_panel_enabled and self.control_panel_rect and not self.control_panel_collapsed:
                        edge_threshold = 8
                        mx, my = mouse_pos
                        rect = self.control_panel_rect
                        
                        if abs(mx - rect.left) < edge_threshold and rect.top <= my <= rect.bottom:
                            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_SIZEWE)
                        elif abs(mx - rect.right) < edge_threshold and rect.top <= my <= rect.bottom:
                            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_SIZEWE)
                        elif abs(my - rect.bottom) < edge_threshold and rect.left <= mx <= rect.right:
                            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_SIZENS)
                        else:
                            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)
                    elif self.dragging:
                        dx = self.drag_start[0] - event.pos[0]
                        dy = self.drag_start[1] - event.pos[1]
                        self.view_offset_x += dx
                        self.view_offset_y += dy
                        self.drag_start = event.pos
                        # Clamp offsets
                        self._clamp_view_offset()
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.fullscreen:
                            self._toggle_fullscreen()
                        else:
                            running = False
                    
                    elif event.key == pygame.K_F11:
                        self._toggle_fullscreen()
                    
                    elif event.key == pygame.K_h:
                        # Toggle heatmap overlay (H key)
                        if not self.show_help:  # Don't toggle if help shown
                            self.show_heatmap = not self.show_heatmap
                            self.feature_flags['show_heatmap'] = self.show_heatmap
                            if self.renderer:
                                self.renderer.show_heatmap = self.show_heatmap
                            # Update checkbox widget if available
                            if self.widget_manager:
                                for widget in self.widget_manager.widgets:
                                    if isinstance(widget, CheckboxWidget) and hasattr(widget, 'flag_name') and widget.flag_name == 'show_heatmap':
                                        widget.checked = self.show_heatmap
                            self.message = f"Heatmap: {'ON' if self.show_heatmap else 'OFF'}"
                    
                    elif event.key == pygame.K_F1:
                        self.show_help = not self.show_help
                    
                    elif event.key == pygame.K_TAB:
                        # Toggle control panel with Tab key
                        if self.control_panel_enabled:
                            # Animate toggle for Tab key as well
                            if not getattr(self, 'control_panel_animating', False):
                                target_collapsed = not self.control_panel_collapsed
                                self._start_toggle_panel_animation(target_collapsed)
                    
                    elif event.key == pygame.K_m:
                        # Toggle minimap
                        self.show_minimap = not self.show_minimap
                        self.feature_flags['show_minimap'] = self.show_minimap
                        # Update checkbox widget if available
                        if self.widget_manager:
                            for widget in self.widget_manager.widgets:
                                if isinstance(widget, CheckboxWidget) and hasattr(widget, 'flag_name') and widget.flag_name == 'show_minimap':
                                    widget.checked = self.show_minimap
                        self.message = f"Minimap: {'ON' if self.show_minimap else 'OFF'}"
                    
                    elif event.key == pygame.K_RIGHTBRACKET or event.key == pygame.K_PERIOD:
                        # Increase speed
                        self.speed_index = min(len(self.speed_levels) - 1, self.speed_index + 1)
                        self.speed_multiplier = self.speed_levels[self.speed_index]
                        self.message = f"Speed: {self.speed_multiplier}x"
                    
                    elif event.key == pygame.K_LEFTBRACKET or event.key == pygame.K_COMMA:
                        # Decrease speed
                        self.speed_index = max(0, self.speed_index - 1)
                        self.speed_multiplier = self.speed_levels[self.speed_index]
                        self.message = f"Speed: {self.speed_multiplier}x"
                    
                    elif event.key == pygame.K_SPACE:
                        self._start_auto_solve()
                    
                    elif event.key == pygame.K_r:
                        self._load_current_map()
                        self._center_view()
                        # Clear effects
                        if self.effects:
                            self.effects.clear()
                        # Reset step count
                        self.step_count = 0
                        self.message = "Map Reset"
                    
                    elif event.key == pygame.K_n:
                        self.current_map_idx = (self.current_map_idx + 1) % len(self.maps)
                        self._load_current_map()
                        self._center_view()
                        # Clear effects and reset step count
                        if self.effects:
                            self.effects.clear()
                        self.step_count = 0
                    
                    elif event.key == pygame.K_p:
                        self.current_map_idx = (self.current_map_idx - 1) % len(self.maps)
                        self._load_current_map()
                        self._center_view()
                        # Clear effects and reset step count
                        if self.effects:
                            self.effects.clear()
                        self.step_count = 0
                    
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        self._change_zoom(1)
                    
                    elif event.key == pygame.K_MINUS:
                        self._change_zoom(-1)
                    
                    elif event.key == pygame.K_0:
                        # Reset zoom to default
                        self.zoom_idx = self.DEFAULT_ZOOM_IDX
                        self.TILE_SIZE = self.ZOOM_LEVELS[self.zoom_idx]
                        self._load_assets()
                        self._center_view()
                        self.message = "Zoom reset to default"
                    
                    elif event.key == pygame.K_f:
                        # Auto-fit zoom to show entire map
                        self._auto_fit_zoom()
                        self.message = f"Auto-fit: {self.TILE_SIZE}px"
                    
                    elif event.key == pygame.K_c:
                        # Center view on player
                        self._center_on_player()
                    
                    elif event.key == pygame.K_l:
                        ok = self.load_visual_map(os.path.join(os.getcwd(), 'screenshot.png'))
                        if not ok:
                            self.message = "Failed to load ./screenshot.png"
                    
                    # Track key holds for continuous movement
                    elif event.key in self.keys_held and not self.auto_mode:
                        self.keys_held[event.key] = True
                        self.move_timer = 0.0  # Reset timer for immediate first move
                        
                    elif not self.auto_mode:
                        # Manual movement - check for diagonal combos first
                        keys = pygame.key.get_pressed()
                        action = None
                        
                        # Check diagonal combinations (two arrow keys pressed)
                        if keys[pygame.K_UP] and keys[pygame.K_LEFT]:
                            action = Action.UP_LEFT
                        elif keys[pygame.K_UP] and keys[pygame.K_RIGHT]:
                            action = Action.UP_RIGHT
                        elif keys[pygame.K_DOWN] and keys[pygame.K_LEFT]:
                            action = Action.DOWN_LEFT
                        elif keys[pygame.K_DOWN] and keys[pygame.K_RIGHT]:
                            action = Action.DOWN_RIGHT
                        # Cardinal directions (single key)
                        elif event.key == pygame.K_UP:
                            action = Action.UP
                        elif event.key == pygame.K_DOWN:
                            action = Action.DOWN
                        elif event.key == pygame.K_LEFT:
                            action = Action.LEFT
                        elif event.key == pygame.K_RIGHT:
                            action = Action.RIGHT
                        
                        if action is not None:
                            self._manual_step(action)
                            self._center_on_player()
            
            # Auto-solve stepping
            if self.auto_mode and not self.env.done:
                self._auto_step()
                self._center_on_player()
            
            # Update widget manager with mouse position
            if self.widget_manager:
                mouse_pos = pygame.mouse.get_pos()
                self.widget_manager.update(mouse_pos, self.delta_time)
            
            # Handle continuous movement (hold key to move) with diagonal support
            if not self.auto_mode and any(self.keys_held.values()):
                self.move_timer += self.delta_time
                if self.move_timer >= self.move_delay:
                    self.move_timer = 0.0
                    # Check for diagonal combinations FIRST (two keys held)
                    if self.keys_held[pygame.K_UP] and self.keys_held[pygame.K_LEFT]:
                        self._manual_step(Action.UP_LEFT)
                    elif self.keys_held[pygame.K_UP] and self.keys_held[pygame.K_RIGHT]:
                        self._manual_step(Action.UP_RIGHT)
                    elif self.keys_held[pygame.K_DOWN] and self.keys_held[pygame.K_LEFT]:
                        self._manual_step(Action.DOWN_LEFT)
                    elif self.keys_held[pygame.K_DOWN] and self.keys_held[pygame.K_RIGHT]:
                        self._manual_step(Action.DOWN_RIGHT)
                    # Cardinal directions (single key)
                    elif self.keys_held[pygame.K_UP]:
                        self._manual_step(Action.UP)
                    elif self.keys_held[pygame.K_DOWN]:
                        self._manual_step(Action.DOWN)
                    elif self.keys_held[pygame.K_LEFT]:
                        self._manual_step(Action.LEFT)
                    elif self.keys_held[pygame.K_RIGHT]:
                        self._manual_step(Action.RIGHT)
            
            # Update toast notifications
            self._update_toasts()

            # Update animated control panel state (if active)
            self._update_control_panel_animation()

            # Update control panel scroll inertia (momentum)
            self._update_control_panel_scroll()

            # If parallel search ran in background, handle result on main thread
            if getattr(self, 'parallel_search_done', False) and getattr(self, 'parallel_search_result', None):
                best = self.parallel_search_result
                # Convert alg index to name
                alg_names = ['A*','BFS','Dijkstra','Greedy']
                name = alg_names[best['alg']] if best['alg'] < len(alg_names) else f"Alg{best['alg']}"
                self._set_message(f"Parallel best: {name} ({best['nodes']} nodes, {best['time_ms']:.0f}ms)")
                self.parallel_search_done = False
                self.parallel_search_result = None
                # Use found path
                _handle_found_path = None
                try:
                    # Reuse the same handling as in _start_auto_solve: set auto_path and show preview
                    self.auto_path = best['path']
                    self.preview_overlay_visible = True
                    self.path_preview_dialog = PathPreviewDialog(path=self.auto_path, env=self.env, solver_result={}, speed_multiplier=self.speed_multiplier)
                    self._set_message('Parallel result ready (sidebar preview)')
                except Exception as e:
                    logger.warning(f"Failed to display parallel search preview: {e}")

            # Render
            self._render()

                # Log mouse clicks for debug overlay
                # Note: capture after render to show latest visual state
            
            
            # Cap framerate
            self.clock.tick(30 if not self.auto_mode else 10)
        
        pygame.quit()
    
    def _clamp_view_offset(self):
        """Clamp view offset to valid range."""
        if self.env is None:
            return
        map_w = self.env.width * self.TILE_SIZE
        map_h = self.env.height * self.TILE_SIZE
        view_w = self.screen_w - self.SIDEBAR_WIDTH
        view_h = self.screen_h - self.HUD_HEIGHT
        
        max_offset_x = max(0, map_w - view_w)
        max_offset_y = max(0, map_h - view_h)
        
        self.view_offset_x = max(0, min(self.view_offset_x, max_offset_x))
        self.view_offset_y = max(0, min(self.view_offset_y, max_offset_y))
    
    def _center_on_player(self):
        """Center the view on the player position."""
        if self.env is None:
            return
        r, c = self.env.state.position
        player_x = c * self.TILE_SIZE
        player_y = r * self.TILE_SIZE
        
        view_w = self.screen_w - self.SIDEBAR_WIDTH
        view_h = self.screen_h - self.HUD_HEIGHT
        
        self.view_offset_x = player_x - view_w // 2
        self.view_offset_y = player_y - view_h // 2
        self._clamp_view_offset()
    
    def _start_auto_solve(self):
        """Start auto-solve mode using state-space solver with inventory tracking."""
        if self.env.done:
            self._load_current_map()
        
        # Scan and mark all items before starting
        self._scan_and_mark_items()
        
        algorithm_names = ["A*", "BFS", "Dijkstra", "Greedy", "D* Lite"]
        solver_name = algorithm_names[self.algorithm_idx] if hasattr(self, 'algorithm_idx') else 'A*'
        self.message = f"Solving ({solver_name})..."
        self._render()
        pygame.display.flip()
        
        # Use either the high-level graph solver (DungeonSolver) or the selected grid-based algorithm.
        from Data.zelda_core import DungeonSolver, ValidationMode
        current_dungeon = self.maps[self.current_map_idx]

        # Helper to display/execute a found path
        def _handle_found_path(path, teleports, solver_result=None):
            # Always record planned path for overlay or execution
            self.auto_path = path

            if VISUALIZATION_AVAILABLE and getattr(self, 'preview_modal_enabled', False):
                # Modal behavior (legacy): show full blocking dialog
                try:
                    self.path_preview_dialog = PathPreviewDialog(
                        path=path,
                        env=self.env,
                        solver_result=solver_result,
                        speed_multiplier=self.speed_multiplier
                    )
                    self.path_preview_mode = True
                    self.preview_overlay_visible = False
                    self.message = "Path preview - Press ENTER to start or ESC to cancel"
                except Exception as e:
                    logger.warning(f"Failed to create path preview: {e}")
                    self._execute_auto_solve(path, solver_result or {}, teleports)
            elif VISUALIZATION_AVAILABLE:
                # Non-modal preview: keep overlay visible on map and show a sidebar summary
                try:
                    # Keep a dialog instance for overlay rendering, but do NOT show the blocking dialog
                    self.path_preview_dialog = PathPreviewDialog(
                        path=path,
                        env=self.env,
                        solver_result=solver_result,
                        speed_multiplier=self.speed_multiplier
                    )
                except Exception as e:
                    logger.warning(f"Failed to create path preview overlay: {e}")
                    self.path_preview_dialog = None
                self.path_preview_mode = False
                self.preview_overlay_visible = True
                self.message = "Path preview available in sidebar (Enter to start, Esc to dismiss)"
            else:
                # Visualization not available: just execute immediately
                self._execute_auto_solve(path, solver_result or {}, teleports)

        # If D* Lite is explicitly chosen or the flag is on, try D* Lite first for full state-space replanning
        if (self.feature_flags.get('dstar_lite', False) or getattr(self, 'algorithm_idx', 0) == 4):
            try:
                from simulation.dstar_lite import DStarLiteSolver
                start_state = self.env.get_state() if hasattr(self.env, 'get_state') else GameState(position=self.env.start_pos)
                ds = DStarLiteSolver(self.env)
                success, path, nodes = ds.solve(start_state)
                elapsed = 0.0
                if success:
                    self.dstar_solver = ds
                    self.dstar_active = True
                    self._set_message(f"D* Lite plan found ({nodes} nodes)")
                    _handle_found_path(path, 0, solver_result={'dstar_nodes': nodes})
                    return
                else:
                    # Fall through to other planners
                    self._set_message("D* Lite: no plan found, trying others", 2.0)
            except Exception as e:
                logger.warning(f"D* Lite integration failed: {e}")

        # If graph exists and the user did not force grid algorithms, prefer the graph solver
        if hasattr(current_dungeon, 'graph') and current_dungeon.graph and not getattr(self, 'force_grid_algorithm', False):
            solver = DungeonSolver()
            result = solver.solve(current_dungeon, mode=ValidationMode.FULL)

            if not result.get('solvable', False):
                self.auto_mode = False
                reason = result.get('reason', 'Unknown')
                self.message = f"Not solvable: {reason}"
                return

            # Store solver result and try to use a grid path for preview if possible
            self.solver_result = result
            edge_types = result.get('edge_types', [])
            keys_avail = result.get('keys_available', 0)
            keys_used = result.get('keys_used', 0)

            start_t = time.time()
            success, path, teleports = self._smart_grid_path()
            elapsed_ms = (time.time() - start_t) * 1000
            nodes = getattr(self, 'last_search_iterations', 0)
            if success:
                self._set_message(f"{solver_name} preview found ({nodes} nodes, {elapsed_ms:.0f}ms)")
                _handle_found_path(path, teleports, solver_result=result)
                return

            # If the grid path is blocked, try graph-guided room teleportation path
            success2, path2, teleports2 = self._graph_guided_path()
            if success2:
                _handle_found_path(path2, teleports2, solver_result=result)
                return

            self.auto_mode = False
            self.message = "Graph OK but grid blocked"
            return

        # If Parallel Search flag is enabled, run grid algorithms in background and pick best result
        if self.feature_flags.get('parallel_search', False):
            import threading
            # Use process-based parallel runner (multiprocessing) to avoid GIL
            try:
                import multiprocessing as mp
                from scripts.parallel_worker import run_grid_algorithm

                # Serialize grid to list for pickling
                cur = current_dungeon
                if hasattr(cur, 'global_grid'):
                    grid = cur.global_grid.tolist()
                else:
                    grid = cur.tolist()
                start = self.env.start_pos
                goal = self.env.goal_pos

                def worker_call(alg_idx):
                    return run_grid_algorithm(grid, start, goal, alg_idx)

                with mp.get_context('spawn').Pool(processes=4) as pool:
                    results = pool.map(worker_call, [0,1,2,3])

                best = None
                for idx, res in enumerate(results):
                    if res.get('success'):
                        score = (res['time_ms'], res['path'] and len(res['path']))
                        if best is None or score < best['score']:
                            best = {'alg': idx, 'path': res['path'], 'nodes': res['nodes'], 'time_ms': res['time_ms'], 'score': score}
                if best:
                    self._set_message('Parallel search completed', 2.0)
                    # show preview using the best path
                    try:
                        self.auto_path = best['path']
                        self.preview_overlay_visible = True
                        self.path_preview_dialog = PathPreviewDialog(path=self.auto_path, env=self.env, solver_result={}, speed_multiplier=self.speed_multiplier)
                        self._set_message(f"Parallel best: alg{best['alg']} ({best['nodes']} nodes, {best['time_ms']:.0f}ms)")
                    except Exception as e:
                        logger.warning(f"Failed to display parallel search preview: {e}")
                else:
                    self._set_message('Parallel search failed to find path', 2.0)
                return
            except Exception as e:
                # Fallback to single-threaded runner if multiprocessing fails
                logger.warning(f"Parallel process runner failed, falling back to thread: {e}")
                import threading
                def _parallel_runner_thread():
                    alg_list = [0,1,2,3]
                    best = None
                    for alg in alg_list:
                        try:
                            saved = getattr(self, 'algorithm_idx', 0)
                            self.algorithm_idx = alg
                            t0 = time.time()
                            succ, pth, tel = self._smart_grid_path()
                            elapsed = (time.time() - t0) * 1000
                            nodes = getattr(self, 'last_search_iterations', 0)
                            if succ:
                                score = (elapsed, len(pth))
                                if best is None or score < best['score']:
                                    best = {'alg': alg, 'path': pth, 'teleports': tel, 'nodes': nodes, 'time_ms': elapsed, 'score': score}
                        finally:
                            self.algorithm_idx = saved
                    if best:
                        self.parallel_search_result = best
                        self.parallel_search_done = True
                        self._set_message('Parallel search completed', 2.0)
                    else:
                        self._set_message('Parallel search failed to find path', 2.0)
                self.parallel_search_done = False
                self.parallel_search_result = None
                self.parallel_search_thread = threading.Thread(target=_parallel_runner_thread, daemon=True)
                self.parallel_search_thread.start()
                self._set_message('Parallel search started (thread fallback)...', 2.0)
                return

        # Otherwise (no graph or user forced grid), run the selected grid algorithm
        start_t = time.time()
        success, path, teleports = self._smart_grid_path()
        elapsed_ms = (time.time() - start_t) * 1000
        nodes = getattr(self, 'last_search_iterations', 0)
        if success:
            self._set_message(f"{solver_name} found ({nodes} nodes, {elapsed_ms:.0f}ms)")
            _handle_found_path(path, teleports, solver_result=None)
            return

        # As a final fallback, run the state-space A* solver (original fallback behavior)
        success2, path2, states = self.solver.solve()
        if success2:
            self.auto_path = path2
            self.auto_step_idx = 0
            self.auto_mode = True
            self.env.reset()
            self.message = f"Solution found! Path length: {len(path2)}"
        else:
            self.auto_mode = False
            self.message = f"No solution found (explored {states} states)"
    
    def _execute_auto_solve(self, path, solver_result, teleports=0):
        """
        Execute auto-solve immediately without preview (fallback).
        
        Args:
            path: Planned path
            solver_result: Solver metadata
            teleports: Number of teleport/warp moves
        """
        self.auto_path = path
        self.auto_step_idx = 0
        self.auto_mode = True
        self.env.reset()
        
        # Build informative message
        keys_used = solver_result.get('keys_used', 0)
        keys_avail = solver_result.get('keys_available', 0)
        key_info = f"Keys: {keys_avail}->{keys_avail - keys_used}" if keys_used > 0 else ""
        
        if teleports > 0:
            self.message = f"Path: {len(path)} ({teleports} warps) {key_info}"
        else:
            self.message = f"Path: {len(path)} steps {key_info}"
    
    def _execute_auto_solve_from_preview(self):
        """
        Start auto-solve after user confirms path preview.
        """
        self.auto_step_idx = 0
        self.auto_mode = True
        self.env.reset()
        
        # Dismiss preview / overlay
        self.path_preview_mode = False
        preview_dialog = self.path_preview_dialog
        self.path_preview_dialog = None
        self.preview_overlay_visible = False
        
        # Use stored path and show message
        if preview_dialog:
            self.message = f"Auto-solve started! Path: {len(self.auto_path)} steps"
        else:
            self.message = "Auto-solve started!"
    
    def _smart_grid_path(self):
        """
        Smart pathfinding that prioritizes walking and only warps via STAIRs.
        Returns (success, path, teleport_count).
        """
        from collections import deque
        import networkx as nx
        
        current_dungeon = self.maps[self.current_map_idx]
        # Support both StitchedDungeon objects (with .global_grid) and raw numpy grids
        if hasattr(current_dungeon, 'global_grid'):
            grid = current_dungeon.global_grid
            room_positions = getattr(current_dungeon, 'room_positions', {})
            room_to_node = getattr(current_dungeon, 'room_to_node', {})
            graph = getattr(current_dungeon, 'graph', None)
        else:
            grid = current_dungeon
            room_positions = {}
            room_to_node = {}
            graph = None
        H, W = grid.shape
        
        # Constants
        ROOM_HEIGHT = 16
        ROOM_WIDTH = 11
        FLOOR = 1
        WALL = 2
        BLOCK = 3
        VOID = 0
        DOOR_OPEN = 10
        DOOR_LOCKED = 11
        START = 21
        TRIFORCE = 22
        KEY = 30
        STAIR = 42
        
        WALKABLE = {FLOOR, DOOR_OPEN, KEY, START, TRIFORCE, STAIR, DOOR_LOCKED}
        
        start = self.env.start_pos
        goal = self.env.goal_pos
        
        # Guard: if start or goal missing (tests may simulate missing), bail out gracefully
        if start is None or goal is None:
            return False, [], 0

        # Reset and optionally collect search heatmap data
        if getattr(self, 'show_heatmap', False):
            self.search_heatmap = {}

        # Helper: get room for a position
        def get_room(pos):
            y, x = pos
            for room_pos, (ry, rx) in room_positions.items():
                if ry <= y < ry + ROOM_HEIGHT and rx <= x < rx + ROOM_WIDTH:
                    return room_pos
            return None

        # Helper: find all STAIR tiles in a room
        def get_stairs_in_room(room_pos):
            if room_pos not in room_positions:
                return []
            ry, rx = room_positions[room_pos]
            stairs = []
            for dy in range(ROOM_HEIGHT):
                for dx in range(ROOM_WIDTH):
                    y, x = ry + dy, rx + dx
                    if 0 <= y < H and 0 <= x < W and grid[y, x] == STAIR:
                        stairs.append((y, x))
            return stairs

        # Helper: find entry point in room
        def find_entry(room_pos):
            if room_pos not in room_positions:
                return None
            ry, rx = room_positions[room_pos]
            # Prefer stair, then center, then any walkable
            for dy in range(ROOM_HEIGHT):
                for dx in range(ROOM_WIDTH):
                    y, x = ry + dy, rx + dx
                    if 0 <= y < H and 0 <= x < W and grid[y, x] == STAIR:
                        return (y, x)
            cy, cx = ry + ROOM_HEIGHT // 2, rx + ROOM_WIDTH // 2
            if 0 <= cy < H and 0 <= cx < W and grid[cy, cx] in WALKABLE:
                return (cy, cx)
            for dy in range(ROOM_HEIGHT):
                for dx in range(ROOM_WIDTH):
                    y, x = ry + dy, rx + dx
                    if 0 <= y < H and 0 <= x < W and grid[y, x] in WALKABLE:
                        return (y, x)
            return None
        
        # Get graph info for stair connections (if available)
        node_to_room = {v: k for k, v in room_to_node.items()}
        # 'graph' variable already set earlier based on current_dungeon
        
        # Find stair destinations based on graph connectivity
        def get_stair_destinations(pos):
            """Find where stairs lead to based on graph."""
            if grid[pos[0], pos[1]] != STAIR:
                return []
            
            if not graph:
                return []
            
            current_room = get_room(pos)
            if not current_room:
                return []
            
            current_node = room_to_node.get(current_room)
            if current_node is None:
                return []
            
            destinations = []
            # Check graph neighbors
            for neighbor_node in graph.neighbors(current_node):
                neighbor_room = node_to_room.get(neighbor_node)
                if neighbor_room and neighbor_room in current_dungeon.room_positions:
                    # Find stairs in neighbor room, or entry point
                    neighbor_stairs = get_stairs_in_room(neighbor_room)
                    if neighbor_stairs:
                        destinations.extend(neighbor_stairs)
                    else:
                        entry = find_entry(neighbor_room)
                        if entry:
                            destinations.append(entry)
            
            return destinations
        
        # Choose search algorithm based on UI selection
        alg = getattr(self, 'algorithm_idx', 0)
        # 0: A*, 1: BFS, 2: Dijkstra, 3: Greedy, 4: D* Lite (fallback to A*)
        # Note: D* Lite not yet implemented — selecting it currently uses A* fallback.
        if alg == 4:
            logger.info("D* Lite selected but not implemented; using A* fallback")
            self._set_message("D* Lite selected: using A* fallback (not implemented)", 2.5)

        def heuristic(a, b):
            # Use ML heuristic if enabled and model available, else Manhattan
            if self.feature_flags.get('ml_heuristic', False) and getattr(self, 'ml_model', None):
                try:
                    return self._ml_heuristic(a, b)
                except Exception as e:
                    logger.warning(f"ML heuristic failed, falling back to Manhattan: {e}")
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        # Placeholder ML heuristic loader
        if not hasattr(self, 'ml_model'):
            self.ml_model = None
        def _ml_heuristic_stub(a, b):
            # For now, a simple biased Manhattan (simulates ML model outputs)
            return 0.9 * (abs(a[0] - b[0]) + abs(a[1] - b[1]))
        if getattr(self, 'ml_model', None) is None:
            self._ml_heuristic = _ml_heuristic_stub
        else:
            # If you add a real model loader, assign self._ml_heuristic accordingly
            pass
        max_iterations = 200000
        counter = 0
        # Track iterations for diagnostics (nodes expanded)
        iterations = 0
        self.last_search_iterations = 0

        if alg == 1:
            # BFS with stair teleportation (existing behavior)
            initial = (start, frozenset())
            visited = {start}
            queue = deque([(start, [start], 0)])  # pos, path, teleport_count
            iterations = 0
            while queue and iterations < max_iterations:
                iterations += 1
                pos, path, teleports = queue.popleft()
                y, x = pos
                if getattr(self, 'show_heatmap', False):
                    self.search_heatmap[pos] = self.search_heatmap.get(pos, 0) + 1
                if pos == goal:
                    self.last_search_iterations = iterations
                    return True, path, teleports
                # 4-directional walking
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and grid[ny, nx] in WALKABLE:
                        npos = (ny, nx)
                        if npos not in visited:
                            visited.add(npos)
                            queue.append((npos, path + [npos], teleports))
                # Stair teleportation
                if grid[y, x] == STAIR:
                    for dest in get_stair_destinations(pos):
                        if dest not in visited:
                            visited.add(dest)
                            queue.append((dest, path + [dest], teleports + 1))
            # No grid path found within BFS limits
            self.last_search_iterations = iterations
            return self._graph_guided_path()
        else:
            # Implement priority-search-based algorithms (A*, Dijkstra, Greedy)
            import heapq
            # Node state: (pos, visited_stairs_frozenset)
            start_state = (start, frozenset())
            # Priority queue entries: (priority, g_cost, counter, pos, visited_stairs, teleports, path)
            start_h = heuristic(start, goal)
            if alg == 0 or alg == 4:
                start_f = start_h
            elif alg == 2:
                start_f = 0
            elif alg == 3:
                start_f = start_h
            else:
                start_f = start_h
            heap = []
            heapq.heappush(heap, (start_f, 0, counter, start, frozenset(), 0, [start]))
            counter += 1
            best = {}  # (pos, stairs) -> best_g
            iterations = 0
            while heap and iterations < max_iterations:
                iterations += 1
                f, g, _cnt, pos, stairs, teleports, path = heapq.heappop(heap)
                if getattr(self, 'show_heatmap', False):
                    self.search_heatmap[pos] = self.search_heatmap.get(pos, 0) + 1
                if pos == goal:
                    self.last_search_iterations = iterations
                    return True, path, teleports
                key = (pos, stairs)
                if key in best and g > best[key]:
                    continue
                best[key] = g

                y, x = pos
                # 4-directional neighbors
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and grid[ny, nx] in WALKABLE:
                        npos = (ny, nx)
                        new_g = g + 1
                        new_stairs = stairs
                        new_teleports = teleports
                        # compute priority
                        h = heuristic(npos, goal)
                        if alg == 2:
                            new_f = new_g
                        elif alg == 3:
                            new_f = h
                        else:
                            new_f = new_g + h
                        nkey = (npos, new_stairs)
                        if nkey in best and new_g >= best[nkey]:
                            continue
                        heapq.heappush(heap, (new_f, new_g, counter, npos, new_stairs, new_teleports, path + [npos]))
                        counter += 1
                # Stair teleportation
                if grid[y, x] == STAIR:
                    for dest in get_stair_destinations(pos):
                        npos = dest
                        if npos in stairs:
                            continue
                        new_stairs = set(stairs)
                        new_stairs.add(npos)
                        new_stairs = frozenset(new_stairs)
                        new_g = g + 1
                        new_teleports = teleports + 1
                        h = heuristic(npos, goal)
                        if alg == 2:
                            new_f = new_g
                        elif alg == 3:
                            new_f = h
                        else:
                            new_f = new_g + h
                        nkey = (npos, new_stairs)
                        if nkey in best and new_g >= best[nkey]:
                            continue
                        heapq.heappush(heap, (new_f, new_g, counter, npos, new_stairs, new_teleports, path + [npos]))
                        counter += 1
            # No path found within priority-search limits
            self.last_search_iterations = iterations
            return self._graph_guided_path()    
    def _graph_guided_path(self):
        """Fallback: follow graph path with teleportation when needed."""
        import networkx as nx
        from collections import deque
        
        current_dungeon = self.maps[self.current_map_idx]
        # Support both StitchedDungeon objects (with .global_grid) and raw numpy grids
        if hasattr(current_dungeon, 'global_grid'):
            grid = current_dungeon.global_grid
            room_positions = getattr(current_dungeon, 'room_positions', {})
            room_to_node = getattr(current_dungeon, 'room_to_node', {})
            graph = getattr(current_dungeon, 'graph', None)
        else:
            grid = current_dungeon
            room_positions = {}
            room_to_node = {}
            graph = None
        H, W = grid.shape
        
        ROOM_HEIGHT = 16
        ROOM_WIDTH = 11
        FLOOR = 1
        DOOR_OPEN = 10
        DOOR_LOCKED = 11
        START = 21
        TRIFORCE = 22
        KEY = 30
        STAIR = 42
        WALKABLE = {FLOOR, DOOR_OPEN, KEY, START, TRIFORCE, STAIR, DOOR_LOCKED}
        
        start = self.env.start_pos
        goal = self.env.goal_pos
        
        def get_room(pos):
            y, x = pos
            for room_pos, (ry, rx) in current_dungeon.room_positions.items():
                if ry <= y < ry + ROOM_HEIGHT and rx <= x < rx + ROOM_WIDTH:
                    return room_pos
            return None
        
        def find_entry(room_pos):
            if room_pos not in current_dungeon.room_positions:
                return None
            ry, rx = current_dungeon.room_positions[room_pos]
            for dy in range(ROOM_HEIGHT):
                for dx in range(ROOM_WIDTH):
                    y, x = ry + dy, rx + dx
                    if 0 <= y < H and 0 <= x < W and grid[y, x] == STAIR:
                        return (y, x)
            cy, cx = ry + ROOM_HEIGHT // 2, rx + ROOM_WIDTH // 2
            if 0 <= cy < H and 0 <= cx < W and grid[cy, cx] in WALKABLE:
                return (cy, cx)
            for dy in range(ROOM_HEIGHT):
                for dx in range(ROOM_WIDTH):
                    y, x = ry + dy, rx + dx
                    if 0 <= y < H and 0 <= x < W and grid[y, x] in WALKABLE:
                        return (y, x)
            return None
        
        def local_bfs(from_pos, to_pos, max_steps=5000):
            if from_pos == to_pos:
                return [from_pos]
            visited = {from_pos}
            queue = deque([(from_pos, [from_pos])])
            while queue and len(visited) < max_steps:
                pos, path = queue.popleft()
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = pos[0] + dy, pos[1] + dx
                    if 0 <= ny < H and 0 <= nx < W and (ny, nx) not in visited:
                        if grid[ny, nx] in WALKABLE:
                            if (ny, nx) == to_pos:
                                return path + [(ny, nx)]
                            visited.add((ny, nx))
                            queue.append(((ny, nx), path + [(ny, nx)]))
            return None
        
        # Get graph path
        room_to_node = getattr(current_dungeon, 'room_to_node', {})
        node_to_room = {v: k for k, v in room_to_node.items()}
        graph = current_dungeon.graph
        
        start_room = get_room(start)
        goal_room = get_room(goal)
        start_node = room_to_node.get(start_room)
        goal_node = room_to_node.get(goal_room)
        
        if start_node is None or goal_node is None:
            return False, [], 0
        
        try:
            graph_path = nx.shortest_path(graph, start_node, goal_node)
        except nx.NetworkXNoPath:
            return False, [], 0
        
        # Build room sequence
        room_sequence = []
        for node in graph_path:
            room = node_to_room.get(node)
            if room and room in current_dungeon.room_positions:
                room_sequence.append(room)
        
        # Build path with teleportation
        full_path = [start]
        current_pos = start
        teleports = 0
        
        for target_room in room_sequence:
            target_pos = find_entry(target_room)
            if not target_pos or current_pos == target_pos:
                continue
            
            segment = local_bfs(current_pos, target_pos)
            if segment:
                full_path.extend(segment[1:])
                current_pos = segment[-1]
            else:
                full_path.append(target_pos)
                current_pos = target_pos
                teleports += 1
        
        # Final to goal
        if current_pos != goal:
            segment = local_bfs(current_pos, goal)
            if segment:
                full_path.extend(segment[1:])
            else:
                full_path.append(goal)
                teleports += 1
        
        if full_path[-1] != goal:
            full_path.append(goal)
        
        return True, full_path, teleports

    def _hybrid_graph_grid_path(self):
        """
        Hybrid pathfinding: use graph to find room sequence, 
        then BFS within each room and teleport between disconnected clusters.
        """
        import networkx as nx
        from collections import deque
        
        current_dungeon = self.maps[self.current_map_idx]
        grid = current_dungeon.global_grid
        H, W = grid.shape
        
        # Constants
        ROOM_HEIGHT = 16
        ROOM_WIDTH = 11
        FLOOR = 1
        WALL = 2
        BLOCK = 3
        VOID = 0
        DOOR_OPEN = 10
        DOOR_LOCKED = 11
        START = 21
        TRIFORCE = 22
        KEY = 30
        STAIR = 42
        
        WALKABLE = {FLOOR, DOOR_OPEN, KEY, START, TRIFORCE, STAIR, DOOR_LOCKED}
        
        start = self.env.start_pos
        goal = self.env.goal_pos
        
        if not start or not goal:
            return False, []
        
        # Helper: find room containing position
        def get_room(pos):
            y, x = pos
            for room_pos, (ry, rx) in current_dungeon.room_positions.items():
                if ry <= y < ry + ROOM_HEIGHT and rx <= x < rx + ROOM_WIDTH:
                    return room_pos
            return None
        
        # Helper: find passable position in room (prefer stairs, then center)
        def find_entry_point(room_pos):
            if room_pos not in current_dungeon.room_positions:
                return None
            ry, rx = current_dungeon.room_positions[room_pos]
            
            # First try to find a stair
            for dy in range(ROOM_HEIGHT):
                for dx in range(ROOM_WIDTH):
                    y, x = ry + dy, rx + dx
                    if 0 <= y < H and 0 <= x < W and grid[y, x] == STAIR:
                        return (y, x)
            
            # Then try center
            cy, cx = ry + ROOM_HEIGHT // 2, rx + ROOM_WIDTH // 2
            if 0 <= cy < H and 0 <= cx < W and grid[cy, cx] in WALKABLE:
                return (cy, cx)
            
            # Then any walkable tile
            for dy in range(ROOM_HEIGHT):
                for dx in range(ROOM_WIDTH):
                    y, x = ry + dy, rx + dx
                    if 0 <= y < H and 0 <= x < W and grid[y, x] in WALKABLE:
                        return (y, x)
            return None
        
        # Helper: BFS within connected tiles (allows room-to-room through doors)
        def local_bfs(from_pos, to_pos, max_steps=5000):
            if from_pos == to_pos:
                return [from_pos]
            
            visited = {from_pos}
            queue = deque([(from_pos, [from_pos])])
            
            while queue and len(visited) < max_steps:
                pos, path = queue.popleft()
                y, x = pos
                
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and (ny, nx) not in visited:
                        tile = grid[ny, nx]
                        if tile in WALKABLE:
                            if (ny, nx) == to_pos:
                                return path + [(ny, nx)]
                            visited.add((ny, nx))
                            queue.append(((ny, nx), path + [(ny, nx)]))
            return None
        
        # Step 1: Get graph path from start room to goal room
        start_room = get_room(start)
        goal_room = get_room(goal)
        
        if not start_room or not goal_room:
            return False, []
        
        room_to_node = getattr(current_dungeon, 'room_to_node', {})
        node_to_room = {v: k for k, v in room_to_node.items()}
        
        start_node = room_to_node.get(start_room)
        goal_node = room_to_node.get(goal_room)
        
        if start_node is None or goal_node is None:
            # Fallback to direct BFS
            path = local_bfs(start, goal, max_steps=50000)
            return (True, path) if path else (False, [])
        
        # Get graph path
        graph = current_dungeon.graph
        try:
            graph_path = nx.shortest_path(graph, start_node, goal_node)
        except nx.NetworkXNoPath:
            return False, []
        
        # Step 2: Build room sequence (skip unmapped nodes)
        room_sequence = []
        for node in graph_path:
            room = node_to_room.get(node)
            if room and room in current_dungeon.room_positions:
                room_sequence.append(room)
        
        if not room_sequence:
            return False, []
        
        # Step 3: Build full path by connecting rooms
        full_path = [start]
        current_pos = start
        
        for target_room in room_sequence:
            target_pos = find_entry_point(target_room)
            if not target_pos:
                continue
            
            # Skip if we're already at this target
            if current_pos == target_pos:
                continue
            
            # Try BFS to target
            path_segment = local_bfs(current_pos, target_pos)
            
            if path_segment:
                # Connected physically - add path
                full_path.extend(path_segment[1:])
                current_pos = path_segment[-1]
            else:
                # Not connected physically - teleport!
                full_path.append(target_pos)
                current_pos = target_pos
        
        # Final segment to goal
        if current_pos != goal:
            final_segment = local_bfs(current_pos, goal)
            if final_segment:
                full_path.extend(final_segment[1:])
            else:
                # Teleport to goal as last resort
                full_path.append(goal)
        
        # Validate path ends at goal
        if full_path[-1] != goal:
            full_path.append(goal)
        
        return True, full_path
    
    def _auto_step(self):
        """Execute one step of auto-solve with comprehensive error handling."""
        try:
            # Validate auto-mode state
            if not self.auto_mode:
                return
            
            if not hasattr(self, 'auto_path') or not self.auto_path:
                self._show_error("No solution path available")
                self.auto_mode = False
                return
            
            if self.auto_step_idx >= len(self.auto_path) - 1:
                self.auto_mode = False
                self._set_message("Solution complete!")
                self.status_message = "Completed"
                return

            # D* Lite replanning check (if active)
            if self.feature_flags.get('dstar_lite', False) and getattr(self, 'dstar_active', False) and getattr(self, 'dstar_solver', None):
                try:
                    current_state = self.env.get_state() if hasattr(self.env, 'get_state') else self.env.state
                    if self.dstar_solver.needs_replan(current_state):
                        success, new_path, updated = self.dstar_solver.replan(current_state)
                        if success and new_path:
                            # Align auto_step_idx to current position in new path
                            curpos = self.env.state.position
                            try:
                                idx = new_path.index(curpos)
                            except ValueError:
                                idx = 0
                            self.auto_path = new_path
                            self.auto_step_idx = idx
                            self._set_message(f"D* Lite replanned ({updated} updates)")
                except Exception as e:
                    logger.warning(f"D* Lite replanning failed: {e}")

            
            # Validate environment
            if self.env is None:
                self._show_error("Environment not initialized")
                self.auto_mode = False
                return
            
            if not hasattr(self.env, 'state') or self.env.state is None:
                self._show_error("Invalid environment state")
                self.auto_mode = False
                return
            
            self.auto_step_idx += 1
            target = self.auto_path[self.auto_step_idx]
            current = self.env.state.position
            
            # Check if this is a teleport (non-adjacent move)
            dr = target[0] - current[0]
            dc = target[1] - current[1]
            
            if abs(dr) > 1 or abs(dc) > 1 or (abs(dr) == 1 and abs(dc) == 1):
                # Teleport - directly set position
                old_state = GameState(
                    position=self.env.state.position,
                    keys=self.env.state.keys,
                    has_bomb=self.env.state.has_bomb,
                    has_boss_key=self.env.state.has_boss_key,
                    opened_doors=self.env.state.opened_doors.copy() if hasattr(self.env.state.opened_doors, 'copy') else set(self.env.state.opened_doors),
                    collected_items=self.env.state.collected_items.copy() if hasattr(self.env.state.collected_items, 'copy') else set(self.env.state.collected_items)
                )
                
                self.env.state.position = target
                self._set_message(f"Teleport! {current} -> {target}")
                self.status_message = "Teleporting..."
                
                # Track item changes (with error handling)
                try:
                    self._track_item_collection(old_state, self.env.state)
                    self._track_item_usage(old_state, self.env.state)
                except Exception as e:
                    logger.warning(f"Item tracking failed: {e}")
                
                # Update visual position (instant for teleport)
                if self.renderer:
                    try:
                        self.renderer.set_agent_position(target[0], target[1], immediate=True)
                    except Exception as e:
                        logger.warning(f"Renderer update failed: {e}")
                
                if self.effects:
                    try:
                        # Add ripple effect at teleport destination (grid coordinates)
                        self.effects.add_effect(RippleEffect(target, (100, 200, 255)))
                    except Exception as e:
                        logger.warning(f"Effect creation failed: {e}")
                
                # Check if reached goal
                if target == self.env.goal_pos:
                    self.env.won = True
                    self.env.done = True
                    self.auto_mode = False
                    self._set_message("AUTO-SOLVE: Victory!")
                    self.status_message = "Victory!"
            else:
                # Normal move - capture old state
                old_state = GameState(
                    position=self.env.state.position,
                    keys=self.env.state.keys,
                    has_bomb=self.env.state.has_bomb,
                    has_boss_key=self.env.state.has_boss_key,
                    opened_doors=self.env.state.opened_doors.copy() if hasattr(self.env.state.opened_doors, 'copy') else set(self.env.state.opened_doors),
                    collected_items=self.env.state.collected_items.copy() if hasattr(self.env.state.collected_items, 'copy') else set(self.env.state.collected_items)
                )
            
            if dr == -1:
                action = Action.UP
            elif dr == 1:
                action = Action.DOWN
            elif dc == -1:
                action = Action.LEFT
            else:
                action = Action.RIGHT
            
            state, reward, done, info = self.env.step(int(action))
            
            # Get new position immediately after step
            new_pos = self.env.state.position
            
            # Increment step counter
            self.step_count += 1
            
            # Track item collection and usage (single call, not duplicated)
            self._track_item_collection(old_state, self.env.state)
            self._track_item_usage(old_state, self.env.state)
            
            # Update modern HUD during auto-solve with collected counts
            if self.modern_hud:
                self.modern_hud.update_game_state(
                    keys=self.env.state.keys,
                    bombs=1 if self.env.state.has_bomb else 0,
                    has_boss_key=self.env.state.has_boss_key,
                    position=new_pos,
                    steps=self.step_count,
                    message=self.message
                )
                # Also update inventory display with collection counts
                if hasattr(self.modern_hud, 'keys_collected'):
                    self.modern_hud.keys_collected = self.keys_collected
                    self.modern_hud.bombs_collected = self.bombs_collected
                    self.modern_hud.boss_keys_collected = self.boss_keys_collected
            
            # Update visual position (smooth)
            if self.renderer:
                self.renderer.set_agent_position(new_pos[0], new_pos[1], immediate=False)
            
            # Check if done (NOT indented under renderer)
            if done:
                self.auto_mode = False
                if self.env.won:
                    self._set_message("AUTO-SOLVE: Victory!")
                    self.status_message = "Victory!"
                    if self.effects:
                        try:
                            # Victory flash effect at goal position
                            goal_pos = self.env.goal_pos
                            self.effects.add_effect(FlashEffect(goal_pos, (255, 215, 0), 0.5))
                        except Exception as e:
                            logger.warning(f"Victory effect failed: {e}")
                else:
                    self._set_message(f"AUTO-SOLVE: Failed - {info.get('msg', '')}")
                    self.status_message = "Failed"
        
        except KeyError as e:
            self._show_error(f"State access error: {str(e)}")
            self.auto_mode = False
        except IndexError as e:
            self._show_error(f"Path index error: {str(e)}")
            self.auto_mode = False
        except AttributeError as e:
            self._show_error(f"Invalid state attribute: {str(e)}")
            self.auto_mode = False
        except Exception as e:
            self._show_error(f"Auto-solve error: {str(e)}")
            self.auto_mode = False
            import traceback
            traceback.print_exc()
    
    def _show_error(self, message: str):
        """Display error message to user with visual feedback."""
        logger.error(message)
        self.error_message = message
        self.error_time = time.time()
        self.status_message = "Error"
        print(f"[!] ERROR: {message}")
    
    def _show_message(self, message: str, duration: float = 3.0):
        """Display informational message to user."""
        logger.info(message)
        self.message = message
        self.message_time = time.time()
        self.message_duration = duration
        self.status_message = "Info"

    # --- Topology helpers ---
    def _export_topology(self):
        """Export current map topology to a DOT file (if available)."""
        current = self.maps[self.current_map_idx]
        graph = getattr(current, 'graph', None)
        room_positions = getattr(current, 'room_positions', None)
        if graph is None:
            self._set_message('No topology graph available for this map', 3.0)
            return
        # Try to use networkx pydot writer
        try:
            import networkx as nx
            try:
                fname = f"topology_map_{self.current_map_idx+1}.dot"
                nx.nx_pydot.write_dot(graph, fname)
                # Add node positions as comments if available
                if room_positions:
                    with open(fname, 'a') as f:
                        f.write('\n// room positions\n')
                        for room, (ry, rx) in room_positions.items():
                            f.write(f"// {room}: {ry},{rx}\n")
                self._set_message(f"Topology exported to {fname}")
                self.topology_export_path = fname
            except Exception as e:
                # Fallback to manual DOT generation
                fname = f"topology_map_{self.current_map_idx+1}.dot"
                with open(fname, 'w', encoding='utf-8') as f:
                    f.write('graph topology {\n')
                    for n in graph.nodes():
                        f.write(f'  "{n}";\n')
                    for u, v in graph.edges():
                        f.write(f'  "{u}" -- "{v}";\n')
                    f.write('}\n')
                self._set_message(f"Topology exported to {fname} (manual)\n{e}")
                self.topology_export_path = fname
        except ImportError:
            self._set_message('NetworkX not available - cannot export DOT automatically', 4.0)

    def _render_topology_overlay(self, surface: pygame.Surface):
        """Draw room nodes and edges on the map area."""
        current = self.maps[self.current_map_idx]
        if not hasattr(current, 'graph') or not current.graph:
            return
        graph = current.graph
        room_positions = getattr(current, 'room_positions', {})
        # Draw edges
        for u, v in graph.edges():
            ru = room_positions.get(u)
            rv = room_positions.get(v)
            if not ru or not rv:
                continue
            # room centers
            cu = (ru[1] * self.TILE_SIZE + self.TILE_SIZE * 0.5 - self.view_offset_x,
                  ru[0] * self.TILE_SIZE + self.TILE_SIZE * 0.5 - self.view_offset_y)
            cv = (rv[1] * self.TILE_SIZE + self.TILE_SIZE * 0.5 - self.view_offset_x,
                  rv[0] * self.TILE_SIZE + self.TILE_SIZE * 0.5 - self.view_offset_y)
            try:
                pygame.draw.line(surface, (120, 200, 255), cu, cv, 2)
            except Exception:
                pass
        # Draw nodes
        for room, (ry, rx) in room_positions.items():
            cx = rx * self.TILE_SIZE + self.TILE_SIZE * 0.5 - self.view_offset_x
            cy = ry * self.TILE_SIZE + self.TILE_SIZE * 0.5 - self.view_offset_y
            r = max(4, int(self.TILE_SIZE * 0.25))
            try:
                pygame.draw.circle(surface, (255, 200, 100), (int(cx), int(cy)), r)
                font = pygame.font.SysFont('Arial', max(10, int(self.TILE_SIZE // 6)))
                txt = font.render(str(room), True, (20, 20, 30))
                surface.blit(txt, (int(cx) - txt.get_width() // 2, int(cy) - txt.get_height() // 2))
            except Exception:
                pass

    # --- Solver comparison helpers ---
    def _set_last_solver_metrics(self, name, nodes, time_ms, path_len):
        self.last_solver_metrics = {
            'name': name,
            'nodes': nodes,
            'time_ms': time_ms,
            'path_len': path_len
        }

    def _run_solver_comparison(self):
        """Entry point for Compare Solvers button - runs and displays results."""
        # Run comparison on main thread (fast maps); if long, we can spawn a worker
        results = []
        alg_names = ['A*', 'BFS', 'Dijkstra', 'Greedy', 'D* Lite', 'StateSpace']
        for idx, name in enumerate(alg_names):
            start_t = time.time()
            # D* Lite special-case: use D* Lite implementation
            if name == 'D* Lite':
                try:
                    from simulation.dstar_lite import DStarLiteSolver
                    start_state = GameState(position=self.env.start_pos, opened_doors=self.env.state.opened_doors.copy() if hasattr(self.env, 'state') else set())
                    ds = DStarLiteSolver(self.env)
                    success, path, nodes = ds.solve(start_state)
                    elapsed = (time.time() - start_t) * 1000
                    results.append({'name': name, 'success': success, 'path_len': len(path), 'nodes': nodes, 'time_ms': elapsed})
                    if success:
                        self._set_last_solver_metrics(name, nodes, elapsed, len(path))
                    continue
                except Exception as e:
                    results.append({'name': name, 'success': False, 'path_len': 0, 'nodes': 0, 'time_ms': 0, 'error': str(e)})
                    continue
            if name == 'StateSpace':
                # Use state-space solver as final baseline
                try:
                    start_state = self.env.get_state() if hasattr(self.env, 'get_state') else GameState(position=self.env.start_pos)
                    success, path = self.solver.solve()
                    elapsed = (time.time() - start_t) * 1000
                    nodes = getattr(self.solver, 'last_states_explored', 0)
                    results.append({'name': name, 'success': success, 'path_len': len(path), 'nodes': nodes, 'time_ms': elapsed})
                    if success and not self.last_solver_metrics:
                        self._set_last_solver_metrics(name, nodes, elapsed, len(path))
                    continue
                except Exception as e:
                    results.append({'name': name, 'success': False, 'path_len': 0, 'nodes': 0, 'time_ms': 0, 'error': str(e)})
                    continue
            # For other algorithms use existing mechanism (set algorithm_idx temporarily)
            saved_alg = getattr(self, 'algorithm_idx', 0)
            saved_preview = self.preview_overlay_visible
            saved_modal = self.preview_modal_enabled
            saved_path = list(self.auto_path) if self.auto_path else []
            try:
                self.preview_overlay_visible = False
                self.preview_modal_enabled = False
                self.algorithm_idx = idx if idx < 5 else 0
                t0 = time.time()
                success, path, teleports = self._smart_grid_path()
                elapsed = (time.time() - t0) * 1000
                nodes = getattr(self, 'last_search_iterations', 0)
                results.append({'name': name, 'success': success, 'path_len': len(path), 'nodes': nodes, 'time_ms': elapsed})
                if success and not self.last_solver_metrics:
                    self._set_last_solver_metrics(name, nodes, elapsed, len(path))
            finally:
                self.algorithm_idx = saved_alg
                self.preview_overlay_visible = saved_preview
                self.preview_modal_enabled = saved_modal
                self.auto_path = saved_path
        self.solver_comparison_results = results
        self.show_solver_comparison_overlay = True
        self._set_message('Solver comparison complete', 3.0)

    def _render_solver_comparison_overlay(self, surface: pygame.Surface):
        """Render a small sidebar table with solver comparison results."""
        if not getattr(self, 'solver_comparison_results', None):
            return
        sidebar_x = self.screen_w - self.SIDEBAR_WIDTH
        box_w = self.SIDEBAR_WIDTH - 20
        box_h = min(220, 24 + 20 * len(self.solver_comparison_results))
        box_y = 220
        box_rect = pygame.Rect(sidebar_x + 10, box_y, box_w, box_h)
        pygame.draw.rect(surface, (38, 38, 55), box_rect)
        pygame.draw.rect(surface, (100, 150, 255), box_rect, 1)
        font = pygame.font.SysFont('Arial', 12, bold=True)
        header = font.render('Solver   Success   Len   Nodes   ms', True, (200, 200, 255))
        surface.blit(header, (box_rect.x + 6, box_rect.y + 6))
        y = box_rect.y + 28
        small = pygame.font.SysFont('Arial', 11)
        for r in self.solver_comparison_results:
            text = f"{r['name'][:7]:7}   {str(r.get('success',False))[:5]:5}   {r.get('path_len',0):3}   {r.get('nodes',0):6}   {int(r.get('time_ms',0)):4}"
            surface.blit(small.render(text, True, (200,200,200)), (box_rect.x + 6, y))
            y += 18
        # Dismiss hint
        hint = small.render('Press Esc to close', True, (150,150,150))
        surface.blit(hint, (box_rect.x + 6, box_rect.y + box_rect.h - 18))
    
    def _set_message(self, message: str, duration: float = 3.0):
        """Set status message with timestamp for auto-hide."""
        self.message = message
        self.message_time = time.time()
        self.message_duration = duration
    
    def _show_toast(self, message: str, duration: float = 3.0, toast_type: str = 'info'):
        """Show a floating toast notification."""
        if hasattr(self, 'toast_notifications'):
            self.toast_notifications.append(ToastNotification(message, duration, toast_type))
    
    def _update_toasts(self):
        """Update and remove expired toasts."""
        if hasattr(self, 'toast_notifications'):
            self.toast_notifications = [t for t in self.toast_notifications if not t.is_expired()]
    
    def _render_toasts(self, surface: pygame.Surface):
        """Render all active toast notifications."""
        if not hasattr(self, 'toast_notifications'):
            return
        base_y = self.screen_h - 140  # Above bottom panel
        for i, toast in enumerate(self.toast_notifications):
            toast.render(surface, self.screen_w // 2, base_y - i * 70)
    
    def _show_warning(self, message: str):
        """Display warning message to user."""
        logger.warning(message)
        self.message = f"[!] {message}"
    
    def _manual_step(self, action: Action):
        """Execute manual step."""
        if self.env.done:
            return
        
        old_pos = self.env.state.position
        old_keys = self.env.state.keys
        old_bombs = self.env.state.has_bomb
        old_boss_key = self.env.state.has_boss_key
        
        state, reward, done, info = self.env.step(int(action))
        new_pos = self.env.state.position
        
        # Increment step counter
        self.step_count += 1
        
        # Check for item pickups and add visual feedback
        if self.env.state.keys > old_keys:
            # Key picked up!
            keys_gained = self.env.state.keys - old_keys
            self.keys_collected += keys_gained
            if self.effects:
                self.effects.add_effect(PopEffect(new_pos, (255, 215, 0)))  # Gold flash
            self.item_pickup_times['key'] = time.time()
            self.message = f"Key collected! ({self.keys_collected}/{self.total_keys}, {self.env.state.keys} held)"
        
        if self.env.state.has_bomb and not old_bombs:
            # Bomb acquired!
            self.bombs_collected += 1
            if self.effects:
                self.effects.add_effect(PopEffect(new_pos, (200, 80, 80)))  # Red flash
            self.item_pickup_times['bomb'] = time.time()
            self.message = f"Bomb acquired! ({self.bombs_collected}/{self.total_bombs})"
        
        if self.env.state.has_boss_key and not old_boss_key:
            # Boss key found!
            self.boss_keys_collected += 1
            if self.effects:
                self.effects.add_effect(FlashEffect(new_pos, (180, 40, 180), 0.5))  # Purple flash
            self.item_pickup_times['boss_key'] = time.time()
            self.message = f"BOSS KEY acquired! ({self.boss_keys_collected}/{self.total_boss_keys})"
        
        # Update modern HUD with current game state
        if self.modern_hud:
            self.modern_hud.update_game_state(
                keys=self.env.state.keys,
                bombs=1 if self.env.state.has_bomb else 0,
                has_boss_key=self.env.state.has_boss_key,
                position=new_pos,
                steps=self.step_count,
                message=self.message
            )
        
        # Update visual position (smooth animation)
        if self.renderer and new_pos != old_pos:
            self.renderer.set_agent_position(new_pos[0], new_pos[1], immediate=False)
            # Add pop effect at new position (grid coordinates)
            if self.effects:
                self.effects.add_effect(PopEffect(new_pos, (100, 255, 100)))
        
        if done:
            if self.env.won:
                self.message = "YOU WIN!"
                if self.effects:
                    goal_pos = self.env.goal_pos
                    self.effects.add_effect(FlashEffect(goal_pos, (255, 215, 0), 0.5))
            else:
                self.message = f"Game Over: {info.get('msg', '')}"
        else:
            msg = info.get('msg', '')
            if msg:
                self.message = msg
    
    def _render(self):
        """Render the current state using new visualization system or fallback."""
        # Clear screen
        self.screen.fill((25, 25, 35))
        
        h, w = self.env.height, self.env.width
        view_w = self.screen_w - self.SIDEBAR_WIDTH
        view_h = self.screen_h - self.HUD_HEIGHT
        
        # Create map surface for the main view area
        map_surface = pygame.Surface((view_w, view_h))
        map_surface.fill((20, 20, 30))
        
        # Apply speed multiplier to animation updates
        effective_dt = self.delta_time * self.speed_multiplier
        
        # Update new renderer if available
        if self.renderer:
            self.renderer.update(effective_dt)
        if self.effects:
            self.effects.update(effective_dt)
        
        # Update modern HUD with current game state every frame (real-time)
        if self.modern_hud and self.env:
            self.modern_hud.update_game_state(
                keys=self.env.state.keys,
                bombs=1 if self.env.state.has_bomb else 0,
                has_boss_key=self.env.state.has_boss_key,
                position=self.env.state.position,
                steps=self.step_count,
                message=self.message
            )
        
        # Draw grid (only visible tiles for performance)
        start_c = max(0, self.view_offset_x // self.TILE_SIZE)
        start_r = max(0, self.view_offset_y // self.TILE_SIZE)
        end_c = min(w, start_c + (view_w // self.TILE_SIZE) + 2)
        end_r = min(h, start_r + (view_h // self.TILE_SIZE) + 2)
        
        # Use new renderer for map tiles if available
        if self.renderer:
            for r in range(start_r, end_r):
                for c in range(start_c, end_c):
                    tile_id = self.env.grid[r, c]
                    screen_x = c * self.TILE_SIZE - self.view_offset_x
                    screen_y = r * self.TILE_SIZE - self.view_offset_y
                    # Use sprite manager (with procedural fallback)
                    tile_surface = self.renderer.sprite_manager.get_tile(tile_id, self.TILE_SIZE)
                    map_surface.blit(tile_surface, (screen_x, screen_y))
                    # Draw stair sprite overlay if tile is stair
                    if tile_id == SEMANTIC_PALETTE['STAIR'] and getattr(self, 'stair_sprite', None):
                        try:
                            alpha = int(140 + 90 * math.sin(time.time() * 3.0))
                            s = self.stair_sprite.copy()
                            s.set_alpha(max(20, alpha))
                            sx = screen_x + (self.TILE_SIZE - s.get_width()) // 2
                            sy = screen_y + (self.TILE_SIZE - s.get_height()) // 2
                            map_surface.blit(s, (sx, sy))
                        except Exception:
                            pass
        else:
            # Fallback rendering
            for r in range(start_r, end_r):
                for c in range(start_c, end_c):
                    tile_id = self.env.grid[r, c]
                    img = self.images.get(tile_id, self.images.get(SEMANTIC_PALETTE['FLOOR']))
                    screen_x = c * self.TILE_SIZE - self.view_offset_x
                    screen_y = r * self.TILE_SIZE - self.view_offset_y
                    map_surface.blit(img, (screen_x, screen_y))
                    # Draw stair sprite overlay for fallback tiles
                    if tile_id == SEMANTIC_PALETTE['STAIR'] and getattr(self, 'stair_sprite', None):
                        try:
                            alpha = int(140 + 90 * math.sin(time.time() * 3.0))
                            s = self.stair_sprite.copy()
                            s.set_alpha(max(20, alpha))
                            sx = screen_x + (self.TILE_SIZE - s.get_width()) // 2
                            sy = screen_y + (self.TILE_SIZE - s.get_height()) // 2
                            map_surface.blit(s, (sx, sy))
                        except Exception:
                            pass
        
        # Draw heatmap overlay if enabled and we have search data
        if self.show_heatmap and self.search_heatmap:
            max_visits = max(self.search_heatmap.values()) if self.search_heatmap else 1
            for (r, c), visits in self.search_heatmap.items():
                if start_r <= r < end_r and start_c <= c < end_c:
                    # Normalize intensity 0.0 - 1.0
                    intensity = visits / max_visits
                    # Blue (cold) to Red (hot) gradient
                    red = int(255 * intensity)
                    blue = int(255 * (1 - intensity))
                    heat_color = (red, 0, blue, 100)
                    
                    heat_surf = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
                    heat_surf.fill(heat_color)
                    screen_x = c * self.TILE_SIZE - self.view_offset_x
                    screen_y = r * self.TILE_SIZE - self.view_offset_y
                    map_surface.blit(heat_surf, (screen_x, screen_y))
        
        # Draw solution path (if auto-solving)
        if self.auto_mode and self.auto_path:
            for i, pos in enumerate(self.auto_path[:self.auto_step_idx + 1]):
                pr, pc = pos
                path_surf = pygame.Surface((self.TILE_SIZE, self.TILE_SIZE), pygame.SRCALPHA)
                
                # Check if this position is a recently unlocked door
                current_time = time.time()
                is_recent_unlock = pos in self.door_unlock_times and (current_time - self.door_unlock_times[pos]) < 2.0
                
                if is_recent_unlock:
                    # Flash effect for recently unlocked doors (yellow/gold)
                    flash_alpha = (math.sin(current_time * 8) + 1) / 2  # 0 to 1
                    alpha = int(150 + 105 * flash_alpha)
                    path_surf.fill((255, 215, 0, alpha))  # Gold
                else:
                    # Use green with slight gradient based on progress
                    alpha = 40 + int(20 * (i / max(1, len(self.auto_path))))
                    path_surf.fill((0, 255, 0, alpha))
                
                screen_x = pc * self.TILE_SIZE - self.view_offset_x
                screen_y = pr * self.TILE_SIZE - self.view_offset_y
                map_surface.blit(path_surf, (screen_x, screen_y))
        
        # Draw Link (use smooth animation if renderer available)
        if self.renderer and self.renderer.agent_visual_pos:
            # Smooth animated position
            visual_pos = self.renderer.agent_visual_pos
            link_x = int(visual_pos.x * self.TILE_SIZE - self.view_offset_x + 2)
            link_y = int(visual_pos.y * self.TILE_SIZE - self.view_offset_y + 2)
        else:
            # Direct grid position
            pr, pc = self.env.state.position
            link_x = pc * self.TILE_SIZE - self.view_offset_x + 2
            link_y = pr * self.TILE_SIZE - self.view_offset_y + 2
        map_surface.blit(self.link_img, (link_x, link_y))
        
        # Render visual effects on map surface
        if self.effects:
            self.effects.render(map_surface, self.TILE_SIZE, (self.view_offset_x, self.view_offset_y))
        
        # Blit map surface to screen
        self.screen.blit(map_surface, (0, 0))
        
        # Get current position for display (use actual grid position)
        pr, pc = self.env.state.position
        
        # Draw sidebar background
        sidebar_x = self.screen_w - self.SIDEBAR_WIDTH
        pygame.draw.rect(self.screen, (35, 35, 50), (sidebar_x, 0, self.SIDEBAR_WIDTH, self.screen_h))
        pygame.draw.line(self.screen, (60, 60, 80), (sidebar_x, 0), (sidebar_x, self.screen_h), 2)
        
        # Sidebar content
        y_pos = 10
        
        # Title
        title = self.big_font.render("ZAVE", True, (100, 200, 255))
        self.screen.blit(title, (sidebar_x + 10, y_pos))
        y_pos += 28
        
        # Dungeon name
        if self.current_map_idx < len(self.map_names):
            name = self.map_names[self.current_map_idx]
        else:
            name = f"Map {self.current_map_idx + 1}"
        name_surf = self.font.render(name, True, (255, 220, 100))
        self.screen.blit(name_surf, (sidebar_x + 10, y_pos))
        y_pos += 20
        
        # Map number
        map_num = f"({self.current_map_idx + 1}/{len(self.maps)})"
        num_surf = self.small_font.render(map_num, True, (150, 150, 150))
        self.screen.blit(num_surf, (sidebar_x + 10, y_pos))
        y_pos += 18
        
        size_info = f"Size: {w}x{h}"
        size_surf = self.small_font.render(size_info, True, (150, 150, 150))
        self.screen.blit(size_surf, (sidebar_x + 10, y_pos))
        y_pos += 20
        
        # Divider
        pygame.draw.line(self.screen, (60, 60, 80), (sidebar_x + 10, y_pos), (self.screen_w - 10, y_pos))
        y_pos += 10
        
        # Inventory section
        inv_title = self.font.render("Inventory", True, (255, 200, 100))
        self.screen.blit(inv_title, (sidebar_x + 10, y_pos))
        y_pos += 22
        
        # Check if items were recently picked up (within last 1 second) for highlight effect
        current_time = time.time()
        key_highlight = 'key' in self.item_pickup_times and (current_time - self.item_pickup_times['key']) < 1.0
        bomb_highlight = 'bomb' in self.item_pickup_times and (current_time - self.item_pickup_times['bomb']) < 1.0
        boss_key_highlight = 'boss_key' in self.item_pickup_times and (current_time - self.item_pickup_times['boss_key']) < 1.0
        
        # Keys with flash animation and X/Y collected format
        if self.total_keys > 0:
            keys_text = f"Keys: {self.keys_collected}/{self.total_keys} ({self.env.state.keys} held)"
        else:
            keys_text = f"Keys: {self.env.state.keys}"
        if key_highlight:
            # Flash between yellow and white
            flash_alpha = (math.sin(current_time * 15) + 1) / 2  # 0 to 1
            keys_color = (255, int(220 + 35 * flash_alpha), int(100 + 155 * flash_alpha))
        else:
            keys_color = (255, 220, 100)
        keys_surf = self.small_font.render(keys_text, True, keys_color)
        self.screen.blit(keys_surf, (sidebar_x + 15, y_pos))
        y_pos += 18
        
        # Bombs with highlight and collected count
        if self.total_bombs > 0:
            bomb_text = f"Bombs: {self.bombs_collected}/{self.total_bombs} {'[Y]' if self.env.state.has_bomb else '[N]'}"
        else:
            bomb_text = f"Bomb: {'[Y]' if self.env.state.has_bomb else '[N]'}"
        if bomb_highlight:
            flash_alpha = (math.sin(current_time * 15) + 1) / 2
            bomb_color = (int(200 + 55 * flash_alpha), int(80 + 175 * flash_alpha), int(80 + 175 * flash_alpha))
        else:
            bomb_color = (100, 255, 100) if self.env.state.has_bomb else (150, 150, 150)
        bomb_surf = self.small_font.render(bomb_text, True, bomb_color)
        self.screen.blit(bomb_surf, (sidebar_x + 15, y_pos))
        y_pos += 18
        
        # Boss key with highlight and collected status
        if self.total_boss_keys > 0:
            boss_key_text = f"Boss Key: {self.boss_keys_collected}/{self.total_boss_keys} {'[Y]' if self.env.state.has_boss_key else '[N]'}"
        else:
            boss_key_text = f"Boss Key: {'[Y]' if self.env.state.has_boss_key else '[N]'}"
        if boss_key_highlight:
            flash_alpha = (math.sin(current_time * 15) + 1) / 2
            boss_color = (int(180 + 75 * flash_alpha), int(40 + 215 * flash_alpha), int(180 + 75 * flash_alpha))
        else:
            boss_color = (255, 150, 100) if self.env.state.has_boss_key else (150, 150, 150)
        boss_surf = self.small_font.render(boss_key_text, True, boss_color)
        self.screen.blit(boss_surf, (sidebar_x + 15, y_pos))
        y_pos += 18
        
        # State-space solver info (if available)
        if self.solver_result:
            y_pos += 5
            pygame.draw.line(self.screen, (60, 60, 80), (sidebar_x + 10, y_pos), (self.screen_w - 10, y_pos))
            y_pos += 8
            
            solver_title = self.font.render("Path Analysis", True, (100, 200, 255))
            self.screen.blit(solver_title, (sidebar_x + 10, y_pos))
            y_pos += 20
            
            # Keys info from solver
            keys_avail = self.solver_result.get('keys_available', 0)
            keys_used = self.solver_result.get('keys_used', 0)
            key_info = f"Keys: {keys_avail} found, {keys_used} used"
            key_color = (255, 220, 100) if keys_used > 0 else (150, 200, 150)
            key_surf = self.small_font.render(key_info, True, key_color)
            self.screen.blit(key_surf, (sidebar_x + 15, y_pos))
            y_pos += 16
            
            # Edge types breakdown
            edge_types = self.solver_result.get('edge_types', [])
            if edge_types:
                type_counts = {}
                for et in edge_types:
                    type_counts[et] = type_counts.get(et, 0) + 1
                
                # Display each edge type with color
                edge_colors = {
                    'open': (100, 255, 100),       # Green - normal door
                    'key_locked': (255, 220, 100), # Yellow - key door
                    'bombable': (255, 150, 50),    # Orange - bomb wall
                    'soft_locked': (180, 100, 255),# Purple - one-way
                    'stair': (100, 200, 255),      # Cyan - teleport
                }
                
                for etype, count in type_counts.items():
                    color = edge_colors.get(etype, (150, 150, 150))
                    type_name = etype.replace('_', ' ').title()
                    et_text = f"  {type_name}: {count}"
                    et_surf = self.small_font.render(et_text, True, color)
                    self.screen.blit(et_surf, (sidebar_x + 15, y_pos))
                    y_pos += 14
        y_pos += 7
        
        # Divider
        pygame.draw.line(self.screen, (60, 60, 80), (sidebar_x + 10, y_pos), (self.screen_w - 10, y_pos))
        y_pos += 10
        
        # === STATUS SECTION ===
        status_title = self.font.render("STATUS", True, (180, 220, 255))
        self.screen.blit(status_title, (sidebar_x + 10, y_pos))
        y_pos += 22
        
        # Map name and position
        if self.env:
            map_name = self.map_names[self.current_map_idx] if self.current_map_idx < len(self.map_names) else f"Map {self.current_map_idx + 1}"
            pos = self.env.state.position
            status_text = f"{map_name}"
            status_surf = self.small_font.render(status_text, True, (200, 220, 255))
            self.screen.blit(status_surf, (sidebar_x + 15, y_pos))
            y_pos += 16
            
            pos_text = f"Pos: ({pos[0]}, {pos[1]})"
            pos_surf = self.small_font.render(pos_text, True, (150, 150, 150))
            self.screen.blit(pos_surf, (sidebar_x + 15, y_pos))
            y_pos += 16
            
            # Status message
            ready_text = "Auto-solving" if self.auto_mode else "Ready"
            ready_color = (100, 255, 100) if self.auto_mode else (200, 200, 200)
            ready_surf = self.small_font.render(ready_text, True, ready_color)
            self.screen.blit(ready_surf, (sidebar_x + 15, y_pos))
            y_pos += 18

            # D* Lite indicator (show when enabled or active)
            try:
                if getattr(self, 'dstar_active', False) and getattr(self, 'dstar_solver', None):
                    replans = getattr(self.dstar_solver, 'replans_count', 0)
                    ds_text = f"D* Lite: ACTIVE ({replans} replans)"
                    ds_surf = self.small_font.render(ds_text, True, (100, 220, 255))
                    self.screen.blit(ds_surf, (sidebar_x + 15, y_pos))
                    y_pos += 16
                elif self.feature_flags.get('dstar_lite', False):
                    ds_text = "D* Lite: enabled"
                    ds_surf = self.small_font.render(ds_text, True, (180, 180, 255))
                    self.screen.blit(ds_surf, (sidebar_x + 15, y_pos))
                    y_pos += 16
            except Exception:
                pass

            # Stair debug: count stairs and show sprite status
            try:
                stair_positions = list(map(tuple, self.env._find_all_positions(SEMANTIC_PALETTE['STAIR']))) if hasattr(self.env, '_find_all_positions') else []
                stair_count = len(stair_positions)
                stair_text = f"Stairs: {stair_count} | Sprite: {'Full' if getattr(self, 'stair_sprite', None) else 'No'}"
                stair_surf = self.small_font.render(stair_text, True, (200, 200, 150))
                self.screen.blit(stair_surf, (sidebar_x + 15, y_pos))
                y_pos += 16
            except Exception:
                pass
        
        # Divider
        pygame.draw.line(self.screen, (60, 60, 80), (sidebar_x + 10, y_pos), (self.screen_w - 10, y_pos))
        y_pos += 10
        
        # === MESSAGE SECTION ===
        message_title = self.font.render("MESSAGE", True, (100, 255, 150))
        self.screen.blit(message_title, (sidebar_x + 10, y_pos))
        y_pos += 22
        
        # Show current message with fade effect
        if self.message and (time.time() - self.message_time) < self.message_duration:
            elapsed = time.time() - self.message_time
            remaining = self.message_duration - elapsed
            alpha = min(1.0, remaining / 0.5) if remaining < 0.5 else 1.0
            
            # Word wrap message to fit sidebar
            max_chars = 28
            words = self.message.split()
            lines = []
            current_line = ""
            
            for word in words:
                test_line = f"{current_line} {word}".strip()
                if len(test_line) <= max_chars:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            if current_line:
                lines.append(current_line)
            
            # Render each line
            for line in lines[:3]:  # Max 3 lines
                msg_color = tuple(int(c * alpha) for c in (150, 255, 200))
                msg_surf = self.small_font.render(line, True, msg_color)
                self.screen.blit(msg_surf, (sidebar_x + 15, y_pos))
                y_pos += 16
        else:
            # Show default message
            default_msg = "Press SPACE to solve"
            msg_surf = self.small_font.render(default_msg, True, (120, 120, 120))
            self.screen.blit(msg_surf, (sidebar_x + 15, y_pos))
            y_pos += 16
        
        y_pos += 8
        
        # Divider
        pygame.draw.line(self.screen, (60, 60, 80), (sidebar_x + 10, y_pos), (self.screen_w - 10, y_pos))
        y_pos += 10
        
        # === METRICS SECTION ===
        metrics_title = self.font.render("Metrics", True, (150, 200, 255))
        self.screen.blit(metrics_title, (sidebar_x + 10, y_pos))
        y_pos += 22
        
        # Steps
        steps_text = f"Steps: {self.step_count}"
        steps_surf = self.small_font.render(steps_text, True, (200, 200, 200))
        self.screen.blit(steps_surf, (sidebar_x + 15, y_pos))
        y_pos += 16
        
        # Speed
        speed_color = (100, 255, 100) if self.speed_multiplier == 1.0 else (255, 200, 100)
        speed_text = f"Speed: {self.speed_multiplier}x"
        speed_surf = self.small_font.render(speed_text, True, speed_color)
        self.screen.blit(speed_surf, (sidebar_x + 15, y_pos))
        y_pos += 16
        
        # Zoom
        zoom_text = f"Zoom: {self.TILE_SIZE}px"
        zoom_surf = self.small_font.render(zoom_text, True, (150, 150, 150))
        self.screen.blit(zoom_surf, (sidebar_x + 15, y_pos))
        y_pos += 16
        
        # FPS
        fps = int(self.clock.get_fps())
        fps_color = (100, 255, 100) if fps >= 25 else (255, 150, 150)
        fps_text = f"FPS: {fps}"
        fps_surf = self.small_font.render(fps_text, True, fps_color)
        self.screen.blit(fps_surf, (sidebar_x + 15, y_pos))
        y_pos += 18
        
        # Divider
        pygame.draw.line(self.screen, (60, 60, 80), (sidebar_x + 10, y_pos), (self.screen_w - 10, y_pos))
        y_pos += 10
        
        # Controls section
        ctrl_title = self.font.render("Controls", True, (100, 200, 100))
        self.screen.blit(ctrl_title, (sidebar_x + 10, y_pos))
        y_pos += 20
        
        controls = [
            "Arrows Move",
            "SPACE Auto-solve",
            "R     Reset map",
            "N/P   Next/Prev",
            "+/-   Zoom",
            "0     Reset zoom",
            "C     Center view",
            "M     Minimap",
            "[/]   Speed+/-",
            "F11   Fullscreen",
            "F1    Help",
            "H     Heatmap",
            "ESC   Quit",
        ]
        
        for ctrl in controls:
            ctrl_surf = self.small_font.render(ctrl, True, (120, 120, 120))
            self.screen.blit(ctrl_surf, (sidebar_x + 15, y_pos))
            y_pos += 15
        
        # Draw HUD at bottom
        hud_y = self.screen_h - self.HUD_HEIGHT
        pygame.draw.rect(self.screen, (30, 30, 45), (0, hud_y, self.screen_w - self.SIDEBAR_WIDTH, self.HUD_HEIGHT))
        pygame.draw.line(self.screen, (60, 60, 80), (0, hud_y), (self.screen_w - self.SIDEBAR_WIDTH, hud_y), 2)
        
        # Status message
        msg_color = (255, 255, 100) if self.env.won else (200, 200, 200)
        msg_surf = self.font.render(self.message, True, msg_color)
        self.screen.blit(msg_surf, (10, hud_y + 10))
        
        # Position info
        pos_text = f"Position: ({pr}, {pc})"
        pos_surf = self.small_font.render(pos_text, True, (150, 150, 150))
        self.screen.blit(pos_surf, (10, hud_y + 35))
        
        # Win state
        if self.env.won:
            win_text = "*** VICTORY! ***"
            win_surf = self.big_font.render(win_text, True, (255, 215, 0))
            self.screen.blit(win_surf, (10, hud_y + 55))
        
        # Render minimap if enabled
        if self.show_minimap:
            self._render_minimap()
        
        # Help overlay
        if self.show_help:
            self._render_help_overlay()
        
        # Path preview dialog (Feature 5) - render on top of everything
        if self.path_preview_mode and self.path_preview_dialog:
            # Render path overlay on map
            try:
                self.path_preview_dialog.render_path_overlay(
                    self.screen,
                    self.TILE_SIZE,
                    self.view_offset_x,
                    self.view_offset_y,
                    self.SIDEBAR_WIDTH,
                    self.HUD_HEIGHT
                )
            except Exception as e:
                logger.warning(f"Failed to render path overlay: {e}")
            
            # Render dialog box
            try:
                self.path_preview_dialog.render(self.screen)
            except Exception as e:
                logger.warning(f"Failed to render path preview dialog: {e}")
        elif getattr(self, 'preview_overlay_visible', False) and getattr(self, 'path_preview_dialog', None):
            # Non-modal overlay: render only the path overlay (no blocking dialog)
            try:
                self.path_preview_dialog.render_path_overlay(
                    self.screen,
                    self.TILE_SIZE,
                    self.view_offset_x,
                    self.view_offset_y,
                    self.SIDEBAR_WIDTH,
                    self.HUD_HEIGHT
                )
            except Exception as e:
                logger.warning(f"Failed to render path overlay (non-modal): {e}")

            # Minimal sidebar preview box with start/dismiss buttons
            try:
                sidebar_x = self.screen_w - self.SIDEBAR_WIDTH
                box_h = 80
                box_y = 120  # fixed area near top of sidebar (below header area)
                box_rect = pygame.Rect(sidebar_x + 10, box_y, self.SIDEBAR_WIDTH - 20, box_h)
                pygame.draw.rect(self.screen, (40, 40, 60), box_rect)
                pygame.draw.rect(self.screen, (100, 150, 255), box_rect, 2)

                # Text details
                font = pygame.font.SysFont('Arial', 14, bold=True)
                small = pygame.font.SysFont('Arial', 12)
                path_len = len(self.auto_path) if getattr(self, 'auto_path', None) else 0
                text1 = font.render(f"Preview: {path_len} steps", True, (200, 200, 255))
                self.screen.blit(text1, (box_rect.x + 8, box_rect.y + 8))

                # Keys info (if available)
                keys_used = getattr(self, 'solver_result', {}).get('keys_used', 0) if getattr(self, 'solver_result', None) else 0
                keys_avail = getattr(self, 'solver_result', {}).get('keys_available', 0) if getattr(self, 'solver_result', None) else 0
                keys_text = f"Keys: {keys_used} / {keys_avail}" if keys_avail > 0 else "Keys: None"
                self.screen.blit(small.render(keys_text, True, (200, 200, 200)), (box_rect.x + 8, box_rect.y + 34))

                # Start & Dismiss buttons
                start_rect = pygame.Rect(box_rect.x + 8, box_rect.y + 48, 140, 24)
                dismiss_rect = pygame.Rect(box_rect.x + 156, box_rect.y + 48, 60, 24)
                pygame.draw.rect(self.screen, (40, 140, 40), start_rect)
                pygame.draw.rect(self.screen, (140, 40, 40), dismiss_rect)
                pygame.draw.rect(self.screen, (100, 255, 100), start_rect, 1)
                pygame.draw.rect(self.screen, (255, 100, 100), dismiss_rect, 1)
                self.sidebar_start_button_rect = start_rect
                self.sidebar_dismiss_button_rect = dismiss_rect

                start_text = small.render("Start Auto-Solve", True, (255, 255, 255))
                dismiss_text = small.render("Dismiss", True, (255, 255, 255))
                self.screen.blit(start_text, (start_rect.x + 8, start_rect.y + 4))
                self.screen.blit(dismiss_text, (dismiss_rect.x + 6, dismiss_rect.y + 4))

            except Exception as e:
                logger.warning(f"Failed to render sidebar preview box: {e}")
        else:
            # Ensure stale sidebar button rects are cleared
            self.sidebar_start_button_rect = None
            self.sidebar_dismiss_button_rect = None

        # Render topology overlay (if enabled)
        if getattr(self, 'show_topology', False):
            try:
                self._render_topology_overlay(self.screen)
            except Exception as e:
                logger.warning(f"Topology overlay failed: {e}")

        # Render solver comparison overlay (if available)
        if getattr(self, 'show_solver_comparison_overlay', False):
            try:
                self._render_solver_comparison_overlay(self.screen)
            except Exception as e:
                logger.warning(f"Solver comparison overlay failed: {e}")
        
        # Render control panel
        if self.control_panel_enabled:
            self._render_control_panel(self.screen)

        # Render developer debug overlay (toggle with F12)
        if getattr(self, 'debug_overlay_enabled', False):
            try:
                self._render_debug_overlay(self.screen)
            except Exception as e:
                logger.warning(f"Debug overlay render failed: {e}")
        
        # Render item legend
        if self.auto_mode:
            self._render_item_legend(self.screen)
        
        # Render error banner (on top of everything)
        self._render_error_banner(self.screen)
        
        # Render toast notifications (on top of everything)
        self._render_toasts(self.screen)
        
        pygame.display.flip()

    def _render_debug_overlay(self, surface: pygame.Surface):
        """Render debug overlay with mouse coords, widget rects, and recent clicks.
        Toggle with F12. Shift-F11 clears click log.
        """
        try:
            font = pygame.font.SysFont('Arial', 12)
        except Exception:
            return

        # Background box
        box_w = 380
        box_h = 24 + 16 * min(10, len(self.widget_manager.widgets) if self.widget_manager else 0)
        box_surf = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        box_surf.fill((20, 20, 30, 220))
        surface.blit(box_surf, (10, 10))

        x = 14
        y = 14
        mouse_pos = pygame.mouse.get_pos()
        surface.blit(font.render(f"Mouse: {mouse_pos}", True, (220, 220, 220)), (x, y))
        y += 16
        panel_rect = getattr(self, 'control_panel_rect', None)
        surface.blit(font.render(f"Panel rect: {panel_rect}", True, (220, 220, 220)), (x, y))
        y += 16
        surface.blit(font.render(f"Collapse btn: {getattr(self,'collapse_button_rect',None)}", True, (220, 220, 220)), (x, y))
        y += 18

        # List first few widgets
        if self.widget_manager:
            for w in self.widget_manager.widgets[:8]:
                info = f"{getattr(w,'control_name',w.__class__.__name__)} rect={w.rect} open={getattr(w,'is_open',False)} state={w.state}"
                surface.blit(font.render(info, True, (200, 200, 255)), (x, y))
                y += 14

        # Draw outlines for panel, collapse button, and open dropdown menus
        if panel_rect:
            try:
                pygame.draw.rect(surface, (200, 80, 80), panel_rect, 2)
            except Exception:
                pass
        if getattr(self, 'collapse_button_rect', None):
            try:
                pygame.draw.rect(surface, (80, 200, 120), self.collapse_button_rect, 2)
            except Exception:
                pass

        # Recent clicks
        cx = 14
        cy = box_h + 30
        surface.blit(font.render("Recent clicks (latest first):", True, (200, 200, 180)), (cx, cy))
        cy += 14
        for pos, ts in (self.debug_click_log[:8] if getattr(self,'debug_click_log',None) else []):
            surface.blit(font.render(f"{pos} @ {int(ts)}", True, (220, 220, 180)), (cx, cy))
            cy += 12

    def _render_unified_bottom_panel(self):
        """Render unified bottom HUD panel - STATUS and MESSAGE only (inventory moved to sidebar)."""
        # Panel dimensions - reduced height since no inventory
        panel_height = 80
        panel_y = self.screen_h - panel_height - 5
        panel_x = 5
        panel_width = self.screen_w - self.SIDEBAR_WIDTH - 15
        
        # Create rounded background
        panel_surf = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel_bg = pygame.Rect(0, 0, panel_width, panel_height)
        pygame.draw.rect(panel_surf, (35, 35, 50, 230), panel_bg, border_radius=8)
        pygame.draw.rect(panel_surf, (60, 60, 80), panel_bg, 2, border_radius=8)
        
        self.screen.blit(panel_surf, (panel_x, panel_y))
        
        # Calculate section widths (only status and message)
        padding = 15
        total_inner_width = panel_width - (padding * 2)
        
        status_width = int(total_inner_width * 0.50)  # 50% for status
        message_width = total_inner_width - status_width  # 50% for message
        
        # Section X positions
        status_x = panel_x + padding
        message_x = status_x + status_width + 20
        
        content_y = panel_y + 10
        section_height = panel_height - 20
        
        # Draw vertical divider
        divider_color = (60, 60, 80)
        pygame.draw.line(self.screen, divider_color,
                        (message_x - 10, content_y),
                        (message_x - 10, content_y + section_height), 2)
        
        # Render sections
        self._render_status_section(status_x, content_y, status_width, section_height)
        self._render_message_section(message_x, content_y, message_width, section_height)
    
    def _render_message_section(self, x: int, y: int, width: int, height: int):
        """Render message/status section in bottom panel."""
        # Title
        title_surf = self.font.render("MESSAGE", True, (100, 255, 150))
        self.screen.blit(title_surf, (x, y))
        y += 22
        
        # Current message with appropriate color
        msg_color = (255, 255, 100) if (self.env and self.env.won) else (200, 200, 200)
        
        # Wrap long messages
        if len(self.message) > 35:
            msg_lines = [self.message[i:i+35] for i in range(0, len(self.message), 35)]
            for line in msg_lines[:2]:  # Max 2 lines
                msg_surf = self.small_font.render(line, True, msg_color)
                self.screen.blit(msg_surf, (x, y))
                y += 16
        else:
            msg_surf = self.small_font.render(self.message, True, msg_color)
            self.screen.blit(msg_surf, (x, y))
    
    def _render_progress_bar(self, surface: pygame.Surface, x: int, y: int, width: int, height: int, 
                             filled: int, total: int, color_filled: tuple, color_empty: tuple):
        """Render a segmented progress bar with filled/empty indicators."""
        if total == 0:
            return
        
        segments = min(total, 10)  # Max 10 segments for visual clarity
        segment_width = width // max(1, segments)
        items_per_segment = total / segments
        
        for i in range(segments):
            segment_x = x + i * segment_width
            segment_rect = pygame.Rect(segment_x + 1, y, segment_width - 2, height)
            
            # Determine if this segment should be filled
            segment_threshold = (i + 1) * items_per_segment
            is_filled = filled >= segment_threshold
            
            if is_filled:
                # Filled segment with gradient effect
                pygame.draw.rect(surface, color_filled, segment_rect, border_radius=2)
                # Lighter border
                highlight = tuple(min(c + 40, 255) for c in color_filled[:3])
                pygame.draw.rect(surface, highlight, segment_rect, 1, border_radius=2)
            else:
                # Empty segment
                pygame.draw.rect(surface, color_empty, segment_rect, border_radius=2)
                pygame.draw.rect(surface, (60, 60, 80), segment_rect, 1, border_radius=2)
    
    def _render_inventory_section(self, x: int, y: int, width: int, height: int):
        """Render inventory section with progress bars and icons."""
        # Title
        title_surf = self.font.render("INVENTORY", True, (100, 200, 255))
        self.screen.blit(title_surf, (x, y))
        
        y_offset = y + 25
        line_height = 24
        bar_width = width - 10
        bar_height = 8
        
        if not self.env:
            return
        
        # Keys with progress bar
        held_keys = self.env.state.keys if hasattr(self.env.state, 'keys') else 0
        keys_color = (255, 215, 0)  # Gold
        
        # Flash animation if recently picked up
        current_time = time.time()
        for pos, pickup_time in list(self.item_pickup_times.items()):
            if current_time - pickup_time < 0.5:
                keys_color = (255, 255, 150)
                break
        
        # Text: "K: 4/7 (3 held)"
        keys_text = f"K: {self.keys_collected}/{self.total_keys}"
        if held_keys > 0:
            keys_text += f" ({held_keys} held)"
        keys_surf = self.small_font.render(keys_text, True, keys_color)
        self.screen.blit(keys_surf, (x, y_offset))
        
        # Progress bar
        if self.total_keys > 0:
            self._render_progress_bar(self.screen, x, y_offset + 16, bar_width, bar_height,
                                     self.keys_collected, self.total_keys,
                                     keys_color, (40, 40, 50))
        y_offset += line_height + 10
        
        # Bombs
        has_bomb = hasattr(self.env.state, 'has_bomb') and self.env.state.has_bomb
        bombs_color = (255, 107, 53) if has_bomb else (100, 100, 100)  # Orange
        bombs_status = "[YES]" if has_bomb else "[NO]"
        
        if self.total_bombs > 0:
            bombs_text = f"B: {bombs_status} Bomb"
            bombs_surf = self.small_font.render(bombs_text, True, bombs_color)
            self.screen.blit(bombs_surf, (x, y_offset))
            y_offset += line_height
        
        # Boss Key with progress bar
        has_boss_key = hasattr(self.env.state, 'has_boss_key') and self.env.state.has_boss_key
        boss_key_color = (176, 66, 255) if has_boss_key else (100, 100, 100)  # Purple
        
        if self.total_boss_keys > 0:
            boss_key_text = f"Boss Key: {self.boss_keys_collected}/{self.total_boss_keys}"
            if has_boss_key:
                boss_key_text += " [Y]"
            boss_key_surf = self.small_font.render(boss_key_text, True, boss_key_color)
            self.screen.blit(boss_key_surf, (x, y_offset))
            
            # Progress bar
            self._render_progress_bar(self.screen, x, y_offset + 16, bar_width, bar_height,
                                     self.boss_keys_collected, self.total_boss_keys,
                                     boss_key_color, (40, 40, 50))
    
    def _render_metrics_section(self, x: int, y: int, width: int, height: int):
        """Render metrics section (steps, speed, zoom, env)."""
        # Title
        title_surf = self.font.render("METRICS", True, (150, 200, 255))
        self.screen.blit(title_surf, (x, y))
        
        y_offset = y + 25
        line_height = 20
        
        # Steps
        steps_surf = self.small_font.render(f"Steps: {self.step_count}", True, (200, 200, 200))
        self.screen.blit(steps_surf, (x, y_offset))
        y_offset += line_height
        
        # Speed
        speed_color = (100, 255, 100) if self.speed_multiplier == 1.0 else (255, 200, 100)
        speed_surf = self.small_font.render(f"Speed: {self.speed_multiplier}x", True, speed_color)
        self.screen.blit(speed_surf, (x, y_offset))
        y_offset += line_height
        
        # Zoom
        zoom_surf = self.small_font.render(f"Zoom: {self.TILE_SIZE}px", True, (150, 150, 150))
        self.screen.blit(zoom_surf, (x, y_offset))
        y_offset += line_height
        
        # Env Steps
        env_steps = self.env.step_count if self.env and hasattr(self.env, 'step_count') else 0
        env_surf = self.small_font.render(f"Env: {env_steps}", True, (150, 150, 150))
        self.screen.blit(env_surf, (x, y_offset))
    
    def _render_controls_section(self, x: int, y: int, width: int, height: int):
        """Render controls section in two-column layout."""
        # Title
        title_surf = self.font.render("CONTROLS", True, (100, 200, 100))
        self.screen.blit(title_surf, (x, y))
        
        y_offset = y + 25
        line_height = 16
        col_width = width // 2
        
        # Two-column layout
        controls_left = [
            ("ARROWS", "Move"),
            ("SPACE", "Solve"),
            ("R", "Reset"),
            ("N/P", "Maps"),
            ("[/]", "Speed"),
        ]
        
        controls_right = [
            ("M", "Minimap"),
            ("H", "Heatmap"),
            ("+/-", "Zoom"),
            ("F11", "Full"),
            ("ESC", "Quit"),
        ]
        
        text_color = (120, 120, 120)
        
        # Left column
        for key, desc in controls_left:
            control_surf = self.small_font.render(f"{key:4s} {desc}", True, text_color)
            self.screen.blit(control_surf, (x, y_offset))
            y_offset += line_height
        
        # Right column
        y_offset = y + 25
        for key, desc in controls_right:
            control_surf = self.small_font.render(f"{key:4s} {desc}", True, text_color)
            self.screen.blit(control_surf, (x + col_width, y_offset))
            y_offset += line_height
    
    def _render_status_section(self, x: int, y: int, width: int, height: int):
        """Render status section with game state information."""
        # Title
        title_surf = self.font.render("STATUS", True, (180, 220, 255))
        self.screen.blit(title_surf, (x, y))
        
        y_offset = y + 25
        line_height = 18
        
        # Victory or current status
        if self.env and self.env.won:
            status_text = "*** VICTORY! ***"
            status_color = (255, 215, 0)
            status_surf = self.big_font.render(status_text, True, status_color)
            self.screen.blit(status_surf, (x, y_offset))
        else:
            # Current map/dungeon
            if self.current_map_idx < len(self.map_names):
                map_name = self.map_names[self.current_map_idx]
                map_text = f"Map: {map_name[:15]}"
                map_surf = self.small_font.render(map_text, True, (150, 200, 255))
                self.screen.blit(map_surf, (x, y_offset))
                y_offset += line_height
            
            # Position
            if self.env and hasattr(self.env.state, 'position'):
                pos = self.env.state.position
                pos_text = f"Pos: ({pos[0]}, {pos[1]})"
                pos_surf = self.small_font.render(pos_text, True, (150, 150, 150))
                self.screen.blit(pos_surf, (x, y_offset))
                y_offset += line_height
            
            # Auto-solve progress
            if self.auto_mode and self.auto_path:
                progress_text = f"Auto: {self.auto_step_idx}/{len(self.auto_path)}"
                progress_surf = self.small_font.render(progress_text, True, (100, 255, 150))
                self.screen.blit(progress_surf, (x, y_offset))
                y_offset += line_height
            
            # Status message
            if self.status_message:
                status_surf = self.small_font.render(self.status_message[:20], True, (180, 220, 255))
                self.screen.blit(status_surf, (x, y_offset))
    
    def _render_minimap(self):
        """Render small dungeon overview map in bottom-right corner."""
        if not self.env:
            return
        
        # Minimap positioning (bottom-right, above HUD)
        minimap_margin = 20
        minimap_x = self.screen_w - self.SIDEBAR_WIDTH - self.minimap_size - minimap_margin
        minimap_y = self.screen_h - self.HUD_HEIGHT - self.minimap_size - minimap_margin
        
        # Create semi-transparent minimap surface
        minimap = pygame.Surface((self.minimap_size, self.minimap_size), pygame.SRCALPHA)
        pygame.draw.rect(minimap, (40, 40, 60, 220), minimap.get_rect(), border_radius=8)
        
        # Draw title
        title_font = pygame.font.SysFont('Arial', 10, bold=True)
        title_surf = title_font.render("Dungeon Map", True, (180, 180, 200))
        minimap.blit(title_surf, (5, 3))
        
        # Calculate scaling factor to fit dungeon in minimap
        map_h, map_w = self.env.height, self.env.width
        content_area = self.minimap_size - 30  # Leave room for title and padding
        scale_x = content_area / map_w
        scale_y = content_area / map_h
        scale = min(scale_x, scale_y)
        
        # Calculate offset to center the minimap content
        scaled_w = int(map_w * scale)
        scaled_h = int(map_h * scale)
        offset_x = (self.minimap_size - scaled_w) // 2
        offset_y = 18 + (self.minimap_size - 18 - scaled_h) // 2
        
        # Draw simplified dungeon layout
        # Use different colors for different tile types
        for r in range(map_h):
            for c in range(map_w):
                tile_id = self.env.grid[r, c]
                
                # Determine tile color based on semantic type
                if tile_id == SEMANTIC_PALETTE['VOID']:
                    continue  # Skip void tiles (transparent)
                elif tile_id == SEMANTIC_PALETTE['WALL'] or tile_id == SEMANTIC_PALETTE['BLOCK']:
                    color = (60, 60, 80)  # Dark gray for walls
                elif tile_id == SEMANTIC_PALETTE['START']:
                    color = (80, 180, 80)  # Green for start
                elif tile_id == SEMANTIC_PALETTE['TRIFORCE']:
                    color = (255, 215, 0)  # Gold for goal
                elif tile_id in [SEMANTIC_PALETTE['KEY_SMALL'], SEMANTIC_PALETTE['KEY_BOSS']]:
                    color = (255, 200, 50)  # Yellow for keys
                elif tile_id in [SEMANTIC_PALETTE['DOOR_LOCKED'], SEMANTIC_PALETTE['DOOR_BOMB'], SEMANTIC_PALETTE['DOOR_BOSS']]:
                    color = (180, 100, 50)  # Brown for locked doors
                elif tile_id == SEMANTIC_PALETTE['STAIR']:
                    color = (100, 150, 255)  # Blue for stairs
                elif tile_id == SEMANTIC_PALETTE['ENEMY']:
                    color = (200, 50, 50)  # Red for enemies
                else:
                    color = (100, 120, 140)  # Light gray for floors
                
                # Draw mini-tile
                mini_x = offset_x + int(c * scale)
                mini_y = offset_y + int(r * scale)
                mini_w = max(1, int(scale))
                mini_h = max(1, int(scale))
                pygame.draw.rect(minimap, color, (mini_x, mini_y, mini_w, mini_h))
        
        # Draw current player position (bright dot)
        pr, pc = self.env.state.position
        player_x = offset_x + int(pc * scale)
        player_y = offset_y + int(pr * scale)
        player_size = max(2, int(scale * 1.5))
        pygame.draw.circle(minimap, (255, 100, 100), (player_x, player_y), player_size)
        # Add white outline for visibility
        pygame.draw.circle(minimap, (255, 255, 255), (player_x, player_y), player_size + 1, 1)
        
        # Highlight uncollected items with pulsing effect
        current_time = time.time()
        pulse = (math.sin(current_time * 3) + 1) / 2  # 0 to 1
        
        # Draw uncollected keys (yellow pulsing dots)
        for pos in self.env._find_all_positions(SEMANTIC_PALETTE['KEY_SMALL']):
            if pos not in self.env.state.collected_items:
                r, c = pos
                mini_x = offset_x + int(c * scale)
                mini_y = offset_y + int(r * scale)
                size = int(2 + pulse * 2)
                pygame.draw.circle(minimap, (255, 255, 0), (mini_x, mini_y), size)
        
        # Draw uncollected boss keys (orange pulsing dots)
        for pos in self.env._find_all_positions(SEMANTIC_PALETTE['KEY_BOSS']):
            if pos not in self.env.state.collected_items:
                r, c = pos
                mini_x = offset_x + int(c * scale)
                mini_y = offset_y + int(r * scale)
                size = int(2 + pulse * 2)
                pygame.draw.circle(minimap, (255, 150, 0), (mini_x, mini_y), size)
        
        # Draw border
        pygame.draw.rect(minimap, (70, 70, 100), minimap.get_rect(), 2, border_radius=8)
        
        # Blit minimap to screen
        self.screen.blit(minimap, (minimap_x, minimap_y))
    
    def _handle_minimap_click(self, mouse_pos: Tuple[int, int]) -> bool:
        """Handle mouse click on minimap to jump to that location."""
        if not self.show_minimap or not self.env:
            return False
        
        # Calculate minimap position
        minimap_margin = 20
        minimap_x = self.screen_w - self.SIDEBAR_WIDTH - self.minimap_size - minimap_margin
        minimap_y = self.screen_h - self.HUD_HEIGHT - self.minimap_size - minimap_margin
        
        # Check if click is within minimap bounds
        mx, my = mouse_pos
        if not (minimap_x <= mx <= minimap_x + self.minimap_size and
                minimap_y <= my <= minimap_y + self.minimap_size):
            return False
        
        # Convert mouse position to map coordinates
        map_h, map_w = self.env.height, self.env.width
        content_area = self.minimap_size - 30
        scale_x = content_area / map_w
        scale_y = content_area / map_h
        scale = min(scale_x, scale_y)
        
        scaled_w = int(map_w * scale)
        scaled_h = int(map_h * scale)
        offset_x = (self.minimap_size - scaled_w) // 2
        offset_y = 18 + (self.minimap_size - 18 - scaled_h) // 2
        
        # Calculate clicked tile
        local_x = mx - minimap_x - offset_x
        local_y = my - minimap_y - offset_y
        
        if local_x < 0 or local_y < 0:
            return True
        
        tile_c = int(local_x / scale)
        tile_r = int(local_y / scale)
        
        if 0 <= tile_r < map_h and 0 <= tile_c < map_w:
            # Center view on clicked tile
            self.view_offset_x = int(tile_c * self.TILE_SIZE - (self.screen_w - self.SIDEBAR_WIDTH) / 2)
            self.view_offset_y = int(tile_r * self.TILE_SIZE - (self.screen_h - self.HUD_HEIGHT) / 2)
            self._clamp_view_offset()
            self.message = f"Jumped to ({tile_r}, {tile_c})"
        
        return True
    
    def _render_help_overlay(self):
        """Render help overlay."""
        overlay = pygame.Surface((self.screen_w, self.screen_h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        self.screen.blit(overlay, (0, 0))
        
        help_lines = [
            "ZAVE - Zelda AI Validation Environment",
            "",
            "Movement:",
            "  Arrow Keys - Move Link",
            "  Mouse Wheel - Zoom in/out",
            "  Middle Mouse Drag - Pan view",
            "",
            "Actions:",
            "  SPACE - Run A* auto-solver",
            "  R - Reset current map",
            "  N - Next map",
            "  P - Previous map",
            "",
            "View:",
            "  +/- or Wheel - Zoom in/out",
            "  0 - Reset zoom to default",
            "  C - Center view on player",
            "  M - Toggle minimap",
            "  H - Toggle A* heatmap",
            "  TAB - Toggle control panel",
            "  F11 - Toggle fullscreen",
            "",
            "Speed Control:",
            "  [ or , - Decrease speed",
            "  ] or . - Increase speed",
            "  (Speeds: 0.25x, 0.5x, 1x, 2x, 5x, 10x)",
            "",
            "Press F1 or ESC to close this help",
        ]
        
        y = 50
        for line in help_lines:
            if line.startswith("ZAVE"):
                surf = self.big_font.render(line, True, (100, 200, 255))
            elif line.endswith(":") and not line.startswith(" "):
                surf = self.font.render(line, True, (255, 200, 100))
            else:
                surf = self.small_font.render(line, True, (200, 200, 200))
            self.screen.blit(surf, (50, y))
            y += 22 if line else 10


def load_maps_from_adapter():
    """Load processed maps from data adapter using new zelda_core - ALL 18 variants."""
    try:
        from Data.zelda_core import ZeldaDungeonAdapter, DungeonSolver
        from pathlib import Path
        
        data_root = Path(__file__).parent / "Data" / "The Legend of Zelda"
        
        if not data_root.exists():
            print(f"Data folder not found: {data_root}")
            return None, None
        
        adapter = ZeldaDungeonAdapter(str(data_root))
        solver = DungeonSolver()
        
        maps = []  # Store full StitchedDungeon objects
        map_names = []  # Track dungeon names
        print("Loading all 18 dungeon variants (9 dungeons x 2 variants)...")
        
        for dungeon_num in range(1, 10):
            for variant in [1, 2]:
                try:
                    dungeon = adapter.load_dungeon(dungeon_num, variant=variant)
                    stitched = adapter.stitch_dungeon(dungeon)
                    
                    # Store the full stitched dungeon (includes graph and room mappings)
                    maps.append(stitched)
                    
                    # Store name
                    quest_name = "Quest 1" if variant == 1 else "Quest 2"
                    map_names.append(f"Dungeon {dungeon_num} ({quest_name})")
                    
                    # Check solvability
                    result = solver.solve(stitched)
                    status = "[OK]" if result['solvable'] else "[X]"
                    print(f"  D{dungeon_num}-{variant}: {status} - {stitched.global_grid.shape}")
                    
                except Exception as e:
                    print(f"  D{dungeon_num}-{variant}: Error - {e}")
        
        return maps if maps else None, map_names if map_names else None
        
    except Exception as e:
        print(f"Error loading maps: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    """Main entry point."""
    print("=== ZAVE GUI Runner ===\n")
    
    if not PYGAME_AVAILABLE:
        print("Pygame is not installed. Please run: pip install pygame")
        return
    
    # Try to load processed maps
    maps, map_names = load_maps_from_adapter()
    
    if maps:
        print(f"Loaded {len(maps)} maps from data adapter")
    else:
        print("Using test map")
        maps = [create_test_map()]
        map_names = ["Test Map"]
    
    # Start GUI
    gui = ZeldaGUI(maps, map_names)
    gui.run()


if __name__ == "__main__":
    main()
